"""app/ratelimit.py — 提交端点令牌桶限流（产品化 M1-R3）。

公开访问的体检提交要防 LLM 费用被刷穿（调研 §3.6）：每 IP 内存令牌桶，
默认匿名 5 次/小时（容量 5，每小时回满）。进程内实现、不引入 Redis——
与任务表同理（演示量级取舍）；多副本部署时各副本独立计数，需共享限流
再升级（R4 之后按真实流量决定）。

IP 取自 ``request.client.host``：R4 部署在 nginx 后时须以
``uvicorn --proxy-headers --forwarded-allow-from <nginx网段>`` 启动，
让 client.host 反映真实客户端 IP（XFF 由 uvicorn 在受信代理下解析）；
直接暴露时即对端地址。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class TokenBucketLimiter:
    """按 key（IP）的内存令牌桶：容量 capacity，refill_seconds 内回满。"""

    def __init__(
        self,
        *,
        capacity: int = 5,
        refill_seconds: float = 3600.0,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity 必须 ≥ 1")
        self._capacity = capacity
        self._refill_seconds = refill_seconds
        self._now = now_fn
        self._buckets: dict[str, tuple[float, float]] = {}  # key → (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """取一枚令牌；成功 True（扣 1 枚），失败 False（桶空，原样保留余量）。"""
        now = self._now()
        with self._lock:
            self._purge_idle_locked(now)
            tokens, last = self._buckets.get(key, (float(self._capacity), now))
            tokens = min(
                float(self._capacity),
                tokens + (now - last) / self._refill_seconds * self._capacity,
            )
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return True
            self._buckets[key] = (tokens, now)
            return False

    def retry_after_seconds(self, key: str) -> int:
        """桶空时距下一枚令牌的秒数（向上取整，供 Retry-After 头）。

        与 :meth:`allow` 同一回充公式——按已流逝时间回充后再算缺口。
        """
        now = self._now()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self._capacity), now))
        tokens = min(
            float(self._capacity),
            tokens + (now - last) / self._refill_seconds * self._capacity,
        )
        deficit = 1.0 - tokens
        if deficit <= 0:
            return 0
        return int(-(-deficit * self._refill_seconds / self._capacity // 1))  # ceil

    def _purge_idle_locked(self, now: float) -> None:
        """清掉超过两个回满周期无活动的桶（防 IP 无界增长；持锁调用）。"""
        idle_cutoff = now - 2 * self._refill_seconds
        stale = [k for k, (_, last) in self._buckets.items() if last < idle_cutoff]
        for k in stale:
            del self._buckets[k]
        if stale:
            logger.info("限流器清理空闲 IP 桶 %d 个", len(stale))
