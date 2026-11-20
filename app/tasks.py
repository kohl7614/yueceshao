"""app/tasks.py — 进程内体检任务表（产品化 M1-R2）。

CPU 重排跑不进小程序 60s 请求超时（调研 §3.3），服务契约从同步改为
「提交 → task_id → 轮询」。本模块是进程内的最小任务表：

- **不引入 Redis/Celery**（路线图 D：演示量级不需要新基础设施）；进程重启
  任务即丢——重启丢任务对演示可接受，状态轮询 404 后由客户端重新提交；
- 单 worker 串行（默认）：管线含 CPU 重排与共享命中计划，并发跑会相互
  踢掉缓存并争抢 CPU；排队由 ``queued`` 状态如实呈现，R3 限流防灌爆；
- 任务保留 1 小时（TTL 惰性清理：submit/get 时顺手扫），不提供持久化。

状态机：queued → running → done | failed（终态，不可逆）。
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

#: 任务保留时长（超过即被惰性清理；轮询到 404 客户端应重新提交）。
TASK_TTL = timedelta(hours=1)

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class TaskStore:
    """进程内任务表：提交即排队，单 worker 串行执行，TTL 惰性清理。"""

    def __init__(self, *, max_workers: int = 1, ttl: timedelta = TASK_TTL) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="checkup")
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    # —— 提交与查询 ——
    def submit(self, fn: Callable[[], dict]) -> str:
        """入队一个体检任务（fn 无参、返回结果 payload dict）；返回 task_id。

        fn 抛异常 → 任务转 failed，``str(exc)`` 会作为 failed.error **返回给
        客户端**——fn 必须抛携带安全文案的异常（内部细节由 fn 自己落日志），
        参见 app/main.py ``_execute_checkup``。
        """
        self._purge_expired()
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": STATUS_QUEUED,
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "elapsed_ms": None,
                "result": None,
                "error": None,
            }
        self._executor.submit(self._run, task_id, fn)
        return task_id

    def get(self, task_id: str) -> dict | None:
        """按 id 取任务快照（dict 拷贝；未知或已过期返回 None）。"""
        self._purge_expired()
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            return dict(record)

    # —— 执行 ——
    def _run(self, task_id: str, fn: Callable[[], dict]) -> None:
        started = datetime.now(timezone.utc).astimezone()
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:  # 已被 TTL 清理（极端：入队后 1 小时未调度）
                return
            record["status"] = STATUS_RUNNING
            record["started_at"] = _now()
        try:
            payload = fn()
        except Exception as exc:  # noqa: BLE001 - 安全文案由路由层给出，此处只落日志
            logger.exception("体检任务 %s 执行失败", task_id)
            with self._lock:
                record = self._tasks.get(task_id)
                if record is not None:
                    record["status"] = STATUS_FAILED
                    record["error"] = str(exc) or exc.__class__.__name__
                    record["finished_at"] = _now()
                    record["elapsed_ms"] = int(
                        (datetime.now(timezone.utc).astimezone() - started).total_seconds() * 1000)
            return
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                record["status"] = STATUS_DONE
                record["result"] = payload
                record["finished_at"] = _now()
                record["elapsed_ms"] = int(
                    (datetime.now(timezone.utc).astimezone() - started).total_seconds() * 1000)

    # —— 清理 ——
    def _purge_expired(self) -> None:
        cutoff = datetime.now(timezone.utc).astimezone() - self._ttl
        cutoff_str = cutoff.isoformat(timespec="seconds")
        with self._lock:
            expired = [tid for tid, rec in self._tasks.items()
                       if rec["created_at"] < cutoff_str]
            for tid in expired:
                del self._tasks[tid]
        if expired:
            logger.info("清理过期任务 %d 个", len(expired))
