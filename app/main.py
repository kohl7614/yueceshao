"""app/main.py — FastAPI 服务（骨架 §3.5 ＋ R2 异步任务化 ＋ R3 安全与合规）。

路由：
- POST /api/checkup：接收项目描述 → **限流检查 → 入队体检任务** → 202＋task_id
  ＋轮询地址（CPU 重排跑不进小程序 60s 请求超时，调研 §3.3；同步契约自 R2 起
  废弃）。限流＝内存令牌桶每 IP（默认 5 次/时，app/ratelimit.py），超频 429；
- GET /api/tasks/{task_id}：轮询任务状态（queued/running/done/failed）；
  done 时 result 携带完整四段式报告（结构化 JSON＋渲染文本＋免责声明）；
- GET /health：存活检查＋当前管线名（真实管线另带装配事实，见 health_info）。

安全与合规（M1-R3）：
- **AI 生成标识**（《人工智能生成合成内容标识办法》2025-09-01 施行）：显式标识
  由本服务端注入渲染文本（插在免责声明之前——免责声明保持末行的红线不变）；
  隐式标识＝响应结构化字段 ``ai_generated: true``（机器可读）。注入点在 API
  worker（服务端渲染层），评测管线（gen/）产物不受影响——客户端只展示；
- **输入上限**：description ≤ 2000 字（422 拒收；评测案例 150–300 字，余量充足）；
- **限流**：默认匿名 5 次/时/IP，防 LLM 费用被刷穿；Retry-After 头给出回满秒数。

配置与装配：
- 管线经 create_app(pipeline=...) 注入（依赖倒置：create_app 缺省 Mock（测试用）；
  真实装配入口 = app/real_main.py（RealCheckupPipeline＋真实 LLM），实现同一
  CheckupPipeline 协议，路由零改动）；
- 任务表 = app/tasks.py 进程内实现（单 worker 串行＋TTL 1 小时；重启丢任务，
  轮询 404 后客户端重新提交）；
- 配置走环境变量 / config.json（不硬编码 key）；日志用 logging。

免责声明红线：渲染层保证其存在（gen.validator/渲染管线），worker 在任务完成时
复核哨兵，缺失即 failed（安全文案，不泄内部细节）。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from app.pipeline import CheckupPipeline, MockCheckupPipeline, PipelineResult
from app.ratelimit import TokenBucketLimiter
from app.tasks import STATUS_QUEUED, TaskStore
from gen.schema import DISCLAIMER_TEXT

logger = logging.getLogger(__name__)

# 免责声明哨兵＝完整产品级原文（渲染层注入的即此文本，逐字比对最严格）
DISCLAIMER_SENTINEL = DISCLAIMER_TEXT

# AI 生成标识（《人工智能生成合成内容标识办法》2025-09-01 施行，红线级）：
# 显式标识行——服务端注入渲染文本，插在「免责声明：」之前（免责声明保持末行）。
AI_GENERATED_LINE = "AI 生成标识：本报告由人工智能自动生成，仅供参考。"
_DISCLAIMER_MARKER = "免责声明："

# 失败安全文案（与任务 failed 状态一起返回；内部细节只落服务端日志）
_GENERIC_FAILURE = "体检执行失败，请稍后重试"
_SENTINEL_FAILURE = "服务内部校验失败"

#: 轮询预计耗时提示（UX 参考值，非承诺）：真实管线（CPU 重排）约两分钟，其余取缺省。
DEFAULT_ESTIMATED_SECONDS = 5

#: 输入长度上限（字）：评测案例 150–300 字，2000 为申报场景留足余量。
MAX_DESCRIPTION_CHARS = 2000

#: 限流默认值：匿名 5 次/时/IP（路线图 R3；评审演示量级）。
RATE_LIMIT_CAPACITY = 5
RATE_LIMIT_REFILL_SECONDS = 3600.0

#: 限流拒绝文案与 Retry-After 兜底秒数。
_RATE_LIMIT_DETAIL = "提交过于频繁（每 IP 每小时 5 次），请稍后再试"


class CheckupRequest(BaseModel):
    """体检请求体：项目描述（简介／资金用途／建设内容）。"""

    description: str

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("description 不能为空")
        text = v.strip()
        if len(text) < 20:
            raise ValueError("description 过短（至少 20 字），请按「简介＋资金用途＋建设内容」补全")
        if len(text) > MAX_DESCRIPTION_CHARS:
            raise ValueError(f"description 过长（上限 {MAX_DESCRIPTION_CHARS} 字）")
        return text


class TaskSubmissionResponse(BaseModel):
    """202 响应：任务受理凭据（轮询地址＋预计耗时提示）。"""

    task_id: str
    status: str
    poll_url: str
    estimated_seconds: int


class TaskStatusResponse(BaseModel):
    """轮询响应：状态与产物（终态前 result/error 均为空）。"""

    task_id: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_ms: int | None = None
    result: dict | None = None
    error: str | None = None


class CheckupResponse(BaseModel):
    """体检结果（done 任务的 result 形状）。

    ``ai_generated`` 是隐式标识字段（机器可读，《标识办法》）；显式标识在
    ``rendered_text`` 内（AI_GENERATED_LINE，服务端注入）。
    """

    pipeline: str
    report: dict
    rendered_text: str
    degraded: bool
    attempts: int
    failed_sections: list[str]
    disclaimer: str
    ai_generated: bool = True


class _SentinelFailure(RuntimeError):
    """免责声明哨兵复核失败（消息即安全文案，可原样进 failed.error）。"""


def _insert_ai_line(rendered_text: str) -> str:
    """显式 AI 标识插到「免责声明：」之前（免责声明保持末行的红线不变）。"""
    marker = _DISCLAIMER_MARKER
    if marker in rendered_text:
        index = rendered_text.rindex(marker)
        return rendered_text[:index] + AI_GENERATED_LINE + "\n\n" + rendered_text[index:]
    return rendered_text + "\n\n" + AI_GENERATED_LINE


def _execute_checkup(pipe: CheckupPipeline, description: str) -> dict:
    """跑一次体检并构造结果 payload（worker 线程内执行）。

    失败一律转成携带**安全文案**的异常——TaskStore 把异常消息存进
    failed.error 返回给客户端，内部细节（路径/堆栈/key）只落服务端日志。
    整个构造过程包在 try 内：管线违约（如字段类型不符）也不裸传异常原文。
    AI 生成标识（显式行＋隐式字段）由本 worker（服务端渲染层）注入。
    """
    try:
        result: PipelineResult = pipe.run(description)
        if DISCLAIMER_SENTINEL not in result.rendered_text:
            logger.error("管线产物缺少免责声明，任务转失败")
            raise _SentinelFailure(_SENTINEL_FAILURE)
        return CheckupResponse(
            pipeline=pipe.name,
            report=result.to_api_dict()["report"],
            rendered_text=_insert_ai_line(result.rendered_text),
            degraded=result.degraded,
            attempts=result.attempts,
            failed_sections=list(result.failed_sections),
            disclaimer=result.disclaimer,
            ai_generated=True,
        ).model_dump()
    except _SentinelFailure:
        raise  # 消息即安全文案，原样上抛
    except Exception:  # noqa: BLE001 —— 客户端只见安全文案（原同步 500 的 detail 口径）
        logger.exception("体检管线执行失败")
        raise RuntimeError(_GENERIC_FAILURE) from None


def create_app(
    pipeline: CheckupPipeline | None = None,
    *,
    rate_limit_capacity: int = RATE_LIMIT_CAPACITY,
    rate_limit_refill_seconds: float = RATE_LIMIT_REFILL_SECONDS,
) -> FastAPI:
    """装配 FastAPI 实例（pipeline 缺省为 Mock——测试用；真实入口见 app/real_main.py）。

    ``rate_limit_*`` 可调小供测试（生产走默认 5 次/时/IP）。
    """
    pipe = pipeline or MockCheckupPipeline()
    store = TaskStore()
    limiter = TokenBucketLimiter(
        capacity=rate_limit_capacity, refill_seconds=rate_limit_refill_seconds)
    app = FastAPI(
        title="粤策哨 · 申报前体检 API",
        version="0.4.0-compliance",
        description="县域强村公司与合作社申报前自查辅助（不构成合规判定；"
                    "真实装配入口见 app/real_main.py，Mock 仅用于测试）",
    )

    @app.get("/health")
    def health() -> dict:
        info: dict = {"status": "ok", "pipeline": pipe.name}
        # 真实管线可选携带装配事实（索引块数/检索口径/LLM 后端）；Mock 无此方法，
        # /health 保持骨架形状——响应字段按管线能力如实增减，不硬造。
        extra = getattr(pipe, "health_info", None)
        if callable(extra):
            info.update(extra())
        return info

    @app.post("/api/checkup", status_code=202, response_model=TaskSubmissionResponse)
    def checkup(req: CheckupRequest, request: Request) -> TaskSubmissionResponse:
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.allow(client_ip):
            retry_after = limiter.retry_after_seconds(client_ip)
            logger.warning("限流拒绝：ip=%s（Retry-After=%ss）", client_ip, retry_after)
            raise HTTPException(
                status_code=429, detail=_RATE_LIMIT_DETAIL,
                headers={"Retry-After": str(max(retry_after, 1))})
        logger.info("体检任务受理：%d 字", len(req.description))
        try:
            task_id = store.submit(lambda: _execute_checkup(pipe, req.description))
        except Exception as exc:  # noqa: BLE001 —— 入队失败（线程池已关等）兜底成 500
            logger.exception("体检任务入队失败")
            raise HTTPException(status_code=500, detail="服务暂不可用，请稍后重试") from exc
        return TaskSubmissionResponse(
            task_id=task_id,
            status=STATUS_QUEUED,
            poll_url=f"/api/tasks/{task_id}",
            estimated_seconds=getattr(pipe, "estimated_seconds", DEFAULT_ESTIMATED_SECONDS),
        )

    @app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
    def task_status(task_id: str) -> TaskStatusResponse:
        record = store.get(task_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail="任务不存在或已过期（保留 1 小时），请重新提交")
        return TaskStatusResponse(**record)

    return app


app = create_app()
