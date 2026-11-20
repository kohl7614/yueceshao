"""app/main.py — FastAPI 服务（手册 §3.5 骨架 ＋ 产品化 M1-R2 异步任务化）。

路由：
- POST /api/checkup：接收项目描述 → **入队体检任务** → 202＋task_id＋轮询地址
  （CPU 重排跑不进小程序 60s 请求超时，调研 §3.3；同步契约自 R2 起废弃）；
- GET /api/tasks/{task_id}：轮询任务状态（queued/running/done/failed）；
  done 时 result 携带完整四段式报告（结构化 JSON＋渲染文本＋免责声明）；
- GET /health：存活检查＋当前管线名（真实管线另带装配事实，见 health_info）。

配置与装配：
- 管线经 create_app(pipeline=...) 注入（依赖倒置：create_app 缺省 Mock（测试用）；
  真实装配入口 = app/real_main.py（RealCheckupPipeline＋真实 LLM），实现同一
  CheckupPipeline 协议，路由零改动）；
- 任务表 = app/tasks.py 进程内实现（单 worker 串行＋TTL 1 小时；重启丢任务，
  轮询 404 后客户端重新提交）；R3 在提交端点上限流；
- 配置走环境变量 / config.json（不硬编码 key）；日志用 logging。

免责声明红线：渲染层保证其存在（gen.validator/渲染管线），worker 在任务完成时
复核哨兵，缺失即 failed（安全文案，不泄内部细节）。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from app.pipeline import CheckupPipeline, MockCheckupPipeline, PipelineResult
from app.tasks import STATUS_QUEUED, TaskStore
from gen.schema import DISCLAIMER_TEXT

logger = logging.getLogger(__name__)

# 免责声明哨兵＝完整产品级原文（渲染层注入的即此文本，逐字比对最严格）
DISCLAIMER_SENTINEL = DISCLAIMER_TEXT

# 失败安全文案（与任务 failed 状态一起返回；内部细节只落服务端日志）
_GENERIC_FAILURE = "体检执行失败，请稍后重试"
_SENTINEL_FAILURE = "服务内部校验失败"

#: 轮询预计耗时提示（UX 参考值，非承诺）：真实管线（CPU 重排）约两分钟，其余取缺省。
DEFAULT_ESTIMATED_SECONDS = 5


class CheckupRequest(BaseModel):
    """体检请求体：项目描述（简介／资金用途／建设内容）。"""

    description: str

    @field_validator("description")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("description 不能为空")
        if len(v.strip()) < 20:
            raise ValueError("description 过短（至少 20 字），请按「简介＋资金用途＋建设内容」补全")
        return v.strip()


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
    """体检结果（done 任务的 result 形状；同步契约时代即此形状，不变）。"""

    pipeline: str
    report: dict
    rendered_text: str
    degraded: bool
    attempts: int
    failed_sections: list[str]
    disclaimer: str


def _execute_checkup(pipe: CheckupPipeline, description: str) -> dict:
    """跑一次体检并构造结果 payload（worker 线程内执行；免责声明哨兵复核）。

    失败一律转成携带**安全文案**的 RuntimeError——TaskStore 把异常消息存进
    failed.error 返回给客户端，内部细节（路径/堆栈/key）只落服务端日志。
    """
    try:
        result: PipelineResult = pipe.run(description)
    except Exception:  # noqa: BLE001 —— 客户端只见安全文案（原同步 500 的 detail 口径）
        logger.exception("体检管线执行失败")
        raise RuntimeError(_GENERIC_FAILURE) from None
    if DISCLAIMER_SENTINEL not in result.rendered_text:
        logger.error("管线产物缺少免责声明，任务转失败")
        raise RuntimeError(_SENTINEL_FAILURE)
    return CheckupResponse(
        pipeline=pipe.name,
        report=result.to_api_dict()["report"],
        rendered_text=result.rendered_text,
        degraded=result.degraded,
        attempts=result.attempts,
        failed_sections=list(result.failed_sections),
        disclaimer=result.disclaimer,
    ).model_dump()


def create_app(pipeline: CheckupPipeline | None = None) -> FastAPI:
    """装配 FastAPI 实例（pipeline 缺省为 Mock——测试用；真实入口见 app/real_main.py）。"""
    pipe = pipeline or MockCheckupPipeline()
    store = TaskStore()
    app = FastAPI(
        title="粤策哨 · 申报前体检 API",
        version="0.3.0-async-tasks",
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
    def checkup(req: CheckupRequest) -> TaskSubmissionResponse:
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
