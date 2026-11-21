"""app/real_main.py — 真实装配入口（产品化 M1-R1，路线图 docs/产品化路线图-20261120.md）。

与评测同一套装配（app/real_components）＝线上口径与评测背书同源：R0 复测
（runs/hybrid-final-20261120）三指标全过，线上采用 hybrid 产线口径。

启动（factory 模式——管线装配加载索引与两个本地模型约 2–4 分钟，不在 import 期执行，
CI 导入本模块零副作用）：

    uvicorn app.real_main:create_real_app --factory --host 0.0.0.0 --port 8000

路由继承 app/main.py（M1-R2 起异步契约，M1-R3 起带限流与 AI 标识）：
``POST /api/checkup`` 提交体检任务（202＋task_id；每 IP 限流 5 次/时，
超频 429），``GET /api/tasks/{task_id}`` 轮询（done 携带四段式报告＋AI 生成
标识），``GET /health`` 含 index_chunks / hybrid_enabled / llm_backend 装配
事实。

拒启条件：LLM key 不到位即抛 RealAdapterError——Mock 报告对外服务没有实验意义
（红线⑤：demo 不虚构；服务端宁可起不来也不能带假装配上线）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from app.main import create_app
from app.real_components import (
    DEFAULT_CONFIG_PATH,
    MockLLM,
    QwenLLM,
    RealAdapterError,
    RealCheckupPipeline,
    build_llm,
)

logger = logging.getLogger(__name__)


def build_real_pipeline(
    config_path: Path | str | None = None,
    llm: MockLLM | QwenLLM | None = None,
) -> RealCheckupPipeline:
    """装配真实管线；LLM 未注入时按 config 装配，key 不到位**拒启**（见模块 docstring）。

    ``llm`` 参数仅测试注入通道（生产 uvicorn factory 不传；运行时不校验注入类型，
    非生产类型的行为由注入方负责）。
    """
    target = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if llm is None:
        llm = build_llm(target)
    if isinstance(llm, MockLLM) or llm is None:
        raise RealAdapterError(
            "真实服务需要 LLM key：请在 .env 配置 DASHSCOPE_API_KEY（或 config.json "
            "generation.api_key_env 指定的变量名）。MockLLM 只用于测试，不得对外服务。")
    pipeline = RealCheckupPipeline(target, llm=llm)
    logger.info(
        "真实管线装配完成：%s（index_chunks=%s, hybrid_enabled=%s）",
        pipeline.name,
        pipeline.health_info().get("index_chunks"),
        pipeline.health_info().get("hybrid_enabled"),
    )
    return pipeline


def create_real_app(
    config_path: Path | str | None = None,
    llm: MockLLM | QwenLLM | None = None,
) -> FastAPI:
    """装配真实服务实例（uvicorn --factory 入口；测试可注入 llm/config）。"""
    return create_app(pipeline=build_real_pipeline(config_path, llm))
