"""tests/test_real_main.py — R1 真实装配入口与检索口径开关。

契约：``retrieval.hybrid_enabled=false`` 走纯稠密回退（pre-#91 路径，final-v3
基线口径，R0 回退闸门的实现钩子）；默认/显式 true 走 A5 混合装配；真实管线
``health_info`` 如实上报装配事实；服务入口在 LLM 不到位的场合**拒启**（Mock
不得对外服务，红线⑤）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.main import MockCheckupPipeline, create_app
from app.real_components import (
    MockLLM,
    RealAdapterError,
    RealCheckupPipeline,
    _build_retriever,
)
from app.real_main import build_real_pipeline
from fastapi.testclient import TestClient

# ==================== hybrid_enabled 检索口径开关 ====================

def test_build_retriever_dense_fallback(monkeypatch) -> None:
    """hybrid_enabled=false → 纯稠密（build_retriever 原样，pre-#91 路径）。"""
    sentinel = SimpleNamespace(marker="dense")
    monkeypatch.setattr("rag.retriever.build_retriever", lambda cfg: sentinel)
    config = {"retrieval": {"hybrid_enabled": False}}
    assert _build_retriever(config) is sentinel


def test_build_retriever_hybrid_default_and_explicit(monkeypatch) -> None:
    """缺键与显式 true 都走混合装配；开关从 retrieval 段读取。"""
    marker = SimpleNamespace(marker="hybrid")
    seen: list[dict] = []

    def _fake_build_hybrid(config, *, temporal):
        seen.append(config)
        return marker

    monkeypatch.setattr("app.real_components._build_hybrid", _fake_build_hybrid)
    assert _build_retriever({"retrieval": {}}) is marker  # 缺键＝默认 true
    assert _build_retriever(
        {"retrieval": {"hybrid_enabled": True}}) is marker  # 显式 true
    assert len(seen) == 2


# ==================== health_info 装配事实 ====================

def _pipeline_with_components(components: SimpleNamespace) -> RealCheckupPipeline:
    """注入组件构造真实管线（不触发真实装配；CI 无索引/模型可跑）。"""
    return RealCheckupPipeline(components=components)  # type: ignore[arg-type]


def test_health_info_reports_facts() -> None:
    """index_chunks/hybrid_enabled/llm_backend 与装配事实一致。"""
    components = SimpleNamespace(
        _retriever=SimpleNamespace(backend=SimpleNamespace(count=lambda: 20099)),
        _config={"retrieval": {"hybrid_enabled": False}},
        llm_backend="QwenLLM",
    )
    info = _pipeline_with_components(components).health_info()
    assert info == {
        "index_chunks": 20099,
        "hybrid_enabled": False,
        "llm_backend": "QwenLLM",
    }


def test_health_info_default_hybrid_true() -> None:
    """配置缺 hybrid_enabled 键时如实报 true（默认口径＝A5 混合）。"""
    components = SimpleNamespace(
        _retriever=SimpleNamespace(backend=SimpleNamespace(count=lambda: 3)),
        _config={"retrieval": {}},
        llm_backend="MockLLM",
    )
    assert _pipeline_with_components(components).health_info()["hybrid_enabled"] is True


# ==================== /health 路由合并 ====================

class _ServingStub:
    """带 health_info 的管线替身（模拟真实管线的服务外观）。"""

    name = "real-retrieval-qwen-llm"

    def health_info(self) -> dict:
        return {"index_chunks": 7, "hybrid_enabled": True, "llm_backend": "QwenLLM"}

    def run(self, description: str):  # pragma: no cover - 本测试不触发体检
        raise AssertionError("本测试不应触发体检")


def test_health_merges_pipeline_facts() -> None:
    """/health 合并管线携带的装配事实（真实管线的服务外观）。"""
    client = TestClient(create_app(pipeline=_ServingStub()))  # type: ignore[arg-type]
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pipeline"] == "real-retrieval-qwen-llm"
    assert body["index_chunks"] == 7
    assert body["hybrid_enabled"] is True


def test_health_mock_keeps_skeleton_shape() -> None:
    """Mock 管线无 health_info：/health 保持骨架形状，不硬造字段。"""
    client = TestClient(create_app(pipeline=MockCheckupPipeline()))
    body = client.get("/health").json()
    assert body == {"status": "ok", "pipeline": "mock-skeleton"}


# ==================== 服务入口拒启条件 ====================

def test_build_real_pipeline_rejects_mock_llm() -> None:
    """显式注入 MockLLM → 拒启（Mock 报告不得对外服务）。"""
    with pytest.raises(RealAdapterError, match="MockLLM 只用于测试"):
        build_real_pipeline(llm=MockLLM())


def test_build_real_pipeline_rejects_missing_key(monkeypatch) -> None:
    """key 不到位（build_llm 返回 None）→ 拒启，错误信息指路 .env。"""
    monkeypatch.setattr("app.real_main.build_llm", lambda _target: None)
    with pytest.raises(RealAdapterError, match="DASHSCOPE_API_KEY"):
        build_real_pipeline()


def test_build_real_pipeline_passes_injected_llm(monkeypatch) -> None:
    """注入非 Mock LLM → 透传给 RealCheckupPipeline（重装配被替换，测试轻量）。"""
    fake_llm = SimpleNamespace(marker="fake-llm")
    captured: dict = {}

    class _FakePipeline:
        name = "fake-real"

        def __init__(self, target, *, llm) -> None:
            captured["target"] = target
            captured["llm"] = llm

        def health_info(self) -> dict:
            return {"index_chunks": 1, "hybrid_enabled": True}

    monkeypatch.setattr("app.real_main.RealCheckupPipeline", _FakePipeline)
    build_real_pipeline(config_path="/tmp/x.json", llm=fake_llm)  # type: ignore[arg-type]
    assert captured["llm"] is fake_llm
