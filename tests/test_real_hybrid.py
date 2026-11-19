"""tests/test_real_hybrid.py — A5 混合检索接入管线的适配层回归。

契约：``_HybridPipelineRetriever.retrieve`` 透传 FusedHit.hit（chunk_id/metadata/
text 契约下游零改动）；时效消融变体共享同一份 BM25 索引；空集合在 BM25 构建
前就被拒（RealAdapterError，与既有错误信息一致）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.real_components import (
    RealAdapterError,
    _build_hybrid,
    _HybridPipelineRetriever,
)

from eval.dataset import EvalCase
from rag.hybrid import BM25Index, HybridConfig
from rag.retriever import Hit, RetrievalResult

_CORPUS = [
    {
        "chunk_id": "1-1",
        "text": "涉农资金统筹整合管理办法：项目单位应当专账核算，严格执行财政资金使用规定。",
        "metadata": {"doc_no": "粤测农〔2099〕1号", "is_current": True,
                     "category": "资金管理办法"},
    },
    {
        "chunk_id": "2-1",
        "text": "粮食烘干设施建设项目申报指南：财政资金可用于购置稻谷烘干机及配套设施。",
        "metadata": {"doc_no": "粤测农〔2099〕2号", "is_current": True,
                     "category": "申报指南"},
    },
    {
        "chunk_id": "3-1",
        "text": "合作社申报材料清单：营业执照、项目用地证明、自筹资金承诺函。",
        "metadata": {"doc_no": "粤测农〔2099〕3号", "is_current": True,
                     "category": "申报指南"},
    },
]


class _FakeDense:
    """稠密一路替身：固定把查询编码成向量、按余弦返回固定排序。"""

    def __init__(self) -> None:
        self.config = HybridConfig().enable_temporal_filter and object()

    @property
    def backend(self) -> _FakeBackend:
        return _FakeBackend()

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        # hybrid 侧会显式传 top_k/where/count_candidates——替身按固定排序返回。
        hits = [
            Hit(chunk_id="3-1", text=_CORPUS[2]["text"], score=0.9,
                distance=0.1, metadata=_CORPUS[2]["metadata"]),
            Hit(chunk_id="2-1", text=_CORPUS[1]["text"], score=0.8,
                distance=0.2, metadata=_CORPUS[1]["metadata"]),
        ]
        return RetrievalResult(query=query, hits=hits)


class _FakeBackend:
    """Chroma 后端替身：语料来自 _CORPUS（验证 iter 协议路径）。"""

    def count(self) -> int:
        return len(_CORPUS)

    def iter_corpus_records(self, page_size: int = 2):
        yield from _CORPUS


def _hybrid() -> _HybridPipelineRetriever:
    return _build_hybrid(
        {"retrieval": {}, "hybrid": {"top_k": 3, "dense_k": 3, "bm25_k": 3}},
        temporal=True, sparse=None, dense=_FakeDense())


def test_adapter_passthrough_hit_contract() -> None:
    """检索契约不变：retrieve 返回 RetrievalResult，hits 是 Hit 且元数据齐全。"""
    adapter = _hybrid()
    result = adapter.retrieve("烘干设施建设的申报条件")
    assert isinstance(result, RetrievalResult)
    assert result.hits and all(isinstance(h, Hit) for h in result.hits)
    ids = {h.chunk_id for h in result.hits}
    assert "2-1" in ids  # 稠密一路的命中出现在融合结果里
    for hit in result.hits:
        assert hit.metadata.get("doc_no")


def test_rebuild_shares_sparse() -> None:
    """时效消融变体：rebuild 复用同一份 BM25 索引对象（不二次全量分词）。"""
    adapter = _hybrid()
    plain = adapter.rebuild(temporal=False)
    assert plain is not adapter
    assert plain._sparse is adapter._sparse
    assert plain._hybrid.config.enable_temporal_filter is False


def test_build_plain_retriever_branch(monkeypatch) -> None:
    """RealComponents._build_plain_retriever 对混合适配器走 rebuild 分支。"""
    from app.real_components import RealComponents

    adapter = _hybrid()
    fake_self = SimpleNamespace(_retriever=adapter)
    plain = RealComponents._build_plain_retriever(fake_self, None)
    assert plain._sparse is adapter._sparse


def test_empty_collection_rejected_before_bm25() -> None:
    """空集合在 BM25 构建前被拒（避免 BM25Okapi 除零），错误信息与既有口径一致。"""

    class _EmptyBackend:
        def count(self) -> int:
            return 0

    class _EmptyDense:
        backend = _EmptyBackend()

        def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(query="", hits=[])

    with pytest.raises(RealAdapterError, match="检索集合为空"):
        _build_hybrid(
            {"retrieval": {}}, temporal=True, sparse=None, dense=_EmptyDense())


def test_injected_sparse_reused() -> None:
    """显式传入的 BM25 索引被原样复用（不重复构建）。"""
    sparse = BM25Index(_CORPUS)
    adapter = _build_hybrid(
        {"retrieval": {}, "hybrid": {"top_k": 3}}, temporal=True,
        sparse=sparse, dense=_FakeDense())
    assert adapter._sparse is sparse
    # 兜底冒烟：EvalCase→查询路径的文本契约（query 非 None）
    case = EvalCase(case_id="T1", query="烘干设施")
    assert case.query


# ==================== A7 质量门接线 ====================

class _FakeGate:
    """门控替身：构造时给定 passed，记录收到的候选数。"""

    def __init__(self, passed: bool) -> None:
        self.passed = passed
        self.seen_candidates: list[int] = []
        self.reason = "test"
        self.tau = 0.55
        self.top_score = 0.9 if passed else 0.1

    def evaluate(self, result) -> SimpleNamespace:
        self.seen_candidates.append(len(result.hits))
        return SimpleNamespace(passed=self.passed, reason=self.reason,
                               tau=self.tau, top_score=self.top_score)


class _FakeGateRunner:
    """GateRunner 替身：fetch(n) 收 n 条候选，按 passed 决定门控结果。"""

    def __init__(self, passed: bool) -> None:
        self._passed = passed
        self.gate = _FakeGate(passed)
        self.requested = None  # fetch 被请求的候选数

    def run(self, query: str, fetch, top_k=None):
        self.requested = 3
        hits = fetch(self.requested)
        gate = self.gate.evaluate(SimpleNamespace(hits=hits))
        # 真契约：GateRunner 返回的 hits 是 RerankedHit（适配层读 .hit 取原命中）
        ranked = [SimpleNamespace(hit=h) for h in hits]
        return SimpleNamespace(hits=ranked, gate=gate)


def test_gate_pass_returns_reranked_hits() -> None:
    """门过 → 命中为重排后的 hit（fetch 候选数正确）。"""
    adapter = _hybrid()
    runner = _FakeGateRunner(passed=True)
    adapter._gate_runner = runner
    result = adapter.retrieve("烘干设施建设的申报条件")
    # fetch 被请求 3 条候选；融合并集不足 3 时按实际并集数返回（上限语义）
    assert runner.requested == 3
    assert len(runner.gate.seen_candidates) == 1
    assert 0 < runner.gate.seen_candidates[0] <= 3
    assert result.hits and all(isinstance(h, Hit) for h in result.hits)


def test_gate_fail_degrades_to_uncovered() -> None:
    """τ 门不过 → A11 诚实降级：空命中（①段渲染为未覆盖），绝不输出低置信候选。"""
    adapter = _hybrid()
    adapter._gate_runner = _FakeGateRunner(passed=False)
    result = adapter.retrieve("烘干设施建设的申报条件")
    assert result.hits == []


def test_rebuild_carries_gate_runner() -> None:
    """时效消融变体继承同一份门控（重排模型只加载一次）。"""
    adapter = _hybrid()
    runner = _FakeGateRunner(passed=True)
    adapter._gate_runner = runner
    plain = adapter.rebuild(temporal=False)
    assert plain._gate_runner is runner


def test_build_gate_runner_section_rules() -> None:
    """无 reranker 段 / enable_gate=false → None（纯 A5 路径）；有段则装配门控。"""
    from app.real_components import _build_gate_runner
    assert _build_gate_runner({"retrieval": {}}) is None
    assert _build_gate_runner(
        {"reranker": {"enable_gate": False, "tau": 0.55}}) is None
    runner = _build_gate_runner(
        {"reranker": {"enable_gate": True, "tau": 0.55,
                      "model_dir": "models/bge-reranker-v2-m3"}})
    assert runner is not None  # 懒加载：构造不触发模型下载/加载
