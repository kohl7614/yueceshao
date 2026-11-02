"""rag.indexer 的单元测试。

对应《开发手册 v1.0》§3.1 PR#3 验收标准：

1. 全量入库可断点续跑——本文件覆盖「续跑跳过已入库」「全量重建先删后建」两条机制，
   真实 719 篇全量跑批的输出见 PR 描述；
2. 抽 10 块元数据完整——``test_check_metadata_flags_missing_field`` 与 Chroma 集成用例。

编码器与向量库默认用假实现：CI 不装 sentence-transformers（权重 1.2GB），
装了 chromadb 时真实向量库用例自动启用，未装则跳过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rag.indexer import (
    AUDIT_CATEGORY,
    REQUIRED_METADATA,
    ChromaStore,
    ChunkRecord,
    IndexConfig,
    build_index,
    check_metadata,
    dump_chunks,
    format_query,
    load_chunks,
    scalarize_metadata,
    to_record,
)


# ================================================================ 测试替身


class FakeEmbedder:
    """确定性假编码器：记录每次批大小，可配置前若干次调用故意失败。"""

    def __init__(self, dim: int = 4, fail_times: int = 0) -> None:
        self.dim = dim
        self.batches: list[int] = []
        self.queries: list[str] = []
        self._remaining_failures = fail_times

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """返回按序号递增的固定向量，便于断言写入顺序。"""
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("故意失败：模拟编码器异常")
        self.batches.append(len(texts))
        return [[float(index + 1), 0.1, 0.2, 0.3][: self.dim] for index in range(len(texts))]

    def encode_query(self, text: str) -> list[float]:
        """记录查询串并返回固定向量。"""
        self.queries.append(text)
        return [1.0, 0.1, 0.2, 0.3][: self.dim]


class FakeStore:
    """内存假向量库，方法面与 :class:`~rag.indexer.ChromaStore` 对齐。"""

    def __init__(self) -> None:
        self.rows: dict[str, ChunkRecord] = {}
        self.vectors: dict[str, list[float]] = {}
        self.reset_called = 0

    def count(self) -> int:
        """已入库块数。"""
        return len(self.rows)

    def ids(self) -> set[str]:
        """已入库的 chunk_id。"""
        return set(self.rows)

    def upsert(self, records: list[ChunkRecord], vectors: list[list[float]]) -> None:
        """按 chunk_id 幂等写入。"""
        assert len(records) == len(vectors), "记录数与向量数必须一致"
        for record, vector in zip(records, vectors):
            self.rows[record.chunk_id] = record
            self.vectors[record.chunk_id] = list(vector)

    def sample(self, limit: int = 10) -> list[tuple[str, dict[str, Any]]]:
        """取前 N 块元数据。"""
        return [(key, dict(value.metadata)) for key, value in list(self.rows.items())[:limit]]

    def reset(self) -> None:
        """清空。"""
        self.rows.clear()
        self.vectors.clear()
        self.reset_called += 1


@dataclass
class ChunkLike:
    """模仿 chunker 的 Chunk：验证鸭子类型取值与 ``as_metadata()`` 约定。"""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        """chunker 侧的真实接口。"""
        return dict(self.metadata)


def make_record(chunk_id: str = "1-0000", **overrides: Any) -> dict[str, Any]:
    """构造一条最小可用的 chunk 记录（文号取自数据仓真实条目）。"""
    metadata: dict[str, Any] = {
        "chunk_id": chunk_id,
        "serial": 1,
        "doc_no": "粤财农〔2020〕106号",
        "title": "广东省涉农资金统筹整合管理办法",
        "structure": "clause",
        "chapter": "第二章 资金使用",
        "section": "",
        "clause": "第十条",
        "clause_index": 10,
        "part": 1,
        "parts_total": 1,
        "category": "资金管理办法",
        "validity": "有效",
        "is_current": True,
        "has_doc_no": True,
        "publish_date": "2020-06-01",
    }
    metadata.update(overrides)
    return {"chunk_id": chunk_id, "text": f"第十条 正文（{chunk_id}）", "metadata": metadata}


# ================================================================ 记录归一化


def test_to_record_from_mapping() -> None:
    """映射形态（JSONL 行）可直接归一化。"""
    record = to_record(make_record("7-0003"))
    assert record.chunk_id == "7-0003"
    assert record.metadata["doc_no"] == "粤财农〔2020〕106号"


def test_to_record_from_chunk_like_object() -> None:
    """chunker 的 Chunk 对象按属性取值，走 ``as_metadata()``。"""
    chunk = ChunkLike(chunk_id="9-0001", text="第九条 正文", metadata={"category": "申报指南"})
    record = to_record(chunk)
    assert record.chunk_id == "9-0001"
    assert record.metadata["category"] == "申报指南"
    assert record.metadata["chunk_id"] == "9-0001"


def test_to_record_rejects_missing_id() -> None:
    """缺 chunk_id 直接拒绝，避免脏数据拿到随机 id。"""
    with pytest.raises(ValueError, match="chunk_id"):
        to_record({"text": "第十条 正文", "metadata": {}})


def test_to_record_rejects_empty_text() -> None:
    """空块不占检索名额。"""
    with pytest.raises(ValueError, match="入库文本为空"):
        to_record({"chunk_id": "1-0000", "text": "   ", "metadata": {}})


def test_metadata_scalarized_to_chroma_types() -> None:
    """None 转空串、非标量丢弃、bool 不当 int 处理。"""
    flattened = scalarize_metadata(
        {
            "clause": None,
            "is_current": False,
            "serial": 12,
            "score": 0.5,
            "keywords": ["帮扶", "资金"],
            "extra": {"a": 1},
        }
    )
    assert flattened == {
        "clause": "",
        "is_current": False,
        "serial": 12,
        "score": 0.5,
    }
    assert isinstance(flattened["is_current"], bool)


# ================================================================ 入库主流程


def test_build_index_in_batches() -> None:
    """按 batch_size 分批编码，块数与向量一一对应。"""
    embedder = FakeEmbedder()
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(5)]

    report = build_index(
        records, embedder=embedder, store=store, config=IndexConfig(batch_size=2)
    )

    assert report.indexed == 5
    assert report.failed == 0
    assert embedder.batches == [2, 2, 1], "应按 batch_size 切批"
    assert store.count() == 5
    assert report.collection_count == 5


def test_resume_skips_indexed_chunks() -> None:
    """断点续跑：重复执行不重复编码已入库块。"""
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(3)]

    first = build_index(records, embedder=FakeEmbedder(), store=store)
    second = build_index(records, embedder=FakeEmbedder(), store=store)

    assert first.indexed == 3
    assert second.indexed == 0
    assert second.skipped_existing == 3
    assert store.count() == 3, "续跑不得产生副本"


def test_no_resume_reindexes_everything() -> None:
    """``resume=False`` 时全量重写（配合 --full-rebuild）。"""
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(3)]

    build_index(records, embedder=FakeEmbedder(), store=store)
    again = build_index(records, embedder=FakeEmbedder(), store=store, resume=False)

    assert again.indexed == 3
    assert again.skipped_existing == 0
    assert store.count() == 3


def test_duplicate_chunk_id_counted_once() -> None:
    """同一批里出现重复 chunk_id 只入一次。"""
    store = FakeStore()
    records = [make_record("1-0000"), make_record("1-0000")]

    report = build_index(records, embedder=FakeEmbedder(), store=store)

    assert report.indexed == 1
    assert report.skipped_existing == 1


def test_audit_category_blocked() -> None:
    """红线④：审计公告一律拒绝入库，且不送去编码。"""
    embedder = FakeEmbedder()
    store = FakeStore()
    records = [
        make_record("1-0000", category=AUDIT_CATEGORY),
        make_record("2-0000"),
    ]

    report = build_index(records, embedder=embedder, store=store)

    assert report.skipped_audit == 1
    assert report.indexed == 1
    assert "1-0000" not in store.rows
    assert embedder.batches == [1], "审计公告不应进入编码环节"


def test_batch_failure_does_not_stop_run() -> None:
    """单批失败只计失败数，后续批次继续。"""
    embedder = FakeEmbedder(fail_times=1)
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(4)]

    report = build_index(
        records, embedder=embedder, store=store, config=IndexConfig(batch_size=2)
    )

    assert report.failed == 2
    assert report.indexed == 2
    assert store.count() == 2


# ================================================================ 元数据抽检


def test_check_metadata_passes_for_complete_chunks() -> None:
    """10 块元数据齐全 → 抽检通过（验收项）。"""
    store = FakeStore()
    records = [make_record(f"1-{index:04d}") for index in range(10)]
    build_index(records, embedder=FakeEmbedder(), store=store)

    audit = check_metadata(store, sample=10)

    assert audit.sampled == 10
    assert audit.ok, audit.summary()


def test_check_metadata_flags_missing_field() -> None:
    """缺 ``validity`` 必须被抓出来——时效过滤靠它，缺了就静默失效。"""
    store = FakeStore()
    record = make_record("1-0001")
    record["metadata"].pop("validity")
    build_index([record], embedder=FakeEmbedder(), store=store)

    audit = check_metadata(store, sample=10)

    assert not audit.ok
    assert "validity" in audit.problems[0]


def test_required_metadata_matches_chunker_fields() -> None:
    """必备字段清单不得漏掉 chunker 实际产出的关键字段。"""
    assert set(REQUIRED_METADATA) <= set(make_record()["metadata"])
    assert "is_current" in REQUIRED_METADATA


# ================================================================ JSONL


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """dump → load 后内容一致，元数据仍是扁平标量。"""
    target = tmp_path / "chunks.jsonl"
    chunks = [
        ChunkLike(chunk_id="1-0000", text="第十条 正文", metadata={"is_current": True}),
        make_record("2-0001"),
    ]

    written = dump_chunks(chunks, target)
    loaded = list(load_chunks(target))

    assert written == 2
    assert [record.chunk_id for record in loaded] == ["1-0000", "2-0001"]
    assert loaded[0].metadata["is_current"] is True
    first_line = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    assert set(first_line) == {"chunk_id", "text", "metadata"}


def test_load_chunks_skips_broken_line(tmp_path: Path) -> None:
    """坏行跳过，不中断整批。"""
    target = tmp_path / "chunks.jsonl"
    target.write_text(
        json.dumps(make_record("1-0000"), ensure_ascii=False)
        + "\n{坏行\n"
        + json.dumps(make_record("1-0001"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    assert [record.chunk_id for record in load_chunks(target)] == ["1-0000", "1-0001"]


# ================================================================ 查询编码


def test_format_query_follows_model_card() -> None:
    """Qwen3-Embedding 的查询串格式：Instruct 行 + Query: 前缀。"""
    text = format_query("村里想申请产业园建设资金")
    assert text.startswith("Instruct: ")
    assert "\nQuery:村里想申请产业园建设资金" in text


def test_qwen_embedder_is_lazy() -> None:
    """导入与构造不加载权重（CI 没装 sentence-transformers 也能跑到这里）。"""
    from rag.indexer import EmbedderConfig, Qwen3Embedder

    embedder = Qwen3Embedder(EmbedderConfig(model_dir="models/不存在的目录"))
    assert embedder._model is None  # noqa: SLF001 - 断言懒加载状态


# ================================================================ Chroma 集成


def test_chroma_store_roundtrip(tmp_path: Path) -> None:
    """真实 Chroma：写入、where 时效过滤、抽检、重置。"""
    chromadb = pytest.importorskip("chromadb")
    config = IndexConfig(persist_dir=tmp_path / "chroma", collection="test_clauses")
    store = ChromaStore(config, client=chromadb.EphemeralClient())
    records = [
        make_record("1-0000", is_current=True),
        make_record("1-0001", is_current=False, validity="已失效"),
        make_record("1-0002", is_current=True),
    ]

    report = build_index(records, embedder=FakeEmbedder(), store=store, config=config)

    assert report.indexed == 3
    assert store.count() == 3
    assert store.ids() == {"1-0000", "1-0001", "1-0002"}

    current = store.collection.get(where={"is_current": True}, include=["metadatas"])
    assert set(current["ids"]) == {"1-0000", "1-0002"}, "A6 时效硬过滤依赖 is_current 元数据"
    assert check_metadata(store, sample=10).ok

    store.reset()
    assert store.count() == 0
