"""粤策哨 · Chroma 入库与 embedding 封装。

职责
----
把 chunker 产出的 chunk 写进**带 where 元数据的 Chroma 集合**，并封装本地
embedding（Qwen3-Embedding-0.6B）。对应《粤策哨开发手册 v1.0》§3.1 PR-3（A13）：

- 入库：chunk 原文进 ``documents``、向量进 ``embeddings``、可过滤字段进 ``metadatas``；
- 封装：文档侧与查询侧分开编码（查询侧带 Qwen3 的指令前缀），权重首次编码时才加载；
- 模式：增量（幂等 upsert＋跳过已入库 id，可断点续跑）与全量重建（先删集合再写）；
- 红线④兜底：``category == 审计公告`` 的 chunk 一律拒绝入库。loader 已过滤一次，
  索引层必须再挡一次——审计公告是评测出卷集，进索引就等于答案泄露，这一条不能
  只靠上游守着。

模块边界
--------
输入是 **chunk 记录**：chunker 的 ``Chunk`` 对象（鸭子类型取属性），或 JSONL 里的
``{"chunk_id", "text", "metadata"}``。因此本模块不 import ``rag.chunker``，
与分块器实现解耦，可独立合并、独立测试。

为什么元数据必须扁平化
----------------------
Chroma 的 ``metadatas`` 只接受 ``str/int/float/bool`` 标量：``None``、列表、字典都会
抛错，且不同版本对 ``None`` 的处理不一致。故入库前统一收敛（:func:`scalarize_metadata`）：
空值转空串，非标量丢弃——这样 retriever 才能稳定地写 ``where={"is_current": True}``
做时效硬过滤（A6）。
"""

import argparse
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

AUDIT_CATEGORY = "审计公告"
"""红线④：该类别的 chunk 永不进索引。"""

REQUIRED_METADATA: tuple[str, ...] = (
    "chunk_id",
    "serial",
    "doc_no",
    "title",
    "structure",
    "clause",
    "clause_index",
    "part",
    "parts_total",
    "category",
    "validity",
    "is_current",
    "has_doc_no",
)
"""验收用的必备元数据字段（抽检对照表）；与 chunker 的 ``as_metadata()`` 对齐。"""

DEFAULT_QUERY_INSTRUCTION = "给定一个涉农资金申报项目描述，检索适用于该项目的政策条款"
"""Qwen3-Embedding 的查询指令（非对称检索的任务描述）。"""


# ================================================================ 记录与元数据


def format_query(text: str, instruction: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    """按 Qwen3-Embedding 模型卡约定拼查询串。

    格式为 ``Instruct: {任务描述}\\nQuery:{问题}``。文档侧不加前缀——指令的作用是把
    「用户问题」映射到「文档检索」这一个任务上，加在文档侧只会污染向量。

    :param text: 用户查询
    :param instruction: 任务描述
    :return: 可直接送进模型的字符串
    """
    return f"Instruct: {instruction}\nQuery:{text}"


def scalarize_metadata(
    metadata: Mapping[str, Any], *, chunk_id: str = ""
) -> dict[str, Any]:
    """把元数据收敛成 Chroma 可接受的扁平标量字典。

    规则：``None`` → ``""``；``bool/int/float/str`` 原样保留；列表、字典、集合及其他
    对象直接丢弃（记 debug 日志）。丢弃而非字符串化，是为了避免 retriever 拿到
    ``"[1, 2]"`` 这种既不能等值也不能范围比较的假值。

    :param metadata: 原始元数据（chunker 的 ``as_metadata()`` 或 JSONL 的 metadata 段）
    :param chunk_id: 仅用于日志定位
    :return: 新字典，不修改入参
    """
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        name = str(key)
        if value is None:
            result[name] = ""
        elif isinstance(value, bool):  # 必须在 int 之前判断：bool 是 int 的子类
            result[name] = value
        elif isinstance(value, (int, float, str)):
            result[name] = value
        else:
            logger.debug(
                "chunk %s 元数据 %s 非标量（%s），入库时丢弃",
                chunk_id or "?",
                name,
                type(value).__name__,
            )
    return result


@dataclass(frozen=True)
class ChunkRecord:
    """入库单元：一个 chunk 的标识、入库文本与扁平元数据。"""

    chunk_id: str
    """稳定标识，来自 chunker（``{序号}-{顺序号}``）。"""

    text: str
    """实际送去编码的文本（已含层级头）。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """where 过滤用的扁平元数据。"""


def to_record(raw: ChunkRecord | Mapping[str, Any] | Any) -> ChunkRecord:
    """把各种形态的 chunk 统一成 :class:`ChunkRecord`。

    兼容三种输入：

    1. 已经是 :class:`ChunkRecord`；
    2. 映射（JSONL 行或手工构造的 dict），取 ``chunk_id`` / ``text`` / ``metadata``；
    3. chunker 的 ``Chunk`` 对象，取属性 ``chunk_id`` / ``text`` 与 ``as_metadata()``。

    :param raw: 待归一化的 chunk
    :return: 元数据已扁平化的入库单元
    :raises ValueError: 缺 ``chunk_id``，或入库文本为空（空块不占检索名额）
    """
    if isinstance(raw, ChunkRecord):
        chunk_id, text, metadata = raw.chunk_id, raw.text, dict(raw.metadata)
    elif isinstance(raw, Mapping):
        section = raw.get("metadata")
        metadata = dict(section) if isinstance(section, Mapping) else {}
        chunk_id = str(raw.get("chunk_id") or metadata.get("chunk_id") or "")
        text = str(raw.get("text") or raw.get("body") or "")
    else:
        chunk_id = str(getattr(raw, "chunk_id", "") or "")
        text = str(getattr(raw, "text", "") or getattr(raw, "body", "") or "")
        as_metadata = getattr(raw, "as_metadata", None)
        metadata = dict(as_metadata()) if callable(as_metadata) else {}

    if not chunk_id:
        raise ValueError("chunk 缺少 chunk_id，无法入库")
    if not text.strip():
        raise ValueError(f"chunk {chunk_id} 入库文本为空")

    flattened = scalarize_metadata(metadata, chunk_id=chunk_id)
    flattened.setdefault("chunk_id", chunk_id)
    return ChunkRecord(chunk_id=chunk_id, text=text, metadata=flattened)


# ================================================================ 配置


@dataclass(frozen=True)
class EmbedderConfig:
    """embedding 参数，对应 ``config.example.json`` 的 ``embedding`` 段。"""

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    model_dir: str = "models/Qwen3-Embedding-0.6B"
    device: str = "cpu"
    batch_size: int = 16
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> "EmbedderConfig":
        """从整份配置的 ``embedding`` 段构造；缺字段回落到默认值。"""
        section = _section(raw, "embedding")
        return cls(
            model_name=str(section.get("model_name") or cls.model_name),
            model_dir=str(section.get("model_dir") or cls.model_dir),
            device=str(section.get("device") or cls.device),
            batch_size=int(section.get("batch_size") or cls.batch_size),
            query_instruction=str(section.get("query_instruction") or cls.query_instruction),
        )


@dataclass(frozen=True)
class IndexConfig:
    """入库参数，对应 ``config.example.json`` 的 ``index`` 段。"""

    persist_dir: Path = Path(".chroma")
    collection: str = "yueceshao_clauses"
    distance: str = "cosine"
    batch_size: int = 16
    embedding: EmbedderConfig = field(default_factory=EmbedderConfig)

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> "IndexConfig":
        """从整份配置构造；``index`` 段缺字段回落到默认值。"""
        section = _section(raw, "index")
        embedding = EmbedderConfig.from_config(raw)
        return cls(
            persist_dir=Path(str(section.get("persist_dir") or ".chroma")),
            collection=str(section.get("collection") or "yueceshao_clauses"),
            distance=str(section.get("distance") or "cosine"),
            batch_size=embedding.batch_size,
            embedding=embedding,
        )


def _section(raw: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    """安全取配置子段：缺失或类型不对时返回空映射。"""
    if not isinstance(raw, Mapping):
        return {}
    section = raw.get(key)
    return section if isinstance(section, Mapping) else {}


# ================================================================ embedding


@runtime_checkable
class Embedder(Protocol):
    """embedding 后端协议。"""

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文档（入库侧）。"""
        ...

    def encode_query(self, text: str) -> list[float]:
        """编码单条查询（检索侧，带指令前缀）。"""
        ...


class Qwen3Embedder:
    """Qwen3-Embedding-0.6B 的 sentence-transformers 封装。

    依赖与权重都是**懒加载**：``import rag.indexer`` 不需要 torch，只有真正编码时才加载，
    所以 CI（不装 sentence-transformers）也能跑本模块的测试。

    模型路径优先取 ``model_dir``（团队约定：权重放本地 ``models/`` 且不入库）；
    该目录不存在时退回 ``model_name``，交给 sentence-transformers 走缓存/Hub 解析。

    关于指令前缀：权重自带的 ``config_sentence_transformers.json`` 里有一条英文通用
    query 指令（"Given a web search query…"），本模块**不用它**——编码时不传
    ``prompt_name``，改为在 :meth:`encode_query` 里自己拼中文领域指令，避免两条指令叠加，
    也便于把任务描述写进配置做消融。
    """

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self._model: Any = None

    @property
    def model(self) -> Any:
        """懒加载的 SentenceTransformer 实例。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # 延迟导入，CI 不装

            source = (
                self.config.model_dir
                if Path(self.config.model_dir).is_dir()
                else self.config.model_name
            )
            logger.info("加载 embedding 模型：%s（device=%s）", source, self.config.device)
            kwargs: dict[str, Any] = {}
            if self.config.device == "cpu":
                # 模型卡权重是 bf16；CPU 上 bf16 反而慢，显式用 float32
                import torch

                kwargs["model_kwargs"] = {"dtype": torch.float32}
            self._model = SentenceTransformer(source, device=self.config.device, **kwargs)
        return self._model

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码，返回归一化向量（与集合的 cosine 距离配套）。"""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def encode_query(self, text: str) -> list[float]:
        """编码查询：加指令前缀后编码。"""
        prefixed = format_query(text, self.config.query_instruction)
        return self.encode_documents([prefixed])[0]


# ================================================================ 向量库


@runtime_checkable
class VectorStore(Protocol):
    """向量库后端协议；测试用内存实现替换，不必装 chromadb。"""

    def count(self) -> int:
        """集合内块数。"""
        ...

    def ids(self) -> set[str]:
        """集合内已有的 chunk_id。"""
        ...

    def upsert(self, records: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        """按 chunk_id 幂等写入。"""
        ...

    def sample(self, limit: int = 10) -> list[tuple[str, dict[str, Any]]]:
        """取前 ``limit`` 块（id, 元数据）用于抽检。"""
        ...

    def reset(self) -> None:
        """清空集合。"""
        ...


class ChromaStore:
    """Chroma 集合的薄封装（实现 :class:`VectorStore` 协议）。

    - 懒加载 ``chromadb``：导入本模块不要求装向量库；
    - 集合按配置的距离度量创建（默认 cosine，与归一化向量配套）；
    - :meth:`ids` 分页读取，供断点续跑跳过已入库的块。
    """

    PAGE_SIZE = 5000
    """``ids()`` 单次拉取的条数，避免大集合一次性进内存。"""

    def __init__(self, config: IndexConfig | None = None, client: Any = None) -> None:
        self.config = config or IndexConfig()
        self._client = client
        self._collection: Any = None

    @property
    def client(self) -> Any:
        """懒加载的 Chroma 客户端（默认持久化到 ``persist_dir``）。"""
        if self._client is None:
            import chromadb  # 延迟导入

            self.config.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.config.persist_dir))
        return self._client

    @property
    def collection(self) -> Any:
        """懒加载的集合句柄。"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.config.collection,
                metadata={"hnsw:space": self.config.distance},
            )
        return self._collection

    def count(self) -> int:
        """集合内块数。"""
        return int(self.collection.count())

    def ids(self) -> set[str]:
        """集合内全部 chunk_id（分页读取）。"""
        found: set[str] = set()
        offset = 0
        while True:
            page = self.collection.get(
                limit=self.PAGE_SIZE, offset=offset, include=[]
            )
            ids = page.get("ids") or []
            if not ids:
                return found
            found.update(str(item) for item in ids)
            if len(ids) < self.PAGE_SIZE:
                return found
            offset += len(ids)

    def upsert(self, records: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        """按 chunk_id 幂等写入（重复写入不产生副本）。"""
        if not records:
            return
        self.collection.upsert(
            ids=[record.chunk_id for record in records],
            embeddings=[list(vector) for vector in vectors],
            documents=[record.text for record in records],
            metadatas=[record.metadata for record in records],
        )

    def sample(self, limit: int = 10) -> list[tuple[str, dict[str, Any]]]:
        """取前 ``limit`` 块用于元数据抽检。"""
        page = self.collection.get(limit=limit, include=["metadatas"])
        ids = page.get("ids") or []
        metadatas = page.get("metadatas") or []
        return [
            (str(chunk_id), dict(metadata or {}))
            for chunk_id, metadata in zip(ids, metadatas)
        ]

    def reset(self) -> None:
        """删除并重建集合（全量重建用）。

        集合不存在时直接跳过：各版本 chromadb 抛出的异常类型不一致，这里按「删不掉
        就重建」处理，随后立即重建集合即可。
        """
        try:
            self.client.delete_collection(self.config.collection)
        except Exception as error:  # noqa: BLE001 - 版本差异兜底，见 docstring
            logger.debug("删除集合 %s 未成功（可能不存在）：%s", self.config.collection, error)
        self._collection = None

    def close(self) -> None:
        """释放句柄（持久化客户端落盘由 chromadb 自行管理，这里只清引用）。"""
        self._collection = None
        self._client = None


# ================================================================ 入库


@dataclass
class IndexReport:
    """一次入库的统计结果。"""

    total: int = 0
    indexed: int = 0
    skipped_existing: int = 0
    skipped_audit: int = 0
    skipped_invalid: int = 0
    failed: int = 0
    collection_count: int = 0

    def summary(self) -> str:
        """一行摘要，供命令行与 PR 输出样例使用。"""
        return (
            f"候选 {self.total} 块：新入库 {self.indexed}，"
            f"已存在跳过 {self.skipped_existing}，审计公告拦截 {self.skipped_audit}，"
            f"无效跳过 {self.skipped_invalid}，失败 {self.failed}；"
            f"集合现有 {self.collection_count} 块"
        )


def build_index(
    records: Iterable[ChunkRecord | Mapping[str, Any] | Any],
    *,
    embedder: Embedder,
    store: VectorStore,
    config: IndexConfig | None = None,
    resume: bool = True,
) -> IndexReport:
    """把 chunk 记录编码后写入向量库。

    断点续跑的实现：先读集合里已有的 ``chunk_id``，已入库的直接跳过；写入用幂等
    ``upsert``，因此中途中断后重跑不会产生重复块。单批编码/写入失败只记失败数，
    不中断整批（与 chunker「单篇失败不打断整批」的容错口径一致）。

    :param records: chunk 记录，接受 ``Chunk`` 对象、映射或 :class:`ChunkRecord`
    :param embedder: 满足 :class:`Embedder` 协议的编码器
    :param store: 向量库封装
    :param config: 入库参数，默认 :class:`IndexConfig`
    :param resume: ``True`` 跳过已入库块；``False`` 全量重写
    :return: 统计结果
    """
    cfg = config or IndexConfig()
    report = IndexReport()
    existing = store.ids() if resume else set()

    pending: list[ChunkRecord] = []
    seen: set[str] = set()
    for raw in records:
        report.total += 1
        try:
            record = to_record(raw)
        except ValueError as error:
            report.skipped_invalid += 1
            logger.warning("跳过无效 chunk：%s", error)
            continue
        if record.metadata.get("category") == AUDIT_CATEGORY:
            report.skipped_audit += 1
            logger.warning("chunk %s 属于审计公告，拒绝入库（红线④）", record.chunk_id)
            continue
        if record.chunk_id in existing or record.chunk_id in seen:
            report.skipped_existing += 1
            continue
        seen.add(record.chunk_id)
        pending.append(record)

    total_batches = max(1, -(-len(pending) // cfg.batch_size))
    step = max(1, total_batches // 10)
    for start in range(0, len(pending), cfg.batch_size):
        batch = pending[start : start + cfg.batch_size]
        number = start // cfg.batch_size + 1
        try:
            vectors = embedder.encode_documents([record.text for record in batch])
            store.upsert(batch, vectors)
        except Exception:  # noqa: BLE001 - 单批失败要能继续跑完剩下的批次
            report.failed += len(batch)
            logger.exception("第 %d 批入库失败，跳过该批 %d 块", number, len(batch))
            continue
        report.indexed += len(batch)
        logger.debug("第 %d 批入库完成：%d 块", number, len(batch))
        if number % step == 0 or number == total_batches:
            logger.info(
                "入库进度 %d/%d 批（已写入 %d 块）", number, total_batches, report.indexed
            )

    report.collection_count = store.count()
    logger.info("入库结束 → %s", report.summary())
    return report


@dataclass
class MetadataAudit:
    """元数据完整性抽检结果（对应验收项「抽 10 块元数据完整」）。"""

    sampled: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """抽检是否全部通过。"""
        return not self.problems

    def summary(self) -> str:
        """一行摘要。"""
        if self.ok:
            return f"抽检 {self.sampled} 块，元数据字段完整"
        return (
            f"抽检 {self.sampled} 块，{len(self.problems)} 块缺字段："
            + "；".join(self.problems[:5])
        )


def check_metadata(
    store: VectorStore,
    *,
    sample: int = 10,
    required: Sequence[str] = REQUIRED_METADATA,
) -> MetadataAudit:
    """抽检集合里的元数据字段是否齐全。

    :param store: 向量库封装
    :param sample: 抽检块数
    :param required: 必备字段
    :return: 抽检结果
    """
    audit = MetadataAudit()
    for chunk_id, metadata in store.sample(sample):
        audit.sampled += 1
        missing = [name for name in required if name not in metadata]
        if missing:
            audit.problems.append(f"{chunk_id} 缺 {','.join(missing)}")
    logger.info("元数据抽检 → %s", audit.summary())
    return audit


# ================================================================ JSONL 与现场分块


def dump_chunks(chunks: Iterable[ChunkRecord | Mapping[str, Any] | Any], path: Path | str) -> int:
    """把 chunk 序列写成 JSONL（入库管线的中间产物），返回写出的块数。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for raw in chunks:
            record = to_record(raw)
            line = {
                "chunk_id": record.chunk_id,
                "text": record.text,
                "metadata": record.metadata,
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1
    logger.info("已写出 %d 块 → %s", written, target)
    return written


def load_chunks(path: Path | str) -> Iterator[ChunkRecord]:
    """逐行读回 chunk JSONL。坏行跳过并记警告，不中断整批。"""
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield to_record(json.loads(text))
            except (json.JSONDecodeError, ValueError) as error:
                logger.warning("%s 第 %d 行无法解析，跳过：%s", target.name, number, error)


def chunks_from_corpus(config: Mapping[str, Any] | None = None) -> list[Any]:
    """从数据仓现场加载并分块（供「一条命令全量重建」）。

    标准入库管线的中间产物是 chunker 输出的 JSONL；本函数只是为了省掉一步手工操作。
    用局部导入，避免本模块对分块器产生硬依赖（分块器 PR 未合并时也能单独合并本模块）。

    :param config: 整份配置（取 ``data_repo`` / ``registry_path``）
    :return: chunk 列表
    :raises RuntimeError: 仓库内尚无 loader/chunker
    """
    try:
        from rag.chunker import chunk_corpus
        from rag.loader import load_registry
    except ImportError as error:  # pragma: no cover - 取决于合并顺序
        raise RuntimeError(
            f"现场分块需要 rag.loader 与 rag.chunker（feat/rag-chunker 已合并），当前缺失：{error}"
        ) from error

    settings = config if isinstance(config, Mapping) else {}
    data_repo = Path(str(settings.get("data_repo") or "../yueceshao-data"))
    registry = Path(str(settings.get("registry_path") or "00_工作区/registry.json"))
    loaded = load_registry(data_repo / registry)
    return chunk_corpus(loaded.documents, data_repo)


# ================================================================ 命令行


def _load_config(path: Path | str) -> dict[str, Any]:
    """读取配置文件；不存在时返回空配置（全部走默认值）。"""
    target = Path(path)
    if not target.is_file():
        logger.warning("配置文件 %s 不存在，使用默认配置", target)
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：``python -m rag.indexer``。"""
    parser = argparse.ArgumentParser(
        prog="python -m rag.indexer",
        description="把 chunk 写入 Chroma 向量库（开发手册 §3.1 PR-3）",
    )
    parser.add_argument("--config", default="config.json", help="配置文件路径，缺省 config.json")
    parser.add_argument("--chunks", help="chunk JSONL 路径（与 --from-corpus 二选一）")
    parser.add_argument(
        "--from-corpus",
        action="store_true",
        help="从数据仓现场分块后入库（需要 loader＋chunker）",
    )
    parser.add_argument("--full-rebuild", action="store_true", help="先删集合再全量重写")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 块（冒烟用）")
    parser.add_argument("--check", action="store_true", help="入库后抽检 10 块元数据完整性")
    parser.add_argument("--log-level", default="INFO", help="日志级别，缺省 INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = _load_config(args.config)
    index_config = IndexConfig.from_config(config)

    if args.from_corpus:
        records: Iterable[Any] = chunks_from_corpus(config)
    elif args.chunks:
        records = load_chunks(args.chunks)
    else:
        parser.error("需要 --chunks 或 --from-corpus 之一")
        return 2
    if args.limit > 0:
        records = list(records)[: args.limit]

    store = ChromaStore(index_config)
    if args.full_rebuild:
        logger.info("全量重建：先删除集合 %s", index_config.collection)
        store.reset()

    embedder = Qwen3Embedder(index_config.embedding)
    report = build_index(records, embedder=embedder, store=store, config=index_config)
    print(report.summary())

    if args.check:
        audit = check_metadata(store)
        print(audit.summary())
        if not audit.ok:
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
