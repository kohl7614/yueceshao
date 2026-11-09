"""评测数据集的读取与校验（3.4 评测与消融 · PR2 · feat/eval-runner）。

数据来源：**团队数据仓**的 `审计公告/案例标注表.csv`（120 条），不进本仓库
（`.gitignore` 已挡 `data/`；红线第 4 条：审计公告是评测出题集，进检索索引＝答案泄露）。

本模块只负责"按约定 schema 读入"，不内置任何案例内容、不写任何文号常量。

**列名别名（feat/eval-rules-pipeline 起）**：团队数据仓的实际标注表使用中文列名
（案例编号／改编后申报场景描述_体检输入／预期命中预警／历史适用条款_文号及条款号／备注），
读取时自动映射到标准列；「预期命中预警」按文案推断 hit／miss（含「应命中／应提示」→hit，
含「不应命中」→miss，其余→未标注），「历史适用条款」按正则提取「文号＋条款号」对。
schema 如下（UTF-8，带表头）：

===============  ======  ==========================================================
列名             必填    说明
===============  ======  ==========================================================
case_id          是      案例唯一标识，如 ``AUD-2024-001``
query            是      送进系统的项目描述原文（审计公告摘录，勿改写）
expected_verdict 否      期望判定：``hit``（应命中负面清单）或 ``miss``（不应命中）；
                         缺省按 ``miss`` 处理，并在明细里标记 ``verdict_unlabeled``
expected_refs    否      期望命中的条款标识，多条用 ``;`` 分隔。**只放 registry 里已核验的
                         标识符**，不要在这里手写文号（红线第 1 条）
notes            否      人工备注，仅进明细，不参与指标
===============  ======  ==========================================================
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DatasetError",
    "RegistryError",
    "EvalCase",
    "load_cases",
    "REQUIRED_COLUMNS",
    "VERDICTS",
    "COLUMN_ALIASES",
    "extract_refs_from_text",
    "summarize_annotation_quality",
]

logger = logging.getLogger(__name__)

#: 必需列名。缺失即报错——宁可当场炸，也不要带着半张表跑出假指标。
REQUIRED_COLUMNS: tuple[str, ...] = ("case_id", "query")

#: 合法判定取值。
VERDICTS: tuple[str, ...] = ("hit", "miss")

#: 中文列名 → 标准列名（团队数据仓实际标注表的表头）。
COLUMN_ALIASES: dict[str, str] = {
    "case_id": "案例编号",
    "query": "改编后申报场景描述_体检输入",
    "expected_verdict": "预期命中预警",
    "expected_refs": "历史适用条款_文号及条款号",
    "notes": "备注",
}

#: 「文号＋条款号」提取：条款号**必需**（条级引擎与条级标注对齐；纯文号与
#: 「第（二）项」项级引用一律不收——宁漏勿错）。只认全形文号，
#: 不追「106号第十九条」式简写。
REF_EXTRACT_PATTERN = re.compile(
    r"([\u4e00-\u9fa5]{2,10}〔\d{4}〕\d+号)第([一二三四五六七八九十百零]+)条")

#: 提取结果的引导词剥除表：正则前缀字符类可能把普通行文（如「现行参照载体」）
#: 吃进文号前缀，而机关简称不会以这些词开头——逐个剥除即可。
_LEADING_NOISE_PREFIXES: tuple[str, ...] = (
    "现行参照载体", "参照载体", "现行参照", "参照", "现行", "依据", "对应", "载体",
)


def _strip_leading_noise(ref: str) -> str:
    for prefix in _LEADING_NOISE_PREFIXES:
        while ref.startswith(prefix):
            ref = ref[len(prefix):]
    return ref


def normalize_ref(doc_no: str, clause_no: str) -> str:
    """把「文号＋条款号」归一为跨形态可比的标识：``文号#条款序号``。

    标注侧写「第二十条」、规则库写「二十」、组员展示「十八」——字符串级
    比对永不相交（feat/eval-rules-pipeline 首跑发现）；统一转阿拉伯序号后
    才可比。条款号无法转数字（如「未分条」）时退回原文，宁缺勿错。
    """
    from rag.chunker import zh_number_to_int  # 局部 import：复用有测试的数字转换

    seq = zh_number_to_int(clause_no)
    return f"{doc_no}#{seq}" if seq is not None else f"{doc_no}#{clause_no.strip()}"


def extract_refs_from_text(text: str) -> tuple[str, ...]:
    """从标注长文本提取期望条款（归一化标识，去重保序）。

    按「；/;」分段后跳过含「不覆盖本情形」的段——那是"该规则不适用此案例"的
    标注语义（与 scripts/eval_rules_engine.py 口径一致），其文号不得计入期望命中。
    """
    seen: dict[str, None] = {}
    for segment in re.split(r"[；;]", text or ""):
        if "不覆盖本情形" in segment or "不覆盖此情形" in segment:
            continue
        for doc_no, clause_cn in REF_EXTRACT_PATTERN.findall(segment):
            seen.setdefault(normalize_ref(_strip_leading_noise(doc_no), clause_cn), None)
    seen.pop("", None)
    return tuple(seen)


def _verdict_from_warning(text: str) -> str | None:
    """从「预期命中预警」文案推断判定：应命中/应提示→hit；不应命中→miss；其余→未标注。"""
    if "不应命中" in text or "不应提示" in text:
        return "miss"
    if "应命中" in text or "应提示" in text:
        return "hit"
    return None


def _cell(row: dict, column: str) -> str:
    """取标准列的值：标准列名优先，缺失时回落中文别名列。"""
    if column in row:
        return (row.get(column) or "").strip()
    alias = COLUMN_ALIASES.get(column)
    if alias and alias in row:
        return (row.get(alias) or "").strip()
    return ""


class DatasetError(ValueError):
    """标注表不合规（缺列、重复 case_id、判定取值非法等）。"""


class RegistryError(ValueError):
    """文号注册表导出不合规（空索引、列表元素缺 ``ref_id``、格式不支持）。

    与 :class:`DatasetError` 同等严厉：registry 建不出索引时若静默放行，所有引用都会被
    判为「无法核验」，引用错误率显示成 100%——这个数字看起来像"系统引用了大量假文号"，
    实际只是输入格式不对，属最容易误导人的假结果。
    """


@dataclass(frozen=True)
class EvalCase:
    """一条评测案例。

    :param case_id: 案例唯一标识
    :param query: 送进系统的项目描述原文
    :param expected_verdict: 期望判定（``hit`` / ``miss``）；``None`` 表示未标注
    :param expected_refs: 期望命中的条款标识集合
    :param notes: 人工备注
    """

    case_id: str
    query: str
    expected_verdict: str | None = None
    expected_refs: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
    expected_refs_rejected: tuple[str, ...] = field(default_factory=tuple)
    """标注列里**不是条款引用**的原文片段（说明文字/库内缺口备注）。

    不计入召回分母，但要能报数：这是"题库缺口"与"系统未命中"的分界证据。
    """

    @property
    def is_positive(self) -> bool:
        """标注为应命中的正例。"""
        return self.expected_verdict == "hit"


#: 紧凑写法「文号#序号」：文号必须是全形文号，序号为阿拉伯或中文数字。
#: 这是标注表里唯一被接受的非「第X条」写法；其余一律不当条款引用——
#: 实测标注列里混有「106号第十八条不覆盖（用地合规…）」「知识库暂无对应文件，
#: 建议补采…」这类说明文字，旧实现把它原样当期望条款，召回分母被污染成 0。
#: 未分条条款的规范写法（与 rules/schema.py 的 clause_no、registry 导出同键）。
_CLAUSE_UNNUMBERED = "未分条（原文无条款号）"

_STANDARD_REF_PATTERN = re.compile(
    r"^(?P<doc>[\u4e00-\u9fa5]{2,10}〔\d{4}〕\d+号)"
    r"#(?P<clause>[0-9]+|[一二三四五六七八九十百零两]+|" + re.escape(_CLAUSE_UNNUMBERED) + r")$")


def _split_refs(raw: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """拆分标注列：返回 ``(合法条款引用, 被拒的原文片段)``。

    合法＝``文号#序号`` 紧凑写法（全形文号 + 数字序号）。被拒片段**不丢弃**，
    由 :class:`EvalCase` 带着走并在产物里报数——口径外/库内缺口是事实，
    既不能算成"未命中"，也不能悄悄消失。
    """
    accepted: dict[str, None] = {}
    rejected: dict[str, None] = {}
    for token in (part.strip() for part in raw.replace("；", ";").split(";")):
        if not token:
            continue
        match = _STANDARD_REF_PATTERN.match(token)
        if match:
            # 归一：中文序号也要转成阿拉伯序号（预测侧键是 ``文号#20``），
            # 否则标注写「#二十」会永远与预测侧的「#20」错位、静默记 miss。
            accepted.setdefault(normalize_ref(match.group("doc"), match.group("clause")), None)
        else:
            rejected.setdefault(token, None)
    return tuple(accepted), tuple(rejected)


def load_cases(path: str | Path) -> list[EvalCase]:
    """读取标注表 CSV。

    :param path: CSV 路径（团队数据仓内，不入库）
    :raises DatasetError: 文件不存在、缺必需列、case_id 重复、判定取值非法
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise DatasetError(
            f"找不到标注表：{csv_path}\n"
            "标注表在团队数据仓（审计公告/案例标注表.csv），不随仓库分发；"
            "请用 --cases 指向本地副本。"
        )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        missing = [
            column for column in REQUIRED_COLUMNS
            if column not in header and COLUMN_ALIASES.get(column) not in header
        ]
        if missing:
            raise DatasetError(
                f"标注表缺少必需列：{', '.join(missing)}；实际表头：{', '.join(header) or '(空)'}"
            )

        cases: list[EvalCase] = []
        seen_ids: set[str] = set()
        for line_no, row in enumerate(reader, start=2):
            case_id = _cell(row, "case_id")
            query = _cell(row, "query")
            if not case_id:
                raise DatasetError(f"第 {line_no} 行 case_id 为空")
            if not query:
                raise DatasetError(f"第 {line_no} 行（{case_id}）query 为空")
            if case_id in seen_ids:
                raise DatasetError(f"第 {line_no} 行 case_id 重复：{case_id}")
            seen_ids.add(case_id)

            # 两条契约：标准列名下非法取值当场炸（原契约，宁炸勿假）；
            # 中文「预期命中预警」列按文案推断，推断不了视为未标注（不炸）。
            has_standard_verdict_column = "expected_verdict" in row
            verdict_cell = _cell(row, "expected_verdict").lower()
            if has_standard_verdict_column and verdict_cell and verdict_cell not in VERDICTS:
                raise DatasetError(
                    f"第 {line_no} 行（{case_id}）expected_verdict 非法：{verdict_cell!r}；"
                    f"应为 {' / '.join(VERDICTS)} 或留空"
                )
            if verdict_cell in VERDICTS:
                verdict_raw = verdict_cell
            else:
                verdict_raw = _verdict_from_warning(verdict_cell) or ""

            raw_refs = _cell(row, "expected_refs")
            parsed_refs = extract_refs_from_text(raw_refs)
            if parsed_refs:
                refs, rejected = parsed_refs, ()
            else:
                refs, rejected = _split_refs(raw_refs)
            cases.append(
                EvalCase(
                    case_id=case_id,
                    query=query,
                    expected_verdict=verdict_raw or None,
                    expected_refs=refs,
                    notes=_cell(row, "notes"),
                    expected_refs_rejected=rejected,
                )
            )

    if not cases:
        raise DatasetError(f"标注表为空：{csv_path}")
    logger.info("读入 %d 条评测案例：%s", len(cases), csv_path)
    return cases


def take(cases: Iterable[EvalCase], limit: int | None) -> list[EvalCase]:
    """取前 ``limit`` 条（smoke 用），``None`` 表示全量。"""
    materialized: Sequence[EvalCase] = list(cases)
    if limit is None:
        return list(materialized)
    if limit <= 0:
        raise ValueError(f"limit 必须为正整数或 None，收到 {limit!r}")
    return list(materialized)[:limit]


def summarize_annotation_quality(cases: Sequence[EvalCase]) -> dict[str, object]:
    """统计标注列质量：合法条款引用 / 被拒片段 / 无有效引用的应命中案例。

    用途：写进 ``metrics.json``，让"召回分母里到底有多少真条款"可被审阅——
    分母缩水必须伴随证据，不能只体现在指标数字变小。
    """
    valid = sum(len(case.expected_refs) for case in cases)
    rejected = sum(len(case.expected_refs_rejected) for case in cases)
    positives = [case for case in cases if case.expected_verdict == "hit"]
    no_ref_positives = [case for case in positives if not case.expected_refs]
    samples: dict[str, None] = {}
    for case in cases:
        for item in case.expected_refs_rejected:
            if len(samples) < 10:
                samples.setdefault(item[:80], None)
    return {
        "cases": len(cases),
        "expected_refs_valid": valid,
        "expected_refs_rejected": rejected,
        "positive_cases": len(positives),
        "positive_cases_without_ref": len(no_ref_positives),
        "positive_cases_without_ref_ids": [case.case_id for case in no_ref_positives][:30],
        "rejected_samples": list(samples),
    }
