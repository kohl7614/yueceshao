"""评测数据集的读取与校验（3.4 评测与消融 · PR2 · feat/eval-runner）。

数据来源：**团队数据仓**的 `审计公告/案例标注表.csv`（120 条），不进本仓库
（`.gitignore` 已挡 `data/`；红线第 4 条：审计公告是评测出题集，进检索索引＝答案泄露）。

本模块只负责"按约定 schema 读入"，不内置任何案例内容、不写任何文号常量。
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "DatasetError",
    "RegistryError",
    "EvalCase",
    "load_cases",
    "REQUIRED_COLUMNS",
    "VERDICTS",
]

logger = logging.getLogger(__name__)

#: 必需列名。缺失即报错——宁可当场炸，也不要带着半张表跑出假指标。
REQUIRED_COLUMNS: tuple[str, ...] = ("case_id", "query")

#: 合法判定取值。
VERDICTS: tuple[str, ...] = ("hit", "miss")


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

    @property
    def is_positive(self) -> bool:
        """标注为应命中的正例。"""
        return self.expected_verdict == "hit"


def _split_refs(raw: str) -> tuple[str, ...]:
    """把 ``a;b; c`` 拆成去空、去重的有序元组。"""
    parts = (token.strip() for token in raw.replace("；", ";").split(";"))
    seen: dict[str, None] = {}
    for token in parts:
        if token:
            seen.setdefault(token, None)
    return tuple(seen)


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
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            raise DatasetError(
                f"标注表缺少必需列：{', '.join(missing)}；实际表头：{', '.join(header) or '(空)'}"
            )

        cases: list[EvalCase] = []
        seen_ids: set[str] = set()
        for line_no, row in enumerate(reader, start=2):
            case_id = (row.get("case_id") or "").strip()
            query = (row.get("query") or "").strip()
            if not case_id:
                raise DatasetError(f"第 {line_no} 行 case_id 为空")
            if not query:
                raise DatasetError(f"第 {line_no} 行（{case_id}）query 为空")
            if case_id in seen_ids:
                raise DatasetError(f"第 {line_no} 行 case_id 重复：{case_id}")
            seen_ids.add(case_id)

            verdict_raw = (row.get("expected_verdict") or "").strip().lower()
            if verdict_raw and verdict_raw not in VERDICTS:
                raise DatasetError(
                    f"第 {line_no} 行（{case_id}）expected_verdict 非法：{verdict_raw!r}；"
                    f"应为 {' / '.join(VERDICTS)} 或留空"
                )

            cases.append(
                EvalCase(
                    case_id=case_id,
                    query=query,
                    expected_verdict=verdict_raw or None,
                    expected_refs=_split_refs(row.get("expected_refs") or ""),
                    notes=(row.get("notes") or "").strip(),
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
