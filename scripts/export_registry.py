"""scripts/export_registry.py — 从数据仓规则库导出「条款级」文号注册表。

引用错误率需要「文号#条款序号 → 元数据」的核验表；数据仓 `registry.json` 是
文档级（733 篇），不含条款号，无法逐条款核验。本脚本读规则库（负面清单_条款
归并.json）里的每条现行条款，归一化文号+条款号，输出 ``ref_id → 元数据``，
供 ``eval.runner --registry`` 与 ``eval.merge`` 核验引用。

用法::

    python -m scripts.export_registry --config config.json --out eval/registry-export.json

输出为 JSON 对象：``{"文号#条款序号": {"doc_no": "...", "clause_no": "..."}, ...}``。
只含文号与条款号标识、不含条款正文（遵守红线⑥：政策语料原文不进代码仓）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.dataset import normalize_ref
from rules.loader import load_store_from_config

__all__ = ["main"]

# 与 eval/real_pipeline.py::_raw_clause 同源：把「第四条」→「四」供 normalize_ref 归一。
_CLAUSE_NUM = re.compile(r"^第?\s*([0-9一二三四五六七八九十百零〇两]+)\s*条?$")


def _raw_clause(clause_no: str) -> str:
    """把「第四条」→「四」（去掉第/条前后缀，供 normalize_ref 归一到阿拉伯序号）。"""
    text = (clause_no or "").strip()
    match = _CLAUSE_NUM.match(text)
    return match.group(1) if match else text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.export_registry",
        description="从数据仓规则库导出条款级文号注册表（供引用错误率核验）",
    )
    parser.add_argument("--config", default="config.json", help="本机配置文件（数据仓路径）")
    parser.add_argument(
        "--out", default="eval/registry-export.json", help="输出 JSON 路径")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    store = load_store_from_config(Path(args.config))
    registry: dict[str, dict[str, str]] = {}
    for record in store.valid_records():
        ref = normalize_ref(record.doc_no, _raw_clause(record.clause_no))
        registry.setdefault(ref, {"doc_no": record.doc_no, "clause_no": record.clause_no})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"导出 {len(registry)} 条条款级注册表 → {out.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
