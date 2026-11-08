"""eval/propose_annotations.py — 为「待P0」案例生成候选条款草稿（辅助标注）。

背景（评测口径 · 2026-11-08）
--------------------------
142 条标注表里 107 条「预期命中预警」是「待P0」占位。按《docs/评测口径.md》：
- ~59 条「禁止用途」案例要由数据负责人逐条核对原文、填真实条款号；
- ~22 条「执行率」与 ~26 条「库内暂无」标记为不进②段召回分母。

本脚本对每条「待P0」案例跑**规则引擎**（确定性、免费、无 LLM），产出候选条款
（文号#条款号 + 条款原文），生成草稿 CSV 供数据负责人**审改**（采纳/改/否）。
候选仅供参考——标准答案以人终审为准，保证评测独立性（AI 不代标）。

用法::

    python -m eval.propose_annotations --config config.json \
      --cases <数据仓>/审计公告/案例标注表.csv --out 候选草稿.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.dataset import normalize_ref
from eval.real_pipeline import _is_placeholder_doc_no
from rules.engine import NegRuleMatcher
from rules.loader import load_store_from_config

__all__ = ["main"]

_CLAUSE_NUM = re.compile(r"^第?\s*([0-9一二三四五六七八九十百零〇两]+)\s*条?$")


def _raw_clause(clause_no: str) -> str:
    text = (clause_no or "").strip()
    match = _CLAUSE_NUM.match(text)
    return match.group(1) if match else text


def _ref(doc_no: str, clause_no: str) -> str:
    return normalize_ref(doc_no, _raw_clause(clause_no))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.propose_annotations",
        description="为「待P0」案例生成候选条款草稿（辅助标注，人终审）",
    )
    parser.add_argument("--config", default="config.json", help="本机配置（数据仓路径）")
    parser.add_argument("--cases", required=True, help="案例标注表 CSV")
    parser.add_argument("--out", default="候选草稿.csv", help="输出草稿 CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    store = load_store_from_config(Path(args.config))
    matcher = NegRuleMatcher(store)

    with open(args.cases, encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    pending = [r for r in rows if (r.get("预期命中预警") or "").startswith("待P0")]

    fieldnames = [
        "案例编号", "违规类型", "违规情形描述",
        "候选条款(文号#条款号)", "候选条款原文", "数据负责人确认",
    ]
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pending:
            query = row.get("改编后申报场景描述_体检输入") or ""
            match = matcher.match(query)
            refs, texts = [], []
            for hit in match.hits:
                if _is_placeholder_doc_no(hit.rule.doc_no):
                    continue  # 占位文号不是可核验候选，跳过（数据②补全真实文号后自然消失）
                refs.append(_ref(hit.rule.doc_no, hit.rule.clause_no))
                texts.append((hit.rule.clause_text or "").replace("\n", " "))
            writer.writerow({
                "案例编号": row.get("案例编号", ""),
                "违规类型": row.get("违规类型", ""),
                "违规情形描述": (row.get("违规情形描述_原文摘录") or "")[:120],
                "候选条款(文号#条款号)": ";".join(dict.fromkeys(refs)),
                "候选条款原文": ";".join(dict.fromkeys(texts))[:300],
                "数据负责人确认": "",
            })

    print(f"已生成 {len(pending)} 条「待P0」案例的候选条款草稿 → {out_path.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
