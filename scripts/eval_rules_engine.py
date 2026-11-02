"""本地评测脚本：规则引擎 × 案例标注表（等价组命中率，数据不入库）。

用法（代码仓根目录）：python scripts/eval_rules_engine.py
前置：根目录 config.json 已按 config.example.json 配置数据仓、规则源与案例表路径。

口径：
1. 历史适用条款 → 等价组映射：按（文号+条款号）匹配规则库记录，取组号集合（排除「结构标注」组，
   如总括句；106号第十八条 → G01-G11 等价组映射到现行组）；
2. 引擎命中规则所属等价组与期望组相交即算「等价组命中」；
3. 抽样：可映射案例按编号升序，用固定随机种子抽 20 条；另附全量与负面清单型子集参考值。
"""

from __future__ import annotations

import csv
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rules.engine import NegRuleMatcher  # noqa: E402
from rules.loader import load_config, load_store_from_config  # noqa: E402
from rules.schema import RuleStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval")

_DOC_PAT = re.compile(r"([\u4e00-\u9fff]{2,10}〔\d{4}〕\d+号)")
_CLAUSE_PAT = re.compile(r"第([一二三四五六七八九十百零]+)条")
_SAMPLE_SIZE = 20
_SEED = 20261102


def expected_groups(history: str, store: RuleStore) -> frozenset[str]:
    """历史适用条款 → 等价组集合（排除结构标注组）。"""
    docs = _DOC_PAT.findall(history)
    clauses = _CLAUSE_PAT.findall(history)
    groups = {
        r.group_id
        for r in store.records
        if r.doc_no in docs and r.clause_no in clauses and r.group_type != "结构标注"
    }
    return frozenset(groups)


def main() -> int:
    """跑抽样 20 条与全量参考，逐条日志输出明细与命中率。"""
    config = load_config()
    store = load_store_from_config()
    matcher = NegRuleMatcher(store)
    cases_file = Path(config["data_repo"]) / config["audit_cases_file"]
    rows = list(csv.DictReader(open(cases_file, encoding="utf-8-sig")))
    mappable = sorted((r for r in rows if expected_groups(r["历史适用条款_文号及条款号"], store)),
                      key=lambda r: r["案例编号"])
    logger.info("案例总数 %d，历史条款可等价组映射 %d 条", len(rows), len(mappable))

    sample = random.Random(_SEED).sample(mappable, _SAMPLE_SIZE)
    hits = 0
    for row in sample:
        expected = expected_groups(row["历史适用条款_文号及条款号"], store)
        result = matcher.match(row["改编后申报场景描述_体检输入"] or "")
        got = result.groups()
        ok = bool(got & expected)
        hits += ok
        logger.info(
            "%s %s 期望组=%s 命中组=%s 命中规则=%s",
            row["案例编号"], "✓" if ok else "✗",
            ",".join(sorted(expected)) or "-", ",".join(sorted(got)) or "-",
            "；".join(f"{h.rule.doc_no}{h.rule.clause_no}" for h in result.hits[:3]) or "-",
        )
    logger.info("等价组命中率（%d 条抽样）：%d/%d = %.1f%%",
                len(sample), hits, len(sample), 100.0 * hits / len(sample))

    all_hits = sum(
        1 for r in mappable
        if matcher.match(r["改编后申报场景描述_体检输入"] or "").groups()
        & expected_groups(r["历史适用条款_文号及条款号"], store)
    )
    logger.info("全量参考（可映射 %d 条）：%d 命中 = %.1f%%",
                len(mappable), all_hits, 100.0 * all_hits / len(mappable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
