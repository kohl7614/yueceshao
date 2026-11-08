"""eval/reevaluate.py — 数据①（P0 口径）定稿后，免费重算三指标。

背景（数据侧任务书）
-------------------
队友跑全量时，案例标注表的「预期命中条款」还是「待P0」占位符（统一填
``粤财农〔2020〕106号#20``），导致召回率被算成 0%——这是**标准答案没定稿**，
不是系统没查到。但系统预测（``predicted_verdict`` / ``predicted_refs`` /
``citations_total/bad``）已冻在各分片的 ``group_details.csv`` 里，是真实、可复用的。

数据①定稿后，只需用新标注表的标准答案**重新对账**，不必重跑管线、不再花 token：

    python -m eval.reevaluate \
      --cases <新案例标注表.csv> \
      --inputs runs/kelier1 runs/xzj0202 runs/magic-42 runs/kohl \
      --out runs/final

重算口径与 ``eval.metrics`` 完全一致：匹配率/召回率按**新标准答案**重新对账；
引用错误率沿用各分片已冻的 ``citations_bad/total``（系统输出侧未变，不必重算）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from eval.dataset import EvalCase, load_cases
from eval.merge import _read_group_details, _row_to_outcome, _write_matrix
from eval.metrics import CaseOutcome, EvalThresholds, summarize

__all__ = ["main"]


def _split(value: str) -> tuple[str, ...]:
    return tuple(x for x in (value or "").split(";") if x)


def _recompute_row(row: dict[str, str], case: EvalCase) -> CaseOutcome:
    """用新标注表的标准答案，对一条已冻结的预测重新对账。

    预测侧（predicted_verdict / predicted_refs）与引用侧（citations_total/bad）
    沿用分片冻值；预期侧换成新标注表。
    """
    frozen = _row_to_outcome(row)
    predicted = set(frozen.predicted_refs)
    expected = set(case.expected_refs)
    verdict_match: bool | None = None
    if case.expected_verdict is not None:
        verdict_match = frozen.predicted_verdict == case.expected_verdict
    refs_hit = tuple(ref for ref in case.expected_refs if ref in predicted)
    if case.expected_verdict == "hit" and expected:
        recall_mode = "ref-level"
    elif case.expected_verdict == "hit":
        recall_mode = "case-level"
    else:
        recall_mode = "n/a"
    return CaseOutcome(
        case_id=case.case_id,
        expected_verdict=case.expected_verdict,
        predicted_verdict=frozen.predicted_verdict,
        verdict_match=verdict_match,
        expected_refs=case.expected_refs,
        predicted_refs=frozen.predicted_refs,
        refs_hit=refs_hit,
        recall_mode=recall_mode,
        citations_total=frozen.citations_total,
        citations_bad=frozen.citations_bad,
        forbidden_hits=frozen.forbidden_hits,
        notes=case.notes,
        degraded=frozen.degraded,
        attempts=frozen.attempts,
        entail_failed=frozen.entail_failed,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.reevaluate",
        description="数据①定稿后，用新标注表免费重算三指标（不重跑管线）",
    )
    parser.add_argument("--cases", required=True, help="定稿后的案例标注表 CSV")
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="各分片输出目录（含 group_details.csv）")
    parser.add_argument("--out", default="runs/final", help="重算输出目录")
    parser.add_argument("--match-threshold", type=float, default=0.85)
    parser.add_argument("--recall-threshold", type=float, default=0.80)
    parser.add_argument("--citation-error-threshold", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    thresholds = EvalThresholds(
        match_rate=args.match_threshold,
        recall=args.recall_threshold,
        citation_error_rate=args.citation_error_threshold,
    )

    new_cases: dict[str, EvalCase] = {c.case_id: c for c in load_cases(args.cases)}
    rows: list[dict[str, str]] = []
    try:
        for directory in args.inputs:
            rows.extend(_read_group_details(Path(directory)))
    except FileNotFoundError as exc:
        print(f"重算失败：{exc}", file=sys.stderr)
        return 2

    # 只重算新标注表里有（且各分片已预测）的案例；缺预测的跳过并提示。
    grouped: dict[str, list[CaseOutcome]] = {}
    order: dict[str, None] = {}
    missing = 0
    for row in rows:
        case = new_cases.get(row["case_id"])
        if case is None:
            missing += 1
            continue
        label = row["group"]
        order.setdefault(label, None)
        grouped.setdefault(label, []).append(_recompute_row(row, case))
    if missing:
        print(f"提示：{missing} 行在新标注表中无对应案例，已跳过。", file=sys.stderr)

    if not grouped:
        print("重算失败：没有可重算的预测数据。", file=sys.stderr)
        return 2

    labels = list(order)
    summaries = {
        label: summarize(outcomes, thresholds, registry_provided=True)
        for label, outcomes in grouped.items()
    }
    primary = summaries[labels[0]]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).astimezone()
                .isoformat(timespec="seconds"),
                "reevaluated_from": [str(Path(d).resolve()) for d in args.inputs],
                "cases_file": str(Path(args.cases).resolve()),
                **primary.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_matrix(out_dir / "ablation_matrix.csv", labels, summaries)

    def _fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    print("\n".join([
        f"重算案例 : {primary.case_count} 条（新标准答案）",
        f"匹配率   : {_fmt(primary.match_rate)}  线 {thresholds.match_rate:.0%}",
        f"召回率   : {_fmt(primary.recall)}  线 {thresholds.recall:.0%}",
        f"引用错误 : {_fmt(primary.citation_error_rate)}"
        f"  上限 {thresholds.citation_error_rate:.0%}",
        f"姿态违规 : {primary.forbidden_violations} 处",
        f"输出目录 : {out_dir.resolve()}",
    ]))
    return 0 if primary.all_pass() else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
