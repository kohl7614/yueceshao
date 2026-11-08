"""eval/merge.py — 合并多份分片跑批结果，重算三指标与消融矩阵。

背景（feat/eval-split，三人分片跑 `--pipeline real --ablate-all`）
-------------------------------------------------------------
三人在各自机器上按案例区间分片跑批，各产出 ``group_details.csv``（组×逐案例
底层数据）。匹配率/召回率/引用错误率都是**比率**，不能跨片简单平均，必须回
底层逐案例数据重算。本脚本把这些分片拼回全量，按组重算三指标，输出最终
``metrics.json`` + ``ablation_matrix.csv`` + ``details.csv``（主组）。

用法::

    python -m eval.merge --inputs runs/kelier1 runs/xzj0202 runs/magic-42 \\
        --out runs/final

引用错误率：各分片跑批时**必须带同一份 registry 导出**（``--registry``），
这样逐案例的 ``citations_bad`` 已按同一口径算好，合并直接求和即可；若分片
未带 registry，则最终引用错误率记 N/A（宁缺勿错，不冒充 0）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.metrics import CaseOutcome, EvalThresholds, summarize

__all__ = ["main"]


def _split(value: str) -> tuple[str, ...]:
    """CSV 里的 ``;`` 分隔列 → 元组（空串 → 空元组）。"""
    return tuple(x for x in (value or "").split(";") if x)


def _opt_bool(value: str) -> bool | None:
    return None if value == "" else bool(int(value))


def _opt_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _row_to_outcome(row: dict[str, str]) -> CaseOutcome:
    """把 ``group_details.csv`` 的一行还原成 :class:`CaseOutcome`。"""
    return CaseOutcome(
        case_id=row["case_id"],
        expected_verdict=row["expected_verdict"] or None,
        predicted_verdict=row["predicted_verdict"],
        verdict_match=_opt_bool(row["verdict_match"]),
        expected_refs=_split(row["expected_refs"]),
        predicted_refs=_split(row["predicted_refs"]),
        refs_hit=_split(row["refs_hit"]),
        recall_mode=row["recall_mode"],
        citations_total=int(row["citations_total"]),
        citations_bad=int(row["citations_bad"]),
        forbidden_hits=_split(row["forbidden_hits"]),
        notes=row["notes"],
        degraded=_opt_bool(row["degraded"]),
        attempts=_opt_int(row["attempts"]),
        entail_failed=_opt_bool(row["entail_failed"]),
    )


def _read_group_details(directory: Path) -> list[dict[str, str]]:
    """读一个分片目录的 ``group_details.csv``，返回原始行列表。"""
    path = directory / "group_details.csv"
    if not path.is_file():
        raise FileNotFoundError(f"分片目录缺 group_details.csv：{directory}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _grouped(rows: Sequence[dict[str, str]]) -> dict[str, list[CaseOutcome]]:
    """按 ``group`` 列归并，保持各分片首次出现顺序（full 在前）。"""
    order: dict[str, None] = {}
    grouped: dict[str, list[CaseOutcome]] = {}
    for row in rows:
        label = row["group"]
        order.setdefault(label, None)
        grouped.setdefault(label, []).append(_row_to_outcome(row))
    return {label: grouped[label] for label in order}


def _write_matrix(
    path: Path,
    labels: Sequence[str],
    summaries: dict[str, Any],
) -> None:
    """多组对比矩阵（与 runner 的 ablation_matrix.csv 同构）。"""
    fieldnames = [
        "group", "disabled_switches", "case_count", "match_rate", "recall",
        "citation_error_rate", "forbidden_violations", "all_pass",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label in labels:
            summary = summaries[label]
            writer.writerow(
                {
                    "group": label,
                    "disabled_switches": "" if label == "full" else "+".join(
                        label[len("w/o:"):].split("+")),
                    "case_count": summary.case_count,
                    "match_rate": "" if summary.match_rate is None else f"{summary.match_rate:.4f}",
                    "recall": "" if summary.recall is None else f"{summary.recall:.4f}",
                    "citation_error_rate": (
                        ""
                        if summary.citation_error_rate is None
                        else f"{summary.citation_error_rate:.4f}"
                    ),
                    "forbidden_violations": summary.forbidden_violations,
                    "all_pass": int(summary.all_pass()),
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.merge",
        description="合并多份分片跑批结果，重算三指标与消融矩阵",
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="各分片的输出目录（每个都含 group_details.csv），可多个")
    parser.add_argument("--out", default="runs/final", help="合并输出目录")
    parser.add_argument(
        "--match-threshold", type=float, default=0.85, help="匹配率验收线，默认 0.85")
    parser.add_argument(
        "--recall-threshold", type=float, default=0.80, help="召回验收线，默认 0.80")
    parser.add_argument(
        "--citation-error-threshold", type=float, default=0.05, help="引用错误率上限，默认 0.05")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；返回 0=全绿，1=有指标未达标，2=配置错误。"""
    args = _build_parser().parse_args(argv)
    thresholds = EvalThresholds(
        match_rate=args.match_threshold,
        recall=args.recall_threshold,
        citation_error_rate=args.citation_error_threshold,
    )

    rows: list[dict[str, str]] = []
    registry_provided = True
    try:
        for directory in args.inputs:
            dir_path = Path(directory)
            part = _read_group_details(dir_path)
            rows.extend(part)
            # 引用错误率能否计算，取决于各分片是否带了 registry。读各片 metrics.json
            # 的 registry_provided 字段（runner 写出的权威来源），而非猜 CSV 列——
            # to_row 恒含 citations_total，猜列存在会恒 True、把「漏传 registry」误报成
            # 0% 引用错误率（#70 终审发现）。
            metrics_path = dir_path / "metrics.json"
            if metrics_path.is_file():
                meta = json.loads(metrics_path.read_text(encoding="utf-8"))
                if not meta.get("registry_provided", False):
                    registry_provided = False
            else:
                registry_provided = False  # 无法确认 → 保守记 N/A
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"合并失败：{exc}", file=sys.stderr)
        return 2

    if not rows:
        print("合并失败：没有任何逐案例数据。", file=sys.stderr)
        return 2

    grouped = _grouped(rows)
    labels = list(grouped)
    summaries = {
        label: summarize(outcomes, thresholds, registry_provided=registry_provided)
        for label, outcomes in grouped.items()
    }
    primary_label = labels[0]
    primary = summaries[primary_label]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).astimezone()
                .isoformat(timespec="seconds"),
                "merged_from": [str(Path(d).resolve()) for d in args.inputs],
                **primary.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_matrix(out_dir / "ablation_matrix.csv", labels, summaries)
    # 主组逐案例明细（供回看；结构与 runner 的 details.csv 同构）。
    with (out_dir / "details.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        outcomes = grouped[primary_label]
        if outcomes:
            writer = csv.DictWriter(handle, fieldnames=list(outcomes[0].to_row().keys()))
            writer.writeheader()
            writer.writerows(item.to_row() for item in outcomes)

    def _fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    print("\n".join([
        f"合并分片 : {', '.join(args.inputs)}",
        f"主组案例 : {primary.case_count} 条",
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
