"""tests/test_reevaluate.py — 数据①定稿后的免费重算链路。

验证 :func:`eval.reevaluate._recompute_row`：预测侧冻值不变、预期侧换新标准答案，
召回率从「旧占位符 → 0%」变成「新标准答案 → 真实值」，无需重跑管线。
另覆盖 main() 的护栏：registry 口径（N/A 不冒充 0）与分片重叠拒算。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from eval.dataset import EvalCase
from eval.metrics import CaseOutcome
from eval.reevaluate import _recompute_row
from eval.reevaluate import main as reevaluate_main


def _frozen_row(expected_refs: tuple[str, ...], predicted_refs: tuple[str, ...]) -> dict[str, str]:
    frozen = CaseOutcome(
        case_id="C1", expected_verdict="hit", predicted_verdict="hit",
        verdict_match=False, expected_refs=expected_refs, predicted_refs=predicted_refs,
        refs_hit=(), recall_mode="ref-level", citations_total=3, citations_bad=0,
        forbidden_hits=(),
    )
    return {"group": "full", **frozen.to_row()}


def test_recompute_row_replaces_stale_ground_truth() -> None:
    """旧标注是占位符（不对账命中），换成新标准答案后召回从 0 → 命中。"""
    row = _frozen_row(("旧占位#20",), ("粤测农〔2099〕1号#4",))
    new_case = EvalCase(
        case_id="C1", query="自造描述", expected_verdict="hit",
        expected_refs=("粤测农〔2099〕1号#4",),
    )
    out = _recompute_row(row, new_case)
    assert out.verdict_match is True
    assert out.refs_hit == ("粤测农〔2099〕1号#4",)
    assert out.recall_mode == "ref-level"


def test_recompute_row_keeps_frozen_citations() -> None:
    """预测与引用侧冻值不变，只换预期侧。"""
    row = _frozen_row(("旧占位#20",), ("粤测农〔2099〕1号#4",))
    new_case = EvalCase(
        case_id="C1", query="自造描述", expected_verdict="miss", expected_refs=(),
    )
    out = _recompute_row(row, new_case)
    assert out.predicted_verdict == "hit"  # 预测冻值不变
    assert out.citations_total == 3  # 引用冻值不变
    assert out.verdict_match is False  # 新预期 miss，预测 hit → 不匹配


# ---------------------------------------------------------------------------
# main() 护栏：registry 口径与分片重叠拒算（与 merge 同款，#70 终审教训）
# ---------------------------------------------------------------------------

def _shard(tmp_path: Path, name: str, *, registry_provided: bool) -> Path:
    """造一个分片目录：group_details.csv 含 1 行 C1 冻结预测＋metrics.json。"""
    directory = tmp_path / name
    directory.mkdir()
    row = _frozen_row(("粤测农〔2099〕1号#4",), ("粤测农〔2099〕1号#4",))
    with (directory / "group_details.csv").open(
            "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (directory / "metrics.json").write_text(
        json.dumps({"registry_provided": registry_provided}), encoding="utf-8")
    return directory


def _cases(tmp_path: Path) -> Path:
    """新标注表（定稿）：C1 应命中且标准答案与冻结预测一致。"""
    path = tmp_path / "cases.csv"
    path.write_text(
        "case_id,query,expected_verdict,expected_refs\n"
        "C1,自造描述,hit,粤测农〔2099〕1号#4\n",
        encoding="utf-8-sig")
    return path


def test_reevaluate_missing_registry_reports_na_not_zero(tmp_path: Path) -> None:
    """分片没带 registry → 重算引用错误率记 N/A，而非拿 citations_bad=0 冒充 0%。"""
    shard = _shard(tmp_path, "r1", registry_provided=False)
    final = tmp_path / "final"
    # 引用错误率 N/A → all_pass 不成立 → 退出码 1（不能靠缺失项过关）
    assert reevaluate_main(
        ["--cases", str(_cases(tmp_path)), "--inputs", str(shard),
         "--out", str(final)]) == 1
    metrics = json.loads((final / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics"]["citation_error_rate"] is None


def test_reevaluate_overlapping_shards_rejected(tmp_path: Path) -> None:
    """分片区间重叠（同组同案例两次）→ 拒绝重算（退出码 2），不带病出数。"""
    shard = _shard(tmp_path, "r1", registry_provided=True)
    assert reevaluate_main(
        ["--cases", str(_cases(tmp_path)), "--inputs", str(shard), str(shard),
         "--out", str(tmp_path / "final")]) == 2
