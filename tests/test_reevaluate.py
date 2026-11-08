"""tests/test_reevaluate.py — 数据①定稿后的免费重算链路。

验证 :func:`eval.reevaluate._recompute_row`：预测侧冻值不变、预期侧换新标准答案，
召回率从「旧占位符 → 0%」变成「新标准答案 → 真实值」，无需重跑管线。
"""

from __future__ import annotations

from eval.dataset import EvalCase
from eval.metrics import CaseOutcome
from eval.reevaluate import _recompute_row


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
