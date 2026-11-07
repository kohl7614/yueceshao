"""tests/test_eval_real_pipeline.py — RealPipeline 适配器（真管线 → 评测预测）。

用假 RealCheckupPipeline 注入测 predict 的裁决/引用映射；不依赖真实索引/模型/key，
CI 可跑。红线：夹具用〔2099〕人造文号。
"""

from __future__ import annotations

from app.pipeline import PipelineResult

from eval.dataset import EvalCase
from eval.real_pipeline import RealPipeline, _raw_clause
from gen.schema import CheckupReport, NegativeHit


class TestRawClause:
    def test_strips_di_tiao_prefix_suffix(self) -> None:
        assert _raw_clause("第四条") == "四"
        assert _raw_clause("十八") == "十八"
        assert _raw_clause("未分条（原文无条款号）") == "未分条（原文无条款号）"


def _fake_pipeline(hit_type: str = "疑似命中（建议对照原文）"):
    """构造假 RealCheckupPipeline：report 带一条②段命中，retrieve 返回 [C1] 映射。"""
    hits = (
        {"chunk_id": "[C1]", "text": "自造原文", "metadata": {
            "doc_no": "粤测农〔2099〕1号", "clause_no": "第四条",
            "validity": "有效", "is_current": True}},
    )

    class FakeComponents:
        def retrieve(self, description, extraction):
            return hits

    class FakePipeline:
        components = FakeComponents()

        def run(self, description):
            report = CheckupReport(
                project_summary="",
                policy_items=(),
                negative_hits=(
                    NegativeHit(citation="[C1]", hit_type=hit_type, matched_words=()),),
                missing_items=(),
                suggestions=(),
            )
            return PipelineResult(
                report=report, rendered_text="免责声明：本报告仅为申报前自查辅助参考",
                degraded=False, attempts=1)

    return FakePipeline()


class TestRealPipeline:
    def test_verdict_hit_and_refs_normalized(self) -> None:
        pipeline = object.__new__(RealPipeline)
        pipeline._pipeline = _fake_pipeline()
        pred = pipeline.predict(EvalCase(case_id="C1", query="描述"))
        assert pred.verdict == "hit"
        assert pred.refs == ("粤测农〔2099〕1号#4",)
        assert pred.citations == ("粤测农〔2099〕1号#4",)

    def test_verdict_miss_when_no_negative_hits(self) -> None:
        def empty_run(description):
            report = CheckupReport(
                project_summary="", policy_items=(), negative_hits=(), missing_items=(),
                suggestions=())
            return PipelineResult(
                report=report, rendered_text="免责声明：…", degraded=False, attempts=1)

        class EmptyPipeline:
            class _C:
                def retrieve(self, description, extraction):
                    return ()

            components = _C()
            run = staticmethod(empty_run)

        pipeline = object.__new__(RealPipeline)
        pipeline._pipeline = EmptyPipeline()
        pred = pipeline.predict(EvalCase(case_id="C2", query="描述"))
        assert pred.verdict == "miss"
        assert pred.refs == ()
