"""tests/test_real_ablation_gating.py — 消融开关真接线（任务B · 通道B · TDD 先红后绿）。

验收口径（docs/消融接线设计-20261108.md）：开关关掉的路径可被测试观测到——
实现函数调用次数为 0 ＋ trace 记 bypass ＋ 预测行为按设计变化。

全部 mock：不碰数据仓、不跑真实索引、不用真 LLM；夹具只用自造文字与〔2099〕
人造文号（红线⑤⑥）。

TDD 纪律：本文件先于实现提交——首批运行必须失败（红），实现后转绿。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from app.real_components import RealCheckupPipeline, RealComponents

from eval.config import SWITCH_NAMES, AblationSwitches
from eval.dataset import EvalCase
from eval.gating import SwitchTrace, gated
from eval.metrics import CasePrediction, EvalThresholds, evaluate_case
from eval.real_pipeline import RealPipeline
from eval.runner import main as runner_main
from eval.runner import run_group
from rag.retriever import Hit, RetrievalResult

# ---------------------------------------------------------------- 假组件

DENSE_HITS = [
    Hit(
        chunk_id="1-1",
        text="自造条款原文甲：专项资金的使用与披露要求（与任何真实文号无关）",
        score=0.9,
        distance=0.1,
        metadata={"doc_no": "粤测农〔2099〕1号", "clause": "第一条",
                  "validity": "现行有效", "is_current": True},
    ),
]

GOOD_GENERATION_JSON = json.dumps({
    "policy_items": [
        {"citation": "[C1]",
         "summary": "该条款支持产业道路与灌溉设施类建设，与项目内容直接相关",
         "confidence": None},
    ],
    "missing_items": [
        {"citation": "[C1]", "name": "项目申报书", "note": "按申报指南建议核对材料清单"},
    ],
    "suggestions": [
        {"text": "建议在描述中写明道路长度、灌溉覆盖面积等量化信息"},
    ],
}, ensure_ascii=False)

BAD_QUOTE_GENERATION_JSON = json.dumps({
    "policy_items": [
        {"citation": "[C1]",
         "summary": "按「这句话绝对不在原始条款文本之中」执行相关要求",
         "confidence": None},
    ],
    "missing_items": [
        {"citation": "[C1]", "name": "项目申报书", "note": "按申报指南建议核对材料清单"},
    ],
    "suggestions": [
        {"text": "建议在描述中写明道路长度、灌溉覆盖面积等量化信息"},
    ],
}, ensure_ascii=False)


class FakeRetriever:
    """记录调用次数的假稠密检索器。"""

    def __init__(self, hits=None) -> None:
        self.calls = 0
        self._hits = list(hits or [])

    def retrieve(self, query: str) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(query=query, hits=self._hits)


class FakeStore:
    def valid_records(self):
        return []


class FakeMatcher:
    """记录调用次数的假规则匹配器（store 供载体清单读取，返回空载体）。"""

    def __init__(self) -> None:
        self.calls = 0
        self.store = FakeStore()

    def match(self, description: str):
        from rules.engine import MatchResult

        self.calls += 1
        return MatchResult(hits=(), description_terms=())


class CountingLLM:
    """记录调用的假 LLM；generate 输出可注入。"""

    def __init__(self, generation_outputs=None) -> None:
        self.extract_calls = 0
        self.entail_calls = 0
        self.generate_calls = 0
        self.generate_args: list[str] = []
        self._gen = list(generation_outputs) if generation_outputs else [GOOD_GENERATION_JSON]

    def extract(self, description: str) -> str:
        self.extract_calls += 1
        return json.dumps(
            {"subject_type": "其他", "construction_items": ["自造条目"], "use_items": [],
             "terms": []}, ensure_ascii=False)

    def entail_review(self, description: str, hits, feedback=None) -> str:
        self.entail_calls += 1
        return "[]"

    def generate(self, description: str, extraction_json: str, context_blocks: str,
                 feedback=None) -> str:
        self.generate_calls += 1
        self.generate_args.append(extraction_json)
        return self._gen[min(self.generate_calls, len(self._gen)) - 1]


def _pipeline(switches=None, *, retriever=None, matcher=None, llm=None,
              retriever_plain=None):
    """注入假组件装配真实管线；switches 非 None 时显式注入（不经 contextvar）。"""
    components = RealComponents(
        Path("nonexistent-config.json"),
        retriever=retriever if retriever is not None else FakeRetriever(DENSE_HITS),
        matcher=matcher if matcher is not None else FakeMatcher(),
        llm=llm if llm is not None else CountingLLM(),
        retriever_plain=retriever_plain,
    )
    if switches is not None:
        components.set_ablation(switches, SwitchTrace())
    return RealCheckupPipeline(components=components), components


# ---------------------------------------------------------------- R1 gating source


class TestGatedSource:
    def test_explicit_source_skips_and_records(self) -> None:
        calls: list[int] = []
        trace = SwitchTrace()
        switches = AblationSwitches.default().with_overrides(dense_retrieval=False)

        @gated("dense_retrieval", stage="unit.x",
               source=lambda: (switches, trace))
        def fn() -> int:
            calls.append(1)
            return 42

        assert fn() is None
        assert calls == []
        assert trace.was_bypassed("dense_retrieval")

    def test_explicit_source_enabled_runs(self) -> None:
        calls: list[int] = []
        trace = SwitchTrace()

        @gated("dense_retrieval", stage="unit.x",
               source=lambda: (AblationSwitches.default(), trace))
        def fn() -> int:
            calls.append(1)
            return 42

        assert fn() == 42
        assert calls == [1]
        assert not trace.was_bypassed("dense_retrieval")

    def test_fallback_used_when_disabled(self) -> None:
        trace = SwitchTrace()
        switches = AblationSwitches.default().with_overrides(rule_scan=False)

        @gated("rule_scan", stage="unit.y",
               fallback=lambda query: [], source=lambda: (switches, trace))
        def fn(query: str) -> list[str]:
            raise AssertionError("关闭时不应执行")

        assert fn("q") == []
        assert trace.was_bypassed("rule_scan")


# ---------------------------------------------------------------- R2–R7 真实管线


class TestRealPipelineSwitches:
    def test_dense_retrieval_off_skips_retrieval(self) -> None:
        retriever = FakeRetriever(DENSE_HITS)
        pipeline, components = _pipeline(
            AblationSwitches.default().with_overrides(dense_retrieval=False),
            retriever=retriever)
        pipeline.run("自造项目描述")
        assert retriever.calls == 0, "关掉稠密检索后不得调用检索器"
        switches, trace = components.ablation_state()
        assert trace.was_bypassed("dense_retrieval")
        assert all(
            hit["metadata"].get("origin") != "retrieval"
            for hit in components._plan("自造项目描述").hits)

    def test_rule_scan_off_skips_matcher(self) -> None:
        matcher = FakeMatcher()
        pipeline, components = _pipeline(
            AblationSwitches.default().with_overrides(rule_scan=False),
            matcher=matcher)
        result = pipeline.run("自造项目描述")
        assert matcher.calls == 0, "关掉规则扫描后不得调用匹配器"
        _, trace = components.ablation_state()
        assert trace.was_bypassed("rule_scan")
        assert result.report.negative_hits == ()

    def test_entail_review_off_skips_llm_entail(self) -> None:
        llm = CountingLLM()
        pipeline, components = _pipeline(
            AblationSwitches.default().with_overrides(entail_review=False), llm=llm)
        pipeline.run("自造项目描述")
        assert llm.entail_calls == 0, "关掉蕴含复核后不得调用 LLM 复核"
        assert llm.generate_calls >= 1, "生成段不受影响仍应执行"
        _, trace = components.ablation_state()
        assert trace.was_bypassed("entail_review")

    def test_temporal_filter_off_uses_plain_retriever(self) -> None:
        temporal_on = FakeRetriever(DENSE_HITS)
        plain = FakeRetriever(DENSE_HITS)
        pipeline, components = _pipeline(
            AblationSwitches.default().with_overrides(temporal_filter=False),
            retriever=temporal_on, retriever_plain=plain)
        pipeline.run("自造项目描述")
        assert plain.calls == 1 and temporal_on.calls == 0, "关时效过滤应换用无时效检索器"
        _, trace = components.ablation_state()
        assert trace.was_bypassed("temporal_filter")

    def test_temporal_filter_on_uses_default_retriever(self) -> None:
        temporal_on = FakeRetriever(DENSE_HITS)
        plain = FakeRetriever(DENSE_HITS)
        pipeline, _components = _pipeline(
            None, retriever=temporal_on, retriever_plain=plain)
        pipeline.run("自造项目描述")
        assert temporal_on.calls == 1 and plain.calls == 0

    def test_citation_validator_off_lets_bad_quote_through(self) -> None:
        llm = CountingLLM([BAD_QUOTE_GENERATION_JSON])
        pipeline, components = _pipeline(
            AblationSwitches.default().with_overrides(citation_validator=False),
            llm=llm)
        result = pipeline.run("自造项目描述")
        assert result.degraded is False, "校验器关闭后坏引语不应被拦截降级"
        assert "这句话绝对不在原始条款文本之中" in result.rendered_text
        _, trace = components.ablation_state()
        assert trace.was_bypassed("citation_validator")

    def test_citation_validator_on_catches_bad_quote(self) -> None:
        llm = CountingLLM([BAD_QUOTE_GENERATION_JSON])
        pipeline, _components = _pipeline(None, llm=llm)
        result = pipeline.run("自造项目描述")
        assert result.degraded is True, "校验器开启时坏引语必须被拦下（重试耗尽→降级）"

    def test_query_decomposition_off_skips_extract(self) -> None:
        llm = CountingLLM()
        pipeline, components = _pipeline(
            AblationSwitches.default().with_overrides(query_decomposition=False),
            llm=llm)
        pipeline.run("自造项目描述")
        assert llm.extract_calls == 0, "关掉查询分解后不得调用 LLM 抽取"
        assert llm.generate_calls >= 1, "生成段仍应执行"
        assert llm.generate_args and llm.generate_args[0] != "", (
            "生成段应收到明确的空抽取占位，而不是报错")
        _, trace = components.ablation_state()
        assert trace.was_bypassed("query_decomposition")


# ---------------------------------------------------------------- R8 runner 接线


class RecordingPipeline:
    name = "fake"

    def __init__(self) -> None:
        self.configured: list = []

    def configure(self, switches, trace) -> None:
        self.configured.append((switches, trace))

    def predict(self, case: EvalCase) -> CasePrediction:
        return CasePrediction(verdict="miss", refs=(), citations=(), text="")


class TestRunnerWiring:
    def test_run_group_passes_switches_to_pipeline(self) -> None:
        pipeline = RecordingPipeline()
        switches = AblationSwitches.default().with_overrides(dense_retrieval=False)
        group = run_group(
            [EvalCase(case_id="C1", query="自造描述")], switches, pipeline,
            EvalThresholds(), None)
        assert pipeline.configured, "run_group 必须把开关真正传给管线"
        assert pipeline.configured[0][0] is switches
        assert group.label == "w/o:dense_retrieval"

    def test_real_pipeline_configure_propagates(self) -> None:
        real, components = _pipeline(None)
        rp = object.__new__(RealPipeline)
        rp._pipeline = real
        switches = AblationSwitches.default().with_overrides(rule_scan=False)
        trace = SwitchTrace()
        rp.configure(switches, trace)
        assert components.ablation_state() == (switches, trace)

    def test_ablate_all_group_labels_follow_new_switches(self, tmp_path) -> None:
        cases = tmp_path / "cases.csv"
        cases.write_text("case_id,query\nC1,自造描述\n", encoding="utf-8-sig")
        out = tmp_path / "runs"
        runner_main(["--cases", str(cases), "--pipeline", "null",
                     "--ablate-all", "--out", str(out)])
        with (out / "ablation_matrix.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["group"] for row in rows] == [
            "full"] + [f"w/o:{name}" for name in SWITCH_NAMES]


# ---------------------------------------------------------------- R9 config 口径


class TestSwitchConfigRedefinition:
    def test_switch_names_redefined(self) -> None:
        assert SWITCH_NAMES == (
            "temporal_filter", "dense_retrieval", "rule_scan", "entail_review",
            "query_decomposition", "citation_validator")

    @pytest.mark.parametrize("name", ["rerank", "hybrid_retrieval", "negative_hard_constraint"])
    def test_removed_switch_names_rejected(self, name: str) -> None:
        with pytest.raises(ValueError):
            AblationSwitches.from_mapping({name: False})

    def test_removed_env_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            AblationSwitches.from_env(environ={"YUECESHAO_ABLATION_RERANK": "0"})

    def test_runner_rejects_removed_switch(self, tmp_path) -> None:
        cases = tmp_path / "cases.csv"
        cases.write_text("case_id,query\nC1,自造描述\n", encoding="utf-8-sig")
        rc = runner_main(["--cases", str(cases), "--pipeline", "null",
                          "--ablate", "rerank", "--out", str(tmp_path / "runs")])
        assert rc == 2, "已移除的旧开关名必须当场报错（退出码 2=配置错误）"

    def test_example_json_loads(self) -> None:
        switches = AblationSwitches.from_json_file(
            Path(__file__).resolve().parent.parent / "eval" / "ablation.example.json")
        assert switches.is_enabled("dense_retrieval")


# ---------------------------------------------------------------- 明细三列


class TestDetailColumns:
    def test_case_prediction_carries_degraded_columns(self) -> None:
        case = EvalCase(case_id="C1", query="自造描述", expected_verdict="miss")
        prediction = CasePrediction(
            verdict="miss", refs=(), citations=(), text="",
            degraded=True, attempts=3, entail_failed=True)
        outcome = evaluate_case(case, prediction, None)
        row = outcome.to_row()
        assert row["degraded"] == 1
        assert row["attempts"] == 3
        assert row["entail_failed"] == 1

    def test_null_prediction_leaves_detail_columns_blank(self) -> None:
        case = EvalCase(case_id="C1", query="自造描述", expected_verdict="miss")
        outcome = evaluate_case(
            case, CasePrediction(verdict="miss", refs=(), citations=(), text=""), None)
        row = outcome.to_row()
        assert row["degraded"] == ""
        assert row["attempts"] == ""
        assert row["entail_failed"] == ""
