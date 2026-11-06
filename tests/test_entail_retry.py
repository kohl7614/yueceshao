"""tests/test_entail_retry.py — entail 节点重试 + 姿态禁语净化（TDD 先红后绿）。

方向4 验证踩到两个 gap：
1. qwen 在 reason 里写「违规」被姿态校验拒收 → 加「净化」（确定性替换成中性口径）；
2. entail 只做一次调用＋解析、失败即降级 → 对齐 A9 加「带反馈重试 ≤2」。

红线：测试夹具用〔2099〕人造文号与自造条款原文。
"""

from __future__ import annotations

import json

from app.graph import OrchestrationComponents, build_checkup_graph
from app.pipeline import MOCK_GENERATION_JSON, MOCK_HITS

from gen.schema import HIT_SUSPECT, sanitize_entailment_text

VALID = "我村拟申请帮扶资金，翻新改建村委会旧瓦房为会议室和接待室。"

EXTRACTION = json.dumps({
    "subject_type": "村集体经济组织",
    "construction_items": ["翻新改建村委会"],
    "use_items": [],
    "terms": [],
}, ensure_ascii=False)

# 姿态违规的蕴含 JSON（reason 含「违规」→ 净化后应可解析）
BAD_POSTURE = json.dumps([
    {"citation": "[C1]", "verdict": "蕴含", "reason": "描述明确违规，落入条款禁止范围"},
], ensure_ascii=False)

# 合规的蕴含 JSON（中性口径「落入条款禁止范围」）
CLEAN_ENTAIL = json.dumps([
    {"citation": "[C1]", "verdict": "蕴含", "reason": "描述落入条款禁止范围"},
], ensure_ascii=False)

# 非法 JSON（净化不救、重试也不救，用于测「重试耗尽降级」）
MALFORMED = "这不是合法的 JSON 数组"


def _multi_entail(outputs: tuple[str, ...]):
    """按调用次数逐次返回蕴含 JSON；耗尽后重复最后一个。"""
    served: list[int] = [0]

    def entail(description, hits, feedback=None):
        index = served[0]
        served[0] += 1
        return outputs[min(index, len(outputs) - 1)]

    return entail


def _components(entail) -> OrchestrationComponents:
    return OrchestrationComponents(
        extract=lambda d: EXTRACTION,
        retrieve=lambda d, e: MOCK_HITS,
        rule_scan=lambda d: (),
        entail_review=entail,
        generate=lambda d, ej, cb, fb: MOCK_GENERATION_JSON,
    )


class TestSanitizeEntailment:
    def test_replaces_posture_phrases(self) -> None:
        assert sanitize_entailment_text("描述违规") == "描述落入条款禁止范围"
        assert sanitize_entailment_text("描述不合规") == "描述落入条款禁止范围"
        assert sanitize_entailment_text("描述合规") == "描述符合允许范围"

    def test_order_handles_bu_hegui_before_hegui(self) -> None:
        # 「不合规」含「合规」子串，须先替换长的，否则会残留成「不符合允许范围」
        assert sanitize_entailment_text("不合规") == "落入条款禁止范围"

    def test_no_replacement_when_clean(self) -> None:
        assert sanitize_entailment_text("落入条款禁止范围") == "落入条款禁止范围"


class TestEntailRetry:
    def test_posture_is_sanitized_before_parse(self) -> None:
        """reason 含「违规」→ 净化后首次解析即成功（无需重试），②段出疑似命中。"""
        final = build_checkup_graph(
            _components(_multi_entail((BAD_POSTURE,)))
        ).invoke({"description": VALID})
        assert final.get("entail_failed") is not True
        hits = final["negative_hits"]
        assert len(hits) == 1
        assert hits[0].hit_type == HIT_SUSPECT
        assert hits[0].citation == "[C1]"

    def test_malformed_json_retries_then_succeeds(self) -> None:
        """首版非法 JSON → 带反馈重试 → 第二版合法 → ②段出疑似命中。"""
        final = build_checkup_graph(
            _components(_multi_entail((MALFORMED, CLEAN_ENTAIL)))
        ).invoke({"description": VALID})
        assert final.get("entail_failed") is not True
        assert len(final["negative_hits"]) == 1

    def test_retry_exhausted_degrades_to_rule_only(self) -> None:
        """连续 3 次非法 JSON → 重试耗尽 → ②段退化为仅规则命中 + 显式标注。"""
        final = build_checkup_graph(
            _components(_multi_entail((MALFORMED, MALFORMED, MALFORMED)))
        ).invoke({"description": VALID})
        assert final["entail_failed"] is True
        assert final["negative_hits"] == ()
        assert "LLM 蕴含复核未完成" in final["rendered_text"]

    def test_network_failure_degrades_without_retry(self) -> None:
        """非 SchemaError（如网络异常）→ 立即降级，不重试。"""
        def broken_entail(d, h, fb=None):
            raise RuntimeError("LLM 超时")

        final = build_checkup_graph(_components(broken_entail)).invoke({"description": VALID})
        assert final["entail_failed"] is True
        assert final["negative_hits"] == ()
