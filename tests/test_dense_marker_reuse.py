"""tests/test_dense_marker_reuse.py — 规则命中复用稠密标记前的原文覆盖校验（issue #52）。

问题：`_compute_plan` 里规则命中若与稠密命中落在同一 (文件序号, 条序号)，旧实现直接复用
稠密标记、不追加规则块。但真实索引同一 (文件序号, 条序号) 下可能有**多个不同原文的分块**
（实测数千组），此时②段引用 `[Cn]` 指向的块正文与该规则条文对不上——条款号对、原文不对。

修法：复用前校验「该稠密块正文**包含**规则条款原文」，不包含就分配独立标记并追加规则块。

真数据实测（60 条案例描述）：修复前规则命中 2327 条、复用稠密标记 15 次、其中 **4 次**
引用块不含规则原文；修复后复用 11 次、不一致 **0** 次。

红线：夹具全为人造措辞（〔2099〕文号 + 自造句），已核对真实语料 0 命中。
"""

from __future__ import annotations

from pathlib import Path

from app.real_components import ORIGIN_RULES, RealComponents

from rag.retriever import Hit, RetrievalResult
from rules.engine import NegRuleMatcher
from rules.loader import records_from_dicts
from rules.schema import RuleStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"

# 合成条款（真实语料 0 命中，见模块 docstring）
RULE_TEXT = "项目资金不得用于购置公务用车。"
OTHER_TEXT = "项目资金不得用于修建楼堂馆所。"
DESC = "我村申报产业项目，拟购置公务用车一辆"


def _rule(doc_index: int = 1, clause_no: str = "四", clause_text: str = RULE_TEXT) -> dict:
    """合成规则记录（人造文号，G04 组，含禁止措辞 → 同时满足载体判据）。"""
    return {
        "条款原文逐字": clause_text,
        "出处文号+条款号": f"粤测农〔2099〕{doc_index}号 {clause_no}",
        "条款号": clause_no,
        "承接列举": False,
        "层级": "省",
        "所在文件效力状态": "有效",
        "等价组建议": {"组号": "G04", "组名": "测试组", "组类型": "锚词组"},
        "负面清单载体文件": False,
        "溯源_文件序号": doc_index,
        "溯源_文号": f"粤测农〔2099〕{doc_index}号",
        "溯源_标题": "测试用办法（人造数据）",
        "溯源_发布机关": "测试机关",
        "溯源_发布日期": "2099-01-01",
        "溯源_类别": "资金管理办法",
        "组裁定": "接受",
    }


def _dense_hit(chunk_id: str, text: str, *, serial: int = 1, clause_index: int = 4) -> Hit:
    """假稠密命中：与规则同 (文件序号, 条序号)，正文由调用方决定是否覆盖规则原文。"""
    return Hit(
        chunk_id=chunk_id,
        text=text,
        score=0.9,
        distance=0.1,
        metadata={
            "doc_no": "粤测农〔2099〕1号",
            "clause": "第四条",
            "clause_index": clause_index,
            "serial": serial,
            "title": "测试用办法（人造数据）",
            "category": "资金管理办法",
            "validity": "有效",
            "is_current": True,
            "has_doc_no": True,
            "structure": "clause",
        },
    )


class FakeRetriever:
    def __init__(self, hits: tuple[Hit, ...]) -> None:
        self._hits = hits

    def retrieve(self, query: str, **_kwargs) -> RetrievalResult:
        return RetrievalResult(query=query, hits=list(self._hits), where={"is_current": True})


def _plan(rules: list[dict], hits: tuple[Hit, ...]):
    store = RuleStore(records_from_dicts(rules))
    components = RealComponents(
        CONFIG_PATH, retriever=FakeRetriever(hits), matcher=NegRuleMatcher(store))
    return components, components._plan(DESC)


class TestReuseRequiresTextCoverage:
    def test_no_reuse_when_dense_block_lacks_rule_text(self) -> None:
        """稠密块正文不含规则原文 → 不得复用它的标记，必须追加独立规则块。"""
        _, plan = _plan([_rule()], (_dense_hit("1-0004", OTHER_TEXT),))

        block_texts = {h["chunk_id"]: h["text"] for h in plan.hits}
        dense_citation = plan.negative_hits[0].citation
        assert dense_citation != "[C1]", "稠密块不含规则原文却复用了它的标记"
        assert RULE_TEXT in block_texts[dense_citation], (
            f"②段引用的块 {dense_citation} 里找不到规则原文")

    def test_reuse_when_dense_block_covers_rule_text(self) -> None:
        """稠密块正文包含规则原文 → 复用同一标记，不重复追加块。"""
        _, plan = _plan([_rule()], (_dense_hit("1-0004", "（款项说明）" + RULE_TEXT),))

        assert plan.negative_hits[0].citation == "[C1]", "覆盖成立时应复用稠密标记"
        rule_blocks = [h for h in plan.hits if h["metadata"]["origin"] == ORIGIN_RULES]
        assert not rule_blocks, "覆盖成立时不应再追加重复的规则块"

    def test_picks_covering_block_among_same_article_siblings(self) -> None:
        """同一 (文件序号, 条序号) 有多块时，选中**正文覆盖规则原文**的那块。"""
        hits = (
            _dense_hit("1-0004", OTHER_TEXT, clause_index=4),
            _dense_hit("1-0004b", RULE_TEXT + "（补充说明）", clause_index=4),
        )
        _, plan = _plan([_rule()], hits)

        assert plan.negative_hits[0].citation == "[C2]", "应选中覆盖规则原文的那块（第二块）"
        block_texts = {h["chunk_id"]: h["text"] for h in plan.hits}
        assert RULE_TEXT in block_texts["[C2]"]

    def test_every_citation_block_contains_its_clause_terms(self) -> None:
        """不变量：②段每条引用所指的块正文，必须含该条命中词表里的**条款侧术语**。

        这是 issue #52 的可执行版验收：块对不上条款时，matched_words 的「条款侧」词
        在块正文里就找不到。
        """
        cases = [
            ([_rule()], (_dense_hit("1-0004", OTHER_TEXT),), DESC),
            ([_rule()], (_dense_hit("1-0004", RULE_TEXT),), DESC),
            ([_rule()], (), DESC),
            (
                [_rule(doc_index=2, clause_no="七", clause_text="专项经费不得用于发放奖金津贴。")],
                (_dense_hit("2-0007", "（无关正文）", serial=2, clause_index=7),),
                "拟用专项经费发放奖金津贴",
            ),
        ]
        for rules, hits, description in cases:
            store = RuleStore(records_from_dicts(rules))
            components = RealComponents(
                CONFIG_PATH, retriever=FakeRetriever(hits), matcher=NegRuleMatcher(store))
            plan = components._plan(description)
            block_texts = {h["chunk_id"]: h["text"] for h in plan.hits}
            assert plan.negative_hits, f"夹具应产生②段命中：{description}"
            for hit in plan.negative_hits:
                block = block_texts[hit.citation]
                clause_terms = [clause for (_desc, clause) in hit.matched_words if clause]
                assert clause_terms, "命中词应至少有一对条款侧术语"
                assert any(term in block for term in clause_terms), (
                    f"引用 {hit.citation} 的块正文找不到条款侧术语 {clause_terms}"
                    f"（块：{block[:20]}…）")
