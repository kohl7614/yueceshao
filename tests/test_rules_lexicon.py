"""tests/test_rules_lexicon.py — 禁止用途词典扩充的契约测试（TDD 先红后绿）。

覆盖两批扩充（见 06_过程档案/词典覆盖分析/词典扩充与组裁定建议清单-20261106.md）：
- 批 1 归并扩词：现有概念组补近义词（挪用截留套取/耕地保护/分红理财/垫付垫资）；
- 批 2 新开程序性禁止概念组：仅收「肯定式违规词」（做了被禁止的事，可关键词共现），
  不收「否定式/中性词」（未公示/未入库/招标/入股 等——中性词会误伤合规负对照）。

红线遵守：本文件不使用真实政策语料原文、不引用真实文号。
"""

from __future__ import annotations

from rules.engine import NegRuleMatcher
from rules.lexicon import FORBIDDEN_USE_CONCEPTS
from rules.loader import records_from_dicts
from rules.schema import RuleStore

_ALL_TERMS = {t for terms in FORBIDDEN_USE_CONCEPTS.values() for t in terms}


def _record(clause_text: str, doc_index: int = 1, group_id: str = "未分组") -> dict:
    """人造规则记录（不引用真实文号）。"""
    return {
        "条款原文逐字": clause_text,
        "出处文号+条款号": f"〔测试样例〕X-{doc_index:03d}号 第一条",
        "条款号": "一",
        "承接列举": False,
        "层级": "省",
        "所在文件效力状态": "有效",
        "等价组建议": {"组号": group_id, "组名": group_id, "组类型": "锚词组（建议稿，人工裁定）"},
        "负面清单载体文件": False,
        "溯源_文件序号": doc_index,
        "溯源_文号": f"〔测试样例〕X-{doc_index:03d}号",
        "溯源_标题": f"测试样例文件{doc_index}（人造数据）",
        "溯源_发布机关": "测试样例机关",
        "溯源_发布日期": "2024-11-01",
        "溯源_类别": "测试样例类别",
        "组裁定": "接受（人造等价组）",
    }


class TestBatch1MergeWords:
    """批 1：现有概念组补近义词，不新增维度。"""

    def test_misappropriation_terms_extended(self) -> None:
        assert {"滞留", "克扣", "骗取", "虚报", "冒领"} <= set(
            FORBIDDEN_USE_CONCEPTS["挪用截留套取"]
        )

    def test_cropland_terms_extended(self) -> None:
        assert {"占用耕地", "造林"} <= set(FORBIDDEN_USE_CONCEPTS["耕地保护与非粮化"])

    def test_dividend_terms_extended(self) -> None:
        assert "兑付" in FORBIDDEN_USE_CONCEPTS["分红理财"]

    def test_advance_terms_extended(self) -> None:
        assert "代垫" in FORBIDDEN_USE_CONCEPTS["垫付垫资"]


class TestBatch2ProceduralGroups:
    """批 2：新开程序性禁止概念组（仅肯定式违规词）。"""

    def test_procurement_group_exists(self) -> None:
        assert {"化整为零", "串通投标", "转包", "支解发包"} <= set(
            FORBIDDEN_USE_CONCEPTS["招投标程序"]
        )

    def test_funds_custody_group_exists(self) -> None:
        assert {"坐支", "私存", "账外账"} <= set(
            FORBIDDEN_USE_CONCEPTS["资金存放与支付"]
        )

    def test_receipt_group_exists(self) -> None:
        assert "白头单" in FORBIDDEN_USE_CONCEPTS["票据与账务"]

    def test_asset_misappropriation_group_exists(self) -> None:
        assert {"侵占", "私分", "侵吞"} <= set(FORBIDDEN_USE_CONCEPTS["资产侵占"])

    def test_duplicate_claim_group_exists(self) -> None:
        assert {"重复申报", "虚增"} <= set(FORBIDDEN_USE_CONCEPTS["重复申报"])

    def test_levy_group_exists(self) -> None:
        assert {"摊派", "违规收费"} <= set(FORBIDDEN_USE_CONCEPTS["摊派收费"])


class TestExcludedWords:
    """排除词回归守卫：中性/否定式词不得入库（会误伤合规负对照）。"""

    def test_neutral_words_excluded(self) -> None:
        for word in ("入股", "招标", "采购", "公示", "项目库", "绩效", "验收", "专户"):
            assert word not in _ALL_TERMS, f"中性词「{word}」不应进词典（会误伤合规负对照）"

    def test_high_weight_trigger_excluded(self) -> None:
        # 「补助」「补贴」是高权重触发词（负对照 N1 写作纪律点名），不得新增进词典
        assert "补助" in _ALL_TERMS  # 存量已在「奖金津贴福利」，保留
        assert "补助标准" not in _ALL_TERMS  # G54 暂缓，不得新增


class TestProceduralMatching:
    """批 2 维度确实能被引擎命中（肯定式违规词 ↔ 条款侧同概念共现）。"""

    def test_procurement_fragmentation_hit(self) -> None:
        clause = "采购人不得将应当公开招标的项目化整为零或者以其他任何方式规避公开招标。"
        store = RuleStore(records_from_dicts([
            _record(clause, 1, "招投标程序"),
        ]))
        result = NegRuleMatcher(store).match("拟将三百万元工程化整为零拆成三笔小额采购。")
        assert len(result.hits) == 1
        assert "化整为零" in result.hits[0].matched_description_terms

    def test_funds_custody_hit(self) -> None:
        store = RuleStore(records_from_dicts([
            _record("项目资金应当专户管理，不得坐支现金。", 2, "资金存放与支付"),
        ]))
        result = NegRuleMatcher(store).match("项目款到账后拟直接坐支用于日常开销。")
        assert len(result.hits) == 1
        assert "坐支" in result.hits[0].matched_description_terms

    def test_compliant_procurement_no_hit(self) -> None:
        """合规表述（公开招标，无违规词）不因新增维度误报。"""
        store = RuleStore(records_from_dicts([
            _record("采购人不得将应当公开招标的项目化整为零规避公开招标。", 3, "招投标程序"),
        ]))
        result = NegRuleMatcher(store).match("项目依法公开招标，整体发包，三家以上比价。")
        assert result.hits == ()
