"""tests/test_metrics_forbidden_scan.py — 姿态禁语扫描器的上下文豁免回归。

全量四分片首跑（2026-11-19）发现两类假阳性：①报告开头「体检输入：…」回显申报人
描述，其原文措辞（「是否符合规定」「现场核查确认合规」）天然含禁语子串；②「符合
规定」四字的子串含「合规」。两者都是申报人/中性短语而非系统结论——姿态红线只约束
系统自己的话。豁免后真违规（系统侧「合规」「可以申报」）必须仍被抓住。
"""

from __future__ import annotations

from eval.metrics import _scan_forbidden

_C75_ECHO = (
    "体检输入：我是某县某镇涉农项目的报账经办人。"
    "请审核这种列支方式是否符合规定。\n\n"
    "① 适用政策清单\n· 建议明确说明资金拨付方式，确保符合财务规章制度。"
)

_C144_ECHO = (
    "体检输入：我们合作社计划新建一批粮食烘干点。"
    "选址经过县自然资源部门现场核查确认合规。\n\n"
    "① 适用政策清单\n· 该政策支持建设粮食烘干设施。"
)


def test_input_echo_exempt() -> None:
    """体检输入回显段整段豁免：申报人措辞里的「符合规定/确认合规」不算违规。"""
    assert _scan_forbidden(_C75_ECHO) == ()
    assert _scan_forbidden(_C144_ECHO) == ()


def test_system_side_statement_still_caught() -> None:
    """系统自己的话出现「合规」不因豁免而漏抓（回望只豁免「符合规定」子串）。"""
    text = "① 适用政策清单\n· 现场核查确认合规。\n· 本项目合规，可以申报。"
    assert "合规" in _scan_forbidden(text)
    assert "可以申报" in _scan_forbidden(text)


def test_fu_he_gui_ding_substring_not_hit_outside_echo() -> None:
    """「是否符合规定」的子串「合规」即使在系统段也不计（(?<!符) 回望豁免）。"""
    text = "④ 修改建议\n· 请审核这种列支方式是否符合规定。"
    assert "合规" not in _scan_forbidden(text)


def test_real_violation_baseline() -> None:
    """基线：真违规在免责声明/条款原文/输入回显三重豁免之外仍全部命中。"""
    text = (
        "体检输入：拟新建烘干点。\n\n"
        "② 负面清单命中提示\n· 条款原文：严禁合规票据以外的支出。"
        "\n· 该项目合规，可以申报。"
    )
    assert set(_scan_forbidden(text)) == {"合规", "可以申报"}
