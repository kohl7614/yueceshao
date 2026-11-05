"""app/graph.py — LangGraph 编排（A12 定稿落地）：把已合并模块串成状态图。

编排拓扑（与 README 架构图一致）：

    parse → [retrieve ‖ rule_scan]（并行）→ entail → generate → render → END

- 节点＝已合并模块的真实调用（gen.schema 解析防线／gen.citation 查表与置信／
  gen.validator 四类校验＋A9 重试降级＋render 免责声明——**无一是 mock**）；
  外部依赖（检索命中／LLM 输出／规则命中）经 :class:`OrchestrationComponents`
  回调注入——骨架阶段为夹具，真实适配器实现同一组回调即完成替换；
- A9 重试子循环封装在 generate 节点内（gen.validator.run_with_retry 已独立测试），
  降级路由由 render 节点按 outcome.degraded 处理（降级段标题附显式标注）；
  图级条件边留给真实 LLM 节点接入后的编排扩展（A12 后续）；
- 并行汇合：retrieve 与 rule_scan 在同一超步并发，分别写 ``hits``／
  ``negative_hits`` 不同 key，无 channel 冲突；entail 在两路都完成后触发
  （LangGraph 默认汇合语义）。

Mock 夹具全部〔2099〕人造文号＋自造条款原文（红线⑤⑥）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from gen.citation import build_meta_table, build_report, render_report
from gen.schema import (
    CheckupReport,
    QueryExtraction,
    parse_extraction,
)
from gen.validator import RetryOutcome, check_disclaimer, run_with_retry

logger = logging.getLogger(__name__)


class CheckupState(TypedDict, total=False):
    """图状态：节点间传递的全部产物（内存编译不序列化，允许承载 dataclass）。"""

    description: str
    extraction: QueryExtraction
    extraction_json: str
    hits: tuple[dict, ...]
    meta_table: dict[str, Any]
    chunk_texts: dict[str, str]
    negative_hits: tuple
    outcome: RetryOutcome
    report: CheckupReport
    rendered_text: str
    degraded: bool
    attempts: int
    failed_sections: tuple[str, ...]


@dataclass(frozen=True)
class OrchestrationComponents:
    """编排的外部依赖回调集（骨架＝Mock 夹具；真实适配器逐一替换）。

    签名约定（与 app.pipeline.MockCheckupPipeline 的同名私有方法一致）：
    - ``extract(description) -> LLM JSON 字符串``
    - ``retrieve(description, extraction) -> 命中 dict 元组（chunk_id/text/metadata）``
    - ``rule_scan(description) -> NegativeHit 元组（②段，确定性模板渲染）``
    - ``entail_review(description, hits) -> 蕴含复核 LLM JSON 字符串``
    - ``generate(description, extraction_json, context_blocks, feedback) -> 生成 LLM JSON``
    """

    extract: Callable[[str], str]
    retrieve: Callable[[str, Any], tuple[dict, ...]]
    rule_scan: Callable[[str], tuple]
    entail_review: Callable[[str, Any], str]
    generate: Callable[[str, str, str, str | None], str]


def build_checkup_graph(components: OrchestrationComponents) -> Any:
    """编译体检编排图（返回 CompiledGraph；``invoke({"description": ...})`` 触发）。"""

    def parse_node(state: CheckupState) -> dict:
        extraction_json = components.extract(state["description"])
        return {"extraction": parse_extraction(extraction_json), "extraction_json": extraction_json}

    def retrieve_node(state: CheckupState) -> dict:
        hits = components.retrieve(state["description"], state["extraction"])
        table = build_meta_table(hits)
        return {
            "hits": hits,
            "meta_table": table,
            "chunk_texts": {h["chunk_id"]: h["text"] for h in hits},
        }

    def rule_scan_node(state: CheckupState) -> dict:
        return {"negative_hits": components.rule_scan(state["description"])}

    def entail_node(state: CheckupState) -> dict:
        components.entail_review(state["description"], state["hits"])
        return {}  # 骨架阶段蕴含结果由 components 侧并入 negative_hits；保留节点为拓扑占位

    def generate_node(state: CheckupState) -> dict:
        outcome = run_with_retry(
            lambda feedback: components.generate(
                state["description"], state["extraction_json"], "", feedback
            ),
            allowed=set(state["meta_table"].keys()),
            chunk_texts=state["chunk_texts"],
        )
        return {"outcome": outcome}

    def render_node(state: CheckupState) -> dict:
        outcome = state["outcome"]
        report = build_report(
            state["description"],
            outcome.delivered,
            negative_hits=state["negative_hits"],
            table=state["meta_table"],
        )
        degraded_sections = outcome.failed_sections if outcome.degraded else ()
        text = render_report(report, state["meta_table"], degraded_sections=degraded_sections)
        if check_disclaimer(text):  # 免责声明最后防线复核（渲染层保证，此处兜底）
            logger.error("渲染成品缺少免责声明（不应发生）")
        return {
            "report": report,
            "rendered_text": text,
            "degraded": outcome.degraded,
            "attempts": outcome.attempts,
            "failed_sections": outcome.failed_sections,
        }

    builder = StateGraph(CheckupState)
    builder.add_node("parse", parse_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rule_scan", rule_scan_node)
    builder.add_node("entail", entail_node)
    builder.add_node("generate", generate_node)
    builder.add_node("render", render_node)

    builder.add_edge(START, "parse")
    builder.add_edge("parse", "retrieve")     # 并行分支一：检索
    builder.add_edge("parse", "rule_scan")    # 并行分支二：规则（README 架构：两路并行）
    builder.add_edge("retrieve", "entail")    # 汇合后负面复核
    builder.add_edge("rule_scan", "entail")
    builder.add_edge("entail", "generate")
    builder.add_edge("generate", "render")
    builder.add_edge("render", END)
    return builder.compile()
