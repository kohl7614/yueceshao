"""app/pipeline.py — 体检编排层：协议＋Mock 端到端实现（手册 §3.5 前置）。

职责：把已合并的各模块串成一次完整体检——

    查询抽取 → 检索命中 → 规则命中 → 负面蕴含复核
    → 生成（①③④段，run_with_retry 校验重试）→ 引用渲染 → 免责声明

其中**真实执行**的部分（不依赖数据仓/索引/API）：gen.schema 的结构解析与
防线、gen.validator 的四类校验与 A9 重试降级、gen.citation 的查表注入与
置信填充、render_report 的四段文本与免责声明强制附加。
**Mock 注入**的部分：检索命中对象、LLM 三段输出、②段规则命中——骨架阶段
以夹具代替，真实适配器（rag 检索接索引／rules 引擎接数据仓／qwen API）由
后续 PR 替换（CheckupPipeline 协议即替换点，接口不变）。

红线：Mock 夹具只用〔2099〕人造文号与自造条款原文（语料不入代码仓）；
编排全程不产生任何结论性措辞；免责声明由渲染层保证（check_disclaimer 复核）。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from gen.citation import build_meta_table, build_report, render_report
from gen.schema import (
    DISCLAIMER_TEXT,
    CheckupReport,
    parse_extraction,
)
from gen.validator import (
    check_disclaimer,
    run_with_retry,
)

logger = logging.getLogger(__name__)

# ---- Mock 夹具：自洽的检索命中（〔2099〕人造文号；原文自造）----
MOCK_HITS: tuple[dict, ...] = (
    {
        "chunk_id": "[C1]",
        "text": "帮扶资金主要用于产业道路、灌溉设施等生产性基础设施建设，"
                "不得用于楼堂馆所建设和偿还债务。",
        "metadata": {
            "doc_no": "粤测农〔2099〕1号", "clause_no": "第四条",
            "effective_date": "2099-01-01", "validity": "有效", "level": "省",
            "title": "测试用涉农资金统筹整合管理办法", "is_current": True,
        },
    },
    {
        "chunk_id": "[C2]",
        "text": "申报主体应当提交项目申报书、实施方案和村集体经济组织成员大会决议材料。",
        "metadata": {
            "doc_no": "无（申报通知未载文号）", "clause_no": "第三条",
            "effective_date": "2099-02-01", "validity": "有效", "level": "县",
            "title": "测试用申报指南", "is_current": True,
        },
    },
)

# Mock 的 LLM 输出（合法 JSON，能通过 parse＋四类校验；引语逐字取自上方原文）
MOCK_EXTRACTION_JSON = json.dumps({
    "subject_type": "村集体经济组织",
    "construction_items": ["产业道路", "灌溉设施"],
    "use_items": ["生产性基础设施建设"],
    "terms": [],
}, ensure_ascii=False)

MOCK_GENERATION_JSON = json.dumps({
    "policy_items": [
        {"citation": "[C1]",
         "summary": "该条款支持产业道路与灌溉设施类建设，与项目内容直接相关",
         "confidence": None},
    ],
    "missing_items": [
        {"citation": "[C2]", "name": "项目申报书", "note": "按申报指南建议核对材料清单"},
        {"citation": "[C2]", "name": "实施方案", "note": "按申报指南建议核对材料清单"},
    ],
    "suggestions": [
        {"text": "建议在描述中写明道路长度、灌溉覆盖面积等量化信息"},
    ],
}, ensure_ascii=False)

MOCK_ENTAILMENT_JSON = json.dumps([
    {"citation": "[C1]", "verdict": "不蕴含",
     "reason": "描述为生产性基础设施建设，未涉及该条款禁止的用途"},
], ensure_ascii=False)


# ---------------------------------------------------------------------------
# 协议与结果
# ---------------------------------------------------------------------------

@runtime_checkable
class CheckupPipeline(Protocol):
    """体检管线协议（后续真实适配器实现同一接口无缝替换 Mock）。"""

    name: str

    def run(self, description: str) -> PipelineResult:
        """输入项目描述，输出完整四段式报告（结构化＋文本）。"""
        ...


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """一次体检的完整产物。"""

    report: CheckupReport            # 结构化四段（置信已由系统侧填充）
    rendered_text: str               # 四段式文本（含免责声明）
    degraded: bool                   # 是否触发 A9 降级
    attempts: int                    # 生成侧实际尝试次数
    failed_sections: tuple[str, ...] = field(default=())
    elapsed_ms: int = 0
    disclaimer: str = DISCLAIMER_TEXT  # 响应显式携带（客户端无需解析文本）

    def to_api_dict(self) -> dict:
        """API 序列化（显式展开；pipeline 名由路由层组装响应时加入）。"""
        from dataclasses import asdict
        return {
            "report": asdict(self.report),
            "rendered_text": self.rendered_text,
            "degraded": self.degraded,
            "attempts": self.attempts,
            "failed_sections": list(self.failed_sections),
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Mock 端到端实现
# ---------------------------------------------------------------------------

class MockCheckupPipeline:
    """Mock 适配器的端到端编排：全链路真实执行，外部依赖以夹具注入。

    generation_outputs：可注入多版生成 JSON——首版故意不合法时可用于演示/测试
    A9 重试与降级路径（默认单版合法输出，一次通过）。
    """

    name = "mock-skeleton"

    def __init__(
        self,
        hits: tuple[dict, ...] = MOCK_HITS,
        extraction_json: str = MOCK_EXTRACTION_JSON,
        generation_outputs: tuple[str, ...] = (MOCK_GENERATION_JSON,),
        entailment_json: str = MOCK_ENTAILMENT_JSON,
        negative_hits: tuple = (),
    ) -> None:
        self._hits = hits
        self._extraction_json = extraction_json
        self._generation_outputs = list(generation_outputs)
        self._entailment_json = entailment_json
        self._negative_hits = negative_hits

    # —— Mock 外部依赖（后续 PR 换成真实适配器）——
    def _extract(self, description: str) -> str:
        return self._extraction_json

    def _retrieve(self, description: str, extraction) -> tuple[dict, ...]:
        return self._hits

    def _rule_scan(self, description: str) -> tuple:
        return self._negative_hits

    def _entail_review(self, description: str, hits) -> str:
        return self._entailment_json

    def _generate(self, description: str, extraction_json: str, context_blocks: str,
                  feedback: str | None) -> str:
        """按轮次弹出预置输出；重试反馈写入日志（Mock 不真的改写输出）。"""
        if feedback:
            logger.info("第 %d 版生成未过校验，按反馈重试：%s",
                        self._served, feedback.splitlines()[0])
        if self._served < len(self._generation_outputs):
            out = self._generation_outputs[self._served]
            self._served += 1
            return out
        return self._generation_outputs[-1]

    # —— 编排（全真实链路）——
    def run(self, description: str) -> PipelineResult:
        started = time.monotonic()
        self._served = 0

        # 1. 查询抽取（结构解析真实校验）
        extraction = parse_extraction(self._extract(description))

        # 2. 检索命中 → 元数据表＋原文映射
        hits = self._retrieve(description, extraction)
        table = build_meta_table(hits)
        chunk_texts = {h["chunk_id"]: h["text"] for h in hits}

        # 3. 规则命中（②段；Mock 阶段由夹具给出）
        negative_hits = self._rule_scan(description)

        # 4. 负面蕴含复核（结果并入②段；骨架阶段夹具已合法，真实适配器 PR 接入）

        # 5. 生成＋校验重试（A9 全真实：四类检查、反馈重试、降级）
        outcome = run_with_retry(
            lambda feedback: self._generate(description, self._extraction_json, "", feedback),
            allowed=set(table.keys()),
            chunk_texts=chunk_texts,
        )

        # 6. 引用渲染（查表注入＋置信填充，全真实）
        report = build_report(description, outcome.delivered,
                              negative_hits=negative_hits, table=table)
        degraded_sections = outcome.failed_sections if outcome.degraded else ()
        text = render_report(report, table, degraded_sections=degraded_sections)

        # 7. 免责声明最后防线复核（渲染层唯一出口，此处双保险）
        issues = check_disclaimer(text)
        if issues:
            logger.error("渲染成品缺少免责声明（不应发生）：%s", issues[0].message)

        elapsed = int((time.monotonic() - started) * 1000)
        return PipelineResult(
            report=report,
            rendered_text=text,
            degraded=outcome.degraded,
            attempts=outcome.attempts,
            failed_sections=outcome.failed_sections,
            elapsed_ms=elapsed,
        )
