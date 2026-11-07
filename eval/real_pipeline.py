"""eval/real_pipeline.py — 真实管线适配器（真检索＋真规则＋真 LLM 蕴含复核）。

把 :class:`app.real_components.RealCheckupPipeline` 适配为 eval 的 :class:`Pipeline`
协议，使评测跑批可以在**真实 Chroma 索引＋qwen**上跑三指标全量回溯——这是召回
15.38%→80% 验证的正式跑批入口（对比基线＝`--pipeline rules` 的规则引擎单路）。

与 :mod:`eval.rules_pipeline` 的关系：rules_pipeline 是「关掉检索与 LLM」的规则独立
基线；本模块是完整系统（检索＋规则＋LLM 蕴含复核＋生成校验）——两者对比即得出
「LLM 蕴含复核带来的召回增量」。

红线：refs/citations 全部来自渲染层 meta_table（溯源自数据仓 registry 的字段），
本模块不生成、不改写任何文号（README 红线①②）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from eval.dataset import EvalCase, normalize_ref
from eval.metrics import CasePrediction
from gen.citation import build_meta_table

__all__ = ["RealPipeline"]

logger = logging.getLogger(__name__)

# 「第四条」→「四」：meta_table 的 clause_no 带第/条前后缀，normalize_ref 需要裸序号
_CLAUSE_NUM = re.compile(r"^第?\s*([0-9一二三四五六七八九十百零〇两]+)\s*条?$")


def _raw_clause(clause_no: str) -> str:
    """把「第四条」→「四」（去掉第/条前后缀，供 normalize_ref 归一到阿拉伯序号）。"""
    text = (clause_no or "").strip()
    match = _CLAUSE_NUM.match(text)
    return match.group(1) if match else text


class RealPipeline:
    """真实管线：项目描述 → 完整四段式报告 → 评测侧预测。

    :param config_path: 本机 config.json（索引/模型/数据仓路径）；None 用仓库根默认
    :param llm: 注入 LLM（测试用）；None 时按 config 装配（MockLLM 或 qwen 取决于装配方）
    """

    name = "real"

    def __init__(self, config_path: Path | str | None = None, llm: Any = None) -> None:
        # 局部 import：本模块被 eval.runner 以 --pipeline real 加载时才引入 app 重依赖；
        # CI（无索引/无 qwen key）跑其他 pipeline 不会因此崩。
        from app.real_components import RealCheckupPipeline

        self._pipeline = RealCheckupPipeline(config_path, llm=llm)
        logger.info("RealPipeline 就绪（真实检索＋规则＋LLM 蕴含复核）")

    def predict(self, case: EvalCase) -> CasePrediction:
        """对一条案例跑完整真实管线，产出评测侧预测。

        ②段命中（规则命中 ∪ LLM 蕴含命中）→ verdict=hit；refs 取命中的
        ``文号#条款号`` 归一标识（来自 meta_table 溯源字段，非手写文号）。
        """
        result = self._pipeline.run(case.query)
        report = result.report
        # run() 内部已按 description 缓存命中计划，retrieve 复用同一计划不重算
        hits = self._pipeline.components.retrieve(case.query, None)
        table = build_meta_table(hits)
        refs: dict[str, None] = {}
        for hit in report.negative_hits:
            meta = table.get(hit.citation)
            if meta is None or not meta.doc_no:
                continue
            refs.setdefault(normalize_ref(meta.doc_no, _raw_clause(meta.clause_no)), None)
        return CasePrediction(
            verdict="hit" if report.negative_hits else "miss",
            refs=tuple(refs),
            citations=tuple(refs),
            text=result.rendered_text,
        )
