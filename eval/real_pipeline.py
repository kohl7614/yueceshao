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
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.dataset import EvalCase, normalize_ref
from eval.metrics import CasePrediction
from gen.citation import build_meta_table

__all__ = ["RealPipeline"]

logger = logging.getLogger(__name__)

# 「第四条」→「四」：meta_table 的 clause_no 带第/条前后缀，normalize_ref 需要裸序号
_CLAUSE_NUM = re.compile(r"^第?\s*([0-9一二三四五六七八九十百零〇两]+)\s*条?$")

# 占位文号：规则库采集时未拿到真实文号处，doc_no 是「无（…）」或「页面未载」。
# 它们不是真实文号、无法在 registry 核验，不应作为引用输出——否则引用错误率被占位符虚高
# （数据侧彻底版＝补全 440 条真实文号，见数据仓任务书）。
_PLACEHOLDER_DOC_NO_PREFIX = "无（"
_PLACEHOLDER_DOC_NO_MARKER = "页面未载"


#: 条款号缺失时的规范取值（与数据仓规则库/registry 导出同键）。
CLAUSE_UNNUMBERED = "未分条（原文无条款号）"


def _raw_clause(clause_no: str) -> str:
    """把「第四条」→「四」（去掉第/条前后缀，供 normalize_ref 归一到阿拉伯序号）。

    条款号为空时归一为 :data:`CLAUSE_UNNUMBERED`——索引里「未分条」文件的正文块
    ``clause`` 元数据是空串，若直接拼成 ``文号#`` 会与 registry 的规范键
    ``文号#未分条（原文无条款号）`` 对不上，导致每条这类引用都被判"无法核验"。
    实测（kohl 分片 36 条）：184 个空条款引用全部属于正文块，按规范名归一后
    184/184 可核验，是把引用错误率从 42.8% 拉回来的关键。
    """
    text = (clause_no or "").strip()
    if not text:
        return CLAUSE_UNNUMBERED
    match = _CLAUSE_NUM.match(text)
    return match.group(1) if match else text


def _build_llm_for_eval(config_path: Path) -> Any:
    """按本机 ``config.json`` 的 ``generation`` 段装配真实 LLM（OpenAI 兼容客户端）。

    评测口径要求「real ＝ 真检索 ＋ 真规则 ＋ 真 LLM」：不显式装配时
    :class:`app.real_components.RealComponents` 会退回 :class:`MockLLM`，三指标随即
    失去实验意义——故此处显式装配；key 不到位时**大声告警**而非静默降级。

    :param config_path: 本机 config.json 路径
    :return: 装配好的 LLM；配置段或 key 不到位时返回 ``None``（沿用 Mock）
    """
    from app.real_components import DEFAULT_ENV_PATH, QwenLLM, _load_config, read_env_file

    config = _load_config(config_path)
    section = config.get("generation") if isinstance(config, Mapping) else None
    if not isinstance(section, Mapping):
        logger.warning(
            "config.json 缺 generation 段：LLM 退回 MockLLM，本次指标不代表真实系统")
        return None
    api_key_env = str(section.get("api_key_env") or "DASHSCOPE_API_KEY")
    if not (os.environ.get(api_key_env) or read_env_file(DEFAULT_ENV_PATH).get(api_key_env)):
        logger.warning(
            "环境变量与 %s 均无 %s：LLM 退回 MockLLM，本次指标不代表真实系统",
            DEFAULT_ENV_PATH, api_key_env)
        return None
    logger.info(
        "LLM 装配：model=%s base_url=%s（key 来自环境变量/%s，不打印）",
        section.get("model"), section.get("base_url"), api_key_env)
    return QwenLLM(config)


def _is_placeholder_doc_no(doc_no: str) -> bool:
    """doc_no 是否为「文号缺失」占位符（非真实文号，不应作为引用输出）。

    真实文号形如 ``粤财农〔2020〕106号``；占位符为 ``无（页面未载/仅正文引用，不计）``、
    ``无（地方性法规）``、``页面未载`` 等。政府令/部令/国令（如 ``国令第498号``）是
    真实文号，只是格式不同，**不**视为占位符。
    """
    return doc_no.startswith(_PLACEHOLDER_DOC_NO_PREFIX) or _PLACEHOLDER_DOC_NO_MARKER in doc_no


class RealPipeline:
    """真实管线：项目描述 → 完整四段式报告 → 评测侧预测。

    :param config_path: 本机 config.json（索引/模型/数据仓路径）；None 用仓库根默认
    :param llm: 注入 LLM（测试用）；None 时按 config 装配（MockLLM 或 qwen 取决于装配方）
    :param switches: 消融开关快照（任务B）；None＝完整系统，之后可用 :meth:`configure` 覆盖
    """

    name = "real"

    def __init__(
        self,
        config_path: Path | str | None = None,
        llm: Any = None,
        switches: Any = None,
    ) -> None:
        # 局部 import：本模块被 eval.runner 以 --pipeline real 加载时才引入 app 重依赖；
        # CI（无索引/无 qwen key）跑其他 pipeline 不会因此崩。
        from app.real_components import DEFAULT_CONFIG_PATH, RealCheckupPipeline

        # ⚠ 主分支把 config_path=None 直接透传给 RealCheckupPipeline，而其内部
        # ``Path(config_path)`` 会 TypeError——故先归一到默认 config.json；
        # 再按 generation 段装配真实 LLM（否则 RealComponents 默认 MockLLM，三指标是假的）。
        target = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        if llm is None:
            llm = _build_llm_for_eval(target)
        self._pipeline = RealCheckupPipeline(target, llm=llm)
        if switches is not None:
            self.configure(switches)
        logger.info("RealPipeline 就绪（真实检索＋规则＋LLM 蕴含复核）")

    @property
    def components(self) -> Any:
        """内层 RealCheckupPipeline 的组件视图（runner 读 ``llm_backend`` 输出诚实文案）。

        runner 持有的是本包装器而非内层管线——不委托的话 ``getattr(pipeline,
        "components")`` 取不到，真跑会被误报成「LLM 退回 MockLLM」（2026-11-12
        kohl-v4 实跑暴露：973 次真调用却被标 mock，默认值方向反了）。
        """
        return self._pipeline.components

    def configure(self, switches: Any, trace: Any | None = None) -> None:
        """显式注入消融开关与 trace（任务B：runner 把 AblationSwitches 真正传给管线）。

        :param switches: :class:`eval.config.AblationSwitches` 快照
        :param trace: 与 runner 共享的 :class:`eval.gating.SwitchTrace`；None 时自建
        """
        from eval.gating import SwitchTrace

        self._pipeline.components.set_ablation(switches, trace or SwitchTrace())

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
            if _is_placeholder_doc_no(meta.doc_no):
                # 文号缺失占位符不是可核验引用，跳过（不影响召回，只降引用错误率）
                continue
            refs.setdefault(normalize_ref(meta.doc_no, _raw_clause(meta.clause_no)), None)
        return CasePrediction(
            verdict="hit" if report.negative_hits else "miss",
            refs=tuple(refs),
            citations=tuple(refs),
            text=result.rendered_text,
            degraded=result.degraded,
            attempts=result.attempts,
            entail_failed=result.entail_failed,
        )
