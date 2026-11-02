"""rules/ — 负面清单结构化规则库与规则引擎（纯 Python 确定性匹配，无 LLM）。

- schema.py：规则数据结构（RuleRecord）与规则库容器（RuleStore）；
- loader.py：从数据仓条款归并 json 加载（效力过滤、等价组索引）。
红线遵守：本包不包含政策语料原文；数据仓位置读 config.json。
"""

from rules.loader import load_config, load_records, load_store_from_config, records_from_dicts
from rules.schema import (
    EFFECT_INVALIDATED, EFFECT_PENDING, EFFECT_VALID, GROUP_NONE, LEVEL_RANK,
    RuleRecord, RuleStore, display_priority_key,
)

__all__ = [
    "RuleRecord", "RuleStore", "display_priority_key",
    "load_config", "load_records", "load_store_from_config", "records_from_dicts",
    "EFFECT_VALID", "EFFECT_PENDING", "EFFECT_INVALIDATED", "GROUP_NONE", "LEVEL_RANK",
]
