"""eval 包：评测与消融（团队手册 3.4）。

子模块：
* :mod:`eval.config` —— 六组消融开关的集中配置（唯一来源，不得旁路硬编码）
* :mod:`eval.gating` —— 开关的运行时观测点（bypass 记录，供测试与报告使用）
"""

from __future__ import annotations

from eval.config import (
    SWITCH_DOCS,
    SWITCH_NAMES,
    AblationSwitches,
)
from eval.gating import (
    BypassEvent,
    SwitchTrace,
    ablation_context,
    current_switches,
    current_trace,
    gated,
)

__all__ = [
    "SWITCH_DOCS",
    "SWITCH_NAMES",
    "AblationSwitches",
    "BypassEvent",
    "SwitchTrace",
    "ablation_context",
    "current_switches",
    "current_trace",
    "gated",
]
