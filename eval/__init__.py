"""eval 包：评测与消融（团队手册 3.4）。

本 PR（``feat/eval-config``）只交付配置层与观测点：

* :mod:`eval.config` —— 六组消融开关的集中配置（唯一来源，不得旁路硬编码）
* :mod:`eval.gating` —— 开关的运行时观测点（bypass 记录，供测试与报告使用）

跑批、指标与管线（``eval.dataset`` / ``eval.metrics`` / ``eval.pipeline`` /
``eval.runner``）在 PR2（``feat/eval-runner``）落地，届时本文件会补充相应导出。
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
    # config
    "SWITCH_DOCS",
    "SWITCH_NAMES",
    "AblationSwitches",
    # gating
    "BypassEvent",
    "SwitchTrace",
    "ablation_context",
    "current_switches",
    "current_trace",
    "gated",
]
