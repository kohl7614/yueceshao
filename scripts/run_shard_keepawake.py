"""scripts/run_shard_keepawake.py — 分片跑批启动器（跑批期间阻止系统休眠）。

背景：第二轮跑批在 23:06 于第二消融组中途被外部终止（日志无崩溃痕迹、退出码 127），
时间点与整机休眠一致。本启动器在跑批线程上声明 ES_SYSTEM_REQUIRED，
Windows 不进入**自动空闲睡眠**；进程退出（含崩溃/被杀）后系统自动恢复原策略，
无常驻副作用。

注意边界：ES_SYSTEM_REQUIRED 只拦「无人操作的自动休眠」——合上盖子、按电源
按钮、开始菜单手动睡眠仍然照常生效，跑批期间请勿做这些操作。

用法（在仓库根目录）::

    python scripts/run_shard_keepawake.py -- <eval.runner 的全部参数>
"""

from __future__ import annotations

import ctypes
import subprocess
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set_keepawake(on: bool) -> bool:
    """声明/解除系统保持唤醒（仅对当前线程生效，进程退出即失效）。

    :return: API 是否成功；返回 0 表示系统拒绝了该请求。
    """
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


def main(argv: list[str]) -> int:
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not _set_keepawake(True):
        # 声明失败＝跑批期间系统仍可能自动休眠、中途白跑——宁可不开跑。
        print(
            "错误：SetThreadExecutionState 声明防休眠失败（返回 0），"
            "跑批期间系统仍可能自动休眠。已阻止启动；请检查系统电源策略后重试。",
            file=sys.stderr,
        )
        return 1
    try:
        return subprocess.call([sys.executable, "-m", "eval.runner", *argv])
    finally:
        _set_keepawake(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
