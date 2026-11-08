"""tests/test_eval_split.py — 分片跑批的切分（--offset）与合并（merge）链路。

承接 feat/eval-split：三人按案例区间分片跑 --pipeline real --ablate-all，
各产 group_details.csv，最后 merge 回全量重算三指标。本文件用 null 管线
（无 LLM）验证「切分 → 合并」的底层数据不丢、比率重算正确。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from eval.merge import main as merge_main
from eval.runner import main as runner_main


def _write_cases(path: Path, n: int) -> None:
    rows = ["case_id,query"] + [f"C{i},自造描述{i}" for i in range(1, n + 1)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8-sig")


def _group_rows(out: Path) -> list[dict[str, str]]:
    with (out / "group_details.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_offset_skips_leading_cases(tmp_path: Path) -> None:
    """--offset 跳过前 N 条，--limit 截取其后若干条。"""
    cases = tmp_path / "cases.csv"
    _write_cases(cases, 4)
    out = tmp_path / "r1"
    runner_main(
        ["--cases", str(cases), "--pipeline", "null",
         "--offset", "1", "--limit", "2", "--out", str(out)])
    rows = _group_rows(out)
    assert [r["case_id"] for r in rows] == ["C2", "C3"]
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["case_count"] == 2


def test_split_then_merge_reassembles_full(tmp_path: Path) -> None:
    """两片各半 → merge 拼回全量，主组 case_count 等于总和，比率按全量重算。"""
    cases = tmp_path / "cases.csv"
    _write_cases(cases, 4)
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    runner_main(["--cases", str(cases), "--pipeline", "null",
                 "--offset", "0", "--limit", "2", "--out", str(out1)])
    runner_main(["--cases", str(cases), "--pipeline", "null",
                 "--offset", "2", "--limit", "2", "--out", str(out2)])
    assert len(_group_rows(out1)) == 2 and len(_group_rows(out2)) == 2

    final = tmp_path / "final"
    merge_main(["--inputs", str(out1), str(out2), "--out", str(final)])
    merged = json.loads((final / "metrics.json").read_text(encoding="utf-8"))
    assert merged["case_count"] == 4
    # 两片都是 null 管线：verdict 全 miss，未标注 → 匹配率 N/A、召回 N/A。
    assert merged["metrics"]["match_rate"] is None
    assert merged["metrics"]["recall"] is None
    # 分片未带 registry → 引用错误率 N/A（宁缺勿错）。
    assert merged["metrics"]["citation_error_rate"] is None


def test_merge_missing_group_details_fails(tmp_path: Path) -> None:
    """输入目录缺 group_details.csv 时返回配置错误（退出码 2）。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert merge_main(["--inputs", str(empty), "--out", str(tmp_path / "out")]) == 2
