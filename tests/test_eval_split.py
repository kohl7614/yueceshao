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
from eval.metrics import CaseOutcome
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


def test_merge_missing_registry_reports_na_not_zero(tmp_path: Path) -> None:
    """分片漏传 --registry（metrics.json 标 registry_provided=False）时，
    合并出的引用错误率必须是 N/A，而非拿 citations_bad=0 误报成 0%（#70 终审发现）。"""
    directory = tmp_path / "r1"
    directory.mkdir()
    outcome = CaseOutcome(
        case_id="C1", expected_verdict=None, predicted_verdict="miss",
        verdict_match=None, expected_refs=(), predicted_refs=(), refs_hit=(),
        recall_mode="n/a", citations_total=1, citations_bad=0, forbidden_hits=(),
    )
    row = {"group": "full", **outcome.to_row()}
    with (directory / "group_details.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (directory / "metrics.json").write_text(
        json.dumps({"registry_provided": False}), encoding="utf-8")

    final = tmp_path / "final"
    merge_main(["--inputs", str(directory), "--out", str(final)])
    merged = json.loads((final / "metrics.json").read_text(encoding="utf-8"))
    assert merged["metrics"]["citation_error_rate"] is None  # N/A，而非 0


def test_merge_overlapping_shards_rejected(tmp_path: Path) -> None:
    """分片区间重叠（同组同案例出现两次）→ 拒绝合并（退出码 2），不带病出数。

    比率类指标的分子分母都按行数走：重复案例被静默双计则指标全歪。这是分发文档
    「--offset/--limit 必须严格照表填，否则会重复/漏跑案例」的机械兜底。
    """
    cases = tmp_path / "cases.csv"
    _write_cases(cases, 2)
    out1 = tmp_path / "r1"
    runner_main(["--cases", str(cases), "--pipeline", "null", "--out", str(out1)])
    final = tmp_path / "final"
    # 同一目录传两次＝区间完全重叠
    assert merge_main(["--inputs", str(out1), str(out1), "--out", str(final)]) == 2
    assert not (final / "metrics.json").exists()


class TestVintageFingerprintGuard:
    """标注 vintage 指纹护栏：分片用不同版本标注表跑批时 merge 拒并。"""

    def _shard(self, tmp_path: Path, name: str, fingerprint: str | None) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "group_details.csv").write_text(
            "group,case_id,expected_verdict,predicted_verdict,verdict_match,expected_refs,"
            "predicted_refs,refs_hit,recall_mode,citations_total,citations_bad,"
            "forbidden_hits,notes,degraded,attempts,entail_failed\n"
            f"full,{name}-1,hit,hit,1,粤测农〔2099〕1号#4,粤测农〔2099〕1号#4,"
            "粤测农〔2099〕1号#4,ref-level,1,0,,,0,1,0\n",
            encoding="utf-8-sig")
        meta = {"registry_provided": True}
        if fingerprint is not None:
            meta["cases_fingerprint"] = fingerprint
        (d / "metrics.json").write_text(
            json.dumps(meta), encoding="utf-8")
        return d

    def test_mixed_fingerprints_refused(self, tmp_path: Path, capsys) -> None:
        a = self._shard(tmp_path, "a", "aaaa1111bbbb2222")
        b = self._shard(tmp_path, "b", "cccc3333dddd4444")
        assert merge_main(["--inputs", str(a), str(b), "--out",
                           str(tmp_path / "out")]) == 2
        assert "标注 vintage 不一致" in capsys.readouterr().err

    def test_same_fingerprints_merge(self, tmp_path: Path) -> None:
        a = self._shard(tmp_path, "a", "aaaa1111bbbb2222")
        b = self._shard(tmp_path, "b", "aaaa1111bbbb2222")
        assert merge_main(["--inputs", str(a), str(b), "--out",
                           str(tmp_path / "out")]) == 0

    def test_all_unknown_not_blocked(self, tmp_path: Path) -> None:
        """老产物（无指纹字段）之间不拦——保持对历史分片的兼容。"""
        a = self._shard(tmp_path, "a", None)
        b = self._shard(tmp_path, "b", None)
        assert merge_main(["--inputs", str(a), str(b), "--out",
                           str(tmp_path / "out")]) == 0

    def test_runner_writes_fingerprint(self, tmp_path: Path) -> None:
        """runner 真跑 null 管线后，metrics.json 的 cases_fingerprint 与标注表
        sha256 前 16 位一致（runner↔merge 的字段名契约由本测试钉住——
        预审发现初版是空转断言，只测了 hashlib 本身）。"""
        import hashlib

        cases = tmp_path / "cases.csv"
        cases.write_text(
            "案例编号,改编后申报场景描述_体检输入,预期命中预警,历史适用条款_文号及条款号\n"
            "T1,拟用资金建设楼堂馆所,应命中,粤测农〔2099〕1号#4\n",
            encoding="utf-8-sig")
        out = tmp_path / "out"
        rc = runner_main(["--cases", str(cases), "--pipeline", "null",
                          "--out", str(out)])
        assert rc == 0 or rc == 1  # 指标未达线返回 1 也算跑完
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        expect = hashlib.sha256(cases.read_bytes()).hexdigest()[:16]
        assert metrics["cases_fingerprint"] == expect
