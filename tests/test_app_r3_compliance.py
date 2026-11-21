"""tests/test_app_r3_compliance.py — M1-R3 安全与合规：限流＋输入上限＋AI 生成标识。

验收（路线图 R3）：超频被拒（429＋Retry-After）；description ≤ 2000 字；
报告含 AI 生成标识（显式行插在免责声明之前——免责声明保持末行的红线不变；
隐式标识＝响应字段 ai_generated=true）。注入点在 API worker（服务端渲染层），
评测管线产物不受影响。
"""

from __future__ import annotations

import time

from app.main import (
    AI_GENERATED_LINE,
    DISCLAIMER_SENTINEL,
    MAX_DESCRIPTION_CHARS,
    create_app,
)
from app.pipeline import PipelineResult
from app.ratelimit import TokenBucketLimiter
from fastapi.testclient import TestClient

from gen.schema import DISCLAIMER_TEXT, CheckupReport

VALID = {"description": (
    "我村拟申请帮扶资金 60 万元，用于修建 2 公里产业道路和配套灌溉设施，"
    "由村集体经济组织作为实施主体，资金按项目进度拨付使用。"
)}


def _wait_done(http: TestClient, poll_url: str, *, timeout_s: float = 10) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        task = http.get(poll_url).json()
        if task["status"] in ("done", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError("任务超时未到终态")


def _stub_result(description: str) -> PipelineResult:
    report = CheckupReport(
        project_summary=description, policy_items=(), negative_hits=(),
        missing_items=(), suggestions=(), uncovered_sections=(),
    )
    return PipelineResult(
        report=report,
        rendered_text=f"体检输入：{description}\n\n免责声明：{DISCLAIMER_TEXT}",
        degraded=False, attempts=1,
    )


# ==================== 输入长度上限 ====================

def test_description_over_2000_chars_422() -> None:
    http = TestClient(create_app())
    long_desc = "我村拟申请帮扶资金修建产业道路。" + "补" * MAX_DESCRIPTION_CHARS
    r = http.post("/api/checkup", json={"description": long_desc})
    assert r.status_code == 422
    assert "2000" in r.json()["detail"][0]["msg"]


def test_description_exactly_2000_chars_accepted() -> None:
    http = TestClient(create_app(rate_limit_capacity=5))
    desc = "我村拟申请帮扶资金修建产业道路。"
    desc = desc + "补" * (MAX_DESCRIPTION_CHARS - len(desc))
    assert len(desc) == MAX_DESCRIPTION_CHARS
    r = http.post("/api/checkup", json={"description": desc})
    assert r.status_code == 202


# ==================== 限流（HTTP 契约） ====================

def test_rate_limit_429_with_retry_after() -> None:
    http = TestClient(create_app(rate_limit_capacity=2))
    assert http.post("/api/checkup", json=VALID).status_code == 202
    assert http.post("/api/checkup", json=VALID).status_code == 202
    r3 = http.post("/api/checkup", json=VALID)
    assert r3.status_code == 429
    assert "5 次" in r3.json()["detail"]
    retry_after = r3.headers.get("retry-after")
    assert retry_after is not None and int(retry_after) >= 1


# ==================== 限流（TokenBucketLimiter 单元） ====================

def test_limiter_refills_over_time() -> None:
    now = [1000.0]
    limiter = TokenBucketLimiter(capacity=2, refill_seconds=100.0, now_fn=lambda: now[0])
    assert limiter.allow("ip1") and limiter.allow("ip1")
    assert not limiter.allow("ip1")            # 桶空
    now[0] += 50.0                              # 半个周期 → 回充 1 枚
    assert limiter.allow("ip1")
    assert not limiter.allow("ip1")


def test_limiter_keys_independent() -> None:
    limiter = TokenBucketLimiter(capacity=1, refill_seconds=3600.0)
    assert limiter.allow("ip1")
    assert not limiter.allow("ip1")
    assert limiter.allow("ip2")                 # 另一 IP 不受影响
    assert limiter.allow("ip3")


def test_limiter_retry_after_math() -> None:
    now = [2000.0]
    limiter = TokenBucketLimiter(capacity=2, refill_seconds=100.0, now_fn=lambda: now[0])
    limiter.allow("ip1")
    limiter.allow("ip1")
    now[0] += 10.0                              # 回充 0.2 枚 → 还差 0.8 枚 = 40s
    assert limiter.retry_after_seconds("ip1") == 40
    assert limiter.retry_after_seconds("fresh-ip") == 0


def test_limiter_purges_idle_buckets() -> None:
    now = [3000.0]
    limiter = TokenBucketLimiter(capacity=2, refill_seconds=100.0, now_fn=lambda: now[0])
    limiter.allow("old-ip")
    now[0] += 250.0                             # 超过 2×refill → 清理
    limiter.allow("new-ip")                     # 触发 purge
    now[0] += 1.0
    assert limiter.allow("old-ip")              # 桶已被清 → 按满桶处理（=新桶）


# ==================== AI 生成标识（显式＋隐式） ====================

def test_ai_label_explicit_line_before_disclaimer() -> None:
    """done 的渲染文本含 AI 标识行，且免责声明仍为末行（红线不变）。"""
    http = TestClient(create_app())
    task_id = http.post("/api/checkup", json=VALID).json()["task_id"]
    task = _wait_done(http, f"/api/tasks/{task_id}")
    assert task["status"] == "done"
    text = task["result"]["rendered_text"]
    assert AI_GENERATED_LINE in text
    assert text.rstrip().endswith(f"免责声明：{DISCLAIMER_TEXT}")
    assert text.index(AI_GENERATED_LINE) < text.rindex("免责声明：")


def test_ai_label_implicit_field_in_result() -> None:
    """响应结构化字段 ai_generated=true（隐式标识，机器可读）。"""
    http = TestClient(create_app())
    task_id = http.post("/api/checkup", json=VALID).json()["task_id"]
    task = _wait_done(http, f"/api/tasks/{task_id}")
    assert task["result"]["ai_generated"] is True


def test_ai_line_inserted_without_disclaimer_marker_appended() -> None:
    """无免责声明标记的文本（不可能出现在 done 产物，防御性）→ 标识行追加末尾。"""
    from app.main import _insert_ai_line
    text = _insert_ai_line("纯文本无标记")
    assert text == "纯文本无标记\n\n" + AI_GENERATED_LINE


# ==================== 免责声明哨兵不受注入影响 ====================

def test_sentinel_check_unchanged_by_ai_injection() -> None:
    """哨兵仍比对管线产物（注入前的文本）——AI 行缺失不触发，免责缺失仍 failed。"""

    class _NoDisclaimer:
        name = "no-disclaimer"

        def run(self, description: str) -> PipelineResult:
            result = _stub_result(description)
            return PipelineResult(
                report=result.report,
                rendered_text=f"体检输入：{description}",  # 无免责声明
                degraded=False, attempts=1,
            )

    http = TestClient(create_app(_NoDisclaimer()))  # type: ignore[arg-type]
    task_id = http.post("/api/checkup", json=VALID).json()["task_id"]
    task = _wait_done(http, f"/api/tasks/{task_id}")
    assert task["status"] == "failed"
    assert task["error"] == "服务内部校验失败"


# ==================== 现役断言的常量一致性 ====================

def test_ai_line_contains_no_forbidden_phrases() -> None:
    """标识行本身不触碰姿态禁语（红线③：无「合规/可以申报」类表述）。"""
    for phrase in ("合规", "可以申报", "准予"):
        assert phrase not in AI_GENERATED_LINE
    assert DISCLAIMER_SENTINEL  # 哨兵常量非空（防手滑改成空串让哨兵恒过）
