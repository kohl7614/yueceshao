"""tests/test_async_tasks.py — M1-R2 异步任务化：提交/轮询契约与任务表行为。

契约：POST /api/checkup → 202＋task_id＋poll_url（同步契约废弃）；GET
/api/tasks/{id} → queued/running/done/failed；failed.error 只含安全文案
（内部细节不泄给客户端，红线⑤的 API 形态）；任务 TTL 1 小时惰性清理；
单 worker 串行（共享命中计划/CPU 重排不并发）。
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from app.main import create_app
from app.pipeline import MockCheckupPipeline, PipelineResult
from app.real_components import RealCheckupPipeline
from app.tasks import TaskStore
from fastapi.testclient import TestClient

VALID = {"description": (
    "我村拟申请帮扶资金 60 万元，用于修建 2 公里产业道路和配套灌溉设施，"
    "由村集体经济组织作为实施主体，资金按项目进度拨付使用。"
)}


def _wait_terminal(http: TestClient, poll_url: str, *, timeout_s: float = 10) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        task = http.get(poll_url).json()
        if task["status"] in ("done", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError("任务超时未到终态")


# ==================== 提交/轮询契约 ====================

def test_submit_returns_202_with_poll_contract() -> None:
    http = TestClient(create_app())
    r = http.post("/api/checkup", json=VALID)
    assert r.status_code == 202
    body = r.json()
    assert set(body) == {"task_id", "status", "poll_url", "estimated_seconds"}
    assert body["status"] == "queued"
    assert body["poll_url"] == f"/api/tasks/{body['task_id']}"
    assert body["estimated_seconds"] == 5  # Mock 缺省提示值


def test_poll_unknown_task_404() -> None:
    http = TestClient(create_app())
    r = http.get("/api/tasks/000000000000")
    assert r.status_code == 404 and "重新提交" in r.json()["detail"]


def test_poll_running_shape_before_done() -> None:
    """轮询响应始终携带完整状态字段（终态前 result/error 为 null）。"""
    http = TestClient(create_app())
    task_id = http.post("/api/checkup", json=VALID).json()["task_id"]
    task = _wait_terminal(http, f"/api/tasks/{task_id}")
    assert set(task) == {"task_id", "status", "created_at", "started_at",
                         "finished_at", "elapsed_ms", "result", "error"}
    assert task["status"] == "done"
    assert task["started_at"] and task["finished_at"]
    assert isinstance(task["elapsed_ms"], int)


# ==================== 失败路径：安全文案，不泄内部细节 ====================

class _BoomPipeline:
    name = "boom"

    def run(self, description: str) -> PipelineResult:
        raise RuntimeError("内部路径 /secret/config.json 读取失败")


def test_failed_error_is_safe_no_leak() -> None:
    """管线异常 → failed＋安全文案；内部细节（路径等）不进响应。"""
    http = TestClient(create_app(_BoomPipeline()))  # type: ignore[arg-type]
    task_id = http.post("/api/checkup", json=VALID).json()["task_id"]
    task = _wait_terminal(http, f"/api/tasks/{task_id}")
    assert task["status"] == "failed"
    assert task["error"] == "体检执行失败，请稍后重试"
    assert "secret" not in task["error"]
    assert task["result"] is None


def _stub_result(description: str) -> PipelineResult:
    from gen.schema import CheckupReport

    report = CheckupReport(
        project_summary=description, policy_items=(), negative_hits=(),
        missing_items=(), suggestions=(), uncovered_sections=(),
    )
    return PipelineResult(report=report, rendered_text=f"体检输入：{description}",
                          degraded=False, attempts=1)


def test_sentinel_failure_marks_task_failed() -> None:
    """渲染产物缺免责声明 → failed＋「服务内部校验失败」（原同步 500 兜底口径）。"""

    class _NoDisclaimer:
        name = "no-disclaimer"

        def run(self, description: str) -> PipelineResult:
            return _stub_result(description)  # rendered_text 无「免责声明：」

    http = TestClient(create_app(_NoDisclaimer()))  # type: ignore[arg-type]
    task_id = http.post("/api/checkup", json=VALID).json()["task_id"]
    task = _wait_terminal(http, f"/api/tasks/{task_id}")
    assert task["status"] == "failed"
    assert task["error"] == "服务内部校验失败"


# ==================== 预计耗时提示 ====================

def test_estimated_seconds_from_pipeline() -> None:
    """管线自带 estimated_seconds 时如实透传（真实管线 120s）。"""
    assert RealCheckupPipeline.estimated_seconds == 120

    class _SlowStub(MockCheckupPipeline):
        estimated_seconds = 30

    http = TestClient(create_app(_SlowStub()))
    body = http.post("/api/checkup", json=VALID).json()
    assert body["estimated_seconds"] == 30


# ==================== TaskStore 单元 ====================

def test_store_serializes_and_queues_second_task() -> None:
    """单 worker：第一个任务占用时第二个如实排队（queued），串行不并发。"""
    store = TaskStore(max_workers=1)
    gate = threading.Event()
    order: list[str] = []

    def slow() -> dict:
        order.append("slow-start")
        gate.wait(timeout=5)
        order.append("slow-end")
        return {}

    store.submit(slow)
    time.sleep(0.2)  # 让第一个任务先进 running
    fast_id = store.submit(lambda: order.append("fast") or {})
    assert store.get(fast_id)["status"] == "queued"

    gate.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and store.get(fast_id)["status"] != "done":
        time.sleep(0.05)
    assert store.get(fast_id)["status"] == "done"
    assert order == ["slow-start", "slow-end", "fast"]


def test_store_stores_exception_message() -> None:
    """fn 抛异常 → failed，error=str(exc)（安全文案责任在 fn，见 docstring）。"""
    store = TaskStore()

    def boom() -> dict:
        raise ValueError("boom")

    task_id = store.submit(boom)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and store.get(task_id)["status"] != "failed":
        time.sleep(0.05)
    record = store.get(task_id)
    assert record["status"] == "failed" and record["error"] == "boom"


def test_store_purges_expired_tasks() -> None:
    """TTL 惰性清理：created_at 超过保留期的任务在访问时被清掉。"""
    store = TaskStore(ttl=timedelta(hours=1))
    old_id = store.submit(lambda: {})
    # 直接把 created_at 拨回 2 小时前（模拟过期；私有字段仅测试操纵）
    with store._lock:  # noqa: SLF001 - 测试操纵时间戳
        store._tasks[old_id]["created_at"] = datetime_now_minus(2)
    assert store.get(old_id) is None  # get 即触发惰性清理

    fresh_id = store.submit(lambda: {})
    assert store.get(old_id) is None
    assert store.get(fresh_id) is not None


def datetime_now_minus(hours: int) -> str:
    return (datetime.now(timezone.utc).astimezone()
            - timedelta(hours=hours)).isoformat(timespec="seconds")
