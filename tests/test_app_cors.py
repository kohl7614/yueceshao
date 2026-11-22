"""tests/test_app_cors.py — M2-R6 跨源放行：在线模式页面（GitHub Pages）
跨源调用本 API 的 CORS 白名单契约。

契约：白名单源的响应携带 access-control-allow-origin；白名单外源不带该头
（预检被拒）；凭据不开放（allow_credentials=False）。
"""

from __future__ import annotations

from app.main import CORS_ALLOW_ORIGINS, create_app
from fastapi.testclient import TestClient

PAGES_ORIGIN = "https://kohl7614.github.io"
VALID = {"description": (
    "我村拟申请帮扶资金 60 万元，用于修建 2 公里产业道路和配套灌溉设施，"
    "由村集体经济组织作为实施主体，资金按项目进度拨付使用。"
)}


def test_allowed_origin_gets_cors_header() -> None:
    http = TestClient(create_app())
    r = http.post(
        "/api/checkup", json=VALID,
        headers={"Origin": PAGES_ORIGIN})
    assert r.status_code == 202
    assert r.headers["access-control-allow-origin"] == PAGES_ORIGIN


def test_preflight_allowed() -> None:
    """预检（OPTIONS）通过并回白名单源（浏览器在 POST 前自动发）。"""
    http = TestClient(create_app())
    r = http.options(
        "/api/checkup",
        headers={"Origin": PAGES_ORIGIN,
                 "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "Content-Type"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == PAGES_ORIGIN


def test_disallowed_origin_rejected() -> None:
    """白名单外源：预检被拒（400），简单请求不带 allow-origin 头。"""
    http = TestClient(create_app())
    pre = http.options(
        "/api/checkup",
        headers={"Origin": "https://evil.example",
                 "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 400
    simple = http.post("/api/checkup", json=VALID,
                       headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in simple.headers


def test_429_retry_after_exposed_cross_origin() -> None:
    """429 的 Retry-After 在跨源下可读（CORS expose_headers，PR#104 预审缺口）。"""
    http = TestClient(create_app(rate_limit_capacity=1))
    first = http.post("/api/checkup", json=VALID,
                      headers={"Origin": PAGES_ORIGIN})
    assert first.status_code == 202
    limited = http.post("/api/checkup", json=VALID,
                        headers={"Origin": PAGES_ORIGIN})
    assert limited.status_code == 429
    assert limited.headers.get("access-control-expose-headers") == "Retry-After"
    assert "retry-after" in limited.headers


def test_local_dev_origins_in_allowlist() -> None:
    """本地开发端口（http.server 8080 / Mock 8000）在白名单（任务书 R6 依赖）。"""
    for origin in ("http://127.0.0.1:8080", "http://localhost:8000"):
        assert origin in CORS_ALLOW_ORIGINS
