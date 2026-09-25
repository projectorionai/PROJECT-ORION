"""
Security tests for the RemoteGateway (Improvement Pass, Priority 1.3).

Covers the Cloud Roadmap Phase C1 auth scheme — one-time pairing codes,
revocable refresh tokens, 15-minute HMAC access tokens — and the Phase C4
capability manifest: a remote-origin request must not be able to reach
desktop_control, file_controller writes or mail sending even when it asks
for them by name. Also pins the Mark X.7 hardening (rate limiting, security
headers, constant-time comparisons) so auth changes cannot regress them.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.remote import (
    REMOTE_HARD_DENY,
    REMOTE_TOOL_WHITELIST,
    RemoteAgentQueue,
    RemoteAuthManager,
    RemoteGateway,
)


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _StubRouter:
    def has_text_fallback(self):
        return True

    async def generate_text(self, message, system_extra=""):
        profile = type("P", (), {"name": "stub-model"})()
        return profile, f"echo: {message}"


class _StubMemory:
    def __init__(self):
        self.episodes = []

    def log_episode(self, role, content):
        self.episodes.append((role, content))

    def prompt_context(self, limit=18):
        return ""


class _RecordingDispatcher:
    """Records every dispatch; the test fails if a denied tool reaches it."""

    def __init__(self):
        self.calls = []

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        from orion_core.data import ToolResult
        return ToolResult(f"ran {name}")


def _auth(tmp_path) -> RemoteAuthManager:
    return RemoteAuthManager(config_dir=tmp_path)


def _gateway(tmp_path, dispatcher=None) -> RemoteGateway:
    return RemoteGateway(
        _StubRouter(), _StubMemory(), _StubBus(),
        dispatcher=dispatcher, config_dir=tmp_path,
    )


async def _client(gateway):
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(gateway._build_app()))
    await client.start_server()
    return client


def _run(coro):
    return asyncio.run(coro)


# ── RemoteAuthManager: pairing ────────────────────────────────────────────────

def test_pairing_code_single_use(tmp_path):
    auth = _auth(tmp_path)
    code = auth.begin_pairing()
    assert auth.complete_pairing(code, "phone") is not None
    assert auth.complete_pairing(code, "phone-again") is None      # consumed


def test_wrong_pairing_code_rejected(tmp_path):
    auth = _auth(tmp_path)
    auth.begin_pairing()
    assert auth.complete_pairing("0000-0000", "intruder") is None


def test_expired_pairing_code_rejected(tmp_path, monkeypatch):
    auth = _auth(tmp_path)
    code = auth.begin_pairing()
    real_now = time.time()
    monkeypatch.setattr(time, "time", lambda: real_now + RemoteAuthManager.PAIR_TTL_S + 1)
    assert auth.complete_pairing(code, "late") is None


def test_pairing_case_insensitive(tmp_path):
    auth = _auth(tmp_path)
    code = auth.begin_pairing()
    assert auth.complete_pairing(code.lower(), "phone") is not None


# ── RemoteAuthManager: token issue / verify ───────────────────────────────────

def test_access_token_roundtrip(tmp_path):
    auth = _auth(tmp_path)
    device_id, refresh = auth.complete_pairing(auth.begin_pairing(), "phone")
    token, expires_in = auth.issue_access(device_id, refresh)
    assert expires_in == RemoteAuthManager.ACCESS_TTL_S == 15 * 60
    assert auth.verify_access(token) == device_id


def test_wrong_refresh_token_rejected(tmp_path):
    auth = _auth(tmp_path)
    device_id, _refresh = auth.complete_pairing(auth.begin_pairing(), "phone")
    assert auth.issue_access(device_id, "forged-refresh") is None


def test_refresh_tokens_stored_only_hashed(tmp_path):
    auth = _auth(tmp_path)
    _device_id, refresh = auth.complete_pairing(auth.begin_pairing(), "phone")
    registry = (tmp_path / "remote_devices.json").read_text(encoding="utf-8")
    assert refresh not in registry
    assert hashlib.sha256(refresh.encode()).hexdigest() in registry


def test_tampered_signature_rejected(tmp_path):
    auth = _auth(tmp_path)
    device_id, refresh = auth.complete_pairing(auth.begin_pairing(), "phone")
    token, _ = auth.issue_access(device_id, refresh)
    encoded, _, signature = token.rpartition(".")
    flipped = ("0" if signature[0] != "0" else "1") + signature[1:]
    assert auth.verify_access(f"{encoded}.{flipped}") is None


def test_forged_payload_with_foreign_secret_rejected(tmp_path):
    auth = _auth(tmp_path)
    device_id, _r = auth.complete_pairing(auth.begin_pairing(), "phone")
    payload = f"{device_id}:{int(time.time()) + 900}"
    encoded = base64.urlsafe_b64encode(payload.encode()).decode()
    forged_sig = hmac.new(b"attacker-secret", payload.encode(), hashlib.sha256).hexdigest()
    assert auth.verify_access(f"{encoded}.{forged_sig}") is None


def test_expired_access_token_rejected(tmp_path):
    auth = _auth(tmp_path)
    device_id, _r = auth.complete_pairing(auth.begin_pairing(), "phone")
    payload = f"{device_id}:{int(time.time()) - 5}"            # already expired
    encoded = base64.urlsafe_b64encode(payload.encode()).decode()
    signature = hmac.new(auth._secret, payload.encode(), hashlib.sha256).hexdigest()
    assert auth.verify_access(f"{encoded}.{signature}") is None


def test_revoked_device_loses_both_token_paths(tmp_path):
    auth = _auth(tmp_path)
    device_id, refresh = auth.complete_pairing(auth.begin_pairing(), "phone")
    token, _ = auth.issue_access(device_id, refresh)
    assert auth.revoke_device(device_id)
    assert auth.issue_access(device_id, refresh) is None       # refresh dead
    assert auth.verify_access(token) is None                   # access dead too


def test_garbage_tokens_rejected(tmp_path):
    auth = _auth(tmp_path)
    for garbage in ("", "no-dot", "a.b", "!!!.???", "AAAA.0000"):
        assert auth.verify_access(garbage) is None


def test_devices_persist_across_manager_restart(tmp_path):
    auth = _auth(tmp_path)
    device_id, refresh = auth.complete_pairing(auth.begin_pairing(), "phone")
    reloaded = _auth(tmp_path)                                  # fresh instance
    assert reloaded.issue_access(device_id, refresh) is not None


# ── capability manifest: the remote execution boundary ────────────────────────

def test_whitelist_and_hard_deny_are_disjoint():
    assert not (REMOTE_TOOL_WHITELIST & REMOTE_HARD_DENY)


@pytest.mark.parametrize("tool", ["desktop_control", "file_controller", "outlook_mail"])
def test_remote_request_cannot_reach_desktop_tools_by_name(tool):
    dispatcher = _RecordingDispatcher()
    queue = RemoteAgentQueue(dispatcher=dispatcher)
    task = _run(queue.run(tool, {"action": "send_draft"}))
    assert task["status"] == "denied"
    assert dispatcher.calls == []                  # never reached the dispatcher


def test_hard_deny_wins_even_if_whitelist_is_tampered(monkeypatch):
    # Defence in depth: a future whitelist edit must not expose desktop tools.
    import orion_core.remote as remote_module
    monkeypatch.setattr(
        remote_module, "REMOTE_TOOL_WHITELIST",
        frozenset(REMOTE_TOOL_WHITELIST | {"desktop_control"}),
    )
    dispatcher = _RecordingDispatcher()
    queue = RemoteAgentQueue(dispatcher=dispatcher)
    task = _run(queue.run("desktop_control", {"action": "click"}))
    assert task["status"] == "denied" and dispatcher.calls == []


def test_unlisted_tool_denied_by_default():
    dispatcher = _RecordingDispatcher()
    queue = RemoteAgentQueue(dispatcher=dispatcher)
    task = _run(queue.run("globe", {}))
    assert task["status"] == "denied" and dispatcher.calls == []


def test_whitelisted_tool_dispatches():
    dispatcher = _RecordingDispatcher()
    queue = RemoteAgentQueue(dispatcher=dispatcher)
    task = _run(queue.run("query_intelligence", {"query": "status"}))
    assert task["status"] == "done" and dispatcher.calls


def test_no_dispatcher_reports_unavailable():
    queue = RemoteAgentQueue(dispatcher=None)
    task = _run(queue.run("query_intelligence", {}))
    assert task["status"] == "unavailable"


# ── HTTP surface: end-to-end over aiohttp ─────────────────────────────────────

def test_http_pair_token_chat_flow(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        code = gateway.begin_pairing()
        client = await _client(gateway)
        try:
            paired = await client.post("/v1/auth/pair",
                                       json={"code": code, "device_name": "pytest"})
            assert paired.status == 200
            creds = await paired.json()

            minted = await client.post("/v1/auth/token", json={
                "device_id": creds["device_id"],
                "refresh_token": creds["refresh_token"]})
            assert minted.status == 200
            token = (await minted.json())["access_token"]

            chat = await client.post(
                "/api/chat", json={"message": "hello orion"},
                headers={"Authorization": f"Bearer {token}"})
            assert chat.status == 200
            body = await chat.json()
            assert body["ok"] and "hello orion" in body["reply"]
        finally:
            await client.close()
    _run(flow())


def test_http_chat_unauthenticated_is_401(tmp_path):
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            for payload in ({"message": "hi"},
                            {"message": "hi", "access_token": "bogus"},
                            {"message": "hi", "token": "legacy-static-token"}):
                response = await client.post("/api/chat", json=payload)
                assert response.status == 401
        finally:
            await client.close()
    _run(flow())


def test_http_legacy_static_token_no_longer_accepted(tmp_path):
    async def flow():
        # Even a byte-for-byte copy of the old static token file is refused.
        (tmp_path / "remote_token.txt").write_text("old-static-token", encoding="utf-8")
        gateway = _gateway(tmp_path)
        client = await _client(gateway)
        try:
            response = await client.post(
                "/api/chat", json={"message": "hi", "token": "old-static-token"},
                headers={"Authorization": "Bearer old-static-token"})
            assert response.status == 401
        finally:
            await client.close()
    _run(flow())


def test_http_agent_task_denied_tool_403(tmp_path):
    async def flow():
        dispatcher = _RecordingDispatcher()
        gateway = _gateway(tmp_path, dispatcher=dispatcher)
        code = gateway.begin_pairing()
        client = await _client(gateway)
        try:
            creds = await (await client.post(
                "/v1/auth/pair", json={"code": code})).json()
            token = (await (await client.post("/v1/auth/token", json={
                "device_id": creds["device_id"],
                "refresh_token": creds["refresh_token"]})).json())["access_token"]

            denied = await client.post(
                "/v1/agent/tasks",
                json={"tool": "desktop_control", "args": {"action": "click"}},
                headers={"Authorization": f"Bearer {token}"})
            assert denied.status == 403
            assert dispatcher.calls == []

            allowed = await client.post(
                "/v1/agent/tasks",
                json={"tool": "query_intelligence", "args": {"query": "x"}},
                headers={"Authorization": f"Bearer {token}"})
            assert allowed.status == 200
            assert dispatcher.calls == [("query_intelligence", {"query": "x"})]
        finally:
            await client.close()
    _run(flow())


def test_http_agent_task_requires_auth(tmp_path):
    async def flow():
        client = await _client(_gateway(tmp_path, dispatcher=_RecordingDispatcher()))
        try:
            response = await client.post(
                "/v1/agent/tasks", json={"tool": "query_intelligence", "args": {}})
            assert response.status == 401
        finally:
            await client.close()
    _run(flow())


def test_json_endpoints_reject_non_object_bodies(tmp_path):
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            for route in ("/v1/auth/pair", "/v1/auth/token",
                          "/v1/agent/tasks", "/api/chat", "/api/confirm"):
                for body in ([], 42, None):
                    response = await client.post(route, json=body)
                    assert response.status == 400, (route, body, response.status)
        finally:
            await client.close()
    _run(flow())


def test_task_status_is_private_to_owning_device(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path, dispatcher=_RecordingDispatcher())
        client = await _client(gateway)
        try:
            async def pair():
                creds = await (await client.post("/v1/auth/pair", json={
                    "code": gateway.begin_pairing()})).json()
                token = (await (await client.post("/v1/auth/token", json={
                    "device_id": creds["device_id"],
                    "refresh_token": creds["refresh_token"]})).json())["access_token"]
                return token

            owner_token = await pair()
            other_token = await pair()
            created = await client.post("/v1/agent/tasks", json={
                "tool": "query_intelligence", "args": {"query": "private"}},
                headers={"Authorization": f"Bearer {owner_token}"})
            task_id = (await created.json())["id"]
            route = f"/v1/agent/tasks/{task_id}"
            other = await client.get(route, headers={
                "Authorization": f"Bearer {other_token}"})
            owner = await client.get(route, headers={
                "Authorization": f"Bearer {owner_token}"})
            assert other.status == 404
            assert owner.status == 200
        finally:
            await client.close()
    _run(flow())


def test_http_security_headers_present(tmp_path):
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            response = await client.get("/api/health")
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert "Content-Security-Policy" in response.headers
        finally:
            await client.close()
    _run(flow())


def test_http_face_is_same_origin_framable(tmp_path):
    """The uplink page frames /face from the same origin, so /face must NOT
    inherit the global DENY — otherwise the browser refuses to render it and
    shows 'refused to connect' with only a fallback mini-orb."""
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            response = await client.get("/face")
            assert response.status == 200
            assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
            assert "frame-ancestors 'self'" in response.headers["Content-Security-Policy"]
        finally:
            await client.close()
    _run(flow())


def test_http_pair_endpoint_rate_limited(tmp_path):
    async def flow():
        client = await _client(_gateway(tmp_path))
        try:
            statuses = []
            for _ in range(12):
                response = await client.post(
                    "/v1/auth/pair", json={"code": "0000-0000"})
                statuses.append(response.status)
            assert 429 in statuses                # brute force cut off inside 60 s
            assert statuses.count(429) >= 2
        finally:
            await client.close()
    _run(flow())


def test_health_endpoint_leaks_no_secrets(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        gateway.begin_pairing()
        client = await _client(gateway)
        try:
            body = await (await client.get("/api/health")).text()
            assert "refresh" not in body and "secret" not in body
        finally:
            await client.close()
    _run(flow())
