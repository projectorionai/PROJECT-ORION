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
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import request_trace
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
            for route in ("/v1/auth/pair", "/v1/auth/token", "/v1/agent/tasks",
                          "/api/chat", "/api/chat/stream", "/api/chat/cancel",
                          "/api/confirm"):
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


# ── streamed replies: /api/chat/stream, /api/chat/cancel, /api/history ───────

class _StreamingRouter:
    """The streaming contract ProviderRouter.generate_text is gaining:
    ``generate_text(prompt, system_extra="", *, on_delta, on_restart)``.
    *body* plays the provider: it is handed both callbacks and returns the
    finished reply."""

    def __init__(self, body):
        self.body = body
        self.calls = 0
        self.cancelled = asyncio.Event()

    def has_text_fallback(self):
        return True

    async def generate_text(self, prompt, system_extra="", *, on_delta=None, on_restart=None):
        self.calls += 1
        # A real router marks each provider attempt on the request in flight.
        request_trace.mark("provider_attempt", provider="fake-stream", ok=True)
        try:
            reply = await self.body(on_delta, on_restart)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return type("P", (), {"name": "fake-stream"})(), reply


class _Brain:
    """A local brain: *task* answers the task path, *offline* the fallback."""

    def __init__(self, task=None, offline="Offline answer."):
        self.task, self.offline = task, offline

    async def try_task(self, message):
        return self.task

    async def respond(self, message):
        return self.offline


def _paired(gateway):
    device_id, refresh = gateway.auth.complete_pairing(gateway.begin_pairing(), "phone")
    token, _expires = gateway.auth.issue_access(device_id, refresh)
    return device_id, {"Authorization": f"Bearer {token}"}


async def _next_event(response, timeout=5.0):
    """The next event on a streamed reply (None at the end), stamped with
    when it arrived."""
    while True:
        line = await asyncio.wait_for(response.content.readline(), timeout)
        if not line:
            return None
        line = line.strip()
        if line.startswith(b"data:"):
            event = json.loads(line[5:])
            event["_at"] = time.monotonic()
            return event


async def _all_events(response, timeout=5.0):
    events = []
    while (event := await _next_event(response, timeout)) is not None:
        events.append(event)
    return events


def _trace(request_id):
    return next(t for t in request_trace.TRACES.all() if t.request_id == request_id)


def _streamed(tmp_path, router, message="tell me", **kwargs):
    """Run one streamed turn; (response, events, gateway)."""
    async def flow():
        memory = _StubMemory()
        gateway = RemoteGateway(router, memory, _StubBus(), config_dir=tmp_path, **kwargs)
        _device, headers = _paired(gateway)
        client = await _client(gateway)
        try:
            response = await client.post("/api/chat/stream", json={"message": message},
                                         headers=headers)
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/event-stream")
            return response, await _all_events(response), gateway
        finally:
            await client.close()
    return _run(flow())


def test_stream_shows_text_before_the_answer_completes(tmp_path):
    async def body(on_delta, on_restart):
        for word in ("The ", "quick ", "brown ", "fox ", "jumps."):
            on_delta(word)
            await asyncio.sleep(0.08)
        return "The quick brown fox jumps."

    started = time.monotonic()
    response, events, gateway = _streamed(tmp_path, _StreamingRouter(body))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert set(kinds[1:-1]) == {"delta"}
    request_id = response.headers["X-Request-Id"]
    assert all(e["request_id"] == request_id for e in events)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(set(seqs))                       # strictly rising
    first, done = events[1], events[-1]
    assert first["_at"] - started < done["_at"] - started - 0.2
    assert "".join(e["text"] for e in events if e["type"] == "delta") == done["reply"]
    assert done["provider"] == "fake-stream"
    # The whole final answer is persisted once, after the phone's message.
    assert gateway.memory.episodes == [("user (remote)", "tell me"),
                                       ("orion (remote)", "The quick brown fox jumps.")]
    trace = _trace(request_id)
    stages = [e.stage for e in trace.events]
    assert stages == ["COMMAND_RECEIVED", "API_REQUEST_START",
                      "API_RESPONSE_RECEIVED", "REPLY_DELIVERED"]
    assert trace.complete and trace.since_start_ms("first_token") is not None
    assert trace.marks_named("provider_attempt")          # the router's marks land here
    assert gateway._chat_streams == {}


def test_stream_without_the_router_contract_sends_one_done(tmp_path):
    _response, events, _gateway = _streamed(tmp_path / "plain", _StubRouter(), "hello")
    assert [e["type"] for e in events] == ["start", "done"]
    assert events[-1]["reply"] == "echo: hello"

    class _Loose:
        """Takes **kwargs but refuses the new keywords: asked once more, plainly."""
        def has_text_fallback(self):
            return True

        async def generate_text(self, prompt, system_extra="", **kwargs):
            if "on_delta" in kwargs:
                raise TypeError("generate_text() got an unexpected keyword argument 'on_delta'")
            return type("P", (), {"name": "loose"})(), "plain answer"

    _response, events, _gateway = _streamed(tmp_path / "loose", _Loose())
    assert [e["type"] for e in events] == ["start", "done"]
    assert events[-1]["reply"] == "plain answer"


def test_streaming_contract_is_detected_not_assumed():
    from orion_core.remote import _stream_support

    async def named(prompt, system_extra="", *, on_delta=None, on_restart=None):
        pass

    async def plain(prompt, system_extra=""):
        pass

    async def loose(prompt, **kwargs):
        pass

    assert _stream_support(named) is True
    assert _stream_support(plain) is False
    assert _stream_support(loose) is None


def test_tokens_reported_from_a_worker_thread_arrive_in_order(tmp_path):
    async def body(on_delta, on_restart):
        for word in ("one ", "two ", "three"):
            await asyncio.to_thread(on_delta, word)
        await asyncio.sleep(0.05)
        return "one two three"

    _response, events, _gateway = _streamed(tmp_path, _StreamingRouter(body))
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "one two three"


def test_restart_clears_the_abandoned_attempt(tmp_path):
    async def body(on_delta, on_restart):
        on_delta("Wrong start")
        await asyncio.sleep(0.15)            # written to the phone by now
        on_delta(" still wrong")             # not yet written when the provider fails
        on_restart("provider-a failed part-way")
        on_delta("Right")
        await asyncio.sleep(0.15)
        return "Right answer"

    _response, events, _gateway = _streamed(tmp_path, _StreamingRouter(body))
    kinds = [e["type"] for e in events]
    assert kinds == ["start", "delta", "restart", "delta", "done"]
    assert events[2]["reason"] == "provider-a failed part-way"
    assert not any("still wrong" in e.get("text", "") for e in events)
    assert events[3]["text"] == "Right" and events[-1]["reply"] == "Right answer"


def test_a_stream_that_fails_part_way_is_restarted_or_reported_never_cut_short(tmp_path):
    async def body(on_delta, on_restart):
        on_delta("Half an ans")
        await asyncio.sleep(0.1)
        # What the provider transport now raises for a stream that ends
        # without [DONE]/a finish reason, once no other provider is left.
        raise RuntimeError("stream ended before the answer was complete")

    # With an offline brain: the partial text is withdrawn, then replaced.
    _response, events, gateway = _streamed(tmp_path / "brain", _StreamingRouter(body),
                                           local_brain=_Brain())
    assert [e["type"] for e in events] == ["start", "delta", "restart", "done"]
    assert events[-1]["reply"] == "Offline answer."
    assert events[-1]["provider"] == "local-brain"
    assert ("orion (remote)", "Half an ans") not in gateway.memory.episodes

    # Without one: withdrawn, then an error. Never a `done` carrying the fragment.
    response, events, gateway = _streamed(tmp_path / "none", _StreamingRouter(body))
    assert [e["type"] for e in events] == ["start", "delta", "restart", "error"]
    assert all(role != "orion (remote)" for role, _ in gateway.memory.episodes)
    assert _trace(response.headers["X-Request-Id"]).failed_stage == "API_REQUEST_START"


def test_the_task_path_answers_in_one_done(tmp_path):
    async def body(on_delta, on_restart):
        raise AssertionError("a handled task must not reach the model")

    router = _StreamingRouter(body)
    _response, events, _gateway = _streamed(tmp_path, router, "open my calendar on my PC",
                                            local_brain=_Brain(task="Opened your calendar."))
    assert [e["type"] for e in events] == ["start", "done"]
    assert events[-1]["reply"] == "Opened your calendar."
    assert events[-1]["provider"] == "orion-remote"
    assert router.calls == 0


def test_stream_route_refuses_exactly_like_plain_chat(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        _device, headers = _paired(gateway)
        client = await _client(gateway)
        try:
            for payload, auth, status in (
                    ({"message": "hi"}, {}, 401),
                    ({"message": "hi", "access_token": "bogus"}, {}, 401),
                    ({"message": "   "}, headers, 400),
                    ({"message": {"text": "hi"}}, headers, 400),
                    ({"message": "rm -rf /"}, headers, 403)):
                response = await client.post("/api/chat/stream", json=payload, headers=auth)
                assert response.status == status, (payload, response.status)
                assert response.headers["Content-Type"].startswith("application/json")
            gateway._RATE_MAX = 0
            response = await client.post("/api/chat/stream", json={"message": "hi"},
                                         headers=headers)
            assert response.status == 429
        finally:
            await client.close()
    _run(flow())


def _hanging_router():
    async def body(on_delta, on_restart):
        on_delta("Thinking about it")
        await asyncio.Event().wait()          # never finishes by itself
    return _StreamingRouter(body)


def test_cancel_is_owner_only_and_stops_the_generation(tmp_path):
    async def flow():
        router = _hanging_router()
        memory = _StubMemory()
        gateway = RemoteGateway(router, memory, _StubBus(), config_dir=tmp_path)
        _owner_id, owner = _paired(gateway)
        _other_id, other = _paired(gateway)
        client = await _client(gateway)
        try:
            response = await client.post("/api/chat/stream", json={"message": "think"},
                                         headers=owner)
            start = await _next_event(response)
            request_id = start["request_id"]
            assert (await _next_event(response))["type"] == "delta"
            for body, headers, status in (
                    ({"request_id": request_id}, other, 404),     # not theirs
                    ({"request_id": "ORION_REQ_999999"}, owner, 404),
                    ({"request_id": request_id}, {}, 401)):
                refused = await client.post("/api/chat/cancel", json=body, headers=headers)
                assert refused.status == status
            assert not router.cancelled.is_set()
            cancelled = await client.post("/api/chat/cancel",
                                          json={"request_id": request_id}, headers=owner)
            assert cancelled.status == 200
            assert (await cancelled.json())["status"] == "cancelled"
            end = await _next_event(response)
            assert end["type"] == "error" and end["cancelled"] is True
            await asyncio.wait_for(router.cancelled.wait(), timeout=2)
            assert await _next_event(response) is None
        finally:
            await client.close()
        assert gateway._chat_streams == {}
        assert all(role != "orion (remote)" for role, _ in memory.episodes)
        mark = _trace(request_id).marks_named("cancelled")[0]
        assert mark.fields["reason"] == "phone" and mark.fields["settled"] is True
        assert _trace(request_id).complete               # not left as a stalled request
    _run(flow())


@pytest.mark.parametrize("server", ["production", "test-server"])
def test_a_phone_that_disconnects_cancels_its_generation(tmp_path, monkeypatch, server):
    """The production runner does not cancel a handler when its client goes
    (aiohttp 3.9+), so the stream must notice the closed socket itself;
    aiohttp's test server does cancel it. Both must stop the generation."""
    import socket

    import aiohttp

    monkeypatch.setattr(RemoteGateway, "_STREAM_TICK_S", 0.05)
    monkeypatch.setenv("ORION_REMOTE_FIREWALL", "0")
    monkeypatch.delenv("ORION_REMOTE_TLS", raising=False)

    async def flow():
        router = _hanging_router()
        gateway = RemoteGateway(router, _StubMemory(), _StubBus(), config_dir=tmp_path)
        _device, headers = _paired(gateway)
        if server == "production":
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                gateway.port = probe.getsockname()[1]
            gateway.host = "127.0.0.1"
            await gateway.start()
            session = aiohttp.ClientSession()
            post = lambda: session.post(                      # noqa: E731
                f"http://127.0.0.1:{gateway.port}/api/chat/stream",
                json={"message": "think"}, headers=headers)
            closing = [session.close(), gateway.stop()]
        else:
            client = await _client(gateway)
            post = lambda: client.post("/api/chat/stream",    # noqa: E731
                                       json={"message": "think"}, headers=headers)
            closing = [client.close()]
        try:
            response = await post()
            request_id = (await _next_event(response))["request_id"]
            await _next_event(response)
            started = time.monotonic()
            response.close()                                  # the phone goes away
            await asyncio.wait_for(router.cancelled.wait(), timeout=3)
            assert time.monotonic() - started < 1.0
            for _ in range(60):
                if not gateway._chat_streams:
                    break
                await asyncio.sleep(0.05)
            assert gateway._chat_streams == {}
        finally:
            for step in closing:
                await step
        assert _trace(request_id).marks_named("cancelled")[0].fields["reason"] == "disconnect"
    _run(flow())


def test_stop_ends_a_streamed_reply_in_progress(tmp_path):
    async def flow():
        router = _hanging_router()
        gateway = RemoteGateway(router, _StubMemory(), _StubBus(), config_dir=tmp_path)
        _device, headers = _paired(gateway)
        client = await _client(gateway)
        try:
            response = await client.post("/api/chat/stream", json={"message": "think"},
                                         headers=headers)
            await _next_event(response)
            await _next_event(response)
            await asyncio.wait_for(gateway.stop(), timeout=5)
            end = await _next_event(response)
            assert end["type"] == "error" and end["message"] == "ORION is shutting down."
            await asyncio.wait_for(router.cancelled.wait(), timeout=2)
        finally:
            await client.close()
    _run(flow())


@pytest.mark.parametrize("blocked", ["power", "socket", "amplitude", "runner"])
def test_gateway_stop_bounds_each_independent_resource(tmp_path, monkeypatch, blocked):
    """A phone that never acknowledges close must not hold desktop teardown."""
    from types import SimpleNamespace

    async def flow():
        gateway = _gateway(tmp_path)
        monkeypatch.setattr(gateway, "_STOP_TIMEOUT_S", 0.025, raising=False)
        monkeypatch.setattr(gateway, "_STOP_CANCEL_GRACE_S", 0.025, raising=False)
        released = asyncio.Event()
        began, finished = [], []

        async def close(name, **kwargs):
            began.append(name)
            if name == blocked:
                while not released.is_set():
                    try:
                        await released.wait()
                    except asyncio.CancelledError:
                        continue              # a badly behaved third-party close
            finished.append(name)

        gateway.power_controller = SimpleNamespace(close=lambda: close("power"))
        gateway._audio_sessions = {"phone": SimpleNamespace(
            close=lambda **kw: close("socket", **kw))}
        gateway._runner = SimpleNamespace(cleanup=lambda: close("runner"))
        amplitude = gateway._amp_task = asyncio.create_task(close("amplitude"))
        await asyncio.sleep(0)                 # amplitude is already running
        stopping = asyncio.create_task(gateway.stop())
        done, _pending = await asyncio.wait({stopping}, timeout=0.15)
        bounded = stopping in done
        in_flight = [t.get_name() for t in getattr(gateway, "_shutdown_tasks", ())
                     if not t.done()]
        owned = list(getattr(gateway, "_shutdown_tasks", ()))
        before_release = list(began)
        released.set()                        # never leave this regression hung
        await stopping
        await asyncio.gather(amplitude, return_exceptions=True)
        await asyncio.gather(*owned, return_exceptions=True)
        assert bounded, f"{blocked} blocked gateway shutdown past its budget"
        assert set(before_release) == {"power", "socket", "amplitude", "runner"}
        assert in_flight, "an unfinished close must stay owned and visible"
        assert any("still running" in str(line) for line in gateway.bus.log.emitted)
        await asyncio.sleep(0)
        assert not getattr(gateway, "_shutdown_tasks", set())
    _run(flow())


def test_gateway_stop_failure_still_closes_other_resources(tmp_path):
    from types import SimpleNamespace

    async def flow():
        gateway = _gateway(tmp_path)
        finished = []
        async def broken():
            raise OSError("synthetic ledger failure")
        async def cleaned(name, **kwargs):
            finished.append(name)
        gateway.power_controller = SimpleNamespace(close=broken)
        gateway._audio_sessions = {"phone": SimpleNamespace(
            close=lambda **kw: cleaned("socket", **kw))}
        gateway._runner = SimpleNamespace(cleanup=lambda: cleaned("runner"))
        await gateway.stop()
        assert set(finished) == {"socket", "runner"}
        assert any("power" in str(line) and "OSError" in str(line)
                   for line in gateway.bus.log.emitted)
    _run(flow())


def test_gateway_stop_has_one_owner_when_callers_overlap_or_cancel(tmp_path):
    from types import SimpleNamespace

    async def flow():
        gateway = _gateway(tmp_path)
        started, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def close():
            calls.append("runner")
            started.set()
            await release.wait()
        gateway._runner = SimpleNamespace(cleanup=close)
        first = asyncio.create_task(gateway.stop())
        await started.wait()
        second = asyncio.create_task(gateway.stop())
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not gateway._stop_task.done(), "caller cancellation abandoned the gateway"
        release.set()
        await second
        await gateway.stop()
        assert calls == ["runner"]
        assert not gateway._shutdown_tasks
    _run(flow())


def test_gateway_does_not_start_while_old_cleanup_is_still_owned(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        pending = asyncio.create_task(asyncio.Event().wait())
        gateway._shutdown_tasks.add(pending)
        try:
            with pytest.raises(RuntimeError, match="still closing"):
                await gateway.start()
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    _run(flow())


# ── W1.3: stop() with real phone connections open (11 Oct 2026) ─────────────
# The real gateway on loopback, every kind of long-lived phone connection
# open, and phones that never answer: an asleep phone does not reply to a
# websocket close frame, and one that stopped reading leaves the server's
# writes parked in drain(). MEASURED before: 4.0 s with the companion app
# connected (its /v1/device/uplink socket was never closed by stop(), so
# aiohttp's 2.0 s grace ran out twice) and 4.0 s for a stalled reader, with
# 2-3 handler tasks per phone still pending afterwards. After: 0.02-0.09 s
# and 0.25 s, nothing pending.

def _loopback_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_REMOTE_FIREWALL", "0")
    monkeypatch.delenv("ORION_REMOTE_TLS", raising=False)
    monkeypatch.delenv("ORION_REMOTE_EXPOSURE", raising=False)
    gateway = RemoteGateway(_hanging_router(), _StubMemory(), _StubBus(), config_dir=tmp_path)
    gateway.host, gateway.port = "127.0.0.1", _free_port()
    return gateway


def _other_tasks():
    mine = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not mine and not t.done()]


def test_stop_is_prompt_with_the_companion_app_connected(tmp_path, monkeypatch):
    import aiohttp

    async def flow():
        gateway = _loopback_gateway(tmp_path, monkeypatch)
        _device, headers = _paired(gateway)
        token = headers["Authorization"][7:]
        await gateway.start()
        base = f"http://127.0.0.1:{gateway.port}"
        session = aiohttp.ClientSession()
        sockets = []
        try:
            # An asleep phone: it never reads, so it never answers a close.
            for path in ("/v1/device/uplink", "/v1/audio/stream"):
                sockets.append(await session.ws_connect(
                    base + path, headers=headers,
                    autoclose=False, autoping=False, heartbeat=None))
            events = await session.get(f"{base}/api/events?token={token}")
            await asyncio.wait_for(events.content.readline(), timeout=5)
            chat = await session.post(f"{base}/api/chat/stream",
                                      json={"message": "think"}, headers=headers)
            await _next_event(chat)                        # start
            await _next_event(chat)                        # the first words
            for _ in range(100):
                if gateway._audio_sessions and gateway.mobile_viewport.open_sockets():
                    break
                await asyncio.sleep(0.02)
            assert gateway._audio_sessions and gateway.mobile_viewport.open_sockets()
            started = time.monotonic()
            await asyncio.wait_for(gateway.stop(), timeout=10)
            took = time.monotonic() - started
            await asyncio.sleep(0.05)
            assert took < 1.0, f"stop() took {took:.2f} s with the companion app connected"
            assert _port_is_closed(gateway.port)
            assert _other_tasks() == [], "a phone handler outlived stop()"
            assert not any("deadline" in str(line) for line in gateway.bus.log.emitted)
        finally:
            for socket in sockets:
                await socket.close()
            await session.close()
    _run(flow())


def test_stop_drops_a_phone_that_stopped_reading(tmp_path, monkeypatch):
    import socket as _socket

    async def flow():
        gateway = _loopback_gateway(tmp_path, monkeypatch)
        _device, headers = _paired(gateway)
        token = headers["Authorization"][7:]
        await gateway.start()
        raw = _socket.socket()
        try:
            raw.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 1024)
            raw.connect(("127.0.0.1", gateway.port))
            raw.sendall(f"GET /api/events?token={token} HTTP/1.1\r\n"
                        f"Host: 127.0.0.1\r\n\r\n".encode())
            # Fill the pipe until the stream's writer is parked in drain():
            # its queue then stays full because nothing takes from it.
            event = {"type": "thought", "text": "x" * 600, "kind": "test"}
            deadline, full_since = time.monotonic() + 20, None
            while time.monotonic() < deadline:
                for _ in range(64):
                    gateway._push_event(event)
                await asyncio.sleep(0.01)
                queues = list(gateway._event_subs)
                if queues and all(q.qsize() >= 60 for q in queues):
                    full_since = full_since or time.monotonic()
                    if time.monotonic() - full_since > 0.5:
                        break
                else:
                    full_since = None
            assert full_since is not None, "could not stall the stream's writer"
            started = time.monotonic()
            await asyncio.wait_for(gateway.stop(), timeout=10)
            took = time.monotonic() - started
            await asyncio.sleep(0.05)
            assert took < 1.0, f"stop() took {took:.2f} s for a phone that stopped reading"
            assert _port_is_closed(gateway.port)
            assert _other_tasks() == [], "the stalled stream's handler outlived stop()"
            assert not gateway._event_subs
        finally:
            raw.close()
    _run(flow())


def test_plain_chat_names_its_request_and_completes_its_trace(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        _device, headers = _paired(gateway)
        client = await _client(gateway)
        try:
            response = await client.post("/api/chat", json={"message": "hi"}, headers=headers)
            body = await response.json()
        finally:
            await client.close()
        assert body["ok"] and body["reply"] == "echo: hi" and body["provider"] == "stub-model"
        assert body["request_id"] == response.headers["X-Request-Id"]
        assert _trace(body["request_id"]).complete
    _run(flow())


def test_history_pages_back_only_through_the_owners_conversation(tmp_path):
    async def flow():
        gateway = _gateway(tmp_path)
        _owner_id, owner = _paired(gateway)
        _other_id, other = _paired(gateway)
        client = await _client(gateway)

        async def get(query, headers):
            response = await client.get(f"/api/history{query}", headers=headers)
            return response.status, (await response.json())

        try:
            for i in range(3):
                await client.post("/api/chat", json={"message": f"m{i}"}, headers=owner)
            status, page = await get("?limit=2", owner)
            assert status == 200 and page["more"] is True
            assert [(e["id"], e["role"], e["text"]) for e in page["items"]] == [
                (5, "user", "m2"), (6, "orion", "echo: m2")]
            _status, older = await get("?before=5&limit=50", owner)
            assert [e["id"] for e in older["items"]] == [1, 2, 3, 4] and older["more"] is False
            _status, probe = await get("?limit=0", owner)
            assert probe["items"] == [] and probe["more"] is True and probe["next"] == 7
            _status, theirs = await get("?limit=50", other)
            assert theirs["items"] == [] and theirs["more"] is False
            assert (await get("", {}))[0] == 401
            for bad in ("?limit=abc", "?limit=500", "?limit=-1", "?before=0"):
                assert (await get(bad, owner))[0] == 400, bad
        finally:
            await client.close()
    _run(flow())


def test_history_is_bounded_per_device_and_in_devices():
    from orion_core.remote import _PhoneHistory
    history = _PhoneHistory()
    for i in range(_PhoneHistory.PER_DEVICE + 50):
        history.add("phone", "user", "x" * (_PhoneHistory.TEXT_LIMIT + 10))
    items, more, next_id = history.page("phone", None, _PhoneHistory.PAGE_MAX)
    assert next_id == _PhoneHistory.PER_DEVICE + 51
    assert len(history._by_device["phone"][1]) <= _PhoneHistory.PER_DEVICE
    assert sum(len(item["text"]) for item in history._by_device["phone"][1]) <= _PhoneHistory.CHAR_BUDGET
    assert all(len(item["text"]) == _PhoneHistory.TEXT_LIMIT for item in items)
    assert all(item["truncated"] for item in items)
    for i in range(_PhoneHistory.DEVICES + 5):
        history.add(f"device-{i}", "user", "hi")
    assert len(history._by_device) == _PhoneHistory.DEVICES


def test_event_streams_are_bounded_and_dropped_when_the_phone_goes(tmp_path, monkeypatch):
    monkeypatch.setattr(RemoteGateway, "_EVENTS_TICK_S", 0.05)

    async def flow():
        gateway = _gateway(tmp_path)
        _device, headers = _paired(gateway)
        token = headers["Authorization"][7:]
        client = await _client(gateway)
        try:
            streams = []
            for _ in range(4):                      # a phone reconnecting repeatedly
                response = await client.get(f"/api/events?token={token}")
                await asyncio.wait_for(response.content.readline(), timeout=5)
                streams.append(response)
            assert len(gateway._event_subs) == RemoteGateway._EVENT_STREAMS_PER_DEVICE
            # The oldest was retired: its stream ends by itself.
            await asyncio.wait_for(streams[0].content.read(), timeout=5)
            for response in streams[1:]:
                response.close()                    # the phone drops off the network
            for _ in range(60):
                if not gateway._event_subs:
                    break
                await asyncio.sleep(0.05)
            assert not gateway._event_subs and not gateway._event_devices
            # The total is capped too; another device is refused, not let in
            # by evicting someone else's stream.
            gateway._EVENT_STREAMS_MAX = 1
            _other, other_headers = _paired(gateway)
            first = await client.get(f"/api/events?token={token}")
            await asyncio.wait_for(first.content.readline(), timeout=5)
            refused = await client.get(
                f"/api/events?token={other_headers['Authorization'][7:]}")
            assert refused.status == 503
            first.close()
        finally:
            await client.close()
    _run(flow())


# ── /api/tasks: the phone's read-only task list ───────────────────────────────

class _TaskJobs:
    """The JobManager snapshot surface /api/tasks reads."""

    def snapshot(self, limit=50, since_seq=0):
        return {
            "seq": 7, "counts": {"running": 1, "completed": 1},
            "jobs": [
                {"id": "job-aaaaaaaaaaaa", "kind": "plan", "name": "tidy downloads",
                 "label": "", "state": "running", "cls": "user", "fraction": 0.5,
                 "step": 2, "total": 4, "note": "moving invoices", "args": {"path": "C:/secret"},
                 "checkpoint": {"next": 3}, "result": "", "updated_at": 1.0},
                {"id": "job-bbbbbbbbbbbb", "kind": "tool", "name": "research",
                 "state": "completed", "result": "x" * 900, "ok": True},
            ],
            "approvals": [{"id": "a1", "job_id": "job-aaaaaaaaaaaa", "step_idx": 1,
                           "tool": "send_email", "reason": "sends an e-mail",
                           "expires_at": 99.0}],
        }


def test_tasks_route_is_authenticated_bounded_and_carries_no_arguments(tmp_path):
    async def flow():
        dispatcher = type("D", (), {"jobs": _TaskJobs()})()
        gateway = _gateway(tmp_path, dispatcher=dispatcher)
        _device, headers = _paired(gateway)
        client = await _client(gateway)
        try:
            assert (await client.get("/api/tasks")).status == 401
            assert (await client.get("/api/tasks?limit=0", headers=headers)).status == 400
            assert (await client.get("/api/tasks?limit=x", headers=headers)).status == 400
            response = await client.get("/api/tasks?limit=10", headers=headers)
            body = await response.json()
        finally:
            await client.close()
        assert response.status == 200 and body["ok"]
        running, done = body["jobs"]
        assert running["state"] == "running" and running["fraction"] == 0.5
        assert "args" not in running and "checkpoint" not in running
        assert "result" not in running and running["note"] == "moving invoices"
        assert len(done["result"]) == 400
        assert body["approvals"] == [{"tool": "send_email", "reason": "sends an e-mail",
                                      "expires_at": 99.0}]
        assert body["approve_on"] == "desktop"
    _run(flow())


def test_tasks_route_falls_back_to_the_shared_job_store(tmp_path, monkeypatch):
    from orion_core import job_store

    store = job_store.JobStore(None)
    store.create(kind="workflow", name="morning routine")
    monkeypatch.setattr(job_store, "default_store", lambda: store)

    async def flow():
        gateway = _gateway(tmp_path)
        _device, headers = _paired(gateway)
        client = await _client(gateway)
        try:
            body = await (await client.get("/api/tasks", headers=headers)).json()
        finally:
            await client.close()
        assert [j["name"] for j in body["jobs"]] == ["morning routine"]
        assert body["jobs"][0]["state"] == "queued"
    _run(flow())


# ── requested TLS fails closed (audit F01) ────────────────────────────────────

class _HealthRecorder:
    def __init__(self):
        self.beats = []

    def register(self, name):
        pass

    def beat(self, name, status="OK", detail="", ttl=0.0):
        self.beats.append((name, status, detail))


def _free_port():
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _port_is_closed(port):
    import socket
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def test_plain_http_only_when_tls_was_not_requested(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REMOTE_TLS", raising=False)
    gateway = RemoteGateway(_StubRouter(), _StubMemory(), _StubBus(), config_dir=tmp_path)
    assert gateway._maybe_ssl_context() is None
    assert not (tmp_path / "remote_tls").exists()


def test_requested_tls_issues_and_reuses_a_pinned_certificate(tmp_path, monkeypatch):
    import ssl

    monkeypatch.setenv("ORION_REMOTE_TLS", "1")
    gateway = RemoteGateway(_StubRouter(), _StubMemory(), _StubBus(), config_dir=tmp_path)
    monkeypatch.setattr(gateway, "_lan_ip", lambda: "192.168.1.20")
    assert isinstance(gateway._maybe_ssl_context(), ssl.SSLContext)
    first = (tmp_path / "remote_tls" / "cert.pem").read_bytes()
    assert isinstance(gateway._maybe_ssl_context(), ssl.SSLContext)
    assert (tmp_path / "remote_tls" / "cert.pem").read_bytes() == first   # never re-issued


def test_direct_tailnet_certificate_covers_magicdns_and_bind_ip(tmp_path, monkeypatch):
    from cryptography import x509

    monkeypatch.setenv("ORION_REMOTE_TLS", "1")
    monkeypatch.setenv("ORION_REMOTE_DNS_NAME", "orion-host.tail1234.ts.net")
    monkeypatch.setenv("ORION_REMOTE_HOST", "100.101.102.103")
    gateway = _gateway(tmp_path)
    monkeypatch.setattr(gateway, "_lan_ip", lambda: "192.168.1.20")
    gateway._maybe_ssl_context()
    cert = x509.load_pem_x509_certificate(
        (tmp_path / "remote_tls" / "cert.pem").read_bytes())
    sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "orion-host.tail1234.ts.net" in sans.get_values_for_type(x509.DNSName)
    assert "100.101.102.103" in [str(ip) for ip in sans.get_values_for_type(x509.IPAddress)]


def test_quick_tunnel_uses_complete_replies_and_no_internal_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_REMOTE_EXPOSURE", "quick-tunnel")
    gateway = _gateway(tmp_path)

    async def check():
        client = await _client(gateway)
        try:
            response = await client.get("/")
            page = await response.text()
            assert "const QUICK_TUNNEL=true;" in page
            assert "const route=QUICK_TUNNEL?'/api/chat/frames':'/api/chat/stream';" in page
            assert await gateway._endpoints() == []
        finally:
            await client.close()

    _run(check())


@pytest.mark.parametrize("fault", ["corrupt certificate", "unreadable key", "generation"])
def test_requested_tls_failure_never_serves_plain_http(tmp_path, monkeypatch, fault):
    from orion_core.remote import RemoteTLSError

    monkeypatch.setenv("ORION_REMOTE_TLS", "1")
    monkeypatch.setenv("ORION_REMOTE_FIREWALL", "0")
    telemetry = type("T", (), {"health": _HealthRecorder()})()
    bus = _StubBus()
    gateway = RemoteGateway(_StubRouter(), _StubMemory(), bus, config_dir=tmp_path,
                            telemetry=telemetry)
    monkeypatch.setattr(gateway, "_lan_ip", lambda: "192.168.1.20")
    tls_dir = tmp_path / "remote_tls"
    if fault == "generation":
        def _cannot_write(*_args):
            raise PermissionError("access is denied")
        monkeypatch.setattr(gateway, "_generate_tls_files", _cannot_write)
    else:
        gateway._generate_tls_files(tls_dir, tls_dir / "cert.pem", tls_dir / "key.pem")
        broken = "cert.pem" if fault == "corrupt certificate" else "key.pem"
        (tls_dir / broken).write_text("-----BEGIN GARBAGE-----\nnot a pem\n", encoding="ascii")
        before = (tls_dir / "cert.pem").read_bytes()
    gateway.host, gateway.port = "127.0.0.1", _free_port()

    with pytest.raises(RemoteTLSError) as raised:
        _run(gateway.start())

    expected = "certificate generation" if fault == "generation" else "certificate loading"
    assert raised.value.stage == expected
    assert "ORION_REMOTE_TLS" in raised.value.repair
    assert gateway._runner is None and gateway._https_enabled is False
    assert _port_is_closed(gateway.port)                 # nothing listens, not even HTTP
    assert ("remote", "DOWN", f"TLS {expected} failed") in telemetry.health.beats
    logged = " ".join(str(args[0]) for args in bus.log.emitted)
    assert "NOT started" in logged and "serving plain HTTP" not in logged
    if fault != "generation":
        assert (tls_dir / "cert.pem").read_bytes() == before   # the pinned files are left alone
