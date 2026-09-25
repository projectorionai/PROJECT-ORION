"""
Tests for MCPHost's reconnect/health/supervision (Mark XXI, Track E2) —
previously a crashed or dropped MCP server stayed gone until the next full
app restart, with no health signal anywhere. MCPServerConn.start/stop are
monkeypatched throughout (no real subprocess/JSON-RPC needed); these tests
exercise the reconnect/backoff/supervision LOGIC in mcp_host.py directly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.mcp_host as mh
from orion_core.mcp_host import MCPHost, MCPServerConn


import pytest


def test_config_migration_preserves_explicitly_disabled_servers():
    config = {"_revision": "earlier", "servers": {
        "fetch": {"enabled": False, "command": "uvx", "args": ["mcp-server-fetch"]},
        "notion": {"enabled": False, "command": "npx", "args": ["notion"]},
    }}

    assert mh.migrate_config(config)
    assert config["servers"]["fetch"]["enabled"] is False
    assert config["servers"]["notion"]["enabled"] is False
    assert config["servers"]["duckduckgo"]["enabled"] is True


@pytest.fixture(autouse=True)
def _own_tool_cache(tmp_path, monkeypatch):
    """Each test gets its own tool-list cache, so one test's successful start
    cannot make the next test's server start lazily."""
    monkeypatch.setattr(mh, "MCP_CATALOGUE_PATH", tmp_path / "mcp_catalogue.json")


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubHealthRegistry:
    def __init__(self) -> None:
        self.beats: list[tuple[str, str, str]] = []

    def register(self, name):
        pass

    def beat(self, name, status, detail=""):
        self.beats.append((name, status, detail))


class _StubTelemetry:
    def __init__(self) -> None:
        self.health = _StubHealthRegistry()


def _config_with(servers: dict) -> dict:
    return {"servers": servers}


def _spec(enabled=True) -> dict:
    return {"enabled": enabled, "command": "npx", "args": [], "env": {}, "description": "x"}


# ── MCPServerConn.is_alive() ─────────────────────────────────────────────────

def test_is_alive_false_before_start():
    conn = MCPServerConn("gmail", _spec(), _StubBus())
    assert conn.is_alive() is False


def test_is_alive_true_when_ready_and_process_running():
    conn = MCPServerConn("gmail", _spec(), _StubBus())
    conn.ready = True

    class _FakeProc:
        returncode = None

    conn.proc = _FakeProc()
    assert conn.is_alive() is True


def test_is_alive_false_when_process_has_exited():
    conn = MCPServerConn("gmail", _spec(), _StubBus())
    conn.ready = True

    class _FakeProc:
        returncode = 1   # exited

    conn.proc = _FakeProc()
    assert conn.is_alive() is False


def test_server_stderr_does_not_echo_a_configured_credential():
    secret = "example-private-token-12345"
    conn = MCPServerConn("gmail", {"command": "npx", "env": {
        "GMAIL_AUTH_TOKEN": secret,
    }}, _StubBus())

    class _Reader:
        def __init__(self):
            self.lines = [f"server error: {secret}\n".encode(), b""]

        async def readline(self):
            return self.lines.pop(0)

    conn.proc = type("Proc", (), {"stderr": _Reader()})()

    async def _capture():
        conn._drain_stderr()
        await conn._stderr_task

    asyncio.run(_capture())
    assert secret not in " ".join(conn._stderr_tail)
    assert "[redacted]" in " ".join(conn._stderr_tail)


def test_server_tool_reply_does_not_echo_a_configured_credential():
    secret = "example-private-token-12345"
    conn = MCPServerConn("gmail", {"command": "npx", "env": {
        "GMAIL_AUTH_TOKEN": secret,
    }}, _StubBus())

    async def _reply(*_args):
        return {"content": [{"type": "text", "text": f"bad token: {secret}"}]}

    conn._request = _reply
    reply = asyncio.run(conn.call_tool("search_emails", {}))
    assert secret not in reply
    assert "[redacted]" in reply


def test_server_tool_failure_does_not_echo_a_configured_credential():
    secret = "example-private-token-12345"
    host = MCPHost(_StubBus())
    conn = MCPServerConn("gmail", {"command": "npx", "env": {
        "GMAIL_AUTH_TOKEN": secret,
    }}, _StubBus())
    conn.tools = [{"name": "search_emails"}]

    async def _fail(*_args):
        raise RuntimeError(f"bad token: {secret}")

    conn.call_tool = _fail
    host.servers["gmail"] = conn
    reply = asyncio.run(host.call("gmail", "search_emails", {}))
    assert secret not in reply
    assert "[redacted]" in reply


# ── connect_all(): health beats ──────────────────────────────────────────────

def test_connect_all_beats_ok_for_a_successful_server(monkeypatch):
    telemetry = _StubTelemetry()
    host = MCPHost(_StubBus(), telemetry)
    monkeypatch.setattr(mh, "load_config", lambda: _config_with({"gmail": _spec()}))

    async def _start(self):
        self.ready = True
        self.tools = [{"name": "send_email"}]

        class _FakeProc:
            returncode = None
        self.proc = _FakeProc()
        return True

    monkeypatch.setattr(MCPServerConn, "start", _start)
    asyncio.run(host.connect_all())
    assert ("mcp.gmail", "OK", "1 tool(s)") in telemetry.health.beats
    assert host.health_snapshot() == {"gmail": "OK"}


def test_connect_all_beats_down_for_a_failed_server(monkeypatch):
    telemetry = _StubTelemetry()
    host = MCPHost(_StubBus(), telemetry)
    monkeypatch.setattr(mh, "load_config", lambda: _config_with({"gmail": _spec()}))

    async def _false(self):
        return False
    monkeypatch.setattr(MCPServerConn, "start", _false)
    asyncio.run(host.connect_all())
    assert any(b[0] == "mcp.gmail" and b[1] == "DOWN" for b in telemetry.health.beats)
    assert host.health_snapshot() == {"gmail": "DOWN"}


def test_connect_all_without_telemetry_never_raises(monkeypatch):
    host = MCPHost(_StubBus(), telemetry=None)
    monkeypatch.setattr(mh, "load_config", lambda: _config_with({"gmail": _spec()}))

    async def _false(self):
        return False
    monkeypatch.setattr(MCPServerConn, "start", _false)
    asyncio.run(host.connect_all())   # must not raise


# ── reconnect() ──────────────────────────────────────────────────────────────

def test_reconnect_succeeds_and_updates_servers(monkeypatch):
    host = MCPHost(_StubBus())
    monkeypatch.setattr(mh, "load_config", lambda: _config_with({"gmail": _spec()}))

    async def _start(self):
        self.ready = True
        self.tools = [{"name": "send_email"}]
        class _FakeProc:
            returncode = None
        self.proc = _FakeProc()
        return True
    monkeypatch.setattr(MCPServerConn, "start", _start)

    ok = asyncio.run(host.reconnect("gmail"))
    assert ok is True
    assert "gmail" in host.servers
    assert host._reconnect_attempts["gmail"] == 0


def test_reconnect_stops_the_dead_connection_first(monkeypatch):
    host = MCPHost(_StubBus())
    monkeypatch.setattr(mh, "load_config", lambda: _config_with({"gmail": _spec()}))

    old = MCPServerConn("gmail", _spec(), _StubBus())
    stopped = []

    async def _record_stop():
        stopped.append(1)
    old.stop = _record_stop
    host.servers["gmail"] = old

    async def _start(self):
        self.ready = True
        class _FakeProc:
            returncode = None
        self.proc = _FakeProc()
        return True
    monkeypatch.setattr(MCPServerConn, "start", _start)

    asyncio.run(host.reconnect("gmail"))
    assert stopped == [1]


def test_reconnect_of_a_now_disabled_server_removes_it(monkeypatch):
    host = MCPHost(_StubBus())
    host._specs["gmail"] = _spec(enabled=True)
    host.servers["gmail"] = MCPServerConn("gmail", _spec(), _StubBus())
    monkeypatch.setattr(mh, "load_config", lambda: _config_with({"gmail": _spec(enabled=False)}))

    ok = asyncio.run(host.reconnect("gmail"))
    assert ok is False
    assert "gmail" not in host.servers
    assert "gmail" not in host._specs


def test_reconnect_of_an_unknown_server_returns_false(monkeypatch):
    host = MCPHost(_StubBus())
    monkeypatch.setattr(mh, "load_config", lambda: _config_with({}))
    assert asyncio.run(host.reconnect("ghost")) is False


# ── supervise() ──────────────────────────────────────────────────────────────

def test_supervise_reconnects_a_dead_server_and_fires_the_callback(monkeypatch):
    host = MCPHost(_StubBus())
    host._specs["gmail"] = _spec()
    # no live connection at all -> immediately due for reconnect

    async def _start(self):
        self.ready = True
        self.tools = [{"name": "send_email"}]
        class _FakeProc:
            returncode = None
        self.proc = _FakeProc()
        return True
    monkeypatch.setattr(MCPServerConn, "start", _start)
    monkeypatch.setattr(mh, "load_config", lambda: _config_with({"gmail": _spec()}))

    fired = []

    async def _run_one_tick():
        task = asyncio.create_task(
            host.supervise(interval_s=0, on_reconnect=lambda n, c: fired.append((n, c))))
        await asyncio.sleep(0.05)
        host.stop_supervising()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

    asyncio.run(_run_one_tick())
    assert fired and fired[0][0] == "gmail"
    assert "gmail" in host.servers


def test_supervise_skips_a_server_that_is_already_alive(monkeypatch):
    host = MCPHost(_StubBus())
    host._specs["gmail"] = _spec()
    alive = MCPServerConn("gmail", _spec(), _StubBus())
    alive.ready = True
    class _FakeProc:
        returncode = None
    alive.proc = _FakeProc()
    host.servers["gmail"] = alive

    reconnect_calls = []
    original_reconnect = host.reconnect

    async def _spy(name):
        reconnect_calls.append(name)
        return await original_reconnect(name)
    host.reconnect = _spy

    async def _run_one_tick():
        task = asyncio.create_task(host.supervise(interval_s=0))
        await asyncio.sleep(0.05)
        host.stop_supervising()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

    asyncio.run(_run_one_tick())
    assert reconnect_calls == []


def test_supervise_never_raises_when_reconnect_itself_raises(monkeypatch):
    host = MCPHost(_StubBus())
    host._specs["gmail"] = _spec()

    async def _boom(name):
        raise RuntimeError("boom")
    host.reconnect = _boom

    async def _run_one_tick():
        task = asyncio.create_task(host.supervise(interval_s=0))
        await asyncio.sleep(0.05)
        host.stop_supervising()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()

    asyncio.run(_run_one_tick())   # must not raise out of the test


def test_backoff_grows_with_repeated_failures():
    host = MCPHost(_StubBus())
    now = 1000.0
    host._schedule_next_reconnect("gmail", now)
    first_delay = host._next_reconnect_at["gmail"] - now
    host._schedule_next_reconnect("gmail", now)
    second_delay = host._next_reconnect_at["gmail"] - now
    assert second_delay > first_delay


def test_stop_supervising_ends_the_loop_promptly():
    host = MCPHost(_StubBus())

    async def _run():
        task = asyncio.create_task(host.supervise(interval_s=0.01))
        await asyncio.sleep(0.02)
        host.stop_supervising()
        await asyncio.wait_for(task, timeout=2.0)   # must actually finish

    asyncio.run(_run())


# ── idle parking ─────────────────────────────────────────────────────────────

def _live_host(monkeypatch, names=("gmail",)):
    host = MCPHost(_StubBus())
    monkeypatch.setattr(mh, "load_config",
                        lambda: _config_with({n: _spec() for n in names}))
    starts: list[str] = []

    async def _start(self):
        starts.append(self.name)
        self.ready = True
        self.tools = [{"name": "send_email"}]

        class _FakeProc:
            returncode = None
        self.proc = _FakeProc()
        return True

    async def _stop(self):
        self.proc = None
        self.ready = False

    async def _call_tool(self, tool, arguments):
        return f"{self.name}:{tool}"

    monkeypatch.setattr(MCPServerConn, "start", _start)
    monkeypatch.setattr(MCPServerConn, "stop", _stop)
    monkeypatch.setattr(MCPServerConn, "call_tool", _call_tool)
    asyncio.run(host.connect_all())
    return host, starts


def test_an_idle_server_is_parked_but_its_tools_stay_listed(monkeypatch):
    host, _ = _live_host(monkeypatch, ("gmail", "fetch"))
    host.idle_park_s = 60.0
    host.servers["gmail"].last_used -= 120.0
    parked = asyncio.run(host.park_idle())
    assert parked == ["gmail"]
    assert not host.servers["gmail"].is_alive()
    assert host.health_snapshot() == {"gmail": "IDLE", "fetch": "OK"}
    assert host.catalogue()["gmail"]["tools"][0]["name"] == "send_email"
    assert "paused to save memory" in host.list_servers()


def test_a_call_wakes_a_parked_server_first(monkeypatch):
    host, starts = _live_host(monkeypatch)
    host.idle_park_s = 60.0
    host.servers["gmail"].last_used -= 120.0
    asyncio.run(host.park_idle())
    woken: list[str] = []
    host._on_reconnect = lambda name, conn: woken.append(name)
    assert asyncio.run(host.call("gmail", "send_email", {})) == "gmail:send_email"
    assert starts == ["gmail", "gmail"] and woken == ["gmail"]
    assert "gmail" not in host.parked and host.servers["gmail"].is_alive()


def test_manual_reconnect_clears_a_parked_server(monkeypatch):
    host, starts = _live_host(monkeypatch)
    host.idle_park_s = 60.0
    host.servers["gmail"].last_used -= 120.0
    asyncio.run(host.park_idle())

    assert asyncio.run(host.reconnect("gmail")) is True
    assert host.health_snapshot() == {"gmail": "OK"}
    assert "gmail" not in host.parked
    assert asyncio.run(host.call("gmail", "send_email", {})) == "gmail:send_email"
    assert starts == ["gmail", "gmail"]


def test_a_busy_or_recent_server_is_never_parked(monkeypatch):
    host, _ = _live_host(monkeypatch, ("gmail", "fetch"))
    host.idle_park_s = 60.0
    host.servers["gmail"].last_used -= 120.0

    async def _park_while_busy():
        async with host.servers["gmail"]._lock:
            return await host.park_idle()

    assert asyncio.run(_park_while_busy()) == []


def test_memory_pressure_parks_sooner_and_zero_disables_everyday_parking(monkeypatch):
    host, _ = _live_host(monkeypatch)
    host.idle_park_s = 0.0
    host.servers["gmail"].last_used -= 300.0
    assert asyncio.run(host.park_idle()) == []
    host.set_pressure_parking(120.0)
    assert asyncio.run(host.park_idle()) == ["gmail"]
    host.set_pressure_parking(None)
    assert host._park_limit() is None


# ── lazy start from the cached tool list ─────────────────────────────────────

def test_a_started_server_saves_its_tools_and_the_next_boot_starts_it_lazily(monkeypatch):
    host, starts = _live_host(monkeypatch)              # first boot: starts it
    assert starts == ["gmail"]
    assert "gmail" in mh._load_catalogue()

    host2, starts2 = _live_host(monkeypatch)            # next boot: from cache
    assert starts2 == []
    assert "gmail" in host2.parked
    assert host2.catalogue()["gmail"]["tools"][0]["name"] == "send_email"
    assert host2.health_snapshot() == {"gmail": "IDLE"}
    assert asyncio.run(host2.call("gmail", "send_email", {})) == "gmail:send_email"
    assert starts2 == ["gmail"] and "gmail" not in host2.parked


def test_a_changed_launch_spec_ignores_the_cached_tools(monkeypatch):
    _live_host(monkeypatch)
    catalogue = mh._load_catalogue()
    catalogue["gmail"]["fingerprint"] = "stale"
    mh.MCP_CATALOGUE_PATH.write_text(__import__("json").dumps(catalogue), encoding="utf-8")
    _, starts = _live_host(monkeypatch)
    assert starts == ["gmail"]


def test_lazy_start_can_be_switched_off(monkeypatch):
    _live_host(monkeypatch)
    monkeypatch.setenv("ORION_MCP_LAZY", "0")
    _, starts = _live_host(monkeypatch)
    assert starts == ["gmail"]
