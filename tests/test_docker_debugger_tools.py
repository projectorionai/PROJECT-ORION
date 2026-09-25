"""
Tests for the docker/debugger dispatcher tools (dispatch_files.py) — the
voice/text-reachable surface over docker_control.py and debugger.py.
docker_tool is tested with docker_control's functions monkeypatched
(hermetic, no real Docker install needed). debugger_tool is tested with a
real DebuggerService driving a real pdb subprocess against a tiny fixture
script, same as test_debugger.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import docker_control as dc
from orion_core.data import ToolResult
from orion_core.debugger import DebuggerService
from orion_core.dispatcher import OrionDispatcher


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


def _dispatcher() -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _StubBus()
    d.debugger = None
    return d


def _fixture(tmp_path: Path) -> Path:
    script = tmp_path / "fixture.py"
    script.write_text("x = 1\nprint(x)\n", encoding="utf-8")
    return script


# ── docker_tool ──────────────────────────────────────────────────────────────

async def test_docker_list_reports_unavailable_message(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: False)
    d = _dispatcher()
    result = await d.docker_tool({"action": "list"})
    assert result.ok
    assert "not found" in result.text.lower()


async def test_docker_list_renders_containers(monkeypatch):
    monkeypatch.setattr(
        dc, "list_containers",
        lambda show_all=True: [{"Names": "web", "Image": "nginx", "Status": "Up"}])
    d = _dispatcher()
    result = await d.docker_tool({"action": "list"})
    assert result.ok
    assert "web" in result.text and "nginx" in result.text


async def test_docker_images_renders_images(monkeypatch):
    monkeypatch.setattr(
        dc, "list_images",
        lambda: [{"Repository": "nginx", "Tag": "latest", "Size": "142MB"}])
    d = _dispatcher()
    result = await d.docker_tool({"action": "images"})
    assert result.ok
    assert "nginx:latest" in result.text


async def test_docker_start_calls_start_container(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dc, "start_container", lambda name: calls.append(name) or ToolResult("started"))
    d = _dispatcher()
    result = await d.docker_tool({"action": "start", "container": "web"})
    assert result.ok
    assert calls == ["web"]


async def test_docker_logs_passes_tail(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dc, "container_logs",
        lambda name, tail: calls.append((name, tail)) or ToolResult("log text"))
    d = _dispatcher()
    result = await d.docker_tool({"action": "logs", "container": "web", "tail": 25})
    assert result.ok
    assert calls == [("web", 25)]


async def test_docker_unknown_action_fails_cleanly():
    d = _dispatcher()
    result = await d.docker_tool({"action": "explode"})
    assert not result.ok
    assert "list" in result.text


async def test_docker_status_calls_describe(monkeypatch):
    monkeypatch.setattr(dc, "describe", lambda: "Docker is available and the daemon is responding.")
    d = _dispatcher()
    result = await d.docker_tool({"action": "status"})
    assert result.ok
    assert "available" in result.text.lower()


def test_docker_tool_is_registered_in_the_handler_table():
    d = _dispatcher()
    assert d.handler_table()["docker"] == d.docker_tool


# ── debugger_tool ────────────────────────────────────────────────────────────

async def test_debugger_tool_reports_unavailable_when_no_service_attached():
    d = _dispatcher()
    result = await d.debugger_tool({"action": "status"})
    assert not result.ok
    assert "not available" in result.text.lower()


async def test_debugger_tool_start_requires_a_path():
    d = _dispatcher()
    d.debugger = DebuggerService(d.bus)
    result = await d.debugger_tool({"action": "start"})
    assert not result.ok


async def test_debugger_tool_full_lifecycle(tmp_path):
    script = _fixture(tmp_path)
    d = _dispatcher()
    d.debugger = DebuggerService(d.bus)
    try:
        start_result = await d.debugger_tool({"action": "start", "path": str(script)})
        assert start_result.ok
        status_result = await d.debugger_tool({"action": "status"})
        assert "running" in status_result.text.lower()
        step_result = await d.debugger_tool({"action": "next"})
        assert step_result.ok
    finally:
        stop_result = await d.debugger_tool({"action": "stop"})
        assert stop_result.ok


async def test_debugger_tool_print_builds_the_p_command(tmp_path):
    script = _fixture(tmp_path)
    d = _dispatcher()
    d.debugger = DebuggerService(d.bus)
    try:
        await d.debugger_tool({"action": "start", "path": str(script)})
        result = await d.debugger_tool({"action": "print", "expression": "x"})
        assert result.ok
    finally:
        await d.debugger_tool({"action": "stop"})


async def test_debugger_tool_unknown_action_fails_cleanly():
    d = _dispatcher()
    d.debugger = DebuggerService(d.bus)
    result = await d.debugger_tool({"action": "explode"})
    assert not result.ok


def test_debugger_tool_is_registered_in_the_handler_table():
    d = _dispatcher()
    assert d.handler_table()["debugger"] == d.debugger_tool
