"""
Tests for debugger.py — a real `python -m pdb` subprocess driver, not a
simulation. These launch actual pdb subprocesses against small fixture
scripts (pdb is stdlib, always available, no network/external binary
needed), so unlike docker_control's tests these are not mocked — they
exercise the genuine start/command/stop protocol end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.debugger import DebugSession, DebuggerService


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


def _write_fixture(tmp_path: Path) -> Path:
    script = tmp_path / "fixture.py"
    script.write_text(
        "def add(a, b):\n"
        "    total = a + b\n"
        "    return total\n"
        "\n"
        "x = 1\n"
        "y = 2\n"
        "result = add(x, y)\n"
        "print(result)\n",
        encoding="utf-8",
    )
    return script


# ── DebugSession ─────────────────────────────────────────────────────────────

def test_start_rejects_a_missing_script(tmp_path):
    session = DebugSession(str(tmp_path / "nope.py"))
    result = session.start()
    assert not result.ok
    assert "no such file" in result.text.lower()


def test_start_launches_pdb_and_reaches_the_first_prompt(tmp_path):
    script = _write_fixture(tmp_path)
    session = DebugSession(str(script))
    try:
        result = session.start()
        assert result.ok
        assert session.is_running()
    finally:
        session.stop()


def test_cannot_start_a_second_session_while_one_is_running(tmp_path):
    script = _write_fixture(tmp_path)
    session = DebugSession(str(script))
    try:
        assert session.start().ok
        second = session.start()
        assert not second.ok
        assert "already running" in second.text.lower()
    finally:
        session.stop()


def test_send_next_steps_through_the_script(tmp_path):
    script = _write_fixture(tmp_path)
    session = DebugSession(str(script))
    try:
        assert session.start().ok
        result = session.send("next")
        assert result.ok
    finally:
        session.stop()


def test_send_print_reports_a_variable_value(tmp_path):
    script = _write_fixture(tmp_path)
    session = DebugSession(str(script))
    try:
        assert session.start().ok
        # step past `x = 1` and `y = 2` so both are bound, then inspect one.
        session.send("next")
        session.send("next")
        result = session.send("p x")
        assert result.ok
        assert "1" in result.text
    finally:
        session.stop()


def test_send_continue_runs_to_completion(tmp_path):
    script = _write_fixture(tmp_path)
    session = DebugSession(str(script))
    try:
        assert session.start().ok
        result = session.send("continue")
        assert result.ok
        # either pdb prints the script's own stdout ("3") or reports exit.
        assert "3" in result.text or "exit" in result.text.lower()
    finally:
        session.stop()


def test_send_without_a_running_session_fails_cleanly(tmp_path):
    session = DebugSession(str(_write_fixture(tmp_path)))
    result = session.send("next")
    assert not result.ok
    assert "no debug session" in result.text.lower()


def test_stop_terminates_the_process(tmp_path):
    script = _write_fixture(tmp_path)
    session = DebugSession(str(script))
    session.start()
    result = session.stop()
    assert result.ok
    assert not session.is_running()


def test_stop_when_nothing_is_running_is_a_safe_no_op(tmp_path):
    session = DebugSession(str(_write_fixture(tmp_path)))
    result = session.stop()
    assert result.ok
    assert "no debug session" in result.text.lower()


def test_breakpoint_then_continue_stops_at_the_line(tmp_path):
    script = _write_fixture(tmp_path)
    session = DebugSession(str(script))
    try:
        assert session.start().ok
        break_result = session.send(f"break {script}:3")
        assert break_result.ok
        cont_result = session.send("continue")
        assert cont_result.ok
        # stopped inside add(), 'total' should be reachable even before its
        # own assignment line finishes (pdb stops *before* executing line 3).
        state = session.send("p a, b")
        assert "1" in state.text and "2" in state.text
    finally:
        session.stop()


# ── DebuggerService ──────────────────────────────────────────────────────────

def test_service_reports_not_running_initially():
    service = DebuggerService(_StubBus())
    assert service.is_running() is False


def test_service_command_without_a_session_fails_cleanly():
    service = DebuggerService(_StubBus())
    result = service.command("next")
    assert not result.ok
    assert "start one first" in result.text.lower()


def test_service_stop_without_a_session_is_a_safe_no_op():
    service = DebuggerService(_StubBus())
    result = service.stop()
    assert result.ok


def test_service_start_command_stop_lifecycle(tmp_path):
    script = _write_fixture(tmp_path)
    service = DebuggerService(_StubBus())
    try:
        start_result = service.start(str(script))
        assert start_result.ok
        assert service.is_running()
        step_result = service.command("next")
        assert step_result.ok
    finally:
        service.stop()
    assert not service.is_running()


def test_service_refuses_a_second_concurrent_session(tmp_path):
    script = _write_fixture(tmp_path)
    service = DebuggerService(_StubBus())
    try:
        assert service.start(str(script)).ok
        second = service.start(str(script))
        assert not second.ok
    finally:
        service.stop()
