"""
Clean exits, and ORION knowing what actually broke.

Both come from one report: every shutdown ended in a wall of tracebacks that
described the bus being deleted rather than any real fault, and self-repair
announced that it had captured something without saying what.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from orion_core.selfrepair import Incident, SelfRepairAgent  # noqa: E402


class _Log:
    def __init__(self): self.lines = []
    def emit(self, text): self.lines.append(text)


class _DeadLog:
    def emit(self, text):
        raise RuntimeError("wrapped C/C++ object of type OrionBus has been deleted")


class _Bus:
    def __init__(self, log=None):
        self.log = log or _Log()
        self.dashboard_event = _Log()
        self.connection_state = type("_S", (), {"connect": lambda *a: None})()


def _agent(bus=None):
    agent = SelfRepairAgent.__new__(SelfRepairAgent)
    agent.bus = bus or _Bus()
    agent.telemetry = None
    agent._loop = None
    agent._tasks = set()
    agent._shutting_down = False
    return agent


def _incident(**kw):
    fields = dict(id="inc-1", at="now", error_type="AttributeError",
                  message="'NoneType' object has no attribute 'render'",
                  file="C:/orion/orion_core/gui/face.py", line=214,
                  module="orion_core.gui.face")
    fields.update(kw)
    return Incident(**fields)


# ── the shutdown wall of tracebacks ──────────────────────────────────────────

def test_a_loop_error_after_the_bus_dies_does_not_raise():
    """The handler outlived the bus, so reporting a pending task raised FROM
    INSIDE the exception handler — which asyncio then reported again, once per
    task. A dozen tracebacks on every clean exit, none about a real fault."""
    agent = _agent(_Bus(log=_DeadLog()))
    agent._loop_exception_handler(None, {"message": "Task was destroyed"})


def test_a_loop_error_is_still_reported_while_the_bus_is_alive():
    agent = _agent()
    agent._loop_exception_handler(None, {"message": "Task was destroyed"})
    assert any("Task was destroyed" in line for line in agent.bus.log.lines)


def test_shutdown_cancels_repairs_still_in_flight():
    """Otherwise asyncio destroys them pending and prints a traceback each.

    Shutdown is now two steps, and the split is the whole point.
    ``shutdown()`` cancels but deliberately KEEPS its references; ``drain()``
    then awaits them. Clearing the set inside shutdown() — which is what this
    test used to assert — dropped the last reference while every task was still
    in the 'cancelling' state, so the loop never got a turn to deliver a single
    CancelledError. The user's console showed the result: over a hundred
    "Task was destroyed but it is pending!" lines on one exit.
    """
    async def _run():
        agent = _agent()

        async def _slow():
            await asyncio.sleep(30)

        task = asyncio.get_running_loop().create_task(_slow())
        agent._tasks.add(task)

        agent.shutdown()
        assert task.cancelling() or task.cancelled() or task.done()
        assert agent._tasks, (
            "references dropped mid-cancellation — the task will be destroyed "
            "while still pending")

        settled = await agent.drain(timeout=2.0)
        assert settled == 1
        assert task.done(), "the cancellation was never actually delivered"
        assert agent._tasks == set()

    asyncio.run(_run())


def test_shutdown_stops_new_repairs_being_scheduled():
    """A coroutine created after shutdown begins is never awaited — which is
    exactly the "coroutine ... was never awaited" warning."""
    agent = _agent()
    agent.shutdown()
    agent._schedule_auto_repair(_incident())
    assert agent._tasks == set()


def test_shutdown_is_safe_with_no_loop_and_twice_over():
    agent = _agent()
    agent.shutdown()
    agent.shutdown()


# ── he says what went wrong ──────────────────────────────────────────────────

def test_the_diagnosis_names_the_fault_in_plain_terms():
    text = _agent().diagnose(_incident())
    assert "does not exist" in text


def test_the_diagnosis_says_where():
    text = _agent().diagnose(_incident())
    assert "face.py" in text and "214" in text


def test_the_diagnosis_says_whether_he_can_fix_it():
    agent = _agent()
    missing = agent.diagnose(_incident(error_type="ModuleNotFoundError",
                                       message="No module named 'chess'"))
    assert "install it himself" in missing
    denied = agent.diagnose(_incident(error_type="PermissionError"))
    assert "cannot repair" in denied


def test_a_transient_fault_is_named_as_external():
    text = _agent().diagnose(_incident(error_type="ConnectionError",
                                       transient=True))
    assert "self-recovering" in text


def test_an_informational_incident_is_not_presented_as_repairable():
    text = _agent().diagnose(_incident(actionable=False))
    assert "nothing to repair" in text


def test_an_unknown_error_type_still_gets_a_useful_line():
    text = _agent().diagnose(_incident(error_type="WeirdCustomError"))
    assert "unrecognised" in text
    assert "face.py" in text


def test_a_fault_with_no_traceback_still_diagnoses():
    text = _agent().diagnose(_incident(file="", line=0, module=""))
    assert "without a traceback" in text


def test_the_diagnosis_is_one_line():
    """It goes to the log beside the summary; a paragraph would bury it."""
    for error in ("AttributeError", "KeyError", "RuntimeError", "Whatever"):
        assert "\n" not in _agent().diagnose(_incident(error_type=error))


# ── standby is silence until he is asked for ─────────────────────────────────

def _worker():
    from orion_core.live_worker import GenAILiveWorker
    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    worker.quiet_mode = False
    worker.standby_mode = False
    worker.bus = type("_B", (), {
        "log": _Log(), "state": _Log(),
    })()
    worker.spoken = []
    worker._say = lambda text: worker.spoken.append(text)
    return worker


def test_going_to_standby_makes_him_silent_not_merely_unprompted():
    """quiet_mode alone still answered. The ask was: say nothing at all."""
    worker = _worker()
    assert worker._handle_quiet_command("orion go to standby") is True
    assert worker.standby_mode is True
    assert worker._held_in_standby("what is the weather") is True


def test_standby_collapses_him_to_the_orb():
    """face3d's _ORB_STATES turns STANDBY into the orb form, so the visual
    matches the behaviour rather than a materialised ORION sitting mute."""
    worker = _worker()
    worker._handle_quiet_command("orion, standby")
    assert "STANDBY" in worker.bus.state.lines


def test_the_wake_phrase_the_user_actually_uses_releases_him(monkeypatch):
    from orion_core import audio_devices
    ready = audio_devices.DeviceCheck(
        "output", "test", True, True, 1, "test output", "output verified")
    ready_in = audio_devices.DeviceCheck(
        "input", "test", True, True, 2, "test input", "input verified")
    monkeypatch.setattr(audio_devices, "verify",
                        lambda kind, spec="", **_k: ready if kind == "output" else ready_in)
    monkeypatch.setattr(audio_devices, "preferred_index", lambda kind: 1 if kind == "output" else 2)
    worker = _worker()
    worker._handle_quiet_command("go into standby")
    assert worker._handle_quiet_command("orion i need you") is True
    assert worker.standby_mode is False
    assert worker.quiet_mode is False
    assert "LISTENING" in worker.bus.state.lines


def test_the_wake_phrase_is_never_swallowed_by_standby():
    """Standby is full silence, so nothing else will release it — if the wake
    phrase were held back too, he could not be brought out at all."""
    worker = _worker()
    worker._handle_quiet_command("standby")
    assert worker._held_in_standby("orion i need you") is False


def test_a_power_command_still_reaches_him_in_standby():
    """Otherwise standby would be a state ORION cannot be shut down from."""
    worker = _worker()
    worker._handle_quiet_command("standby")
    assert worker._held_in_standby("orion shut down") is False


def test_nothing_is_held_back_when_he_is_not_in_standby():
    assert _worker()._held_in_standby("what is the weather") is False


def test_he_acknowledges_once_and_then_stops():
    worker = _worker()
    worker._handle_quiet_command("orion go to standby")
    assert worker.spoken == ["Standing by."]


def test_explicit_standby_disables_microphone_capture():
    worker = _worker()

    class _Mic:
        def __init__(self): self.enabled = True; self.drained = False
        def set_enabled(self, enabled): self.enabled = enabled
        def _drain(self): self.drained = True

    mic = _Mic()
    worker.microphone_enabled = True
    worker.mic = mic
    worker.tool_busy = True
    worker._clear_turn = lambda: None

    worker._handle_quiet_command("go into standby")

    assert worker.standby_mode is True
    assert worker.microphone_enabled is False
    assert mic.enabled is False
    assert mic.drained is True
