"""
The self-repair spiral: "Cannot enter into task", five times in a row.

From the user's log, on shutdown:

    RuntimeError: Cannot enter into task <Task-207 name='Task-207'
    coro=<SelfRepairAgent.attempt_auto_repair() ...>> while another task
    <Task-1 coro=<run_application() ...>> is being executed
      File ".../asyncio/events.py", line 89, in _run
    REPAIR: captured [inc-29] RuntimeError ...
    REPAIR: captured [inc-30] RuntimeError ...   (and 31, 32, 33)
    Health degraded components: self_repair

Nothing there was a defect in ORION's code. The chain was:

  1. shutdown spoke the farewell, which PUMPED QT to wait for the sentence to
     finish (_await_farewell — it awaits now, see the last test);
  2. qasync dispatches every asyncio callback as a Qt timer event, so pumping
     Qt from inside run_application() ran a pending repair task's first step
     RE-ENTRANTLY;
  3. asyncio refused, raising the RuntimeError above;
  4. the loop exception handler CAPTURED IT AS AN INCIDENT, which scheduled a
     repair, which was dispatched by the next pump of the same wait loop — and
     round again, until shutdown finished.

So the fault was manufactured entirely by the handling of the previous one, and
it pinned self_repair DEGRADED on the way out.

Three independent things are locked in here, because any one of them alone
leaves the spiral one regression away from returning:

  * self-repair is disarmed BEFORE the farewell wait (ordering, in app.py);
  * a fault in the repair machinery is never itself an incident (the cut);
  * a repair task is not created while re-entrantly dispatched (the guard).
"""

from __future__ import annotations

import asyncio
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.selfrepair import SelfRepairAgent  # noqa: E402


class _Signal:
    def __init__(self):
        self.messages: list[str] = []

    def emit(self, *args):
        self.messages.append(" ".join(str(a) for a in args))

    def connect(self, *a, **k):
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Health:
    def __init__(self):
        self.status: dict[str, tuple[str, str]] = {}

    def register(self, name, **k):
        pass

    def beat(self, name, status="OK", detail="", ttl=0.0):
        self.status[name] = (status, detail)


class _Telemetry:
    def __init__(self):
        self.health = _Health()
        self.metrics = type("M", (), {"incr": lambda *a, **k: None})()
        self.log = type("L", (), {"recent": lambda *a, **k: []})()


def _agent(monkeypatch, auto="0"):
    monkeypatch.setenv("ORION_AUTOREPAIR", auto)
    return SelfRepairAgent(_StubBus(), telemetry=_Telemetry(), router=None)


def _reentrancy_error() -> RuntimeError:
    """The literal exception from the log."""
    return RuntimeError(
        "Cannot enter into task <Task-207 name='Task-207' "
        "coro=<SelfRepairAgent.attempt_auto_repair() running>> while another "
        "task <Task-1 coro=<run_application() running>> is being executed")


# ── the cut: repair-machinery faults are not incidents ───────────────────────

def test_the_reentrancy_error_is_recognised_as_self_inflicted(monkeypatch):
    agent = _agent(monkeypatch)
    assert agent._is_self_inflicted(_reentrancy_error())


def test_it_never_becomes_an_incident(monkeypatch):
    agent = _agent(monkeypatch)
    agent._loop_exception_handler(None, {"exception": _reentrancy_error()})
    assert agent._incidents == {}, "the repairer filed a bug report about itself"


def test_it_never_degrades_self_repair(monkeypatch):
    """The user's actual complaint: 'health degraded components: self_repair'."""
    agent = _agent(monkeypatch)
    for _ in range(5):                       # inc-29 … inc-33
        agent._loop_exception_handler(None, {"exception": _reentrancy_error()})
    assert agent.telemetry.health.status.get("self_repair", ("OK", ""))[0] == "OK"


def test_it_is_reported_once_not_five_times(monkeypatch):
    """Silence would hide a real scheduling regression; five lines is the spam."""
    agent = _agent(monkeypatch)
    for _ in range(5):
        agent._loop_exception_handler(None, {"exception": _reentrancy_error()})
    said = [m for m in agent.bus.log.messages if "repair machinery" in m]
    assert len(said) == 1, f"expected one explanation, got {len(said)}"


def test_a_traceback_through_the_repairer_is_self_inflicted(monkeypatch):
    """Message matching alone is brittle — provenance is the stronger signal."""
    agent = _agent(monkeypatch)

    def _inside_the_repairer():
        # A frame whose code object genuinely lives in selfrepair.py.
        return agent._diff_lines("a\n", None)      # raises inside the module

    try:
        _inside_the_repairer()
    except Exception as exc:
        assert agent._is_self_inflicted(exc)
    else:
        raise AssertionError("expected the helper to raise")


def test_a_real_defect_is_still_captured_and_still_degrades(monkeypatch):
    """The cut must not blunt the thing it protects."""
    agent = _agent(monkeypatch)
    try:
        raise ValueError("a genuine defect in ORION's own logic")
    except ValueError as exc:
        agent._loop_exception_handler(None, {"exception": exc})
    assert len(agent._incidents) == 1
    assert agent.telemetry.health.status["self_repair"][0] == "DEGRADED"


def test_an_unrelated_runtimeerror_is_still_captured(monkeypatch):
    """RuntimeError is far too common to blanket-ignore."""
    agent = _agent(monkeypatch)
    try:
        raise RuntimeError("dictionary changed size during iteration")
    except RuntimeError as exc:
        agent._loop_exception_handler(None, {"exception": exc})
    assert len(agent._incidents) == 1


# ── the guard: no task creation while re-entrantly dispatched ────────────────

def test_no_repair_task_is_created_while_a_task_is_current(monkeypatch):
    """Reproduces step 2 of the chain.

    Modelling this correctly matters. Under plain asyncio a
    call_soon_threadsafe callback NEVER runs with a task current, so the
    re-entrancy simply cannot occur and a naive test passes without exercising
    anything. qasync is what makes it possible: it dispatches asyncio callbacks
    as Qt timer events, so a Qt pump inside a coroutine runs them **immediately,
    on the current task's stack**. That is what is emulated here.
    """
    agent = _agent(monkeypatch, auto="1")

    async def _drive():
        loop = asyncio.get_running_loop()
        agent._loop = loop
        incident = type("I", (), {"error_type": "X", "file": "", "line": 0})()
        created: list = []
        real_create = loop.create_task
        loop.create_task = lambda *a, **k: (created.append(1), real_create(*a, **k))[1]
        # A Qt pump: the queued callback fires here and now, re-entrantly.
        loop.call_soon_threadsafe = lambda fn, *a: fn(*a)
        try:
            agent._schedule_auto_repair(incident)
            assert not created, "a repair task was created re-entrantly"
        finally:
            loop.create_task = real_create
            agent._shutting_down = True

    asyncio.run(_drive())


def test_the_repair_still_happens_once_the_pump_is_over(monkeypatch):
    """Deferring must not mean dropping — the fix cannot cost real repairs."""
    agent = _agent(monkeypatch, auto="1")

    async def _drive():
        loop = asyncio.get_running_loop()
        agent._loop = loop
        incident = type("I", (), {"error_type": "X", "file": "", "line": 0})()
        ran: list = []

        async def _fake_repair(inc):
            ran.append(inc)

        agent.attempt_auto_repair = _fake_repair
        loop.call_soon_threadsafe = lambda fn, *a: fn(*a)   # re-entrant pump
        agent._schedule_auto_repair(incident)
        assert not ran, "ran re-entrantly"
        await asyncio.sleep(0.25)          # the pump ends; the loop idles
        assert ran, "the deferred repair never ran at all"

    asyncio.run(_drive())


def test_the_deferral_is_bounded(monkeypatch):
    """A program that never leaves its pump drops the repair, not loops."""
    agent = _agent(monkeypatch, auto="1")
    source = inspect.getsource(agent._schedule_auto_repair)
    assert "deferrals" in source and "< 5" in source


# ── the ordering: disarmed before the farewell wait ──────────────────────────

def test_self_repair_is_shut_down_before_the_farewell():
    """The root cause was _await_farewell() pumping Qt while repairs were still
    queued. Nothing repairable may be queued by the time it waits, or the
    spiral starts again."""
    app = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py")
    text = app.read_text(encoding="utf-8", errors="replace")
    disarm = text.find('_safe("self-repair", selfrepair.shutdown)')
    farewell = text.find("_await_farewell(worker.speech)")
    assert disarm != -1 and farewell != -1
    assert disarm < farewell, (
        "selfrepair.shutdown() must run BEFORE the farewell wait")


def test_the_farewell_wait_awaits_instead_of_pumping_qt():
    """Pumping Qt from inside run_application() steps other tasks
    re-entrantly ("Cannot enter into task ..."), for every task, not only
    repairs — the turn watchdog, sentinel and reminders all hit it on
    2026-09-23. The wait yields to the running loop instead."""
    app = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py")
    text = app.read_text(encoding="utf-8", errors="replace")
    start = text.index("async def _await_farewell(")
    # Up to the next top-level statement (the next unindented line).
    body = text[start:start + 1 + re.search(r"\n\S", text[start + 1:]).start()]
    assert ".processEvents(" not in body      # a call, not the docstring's history
    assert "await _await_farewell(worker.speech)" in text
