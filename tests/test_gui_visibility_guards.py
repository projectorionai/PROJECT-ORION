"""
Visibility guards on timer-driven GUI work (performance pass).

Several QTimers kept running their full refresh body while the widget they
belong to was genuinely off-screen — a Command Deck page the user had
navigated away from, or the Command Centre window after "closing" it (its
closeEvent hides rather than destroys). Each timer target now starts with
``if not self.isVisible(): return``, the same one-line convention
automation_deck/mission_deck already used.

The subtlety these tests exist to lock down: three of these widgets prime
their own first paint by calling the refresh directly from __init__, before
anything is visible. A naive top-line guard would silently swallow that
priming call and leave the widget blank until its first post-show tick (up
to 30s for the deadline list). So each guarded method was split into a
guarded public target and an unguarded private body, with __init__ calling
the body — both halves of that arrangement are asserted here.

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.bus import OrionBus  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _CountingLedger:
    """Stands in for TokenUsageLedger, counting the expensive query."""

    def __init__(self) -> None:
        self.calls = 0

    def timeseries_by(self, *_a, **_k):
        self.calls += 1
        return {}


class _CountingMemory:
    def __init__(self) -> None:
        self.recall_calls = 0

    def recall(self, *_a, **_k):
        self.recall_calls += 1
        return []

    def remember(self, *_a, **_k):
        return None


class _CountingReminders:
    def __init__(self) -> None:
        self.active_calls = 0

    def active(self):
        self.active_calls += 1
        return []

    def cancel(self):
        return None


# ── TokenUsageGraph ──────────────────────────────────────────────────────────

def test_token_graph_primes_itself_at_construction(_app):
    from orion_core.gui.token_graph import TokenUsageGraph

    ledger = _CountingLedger()
    TokenUsageGraph(ledger)
    assert ledger.calls == 1, "the first paint must bypass the visibility guard"


def test_token_graph_refresh_is_idle_while_hidden(_app):
    from orion_core.gui.token_graph import TokenUsageGraph

    ledger = _CountingLedger()
    graph = TokenUsageGraph(ledger)
    before = ledger.calls
    graph.refresh()
    assert ledger.calls == before, "hidden panel must not query the ledger"


def test_token_graph_refresh_runs_once_shown(_app):
    from orion_core.gui.token_graph import TokenUsageGraph

    ledger = _CountingLedger()
    graph = TokenUsageGraph(ledger)
    graph.show()
    before = ledger.calls
    graph.refresh()
    assert ledger.calls == before + 1
    graph.hide()


def test_token_graph_attach_ledger_renders_immediately(_app):
    from orion_core.gui.token_graph import TokenUsageGraph

    graph = TokenUsageGraph(None)      # no ledger yet — nothing to query
    ledger = _CountingLedger()
    graph.attach_ledger(ledger)
    assert ledger.calls == 1, "attaching a ledger must render it, not wait for a tick"


# ── HologramFace ─────────────────────────────────────────────────────────────

def test_hologram_face_tick_is_idle_while_hidden(_app):
    from orion_core.gui.face import HologramFace

    face = HologramFace()
    before = face._t
    face._tick()
    assert face._t == before, "hidden face must not advance its animation clock"


def test_hologram_face_tick_advances_once_shown(_app):
    from orion_core.gui.face import HologramFace

    face = HologramFace()
    face.show()
    before = face._t
    face._tick()
    assert face._t > before
    face.hide()


# ── entrepreneur.py widgets (the TOOLKIT page) ──────────────────────────────

def test_world_clock_primes_itself_at_construction(_app):
    from orion_core.gui.entrepreneur import WorldClock

    clock = WorldClock()
    assert clock.labels[0].text().strip(), "clocks must be populated before first show"


def test_world_clock_tick_is_idle_while_hidden(_app):
    from orion_core.gui.entrepreneur import WorldClock

    clock = WorldClock()
    clock.labels[0].setText("SENTINEL")
    clock._tick()
    assert clock.labels[0].text() == "SENTINEL"


def test_world_clock_tick_renders_once_shown(_app):
    from orion_core.gui.entrepreneur import WorldClock

    clock = WorldClock()
    clock.show()
    clock.labels[0].setText("SENTINEL")
    clock._tick()
    assert clock.labels[0].text() != "SENTINEL"
    clock.hide()


def test_deadline_countdown_primes_its_list_at_construction(_app):
    from orion_core.gui.entrepreneur import DeadlineCountdown

    memory = _CountingMemory()
    DeadlineCountdown(OrionBus(), memory)
    assert memory.recall_calls == 1, "the deadline list must load before first show"


def test_deadline_countdown_render_is_idle_while_hidden(_app):
    from orion_core.gui.entrepreneur import DeadlineCountdown

    widget = DeadlineCountdown(OrionBus(), _CountingMemory())
    widget._deadlines.append(("ship it", "2099-01-01"))
    widget._render()
    assert widget.list.count() == 0, "hidden widget must not rebuild its list"


def test_deadline_countdown_render_runs_once_shown(_app):
    from orion_core.gui.entrepreneur import DeadlineCountdown

    widget = DeadlineCountdown(OrionBus(), _CountingMemory())
    widget.show()
    widget._deadlines.append(("ship it", "2099-01-01"))
    widget._render()
    assert widget.list.count() == 1
    widget.hide()


def test_reminders_widget_primes_its_list_at_construction(_app):
    from orion_core.gui.entrepreneur import RemindersWidget

    reminders = _CountingReminders()
    RemindersWidget(OrionBus(), reminders)
    assert reminders.active_calls == 1, "reminders must render before first show"


def test_reminders_refresh_is_idle_while_hidden(_app):
    from orion_core.gui.entrepreneur import RemindersWidget

    reminders = _CountingReminders()
    widget = RemindersWidget(OrionBus(), reminders)
    before = reminders.active_calls
    widget._refresh()
    assert reminders.active_calls == before


def test_reminders_refresh_runs_once_shown(_app):
    from orion_core.gui.entrepreneur import RemindersWidget

    reminders = _CountingReminders()
    widget = RemindersWidget(OrionBus(), reminders)
    widget.show()
    before = reminders.active_calls
    widget._refresh()
    assert reminders.active_calls == before + 1
    widget.hide()


def test_focus_timer_is_deliberately_not_guarded(_app):
    # A running Pomodoro must keep counting down while the user works on
    # another Command Deck page — guarding this one would be a functional
    # regression, not a performance win. Locked down so a future sweep
    # doesn't "helpfully" add the guard everywhere.
    from orion_core.gui.entrepreneur import FocusTimer

    widget = FocusTimer(OrionBus())
    before = widget._remaining
    widget._tick()
    assert widget._remaining == before - 1, "the focus timer must tick while hidden"


# ── OrionCoreWindow ──────────────────────────────────────────────────────────

class _CountingAvatar:
    def __init__(self) -> None:
        self.ticks = 0

    def attach(self, _face) -> None:
        pass

    def tick(self) -> None:
        self.ticks += 1

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _core_window(tmp_path):
    from orion_core.memory import MemoryAgent, OrionMemoryMatrix
    from orion_core.gui.core_window import OrionCoreWindow

    bus = OrionBus()
    matrix = OrionMemoryMatrix(tmp_path / "core.db", tmp_path, bus)
    return OrionCoreWindow(bus, MemoryAgent(matrix, bus))


def test_avatar_tick_is_idle_while_the_window_is_hidden(_app, tmp_path):
    window = _core_window(tmp_path)
    avatar = _CountingAvatar()
    window.attach_avatar(avatar)
    window._tick_avatar_if_visible()
    assert avatar.ticks == 0


def test_avatar_ticks_once_the_window_is_shown(_app, tmp_path):
    window = _core_window(tmp_path)
    avatar = _CountingAvatar()
    window.attach_avatar(avatar)
    window.show()
    window._tick_avatar_if_visible()
    assert avatar.ticks == 1
    window.hide()


def test_avatar_tick_wrapper_is_safe_with_no_avatar_attached(_app, tmp_path):
    window = _core_window(tmp_path)
    window.show()
    window._tick_avatar_if_visible()   # must not raise
    window.hide()


def test_clock_tick_is_idle_while_the_window_is_hidden(_app, tmp_path):
    window = _core_window(tmp_path)
    window.clock_label.setText("SENTINEL")
    window._tick_clock()
    assert window.clock_label.text() == "SENTINEL"


def test_clock_tick_updates_once_the_window_is_shown(_app, tmp_path):
    window = _core_window(tmp_path)
    window.show()
    window.clock_label.setText("SENTINEL")
    window._tick_clock()
    assert window.clock_label.text() != "SENTINEL"
    window.hide()


# ── CommandCentreWindow ──────────────────────────────────────────────────────
#
# Its real __init__ wants seven live subsystems (worker, display, workspace,
# dispatcher…). The guard under test runs before any of them are touched, so
# the window is built by bypassing __init__ and initialising only its Qt base
# — a real widget with a real isVisible(), and nothing else needed.

def _bare_command_centre():
    from PyQt6.QtWidgets import QMainWindow
    from orion_core.gui.command_centre import CommandCentreWindow

    window = CommandCentreWindow.__new__(CommandCentreWindow)
    QMainWindow.__init__(window)
    return window


class _ExplodingTelemetry:
    """Every attribute access is a failure — _refresh must never get here
    while hidden. If the guard regresses, this raises loudly."""

    def __getattr__(self, name):
        raise AssertionError(f"hidden Command Centre touched telemetry.{name}")


def test_command_centre_refresh_is_idle_while_hidden(_app):
    window = _bare_command_centre()
    window.telemetry = _ExplodingTelemetry()
    window._refresh()   # must return before touching anything


def test_command_centre_refresh_proceeds_once_shown(_app):
    # The mirror image: once visible, _refresh does reach its body — proved
    # by it getting far enough to touch the (still absent) real widgets.
    window = _bare_command_centre()
    window.show()
    with pytest.raises(AttributeError):
        window._refresh()
    window.hide()
