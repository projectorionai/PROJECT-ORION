"""
Frames spent only where they show.

  "ORION stutters a lot (his graphical face and orb plus his voice) alongside
   his GUI ... Alongside this I do experience quite a bit of lag."

Measured, not guessed. Both hand-painted widgets cost about 7 ms of QPainter
work per frame and both ran at a fixed 30 Hz:

    CentralHud    7.04 ms/frame -> 211 ms/sec  (21% of a core)
    HologramFace  7.66 ms/frame -> 230 ms/sec  (23% of a core)
                                   ---------
                                   441 ms/sec  (44% of one core)

Under qasync the Qt thread IS the asyncio event loop, so that 44% is not
running alongside ORION's brain — it is inside it. Every 7 ms paint is 7 ms the
audio callback does not get, on a pipeline with a hard deadline. That is the
stutter.

The tick MATHS was measured first and came back at 0.9 ms/sec — nothing. The
cost is entirely in painting, which is why the fix is about how often an
unchanged picture is redrawn rather than about making the maths cheaper.

After: 175 ms/sec idle, a 60% reduction, with full 30 Hz restored the instant
anything actually moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.gui.animation_budget import (  # noqa: E402
    ACTIVE_HZ, IDLE_HZ, SETTLE_FRAMES, AnimationBudget,
)


class _Timer:
    """A QTimer stand-in — these tests are about the policy, not about Qt."""

    def __init__(self, interval: int = 33) -> None:
        self._interval = interval
        self.changes: list[int] = []

    def interval(self) -> int:
        return self._interval

    def setInterval(self, value: int) -> None:
        self._interval = int(value)
        self.changes.append(self._interval)


@pytest.fixture
def budget():
    return AnimationBudget(_Timer())


# ── it engages only once motion has genuinely stopped ────────────────────────

def test_it_starts_at_full_rate(budget):
    assert budget.idling is False


def test_a_settled_widget_drops_to_the_idle_rate(budget):
    for _ in range(SETTLE_FRAMES + 2):
        budget.note((0.5, 0.5))
    assert budget.idling is True
    assert budget._timer.interval() == round(1000 / IDLE_HZ)


def test_it_does_not_drop_on_a_brief_pause(budget):
    """A gap between two words must not cause a visible cadence change."""
    for _ in range(SETTLE_FRAMES - 1):
        budget.note((0.5, 0.5))
    assert budget.idling is False


def test_motion_restores_full_rate_on_the_very_next_frame(budget):
    for _ in range(SETTLE_FRAMES + 5):
        budget.note((0.5, 0.5))
    assert budget.idling is True
    budget.note((0.9, 0.5))
    assert budget.idling is False
    assert budget._timer.interval() == round(1000 / ACTIVE_HZ)


def test_a_settled_frame_needs_no_repaint(budget):
    """This is where the saving comes from — not a cheaper paint, no paint."""
    for _ in range(SETTLE_FRAMES + 2):
        budget.note((0.5, 0.5))
    assert budget.note((0.5, 0.5)) is False


def test_a_moving_frame_always_repaints(budget):
    assert budget.note((0.1, 0.0)) is True
    assert budget.note((0.4, 0.0)) is True


def test_tiny_drift_below_the_epsilon_counts_as_still(budget):
    """An eased channel never reaches its target exactly; it converges. Waiting
    for equality would mean never idling at all."""
    value = 0.5
    for _ in range(SETTLE_FRAMES + 4):
        value += 0.0001
        budget.note((value,))
    assert budget.idling is True


def test_a_channel_count_change_counts_as_motion(budget):
    budget.note((0.5, 0.5))
    assert budget.note((0.5, 0.5, 0.5)) is True


def test_wake_forces_full_rate_immediately(budget):
    """For a state change the eased channels have not caught up to yet."""
    for _ in range(SETTLE_FRAMES + 2):
        budget.note((0.5,))
    assert budget.idling is True
    budget.wake()
    assert budget.idling is False


def test_the_idle_rate_is_never_a_freeze():
    """A frozen picture reads as a hang. Calm is not the same as stopped."""
    assert IDLE_HZ >= 5.0


def test_a_custom_idle_rate_is_honoured():
    timer = _Timer()
    budget = AnimationBudget(timer, idle_hz=15.0)
    for _ in range(SETTLE_FRAMES + 2):
        budget.note((0.5,))
    assert timer.interval() == round(1000 / 15.0)


def test_a_dead_timer_does_not_raise(budget):
    class _Dead:
        def interval(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

        def setInterval(self, value):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    budget._timer = _Dead()
    for _ in range(SETTLE_FRAMES + 2):
        budget.note((0.5,))       # must not raise from inside a paint tick


def test_the_interval_is_not_set_when_already_correct(budget):
    for _ in range(SETTLE_FRAMES + 10):
        budget.note((0.5,))
    assert budget._timer.changes.count(round(1000 / IDLE_HZ)) == 1


# ── the widgets actually use it ──────────────────────────────────────────────

def _gui(module: str) -> str:
    return (Path(__file__).resolve().parents[1] / "orion_core" / "gui"
            / module).read_text(encoding="utf-8", errors="replace")


def test_the_hud_is_budgeted():
    source = _gui("hud.py")
    assert "AnimationBudget" in source
    assert "_budget.note(" in source


def test_the_face_is_budgeted():
    source = _gui("face.py")
    assert "AnimationBudget" in source
    assert "_budget.note(" in source


def test_the_face_idles_more_gently_than_the_hud():
    """The face is never truly still — scanlines drift, the gaze wanders, it
    blinks — so dropping it as far as the HUD would coarsen motion meant to
    read as alive."""
    assert "idle_hz=15.0" in _gui("face.py")


def test_the_face_still_skips_everything_when_hidden():
    """The cheapest frame is the one never computed. This guard predates the
    budget and must survive it."""
    assert "if not self.isVisible():" in _gui("face.py")


def test_the_face_excludes_its_always_moving_channels():
    """Including _pulse or _scan would mean 'moving' every single frame and the
    budget could never engage."""
    source = _gui("face.py")
    marker = source.index("_budget.note((")
    window = source[marker:marker + 320]
    assert "_pulse" not in window
    assert "_scan" not in window
    assert "self.amplitude" in window


# ── end to end, with real widgets ────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_hud_really_drops_its_rate_when_idle(qapp):
    from orion_core.gui.hud import CentralHud

    hud = CentralHud()
    hud.resize(400, 300)
    hud.set_amplitude(0.0)
    for _ in range(SETTLE_FRAMES + 40):
        hud._tick()
    assert hud.timer.interval() > 33, "still repainting a still picture at 30 Hz"


def test_the_hud_returns_to_full_rate_when_he_speaks(qapp):
    from orion_core.gui.hud import CentralHud

    hud = CentralHud()
    hud.resize(400, 300)
    hud.set_amplitude(0.0)
    for _ in range(SETTLE_FRAMES + 40):
        hud._tick()
    hud.set_amplitude(0.9)
    hud._tick()
    assert hud.timer.interval() == 33


def test_the_face_drops_its_rate_when_idle(qapp, monkeypatch):
    from orion_core.gui.face import HologramFace

    face = HologramFace()
    face.resize(300, 300)
    # Force visibility rather than relying on show()+processEvents(): under the
    # offscreen platform, and depending on what earlier Qt tests left behind,
    # isVisible() is not reliable, which made this test order-dependent. The
    # subject here is the BUDGET dropping the rate, not Qt's window-manager
    # reporting — so pin the one thing that is incidental to it.
    monkeypatch.setattr(face, "isVisible", lambda: True)
    # A blink is real motion the budget SHOULD honour at full rate, and its
    # timing is drawn from the global RNG (random.randint) — which is what made
    # this test order-dependent: a blink landing inside the settle window kept
    # resetting the idle counter. Push the next blink past the window so we
    # measure the quiet-idle drop itself, not blink timing.
    face._blink = 1.0
    face._blink_in = 10_000
    face.set_amplitude(0.0)
    for _ in range(SETTLE_FRAMES + 60):
        face._tick()
    assert face.timer.interval() > 33
