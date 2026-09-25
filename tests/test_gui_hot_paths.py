"""
GUI hot paths (Mark XXVI performance).

Under qasync the Qt event loop IS the asyncio loop, so every animation timer
callback runs on the same thread as the audio callback and the face paint. There
are 18 periodic timers across the deck; 8 of them fire faster than every 50 ms.
Work done in those ticks is the most expensive work in ORION per unit of code.

Two defects fixed here, both of the same shape — paying repeatedly for something
that only needed doing once, or not at all:

* ``cursor_overlay._cursor_pos`` declared a ``ctypes.Structure`` subclass INSIDE
  the function, so 33 times a second it built a new class through the ctypes
  metaclass and re-resolved ``windll.user32.GetCursorPos``. Measured 9.14 us;
  hoisted, 1.21 us. It also called ``move()`` on a frameless always-on-top
  window every tick regardless of whether the cursor had moved — a real
  SetWindowPos into the compositor, for nothing.

* The status orb in ``widgets.py`` ticked and requested a repaint while hidden.
  Every other animated widget in the deck already guards on visibility; this one
  was missed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.gui import cursor_overlay as co  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ── cursor position ──────────────────────────────────────────────────────────

def test_the_ctypes_plumbing_is_resolved_once_at_import():
    if sys.platform != "win32":
        pytest.skip("windll path is Windows-only")
    assert co._GET_CURSOR_POS is not None
    assert co._CURSOR_PT is not None and co._CURSOR_REF is not None


def test_the_tick_does_not_rebuild_a_ctypes_structure():
    """The regression guard: a Structure subclass declared inside the function
    is the defect, and it is invisible in a profile that only looks at totals."""
    import inspect
    src = inspect.getsource(co.CursorOverlay._cursor_pos)
    assert "class POINT" not in src, (
        "the ctypes struct is being rebuilt every tick again")
    assert "ctypes.windll" not in src, "windll is being re-resolved every tick"


def test_cursor_pos_returns_a_sane_pair():
    x, y = co.CursorOverlay._cursor_pos(None)
    assert isinstance(x, int) and isinstance(y, int)


def test_cursor_pos_is_cheap_enough_for_33_hz():
    co.CursorOverlay._cursor_pos(None)
    start = time.perf_counter()
    for _ in range(3000):
        co.CursorOverlay._cursor_pos(None)
    per_call_us = (time.perf_counter() - start) * 1e6 / 3000
    # 1.21 us measured; 6 us is loose for a slow box but well under the 9.14 us
    # the in-function class construction cost.
    assert per_call_us < 6.0, "cursor lookup costs %.2f us a call" % per_call_us


def test_cursor_pos_falls_back_when_the_win32_call_is_unavailable(monkeypatch):
    """A locked-down host with no usable user32 must still animate, not crash."""
    monkeypatch.setattr(co, "_GET_CURSOR_POS", None)
    result = co.CursorOverlay._cursor_pos(None)
    assert isinstance(result, tuple) and len(result) == 2


def test_a_raising_win32_call_falls_through(monkeypatch):
    def boom(_ref):
        raise OSError("user32 went away")
    monkeypatch.setattr(co, "_GET_CURSOR_POS", boom)
    result = co.CursorOverlay._cursor_pos(None)
    assert isinstance(result, tuple) and len(result) == 2


# ── redundant window moves ───────────────────────────────────────────────────

def test_the_overlay_only_moves_when_the_cursor_moves(qapp, monkeypatch):
    overlay = co.CursorOverlay.__new__(co.CursorOverlay)
    overlay._last_cursor = (-1, -1)
    overlay._phase = 0.0
    overlay._flare = 0.0
    moves: list[tuple[int, int]] = []
    monkeypatch.setattr(co.CursorOverlay, "_cursor_pos", lambda _s: (500, 400))
    monkeypatch.setattr(co.CursorOverlay, "move",
                        lambda _s, x, y: moves.append((x, y)))
    monkeypatch.setattr(co.CursorOverlay, "update", lambda _s: None)

    for _ in range(10):
        co.CursorOverlay._tick(overlay)
    assert len(moves) == 1, (
        "the window was moved %d times for a stationary cursor" % len(moves))


def test_the_overlay_does_move_when_the_cursor_moves(qapp, monkeypatch):
    overlay = co.CursorOverlay.__new__(co.CursorOverlay)
    overlay._last_cursor = (-1, -1)
    overlay._phase = 0.0
    overlay._flare = 0.0
    moves: list[tuple[int, int]] = []
    positions = iter([(10, 10), (11, 10), (11, 12), (11, 12), (40, 40)])
    monkeypatch.setattr(co.CursorOverlay, "_cursor_pos", lambda _s: next(positions))
    monkeypatch.setattr(co.CursorOverlay, "move",
                        lambda _s, x, y: moves.append((x, y)))
    monkeypatch.setattr(co.CursorOverlay, "update", lambda _s: None)

    for _ in range(5):
        co.CursorOverlay._tick(overlay)
    assert len(moves) == 4, "a real cursor move was dropped"


def test_the_hotspot_offset_is_still_applied(qapp, monkeypatch):
    """The halo anchors its arrow tip on the OS cursor point; an optimisation
    that shifted the sprite would be very visible and very annoying."""
    overlay = co.CursorOverlay.__new__(co.CursorOverlay)
    overlay._last_cursor = (-1, -1)
    overlay._phase = 0.0
    overlay._flare = 0.0
    moves: list[tuple[int, int]] = []
    monkeypatch.setattr(co.CursorOverlay, "_cursor_pos", lambda _s: (500, 400))
    monkeypatch.setattr(co.CursorOverlay, "move",
                        lambda _s, x, y: moves.append((x, y)))
    monkeypatch.setattr(co.CursorOverlay, "update", lambda _s: None)
    co.CursorOverlay._tick(overlay)
    hx, hy = co.CursorOverlay.HOTSPOT
    assert moves == [(500 - hx, 400 - hy)]


# ── the hidden orb ───────────────────────────────────────────────────────────

def _orb_class():
    from orion_core.gui import widgets
    for value in vars(widgets).values():
        if isinstance(value, type) and hasattr(value, "_tick") \
                and hasattr(value, "set_amplitude"):
            return value
    pytest.skip("status orb widget not found")


def test_the_orb_does_no_work_while_hidden(qapp):
    orb = _orb_class()()
    orb.hide()
    orb.set_amplitude(1.0)
    before = (orb.amplitude, orb._pulse, orb.rotation)
    for _ in range(20):
        orb._tick()
    assert (orb.amplitude, orb._pulse, orb.rotation) == before, (
        "the hidden orb is still animating 30 times a second")


def test_the_orb_animates_when_visible(qapp):
    orb = _orb_class()()
    orb.show()
    orb.set_amplitude(1.0)
    before = orb.rotation
    for _ in range(5):
        orb._tick()
    assert orb.rotation != before, "the visibility guard broke the animation"
    assert orb.amplitude > 0.0
    orb.hide()


def test_every_fast_timer_widget_pauses_when_hidden():
    """The general rule, enforced across the deck: anything ticking faster than
    every 50 ms must have some way of not running while nobody can see it."""
    import re
    root = Path(__file__).resolve().parents[1] / "orion_core" / "gui"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        if "QTimer" not in src:
            continue
        intervals = [int(m) for m in re.findall(r"setInterval\(\s*(\d{1,6})\s*\)", src)]
        if not any(i < 50 for i in intervals):
            continue
        # Four mechanisms count, because all four genuinely stop the work:
        # a visibility test in the tick, a hideEvent that stops the timer, the
        # shared animation budget, or an explicit lifecycle that stops the timer
        # and hides together (cursor_overlay's start/stop pair — it is a
        # top-level always-on-top window, so it has no parent to hide it).
        lifecycle = "_timer.stop()" in src and (".hide()" in src or "hide()" in src)
        guarded = ("isVisible()" in src or "def hideEvent" in src
                   or "animation_budget" in src or "AnimationBudget" in src
                   or lifecycle)
        if not guarded:
            offenders.append(path.name)
    assert not offenders, (
        "these animate faster than 20 fps with nothing stopping them while "
        "hidden: " + ", ".join(offenders))
