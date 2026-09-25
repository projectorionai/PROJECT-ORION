"""
The Command Deck actually filling the screen, not just claiming to.

  "when ORION starts up the command deck (2nd part of the GUI) it doesn't
   maximise fully ... it tells me that it's fully maximised but it just shows up
   on the top left as 'maximised' but it's not actually fully maximised."

That is the exact signature of ``showMaximized()`` on a window whose platform
peer does not exist yet: Qt records the maximised STATE but computes the
maximised GEOMETRY against a screen it has not resolved, so the frame never
grows. It bites the SECOND window (the deck) far more than the first (the
face), which is precisely what was reported.

The fix applies the maximised state after the window is shown and the event
loop has run once, with an explicit resize to the work area as a fallback for
platforms that ignore the state flag alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _app_source() -> str:
    return (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8", errors="replace")


def test_no_layout_path_uses_bare_show_maximized():
    """Every placement must go through the robust helper. A raw
    showMaximized() in a layout function is the bug coming back."""
    source = _app_source()
    # Isolate the three layout functions.
    start = source.index("def _apply_unified_layout")
    end = source.index("async def run_application")
    layouts = source[start:end]
    assert "showMaximized()" not in layouts, (
        "a layout path still calls showMaximized() directly")
    assert layouts.count("_maximise_properly(") >= 5


def test_the_helper_defers_to_the_next_tick():
    """The whole point: the not-yet-realised case is fixed by re-applying on
    the next event-loop tick."""
    source = _app_source()
    helper = source[source.index("def _maximise_properly"):
                    source.index("def _apply_unified_layout")]
    assert "QTimer.singleShot(0" in helper
    assert "WindowMaximized" in helper


def test_the_helper_sets_an_explicit_full_geometry():
    """Belt and braces: even if the maximised flag is later dropped (a
    window-flags change rebuilds the peer), the window still fills the work
    area rather than snapping to a tiny default."""
    source = _app_source()
    helper = source[source.index("def _maximise_properly"):
                    source.index("def _apply_unified_layout")]
    assert "availableGeometry()" in helper
    assert "setGeometry(area)" in helper


def test_the_helper_runs_against_a_real_offscreen_window(monkeypatch):
    """Exercise the real code path once, headlessly. Offscreen Qt will not
    truly maximise, but the helper must run without raising and must set a
    geometry and a maximised state."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])
    from orion_core.app import _maximise_properly

    win = QWidget()
    win.resize(200, 150)          # start deliberately small, like the bug
    _maximise_properly(win)
    app.processEvents()           # let the singleShot(0) fire

    assert win.isVisible()
    assert win.windowState() & Qt.WindowState.WindowMaximized, (
        "the window was not put into the maximised state")
    win.close()
