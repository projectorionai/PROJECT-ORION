"""
The Command Centre thought stream reveals ORION's thoughts character-by-character
(a live "typing" effect) rather than pasting finished paragraphs, and queues
overlapping thoughts.  Headless (offscreen Qt).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ORION_REMOTE_ACCESS", "0")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _centre(_app):
    from orion_core.bus import OrionBus
    from orion_core.gui.command_centre import CommandCentreWindow

    cc = CommandCentreWindow(OrionBus(), None, None, None, None, None, None)
    cc._timer.stop()  # don't let the 1 s refresh touch the None telemetry
    # SHOW it. The typewriter reveals characters one at a time only when the
    # panel is on screen — animating into a hidden widget costs ~70 ms of every
    # second on the thread that also carries audio, for pixels nobody sees. A
    # fixture that never shows the window was testing the wrong branch.
    cc.show()
    return cc


def _tick(cc, n):
    for _ in range(n):
        cc._type_tick()


def test_thought_types_out_incrementally(_app):
    cc = _centre(_app)
    cc._on_thought({"at": "09:00:00", "kind": "reflection", "text": "Watching the queue."})
    _tick(cc, 4)
    mid = cc.thought_box.toPlainText()
    assert "▌" in mid                      # a live caret is trailing
    assert mid.replace("▌", "") != "09:00:00 ◈ Watching the queue."  # not all at once
    _tick(cc, 60)
    done = cc.thought_box.toPlainText()
    assert "Watching the queue." in done
    assert "▌" not in done                 # caret cleared when finished


def test_overlapping_thoughts_queue(_app):
    cc = _centre(_app)
    cc._on_thought({"at": "09:00:00", "kind": "reflection", "text": "First thought here."})
    cc._on_thought({"at": "09:00:01", "kind": "decision", "text": "Second thought here."})
    _tick(cc, 5)
    assert "Second thought" not in cc.thought_box.toPlainText()
    assert len(cc._thought_queue) == 1
    _tick(cc, 60)                          # finish the first
    cc._begin_next_thought()               # app uses a singleShot pause; drive it
    _tick(cc, 60)
    final = cc.thought_box.toPlainText()
    assert "First thought here." in final and "Second thought here." in final
    assert "▌" not in final


def test_decision_and_reflection_markers(_app):
    cc = _centre(_app)
    cc._on_thought({"at": "09:00:00", "kind": "decision", "text": "Act now."})
    _tick(cc, 40)
    assert "⚙" in cc.thought_box.toPlainText()   # decisions get the gear marker


def test_a_hidden_panel_does_not_animate(_app):
    """Measured at 1.53 ms per tick and 45 Hz: ~70 ms of every second on the
    event loop, spent revealing text into a widget that is not on screen. The
    thought still arrives complete — it simply stops being animated."""
    cc = _centre(_app)
    cc.hide()
    cc._on_thought({"at": "09:00:00", "kind": "reflection", "text": "Unseen thought."})
    cc._type_tick()
    text = cc.thought_box.toPlainText()
    assert "Unseen thought." in text, "the thought was lost, not just un-animated"
    assert "▌" not in text, "a caret is still being drawn for nobody"
    assert not cc._type_timer.isActive(), "the timer is still ticking unseen"
