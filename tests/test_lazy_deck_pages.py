"""Pages you never open should not be built.

All 26 deck pages were constructed before the deck existed, including five
native surfaces. Measured on this machine: the first QWebEngineView costs
207 MB and 2.5 s; a second one costs a further 73 MB, 0.9 s and another
Chromium child process. ORION's own code is 82 MB — nearly all of his ~750 MB
is runtime like this.

GLOBE, CHESS and MISSION are built on first navigation now. They qualified
because nothing touches them between construction and the user opening the
page; BRAIN deliberately stays eager, because the dispatcher pulses it on
every tool call and the face uses it as its centre.

Rendered in a subprocess: a QApplication built inline hangs this suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication, QLabel
app = QApplication([])
from orion_core.bus import OrionBus
from orion_core.gui.unified_dashboard import UnifiedDashboard

built = []
def factory(name):
    def _f():
        built.append(name)
        return QLabel("real " + name)
    return _f

def boom():
    raise RuntimeError("kaboom")

deck = UnifiedDashboard(OrionBus(), [
    ("LOG", QLabel("eager")),
    ("GLOBE", factory("GLOBE")),
    ("CHESS", factory("CHESS")),
    ("MISSION", factory("MISSION")),
    ("BAD", boom),
])
out = {{}}
out["built_at_construction"] = list(built)
out["page_names"] = deck.page_names()
out["stack_count"] = deck.stack.count()
out["globe_built_before"] = deck.page_is_built("GLOBE")

deck.show_page_named("chess")
out["built_after_opening_chess"] = list(built)
out["chess_widget_is_real"] = deck._pages[2][1].text() == "real CHESS"
out["stack_count_after"] = deck.stack.count()

w = deck.page_widget("globe")
out["page_widget_builds"] = list(built)
out["page_widget_returns_real"] = getattr(w, "text", lambda: "")() == "real GLOBE"

deck.show_page_named("BAD")
out["survived_a_broken_factory"] = deck.stack.count()
out["mission_never_built"] = "MISSION" not in built
out["failed_page_not_built"] = not deck.page_is_built("BAD")
out["unknown_page_not_built"] = not deck.page_is_built("DOES_NOT_EXIST")
deck._factories["BAD"] = lambda: QLabel("recovered")
out["failed_page_can_retry"] = deck.page_widget("BAD").text() == "recovered"
out["retry_preserves_stack_count"] = deck.stack.count()
print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def probe():
    code = _PROBE.format(root=str(ROOT))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=300)
    marker = [l for l in result.stdout.splitlines() if l.startswith("@@")]
    if not marker:
        pytest.fail(f"probe failed:\n{result.stdout[-1500:]}\n{result.stderr[-2500:]}")
    return json.loads(marker[0][2:])


def test_nothing_deferred_is_built_up_front(probe):
    assert probe["built_at_construction"] == []


def test_every_page_still_exists_to_navigate_to(probe):
    """Deferring must not hide a page: the name list drives the tabs, the
    command palette, voice navigation and the swarm's node graph."""
    assert probe["page_names"] == ["LOG", "GLOBE", "CHESS", "MISSION", "BAD"]
    assert probe["stack_count"] == 5


def test_opening_a_page_builds_exactly_that_one(probe):
    assert probe["built_after_opening_chess"] == ["CHESS"]
    assert probe["chess_widget_is_real"] is True
    assert probe["stack_count_after"] == 5, "the stack lost or gained a page"


def test_the_real_widget_replaces_the_placeholder(probe):
    """Callers that need the live page — starting the camera, say — must get
    the real thing, not the stand-in."""
    assert probe["page_widget_returns_real"] is True
    assert probe["page_widget_builds"] == ["CHESS", "GLOBE"]


def test_a_page_that_fails_to_build_does_not_take_the_deck_with_it(probe):
    """Losing one page is a bad afternoon; losing the deck is a broken
    assistant."""
    assert probe["survived_a_broken_factory"] == 5


def test_a_page_never_opened_is_never_built(probe):
    assert probe["mission_never_built"] is True


def test_failed_page_can_be_retried_without_claiming_it_was_built(probe):
    assert probe["failed_page_not_built"]
    assert probe["unknown_page_not_built"]
    assert probe["failed_page_can_retry"]
    assert probe["retry_preserves_stack_count"] == 5


def test_the_heavy_pages_are_actually_deferred_in_the_app():
    """A mechanism nothing uses saves nothing."""
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    for name in ("globe", "chess_view", "mission_deck"):
        assert f"def {name}() -> Any:" in source, f"{name} is still built eagerly"


def test_the_overview_stays_eager():
    """The dispatcher can pulse the 2D overview before it is opened."""
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "swarm_view = CommandOverview(" in source
    assert "def swarm_view() -> Any:" not in source
