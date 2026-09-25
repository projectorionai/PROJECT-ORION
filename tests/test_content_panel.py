"""
A surface for what ORION found.

He already fetches real material — the briefing pulls news stories with titles
and links, the research engine gathers sources the same way — and all of it
arrived as spoken prose and a line in the activity log. A headline read aloud
cannot be clicked, and a log is not a reading surface: by the third story the
first has scrolled away and the URL was never anywhere you could reach.

Nothing here fetches or ranks anything. It renders rows the rest of ORION
already produces, which is why the producers needed no reshaping to feed it.

Two behaviours are load-bearing and easy to get wrong:

  * it is populated whether or not it is visible, and it does NOT raise itself
    when results land. A window that appears over your work in the middle of a
    briefing is not a feature.
  * a row with no title is dropped. A card showing an empty box and a URL is
    worse than one fewer card.

The panel is rendered in a subprocess; the wiring is read from source.
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
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.gui.content_panel import MAX_CARDS, ContentPanel, _host

panel = ContentPanel()
panel.resize(460, 480)

out = {{"empty_at_start": panel.count}}

rows = [
    {{"title": "Bank of England holds rates", "url": "https://www.bbc.co.uk/news/1", "topic": "Markets"}},
    {{"title": "Transformer variant cuts cost", "url": "https://arxiv.org/abs/1", "topic": "AI"}},
    {{"title": "An item with no link at all", "url": "", "topic": "Misc"}},
    {{"title": "", "url": "https://example.com/skipped"}},
    {{"not_a_row": True}},
    "a bare string",
]
out["added"] = panel.show_results(rows)
out["count"] = panel.count
out["titles"] = panel.titles()
out["visible_after_results"] = panel.isVisible()

# Newest first.
out["added_second"] = panel.show_results([{{"title": "Newest story", "url": "https://x/y"}}])
out["first_title"] = panel.titles()[0]

# The cap.
panel.show_results([{{"title": f"filler {{i}}", "url": "https://x/{{i}}"}}
                    for i in range(MAX_CARDS + 15)])
out["capped_at"] = panel.count
out["max_cards"] = MAX_CARDS

panel.clear()
out["after_clear"] = panel.count

out["host_plain"] = _host("https://www.theguardian.com/uk/rail")
out["host_none"] = _host("not a url")
print("@@" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def panel() -> dict:
    proc = subprocess.run([sys.executable, "-c", _PROBE.format(
        root=str(ROOT).replace("\\", "/"))], cwd=ROOT,
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail(f"probe produced nothing:\n{proc.stdout[-1500:]}")


# ── rendering ────────────────────────────────────────────────────────────────

def test_it_starts_empty_and_says_so(panel):
    assert panel["empty_at_start"] == 0


def test_real_rows_become_cards(panel):
    assert panel["added"] == 3
    assert "Bank of England holds rates" in panel["titles"]


def test_a_row_with_no_title_is_dropped(panel):
    """A card showing an empty box and a URL is worse than one fewer card."""
    assert not any(title.strip() == "" for title in panel["titles"])


def test_rubbish_rows_do_not_break_it(panel):
    """A dict with no title, a bare string — neither should raise."""
    assert panel["count"] == 3


def test_an_item_with_no_link_still_appears(panel):
    """Not every source has a URL; dropping it would lose the finding."""
    assert "An item with no link at all" in panel["titles"]


def test_newest_arrives_at_the_top(panel):
    assert panel["added_second"] == 1
    assert panel["first_title"] == "Newest story"


def test_the_stack_is_capped(panel):
    """A view of what just happened, not an archive — the knowledge store is
    the archive."""
    assert panel["capped_at"] == panel["max_cards"]


def test_it_can_be_emptied(panel):
    assert panel["after_clear"] == 0


def test_the_link_is_shown_readably(panel):
    """A full URL on a card is noise; the host is the useful part."""
    assert panel["host_plain"] == "theguardian.com"
    assert panel["host_none"] == ""


# ── the behaviour that matters most ──────────────────────────────────────────

def test_results_do_not_make_a_window_appear(panel):
    """It is filled whether or not anyone is looking. A panel that raises
    itself over your work during a morning briefing is not a feature."""
    assert panel["visible_after_results"] is False


# ── wiring ───────────────────────────────────────────────────────────────────

def _source(name: str) -> str:
    return (ROOT / "orion_core" / name).read_text(encoding="utf-8")


def test_the_signal_has_a_producer_and_a_consumer():
    """ORION has shipped a declared, subscribed, never-emitted signal before."""
    assert "content_results" in _source("bus.py")
    assert "content_results.emit" in _source("briefing.py"), (
        "the briefing's stories never reach the panel"
    )
    assert "content_results.emit" in _source("research.py"), (
        "research sources never reach the panel"
    )
    window = _source("gui/core_window.py")
    assert "content_results.connect" in window
    assert "def _on_content_results" in window


def test_the_panel_is_reachable_on_request():
    window = _source("gui/core_window.py")
    assert "def open_content_panel" in window
    assert '"results", "findings", "what_you_found"' in window


def test_arriving_results_are_acknowledged_by_the_face():
    """The wordless "that landed" — people look at a thing before commenting."""
    window = _source("gui/core_window.py")
    start = window.index("def _on_content_results")
    body = window[start:start + 800]
    assert '_to_face("glance"' in body
