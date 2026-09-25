"""Asking ORION to open a page has to open that page, or say it cannot.

`show_page_named` matched with ``name.lower() in page_name.lower()`` and took
the first hit. That is backwards for anything a person actually says: the
REQUEST had to be a substring of the PAGE, so "the command centre page" matched
nothing at all, while a one-letter request matched whichever page happened to
contain that letter first.

Both failures looked identical from outside, because `interface_control`
reported "done" either way — the command went onto the bus, nothing matched at
the other end, and the deck did not move while ORION said it had.

Offline: the resolver is pure over a page list, so no QApplication.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.gui.unified_dashboard import UnifiedDashboard  # noqa: E402

PAGES = ["DEVELOPMENT", "CODING", "WIDGETS", "TOOLKIT", "MARKETING", "STUDIO",
         "DESIGN", "FASHION", "ENTERTAINMENT", "LIBRARY", "OPS", "COGNITION",
         "AUTOMATION", "COMMAND CENTRE", "PLUGINS", "DIAGNOSTICS", "GLOBE",
         "LOG", "MEMORY", "TELEMETRY", "CHESS", "SECURITY", "BRAIN",
         "MISSION", "WORKBENCH"]


def _resolve(spoken: str):
    deck = types.SimpleNamespace(
        _pages=[(name, object()) for name in PAGES],
        _NAV_FILLER=UnifiedDashboard._NAV_FILLER,
        _NAV_ALIASES=UnifiedDashboard._NAV_ALIASES,
    )
    return UnifiedDashboard.resolve_page_name(deck, spoken)


@pytest.mark.parametrize("spoken,expected", [
    ("LOG", "LOG"),
    ("log", "LOG"),
    ("the log page", "LOG"),                  # old code: no match
    ("command centre", "COMMAND CENTRE"),
    ("the command centre page", "COMMAND CENTRE"),
    ("open the command center", "COMMAND CENTRE"),   # American spelling
    ("go to diagnostics", "DIAGNOSTICS"),     # old code: no match
    ("show me the globe", "GLOBE"),
    ("memory", "MEMORY"),
    ("widget", "WIDGETS"),
])
def test_a_page_is_found_from_how_people_ask(spoken, expected):
    assert _resolve(spoken) == expected


@pytest.mark.parametrize("spoken,expected", [
    ("spellscape", "BRAIN"),     # what it used to be called
    ("swarm", "BRAIN"),          # what it replaced
    ("bring up the brain", "BRAIN"),
    ("world map", "GLOBE"),
    ("camera", "WORKBENCH"),
])
def test_the_names_people_actually_use_resolve(spoken, expected):
    """A page's label is rarely the word someone says for it."""
    assert _resolve(spoken) == expected


@pytest.mark.parametrize("spoken", ["", "   ", "the page", "nonsense zzzz"])
def test_nothing_is_better_than_the_wrong_page(spoken):
    """The old matcher returned DEVELOPMENT for the single letter "o" — the
    first page containing one. Silently opening the wrong page is worse than
    saying it could not be found."""
    assert _resolve(spoken) is None


def test_a_bare_letter_no_longer_opens_whatever_matched_first():
    assert _resolve("o") is None


def test_ambiguity_is_a_failure_not_a_guess():
    """Two equally good matches mean the request did not identify a page."""
    deck = types.SimpleNamespace(
        _pages=[("ALPHA ONE", object()), ("ALPHA TWO", object())],
        _NAV_FILLER=UnifiedDashboard._NAV_FILLER,
        _NAV_ALIASES=UnifiedDashboard._NAV_ALIASES,
    )
    assert UnifiedDashboard.resolve_page_name(deck, "alpha") is None


def test_show_page_named_reports_whether_it_worked():
    """The caller had no way to tell navigation from a silent miss."""
    import inspect

    source = inspect.getsource(UnifiedDashboard.show_page_named)
    assert "-> bool" in source
    assert "return False" in source and "return True" in source


def test_the_tool_does_not_report_done_for_a_page_that_does_not_exist():
    import inspect

    from orion_core.dispatch_desktop import DesktopDispatchMixin

    source = inspect.getsource(DesktopDispatchMixin.interface_control)
    assert "page_resolver" in source, (
        "interface_control emits onto the bus; without resolving the name "
        "first it cannot know whether the deck moved")
    assert source.index("page_resolver") < source.index("gui_command.emit"), (
        "the name must be checked before the command is sent")


def test_the_resolver_is_actually_wired_up():
    """A resolver nothing calls is the same as no resolver."""
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "dispatcher.page_resolver = deck.resolve_page_name" in source
    assert "dispatcher.page_names = deck.page_names" in source
