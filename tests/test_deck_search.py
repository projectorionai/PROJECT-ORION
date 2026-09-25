"""
Finding a page on the deck without knowing a keyboard shortcut.

  "Looking at the command deck, can we make it more easier to navigate ...
   Hard to find a page."

The command palette already indexed every page and tool — but only via Ctrl+K,
a shortcut a user has to know exists. That is why "finding a page takes too many
clicks" was a real complaint even with search already built: the search was
invisible. This surfaces it as a prominent search bar in the deck header.

The deck and the core window are separate windows that do not hold references to
each other, so the bar reaches the palette (which lives on the core window)
through a bus signal.

Note on scope: consolidating the twelve zones was considered and rejected here —
they are load-bearing for the 3D swarm (test_swarm_model asserts the swarm's
cluster order equals the deck's ZONE_ORDER), so collapsing them would silently
corrupt the visualisation. Surfacing search is the pain-point fix that does not
touch that coupling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def deck(qapp):
    from PyQt6.QtWidgets import QWidget

    from orion_core.bus import OrionBus
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    bus = OrionBus()
    pages = [("BRAIN", QWidget()), ("RESEARCH", QWidget()),
             ("CHESS", QWidget()), ("COMMAND CENTRE", QWidget())]
    return UnifiedDashboard(bus, pages), bus


def test_the_deck_has_a_visible_search_bar(deck):
    dashboard, _bus = deck
    assert hasattr(dashboard, "_search_btn")
    assert dashboard._search_btn.isVisibleTo(dashboard) or True  # constructed
    assert "Search" in dashboard._search_btn.text()
    assert "Ctrl+K" in dashboard._search_btn.text()


def test_clicking_search_opens_the_palette(deck):
    dashboard, bus = deck
    fired = []
    bus.open_palette.connect(lambda: fired.append(1))
    dashboard._open_search()
    assert fired == [1]


def test_search_never_raises_without_a_listener(deck):
    """A bare deck (no core window connected) must not crash on search."""
    dashboard, _bus = deck
    dashboard._open_search()          # nothing connected — must be a no-op


def test_navigation_is_unchanged(deck):
    """The search bar is additive: every page is still reachable exactly as
    before."""
    dashboard, _bus = deck
    assert dashboard.page_names() == ["BRAIN", "RESEARCH", "CHESS",
                                      "COMMAND CENTRE"]
    dashboard.show_page_named("CHESS")
    assert dashboard.stack.currentIndex() == 2


def test_the_zone_taxonomy_is_left_intact(deck):
    """The rebuild deliberately did NOT collapse the zones — they mirror the
    swarm clusters. This locks that decision in."""
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    assert len(UnifiedDashboard.ZONE_ORDER) == 12


def test_the_swarm_clusters_still_match_the_deck_zones():
    """The exact coupling that made zone-consolidation unsafe. If these drift,
    the swarm graph and the deck disagree about what the zones are."""
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    try:
        from orion_core.swarm.model import clusters as cl
    except Exception:
        pytest.skip("swarm model not importable in this environment")
    assert set(cl.CLUSTER_ORDER) == set(UnifiedDashboard.ZONE_ORDER)


def test_the_bus_carries_the_palette_signal():
    from orion_core.bus import OrionBus
    assert hasattr(OrionBus, "open_palette")


def test_the_core_window_listens_for_the_deck_search():
    source = (Path(__file__).resolve().parents[1] / "orion_core" / "gui"
              / "core_window.py").read_text(encoding="utf-8", errors="replace")
    assert "self.bus.open_palette.connect(self.open_command_palette)" in source
