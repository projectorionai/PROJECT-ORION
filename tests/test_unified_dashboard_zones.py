"""
Tests for the zone-grouped Command Deck navigation (Mark XX design-spec §4,
High Impact item 8, widened to twelve zones in the architectural-audit
pass): the old header was one flat row of thirteen co-equal page buttons in
alphabetically-arbitrary order (a Hick's Law cost — decision time scales
with the log of undifferentiated choices). It's replaced with two rows:
twelve intent-based zones on top, the active zone's pages below. SECURITY
is a genuinely empty, disabled zone — security_recon.py exists as a real
backend with no GUI page yet, so the honest nav state is "named but
disabled", same as DEVELOPMENT/AUTOMATION were before they got real pages.
CHESS is deliberately excluded from every zone (§3's verdict: it shouldn't
occupy permanent top-level real estate) — it keeps its page, reachable via
the Command Palette and the existing chevron/swipe/keyboard navigation,
which still cycle the full flat page list exactly as before.

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
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from orion_core.bus import OrionBus
from orion_core.gui.unified_dashboard import UnifiedDashboard

# The exact page labels app.py registers, in order — kept in sync manually;
# test_zone_mapping_covers_every_real_app_page below fails loudly if this
# list and ZONE_PAGES drift apart.
_REAL_PAGE_LABELS = [
    "MISSION", "RESEARCH", "WORKBENCH", "DEVELOPMENT", "WIDGETS", "TOOLKIT",
    "MARKETING", "LIBRARY", "OPS", "COGNITION", "AUTOMATION", "COMMAND CENTRE",
    "DIAGNOSTICS", "GLOBE", "LOG", "MEMORY", "TELEMETRY", "CHESS", "SECURITY",
]


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _deck(labels: list[str]) -> UnifiedDashboard:
    pages = [(label, QWidget()) for label in labels]
    return UnifiedDashboard(OrionBus(), pages)


def test_zone_buttons_exist_for_all_twelve_zones(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    assert set(deck._zone_buttons) == set(UnifiedDashboard.ZONE_ORDER)
    assert len(deck._zone_buttons) == 12


def test_a_zone_with_no_pages_is_disabled(_app, monkeypatch):
    # DEVELOPMENT and AUTOMATION now both have real pages (AutomationDeckView/
    # DevelopmentDeckView, app.py) so no zone in the real app is empty any
    # more. The enabled/disabled check is keyed off the STATIC ZONE_PAGES
    # mapping, not whether this particular deck instance's page list
    # happens to include a zone's pages — so covering the still-live
    # disabled-zone code path means patching that mapping directly, not
    # just constructing a deck with a shorter page list.
    synthetic = dict(UnifiedDashboard.ZONE_PAGES)
    synthetic["BUSINESS"] = ()
    monkeypatch.setattr(UnifiedDashboard, "ZONE_PAGES", synthetic)
    deck = _deck(_REAL_PAGE_LABELS)
    assert deck._zone_buttons["BUSINESS"].isEnabled() is False


def test_security_zone_is_enabled_and_has_its_page(_app):
    """SECURITY was previously a deliberately EMPTY, disabled zone — the
    backend was real but had no GUI page. That empty tuple is precisely what
    made the security tab 'unclickable with nothing in it'. It now owns
    SecurityCentreView (posture, monitoring toggle, alert history)."""
    deck = _deck(_REAL_PAGE_LABELS)
    assert UnifiedDashboard.ZONE_PAGES["SECURITY"] == ("SECURITY",)
    assert deck._zone_buttons["SECURITY"].isEnabled() is True


def test_every_zone_is_enabled_in_the_real_app(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    for zone, btn in deck._zone_buttons.items():
        assert btn.isEnabled() is True, f"{zone} should be enabled"


def test_development_and_automation_zones_are_enabled(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    assert deck._zone_buttons["DEVELOPMENT"].isEnabled() is True
    assert deck._zone_buttons["AUTOMATION"].isEnabled() is True


def test_zones_with_pages_are_enabled(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    assert deck._zone_buttons["INTELLIGENCE"].isEnabled() is True
    assert deck._zone_buttons["MONITORING"].isEnabled() is True


def test_initial_page_checks_its_zone(_app):
    deck = _deck(_REAL_PAGE_LABELS)   # page 0 = MISSION
    assert deck._zone_buttons["INTELLIGENCE"].isChecked() is True
    assert deck._zone_buttons["MONITORING"].isChecked() is False


def test_page_row_shows_only_the_active_zones_pages(_app):
    deck = _deck(_REAL_PAGE_LABELS)   # starts on MISSION -> INTELLIGENCE
    shown = {deck._pages[i][0] for i in deck._tabs}
    assert shown == {"MISSION", "GLOBE"}


def test_research_zone_holds_research_and_library(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    research_index = deck.page_names().index("RESEARCH")
    deck._select(research_index)
    shown = {deck._pages[i][0] for i in deck._tabs}
    assert shown == {"RESEARCH", "LIBRARY"}


def test_memory_is_its_own_zone(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    memory_index = deck.page_names().index("MEMORY")
    deck._select(memory_index)
    assert deck._zone_buttons["MEMORY"].isChecked() is True
    shown = {deck._pages[i][0] for i in deck._tabs}
    assert shown == {"MEMORY"}


def test_system_zone_holds_command_centre(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    cc_index = deck.page_names().index("COMMAND CENTRE")
    deck._select(cc_index)
    assert deck._zone_buttons["SYSTEM"].isChecked() is True
    assert deck._zone_buttons["OPERATIONS"].isChecked() is False


def test_selecting_a_page_in_a_different_zone_rebuilds_the_row(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    log_index = deck.page_names().index("LOG")
    deck._select(log_index)
    assert deck._zone_buttons["MONITORING"].isChecked() is True
    assert deck._zone_buttons["INTELLIGENCE"].isChecked() is False
    shown = {deck._pages[i][0] for i in deck._tabs}
    assert shown == {"DIAGNOSTICS", "LOG", "TELEMETRY"}


def test_clicking_a_zone_jumps_to_its_first_page(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    deck._select_zone("RESEARCH")
    assert deck._pages[deck.stack.currentIndex()][0] == "RESEARCH"


def test_clicking_an_empty_zone_does_nothing(_app):
    deck = _deck(["MISSION"])   # BUSINESS has no page in this synthetic set
    before = deck.stack.currentIndex()
    deck._select_zone("BUSINESS")   # no pages -> no-op
    assert deck.stack.currentIndex() == before


def test_a_page_in_no_zone_still_gets_its_own_single_entry_row(_app):
    """The orphan mechanism itself, exercised with a genuinely zone-less page.

    CHESS used to be the example. It is now in SYSTEM, because being in no
    zone stopped being a soft preference the moment swarm navigation retired
    the tab bars: zone routing became the only way in, and "I cannot go on the
    chess page at all" was the result."""
    deck = _deck(["MISSION", "ORPHAN"])
    deck._select(deck.page_names().index("ORPHAN"))
    shown = {deck._pages[i][0] for i in deck._tabs}
    assert shown == {"ORPHAN"}
    assert all(not btn.isChecked() for btn in deck._zone_buttons.values())


def test_chess_is_reachable_through_a_zone(_app):
    """Swarm navigation removed the tab bars, so a page in no zone has no
    route at all."""
    deck = _deck(_REAL_PAGE_LABELS)
    assert deck._zone_for_page("CHESS") == "SYSTEM"


def test_adding_chess_did_not_displace_what_the_system_zone_opens_on(_app):
    assert UnifiedDashboard.ZONE_PAGES["SYSTEM"][0] == "COMMAND CENTRE"


def test_next_page_still_cycles_the_full_flat_list_including_orphans(_app):
    deck = _deck(["A", "B", "CHESS"])
    deck._select(0)
    deck.next_page()
    deck.next_page()
    assert deck._pages[deck.stack.currentIndex()][0] == "CHESS"


def test_show_page_named_still_reaches_a_page_with_no_zone(_app):
    deck = _deck(_REAL_PAGE_LABELS)
    deck.show_page_named("chess")
    assert deck._pages[deck.stack.currentIndex()][0] == "CHESS"


def test_zone_mapping_covers_every_real_app_page_exactly_once_or_not_at_all(_app):
    # Regression guard: every real page label must resolve to at most one
    # zone, CHESS must resolve to none, and every other page must resolve
    # to exactly one — if app.py's page list ever changes without updating
    # ZONE_PAGES, this fails loudly instead of silently dropping a page
    # from the nav.
    deck = _deck(_REAL_PAGE_LABELS)
    zoned = {name: deck._zone_for_page(name) for name in _REAL_PAGE_LABELS}
    for name, zone in zoned.items():
        assert zone is not None, f"{name} has no zone mapping"
        assert zone in UnifiedDashboard.ZONE_ORDER
