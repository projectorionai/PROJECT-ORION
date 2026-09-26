"""
Tests for the swarm as ORION's navigation surface (Mark XXII).

"Migrate each part of the Command Deck into the swarm" — every zone is
reachable by opening the cluster node that represents it, so navigating the
graph and navigating ORION are the same gesture.

The load-bearing rule: zone-to-page resolution is asked of the deck's own
ZONE_PAGES, never duplicated in the window, so the swarm can never disagree
with the deck about which pages a zone owns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ORION_REMOTE_ACCESS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from orion_core.bus import OrionBus  # noqa: E402
from orion_core.gui import core_window as core_window_mod  # noqa: E402
from orion_core.gui.core_window import OrionCoreWindow  # noqa: E402
from orion_core.gui.swarm_view import SwarmDeckView  # noqa: E402
from orion_core.gui.unified_dashboard import UnifiedDashboard  # noqa: E402
from orion_core.memory import MemoryAgent, OrionMemoryMatrix  # noqa: E402
from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.inspect import build_report  # noqa: E402
from orion_core.swarm.model import NodeKind, SwarmNode  # noqa: E402
from orion_core.swarm.sources import cluster_id, compose  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_webengine_face(monkeypatch):
    # See test_unified_shell.py: the WebEngine face cannot survive a native
    # window recreation under offscreen Qt.
    monkeypatch.setattr(OrionCoreWindow, "_build_face",
                        lambda self: core_window_mod.HologramFace())
    yield


class _Signal:
    def __init__(self): self._slots = []
    def connect(self, slot): self._slots.append(slot)
    def emit(self, *a, **k):
        for s in self._slots:
            s(*a, **k)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def _window(tmp_path) -> OrionCoreWindow:
    bus = OrionBus()
    matrix = OrionMemoryMatrix(tmp_path / "core.db", tmp_path, bus)
    return OrionCoreWindow(bus, MemoryAgent(matrix, bus))


def _deck(bus) -> UnifiedDashboard:
    # BRAIN is INTELLIGENCE's first page on the real deck, so the stand-in
    # must carry it too — otherwise this fixture, not the code, decides which
    # page a zone opens on, and the test stops testing zone resolution.
    return UnifiedDashboard(bus, [("BRAIN", QLabel("brain")),
                                  ("MISSION", QLabel("mission")),
                                  ("GLOBE", QLabel("globe")),
                                  ("SWARM", QLabel("swarm")),
                                  ("OPS", QLabel("ops")),
                                  ("MEMORY", QLabel("memory"))])


# ── the model knows what a cluster is ────────────────────────────────────────

def test_cluster_anchors_are_identifiable():
    node = SwarmNode(id=cluster_id(cl.MEMORY), kind=NodeKind.SUBSYSTEM,
                     label="Memory", cluster=cl.MEMORY)
    assert node.is_cluster is True
    assert node.cluster_zone == "MEMORY"


def test_a_real_subsystem_is_not_a_cluster():
    node = SwarmNode(id="module:memory", kind=NodeKind.SUBSYSTEM,
                     label="memory", cluster=cl.MEMORY)
    assert node.is_cluster is False
    assert node.cluster_zone == ""


def test_every_composed_cluster_anchor_reports_its_zone():
    for node in compose().nodes:
        if node.is_cluster:
            assert node.cluster_zone in cl.CLUSTER_ORDER


# ── the inspector offers the zone as an action ───────────────────────────────

def test_a_cluster_offers_opening_its_zone():
    node = SwarmNode(id=cluster_id(cl.SECURITY), kind=NodeKind.SUBSYSTEM,
                     label="Security", cluster=cl.SECURITY)
    report = build_report(node)
    assert report.action_target == "SECURITY"
    assert "Security" in report.action_label


def test_every_zone_is_openable_from_its_cluster_node():
    """'Everything in the swarm' — no zone may be unreachable from the graph."""
    for node in compose().nodes:
        if node.is_cluster:
            assert build_report(node).action_target == node.cluster_zone


def test_a_cluster_action_does_not_require_an_agent_handler():
    node = SwarmNode(id=cluster_id(cl.SYSTEM), kind=NodeKind.SUBSYSTEM,
                     label="System", cluster=cl.SYSTEM)
    assert build_report(node, can_open_agent=False).action_target == "SYSTEM"


# ── the window resolves a zone to a real page ────────────────────────────────

def test_opening_a_zone_navigates_the_deck_to_its_first_page(_app, tmp_path):
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_dashboard(deck)
    win._open_deck_zone("OPERATIONS")
    assert "OPS" in deck.page_names()[deck.stack.currentIndex()]


def test_zone_resolution_uses_the_decks_own_mapping(_app, tmp_path):
    """Never a second copy of the zone->page table."""
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_dashboard(deck)
    win._open_deck_zone("INTELLIGENCE")
    expected = UnifiedDashboard.ZONE_PAGES["INTELLIGENCE"][0]
    assert expected in deck.page_names()[deck.stack.currentIndex()]


def test_zone_names_are_case_insensitive(_app, tmp_path):
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_dashboard(deck)
    win._open_deck_zone("memory")
    assert "MEMORY" in deck.page_names()[deck.stack.currentIndex()]


def test_an_unknown_zone_logs_instead_of_navigating(_app, tmp_path):
    win = _window(tmp_path)
    win.attach_dashboard(_deck(win.bus))
    received: list[str] = []
    win.bus.log.connect(received.append)
    win._open_deck_zone("NOT_A_ZONE")
    assert any("no pages yet" in m for m in received)


def test_opening_a_zone_with_no_deck_logs_cleanly(_app, tmp_path):
    win = _window(tmp_path)
    received: list[str] = []
    win.bus.log.connect(received.append)
    win._open_deck_zone("MEMORY")
    assert any("not attached" in m for m in received)


def test_every_zone_with_pages_resolves_to_a_real_page(_app, tmp_path):
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_dashboard(deck)
    for zone, pages in UnifiedDashboard.ZONE_PAGES.items():
        if not pages:
            continue
        received: list[str] = []
        win.bus.log.connect(received.append)
        win._open_deck_zone(zone)
        assert not any("no pages yet" in m for m in received), zone


# ── end to end: clicking a cluster in the swarm navigates ORION ──────────────

def test_the_inspector_action_opens_the_zone(_app):
    opened: list[str] = []
    view = SwarmDeckView(_StubBus(), on_open_zone=opened.append)
    view._on_swarm_event({"id": cluster_id(cl.CREATIVE), "kind": "subsystem",
                          "label": "Creative", "cluster": cl.CREATIVE})
    assert view.inspector_action.isEnabled() is True
    view.inspector_action.click()
    assert opened == ["CREATIVE"]


def test_the_action_button_names_the_zone(_app):
    view = SwarmDeckView(_StubBus(), on_open_zone=lambda z: None)
    view._on_swarm_event({"id": cluster_id(cl.AUTOMATION), "kind": "subsystem",
                          "label": "Automation", "cluster": cl.AUTOMATION})
    assert "Automation" in view.inspector_action.text()


def test_a_cluster_without_a_zone_handler_does_not_crash(_app):
    view = SwarmDeckView(_StubBus(), on_open_zone=None)
    view._on_swarm_event({"id": cluster_id(cl.MEMORY), "kind": "subsystem",
                          "label": "Memory", "cluster": cl.MEMORY})
    view._on_inspector_action()          # must not raise


def test_a_failing_zone_handler_is_swallowed(_app):
    view = SwarmDeckView(_StubBus(),
                         on_open_zone=lambda z: (_ for _ in ()).throw(RuntimeError))
    view._on_swarm_event({"id": cluster_id(cl.MEMORY), "kind": "subsystem",
                          "label": "Memory", "cluster": cl.MEMORY})
    view._on_inspector_action()          # must not raise


def test_agent_nodes_still_open_their_workspace(_app):
    """Adding zone navigation must not break the existing agent action."""
    opened: list[str] = []
    view = SwarmDeckView(_StubBus(), on_open_agent=opened.append,
                         on_open_zone=lambda z: None)
    view._on_swarm_event({"id": "agent:coding", "kind": "agent",
                          "label": "Coding Agent"})
    view.inspector_action.click()
    assert opened == ["coding"]


# ── the Command Deck lives INSIDE the swarm ──────────────────────────────────

def test_every_deck_page_becomes_a_node():
    """'The deck must be migrated into the subsystem' — the graph holds all
    of ORION, not just his agents."""
    pages = {"MISSION": "INTELLIGENCE", "GLOBE": "INTELLIGENCE",
             "OPS": "OPERATIONS", "MEMORY": "MEMORY"}
    ids = compose(deck_pages=pages).node_ids()
    for page in pages:
        assert f"page:{page}" in ids


def test_a_page_node_lands_in_its_own_zone_cluster():
    snap = compose(deck_pages={"OPS": "OPERATIONS"})
    node = snap.by_id()["page:OPS"]
    assert node.kind is NodeKind.PAGE
    assert node.cluster == cl.OPERATIONS
    assert node.parent == cluster_id(cl.OPERATIONS)


def test_a_page_node_is_wired_to_its_cluster():
    snap = compose(deck_pages={"OPS": "OPERATIONS"})
    assert any(e.source == cluster_id(cl.OPERATIONS) and e.target == "page:OPS"
               for e in snap.edges)


def test_a_page_outside_every_zone_still_gets_a_node():
    """CHESS is deliberately in no zone; it must not be the one part of ORION
    the swarm cannot reach."""
    snap = compose(deck_pages={"CHESS": "SYSTEM"})
    assert "page:CHESS" in snap.node_ids()
    assert snap.by_id()["page:CHESS"].cluster == cl.SYSTEM


def test_an_unknown_zone_falls_back_rather_than_dropping_the_page():
    snap = compose(deck_pages={"WEIRD": "NOT_A_ZONE"})
    assert cl.is_known_cluster(snap.by_id()["page:WEIRD"].cluster)


def test_composing_without_deck_pages_is_unchanged():
    assert not any(n.kind is NodeKind.PAGE for n in compose().nodes)


def test_a_page_node_offers_opening_itself():
    snap = compose(deck_pages={"GLOBE": "INTELLIGENCE"})
    report = build_report(snap.by_id()["page:GLOBE"])
    assert report.action_target == "GLOBE"
    assert "GLOBE" in report.action_label


def test_the_inspector_action_opens_the_page(_app):
    opened: list[str] = []
    view = SwarmDeckView(_StubBus(), on_open_page=opened.append)
    view._on_swarm_event({"id": "page:GLOBE", "kind": "page", "label": "GLOBE",
                          "cluster": cl.INTELLIGENCE})
    assert view.inspector_action.isEnabled() is True
    view.inspector_action.click()
    assert opened == ["GLOBE"]


def test_attach_deck_pages_adds_them_to_the_graph(_app):
    """The deck is built after the swarm page it contains, so the page list
    has to be attachable late."""
    view = SwarmDeckView(_StubBus())
    assert not any(n.kind is NodeKind.PAGE for n in view.compose_typed().nodes)
    view.attach_deck_pages({"OPS": "OPERATIONS", "GLOBE": "INTELLIGENCE"})
    ids = view.compose_typed().node_ids()
    assert "page:OPS" in ids and "page:GLOBE" in ids


def test_attaching_pages_invalidates_the_cached_structure(_app):
    """Page nodes are part of the graph's shape; a stale cache would hide
    them until something else happened to change."""
    view = SwarmDeckView(_StubBus())
    view.compose_typed()
    view.attach_deck_pages({"OPS": "OPERATIONS"})
    assert "page:OPS" in view.compose_typed().node_ids()


def test_the_swarm_reaches_every_page_the_real_deck_offers(_app, tmp_path):
    """End to end against the REAL zone table: no page is unreachable."""
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_dashboard(deck)
    zone_of = {p: z for z, ps in UnifiedDashboard.ZONE_PAGES.items() for p in ps}
    mapping = {name: zone_of.get(name, "SYSTEM") for name in deck.page_names()}

    view = SwarmDeckView(_StubBus(), on_open_page=win._open_deck_page)
    view.attach_deck_pages(mapping)
    ids = view.compose_typed().node_ids()
    for name in deck.page_names():
        assert f"page:{name}" in ids


def test_opening_a_page_node_navigates_the_real_deck(_app, tmp_path):
    win = _window(tmp_path)
    deck = _deck(win.bus)
    win.attach_dashboard(deck)
    view = SwarmDeckView(_StubBus(), on_open_page=win._open_deck_page)
    view._on_swarm_event({"id": "page:OPS", "kind": "page", "label": "OPS",
                          "cluster": cl.OPERATIONS})
    view.inspector_action.click()
    assert "OPS" in deck.page_names()[deck.stack.currentIndex()]


def test_double_clicking_a_cluster_navigates(_app):
    """The GL surface routes double-click through the same action path."""
    opened: list[str] = []
    view = SwarmDeckView(_StubBus(), on_open_zone=opened.append)

    class _Scene:
        def node(self, node_id):
            return SwarmNode(id=node_id, kind=NodeKind.SUBSYSTEM,
                             label="Research", cluster=cl.RESEARCH)

    class _GL:
        scene = _Scene()

    view.gl_view = _GL()
    view._on_gl_node_activated(cluster_id(cl.RESEARCH))
    assert opened == ["RESEARCH"]
