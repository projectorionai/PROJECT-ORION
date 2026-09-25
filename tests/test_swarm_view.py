"""
Tests for swarm_view.py (Mark XXI, Track B) — the 3D agent-swarm map.

compose_snapshot() (the real data source) gets the most thorough coverage
since it's pure Python, fully independent of Qt/WebEngine. SwarmDeckView's
construction and non-rendering logic (inspector, bus-state tracking, the
visibility guard) are tested WITHOUT ever calling .show()/triggering
_ensure_built() when WEBENGINE_OK is True — that would spin up a real
QWebEngineView/Chromium context, which this codebase has no precedent for
doing in a test (test_globe_zoom.py, the nearest analogue, tests the pure
zoom-controller logic, never GlobeView itself). The WEBENGINE_OK=False
fallback path (a plain QListWidget) IS fully constructible and is tested
directly.

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
from PyQt6.QtWidgets import QApplication  # noqa: E402

import orion_core.gui.swarm_view as sv  # noqa: E402
from orion_core.bus import OrionBus  # noqa: E402
from orion_core.gui.swarm_view import SwarmDeckView, compose_snapshot  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


# ── compose_snapshot(): the real data source ────────────────────────────────

def test_snapshot_always_includes_the_core_node():
    snap = compose_snapshot()
    assert snap["nodes"] == [{"id": "core", "kind": "core", "label": "ORION", "state": "STANDBY"}]
    assert snap["edges"] == []


def test_snapshot_reflects_the_supplied_core_state():
    snap = compose_snapshot(core_state="LISTENING")
    assert snap["nodes"][0]["state"] == "LISTENING"


class _StubAgents:
    def describe(self):
        return [
            {"name": "coding", "title": "Coding Agent", "focus": "code review", "calls": 5, "last_active": "t"},
            {"name": "research", "title": "Research Agent", "focus": "analysis", "calls": 0, "last_active": ""},
        ]


def test_snapshot_includes_agent_nodes_and_core_edges():
    snap = compose_snapshot(agents=_StubAgents())
    agent_nodes = [n for n in snap["nodes"] if n["kind"] == "agent"]
    assert {n["id"] for n in agent_nodes} == {"agent:coding", "agent:research"}
    coding = next(n for n in agent_nodes if n["id"] == "agent:coding")
    assert coding["label"] == "Coding Agent"
    assert coding["calls"] == 5
    assert {"from": "core", "to": "agent:coding"} in snap["edges"]


class _BrokenAgents:
    def describe(self):
        raise RuntimeError("boom")


def test_snapshot_survives_a_broken_agents_source():
    snap = compose_snapshot(agents=_BrokenAgents())
    assert snap["nodes"] == [{"id": "core", "kind": "core", "label": "ORION", "state": "STANDBY"}]


class _StubModuleRegistry:
    def all_described(self, telemetry):
        return {
            "dispatcher": {"role": "tool router", "health": "OK", "dependencies": ["memory"]},
            "memory": {"role": "matrix", "health": "DEGRADED", "dependencies": []},
        }


class _StubRegistries:
    def __init__(self):
        self.modules = _StubModuleRegistry()


def test_snapshot_includes_module_nodes_and_dependency_edges():
    snap = compose_snapshot(registries=_StubRegistries())
    module_nodes = {n["id"]: n for n in snap["nodes"] if n["kind"] == "module"}
    assert module_nodes["module:dispatcher"]["health"] == "OK"
    assert module_nodes["module:memory"]["health"] == "DEGRADED"
    assert {"from": "module:dispatcher", "to": "module:memory"} in snap["edges"]
    # module nodes are NOT wired to core directly (only via their own deps)
    assert {"from": "core", "to": "module:dispatcher"} not in snap["edges"]


def test_snapshot_drops_a_dependency_edge_to_an_unknown_module():
    class _Reg:
        def all_described(self, telemetry):
            return {"dispatcher": {"role": "x", "health": "OK", "dependencies": ["ghost_module"]}}

    class _R:
        modules = _Reg()

    snap = compose_snapshot(registries=_R())
    assert snap["edges"] == []   # "ghost_module" was never itself a node


class _StubMCPHost:
    def catalogue(self):
        return {"gmail": {"description": "x", "tools": [{"name": "send_email"}, {"name": "list"}]}}

    def health_snapshot(self):
        return {"gmail": "OK"}


def test_snapshot_includes_mcp_nodes():
    snap = compose_snapshot(mcp_host=_StubMCPHost())
    mcp_nodes = [n for n in snap["nodes"] if n["kind"] == "mcp"]
    assert len(mcp_nodes) == 1
    assert mcp_nodes[0]["id"] == "mcp:gmail"
    assert mcp_nodes[0]["health"] == "OK"
    assert "2 tool" in mcp_nodes[0]["detail"]
    assert {"from": "core", "to": "mcp:gmail"} in snap["edges"]


class _StubWorkflowEngine:
    definitions = {"morning": {"steps": [{"tool": "briefing"}, {"tool": "weather"}]}}


def test_snapshot_includes_workflow_nodes():
    snap = compose_snapshot(workflow_engine=_StubWorkflowEngine())
    wf = next(n for n in snap["nodes"] if n["kind"] == "workflow")
    assert wf["id"] == "workflow:morning"
    assert "2 step" in wf["detail"]


class _StubMemory:
    def tiers_snapshot(self):
        return {"short_term": 4, "long_term": 200}


def test_snapshot_includes_memory_tier_nodes():
    snap = compose_snapshot(memory=_StubMemory())
    ids = {n["id"] for n in snap["nodes"] if n["kind"] == "memory"}
    assert ids == {"memory:short_term", "memory:long_term"}


def test_snapshot_composes_every_source_together_without_interference():
    snap = compose_snapshot(
        core_state="THINKING", agents=_StubAgents(), registries=_StubRegistries(),
        mcp_host=_StubMCPHost(), workflow_engine=_StubWorkflowEngine(), memory=_StubMemory(),
    )
    kinds = {n["kind"] for n in snap["nodes"]}
    assert kinds == {"core", "agent", "module", "mcp", "workflow", "memory"}


# ── SwarmDeckView: construction + non-rendering logic ───────────────────────

class _Signal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *a, **k):
        for s in self._slots:
            s(*a, **k)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def test_construction_does_not_require_webengine_to_actually_build(_app):
    # Constructing must never call _ensure_built() itself — that only
    # happens on showEvent, precisely so first paint isn't blocked and so
    # this stays testable without a real Chromium context.
    view = SwarmDeckView(_StubBus())
    assert view._built is False
    assert view.view is None


def test_bus_state_updates_core_state(_app):
    bus = _StubBus()
    view = SwarmDeckView(bus)
    bus.state.emit("SPEAKING")
    assert view._core_state == "SPEAKING"


def test_compose_uses_the_constructors_stored_backends(_app):
    view = SwarmDeckView(_StubBus(), agents=_StubAgents())
    snap = view._compose()
    assert any(n["id"] == "agent:coding" for n in snap["nodes"])


def test_refresh_is_a_no_op_while_hidden(_app):
    view = SwarmDeckView(_StubBus(), agents=_StubAgents())
    calls = []
    view._push_snapshot = lambda: calls.append(1)
    view._refresh()
    assert calls == []   # never shown


def test_refresh_pushes_a_snapshot_once_visible(_app):
    view = SwarmDeckView(_StubBus())
    # showEvent -> _ensure_built() would construct a real QWebEngineView
    # and load SWARM_HTML (CDN Three.js) — exactly the heavy, possibly
    # network-dependent init this test suite has no precedent for
    # triggering (see the module docstring). Neutralise it so .show()
    # only exercises the visibility flag this test actually cares about.
    view._ensure_built = lambda: None
    view.show()
    calls = []
    view._push_snapshot = lambda: calls.append(1)
    view._refresh()
    assert calls == [1]
    view.hide()


# ── inspector logic (Track B4) ──────────────────────────────────────────────

def test_swarm_event_updates_the_inspector_for_an_agent_node(_app):
    opened = []
    view = SwarmDeckView(_StubBus(), on_open_agent=lambda name: opened.append(name))
    view._on_swarm_event({"id": "agent:coding", "kind": "agent", "label": "Coding Agent",
                          "calls": 3, "detail": "code review"})
    assert "Coding Agent" in view.inspector_label.text()
    assert "3" in view.inspector_detail.text()
    assert view.inspector_action.isEnabled() is True
    view.inspector_action.click()
    assert opened == ["coding"]


def test_swarm_event_disables_the_action_for_a_non_agent_node(_app):
    view = SwarmDeckView(_StubBus())
    view._on_swarm_event({"id": "module:dispatcher", "kind": "module", "label": "dispatcher",
                          "health": "OK", "detail": "tool router"})
    assert view.inspector_action.isEnabled() is False
    assert "OK" in view.inspector_detail.text()


def test_swarm_event_with_no_open_callback_disables_the_action_even_for_an_agent(_app):
    view = SwarmDeckView(_StubBus(), on_open_agent=None)
    view._on_swarm_event({"id": "agent:coding", "kind": "agent", "label": "Coding Agent"})
    assert view.inspector_action.isEnabled() is False


def test_swarm_event_ignores_a_malformed_payload(_app):
    view = SwarmDeckView(_StubBus())
    before = view.inspector_label.text()
    view._on_swarm_event("not a dict")   # must not raise
    assert view.inspector_label.text() == before


def test_inspector_action_with_nothing_selected_is_a_safe_no_op(_app):
    view = SwarmDeckView(_StubBus(), on_open_agent=lambda name: (_ for _ in ()).throw(AssertionError))
    view._on_inspector_action()   # must not raise / must not call the callback


def test_inspector_action_callback_exception_is_swallowed(_app):
    view = SwarmDeckView(_StubBus(), on_open_agent=lambda name: 1 / 0)
    view._on_swarm_event({"id": "agent:coding", "kind": "agent", "label": "Coding"})
    view._on_inspector_action()   # must not raise despite the callback failing


# ── pulse() (Track B5) never raises with no live view ───────────────────────

def test_pulse_is_a_safe_no_op_before_the_view_is_built(_app):
    view = SwarmDeckView(_StubBus())
    view.pulse("core", "agent:coding")   # must not raise


# ── WEBENGINE_OK=False fallback (fully constructible, no Chromium needed) ──

def test_fallback_list_view_is_used_when_neither_renderer_is_available(_app, monkeypatch):
    # Both the GL renderer and WebEngine have to be off for the plain list to
    # be the surface — the GL path is the default now, so disabling only
    # WebEngine no longer reaches this branch.
    monkeypatch.setenv("ORION_SWARM_GL", "0")
    monkeypatch.setattr(sv, "WEBENGINE_OK", False)
    view = SwarmDeckView(_StubBus(), agents=_StubAgents())
    assert hasattr(view, "list_view")
    view._push_snapshot()
    assert view.list_view.count() >= 1
    labels = [view.list_view.item(i).text() for i in range(view.list_view.count())]
    assert any("Coding Agent" in label for label in labels)


def test_fallback_list_selection_drives_the_inspector(_app, monkeypatch):
    monkeypatch.setenv("ORION_SWARM_GL", "0")
    monkeypatch.setattr(sv, "WEBENGINE_OK", False)
    view = SwarmDeckView(_StubBus(), agents=_StubAgents(), on_open_agent=lambda n: None)
    view._push_snapshot()
    view.list_view.setCurrentRow(1)   # first agent row (index 0 is core)
    assert "agent" in view.inspector_label.text().lower() or view.inspector_action.text()


# ── console-message routing (_route_console_message) ────────────────────────
#
# The real _SwarmPage.javaScriptConsoleMessage is a one-line call into this
# plain function — tested directly here so nothing in this file ever
# imports QtWebEngineCore/Widgets (see the module docstring: that import
# must not happen during pytest collection, or it destabilises OpenGL for
# unrelated real-widget tests later in the full suite).

def test_swarm_page_routes_a_prefixed_console_message():
    received = []
    message = sv._EVENT_PREFIX + '{"id": "agent:coding", "kind": "agent"}'
    sv._route_console_message(lambda payload: received.append(payload), message)
    assert received == [{"id": "agent:coding", "kind": "agent"}]


def test_swarm_page_ignores_an_unprefixed_message():
    received = []
    sv._route_console_message(lambda payload: received.append(payload), "some unrelated log line")
    assert received == []


def test_swarm_page_ignores_malformed_json_after_the_prefix():
    received = []
    sv._route_console_message(lambda payload: received.append(payload), sv._EVENT_PREFIX + "{not json")
    assert received == []


def test_swarm_page_never_raises_when_the_callback_itself_raises():
    def _boom(payload):
        raise RuntimeError("boom")
    message = sv._EVENT_PREFIX + '{"id": "x"}'
    sv._route_console_message(_boom, message)   # must not raise


def test_swarm_page_ignores_a_message_with_no_callback_attached():
    message = sv._EVENT_PREFIX + '{"id": "x"}'
    sv._route_console_message(None, message)   # must not raise


def test_swarm_page_class_is_not_built_until_first_requested():
    # WEBENGINE_OK's own import-deferral (module docstring) only holds if
    # nothing forces the lazy factory this early — confirms the class cache
    # starts empty rather than a real QWebEnginePage subclass sneaking in
    # via some other import path.
    assert sv._SwarmPageCls is None


# ── native GPU renderer, Mark XXII Phase 1 ───────────────────────────────────
#
# None of these construct a SwarmGLView. Creating a real GL context in a test
# carries the same hazard as the QWebEnginePage that crashed this suite with a
# native STATUS_STACK_BUFFER_OVERRUN, so the widget is only ever built from a
# real showEvent — which no test triggers.

def test_the_gl_renderer_is_the_default(monkeypatch):
    """Verified against a live OpenGL 4.1 core context on real hardware, so
    it is opt-OUT now rather than opt-in."""
    monkeypatch.delenv("ORION_SWARM_GL", raising=False)
    assert sv.gl_renderer_enabled() is True


def test_the_gl_flag_accepts_the_usual_truthy_spellings(monkeypatch):
    for value in ("1", "true", "on", "YES"):
        monkeypatch.setenv("ORION_SWARM_GL", value)
        assert sv.gl_renderer_enabled() is True, value


def test_an_explicit_false_forces_the_old_webengine_renderer(monkeypatch):
    for value in ("0", "false", "off", "NO"):
        monkeypatch.setenv("ORION_SWARM_GL", value)
        assert sv.gl_renderer_enabled() is False, value


def test_an_unrecognised_value_does_not_silently_disable_the_renderer(monkeypatch):
    """Only an explicit falsey word turns it off; a typo must not quietly
    drop the user back onto the old renderer."""
    monkeypatch.setenv("ORION_SWARM_GL", "nonsense")
    assert sv.gl_renderer_enabled() is True


def test_construction_in_gl_mode_still_builds_nothing_heavy(_app, monkeypatch):
    monkeypatch.setenv("ORION_SWARM_GL", "1")
    view = SwarmDeckView(_StubBus())
    assert view._gl_mode is True
    assert view._built is False
    assert view.gl_view is None          # deferred to showEvent


def test_gl_mode_does_not_create_the_list_fallback(_app, monkeypatch):
    monkeypatch.setenv("ORION_SWARM_GL", "1")
    view = SwarmDeckView(_StubBus())
    assert not hasattr(view, "list_view")


def test_compose_typed_returns_the_mark_xxii_snapshot(_app):
    view = SwarmDeckView(_StubBus(), agents=_StubAgents())
    snapshot = view.compose_typed()
    ids = snapshot.node_ids()
    assert "core" in ids
    assert "agent:coding" in ids
    # clusters are synthesised, so agents no longer hang off core directly
    assert "cluster:DEVELOPMENT" in ids


def test_compose_typed_reflects_the_live_bus_state(_app):
    bus = _StubBus()
    view = SwarmDeckView(bus)
    bus.state.emit("THINKING")
    assert view.compose_typed().core_state == "THINKING"


class _StubGLView:
    """Stands in for SwarmGLView — same surface, no GL context."""

    def __init__(self):
        self.applied = []
        self.scene = self

    def apply_snapshot(self, snapshot):
        self.applied.append(snapshot)

    def node(self, node_id):
        return None


def test_gl_mode_pushes_typed_snapshots_straight_to_the_renderer(_app, monkeypatch):
    """No JSON, no bridge — the whole point of replacing the WebEngine path."""
    monkeypatch.setenv("ORION_SWARM_GL", "1")
    view = SwarmDeckView(_StubBus(), agents=_StubAgents())
    stub = _StubGLView()
    view.gl_view = stub
    view._push_snapshot()
    assert len(stub.applied) == 1
    assert "agent:coding" in stub.applied[0].node_ids()


def test_gl_mode_never_serialises_to_json(_app, monkeypatch):
    monkeypatch.setenv("ORION_SWARM_GL", "1")
    view = SwarmDeckView(_StubBus())
    view.gl_view = _StubGLView()
    called = []
    view._js = lambda code: called.append(code)
    view._push_snapshot()
    assert called == []


def test_a_gl_pick_drives_the_same_inspector_as_every_other_surface(_app):
    from orion_core.swarm.model import NodeKind, SwarmNode

    view = SwarmDeckView(_StubBus(), on_open_agent=lambda name: None)

    class _Scene:
        def node(self, node_id):
            return SwarmNode(id="agent:coding", kind=NodeKind.AGENT,
                             label="Coding Agent", cluster="DEVELOPMENT")

    class _GL:
        scene = _Scene()

    view.gl_view = _GL()
    view._on_gl_node_selected("agent:coding")
    assert "Coding Agent" in view.inspector_label.text()
    assert view.inspector_action.isEnabled() is True


def test_a_gl_pick_for_an_unknown_node_is_a_safe_no_op(_app):
    view = SwarmDeckView(_StubBus())
    before = view.inspector_label.text()
    view.gl_view = _StubGLView()          # its scene.node() returns None
    view._on_gl_node_selected("ghost")
    assert view.inspector_label.text() == before


def test_the_gl_module_is_not_imported_merely_by_importing_the_page(_app):
    """gl_view.py imports Qt's OpenGL modules at module level; pulling it in
    at collection time is exactly the mistake that crashed this suite before.

    Asserted against the SOURCE rather than sys.modules: another test file
    may legitimately import gl_view, and a global-state check would then pass
    or fail purely on test ordering rather than on the property it claims to
    protect."""
    from pathlib import Path

    source = Path(sv.__file__).read_text(encoding="utf-8")
    module_level = source.split("def _build_gl_view")[0]
    assert "from ..swarm.render.gl_view import" not in module_level
