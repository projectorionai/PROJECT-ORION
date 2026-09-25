"""
The Mark XXIII systems reaching the running app.

Every one of these features is only worth anything if a real dispatch drives
it. These tests exercise the actual seams — dispatcher -> swarm view ->
ledger — rather than the pure models, which their own suites already cover.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.bridges import Phase  # noqa: E402


class _Memory:
    def tiers_snapshot(self):
        return {"working": 12, "long_term": 400, "archived": 9000}


class _SwarmStub:
    """Stands in for SwarmDeckView at the dispatcher seam."""

    def __init__(self) -> None:
        from orion_core.swarm.bridges import BridgeLedger
        self.bridges = BridgeLedger()
        self.launched: list[str] = []
        self.resolved: list[tuple] = []

    def launch_bridge(self, tool, args=None):
        self.launched.append(tool)
        return self.bridges.launch(["core", f"module:{tool}"], label=tool)

    def pulse(self, _from, _to, tool="", args=None, flight=None, ok=True):
        self.resolved.append((flight, ok))
        if flight is not None:
            self.bridges.resolve(flight, ok=ok)


# ── the dispatcher opens and closes a bridge around every tool call ──────────

def _dispatcher(swarm):
    from orion_core.dispatcher import OrionDispatcher

    dispatcher = OrionDispatcher.__new__(OrionDispatcher)
    dispatcher.active_tools = 0
    dispatcher.recent_tools = []
    dispatcher.telemetry = None
    dispatcher.swarm_view = swarm
    return dispatcher


def test_a_bridge_opens_before_the_tool_runs_not_after():
    """The whole point: the packet is in flight WHILE the work happens."""
    swarm = _SwarmStub()
    dispatcher = _dispatcher(swarm)
    seen: list[int] = []

    async def slow(_args):
        from orion_core.data import ToolResult
        seen.append(swarm.bridges.live())
        return ToolResult("done")

    dispatcher.handler_table = lambda: {"probe": slow}
    asyncio.run(dispatcher.dispatch("probe", {}))
    assert seen == [1]                     # already in flight inside the tool


def test_the_bridge_resolves_when_the_tool_finishes():
    swarm = _SwarmStub()
    dispatcher = _dispatcher(swarm)

    async def quick(_args):
        from orion_core.data import ToolResult
        return ToolResult("done")

    dispatcher.handler_table = lambda: {"probe": quick}
    asyncio.run(dispatcher.dispatch("probe", {}))
    flight, ok = swarm.resolved[0]
    assert flight is not None and ok is True


def test_a_failing_tool_resolves_its_bridge_as_a_failure():
    swarm = _SwarmStub()
    dispatcher = _dispatcher(swarm)

    async def broken(_args):
        raise RuntimeError("boom")

    dispatcher.handler_table = lambda: {"probe": broken}
    asyncio.run(dispatcher.dispatch("probe", {}))
    _flight, ok = swarm.resolved[0]
    assert ok is False


def test_a_broken_visualisation_can_never_fail_a_tool_call():
    """A bridge that raised must not cost ORION the dispatch."""
    from orion_core.data import ToolResult

    class _Broken(_SwarmStub):
        def launch_bridge(self, tool, args=None):
            raise RuntimeError("renderer gone")

    dispatcher = _dispatcher(_Broken())

    async def quick(_args):
        return ToolResult("done")

    dispatcher.handler_table = lambda: {"probe": quick}
    assert asyncio.run(dispatcher.dispatch("probe", {})).ok is True


def test_no_swarm_attached_is_simply_no_bridge():
    from orion_core.data import ToolResult

    dispatcher = _dispatcher(None)

    async def quick(_args):
        return ToolResult("done")

    dispatcher.handler_table = lambda: {"probe": quick}
    assert asyncio.run(dispatcher.dispatch("probe", {})).ok is True


# ── the swarm view builds real routes ────────────────────────────────────────

@pytest.fixture(scope="module")
def _app():
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _view(_app, **kw):
    from orion_core.bus import OrionBus
    from orion_core.gui.swarm_view import SwarmDeckView
    return SwarmDeckView(OrionBus(), **kw)


def test_a_memory_tool_routes_through_the_real_constellation(_app):
    """Not the generic cluster route — the tiers ORION actually holds."""
    view = _view(_app, memory=_Memory())
    route = view._route_for("recall", {"query": "invoice"})
    assert route[:2] == ["core", f"cluster:{cl.MEMORY}"]
    assert "memory:working" in route
    assert route.index("memory:working") < route.index("memory:archived")


def test_a_non_memory_tool_keeps_the_ordinary_route(_app):
    view = _view(_app, memory=_Memory())
    route = view._route_for("agent_dispatch", {"agent": "research"})
    assert route[-1] == "agent:research"
    assert not any(r.startswith("memory:") for r in route)


def test_a_memory_tool_with_no_memory_backend_still_routes(_app):
    view = _view(_app)
    assert view._route_for("recall", {}) == ["core", f"cluster:{cl.INTELLIGENCE}",
                                             "module:reasoning",
                                             f"cluster:{cl.MEMORY}"]


def test_launching_and_resolving_through_the_view(_app):
    view = _view(_app, memory=_Memory())
    flight = view.launch_bridge("recall", {"query": "x"})
    assert flight is not None
    assert view.bridges.live() == 1
    view.resolve_bridge(flight, ok=True)
    pulses = view.bridges.pulses()
    assert pulses and pulses[0].phase in (Phase.OUTBOUND, Phase.RETURNING)


def test_resolving_nothing_is_harmless(_app):
    _view(_app).resolve_bridge(None, ok=True)


def test_pulse_still_works_with_the_old_two_argument_call(_app):
    """Older call sites must keep working — pulse gained three parameters."""
    _view(_app).pulse("core", "module:vision")


def test_a_dispatch_before_the_renderer_exists_is_still_recorded(_app):
    """Startup dispatches tools before the GL view is built; those commands
    must still be in flight when it appears."""
    view = _view(_app)
    assert view.gl_view is None
    assert view.launch_bridge("probe", {}) is not None
    assert view.bridges.live() == 1


# ── the renderer's own seams, without a GL context ───────────────────────────

def test_the_renderer_exposes_hover_and_bridges_without_a_context(_app):
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    ledger_view = _view(_app)
    view.attach_bridges(ledger_view.bridges)
    assert view.bridges is ledger_view.bridges
    assert view.hover.target is None
    view.set_labels_enabled(False)


def test_bridge_marks_drop_packets_whose_nodes_are_not_on_screen(_app):
    """A route can name a node removed mid-flight; drawing it at the origin
    would invent a location."""
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    ledger_view = _view(_app)
    view.attach_bridges(ledger_view.bridges)
    ledger_view.bridges.launch(["ghost:a", "ghost:b"], label="gone")
    assert view._window._bridge_marks(1.0) == []


def test_the_hovered_node_is_labelled_alongside_the_cluster_anchors(_app):
    from orion_core.swarm.render.gl_view import SwarmGLView
    from orion_core.swarm.sources import compose

    view = SwarmGLView()
    window = view._window
    window.apply_snapshot(compose(memory=_Memory()))
    anchors = {e[0] for e in window._callout_entries()}
    assert "core" in anchors
    assert f"cluster:{cl.MEMORY}" in anchors
    assert "memory:archived" not in anchors

    window.hover.point_at("memory:archived")
    window.hover.unfold()
    assert "memory:archived" in {e[0] for e in window._callout_entries()}


def test_an_unfolding_readout_keeps_the_view_repainting(_app):
    """A CPU-driven animation is not covered by the shader-driven cases."""
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    window = view._window
    window._particle_count = 0
    assert window._has_animation() is False
    window.hover.point_at("memory:working")
    window.hover.unfold()
    assert window._has_animation() is True


def test_a_command_in_flight_keeps_the_view_repainting(_app):
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    window = view._window
    window._particle_count = 0
    ledger_view = _view(_app)
    view.attach_bridges(ledger_view.bridges)
    assert window._has_animation() is False
    ledger_view.bridges.launch(["core", "module:x"])
    assert window._has_animation() is True


def test_a_settled_idle_view_still_stops_repainting(_app):
    """The GPU must go quiet when nothing is happening — adding two new
    animation sources must not have broken that."""
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    window = view._window
    window._particle_count = 0
    assert window._has_animation() is False


def test_static_particle_depth_does_not_keep_the_renderer_animating(_app):
    """The particle field should add depth, not force an idle 60 FPS loop."""
    from orion_core.swarm.render.gl_view import SwarmGLView

    view = SwarmGLView()
    window = view._window
    window._particle_count = 2048
    assert window._has_animation() is False
