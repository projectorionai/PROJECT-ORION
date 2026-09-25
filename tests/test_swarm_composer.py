"""
Tests for orion_core.swarm.composer (Mark XXII, Phase 2).

The composer is a pure optimisation, so the first duty of these tests is to
prove it changes nothing observable: a cached compose must be equivalent to
sources.compose. The second is to prove it actually optimises — that a steady
state reuses node objects instead of reallocating ~3,600 dataclasses per tick.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.composer import SnapshotComposer, _structure_signature  # noqa: E402
from orion_core.swarm.model import Health  # noqa: E402
from orion_core.swarm.sources import compose as compose_full  # noqa: E402
from orion_core.telemetry import Telemetry  # noqa: E402


class _Signal:
    def emit(self, *a): pass
    def connect(self, *a): pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Agents:
    def __init__(self, names=("coding", "research")): self.names = list(names)
    def describe(self):
        return [{"name": n, "title": f"{n.title()} Agent", "focus": "x",
                 "calls": 0, "last_active": ""} for n in self.names]


class _Modules:
    def __init__(self, names=("memory", "dispatcher"), sick=()):
        self.names, self.sick = list(names), set(sick)
    def all_described(self, telemetry):
        return {n: {"role": "r", "health": "DOWN" if n in self.sick else "OK",
                    "dependencies": []} for n in self.names}


class _Reg:
    def __init__(self, names=("memory", "dispatcher"), sick=()):
        self.modules = _Modules(names, sick)


class _MCP:
    def __init__(self, tools=("list_repos", "open_pr")): self.tools = list(tools)
    def catalogue(self):
        return {"github": {"tools": [{"name": t} for t in self.tools]}}
    def health_snapshot(self): return {"github": "OK"}


class _WF:
    def __init__(self, names=("morning",)):
        self.definitions = {n: {"steps": [{"tool": "t"}]} for n in names}


class _Mem:
    def __init__(self, tiers=("short_term", "long_term")): self.tiers = list(tiers)
    def tiers_snapshot(self): return {t: 3 for t in self.tiers}


def _kwargs(**overrides):
    base = dict(core_state="THINKING", agents=_Agents(), registries=_Reg(),
                telemetry=None, mcp_host=_MCP(), workflow_engine=_WF(),
                memory=_Mem())
    base.update(overrides)
    return base


def _telemetry() -> Telemetry:
    tel = Telemetry(_StubBus())
    tel.health.register("memory")
    tel.health.beat("memory", status="OK", detail="")
    return tel


# ── equivalence with the authoritative composer ──────────────────────────────

def test_a_cold_compose_matches_sources_compose_exactly():
    kw = _kwargs()
    assert SnapshotComposer().compose(**kw).node_ids() == compose_full(**kw).node_ids()


def test_a_cached_compose_still_matches_sources_compose():
    kw = _kwargs()
    composer = SnapshotComposer()
    composer.compose(**kw)                      # warm
    cached = composer.compose(**kw)             # cached path
    fresh = compose_full(**kw)
    assert cached.node_ids() == fresh.node_ids()
    assert len(cached.edges) == len(fresh.edges)


def test_node_clusters_and_kinds_survive_the_cached_path():
    kw = _kwargs()
    composer = SnapshotComposer()
    composer.compose(**kw)
    cached = composer.compose(**kw).by_id()
    fresh = compose_full(**kw).by_id()
    for node_id, node in fresh.items():
        assert cached[node_id].cluster == node.cluster
        assert cached[node_id].kind is node.kind


def test_the_core_state_is_honoured_on_the_cached_path():
    composer = SnapshotComposer()
    composer.compose(**_kwargs(core_state="IDLE"))
    assert composer.compose(**_kwargs(core_state="SPEAKING")).core_state == "SPEAKING"


# ── the optimisation itself ──────────────────────────────────────────────────

def test_an_unchanged_structure_is_reused_rather_than_rebuilt():
    composer = SnapshotComposer()
    kw = _kwargs()
    for _ in range(10):
        composer.compose(**kw)
    assert composer.structure_rebuilds == 1
    assert composer.structure_reuses == 9


def test_a_steady_state_allocates_no_new_nodes():
    """The whole point: ~3,600 dataclasses per tick becomes zero."""
    composer = SnapshotComposer()
    kw = _kwargs()
    composer.compose(**kw)
    baseline = composer.node_allocations
    for _ in range(20):
        composer.compose(**kw)
    assert composer.node_allocations == baseline


def test_the_same_node_objects_are_handed_back_when_nothing_changed():
    composer = SnapshotComposer()
    kw = _kwargs()
    first = composer.compose(**kw).by_id()
    second = composer.compose(**kw).by_id()
    assert second["module:memory"] is first["module:memory"]


# ── invalidation ─────────────────────────────────────────────────────────────

def test_adding_an_agent_rebuilds_the_structure():
    composer = SnapshotComposer()
    composer.compose(**_kwargs())
    snapshot = composer.compose(**_kwargs(agents=_Agents(("coding", "research", "legal"))))
    assert composer.structure_rebuilds == 2
    assert "agent:legal" in snapshot.node_ids()


def test_removing_a_module_rebuilds_the_structure():
    composer = SnapshotComposer()
    composer.compose(**_kwargs())
    snapshot = composer.compose(**_kwargs(registries=_Reg(("memory",))))
    assert "module:dispatcher" not in snapshot.node_ids()


def test_a_new_mcp_tool_rebuilds_the_structure():
    """MCP servers appear and disappear at runtime; a tool arriving must show
    up without waiting for anything else to change."""
    composer = SnapshotComposer()
    composer.compose(**_kwargs())
    snapshot = composer.compose(**_kwargs(mcp_host=_MCP(("list_repos", "open_pr", "merge"))))
    assert "mcp_tool:github:merge" in snapshot.node_ids()


def test_a_new_workflow_rebuilds_the_structure():
    composer = SnapshotComposer()
    composer.compose(**_kwargs())
    snapshot = composer.compose(**_kwargs(workflow_engine=_WF(("morning", "nightly"))))
    assert "workflow:nightly" in snapshot.node_ids()


def test_a_new_memory_tier_rebuilds_the_structure():
    composer = SnapshotComposer()
    composer.compose(**_kwargs())
    snapshot = composer.compose(**_kwargs(memory=_Mem(("short_term", "long_term", "episodic"))))
    assert "memory:episodic" in snapshot.node_ids()


def test_telemetry_change_alone_does_not_rebuild_the_structure():
    """Measurements moving is the common case and must stay on the fast path."""
    composer = SnapshotComposer()
    telemetry = _telemetry()
    kw = _kwargs(telemetry=telemetry)
    composer.compose(**kw)
    for i in range(5):
        telemetry.record_tool_call("weather", ok=True)
        composer.compose(**kw)
    assert composer.structure_rebuilds == 1


def test_invalidate_forces_a_rebuild():
    composer = SnapshotComposer()
    kw = _kwargs()
    composer.compose(**kw)
    composer.invalidate()
    composer.compose(**kw)
    assert composer.structure_rebuilds == 2


# ── live data still flows through the cached path ────────────────────────────

def test_a_module_going_down_is_reflected_without_a_structure_rebuild():
    composer = SnapshotComposer()
    composer.compose(**_kwargs())
    snapshot = composer.compose(**_kwargs(registries=_Reg(sick={"memory"})))
    assert composer.structure_rebuilds == 1          # same module NAMES
    assert snapshot.by_id()["module:memory"].health is Health.DOWN


def test_mcp_tool_telemetry_is_rebound_on_the_cached_path():
    telemetry = _telemetry()
    composer = SnapshotComposer()
    kw = _kwargs(telemetry=telemetry)
    composer.compose(**kw)
    for _ in range(6):
        telemetry.record_tool_call("mcp__github__list_repos", ok=True)
    snapshot = composer.compose(**kw)
    assert snapshot.by_id()["mcp_tool:github:list_repos"].telemetry.calls == 6


def test_core_aggregate_counters_update_on_the_cached_path():
    telemetry = _telemetry()
    composer = SnapshotComposer()
    kw = _kwargs(telemetry=telemetry)
    composer.compose(**kw)
    for _ in range(4):
        telemetry.record_tool_call("weather", ok=True)
    assert composer.compose(**kw).by_id()["core"].telemetry.calls == 4


# ── robustness ───────────────────────────────────────────────────────────────

def test_a_broken_source_does_not_poison_the_cache():
    class _Broken:
        def describe(self): raise RuntimeError("boom")

    composer = SnapshotComposer()
    snapshot = composer.compose(**_kwargs(agents=_Broken()))
    assert "agents" in snapshot.degraded_sources
    assert "module:memory" in snapshot.node_ids()


def test_a_source_that_starts_failing_forces_a_rebuild():
    """A backend dying changes what the graph should show, so the cached
    structure must not survive it."""
    class _Broken:
        def describe(self): raise RuntimeError("boom")

    composer = SnapshotComposer()
    composer.compose(**_kwargs())
    composer.compose(**_kwargs(agents=_Broken()))
    assert composer.structure_rebuilds == 2


def test_compose_with_no_backends_is_valid_and_cacheable():
    composer = SnapshotComposer()
    first = composer.compose()
    second = composer.compose()
    assert first.node_ids() == second.node_ids()
    assert composer.structure_rebuilds == 1


def test_signature_ignores_measurements_entirely():
    """If the signature saw telemetry it would never hit the cache."""
    a = _structure_signature(_Agents(), _Reg(), _MCP(), _WF(), _Mem())
    b = _structure_signature(_Agents(), _Reg(sick={"memory"}), _MCP(), _WF(), _Mem())
    assert a == b


def test_stats_report_the_reuse_ratio():
    composer = SnapshotComposer()
    kw = _kwargs()
    for _ in range(10):
        composer.compose(**kw)
    stats = composer.stats()
    assert stats["composes"] == 10
    assert stats["reuse_ratio"] == 0.9
