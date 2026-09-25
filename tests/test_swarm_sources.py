"""
Tests for orion_core.swarm.telemetry_bind and .sources (Mark XXII, Phase 0).

Deliberately exercised against the REAL Telemetry/MetricsRegistry objects, not
fakes — the audit's finding was that live metrics existed and were being
discarded, so a test that stubs the metrics layer would prove nothing about
whether the binding actually reads what the dispatcher actually writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.model import (  # noqa: E402
    Activity,
    EdgeKind,
    Health,
    NodeKind,
    NodeTelemetry,
)
from orion_core.swarm.sources import CORE_ID, cluster_id, compose  # noqa: E402
from orion_core.swarm.telemetry_bind import (  # noqa: E402
    TelemetryIndex,
    activity_for,
    bind_node,
)
from orion_core.telemetry import Telemetry  # noqa: E402


class _Signal:
    def __init__(self): self.messages = []
    def emit(self, *a): self.messages.append(a)
    def connect(self, *a): pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def _real_telemetry() -> Telemetry:
    """A real Telemetry primed exactly the way the dispatcher primes it."""
    tel = Telemetry(_StubBus())
    for _ in range(8):
        tel.record_tool_call("weather", ok=True)
        tel.metrics.observe("tool.weather.ms", 12.0)
    tel.record_tool_call("flaky", ok=False)
    for _ in range(9):
        tel.record_tool_call("flaky", ok=False)
    tel.metrics.observe("tool.flaky.ms", 400.0)
    tel.health.register("vision")
    tel.health.beat("vision", status="DEGRADED", detail="camera busy")
    return tel


# ── TelemetryIndex ───────────────────────────────────────────────────────────

def test_index_reads_the_counters_the_dispatcher_actually_writes():
    idx = TelemetryIndex(_real_telemetry())
    tel = idx.for_tool("weather")
    assert tel.calls == 8
    assert tel.failures == 0


def test_index_reads_the_latency_percentiles_already_being_computed():
    idx = TelemetryIndex(_real_telemetry())
    tel = idx.for_tool("weather")
    assert tel.latency_p50_ms == 12.0
    assert tel.latency_max_ms == 12.0
    assert tel.latency_avg_ms == 12.0


def test_index_computes_a_real_error_rate_from_real_counters():
    idx = TelemetryIndex(_real_telemetry())
    assert idx.for_tool("flaky").error_rate == 1.0


def test_index_reads_component_health_and_detail():
    idx = TelemetryIndex(_real_telemetry())
    assert idx.health_of("vision") is Health.DEGRADED
    assert idx.health_detail("vision") == "camera busy"
    assert idx.heartbeat_age("vision") is not None


def test_index_is_total_for_unknown_names():
    idx = TelemetryIndex(_real_telemetry())
    assert idx.for_tool("never_called") == NodeTelemetry()
    assert idx.health_of("no_such_component") is Health.UNKNOWN
    assert idx.heartbeat_age("no_such_component") is None


def test_index_with_no_telemetry_at_all_is_safe_and_reports_unavailable():
    idx = TelemetryIndex(None)
    assert idx.available is False
    assert idx.for_tool("weather") == NodeTelemetry()


def test_index_survives_a_telemetry_object_that_raises():
    class _Broken:
        @property
        def metrics(self): raise RuntimeError("boom")
        @property
        def health(self): raise RuntimeError("boom")

    idx = TelemetryIndex(_Broken())
    assert idx.available is False
    assert idx.for_tool("x") == NodeTelemetry()


def test_index_takes_only_one_metrics_snapshot_regardless_of_lookups():
    """MetricsRegistry.snapshot() sorts every bucket (1.32 ms measured); doing
    it per node would multiply that by the node count on every refresh."""
    tel = _real_telemetry()
    calls = {"n": 0}
    original = tel.metrics.snapshot

    def counting():
        calls["n"] += 1
        return original()

    tel.metrics.snapshot = counting
    idx = TelemetryIndex(tel)
    for _ in range(50):
        idx.for_tool("weather")
        idx.for_component("vision")
    assert calls["n"] == 1


# ── activity derivation ──────────────────────────────────────────────────────

def test_a_down_component_reads_as_offline():
    assert activity_for(NodeTelemetry(), Health.DOWN) is Activity.OFFLINE


def test_a_high_error_rate_on_a_real_sample_reads_as_error():
    assert activity_for(NodeTelemetry(calls=10, failures=9), Health.OK) is Activity.ERROR


def test_one_early_failure_does_not_paint_a_node_red():
    """Over-alarming would be a worse replacement for 'no activity concept'
    than having none at all."""
    assert activity_for(NodeTelemetry(calls=2, failures=1), Health.OK) is not Activity.ERROR


def test_a_node_reporting_a_current_action_reads_as_busy():
    assert activity_for(NodeTelemetry(current_action="indexing"), Health.OK) is Activity.BUSY


def test_a_never_called_node_reads_as_idle():
    assert activity_for(NodeTelemetry(), Health.OK) is Activity.IDLE


def test_bind_node_applies_telemetry_and_derives_activity():
    from orion_core.swarm.model import SwarmNode
    node = SwarmNode(id="module:vision", kind=NodeKind.SUBSYSTEM,
                     label="vision", cluster=cl.SYSTEM)
    bound = bind_node(node, NodeTelemetry(calls=10, failures=9), Health.OK)
    assert bound.telemetry.calls == 10
    assert bound.activity is Activity.ERROR
    assert node.activity is Activity.IDLE       # original untouched


# ── compose() ────────────────────────────────────────────────────────────────

class _Agents:
    def describe(self):
        return [{"name": "coding", "title": "Coding Agent", "focus": "review",
                 "calls": 3, "last_active": ""},
                {"name": "research", "title": "Research Agent", "focus": "analysis",
                 "calls": 0, "last_active": ""}]


class _Modules:
    def all_described(self, telemetry):
        return {
            "memory": {"role": "the matrix", "health": "OK", "dependencies": []},
            "dispatcher": {"role": "tool router", "health": "OK",
                           "dependencies": ["memory", "ghost"]},
        }


class _Registries:
    def __init__(self): self.modules = _Modules()


class _MCP:
    def catalogue(self):
        return {"github": {"description": "repos",
                           "tools": [{"name": "list_repos"}, {"name": "open_pr"}]}}
    def health_snapshot(self):
        return {"github": "OK"}


class _WF:
    definitions = {"morning": {"steps": [{"tool": "briefing"}, {"tool": "weather"}]}}


class _Mem:
    def tiers_snapshot(self): return {"short_term": 4, "long_term": 0}


def _full():
    return compose(core_state="THINKING", agents=_Agents(),
                   registries=_Registries(), telemetry=_real_telemetry(),
                   mcp_host=_MCP(), workflow_engine=_WF(), memory=_Mem())


def test_compose_always_produces_core_and_all_twelve_clusters():
    snap = compose()
    ids = snap.node_ids()
    assert CORE_ID in ids
    for cluster in cl.CLUSTER_ORDER:
        assert cluster_id(cluster) in ids


def test_every_cluster_hangs_off_core():
    snap = compose()
    memberships = {(e.source, e.target) for e in snap.edges}
    for cluster in cl.CLUSTER_ORDER:
        assert (CORE_ID, cluster_id(cluster)) in memberships


def test_agents_land_in_their_own_cluster_not_on_core():
    snap = _full()
    coding = snap.by_id()["agent:coding"]
    assert coding.cluster == cl.DEVELOPMENT
    assert coding.parent == cluster_id(cl.DEVELOPMENT)
    # the flat "everything orbits core" ring is gone
    assert (CORE_ID, "agent:coding") not in {(e.source, e.target) for e in snap.edges}


def test_modules_are_clustered_by_their_real_zone():
    snap = _full()
    assert snap.by_id()["module:memory"].cluster == cl.MEMORY
    assert snap.by_id()["module:dispatcher"].cluster == cl.SYSTEM


def test_a_dangling_dependency_never_produces_an_edge():
    snap = _full()
    targets = {e.target for e in snap.edges if e.kind is EdgeKind.DEPENDENCY}
    assert "module:memory" in targets
    assert "module:ghost" not in targets


def test_mcp_tools_become_their_own_nodes_under_their_server():
    snap = _full()
    tool = snap.by_id()["mcp_tool:github:list_repos"]
    assert tool.kind is NodeKind.MCP_TOOL
    assert tool.parent == "mcp:github"
    assert tool.cluster == cl.COMMUNICATION


def test_an_mcp_server_lists_its_tools_as_capabilities():
    server = _full().by_id()["mcp:github"]
    assert set(server.capabilities) == {"list_repos", "open_pr"}


def test_workflows_and_memory_tiers_reach_their_clusters():
    snap = _full()
    assert snap.by_id()["workflow:morning"].cluster == cl.AUTOMATION
    assert snap.by_id()["memory:short_term"].cluster == cl.MEMORY


def test_a_populated_memory_tier_is_active_and_an_empty_one_is_idle():
    snap = _full()
    assert snap.by_id()["memory:short_term"].activity is Activity.ACTIVE
    assert snap.by_id()["memory:long_term"].activity is Activity.IDLE


def test_module_health_comes_through_from_the_registry():
    assert _full().by_id()["module:memory"].health is Health.OK


def test_core_carries_the_aggregate_tool_counters():
    core = _full().by_id()[CORE_ID]
    assert core.telemetry.calls == 18      # 8 weather + 10 flaky
    assert core.telemetry.failures == 10


# ── failure isolation ────────────────────────────────────────────────────────

def test_one_broken_source_never_blanks_the_others():
    class _Broken:
        def describe(self): raise RuntimeError("boom")

    snap = compose(agents=_Broken(), registries=_Registries(), memory=_Mem())
    assert "agents" in snap.degraded_sources
    assert "module:memory" in snap.node_ids()      # survived
    assert "memory:short_term" in snap.node_ids()


def test_degraded_sources_names_which_part_of_the_picture_is_missing():
    """The old version swallowed exceptions, so a crashed MCP host looked
    exactly like having no MCP servers configured."""
    class _BrokenMCP:
        def catalogue(self): raise RuntimeError("host died")

    snap = compose(mcp_host=_BrokenMCP())
    assert snap.degraded_sources == ("mcp",)


def test_a_healthy_compose_reports_no_degraded_sources():
    assert _full().degraded_sources == ()


def test_compose_with_no_backends_at_all_still_produces_a_valid_graph():
    snap = compose()
    assert len(snap.nodes) == 13          # core + 12 clusters
    assert snap.degraded_sources == ()


def test_duplicate_node_ids_are_dropped_rather_than_duplicated():
    class _Dupes:
        def describe(self):
            return [{"name": "coding", "title": "A", "focus": "", "calls": 0},
                    {"name": "coding", "title": "B", "focus": "", "calls": 0}]

    snap = compose(agents=_Dupes())
    assert len([n for n in snap.nodes if n.id == "agent:coding"]) == 1


def test_snapshot_records_when_it_was_captured():
    assert compose().captured_at > 0.0
