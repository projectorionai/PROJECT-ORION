"""
Tests for the Track C toolbelt now reaching the everyday agent_dispatch path
(AgentManager.dispatch), not just the separate, rarer `reason` tool. Before
this, dispatch() always called BaseAgent.handle() — a blind persona + one
provider call with no lookups available at all, even though investigate()
(with a real read-only instrument tray) already existed and just wasn't
reachable from here.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.agents import AgentFinding, AgentManager, AgentToolbelt, BaseAgent, ToolInvocation
from orion_core.data import ToolResult


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubRouter:
    def has_text_fallback(self):
        return False


def _manager() -> AgentManager:
    return AgentManager(_StubRouter(), _StubBus())


# ── toolbelt reaches investigate() ──────────────────────────────────────────

def test_dispatch_passes_the_attached_toolbelt_to_investigate(monkeypatch):
    mgr = _manager()
    captured: dict = {}

    async def _fake_investigate(self, request, context="", toolbelt=None, brief=""):
        captured["toolbelt"] = toolbelt
        return AgentFinding(self.name, self.title, "answer text")

    monkeypatch.setattr(BaseAgent, "investigate", _fake_investigate)
    sentinel = AgentToolbelt()   # a real (empty) toolbelt, just needs identity
    mgr.attach_toolbelt_factory(lambda: sentinel)
    asyncio.run(mgr.dispatch("debug this python traceback", agent_name="coding"))
    assert captured["toolbelt"] is sentinel


def test_dispatch_passes_none_when_no_factory_attached(monkeypatch):
    mgr = _manager()
    captured: dict = {}

    async def _fake_investigate(self, request, context="", toolbelt=None, brief=""):
        captured["toolbelt"] = toolbelt
        return AgentFinding(self.name, self.title, "answer text")

    monkeypatch.setattr(BaseAgent, "investigate", _fake_investigate)
    asyncio.run(mgr.dispatch("debug this python traceback", agent_name="coding"))
    assert captured["toolbelt"] is None


def test_dispatch_builds_a_fresh_toolbelt_from_the_factory_each_call():
    mgr = _manager()
    built: list[AgentToolbelt] = []

    def _factory():
        belt = AgentToolbelt()
        built.append(belt)
        return belt

    mgr.attach_toolbelt_factory(_factory)
    asyncio.run(mgr.dispatch("debug this", agent_name="coding"))
    asyncio.run(mgr.dispatch("debug that", agent_name="coding"))
    assert len(built) == 2
    assert built[0] is not built[1]


# ── result shape: AgentFinding adapted into ToolResult ─────────────────────

def test_dispatch_result_carries_the_finding_answer_and_title(monkeypatch):
    mgr = _manager()

    async def _fake_investigate(self, request, context="", toolbelt=None, brief=""):
        return AgentFinding(self.name, self.title, "here is the fix", ok=True)

    monkeypatch.setattr(BaseAgent, "investigate", _fake_investigate)
    result = asyncio.run(mgr.dispatch("debug this python traceback", agent_name="coding"))
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "here is the fix" in result.text
    assert "Coding" in result.text or "coding" in result.text.lower()


def test_dispatch_result_reflects_a_failed_finding(monkeypatch):
    mgr = _manager()

    async def _fake_investigate(self, request, context="", toolbelt=None, brief=""):
        return AgentFinding(self.name, self.title, "could not reach a provider", ok=False)

    monkeypatch.setattr(BaseAgent, "investigate", _fake_investigate)
    result = asyncio.run(mgr.dispatch("debug this python traceback", agent_name="coding"))
    assert not result.ok


# ── activity tracking surfaces through describe() ───────────────────────────

def test_describe_shows_zero_calls_before_any_dispatch():
    rows = _manager().describe()
    coding = next(r for r in rows if r["name"] == "coding")
    assert coding["calls"] == 0
    assert coding["last_active"] == ""


def test_describe_reflects_call_count_and_last_active_after_dispatch(monkeypatch):
    mgr = _manager()

    async def _fake_investigate(self, request, context="", toolbelt=None, brief=""):
        return AgentFinding(self.name, self.title, "answer")

    monkeypatch.setattr(BaseAgent, "investigate", _fake_investigate)
    asyncio.run(mgr.dispatch("debug this python traceback", agent_name="coding"))
    asyncio.run(mgr.dispatch("debug this python traceback again", agent_name="coding"))
    rows = mgr.describe()
    coding = next(r for r in rows if r["name"] == "coding")
    assert coding["calls"] == 2
    assert coding["last_active"] != ""


def test_describe_only_tracks_the_agent_actually_dispatched(monkeypatch):
    mgr = _manager()

    async def _fake_investigate(self, request, context="", toolbelt=None, brief=""):
        return AgentFinding(self.name, self.title, "answer")

    monkeypatch.setattr(BaseAgent, "investigate", _fake_investigate)
    asyncio.run(mgr.dispatch("debug this python traceback", agent_name="coding"))
    rows = {r["name"]: r for r in mgr.describe()}
    assert rows["coding"]["calls"] == 1
    assert rows["marketing"]["calls"] == 0


# ── evidence trail (Mark XX design-spec §6, High Impact item 6) ────────────
#
# finding.tools — every instrument reading a specialist took — used to be
# computed by investigate() and then discarded right at this dispatch()
# boundary; only the prose answer ever reached the caller. It now rides
# along on ToolResult.evidence so a GUI surface can show what grounded the
# answer instead of the grounding being invisible.

def test_dispatch_surfaces_the_toolbelt_evidence_trail(monkeypatch):
    mgr = _manager()

    async def _fake_investigate(self, request, context="", toolbelt=None, brief=""):
        finding = AgentFinding(self.name, self.title, "answer text")
        finding.tools.append(ToolInvocation(
            tool="memory_search", args={"query": "x"}, ok=True, observation="3 hits"))
        finding.tools.append(ToolInvocation(
            tool="knowledge_graph", args={}, ok=False, observation="graph offline"))
        return finding

    monkeypatch.setattr(BaseAgent, "investigate", _fake_investigate)
    result = asyncio.run(mgr.dispatch("debug this python traceback", agent_name="coding"))
    assert result.evidence is not None
    assert len(result.evidence) == 2
    assert result.evidence[0]["tool"] == "memory_search"
    assert result.evidence[0]["ok"] is True
    assert "memory_search" in result.evidence[0]["summary"]
    assert result.evidence[1]["ok"] is False


def test_dispatch_evidence_is_none_when_the_specialist_looked_nothing_up(monkeypatch):
    mgr = _manager()

    async def _fake_investigate(self, request, context="", toolbelt=None, brief=""):
        return AgentFinding(self.name, self.title, "answer from persona alone")

    monkeypatch.setattr(BaseAgent, "investigate", _fake_investigate)
    result = asyncio.run(mgr.dispatch("debug this python traceback", agent_name="coding"))
    assert result.evidence is None


# ── available_instruments() (agent workspace "Instruments" panel) ──────────
#
# Pure introspection — constructing a toolbelt costs nothing until .run() is
# called — so a GUI workspace can show "what can this agent look up right
# now" without spending any of the investigation budget.

def test_available_instruments_is_empty_with_no_factory_attached():
    mgr = _manager()
    assert mgr.available_instruments() == []


def test_available_instruments_lists_the_attached_toolbelts_names():
    mgr = _manager()
    belt = AgentToolbelt(allowed={"memory_search": "search memory", "web_search": "search the web"})
    mgr.attach_toolbelt_factory(lambda: belt)
    assert mgr.available_instruments() == ["memory_search", "web_search"]


def test_available_instruments_never_raises_on_a_broken_factory():
    mgr = _manager()
    mgr.attach_toolbelt_factory(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert mgr.available_instruments() == []


def test_available_instruments_does_not_spend_any_budget():
    mgr = _manager()
    belt = AgentToolbelt(allowed={"memory_search": "search memory"}, budget=2)
    mgr.attach_toolbelt_factory(lambda: belt)
    mgr.available_instruments()
    mgr.available_instruments()
    assert belt.remaining() == 2
