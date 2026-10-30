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
from collections import deque
import sys
from pathlib import Path

import pytest

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


def test_dispatch_degrades_to_a_specialist_answer_when_toolbelt_factory_fails():
    mgr = _manager()
    mgr.attach_toolbelt_factory(lambda: (_ for _ in ()).throw(RuntimeError("offline store")))
    result = asyncio.run(mgr.dispatch("debug this", agent_name="coding"))
    assert result.ok and "Specialist brief" in result.text


def test_dispatch_preserves_a_forced_close_failure_and_its_evidence():
    class _Profile:
        name = "scripted"

    class _Router:
        def __init__(self):
            self.calls = 0

        def has_text_fallback(self):
            return True

        async def generate_text(self, *args, **kwargs):
            self.calls += 1
            if self.calls > BaseAgent.MAX_INVESTIGATION_STEPS:
                raise RuntimeError("synthesis unavailable")
            return _Profile(), 'TOOL: query_intelligence\nARGS: {"query": "build"}'

    mgr = AgentManager(_Router(), _StubBus())

    async def _lookup(_name, _args):
        return ToolResult("Found the build settings")

    mgr.attach_toolbelt_factory(lambda: AgentToolbelt(
        dispatch=_lookup, allowed={"query_intelligence": "memory"}, budget=99))
    result = asyncio.run(mgr.dispatch("debug this", agent_name="coding"))
    assert not result.ok
    assert result.evidence and len(result.evidence) == BaseAgent.MAX_INVESTIGATION_STEPS


def test_provider_schema_advertises_all_registered_specialists():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    declaration = next(row for row in TOOL_DECLARATIONS if row["name"] == "agent_dispatch")
    selector = declaration["parameters"]["properties"]["agent"]
    for name in _manager().agent_names():
        assert name in declaration["description"]
        assert name in selector["description"]


@pytest.mark.parametrize("question,name,instrument", [
    ("Diagnose why ORION stopped talking at 23:41", "reliability", "incident_lookup"),
    ("Review the API integration contract and OAuth retry policy", "integrations", "capability_status"),
    ("Resume my project and identify unfinished next steps", "continuity", "situation"),
])
@pytest.mark.parametrize("selection", ["auto", "named"])
@pytest.mark.parametrize("voice", [False, True])
def test_operational_specialists_use_the_real_voice_and_text_dispatch_route(
        monkeypatch, question, name, instrument, selection, voice):
    from orion_core import dispatcher as dispatcher_module, speaker_gate
    from orion_core.reasoning import BUDGETS, ReasoningEngine, ReasoningTier
    from orion_core.tool_resolver import ResolverState, resolve

    class _Profile:
        name = "scripted"

    class _Router:
        def __init__(self):
            self.calls = []

        def has_text_fallback(self):
            return True

        async def generate_text(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            if len(self.calls) == 1:
                assert "Saved project evidence" in prompt
                return _Profile(), f'TOOL: {instrument}\nARGS: {{"query": "orion"}}'
            assert "Observed state" in prompt
            return _Profile(), "The observed state supports this next step."

    observed = []
    router, bus = _Router(), _StubBus()
    manager = AgentManager(router, bus)

    async def _observe(tool, args):
        observed.append((tool, args))
        return ToolResult("Observed state")

    engine = ReasoningEngine(bus, router, manager, dispatch=_observe)
    manager.attach_toolbelt_factory(lambda: engine.toolbelt(BUDGETS[ReasoningTier.STANDARD]))
    dispatcher = dispatcher_module.OrionDispatcher.__new__(dispatcher_module.OrionDispatcher)
    dispatcher.bus = bus
    dispatcher.agent_manager = manager
    dispatcher.telemetry = None
    dispatcher.active_tools = 0
    dispatcher.recent_tools = deque(maxlen=60)
    monkeypatch.setattr(dispatcher_module, "_note_gap", lambda *_a: None)

    selected = resolve(question, ResolverState(), dispatcher_module.TOOL_DECLARATIONS)
    assert "agent_dispatch" in {row["name"] for row in selected}

    async def _scenario():
        args = {"request": question, "agent": name if selection == "named" else "auto",
                "context": "Saved project evidence"}
        if voice:
            with speaker_gate.voice_turn():
                return await dispatcher.dispatch("agent_dispatch", args)
        return await dispatcher.dispatch("agent_dispatch", args)

    result = asyncio.run(_scenario())
    assert result.ok and manager.get(name).title in result.text
    assert "observed state supports" in result.text
    assert result.evidence and result.evidence[0]["tool"] == instrument
    assert result.evidence[0]["ok"] is True
    assert observed
    assert next(row for row in manager.describe() if row["name"] == name)["calls"] == 1


@pytest.mark.parametrize("question", ["connection", "continue", "slow"])
def test_weak_words_keep_the_general_dispatch_route(question):
    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    class _Dispatch(ProductivityDispatchMixin):
        agent_manager = _manager()

    result = asyncio.run(_Dispatch().agent_dispatch({"request": question}))
    assert not result.ok and "Nothing was dispatched" in result.text      # Mark XXXX.4
    assert all(row["calls"] == 0 for row in _Dispatch.agent_manager.describe())



@pytest.mark.parametrize("key", ["request", "task", "prompt", "question"])
def test_agent_dispatch_takes_the_request_however_the_model_names_it(key):
    from types import SimpleNamespace
    """26-27 Oct 2026: "No request supplied for agent dispatch" failed two
    owner requests when the model named the field something else."""
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    seen = []

    async def dispatch(**kwargs):
        seen.append(kwargs["request"])
        return ToolResult("ok")

    owner = SimpleNamespace(agent_manager=SimpleNamespace(dispatch=dispatch), jobs=None)
    result = asyncio.run(ProductivityDispatchMixin.agent_dispatch(owner, {key: "check my sleep data"}))
    assert result.ok and seen == ["check my sleep data"]
    empty = asyncio.run(ProductivityDispatchMixin.agent_dispatch(owner, {}))
    assert not empty.ok and "'request'" in empty.text


def test_what_are_you_working_on_includes_research_in_progress():
    from types import SimpleNamespace
    """30 Oct 2026: "how do I know if he's actually still doing something?" -
    the running research was not in the job list."""
    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    class Running:
        def done(self):
            return False

    research = SimpleNamespace(_active={"task": Running()},
                               status=lambda: ToolResult("Researching 'sleep' - 4.0 of 30 minutes elapsed."))
    jobs = SimpleNamespace(report=lambda: ToolResult("No background jobs."))
    owner = SimpleNamespace(jobs=jobs, research=research)
    owner._running_research = lambda: ProductivityDispatchMixin._running_research(owner)
    result = asyncio.run(ProductivityDispatchMixin.job_tool(owner, {"action": "list"}))
    assert result.text.startswith("Researching 'sleep'") and "No background jobs." in result.text
