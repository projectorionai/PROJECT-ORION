"""
Phase 3 core tests — ExecutiveCore, EvidenceEngine, ResearchDirector,
SystemRegistries and the DynamicBriefingEngine.

Everything runs offline against stub buses/cognition; no model, no network,
no Qt event loop.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.briefing_engine import (ContextMonitor, DynamicBriefingEngine,
                                        EventMonitor, PriorityMonitor)
from orion_core.data import ToolResult
from orion_core.evidence import EvidenceEngine, score_confidence
from orion_core.executive_core import (DecisionEngine, ExecutiveCore,
                                       PriorityEngine, StrategicEngine)
from orion_core.registries import SystemRegistries
from orion_core.research_director import ResearchDirector


class _Signal:
    def __init__(self, sink, name):
        self._sink = sink
        self._name = name

    def emit(self, *payload):
        self._sink.append((self._name, payload))

    def connect(self, *_a, **_k):
        pass


class StubBus:
    def __init__(self):
        self.emitted = []

    def __getattr__(self, name):
        sig = _Signal(self.emitted, name)
        object.__setattr__(self, name, sig)
        return sig


def _stamp(delta_hours: float = 0.0) -> str:
    moment = datetime.now(timezone.utc) + timedelta(hours=delta_hours)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class StubCognition:
    """Just enough CognitiveStateManager for the Phase 3 engines."""

    def __init__(self, state=None):
        self.state = state or {
            "active_projects": {}, "pending_tasks": {}, "goals": {},
            "open_workflows": {}, "user_priorities": [],
            "research_sessions": {},
        }

    def snapshot(self):
        import copy
        return copy.deepcopy(self.state)

    def upsert_research_session(self, topic, details=None):
        key = topic.lower().replace(" ", "-")
        sessions = self.state.setdefault("research_sessions", {})
        record = dict(sessions.get(key) or {})
        record.update(details or {})
        record.update({"id": key, "topic": topic, "updated_at": _stamp()})
        sessions[key] = record
        return record


# ── DecisionEngine / PriorityEngine / StrategicEngine ─────────────────────────

def test_decision_engine_challenges_launch_ideas_offline():
    engine = DecisionEngine(router=None)
    points = engine.challenge_offline("We should launch this product next week")
    text = " ".join(points).lower()
    assert "market saturation" in text
    assert len(points) >= 4
    # Deterministic: the same decision yields the same challenge.
    assert points == engine.challenge_offline("We should launch this product next week")


def test_decision_engine_full_challenge_without_model():
    result = asyncio.run(DecisionEngine(router=None).challenge("hire a video editor"))
    assert "Before committing to" in result
    assert "ramp-up" in result.lower() or "hire" in result.lower()


def test_priority_engine_ranks_overdue_first():
    state = {
        "pending_tasks": {
            "a": {"title": "pay the client invoice", "due": _stamp(-30)},
            "b": {"title": "tidy the desktop"},
        },
        "goals": {}, "open_workflows": {},
    }
    ranked = PriorityEngine().rank(state)
    assert ranked[0]["label"].startswith("pay the client invoice")
    assert ranked[0]["reason"] == "OVERDUE"


def test_priority_engine_includes_goals_and_workflows():
    state = {
        "pending_tasks": {},
        "goals": {"g1": {"title": "Grow creator roster", "status": "active",
                         "priority": 0.9, "progress": 0.2,
                         "updated_at": _stamp()}},
        "open_workflows": {"weekly_report": {}},
    }
    kinds = {item["kind"] for item in PriorityEngine().rank(state)}
    assert kinds == {"goal", "workflow"}


def test_strategic_engine_flags_stalled_projects_and_overdue():
    state = {
        "active_projects": {"orion": {"updated_at": _stamp(-24 * 20)}},
        "pending_tasks": {"t": {"title": "old task", "due": _stamp(-48)}},
        "goals": {}, "open_workflows": {}, "research_sessions": {},
        "user_priorities": [],
    }
    recs = " ".join(StrategicEngine().recommendations(state))
    assert "no update in" in recs
    assert "overdue" in recs


def test_strategic_engine_blind_spots():
    spots = " ".join(StrategicEngine().blind_spots({
        "goals": {}, "user_priorities": [],
        "active_projects": {"p": {}}, "pending_tasks": {},
    }))
    assert "No goals" in spots
    assert "priorities" in spots
    assert "next action" in spots


def test_executive_core_focus_composes_queue_and_recommendations():
    cognition = StubCognition({
        "active_projects": {}, "open_workflows": {}, "research_sessions": {},
        "user_priorities": ["ship phase 3"],
        "pending_tasks": {"t": {"title": "urgent deck review", "due": _stamp(3)}},
        "goals": {},
    })
    core = ExecutiveCore(StubBus(), cognition)
    result = asyncio.run(core.focus())
    assert result.ok
    assert "Priority queue:" in result.text
    assert "Recommendations:" in result.text


def test_executive_core_challenge_requires_a_decision():
    core = ExecutiveCore(StubBus(), StubCognition())
    result = asyncio.run(core.challenge(""))
    assert not result.ok


# ── EvidenceEngine ────────────────────────────────────────────────────────────

@pytest.fixture()
def evidence(tmp_path):
    engine = EvidenceEngine(StubBus(), path=tmp_path / "evidence.db")
    yield engine
    engine.close()


def test_score_confidence_hedges_and_strength():
    assert score_confidence("It may possibly improve results") < 0.5
    assert score_confidence("The study measured a 40% lift in the trial") > 0.7


def test_record_and_read_claims_with_provenance(evidence):
    claim_id = evidence.record_claim(
        "tiktok cpm", "UK TikTok CPMs average 4 pounds for cold traffic",
        source="research:2026-07-17/notes/01.md", url="https://example.org")
    assert claim_id > 0
    claims = evidence.claims_for("tiktok cpm")
    assert len(claims) == 1
    assert claims[0]["source"].startswith("research:")
    record = evidence.provenance(claim_id)
    assert record is not None and record["url"] == "https://example.org"


def test_duplicate_claims_refresh_not_duplicate(evidence):
    first = evidence.record_claim("t", "Short hooks lift retention rates", "a")
    second = evidence.record_claim("t", "Short hooks lift retention rates", "b")
    assert first == second
    assert len(evidence.claims_for("t")) == 1


def test_corroborating_claims_raise_confidence(evidence):
    evidence.record_claim(
        "posting", "Daily posting on TikTok drives channel growth",
        source="a", confidence=0.6)
    evidence.record_claim(
        "posting", "Posting daily on TikTok is key for growth of a channel",
        source="b", confidence=0.6)
    confidences = [c["confidence"] for c in evidence.claims_for("posting")]
    assert max(confidences) >= 0.65


def test_contradiction_by_polarity_detected(evidence):
    evidence.record_claim(
        "hooks", "Short hooks increase viewer retention on TikTok", "a")
    evidence.record_claim(
        "hooks", "Short hooks do not increase viewer retention on TikTok", "b")
    conflicts = evidence.detect_contradictions("hooks")
    assert conflicts and conflicts[0]["kind"] == "polarity"


def test_contradiction_by_figures_detected(evidence):
    evidence.record_claim("cpm", "TikTok CPM averages 5 dollars in the UK", "a")
    evidence.record_claim("cpm", "TikTok CPM averages 9 dollars in the UK", "b")
    conflicts = evidence.detect_contradictions("cpm")
    assert conflicts and conflicts[0]["kind"] == "figures"


def test_evidence_stats(evidence):
    evidence.record_claim("a", "Creators respond well to clear briefs", "s")
    stats = evidence.stats()
    assert stats["claims"] == 1 and stats["topics"] == 1


# ── ResearchDirector ──────────────────────────────────────────────────────────

class StubResearch:
    def __init__(self):
        self._active = {}
        self.on_complete = None
        self.started = []

    def start_research(self, topic, minutes=30.0):
        self.started.append((topic, minutes))
        return ToolResult(f"researching {topic}")


@pytest.fixture()
def director(tmp_path):
    evidence_engine = EvidenceEngine(StubBus(), path=tmp_path / "ev.db")
    d = ResearchDirector(StubBus(), StubResearch(), StubCognition(),
                         evidence_engine)
    yield d
    evidence_engine.close()


def test_queue_topic_starts_immediately_when_idle(director):
    result = asyncio.run(director.queue_topic("ai operating systems", 15))
    assert result.ok and "underway" in result.text
    assert director.research.started == [("ai operating systems", 15.0)]
    sessions = director.cognition.state["research_sessions"]
    assert list(sessions.values())[0]["status"] == "running"


def test_queue_topic_waits_when_runner_busy(director):
    class BusyTask:
        @staticmethod
        def done():
            return False
    director.research._active = {"task": BusyTask()}
    result = asyncio.run(director.queue_topic("second topic"))
    assert result.ok and "agenda" in result.text
    assert director.research.started == []


def test_completion_harvests_and_advances(director, tmp_path):
    asyncio.run(director.queue_topic("topic one", 10))
    asyncio.run(director.queue_topic("topic two", 10))
    folder = tmp_path / "run"
    (folder / "notes").mkdir(parents=True)
    (folder / "SUMMARY.md").write_text(
        "TikTok Shop commissions are around 5 percent for new sellers. "
        "Creator content with strong hooks will outperform polished adverts.",
        encoding="utf-8")
    asyncio.run(director._on_run_complete("topic one", folder))
    sessions = director.cognition.state["research_sessions"]
    assert sessions["topic-one"]["status"] == "complete"
    assert sessions["topic-one"]["claims"] >= 1
    # The next queued topic started automatically.
    assert ("topic two", 10.0) in director.research.started
    findings = asyncio.run(director.findings("topic one"))
    assert "Findings" in findings.text


def test_resume_pending_requeues_interrupted_runs(director):
    director.cognition.upsert_research_session("interrupted", {"status": "running"})
    started = asyncio.run(director.resume_pending())
    assert started == "interrupted"
    assert ("interrupted", 20.0) in director.research.started


def test_agenda_and_reviewed(director):
    asyncio.run(director.queue_topic("alpha"))
    agenda = asyncio.run(director.agenda())
    assert "alpha" in agenda.text
    result = asyncio.run(director.mark_reviewed("alpha"))
    assert result.ok
    assert director.cognition.state["research_sessions"]["alpha"]["reviewed"] is True


# ── SystemRegistries ──────────────────────────────────────────────────────────

class StubDispatcher:
    @staticmethod
    def handler_table():
        return {"research": None, "skill": None, "creator_intel": None}


def test_registries_capabilities_and_search():
    registries = SystemRegistries()
    count = registries.capabilities.register_from_dispatcher(
        StubDispatcher(),
        [{"name": "skill", "description": "installable skill packages"}])
    assert count == 3
    assert registries.capabilities.has("research")
    assert registries.capabilities.search("skill") == ["skill"]
    report = registries.report("skill")
    assert "skill" in report.text


def test_registries_modules_and_features():
    registries = SystemRegistries()
    registries.modules.register("evidence", object(), "claims")
    registries.modules.register("missing", None, "absent service")
    registries.features.register("phase3", "the OS layer")
    assert registries.modules.available() == ["evidence"]
    assert registries.features.has("phase3")
    report = registries.report()
    assert "1/2 modules" in report.text
    assert "missing" in report.text


# ── DynamicBriefingEngine ─────────────────────────────────────────────────────

def test_event_monitor_keeps_terminal_events_only():
    monitor = EventMonitor()
    monitor.observe("research", {"topic": "x", "status": "running"})
    monitor.observe("research", {"topic": "x", "status": "done"})
    monitor.observe("workflow", {"workflow": "wf", "status": "complete"})
    monitor.observe("irrelevant_channel", {"status": "done"})
    events = monitor.drain()
    assert len(events) == 2
    assert monitor.events == []


def test_priority_monitor_flags_overdue_and_due_soon():
    state = {"pending_tasks": {
        "a": {"title": "send contract", "due": _stamp(-2)},
        "b": {"title": "call creator", "due": _stamp(3)},
        "c": {"title": "someday item"},
    }}
    pressing = PriorityMonitor().due_soon(state)
    assert any("OVERDUE" in p for p in pressing)
    assert any("due in" in p for p in pressing)
    assert len(pressing) == 2


def test_context_monitor_reports_delta():
    cognition = StubCognition()
    monitor = ContextMonitor(cognition)
    assert monitor.delta()["first"] is True
    cognition.state["pending_tasks"]["t1"] = {"title": "new"}
    delta = monitor.delta()
    assert delta["tasks_added"] == 1 and delta["tasks_cleared"] == 0


def _briefing_engine(state=None):
    return DynamicBriefingEngine(StubBus(), StubCognition(state))


def test_brief_includes_away_section_after_events():
    engine = _briefing_engine()
    engine.observe_event("research", {"topic": "ai systems", "status": "done"})
    result = asyncio.run(engine.brief("evening"))
    assert result.ok
    assert "While you were away:" in result.text
    assert "ai systems" in result.text
    # Events are drained by the brief.
    assert asyncio.run(engine.brief("evening")).text.count("away") == 0


def test_brief_adapts_to_period_and_deadlines():
    engine = _briefing_engine({
        "active_projects": {}, "goals": {}, "open_workflows": {},
        "research_sessions": {}, "user_priorities": ["phase 3"],
        "pending_tasks": {"t": {"title": "review edit", "due": _stamp(2)}},
    })
    morning = asyncio.run(engine.brief("morning"))
    assert "Morning briefing" in morning.text
    assert "Standing priorities" in morning.text
    assert "Deadline watch:" in morning.text


def test_check_triggers_quiet_and_urgent():
    quiet = asyncio.run(_briefing_engine().check_triggers())
    assert "Nothing needs your attention" in quiet.text
    engine = _briefing_engine({
        "active_projects": {}, "goals": {}, "open_workflows": {},
        "research_sessions": {}, "user_priorities": [],
        "pending_tasks": {"t": {"title": "overdue thing", "due": _stamp(-1)}},
    })
    urgent = asyncio.run(engine.check_triggers())
    assert "Time-sensitive:" in urgent.text
