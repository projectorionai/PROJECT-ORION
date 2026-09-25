"""
Specialist briefing + research extension tests — the ULTRON-pass additions:
research / mission / security / opportunity briefings, SourceValidator
domain tiers, and ResearchDirector.opportunities().

All offline: stub bus, stub cognition, temporary evidence store.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.briefing_engine import DynamicBriefingEngine
from orion_core.evidence import EvidenceEngine
from orion_core.missions import MissionEngine
from orion_core.research_director import ResearchDirector, SourceValidator


class _Signal:
    def __init__(self, sink, name):
        self._sink = sink
        self._name = name

    def emit(self, *payload):
        self._sink.append((self._name, payload))

    def connect(self, *_):
        pass


class StubBus:
    def __init__(self):
        self.emitted = []

    def __getattr__(self, name):
        sig = _Signal(self.emitted, name)
        object.__setattr__(self, name, sig)
        return sig


class StubCognition:
    def __init__(self, state=None):
        self.state = state or {}

    def snapshot(self):
        return dict(self.state)

    def upsert_research_session(self, topic, patch):
        sessions = self.state.setdefault("research_sessions", {})
        record = sessions.setdefault(topic, {"topic": topic})
        record.update(patch)
        return dict(record)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── source validation ─────────────────────────────────────────────────────────

def test_source_validator_domain_tiers():
    score_gov, label_gov = SourceValidator.credibility("https://www.nih.gov/x")
    score_ref, _ = SourceValidator.credibility("https://en.wikipedia.org/wiki/Y")
    score_soc, label_soc = SourceValidator.credibility("https://reddit.com/r/z")
    score_web, _ = SourceValidator.credibility("https://random-blog.example.com")
    assert score_gov > score_ref > score_web > score_soc
    assert label_gov == "primary/institutional"
    assert label_soc == "social/forum"


def test_source_validator_assess_text_mean():
    report = SourceValidator.assess_text(
        "See https://www.nature.com/articles/a and "
        "https://reddit.com/r/science/comments/b for details.")
    assert len(report["sources"]) == 2
    assert 0.35 < report["mean"] < 0.90


def test_validate_sources_tool(tmp_path):
    evidence = EvidenceEngine(path=tmp_path / "evidence.db")
    director = ResearchDirector(StubBus(), None, StubCognition(), evidence)
    result = _run(director.validate_sources(
        "Sources: https://pubmed.ncbi.nlm.nih.gov/123 and "
        "https://www.tiktok.com/@creator/video/9"))
    assert result.ok
    assert "mean credibility" in result.text
    assert "primary/institutional" in result.text
    assert not _run(director.validate_sources("")).ok
    evidence.close()


# ── opportunity scanning ──────────────────────────────────────────────────────

def test_opportunities_surface_contradictions_and_reviews(tmp_path):
    evidence = EvidenceEngine(path=tmp_path / "evidence.db")
    cognition = StubCognition({"research_sessions": {
        "sleep science": {"topic": "sleep science", "status": "complete",
                          "reviewed": False},
    }})
    evidence.record_claim("sleep science",
                          "Adults need eight hours of sleep for recovery",
                          "study-a")
    evidence.record_claim("sleep science",
                          "Adults do not need eight hours of sleep for recovery",
                          "study-b")
    director = ResearchDirector(StubBus(), None, cognition, evidence)
    result = _run(director.opportunities())
    assert "contradiction" in result.text.lower()
    assert "Review completed research" in result.text
    evidence.close()


def test_opportunities_quiet_when_consistent(tmp_path):
    evidence = EvidenceEngine(path=tmp_path / "evidence.db")
    director = ResearchDirector(StubBus(), None, StubCognition(), evidence)
    result = _run(director.opportunities())
    assert "No pressing research opportunities" in result.text
    evidence.close()


# ── specialist briefings ──────────────────────────────────────────────────────

@pytest.fixture()
def engine_kit(tmp_path):
    bus = StubBus()
    cognition = StubCognition({"research_sessions": {
        "hooks": {"topic": "hooks", "status": "running"},
    }})
    evidence = EvidenceEngine(path=tmp_path / "evidence.db")
    director = ResearchDirector(bus, None, cognition, evidence)
    missions = MissionEngine(bus, path=tmp_path / "missions.json")
    briefings = DynamicBriefingEngine(
        bus, cognition, research_director=director, missions=missions)
    yield briefings, missions
    evidence.close()


def test_research_briefing(engine_kit):
    briefings, _missions = engine_kit
    result = _run(briefings.brief("research"))
    assert "Research briefing" in result.text
    assert "Running: hooks" in result.text


def test_mission_briefing_leads_with_current(engine_kit):
    briefings, missions = engine_kit
    missions.add_task("Develop ORION", "Ship the avatar system")
    result = _run(briefings.brief("mission"))
    assert "Mission briefing" in result.text
    assert "▶ Develop ORION" in result.text
    assert "Recommended next move" in result.text


def test_security_briefing_reports_events(engine_kit):
    briefings, _missions = engine_kit
    briefings.observe_event("security", {"status": "alert",
                                         "detail": "port scan detected"})
    result = _run(briefings.brief("security"))
    assert "Security briefing" in result.text
    assert "port scan detected" in result.text
    # With no events at all the perimeter reads quiet.
    calm = _run(DynamicBriefingEngine(StubBus(), StubCognition())
                .brief("security"))
    assert "perimeter quiet" in calm.text


def test_opportunity_briefing_flags_idle_missions(engine_kit):
    briefings, _missions = engine_kit
    result = _run(briefings.brief("opportunity"))
    assert "Opportunity briefing" in result.text
    assert "no queued work" in result.text        # seeded missions are idle


def test_time_of_day_briefs_still_work(engine_kit):
    briefings, _missions = engine_kit
    result = _run(briefings.brief("morning"))
    assert "Morning briefing" in result.text
