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


def test_scholarly_apis_keep_provenance_and_do_not_verify_claims(monkeypatch):
    import json

    from orion_core import scholarly
    called = []

    async def fake(source, query, limit):
        called.append((source, query, limit))
        if source == "crossref":
            return {"status": "ok", "message": {"items": [{"title": ["A <i>synthetic</i> study"],
                "DOI": "10.5555/example", "author": [{"given": "A", "family": "Reader"}],
                "published": {"date-parts": [[2026]]}, "is-referenced-by-count": 2}]}}
        return {"resultList": {"result": [{"title": "A synthetic study", "doi": "10.5555/example",
            "id": "123", "source": "MED", "pubYear": "2026"}]}}

    monkeypatch.setattr(scholarly, "_request", fake)
    result = asyncio.run(scholarly.search({"query": " synthetic ", "limit": 3}))
    report = json.loads(result.text)
    assert result.ok and not report["partial"]
    assert sorted(called) == [("crossref", "synthetic", 3), ("europepmc", "synthetic", 3)]
    [record] = report["records"]
    assert record["sources"] == ["crossref", "europepmc"]
    assert record["url"] == "https://doi.org/10.5555/example"
    assert record["title"] == "A synthetic study"
    assert record["publication_date"] == "2026"  # never manufacture month/day
    assert not record["full_text_read"] and not record["claim_verified"]
    assert report["observed_at"].endswith("+00:00")


def test_scholarly_combined_search_does_not_hide_the_second_index(monkeypatch):
    import json

    from orion_core import scholarly

    async def fake(source, *_):
        if source == "crossref":
            return {"status": "ok", "message": {"items": [
                {"title": [f"Crossref {i}"], "DOI": f"10.5555/crossref{i}"} for i in range(3)]}}
        return {"resultList": {"result": [
            {"title": f"Europe PMC {i}", "doi": f"10.5555/europepmc{i}"} for i in range(3)]}}

    monkeypatch.setattr(scholarly, "_request", fake)
    records = json.loads(asyncio.run(scholarly.search({"query": "synthetic", "limit": 3})).text)["records"]
    assert len(records) == 3
    assert [r["sources"] for r in records] == [["crossref"], ["europepmc"], ["crossref"]]


def test_scholarly_duplicate_doi_preserves_conflicting_index_observations(monkeypatch):
    import json

    from orion_core import scholarly

    async def fake(source, *_):
        if source == "crossref":
            return {"status": "ok", "message": {"items": [{"title": ["Original title"],
                "DOI": "10.5555/example", "published": {"date-parts": [[2025]]},
                "author": [{"given": "A", "family": "Reader"}], "is-referenced-by-count": 2}]}}
        return {"resultList": {"result": [{"title": "Revised title", "doi": "10.5555/EXAMPLE",
            "authorString": "B Reader", "pubYear": "2026", "citedByCount": 7}]}}

    monkeypatch.setattr(scholarly, "_request", fake)
    [record] = json.loads(asyncio.run(scholarly.search({"query": "synthetic"})).text)["records"]
    assert record["sources"] == ["crossref", "europepmc"]
    observations = record["source_records"]
    assert observations["crossref"]["title"] == "Original title"
    assert observations["europepmc"]["title"] == "Revised title"
    assert observations["crossref"]["publication_date"] == "2025"
    assert observations["europepmc"]["publication_date"] == "2026"
    assert observations["crossref"]["authors"] == ["A Reader"]
    assert observations["europepmc"]["authors"] == ["B Reader"]
    assert observations["crossref"]["citation_count"] == 2
    assert observations["europepmc"]["citation_count"] == 7
    assert not record["claim_verified"]


@pytest.mark.parametrize("args", [{"query": ""}, {"query": "x" * 501},
    {"query": "x", "source": "https://example.com"}, {"query": "x", "limit": True},
    {"query": "x", "limit": 11}, {"query": "x", "limit": "2"}])
def test_scholarly_rejects_invalid_requests_before_transport(args, monkeypatch):
    from orion_core import scholarly

    async def forbidden(*_):
        raise AssertionError("Invalid request reached transport")
    monkeypatch.setattr(scholarly, "_request", forbidden)
    assert not asyncio.run(scholarly.search(args)).ok


def test_scholarly_empty_partial_and_failed_sources_are_distinct(monkeypatch):
    import json

    from orion_core import scholarly

    async def fake(source, *_):
        if source == "crossref":
            raise ValueError("HTTP 429")
        return {"resultList": {"result": []}}
    monkeypatch.setattr(scholarly, "_request", fake)
    partial = asyncio.run(scholarly.search({"query": "synthetic"}))
    report = json.loads(partial.text)
    assert partial.ok and report["partial"] and report["records"] == []
    assert report["sources"][0] == {"source": "crossref", "ok": False, "error": "HTTP 429"}
    failed = asyncio.run(scholarly.search({"query": "synthetic", "source": "crossref"}))
    assert not failed.ok
    assert not json.loads(failed.text)["partial"]
    empty = asyncio.run(scholarly.search({"query": "synthetic", "source": "europepmc"}))
    assert empty.ok and not json.loads(empty.text)["partial"]


@pytest.mark.parametrize("payload", [[], {}, {"resultList": {"result": "broken"}},
    {"resultList": {"result": ["broken"]}}])
def test_scholarly_malformed_response_is_not_reported_as_no_results(payload, monkeypatch):
    from orion_core import scholarly

    async def fake(*_):
        return payload
    monkeypatch.setattr(scholarly, "_request", fake)
    assert not asyncio.run(scholarly.search({"query": "synthetic", "source": "europepmc"})).ok


def test_scholarly_invalid_date_cannot_discard_other_bibliographic_metadata(monkeypatch):
    import json

    from orion_core import scholarly

    async def fake(*_):
        return {"status": "ok", "message": {"items": [{"title": ["Synthetic"],
            "DOI": "10.5555/example", "published": {"date-parts": [[10**100]]}}]}}
    monkeypatch.setattr(scholarly, "_request", fake)
    result = asyncio.run(scholarly.search({"query": "synthetic", "source": "crossref"}))
    assert result.ok
    [record] = json.loads(result.text)["records"]
    assert record["title"] == "Synthetic" and record["publication_date"] == ""


@pytest.mark.parametrize("status,chunks,ok", [(200, [b'{"resultList":{"result":[]}}'], True),
    (429, [], False), (302, [], False), (200, [b"x" * 30, b"x" * 30], False)])
def test_scholarly_transport_is_fixed_bounded_and_does_not_follow_redirects(status, chunks, ok, monkeypatch):
    from orion_core import scholarly

    class Fake:
        content_length = None
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            pass
        def get(self, endpoint, **options):
            assert endpoint == scholarly.ENDPOINTS["europepmc"]
            assert options["allow_redirects"] is False
            assert options["params"]["query"] == "https://untrusted.example/path"
            self.status = status
            self.content = self
            return self
        async def iter_chunked(self, _):
            for chunk in chunks:
                yield chunk

    def session(**options):
        assert options["timeout"].total == 8
        return Fake()
    monkeypatch.setattr(scholarly.aiohttp, "ClientSession", session)
    monkeypatch.setattr(scholarly, "MAX_BYTES", 50)
    assert asyncio.run(scholarly.search({"query": "https://untrusted.example/path", "source": "europepmc"})).ok is ok


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
