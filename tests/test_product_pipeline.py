"""
Product intelligence pipeline tests — CTAAnalyzer, CompetitorAnalysisAgent,
TrendTracker and the full ProductIntelligencePipeline through the suite.

All offline: stub bus, temporary paths, no provider router.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.creator_intel import (CTAAnalyzer, CompetitorAnalysisAgent,
                                      CreatorIntelSuite, HookAnalyzer,
                                      TrendTracker)


class _Signal:
    def __init__(self, sink, name):
        self._sink = sink
        self._name = name

    def emit(self, *payload):
        self._sink.append((self._name, payload))


class StubBus:
    def __init__(self):
        self.emitted = []

    def __getattr__(self, name):
        sig = _Signal(self.emitted, name)
        object.__setattr__(self, name, sig)
        return sig


class StubMemory:
    def __init__(self):
        self.saved = []

    def save(self, category, key, value, **kwargs):
        self.saved.append((category, key, value))
        return key


@pytest.fixture()
def suite(tmp_path):
    return CreatorIntelSuite(StubBus(), memory=StubMemory(),
                             creators_path=tmp_path / "creators.json")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── CTA analyser ──────────────────────────────────────────────────────────────

def test_cta_scoring_rewards_single_specific_action():
    ctas = CTAAnalyzer()
    strong, _ = ctas.score("Comment MORE and grab yours — link in bio today")
    weak, weak_notes = ctas.score(
        "Thanks so much for watching everyone, see you in the next "
        "one hopefully")
    assert strong > weak
    assert any("No action verb" in n for n in weak_notes)
    improved = ctas.improve("", "the posture corrector")
    assert len(improved) >= 3
    assert any("posture corrector" in c for c in improved)


def test_cta_penalises_competing_actions():
    ctas = CTAAnalyzer()
    single, _ = ctas.score("Follow for part two")
    crowded, notes = ctas.score(
        "Follow, comment, share, save this and shop the link")
    assert single > crowded
    assert any("competing actions" in n for n in notes)


# ── competitor teardown ───────────────────────────────────────────────────────

def test_competitor_teardown_finds_gaps():
    agent = CompetitorAnalysisAgent(HookAnalyzer(), CTAAnalyzer())
    lines = agent.analyse(
        "Stop scrolling — the truth about your desk chair. It ruins your "
        "back. I tested proof for 30 days with results. Follow for more.")
    text = "\n".join(lines)
    assert "hook scores" in text
    assert "CTA" in text
    assert "triggers they lean on" in text
    assert agent.analyse("")[0].startswith("Paste the competitor")


# ── trend tracker ─────────────────────────────────────────────────────────────

def test_trend_tracker_accumulates_and_reports(tmp_path):
    tracker = TrendTracker(tmp_path / "trends.json")
    assert "No trend data" in tracker.report()[0]
    for score, triggers in ((6.0, ["curiosity"]), (7.5, ["curiosity", "proof"]),
                            (8.0, ["proof"])):
        tracker.record("product_intel", "posture corrector",
                       {"hook_score": score, "triggers": triggers})
    report = "\n".join(tracker.report())
    assert "3 recorded" in report
    assert "curiosity" in report and "proof" in report
    assert "Hook quality" in report


# ── the full pipeline ─────────────────────────────────────────────────────────

def test_pipeline_produces_complete_package(suite):
    result = suite.pipeline.run(
        "LED posture corrector",
        competitor="Stop buying posture braces before you watch this — "
                   "I tested proof. Shop the link in bio now.",
        script="Hey guys welcome back to my channel. Today I review a "
               "posture thing. It is okay. Bye.")
    report = result["report"]
    for section in ("PRODUCT INTELLIGENCE", "Competitor teardown",
                    "SUPPLIED SCRIPT", "IMPROVED HOOKS", "IMPROVED CTAs",
                    "CREATOR BRIEF", "TESTING & SCALING"):
        assert section in report
    assert result["hook_score"] is not None
    # The run compounded into trend history.
    assert "1 recorded" in "\n".join(suite.trends.report())


def test_pipeline_via_tool_surface_stores_memory(suite):
    result = _run(suite.handle({"action": "pipeline",
                                "product": "collapsible water bottle"}))
    assert result.ok
    assert "CREATOR BRIEF" in result.text
    assert suite.memory.saved                     # stored to long-term memory
    category, key, _value = suite.memory.saved[0]
    assert category == "KNOWLEDGE"
    assert "collapsible water bottle" in key


def test_new_tool_actions_route(suite):
    cta = _run(suite.handle({"action": "cta",
                             "cta": "Grab yours from the link in bio today"}))
    assert cta.ok and "CTA score" in cta.text
    teardown = _run(suite.handle({"action": "competitor",
                                  "competitor": "POV: you found the secret "
                                                "desk setup. Follow now."}))
    assert teardown.ok and "Competitor teardown" in teardown.text
    trends = _run(suite.handle({"action": "trends"}))
    assert trends.ok
    missing = _run(suite.handle({"action": "pipeline", "product": "  "}))
    assert not missing.ok
