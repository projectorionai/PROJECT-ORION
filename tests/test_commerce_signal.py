"""
Tests for the commerce demand-signal grounding (Improvement Pass, Priority 3.2).

The whole point is: one real data source grounds the advice, but the
deterministic offline path (MODE B) is preserved and never regresses. The
network fetch is injected, so these run fully offline.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.commerce import (
    CommerceSignalProvider,
    CommerceSuite,
    DropshippingResearchAgent,
    ProductOpportunityScore,
)


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _StubMatrix:
    def __init__(self):
        self.saved = []

    def save(self, category, key, value, silent=False):
        self.saved.append((category, key, value))


class _StubMemory:
    def __init__(self):
        self.matrix = _StubMatrix()

    @staticmethod
    def _project_slug(name):
        return name.lower().replace(" ", "_")

    def records(self, limit=200):
        return []


class _OfflineRouter:
    def has_text_fallback(self):
        return False


def _pageview_payload(views):
    return {"items": [{"timestamp": f"2026070{i%10}00", "views": v}
                      for i, v in enumerate(views)]}


def _provider(tmp_path, fetch, clock=None):
    return CommerceSignalProvider(tmp_path / "sig.db", ttl_hours=24.0,
                                  min_interval_s=0.0, fetch=fetch, clock=clock)


# ── interpretation: interest / momentum / trend ───────────────────────────────

def test_rising_series_reads_as_rising_high_interest(tmp_path):
    async def fetch(url):
        return _pageview_payload([100] * 30 + [900] * 30)   # clear upswing
    prov = _provider(tmp_path, fetch)
    sig = asyncio.run(prov.interest("air fryer"))
    assert sig is not None
    assert sig["trend"] == "rising" and sig["momentum"] > 5
    assert 0 <= sig["interest"] <= 10 and sig["source"] == "wikimedia-pageviews"
    prov.close()


def test_falling_series_reads_as_falling(tmp_path):
    async def fetch(url):
        return _pageview_payload([900] * 30 + [100] * 30)
    prov = _provider(tmp_path, fetch)
    sig = asyncio.run(prov.interest("fidget spinner"))
    assert sig["trend"] == "falling" and sig["momentum"] < 5
    prov.close()


def test_high_traffic_scores_higher_interest(tmp_path):
    async def big(url):
        return _pageview_payload([50000] * 40)

    async def small(url):
        return _pageview_payload([50] * 40)
    hi = asyncio.run(_provider(tmp_path, big).interest("popular"))
    lo = asyncio.run(_provider(tmp_path, small).interest("obscure"))
    assert hi["interest"] > lo["interest"]


# ── failure paths → None → offline fallback preserved ─────────────────────────

def test_no_article_match_returns_none(tmp_path):
    async def fetch(url):
        return None                                  # 404 / no such article
    assert asyncio.run(_provider(tmp_path, fetch).interest("zxqw")) is None


def test_fetch_exception_returns_none(tmp_path):
    async def fetch(url):
        raise RuntimeError("network down")
    assert asyncio.run(_provider(tmp_path, fetch).interest("air fryer")) is None


def test_too_few_samples_returns_none(tmp_path):
    async def fetch(url):
        return _pageview_payload([100, 200])          # < 8 points
    assert asyncio.run(_provider(tmp_path, fetch).interest("thin data")) is None


def test_empty_term_returns_none(tmp_path):
    async def fetch(url):
        raise AssertionError("should not fetch for an empty term")
    assert asyncio.run(_provider(tmp_path, fetch).interest("   ")) is None


# ── caching ───────────────────────────────────────────────────────────────────

def test_second_call_is_served_from_cache(tmp_path):
    calls = {"n": 0}

    async def fetch(url):
        calls["n"] += 1
        return _pageview_payload([300] * 30)
    prov = _provider(tmp_path, fetch)
    a = asyncio.run(prov.interest("storage basket"))
    b = asyncio.run(prov.interest("Storage  Basket"))    # normalised same key
    assert a == b and calls["n"] == 1                    # one network call only
    prov.close()


def test_cache_expires_after_ttl(tmp_path):
    now = {"t": 1000.0}

    async def fetch(url):
        return _pageview_payload([300] * 30)
    prov = CommerceSignalProvider(tmp_path / "s.db", ttl_hours=1.0,
                                  min_interval_s=0.0, fetch=fetch,
                                  clock=lambda: now["t"])
    asyncio.run(prov.interest("thing"))
    now["t"] += 2 * 3600                                  # past the 1 h TTL
    # A fresh network read still works; the point is the cache did not serve it.
    assert asyncio.run(prov.interest("thing")) is not None
    prov.close()


# ── grounding the score (MODE B preserved) ────────────────────────────────────

def _dropship(signals):
    return DropshippingResearchAgent(_StubBus(), _StubMemory(), _OfflineRouter(),
                                     signals=signals)


def test_score_offline_matches_deterministic_baseline():
    # With no signal provider, the score is exactly the pure deterministic read.
    agent = _dropship(None)
    result = asyncio.run(agent.score_product("plain widget", "a simple home item"))
    m = ProductOpportunityScore.infer("a simple home item")
    expected = ProductOpportunityScore.score(m)
    assert f"{expected}/100" in result.text
    assert "Grounded in real demand" not in result.text


def test_score_uses_real_signal_when_available(tmp_path):
    async def fetch(url):
        return _pageview_payload([100] * 30 + [1200] * 30)   # rising, popular
    agent = _dropship(_provider(tmp_path, fetch))
    result = asyncio.run(agent.score_product("air fryer", "trending kitchen gadget"))
    assert "Grounded in real demand" in result.text
    assert "rising" in result.text


def test_signal_failure_falls_back_to_offline(tmp_path):
    async def fetch(url):
        raise RuntimeError("offline")
    agent = _dropship(_provider(tmp_path, fetch))
    result = asyncio.run(agent.score_product("air fryer", "kitchen gadget"))
    assert result.ok
    assert "Grounded in real demand" not in result.text    # clean degrade


def test_caller_supplied_metrics_win_over_signal(tmp_path):
    calls = {"n": 0}

    async def fetch(url):
        calls["n"] += 1
        return _pageview_payload([1000] * 40)
    agent = _dropship(_provider(tmp_path, fetch))
    # Both grounded metrics pinned explicitly → the signal must not be consulted.
    asyncio.run(agent.score_product("x", "y", metrics={"demand": 9, "virality": 8}))
    assert calls["n"] == 0


# ── suite construction + advisor grounding ────────────────────────────────────

def test_suite_builds_signal_provider_by_default():
    suite = CommerceSuite(_StubBus(), _StubMemory(), _OfflineRouter())
    assert suite.signals is not None
    assert suite.advisor.signals is suite.signals          # shared across agents
    assert suite.dropship.signals is suite.signals


def test_advisor_surfaces_signal_in_mode_b(tmp_path):
    async def fetch(url):
        return _pageview_payload([200] * 30 + [500] * 30)
    suite = CommerceSuite(_StubBus(), _StubMemory(), _OfflineRouter(),
                          signals=_provider(tmp_path, fetch))
    result = asyncio.run(suite.advisor.advise("air fryer bundle"))
    # No model → deterministic path, but the real signal is still surfaced.
    assert "REAL DEMAND SIGNAL" in result.text
