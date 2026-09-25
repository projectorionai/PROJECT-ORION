"""
Tests for the command-deck token graph's data path.

* TokenUsageLedger.timeseries_by — one grouped query returning per-provider
  bucketed series (the graph's source), filter- and window-aware.
* GenAILiveWorker._record_live_usage — Gemini Live turns land in the ledger
  (they previously never did), streaming-tagged, deduped, and never raising
  when no ledger is attached.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.token_usage import TokenUsageLedger, UsageRecord


def _ledger() -> TokenUsageLedger:
    return TokenUsageLedger(":memory:")


def _record(ledger, request_id, provider, tokens, at=None):
    ledger.record(UsageRecord(
        request_id=request_id, provider=provider, model="m",
        input_tokens=tokens // 2, output_tokens=tokens - tokens // 2,
        at=at or time.time(),
    ))


def test_timeseries_by_provider_groups_and_orders():
    ledger = _ledger()
    now = time.time()
    _record(ledger, "a1", "gemini", 100, at=now - 7200)
    _record(ledger, "a2", "gemini", 300, at=now)
    _record(ledger, "b1", "groq", 50, at=now)
    series = ledger.timeseries_by("provider", bucket="hour")
    assert set(series) == {"gemini", "groq"}
    assert len(series["gemini"]) == 2                     # two distinct hours
    assert series["gemini"][-1]["total_tokens"] == 300
    assert series["groq"][0]["total_tokens"] == 50


def test_timeseries_by_respects_since_filter():
    ledger = _ledger()
    now = time.time()
    _record(ledger, "old", "gemini", 999, at=now - 90_000)   # > 24 h ago
    _record(ledger, "new", "gemini", 10, at=now)
    series = ledger.timeseries_by("provider", filters={"since": now - 86_400})
    assert sum(b["total_tokens"] for b in series["gemini"]) == 10


def test_timeseries_by_rejects_unknown_dimension():
    try:
        _ledger().timeseries_by("api_key")
        raised = False
    except ValueError:
        raised = True
    assert raised


class _Router:
    def __init__(self, ledger):
        self.token_ledger = ledger

    def active_key(self, profile):
        return profile.api_key


class _Profile:
    name = "gemini"
    model = "models/gemini-live"
    api_key = "AIza" + "x" * 35


class _Usage:
    prompt_token_count = 120
    response_token_count = 340
    total_token_count = 460


def _bare_worker(ledger):
    from orion_core.live_worker import GenAILiveWorker
    worker = object.__new__(GenAILiveWorker)
    worker.router = _Router(ledger)
    worker.active_live_provider = _Profile()
    worker.active_live_model = "models/gemini-live"
    return worker


def test_live_usage_lands_in_ledger_streaming_tagged():
    ledger = _ledger()
    worker = _bare_worker(ledger)
    worker._record_live_usage(_Usage())
    summary = ledger.summary(filters={"provider": "gemini"})
    assert summary["requests"] == 1
    assert summary["input_tokens"] == 120
    assert summary["output_tokens"] == 340
    assert summary["total_tokens"] == 460
    by_task = ledger.by_dimension("task")
    assert by_task[0]["key"] == "live_audio"


def test_live_usage_without_ledger_or_profile_is_silent():
    worker = _bare_worker(None)
    worker.router.token_ledger = None
    worker._record_live_usage(_Usage())                   # must not raise
    worker2 = _bare_worker(_ledger())
    worker2.active_live_provider = None
    worker2._record_live_usage(_Usage())                  # must not raise
    worker2._record_live_usage(None)                      # must not raise
