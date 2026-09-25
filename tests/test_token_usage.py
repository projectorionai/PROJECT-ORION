"""
Tests for the token-usage ledger and model registry (Section 6).

Covers: non-streamed and streamed responses, retries and reconnect/failover
dedup, failed requests, cached and reasoning tokens, estimate→authoritative
reconciliation, unknown context limits (never fabricated), cost labelling, key
masking, and aggregation by dimension and time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import model_registry
from orion_core.provider_capabilities import NormalizedUsage
from orion_core.token_usage import (
    TokenUsageLedger,
    UsageRecord,
    estimate_tokens,
    mask_key,
)


def _ledger():
    return TokenUsageLedger(db_path=":memory:")


def test_key_is_masked_never_stored_raw():
    alias = mask_key("AIzaSuperSecretKey1234")
    assert "SuperSecret" not in alias          # the key body is never present
    assert "AIzaSuperSecretKey" not in alias
    assert "1234" in alias                      # only the last 4 chars are shown
    assert mask_key("local") == "local"


def test_non_streamed_record_and_summary():
    led = _ledger()
    assert led.record(UsageRecord("r1", "openai", "gpt-4o-mini", key_alias="local",
                                  input_tokens=1000, output_tokens=200))
    s = led.summary()
    assert s["requests"] == 1
    assert s["input_tokens"] == 1000 and s["output_tokens"] == 200
    assert s["total_tokens"] == 1200


def test_duplicate_request_id_is_deduped():
    led = _ledger()
    r = UsageRecord("same-id", "openai", "gpt-4o-mini", input_tokens=100, output_tokens=50)
    assert led.record(r) is True
    # Streaming completion / retry / reconnect re-emits the SAME request id.
    assert led.record(r) is False
    assert led.record(r) is False
    assert led.summary()["requests"] == 1


def test_streamed_response_counts_once_after_reconcile():
    led = _ledger()
    # Streaming: record an estimate at request start...
    led.record(UsageRecord("stream-1", "anthropic", "claude-sonnet-5",
                           input_tokens=estimate_tokens("a" * 4000), estimated=True,
                           tokenizer="heuristic", tokenizer_version="1.0", streaming=True))
    # ...then reconcile with authoritative usage when the stream finishes.
    led.reconcile("stream-1", NormalizedUsage(input_tokens=980, output_tokens=310,
                                              cached_input_tokens=200, total_tokens=1290))
    s = led.summary()
    assert s["requests"] == 1
    assert s["input_tokens"] == 980 and s["output_tokens"] == 310
    assert s["cached_input_tokens"] == 200


def test_failover_records_each_provider_once():
    led = _ledger()
    # A turn that failed over cloud→local produces two DISTINCT physical calls.
    led.record(UsageRecord("call-a", "openai", "gpt-4o-mini", input_tokens=500, output_tokens=0, failed=True))
    led.mark_failed("call-a")
    led.record(UsageRecord("call-b", "local_ollama", "llama3.1", input_tokens=500, output_tokens=120))
    s = led.summary()
    assert s["requests"] == 2
    assert s["failed_requests"] == 1


def test_cached_and_reasoning_tokens_tracked_distinctly():
    led = _ledger()
    led.record(UsageRecord("r", "openai", "o3", input_tokens=400, output_tokens=600,
                           cached_input_tokens=100, reasoning_tokens=350))
    s = led.summary()
    assert s["cached_input_tokens"] == 100
    assert s["reasoning_tokens"] == 350


def test_unknown_model_context_is_not_fabricated():
    led = _ledger()
    led.record(UsageRecord("r", "mystery", "some-unknown-model-x", input_tokens=10, output_tokens=5))
    report = led.context_report("some-unknown-model-x", tokens_in_context=10)
    assert report["context_window"] is None            # Not reported, never guessed
    assert report["context_utilisation_pct"] is None
    # A known model DOES report an authoritative window.
    known = led.context_report("gpt-4o-mini", tokens_in_context=64000)
    assert known["context_window"] == 128_000
    assert known["context_utilisation_pct"] == 50.0
    assert known["remaining_context"] == 64_000


def test_context_window_is_not_confused_with_quota():
    report = TokenUsageLedger(":memory:").context_report("claude-sonnet-5", 1000)
    assert "not account" in report["note"].lower()
    assert report["context_window"] == 200_000


def test_cost_is_labelled_and_absent_when_unknown():
    led = _ledger()
    led.record(UsageRecord("priced", "openai", "gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000))
    led.record(UsageRecord("unpriced", "xai", "grok-4", input_tokens=1000, output_tokens=1000))
    rows = {r["key"]: r for r in led.by_dimension("model")}
    assert rows["gpt-4o-mini"]["estimated_cost_usd"] is not None
    assert rows["grok-4"]["estimated_cost_usd"] is None   # not recorded → not fabricated


def test_aggregation_by_dimension_and_time():
    led = _ledger()
    led.record(UsageRecord("a", "openai", "gpt-4o-mini", session_id="s1", input_tokens=100, output_tokens=50, at=1_700_000_000))
    led.record(UsageRecord("b", "openai", "gpt-4o-mini", session_id="s2", input_tokens=200, output_tokens=60, at=1_700_000_000))
    led.record(UsageRecord("c", "anthropic", "claude-sonnet-5", session_id="s1", input_tokens=300, output_tokens=70, at=1_700_100_000))
    by_provider = {r["key"]: r for r in led.by_dimension("provider")}
    assert by_provider["openai"]["requests"] == 2
    by_session = {r["key"]: r for r in led.by_dimension("session_id")}
    assert by_session["s1"]["requests"] == 2
    days = led.timeseries("day")
    assert len(days) == 2


def test_estimate_tokens_reasonable():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100


def test_registry_reports_versioned_metadata():
    assert model_registry.REGISTRY_VERSION
    assert model_registry.context_window("gemini-2.5-flash-native-audio") == 1_048_576
    assert model_registry.context_window("models/gemini-2.0-flash-live-001") == 1_048_576
    assert model_registry.context_window("totally-made-up") is None
