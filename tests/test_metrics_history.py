"""
Tests for the rolling metrics history + per-tool usage audit
(Improvement Pass, Priority 3.1 and 3.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.metrics_history import MetricsHistory
from orion_core.telemetry import Telemetry


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


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def history(tmp_path):
    clock = _Clock()
    store = MetricsHistory(tmp_path / "hist.db", interval_s=60.0,
                           retention_days=30.0, clock=clock)
    store._clock_obj = clock                    # expose for tests
    yield store
    store.close()


# ── MetricsHistory: sampling + throttle ───────────────────────────────────────

def test_record_writes_series(history):
    assert history.record({"c.tool.calls": 5, "g.queue": 2})
    assert history.trend("c.tool.calls")["last"] == 5.0


def test_sample_is_throttled(history):
    clock = history._clock_obj
    assert history.sample({"c.x": 1}) is True         # first always writes
    clock.advance(30)
    assert history.sample({"c.x": 2}) is False        # inside the 60 s window
    clock.advance(31)
    assert history.sample({"c.x": 3}) is True          # window elapsed
    series = history.series("c.x")
    assert [v for _t, v in series] == [1.0, 3.0]       # the throttled 2 dropped


def test_record_ignores_non_numeric(history):
    assert history.record({"c.x": "not a number", "c.y": None}) is False
    assert history.record({"c.x": 3, "c.bad": "skip"}) is True
    assert history.trend("c.x")["last"] == 3.0
    assert history.trend("c.bad") is None


def test_empty_series_is_noop(history):
    assert history.record({}) is False


# ── trends + series ───────────────────────────────────────────────────────────

def test_trend_summarises_growth(history):
    clock = history._clock_obj
    for value in (100, 120, 150, 210):
        history.record({"g.rss_mb": value})
        clock.advance(60)
    trend = history.trend("g.rss_mb")
    assert trend["first"] == 100.0 and trend["last"] == 210.0
    assert trend["min"] == 100.0 and trend["max"] == 210.0
    assert trend["delta"] == 110.0 and trend["count"] == 4.0


def test_series_returns_chronological_order(history):
    clock = history._clock_obj
    for value in (1, 2, 3):
        history.record({"c.n": value})
        clock.advance(60)
    assert [v for _t, v in history.series("c.n")] == [1.0, 2.0, 3.0]


def test_trend_window_filters_old_points(history):
    clock = history._clock_obj
    history.record({"g.v": 1})
    clock.advance(10_000)
    history.record({"g.v": 9})
    recent = history.trend("g.v", window_s=100.0)
    assert recent["count"] == 1.0 and recent["last"] == 9.0


def test_unknown_metric_trend_is_none(history):
    assert history.trend("does.not.exist") is None


# ── retention / pruning ───────────────────────────────────────────────────────

def test_prune_drops_rows_beyond_retention(tmp_path):
    clock = _Clock()
    store = MetricsHistory(tmp_path / "h.db", interval_s=1.0,
                           retention_days=1.0, clock=clock)
    store.record({"g.v": 1})                    # day 0
    clock.advance(2 * 86400)                      # +2 days
    store.record({"g.v": 2})
    removed = store.prune()
    assert removed == 1
    assert [v for _t, v in store.series("g.v")] == [2.0]
    store.close()


def test_sample_prunes_opportunistically(tmp_path):
    clock = _Clock()
    store = MetricsHistory(tmp_path / "h.db", interval_s=1.0,
                           retention_days=1.0, clock=clock)
    store.sample({"g.v": 1})
    clock.advance(2 * 86400)
    store.sample({"g.v": 2})                       # writes AND prunes the old row
    assert [v for _t, v in store.series("g.v")] == [2.0]
    store.close()


# ── tool usage persistence ────────────────────────────────────────────────────

def test_tool_usage_upsert_and_rank(history):
    history.record_tool_usage({
        "vision_analyse": {"calls": 10, "failures": 1},
        "open_app": {"calls": 3, "failures": 0},
    })
    history.record_tool_usage({"open_app": {"calls": 20, "failures": 2}})  # upsert
    usage = history.tool_usage()
    assert usage[0]["tool"] == "open_app" and usage[0]["calls"] == 20
    assert usage[1]["tool"] == "vision_analyse"
    assert history.tool_usage(top=1) == usage[:1]


def test_summary_reports_counts(history):
    history.record({"c.a": 1, "c.b": 2})
    history.record_tool_usage({"open_app": {"calls": 1}})
    summary = history.summary()
    assert summary["samples"] == 2 and summary["metrics"] == 2
    assert summary["tools_tracked"] == 1


def test_persists_across_reopen(tmp_path):
    store = MetricsHistory(tmp_path / "h.db")
    store.record({"g.v": 42})
    store.record_tool_usage({"open_app": {"calls": 7}})
    store.close()
    reopened = MetricsHistory(tmp_path / "h.db")
    assert reopened.trend("g.v")["last"] == 42.0
    assert reopened.tool_usage()[0]["calls"] == 7
    reopened.close()


# ── Telemetry integration (Priority 3.4 usage audit) ──────────────────────────

def test_record_tool_call_increments_counters():
    tel = Telemetry(_StubBus())
    tel.record_tool_call("vision_analyse", ok=True)
    tel.record_tool_call("vision_analyse", ok=True)
    tel.record_tool_call("vision_analyse", ok=False)
    tel.record_tool_call("open_app", ok=True)
    counters = tel.metrics.snapshot()["counters"]
    assert counters["tool.calls"] == 4
    assert counters["tool.failures"] == 1
    assert counters["tool.vision_analyse.calls"] == 3
    assert counters["tool.vision_analyse.failures"] == 1


def test_tool_usage_ranks_tools():
    tel = Telemetry(_StubBus())
    for _ in range(5):
        tel.record_tool_call("vision_analyse", ok=True)
    for _ in range(2):
        tel.record_tool_call("open_app", ok=True)
    usage = tel.tool_usage()
    assert usage[0] == {"tool": "vision_analyse", "calls": 5, "failures": 0}
    assert usage[1]["tool"] == "open_app"
    assert tel.tool_usage(top=1) == usage[:1]


def test_history_series_excludes_per_tool_counters():
    tel = Telemetry(_StubBus())
    tel.record_tool_call("open_app", ok=True)
    tel.metrics.gauge("audio.playback_latency_ms", 42.0)
    series = tel._history_series()
    assert "c.tool.calls" in series                 # aggregate kept
    assert "g.audio.playback_latency_ms" in series
    assert not any(k.startswith("c.tool.open_app") for k in series)  # per-tool excluded


def test_enable_history_and_tick(tmp_path):
    tel = Telemetry(_StubBus())
    assert tel.enable_history(tmp_path / "h.db") is not None
    tel.record_tool_call("open_app", ok=True)
    tel.metrics.gauge("audio.playback_latency_ms", 30.0)
    assert tel.tick_history() is True               # first tick writes
    assert tel.tick_history() is False              # throttled immediately after
    assert tel.history.trend("g.audio.playback_latency_ms")["last"] == 30.0
    assert tel.history.tool_usage()[0]["tool"] == "open_app"
    tel.history.close()


def test_tick_history_noop_without_store():
    tel = Telemetry(_StubBus())
    assert tel.tick_history() is False              # history disabled → safe no-op
