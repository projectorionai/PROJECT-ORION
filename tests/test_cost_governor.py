"""
The cost governor (CAP-02) — free tier first, and don't cook the CPU.

    "cheapest and best models ... estimated time of usage remaining ... if CPU
     usage is high, can we divert to GPU?"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.cost_governor import CostGovernor, Pressure  # noqa: E402
from orion_core.usage_runway import UsageRunway  # noqa: E402


class FakeLedger:
    """A ledger with just the two methods UsageRunway reads."""

    def __init__(self, total: int = 0, rate_rows=None):
        self._total = total
        self._rows = rate_rows or []

    def summary(self, filters=None):
        return {"total_tokens": self._total}

    def timeseries(self, bucket="hour", filters=None):
        return self._rows


def make_governor(tmp_path, *, used, budget, cpu=10.0, gpu=True, rate_rows=None):
    ledger = FakeLedger(total=used, rate_rows=rate_rows)
    runway = UsageRunway(ledger, path=tmp_path / "runway.json")
    runway.set_budget("test", int(budget))
    return CostGovernor(runway, cpu_percent=lambda: cpu, gpu_available=lambda: gpu)


# ── pressure tiers ───────────────────────────────────────────────────────────

def test_low_usage_is_easy_and_holds_the_model(tmp_path):
    gov = make_governor(tmp_path, used=100, budget=1000)
    d = gov.assess("test")
    assert d.pressure is Pressure.EASY
    assert not d.should_switch
    assert d.suggested_model == ""
    assert "No need to switch" in d.message


def test_over_half_is_watch_but_still_no_switch(tmp_path):
    gov = make_governor(tmp_path, used=600, budget=1000)
    d = gov.assess("test")
    assert d.pressure is Pressure.WATCH
    assert not d.should_switch


def test_tight_recommends_the_cheapest_paid_model(tmp_path):
    gov = make_governor(tmp_path, used=850, budget=1000)
    d = gov.assess("test")
    assert d.pressure is Pressure.TIGHT
    assert d.should_switch
    assert "deepseek" in d.suggested_model.lower()


def test_exhausted_recommends_a_free_model(tmp_path):
    gov = make_governor(tmp_path, used=1000, budget=1000)
    d = gov.assess("test")
    assert d.pressure is Pressure.EXHAUSTED
    assert d.should_switch
    # the suggested model must actually be a free-tier one
    from orion_core import model_advisor
    free_names = {m.name for m in model_advisor.CATALOGUE if m.free_tier}
    assert d.suggested_model in free_names
    assert "free" in d.suggested_reason.lower()


def test_high_burn_rate_triggers_tight_before_the_wall(tmp_path):
    # only 10% of the budget used, but burning fast enough that <30 min remains
    rows = [{"at": time.time(), "tokens": 3000}]
    gov = make_governor(tmp_path, used=100, budget=1000, rate_rows=rows)
    d = gov.assess("test")
    assert d.minutes_left is not None and d.minutes_left <= 30
    assert d.pressure is Pressure.TIGHT
    assert d.should_switch


# ── CPU → GPU advice ─────────────────────────────────────────────────────────

def test_hot_cpu_local_with_gpu_advises_offload(tmp_path):
    gov = make_governor(tmp_path, used=100, budget=1000, cpu=95.0, gpu=True)
    d = gov.assess("test", local=True)
    assert "GPU" in d.gpu_advice
    assert "offload" in d.gpu_advice.lower()


def test_hot_cpu_local_without_gpu_advises_cloud(tmp_path):
    gov = make_governor(tmp_path, used=100, budget=1000, cpu=95.0, gpu=False)
    d = gov.assess("test", local=True)
    assert "free cloud" in d.gpu_advice.lower()


def test_cool_cpu_local_is_reassuring(tmp_path):
    gov = make_governor(tmp_path, used=100, budget=1000, cpu=20.0)
    d = gov.assess("test", local=True)
    assert "comfortable" in d.gpu_advice.lower()


def test_cloud_work_gives_no_gpu_nag_when_cpu_is_calm(tmp_path):
    gov = make_governor(tmp_path, used=100, budget=1000, cpu=20.0)
    d = gov.assess("test", local=False)
    assert d.gpu_advice == ""


def test_cloud_work_disowns_a_hot_cpu(tmp_path):
    gov = make_governor(tmp_path, used=100, budget=1000, cpu=95.0)
    d = gov.assess("test", local=False)
    assert "cloud" in d.gpu_advice.lower()
    assert "not me" in d.gpu_advice.lower()


# ── the decision object ──────────────────────────────────────────────────────

def test_decision_serialises(tmp_path):
    gov = make_governor(tmp_path, used=500, budget=1000)
    d = gov.assess("test")
    data = d.as_dict()
    assert data["provider"] == "test"
    assert 0.0 <= data["fraction_used"] <= 1.0
    assert "pressure" in data


def test_message_always_mentions_usage_and_pressure(tmp_path):
    gov = make_governor(tmp_path, used=500, budget=1000)
    d = gov.assess("test")
    assert "used today" in d.message
    assert "Pressure" in d.message


# ── the tool wiring ──────────────────────────────────────────────────────────

def test_ai_mode_governor_action_is_wired():
    import inspect
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    src = inspect.getsource(ProductivityDispatchMixin.ai_mode)
    assert "cost_governor" in src or "CostGovernor" in src
    assert "governor" in src.lower()


def test_schema_advertises_the_governor_action():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "ai_mode")
    assert "governor" in tool["description"].lower() or "runway" in tool["description"].lower()
