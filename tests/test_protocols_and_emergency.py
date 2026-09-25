"""
Protocols — time-aware, bounded, non-repeating; and the emergency protocol,
which must never invent an emergency.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from orion_core.data import ToolResult
from orion_core.emergency import (
    CATEGORIES, Confidence, EmergencyProtocol, Severity, classify,
)
from orion_core.protocols import BUILTIN_PROTOCOLS, ProtocolManager
from orion_core.time_service import TIME


class _Sig:
    def __init__(self):
        self.messages = []

    def emit(self, *a):
        self.messages.append(a[0] if len(a) == 1 else a)

    def connect(self, *a):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


class _Memory:
    def remember(self, *a, **k):
        pass

    def recall(self, *a, **k):
        return []


def _manager(dispatch=None):
    manager = ProtocolManager(_Bus(), _Memory())
    if dispatch is None:
        async def dispatch(tool, args):
            return ToolResult(f"{tool} ok")
    manager.bind_dispatch(dispatch)
    manager.MIN_RERUN_S = 0.0        # rate limiting is tested explicitly
    return manager


# ── the four protocols the brief asks for, plus the originals ────────────────

def test_all_four_named_protocols_exist():
    for name in ("morning", "afternoon", "evening", "emergency"):
        assert name in BUILTIN_PROTOCOLS, name


def test_the_original_protocols_were_not_removed():
    """The brief: preserve and improve the existing four."""
    for name in ("morning", "focus", "wind_down", "situation_report"):
        assert name in BUILTIN_PROTOCOLS, name


def test_every_protocol_has_a_description():
    for name, spec in BUILTIN_PROTOCOLS.items():
        assert spec.get("description"), name


def test_every_step_names_a_tool():
    for name, spec in BUILTIN_PROTOCOLS.items():
        for step in spec.get("steps", []):
            assert step.get("tool"), f"{name} has a step with no tool"


# ── time awareness (§7: must not assume it is morning) ───────────────────────

def test_running_the_morning_protocol_at_night_says_so():
    manager = _manager()
    TIME.set_clock(lambda: datetime(2026, 8, 8, 23, 10, tzinfo=timezone.utc))
    try:
        context = manager.time_context(BUILTIN_PROTOCOLS["morning"])
    finally:
        TIME.set_clock(None)
    assert "night" in context
    assert "not morning" in context


def test_running_the_morning_protocol_in_the_morning_adds_no_caveat():
    manager = _manager()
    TIME.set_clock(lambda: datetime(2026, 8, 8, 8, 0, tzinfo=timezone.utc))
    try:
        assert manager.time_context(BUILTIN_PROTOCOLS["morning"]) == ""
    finally:
        TIME.set_clock(None)


def test_the_caveat_still_runs_the_protocol():
    """A wrong-time protocol is caveated, never refused."""
    manager = _manager()
    TIME.set_clock(lambda: datetime(2026, 8, 8, 23, 10, tzinfo=timezone.utc))
    try:
        result = asyncio.run(manager.run("morning"))
    finally:
        TIME.set_clock(None)
    assert result.ok
    assert "not morning" in result.text
    assert "morning_briefing" in result.text


def test_afternoon_and_evening_declare_their_own_periods():
    assert BUILTIN_PROTOCOLS["afternoon"]["period"] == "afternoon"
    assert BUILTIN_PROTOCOLS["evening"]["period"] == "evening"
    assert BUILTIN_PROTOCOLS["evening"]["steps"][0]["args"]["period"] == "evening"


# ── bounded execution ─────────────────────────────────────────────────────────

def test_a_hanging_step_times_out_without_taking_the_protocol_down():
    async def dispatch(tool, args):
        if tool == "system_health":
            await asyncio.sleep(30)
        return ToolResult(f"{tool} ok")

    manager = _manager(dispatch)
    manager.STEP_TIMEOUT_S = 0.05
    result = asyncio.run(manager.run("morning"))
    assert "TIMEOUT" in result.text
    assert "morning_briefing: ok" in result.text


def test_a_failing_step_does_not_stop_later_steps():
    async def dispatch(tool, args):
        if tool == "morning_briefing":
            raise RuntimeError("feed down")
        return ToolResult(f"{tool} ok")

    manager = _manager(dispatch)
    result = asyncio.run(manager.run("morning"))
    assert "error" in result.text
    assert "system_health: ok" in result.text


def test_rate_limiting_suppresses_an_immediate_rerun():
    manager = _manager()
    manager.MIN_RERUN_S = 60.0
    assert asyncio.run(manager.run("focus")).ok
    second = asyncio.run(manager.run("focus"))
    assert second.ok is False
    assert "ignoring a repeat" in second.text


def test_an_unwired_native_protocol_is_refused_not_faked():
    """A protocol that cannot run must say so, not report success."""
    manager = _manager()
    result = asyncio.run(manager.run("emergency"))
    assert result.ok is False
    assert "not wired up" in result.text
    assert "will not pretend" in result.text


def test_a_bound_native_protocol_runs_its_own_pipeline():
    manager = _manager()

    async def handler():
        return {"spoken": "Emergency check complete. Nothing found."}

    manager.bind_native("emergency", handler)
    result = asyncio.run(manager.run("emergency"))
    assert result.ok
    assert "Nothing found" in result.text


# ── emergency classification ──────────────────────────────────────────────────

UK = "Birmingham, England, United Kingdom"


def test_ordinary_news_is_not_an_emergency():
    for headline in [
        "Council debates bin collection schedule",
        "Local team wins away fixture",
        "New cafe opens on the high street",
        "Share prices close slightly higher",
    ]:
        assert classify(headline, source="BBC", url="https://bbc.co.uk/x",
                        locality=UK) is None, headline


def test_an_official_source_is_confirmed():
    finding = classify("Met Office issues red warning for severe weather",
                       source="Met Office",
                       url="https://www.metoffice.gov.uk/warnings",
                       locality=UK)
    assert finding.confidence is Confidence.CONFIRMED
    assert finding.category == "severe_weather"


def test_a_news_outlet_is_reported_but_unverified():
    finding = classify("Major delays after signalling fault",
                       source="BBC", url="https://bbc.co.uk/news/x", locality=UK)
    assert finding.confidence is Confidence.REPORTED


def test_an_unattributed_claim_is_speculative_and_downgraded():
    official = classify("Red warning issued for storm",
                        url="https://metoffice.gov.uk/x", locality=UK)
    rumour = classify("Red warning issued for storm", locality=UK)
    assert rumour.confidence is Confidence.SPECULATIVE
    assert rumour.severity < official.severity, (
        "an unattributed claim must never read as severe as a confirmed one")


def test_a_finding_states_its_confidence_out_loud():
    finding = classify("Gas leak forces evacuation and cordon",
                       source="Police", url="https://police.uk/x", locality=UK)
    spoken = finding.spoken()
    assert "confirmed" in spoken
    assert "What to do:" in spoken
    assert "What to avoid:" in spoken


def test_an_event_elsewhere_does_not_claim_to_affect_the_user():
    finding = classify("Severe flood warning issued for Aberdeen",
                       source="Met Office", url="https://metoffice.gov.uk/x",
                       locality=UK)
    assert finding.affects_user is False
    assert "does not name your area" in finding.why_affects


def test_an_event_in_the_users_town_does_affect_them():
    finding = classify("Severe flood warning issued for Birmingham",
                       source="Met Office", url="https://metoffice.gov.uk/x",
                       locality=UK)
    assert finding.affects_user is True
    assert "Birmingham" in finding.why_affects


def test_unknown_location_is_admitted_not_assumed():
    finding = classify("Storm warning issued", source="Met Office",
                       url="https://metoffice.gov.uk/x", locality="")
    assert finding.affects_user is False
    assert "do not know your location" in finding.why_affects


def test_security_outranks_transport():
    security = classify("Major incident declared, evacuation under way",
                        url="https://police.uk/x", locality=UK)
    transport = classify("Major delays on all lines suspended",
                         url="https://nationalrail.co.uk/x", locality=UK)
    assert security.severity > transport.severity


# ── the emergency run ─────────────────────────────────────────────────────────

def _protocol(**kw):
    return EmergencyProtocol(_Bus(), **kw)


def test_no_sources_means_no_emergencies_stated_honestly():
    result = asyncio.run(_protocol().run(force=True))
    assert result["count"] == 0
    assert result["notify"] is False
    assert "no active emergencies" in result["spoken"]


def test_it_never_invents_a_finding():
    result = asyncio.run(_protocol().run(force=True))
    assert result["findings"] == []


def test_rate_limiting_blocks_a_second_immediate_run():
    protocol = _protocol()
    asyncio.run(protocol.run(force=True))
    second = asyncio.run(protocol.run())
    assert "next check available" in second["note"]


def test_duplicate_findings_are_suppressed():
    protocol = _protocol()
    finding = classify("Red warning for severe weather in Birmingham",
                       url="https://metoffice.gov.uk/x", locality=UK)
    first = protocol._deduplicate([finding])
    second = protocol._deduplicate([finding])
    assert len(first) == 1
    assert second == [], "the same storm must not be announced twice"


def test_a_reworded_duplicate_still_collapses():
    protocol = _protocol()
    a = classify("Red warning for severe weather in Birmingham",
                 url="https://metoffice.gov.uk/x", locality=UK)
    b = classify("Birmingham severe weather red warning",
                 url="https://metoffice.gov.uk/x", locality=UK)
    assert len(protocol._deduplicate([a])) == 1
    assert protocol._deduplicate([b]) == []


def test_a_failing_source_does_not_fail_the_protocol():
    protocol = _protocol()

    async def broken(locality):
        raise RuntimeError("feed exploded")

    protocol._news_source = broken
    result = asyncio.run(protocol.run(force=True))
    assert result["count"] == 0
    assert "no active emergencies" in result["spoken"]


def test_a_hanging_source_is_timed_out():
    protocol = _protocol()
    protocol.SOURCE_TIMEOUT_S = 0.05

    async def slow(locality):
        await asyncio.sleep(5)
        return []

    protocol._news_source = slow
    result = asyncio.run(protocol.run(force=True))
    assert result["count"] == 0


def test_only_severe_findings_that_reach_the_user_interrupt_them():
    protocol = _protocol()
    far_away = classify("Red warning for severe weather in Aberdeen",
                        url="https://metoffice.gov.uk/x", locality=UK)
    assert protocol.should_notify([far_away]) is False

    nearby = classify("Red warning for severe weather in Birmingham",
                      url="https://metoffice.gov.uk/x", locality=UK)
    assert protocol.should_notify([nearby]) is True


def test_findings_are_ordered_most_severe_first():
    protocol = _protocol()

    async def source(locality):
        return [
            classify("Major delays on all lines suspended",
                     url="https://nationalrail.co.uk/x", locality=UK),
            classify("Major incident declared with evacuation",
                     url="https://police.uk/x", locality=UK),
        ]

    protocol._news_source = source
    result = asyncio.run(protocol.run(force=True))
    assert result["count"] == 2
    assert result["findings"][0]["category"] == "security"


def test_an_offline_orion_subsystem_is_itself_an_emergency():
    class Health:
        def snapshot(self):
            return {"Speaker": {"status": "OFFLINE", "detail": "no stream"}}

    protocol = _protocol(health=Health())
    result = asyncio.run(protocol.run(force=True))
    assert result["count"] == 1
    assert result["findings"][0]["category"] == "system"
    assert result["findings"][0]["affects_user"] is True


def test_categories_all_carry_advice_and_avoidance():
    for name, spec in CATEGORIES.items():
        assert spec["advice"], name
        assert spec["avoid"], name
