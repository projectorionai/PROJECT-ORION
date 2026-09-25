"""
Tests for the morning-briefing fix: a "morning" period genuinely widens the
news freshness floor and frames the header for an overnight catch-up, instead
of running byte-for-byte the same generic composition every time — and the
period argument actually threads from the protocol/tool call through to
compose_source_material() (previously it would have been silently dropped by
the on_briefing_request fire-and-forget branch).

Hermetic: no real network, no real news/market/crypto fetches.  The SQLite
news-signature cache is redirected to a tmp file so tests never touch the
real config/news_cache.db.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import briefing as briefing_module
from orion_core.briefing import MorningBriefingService
from orion_core.data import ToolResult


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubService:
    available = False


def _service(tmp_path, monkeypatch) -> MorningBriefingService:
    monkeypatch.setattr(briefing_module, "NEWS_CACHE_PATH", tmp_path / "news_cache.db")
    monkeypatch.setattr(briefing_module, "BRIEFING_STATE_PATH", tmp_path / "briefing_state.json")
    return MorningBriefingService(_StubBus(), _StubService(), _StubService())


# ── greeting_period ──────────────────────────────────────────────────────────

def test_greeting_period_boundaries():
    """The shared band table (orion_core.time_service), not a local copy.

    The evening boundary moved from 17:00 to 18:00. This service used 17:00
    while local_brain used 18:00, so between five and six o'clock the briefing
    header and the offline brain named different parts of the same day. One
    table now decides for everybody; 18:00 is the boundary.
    """
    assert MorningBriefingService.greeting_period(datetime(2026, 1, 1, 6, 0)) == "morning"
    assert MorningBriefingService.greeting_period(datetime(2026, 1, 1, 11, 59)) == "morning"
    assert MorningBriefingService.greeting_period(datetime(2026, 1, 1, 12, 0)) == "afternoon"
    assert MorningBriefingService.greeting_period(datetime(2026, 1, 1, 17, 59)) == "afternoon"
    assert MorningBriefingService.greeting_period(datetime(2026, 1, 1, 18, 0)) == "evening"
    assert MorningBriefingService.greeting_period(datetime(2026, 1, 1, 23, 30)) == "evening"
    # Half past midnight is NIGHT, but you still greet with "good evening" —
    # this is the pair that produced "Good morning" at 00:31.
    assert MorningBriefingService.greeting_period(datetime(2026, 1, 1, 0, 31)) == "evening"


# ── _news_report ladder selection ───────────────────────────────────────────

def test_news_report_climbs_the_default_ladder_when_no_ladder_given(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    seen_windows: list[float] = []

    async def _fake_gather(session, window_h):
        seen_windows.append(window_h)
        return []

    monkeypatch.setattr(service, "_gather_candidates", _fake_gather)
    asyncio.run(service._news_report(session=None))
    assert tuple(seen_windows) == service.FRESHNESS_LADDER


def test_news_report_uses_morning_ladder_when_given_one(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    seen_windows: list[float] = []

    async def _fake_gather(session, window_h):
        seen_windows.append(window_h)
        return []

    monkeypatch.setattr(service, "_gather_candidates", _fake_gather)
    asyncio.run(service._news_report(session=None, ladder=service.MORNING_FRESHNESS_LADDER))
    assert tuple(seen_windows) == service.MORNING_FRESHNESS_LADDER
    assert 1.5 not in seen_windows   # the narrow rung that always over-satisfied MIN_STORIES


# ── compose_source_material: period changes header + ladder ────────────────

def _stub_out_sections(service, monkeypatch, captured: dict) -> None:
    async def _news(session, ladder=None):
        captured["ladder"] = ladder
        return "News: stub."

    async def _empty(*_a, **_k):
        return ""

    monkeypatch.setattr(service, "_news_report", _news)
    monkeypatch.setattr(service, "_market_section", _empty)
    monkeypatch.setattr(service, "_crypto_section", _empty)
    monkeypatch.setattr(service, "_calendar_section", _empty)
    monkeypatch.setattr(service, "_tasks_section", _empty)
    monkeypatch.setattr(service, "_email_section", _empty)


def test_the_header_reads_orions_clock_not_the_machines(tmp_path, monkeypatch):
    """ORION's clock and the machine's can be in different zones (a UTC cloud
    node, ORION on UK time). The header's part of the day and spoken time must
    come from ORION's clock, like the greeting does: at 18:11 in London the
    header once said "afternoon" and "five eleven" on a UTC machine."""
    from datetime import datetime, timezone, timedelta

    from orion_core.time_service import TIME

    service = _service(tmp_path, monkeypatch)
    _stub_out_sections(service, monkeypatch, {})
    fixed = datetime(2026, 9, 25, 18, 11, tzinfo=timezone(timedelta(hours=1)))
    TIME.set_clock(lambda: fixed)
    try:
        result = asyncio.run(service.compose_source_material(period="general"))
        assert "Your evening briefing" in result
        assert "six eleven" in result
        assert service.already_briefed_today() is True
    finally:
        TIME.set_clock(None)


def test_compose_source_material_general_period_uses_generic_header(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    captured: dict = {}
    _stub_out_sections(service, monkeypatch, captured)
    result = asyncio.run(service.compose_source_material(period="general"))
    # Named for the part of the day it actually is — never a fixed "morning".
    assert f"Your {service.greeting_period()} briefing" in result
    assert captured["ladder"] is None


def test_compose_source_material_morning_period_uses_morning_header_and_ladder(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    captured: dict = {}
    _stub_out_sections(service, monkeypatch, captured)
    result = asyncio.run(service.compose_source_material(period="morning"))
    assert "Intelligence briefing" not in result
    assert any(g in result for g in ("Good morning", "Good afternoon", "Good evening"))
    assert captured["ladder"] == service.MORNING_FRESHNESS_LADDER


# ── dispatch_productivity.morning_briefing: period threads through ─────────

def test_morning_briefing_tool_passes_period_to_on_briefing_request(tmp_path, monkeypatch):
    import types
    from orion_core.briefing import MorningBriefingService
    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    monkeypatch.setattr(MorningBriefingService, "greeting_period",
                        staticmethod(lambda moment=None: "morning"))

    seen: list[str] = []

    async def _on_request(period: str) -> None:
        seen.append(period)

    d = types.SimpleNamespace()
    d.on_briefing_request = _on_request
    result = asyncio.run(ProductivityDispatchMixin.morning_briefing(d, {"period": "morning"}))
    asyncio.run(asyncio.sleep(0))   # let the created task run
    assert seen == ["morning"]
    assert result.ok


def test_morning_briefing_tool_calls_compose_source_material_directly_without_hook(tmp_path, monkeypatch):
    import types
    from orion_core.briefing import MorningBriefingService
    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    monkeypatch.setattr(MorningBriefingService, "greeting_period",
                        staticmethod(lambda moment=None: "morning"))

    class _FakeBriefing:
        def __init__(self):
            self.calls: list[str] = []

        async def compose_source_material(self, period: str = "general") -> str:
            self.calls.append(period)
            return "briefing text"

    d = types.SimpleNamespace()
    d.on_briefing_request = None
    d.briefing = _FakeBriefing()
    result = asyncio.run(ProductivityDispatchMixin.morning_briefing(d, {"period": "morning"}))
    assert d.briefing.calls == ["morning"]
    assert result.text == "briefing text"


def test_morning_briefing_tool_defaults_to_general_period(tmp_path):
    import types
    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    class _FakeBriefing:
        def __init__(self):
            self.calls: list[str] = []

        async def compose_source_material(self, period: str = "general") -> str:
            self.calls.append(period)
            return "briefing text"

    d = types.SimpleNamespace()
    d.on_briefing_request = None
    d.briefing = _FakeBriefing()
    asyncio.run(ProductivityDispatchMixin.morning_briefing(d, {}))
    assert d.briefing.calls == ["general"]


# ── live_worker._deliver_briefing: period reaches compose_source_material ──

def test_deliver_briefing_passes_period_to_compose_source_material(tmp_path):
    import types
    from orion_core.live_worker import GenAILiveWorker

    class _FakeBriefing:
        def __init__(self):
            self.calls: list[str] = []

        async def compose_source_material(self, period: str = "general") -> str:
            self.calls.append(period)
            return "briefing text"

    class _StopEvent:
        def is_set(self) -> bool:
            return True   # exit right after compose_source_material — no further faking needed

    worker = types.SimpleNamespace()
    worker.dispatcher = types.SimpleNamespace(briefing=_FakeBriefing())
    worker.bus = _StubBus()
    worker.connected = False
    worker.stop_event = _StopEvent()

    asyncio.run(GenAILiveWorker._deliver_briefing(worker, wait_for_connection=0.0, period="morning"))
    assert worker.dispatcher.briefing.calls == ["morning"]


# ── "already briefed today" bookkeeping ─────────────────────────────────────
# So a restart or a new same-day session doesn't re-offer a briefing already
# delivered a few hours ago unless something has genuinely broken since.

def test_not_already_briefed_by_default(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    assert service.already_briefed_today() is False


def test_mark_briefed_sets_already_briefed_today(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.mark_briefed(datetime(2026, 8, 1, 9, 0))
    assert service.already_briefed_today(datetime(2026, 8, 1, 18, 0)) is True


def test_already_briefed_today_is_false_on_a_new_day(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.mark_briefed(datetime(2026, 8, 1, 9, 0))
    assert service.already_briefed_today(datetime(2026, 8, 2, 7, 0)) is False


def test_mark_briefed_persists_across_instances(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    service.mark_briefed(datetime(2026, 8, 1, 9, 0))
    reloaded = _service(tmp_path, monkeypatch)   # simulates a restart
    assert reloaded.already_briefed_today(datetime(2026, 8, 1, 20, 0)) is True


def test_compose_source_material_marks_briefed(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    captured: dict = {}
    _stub_out_sections(service, monkeypatch, captured)
    assert service.already_briefed_today() is False
    asyncio.run(service.compose_source_material(period="general"))
    assert service.already_briefed_today() is True


# ── has_fresh_news_since_last_briefing: read-only, never steals stories ────

def test_has_fresh_news_true_when_unseen_candidates_exist(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    async def _fake_gather(session, window_h):
        return [{"signature": "sig-1", "source_weight": 1.0, "tokens": frozenset({"a"})}]

    monkeypatch.setattr(service, "_gather_candidates", _fake_gather)
    assert asyncio.run(service.has_fresh_news_since_last_briefing()) is True


def test_has_fresh_news_false_when_nothing_new(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    async def _fake_gather(session, window_h):
        return []

    monkeypatch.setattr(service, "_gather_candidates", _fake_gather)
    assert asyncio.run(service.has_fresh_news_since_last_briefing()) is False


def test_has_fresh_news_never_commits_to_the_seen_cache(tmp_path, monkeypatch):
    """The check must be read-only — committing here would make a subsequent
    REAL briefing filter these same stories out as 'already seen'."""
    service = _service(tmp_path, monkeypatch)

    async def _fake_gather(session, window_h):
        return [{"signature": "sig-1", "source_weight": 1.0, "tokens": frozenset({"a"})}]

    monkeypatch.setattr(service, "_gather_candidates", _fake_gather)
    asyncio.run(service.has_fresh_news_since_last_briefing())
    assert service.cache.seen("sig-1") is False


def test_has_fresh_news_false_on_network_failure(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    async def _boom(session, window_h):
        raise RuntimeError("network down")

    monkeypatch.setattr(service, "_gather_candidates", _boom)
    assert asyncio.run(service.has_fresh_news_since_last_briefing()) is False


# ── offer_startup_briefing: SAYS he remembers, rather than going quiet ───────
#
# "ORION seems to not remember if he's given a briefing in the day too, I want
#  him to explicitly remember and say 'It seems you've had your briefing in the
#  past hour'."
#
# He always knew. The startup instruction told him "Do NOT offer or mention the
# briefing", so he said nothing at all — and from the user's side an assistant
# who says nothing is indistinguishable from one who has forgotten.

def test_offer_startup_briefing_says_he_remembers_when_nothing_is_new(tmp_path):
    import types
    from orion_core.live_worker import GenAILiveWorker

    class _FakeBriefing:
        def already_briefed_today(self):
            return True

        async def has_fresh_news_since_last_briefing(self):
            return False

    class _StopEvent:
        def is_set(self):
            return False

    sent: list[str] = []
    worker = types.SimpleNamespace()
    worker.dispatcher = types.SimpleNamespace(briefing=_FakeBriefing())
    worker.bus = _StubBus()
    worker.connected = True
    worker.stop_event = _StopEvent()
    worker.session = None
    worker.awaiting_briefing = False
    worker._greeted_at = 0.0
    worker._refresh_wake_window = lambda: None
    worker._compose_greeting = lambda: _async_return("Good evening, sir.")
    worker._say = lambda text: sent.append(text)

    asyncio.run(GenAILiveWorker.offer_startup_briefing(worker))
    assert len(sent) == 1
    assert "already had your briefing" in sent[0], (
        f"he went quiet instead of saying he remembers: {sent}")
    assert "nothing new" in sent[0]
    assert worker.awaiting_briefing is False   # still never enters the offer/consent flow


def test_offer_startup_briefing_still_asks_when_fresh_news_exists(tmp_path):
    import types
    from orion_core.live_worker import GenAILiveWorker

    class _FakeBriefing:
        def already_briefed_today(self):
            return True

        async def has_fresh_news_since_last_briefing(self):
            return True

    class _StopEvent:
        def is_set(self):
            return False

    sent: list[str] = []
    worker = types.SimpleNamespace()
    worker.dispatcher = types.SimpleNamespace(briefing=_FakeBriefing())
    worker.bus = _StubBus()
    worker.connected = True
    worker.stop_event = _StopEvent()
    worker.session = None
    worker.awaiting_briefing = False
    worker._greeted_at = 0.0
    worker._refresh_wake_window = lambda: None
    worker._compose_greeting = lambda: _async_return("Good evening, sir.")
    worker._say = lambda text: sent.append(text)

    asyncio.run(GenAILiveWorker.offer_startup_briefing(worker))
    # Still asks — but as an UPDATE, not a repeat. Having already been briefed
    # and something having broken since is a specific situation, and a generic
    # "would you like your briefing?" throws away the one useful thing ORION
    # knows: that he checked, and something actually changed.
    assert len(sent) == 1
    assert "something new" in sent[0], f"offered a generic repeat: {sent}"
    assert "had your briefing" in sent[0]
    assert worker.awaiting_briefing is True


def test_offer_startup_briefing_asks_normally_when_not_yet_briefed_today(tmp_path):
    import types
    from orion_core.live_worker import GenAILiveWorker

    class _FakeBriefing:
        def already_briefed_today(self):
            return False

    class _StopEvent:
        def is_set(self):
            return False

    sent: list[str] = []
    worker = types.SimpleNamespace()
    worker.dispatcher = types.SimpleNamespace(briefing=_FakeBriefing())
    worker.bus = _StubBus()
    worker.connected = True
    worker.stop_event = _StopEvent()
    worker.session = None
    worker.awaiting_briefing = False
    worker._greeted_at = 0.0
    worker._refresh_wake_window = lambda: None
    worker._compose_greeting = lambda: _async_return("Good morning, sir.")
    worker._say = lambda text: sent.append(text)

    asyncio.run(GenAILiveWorker.offer_startup_briefing(worker))
    assert sent == ["Good morning, sir. Would you like your briefing?"]
    assert worker.awaiting_briefing is True


async def _async_return(value):
    return value


def test_a_morning_request_in_the_evening_is_the_evening_briefing(monkeypatch):
    """"Here is your morning briefing" at 18:25 — the overnight treatment is
    only honoured in the morning, and the reply names the real part of day."""
    import types
    from orion_core.briefing import MorningBriefingService
    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    monkeypatch.setattr(MorningBriefingService, "greeting_period",
                        staticmethod(lambda moment=None: "evening"))
    seen: list[str] = []

    async def _on_request(period: str) -> None:
        seen.append(period)

    d = types.SimpleNamespace(on_briefing_request=_on_request)
    result = asyncio.run(ProductivityDispatchMixin.morning_briefing(d, {"period": "morning"}))
    asyncio.run(asyncio.sleep(0))
    assert seen == ["general"]
    assert "evening briefing" in result.text
