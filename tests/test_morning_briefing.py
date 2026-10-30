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


def test_a_briefing_composed_during_the_offer_is_used_on_yes():
    # 27 Oct 2026: "Yes, go on" -> six seconds of silence while every feed
    # was fetched. The offer now composes it in the background.
    import types
    from orion_core.live_worker import GenAILiveWorker

    class _FakeBriefing:
        def __init__(self):
            self.calls: list[str] = []

        async def compose_source_material(self, period: str = "general") -> str:
            self.calls.append(period)
            return f"{period} briefing"

    class _StopEvent:
        def is_set(self) -> bool:
            return True

    async def run():
        worker = types.SimpleNamespace(
            dispatcher=types.SimpleNamespace(briefing=_FakeBriefing()), bus=_StubBus(),
            connected=False, stop_event=_StopEvent())
        GenAILiveWorker._prefetch_briefing(worker)
        period = worker._briefing_prefetch[0]
        await asyncio.sleep(0)
        await GenAILiveWorker._deliver_briefing(worker, wait_for_connection=0.0, period=period)
        # Composed once - by the prefetch - and the hold is spent.
        assert worker.dispatcher.briefing.calls == [period]
        assert worker._briefing_prefetch is None
        # Another period does not take a stale composition.
        GenAILiveWorker._prefetch_briefing(worker)
        other = "general" if period == "morning" else "morning"
        assert GenAILiveWorker._take_prefetched_briefing(worker, other) is None

    asyncio.run(run())


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
    from orion_core.voice_variety import BANK
    # The offer is phrased afresh each time (voice_variety), never a script.
    assert len(sent) == 1 and sent[0].startswith("Good morning, sir. ")
    assert sent[0][len("Good morning, sir. "):] in BANK["briefing_offer"]
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


def test_a_second_briefing_request_joins_the_one_already_on_its_way(monkeypatch):
    """3 of 10 sessions heard the briefing twice: every call spawned its own
    delivery. While one is composing or being spoken, another joins it."""
    import types
    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    monkeypatch.setattr(MorningBriefingService, "greeting_period",
                        staticmethod(lambda moment=None: "evening"))
    deliveries: list[str] = []

    async def scenario():
        spoken = asyncio.Event()

        async def deliver(period: str) -> None:
            deliveries.append(period)
            await spoken.wait()                 # composing, then speaking

        d = types.SimpleNamespace(on_briefing_request=deliver)
        first = await ProductivityDispatchMixin.morning_briefing(d, {})
        await asyncio.sleep(0)
        assert ProductivityDispatchMixin.briefing_in_flight(d)
        second = await ProductivityDispatchMixin.morning_briefing(d, {})
        third = await ProductivityDispatchMixin.morning_briefing(d, {"period": "morning"})
        await asyncio.sleep(0)
        assert deliveries == ["general"], "one delivery, however often it is asked for"
        spoken.set()
        await asyncio.sleep(0.01)
        assert not ProductivityDispatchMixin.briefing_in_flight(d)
        again = await ProductivityDispatchMixin.morning_briefing(d, {})
        await asyncio.sleep(0)
        spoken.set()
        await asyncio.sleep(0.01)
        return first, second, third, again

    first, second, third, again = asyncio.run(scenario())
    assert first.ok and first.receipt["status"] == "delivering"
    assert "Do NOT compose, summarise or narrate a briefing yourself" in first.text
    for joined in (second, third):
        assert joined.ok and joined.receipt["status"] == "already_delivering"
        assert "already on its way" in joined.text and "narrate" in joined.text
    assert again.receipt["status"] == "delivering" and deliveries == ["general", "general"], \
        "a later request after the first has finished is a new briefing"
