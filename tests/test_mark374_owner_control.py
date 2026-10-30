"""Mark XXXVII.4 - what the owner asked for on 26 Oct 2026.

* ORION must be able to drive his own Command Deck and his face when asked -
  and the owner's own "open WhatsApp" / "minimise this" / "forget that" must
  run again (every tool call had been deferred as background work).
* He shows and tours his expressions, reads and searches the deck and presses
  ordinary buttons there - never destructive or approval ones.
* A text fallback for words already being answered, or answered by Live, does
  not act twice; nothing is said after the goodbye.
* He reads more feelings, and answers them with an empathetic face.
* Cute or professional voice and manner, switchable.
* Singing uses his own offline voice when Charon cannot, masters cleanly, and
  sings a copyrighted song only as a short cover snippet.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from orion_core import execution_policy as policy


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


# ── the owner's own turn ─────────────────────────────────────────────────────

@pytest.mark.parametrize("effect", ["application_open", "window_minimise", "window_close",
                                    "media_control", "shell_open_file", "forget_confirmation_dialog",
                                    "file_confirmation_dialog", "orion_restart",
                                    # Mark XXXX.2: "can you just press Start Camera?"
                                    "keyboard_input", "move_cursor", "clipboard_write",
                                    "focus_window", "edit_text_focus"])
def test_the_owners_turn_may_do_what_he_asked_for(effect):
    with policy.conversation_scope("voice"):
        assert policy.require_foreground(effect) is None, effect


@pytest.mark.parametrize("origin", ["job", "workflow", "plugin", "agent"])
def test_the_pointer_and_keyboard_stay_his_alone(origin):
    with policy.background_scope(origin):
        for effect in ("keyboard_input", "move_cursor", "clipboard_write"):
            assert policy.require_foreground(effect) is not None, (origin, effect)


@pytest.mark.parametrize("effect", ["generic_browser_automation", "gesture_input_session",
                                    "unrestricted_development_process",
                                    "plugin custom desktop/process effects"])
def test_the_owners_turn_still_never_injects_input_or_automates(effect):
    with policy.conversation_scope("voice"):
        blocked = policy.require_foreground(effect)
    assert blocked is not None and blocked.receipt["status"] == "requires_foreground"


@pytest.mark.parametrize("origin", ["tool", "job", "workflow", "plugin", "agent"])
def test_background_work_still_cannot_open_apps_or_move_windows(origin):
    with policy.background_scope(origin):
        assert policy.require_foreground("application_open") is not None
        assert policy.require_foreground("window_minimise") is not None
        with policy.conversation_scope("voice"):          # nested inside background work
            assert policy.require_foreground("application_open") is not None


def test_a_cancelled_or_detached_turn_loses_the_owners_permission():
    with policy.conversation_scope("voice") as scope:
        scope.detach()
        assert policy.require_foreground("application_open") is not None
    with policy.conversation_scope("voice") as scope:
        scope.revoke()
        assert policy.require_foreground("application_open") is not None


def test_open_an_app_by_voice_is_no_longer_refused(monkeypatch):
    """The direct check that found it: DesktopAgent.open_app refused every
    conversational call with requires_foreground."""
    from orion_core.agents import DesktopAgent

    agent = DesktopAgent.__new__(DesktopAgent)
    monkeypatch.setattr(DesktopAgent, "_is_url", lambda self, name: True)
    opened = []
    monkeypatch.setattr("orion_core.agents.open_url", lambda url: opened.append(url))
    with policy.conversation_scope("voice"):
        result = agent.open_app("https://example.org")
    assert result.ok and opened == ["https://example.org"]
    with policy.background_scope("job"):
        assert agent.open_app("https://example.org").receipt["status"] == "requires_foreground"


def test_a_policy_deferral_is_not_a_missing_capability():
    from orion_core.capability_gaps import classify

    assert classify("'gui_navigation' needs a foreground session; the background task "
                    "left your desktop unchanged.") == "designed"
    assert classify("I won't press 'Delete All' for you - it can't be undone") == "designed"


# ── his face's expressions ───────────────────────────────────────────────────

class _Signal:
    def __init__(self) -> None:
        self.slots: list = []
        self.sent: list = []

    def connect(self, slot) -> None:
        self.slots.append(slot)

    def disconnect(self, slot) -> None:
        self.slots.remove(slot)

    def emit(self, *args) -> None:
        self.sent.append(args)
        for slot in list(self.slots):
            slot(*args)


def _emotion_bus():
    return SimpleNamespace(state=_Signal(), sentiment_payload=_Signal(),
                           emotion_changed=_Signal(), log=_Signal())


@pytest.mark.parametrize("said,expression", [
    ("sad", "sad"), ("your sad face", "sad"), ("in love", "love"), ("Surprised!", "surprised"),
    ("what does it look like when you're happy", "happy"), ("smile", "happy"),
    ("tired", "sleepy"), ("neutral", "neutral"), ("a banana", ""), ("", ""),
])
def test_expressions_are_understood_as_people_say_them(said, expression):
    from orion_core.emotion import face_expression

    assert face_expression(said) == expression


async def test_a_shown_expression_holds_while_he_talks_then_lets_go():
    from orion_core.emotion import EmotionStateManager

    bus = _emotion_bus()
    engine = EmotionStateManager(bus)
    message, ok = engine.show("your sad face", seconds=0.5)
    assert ok and "sad" in message and bus.emotion_changed.sent[-1][0] == "sad"
    engine._on_state("SPEAKING")                 # he says "here's my sad face"
    assert bus.emotion_changed.sent[-1][0] == "sad"
    await asyncio.sleep(0.7)
    assert bus.emotion_changed.sent[-1][0] == "speaking"
    assert not engine.show("banana")[1]


async def test_an_expression_tour_shows_each_in_turn_and_ends_at_rest():
    from orion_core.emotion import EmotionStateManager

    bus = _emotion_bus()
    engine = EmotionStateManager(bus)
    message, ok = engine.tour("happy, love and surprised", step_s=0.8)
    assert ok and "happy, love, surprised" in message
    await asyncio.sleep(2.7)
    worn = [args[0] for args in bus.emotion_changed.sent]
    assert worn[-4:] == ["happy", "love", "surprised", "neutral"], worn


class _Engine:
    def __init__(self) -> None:
        self.calls: list = []

    def show(self, name, seconds=None):
        self.calls.append(("show", name))
        return f"Showing my {name} face now.", True

    def tour(self, names=None, step_s=None):
        self.calls.append(("tour", names))
        return "Showing my expressions.", True


def _host(engine=None, bus=None):
    from orion_core.dispatch_desktop import DesktopDispatchMixin

    host = DesktopDispatchMixin()
    host.emotion = engine
    host.bus = bus or SimpleNamespace(gui_command=_Signal(), gui_result=_Signal())
    return host


def test_interface_control_shows_and_tours_expressions_for_the_owner():
    engine = _Engine()
    host = _host(engine)
    with policy.conversation_scope("voice"):
        assert host.interface_control({"action": "emotion", "target": "sad"}).ok
        assert host.interface_control({"action": "emotion_tour", "target": "happy, sad"}).ok
    assert engine.calls == [("show", "sad"), ("tour", ["happy", "sad"])]
    with policy.background_scope("job"):
        refused = host.interface_control({"action": "emotion", "target": "sad"})
    assert refused.receipt["status"] == "requires_foreground" and len(engine.calls) == 2


# ── reading and pressing the deck ────────────────────────────────────────────

class _Deck:
    """The deck API deck_reader uses, over two real pages of widgets."""

    def __init__(self) -> None:
        self.pages = {}
        for name, widgets in (
                ("CHESS", [QLabel("Chess Practice"), QPushButton("Start &Practice"),
                           QPushButton("Delete All Games")]),
                ("TASKS", [QLabel("Overdue: 2 tasks"), QPushButton("Refresh"),
                           QPushButton("Refresh List")])):
            page = QWidget()
            layout = QVBoxLayout(page)
            for widget in widgets:
                layout.addWidget(widget)
            self.pages[name] = page
        hidden = QLabel("secret hidden label")
        self.pages["TASKS"].layout().addWidget(hidden)
        hidden.hide()
        self.current = "TASKS"

    def page_names(self):
        return list(self.pages)

    def page_widget(self, name):
        return self.pages.get(name)

    live_page = page_widget

    def display_name(self, name):
        return name.title()

    def page_terms(self, name):
        return f"{name} {name.title()} {'Play' if name == 'CHESS' else 'Productivity'}"

    def current_page_name(self):
        return self.current


def test_a_page_is_read_as_its_text_never_its_hidden_widgets(_app):
    from orion_core.gui import deck_reader

    ok, text = deck_reader.read_page(_Deck(), "TASKS")
    assert ok and "Overdue: 2 tasks" in text and "[button] Refresh" in text
    assert "secret" not in text


def test_find_says_which_page_holds_the_words(_app):
    from orion_core.gui import deck_reader

    ok, text, best = deck_reader.find(_Deck(), "practice")
    assert ok and best == "CHESS" and "Chess" in text
    ok, text, best = deck_reader.find(_Deck(), "zebra crossing")
    assert not ok and best is None


def test_an_ordinary_button_is_found_and_a_destructive_one_refused(_app):
    from orion_core.gui import deck_reader

    deck = _Deck()
    ok, text, button, page = deck_reader.locate(deck, "start practice", None)
    assert ok and page == "CHESS" and button.text() == "Start &Practice"
    assert deck_reader._button_label(button) == "Start Practice" and "'Start Practice'" in text
    ok, text, button, page = deck_reader.locate(deck, "delete all games", "CHESS")
    assert not ok and button is None and "won't press" in text
    ok, text, button, page = deck_reader.locate(deck, "refresh", "TASKS")
    assert ok and button.text() == "Refresh"                      # exact beats partial
    ok, text, button, _page = deck_reader.locate(deck, "nothing like this", None)
    assert not ok and button is None


async def test_the_deck_request_reports_what_the_window_answered():
    bus = SimpleNamespace(gui_command=_Signal(), gui_result=_Signal())

    def window(payload):          # the core window, answering on its own thread
        bus.gui_result.emit({"request_id": payload["request_id"], "ok": True,
                             "text": f"read {payload['target'] or payload['page']}", "page": "TASKS"})
    bus.gui_command.connect(window)
    host = _host(bus=bus)
    with policy.conversation_scope("voice"):
        result = await host.interface_control({"action": "read", "page": "tasks"})
    assert result.ok and result.text == "read tasks"
    assert bus.gui_command.sent[0][0]["query"] == "read" and not bus.gui_result.slots


def test_the_compact_face_does_not_replay_its_power_on(_app):
    from orion_core.gui.orb_overlay import OrbOverlay
    from orion_core.gui.screen_face import BOOT_SECONDS

    overlay = OrbOverlay(form="screen")
    try:
        assert overlay.orb._head._boot > BOOT_SECONDS
    finally:
        overlay.deleteLater()


# ── one answer per request ───────────────────────────────────────────────────

def _worker():
    from orion_core.live_worker import GenAILiveWorker

    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    worker.bus = SimpleNamespace(log=_Signal())
    return worker


async def test_words_already_answered_by_text_are_not_started_again(monkeypatch):
    from orion_core.live_worker import GenAILiveWorker

    release = asyncio.Event()
    ran: list[str] = []

    async def fake_run(self, text, reason="", *, epoch=0):
        ran.append(text)
        await release.wait()

    monkeypatch.setattr(GenAILiveWorker, "_run_text_fallback", fake_run)
    worker = _worker()
    first = asyncio.create_task(worker._submit_text_fallback("Alright, let's go for a round."))
    await asyncio.sleep(0)
    await worker._submit_text_fallback("alright lets go for a round")     # the watchdog's copy
    assert ran == ["Alright, let's go for a round."]
    assert any("not starting it twice" in args[0] for args in worker.bus.log.sent)
    release.set()
    await first
    assert not worker._text_fallback_tasks


async def test_a_fallback_stands_down_once_live_answers_its_request(monkeypatch):
    from orion_core.live_worker import GenAILiveWorker

    seen: dict = {}

    async def fake_run(self, text, reason="", *, epoch=0):
        task = asyncio.current_task()
        seen["before"] = GenAILiveWorker._live_took_over(self, task)
        self._note_live_answer()                     # Live speaks for this request
        seen["after"] = GenAILiveWorker._live_took_over(self, task)

    monkeypatch.setattr(GenAILiveWorker, "_run_text_fallback", fake_run)
    worker = _worker()
    worker._turn_request_id = 4
    worker._marked_turn_key = GenAILiveWorker._normalise_phrase("let's go for a round")
    await worker._submit_text_fallback("Let's go for a round.")
    assert seen == {"before": False, "after": True}
    # words Live never had: nothing to stand down for
    seen.clear()
    await worker._submit_text_fallback("something else entirely")
    assert seen == {"before": False, "after": False}


async def test_leaving_retires_every_text_fallback():
    worker = _worker()
    worker._fallback_epoch = 3
    pending = asyncio.create_task(asyncio.sleep(10))
    worker._text_fallback_tasks = {pending}
    worker._retire_text_fallbacks()
    await asyncio.gather(pending, return_exceptions=True)
    assert worker._fallback_epoch == 4 and pending.cancelled()


# ── reading feelings ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("said,feeling", [
    ("I've had a really rough day", "sadness"),
    ("I'm feeling low today", "sadness"),
    ("I'm not happy with this", "sadness"),
    ("haha that's so funny", "amusement"),
    ("thank you so much", "gratitude"),
    ("this keeps crashing, I'm fed up", "frustration"),
    ("I'm so excited, I can't wait", "excitement"),
    ("you're the best, love you", "affection"),
    ("I'm exhausted", "tiredness"),
    ("wow, no way", "surprise"),
])
def test_he_reads_the_feelings_people_voice(said, feeling):
    from orion_core.emotion import SentimentAnalyser

    assert SentimentAnalyser.analyse(said, origin="user")["sentiment"] == feeling


def test_a_negated_feeling_is_not_counted():
    from orion_core.emotion import SentimentAnalyser

    assert SentimentAnalyser.analyse("I'm not excited about it", origin="user")["sentiment"] != "excitement"


@pytest.mark.parametrize("feeling,face", [("sadness", "empathetic"), ("frustration", "reassuring"),
                                          ("affection", "shy"), ("excitement", "excited")])
def test_his_face_answers_the_owners_feeling_with_empathy(feeling, face):
    from orion_core.emotion import EmotionStateManager

    bus = _emotion_bus()
    engine = EmotionStateManager(bus)
    engine._on_sentiment({"sentiment": feeling, "confidence": 0.8, "intensity": 0.6,
                          "origin": "user", "reason": ""})
    assert bus.emotion_changed.sent[-1][0] == face


def test_his_own_confident_reply_is_no_longer_a_blank_face():
    from orion_core.emotion import EmotionStateManager

    bus = _emotion_bus()
    engine = EmotionStateManager(bus)
    engine._on_sentiment({"sentiment": "confident", "confidence": 0.8, "intensity": 0.5,
                          "origin": "orion", "reason": ""})
    assert bus.emotion_changed.sent[-1][0] == "content"


# ── cute or professional ─────────────────────────────────────────────────────

def test_the_cute_style_is_retired_and_he_stays_professional(_app):
    # Mark 39 (owner, 27 Oct 2026): "professional and not cute".
    from orion_core import voice_settings as vs
    from orion_core.bus import OrionBus
    from orion_core.identity import IdentityManager

    identity = IdentityManager(OrionBus())
    assert not vs.apply_style("cute")[1]
    persona = identity.persona_text()
    assert "VOICE STYLE - CUTE" not in persona and "WIT AND SASS" in persona
    assert "JARVIS" in persona and "JARVIS" in identity.style_capsule()
    assert "Okie-dokie" in persona            # named only as something he never says
    assert vs.apply_style("professional")[1]


# ── singing ──────────────────────────────────────────────────────────────────

def test_his_offline_voice_sings_before_the_windows_voice(monkeypatch):
    from orion_core import singing

    monkeypatch.setattr(singing, "voices_on", lambda: True)
    names = [voice.name for voice in singing.voice_chain()]
    assert names[:2] == ["Charon", "offline voice"]
    if "Windows voice" in names:
        assert names[-1] == "Windows voice"


def test_the_master_is_linear_below_the_knee_and_bounded_above():
    import numpy as np

    from orion_core import singing

    t = np.linspace(0, 1, 24000, dtype=np.float32)
    quiet = 0.2 * np.sin(2 * np.pi * 220 * t)
    signal = quiet.copy()
    signal[1000] = 3.0                                    # one stray spike
    out = singing.master(signal)
    assert float(np.max(np.abs(out))) < 1.0
    body = np.delete(np.arange(len(signal)), 1000)
    gain = out[body] / np.where(np.abs(signal[body]) > 1e-6, signal[body], 1)
    mask = np.abs(signal[body]) > 1e-3
    assert np.allclose(gain[mask], gain[mask][0], rtol=1e-4)  # a clean gain, no saturation


async def test_a_cover_is_a_short_snippet_to_his_own_tune():
    from orion_core import singing

    song, note = await singing.plan_song(SimpleNamespace(), {
        "song": "Some Pop Song", "cover": True,
        "lyrics": "first placeholder line / second placeholder line / third line / fourth line"})
    assert len(song.lines) == singing.COVER_MAX_LINES and "snippet" in note
    original, note = await singing.plan_song(SimpleNamespace(), {
        "title": "My Own Song", "lyrics": "one line here / two lines here / three lines here"})
    assert len(original.lines) == 3 and note == "your words"


# ── common sense: the weather, specifically ──────────────────────────────────

def _forecast():
    """Wet until 4 pm today and on Tuesday and Wednesday; Thursday dry."""
    from datetime import datetime, timedelta

    from orion_core import weather

    start = datetime(2026, 10, 26, 0, 0)
    times, codes, chances, temps = [], [], [], []
    for h in range(24 * 7):
        at = start + timedelta(hours=h)
        day = h // 24
        wet = (day == 0 and at.hour < 16) or day in (1, 2) or day == 5
        times.append(at.strftime("%Y-%m-%dT%H:%M"))
        codes.append(61 if wet else 1)
        chances.append(85 if wet else 5)
        temps.append(11.0 + day)
    days = [(start + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(7)]
    daily_wet = [True, True, True, False, False, True, False]
    data = {
        "current": {"time": "2026-10-26T13:00", "temperature_2m": 11.0, "weather_code": 61,
                    "apparent_temperature": 9.0},
        "hourly": {"time": times, "temperature_2m": temps, "apparent_temperature": temps,
                   "precipitation_probability": chances, "weather_code": codes,
                   "wind_gusts_10m": [10] * len(times)},
        "daily": {"time": days, "weather_code": [61 if w else 1 for w in daily_wet],
                  "temperature_2m_max": [12 + d for d in range(7)],
                  "temperature_2m_min": [6 + d for d in range(7)],
                  "apparent_temperature_min": [5 + d for d in range(7)],
                  "precipitation_probability_max": [85 if w else 5 for w in daily_wet],
                  "precipitation_sum": [3 if w else 0 for w in daily_wet],
                  "wind_gusts_10m_max": [12] * 7, "uv_index_max": [1] * 7,
                  "sunrise": [f"{d}T07:00" for d in days], "sunset": [f"{d}T16:45" for d in days]},
    }
    return weather.parse(data, "Colchester")


@pytest.mark.parametrize("question", [
    "when's the weather going to stop being bad",
    "when will it stop raining", "when does it clear up", "when's the next dry day",
])
def test_when_will_it_stop_is_answered_with_the_hour_and_the_day(question):
    from orion_core import weather

    text = weather.answer(_forecast(), question)
    assert "should stop around" in text and "today" in text, text
    assert "Thursday" in text and "degrees" in text, text
    assert "expect rain tomorrow and Wednesday" in text, text   # what to expect before it


def test_the_weather_in_my_own_country_means_my_own_town():
    from orion_core.dispatch_web import _own_country

    here = SimpleNamespace(locality=lambda: "Colchester, England, United Kingdom")
    abroad = SimpleNamespace(locality=lambda: "Paris, Ile-de-France, France")
    assert _own_country("the UK", here) and _own_country("Britain", here)
    assert not _own_country("France", here) and not _own_country("the UK", abroad)
    assert not _own_country("London", here) and not _own_country("UK", None)


# ── his memory of past conversations, attached to the request ───────────────

def test_a_request_that_refers_back_carries_what_was_said_before():
    from orion_core.memory import MemoryAgent

    agent = MemoryAgent.__new__(MemoryAgent)
    agent.recall_said = lambda text, limit=4: [
        {"content": "open the file I mentioned", "created_at": "2026-10-26 14:00:00"},
        {"content": "my essay is called Neural Plasticity Draft", "created_at": "2026-10-20 10:00:00"}]
    note = agent.past_note("open the file I mentioned")
    assert "Neural Plasticity Draft" in note and "2026-10-20" in note
    assert "open the file I mentioned" not in note.split(":", 1)[1]     # not the request itself
    assert agent.past_note("remind me before 5 pm") == ""             # not a reference back
    assert agent.past_note("what's the weather") == ""


def test_the_persona_says_use_what_you_know_be_specific_and_feel(_app):
    from orion_core.bus import OrionBus
    from orion_core.identity import IdentityManager

    text = IdentityManager(OrionBus()).persona_text()
    assert "CONTEXT RULE" in text and "SPECIFICS RULE" in text and "EMOTIONAL INTELLIGENCE" in text
    assert "SPECIFICS RULE" in IdentityManager(OrionBus()).style_capsule()


# ── his name, said right ─────────────────────────────────────────────────────

@pytest.mark.parametrize("written", ["O.R.I.O.N. is shutting down now.", "ORION is restarting",
                                     "Orion here.", "ORIN online"])
def test_kokoro_is_given_his_name_as_it_is_said(written):
    from orion_core.voice_kokoro import spoken_name

    said = spoken_name(written)
    assert "Oh-ryan" in said and "O.R" not in said and "ORI" not in said.upper().replace("OH-RYAN", "")


def test_the_voice_model_never_reads_the_dotted_name_in_the_tools_it_uses_to_leave():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    for name in ("shutdown_orion", "restart_orion"):
        tool = next(d for d in TOOL_DECLARATIONS if d["name"] == name)
        assert "O.R.I.O.N" not in tool["description"], name


# ── away mode and his own manual (TORVIS parity, 29 Oct 2026) ────────────────

def test_away_mode_keeps_the_pc_awake_locks_it_when_idle_and_gives_sleep_back():
    import time as _time

    from orion_core.away_mode import AwayMode
    calls, idle = [], {"s": 0.0}
    away = AwayMode(calls.append, idle=lambda: idle["s"],
                    keep_awake=lambda on: calls.append(("awake", on)) or True,
                    lock=lambda: calls.append("lock") or True, poll_s=0.01, lock_after_s=600)
    assert "stays awake" in away.start() and away.on
    _time.sleep(0.05)
    assert ("awake", True) in calls and "lock" not in calls
    idle["s"] = 700.0
    _time.sleep(0.05)
    assert calls.count("lock") == 1                     # once, not on every check
    _time.sleep(0.05)
    assert calls.count("lock") == 1 and "locked itself" in away.status()
    idle["s"] = 0.0                                      # he is back
    _time.sleep(0.05)
    idle["s"] = 700.0                                    # and leaves again
    _time.sleep(0.05)
    assert calls.count("lock") == 2
    assert "sleep again" in away.stop() and not away.on
    assert calls[-1] == ("awake", False) or ("awake", False) in calls


def test_his_manual_answers_how_he_works():
    from orion_core import self_manual
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    host = KnowledgeDispatchMixin()
    heard = host.troubleshooting_guide_tool({"question": "why can't you hear me, the mic seems dead"})
    assert heard.ok and heard.text.startswith("He can't hear me")
    robot = self_manual.lookup("how do I flash firmware to my arduino")[0][0]
    assert robot == "Writing and flashing robot firmware"
    assert "Manual sections" in host.troubleshooting_guide_tool({}).text
    assert not host.troubleshooting_guide_tool({"question": "zzz qqq"}).ok
