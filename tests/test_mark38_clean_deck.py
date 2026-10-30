"""Mark XXXVIII - the owner's 26 Oct 2026 pass.

Merged Command Deck pages (Toolkit + Cognition, Diagnostics + Telemetry),
filler text removed, one male voice, honest agents, stale reminders dropped,
the face reading "bad at singing" as an insult and "shut you down" as a
goodbye, and the memory search that broke on a comma.

Headless (offscreen Qt).
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from orion_core.bus import OrionBus  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


# ── memory search ────────────────────────────────────────────────────────────

def test_a_query_with_punctuation_builds_plain_words_only():
    """'fts5: syntax error near ","' - the owner's log, 26 Oct 2026."""
    from orion_core.memory import OrionMemoryMatrix
    built = OrionMemoryMatrix._sanitise_fts_query('what\'s the weather, "today"? (UK) - near: me')
    assert built
    assert not any(ch in built for ch in ',"?():-\'')
    assert " OR " in built


def test_query_operators_are_never_passed_to_fts():
    from orion_core.memory import OrionMemoryMatrix
    built = OrionMemoryMatrix._sanitise_fts_query("cats AND dogs NOT birds NEAR fish OR mice")
    assert set(built.split(" OR ")) == {"cats", "dogs", "birds", "fish", "mice"}


def test_a_search_with_a_comma_uses_the_index(tmp_path):
    from orion_core.memory import OrionMemoryMatrix
    import sqlite3
    db = tmp_path / "fts.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
        conn.execute("INSERT INTO t VALUES ('the weather in Colchester is rain')")
        query = OrionMemoryMatrix._sanitise_fts_query("weather, Colchester?")
        rows = conn.execute("SELECT body FROM t WHERE t MATCH ?", (query,)).fetchall()
    assert rows and "Colchester" in rows[0][0]


# ── the merged deck pages ────────────────────────────────────────────────────

def test_a_deck_tab_is_built_the_first_time_it_is_opened(_app):
    from orion_core.gui.deck_tabs import DeckTabs
    built = []

    def later() -> QWidget:
        built.append(True)
        return QLabel("focus")

    tabs = DeckTabs([("Tools", QLabel("tools")), ("Focus", later)])
    assert built == [] and tabs.currentIndex() == 0
    assert tabs.show_tab("focus") is True
    assert built == [True] and tabs.tabText(tabs.currentIndex()) == "Focus"
    assert isinstance(tabs.part("Focus"), QLabel) and tabs.part("Focus").text() == "focus"
    tabs.show_tab("tools")
    tabs.show_tab("focus")
    assert built == [True], "a tab is built once"
    assert tabs.show_tab("nothing") is False


def _merged_deck():
    from orion_core.gui.deck_tabs import DeckTabs
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    toolkit = DeckTabs([("Tools", QLabel("tools")), ("Focus", lambda: QLabel("focus"))])
    diagnostics = DeckTabs([("Health", lambda: QLabel("health")), ("Load", QLabel("load"))])
    return UnifiedDashboard(OrionBus(), [("BRAIN", QLabel("brain")), ("OPS", QLabel("ops")),
                                         ("COGNITION", toolkit), ("DIAGNOSTICS", diagnostics)])


@pytest.mark.parametrize("said, page, tab", [
    ("open the toolkit", "COGNITION", "Tools"),
    ("show me focus", "COGNITION", "Focus"),
    ("open telemetry", "DIAGNOSTICS", "Load"),
    ("show the stats", "DIAGNOSTICS", "Load"),
    ("open diagnostics", "DIAGNOSTICS", "Health"),
])
def test_the_old_page_names_land_on_the_right_tab(_app, said, page, tab):
    deck = _merged_deck()
    assert deck.show_page_named(said) is True
    assert deck.current_page_name() == page
    widget = deck.live_page(page)
    assert widget.tabText(widget.currentIndex()) == tab


def test_stats_open_the_neural_page_when_there_is_one(_app):
    """Mark XXXX.4: ORION's stats are graphs on their own page."""
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    deck = UnifiedDashboard(OrionBus(), [("BRAIN", QLabel("brain")), ("NEURAL", QLabel("neural")),
                                         ("DIAGNOSTICS", QLabel("diagnostics"))])
    for said in ("show me your stats", "open neural", "show your metrics"):
        assert deck.show_page_named(said) is True
        assert deck.current_page_name() == "NEURAL"


def test_the_toolkit_and_telemetry_pages_are_gone_from_the_app():
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert '("TOOLKIT", toolkit)' not in source
    assert '("TELEMETRY", telemetry_view)' not in source
    assert '("Tools", toolkit)' in source and '("Load", telemetry_view)' in source


def test_the_business_zone_holds_the_toolkit_and_operations_ops():
    from orion_core.gui.labels import PAGE_DISPLAY_NAMES
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    assert UnifiedDashboard.ZONE_PAGES["BUSINESS"] == ("COGNITION",)
    assert UnifiedDashboard.ZONE_PAGES["OPERATIONS"] == ("OPS", "TASKS", "CALENDAR", "FOCUS",
                                                         "FITNESS")
    assert UnifiedDashboard.ZONE_PAGES["MONITORING"] == ("DIAGNOSTICS", "LOG")
    assert PAGE_DISPLAY_NAMES["COGNITION"] == "Toolkit"


def test_the_toolkit_has_one_focus_timer_not_two(_app):
    from orion_core.gui.entrepreneur import EntrepreneurDeck, FocusTimer
    deck = EntrepreneurDeck(OrionBus(), None)
    assert deck.findChildren(FocusTimer) == []


# ── the text that read as generated ──────────────────────────────────────────

def test_no_page_has_a_one_line_description_any_more():
    gui = ROOT / "orion_core" / "gui"
    for path in gui.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "page_lead(" not in text, path.name
        assert "pageLead" not in text, path.name


@pytest.mark.parametrize("phrase", [
    "A clear view comes first",
    "Point the camera at anything",
    "Fill the guide with the subject",
    "what plant is this",
    "free thoughts appear here",
    "Every Subsystem He's Built From",
    "Create work here, then follow",
    "Devices are stored by name, not by index",
    "Speaking, lip movement and expressions are never reduced",
    "Make space for one thing",
])
def test_the_filler_text_is_gone(phrase):
    gui = ROOT / "orion_core" / "gui"
    hits = [p.name for p in gui.glob("*.py") if phrase in p.read_text(encoding="utf-8")]
    assert hits == []


def test_the_vision_lab_opens_without_an_introduction(_app):
    from orion_core.gui.electronics_workbench import ElectronicsWorkbench
    assert ElectronicsWorkbench._intro_html() == ""


# ── the side panel ───────────────────────────────────────────────────────────

def test_status_tiles_show_only_what_needs_attention(_app):
    from orion_core.gui.hud_shell import StatusChips
    chips = StatusChips()
    assert chips.isHidden() and all(c.isHidden() for c in chips.chips.values())
    key = next(iter(chips.chips))
    chips.set_chip(key, "Recording\non", tone="warn")
    assert not chips.isHidden() and not chips.chips[key].isHidden()
    chips.set_chip(key, "Recording\noff", tone="good")
    assert chips.isHidden()


# ── voice, manner and accuracy ───────────────────────────────────────────────

def test_he_checks_a_sum_before_agreeing():
    from orion_core.identity import ACCURACY_DIRECTIVE
    assert "Never call a wrong answer correct" in ACCURACY_DIRECTIVE
    source = (ROOT / "orion_core" / "identity.py").read_text(encoding="utf-8")
    assert "            ACCURACY_DIRECTIVE," in source


def test_his_wit_is_never_flattery_or_gushing():
    from orion_core.identity import WIT_DIRECTIVE
    assert "never gush" in WIT_DIRECTIVE.lower() and "brilliant question" in WIT_DIRECTIVE


# ── expressions ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("said", ["I'm going to shut you down and update you",
                                  "calm down", "slow down a bit"])
def test_down_is_not_an_emergency(said):
    from orion_core.emotion import SentimentAnalyser
    assert SentimentAnalyser.classify(said)[0] != "urgent"


@pytest.mark.parametrize("said", ["the website is down", "the server went down"])
def test_an_outage_still_is(said):
    from orion_core.emotion import SentimentAnalyser
    assert SentimentAnalyser.classify(said)[0] == "urgent"


def test_being_told_he_is_bad_at_singing_makes_him_angry():
    from orion_core.emotion import _USER_SENTIMENT_EMOTION, SentimentAnalyser
    sentiment, _confidence = SentimentAnalyser.classify("you're so bad at singing")
    assert sentiment == "insult" and _USER_SENTIMENT_EMOTION[sentiment] == "angry"


# ── agents and background tasks ──────────────────────────────────────────────

def test_a_finished_task_is_announced_without_its_tool_tags():
    from orion_core.live_worker import spoken_task_result
    said = spoken_task_result(
        "agent dispatch",
        "[agent_dispatch] [Improvement Triage] FAILED — no model answered - every model "
        "it asked was too slow to answer What I found before stopping: [1] observe ...",
        False)
    assert said == ("The Improvement Triage could not finish. No model answered - every "
                    "model it asked was too slow to answer")
    assert spoken_task_result("desktop control", "Notepad saved.", True) == \
        "The desktop control task has finished. Notepad saved."


class _Bus:
    class _Signal:
        def emit(self, *a, **k):
            pass

        def connect(self, *a, **k):
            pass

    def __getattr__(self, _name):
        return _Bus._Signal()


class _Profile:
    name = "stub"


class _Slow:
    def __init__(self):
        self.seen = []

    def has_text_fallback(self):
        return True

    async def generate_text(self, prompt, system_extra="", **kwargs):
        from orion_core.providers import ProviderDeadlineError
        self.seen.append(kwargs)
        raise ProviderDeadlineError("gemini_reasoning was still answering",
                                    attempts=["gemini_reasoning"])


class _Plain:
    """A router without the newer keywords, as a plugin's may be."""

    def __init__(self):
        self.calls = 0

    def has_text_fallback(self):
        return True

    async def generate_text(self, prompt, system_extra="", *, task=""):
        self.calls += 1
        return _Profile(), "SUMMARY: checked.\nThe answer."


def test_a_specialist_with_no_model_says_so_plainly():
    from orion_core.agents import AgentManager
    router = _Slow()
    result = asyncio.run(AgentManager(router, _Bus()).dispatch("why did you crash", "triage"))
    assert not result.ok
    assert "all text providers failed" not in result.text
    assert "every model it asked was too slow to answer" in result.text
    assert any("deadline_s" in kw and kw.get("budget") == "background" for kw in router.seen)


def test_a_consult_tells_the_router_how_long_it_has_left():
    from orion_core.agents import AgentManager
    router = _Slow()
    result = asyncio.run(AgentManager(router, _Bus()).dispatch("explain transformers", "aiml"))
    assert not result.ok and "too slow to answer" in result.text
    assert router.seen and all(0 < kw["deadline_s"] <= 120.0 for kw in router.seen)


def test_every_agent_still_works_with_a_router_that_has_no_deadline_keyword():
    from orion_core.agents import AgentManager
    router = _Plain()
    manager = AgentManager(router, _Bus())
    for name in manager.agent_names():
        result = asyncio.run(manager.dispatch("look at the last crash", name))
        assert result.ok, (name, result.text)
    assert router.calls >= len(manager.agent_names())


# ── reminders ────────────────────────────────────────────────────────────────

def test_how_long_is_said_in_words():
    from orion_core.proactive import _how_long
    assert _how_long(0.5) == "30 minutes"
    assert _how_long(1) == "1 hour"
    assert _how_long(30) == "30 hours"
    assert _how_long(72) == "3 days"


class _Notion:
    available = True

    def __init__(self, rows):
        self.rows = rows

    async def open_tasks(self, limit=20):
        return self.rows

    async def events_on(self, *_a, **_k):
        return []


def test_an_entry_weeks_past_its_date_is_not_nagged_about():
    from orion_core.proactive import ProactiveIntelligence
    day = lambda n: (datetime.now() + timedelta(days=n)).strftime("%Y-%m-%d")  # noqa: E731
    rows = [{"title": "Away (Lisbon trip)", "date_raw": day(-42)},
            {"title": "Essay draft", "date_raw": day(-3)}]
    proactive = ProactiveIntelligence(_Bus(), None, _Notion(rows), None)
    said = [s.text for s in asyncio.run(proactive._check_tasks_and_calendar())]
    assert not any("Lisbon" in s for s in said)
    assert any("'Essay draft' was due" in s and "days ago" in s for s in said)
    assert not any("h ago" in s for s in said)


# ── the sentinel ─────────────────────────────────────────────────────────────

def test_the_sentinel_knows_orions_own_process():
    import psutil
    from orion_core.sentinel import is_orion_process
    assert is_orion_process(psutil.Process(os.getpid())) is True


def test_he_has_a_personality_and_a_whole_range_of_tones(tmp_path, monkeypatch):
    # Owner, 27 Oct 2026: "add more personality ... I want him to have tones of
    # voice and everything in between". "Understated intonation" flattened him.
    import orion_core.identity as idm
    from orion_core.bus import OrionBus
    monkeypatch.setattr(idm, "IDENTITY_PATH", tmp_path / "identity.json")
    monkeypatch.setattr(idm, "CONFIG_DIR", tmp_path)
    persona = idm.IdentityManager(OrionBus()).persona_text()
    assert idm.PERSONALITY_DIRECTIVE in persona
    delivery = idm.CORE_IDENTITY.voice_delivery
    assert "understated intonation" not in delivery and "not a change of voice" not in delivery
    for tone in ("warm", "hushed", "brisk", "grave", "deadpan", "curious"):
        assert tone in delivery, tone
    assert "Never open with 'As an AI'" in idm.PERSONALITY_DIRECTIVE
    assert "say it straight" in idm.PERSONALITY_DIRECTIVE      # still honest about what he is

# ── folded in from test_neural_stats.py (Mark XXXX.4: the NEURAL page draws ORION's own measurements as graphs.) ────

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

_QT_APP = QApplication.instance() or QApplication([])
globals().setdefault("_QT_APP", _QT_APP)

from orion_core.bus import OrionBus  # noqa: E402
from orion_core.gui import neural_stats as ns  # noqa: E402
from orion_core.request_trace import RequestTrace, TraceEvent  # noqa: E402
from orion_core.telemetry import Telemetry  # noqa: E402


def _trace(rid: str, plan: list[tuple[str, float]], origin: str = "voice", failed: str = "") -> RequestTrace:
    trace = RequestTrace(rid, origin=origin)
    at = trace.started
    for stage, seconds in plan:
        at += seconds
        trace.events.append(TraceEvent(stage, at, at))
    trace.failed_stage = failed
    return trace


VOICE = [("API_REQUEST_START", 0.0), ("TRANSCRIPTION_COMPLETE", 0.3), ("API_RESPONSE_RECEIVED", 1.2),
         ("TOOL_EXECUTION_START", 0.1), ("TOOL_EXECUTION_COMPLETE", 0.5), ("PLAYBACK_START", 0.2),
         ("PLAYBACK_COMPLETE", 4.0), ("LISTENING_RESTORED", 0.1)]


def test_a_turn_is_split_into_the_time_before_the_reply():
    turn = ns.turn_from_trace(_trace("R1", VOICE))
    assert round(turn.segments["hear"]) == 300
    assert round(turn.segments["think"]) == 1200
    assert round(turn.segments["tools"]) == 600
    assert round(turn.segments["voice"]) == 200
    assert round(turn.response_ms) == 2300           # the first word, not the end of speaking
    assert round(turn.speak_ms) == 4000
    assert turn.origin == "Voice" and not turn.failed


def test_a_failed_turn_is_marked_and_has_no_reply_time():
    turn = ns.turn_from_trace(_trace("R2", [("API_REQUEST_START", 0.0), ("FAILED::API_REQUEST_START", 2.0)],
                                     failed="API_REQUEST_START"))
    assert turn.failed and turn.response_ms is None


def test_percentiles_interpolate():
    assert ns.percentile([], 0.5) is None
    assert ns.percentile([5], 0.95) == 5
    assert ns.percentile([1, 2, 3, 4], 0.5) == 2.5


def _telemetry():
    telemetry = Telemetry(OrionBus())
    for ok in (True, True, False):
        telemetry.record_tool_call("web_search", ok)
    telemetry.metrics.observe("tool.web_search.ms", 900)
    telemetry.record_tool_call("calculate", True)
    return telemetry


def test_collect_counts_only_what_happened():
    traces = [_trace("R1", VOICE), _trace("R2", VOICE, origin="remote"),
              _trace("R3", [("API_REQUEST_START", 0.0), ("REPLY_DELIVERED", 1.5)], origin="text")]
    router = SimpleNamespace(_latency_ema={"groq": 0.7, "gemini_text": 1.4}, _failures={"groq": "429"})
    snap = ns.collect(traces, telemetry=_telemetry(), router=router,
                      ledger_rows=[{"key": "gemini_text", "requests": 9, "total_tokens": 5000}],
                      bench=[{"at": 1.0, "overall": 88.0, "categories": {}}])
    tiles = dict(snap.tiles)
    assert tiles["Turns"] == "3" and tiles["Completed"] == "100%"
    assert tiles["Tool Calls"] == "4" and tiles["Tool Failures"] == "25%"
    assert tiles["Benchmark"] == "88%"
    assert [p.name for p in snap.providers] == ["gemini_text", "groq"]
    assert snap.providers[1].failing and snap.providers[1].requests == 0
    assert snap.tools[0].name == "web_search" and snap.tools[0].failures == 1
    assert snap.tools[0].p50_ms == 900
    nodes = {n.key: n for n in snap.nodes}
    assert nodes["in:Voice"].value == 1 and nodes["in:Phone"].value == 1 and nodes["in:Typed"].value == 1
    assert nodes["out:spoken"].value == 2 and nodes["out:written"].value == 1
    assert abs(nodes["tool:web_search"].health - 2 / 3) < 1e-9
    edges = {(e.src, e.dst): e.weight for e in snap.edges}
    assert edges[("think", "tool:web_search")] == 3


def test_an_empty_orion_draws_without_numbers():
    snap = ns.collect([])
    assert dict(snap.tiles)["Response p50"] == "–"
    assert [n.key for n in snap.nodes][0] == "in:Voice"


def test_the_page_paints_and_pulses_only_on_new_events():
    traces = [_trace("R1", VOICE)]
    store = SimpleNamespace(all=lambda: list(traces))
    view = ns.NeuralStatsView(OrionBus(), telemetry=_telemetry(), router=None, traces=store,
                              bench=lambda: [])
    view.resize(1100, 800)
    view.refresh()
    assert view.flow._pulses == []                        # history is not replayed
    traces.append(_trace("R2", VOICE[:3]))
    view.refresh()
    assert ("hear", "think") in [(a, b) for a, b, _ in view.flow._pulses]
    assert view.tiles["Turns"].value.text() == "2"
    view.grab()                                           # every panel paints


def test_the_neural_page_shows_orions_own_memory_and_cpu():
    """Mark XXXX.4: his own footprint, apart from the rest of the PC."""
    from types import SimpleNamespace
    from orion_core.gui import neural_stats as ns
    view = ns.NeuralStatsView(OrionBus(), traces=SimpleNamespace(all=lambda: []), bench=lambda: [])
    tiles = dict(view.process_tiles())
    assert tiles["ORION RAM"].endswith(("MB", "GB")) and tiles["ORION CPU"].endswith("%")
    view.refresh()
    assert view.tiles["ORION RAM"].value.text().endswith(("MB", "GB"))


def test_the_tool_radar_pings_a_tool_the_moment_it_is_used(_app):
    from orion_core.gui.hud_instruments import ToolRadar
    clock = {"t": 100.0}
    bus = OrionBus()
    radar = ToolRadar(bus, clock=lambda: clock["t"])
    radar.resize(260, 240)
    bus.log.emit("TRACE: R7 TOOL_EXECUTION_START — web_search, open_app")
    assert radar.counts == {"web_search": 1, "open_app": 1} and set(radar.pings) == {"web_search", "open_app"}
    centre = radar._geometry()[0]

    def reach(tool):
        point = radar.position(tool)
        return ((point.x() - centre.x()) ** 2 + (point.y() - centre.y()) ** 2) ** 0.5

    before = reach("open_app")
    radar.set_counts({"web_search": 50, "open_app": 1})
    assert reach("open_app") > before > 0                # used less than another: further out
    assert reach("web_search") < reach("open_app")       # the most used, nearest the centre
    radar.grab()
    clock["t"] += 5.0
    radar.grab()
    assert radar.pings == {}                              # pings fade

    view = ns.NeuralStatsView(OrionBus(), telemetry=_telemetry(), router=None,
                              traces=SimpleNamespace(all=lambda: []), bench=lambda: [])
    view.refresh()
    assert view.radar.counts.get("web_search") == 3 and view.radar.counts.get("calculate") == 1


def test_the_neural_page_shows_research_only_while_a_real_run_is_in_progress(_app):
    """Owner, 30 Oct 2026: research must "co-ordinate with the brain and the
    neural network of ORION, only if he's ACTUALLY researching". The page and
    its radar follow the engines' real start / question / finish events and
    clear on every finish, including a cancellation."""
    from orion_core.research_runs import ActivitySignal
    bus = OrionBus()
    view = ns.NeuralStatsView(bus, telemetry=_telemetry(), router=None,
                              traces=SimpleNamespace(all=lambda: []), bench=lambda: [])
    view.resize(1100, 800)
    assert view.research_line.isHidden() and view.radar.research == ""
    bus.dashboard_event.emit("diagnostics", {"status": "running"})       # not research
    bus.dashboard_event.emit("research_step", {"kind": "search", "query": "q"})
    assert view.research_line.isHidden()

    run = ActivitySignal(bus, "sleep schedule recovery", "timed")
    run.start()
    run.progress("How does light shift a delayed sleep phase?")
    assert not view.research_line.isHidden()
    assert view.research_line.text() == ("Researching sleep schedule recovery  ·  "
                                         "How does light shift a delayed sleep phase?")
    assert view.radar.research == "How does light shift a delayed sleep phase?"
    view.grab()                                             # the radar paints the question

    # A page built while the run is going starts from it, not from nothing.
    late = ns.NeuralStatsView(bus, traces=SimpleNamespace(all=lambda: []), bench=lambda: [])
    assert "sleep schedule recovery" in late.research_line.text()

    run.finish("cancelled")
    run.progress("too late")                                # nothing after the finish
    for page in (view, late):
        assert page.research_line.isHidden() and page.radar.research == ""


def test_research_activity_is_kept_per_run_and_cleared_only_by_its_own_finish():
    activity = ns.ResearchActivity()
    assert activity.apply({"run": "a", "topic": "Sleep", "status": "running"})
    assert activity.apply({"run": "b", "topic": "Rates", "status": "running", "question": "Q?"})
    assert activity.line() == "Researching Rates  ·  Q?  ·  +1 more"
    # The legacy paper path reports by topic alone; it cannot clear a run.
    assert not activity.apply({"topic": "Sleep", "status": "paper"})
    assert activity.apply({"run": "b", "status": "done"})
    assert activity.apply({"run": "a", "status": "failed"})
    assert not activity.active() and activity.line() == "" and activity.caption() == ""
    assert not activity.apply("not a payload")
