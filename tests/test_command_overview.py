"""Native Command Deck overview and persistent navigation regressions."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget

from orion_core.bus import OrionBus
from orion_core.gui.command_overview import CommandOverview
from orion_core.gui.unified_dashboard import UnifiedDashboard


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_overview_updates_only_live_values_and_routes_actions():
    app = _app()
    bus = OrionBus()
    opened = []
    view = CommandOverview(bus, on_open_page=opened.append)
    view.attach_deck_pages({"RESEARCH": "RESEARCH", "LOG": "MONITORING"})
    bus.state.emit("PROCESSING")
    bus.connection_state.emit({"provider": "Gemini", "state": "connected"})
    bus.telemetry_sample.emit({"cpu": 27.3, "ram": 61.8})
    view.pulse(to_id="research", tool="web_search", ok=True)
    app.processEvents()
    assert view.mode.text() == "Thinking"
    assert view.mode.accessibleName() == "ORION's state"
    assert view.model.state == "PROCESSING"
    assert view.iris.state_label(view.model.state) == "Thinking"
    assert view.iris.accessibleName() == "ORION state gauge"
    assert "Gemini" in view.link.text()
    assert view.host.text() == "Processor 27%   ·   Memory 62%"
    assert "Search the Web" in view.route.text()
    assert view._quick_links["RESEARCH"].isEnabled()
    assert not view._quick_links["MISSION"].isEnabled()
    view._quick_links["RESEARCH"].click()
    assert opened == ["RESEARCH"]


def test_split_pane_keeps_every_page_visible_in_navigation():
    _app()
    bus = OrionBus()
    deck = UnifiedDashboard(bus, [(name, QWidget()) for name in
                                  ("BRAIN", "RESEARCH", "GLOBE", "LOG", "CHESS")])
    assert set(deck._sidebar_buttons) == set(deck.page_names())
    assert deck._sidebar.isVisibleTo(deck) or not deck.isVisible()
    deck._sidebar_buttons["GLOBE"].click()
    assert deck.stack.currentIndex() == deck.page_names().index("GLOBE")
    assert deck._sidebar_buttons["GLOBE"].isChecked()
    assert deck.display_name("BRAIN") == "Brain"


def test_overview_shows_the_text_model_route_gpu_and_sound_alerts():
    app = _app()
    bus = OrionBus()
    view = CommandOverview(bus)
    bus.telemetry_sample.emit({"cpu": 10.0, "ram": 20.0, "gpu": 55.4})
    bus.dashboard_event.emit("provider_route", {
        "provider": "groq", "local": False, "latency_s": 1.234, "fallbacks": 1})
    app.processEvents()
    assert view.host.text() == "Processor 10%   ·   Memory 20%   ·   Graphics 55%"
    assert view.model_route.text() == "groq  ·  online  ·  1.2 s  ·  1 alternative tried"

    # A breaker opening on the provider shown says so; another provider's
    # does not overwrite it.
    bus.dashboard_event.emit("provider_breaker", {"provider": "ollama", "cooldown_s": 45})
    app.processEvents()
    assert view.model_route.text().startswith("groq  ·  online")
    bus.dashboard_event.emit("provider_breaker", {"provider": "groq", "cooldown_s": 90.0})
    app.processEvents()
    assert view.model_route.text() == "groq  ·  retrying in 90 s"

    view.show()
    bus.dashboard_event.emit("sound_alert", {"category": "smoke alarm", "score": 0.7})
    app.processEvents()
    assert view.hearing.text().startswith("Smoke alarm  ·  ")
    assert "SOUND: smoke alarm heard" in view.activity.text()
    view.hide()


def test_overview_ignores_malformed_dashboard_payloads():
    app = _app()
    bus = OrionBus()
    view = CommandOverview(bus)
    for channel in ("provider_route", "provider_breaker", "sound_alert"):
        bus.dashboard_event.emit(channel, "not a dict")
    bus.dashboard_event.emit("provider_route", {"provider": "p", "latency_s": "n/a"})
    app.processEvents()
    assert view.hearing.text() == "No sound alerts"
    assert view.model_route.text() == "p  ·  online"


def test_local_context_panel_shows_stop_and_source_staleness(monkeypatch):
    from orion_core.context_service import ContextService
    from orion_core.gui import command_overview

    app = _app()
    bus = OrionBus()
    view = CommandOverview(bus)
    clock = [1000.0]
    monkeypatch.setattr(command_overview.time, "time", lambda: clock[0])
    context = ContextService(bus, clock=lambda: clock[0])
    camera = context.open_source("camera.local")
    camera.available(True)
    camera.visual("appeared")
    app.processEvents()
    assert "Camera active" in view.context.text()
    assert "event 0s ago" in view.context.text()
    clock[0] += 6
    view._refresh_context()
    assert "Camera degraded" in view.context.text()
    camera.close()
    app.processEvents()
    assert "Camera off" in view.context.text()
    assert "event" not in view.context.text()
    from orion_core.perception import PerceptionLoop
    loop = PerceptionLoop(bus)
    loop.set_cloud_sharing(True, max_calls=2)
    app.processEvents()
    assert "Cloud frames on · 2 calls left" in view.context.text()
    loop.set_cloud_sharing(False)
    app.processEvents()
    assert "Cloud frames off" in view.context.text()
    view.close()


def test_a_long_provider_name_is_shortened_inside_the_ring_not_the_state():
    """It was clipped at both ends, so 'Disconnected' itself was lost
    (audit 6 Oct 2026). The provider gives way; the state stays whole."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtGui import QFont, QFontMetricsF
    from PyQt6.QtWidgets import QApplication
    globals().setdefault("_QT_APP", QApplication.instance() or QApplication([]))  # held, or it is collected
    from orion_core.gui.brain_instruments import BrainModel, _fit_link
    model = BrainModel()
    model.set_link("An Extremely Long Synthetic Provider Name", "disconnected")
    font = QFont()
    font.setPixelSize(10)
    metrics = QFontMetricsF(font)
    width = metrics.horizontalAdvance("  ·  Disconnected") + metrics.horizontalAdvance("Synthetic")
    fitted = _fit_link(model, font, width)
    assert fitted.endswith("·  Disconnected") and "…" in fitted
    assert metrics.horizontalAdvance(fitted) <= width
    # Too narrow even for the state: the state alone, shortened if it must be.
    narrow = _fit_link(model, font, metrics.horizontalAdvance("Disconnected"))
    assert narrow == "Disconnected"
    model.set_link("Gemini", "connected")
    assert _fit_link(model, font, 400) == model.link        # fits: untouched


# ── Mark XXXV W1.2: the instruments drawn by a Brain host process ────────────

def test_the_brain_model_hands_its_picture_to_another_process_intact():
    from orion_core.gui.brain_instruments import BrainModel
    _app()
    now = [100.0]
    source = BrainModel(clock=lambda: now[0])
    source.set_link("Gemini", "connected")
    source.heard()
    now[0] += 1.0
    source.set_state("PROCESSING")
    source.tool_started("web_search")
    now[0] += 0.5
    source.tool_finished("web_search", ok=False, elapsed_ms=400)
    source.tool_started("open_app")                          # still running
    source.set_telemetry(31.0, 62.0, 7.0)
    snap = source.snapshot()
    later = [5000.0]                                         # another clock entirely
    copy = BrainModel(clock=lambda: later[0])
    copy.load_snapshot(snap)

    def picture(model, now_value):
        return [(s.lane, round(now_value - s.start, 3),
                 None if s.end is None else round(now_value - s.end, 3), s.label, s.ok, s.tick)
                for s in model.visible_segments(now_value)]
    assert picture(copy, later[0]) == picture(source, now[0])
    assert copy.state == "PROCESSING" and copy.link_ok and (copy.cpu, copy.gpu) == (31.0, 7.0)
    assert set(copy._open) == set(source._open) == {"THINK", "ACT"}
    assert abs(copy.heat("TOOLS", later[0]) - source.heat("TOOLS", now[0])) < 1e-9


class _FakeBrain(QWidget):
    """Stands in for page_host.RemoteBrain: records what it would send."""
    instances: list = []

    def __init__(self, parent=None, *, background="#000"):
        super().__init__(parent)
        self.background = background
        self.sent: list = []
        self.failed = False
        self.on_failed = None
        self.snapshot_provider = None
        self.parts = {"iris": QWidget(), "lanes": QWidget()}
        _FakeBrain.instances.append(self)

    def part(self, name):
        return self.parts[name]

    def forward(self, method, *args):
        self.sent.append((method, args))

    def set_visible(self, visible):
        self.sent.append(("set_visible", (visible,)))

    def set_attended(self, attended):
        self.sent.append(("set_attended", (attended,)))

    def set_stalled(self, count):
        self.sent.append(("set_stalled", (count,)))

    def shutdown(self):
        self.sent.append(("shutdown", ()))


def test_the_brain_page_feeds_its_host_and_draws_itself_if_the_host_fails(monkeypatch):
    from orion_core.gui import command_overview, page_host
    from orion_core.gui.brain_instruments import IrisGauge, SignalLanes
    app = _app()
    _FakeBrain.instances.clear()
    monkeypatch.setattr(page_host, "RemoteBrain", _FakeBrain)
    monkeypatch.setenv("ORION_BRAIN_PROCESS", "1")
    monkeypatch.delenv("ORION_PAGE_PROCESS", raising=False)
    bus = OrionBus()
    view = CommandOverview(bus, remote_instruments=True)
    assert not _FakeBrain.instances, "the host starts on the page's first opening, not before"
    bus.state.emit("LISTENING")                              # before the host: kept locally
    view.show()
    app.processEvents()
    remote = _FakeBrain.instances[0]
    assert remote.background == command_overview._instrument_background()
    assert remote.sent[0][0] == "load_snapshot" and remote.sent[0][1][0]["state"] == "LISTENING"
    assert ("set_visible", (True,)) in remote.sent
    assert view.iris.accessibleName() == "ORION state gauge"
    bus.state.emit("PROCESSING")
    view.pulse(tool="web_search", ok=True, elapsed_ms=300)
    assert ("set_state", ("PROCESSING",)) in remote.sent
    assert ("tool_finished", ("web_search", True, 300)) in remote.sent
    assert view.model.state == "PROCESSING", "ORION's own model is kept as well"
    remote.on_failed()                                       # the host gave up
    app.processEvents()
    assert isinstance(view.iris.findChild(IrisGauge), IrisGauge)
    assert isinstance(view.lanes.findChild(SignalLanes), SignalLanes)
    assert ("shutdown", ()) in remote.sent and view._remote is None
    count = len(remote.sent)
    bus.state.emit("SPEAKING")
    assert len(remote.sent) == count, "nothing goes to a host that has gone"
    view.close()


def test_a_page_built_by_the_tests_never_starts_a_host(monkeypatch):
    from orion_core.gui import page_host
    _FakeBrain.instances.clear()
    monkeypatch.setattr(page_host, "RemoteBrain", _FakeBrain)
    app = _app()
    view = CommandOverview(OrionBus())
    view.show()
    app.processEvents()
    assert not _FakeBrain.instances
    view.close()


def test_the_instruments_background_is_the_panel_made_solid():
    from orion_core.gui import command_overview
    colour = command_overview._instrument_background()
    assert colour.startswith("#") and len(colour) == 7
    # Panel #161616 at 0.72 over the window #060606 at 0.88 on a dark desktop.
    assert colour == "#111111"


def test_background_work_shows_on_the_work_lane_until_it_ends():
    """Mark XXXX.4: agents, jobs, detached tools and installs were missing from
    the BRAIN page; each now has a WORK bar from start to finish."""
    _app()
    bus = OrionBus()
    view = CommandOverview(bus, on_open_page=lambda _name: None)
    lines = [
        "AGENT: routing request to the Cybersecurity Agent.",
        "JOB: job-abc123 started in the background — reliability check: the recorded crash",
        "TOOL: agent_dispatch still running after 45s - answering now and finishing it in the background.",
        "INSTALL: VideoLAN.VLC started — VLC media player",
    ]
    for line in lines:
        bus.log.emit(line)
    assert sorted(seg.key for seg in view.model.working()) == [
        "agent:cybersecurity", "install:VideoLAN.VLC", "job:job-abc123", "tool:agent_dispatch"]
    assert view.model.busy() is False                  # background work holds no turn open
    bus.log.emit("SPECIALIST: Cybersecurity Agent DONE — answered")
    bus.log.emit("JOB: job-abc123 failed after 12.0s — [agent_dispatch] boom")
    bus.log.emit("TOOL: agent_dispatch finished in the background - done.")
    bus.log.emit("INSTALL: VideoLAN.VLC finished — Installed VideoLAN.VLC.")
    assert view.model.working() == []
    ended = {s.key: s.ok for s in view.model.segments if s.lane == "WORK"}
    assert ended == {"agent:cybersecurity": True, "job:job-abc123": False,
                     "tool:agent_dispatch": True, "install:VideoLAN.VLC": True}
    snap = view.model.snapshot()
    assert all(len(row) == 9 for row in snap["segments"])


def test_a_long_running_job_survives_a_snapshot_round_trip():
    from orion_core.gui.brain_instruments import BrainModel
    _app()
    model = BrainModel()
    model.work_started("job:job-1", "Research")
    copy = BrainModel()
    copy.load_snapshot(model.snapshot())
    assert [s.key for s in copy.working()] == ["job:job-1"]
    copy.work_finished("job:job-1", ok=True)
    assert copy.working() == []


def test_the_ecg_beats_for_what_he_does_and_is_flat_when_quiet():
    """Mark XXXX.4: brain activity as an ECG - one heartbeat per real event."""
    from types import SimpleNamespace
    from orion_core.gui.brain_instruments import EcgMonitor, ecg_beat
    from orion_core.gui.neural_stats import NeuralStatsView
    _app()
    assert ecg_beat(-0.1) == 0 and ecg_beat(0.9) == 0 and ecg_beat(0.215) > 0.9   # the R wave
    clock = {"t": 50.0}
    bus = OrionBus()
    view = NeuralStatsView(bus, traces=SimpleNamespace(all=lambda: []), bench=lambda: [])
    view.ecg.clock = lambda: clock["t"]
    assert view.ecg.value("THINK", 50.2) == 0                          # quiet: a flat line
    bus.log.emit("TRACE: R9 API_REQUEST_START")
    bus.log.emit("TRACE: R9 TOOL_EXECUTION_START — web_search")
    assert view.ecg.active("THINK") and view.ecg.active("ACT")
    assert view.ecg.value("THINK", 50.215) > 0.9
    bus.log.emit("TRACE: R9 TOOL_EXECUTION_COMPLETE")
    bus.log.emit("AGENT: routing request to the Coding Agent.")
    assert not view.ecg.active("ACT") and view.ecg.active("WORK")
    bus.log.emit("SPECIALIST: Coding Agent DONE — answered")
    assert not view.ecg.active("WORK")
    monitor = EcgMonitor(clock=lambda: clock["t"])
    monitor.resize(600, 200)
    monitor.advance(50.5)                                               # draws without a screen
    assert monitor._canvas is not None
    for _ in range(40):                                                 # hidden, nothing prunes
        monitor.beat("HEAR")
    assert len(monitor._beats["HEAR"]) <= 12


def test_the_brain_page_has_an_ecg_setting_that_it_remembers():
    """Owner, 28 Oct 2026: "there must be a graph setting (kind of looking
    like a ECG machine)" on the BRAIN page, not only on NEURAL."""
    from orion_core.gui import command_overview
    _app()
    bus = OrionBus()
    view = CommandOverview(bus)
    assert view.graph_stack.currentWidget() is view.lanes            # Lanes by default
    view._graph_buttons["ECG"].click()
    assert view.graph_stack.currentWidget() is view.ecg
    assert command_overview.brain_graph() == "ECG"
    bus.log.emit("TRACE: R4 API_REQUEST_START")
    assert view.ecg.active("THINK")
    again = CommandOverview(OrionBus())
    assert again.graph_stack.currentWidget() is again.ecg            # kept across a restart
    assert again._graph_buttons["ECG"].isChecked()
    again.show_graph("Lanes")
    assert command_overview.brain_graph() == "Lanes"


def test_the_voice_ridgeline_shows_who_is_speaking_and_rests_when_quiet():
    """Owner, 28 Oct 2026: "Let's do the voice ridgeline" - the owner's voice
    from the microphone, ORION's from his playback, a row per frame."""
    import numpy as np

    from orion_core.gui import command_overview
    from orion_core.gui.hud_instruments import (RIDGE_BANDS, RIDGE_ROWS, VoiceRidgeline,
                                                mic_spectrum, voice_spectrum)
    _app()
    n = np.arange(1024) / 16000.0
    tone = (np.sin(2 * np.pi * 1000.0 * n) * 0.3 * 32767).astype(np.int16).tobytes()
    levels = mic_spectrum(tone)
    assert len(levels) == RIDGE_BANDS
    peak = levels.index(max(levels))
    centres = np.geomspace(80.0, 7000.0, RIDGE_BANDS)
    assert 700 < centres[peak] < 1400                     # the 1 kHz tone, where it belongs
    assert max(mic_spectrum(b"")) == 0.0
    assert max(voice_spectrum({"low": 0.6, "mid": 0.9, "high": 0.2}, 0.5)) > 0.3
    assert max(voice_spectrum({}, 0.0)) == 0.0

    sound = {"pcm": tone}
    bus = OrionBus()
    ridge = VoiceRidgeline(bus, source=lambda: sound["pcm"])
    ridge.resize(800, 220)
    for _ in range(5):
        ridge._advance()
    you, own = ridge.rows[-1]
    assert max(you) > 0.3 and max(own) == 0.0              # the owner, not ORION
    bus.voice_spectrum.emit({"low": 0.5, "mid": 0.8, "high": 0.3})
    bus.amplitude.emit(0.6)
    sound["pcm"] = b""
    ridge._advance()
    assert max(ridge.rows[-1][1]) > 0.3                     # ORION speaking
    for _ in range(RIDGE_ROWS * 3):
        ridge._advance()
    assert len(ridge.rows) == RIDGE_ROWS and ridge._quiet > RIDGE_ROWS   # settled: no repaints
    ridge.grab()

    view = CommandOverview(OrionBus())
    assert "Voice" in command_overview.GRAPH_STYLES
    view._graph_buttons["Voice"].click()
    assert view.graph_stack.currentWidget() is view.voice
    assert command_overview.brain_graph() == "Voice"



def test_research_runs_show_on_the_brain_work_lane():
    from types import SimpleNamespace
    """30 Oct 2026: "it says he's researching but it's not showing on the brain"."""
    calls = []
    page = SimpleNamespace(model=SimpleNamespace(
        work_started=lambda key, label="": calls.append(("start", key, label)),
        work_finished=lambda key, ok=True: calls.append(("end", key, ok))))
    CommandOverview._note_work(page, "RESEARCH: started 'sleep strategies' → 2026-10-30_sleep")
    CommandOverview._note_work(page, "RESEARCH: investigating — what shifts the body clock?")
    CommandOverview._note_work(page, "RESEARCH: completed — 6 note(s) in 2026-10-30_sleep.")
    CommandOverview._note_work(page, "RESEARCH: cancelled 'sleep strategies'.")
    assert calls == [("start", "research", "Researching sleep strategies"),
                     ("end", "research", True), ("end", "research", False)]


def test_dossiers_and_papers_get_their_own_work_bar_while_they_run():
    from types import SimpleNamespace
    calls = []
    page = SimpleNamespace(model=SimpleNamespace(
        work_started=lambda key, label="": calls.append(("start", key, label)),
        work_finished=lambda key, ok=True: calls.append(("end", key, ok))))
    running = {"run": "ab12", "topic": "sleep", "engine": "dossier", "status": "running"}
    CommandOverview._on_dashboard_event(page, "research", running)
    CommandOverview._on_dashboard_event(page, "research", dict(running, status="done"))
    CommandOverview._on_dashboard_event(page, "research", dict(running, engine="timed"))
    assert calls == [("start", "research:ab12", "Researching sleep"), ("end", "research:ab12", True)]
