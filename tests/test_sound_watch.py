"""
The background sound watch: local YAMNet over the microphone's rolling buffer,
announcing only the sounds a person would want to be told about.

The decision logic is tested with synthetic score vectors; the end-to-end
test runs the real YAMNet model on a synthetic 3.15 kHz smoke-alarm beep.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")

from orion_core import sound_watch
from orion_core.sound_watch import (
    DEFAULT_RULES, SILENCE_RMS, WINDOW_S, SoundWatch, SoundWatcher, window_rms,
)

LABELS = ["Speech", "Smoke detector, smoke alarm", "Alarm", "Doorbell", "Music", "Knock"]


def _scores(**by_label):
    out = np.zeros(len(LABELS), dtype=np.float32)
    for name, value in by_label.items():
        out[LABELS.index(name.replace("_", " "))] = value
    return out


SMOKE = {"Smoke detector, smoke alarm": 0.7, "Alarm": 0.6}


def _smoke():
    out = np.zeros(len(LABELS), dtype=np.float32)
    for name, value in SMOKE.items():
        out[LABELS.index(name)] = value
    return out


# ── the decision ─────────────────────────────────────────────────────────────

def test_one_loud_frame_is_not_an_alarm():
    watcher = SoundWatcher(LABELS)
    assert watcher.observe(_smoke(), now=0.0) == []


def test_a_sustained_alarm_is_announced_once_and_quiets_the_generic_rule():
    watcher = SoundWatcher(LABELS)
    watcher.observe(_smoke(), now=0.0)
    alerts = watcher.observe(_smoke(), now=1.0)
    assert [a.category for a in alerts] == ["smoke alarm"]
    assert alerts[0].urgent and "smoke alarm" in alerts[0].sentence()
    # Still ringing: no repeat within the cooldown, and "alarm" stays quiet.
    for second in range(2, 30):
        assert watcher.observe(_smoke(), now=float(second)) == []


def test_the_alert_repeats_after_its_cooldown():
    watcher = SoundWatcher(LABELS)
    watcher.observe(_smoke(), now=0.0)
    assert watcher.observe(_smoke(), now=1.0)
    rule = next(r for r in DEFAULT_RULES if r.category == "smoke alarm")
    later = 1.0 + rule.cooldown_s + 1.0
    watcher.observe(_smoke(), now=later)
    assert [a.category for a in watcher.observe(_smoke(), now=later + 1.0)] == ["smoke alarm"]


def test_a_sound_that_stops_does_not_count_towards_the_next():
    watcher = SoundWatcher(LABELS)
    watcher.observe(_smoke(), now=0.0)
    silence = np.zeros(len(LABELS), dtype=np.float32)
    watcher.observe(silence, now=1.0)
    watcher.observe(silence, now=2.0)
    assert watcher.observe(_smoke(), now=3.0) == []


def test_a_single_window_rule_fires_immediately():
    watcher = SoundWatcher(LABELS)
    alerts = watcher.observe(_scores(Doorbell=0.8), now=0.0)
    assert [a.category for a in alerts] == ["doorbell"]
    assert not alerts[0].urgent


def test_ordinary_sounds_raise_nothing():
    watcher = SoundWatcher(LABELS)
    for second in range(10):
        assert watcher.observe(_scores(Speech=0.9, Music=0.8), now=float(second)) == []


def test_rules_whose_labels_are_absent_are_ignored():
    watcher = SoundWatcher(["Speech"])
    assert watcher.observe(np.array([0.9], dtype=np.float32), now=0.0) == []


def test_window_rms():
    assert window_rms(b"") == 0.0
    loud = (np.full(1600, 16384, dtype=np.int16)).tobytes()
    assert window_rms(loud) == pytest.approx(0.5, abs=1e-3)


# ── the runner ───────────────────────────────────────────────────────────────

class _Source:
    def __init__(self, pcm: bytes):
        self.pcm = pcm

    def snapshot(self, seconds):
        return self.pcm


def _pcm(level: float) -> bytes:
    n = int(WINDOW_S * 16000)
    return (np.full(n, int(level * 32767), dtype=np.int16)).tobytes()


def test_silence_is_skipped_without_running_the_model():
    calls = []
    watch = SoundWatch(lambda a: None, source=_Source(_pcm(SILENCE_RMS / 4)),
                       classify=lambda w: calls.append(1) or _smoke(), labels=LABELS)
    assert watch._prepare()
    for _ in range(3):
        watch.step(now=0.0)
    assert calls == [] and watch.skipped == 3


def test_alerts_reach_the_consumer_and_a_broken_one_is_survived():
    delivered, logged = [], []

    def consumer(alert):
        delivered.append(alert)
        raise RuntimeError("consumer broke")

    watch = SoundWatch(consumer, source=_Source(_pcm(0.3)), classify=lambda w: _smoke(),
                       labels=LABELS, log=logged.append)
    assert watch._prepare()
    watch.step(now=0.0)
    alerts = watch.step(now=1.0)
    assert [a.category for a in alerts] == ["smoke alarm"]
    assert delivered == alerts
    assert any("delivery failed" in line for line in logged)
    assert watch.windows == 2


def test_start_and_stop_run_a_real_thread():
    watch = SoundWatch(lambda a: None, source=_Source(b""), classify=lambda w: _smoke(),
                       labels=LABELS, interval_s=0.25)
    ok, message = watch.start()
    assert ok and watch.running
    assert "already" in watch.start()[1]
    watch.stop()
    assert not watch.running
    assert "off" in watch.describe()


# ── the real model ───────────────────────────────────────────────────────────

def _model_ready():
    from orion_core.sound_sense import _Yamnet
    return _Yamnet.load()


def test_the_real_model_hears_a_smoke_alarm_beep():
    if not _model_ready():
        pytest.skip("YAMNet model or onnxruntime unavailable")
    sr = 16000
    t = np.arange(int(WINDOW_S * sr)) / sr
    beep = 0.5 * np.sin(2 * np.pi * 3150 * t) * ((t % 0.5) < 0.25)
    pcm = (beep * 32767).astype(np.int16).tobytes()
    heard = []
    watch = SoundWatch(heard.append, source=_Source(pcm))
    assert watch._prepare()
    watch.step(now=0.0)
    watch.step(now=1.0)
    assert [a.category for a in heard] == ["smoke alarm"]


def test_the_tool_starts_reports_and_stops_the_watch(monkeypatch):
    if not _model_ready():
        pytest.skip("YAMNet model or onnxruntime unavailable")
    from orion_core.dispatch_vision import VisionDispatchMixin

    monkeypatch.setattr(sound_watch, "_WATCH", None)

    class _Signal:
        def __init__(self):
            self.seen = []

        def emit(self, *a):
            self.seen.append(a)

    class _Bus:
        def __init__(self):
            self.log, self.dashboard_event, self.speak_request = _Signal(), _Signal(), _Signal()

    host = VisionDispatchMixin.__new__(VisionDispatchMixin)
    host.bus = _Bus()

    async def flow():
        started = await host.sound_sense_tool({"action": "watch"})
        status = await host.sound_sense_tool({"action": "watch_status"})
        stopped = await host.sound_sense_tool({"action": "watch_off"})
        return started, status, stopped

    started, status, stopped = asyncio.run(flow())
    assert started.ok and "smoke alarms" in started.text
    assert "is on" in status.text
    assert stopped.ok and not sound_watch._WATCH.running
    # The consumer the tool installed speaks and feeds the dashboard.
    sound_watch._WATCH.on_alert(sound_watch.Alert("doorbell", "Doorbell", 0.8, 0.0))
    assert host.bus.speak_request.seen and host.bus.dashboard_event.seen[0][0] == "sound_alert"
