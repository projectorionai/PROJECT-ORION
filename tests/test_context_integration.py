"""Context lifecycle over real producer seams, using synthetic device input."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from orion_core.context_events import AudioObservation
from orion_core.context_service import ContextService
from orion_core.perception import PerceptionLoop
from orion_core.resource_governor import ResourceGovernor
from orion_core.sound_watch import SoundWatch, WINDOW_S


class Signal:
    def __init__(self):
        self.events = []
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot):
        self.slots.remove(slot)

    def emit(self, *args):
        self.events.append(args)
        for slot in list(self.slots):
            slot(*args)


class Bus:
    def __init__(self):
        self.log, self.dashboard_event, self.context_event = Signal(), Signal(), Signal()


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def test_source_stop_rejects_late_results_even_after_a_restart():
    bus, clock = Bus(), Clock()
    context = ContextService(bus, clock=clock)
    old = context.open_source("audio.local")
    old.available(True)
    assert old.audio("doorbell", 0.8)
    old.close()
    assert context.snapshot()["events"] == []
    new = context.open_source("audio.local")
    new.available(True)
    assert not old.audio("old alarm", 0.9)
    old.close(failed=True)
    assert new.audio("new alarm", 0.9)
    assert context.snapshot()["sources"]["audio.local"]["state"] == "active"
    assert len(bus.context_event.events) == 2
    context.close()
    assert not new.audio("late", 0.9)
    assert context.snapshot()["events"] == []
    assert all(s["state"] == "off" for s in context.snapshot()["sources"].values())


def test_unavailable_and_stalled_sources_lose_current_context():
    clock = Clock()
    context = ContextService(clock=clock)
    session = context.open_source("camera.local")
    assert not session.visual("appeared")  # no usable frame yet
    session.available(True)
    assert session.visual("appeared")
    clock.now += 6
    snapshot = context.snapshot()
    assert snapshot["sources"]["camera.local"]["state"] == "degraded"
    assert snapshot["events"] == []
    session.available(True)
    assert session.visual("returned")
    session.available(False)
    assert not session.visual("appeared")
    assert context.snapshot()["events"] == []
    assert "degraded" in context.report()


def test_source_metadata_is_minimal_typed_and_excludes_descriptions():
    context = ContextService(clock=Clock())
    camera = context.open_source("camera.local")
    camera.available(True)
    assert camera.visual("scene_changed")
    assert not camera.visual("described")
    with pytest.raises(ValueError, match="match its source"):
        camera.publish(AudioObservation("speech"), confidence=0.9)
    with pytest.raises(ValueError, match="unknown context source"):
        context.open_source("screen")
    [event] = context.snapshot()["events"]
    assert event["payload"] == {"kind": "visual", "label": "scene changed", "region": None}
    assert event["sensitivity"] == "personal"
    assert event["evidence_ref"] is None
    assert "fixed heuristic" in context.report()


def test_governor_samples_publish_without_renewing_a_failed_probe():
    readings = iter([(71.0, 2000.0), None])
    context = ContextService(clock=Clock())
    governor = ResourceGovernor(reader=lambda: next(readings),
                                cpu_reader=lambda: 25.0, gpu_reader=lambda: None)
    governor.add_listener(context.observe_resources)
    governor.tick()
    assert {e["payload"]["metric"] for e in context.snapshot()["events"]} == {
        "memory_used_percent", "cpu_percent"}
    governor.tick()
    assert context.snapshot()["events"] == []
    assert context.snapshot()["sources"]["system.monitor"]["state"] == "degraded"


async def test_real_camera_loop_publishes_local_change_and_clears_on_stop():
    bus = Bus()
    context = ContextService(bus)
    blank = np.zeros((48, 64), dtype=np.uint8)
    changed = blank.copy()
    changed[10:24, 10:24] = 220
    frames = iter([blank, changed])
    cloud_calls = []

    async def describe(_frame):
        cloud_calls.append(True)
        return "must not be called without consent"

    loop = PerceptionLoop(bus, frame_source=lambda: next(frames, changed),
                          describe=describe, context_service=context, fps=10)
    try:
        loop.start()
        for _ in range(100):
            if context.snapshot()["events"]:
                break
            await asyncio.sleep(0.01)
        assert context.snapshot()["events"]
        assert cloud_calls == []
        assert all(e["payload"]["kind"] == "visual" for e in context.snapshot()["events"])
    finally:
        await loop.stop()
    assert context.snapshot()["events"] == []
    assert context.snapshot()["sources"]["camera.local"]["state"] == "off"


async def test_missing_camera_frame_is_degraded_not_fresh_context():
    context = ContextService()
    loop = PerceptionLoop(Bus(), frame_source=lambda: None, context_service=context)
    try:
        loop.start()
        await asyncio.sleep(0.02)
        assert loop.status()["source_state"] == "degraded"
        assert "unavailable or stale" in loop.report()
    finally:
        await loop.stop()


async def test_stop_releases_an_owned_camera_even_when_its_loop_already_finished():
    camera = SimpleNamespace(stopped=False)

    def stop_camera():
        camera.stopped = True

    camera.stop = stop_camera
    loop = PerceptionLoop(Bus(), camera=camera)
    loop._owns_camera = True
    await loop.stop()
    assert camera.stopped and not loop._owns_camera


async def test_pressure_prevents_starting_sensors_or_loading_the_sound_model(monkeypatch):
    from orion_core import resource_governor
    from orion_core.dispatch_vision import VisionDispatchMixin

    monkeypatch.setattr(resource_governor, "optional_work_allowed", lambda: False)
    context = ContextService()
    watch = _sound(context)
    prepared = []
    monkeypatch.setattr(watch, "_prepare", lambda: prepared.append(True))
    assert not watch.start()[0] and not prepared
    dispatcher = VisionDispatchMixin()
    dispatcher.perception = PerceptionLoop(Bus(), frame_source=lambda: None,
                                           context_service=context)
    result = await dispatcher.perception_tool({"action": "start"})
    assert not result.ok and "memory pressure" in result.text
    assert all(s["state"] == "off" for s in context.snapshot()["sources"].values())


def _sound(context, on_alert=lambda _alert: None, classify=None):
    pcm = np.full(int(WINDOW_S * 16000), 12000, dtype=np.int16).tobytes()
    watch = SoundWatch(on_alert, source=SimpleNamespace(snapshot=lambda _seconds: pcm),
                       labels=["Doorbell"], classify=classify or (lambda _wave: np.array([0.9])),
                       interval_s=0.25)
    watch.context_service = context
    return watch


def test_real_sound_thread_publishes_and_stop_clears_source():
    context = ContextService()
    delivered = threading.Event()
    watch = _sound(context, lambda _alert: delivered.set())
    try:
        assert watch.start()[0]
        assert delivered.wait(2.0)
        [event] = context.snapshot()["events"]
        assert event["source"] == "audio.local"
        assert event["payload"]["label"] == "doorbell"
    finally:
        watch.stop()
    assert not watch.running
    assert context.snapshot()["events"] == []


def test_classifier_finishing_after_revoke_cannot_publish_or_announce():
    context = ContextService()
    entered, release = threading.Event(), threading.Event()
    announced = []

    def classify(_wave):
        entered.set()
        assert release.wait(3)
        return np.array([0.9])

    watch = _sound(context, announced.append, classify)
    try:
        assert watch.start()[0] and entered.wait(2)
        stopper = threading.Thread(target=watch.stop)
        stopper.start()
        assert watch._stop.wait(1)
        release.set()
        stopper.join(2)
        assert not stopper.is_alive()
        assert announced == [] and context.snapshot()["events"] == []
    finally:
        release.set()
        watch.stop()


def test_failed_sound_model_reports_degraded_without_starting(monkeypatch):
    context = ContextService()
    watch = _sound(context)
    monkeypatch.setattr(watch, "_prepare", lambda: False)
    assert not watch.start()[0]
    assert not watch.running
    assert context.snapshot()["sources"]["audio.local"]["state"] == "degraded"


async def test_awareness_and_proactive_reports_read_current_context(monkeypatch):
    from orion_core.cognitive_loop import CognitiveLoopManager
    from orion_core.proactive import ProactiveIntelligence

    bus = Bus()
    context = ContextService(bus, clock=Clock())
    session = context.open_source("audio.local")
    session.available(True)
    session.audio("doorbell", 0.9)
    memory = SimpleNamespace(active_project="")
    cognition = SimpleNamespace(snapshot=lambda: {})
    awareness = CognitiveLoopManager(bus, memory, cognition, context_service=context)
    proactive = ProactiveIntelligence(Bus(), None, None, memory, context_service=context)

    async def survey(announce):
        return []

    monkeypatch.setattr(proactive, "_survey", survey)
    assert "doorbell" in awareness.situation_report().text
    assert "doorbell" in (await proactive.check_now()).text
    assert awareness.last_digest()["context"]["events"]
    session.audio("knocking", 0.8)
    assert any(channel == "awareness" for channel, _data in bus.dashboard_event.events)
    old_event = bus.context_event.events[-1][0]
    session.close()
    count = len(bus.dashboard_event.events)
    awareness._context_changed(old_event)
    assert len(bus.dashboard_event.events) == count
    assert "doorbell" not in awareness.situation_report().text
    assert "doorbell" not in (await proactive.check_now()).text
    assert awareness.last_digest()["context"]["events"] == []
    awareness.stop()
    assert not bus.context_event.slots


def test_quiet_source_heartbeat_updates_visible_freshness_without_events():
    clock, bus = Clock(), Bus()
    context = ContextService(bus, clock=clock)
    session = context.open_source("camera.local")
    session.available(True)
    clock.now += 2
    session.available(True)
    _channel, state = bus.dashboard_event.events[-1]
    assert state["sources"]["camera.local"]["fresh_until"] == clock.now + 5
    assert state["events"] == []


async def test_automatic_camera_cloud_calls_need_a_separate_bounded_grant():
    calls = []

    async def describe(frame):
        calls.append(frame)
        return "cloud description"

    loop = PerceptionLoop(Bus(), describe=describe)
    await loop._understand("frame", "test", 1.0)
    assert calls == []
    loop.set_cloud_sharing(True, max_calls=1)
    await loop._understand("frame", "test", 2.0)
    await loop._understand("frame", "test", 3.0)
    assert calls == ["frame"]
    assert loop.status()["cloud_calls_remaining"] == 0
    await loop.stop()
    assert not loop.cloud_enabled
    await loop._understand("frame", "test", 4.0)
    assert calls == ["frame"]


async def test_revoked_cloud_result_is_discarded_and_cloud_action_requires_consent():
    from orion_core.dispatch_vision import VisionDispatchMixin
    from orion_core.concurrency import classify, ToolClass

    started, released = asyncio.Event(), asyncio.Event()

    async def describe(_frame):
        started.set()
        await released.wait()
        return "late cloud result"

    loop = PerceptionLoop(Bus(), describe=describe)
    dispatcher = VisionDispatchMixin()
    dispatcher.perception = loop
    assert not (await dispatcher.perception_tool({"action": "cloud_on"})).ok
    assert not (await dispatcher.perception_tool({"action": "cloud_on", "consent": "true"})).ok
    assert not (await dispatcher.perception_tool({"action": "cloud_on", "consent": True,
                                                  "max_calls": 1000})).ok
    assert (await dispatcher.perception_tool({"action": "cloud_on", "consent": True,
                                              "max_calls": 2})).ok
    pending = asyncio.create_task(loop._understand("frame", "test", 1.0))
    await started.wait()
    assert (await dispatcher.perception_tool({"action": "cloud_off"})).ok
    released.set()
    await pending
    assert not loop.last_description and not loop.events
    for action in ("cloud_on", "cloud_off"):
        assert classify("perception", {"action": action}) is not ToolClass.PARALLEL
