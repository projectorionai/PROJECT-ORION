"""Camera routing and cancellation at subsystem boundaries, without hardware."""
import asyncio
from types import SimpleNamespace

import numpy as np

from orion_core.data import ToolResult
from orion_core.vision import VisionAgent
from orion_core.local_brain import LocalBrain
from orion_core.dispatch_vision import VisionDispatchMixin


class Signal:
    def __init__(self):
        self.events = []

    def emit(self, payload):
        self.events.append(payload)


def _vision(source):
    vision = VisionAgent.__new__(VisionAgent)
    vision.bus = SimpleNamespace(log=Signal(), camera_frame=Signal())
    vision._live_frame_source = source
    return vision


def test_explicit_camera_does_not_borrow_another_cameras_frame():
    class Tracker:
        camera_index = 0

        def latest_frame(self):
            raise AssertionError("This is the wrong camera")

    vision = _vision(Tracker().latest_frame)
    opened = []
    vision._open_camera = lambda index: (opened.append(index) or None, "")
    result = vision._capture_live_frame_sync(2, 640)
    assert not result.ok and opened == [2]


def test_warming_camera_does_not_open_a_second_handle(monkeypatch):
    class Tracker:
        camera_index = 0

        def is_capturing(self):
            return True

        def latest_frame(self):
            return None

    vision = _vision(Tracker().latest_frame)
    vision._open_camera = lambda index: (_ for _ in ()).throw(AssertionError("Second handle"))
    monkeypatch.setattr("orion_core.vision.time.sleep", lambda _: None)
    result = vision._capture_live_frame_sync(0, 640)
    assert not result.ok and "fresh frame" in result.text


def test_stopped_tracker_does_not_delay_on_demand_capture(monkeypatch):
    class Tracker:
        camera_index = 0

        def is_capturing(self):
            return False

        def latest_frame(self):
            raise AssertionError("Stopped tracker should not be polled")

    vision = _vision(Tracker().latest_frame)
    capture = SimpleNamespace(read=lambda: (True, np.zeros((48, 64, 3), np.uint8)), release=lambda: None)
    vision._open_camera = lambda index: (capture, "fixture")
    monkeypatch.setattr("orion_core.vision.time.sleep", lambda _: (_ for _ in ()).throw(AssertionError("Unexpected warm-up delay")))
    assert vision._capture_live_frame_sync(0, 640).ok


def test_cancelled_voice_scan_publishes_terminal_event():
    async def scenario():
        async def inspect(*args, **kwargs):
            raise asyncio.CancelledError

        bus = SimpleNamespace(**{f"electronics_scan_{name}": Signal() for name in ("started", "result", "error")})
        dispatcher = SimpleNamespace(bus=bus, vision=SimpleNamespace(inspect_electronics=inspect))
        try:
            await VisionDispatchMixin.vision_analyse(dispatcher, {"action": "pcb"})
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation must propagate")
        assert bus.electronics_scan_error.events[0]["request_id"] == bus.electronics_scan_started.events[0]["request_id"]

    asyncio.run(scenario())


def test_offline_electronics_intent_routes_to_real_tool():
    async def scenario():
        calls = []

        async def dispatch(name, args):
            calls.append((name, args))
            return ToolResult("Local checks", evidence=[{"summary": "The markings are unreadable."}])

        brain = LocalBrain.__new__(LocalBrain)
        brain.dispatcher = SimpleNamespace(dispatch=dispatch)
        result = await brain._intent_task("scan this pcb", "Scan this PCB")
        assert calls == [("vision_analyse", {"action": "pcb", "prompt": "Scan this PCB"})]
        assert result == "The markings are unreadable."
        assert await brain._intent_task("explain pcb manufacturing", "Explain PCB manufacturing") is None

    asyncio.run(scenario())
