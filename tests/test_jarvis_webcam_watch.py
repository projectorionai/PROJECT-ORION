"""Motion-aware webcam analysis adapted from JARVIS's watch capability."""

from __future__ import annotations

import asyncio
from io import BytesIO

import pytest
from PIL import Image

from orion_core.data import ToolResult
from orion_core.video_analysis import VideoAnalyser


def _jpeg(colour: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (12, 12), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


class _Vision:
    def _ocr_image_text(self, image: Image.Image) -> str:
        red, _green, blue = image.getpixel((0, 0))
        return "hand raised" if red > blue else "hand lowered"


class _Router:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_text(self, prompt: str):
        self.prompts.append(prompt)
        return None, ("SUMMARY: The hand moves from low to high.\n"
                      "ACTION: Keep the wrist level.")


def test_watch_camera_uses_ordered_frames_without_claiming_audio(monkeypatch):
    router = _Router()
    analyser = VideoAnalyser(router=router, vision=_Vision())
    captures: list[tuple[float, int, int | None]] = []

    def sample(seconds, frames, index, _stop=None):
        captures.append((seconds, frames, index))
        return [(0.0, _jpeg((0, 0, 255))),
                (2.0, _jpeg((255, 0, 0)))], ""

    monkeypatch.setattr(analyser, "_sample_camera", sample)
    report = asyncio.run(analyser.watch_camera(30, "check my movement",
                                                max_frames=20, camera_index=3))
    assert captures == [(15.0, 8, 3)]
    assert [frame.text for frame in report.frames] == ["hand lowered", "hand raised"]
    assert report.action == "Keep the wrist level."
    assert "No audio was captured" in router.prompts[0]
    assert "hand lowered" in router.prompts[0]
    assert "hand raised" in router.prompts[0]
    assert "TRANSCRIPT" not in router.prompts[0]


def test_multiple_camera_frames_cost_one_vision_call(monkeypatch):
    class VisionRouter:
        def __init__(self):
            self.calls = []

        async def generate_vision(self, image, prompt, **options):
            self.calls.append((image, prompt, options))
            return None, "The hand rises between frames. Keep the wrist level."

    router = VisionRouter()
    analyser = VideoAnalyser(router=router, vision=_Vision())
    monkeypatch.setattr(analyser, "_sample_camera", lambda *_args: (
        [(0.0, _jpeg((0, 0, 255))), (1.0, _jpeg((255, 0, 0)))], ""))
    report = asyncio.run(analyser.watch_camera(2, "check my movement"))
    assert report.captured_frames == 2
    assert len(router.calls) == 1
    assert router.calls[0][0].startswith(b"\xff\xd8")
    assert "check my movement" in router.calls[0][1]
    assert "The hand rises" in report.summary
    assert report.frames[0].via == "model"


def test_camera_failure_reports_the_actual_capture_problem(monkeypatch):
    analyser = VideoAnalyser(vision=_Vision())
    monkeypatch.setattr(analyser, "_sample_camera",
                        lambda *_args: ([], "camera blocked by Windows"))
    report = asyncio.run(analyser.watch_camera(0, max_frames=0))
    assert report.duration == 2
    assert report.frames == []
    assert "camera blocked by Windows" in report.notes


def test_camera_capture_borrows_the_live_tracker_and_never_opens_second_handle(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from orion_core import camera_devices, utils
    import time

    class Tracker:
        camera_index = 2

        def __init__(self):
            self.calls = 0

        def is_capturing(self):
            return True

        def frame(self):
            self.calls += 1
            return np.full((20, 30, 3), self.calls * 40, dtype=np.uint8)

    tracker = Tracker()
    vision = type("Vision", (), {"_live_frame_source": tracker.frame})()
    monkeypatch.setattr(camera_devices, "resolve", lambda: 2)
    monkeypatch.setattr(utils, "open_camera_capture",
                        lambda *_args: pytest.fail("a second camera handle was opened"))
    monkeypatch.setattr(time, "sleep", lambda *_args: None)
    frames, problem = VideoAnalyser(vision=vision)._sample_camera(2, 2, None)
    assert len(frames) == 2
    assert all(jpeg.startswith(b"\xff\xd8") for _, jpeg in frames)
    assert tracker.calls == 2
    assert problem == ""


def test_vision_tool_routes_explicit_watch_camera_and_advertises_it():
    from orion_core.concurrency import ToolClass, classify
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.dispatch_vision import VisionDispatchMixin

    class Dispatcher(VisionDispatchMixin):
        async def _watch_camera(self, prompt, args):
            return ToolResult(f"watched: {prompt}; {args['seconds']}")

    result = asyncio.run(Dispatcher().vision_analyse(
        {"action": "watch_camera", "prompt": "my form", "seconds": 6}))
    assert result.ok
    assert result.text == "watched: my form; 6"
    schema = next(tool for tool in TOOL_DECLARATIONS
                  if tool["name"] == "vision_analyse")
    assert "watch_camera" in schema["description"]
    assert "watch_camera" in schema["parameters"]["properties"]["action"]["description"]
    assert classify("vision_analyse", {"action": "watch_camera"}) is ToolClass.SERIAL


def test_recent_camera_history_is_bounded_and_erased_on_stop(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from orion_core.perception import PerceptionLoop

    monkeypatch.setenv("ORION_CAMERA_HISTORY", "1")

    now = [0.0]
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    watcher = PerceptionLoop(bus, frame_source=lambda: None,
                             lab=SimpleNamespace(reset=lambda: None),
                             clock=lambda: now[0])
    watcher._task = SimpleNamespace(done=lambda: False)
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    for moment in range(19):
        now[0] = float(moment)
        watcher._remember_camera_frame(frame, now[0])
    recent = watcher.recent_camera_frames(15, 8)
    assert len(recent) == 8
    assert all(jpeg.startswith(b"\xff\xd8") for _, jpeg in recent)
    assert len(watcher._camera_history) <= 16
    assert watcher._camera_history[0][0] >= 3.0
    watcher._task = None
    assert watcher.recent_camera_frames() == []
    asyncio.run(watcher.stop())
    assert not watcher._camera_history


def test_recent_camera_reuses_ordered_frame_description():
    class Router:
        def __init__(self):
            self.calls = 0

        async def generate_vision(self, image, prompt, **options):
            self.calls += 1
            assert "what changed" in prompt.lower()
            return None, "The hand rose before the cup moved."

    router = Router()
    analyser = VideoAnalyser(router=router, vision=_Vision())
    report = asyncio.run(analyser.analyse_recent_camera(
        [(0.0, _jpeg((0, 0, 255))), (1.0, _jpeg((255, 0, 0)))],
        "what changed?"))
    assert report.summary == "The hand rose before the cup moved."
    assert report.captured_frames == 2 and router.calls == 1


def test_past_camera_request_needs_an_existing_watch():
    from types import SimpleNamespace
    from orion_core.dispatch_vision import VisionDispatchMixin

    class Dispatcher(VisionDispatchMixin):
        perception = SimpleNamespace(running=False)

    result = asyncio.run(Dispatcher().vision_analyse(
        {"action": "watch_camera_past", "prompt": "what just happened?"}))
    assert not result.ok
    assert "Start local camera watching" in result.text
