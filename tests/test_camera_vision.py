"""
Camera background-identification tests (new ask).

ORION should be able to look through the webcam and identify what's in view —
the person and, especially, the background.  These tests cover the dispatch
routing and the text-shaping, with the actual capture stubbed (no hardware).
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.dispatcher import OrionDispatcher


class _Sig:
    def emit(self, *a):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


class _FakeVision:
    """Stands in for VisionAgent; records the camera call."""

    def __init__(self):
        self.calls = []

    async def analyse_camera(self, prompt="", camera_index=0):
        self.calls.append((prompt, camera_index))
        return ToolResult(
            f"camera[{camera_index}] {prompt}",
            media={"data": b"jpeg", "mime_type": "image/jpeg"})

    async def analyse_screen(self, prompt=""):
        return ToolResult("screen")


def _dispatcher_with_vision(vision):
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _Bus()
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)
    d.vision = vision
    return d


def test_camera_action_routes_to_analyse_camera():
    vision = _FakeVision()
    d = _dispatcher_with_vision(vision)
    result = asyncio.run(d.dispatch("vision_analyse", {"action": "camera"}))
    assert result.ok
    # None, not 0: no index named means "the remembered camera", which the
    # vision agent resolves (and can borrow from the face tracker). A forced 0
    # opened the wrong device and refused the tracker's frame.
    assert vision.calls == [("", None)]
    assert result.media and result.media["mime_type"] == "image/jpeg"


def test_camera_index_is_passed_through():
    vision = _FakeVision()
    d = _dispatcher_with_vision(vision)
    asyncio.run(d.dispatch("vision_analyse",
                            {"action": "background", "camera_index": 2, "prompt": "what's behind me"}))
    assert vision.calls == [("what's behind me", 2)]


def test_bad_camera_index_falls_back_to_the_remembered_camera():
    vision = _FakeVision()
    d = _dispatcher_with_vision(vision)
    asyncio.run(d.dispatch("vision_analyse", {"action": "webcam", "camera_index": "oops"}))
    assert vision.calls == [("", None)]


def test_camera_index_zero_is_still_honoured_when_named():
    vision = _FakeVision()
    d = _dispatcher_with_vision(vision)
    asyncio.run(d.dispatch("vision_analyse", {"action": "camera", "camera_index": 0}))
    assert vision.calls == [("", 0)]


def test_analyse_camera_shapes_identification_prompt(monkeypatch):
    from orion_core.vision import VisionAgent

    agent = VisionAgent.__new__(VisionAgent)

    async def _fake_capture(camera_index=0, max_side=768):
        return ToolResult("raw frame", media={"data": b"x", "mime_type": "image/jpeg"})

    agent.capture_live_frame = _fake_capture           # type: ignore
    result = asyncio.run(agent.analyse_camera())
    assert result.ok
    assert "background" in result.text.lower()
    assert result.media["mime_type"] == "image/jpeg"


def test_analyse_camera_propagates_capture_failure():
    from orion_core.vision import VisionAgent

    agent = VisionAgent.__new__(VisionAgent)

    async def _fail(camera_index=0, max_side=768):
        return ToolResult("no camera", ok=False)

    agent.capture_live_frame = _fail                   # type: ignore
    result = asyncio.run(agent.analyse_camera())
    assert not result.ok


def _bare_vision():
    from orion_core.vision import VisionAgent
    agent = VisionAgent.__new__(VisionAgent)
    agent.bus = _Bus()
    agent._live_frame_source = None
    return agent


def test_capture_borrows_tracker_frame_and_never_opens_a_second_camera(monkeypatch):
    """When a live frame source (the face tracker) is available, on-demand
    capture must BORROW its frame — never open a second handle on the same
    webcam (the -1072873821 MSMF grab failure on Windows)."""
    import cv2
    import numpy as np

    def _boom(*_a, **_k):
        raise AssertionError("VideoCapture must not be opened when a frame is borrowable")

    # Patched on the cv2 module itself rather than through orion_core.vision:
    # vision.py now imports cv2 lazily inside the worker methods (so importing
    # it on every app start doesn't drag OpenCV in), and sys.modules caching
    # means those local imports resolve to this very object.
    monkeypatch.setattr(cv2, "VideoCapture", _boom)

    agent = _bare_vision()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    agent._live_frame_source = lambda: frame

    result = agent._capture_live_frame_sync(0, 640)
    assert result.ok
    assert result.media["mime_type"] == "image/jpeg"
    assert result.media["data"]


def test_capture_falls_back_to_camera_when_no_source(monkeypatch):
    """With no borrowable frame, capture opens the camera itself (DirectShow
    path) and warms up before grabbing."""
    import numpy as np

    agent = _bare_vision()

    class _FakeCap:
        def __init__(self):
            self.reads = 0

        def read(self):
            self.reads += 1
            if self.reads < 2:          # first frame cold/black → dropped
                return False, None
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(agent, "_open_camera", lambda idx: (_FakeCap(), "CAP_DSHOW"))
    result = agent._capture_live_frame_sync(0, 640)
    assert result.ok
    assert result.media["mime_type"] == "image/jpeg"


def test_capture_reports_busy_camera_clearly(monkeypatch):
    """If the camera can't be opened at all, the message must name the likely
    cause (in use / privacy) rather than a bare failure."""
    agent = _bare_vision()
    monkeypatch.setattr(agent, "_open_camera", lambda idx: (None, ""))
    result = agent._capture_live_frame_sync(0, 640)
    assert not result.ok
    assert "in use" in result.text.lower()
