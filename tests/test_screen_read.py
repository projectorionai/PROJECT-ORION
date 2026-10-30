"""
Shoulder (CAP-03) — read the active screen on request, describe it, act on it.

    "analyse images ... identify what course of action to take dependent on the
     prompt."
"""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.screen_read import ScreenReader, ScreenReading  # noqa: E402


PNG = b"\x89PNG\r\n\x1a\n fake image bytes"


def _reader(**kw):
    kw.setdefault("capture", lambda: PNG)
    return ScreenReader(**kw)


async def test_reads_and_describes_the_screen():
    async def describe(img, prompt):
        assert img == PNG
        return "A VS Code window with a Python traceback.", "model"

    r = await _reader(describe=describe).read("what's the error?")
    assert r.ok
    assert "traceback" in r.description.lower()
    assert r.via == "model"


async def test_prompt_drives_an_action():
    async def describe(img, prompt):
        return "A login form asking for an API key.", "model"

    async def generate(ask):
        assert "login form" in ask          # the description is handed to the model
        assert "API key" in ask
        return "Paste the key from your password manager into the field."

    r = await _reader(describe=describe, generate=generate).read("what do I do here?")
    assert r.ok
    assert "password manager" in r.action
    assert "password manager" in r.describe()


async def test_no_describer_degrades_gracefully():
    r = await _reader(describe=None).read("anything")
    assert not r.ok
    assert "no way to see" in r.note.lower()


async def test_capture_failure_is_reported_not_raised():
    def boom():
        raise RuntimeError("no display")
    r = await ScreenReader(capture=boom).read("x")
    assert not r.ok
    assert "capture" in r.note.lower()


async def test_empty_capture_is_handled():
    r = await ScreenReader(capture=lambda: b"").read("x")
    assert not r.ok


async def test_a_read_is_logged_once():
    logged = []

    async def describe(img, prompt):
        return "the desktop", "ocr"

    await _reader(describe=describe, log=logged.append).read("hi")
    assert any("screen" in m.lower() for m in logged)


async def test_no_passive_capture_without_a_call():
    # Constructing the reader must NOT capture anything — privacy model.
    calls = []
    ScreenReader(capture=lambda: calls.append(1) or PNG)
    assert calls == []      # nothing captured until read() is awaited


async def test_capture_runs_off_the_event_loop_thread():
    event_thread = threading.get_ident()

    def capture():
        assert threading.get_ident() != event_thread
        return PNG

    async def describe(_image, _prompt):
        assert threading.get_ident() == event_thread
        return "Desktop text.", "model"

    assert (await ScreenReader(capture=capture, describe=describe).read()).ok


def test_default_capture_returns_a_real_jpeg_with_readable_panorama_resolution(monkeypatch):
    import mss
    from orion_core.screen_read import default_capture
    image = Image.new("RGB", (3840, 1080), "white")

    class Capture:
        monitors = [{"left": -1920, "top": 0, "width": 3840, "height": 1080}]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def grab(self, monitor):
            assert monitor == self.monitors[0]
            return SimpleNamespace(size=image.size, rgb=image.tobytes())

    monkeypatch.setattr(mss, "MSS", Capture, raising=False)
    with Image.open(io.BytesIO(default_capture())) as frame:
        assert frame.format == "JPEG"   # matches the model's image/jpeg payload
        assert frame.size == (2560, 720)


def _screen_dispatcher(*, router=None, vision=None, grabber=None):
    from orion_core.dispatch_vision import VisionDispatchMixin
    dispatcher = VisionDispatchMixin()
    dispatcher.bus = None
    dispatcher.ai = None
    dispatcher.router = router
    dispatcher.vision = vision or SimpleNamespace()
    dispatcher.grabber = grabber
    return dispatcher


async def test_screen_tool_sends_screenshot_pixels_and_keeps_the_complete_answer():
    calls = []
    text = "Visible document text.\n" + "a complete paragraph " * 40

    def capture_image(**kwargs):
        calls.append(kwargs)
        return Image.new("RGB", (3840, 1080), "white")

    async def generate_vision(image, request, **kwargs):
        with Image.open(io.BytesIO(image)) as frame:
            assert frame.format == "JPEG"
            assert frame.size == (2560, 720)
        assert "SCREENSHOT" in request
        assert "video frame" not in request
        assert kwargs["max_tokens"] >= 1000
        return None, text

    dispatcher = _screen_dispatcher(
        grabber=SimpleNamespace(capture_image=capture_image),
        router=SimpleNamespace(generate_vision=generate_vision))
    result = await dispatcher.screen_read_tool({})
    assert result.ok and result.text == text.strip()
    assert len(calls) == 1


async def test_screen_read_ocr_fallback_uses_the_native_frame_and_selected_display():
    calls = []

    def capture_image(**kwargs):
        calls.append(kwargs)
        return Image.new("RGB", (3840, 1080), "white")

    def read_image(image):
        assert image.size == (3840, 1080)   # before any outbound scaling
        return "Small text visible on the second monitor."

    vision = SimpleNamespace(
        _ocr_image_text=read_image,
        _exact_window_text=lambda **_kwargs: pytest.fail("another monitor's foreground text must not be used"))
    dispatcher = _screen_dispatcher(vision=vision,
        grabber=SimpleNamespace(capture_image=capture_image))
    result = await dispatcher.screen_read_tool({"monitor": 1})
    assert result.ok and "second monitor" in result.text
    assert calls == [{"max_side": 0, "monitor": 2}]


async def test_a_failed_vision_model_still_reads_exact_application_text():
    async def generate_vision(*_args, **_kwargs):
        raise RuntimeError("provider disconnected")

    dispatcher = _screen_dispatcher(
        router=SimpleNamespace(generate_vision=generate_vision),
        vision=SimpleNamespace(_exact_window_text=lambda **_kwargs: "The exact typed text."),
        grabber=SimpleNamespace(capture_image=lambda **_kwargs: Image.new("RGB", (80, 60))))
    result = await dispatcher.screen_read_tool({})
    assert result.ok and result.text == "The exact typed text."


@pytest.mark.parametrize("monitor", [-1, True, "other", 1.5])
async def test_invalid_screen_monitor_is_rejected_before_capture(monitor):
    dispatcher = _screen_dispatcher(grabber=SimpleNamespace(
        capture_image=lambda **_kwargs: pytest.fail("invalid target must not capture")))
    result = await dispatcher.screen_read_tool({"monitor": monitor})
    assert not result.ok


async def test_capture_screen_maps_display_zero_and_reports_disconnected_targets():
    calls = []

    def capture_jpeg(**kwargs):
        calls.append(kwargs)
        if kwargs.get("monitor") == 100:
            raise ValueError("That monitor is not connected.")
        return b"jpeg"

    dispatcher = _screen_dispatcher(grabber=SimpleNamespace(capture_jpeg=capture_jpeg))
    result = await dispatcher.capture_screen({"monitor": 0})
    assert result.ok and "monitor 0" in result.text
    assert calls[0]["monitor"] == 1
    assert calls[0]["max_side"] == 2560
    assert not (await dispatcher.capture_screen({"monitor": 99})).ok


async def test_screen_capture_matches_display_bounds_when_backend_order_differs():
    from orion_core.vision import VolatileScreenGrabber
    primary = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    secondary = {"left": -1920, "top": 0, "width": 1920, "height": 1080}
    captured = []
    image = Image.new("RGB", (1920, 1080), "white")

    def grab(target):
        captured.append(target)
        return SimpleNamespace(size=image.size, rgb=image.tobytes())

    grabber = VolatileScreenGrabber(None)
    grabber._grabber = lambda: SimpleNamespace(monitors=[{}, secondary, primary], grab=grab)
    dispatcher = _screen_dispatcher(grabber=grabber)
    dispatcher.display = SimpleNamespace(monitor_by_index=lambda index:
        SimpleNamespace(x=0, y=0, width=1920, height=1080) if index == 0 else None)
    result = await dispatcher.capture_screen({"monitor": 0})
    assert result.ok
    assert captured == [primary]    # display zero is mss monitor two here


# ── the tool wiring ──────────────────────────────────────────────────────────

def test_tool_and_schema_present():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"screen_read"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "screen_read")
    assert "action" in tool["parameters"]["properties"]
    assert "prompt" in tool["parameters"]["properties"]
