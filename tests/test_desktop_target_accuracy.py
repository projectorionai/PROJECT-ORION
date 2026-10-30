"""Targeting regressions using recorded input, synthetic displays and UIA."""

import asyncio
import hashlib
import sys
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from orion_core.control import AutonomousControlLayer
from orion_core.display import DisplayTopologyManager, Monitor, Topology, physical_pixels
from orion_core.verification import VisualVerificationEngine
from orion_core.vision import VisionAgent, VolatileScreenGrabber
from test_control_layer import _FakeKB, _FakePG, _StubBus
from test_verification_engine import _StubControl, _StubDisplay, _StubVision, _element
from test_vision_element_matching import _el, _vision


def topology():
    manager = object.__new__(DisplayTopologyManager)
    manager._topology = Topology([
        Monitor(0, 0, 0, 1920, 1080, True, 1.5),
        Monitor(1, -1600, -800, 1600, 900, False, 2.0),
    ], time.monotonic())
    manager.cursor_position = lambda: (40, 50)
    return manager


@pytest.fixture
def layer(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyautogui", _FakePG())
    monkeypatch.setitem(sys.modules, "keyboard", _FakeKB())
    monkeypatch.setenv("ORION_SHOW_WORKING", "0")
    monkeypatch.setenv("ORION_AUTONOMY", "1")
    monkeypatch.setenv("ORION_KEEP_CURSOR", "1")
    return AutonomousControlLayer(_StubBus(), topology())


@pytest.mark.parametrize("action,args", [
    ("click", {"x": -500, "y": 700}),
    ("click", {"x": 999999, "y": 50}),
    ("click", {"x": 50, "y": 50, "monitor": 99}),
    ("click", {"x": -1, "y": 50, "monitor": 1}),
    ("move_cursor", {"x": 50, "y": 50, "monitor": 99}),
    ("drag_cursor", {"x1": 40, "y1": 50, "x2": -500, "y2": 700}),
    ("scroll", {"amount": 1, "x": -500, "y": 700}),
    ("smooth_scroll", {"amount": 1, "x": -500, "y": 700}),
])
def test_invalid_targets_never_move_input_to_another_screen(layer, action, args):
    result = getattr(layer, action)(**args)
    assert not result.ok
    assert layer._pg.calls == []


def test_negative_origin_and_monitor_local_physical_pixels_are_preserved(layer):
    assert layer.click(100, 200, monitor=1).ok
    assert ("click", -1500, -600, 1, "left") in layer._pg.calls
    assert layer.click(-1400, -500).ok
    assert ("click", -1400, -500, 1, "left") in layer._pg.calls


def test_disconnected_monitor_normalisation_is_not_primary_fallback():
    manager = topology()
    assert manager.monitor_by_index(99) is None
    with pytest.raises(ValueError, match="not connected"):
        manager.to_virtual(99, 10, 10)
    with pytest.raises(ValueError, match="not connected"):
        manager.normalise(99, .5, .5)


def test_duplicate_exact_labels_are_ambiguous_and_show_native_ids():
    first = dict(_el("Save", x=0), automation_id="save_document")
    second = dict(_el("Save", x=200), automation_id="save_preferences")
    vision = _vision([first, second])
    result = vision._resolve_element_sync("Save")
    assert result["status"] == "ambiguous" and result["element"] is None
    assert "save_document" in result["detail"] and "save_preferences" in result["detail"]
    assert vision._find_element_sync("Save", "all") is None
    assert vision._resolve_element_sync("Save", automation_id="save_preferences")["element"] == second


def test_shorter_partial_label_does_not_silently_choose_a_different_command():
    result = _vision([_el("Save as"), _el("Save all", x=100)])._resolve_element_sync("Save")
    assert result["status"] == "ambiguous"


@pytest.mark.parametrize("query,label", [("Save", "Saved passwords"), ("Sign in", "Sign information")])
def test_word_prefix_is_not_a_matching_command(query, label):
    assert _vision([_el(label)])._find_element_sync(query, "all") is None


def test_decorative_text_does_not_mask_a_real_button():
    label = dict(_el("Save"), role="Text")
    button = _el("Save", x=100)
    assert _vision([label, button])._resolve_element_sync("Save")["element"] == button


def test_duplicate_wrappers_for_one_identity_do_not_create_false_ambiguity():
    item = dict(_el("Save"), runtime_id=[12, 44])
    assert _vision([item, dict(item)])._resolve_element_sync("Save")["status"] == "found"


def test_ambiguous_native_target_never_falls_back_to_ocr_or_scroll():
    vision = _vision([_el("Submit"), _el("Submit", x=200)])
    async def forbidden(*args, **kwargs):
        pytest.fail("ambiguity must not bypass native target selection")
    vision.find_element_via_ocr = forbidden
    vision.nearby_candidates = forbidden
    control = _StubControl()
    engine = VisualVerificationEngine(_StubBus(), control, vision, _StubDisplay())
    result = asyncio.run(engine.click_text("Submit"))
    assert not result.ok and "Several controls" in result.text
    assert not control.clicks and not control.scrolls


def test_missing_native_id_never_becomes_a_label_search():
    vision = _vision([dict(_el("Save"), automation_id="different")])
    async def forbidden(*args, **kwargs):
        pytest.fail("an explicit ID is never guessed through OCR")
    vision.find_element_via_ocr = forbidden
    control = _StubControl()
    engine = VisualVerificationEngine(_StubBus(), control, vision, _StubDisplay())
    result = asyncio.run(engine.click_text("Save", automation_id="absent"))
    assert not result.ok and not control.clicks and not control.scrolls


def test_stale_target_before_travel_never_moves_or_clicks(layer):
    result = layer.click(100, 200, target_check=lambda: False)
    assert not result.ok and "no click" in result.text
    assert layer._pg.calls == []


def test_hover_rechecks_target_and_never_clicks_a_replacement(layer):
    answers = iter([True, False])
    result = layer.click(100, 200, target_check=lambda: next(answers))
    assert not result.ok and "no click" in result.text
    assert layer._pg.calls == [("moveTo", 100, 200)]


def test_disable_during_glide_prevents_a_click(layer, monkeypatch):
    monkeypatch.setattr(layer, "_travel", lambda *args: layer.set_enabled(False))
    assert not layer.click(100, 200).ok
    assert not any(call[0] == "click" for call in layer._pg.calls)


def test_backend_partial_click_failure_is_never_replayed(layer):
    issued = []
    def fail(*args, **kwargs):
        issued.append(1)
        raise RuntimeError("driver failed after issuing input")
    layer._pg.click = fail
    result = layer.click(100, 200)
    assert not result.ok and "not repeated" in result.text and issued == [1]


def test_successful_native_resolution_enumerates_once(layer, monkeypatch):
    item = dict(_el("Save", x=100, y=100), runtime_id=[1], source_hwnd=42)
    vision = _vision([item])
    calls = []
    vision._detect_elements_sync = lambda *args, **kwargs: calls.append(1) or [item]
    vision.validate_element_sync = lambda element: element == item
    vision.grabber = SimpleNamespace(capture_array=lambda region: object())
    engine = VisualVerificationEngine(_StubBus(), layer, vision, _StubDisplay())
    engine._change_ratio = lambda before, after: .5
    async def immediate(_seconds):
        pass
    monkeypatch.setattr(asyncio, "sleep", immediate)
    assert asyncio.run(engine.click_text("Save")).ok
    assert calls == [1]
    assert len([call for call in layer._pg.calls if call[0] == "click"]) == 1


@pytest.mark.parametrize("phase", ["before", "after"])
def test_capture_fault_reports_truthful_action_state_without_replay(phase, monkeypatch):
    vision = _StubVision([_element("Save", 100, 200)])
    control = _StubControl()
    engine = VisualVerificationEngine(_StubBus(), control, vision, _StubDisplay())
    capture_count = 0
    async def capture(region):
        nonlocal capture_count
        capture_count += 1
        if phase == "before" or capture_count == 2:
            raise RuntimeError("capture unavailable")
        return object()
    async def immediate(_seconds):
        pass
    engine._capture = capture
    monkeypatch.setattr(asyncio, "sleep", immediate)
    result = asyncio.run(engine.click_element("Save"))
    assert not result.ok
    assert len(control.clicks) == (0 if phase == "before" else 1)


class NativeControl:
    def __init__(self, identity, name="Save", rect=(100, 100, 40, 20), enabled=True, parent=None):
        self.element_info = SimpleNamespace(runtime_id=identity, name=name,
            automation_id="save", process_id=70, control_type="Button", enabled=enabled, visible=True)
        self._rect = rect
        self._parent = parent
    def rectangle(self):
        x, y, w, h = self._rect
        return SimpleNamespace(left=x, top=y, width=lambda: w, height=lambda: h)
    def parent(self):
        return self._parent


@pytest.mark.parametrize("fault", [None, "window", "replacement", "moved", "disabled", "name", "covered", "child"])
def test_native_hit_test_checks_identity_geometry_and_foreground(monkeypatch, fault):
    vision = _vision([])
    target = NativeControl([42])
    element = dict(vision._element_dict(target), source_hwnd=77)
    native = target
    if fault == "replacement":
        native = NativeControl([99])
    elif fault == "moved":
        native = NativeControl([42], rect=(200, 100, 40, 20))
    elif fault == "disabled":
        native = NativeControl([42], enabled=False)
    elif fault == "name":
        native = NativeControl([42], name="Delete")
    elif fault == "covered":
        native = NativeControl([99], name="Overlay")
    elif fault == "child":
        native = NativeControl([99], parent=target)
    events = []
    monkeypatch.setitem(sys.modules, "pythoncom", SimpleNamespace(
        CoInitialize=lambda: events.append("init"), CoUninitialize=lambda: events.append("release")))
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(
        Desktop=lambda **kwargs: SimpleNamespace(from_point=lambda *args: native)))
    vision._uia_available = lambda: True
    vision._foreground_hwnd = lambda: 90 if fault == "window" else 77
    assert vision.validate_element_sync(element) == (fault in {None, "child"})
    assert events == ["init", "release"]


def test_ocr_click_requires_unchanged_native_pixels(monkeypatch):
    image = Image.new("RGB", (40, 20), "white")
    vision = _vision([])
    vision._foreground_hwnd = lambda: 5
    vision.grabber = SimpleNamespace(capture_region=lambda rect: image)
    item = dict(_el("Save"), role="OCR", source_hwnd=5,
                visual_token=hashlib.sha256(image.tobytes()).hexdigest())
    assert vision.validate_element_sync(item)
    image.putpixel((0, 0), (0, 0, 0))
    assert not vision.validate_element_sync(item)


def test_outbound_screenshot_carries_real_scale_and_negative_origin(monkeypatch):
    image = Image.new("RGB", (3840, 1080), "grey")
    vision = VisionAgent(_StubBus(), SimpleNamespace(
        capture_image_with_bounds=lambda **kwargs: (image, (-1920, -100, 3840, 1080))), None)
    vision._exact_window_text = lambda **kwargs: ""
    vision._ocr_image_text = lambda image: ""
    vision._visible_window_titles = lambda: []
    monkeypatch.setattr(VisionAgent, "ocr_available", property(lambda self: False))
    result = vision._analyse_screen_sync("")
    assert result.media["coordinates"] == {
        "space": "screenshot", "origin": [-1920, -100],
        "native_size": [3840, 1080], "image_size": [2560, 720]}
    assert "desktop_x = -1920" in result.text


def test_dxgi_primary_output_cannot_replace_a_multiscreen_coordinate_map(monkeypatch):
    grabber = VolatileScreenGrabber(_StubBus())
    grabber._grabber = lambda: SimpleNamespace(monitors=[{}, {}, {}])
    grabber._looks_blank = lambda image: True
    grabber.capture_dxgi = lambda output: pytest.fail("single-screen DXGI must not replace a panorama")
    original = Image.new("RGB", (3840, 1080))
    assert grabber._unless_blank(original, None) is original


@pytest.mark.parametrize("monitor", [-1, 3, True, "1"])
def test_disconnected_or_invalid_capture_monitor_never_silently_becomes_the_desktop(monitor):
    grabber = VolatileScreenGrabber(_StubBus())
    capture = SimpleNamespace(monitors=[{"left": 0}, {"left": 0}, {"left": 1920}])
    with pytest.raises(ValueError, match="not connected"):
        grabber._monitor(capture, monitor)


def test_jpeg_capture_uses_the_same_dxgi_recovery_as_image_capture():
    import io
    monitors = [{"left": 0, "top": 0, "width": 64, "height": 36}] * 2
    black = Image.new("RGB", (64, 36), "black")
    shot = SimpleNamespace(size=black.size, rgb=black.tobytes())
    grabber = VolatileScreenGrabber(_StubBus())
    grabber._grabber = lambda: SimpleNamespace(monitors=monitors, grab=lambda _target: shot)
    grabber.capture_dxgi = lambda output: Image.new("RGB", black.size, "white")
    # Live and screen_read use JPEG. A blank fullscreen GPU surface must use
    # the same recovery as local OCR and screen analysis.
    with Image.open(io.BytesIO(grabber.capture_jpeg(max_side=0))) as decoded:
        assert decoded.size == black.size
        assert decoded.getpixel((0, 0))[0] > 240


def test_dxgi_matches_the_requested_monitor_on_another_adapter(monkeypatch):
    import numpy as np
    primary = {"left": 0, "top": 0, "width": 64, "height": 36}
    secondary = {"left": -64, "top": 0, "width": 64, "height": 36}
    grabber = VolatileScreenGrabber(_StubBus())
    # mss monitor one is secondary, while the first DXGI output is primary.
    grabber._grabber = lambda: SimpleNamespace(monitors=[{}, secondary, primary])
    attempts = []

    def camera(**kwargs):
        attempts.append((kwargs["device_idx"], kwargs["output_idx"]))
        x = 0 if kwargs["device_idx"] == 0 else -64
        return SimpleNamespace(
            _output=SimpleNamespace(desc=SimpleNamespace(DesktopCoordinates=
                SimpleNamespace(left=x, top=0, right=x + 64, bottom=36))),
            grab=lambda: np.full((36, 64, 3), 180, dtype=np.uint8))

    monkeypatch.setitem(sys.modules, "dxcam", SimpleNamespace(
        create=camera, output_info=lambda: "Device[0] Output[0]: Primary:True\nDevice[1] Output[0]: Primary:False"))
    frame = grabber.capture_dxgi(0)
    assert frame is not None and frame.size == (64, 36)
    assert attempts == [(0, 0), (1, 0)]


def test_worker_dpi_context_restores_the_previous_policy_on_failure(monkeypatch):
    import ctypes
    import orion_core.display as display
    changes = []
    class Setter:
        def __call__(self, value):
            changes.append(value.value if isinstance(value, ctypes.c_void_p) else value)
            return 123
    monkeypatch.setattr(display.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=SimpleNamespace(
        SetThreadDpiAwarenessContext=Setter())), raising=False)
    @physical_pixels
    def broken():
        raise RuntimeError("fault")
    with pytest.raises(RuntimeError, match="fault"):
        broken()
    assert changes == [ctypes.c_void_p(-4).value, 123]
