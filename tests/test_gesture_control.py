"""
Tests for GestureEngine — webcam hand-gesture control of PC peripherals.

Hermetic: these never open a camera or download the model. They cover the
pure gesture interpretation (finger-extension, deliberate holds, relative
pinch-drag, completed swipes) against timed synthetic landmark points, plus
start/stop/status lifecycle and graceful degradation when mediapipe is
unavailable.
"""

from __future__ import annotations

import sys
import random
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.gesture_control import DEFAULT_MAPPINGS, GestureEngine, synthetic_hand, synthetic_sequence


@pytest.fixture(autouse=True)
def _private_gesture_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("orion_core.gesture_control.SETTINGS_PATH", tmp_path / "gestures.json")


class _RecordingSignal:
    def __init__(self) -> None:
        self.emitted: list = []

    def emit(self, *args) -> None:
        self.emitted.append(args[0] if len(args) == 1 else args)

    def connect(self, *a, **k) -> None:
        pass


class _FakeBus:
    def __init__(self) -> None:
        self.log = _RecordingSignal()
        self.orb_gesture = _RecordingSignal()


class _FakePeripherals:
    def __init__(self) -> None:
        self.volume_calls: list[float] = []

    def set_volume(self, level: float) -> ToolResult:
        self.volume_calls.append(level)
        return ToolResult(f"volume {level}")

    def change_volume(self, delta: float) -> ToolResult:
        self.volume_calls.append(delta)
        return ToolResult(f"volume changed {delta}")


def _point(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0)


def _landmarks(finger_states: list[bool], pinch_distance: float = 0.15) -> list:
    """Build a synthetic 21-point hand with the four non-thumb fingers set
    extended/curled per finger_states = [index, middle, ring, pinky], and
    the thumb tip placed pinch_distance away from the index tip."""
    lm = [_point(0.5, 0.5) for _ in range(21)]
    lm[0] = _point(0.5, 0.9)  # wrist
    tips = (8, 12, 16, 20)
    pips = (6, 10, 14, 18)
    for tip, pip, extended in zip(tips, pips, finger_states):
        lm[pip] = _point(0.5, 0.6)
        lm[tip] = _point(0.5, 0.4) if extended else _point(0.5, 0.7)  # above/below pip
    # thumb tip near (or far from) the index tip
    index = lm[8]
    lm[4] = _point(index.x + pinch_distance, index.y)
    return lm


def _engine() -> tuple[GestureEngine, _FakeBus, _FakePeripherals, list]:
    bus = _FakeBus()
    peripherals = _FakePeripherals()
    media_calls: list[str] = []
    engine = GestureEngine(bus, peripherals, media_calls.append, camera_index=0)
    engine.action_calls = []
    engine.actions.worker = SimpleNamespace(cancel_turn=lambda: None, toggle_pause=lambda: None)
    engine.actions._on_loop = lambda fn: fn()
    engine.actions._ensure_com = lambda: None

    def action_now(gesture, spec, event):
        engine.action_calls.append(dict(spec))
        engine.actions.perform(spec, event)

    # Exercise the actual action adapter synchronously; there are no detached
    # action threads, real COM calls, voice workers, cameras, or owner settings.
    engine._queue_action = action_now
    return engine, bus, peripherals, media_calls


def _hold(engine, pose, seconds, *, start=0.0, centre=(0.5, 0.55)):
    frames = max(1, round(seconds * 15))
    for index in range(frames):
        engine._handle_hands([synthetic_hand(pose, centre=centre)], start + index / 15)
    return start + frames / 15


# ── availability / lifecycle ───────────────────────────────────────────────

def test_start_refuses_when_mediapipe_unavailable(monkeypatch):
    engine, _, _, _ = _engine()
    monkeypatch.setattr(GestureEngine, "available", False)
    result = engine.start()
    assert isinstance(result, ToolResult) and not result.ok
    assert "mediapipe" in result.text.lower()


def test_status_reports_inactive_before_start():
    engine, _, _, _ = _engine()
    result = engine.status()
    assert result.ok and "off" in result.text.lower()
    assert not engine.peripheral_active()


def test_stop_before_start_is_reported_cleanly():
    engine, _, _, _ = _engine()
    result = engine.stop()
    assert "not active" in result.text.lower()


def test_orb_mode_restores_existing_peripheral_gestures(monkeypatch):
    engine, bus, _, _ = _engine()
    monkeypatch.setattr(GestureEngine, "available", True)
    engine._thread = SimpleNamespace(is_alive=lambda: True)
    assert engine.start_orb().ok
    assert engine._mode == "orb"
    assert engine.stop_orb().ok
    assert engine.peripheral_active()
    assert bus.orb_gesture.emitted[-1] == {"hands": 0}


# ── finger-extension heuristic ─────────────────────────────────────────────

def test_extended_fingers_reads_fist_as_all_curled():
    engine, *_ = _engine()
    landmarks = _landmarks([False, False, False, False])
    assert engine._extended_fingers(landmarks) == [False, False, False, False]


def test_extended_fingers_reads_open_hand_as_all_extended():
    engine, *_ = _engine()
    landmarks = _landmarks([True, True, True, True])
    assert engine._extended_fingers(landmarks) == [True, True, True, True]


# ── gesture classification → actions ───────────────────────────────────────

def test_fist_requires_a_deliberate_hold_and_stops_orion_only_once():
    engine, _, _, media_calls = _engine()
    now = _hold(engine, "fist", 0.3)
    assert engine.action_calls == [], "a transient fist must not cancel a turn"
    now = _hold(engine, "fist", 1.2, start=now)
    assert engine.action_calls == [DEFAULT_MAPPINGS["fist"]] == [{"action": "stop_orion"}]
    _hold(engine, "fist", 2.0, start=now)
    assert engine.action_calls == [{"action": "stop_orion"}], "holding must not repeat-fire"
    assert media_calls == []


def test_releasing_fist_rearms_the_trigger():
    engine, _, _, media_calls = _engine()
    now = _hold(engine, "fist", 1.5)
    assert engine.action_calls.count({"action": "stop_orion"}) == 1
    now = _hold(engine, "open_palm", 1.5, start=now)
    _hold(engine, "fist", 1.5, start=now)
    assert engine.action_calls.count({"action": "stop_orion"}) == 2
    assert media_calls == []


@pytest.mark.parametrize("direction,expected_sign", [(-1, 1), (1, -1)])
def test_pinch_changes_volume_only_after_hold_and_vertical_drag(direction, expected_sign):
    engine, _, peripherals, _ = _engine()
    now = _hold(engine, "pinch", 0.8)
    assert not peripherals.volume_calls, "closing a stationary pinch must not jump the volume"
    for index in range(1, 16):
        hand = synthetic_hand("pinch", centre=(0.5, 0.55 + direction * 0.15 * index / 15))
        engine._handle_hands([hand], now + index / 15)
    assert peripherals.volume_calls and all(delta * expected_sign > 0 for delta in peripherals.volume_calls)
    assert all(abs(delta) == 0.04 for delta in peripherals.volume_calls)


def test_pinch_volume_ignores_sub_threshold_jitter():
    engine, _, peripherals, _ = _engine()
    now = _hold(engine, "pinch", 0.8)
    first_call_count = len(peripherals.volume_calls)
    # Sub-step vertical jitter around an engaged pinch must not move volume.
    for index in range(30):
        hand = synthetic_hand("pinch", centre=(0.5, 0.55 + (0.0002 if index % 2 else -0.0002)))
        engine._handle_hands([hand], now + index / 15)
    assert len(peripherals.volume_calls) == first_call_count


def test_open_hand_is_not_read_as_pinch():
    engine, _, peripherals, media_calls = _engine()
    _hold(engine, "open_palm", 2.0)
    assert not peripherals.volume_calls
    assert not media_calls


def test_two_open_hands_pull_the_orb_apart_without_media_actions():
    engine, bus, peripherals, media_calls = _engine()
    left = _landmarks([True] * 4)
    right = _landmarks([True] * 4)
    for hand, x in ((left, 0.15), (right, 0.85)):
        for point in hand:
            point.x += x - 0.5
    engine._process_orb_hands([left, right])
    sample = bus.orb_gesture.emitted[-1]
    assert sample["hands"] == 2
    assert sample["spread"] > 0.9
    assert abs(sample["x"]) < 0.05
    assert not peripherals.volume_calls and not media_calls


def test_one_hand_moves_the_orb_and_lost_hands_reassemble_it():
    engine, bus, _, _ = _engine()
    hand = _landmarks([True] * 4)
    for point in hand:
        point.x -= 0.3
    engine._process_orb_hands([hand])
    assert bus.orb_gesture.emitted[-1]["hands"] == 1
    assert bus.orb_gesture.emitted[-1]["x"] > 0.5
    engine._process_orb_hands([])
    assert bus.orb_gesture.emitted[-1] == {"hands": 0}


def test_sideways_open_hand_still_moves_orb():
    engine, bus, _, _ = _engine()
    hand = _landmarks([True] * 4)
    for point in hand:
        point.x, point.y = 0.5 + (point.y - 0.5), 0.5 - (point.x - 0.5)
    engine._process_orb_hands([hand])
    assert bus.orb_gesture.emitted[-1]["hands"] == 1


def test_orb_worker_failure_notifies_ui(monkeypatch):
    engine, bus, _, _ = _engine()
    monkeypatch.setattr(GestureEngine, "available", True)
    monkeypatch.setattr(engine, "_loop", lambda: (_ for _ in ()).throw(RuntimeError("model failed")))
    assert engine.start_orb().ok
    engine._thread.join(timeout=1)
    assert bus.orb_gesture.emitted[-1] == {"hands": 0, "stopped": True,
                                          "session": engine.orb_session}


def test_stopping_worker_cannot_be_restarted_until_it_exits(monkeypatch):
    engine, _, _, _ = _engine()
    monkeypatch.setattr(GestureEngine, "available", True)
    engine._mode = "orb"
    engine._thread = SimpleNamespace(is_alive=lambda: True, join=lambda timeout: None)
    assert engine.stop_orb().ok
    assert engine._thread is not None
    assert not engine.start_orb().ok


def test_camera_handoff_releases_own_capture_before_borrowing(monkeypatch):
    import orion_core.gesture_control as module

    engine, _, _, _ = _engine()
    active = [False]
    engine.frame_source = lambda: "borrowed"
    engine.frame_source_active = lambda: active[0]
    engine._ensure_model = lambda: True
    frames = []
    captures = []

    class Capture:
        def __init__(self):
            self.released = False

        def read(self):
            return True, "owned"

        def release(self):
            self.released = True

    def open_capture(_index, *, size, fps):
        assert size == (640, 480) and fps == 15.0
        capture = Capture()
        captures.append(capture)
        return capture, "test"

    timestamps = []

    def process(_mp, _landmarker, frame, timestamp, kind):
        assert kind == "gesture"
        assert not timestamps or timestamp > timestamps[-1]
        timestamps.append(timestamp)
        frames.append(frame)
        if len(frames) == 1:
            active[0] = True
        elif len(frames) == 2:
            assert captures[0].released, "own camera must be released before borrowing"
            active[0] = False
        else:
            engine._stop_event.set()

    engine._process_frame = process
    closed = []
    engine._create_recogniser = lambda: (object(), SimpleNamespace(close=lambda: closed.append(True)), "gesture")
    engine._wake.wait = lambda _timeout: False
    monkeypatch.setattr(module, "open_camera_capture", open_capture)
    engine._loop()
    assert frames == ["owned", "borrowed", "owned"]
    assert len(captures) == 2 and all(capture.released for capture in captures)
    assert closed == [True]


@pytest.mark.parametrize("gesture,action", [("swipe_right", "next"), ("swipe_left", "previous")])
@pytest.mark.parametrize("fps", [15.0, 30.0])
def test_completed_swipe_uses_user_direction_and_triggers_one_track_action(gesture, action, fps):
    engine, _, _, media_calls = _engine()
    for now, hands in synthetic_sequence(gesture, random.Random(7), fps=fps, jitter=0.0):
        engine._handle_hands(hands, now)
    assert media_calls == [action]
    assert engine.action_calls == [{"action": f"media_{action}"}]


def test_pointer_requires_stable_pinch_and_releases_on_hand_loss(monkeypatch):
    import orion_core.gesture_control as module

    bus = _FakeBus()
    actions = []
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    engine = GestureEngine(bus, _FakePeripherals(), lambda *_: None,
                           pointer_action=lambda action, x, y: (
                               actions.append(action) or ToolResult("ok")))
    engine._mode = "pointer"
    pointing = _landmarks([True, False, False, False], pinch_distance=0.02)
    engine._process_pointer_hands([pointing])
    assert "down" not in actions
    now[0] += 0.26
    engine._process_pointer_hands([pointing])
    assert actions.count("down") == 1
    engine._process_pointer_hands([])
    assert actions.count("up") == 1
    engine._process_pointer_hands([])
    assert actions.count("up") == 1


def test_pointer_scroll_needs_two_fingers_and_vertical_motion(monkeypatch):
    import orion_core.gesture_control as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    actions = []
    engine = GestureEngine(_FakeBus(), _FakePeripherals(), lambda *_: None,
                           pointer_action=lambda action, x, y: (
                               actions.append(action) or ToolResult("ok")))
    engine._mode = "pointer"
    fingers = _landmarks([True, True, False, False])
    engine._process_pointer_hands([fingers])
    assert actions == []
    fingers[8].y -= 0.10
    now[0] += 0.35
    engine._process_pointer_hands([fingers])
    assert actions == ["scroll_up"]


def test_hand_pointer_respects_desktop_autonomy_guard():
    from orion_core.control import AutonomousControlLayer

    class Mouse:
        def __init__(self):
            self.actions = []

        def moveTo(self, *args, **kwargs):
            self.actions.append(("move", args))

        def mouseUp(self, **kwargs):
            self.actions.append(("up", kwargs))

    mouse = Mouse()
    control = AutonomousControlLayer.__new__(AutonomousControlLayer)
    control.enabled = False
    control._pg = mouse
    control.display = SimpleNamespace(primary=lambda: SimpleNamespace(
        x=100, y=200, width=800, height=600))
    assert not control.gesture_pointer("ready").ok
    assert not control.gesture_pointer("move").ok
    assert mouse.actions == []
    assert control.gesture_pointer("up").ok
    assert mouse.actions == [("up", {"button": "left"})]
    control.enabled = True
    assert control.gesture_pointer("move", 0.0, 1.0).ok
    assert mouse.actions[-1] == ("move", (899, 799))


def _gesture_tool(engine, args):
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    dispatcher = SimpleNamespace(gestures=engine,
        handler_table=lambda: {"web_search": None, "shutdown_orion": None})
    return DesktopDispatchMixin.gesture_control(dispatcher, args)


def test_gesture_preferences_are_reachable_and_persist_without_a_camera(monkeypatch):
    from orion_core.gesture_control import GestureSettings
    engine, *_ = _engine()
    monkeypatch.setattr(GestureEngine, "available", False)
    listing = _gesture_tool(engine, {"action": "list"})
    assert listing.ok and "Hand gestures:" in listing.text
    assert _gesture_tool(engine, {"action": "map", "gesture": "victory",
                                  "mapped_action": "mute_toggle"}).ok
    assert _gesture_tool(engine, {"action": "sensitivity", "value": "high"}).ok
    assert _gesture_tool(engine, {"action": "low_power", "enabled": True}).ok
    reloaded = GestureSettings(engine.settings.path)
    assert reloaded.mappings["victory"] == {"action": "mute_toggle"}
    assert reloaded.sensitivity == engine.interpreter.sensitivity == 1.4
    assert reloaded.low_power is True
    assert _gesture_tool(engine, {"action": "reset_mappings"}).ok
    assert GestureSettings(engine.settings.path).mappings == DEFAULT_MAPPINGS


def test_gesture_tool_mapping_validates_the_tool_and_preserves_confirmation_gates():
    engine, *_ = _engine()
    result = _gesture_tool(engine, {"action": "map", "gesture": "victory", "tool": "web_search",
                                  "args": '{"query":"test", "confirm":true, "consent":true}'})
    assert result.ok and "left out" in result.text
    assert engine.settings.mappings["victory"] == {"tool": "web_search", "args": {"query": "test"}}
    before = copy.deepcopy(engine.settings.mappings)
    for tool in ("missing_tool", "shutdown_orion"):
        assert not _gesture_tool(engine, {"action": "map", "gesture": "victory", "tool": tool}).ok
    assert engine.settings.mappings == before
    assert not _gesture_tool(engine, {"action": "map", "gesture": "victory", "tool": "web_search",
                                     "args": '["bad"]'}).ok


@pytest.mark.parametrize("action, extra", [("low_power", {"enabled": "false"}),
                                           ("low_power", {}), ("test", {"execute": "false"})])
def test_gesture_tool_does_not_treat_text_false_as_permission(action, extra):
    engine, *_ = _engine()
    assert not _gesture_tool(engine, dict(action=action, **extra)).ok
    assert not engine.settings.low_power and not engine.action_calls


def test_gesture_tests_describe_personal_actions_but_do_not_simulate_owner_answers():
    engine, *_ = _engine()
    assert _gesture_tool(engine, {"action": "test", "gesture": "thumbs_up"}).ok
    result = _gesture_tool(engine, {"action": "test", "gesture": "thumbs_up", "execute": True})
    assert not result.ok and "only your own hand" in result.text
    assert engine.action_calls == []


@pytest.mark.parametrize("data", [[], {"mappings": ["bad"]}, {"mappings": "bad"}])
def test_malformed_gesture_settings_leave_defaults_and_keep_the_file(tmp_path, data):
    from orion_core.gesture_control import GestureSettings
    path = tmp_path / "malformed.json"
    original = json.dumps(data)
    path.write_text(original, encoding="utf-8")
    settings = GestureSettings(path)
    assert settings.load_error and settings.mappings == DEFAULT_MAPPINGS
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("operation", ["map", "reset", "sensitivity", "low_power"])
def test_failed_gesture_setting_write_keeps_the_previous_live_preferences(monkeypatch, operation):
    engine, *_ = _engine()
    assert engine.set_mapping("victory", "mute_toggle").ok
    before_mappings = copy.deepcopy(engine.settings.mappings)
    before_file = engine.settings.path.read_bytes()

    def unavailable():
        raise PermissionError("synthetic write failure")

    monkeypatch.setattr(engine.settings, "save", unavailable)
    operations = {"map": lambda: engine.set_mapping("victory", "media_next"),
                  "reset": engine.reset_mappings,
                  "sensitivity": lambda: engine.set_sensitivity("high"),
                  "low_power": lambda: engine.set_low_power(True)}
    result = operations[operation]()
    assert not result.ok and "nothing changed" in result.text
    assert engine.settings.mappings == before_mappings
    assert engine.settings.sensitivity == engine.interpreter.sensitivity == 1.0
    assert engine.settings.low_power is False
    assert engine.settings.path.read_bytes() == before_file


def test_gesture_calibration_route_rejects_nonfinite_or_invalid_durations():
    engine, *_ = _engine()
    assert not _gesture_tool(engine, {"action": "calibrate"}).ok
    engine._thread = SimpleNamespace(is_alive=lambda: True)
    for seconds in (float("nan"), float("inf"), "bad"):
        assert not _gesture_tool(engine, {"action": "calibrate", "seconds": seconds}).ok
        assert engine._calibrating_until == 0.0
    assert _gesture_tool(engine, {"action": "calibrate", "seconds": 3}).ok
    assert engine._calibrating_until > 0
