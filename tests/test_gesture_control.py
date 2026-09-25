"""
Tests for GestureEngine — webcam hand-gesture control of PC peripherals.

Hermetic: these never open a camera or download the model. They cover the
pure gesture-classification maths (finger-extension, pinch-to-volume,
swipe detection) directly against synthetic landmark points, plus
start/stop/status lifecycle and graceful degradation when mediapipe is
unavailable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.gesture_control import GestureEngine


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


class _FakePeripherals:
    def __init__(self) -> None:
        self.volume_calls: list[float] = []

    def set_volume(self, level: float) -> ToolResult:
        self.volume_calls.append(level)
        return ToolResult(f"volume {level}")


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
    engine = GestureEngine(bus, peripherals, media_calls.append)
    return engine, bus, peripherals, media_calls


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
    assert "inactive" in result.text.lower()


def test_stop_before_start_is_reported_cleanly():
    engine, _, _, _ = _engine()
    result = engine.stop()
    assert "not active" in result.text.lower()


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

def test_fist_triggers_play_pause_once_per_hold():
    engine, _, _, media_calls = _engine()
    fist = _landmarks([False, False, False, False])
    engine._classify_and_act(fist)
    assert media_calls == ["play_pause"]
    # Holding the same fist must not repeat-fire.
    engine._classify_and_act(fist)
    assert media_calls == ["play_pause"]


def test_releasing_fist_rearms_the_trigger():
    engine, _, _, media_calls = _engine()
    fist = _landmarks([False, False, False, False])
    open_hand = _landmarks([True, True, True, True])
    engine._classify_and_act(fist)
    engine._classify_and_act(open_hand)
    engine._last_gesture_at = 0.0  # bypass the cooldown for the test
    engine._classify_and_act(fist)
    assert media_calls == ["play_pause", "play_pause"]


def test_pinch_shape_maps_distance_to_volume():
    engine, _, peripherals, _ = _engine()
    # index extended, others curled, thumb far from index -> wide pinch
    wide = _landmarks([True, False, False, False], pinch_distance=engine._PINCH_MAX)
    engine._classify_and_act(wide)
    assert peripherals.volume_calls
    assert peripherals.volume_calls[-1] > 0.9

    engine2, _, peripherals2, _ = _engine()
    narrow = _landmarks([True, False, False, False], pinch_distance=engine2._PINCH_MIN)
    engine2._classify_and_act(narrow)
    assert peripherals2.volume_calls[-1] < 0.1


def test_pinch_volume_ignores_sub_threshold_jitter():
    engine, _, peripherals, _ = _engine()
    landmarks = _landmarks([True, False, False, False], pinch_distance=0.15)
    engine._classify_and_act(landmarks)
    first_call_count = len(peripherals.volume_calls)
    # A near-identical pinch distance should not re-trigger set_volume.
    landmarks2 = _landmarks([True, False, False, False], pinch_distance=0.151)
    engine._classify_and_act(landmarks2)
    assert len(peripherals.volume_calls) == first_call_count


def test_open_hand_is_not_read_as_pinch():
    engine, _, peripherals, media_calls = _engine()
    open_hand = _landmarks([True, True, True, True])
    engine._classify_and_act(open_hand)
    assert not peripherals.volume_calls
    assert not media_calls


def test_swipe_triggers_next_track():
    engine, _, _, media_calls = _engine()
    now = time.monotonic()
    open_hand = _landmarks([True, True, True, True])
    open_hand[0] = _point(0.1, 0.9)  # wrist starts on the left
    engine._track_swipe(open_hand, now)
    open_hand[0] = _point(0.1 + engine._SWIPE_MIN_DELTA + 0.05, 0.9)  # moves right
    engine._track_swipe(open_hand, now + 0.1)
    assert media_calls == ["next"]


def test_swipe_left_triggers_previous_track():
    engine, _, _, media_calls = _engine()
    now = time.monotonic()
    open_hand = _landmarks([True, True, True, True])
    open_hand[0] = _point(0.9, 0.9)
    engine._track_swipe(open_hand, now)
    open_hand[0] = _point(0.9 - engine._SWIPE_MIN_DELTA - 0.05, 0.9)
    engine._track_swipe(open_hand, now + 0.1)
    assert media_calls == ["previous"]
