"""
Gesture control — webcam hand-tracking mapped to PC-native peripherals
(volume, media playback).

Mirrors FaceTracker's threading shape (orion_core/face_tracking.py) — a
daemon capture-loop thread that never touches Qt. Gesture-to-action is a
local, low-latency loop: it calls PeripheralController.set_volume and the
DesktopAgent media_control callback DIRECTLY, never round-tripping through
the LLM per frame. Only start()/stop()/status() are LLM-invoked, via the
gesture_control dispatcher tool.

Camera sharing: rather than opening a second cv2.VideoCapture handle on the
same webcam FaceTracker may already be using (Windows' MSMF backend does
not reliably support two concurrent handles on one device index), this
engine accepts an optional frame_source callable — the exact "borrow
instead of opening a second handle" convention already used twice elsewhere
(vision.set_live_frame_source, PerceptionLoop(frame_source=...), both in
app.py). If no frame source is given, or it returns nothing, the engine
opens the camera itself, same as FaceTracker does by default.

Uses mediapipe's Tasks API (HandLandmarker) rather than the deprecated
mp.solutions.hands. The model bundle (~7.8MB) is downloaded once to
CONFIG_DIR/models on first use, mirroring the "models download on first
use" convention already documented for Vosk/Whisper in requirements.txt.
"""

from __future__ import annotations

import importlib.util
import time
from collections import deque
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .utils import open_camera_capture


def _mediapipe_available() -> bool:
    """Whether mediapipe can be imported, WITHOUT importing it.

    find_spec only consults the module finders, so this costs nothing —
    where a real module-scope `import mediapipe` would pull in a large
    dependency (and TensorFlow Lite with it) for every process that touches
    this module, gesture control running or not. The real import stays where
    it already was, inside _build_landmarker."""
    return importlib.util.find_spec("mediapipe") is not None

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = CONFIG_DIR / "models" / "hand_landmarker.task"

# Landmark indices (MediaPipe's standard 21-point hand model).
_WRIST = 0
_THUMB_TIP = 4
_FINGER_TIPS = (8, 12, 16, 20)     # index, middle, ring, pinky
_FINGER_PIPS = (6, 10, 14, 18)     # the joint below each tip


class GestureEngine:
    """Webcam hand-gesture control of PC-native peripherals."""

    available = _mediapipe_available()

    # Gesture tuning — starting points, not tuned against real hardware.
    _TARGET_FPS = 20.0
    _PINCH_MIN = 0.03      # normalized distance: fingers touching
    _PINCH_MAX = 0.25      # normalized distance: fingers fully spread
    _VOLUME_STEP_MIN = 0.02  # ignore pinch jitter smaller than this
    _GESTURE_COOLDOWN_S = 1.2   # min gap between discrete triggers (play/pause, swipe)
    _SWIPE_WINDOW_S = 0.5
    _SWIPE_MIN_DELTA = 0.28    # normalized wrist-x displacement within the window

    def __init__(
        self,
        bus: OrionBus,
        peripherals: Any,
        media_control: Callable[[str], Any],
        camera_index: int | None = None,
        frame_source: Callable[[], Any] | None = None,
    ) -> None:
        self.bus = bus
        self.peripherals = peripherals
        self.media_control = media_control
        if camera_index is None:
            from . import camera_devices as cd
            camera_index = cd.resolve()
        self.camera_index = int(camera_index)
        self.frame_source = frame_source

        self._thread: Thread | None = None
        self._stop_event = Event()
        self._last_volume: float | None = None
        self._last_gesture_at = 0.0
        self._wrist_history: deque[tuple[float, float]] = deque(maxlen=32)
        self._fist_held = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> ToolResult:
        if not self.available:
            return ToolResult(
                "Gesture control is unavailable — install mediapipe with "
                "'pip install mediapipe'.", ok=False)
        if self._thread is not None and self._thread.is_alive():
            return ToolResult("Gesture control is already active.")
        self._stop_event.clear()
        self._thread = Thread(target=self._loop, name="orion-gesture-control", daemon=True)
        self._thread.start()
        self.bus.log.emit("GESTURE: starting hand-tracking control.")
        return ToolResult("Watching for hand gestures now — pinch for volume, "
                          "a closed fist to play/pause, swipe to change tracks.")

    def stop(self) -> ToolResult:
        if self._thread is None or not self._thread.is_alive():
            return ToolResult("Gesture control was not active.")
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        self._thread = None
        self.bus.log.emit("GESTURE: stopped.")
        return ToolResult("Gesture control stopped.")

    def status(self) -> ToolResult:
        active = self._thread is not None and self._thread.is_alive()
        return ToolResult("Gesture control is " + ("active." if active else "inactive."))

    # ── model acquisition ───────────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        # Downloaded to a .part file, checked, then renamed. urlretrieve wrote
        # straight onto _MODEL_PATH, so an interrupted download left a truncated
        # model that exists() accepted forever — gestures broken until the
        # file was found and deleted by hand.
        from .model_store import ModelDownloadError, download_verified, usable
        if usable(_MODEL_PATH):
            return True
        try:
            self.bus.log.emit("GESTURE: downloading hand-tracking model (~8MB, once)...")
            download_verified(_MODEL_URL, _MODEL_PATH)
            return True
        except ModelDownloadError as exc:
            self.bus.log.emit(f"GESTURE: model download failed - {exc}")
            return False

    # ── capture loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        if not self._ensure_model():
            self.bus.log.emit("GESTURE: no model available — stopping.")
            return

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision
        import mediapipe as mp

        landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=1,
            )
        )

        own_capture: Any = None
        if self.frame_source is None:
            try:
                own_capture, _backend = open_camera_capture(self.camera_index)
            except Exception as exc:
                self.bus.log.emit(f"GESTURE: camera unavailable - {exc}")
                landmarker.close()
                return

        interval = 1.0 / self._TARGET_FPS
        try:
            while not self._stop_event.is_set():
                frame = self._read_frame(own_capture)
                loop_start = time.monotonic()
                if frame is not None:
                    try:
                        self._process_frame(mp, landmarker, frame)
                    except Exception as exc:
                        self.bus.log.emit(f"GESTURE: frame processing error - {exc}")
                elapsed = time.monotonic() - loop_start
                self._stop_event.wait(max(0.0, interval - elapsed))
        finally:
            landmarker.close()
            if own_capture is not None:
                own_capture.release()

    def _read_frame(self, own_capture: Any) -> Any:
        if self.frame_source is not None:
            try:
                return self.frame_source()
            except Exception:
                return None
        if own_capture is None:
            return None
        ok, frame = own_capture.read()
        return frame if ok else None

    def _process_frame(self, mp: Any, landmarker: Any, frame: Any) -> None:
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            self._fist_held = False
            return
        landmarks = result.hand_landmarks[0]
        self._classify_and_act(landmarks)

    # ── gesture classification (heuristic — tunable starting points) ──────────

    def _classify_and_act(self, landmarks: list) -> None:
        extended = self._extended_fingers(landmarks)
        now = time.monotonic()

        # Fist (all four non-thumb fingers curled) → play/pause on the
        # open->closed transition, with a cooldown so holding the pose
        # doesn't repeat-fire.
        is_fist = extended.count(True) == 0
        if is_fist and not self._fist_held and (now - self._last_gesture_at) > self._GESTURE_COOLDOWN_S:
            self._fist_held = True
            self._last_gesture_at = now
            self.media_control("play_pause")
            return
        if not is_fist:
            self._fist_held = False

        # Pinch (index extended, middle/ring/pinky curled — thumb+index
        # distance maps to volume) — deliberately gated so an open flat
        # hand doesn't get misread as a pinch.
        pinch_shape = extended == [True, False, False, False]
        if pinch_shape:
            self._handle_pinch(landmarks)
            return

        # Open hand, tracked for a horizontal swipe → next/prev track.
        if extended.count(True) >= 3:
            self._track_swipe(landmarks, now)

    def _extended_fingers(self, landmarks: list) -> list[bool]:
        """One bool per non-thumb finger (index, middle, ring, pinky): tip
        above its own PIP joint (smaller normalized y) is read as extended.
        This assumes a roughly upright hand facing the camera — a coarse
        heuristic, good enough to distinguish fist / pinch / open-hand."""
        return [
            landmarks[tip].y < landmarks[pip].y
            for tip, pip in zip(_FINGER_TIPS, _FINGER_PIPS)
        ]

    def _handle_pinch(self, landmarks: list) -> None:
        thumb = landmarks[_THUMB_TIP]
        index = landmarks[_FINGER_TIPS[0]]
        distance = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        span = self._PINCH_MAX - self._PINCH_MIN
        level = (distance - self._PINCH_MIN) / span if span else 0.0
        level = max(0.0, min(1.0, level))
        if self._last_volume is None or abs(level - self._last_volume) >= self._VOLUME_STEP_MIN:
            self._last_volume = level
            self.peripherals.set_volume(level)

    def _track_swipe(self, landmarks: list, now: float) -> None:
        wrist = landmarks[_WRIST]
        self._wrist_history.append((now, wrist.x))
        # Drop samples older than the swipe window.
        while self._wrist_history and now - self._wrist_history[0][0] > self._SWIPE_WINDOW_S:
            self._wrist_history.popleft()
        if len(self._wrist_history) < 2:
            return
        if (now - self._last_gesture_at) <= self._GESTURE_COOLDOWN_S:
            return
        delta = self._wrist_history[-1][1] - self._wrist_history[0][1]
        if abs(delta) < self._SWIPE_MIN_DELTA:
            return
        self._last_gesture_at = now
        self._wrist_history.clear()
        self.media_control("next" if delta > 0 else "previous")
