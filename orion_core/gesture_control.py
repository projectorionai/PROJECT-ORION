"""
Gesture control — webcam hand gestures mapped to ORION and to this PC.

Pipeline (one daemon capture thread, never the Qt/qasync thread):

    camera frame ──► MediaPipe GestureRecognizer (21 hand landmarks + a
    learned pose label) ──► GestureInterpreter (landmark smoothing, pose
    voting, hold-to-fire, latch/re-arm, cooldown, stroke-based swipes and
    waves, pinch-drag) ──► GestureEvent ──► the owner's mapping ──► an action
    run on a separate action thread (ORION-level actions hop onto ORION's
    event loop; tool mappings go through the dispatcher and its gates).

Why it is built this way, with the evidence:

* **Debounce, not per-frame reaction.** The first version fired play/pause
  the instant one frame looked like a fist, and mapped the raw thumb-index
  distance straight to the volume, so any hand drifting through view moved
  the volume. Here a static pose must WIN a vote over the last ~0.45 s and
  then be HELD (0.6–1.0 s), still, before it fires once; it re-arms only
  after the pose has been released. A swipe counts only when a stroke
  COMPLETES in view — lowering a hand out of frame never finishes its stroke,
  so it cannot turn the volume down. Measured on synthetic sequences with
  landmark jitter (see tests/test_gesture_control.py).
* **Rotation-invariant geometry.** Finger extension is the ratio of
  wrist→tip to wrist→PIP distance (≈1.35 extended, ≈0.7 curled on real
  MediaPipe output), in aspect-corrected coordinates, so a tilted or
  sideways hand reads the same. The old rule ("tip above its PIP") only
  worked for an upright hand.
* **Model first, geometry second.** MediaPipe 1.0's GestureRecognizer bundle
  carries both the hand landmarker and a trained pose classifier (fist, open
  palm, thumbs up/down, victory, pointing). A confident model label wins;
  the geometric classifier covers what the model has no class for (pinch)
  and the landmark-only fallback.
* **Low CPU.** MEASURED on this machine (MediaPipe 1.0.0, XNNPACK, synthetic
  frames): 9–14 ms of CPU per frame whatever the input size, because the
  palm detector works on a 192 px letterbox anyway. Cost therefore scales
  with frame rate, so the loop runs 15 fps while a hand is in view and
  drops to an idle scan rate after 2 s without one; low-power mode halves
  both (8 / 4 fps).
* **Camera sharing.** Windows will not reliably give two handles on one
  webcam (MSMF -1072873821), so frames are BORROWED from the face tracker
  whenever it runs. When it does not, this engine opens its own 640x480
  handle and offers its frames to vision through SharedCamera; before the
  tracker opens the device (camera lab, orb, perception) it asks this engine
  to let go first (see share_camera_with), so the hand-off never collides.
* **Gates stay gates.** A gesture can never approve one of the system
  guard's on-screen destructive confirmations; thumbs up answers ORION's
  question as a typed "Yes." only while a conversation is live, and refuses
  when a guarded action is pending. Tool mappings are dispatched through
  OrionDispatcher.dispatch with their release arguments (confirm/consent/
  submit) stripped, so a gesture can propose an outward action but never
  release one. The cursor mode still needs pointer_start AND the desktop
  autonomy guard.

The model ships in assets/hand/ (Apache-2.0, from Google's MediaPipe model
storage); a build without it downloads the same pinned file once into
CONFIG_DIR/models.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import json
import math
import queue
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Any, Callable, Iterable, Sequence

from .bus import OrionBus
from .constants import CONFIG_DIR, resource_path
from .data import ToolResult
from .utils import open_camera_capture


def _mediapipe_available() -> bool:
    """Whether mediapipe can be imported, WITHOUT importing it.

    find_spec only consults the module finders, so this costs nothing —
    where a real module-scope `import mediapipe` would pull in a large
    dependency (and TensorFlow Lite with it) for every process that touches
    this module, gesture control running or not. The real import stays
    inside the capture thread."""
    return importlib.util.find_spec("mediapipe") is not None


# ── model files ──────────────────────────────────────────────────────────────

#: Version-pinned (".../float16/1/") rather than "latest", so the size and
#: SHA-256 below can be checked — model_store.download_verified refuses a
#: truncated or substituted file.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
_MODEL_SIZE = 8_373_440
_MODEL_SHA256 = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"
_MODEL_PATH = CONFIG_DIR / "models" / "gesture_recognizer.task"
#: What the first version downloaded; still usable (landmarks only).
_LEGACY_MODEL_PATH = CONFIG_DIR / "models" / "hand_landmarker.task"

#: Mappings, sensitivity, low-power and calibration. Written only when the
#: owner changes a setting.
SETTINGS_PATH = CONFIG_DIR / "gesture_control.json"


def bundled_model_path() -> Path:
    """The recogniser shipped in assets/hand (frozen or source tree)."""
    return resource_path("assets", "hand", "gesture_recognizer.task")


# ── landmark geometry ────────────────────────────────────────────────────────

# MediaPipe's 21-point hand model.
_WRIST = 0
_THUMB_MCP, _THUMB_TIP = 2, 4
_FINGER_MCPS = (5, 9, 13, 17)      # index, middle, ring, pinky
_FINGER_PIPS = (6, 10, 14, 18)
_FINGER_TIPS = (8, 12, 16, 20)
_PALM = (0, 5, 9, 13, 17)

#: wrist→tip / wrist→PIP. MEASURED on real MediaPipe output for an open hand:
#: 1.35 for the index finger; a curled finger folds back past its PIP (≈0.7).
_EXTENDED_RATIO = 1.12
_CURLED_RATIO = 1.0


@dataclass(frozen=True)
class HandShape:
    """What one hand is doing, in units of its own size (wrist→middle MCP)."""

    ratios: tuple[float, float, float, float]
    extended: tuple[bool | None, bool | None, bool | None, bool | None]
    thumb_out: float        # thumb tip ↔ pinky MCP, hand lengths
    thumb_angle: float      # thumb MCP→tip direction: 0 = image up, ±180 = down
    thumb_above: float      # how far the thumb tip clears every finger point (+up)
    thumb_below: float      # the same, downwards
    pinch: float            # thumb tip ↔ index tip, hand lengths
    spread: float           # index tip ↔ middle tip, hand lengths
    scale: float            # wrist → middle MCP, aspect-corrected image units
    centre: tuple[float, float]   # palm centre, aspect-corrected (x·aspect, y)
    pinch_point: tuple[float, float]


def _xy(point: Any, aspect: float) -> tuple[float, float]:
    if isinstance(point, (tuple, list)):
        return float(point[0]) * aspect, float(point[1])
    return float(point.x) * aspect, float(point.y)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def measure_hand(landmarks: Sequence[Any], aspect: float = 1.0) -> HandShape | None:
    """Rotation- and size-invariant description of a 21-point hand, or None.

    *aspect* is frame width / height: MediaPipe normalises x by the width and
    y by the height, so on a 4:3 or 16:9 frame raw distances are stretched
    and every angle is wrong unless x is scaled back first."""
    if landmarks is None or len(landmarks) < 21:
        return None
    pts = [_xy(p, aspect) for p in landmarks[:21]]
    wrist = pts[_WRIST]
    scale = _dist(wrist, pts[9])
    if not math.isfinite(scale) or scale < 1e-4:
        return None
    ratios: list[float] = []
    extended: list[bool | None] = []
    for pip, tip in zip(_FINGER_PIPS, _FINGER_TIPS):
        ratio = _dist(wrist, pts[tip]) / max(1e-6, _dist(wrist, pts[pip]))
        ratios.append(ratio)
        extended.append(True if ratio >= _EXTENDED_RATIO
                        else False if ratio <= _CURLED_RATIO else None)
    thumb_tip = pts[_THUMB_TIP]
    thumb_mcp = pts[_THUMB_MCP]
    others = [pts[i] for i in range(5, 21)]
    dx, dy = thumb_tip[0] - thumb_mcp[0], thumb_tip[1] - thumb_mcp[1]
    centre_x = sum(pts[i][0] for i in _PALM) / len(_PALM)
    centre_y = sum(pts[i][1] for i in _PALM) / len(_PALM)
    index_tip = pts[_FINGER_TIPS[0]]
    return HandShape(
        ratios=tuple(ratios),            # type: ignore[arg-type]
        extended=tuple(extended),        # type: ignore[arg-type]
        thumb_out=_dist(thumb_tip, pts[17]) / scale,
        thumb_angle=math.degrees(math.atan2(dx, -dy)),
        thumb_above=(min(p[1] for p in others) - thumb_tip[1]) / scale,
        thumb_below=(thumb_tip[1] - max(p[1] for p in others)) / scale,
        pinch=_dist(thumb_tip, index_tip) / scale,
        spread=_dist(index_tip, pts[_FINGER_TIPS[1]]) / scale,
        scale=scale,
        centre=(centre_x, centre_y),
        pinch_point=((thumb_tip[0] + index_tip[0]) / 2, (thumb_tip[1] + index_tip[1]) / 2),
    )


#: Thumb tip within this many hand lengths of the index tip reads as a pinch.
PINCH_THRESHOLD = 0.28


def classify_shape(shape: HandShape, pinch_threshold: float = PINCH_THRESHOLD) -> str:
    """The static pose from geometry alone: one of open_palm, fist,
    thumbs_up, thumbs_down, victory, point, pinch, two_fingers or none.

    Ambiguous fingers (neither clearly straight nor clearly folded) make the
    pose 'none' rather than a guess — a guess is what misfires."""
    ext = shape.extended
    if shape.pinch < pinch_threshold and ext[0] is not False:
        return "pinch"
    if all(e is True for e in ext) and shape.thumb_out >= 0.9:
        return "open_palm"
    if all(e is False for e in ext):
        if shape.thumb_out >= 1.0:
            if abs(shape.thumb_angle) <= 50 and shape.thumb_above >= 0.1:
                return "thumbs_up"
            if abs(shape.thumb_angle) >= 130 and shape.thumb_below >= 0.1:
                return "thumbs_down"
            return "none"        # thumb out sideways: not a sign we use
        return "fist"
    if ext == (True, True, False, False):
        return "victory" if shape.spread >= 0.3 else "two_fingers"
    if ext == (True, False, False, False):
        return "point"
    return "none"


#: MediaPipe GestureRecognizer categories -> this vocabulary.
_MODEL_LABELS = {
    "Closed_Fist": "fist", "Open_Palm": "open_palm", "Pointing_Up": "point",
    "Thumb_Up": "thumbs_up", "Thumb_Down": "thumbs_down", "Victory": "victory",
    "ILoveYou": "none", "None": "none",
}


def fuse_label(shape: HandShape, model_label: str = "", model_score: float = 0.0,
               min_score: float = 0.6, pinch_threshold: float = PINCH_THRESHOLD) -> str:
    """One pose per frame. A confident model label wins for the classes the
    model was trained on; geometry decides pinch (no model class) and
    everything when the model is absent or unsure."""
    geometric = classify_shape(shape, pinch_threshold)
    if geometric == "pinch":
        return "pinch"
    mapped = _MODEL_LABELS.get(str(model_label or ""))
    if mapped is not None and model_score >= min_score:
        return mapped
    return geometric


@dataclass
class HandObservation:
    """One detected hand in one frame."""

    landmarks: Sequence[Any]
    label: str = ""          # MediaPipe gesture category ('' = no classifier)
    score: float = 0.0
    handedness: str = ""


@dataclass(frozen=True)
class GestureEvent:
    name: str
    at: float
    value: int = 0           # +1 / -1 per pinch_drag step


# ── the temporal interpreter ─────────────────────────────────────────────────

class GestureInterpreter:
    """Per-frame hand observations in, debounced gesture events out.

    Pure: no camera, no clock of its own (``now`` is passed in) and no side
    effects — which is what lets a synthetic landmark sequence exercise
    exactly what a webcam would. Distances are in hand lengths (wrist→middle
    MCP), so the same gesture reads the same at any distance from the camera.
    """

    STATIC_HOLD_S = {"open_palm": 1.0, "fist": 0.6, "thumbs_up": 0.7,
                     "thumbs_down": 0.7, "victory": 0.6, "point": 0.9}
    VOTE_WINDOW_S = 0.45
    VOTE_SHARE = 0.65
    RELEASE_S = 0.35         # a fired pose must be gone this long to re-arm
    COOLDOWN_S = 0.9         # between any two discrete triggers
    STILL_SPEED = 1.0        # hand lengths/s: above this, no static trigger
    SETTLE_S = 0.35          # a newly seen hand moves into place first
    LOST_S = 0.3             # frames without a hand before it counts as gone
    MOVE_SPEED = 1.6         # hand lengths/s: a stroke starts
    STOP_SPEED = 0.7         # ... and ends
    SWIPE_DIST = 1.5         # hand lengths along the stroke
    SWIPE_MAX_S = 0.8
    SWIPE_CONFIRM_S = 0.3    # a wave's next stroke arrives sooner than this
    SWIPE_REFRACTORY_S = 0.8
    WAVE_MIN = 0.35
    WAVE_STROKES = 4
    WAVE_WINDOW_S = 2.0
    WAVE_REFRACTORY_S = 2.0
    PINCH_ENGAGE_S = 0.3
    PINCH_RELEASE = 1.6      # × pinch threshold: hysteresis while dragging
    DRAG_STEP = 0.3          # hand lengths of vertical travel per step
    SMOOTH_ALPHA = 0.6

    def __init__(self, sensitivity: float = 1.0, mirror: bool = True,
                 min_score: float = 0.6) -> None:
        self.mirror = bool(mirror)
        self.set_sensitivity(sensitivity)
        self._base_min_score = float(min_score)
        self.calibrated_still_speed: float | None = None
        self.reset()

    def set_sensitivity(self, sensitivity: float) -> float:
        """0.5 (deliberate) … 2.0 (hair-trigger). Shorter holds and shorter
        swipes as it rises; the model's confidence bar drops a little."""
        value = max(0.5, min(2.0, float(sensitivity)))
        self.sensitivity = value
        return value

    @property
    def min_score(self) -> float:
        return max(0.35, min(0.85, getattr(self, "_base_min_score", 0.6)
                             / math.sqrt(self.sensitivity)))

    def hold_for(self, label: str) -> float | None:
        hold = self.STATIC_HOLD_S.get(label)
        return None if hold is None else hold / self.sensitivity

    def reset(self) -> None:
        self._track: list[tuple[float, float, float]] | None = None
        self._lose(clear_latch=True)
        self._last_trigger_at = -1e9
        self._last_swipe_at = -1e9
        self._last_wave_at = -1e9
        self._pending_swipe: tuple[str, float] | None = None
        self.last_label = "none"

    def _lose(self, clear_latch: bool = True) -> None:
        self._track = None
        self._seen_at: float | None = None
        self._last_hand_at: float | None = None
        self._labels: deque[tuple[float, str]] = deque(maxlen=24)
        self._candidate = "none"
        self._candidate_since = 0.0
        if clear_latch:
            self._latched: str | None = None
            self._latch_clear_since: float | None = None
        self._centres: deque[tuple[float, float, float, float, bool]] = deque(maxlen=48)
        self._stroke: dict[str, Any] | None = None
        self._strokes: deque[tuple[float, float, float]] = deque(maxlen=12)
        self._pinch_since: float | None = None
        self._pinch_engaged = False
        self._pinch_anchor = 0.0
        self._pinching = False

    # ── frame entry point ───────────────────────────────────────────────────

    def observe(self, hands: Iterable[Any], now: float,
                aspect: float = 4.0 / 3.0) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        observations = [h if isinstance(h, HandObservation) else HandObservation(h)
                        for h in (hands or [])]
        hand = self._pick(observations, aspect)
        if hand is None:
            if self._last_hand_at is not None and now - self._last_hand_at >= self.LOST_S:
                # A stroke still in flight dies with the hand: lowering a hand
                # out of frame is not a swipe down.
                self._lose(clear_latch=True)
            self._flush_pending(now, events)
            return events
        pts = self._smooth(hand.landmarks, aspect)
        shape = measure_hand(pts, 1.0)
        if shape is None:
            return events
        label = fuse_label(shape, hand.label, hand.score, self.min_score,
                           self._pinch_threshold())
        if self._seen_at is None:
            self._seen_at = now
        self._last_hand_at = now
        self.last_label = label
        self._labels.append((now, label))
        dominant = self._dominant(now)
        if dominant != self._candidate:
            self._candidate, self._candidate_since = dominant, now
        self._update_latch(now, dominant)
        self._update_pinch(now, shape, events)
        self._update_motion(now, shape, label, events)
        self._update_static(now, dominant, events)
        self._flush_pending(now, events)
        return events

    def _pinch_threshold(self) -> float:
        return PINCH_THRESHOLD * math.sqrt(self.sensitivity)

    def _pick(self, hands: list[HandObservation], aspect: float) -> HandObservation | None:
        """The hand already being followed (nearest palm), else the largest."""
        best, best_key = None, None
        for hand in hands:
            shape = measure_hand(hand.landmarks, aspect)
            if shape is None:
                continue
            if self._track is not None:
                tracked = measure_hand(self._track, 1.0)
                key = (_dist(shape.centre, tracked.centre) if tracked is not None else 0.0)
            else:
                key = -shape.scale
            if best_key is None or key < best_key:
                best, best_key = hand, key
        return best

    def _smooth(self, landmarks: Sequence[Any], aspect: float) -> list[tuple[float, float, float]]:
        raw = []
        for p in landmarks[:21]:
            if isinstance(p, (tuple, list)):
                z = float(p[2]) if len(p) > 2 else 0.0
            else:
                z = float(getattr(p, "z", 0.0) or 0.0)
            x, y = _xy(p, aspect)
            raw.append((x, y, z))
        if self._track is None or len(self._track) != len(raw):
            self._track = raw
            return raw
        # A jump of more than a hand length is a different hand, not motion.
        old = measure_hand(self._track, 1.0)
        new = measure_hand(raw, 1.0)
        if old is not None and new is not None and \
                _dist(old.centre, new.centre) > 1.5 * max(old.scale, new.scale):
            self._track = raw
            return raw
        a = self.SMOOTH_ALPHA
        self._track = [(o[0] + a * (n[0] - o[0]), o[1] + a * (n[1] - o[1]),
                        o[2] + a * (n[2] - o[2])) for o, n in zip(self._track, raw)]
        return self._track

    def _dominant(self, now: float) -> str:
        recent = [label for t, label in self._labels if now - t <= self.VOTE_WINDOW_S]
        if len(recent) < 3:
            # Low frame rates: the last three frames within a second still vote.
            recent = [label for t, label in list(self._labels)[-3:] if now - t <= 1.2]
        if len(recent) < 2:
            return "none"
        label, count = Counter(recent).most_common(1)[0]
        return label if count / len(recent) >= self.VOTE_SHARE else "none"

    def _update_latch(self, now: float, dominant: str) -> None:
        if self._latched is None:
            return
        if dominant == self._latched:
            self._latch_clear_since = None
        elif self._latch_clear_since is None:
            self._latch_clear_since = now
        elif now - self._latch_clear_since >= self.RELEASE_S:
            self._latched = None
            self._latch_clear_since = None

    def _speed(self, now: float, window: float = 0.3) -> float:
        """Mean palm speed over the last *window* seconds, hand lengths/s."""
        samples = [c for c in self._centres if now - c[0] <= window]
        if len(samples) < 2:
            return 0.0
        (t0, x0, y0, s0, _), (t1, x1, y1, s1, _) = samples[0], samples[-1]
        if t1 - t0 <= 0:
            return 0.0
        return math.hypot(x1 - x0, y1 - y0) / max(1e-6, (s0 + s1) / 2) / (t1 - t0)

    def _can_trigger(self, now: float) -> bool:
        return now - self._last_trigger_at >= self.COOLDOWN_S

    def _emit(self, events: list[GestureEvent], name: str, now: float, value: int = 0,
              discrete: bool = True) -> None:
        events.append(GestureEvent(name, now, value))
        if discrete:
            self._last_trigger_at = now

    # ── static poses ────────────────────────────────────────────────────────

    def _update_static(self, now: float, dominant: str, events: list[GestureEvent]) -> None:
        hold = self.hold_for(dominant)
        if (hold is None or self._pinch_engaged or self._stroke is not None
                or self._pending_swipe is not None):
            return
        if self._latched == dominant or not self._can_trigger(now):
            return
        if now - self._candidate_since < hold:
            return
        still = self.calibrated_still_speed or self.STILL_SPEED
        if self._speed(now) > still:
            return
        self._latched = dominant
        self._latch_clear_since = None
        self._emit(events, dominant, now)

    # ── strokes: swipes and waves ───────────────────────────────────────────

    def _update_motion(self, now: float, shape: HandShape, label: str,
                       events: list[GestureEvent]) -> None:
        cx, cy = shape.centre
        # User frame: +x = the user's right (the webcam image is not
        # mirrored, so their right is the image's left), +y = up.
        ux = -cx if self.mirror else cx
        uy = -cy
        is_open = sum(1 for e in shape.extended if e is True) >= 3
        previous = self._centres[-1] if self._centres else None
        self._centres.append((now, ux, uy, shape.scale, is_open))
        if self._pinch_engaged or previous is None or self._seen_at is None:
            self._stroke = None
            return
        if now - self._seen_at < self.SETTLE_S:
            return
        dt = now - previous[0]
        if dt <= 0 or dt > 0.5:
            self._stroke = None
            return
        scale = max(1e-6, (shape.scale + previous[3]) / 2)
        vx, vy = (ux - previous[1]) / dt / scale, (uy - previous[2]) / dt / scale
        speed = math.hypot(vx, vy)
        stroke = self._stroke
        if stroke is None:
            if speed >= self.MOVE_SPEED:
                self._stroke = {"t0": previous[0], "x0": previous[1], "y0": previous[2],
                                "dir": (vx / speed, vy / speed), "frames": 1,
                                "open": int(is_open), "scale": scale,
                                "x1": ux, "y1": uy, "t1": now}
            return
        cos = (vx * stroke["dir"][0] + vy * stroke["dir"][1]) / speed if speed > 0 else 1.0
        if speed < self.STOP_SPEED or cos < 0.3:
            self._end_stroke(stroke, events)
            self._stroke = None
            # A pose is held from the moment the hand stopped, not from before
            # it moved: otherwise the open hand that just swiped or waved
            # reads as an open palm held for a second and pauses ORION. Only a
            # stroke with real travel counts; tracker jitter makes tiny ones.
            travel = math.hypot(stroke["x1"] - stroke["x0"],
                                stroke["y1"] - stroke["y0"]) / stroke["scale"]
            if travel >= self.WAVE_MIN:
                self._candidate_since = max(self._candidate_since, now)
            if speed >= self.MOVE_SPEED:          # turned straight round: a new stroke
                self._stroke = {"t0": previous[0], "x0": previous[1], "y0": previous[2],
                                "dir": (vx / speed, vy / speed), "frames": 1,
                                "open": int(is_open), "scale": scale,
                                "x1": ux, "y1": uy, "t1": now}
            return
        stroke.update(frames=stroke["frames"] + 1, open=stroke["open"] + int(is_open),
                      x1=ux, y1=uy, t1=now,
                      scale=(stroke["scale"] * stroke["frames"] + scale) / (stroke["frames"] + 1))

    def _end_stroke(self, stroke: dict[str, Any], events: list[GestureEvent]) -> None:
        now = stroke["t1"]
        scale = stroke["scale"]
        dx = (stroke["x1"] - stroke["x0"]) / scale
        dy = (stroke["y1"] - stroke["y0"]) / scale
        duration = now - stroke["t0"]
        open_share = stroke["open"] / max(1, stroke["frames"])
        horizontal = abs(dx) >= abs(dy)
        major, minor = (dx, dy) if horizontal else (dy, dx)
        # Waves: alternating horizontal strokes of modest size.
        if horizontal and abs(dx) >= self.WAVE_MIN and open_share >= 0.5:
            self._strokes.append((now, dx, dy))
            if self._pending_swipe is not None and now - self._pending_swipe[1] <= self.SWIPE_CONFIRM_S:
                # The hand came straight back: that was the first stroke of a
                # wave (or a return), not a swipe. Fewer false swipes beats
                # catching a flick that is immediately undone.
                self._pending_swipe = None
            if self._is_wave(now):
                self._strokes.clear()
                self._pending_swipe = None
                if now - self._last_wave_at >= self.WAVE_REFRACTORY_S and self._can_trigger(now):
                    self._last_wave_at = now
                    self._emit(events, "wave", now)
                return
        if (abs(major) >= self.SWIPE_DIST / math.sqrt(self.sensitivity)
                and abs(major) >= 2.0 * abs(minor) and duration <= self.SWIPE_MAX_S
                and open_share >= 0.6
                and now - self._last_swipe_at >= self.SWIPE_REFRACTORY_S):
            if horizontal:
                name = "swipe_right" if dx > 0 else "swipe_left"
            else:
                name = "swipe_up" if dy > 0 else "swipe_down"
            self._pending_swipe = (name, now)

    def _is_wave(self, now: float) -> bool:
        recent = [s for s in self._strokes if now - s[0] <= self.WAVE_WINDOW_S]
        run = 0
        last_sign = 0
        for _t, dx, _dy in recent:
            sign = 1 if dx > 0 else -1
            run = run + 1 if sign != last_sign else 1
            last_sign = sign
        return run >= self.WAVE_STROKES

    def _flush_pending(self, now: float, events: list[GestureEvent]) -> None:
        pending = self._pending_swipe
        if pending is None or now - pending[1] < self.SWIPE_CONFIRM_S:
            return
        self._pending_swipe = None
        if self._can_trigger(now):
            self._last_swipe_at = now
            self._strokes.clear()
            self._emit(events, pending[0], now)

    # ── pinch-drag ──────────────────────────────────────────────────────────

    def _update_pinch(self, now: float, shape: HandShape, events: list[GestureEvent]) -> None:
        threshold = self._pinch_threshold()
        limit = threshold * (self.PINCH_RELEASE if self._pinch_engaged else 1.0)
        pinching = shape.pinch < limit and shape.extended[0] is not False
        uy = -shape.pinch_point[1]
        if not pinching:
            self._pinch_since = None
            self._pinch_engaged = False
            self._pinching = False
            return
        self._pinching = True
        if self._pinch_since is None:
            self._pinch_since = now
        if not self._pinch_engaged:
            if (now - self._pinch_since >= self.PINCH_ENGAGE_S / self.sensitivity
                    and self._speed(now) <= self.STILL_SPEED * 1.5):
                self._pinch_engaged = True
                self._pinch_anchor = uy
                self._stroke = None
                self._pending_swipe = None
            return
        step = self.DRAG_STEP * shape.scale
        emitted = 0
        while uy - self._pinch_anchor >= step and emitted < 4:
            self._pinch_anchor += step
            self._emit(events, "pinch_drag", now, value=+1, discrete=False)
            emitted += 1
        while self._pinch_anchor - uy >= step and emitted < 4:
            self._pinch_anchor -= step
            self._emit(events, "pinch_drag", now, value=-1, discrete=False)
            emitted += 1

    # ── introspection ───────────────────────────────────────────────────────

    @property
    def hand_present(self) -> bool:
        return self._last_hand_at is not None

    @property
    def pinch_engaged(self) -> bool:
        return self._pinch_engaged


# ── a canonical hand, for the offline self-test and the tests ────────────────

# Local frame: wrist at the origin, +y towards the fingers, units of hand
# length (wrist → middle MCP). Taken from what real MediaPipe returned for an
# open hand (scratch measurement, 16 Oct 2026), rounded.
_REF_BASES = {1: (-0.36, 0.15), 5: (-0.46, 0.93), 9: (-0.21, 0.98),
              13: (0.03, 0.92), 17: (0.26, 0.79)}
_REF_DIRS = {5: -10.0, 9: -2.0, 13: 7.0, 17: 18.0}            # degrees from +y
_REF_LENGTHS = {5: (0.47, 0.30, 0.25), 9: (0.51, 0.33, 0.28),
                13: (0.47, 0.32, 0.26), 17: (0.35, 0.24, 0.20)}
_THUMB_LENGTHS = (0.42, 0.33, 0.28)

#: Per finger (index, middle, ring, pinky): True extended, False curled.
POSE_FINGERS = {
    "open_palm": (True, True, True, True),
    "fist": (False, False, False, False),
    "thumbs_up": (False, False, False, False),
    "thumbs_down": (False, False, False, False),
    "victory": (True, True, False, False),
    "point": (True, False, False, False),
    "pinch": (True, True, True, True),
    "relaxed": (None, None, False, False),
}


def synthetic_hand(pose: str = "open_palm", *, centre: tuple[float, float] = (0.5, 0.55),
                   size: float = 0.18, angle: float = 0.0, aspect: float = 4.0 / 3.0,
                   right: bool = False, spread: float = 1.0) -> list[tuple[float, float, float]]:
    """21 normalised landmarks for *pose*, built on real MediaPipe proportions.

    *size* is the hand length as a fraction of frame height; *angle* rotates
    the hand (degrees, counter-clockwise). thumbs_up/thumbs_down are the
    thumb-out fist turned so the thumb points up/down, as a webcam sees one.
    """
    fingers = POSE_FINGERS.get(pose, (None, None, None, None))
    flex_curled = (80.0, 100.0, 60.0)       # cumulative 80, 180, 240 degrees
    flex_straight = (4.0, 4.0, 3.0)
    flex_relaxed = (35.0, 45.0, 25.0)
    local: dict[int, tuple[float, float, float]] = {0: (0.0, 0.0, 0.0)}
    for k, (bx, by) in _REF_BASES.items():
        local[k] = (bx, by, 0.0)
    for slot, (mcp, ext) in enumerate(zip(_FINGER_MCPS, fingers)):
        flex = flex_straight if ext is True else flex_curled if ext is False else flex_relaxed
        base_angle = _REF_DIRS[mcp] * (spread if slot != 1 else 1.0)
        if pose == "victory" and slot in (0, 1):
            base_angle += -9.0 if slot == 0 else 9.0     # the V
        ux, uy = math.sin(math.radians(base_angle)), math.cos(math.radians(base_angle))
        x, y, z = local[mcp]
        cumulative = 0.0
        for joint, (length, bend) in enumerate(zip(_REF_LENGTHS[mcp], flex), start=1):
            cumulative += bend
            planar = math.cos(math.radians(cumulative)) * length
            x, y = x + ux * planar, y + uy * planar
            z -= math.sin(math.radians(cumulative)) * length
            local[mcp + joint] = (x, y, z)
    # Thumb.
    if pose in ("fist",):
        thumb_dirs = (35.0, 75.0, 100.0)         # tucked across the fingers
    elif pose in ("thumbs_up", "thumbs_down"):
        thumb_dirs = (-80.0, -88.0, -90.0)       # straight out sideways
    elif pose == "pinch":
        tip = local[8]
        cx_, cy_, _ = local[1]
        mid = ((cx_ + tip[0]) / 2 - 0.25, (cy_ + tip[1]) / 2)
        local[2] = (cx_ + (mid[0] - cx_) * 0.6, cy_ + (mid[1] - cy_) * 0.6, -0.05)
        local[3] = (mid[0], mid[1], -0.08)
        local[4] = (tip[0] - 0.04, tip[1] - 0.06, -0.1)
        thumb_dirs = ()
    elif pose in ("victory", "point"):
        thumb_dirs = (30.0, 70.0, 95.0)
    else:
        thumb_dirs = (-50.0, -45.0, -40.0)       # open hand: thumb out
    x, y, z = local[1]
    for joint, (length, direction) in enumerate(zip(_THUMB_LENGTHS, thumb_dirs), start=2):
        x += math.sin(math.radians(direction)) * length
        y += math.cos(math.radians(direction)) * length
        local[joint] = (x, y, z - 0.04 * joint)
    turn = angle + (-90.0 if pose == "thumbs_up" else 90.0 if pose == "thumbs_down" else 0.0)
    if right:
        turn = -turn
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    out = []
    for i in range(21):
        lx, ly, lz = local[i]
        if right:
            lx = -lx
        rx, ry = lx * c - ly * s, lx * s + ly * c
        # Wrist sits below the palm centre; place the palm centre at *centre*.
        X = centre[0] * aspect + rx * size
        Y = centre[1] + 0.55 * size - ry * size
        out.append((X / aspect, Y, lz * size))
    return out


# ── vocabulary, actions and settings ─────────────────────────────────────────

GESTURES: dict[str, str] = {
    "open_palm":   "open palm, held still for about a second",
    "fist":        "closed fist, held briefly",
    "thumbs_up":   "thumbs up, held briefly",
    "thumbs_down": "thumbs down, held briefly",
    "victory":     "two-finger V, held briefly",
    "point":       "index finger pointing, held",
    "swipe_left":  "open hand swept to your left",
    "swipe_right": "open hand swept to your right",
    "swipe_up":    "open hand swept upwards",
    "swipe_down":  "open hand swept downwards (and stopped in view)",
    "wave":        "a wave — side to side, three or more times",
    "pinch_drag":  "pinch thumb and index, then move up or down",
}

#: name -> (what it does, kind). 'continuous' actions take pinch_drag steps.
ACTIONS: dict[str, tuple[str, str]] = {
    "none":             ("nothing", "any"),
    "pause_orion":      ("pause or resume ORION, holding his speech in place", "discrete"),
    "stop_orion":       ("stop ORION talking and cancel his current request", "discrete"),
    "wake_orion":       ("open ORION's ear without the wake word", "discrete"),
    "confirm":          ("answer yes to ORION's question (never approves an on-screen "
                         "destructive prompt)", "discrete"),
    "cancel":           ("cancel a pending confirmation, or answer no", "discrete"),
    "media_play_pause": ("play or pause media", "discrete"),
    "media_next":       ("next track", "discrete"),
    "media_previous":   ("previous track", "discrete"),
    "volume_up":        ("volume up a step", "discrete"),
    "volume_down":      ("volume down a step", "discrete"),
    "mute_toggle":      ("mute or unmute the speakers", "discrete"),
    "screenshot":       ("save a screenshot to Pictures\\Screenshots", "discrete"),
    "switch_window":    ("Alt+Tab (needs desktop autonomy on)", "discrete"),
    "desktop_next":     ("next virtual desktop (needs desktop autonomy on)", "discrete"),
    "desktop_previous": ("previous virtual desktop (needs desktop autonomy on)", "discrete"),
    "volume_drag":      ("volume follows the pinch up and down", "continuous"),
    "scroll_drag":      ("scroll follows the pinch (needs desktop autonomy on)", "continuous"),
}

DEFAULT_MAPPINGS: dict[str, dict[str, Any]] = {
    "open_palm": {"action": "pause_orion"},
    "fist": {"action": "stop_orion"},
    "thumbs_up": {"action": "confirm"},
    "thumbs_down": {"action": "cancel"},
    "victory": {"action": "screenshot"},
    "point": {"action": "none"},
    "swipe_left": {"action": "media_previous"},
    "swipe_right": {"action": "media_next"},
    "swipe_up": {"action": "volume_up"},
    "swipe_down": {"action": "volume_down"},
    "wave": {"action": "wake_orion"},
    "pinch_drag": {"action": "volume_drag"},
}

#: Never reachable from a gesture: lifecycle, self-modification, and the
#: gesture tool itself (a gesture that reconfigures gestures is a trap).
_FORBIDDEN_TOOLS = frozenset({
    "gesture_control", "shutdown_orion", "restart_orion", "self_repair", "forge",
    "backup", "cleanup_review", "execute_plan", "autoplan", "protocol",
})
#: Arguments that RELEASE a proposed action (speaker_gate.RELEASE_ARGS).
_RELEASE_ARGS = ("confirm", "consent", "submit")
#: Mapped actions that stand in for the owner answering ORION or waking him.
#: Only a hand in front of the camera makes them; a tool call may not (the
#: ``test`` action describes them but never runs them).
_PERSONAL_ACTIONS = frozenset({"confirm", "cancel", "wake_orion"})

#: Frame rates: (hand in view, idle scan). See the module docstring.
_FPS = {"normal": (15.0, 8.0), "low_power": (8.0, 4.0)}
_IDLE_AFTER_S = 2.0
_WORK_WIDTH = 640


def describe_mapping(spec: dict[str, Any] | None) -> str:
    spec = spec or {"action": "none"}
    if "tool" in spec:
        args = spec.get("args") or {}
        detail = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"the {spec['tool']} tool" + (f" ({detail})" if detail else "")
    action = spec.get("action", "none")
    return ACTIONS.get(action, (action, ""))[0]


class GestureSettings:
    """The owner's gesture preferences, persisted as JSON under CONFIG_DIR."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else SETTINGS_PATH
        self.mappings: dict[str, dict[str, Any]] = {k: dict(v) for k, v in DEFAULT_MAPPINGS.items()}
        self.sensitivity = 1.0
        self.low_power = False
        self.mirror = True
        self.calibration: dict[str, Any] = {}
        self.load_error = ""
        self.load()

    def load(self) -> None:
        try:
            if not self.path.is_file():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:          # a bad file must not stop gestures
            self.load_error = str(exc)[:120]
            return
        if not isinstance(data, dict):
            self.load_error = "Gesture settings must be an object"
            return
        mappings = data.get("mappings") or {}
        if not isinstance(mappings, dict):
            self.load_error = "Gesture mappings must be an object"
            return
        for gesture, spec in mappings.items():
            if gesture in GESTURES and isinstance(spec, dict) and (
                    spec.get("action") in ACTIONS or isinstance(spec.get("tool"), str)):
                self.mappings[gesture] = dict(spec)
        try:
            sensitivity = float(data.get("sensitivity", 1.0))
            if math.isfinite(sensitivity):
                self.sensitivity = max(0.5, min(2.0, sensitivity))
        except (TypeError, ValueError):
            pass
        self.low_power = data.get("low_power") is True
        self.mirror = data.get("mirror", True) is not False
        if isinstance(data.get("calibration"), dict):
            self.calibration = dict(data["calibration"])

    def save(self) -> None:
        from .atomic_io import atomic_write_text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps({
            "version": 1, "mappings": self.mappings, "sensitivity": self.sensitivity,
            "low_power": self.low_power, "mirror": self.mirror,
            "calibration": self.calibration}, indent=2))


def _screenshots_dir() -> Path:
    """Windows' own Screenshots folder (it follows OneDrive redirection),
    falling back to ~/Pictures/Screenshots."""
    try:
        import ctypes
        from ctypes import wintypes
        import uuid

        class _GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

        raw = uuid.UUID("{b7bede81-df94-4682-a7d8-57a52620b86f}").bytes_le
        guid = _GUID.from_buffer_copy(raw)
        path_ptr = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(
                ctypes.byref(guid), 0, None, ctypes.byref(path_ptr)) == 0:
            found = Path(path_ptr.value)
            ctypes.windll.ole32.CoTaskMemFree(path_ptr)
            return found
    except Exception:
        pass
    return Path.home() / "Pictures" / "Screenshots"


class GestureActions:
    """Carries out a mapped action. Runs on GestureEngine's action thread.

    Desktop and PC actions go through ORION's own tools (media_control,
    peripherals, desktop_control, capture_screen) via the dispatcher, so they
    meet the same gates — the autonomy guard, the system guard — as a spoken
    request. ORION-level actions (pause, stop, wake, answers) are handed to
    his event loop, where his own buttons run them. Nothing here touches Qt.
    Before ORION is connected, media and volume fall back to the direct
    callables the engine was built with."""

    def __init__(self, bus: Any, *, peripherals: Any = None,
                 media_control: Callable[..., Any] | None = None,
                 pointer_action: Callable[[str, float, float], Any] | None = None,
                 screenshot_dir: Path | None = None) -> None:
        self.bus = bus
        self.peripherals = peripherals
        self.media_control = media_control
        self.pointer_action = pointer_action
        self.screenshot_dir = screenshot_dir
        self.worker: Any = None
        self.dispatcher: Any = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last_conversation_at = -1e9
        self.conversation_window_s = 45.0
        self._com_ready = False

    def connect_orion(self, *, worker: Any = None, dispatcher: Any = None,
                      loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.worker = worker
        self.dispatcher = dispatcher
        self.loop = loop
        speaking = getattr(self.bus, "speaking", None)
        if speaking is not None:
            try:
                speaking.connect(self._on_speaking)
            except Exception:
                pass

    def _on_speaking(self, active: Any) -> None:
        # Thumbs up/down answer only a LIVE exchange: ORION spoke recently.
        self.last_conversation_at = time.monotonic()

    def conversation_live(self) -> bool:
        return time.monotonic() - self.last_conversation_at <= self.conversation_window_s

    # ── plumbing ────────────────────────────────────────────────────────────

    def _on_loop(self, fn: Callable[[], Any], timeout: float = 5.0) -> Any:
        loop = self.loop
        if loop is None or loop.is_closed():
            raise RuntimeError("ORION is not connected to gesture control yet")
        future: concurrent.futures.Future = concurrent.futures.Future()

        def run() -> None:
            try:
                future.set_result(fn())
            except BaseException as exc:     # surfaced to the action thread
                future.set_exception(exc)

        loop.call_soon_threadsafe(run)
        return future.result(timeout)

    def _await_on_loop(self, coro: Any, timeout: float = 20.0) -> Any:
        loop = self.loop
        if loop is None or loop.is_closed():
            coro.close()
            raise RuntimeError("ORION is not connected to gesture control yet")
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)

    def _ensure_com(self) -> None:
        # pycaw's volume endpoint is COM; this is a thread of our own.
        if self._com_ready:
            return
        self._com_ready = True
        try:
            import comtypes  # type: ignore[import-not-found]
            comtypes.CoInitialize()
        except Exception:
            pass

    # ── entry point ─────────────────────────────────────────────────────────

    def perform(self, spec: dict[str, Any], event: GestureEvent | None = None) -> ToolResult:
        if "tool" in spec:
            return self._tool(str(spec["tool"]), dict(spec.get("args") or {}))
        action = str(spec.get("action") or "none")
        handler = getattr(self, f"_do_{action}", None)
        if action == "none":
            return ToolResult("No action is mapped to that gesture.", ok=False)
        if handler is None:
            return ToolResult(f"Unknown gesture action: {action}.", ok=False)
        return handler(event)

    def _tool(self, tool: str, args: dict[str, Any]) -> ToolResult:
        if tool in _FORBIDDEN_TOOLS:
            return ToolResult(f"A gesture may not run {tool}.", ok=False)
        for key in _RELEASE_ARGS:
            args.pop(key, None)
        dispatcher = self.dispatcher
        if dispatcher is None or self.loop is None:
            return ToolResult(f"{tool} needs ORION's dispatcher, which is not connected.",
                              ok=False)
        result = self._await_on_loop(dispatcher.dispatch(tool, args))
        return result if isinstance(result, ToolResult) else ToolResult(str(result))

    def _connected(self) -> bool:
        return self.dispatcher is not None and self.loop is not None

    # ── ORION ───────────────────────────────────────────────────────────────

    def _worker_call(self, method: str, description: str) -> ToolResult:
        worker = self.worker
        fn = getattr(worker, method, None)
        if fn is None:
            return ToolResult(f"ORION's voice is not ready to {description}.", ok=False)
        self._on_loop(fn)
        return ToolResult(f"ORION: {description}.")

    def _do_pause_orion(self, _event: Any) -> ToolResult:
        return self._worker_call("toggle_pause", "pause toggled")

    def _do_stop_orion(self, _event: Any) -> ToolResult:
        return self._worker_call("cancel_turn", "stopped")

    def _do_wake_orion(self, _event: Any) -> ToolResult:
        worker = self.worker
        if worker is None:
            return ToolResult("ORION's voice is not ready to wake.", ok=False)
        try:
            from . import push_to_talk
            if push_to_talk.enabled():
                # The point of push-to-talk is a CLOSED microphone until the
                # key is held; a wave must not be a way round that.
                return ToolResult("Push-to-talk is on — hold the key to talk; a wave "
                                  "won't open the microphone.", ok=False)
        except Exception:
            pass
        if getattr(worker, "standby_mode", False):
            return ToolResult("ORION is in standby — say 'wake up'; a wave won't reopen "
                              "the microphone from standby.", ok=False)

        def wake() -> str:
            if getattr(worker, "paused", False):
                worker.resume()
                return "resumed"
            refresh = getattr(worker, "_refresh_wake_window", None)
            if refresh is not None:
                refresh()
            if getattr(worker, "connected", False) and hasattr(worker, "_emit_state"):
                worker._emit_state("LISTENING")
            return "listening"

        state = self._on_loop(wake)
        return ToolResult("ORION resumed." if state == "resumed"
                          else "ORION is listening — go ahead, no wake word needed.")

    def _guard_pending(self) -> int:
        guard = getattr(self.dispatcher, "system_guard", None)
        if guard is None:
            return 0
        return int(self._on_loop(guard.pending_count))

    def _answer(self, text: str) -> ToolResult:
        submit = getattr(self.worker, "submit_text", None)
        if submit is None:
            return ToolResult("ORION's voice is not ready for an answer.", ok=False)
        self._await_on_loop(submit(text), timeout=30.0)
        return ToolResult(f"Answered ORION: {text}")

    def _do_confirm(self, _event: Any) -> ToolResult:
        if self._guard_pending():
            return ToolResult("A destructive action is waiting for the on-screen Confirm "
                              "button — I won't approve it with a gesture.", ok=False)
        if not self.conversation_live():
            return ToolResult("Thumbs up seen, but ORION isn't waiting for an answer.", ok=False)
        return self._answer("Yes.")

    def _do_cancel(self, _event: Any) -> ToolResult:
        if self._guard_pending():
            cancel = getattr(self.dispatcher, "cancel_system_action", None)
            if cancel is not None:
                result = self._on_loop(cancel)
                return result if isinstance(result, ToolResult) else ToolResult(
                    "Cancelled — nothing was executed.")
        if not self.conversation_live():
            return ToolResult("Thumbs down seen; there was nothing to cancel.", ok=False)
        return self._answer("No.")

    # ── media and sound ─────────────────────────────────────────────────────

    def _media(self, action: str) -> ToolResult:
        if self._connected():
            return self._tool("media_control", {"action": action})
        if self.media_control is None:
            return ToolResult("Media control is not available.", ok=False)
        result = self.media_control(action)
        return result if isinstance(result, ToolResult) else ToolResult(f"Media: {action}.")

    def _do_media_play_pause(self, _event: Any) -> ToolResult:
        return self._media("play_pause")

    def _do_media_next(self, _event: Any) -> ToolResult:
        return self._media("next")

    def _do_media_previous(self, _event: Any) -> ToolResult:
        return self._media("previous")

    def _volume(self, delta: float) -> ToolResult:
        if self._connected():
            return self._tool("peripherals", {"action": "volume_up" if delta > 0 else "volume_down",
                                              "level": round(abs(delta), 3)})
        if self.peripherals is None:
            return ToolResult("Volume control is not available.", ok=False)
        self._ensure_com()
        change = getattr(self.peripherals, "change_volume", None)
        if change is not None:
            return change(delta)
        return ToolResult("Volume control is not available.", ok=False)

    def _do_volume_up(self, _event: Any) -> ToolResult:
        return self._volume(+0.06)

    def _do_volume_down(self, _event: Any) -> ToolResult:
        return self._volume(-0.06)

    def _do_volume_drag(self, event: GestureEvent | None) -> ToolResult:
        step = (event.value if event is not None else 0) or 0
        return self._volume(0.04 * step) if step else ToolResult("No pinch movement.")

    def _do_mute_toggle(self, _event: Any) -> ToolResult:
        if self._connected():
            return self._tool("peripherals", {"action": "toggle_mute"})
        if self.peripherals is None or not hasattr(self.peripherals, "toggle_mute"):
            return ToolResult("Mute control is not available.", ok=False)
        self._ensure_com()
        return self.peripherals.toggle_mute()

    # ── desktop: ORION's desktop_control tool, behind its autonomy guard ────

    def _do_scroll_drag(self, event: GestureEvent | None) -> ToolResult:
        step = (event.value if event is not None else 0) or 0
        if not step:
            return ToolResult("No pinch movement.")
        return self._tool("desktop_control", {"action": "scroll", "amount": 3 if step > 0 else -3,
                                              "why": "hand gesture: pinch-drag scroll"})

    def _hotkey(self, keys: str) -> ToolResult:
        return self._tool("desktop_control", {"action": "hotkey", "keys": keys,
                                              "why": f"hand gesture: {keys}"})

    def _do_switch_window(self, _event: Any) -> ToolResult:
        return self._hotkey("alt+tab")

    def _do_desktop_next(self, _event: Any) -> ToolResult:
        return self._hotkey("ctrl+win+right")

    def _do_desktop_previous(self, _event: Any) -> ToolResult:
        return self._hotkey("ctrl+win+left")

    def _do_screenshot(self, _event: Any) -> ToolResult:
        # capture_screen keeps its pixels in memory only; saving them is the
        # owner's explicit ask when he makes the V.
        captured = self._tool("capture_screen", {"quality": 95, "max_side": 4096})
        data = (captured.media or {}).get("data") if captured.ok and isinstance(
            getattr(captured, "media", None), dict) else None
        if not data:
            return captured if not captured.ok else ToolResult(
                "The screen capture came back empty.", ok=False)
        folder = self.screenshot_dir or _screenshots_dir()
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H%M%S")
        target = folder / f"ORION {stamp}.jpg"
        n = 2
        while target.exists():                      # never overwrite
            target = folder / f"ORION {stamp} ({n}).jpg"
            n += 1
        target.write_bytes(bytes(data))
        return ToolResult(f"Screenshot saved to {target}.")


# ── camera sharing ───────────────────────────────────────────────────────────

class SharedCamera:
    """One frame source over the face tracker and gesture control's own
    handle, for consumers (on-demand vision) that would otherwise open a
    second handle on the webcam. Mirrors the tracker's borrowing interface:
    vision reads ``latest_frame.__self__.camera_index``/``is_capturing``."""

    def __init__(self, tracker: Any, gestures: Any) -> None:
        self.tracker = tracker
        self.gestures = gestures

    @property
    def camera_index(self) -> Any:
        source = self.tracker if self.tracker is not None else self.gestures
        return getattr(source, "camera_index", None)

    def is_capturing(self) -> bool:
        for source in (self.tracker, self.gestures):
            fn = getattr(source, "is_capturing", None)
            try:
                if callable(fn) and fn():
                    return True
            except Exception:
                continue
        return False

    def latest_frame(self) -> Any:
        for source in (self.tracker, self.gestures):
            fn = getattr(source, "latest_frame", None)
            if not callable(fn):
                continue
            try:
                frame = fn()
            except Exception:
                frame = None
            if frame is not None:
                return frame
        return None


# ── the engine ───────────────────────────────────────────────────────────────

class GestureEngine:
    """Webcam hand-gesture control on a daemon thread."""

    available = _mediapipe_available()

    def __init__(
        self,
        bus: OrionBus,
        peripherals: Any,
        media_control: Callable[..., Any],
        camera_index: int | None = None,
        frame_source: Callable[[], Any] | None = None,
        frame_source_active: Callable[[], bool] | None = None,
        pointer_action: Callable[[str, float, float], Any] | None = None,
        *,
        actions: GestureActions | None = None,
        settings_path: Path | None = None,
    ) -> None:
        self.bus = bus
        self.peripherals = peripherals
        self.media_control = media_control
        if camera_index is None:
            from . import camera_devices as cd
            camera_index = cd.resolve()
        self.camera_index = int(camera_index)
        self.frame_source = frame_source
        self.frame_source_active = frame_source_active
        self.pointer_action = pointer_action
        self.settings = GestureSettings(settings_path)
        self.interpreter = GestureInterpreter(self.settings.sensitivity, self.settings.mirror)
        self._apply_calibration()
        self.actions = actions or GestureActions(
            bus, peripherals=peripherals, media_control=media_control,
            pointer_action=pointer_action)

        self._thread: Thread | None = None
        self._stop_event = Event()
        self._wake = Event()
        self._mode = "peripherals"          # 'peripherals' = mapped gestures
        self._resume_mode: str | None = None
        self._orb_session = 0
        self._pointer_pinch_started: float | None = None
        self._pointer_down = False
        self._pointer_scroll_y: float | None = None
        self._pointer_last_scroll = 0.0
        self._model: tuple[Path, str] = (bundled_model_path(), "gesture")
        # Own-camera sharing state.
        self._frame_lock = Lock()
        self._own_frame: Any = None
        self._own_frame_at = 0.0
        self._own_open = False
        self._yield_request = Event()
        self._camera_released = Event()
        self._camera_released.set()
        self._source = "none"               # 'own', 'borrowed' or 'none'
        # Action thread.
        self._action_queue: queue.Queue = queue.Queue(maxsize=16)
        self._action_thread: Thread | None = None
        # Telemetry for status().
        self._infer_ms: deque[float] = deque(maxlen=60)
        self._fps_now = 0.0
        self._last_event: tuple[str, float, str] | None = None
        self._events_seen = 0
        self._stop_reason = ""
        self._calibrating_until = 0.0
        self._calibration_samples: list[tuple[float, float, float]] = []
        self._last_hand_wall = -1e9

    # ── settings ────────────────────────────────────────────────────────────

    def _save_preference(self, field: str, value: Any) -> ToolResult | None:
        """A failed preference write must not change the live gesture mapping."""
        before = getattr(self.settings, field)
        setattr(self.settings, field, value)
        try:
            self.settings.save()
        except Exception as exc:
            setattr(self.settings, field, before)
            return ToolResult(f"Gesture settings could not be saved; nothing changed: "
                              f"{str(exc)[:120]}", ok=False)
        return None

    def _apply_calibration(self) -> None:
        jitter = self.settings.calibration.get("still_speed")
        if isinstance(jitter, (int, float)) and 0.2 <= float(jitter) <= 3.0:
            self.interpreter.calibrated_still_speed = float(jitter)

    def list_gestures(self) -> ToolResult:
        lines = [f"{GESTURES[g]} → {describe_mapping(self.settings.mappings.get(g))}"
                 for g in GESTURES]
        mode = "low-power" if self.settings.low_power else "normal"
        lines.append(f"Sensitivity {self.settings.sensitivity:g} ({mode} mode). "
                     "Cursor mode is separate: ask for the hand pointer.")
        actions = ", ".join(a for a in ACTIONS if a != "none")
        lines.append(f"Actions you can map: {actions}, none, or an ORION tool.")
        return ToolResult("Hand gestures:\n" + "\n".join(f"• {line}" for line in lines))

    def set_mapping(self, gesture: str, action: str = "", tool: str = "",
                    args: dict[str, Any] | None = None,
                    tool_exists: Callable[[str], bool] | None = None) -> ToolResult:
        gesture = _norm(gesture)
        if gesture not in GESTURES:
            return ToolResult(f"'{gesture}' isn't a gesture I recognise. Gestures: "
                              + ", ".join(GESTURES) + ".", ok=False)
        action, tool = _norm(action), str(tool or "").strip()
        notes = ""
        if action and action in ACTIONS:
            kind = ACTIONS[action][1]
            if gesture == "pinch_drag" and kind == "discrete":
                return ToolResult("Pinch-drag moves in steps; map it to volume_drag, "
                                  "scroll_drag or none.", ok=False)
            if gesture != "pinch_drag" and kind == "continuous":
                return ToolResult(f"{action} follows a pinch-drag; map it to pinch_drag.",
                                  ok=False)
            spec: dict[str, Any] = {"action": action}
        else:
            tool = tool or action
            if not tool:
                return ToolResult("Say which action (or ORION tool) the gesture should run.",
                                  ok=False)
            if gesture == "pinch_drag":
                return ToolResult("Pinch-drag can only drive volume_drag or scroll_drag.",
                                  ok=False)
            if tool in _FORBIDDEN_TOOLS:
                return ToolResult(f"A gesture may not run {tool}.", ok=False)
            if tool_exists is not None and not tool_exists(tool):
                return ToolResult(f"'{tool}' is neither a gesture action nor an ORION tool. "
                                  "Actions: " + ", ".join(ACTIONS) + ".", ok=False)
            if args is not None and not isinstance(args, dict):
                return ToolResult("Mapped tool arguments must be an object.", ok=False)
            clean = dict(args or {})
            released = [key for key in _RELEASE_ARGS if key in clean]
            for key in released:
                clean.pop(key)
            if released:
                notes = (" I left out " + ", ".join(released) + ": a gesture can propose that "
                         "action, but ORION's usual confirmation still releases it.")
            spec = {"tool": tool, "args": clean}
        failed = self._save_preference("mappings", dict(self.settings.mappings, **{gesture: spec}))
        if failed is not None:
            return failed
        return ToolResult(f"{GESTURES[gesture].capitalize()} now → {describe_mapping(spec)}."
                          + notes)

    def reset_mappings(self) -> ToolResult:
        failed = self._save_preference("mappings", {k: dict(v) for k, v in DEFAULT_MAPPINGS.items()})
        if failed is not None:
            return failed
        return ToolResult("Gesture mappings are back to the defaults.")

    def set_sensitivity(self, value: Any) -> ToolResult:
        try:
            requested = float(value)
        except (TypeError, ValueError):
            words = {"low": 0.7, "normal": 1.0, "medium": 1.0, "high": 1.4}
            requested = words.get(_norm(str(value)), float("nan"))
        if not math.isfinite(requested):
            return ToolResult("Sensitivity is a number from 0.5 (deliberate) to 2 "
                              "(hair-trigger), or low/normal/high.", ok=False)
        applied = max(0.5, min(2.0, requested))
        failed = self._save_preference("sensitivity", applied)
        if failed is not None:
            return failed
        self.interpreter.set_sensitivity(applied)
        return ToolResult(f"Gesture sensitivity set to {applied:g}"
                          + (" (clamped)." if applied != requested else "."))

    def set_low_power(self, enabled: bool) -> ToolResult:
        if type(enabled) is not bool:
            return ToolResult("Low-power mode needs enabled=true or enabled=false.", ok=False)
        failed = self._save_preference("low_power", enabled)
        if failed is not None:
            return failed
        active, idle = _FPS["low_power" if enabled else "normal"]
        return ToolResult(f"Low-power gestures {'on' if enabled else 'off'}: "
                          f"{active:g} fps with a hand in view, {idle:g} fps scanning.")

    def calibrate(self, seconds: float = 3.0) -> ToolResult:
        if not self._running() or self._mode != "peripherals":
            return ToolResult("Start hand gestures first, then ask me to calibrate and hold "
                              "an open hand still in front of the camera.", ok=False)
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return ToolResult("Calibration takes a number of seconds from 2 to 8.", ok=False)
        if not math.isfinite(seconds):
            return ToolResult("Calibration takes a finite number of seconds from 2 to 8.", ok=False)
        seconds = max(2.0, min(8.0, seconds))
        self._calibration_samples = []
        self._calibrating_until = time.monotonic() + seconds
        return ToolResult(f"Calibrating: hold one open hand still in view for {seconds:g} "
                          "seconds. I'll log the result.")

    def _finish_calibration(self) -> None:
        samples = self._calibration_samples
        self._calibrating_until = 0.0
        self._calibration_samples = []
        if len(samples) < 8:
            self._say("GESTURE: calibration saw too little of your hand — try again with "
                      "it in good light.", speak=True)
            return
        speeds = []
        for (t0, x0, y0), (t1, x1, y1) in zip(samples, samples[1:]):
            if t1 > t0:
                speeds.append(math.hypot(x1 - x0, y1 - y0) / (t1 - t0))
        speeds.sort()
        typical = speeds[int(len(speeds) * 0.9)] if speeds else 0.0
        # Three times the steady-hand jitter, never below the default: a
        # shaky hand or noisy camera still gets its static gestures.
        still = max(GestureInterpreter.STILL_SPEED, min(3.0, typical * 3.0))
        self.interpreter.calibrated_still_speed = still
        self.settings.calibration = {"still_speed": round(still, 3),
                                     "jitter": round(typical, 3),
                                     "at": time.strftime("%Y-%m-%d %H:%M")}
        try:
            self.settings.save()
        except Exception as exc:
            self.bus.log.emit(f"GESTURE: calibration not saved - {exc}")
        self._say(f"GESTURE: calibrated — steady-hand jitter {typical:.2f} hand lengths/s, "
                  f"stillness bar {still:.2f}.")

    def test(self, gesture: str = "", execute: bool = False) -> ToolResult:
        """Without a gesture: an offline self-test of the recogniser on
        synthetic hands (no camera). With one: what it would do, and — with
        execute — do it once now, through the same gates."""
        gesture = _norm(gesture)
        if gesture:
            if gesture not in GESTURES:
                return ToolResult(f"'{gesture}' isn't a gesture I recognise.", ok=False)
            spec = self.settings.mappings.get(gesture, {"action": "none"})
            if not execute:
                return ToolResult(f"{GESTURES[gesture].capitalize()} → {describe_mapping(spec)}.")
            if spec.get("action") in _PERSONAL_ACTIONS:
                return ToolResult(f"{GESTURES[gesture].capitalize()} is mapped to "
                                  f"{describe_mapping(spec)}. That stands in for you answering or "
                                  "waking me, so only your own hand should make it - try the "
                                  "gesture in front of the camera.", ok=False)
            self._queue_action(gesture, spec, GestureEvent(gesture, time.monotonic(),
                                                           1 if gesture == "pinch_drag" else 0))
            return ToolResult(f"Running {describe_mapping(spec)} as if you had made that "
                              "gesture; the outcome will appear in the log.")
        # Judged at the rate the camera loop will really run, so low-power
        # mode shows what it gives up instead of reusing the normal-rate score.
        low_power = bool(self.settings.low_power)
        rate = _FPS["low_power" if low_power else "normal"][0]
        report = self_test(rate)
        if low_power:
            report += f" Measured at the low-power rate of {rate:g} fps."
        model, kind = self._find_model()
        if not self.available:
            report += " MediaPipe is not installed, so live gestures can't start."
        elif model is None:
            report += " MediaPipe is installed but no hand model is present yet."
        else:
            report += f" Live recogniser ready ({kind} model, {model.stat().st_size // 1024} KB)."
        return ToolResult(report)

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _launch(self, name: str) -> None:
        self._stop_event.clear()
        self._stop_reason = ""
        self.interpreter.reset()
        self._thread = Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _unavailable_text(self) -> str:
        return ("Gesture control needs MediaPipe, which isn't installed in this build "
                "(pip install mediapipe).")

    def start(self) -> ToolResult:
        if not self.available:
            return ToolResult(self._unavailable_text(), ok=False)
        if self._running():
            if self._stop_event.is_set():
                return ToolResult("Gesture control is still stopping; try again shortly.", ok=False)
            if self._mode == "orb":
                self._resume_mode = "peripherals"
                return ToolResult("Hand gestures will resume when orb control is turned off.")
            if self._mode == "pointer":
                return ToolResult("Hand pointer is active; stop it before starting gestures.",
                                  ok=False)
            return ToolResult("Hand gestures are already on.")
        self._mode = "peripherals"
        self._launch("orion-gesture-control")
        self.bus.log.emit("GESTURE: starting hand-gesture control.")
        self._banner("HAND GESTURES ON")
        quick = (f"{self._short('open_palm')} with an open palm held still, "
                 f"{self._short('fist')} with a fist, thumbs up/down to answer, "
                 f"swipe for {self._short('swipe_right')}/{self._short('swipe_left')}")
        return ToolResult(f"Hand gestures on — {quick}. Say 'list gestures' for the rest."
                          + (" Low-power mode." if self.settings.low_power else ""))

    def _short(self, gesture: str) -> str:
        spec = self.settings.mappings.get(gesture) or {"action": "none"}
        if "tool" in spec:
            return spec["tool"]
        return {"pause_orion": "pause", "stop_orion": "stop", "media_next": "next track",
                "media_previous": "previous track", "media_play_pause": "play/pause",
                "volume_up": "louder", "volume_down": "quieter", "wake_orion": "wake",
                "screenshot": "a screenshot", "none": "nothing"}.get(
                    spec.get("action", "none"), spec.get("action", "none").replace("_", " "))

    def start_orb(self) -> ToolResult:
        """Use the same hand tracker for the orb, without media side effects."""
        if not self.available:
            return ToolResult("Orb hand control needs mediapipe.", ok=False)
        if self._running() and self._stop_event.is_set():
            return ToolResult("Gesture control is still stopping; try again shortly.", ok=False)
        if self._mode == "orb" and self._running():
            return ToolResult("Orb hand control is already active.")
        if self._mode == "pointer" and self._running():
            return ToolResult("Stop the hand pointer before controlling the orb.", ok=False)
        self._orb_session += 1
        running = self._running()
        self._resume_mode = self._mode if running else None
        self._mode = "orb"
        self.interpreter.reset()
        if not running:
            self._launch("orion-orb-hands")
        self.bus.log.emit("GESTURE: orb hand control starting.")
        return ToolResult("Orb hand control on. Move one open hand to move the orb; "
                          "pull two open hands apart to separate its particles.")

    def stop_orb(self) -> ToolResult:
        """Leave orb mode; restore mapped gestures if they were running."""
        if self._mode != "orb":
            return ToolResult("Orb hand control was not active.")
        if self._resume_mode is not None and self._running():
            self._mode, self._resume_mode = self._resume_mode, None
            self.interpreter.reset()
        else:
            self._resume_mode = None
            self.stop(notify_orb=False)
        self.bus.orb_gesture.emit({"hands": 0})
        self.bus.log.emit("GESTURE: orb hand control stopped.")
        return ToolResult("Orb hand control stopped.")

    def start_pointer(self) -> ToolResult:
        """Arm deliberate hand control of the PC's primary-screen pointer."""
        if not self.available:
            return ToolResult("Hand pointer needs mediapipe.", ok=False)
        if self.pointer_action is None:
            return ToolResult("Hand pointer is not connected to desktop control.", ok=False)
        if self._running() and self._stop_event.is_set():
            return ToolResult("Hand tracking is still stopping; try again shortly.", ok=False)
        if self._mode == "pointer" and self._running():
            return ToolResult("Hand pointer is already active.")
        if self._mode == "orb" and self._running():
            return ToolResult("Stop orb hand control before starting the pointer.", ok=False)
        # Probe the protected desktop path before promising that control works.
        probe = self.pointer_action("ready", 0.5, 0.5)
        if not getattr(probe, "ok", False):
            return ToolResult(getattr(probe, "text", "Hand pointer is unavailable."), ok=False)
        self._release_pointer()
        self._mode = "pointer"
        if not self._running():
            self._launch("orion-hand-pointer")
        return ToolResult("Hand pointer on. Point to move; hold a pinch to click or drag; "
                          "raise two fingers and move vertically to scroll.")

    def stop_pointer(self) -> ToolResult:
        if self._mode != "pointer":
            return ToolResult("Hand pointer was not active.")
        self._release_pointer()
        self._pointer_scroll_y = None
        return self.stop()

    def peripheral_active(self) -> bool:
        return self._mode == "peripherals" and self._running()

    @property
    def orb_session(self) -> int:
        return self._orb_session

    def stop(self, *, notify_orb: bool = True) -> ToolResult:
        thread = self._thread
        was_orb = self._mode == "orb"
        self._stop_event.set()
        self._wake.set()
        self._release_pointer()
        self._pointer_scroll_y = None
        self._mode = "peripherals"
        self._resume_mode = None
        if thread is None or not thread.is_alive():
            self._thread = None
            if was_orb and notify_orb:
                self.bus.orb_gesture.emit({"hands": 0, "stopped": True,
                                           "session": self._orb_session})
            return ToolResult("Gesture control was not active.")
        if thread is not current_thread():
            thread.join(timeout=0.1)
        if not thread.is_alive():
            self._thread = None
        if was_orb and notify_orb:
            self.bus.orb_gesture.emit({"hands": 0, "stopped": True,
                                       "session": self._orb_session})
        self.bus.log.emit("GESTURE: stopped.")
        self._banner("HAND GESTURES OFF")
        return ToolResult("Hand gestures off.")

    def status(self) -> ToolResult:
        if not self._running():
            reason = f" It last stopped because {self._stop_reason}." if self._stop_reason else ""
            extra = "" if self.available else " MediaPipe is not installed."
            return ToolResult("Hand gestures are off." + reason + extra)
        mode = {"peripherals": "gestures", "orb": "orb control",
                "pointer": "hand pointer"}.get(self._mode, self._mode)
        source = {"own": "its own 640x480 camera feed",
                  "borrowed": "the face tracker's camera feed",
                  "none": "waiting for the camera"}.get(self._source, self._source)
        parts = [f"Hand gestures are on ({mode}, using {source}"]
        parts[0] += ", low-power)" if self.settings.low_power else ")"
        if self._infer_ms and self._fps_now:
            per = sorted(self._infer_ms)[len(self._infer_ms) // 2]
            parts.append(f"{self._fps_now:.0f} fps at ~{per:.0f} ms a frame — about "
                         f"{per * self._fps_now / 10:.0f}% of one CPU core")
        parts.append("a hand is in view" if self.interpreter.hand_present else "no hand in view")
        if self._last_event:
            name, at, outcome = self._last_event
            parts.append(f"last gesture {name.replace('_', ' ')} {time.monotonic() - at:.0f} s "
                         f"ago ({outcome})")
        return ToolResult("; ".join(parts) + ".")

    # ── camera sharing ──────────────────────────────────────────────────────

    def latest_frame(self) -> Any:
        """The newest frame from gesture control's OWN camera handle (copy),
        or None — lets vision borrow instead of opening a second handle."""
        with self._frame_lock:
            frame = self._own_frame
            fresh = self._own_open and time.monotonic() - self._own_frame_at < 2.0
        if frame is None or not fresh:
            return None
        try:
            return frame.copy()
        except Exception:
            return frame

    def is_capturing(self) -> bool:
        return self._own_open and self._running()

    def yield_camera(self, timeout: float = 1.5) -> bool:
        """Release our own camera handle before someone else opens the
        device. Blocks the CALLER (the tracker's capture thread, see
        share_camera_with) — never called on the GUI thread."""
        if not self._running() or self._camera_released.is_set():
            return True
        self._yield_request.set()
        self._wake.set()
        released = self._camera_released.wait(timeout)
        return released

    def share_camera_with(self, tracker: Any) -> None:
        """Make *tracker* (FaceTracker) ask us to let go of the webcam before
        its capture thread opens it.

        FaceTracker.start() runs ``Thread(target=self._loop)``, so wrapping the
        instance's ``_loop`` puts the hand-off on the tracker's own thread: the
        GUI thread that pressed 'camera lab' never waits on our release."""
        original = getattr(tracker, "_loop", None)
        if original is None or getattr(original, "_gesture_handoff", False):
            return
        engine = self

        def loop_after_handoff(*args: Any, **kwargs: Any) -> Any:
            engine.yield_camera()
            return original(*args, **kwargs)

        loop_after_handoff._gesture_handoff = True    # type: ignore[attr-defined]
        tracker._loop = loop_after_handoff

    def connect_orion(self, **kwargs: Any) -> None:
        """Wire ORION-level actions (worker, dispatcher, loop, control)."""
        connect = getattr(self.actions, "connect_orion", None)
        if connect is not None:
            connect(**kwargs)

    # ── model acquisition ───────────────────────────────────────────────────

    def _find_model(self) -> tuple[Path | None, str]:
        from .model_store import usable
        bundled = bundled_model_path()
        if bundled.is_file():
            return bundled, "gesture"
        if usable(_MODEL_PATH, size=_MODEL_SIZE):
            return _MODEL_PATH, "gesture"
        if usable(_LEGACY_MODEL_PATH):
            return _LEGACY_MODEL_PATH, "hand"
        return None, ""

    def _ensure_model(self) -> bool:
        # A build without assets/hand fetches the pinned recogniser once,
        # verified (size + SHA-256) and renamed into place only when whole.
        found, kind = self._find_model()
        if found is not None:
            self._model = (found, kind)
            return True
        from .model_store import ModelDownloadError, download_verified
        try:
            self.bus.log.emit("GESTURE: downloading the hand-gesture model (~8 MB, once)...")
            download_verified(_MODEL_URL, _MODEL_PATH, size=_MODEL_SIZE, sha256=_MODEL_SHA256)
            self._model = (_MODEL_PATH, "gesture")
            return True
        except ModelDownloadError as exc:
            self.bus.log.emit(f"GESTURE: model download failed - {exc}")
            return False

    # ── capture loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:
            self._stopped_because(f"of an error: {str(exc)[:120]}")
        finally:
            self._release_pointer()
            self._release_own(None)
            # A failed model download, MediaPipe import or camera startup must
            # uncheck the UI instead of leaving a live-looking but inert mode.
            if (current_thread() is self._thread and self._mode == "orb"
                    and not self._stop_event.is_set()):
                self.bus.orb_gesture.emit({"hands": 0, "stopped": True,
                                           "session": self._orb_session})

    def _stopped_because(self, reason: str) -> None:
        """Unexpected stop: say so, out loud once, rather than going quiet."""
        self._stop_reason = reason
        self._say(f"GESTURE: hand gestures stopped because {reason}.", speak=True)
        self._banner("HAND GESTURES OFF")

    def _create_recogniser(self) -> tuple[Any, Any, str]:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision
        import mediapipe as mp

        path, kind = self._model
        base = BaseOptions(model_asset_path=str(path))
        if kind == "gesture" and hasattr(mp_vision, "GestureRecognizer"):
            recogniser = mp_vision.GestureRecognizer.create_from_options(
                mp_vision.GestureRecognizerOptions(
                    base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=2))
            return mp, recogniser, "gesture"
        recogniser = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=base, running_mode=mp_vision.RunningMode.VIDEO, num_hands=2))
        return mp, recogniser, "hand"

    def _loop(self) -> None:
        if not self._ensure_model():
            self._stopped_because("the hand-gesture model isn't available")
            return
        mp, recogniser, kind = self._create_recogniser()
        own_capture: Any = None
        open_failures = 0
        process_failures = 0
        last_frame_at = time.monotonic()
        last_ts = 0
        try:
            while not self._stop_event.is_set():
                loop_start = time.monotonic()
                borrow = self._source_running()
                if (borrow or self._yield_request.is_set()) and own_capture is not None:
                    own_capture = self._release_own(own_capture)
                if self._yield_request.is_set():
                    self._yield_request.clear()
                    if not borrow:
                        # Someone is about to open the camera: give them a
                        # moment to become the frame source we borrow from.
                        self._stop_event.wait(0.5)
                        continue
                if not borrow and own_capture is None:
                    # Cleared BEFORE the open: a tracker starting meanwhile
                    # waits for this open to finish and be released, rather
                    # than racing it for the device.
                    self._camera_released.clear()
                    try:
                        own_capture, _backend = open_camera_capture(
                            self.camera_index, size=(640, 480), fps=15.0)
                        if own_capture is None:
                            raise RuntimeError("camera capture is disabled for this run"
                                               if _backend == "disabled" else "no camera opened")
                        with self._frame_lock:
                            self._own_open = True
                        open_failures = 0
                    except Exception as exc:
                        self._camera_released.set()
                        open_failures += 1
                        # The face tracker may still be releasing the Windows
                        # camera driver after orb mode or the lab hands it back.
                        if open_failures >= 8:
                            self._stopped_because(f"the camera is unavailable ({exc})")
                            return
                        self._stop_event.wait(0.5)
                        continue
                self._source = "borrowed" if borrow else "own"
                frame = self._read_frame(own_capture, borrow=borrow)
                if frame is not None:
                    last_frame_at = time.monotonic()
                    try:
                        started = time.perf_counter()
                        timestamp = max(last_ts + 1, int(time.monotonic() * 1000))
                        last_ts = timestamp
                        self._process_frame(mp, recogniser, frame, timestamp, kind)
                        self._infer_ms.append((time.perf_counter() - started) * 1000.0)
                        process_failures = 0
                    except Exception as exc:
                        process_failures += 1
                        if process_failures == 1:
                            self.bus.log.emit(f"GESTURE: frame processing error - {exc}")
                        if process_failures >= 5:
                            self._stopped_because("frames kept failing to process")
                            return
                elif time.monotonic() - last_frame_at > 12.0:
                    self._stopped_because("the camera stopped delivering frames")
                    return
                fps = self._target_fps()
                self._fps_now = fps
                elapsed = time.monotonic() - loop_start
                self._wake.clear()
                self._wake.wait(max(0.0, 1.0 / fps - elapsed))
        finally:
            try:
                recogniser.close()
            except Exception:
                pass
            self._release_own(own_capture)
            self._source = "none"

    def _target_fps(self) -> float:
        active, idle = _FPS["low_power" if self.settings.low_power else "normal"]
        if self._mode != "peripherals":
            return active                 # orb/pointer follow the hand continuously
        # Most of the time nobody is gesturing: scan slowly until a hand shows.
        idle_for = time.monotonic() - self._last_hand_wall
        return active if idle_for < _IDLE_AFTER_S else idle

    def _release_own(self, capture: Any) -> None:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        with self._frame_lock:
            self._own_open = False
            self._own_frame = None
        self._camera_released.set()
        return None

    def _source_running(self) -> bool:
        if self.frame_source is None:
            return False
        if self.frame_source_active is None:
            return True
        try:
            return bool(self.frame_source_active())
        except Exception:
            return False

    def _read_frame(self, own_capture: Any, *, borrow: bool = False) -> Any:
        if borrow and self.frame_source is not None:
            try:
                return self.frame_source()
            except Exception:
                return None
        if own_capture is None:
            return None
        ok, frame = own_capture.read()
        if not ok:
            return None
        with self._frame_lock:
            self._own_frame = frame
            self._own_frame_at = time.monotonic()
        return frame

    def _process_frame(self, mp: Any, recogniser: Any, frame: Any,
                       timestamp_ms: int | None = None, kind: str = "gesture") -> None:
        import cv2
        height, width = frame.shape[:2]
        if width > _WORK_WIDTH:
            # INTER_LINEAR: a third of INTER_AREA's cost on a 1080p borrowed
            # frame, and the palm detector letterboxes to 192 px regardless.
            frame = cv2.resize(frame, (_WORK_WIDTH, max(1, round(height * _WORK_WIDTH / width))),
                               interpolation=cv2.INTER_LINEAR)
            height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
        if kind == "gesture" and hasattr(recogniser, "recognize_for_video"):
            result = recogniser.recognize_for_video(mp_image, timestamp)
        else:
            result = recogniser.detect_for_video(mp_image, timestamp)
        hands = list(getattr(result, "hand_landmarks", None) or [])
        if self._mode == "orb":
            self._process_orb_hands(hands)
            return
        if self._mode == "pointer":
            self._process_pointer_hands(hands)
            return
        gestures = list(getattr(result, "gestures", None) or [])
        handedness = list(getattr(result, "handedness", None) or [])
        observations = []
        for i, landmarks in enumerate(hands):
            label, score, side = "", 0.0, ""
            if i < len(gestures) and gestures[i]:
                label = str(getattr(gestures[i][0], "category_name", "") or "")
                score = float(getattr(gestures[i][0], "score", 0.0) or 0.0)
            if i < len(handedness) and handedness[i]:
                side = str(getattr(handedness[i][0], "category_name", "") or "")
            observations.append(HandObservation(landmarks, label, score, side))
        self._handle_hands(observations, time.monotonic(), width / max(1, height))

    def _handle_hands(self, hands: Sequence[Any], now: float, aspect: float = 4.0 / 3.0) -> None:
        """Mapped-gesture mode: interpret, then act on each event."""
        if hands:
            self._last_hand_wall = now
        events = self.interpreter.observe(hands, now, aspect)
        if self._calibrating_until:
            self._collect_calibration(hands, now, aspect)
        for event in events:
            self._on_event(event)

    def _collect_calibration(self, hands: Sequence[Any], now: float, aspect: float) -> None:
        if now >= self._calibrating_until:
            self._finish_calibration()
            return
        if not hands:
            return
        first = hands[0]
        landmarks = first.landmarks if isinstance(first, HandObservation) else first
        shape = measure_hand(landmarks, aspect)
        if shape is not None and classify_shape(shape) == "open_palm":
            self._calibration_samples.append((now, shape.centre[0] / shape.scale,
                                              shape.centre[1] / shape.scale))

    def _on_event(self, event: GestureEvent) -> None:
        self._events_seen += 1
        spec = self.settings.mappings.get(event.name) or {"action": "none"}
        if event.name != "pinch_drag":
            self.bus.log.emit(f"GESTURE: {event.name.replace('_', ' ')} → "
                              f"{describe_mapping(spec)}")
            self._last_event = (event.name, event.at, describe_mapping(spec))
        if spec.get("action", "") == "none" and "tool" not in spec:
            return
        self._queue_action(event.name, spec, event)

    # ── action thread ───────────────────────────────────────────────────────

    def _queue_action(self, gesture: str, spec: dict[str, Any], event: GestureEvent) -> None:
        if self._action_thread is None or not self._action_thread.is_alive():
            self._action_thread = Thread(target=self._action_worker,
                                         name="orion-gesture-actions", daemon=True)
            self._action_thread.start()
        try:
            self._action_queue.put_nowait((gesture, spec, event))
        except queue.Full:
            self.bus.log.emit("GESTURE: actions are backing up; dropped one.")

    def _action_worker(self) -> None:
        # Its own thread: a media-session call or a screenshot can take a
        # few hundred ms, and the capture loop must not stall behind it.
        while True:
            try:
                gesture, spec, event = self._action_queue.get(timeout=30.0)
            except queue.Empty:
                return                        # idle: let the thread go
            try:
                result = self.actions.perform(spec, event)
            except Exception as exc:
                result = ToolResult(f"{describe_mapping(spec)} failed: {str(exc)[:160]}",
                                    ok=False)
            if gesture != "pinch_drag" or not getattr(result, "ok", True):
                self.bus.log.emit(f"GESTURE: {gesture.replace('_', ' ')} → "
                                  f"{getattr(result, 'text', result)}")
            if self._last_event and self._last_event[0] == gesture:
                self._last_event = (gesture, self._last_event[1],
                                    first_words(getattr(result, "text", "")))

    # ── orb and pointer modes (unchanged behaviour) ─────────────────────────

    def _release_pointer(self) -> None:
        if self._pointer_down and self.pointer_action is not None:
            try:
                self.pointer_action("up", 0.5, 0.5)
            except Exception:
                pass
        self._pointer_down = False
        self._pointer_pinch_started = None

    def _process_orb_hands(self, hands: list) -> None:
        """Emit camera-space palm motion and two-hand spread for the 3-D orb."""
        open_hands = []
        for hand in hands:
            wrist = hand[_WRIST]
            extended = 0
            for tip_index, pip_index in zip(_FINGER_TIPS, _FINGER_PIPS):
                tip, pip = hand[tip_index], hand[pip_index]
                tip_r2 = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
                pip_r2 = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
                extended += tip_r2 > pip_r2 * 1.25
            if extended >= 3:
                open_hands.append(hand)
        if not open_hands:
            self.bus.orb_gesture.emit({"hands": 0})
            return
        centres = []
        for hand in open_hands[:2]:
            palm = [hand[i] for i in _PALM]
            centres.append((sum(p.x for p in palm) / 5,
                            sum(p.y for p in palm) / 5))
        x = sum(p[0] for p in centres) / len(centres)
        y = sum(p[1] for p in centres) / len(centres)
        # Mirror camera x so moving a hand right moves the orb right on screen.
        sample = {"hands": len(centres), "x": max(-1.0, min(1.0, (0.5 - x) * 2)),
                  "y": max(-1.0, min(1.0, (0.5 - y) * 2)), "spread": 0.0,
                  "twist": 0.0}
        if len(centres) == 2:
            left, right = sorted(centres)
            distance = ((right[0] - left[0]) ** 2 +
                        (right[1] - left[1]) ** 2) ** 0.5
            sample["spread"] = max(0.0, min(1.0, (distance - 0.16) / 0.48))
            sample["twist"] = max(-0.8, min(0.8, (right[1] - left[1]) * 1.4))
        self.bus.orb_gesture.emit(sample)

    def _process_pointer_hands(self, hands: list) -> None:
        """Require a stable pinch before pressing; release on hand loss."""
        if self._stop_event.is_set() or self._mode != "pointer":
            self._release_pointer()
            return
        if self.pointer_action is None or not hands:
            self._release_pointer()
            self._pointer_scroll_y = None
            return
        points = hands[0]
        extended = self._extended_fingers(points)
        index = points[_FINGER_TIPS[0]]
        now = time.monotonic()
        if extended == [True, True, False, False]:
            self._release_pointer()
            if self._pointer_scroll_y is not None and now - self._pointer_last_scroll >= 0.3:
                change = index.y - self._pointer_scroll_y
                if abs(change) >= 0.08:
                    self.pointer_action("scroll_up" if change < 0 else "scroll_down",
                                        index.x, index.y)
                    self._pointer_last_scroll = now
                    self._pointer_scroll_y = index.y
            else:
                self._pointer_scroll_y = index.y
            return
        self._pointer_scroll_y = None
        if extended != [True, False, False, False]:
            self._release_pointer()
            return
        thumb = points[_THUMB_TIP]
        distance = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
        pinched = distance < 0.045
        if pinched:
            if self._pointer_pinch_started is None:
                self._pointer_pinch_started = now
            elif not self._pointer_down and now - self._pointer_pinch_started >= 0.25:
                result = self.pointer_action("down", index.x, index.y)
                self._pointer_down = bool(getattr(result, "ok", False))
        else:
            self._release_pointer()
        self.pointer_action("move", index.x, index.y)

    def _extended_fingers(self, landmarks: Sequence[Any]) -> list[bool]:
        """One bool per non-thumb finger (index, middle, ring, pinky), by the
        rotation-invariant wrist-distance ratio (see measure_hand)."""
        shape = measure_hand(landmarks, 1.0)
        if shape is None:
            return [False, False, False, False]
        return [ratio >= _EXTENDED_RATIO for ratio in shape.ratios]

    # ── small helpers ───────────────────────────────────────────────────────

    def _banner(self, text: str) -> None:
        banner = getattr(self.bus, "banner", None)
        if banner is not None:
            try:
                banner.emit(text, 1)
            except Exception:
                pass

    def _say(self, line: str, speak: bool = False) -> None:
        try:
            self.bus.log.emit(line)
        except Exception:
            pass
        request = getattr(self.bus, "speak_request", None) if speak else None
        if request is not None:
            try:
                request.emit(line.split(": ", 1)[-1])
            except Exception:
                pass


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def first_words(text: str, limit: int = 60) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def self_test(frames_per_second: float = 15.0) -> str:
    """Run the interpreter over synthetic hands (no camera, no MediaPipe) and
    report how many gestures it recognised and how many it invented."""
    import random
    rng = random.Random(7)
    expected = ["open_palm", "fist", "thumbs_up", "thumbs_down", "victory",
                "swipe_left", "swipe_right", "swipe_up", "swipe_down", "wave"]
    hits = 0
    false = 0
    for gesture in expected:
        events = [e.name for e in run_sequence(synthetic_sequence(gesture, rng,
                                                                  frames_per_second))]
        hits += events.count(gesture) == 1
        false += len([e for e in events if e != gesture]) + max(0, events.count(gesture) - 1)
    idle = [e.name for e in run_sequence(synthetic_sequence("idle", rng, frames_per_second))]
    false += len(idle)
    return (f"Recogniser self-test on synthetic hands: {hits}/{len(expected)} gestures "
            f"recognised, {false} false trigger(s) including a 20-second idle hand.")


def synthetic_sequence(kind: str, rng: Any, fps: float = 15.0, jitter: float = 0.03,
                       ) -> list[tuple[float, list[Any]]]:
    """(time, hands) frames for one scripted gesture. Positions jitter by
    *jitter* hand lengths per landmark per frame, as a real tracker does."""
    size = 0.2
    dt = 1.0 / fps
    frames: list[tuple[float, list[Any]]] = []
    t = 0.0

    def add(pose: str, centre: tuple[float, float], angle: float = 0.0) -> None:
        nonlocal t
        hand = synthetic_hand(pose, centre=centre, size=size, angle=angle)
        noisy = [(x + rng.gauss(0, jitter * size / (4 / 3)), y + rng.gauss(0, jitter * size), z)
                 for x, y, z in hand]
        frames.append((t, [noisy]))
        t += dt

    def blank(seconds: float) -> None:
        nonlocal t
        for _ in range(int(seconds * fps)):
            frames.append((t, []))
            t += dt

    blank(0.3)
    if kind in ("open_palm", "fist", "thumbs_up", "thumbs_down", "victory", "point"):
        for _ in range(int(1.6 * fps)):
            add(kind, (0.5, 0.5))
    elif kind.startswith("swipe_"):
        direction = kind.split("_", 1)[1]
        # Image x is mirrored: the user's right is the image's left.
        vec = {"right": (-1, 0), "left": (1, 0), "up": (0, -1), "down": (0, 1)}[direction]
        start = (0.5 - vec[0] * 0.18, 0.5 - vec[1] * 0.16)
        for _ in range(int(0.5 * fps)):
            add("open_palm", start)
        steps = max(3, int(0.3 * fps))
        for i in range(1, steps + 1):
            f = i / steps
            add("open_palm", (start[0] + vec[0] * 0.36 * f, start[1] + vec[1] * 0.32 * f))
        end = (start[0] + vec[0] * 0.36, start[1] + vec[1] * 0.32)
        for _ in range(int(0.45 * fps)):
            add("open_palm", end)
    elif kind == "wave":
        for _ in range(int(0.4 * fps)):
            add("open_palm", (0.5, 0.5))
        n = int(1.6 * fps)
        for i in range(n):
            phase = 2 * math.pi * 2.5 * i / fps
            add("open_palm", (0.5 + 0.09 * math.sin(phase), 0.5), angle=15 * math.sin(phase))
        for _ in range(int(0.5 * fps)):
            add("open_palm", (0.5, 0.5))
    elif kind == "idle":
        # A relaxed, half-curled hand drifting about, changing pose often
        # enough that none is held, entering and leaving the frame.
        poses = ["relaxed", "relaxed", "point", "relaxed", "victory", "relaxed", "open_palm"]
        x, y = 0.5, 0.7
        for second in range(20):
            pose = poses[second % len(poses)]
            if second % 5 == 4:
                blank(0.6)
                x, y = 0.5, 0.95
            for _ in range(int(0.45 * fps)):
                x = min(0.85, max(0.15, x + rng.gauss(0, 0.012)))
                y = min(0.95, max(0.3, y + rng.gauss(-0.004, 0.012)))
                add(pose, (x, y), angle=rng.gauss(0, 8))
            for _ in range(int(0.55 * fps)):
                x = min(0.85, max(0.15, x + rng.gauss(0, 0.012)))
                y = min(0.95, max(0.3, y + rng.gauss(0, 0.012)))
                add("relaxed", (x, y), angle=rng.gauss(0, 8))
    blank(0.6)
    return frames


def run_sequence(frames: Sequence[tuple[float, list[Any]]],
                 interpreter: GestureInterpreter | None = None) -> list[GestureEvent]:
    interpreter = interpreter or GestureInterpreter()
    events: list[GestureEvent] = []
    for t, hands in frames:
        events.extend(interpreter.observe(hands, t, 4.0 / 3.0))
    return events
