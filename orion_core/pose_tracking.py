"""
Body tracking — MediaPipe Pose, read as gestures a person would recognise.

MediaPipe's PoseLandmarker finds up to two people and 33 landmarks on each
(nose, shoulders, elbows, wrists, hips, knees …) in normalised image
coordinates. On its own that is 66 points of numbers; what makes it useful is
reading them as the things people actually do in front of a camera:

    person          someone is in view (more reliable than motion alone)
    hand_raised     either wrist clearly above its shoulder
    left_hand_up / right_hand_up    — the PERSON's left and right
    both_hands_up   both wrists above the head
    arms_out        both arms held out level to the sides (a T)

Those names are what vision rules trigger on: "when I raise my hand, pause the
music" is a pose rule wired to a workflow.

The interpretation (``interpret``) is a pure function over landmarks so it is
testable without a camera or the model. MediaPipe itself is imported only
when a pose is first read, and its 5.5 MB model is downloaded then — the same
first-use convention gesture control uses — through model_store, so an
interrupted download can never leave a broken model behind.

The standalone .exe is built WITHOUT MediaPipe; there this reports that pose
tracking is not available instead of failing.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

from .constants import CONFIG_DIR

POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
POSE_MODEL_PATH = CONFIG_DIR / "models" / "pose_landmarker_lite.task"
POSE_MODEL_MB = 5.5

# MediaPipe's 33-point body model — the indices this module reads.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

VISIBLE = 0.5                    # landmark visibility that counts as seen
RAISE_MARGIN = 0.04              # wrist must be this far above the shoulder

POSE_GESTURES = ("person", "hand_raised", "left_hand_up", "right_hand_up",
                 "both_hands_up", "arms_out")

#: Spoken names for the gestures.
GESTURE_ALIASES = {
    "hand up": "hand_raised", "raised hand": "hand_raised", "raise my hand": "hand_raised",
    "hand raised": "hand_raised", "wave": "hand_raised", "waving": "hand_raised",
    "left hand": "left_hand_up", "right hand": "right_hand_up",
    "both hands": "both_hands_up", "hands up": "both_hands_up", "arms up": "both_hands_up",
    "t pose": "arms_out", "t-pose": "arms_out", "arms out": "arms_out",
    "someone": "person", "a person": "person", "body": "person",
}


def canonical_gesture(name: str) -> str | None:
    text = " ".join(str(name or "").strip().lower().replace("_", " ").split())
    if text.replace(" ", "_") in POSE_GESTURES:
        return text.replace(" ", "_")
    return GESTURE_ALIASES.get(text)


@dataclass(frozen=True)
class PersonPose:
    landmarks: tuple[tuple[float, float, float], ...]    # (x, y, visibility), normalised
    gestures: frozenset[str]

    @property
    def centre(self) -> tuple[float, float]:
        seen = [(x, y) for x, y, v in self.landmarks if v >= VISIBLE] or \
               [(x, y) for x, y, _ in self.landmarks]
        return (sum(p[0] for p in seen) / len(seen), sum(p[1] for p in seen) / len(seen))


@dataclass(frozen=True)
class PoseReading:
    people: tuple[PersonPose, ...]

    @property
    def gestures(self) -> frozenset[str]:
        found: set[str] = set()
        for person in self.people:
            found |= person.gestures
        return frozenset(found)

    def describe(self) -> str:
        if not self.people:
            return "No one is in view."
        count = len(self.people)
        who = "one person" if count == 1 else f"{count} people"
        doing = sorted(self.gestures - {"person"})
        if not doing:
            return f"I can see {who}, standing or sitting normally."
        words = {"hand_raised": "a hand raised", "left_hand_up": "their left hand up",
                 "right_hand_up": "their right hand up", "both_hands_up": "both hands up",
                 "arms_out": "arms held out"}
        return f"I can see {who}: " + ", ".join(words.get(g, g) for g in doing) + "."


def interpret(landmarks: Sequence[Sequence[float]]) -> frozenset[str]:
    """Gestures from one person's 33 (x, y, visibility) landmarks.

    Image y grows DOWNWARDS, so "above" is a smaller y. Left and right are the
    person's own, as MediaPipe labels them."""
    if len(landmarks) < 25:
        return frozenset()

    def seen(index: int) -> bool:
        return landmarks[index][2] >= VISIBLE

    def y(index: int) -> float:
        return landmarks[index][1]

    def x(index: int) -> float:
        return landmarks[index][0]

    found = {"person"}
    left_up = seen(LEFT_WRIST) and seen(LEFT_SHOULDER) and \
        y(LEFT_WRIST) < y(LEFT_SHOULDER) - RAISE_MARGIN
    right_up = seen(RIGHT_WRIST) and seen(RIGHT_SHOULDER) and \
        y(RIGHT_WRIST) < y(RIGHT_SHOULDER) - RAISE_MARGIN
    if left_up:
        found.add("left_hand_up")
    if right_up:
        found.add("right_hand_up")
    if left_up or right_up:
        found.add("hand_raised")
    if left_up and right_up and seen(NOSE) and \
            y(LEFT_WRIST) < y(NOSE) and y(RIGHT_WRIST) < y(NOSE):
        found.add("both_hands_up")
    if all(seen(i) for i in (LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER)):
        shoulder_width = abs(x(LEFT_SHOULDER) - x(RIGHT_SHOULDER))
        level = (abs(y(LEFT_WRIST) - y(LEFT_SHOULDER)) < 0.08 and
                 abs(y(RIGHT_WRIST) - y(RIGHT_SHOULDER)) < 0.08)
        spread = abs(x(LEFT_WRIST) - x(RIGHT_WRIST))
        if level and shoulder_width > 0.02 and spread > 2.6 * shoulder_width:
            found.add("arms_out")
    return frozenset(found)


class PoseTracker:
    """Lazily-built MediaPipe PoseLandmarker. ``read`` is blocking: run it in
    a worker thread."""

    MAX_PEOPLE = 2

    def __init__(self) -> None:
        self._landmarker: Any = None
        self._mp: Any = None
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._last_ts = 0
        self.error = ""

    @staticmethod
    def library_available() -> bool:
        return importlib.util.find_spec("mediapipe") is not None

    @property
    def available(self) -> bool:
        return self.library_available()

    def describe_state(self) -> str:
        if not self.library_available():
            return ("pose tracking needs MediaPipe, which this build of ORION does "
                    "not include")
        from .model_store import usable
        if not usable(POSE_MODEL_PATH):
            return (f"pose tracking ready — its {POSE_MODEL_MB} MB model downloads "
                    "the first time it is used")
        return "pose tracking ready"

    def _ensure(self) -> bool:
        if self._landmarker is not None:
            return True
        if not self.library_available():
            self.error = "MediaPipe is not installed"
            return False
        from .model_store import ModelDownloadError, download_verified, usable
        if not usable(POSE_MODEL_PATH):
            try:
                download_verified(POSE_MODEL_URL, POSE_MODEL_PATH)
            except ModelDownloadError as exc:
                self.error = f"could not download the pose model: {exc}"
                return False
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision as mp_vision
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(
                mp_vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_poses=self.MAX_PEOPLE,
                ))
            self._mp = mp
            self.error = ""
            return True
        except Exception as exc:
            self.error = f"MediaPipe would not start: {exc}"
            self._landmarker = None
            return False

    def read(self, frame: Any) -> PoseReading | None:
        """People and their gestures in a BGR frame; None if unavailable."""
        if frame is None or getattr(frame, "ndim", 0) != 3:
            return None
        with self._lock:
            if not self._ensure():
                return None
            try:
                import cv2
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                # VIDEO mode tracks between frames (steadier than IMAGE mode)
                # and needs strictly increasing timestamps.
                stamp = max(self._last_ts + 1, int((time.monotonic() - self._started) * 1000))
                self._last_ts = stamp
                result = self._landmarker.detect_for_video(image, stamp)
            except Exception as exc:
                self.error = f"pose read failed: {exc}"
                return None
        people = []
        for person in getattr(result, "pose_landmarks", None) or ():
            points = tuple((float(p.x), float(p.y), float(getattr(p, "visibility", 1.0) or 0.0))
                           for p in person)
            people.append(PersonPose(points, interpret(points)))
        return PoseReading(tuple(people))

    def close(self) -> None:
        with self._lock:
            if self._landmarker is not None:
                try:
                    self._landmarker.close()
                except Exception:
                    pass
                self._landmarker = None


__all__ = ["POSE_GESTURES", "PersonPose", "PoseReading", "PoseTracker",
           "canonical_gesture", "interpret"]
