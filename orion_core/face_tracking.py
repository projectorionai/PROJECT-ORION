"""
FaceTracker — webcam head tracking that lets the avatar hold eye contact.

Design rules (from the Ultron hand-tracker study, see docs/ULTRON_ANALYSIS.md):

    • hysteresis everywhere — presence engages after 3 consecutive detections
      and releases after 8 misses (the AvatarAnimationManager applies this;
      the tracker just reports honest per-frame samples);
    • smooth before you steer — samples are EMA-smoothed here AND pose-smoothed
      in the animation manager, so the head never snaps;
    • performance stays high — 320-px greyscale frames at ~12 Hz on a daemon
      thread, detection every frame is cheap at that size; the capture loop
      never touches the Qt thread (samples go out through a callback the app
      wires to a queued bus signal);
    • graceful degradation — no OpenCV, no camera, or a camera in use simply
      means ``available`` is False / ``start`` returns an explanation, and the
      avatar carries on with idle sway.

The tracker is deliberately dumb about *meaning*: it reports where the largest
face is.  Subtlety (clamped follow angles, drift-home on loss) lives in the
avatar layer, so the same tracker could later drive other systems.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from typing import Any, Callable

from pathlib import Path

from .constants import resource_path
from .data import ToolResult
from .utils import open_camera_capture

#: What ORION reports when he is watching but cannot pick out a face — either
#: nobody is there, or no detection model is installed.
_NO_FACE: dict[str, Any] = {"present": False, "x": 0.0, "y": 0.0, "size": 0.0}

#: A cascade shipped with ORION, used when OpenCV has none of its own. Absent
#: by default: the file is ~900 KB and the live feed does not need it.
ASSET_CASCADE = (Path(__file__).resolve().parent.parent
                 / "assets" / "haarcascade_frontalface_default.xml")


def _cv2_available() -> bool:
    """Whether OpenCV can be imported, WITHOUT importing it.

    find_spec only inspects the module finders, so asking the question costs
    nothing — where a real `import cv2` at module scope would pull OpenCV
    into every process that touches this module, tracker running or not.
    The actual import happens inside the worker methods below."""
    return importlib.util.find_spec("cv2") is not None


__all__ = ["FaceTracker"]

#: How often the capture loop grabs. MEASURED on a Logitech C920 through
#: Media Foundation: 30.7 fps at 1920x1080, which is that camera's hardware
#: ceiling at full resolution — 60 fps exists only at 720p and below. Asking
#: for more than the sensor can give buys nothing and burns a core spinning
#: on repeat frames, so this tracks the real figure rather than a wish.
_TARGET_FPS = float(os.getenv("ORION_CAMERA_FPS", "30"))

#: What ORION asks the camera for. 1080p is what this camera is, and the
#: shared frame feeds the camera lab, on-demand vision and gesture control as
#: well as face tracking — every one of which is better off with the pixels.
_CAPTURE_WIDTH = int(os.getenv("ORION_CAMERA_WIDTH", "1920"))
_CAPTURE_HEIGHT = int(os.getenv("ORION_CAMERA_HEIGHT", "1080"))

#: What detection runs on. See _load_detector for the measurement.
_DETECT_WIDTH = int(os.getenv("ORION_FACE_DETECT_WIDTH", "640"))
_DETECT_HEIGHT = 360

#: The working copy detection runs on. Detection does not need 1080p and
#: paying for it per frame is what would make 30 fps unaffordable.
_FRAME_WIDTH = 320


class FaceTracker:
    """Webcam face-position tracker on a daemon thread."""

    available = _cv2_available()

    def __init__(self, callback: Callable[[dict[str, Any]], None],
                 camera_index: int | None = None) -> None:
        self.callback = callback
        # An explicit index is honoured as given; otherwise the camera is
        # re-resolved BY NAME at every start, so a camera chosen (or plugged
        # in) while ORION runs is used without a restart.
        self._explicit_index = camera_index is not None
        self._prefer_backend = ""
        if camera_index is None:
            from . import camera_devices as cd
            camera_index, self._prefer_backend = cd.resolve_device(hi_res=True)
        self.camera_index = int(camera_index)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        #: Bumped for every capture thread. A thread that is not current may
        #: finish, but may not touch shared state (see _loop).
        self._generation = 0
        self._stop_requested_at = 0.0
        self._smooth: tuple[float, float] | None = None
        self.running = False
        self.last_sample: dict[str, Any] = {"present": False, "x": 0.0, "y": 0.0}
        self.frames = 0
        self.detections = 0
        self.error = ""
        #: Why faces are not being detected, when the camera is otherwise fine.
        #: Kept apart from `error`, which means the CAPTURE failed — conflating
        #: the two is what made a missing data file read as "no camera".
        self.detection_error = ""
        self._logged: set[str] = set()
        # Cache the most recent raw frame so on-demand vision can BORROW the
        # already-open camera instead of opening a second handle (Windows
        # refuses that with the -1072873821 MSMF grab failure).
        self._frame_lock = threading.Lock()
        self._latest_frame: Any = None
        self._frame_time = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> ToolResult:
        if not self.available:
            return ToolResult(
                "Face tracking needs OpenCV (`pip install opencv-python`); "
                "it is not installed, so I'll keep the avatar on idle sway.",
                ok=False)
        if self.running:
            return ToolResult("Face tracking is already running.")
        if self._thread is not None and self._thread.is_alive():
            if time.monotonic() - self._stop_requested_at < 3.0:
                return ToolResult("The camera is still stopping. Try again in a moment.", ok=False)
            # A driver has held the old thread inside read() for seconds after
            # it was told to stop. Waiting on it is how the camera lab stopped
            # starting at all until ORION was restarted. It keeps its own stop
            # event (still set) and its own generation, so it exits whenever
            # the driver lets go and can no longer touch shared state.
            self._log("CAMERA: the previous capture is stuck in the driver - "
                      "starting a fresh one.")
        if not self._explicit_index:
            try:
                from . import camera_devices as cd
                self.camera_index, self._prefer_backend = cd.resolve_device(hi_res=True)
            except Exception:
                pass
        with self._frame_lock:
            self._latest_frame = None
            self._frame_time = 0.0
        self._generation += 1
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, args=(self._stop, self._generation),
            name="orion-face-tracker", daemon=True)
        self.running = True
        self.error = ""
        self._thread.start()
        return ToolResult("Face tracking engaged — I'll keep my eyes on you.")

    def use_camera(self, index: int, backend: str = "") -> None:
        """Switch to a specific camera from the next start()."""
        self.camera_index = int(index)
        self._prefer_backend = backend
        self._explicit_index = True

    def follow_default_camera(self) -> None:
        """Go back to the remembered camera (resolved by name at start())."""
        self._explicit_index = False

    def stop(self) -> ToolResult:
        self._stop_requested_at = time.monotonic()
        self._stop.set()
        self.running = False
        # Retain the thread until it really exits. A driver may block in read()
        # beyond the join timeout; clearing _stop for a second thread then
        # resurrected the first capture loop and opened two camera handles.
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._frame_lock:
            self._latest_frame = None
            self._frame_time = 0.0
        self._emit({"present": False, "x": 0.0, "y": 0.0, "size": 0.0})
        return ToolResult("Face tracking disengaged.")

    def status(self) -> ToolResult:
        if not self.available:
            return ToolResult("Face tracking unavailable: OpenCV not installed.")
        if not self.running:
            return ToolResult("Face tracking is off. Say the word and I'll watch.")
        s = self.last_sample
        if self.detection_error:
            # The camera IS running; only the face detector is missing. Saying
            # "face tracking is off" here would send someone hunting for a
            # camera fault that does not exist.
            detail = (f"The camera is live — {self.frames} frame(s) captured — "
                      f"but I cannot pick out faces: {self.detection_error}. "
                      "The feed, vision and gestures all work.")
            return ToolResult(detail)
        seen = "watching you now" if s.get("present") else "no face in view"
        detail = (f"Face tracking live — {seen}. "
                  f"{self.frames} frame(s), {self.detections} detection(s).")
        if self.error:
            detail += f" Last fault: {self.error}"
        return ToolResult(detail)

    def latest_frame(self) -> Any:
        """The most recent raw BGR frame the tracker captured (a copy), or None.

        Lets other subsystems (on-demand vision / camera analysis) borrow the
        live camera without opening a second handle — the second handle is what
        fails with -1072873821 (MSMF) on Windows."""
        with self._frame_lock:
            frame = self._latest_frame
            fresh = self.running and time.monotonic() - self._frame_time < 2.0
        if frame is None or not fresh:
            return None
        try:
            return frame.copy()
        except Exception:
            return frame

    def is_capturing(self) -> bool:
        return self.running

    # ── capture loop ──────────────────────────────────────────────────────────

    def _current(self, generation: int) -> bool:
        return generation == self._generation

    def _open_failure_reason(self) -> str:
        """Why the camera would not open, in words someone can act on."""
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                "Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager"
                "\\ConsentStore\\webcam")
            value, _kind = winreg.QueryValueEx(key, "Value")
            if str(value).lower() == "deny":
                return ("Windows is blocking camera access - turn on Settings > "
                        "Privacy & security > Camera > 'Let desktop apps access your camera'")
        except Exception:
            pass
        try:
            from . import camera_devices as cd

            cameras = cd.all_cameras()
            if not cameras:
                return "no camera is connected (none found by Windows)"
            names = ", ".join(c.label() for c in cameras)
            return (f"camera {self.camera_index} would not open - it may be in use by "
                    f"another app (Teams, Discord, a browser tab) or was unplugged. "
                    f"Cameras found: {names}")
        except Exception:
            return "camera unavailable"

    def _loop(self, stop: threading.Event | None = None,
              generation: int | None = None) -> None:  # pragma: no cover - requires a physical camera
        stop = stop if stop is not None else self._stop
        generation = self._generation if generation is None else generation
        capture = None
        failures = 0
        try:
            import cv2
            # Size and rate are negotiated INSIDE the opener, before it
            # validates with a read. Setting them here afterwards makes Media
            # Foundation renegotiate a stream it has already delivered from,
            # which fails with "Failed to select stream 0" and then hands back
            # a malformed Mat — zero frames, and a camera that looks broken.
            capture, backend = open_camera_capture(
                self.camera_index, hi_res=True,
                size=(_CAPTURE_WIDTH, _CAPTURE_HEIGHT), fps=_TARGET_FPS,
                prefer=self._prefer_backend)
            if capture is None or not capture.isOpened():
                if self._current(generation):
                    self.error = self._open_failure_reason()
                    self._log(f"CAMERA: {self.error}")
                    self.running = False
                return
            # Preserve small PCB markings for consumers of the shared frame.
            # Detection still uses its 320px working copy; unsupported camera
            # properties are advisory and never prevent capture.
            try:
                self.frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                self._log(f"CAMERA: {self.frame_size[0]}x{self.frame_size[1]} "
                          f"via {backend} at up to {_TARGET_FPS:.0f} fps.")
            except Exception:
                pass
            # YuNet first: it is the only one with a model on this machine,
            # and the only one that yields landmarks. The cascade stays as a
            # fallback for an installation that still has its XMLs.
            detector = self._load_detector()
            cascade = None if detector is not None else self._load_cascade()
            interval = 1.0 / _TARGET_FPS
            while not stop.is_set() and self._current(generation):
                started = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    self.error = "frame read failed"
                    with self._frame_lock:
                        self._latest_frame = None
                    if failures >= 20:
                        # Ten seconds without a frame: unplugged, or taken by
                        # another app. Saying so beats a feed that looks live
                        # and shows nothing.
                        self.error = ("the camera stopped delivering frames - "
                                      "unplugged, or taken by another app")
                        self._log(f"CAMERA: {self.error}")
                        break
                    stop.wait(0.5)
                    continue
                failures = 0
                self.frames += 1
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_time = time.monotonic()
                self.error = ""
                # Detection is OPTIONAL. Capturing frames is not: the camera
                # lab, on-demand vision and gesture control all borrow this
                # one handle, and they need pictures, not faces. A detector
                # that cannot run must therefore cost the face sample and
                # nothing else.
                if detector is not None:
                    try:
                        sample = self._detect_yunet(frame, detector)
                    except Exception as exc:
                        detector = None
                        self.detection_error = str(exc)[:120]
                        self._log(f"CAMERA: face detection disabled ({self.detection_error}); "
                                  "the live feed keeps running.")
                        sample = _NO_FACE.copy()
                elif cascade is not None:
                    try:
                        sample = self._detect(frame, cascade)
                    except Exception as exc:
                        cascade = None
                        self.detection_error = str(exc)[:120]
                        self._log(f"CAMERA: face detection disabled ({self.detection_error}); "
                                  "the live feed keeps running.")
                        sample = _NO_FACE.copy()
                else:
                    sample = _NO_FACE.copy()
                self._emit(sample)
                # Pace to the target rate; detection at 320 px is ~2-4 ms.
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    stop.wait(remaining)
        except Exception as exc:
            if self._current(generation):
                self.error = str(exc)[:120]
        finally:
            if self._current(generation):
                self.running = False
                with self._frame_lock:
                    self._latest_frame = None
                    self._frame_time = 0.0
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass

    def _load_detector(self) -> Any:
        """YuNet, OpenCV's own face detector, or None.

        Preferred over the Haar cascade for two reasons beyond accuracy.
        OpenCV 5 stopped shipping the cascade XMLs at all, so that path has no
        model to load on this machine; and YuNet returns five LANDMARKS with
        every face — both eyes, the nose tip and both mouth corners — which a
        cascade cannot do. It is what makes "track my mouth, nose and eyes"
        possible without adding a heavyweight dependency: the detector is
        built into cv2 and the model is 227 KB.

        Detection runs on a downscaled copy. MEASURED: 98 ms on a full
        1920x1080 frame against 6.7 ms at 640x360 — at 30 fps the first is
        three frames of budget for one detection, and the second is a fifth
        of one.
        """
        import cv2

        if not hasattr(cv2, "FaceDetectorYN"):
            return None
        model = resource_path("assets", "face",
                              "face_detection_yunet_2023mar.onnx")
        try:
            if not Path(model).is_file():
                self.detection_error = "no face-detection model installed"
                self._log("CAMERA: YuNet model missing; falling back to the "
                          "Haar cascade if one exists.")
                return None
            return cv2.FaceDetectorYN.create(
                str(model), "", (_DETECT_WIDTH, _DETECT_HEIGHT), 0.6, 0.3, 5000)
        except Exception as exc:
            self.detection_error = str(exc)[:120]
            return None

    def _detect_yunet(self, frame: Any, detector: Any) -> dict[str, Any]:
        """A face sample WITH landmarks, in normalised frame coordinates.

        Positions are fractions of the frame rather than pixels, so a consumer
        does not have to know what resolution the camera happened to negotiate
        — and they keep working when it changes.
        """
        import cv2

        height, width = frame.shape[:2]
        scale = _DETECT_WIDTH / float(width)
        small = cv2.resize(frame, (_DETECT_WIDTH, max(1, int(height * scale))))
        sh, sw = small.shape[:2]
        detector.setInputSize((sw, sh))
        _count, faces = detector.detect(small)
        if faces is None or len(faces) == 0:
            self._smooth = None
            return _NO_FACE.copy()
        face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        fx, fy, fw, fh = (float(v) for v in face[:4])
        points = [(float(face[4 + i * 2]) / sw, float(face[5 + i * 2]) / sh)
                  for i in range(5)]
        self.detections += 1
        centre_x = (fx + fw / 2.0) / sw
        centre_y = (fy + fh / 2.0) / sh
        return {
            "present": True,
            # -1..1 with 0 centred, matching what the avatar rig already reads.
            "x": (centre_x - 0.5) * 2.0,
            "y": (centre_y - 0.5) * 2.0,
            "size": fw / sw,
            "confidence": float(face[14]),
            "landmarks": {
                "right_eye": points[0],
                "left_eye": points[1],
                "nose": points[2],
                "mouth_right": points[3],
                "mouth_left": points[4],
            },
        }

    def _load_cascade(self) -> Any:
        """The frontal-face cascade, or None when there is no usable model.

        OpenCV 5 ships ``cv2/data/`` with the XML cascades REMOVED, and
        ``CascadeClassifier`` does not raise on a path that is not there — it
        returns an empty classifier, and the error only surfaces later, from
        ``detectMultiScale``, on the first frame. That exception propagated out
        of the capture loop, whose ``finally`` sets ``running = False``, so a
        missing data file read as "camera unavailable" with a working camera
        plugged in.

        Returning None instead lets the feed run without detection.
        """
        import cv2

        candidates = []
        data = getattr(cv2, "data", None)
        base = getattr(data, "haarcascades", "") if data is not None else ""
        if base:
            candidates.append(Path(base) / "haarcascade_frontalface_default.xml")
        candidates.append(ASSET_CASCADE)
        for candidate in candidates:
            try:
                if not candidate.is_file():
                    continue
                cascade = cv2.CascadeClassifier(str(candidate))
                if not cascade.empty():
                    return cascade
            except Exception:
                continue
        self.detection_error = "no face-detection model installed"
        self._log("CAMERA: no face-detection cascade found (OpenCV 5 no longer "
                  "ships one), so ORION will watch without tracking faces. The "
                  "live feed, vision and gestures are unaffected.")
        return None

    def _log(self, message: str) -> None:
        """One line to the bus, once per distinct message. Never raises."""
        if message in self._logged:
            return
        self._logged.add(message)
        try:
            bus = getattr(self, "bus", None)
            if bus is not None:
                bus.log.emit(message)
            else:
                print(message)
        except Exception:
            pass

    def _detect(self, frame: Any, cascade: Any) -> dict[str, Any]:  # pragma: no cover
        import cv2
        height, width = frame.shape[:2]
        scale = _FRAME_WIDTH / float(width)
        small = cv2.resize(frame, (_FRAME_WIDTH, int(height * scale)))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(grey, scaleFactor=1.15, minNeighbors=4,
                                         minSize=(40, 40))
        if len(faces) == 0:
            self._smooth = None
            return {"present": False, "x": 0.0, "y": 0.0, "size": 0.0}
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        self.detections += 1
        sh, sw = grey.shape[:2]
        cx = ((x + w / 2.0) / sw) * 2.0 - 1.0        # [-1, 1], +1 = frame right
        cy = ((y + h / 2.0) / sh) * 2.0 - 1.0        # [-1, 1], +1 = frame bottom
        if self._smooth is None:
            self._smooth = (cx, cy)
        else:                                          # EMA α=0.4 (Ultron rule)
            self._smooth = (self._smooth[0] + (cx - self._smooth[0]) * 0.4,
                            self._smooth[1] + (cy - self._smooth[1]) * 0.4)
        return {"present": True, "x": round(self._smooth[0], 4),
                "y": round(self._smooth[1], 4), "size": round(w / float(sw), 4)}

    def _emit(self, sample: dict[str, Any]) -> None:
        self.last_sample = sample
        try:
            self.callback(sample)
        except Exception:
            pass
