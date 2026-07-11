"""
PresenceMonitor — optional webcam presence detection ("welcome back, sir").

JARVIS greets Tony when he walks in.  This gives ORION the same faculty: when
you sit down at the machine he can greet you, and he can note when you step
away.  It uses OpenCV's built-in Haar-cascade face detector on single frames.

PRIVACY BY DESIGN:
    • OFF by default.  It only runs when ORION_PRESENCE=1 is set.
    • It never records or stores images — a frame is grabbed, a face is
      counted, the frame is discarded.  Only a boolean "someone is present"
      is ever kept.
    • The camera is opened only while the monitor runs and released on stop.

Detection is coarse on purpose (presence, not identity); it debounces so a
brief look-away doesn't trigger a goodbye, and it greets at most once per
absence so it never nags.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from .bus import OrionBus


class PresenceMonitor:
    ABSENCE_GREET_GAP = 90.0     # only greet again after this long away
    LEAVE_AFTER = 25.0           # seconds of no face before "stepped away"
    SAMPLE_INTERVAL = 2.5

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.enabled = os.getenv("ORION_PRESENCE", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._present = False
        self._last_seen = 0.0
        self._last_greet = 0.0
        self._cap: Any = None
        self._cascade: Any = None
        self._stop = asyncio.Event()

    # ── camera lifecycle ──────────────────────────────────────────────────────

    def _open(self) -> bool:
        try:
            import cv2
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._cap = cv2.VideoCapture(0)
            if not self._cap or not self._cap.isOpened():
                self.bus.log.emit("PRESENCE: no webcam available; presence detection disabled.")
                return False
            self.bus.log.emit("PRESENCE: webcam presence detection active (no images are stored).")
            return True
        except Exception as exc:
            self.bus.log.emit(f"PRESENCE: unavailable - {str(exc).splitlines()[0][:100]}")
            return False

    def _release(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def _face_present(self) -> bool:
        try:
            import cv2
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return self._present  # transient read failure — hold state
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(grey, scaleFactor=1.2,
                                                   minNeighbors=5, minSize=(80, 80))
            del frame, grey  # discard the image immediately
            return len(faces) > 0
        except Exception:
            return self._present

    # ── loop ──────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        if not self.enabled:
            return
        if self.telemetry is not None:
            self.telemetry.health.register("presence")
        if not await asyncio.to_thread(self._open):
            return
        try:
            while not self._stop.is_set():
                seen = await asyncio.to_thread(self._face_present)
                now = time.monotonic()
                if seen:
                    self._last_seen = now
                    if not self._present:
                        self._present = True
                        if now - self._last_greet > self.ABSENCE_GREET_GAP:
                            self._last_greet = now
                            self.bus.speak_request.emit("Welcome back, sir.")
                elif self._present and (now - self._last_seen) > self.LEAVE_AFTER:
                    self._present = False
                    self.bus.log.emit("PRESENCE: user appears to have stepped away.")
                if self.telemetry is not None:
                    self.telemetry.metrics.gauge("presence.present", 1.0 if self._present else 0.0)
                    self.telemetry.health.beat("presence", "OK",
                                               "present" if self._present else "away")
                await asyncio.sleep(self.SAMPLE_INTERVAL)
        except asyncio.CancelledError:
            pass
        finally:
            self._release()

    def stop(self) -> None:
        self._stop.set()
