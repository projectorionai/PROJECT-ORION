"""
Camera preview surfaces — "show me what you're looking at".

Two small, dismissible always-on-top windows that make ORION's camera use
visible instead of invisible:

    SnapshotPreviewWindow — when ORION captures a still of the user
                            (``vision.capture_live_frame`` /
                            ``analyse_camera``), the exact frame he sent to
                            the model is shown, so the user can see what was
                            actually taken rather than trusting a description
                            of it. Auto-dismisses; clicking keeps it open.

    LiveCameraWindow      — a small square live view of the webcam while face
                            tracking is running, so the user can see what
                            ORION sees and judge whether the tracking is
                            accurate.

Both are strictly CONSUMERS: they never open a camera handle of their own.
That matters on Windows, where a second handle on the same webcam fails with
MSMF error -1072873821 (the reason ``vision`` borrows the tracker's frame
rather than opening its own). The live view polls
``FaceTracker.latest_frame()`` — already a defensive copy — on a GUI-thread
timer, so the capture loop is untouched and Qt's "widgets only from the GUI
thread" rule is never bent.

Degrades to nothing useful but never crashes: without OpenCV/numpy the live
view shows a plain message, and a snapshot with unreadable bytes is ignored.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import C
from .style import rgba

def bgr_to_qimage(frame: Any) -> Optional[QImage]:
    """Convert an OpenCV BGR frame to a QImage, or None if unusable.

    The QImage is copied because the source numpy buffer is owned by the
    capture loop and may be overwritten (or freed) the moment this returns —
    a QImage referencing it would then paint garbage or crash.

    Qt can consume BGR directly. No colour conversion or OpenCV import is
    needed on the GUI thread, including for the camera's first frame.
    """
    if frame is None:
        return None
    try:
        if len(frame.shape) != 3 or frame.shape[2] != 3 or str(frame.dtype) != "uint8":
            return None
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return None
        if not frame.flags.c_contiguous:
            frame = frame.copy(order="C")
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
        return image.copy()
    except Exception:
        return None


class _PreviewWindow(QWidget):
    """Shared chrome: frameless, always-on-top, titled, with a close button."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setObjectName("cameraPreview")
        self.setStyleSheet(
            f"#cameraPreview {{ background: {rgba(C.PANEL_HI, 0.94)};"
            f" border: 1px solid {rgba(C.WHITE, 0.055)};"
            f" border-top: 1px solid {rgba(C.WHITE, 0.10)};"
            f" border-radius: 14px; }}"
            f"QLabel {{ color: {C.MUTED}; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(6)

        header = QHBoxLayout()
        self.header = header
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {C.ACCENT}; font-weight: 600;")
        header.addWidget(self.title_label, 1)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(18, 18)
        close_btn.setToolTip("Close this preview")
        close_btn.setAccessibleName("Close preview")
        close_btn.clicked.connect(self.hide)
        header.addWidget(close_btn, 0)
        outer.addLayout(header)

        self.image_label = QLabel("…")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(240, 180)
        outer.addWidget(self.image_label, 1)

        self._drag_offset = None

    # Frameless windows have no title bar to drag, so the whole surface drags.
    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        self._drag_offset = None


class SnapshotPreviewWindow(_PreviewWindow):
    """Shows the still ORION just captured, so the user sees the actual frame."""

    AUTO_HIDE_MS = 8000

    def __init__(self, bus: OrionBus, parent: Optional[QWidget] = None) -> None:
        super().__init__("SNAPSHOT", parent)
        self.bus = bus
        self.resize(320, 260)
        self._auto_hide = QTimer(self)
        self._auto_hide.setSingleShot(True)
        self._auto_hide.timeout.connect(self.hide)
        try:
            self.bus.camera_frame.connect(self._on_camera_frame)
        except Exception:
            pass

    def _on_camera_frame(self, payload: Any) -> None:
        if not isinstance(payload, dict) or payload.get("kind") != "snapshot":
            return
        data = payload.get("jpeg")
        if not isinstance(data, (bytes, bytearray)):
            return
        pixmap = QPixmap()
        if not pixmap.loadFromData(bytes(data), "JPEG"):
            return
        self.show_snapshot(pixmap, str(payload.get("note") or ""))

    def show_snapshot(self, pixmap: QPixmap, note: str = "") -> None:
        scaled = pixmap.scaled(
            300, 220,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.title_label.setText(f"SNAPSHOT — {note}" if note else "SNAPSHOT")
        self.show()
        self.raise_()
        self._auto_hide.start(self.AUTO_HIDE_MS)

    def enterEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        # Hovering means the user is actually looking at it — stop the clock.
        self._auto_hide.stop()
        super().enterEvent(event)


class LiveCameraWindow(_PreviewWindow):
    """A small square live view of what the face tracker is seeing.

    Polls the tracker rather than being pushed to: the tracker's capture loop
    runs on its own thread, and pushing frames from there into a widget would
    violate Qt's GUI-thread rule. Polling on a QTimer keeps every widget touch
    on the GUI thread and costs nothing when hidden.
    """

    POLL_MS = 100        # ~10 fps is plenty for "is it tracking me correctly?"
    VIEW_SIDE = 220
    #: How often the window measures a frame itself when perception is not
    #: supplying readings (colours / motion / numbers views).
    OWN_READING_S = 0.5
    #: Detections and poses older than this are not drawn.
    FRESH_S = 2.5

    def __init__(self, bus: OrionBus, tracker: Any | None = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__("LIVE CAMERA", parent)
        self.bus = bus
        self.tracker = tracker
        self.perception: Any = None
        self.overlay_mode = "normal"
        self._lab: Any = None
        self._own_reading: Any = None
        self._own_reading_at = 0.0
        self.resize(self.VIEW_SIDE + 20, self.VIEW_SIDE + 46)
        self.image_label.setMinimumSize(self.VIEW_SIDE, self.VIEW_SIDE)
        # The views the camera can be seen through (vision_lab.OVERLAY_MODES).
        self.mode_button = QPushButton("VIEW")
        self.mode_button.setFixedHeight(18)
        self.mode_button.setToolTip("Cycle the camera view: normal, edges, motion, "
                                    "colours, numbers, detect")
        self.mode_button.setAccessibleName("Camera view: normal")
        self.mode_button.clicked.connect(self._next_mode)
        self.header.insertWidget(1, self.mode_button, 0)
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._refresh)

    def attach_tracker(self, tracker: Any) -> None:
        self.tracker = tracker

    def attach_perception(self, perception: Any) -> None:
        """Readings, detections and poses come from the perception loop."""
        self.perception = perception

    def set_overlay(self, mode: str) -> None:
        from ..vision_lab import OVERLAY_MODES
        if mode not in OVERLAY_MODES:
            return
        self.overlay_mode = mode
        self.title_label.setText("LIVE CAMERA" if mode == "normal"
                                 else f"LIVE CAMERA · {mode.upper()}")
        self.mode_button.setAccessibleName(f"Camera view: {mode}")
        perception = self.perception
        if perception is not None:
            # The detect view asks for the naming layer; every other view is
            # served by the always-on instruments, so it asks for nothing.
            perception.overlay_needs = {"objects", "pose"} if mode == "detect" else set()
            if mode == "detect" and not perception.running:
                perception.start()

    def _next_mode(self) -> None:
        from ..vision_lab import OVERLAY_MODES
        index = OVERLAY_MODES.index(self.overlay_mode) if self.overlay_mode in OVERLAY_MODES else 0
        self.set_overlay(OVERLAY_MODES[(index + 1) % len(OVERLAY_MODES)])

    def start(self) -> None:
        if self.tracker is None:
            self.image_label.setText("Face tracking is not available.")
        self.show()
        self.raise_()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def toggle(self) -> None:
        self.stop() if self._timer.isActive() else self.start()

    def _refresh(self) -> None:
        if not self.isVisible():
            return
        tracker = self.tracker
        if tracker is None:
            return
        try:
            if not tracker.is_capturing():
                self.image_label.setText("Face tracking is off.")
                return
            frame = tracker.latest_frame()
        except Exception:
            return
        frame = self._overlay(frame)
        image = bgr_to_qimage(frame)
        if image is None:
            self.image_label.setText("Waiting for a frame…")
            return
        pixmap = QPixmap.fromImage(image).scaled(
            self.VIEW_SIDE, self.VIEW_SIDE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

    def _overlay(self, frame: Any) -> Any:
        """The frame as the chosen view, with fresh boxes and skeletons drawn.

        Costs 1-4 ms at this size (measured), on the GUI thread, only while
        the window is visible. Falls back to the plain frame on any fault."""
        if frame is None:
            return frame
        import time as _time
        perception = self.perception
        now = _time.monotonic()
        detections, poses, reading = (), (), None
        if perception is not None:
            reading = perception.last_reading if perception.running else None
            if perception.last_detections_at is not None and \
                    perception.clock() - perception.last_detections_at < self.FRESH_S:
                detections = perception.last_detections
            if perception.last_pose is not None and perception.last_pose_at is not None and \
                    perception.clock() - perception.last_pose_at < self.FRESH_S:
                poses = perception.last_pose.people
        if self.overlay_mode == "normal" and not detections and not poses:
            return frame
        if reading is None and self.overlay_mode in ("motion", "colours", "numbers"):
            if now - self._own_reading_at >= self.OWN_READING_S:
                from ..vision_lab import VisionLab
                if self._lab is None:
                    self._lab = VisionLab(fps_hint=1 / self.OWN_READING_S)
                self._own_reading = self._lab.read(frame)
                self._own_reading_at = now
            reading = self._own_reading
        from ..vision_lab import render
        return render(frame, self.overlay_mode, reading, detections, poses,
                      width=self.VIEW_SIDE * 2)

    def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        self._timer.stop()      # never poll a hidden window
        super().hideEvent(event)
