"""A camera-first electronics workbench with honest, frame-bound findings.

The tracker remains the single owner of the physical camera.  This surface
borrows its frames, freezes the selected frame before requesting analysis,
and only draws evidence boxes on that retained capture.  Merely constructing
or showing the widget never activates the camera or imports OpenCV.
"""

from __future__ import annotations

import html
import math
import threading
import time
import uuid
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from ..board_detect import BoardScanner
from ..constants import C
from .camera_preview import bgr_to_qimage
from .labels import plural

#: Live local geometry is drawn as unlabelled silver hairlines; model findings
#: are drawn thick, green and numbered. Never the same colour — the user has to
#: be able to tell "ORION has found an edge" from "ORION has identified a part"
#: at a glance, without reading anything.
#:
#: Silver rather than the obvious scanner-cyan: ORION's visual identity retired
#: the electric-cyan counter-hue in favour of graphite and gunmetal lit by
#: white accent light (see constants.C), and a scanning overlay is exactly the
#: "rim light and data lines" that palette exists for.
LIVE_EDGE = QColor(C.ACCENT)
LIVE_REGION = QColor(C.SILVER)
LIVE_REGION.setAlpha(165)


def _plain(value: Any, limit: int = 3000) -> str:
    """Keep model output as bounded, escaped text rather than executable HTML."""
    return html.escape(str(value or "")[:limit])


class InspectionCanvas(QWidget):
    """Letterboxed camera image; coordinates always refer to the source image."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(280, 210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Electronics camera and captured inspection image")
        self.image: QImage | None = None
        self.captured = False
        self.observations: list[dict[str, Any]] = []
        #: Live local geometry from board_detect — never model findings.
        self.reading: Any = None
        #: (columns, rows) of the reference grid, or (0, 0) for none.
        self.grid: tuple[int, int] = (0, 0)
        #: Whether live board geometry is drawn (electronics mode only).
        self.show_geometry = True
        #: Shapes the last paint drew from `reading`. Lets a test observe what
        #: the paint path actually did, rather than infer it from pixel
        #: colours that the canvas's own labels also happen to use.
        self.geometry_drawn = 0
        #: QR codes / barcodes found in the view: {"points": [[x, y]...] in
        #: 0-1 image fractions, "label": str}. Drawn live and on captures.
        self.codes: list[dict[str, Any]] = []
        self.message = "Camera off"
        self.detail = ""

    def set_image(self, image: QImage, *, captured: bool = False) -> None:
        self.image = image
        self.captured = captured
        if not captured:
            self.observations = []
        self.update()

    def set_reading(self, reading: Any) -> None:
        """Attach this frame's local geometry; only drawn on the live view."""
        self.reading = reading
        self.update()

    def clear_image(self, message: str, detail: str = "") -> None:
        self.image = None
        self.observations = []
        self.reading = None
        self.message, self.detail = message, detail
        self.update()

    def image_rect(self) -> QRectF:
        area = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if self.image is None or self.image.isNull():
            return area
        scale = min(area.width() / self.image.width(), area.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height)

    def _region_count(self) -> int:
        regions = getattr(self.reading, "regions", None)
        return len(regions) if isinstance(regions, list) else 0

    def _paint_grid(self, painter: QPainter, target: QRectF) -> None:
        """The reference grid: the same cells ORION is told about, so "C4"
        means the same square to the user and to him."""
        cols, rows = self.grid
        if cols <= 0 or rows <= 0:
            return
        from ..electronics_inspection import _column_name

        painter.save()
        line = QColor(255, 255, 255, 70)
        painter.setPen(QPen(line, 1))
        for c in range(1, cols):
            x = target.left() + c * target.width() / cols
            painter.drawLine(QPointF(x, target.top()), QPointF(x, target.bottom()))
        for r in range(1, rows):
            y = target.top() + r * target.height() / rows
            painter.drawLine(QPointF(target.left(), y), QPointF(target.right(), y))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 150))
        for c in range(cols):
            x = target.left() + c * target.width() / cols
            painter.drawText(QRectF(x + 3, target.top() + 2, 30, 14),
                             Qt.AlignmentFlag.AlignLeft, _column_name(c))
        for r in range(rows):
            y = target.top() + r * target.height() / rows
            painter.drawText(QRectF(target.left() + 3, y + 14, 30, 14),
                             Qt.AlignmentFlag.AlignLeft, str(r + 1))
        painter.restore()

    def _paint_live_geometry(self, painter: QPainter, target: QRectF) -> None:
        """Draw this instant's local edges over the live feed.

        Hairlines and no labels, deliberately: these are contours, and dressing
        them up as identifications would be a lie the user could not see
        through. The numbered green boxes after a capture are the claims.
        """
        if not self.show_geometry:
            return
        reading = self.reading
        if reading is None or not getattr(reading, "available", False):
            return
        drawn = 0

        def place(box: Any) -> QRectF | None:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return None
            try:
                x, y, width, height = (float(v) for v in box)
            except (TypeError, ValueError):
                return None
            if (not all(math.isfinite(v) for v in (x, y, width, height))
                    or width <= 0 or height <= 0):
                return None
            return QRectF(target.x() + x * target.width(),
                          target.y() + y * target.height(),
                          width * target.width(), height * target.height())

        quad = getattr(reading, "quad", None)
        outline = getattr(reading, "board", None)
        if isinstance(quad, (list, tuple)) and len(quad) == 4:
            points = []
            for point in quad:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    points = []
                    break
                try:
                    px, py = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    points = []
                    break
                if not (math.isfinite(px) and math.isfinite(py)):
                    points = []
                    break
                points.append(QPointF(target.x() + px * target.width(),
                                      target.y() + py * target.height()))
            if len(points) == 4:
                painter.setPen(QPen(QColor(LIVE_EDGE), 2))
                painter.drawPolygon(QPolygonF(points))
                outline = None          # the quad is the more precise outline
                drawn += 1
        if outline is not None:
            rect = place(outline)
            if rect is not None:
                painter.setPen(QPen(QColor(LIVE_EDGE), 2, Qt.PenStyle.DashLine))
                painter.drawRect(rect)
                drawn += 1

        painter.setPen(QPen(QColor(LIVE_REGION), 1))
        for box in (getattr(reading, "regions", None) or [])[:40]:
            rect = place(box)
            # Sub-pixel rectangles are visual noise at preview scale.
            if rect is not None and rect.width() >= 3 and rect.height() >= 3:
                painter.drawRect(rect)
                drawn += 1
        self.geometry_drawn = drawn

    def _paint_codes(self, painter: QPainter, target: QRectF) -> None:
        for code in self.codes:
            points = code.get("points") or []
            if len(points) < 3:
                continue
            try:
                polygon = QPolygonF([QPointF(target.x() + float(x) * target.width(),
                                             target.y() + float(y) * target.height())
                                     for x, y in points])
            except (TypeError, ValueError):
                continue
            painter.setPen(QPen(QColor(C.GOOD), 3))
            painter.drawPolygon(polygon)
            label = str(code.get("label") or "")[:48]
            if label:
                anchor = polygon.boundingRect()
                tag = QRectF(anchor.left(), max(target.top(), anchor.top() - 24),
                             max(80.0, 8.0 * len(label) + 16), 22)
                painter.fillRect(tag, QColor("#15372b"))
                painter.setPen(QColor("#ffffff"))
                painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, label)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        self.geometry_drawn = 0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # The widget ground is chrome, so it takes the palette; the board
        # itself is a photographed object and keeps its own colours.
        painter.fillRect(self.rect(), QColor(C.BG))
        painter.setPen(QPen(QColor(C.ACCENT_DEEP), 1))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 10, 10)
        target = self.image_rect()
        if self.image is None:
            painter.setPen(QColor(C.ACCENT))
            font = painter.font()
            font.setPointSize(15)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(self.rect().adjusted(20, 0, -20, -28), Qt.AlignmentFlag.AlignCenter, self.message)
            font.setPointSize(10)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QColor(C.MUTED))
            painter.drawText(self.rect().adjusted(28, 50, -28, 0), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.detail)
            return
        painter.drawImage(target, self.image)
        painter.save()
        painter.setClipRect(target)
        self._paint_grid(painter, target)
        if not self.captured:
            guide = target.adjusted(target.width() * .07, target.height() * .09,
                                    -target.width() * .07, -target.height() * .09)
            painter.setPen(QPen(QColor(C.ACCENT), 2))
            for x, y, dx, dy in ((guide.left(), guide.top(), 1, 1),
                                 (guide.right(), guide.top(), -1, 1),
                                 (guide.left(), guide.bottom(), 1, -1),
                                 (guide.right(), guide.bottom(), -1, -1)):
                painter.drawLine(QPointF(x, y), QPointF(x + dx * 22, y))
                painter.drawLine(QPointF(x, y), QPointF(x, y + dy * 22))
            self._paint_live_geometry(painter, target)
        else:
            # Most vision models answer an inspection in prose and localise
            # nothing. Without this the captured frame goes blank the moment
            # the guides disappear, and stays blank through a perfectly good
            # text report — so the local geometry stays visible until there
            # are real findings to replace it, and never competes with them.
            if not any(item.get("bbox") for item in self.observations):
                self._paint_live_geometry(painter, target)
            for number, item in enumerate(self.observations, 1):
                box = item.get("bbox")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    x, y, width, height = map(float, box)
                except (TypeError, ValueError):
                    continue
                if (not all(math.isfinite(v) for v in (x, y, width, height))
                        or x < 0 or y < 0 or width <= 0 or height <= 0
                        or x + width > 1.001 or y + height > 1.001):
                    continue
                rect = QRectF(target.x() + x * target.width(), target.y() + y * target.height(),
                              width * target.width(), height * target.height())
                painter.setPen(QPen(QColor(C.GOOD), 2))
                painter.drawRect(rect)
                cells = item.get("cells") or []
                text = str(number)
                if cells:
                    from ..electronics_inspection import cell_span
                    text = f"{number} · {cell_span(cells)}"
                tag_width = 26 if not cells else 26 + 8 * len(text)
                tag = QRectF(rect.left(), max(target.top(), rect.top() - 23), tag_width, 22)
                painter.fillRect(tag, QColor("#15372b"))
                painter.setPen(QColor("#ffffff"))
                painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, text)
        self._paint_codes(painter, target)
        painter.restore()
        tag = "CAPTURED FRAME" if self.captured else "LIVE CAMERA"
        if not self.captured and self._region_count():
            painter.setPen(QColor(LIVE_EDGE))
            painter.drawText(
                QRectF(target.left() + 12, target.bottom() - 34, target.width() - 24, 24),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"TRACKING  ·  {plural(self._region_count(), 'region')}")
        painter.fillRect(QRectF(target.left() + 12, target.top() + 12, 145, 27), QColor(6, 10, 14, 220))
        painter.setPen(QColor(C.ACCENT))
        painter.drawText(QRectF(target.left() + 22, target.top() + 12, 135, 27), Qt.AlignmentFlag.AlignVCenter, tag)


class ElectronicsWorkbench(QWidget):
    """Opt-in camera inspection UI connected through the OrionBus."""

    scan_requested = pyqtSignal(object)
    _camera_released = pyqtSignal(object)
    #: Worker-thread results, delivered on the GUI thread.
    _codes_found = pyqtSignal(object, object)
    _clip_done = pyqtSignal(object, str)
    _scan_read = pyqtSignal(object, str)
    POLL_MS = 100
    STALE_SECONDS = 3.0

    def __init__(self, bus: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.tracker: Any = None
        self._owns_camera = False
        self._camera_enabled = False
        self._stopping = False
        self._frame: Any = None
        self._captured_image: QImage | None = None
        self._observations: list[dict[str, Any]] = []
        self._request_id: str | None = None
        #: The exact BGR frame behind the retained capture, kept so a follow-up
        #: question is answered about the same view rather than a new one.
        self._captured_frame: Any = None
        self._captured_reading: Any = None
        self._busy = False
        self._show_capture = False
        self._cameras: list[Any] = []
        self._restart_after_release = False
        self._stopping_since = 0.0
        self._last_counter: Any = None
        self._last_frame_time = 0.0
        self._last_quality_time = 0.0
        # Local geometry runs on its own thread so a detection pass can never
        # be what makes the preview stutter.
        self._scanner = BoardScanner()
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._refresh)
        self._camera_released.connect(self._on_camera_released)
        # The everyday camera tools (camera_tools.py).
        from ..camera_tools import ClipRecorder
        self._recorder = ClipRecorder()
        self._record_left = 0
        self._record_timer = QTimer(self)
        self._record_timer.setInterval(1000)
        self._record_timer.timeout.connect(self._record_tick)
        self._low_light = False
        self._codes_busy = False
        self._last_codes_scan = 0.0
        self._ocr: Any = None
        self._codes_found.connect(self._on_codes)
        self._clip_done.connect(self._on_clip_done)
        self._scan_read.connect(self._on_scan_read)
        self._build_ui()
        request_signal = getattr(bus, "electronics_scan_requested", None)
        if request_signal is not None:
            self.scan_requested.connect(request_signal.emit)
        for name, slot in (("electronics_scan_started", self.on_scan_started),
                           ("electronics_scan_result", self.on_scan_result),
                           ("electronics_scan_error", self.on_scan_error)):
            signal = getattr(bus, name, None)
            if signal is not None:
                signal.connect(slot)

    def _build_ui(self) -> None:
        self.setObjectName("electronicsWorkbench")
        self.setStyleSheet(f"""
            #electronicsWorkbench {{ background: {C.BG}; }}
            #workbenchTitle {{ color: {C.WHITE}; font-size: 23px; font-weight: 650; }}
            #workbenchMuted {{ color: {C.MUTED}; font-size: 12px; }}
            #inspectionPanel {{ background: {C.PANEL}; border: 1px solid {C.BORDER}; border-radius: 10px; }}
            #inspectionPanel QLabel {{ border: none; background: transparent; }}
            #workbenchState {{ color: {C.ACCENT}; background: {C.PANEL_HI}; border-radius: 7px; padding: 7px 11px; }}
            #workbenchReport {{ background: transparent; border: none; color: {C.WHITE}; font-size: 12px; }}
            #electronicsWorkbench QPushButton {{ min-height: 28px; border: 1px solid {C.ACCENT_DEEP}; border-radius: 7px; padding: 4px 12px; background: {C.PANEL_HI}; color: {C.ACCENT}; }}
            #electronicsWorkbench QPushButton:hover {{ border-color: {C.ACCENT_DIM}; }}
            #electronicsWorkbench QPushButton:disabled {{ color: {C.ACCENT_DEEP}; border-color: {C.BORDER}; }}
            #electronicsWorkbench QPushButton#captureInspection {{ background: {C.PRI}; color: white; border-color: {C.PRI}; font-weight: 650; }}
            #electronicsWorkbench QPushButton#captureInspection:disabled {{ background: {C.PANEL_HI}; color: {C.ACCENT_DIM}; border-color: {C.BORDER}; }}
            #electronicsWorkbench QPushButton#askCapture {{ min-height: 38px; padding: 4px 16px; }}
            #electronicsWorkbench QPushButton#cameraTool {{ min-height: 24px; padding: 2px 9px; }}
            #electronicsWorkbench QCheckBox {{ color: {C.ACCENT}; spacing: 6px; }}
            #electronicsWorkbench QLineEdit {{ background: {C.PANEL}; border: 1px solid {C.BORDER}; border-radius: 7px; padding: 10px; color: {C.WHITE}; }}
            #electronicsWorkbench QSplitter::handle {{ background: transparent; }}
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(14)
        heading = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(4)
        title = QLabel("Vision Lab")
        title.setObjectName("workbenchTitle")
        titles.addWidget(title)
        heading.addLayout(titles, 1)
        self.state_label = QLabel("CAMERA OFF")
        self.state_label.setObjectName("workbenchState")
        heading.addWidget(self.state_label, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(heading)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(16)
        viewer = QWidget()
        video_layout = QVBoxLayout(viewer)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(10)
        self.canvas = InspectionCanvas()
        video_layout.addWidget(self.canvas, 1)
        self.quality_label = QLabel("")
        self.quality_label.setObjectName("workbenchMuted")
        self.quality_label.setWordWrap(True)
        video_layout.addWidget(self.quality_label)

        # ── what to look with, how, and on what grid ─────────────────────────
        options = QHBoxLayout()
        options.setSpacing(8)
        self.camera_box = QComboBox()
        self.camera_box.setAccessibleName("Camera")
        self.camera_box.setToolTip("Which camera to use — remembered by name")
        self.camera_box.activated.connect(self._camera_chosen)
        options.addWidget(self.camera_box, 2)
        self.rescan_button = QPushButton("↻")
        self.rescan_button.setToolTip("Look for cameras again")
        self.rescan_button.setAccessibleName("Rescan cameras")
        self.rescan_button.setFixedWidth(36)
        # A 36 px button under the page's 12 px side padding had 12 px left for
        # the glyph, and drew a sliver of it (":") instead of the arrow.
        self.rescan_button.setStyleSheet("padding: 0px;")
        self.rescan_button.clicked.connect(self.refresh_cameras)
        options.addWidget(self.rescan_button)
        self.mode_box = QComboBox()
        self.mode_box.setAccessibleName("Scan mode")
        for label, value in (("Identify Anything", "anything"),
                             ("Read Text & Documents", "text"),
                             ("Count Items", "count"),
                             ("Electronics / PCB", "electronics"),
                             ("Codes (QR & Barcodes)", "codes")):
            self.mode_box.addItem(label, value)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        options.addWidget(self.mode_box, 1)
        self.grid_box = QComboBox()
        self.grid_box.setAccessibleName("Reference grid")
        self.grid_box.setToolTip("A labelled grid over the view; findings are pinned to its cells")
        for label, value in (("Grid 8 × 6", (8, 6)), ("Grid 12 × 9", (12, 9)),
                             ("Grid 16 × 12", (16, 12)), ("Grid 4 × 3", (4, 3)),
                             ("No Grid", (0, 0))):
            self.grid_box.addItem(label, value)
        self.grid_box.currentIndexChanged.connect(self._grid_changed)
        options.addWidget(self.grid_box, 1)
        # Compact minimums so a narrow deck can still stack the lab rather
        # than being forced wide by the longest camera name.
        for box in (self.camera_box, self.mode_box, self.grid_box):
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(6)
        video_layout.addLayout(options)
        self._options_row = options            # the photo self-timer joins it below

        controls = QHBoxLayout()
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self._toggle_camera)
        self.capture_button = QPushButton("Capture + Inspect")
        self.capture_button.setObjectName("captureInspection")
        self.capture_button.setEnabled(False)
        self.capture_button.setToolTip("Inspect the exact camera frame shown. AI analysis uses the configured vision provider.")
        self.capture_button.clicked.connect(self.capture)
        self.live_button = QPushButton("Live View")
        self.live_button.setVisible(False)
        self.live_button.clicked.connect(self.show_live)
        controls.addWidget(self.camera_button)
        controls.addWidget(self.live_button)
        controls.addStretch(1)
        controls.addWidget(self.capture_button)
        video_layout.addLayout(controls)

        # ── the everyday camera: photo, clip, document, codes, low light ─────
        # All local (camera_tools.py), all on the frame already on screen.
        tools = QHBoxLayout()
        tools.setSpacing(8)
        self.photo_button = QPushButton("Photo")
        self.photo_button.setToolTip("Save this view to Pictures\\ORION")
        self.photo_button.clicked.connect(self.take_photo)
        # Self-timer: off -> 3 s -> 10 s, counted down on the Photo button.
        self._photo_delay = 0
        self._countdown_left = 0
        self._countdown = QTimer(self)
        self._countdown.setInterval(1000)
        self._countdown.timeout.connect(self._countdown_tick)
        self.timer_button = QPushButton("Timer Off")
        self.timer_button.setAccessibleName("Photo self-timer")
        self.timer_button.setToolTip("Self-timer for the photo: off, 3 or 10 seconds")
        self.timer_button.clicked.connect(self._cycle_photo_timer)
        self.record_button = QPushButton("Record")
        self.record_button.setToolTip("Record a short clip to Videos\\ORION (press again to stop)")
        self.record_button.clicked.connect(self.toggle_recording)
        self.document_button = QPushButton("Document")
        self.document_button.setToolTip("Flatten the page in view, save it and read its text")
        self.document_button.clicked.connect(self.scan_document)
        self.codes_button = QPushButton("Codes")
        self.codes_button.setToolTip("Read QR codes and barcodes in view")
        self.codes_button.clicked.connect(self.read_codes)
        self.low_light_box = QCheckBox("Low Light")
        self.low_light_box.setToolTip("Brighten a dim room on screen and in captures")
        self.low_light_box.toggled.connect(self._low_light_toggled)
        for widget in (self.photo_button, self.record_button,
                       self.document_button, self.codes_button):
            widget.setObjectName("cameraTool")
            widget.setEnabled(False)
            tools.addWidget(widget)
        tools.addStretch(1)
        tools.addWidget(self.low_light_box)
        video_layout.addLayout(tools)
        # In the options row, not beside Photo: the tools row sets the lab's
        # minimum width, and one more button there stopped it stacking in a
        # narrow deck (resizeEvent, < 860 px). The options row has room.
        self.timer_button.setObjectName("cameraTool")
        self.timer_button.setEnabled(False)
        self._options_row.addWidget(self.timer_button)
        self.splitter.addWidget(viewer)

        report_panel = QFrame()
        report_panel.setObjectName("inspectionPanel")
        report_panel.setMinimumWidth(230)
        report_layout = QVBoxLayout(report_panel)
        report_layout.setContentsMargins(17, 17, 17, 12)
        report_layout.setSpacing(12)
        self.report_heading = QLabel("Inspection Notes")
        self.report_heading.setStyleSheet(f"color: {C.ACCENT}; font-size: 11px; font-weight: 650;")
        report_layout.addWidget(self.report_heading)
        self.report = QTextBrowser()
        self.report.setObjectName("workbenchReport")
        self.report.setOpenLinks(False)
        self.report.setOpenExternalLinks(False)
        # Actions in the notes ("open the photo", "copy the text") are
        # numbered anchors; the payloads stay here, never in the HTML.
        self._report_actions: list[tuple[str, Any]] = []
        self.report.anchorClicked.connect(self._report_action)
        self.report.setHtml(self._intro_html())
        report_layout.addWidget(self.report, 1)
        self.splitter.addWidget(report_panel)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([700, 290])
        outer.addWidget(self.splitter, 1)
        ask_row = QHBoxLayout()
        ask_row.setSpacing(10)
        self.focus_input = QLineEdit()
        self.focus_input.setMaxLength(1000)
        self.focus_input.setPlaceholderText("Ask about the capture")
        self.focus_input.setAccessibleName("Optional inspection focus")
        self.focus_input.returnPressed.connect(self._focus_entered)
        ask_row.addWidget(self.focus_input, 1)
        # Asking a second question used to mean capturing again — on a board
        # that had been picked up and put down in between, so the answer
        # described a different view from the one on screen. This re-asks
        # about the frame the user is actually looking at.
        self.ask_button = QPushButton("Ask About This Capture")
        self.ask_button.setObjectName("askCapture")
        self.ask_button.setEnabled(False)
        self.ask_button.setToolTip("Ask again about the same capture")
        self.ask_button.clicked.connect(self.ask_again)
        ask_row.addWidget(self.ask_button)
        outer.addLayout(ask_row)

        self._grid_changed()
        self._mode_changed()
        self.refresh_cameras()

    @staticmethod
    def _intro_html() -> str:
        return ""

    # ── camera, mode and grid choices ────────────────────────────────────────

    @staticmethod
    def _read_cameras() -> tuple[list[Any], str]:
        from .. import camera_devices as cd
        try:
            return cd.all_cameras(), cd.preferred_name()
        except Exception:
            return [], ""

    def refresh_cameras(self) -> None:
        """List the cameras Windows can see, by name, remembered choice first.

        On a worker thread when the loop is running: listing them (and the
        OpenCV import under it) held the GUI thread for 400-1360 ms while
        ORION started, because this page is built at boot (28 Oct 2026)."""
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._show_cameras(*self._read_cameras())
            return
        from .. import background

        async def read() -> None:
            cameras, wanted = await asyncio.to_thread(self._read_cameras)
            try:
                self._show_cameras(cameras, wanted)
            except RuntimeError:                  # the page was closed meanwhile
                pass
        background.spawn(read(), name="orion-camera-list")

    def _show_cameras(self, cameras: list[Any], wanted: str) -> None:
        self._cameras = cameras
        self.camera_box.blockSignals(True)
        self.camera_box.clear()
        if not cameras:
            self.camera_box.addItem("Default Camera", None)
        for camera in cameras:
            self.camera_box.addItem(camera.label(), camera.name)
        if wanted:
            index = self.camera_box.findData(wanted)
            if index >= 0:
                self.camera_box.setCurrentIndex(index)
        self.camera_box.blockSignals(False)
        count = len(cameras)
        self.rescan_button.setToolTip(f"Look for cameras again ({count} found)")

    def _camera_chosen(self, _index: int = 0) -> None:
        name = self.camera_box.currentData()
        if not name:
            return
        from .. import camera_devices as cd

        cd.set_device(str(name))
        follow = getattr(self.tracker, "follow_default_camera", None)
        if callable(follow):
            follow()
        if self._camera_enabled:
            # Switch now: stop, then start once the old handle is released.
            self._restart_after_release = True
            self.stop_camera()

    def selected_mode(self) -> str:
        return str(self.mode_box.currentData() or "anything")

    def selected_grid(self) -> tuple[int, int]:
        value = self.grid_box.currentData()
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return (0, 0)

    def _grid_changed(self, _index: int = 0) -> None:
        self.canvas.grid = self.selected_grid()
        self.canvas.update()

    def _mode_changed(self, _index: int = 0) -> None:
        mode = self.selected_mode()
        self.quality_label.setText({
            "electronics": "Board flat, markings in focus",
            "text": "Page flat and square, even light",
            "count": "Spread the items out",
        }.get(mode, ""))
        self.canvas.update()

    def attach_tracker(self, tracker: Any) -> None:
        if tracker is self.tracker:
            return
        if self._camera_enabled:
            self.stop_camera()
        self.tracker = tracker

    def _toggle_camera(self) -> None:
        self.stop_camera() if self._camera_enabled else self.start_camera()

    def start_camera(self) -> None:
        if self._stopping and time.monotonic() - self._stopping_since > 4.0:
            # The release thread is stuck in the camera driver. Waiting on it
            # forever is what left the lab unable to start until a restart;
            # the tracker itself now abandons a stuck capture and opens anew.
            self._stopping = False
            self.camera_button.setEnabled(True)
        if self._stopping or self._camera_enabled:
            return
        if self.tracker is None:
            self.state_label.setText("CAMERA UNAVAILABLE")
            self.canvas.clear_image("Camera unavailable", "ORION is still starting")
            return
        try:
            already_running = self.tracker.is_capturing()
            if not already_running:
                result = self.tracker.start()
                if not getattr(result, "ok", True):
                    raise RuntimeError(getattr(result, "text", "Unable to start the camera."))
            self._owns_camera = not already_running
        except Exception as exc:
            self.state_label.setText("CAMERA UNAVAILABLE")
            self.canvas.clear_image("Could not start the camera", str(exc)[:220])
            return
        self._camera_enabled = True
        self._scanner.start()
        self._last_counter = getattr(self.tracker, "frames", None)
        self._last_frame_time = time.monotonic()
        self._frame = None
        self._show_capture = False
        self.canvas.clear_image("Opening camera…")
        self.state_label.setText("CONNECTING")
        self.camera_button.setText("Stop Camera")
        if self.isVisible():
            self._timer.start()
        self._refresh()

    def stop_camera(self) -> None:
        self._camera_enabled = False
        self._timer.stop()
        self._scanner.stop(wait=False)
        if self._recorder.recording:
            self._recorder.stop()            # the clip ends with the camera
        # A retained capture can still be saved, scanned or read; nothing
        # can be recorded without a live feed.
        for button in (self.photo_button, self.timer_button, self.document_button, self.codes_button):
            button.setEnabled(self._show_capture)
        self.record_button.setEnabled(False)
        if self._countdown.isActive():          # no camera, no delayed photo
            self._countdown.stop()
            self.photo_button.setText("Photo")
        if not self._show_capture:
            # A retained capture keeps the geometry measured from it; only the
            # live view's reading goes stale when the camera stops.
            self.canvas.reading = None
        self._frame = None
        self.capture_button.setEnabled(False)
        self.camera_button.setText("Start Camera")
        self.state_label.setText("CAMERA OFF")
        if not self._show_capture:
            self.canvas.clear_image("Camera paused")
        if self._owns_camera and self.tracker is not None:
            self._owns_camera = False
            self._stopping = True
            self._stopping_since = time.monotonic()
            self.camera_button.setEnabled(False)
            # Never disabled for longer than the tracker's own stuck-driver
            # grace: the button comes back even if the release never returns.
            QTimer.singleShot(4500, self._unlatch_stop)
            tracker = self.tracker

            def release() -> None:
                try:
                    tracker.stop()
                finally:
                    try:
                        self._camera_released.emit(tracker)
                    except RuntimeError:  # widget already destroyed during app shutdown
                        pass

            threading.Thread(target=release, name="orion-workbench-camera-stop", daemon=True).start()

    def _on_camera_released(self, tracker: Any) -> None:
        self._stopping = False
        self.camera_button.setEnabled(True)
        if self._restart_after_release:
            self._restart_after_release = False
            self.start_camera()

    def _unlatch_stop(self) -> None:
        if self._stopping and time.monotonic() - self._stopping_since >= 4.0:
            self._stopping = False
            self.camera_button.setEnabled(True)
            if self._restart_after_release:
                self._restart_after_release = False
                self.start_camera()

    def _refresh(self) -> None:
        if not self._camera_enabled or self.tracker is None:
            return
        try:
            if not self.tracker.is_capturing():
                error = str(getattr(self.tracker, "error", "") or "Camera disconnected. Try starting it again.")
                self.stop_camera()
                self.state_label.setText("CAMERA UNAVAILABLE")
                self.quality_label.setText(error)
                return
            frame = self.tracker.latest_frame()
            if frame is not None and self._low_light:
                from ..camera_tools import enhance_low_light
                frame = enhance_low_light(frame)
            counter = getattr(self.tracker, "frames", None)
            now = time.monotonic()
            if frame is not None and (counter is None or counter != self._last_counter):
                self._last_frame_time = now
                self._last_counter = counter
            if now - self._last_frame_time > self.STALE_SECONDS:
                self._frame = None
                self.capture_button.setEnabled(False)
                self.state_label.setText("WAITING FOR CAMERA")
                self.quality_label.setText("No picture from the camera")
                return
            reading = None
            if frame is not None and not self._show_capture:
                self._scanner.submit(frame)
                snapshot = self._scanner.latest_snapshot()
                if snapshot is not None and now - snapshot[2] < 0.75:
                    frame, reading, _captured_at = snapshot
            image = bgr_to_qimage(frame)
        except Exception as exc:
            self._frame = None
            self.capture_button.setEnabled(False)
            self.quality_label.setText(f"Camera frame unavailable: {str(exc)[:160]}")
            return
        if image is None:
            self._frame = None
            self.capture_button.setEnabled(False)
            return
        self._frame = frame
        self.capture_button.setEnabled(not self._busy and not self._show_capture)
        for button in (self.photo_button, self.timer_button, self.document_button, self.codes_button):
            button.setEnabled(True)
        self.record_button.setEnabled(True)
        if not self._show_capture:
            if self.selected_mode() == "codes":
                self._scan_codes_live(frame, now)
            elif self.canvas.codes:
                self.canvas.codes = []
        if not self._show_capture:
            # Geometry and pixels come from one completed worker snapshot.
            # If processing stalls, show the fresh feed without stale boxes.
            self.canvas.reading = reading
            self.canvas.set_image(image)
            self.state_label.setText("TRACKING" if getattr(reading, "board", None) else "LIVE")
            if now - self._last_quality_time >= .8:
                self._last_quality_time = now
                if reading is not None and getattr(reading, "available", False):
                    self.quality_label.setText(
                        f"{image.width()} x {image.height()}  ·  {reading.guidance()}")
                else:
                    self._update_lighting(image)

    # ── everyday camera tools ─────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        log = getattr(self.bus, "log", None)
        if log is not None:
            try:
                log.emit(message)
            except Exception:
                pass

    def _view_frame(self) -> Any:
        """The frame on screen: the retained capture, or the live frame."""
        frame = self._captured_frame if self._show_capture else self._frame
        if frame is None:
            return None
        try:
            return frame.copy()
        except Exception:
            return None

    def _note(self, heading: str, html_body: str, actions: list[tuple[str, str, Any]] = ()) -> None:
        """Show a result in the notes, with numbered action links."""
        self._report_actions = []
        links = []
        for label, kind, payload in actions:
            self._report_actions.append((kind, payload))
            links.append(f'<a href="orion-action:{len(self._report_actions) - 1}">{_plain(label)}</a>')
        self.report_heading.setText(heading)
        self.report.setHtml(html_body + (f"<p>{'  ·  '.join(links)}</p>" if links else ""))

    def _report_action(self, url: Any) -> None:
        text = url.toString() if hasattr(url, "toString") else str(url)
        if not text.startswith("orion-action:"):
            return
        try:
            kind, payload = self._report_actions[int(text.split(":", 1)[1])]
        except (ValueError, IndexError):
            return
        from .. import camera_tools
        if kind == "open_file":
            camera_tools.open_file(payload)
        elif kind == "open_folder":
            camera_tools.open_file(payload)
        elif kind == "copy":
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(str(payload))
                self.quality_label.setText("Copied to the clipboard.")
        elif kind == "open_link":
            from ..browser import open_url
            open_url(str(payload))

    def _cycle_photo_timer(self) -> None:
        self._photo_delay = {0: 3, 3: 10}.get(self._photo_delay, 0)
        self.timer_button.setText(f"Timer {self._photo_delay} s" if self._photo_delay else "Timer Off")
        self.timer_button.setAccessibleName(f"Photo self-timer: {self._photo_delay} seconds"
                                            if self._photo_delay else "Photo self-timer: off")

    def take_photo(self) -> None:
        """With the self-timer on, count down on the button first."""
        if self._countdown.isActive():
            return
        if self._photo_delay:
            self._countdown_left = self._photo_delay
            self.photo_button.setText(f"{self._countdown_left}…")
            self._countdown.start()
            return
        self._take_photo_now()

    def _countdown_tick(self) -> None:
        self._countdown_left -= 1
        if self._countdown_left > 0:
            self.photo_button.setText(f"{self._countdown_left}…")
            return
        self._countdown.stop()
        self.photo_button.setText("Photo")
        self._take_photo_now()

    def _take_photo_now(self) -> None:
        from .. import camera_tools
        frame = self._view_frame()
        if frame is None:
            self.quality_label.setText("No camera frame to save yet.")
            return
        try:
            path = camera_tools.save_image(frame, "photo")
        except Exception as exc:
            path = None
            self.quality_label.setText(f"The photo could not be saved: {str(exc)[:120]}")
        if path is None:
            return
        self.quality_label.setText(f"Photo saved to {path.parent}")
        self._note("Photo Saved", f"<p>{_plain(path.name)}</p><p>{_plain(str(path.parent))}</p>",
                   [("Open the photo", "open_file", path), ("Open the folder", "open_folder", path.parent)])
        self._log(f"VISION LAB: photo saved - {path}")

    def toggle_recording(self, seconds: int = 10) -> None:
        if self._recorder.recording:
            self._recorder.stop()
            return
        tracker = self.tracker
        if tracker is None or not self._camera_enabled:
            self.quality_label.setText("Start the camera to record a clip.")
            return
        low_light = self._low_light

        def source() -> Any:
            frame = tracker.latest_frame()
            if frame is not None and low_light:
                from ..camera_tools import enhance_low_light
                frame = enhance_low_light(frame)
            return frame

        if self._recorder.start(source, seconds=seconds, fps=15,
                                on_done=lambda path, message: self._clip_done.emit(path, message)):
            self._record_left = int(seconds)
            self.record_button.setText(f"Stop ({self._record_left})")
            self.state_label.setText("RECORDING")
            self._record_timer.start()

    def _record_tick(self) -> None:
        self._record_left = max(0, self._record_left - 1)
        if self._recorder.recording:
            self.record_button.setText(f"Stop ({self._record_left})")
        else:
            self._record_timer.stop()

    def _on_clip_done(self, path: Any, message: str) -> None:
        self._record_timer.stop()
        self.record_button.setText("Record")
        if self._camera_enabled:
            self.state_label.setText("LIVE")
        if path is None:
            self.quality_label.setText(message or "The clip could not be recorded.")
            return
        self.quality_label.setText(f"{message} Saved to {path.parent}")
        self._note("Clip Saved", f"<p>{_plain(path.name)}</p><p>{_plain(message)}</p>",
                   [("Play the clip", "open_file", path), ("Open the folder", "open_folder", path.parent)])
        self._log(f"VISION LAB: clip saved - {path}")

    def scan_document(self) -> None:
        """Flatten the page in view, show it, save it, then read its text."""
        from .. import camera_tools
        frame = self._view_frame()
        if frame is None:
            self.quality_label.setText("No camera frame to scan yet.")
            return
        scan, found = camera_tools.scan_document(frame)
        if scan is None:
            self.quality_label.setText("The page could not be scanned.")
            return
        try:
            import cv2
            shown = cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR) if scan.ndim == 2 else scan
        except Exception:
            shown = scan
        image = bgr_to_qimage(shown)
        if image is not None:
            self._show_capture = True
            self._captured_image = image
            self._captured_frame = shown
            self.canvas.observations = []
            self.canvas.codes = []
            self.canvas.set_image(image, captured=True)
            self.live_button.setText("Live View")
            self.live_button.setVisible(True)
        path = camera_tools.save_image(scan, "scan")
        where = f"Saved to {path.parent}" if path else "Not saved."
        outline = ("Page edges found and straightened." if found
                   else "No page outline found - the whole view was cleaned instead.")
        self.quality_label.setText(f"{outline} {where}")
        self._note("Document Scanned",
                   f"<p>{_plain(outline)}</p><p>Reading the text…</p>",
                   [("Open the scan", "open_file", path)] if path else [])
        self._pending_scan_path = path

        def read() -> None:
            text = ""
            try:
                if self._ocr is None:
                    from ..ocr_engine import OcrEngine
                    self._ocr = OcrEngine(self.bus)
                from PIL import Image
                text = self._ocr.image_to_text(Image.fromarray(scan))
            except Exception as exc:
                text = f"\u0000{exc}"
            self._scan_read.emit(path, text)

        threading.Thread(target=read, name="orion-document-ocr", daemon=True).start()

    def _on_scan_read(self, path: Any, text: str) -> None:
        if text.startswith("\u0000"):
            body = f"<p>The text could not be read: {_plain(text[1:], 200)}</p>"
            actions = [("Open the scan", "open_file", path)] if path else []
        elif not text.strip():
            body = "<p>No text found on the page.</p>"
            actions = [("Open the scan", "open_file", path)] if path else []
        else:
            words = len(text.split())
            body = (f"<p>{words} words read.</p><pre style='white-space:pre-wrap'>"
                    f"{_plain(text, 6000)}</pre>")
            actions = [("Copy the text", "copy", text)]
            if path:
                actions.append(("Open the scan", "open_file", path))
        self._note("Document Scanned", body, actions)

    def read_codes(self) -> None:
        from .. import camera_tools
        frame = self._view_frame()
        if frame is None:
            self.quality_label.setText("No camera frame to read yet.")
            return
        codes = camera_tools.read_codes(frame)
        self._on_codes(codes, (frame.shape[1], frame.shape[0]))
        if not codes:
            self._note("Codes", "<p>No QR code or barcode is readable in this view. Hold it "
                                "flat, fill more of the frame and keep it in focus.</p>")

    def _scan_codes_live(self, frame: Any, now: float) -> None:
        if self._codes_busy or now - self._last_codes_scan < 0.6 or frame is None:
            return
        self._codes_busy = True
        self._last_codes_scan = now
        try:
            copy = frame.copy()
        except Exception:
            self._codes_busy = False
            return

        def work() -> None:
            from .. import camera_tools
            try:
                codes = camera_tools.read_codes(copy)
            except Exception:
                codes = []
            self._codes_found.emit(codes, (copy.shape[1], copy.shape[0]))

        threading.Thread(target=work, name="orion-live-codes", daemon=True).start()

    def _on_codes(self, codes: Any, size: Any) -> None:
        self._codes_busy = False
        width, height = size if size else (1, 1)
        drawn = []
        for code in codes or []:
            points = [[x / max(1, width), y / max(1, height)] for x, y in code.points]
            drawn.append({"points": points, "label": code.summary[:40] or code.kind})
        self.canvas.codes = drawn
        self.canvas.update()
        if not codes:
            return
        rows = []
        actions: list[tuple[str, str, Any]] = []
        for number, code in enumerate(codes, 1):
            rows.append(f"<p><b>{number}. {_plain(code.summary, 200)}</b><br>"
                        f"<span style='color:#9a9aa3'>{_plain(code.kind)}</span><br>"
                        f"{_plain(code.text if code.meaning != 'wifi' else 'Wi-Fi details hidden', 1000)}</p>")
            if code.meaning == "link":
                actions.append((f"Open link {number}", "open_link", code.text))
            if code.meaning != "wifi":
                actions.append((f"Copy {number}", "copy", code.text))
        self._note(f"CODES · {len(codes)}", "".join(rows), actions)

    def _low_light_toggled(self, on: bool) -> None:
        self._low_light = bool(on)
        self.quality_label.setText("Low-light boost on" if on else "Low-light boost off")

    def _update_lighting(self, image: QImage) -> None:
        # Only 192 samples from a tiny thumbnail: no model, OCR or heavy vision
        # processing on the GUI thread, and no claim of component detection.
        thumb = image.scaled(16, 12, Qt.AspectRatioMode.IgnoreAspectRatio)
        values = [thumb.pixelColor(x, y).lightness() for y in range(12) for x in range(16)]
        brightness = sum(values) / len(values)
        if brightness < 45:
            guidance = "  ·  Too dark"
        elif brightness > 222:
            guidance = "  ·  Too bright"
        else:
            guidance = ""
        self.quality_label.setText(f"{image.width()} × {image.height()}{guidance}")

    def capture(self) -> None:
        if self._busy or not self._camera_enabled or self._show_capture:
            return
        # Refresh the readiness check, but inspect the frame actually rendered
        # on the canvas rather than obtaining a newer, unseen camera frame.
        if self._frame is None or time.monotonic() - self._last_frame_time > self.STALE_SECONDS:
            self.capture_button.setEnabled(False)
            return
        try:
            frame = self._frame.copy()
        except Exception:
            self.quality_label.setText("Capture failed")
            return
        image = bgr_to_qimage(frame)
        if image is None:
            return
        self._show_capture = True
        self._captured_image = image
        self._captured_frame = frame
        self._captured_reading = self.canvas.reading
        self._observations = []
        self.canvas.observations = []
        self.canvas.set_image(image, captured=True)
        self.live_button.setText("Live View")
        self.live_button.setVisible(True)
        self._request_inspection(frame, "Captured frame retained. Inspecting the visible details…")

    def _focus_entered(self) -> None:
        """Enter in the focus box: re-ask about a capture, else nothing."""
        if self._show_capture:
            self.ask_again()

    def ask_again(self) -> None:
        """Put a fresh question to the frame already captured."""
        if self._busy or not self._show_capture or self._captured_frame is None:
            return
        self._observations = []
        self.canvas.observations = []
        self.canvas.update()
        self._request_inspection(
            self._captured_frame, "Re-examining the same captured frame…")

    def _request_inspection(self, frame: Any, note: str) -> None:
        """The one route to the inspection service, for a capture or a re-ask."""
        self._request_id = uuid.uuid4().hex
        self._busy = True
        self.state_label.setText("ANALYSING")
        self.capture_button.setEnabled(False)
        self.ask_button.setEnabled(False)
        self.quality_label.setText(note)
        self.report_heading.setText("Inspection in Progress")
        self.report.setHtml('<h3>Inspecting…</h3>')
        self.scan_requested.emit({"request_id": self._request_id, "frame": frame,
                                  "focus": self.focus_input.text().strip(),
                                  "mode": self.selected_mode(),
                                  "grid": self.selected_grid()})

    def show_live(self) -> None:
        if self._show_capture:
            self._show_capture = False
            self.live_button.setText("Captured Frame")
            self.ask_button.setEnabled(False)
            if not self._camera_enabled:
                self.canvas.clear_image("Camera paused")
            self._refresh()
        elif self._captured_image is not None:
            self._show_capture = True
            self.live_button.setText("Live View")
            self.canvas.set_image(self._captured_image, captured=True)
            self.canvas.reading = self._captured_reading
            self.canvas.observations = self._observations
            self.canvas.update()
            self.capture_button.setEnabled(False)
            self.ask_button.setEnabled(not self._busy and self._captured_frame is not None)
            self.state_label.setText("ANALYSING" if self._busy else "CAPTURED")

    def _matches(self, payload: Any) -> bool:
        return isinstance(payload, dict) and bool(self._request_id) and payload.get("request_id") == self._request_id

    def on_scan_started(self, payload: Any) -> None:
        if not isinstance(payload, dict) or not payload.get("request_id"):
            return
        if not self._matches(payload) and not self._busy:
            # Voice scans originate outside the widget. Adopt the request ID
            # before its result arrives, without relabelling the previous board.
            self._request_id = str(payload["request_id"])
            self._busy = True
            self._show_capture = True
            self._captured_image = None
            self._captured_frame = None
            self._captured_reading = None
            self._observations = []
            self.canvas.clear_image("Capturing…")
            self.state_label.setText("ANALYSING")
            self.capture_button.setEnabled(False)
            self.ask_button.setEnabled(False)
            self.report.setHtml("<h3>Inspecting…</h3>")
        if self._matches(payload) and self._busy:
            self.report_heading.setText("Inspection in Progress")

    def on_scan_result(self, payload: Any) -> None:
        if not self._matches(payload):
            return
        self._busy = False
        status = payload.get("status", "complete")
        self.report_heading.setText("Local Checks" if status == "local_only" else
                                    {"text": "Text Read", "count": "Count",
                                     "anything": "What ORION Sees"}.get(
                                        str(payload.get("mode") or ""), "Inspection Notes"))
        observations = payload.get("observations")
        self._observations = [item for item in observations[:24] if isinstance(item, dict)] if isinstance(observations, list) else []
        jpeg = payload.get("jpeg")
        if isinstance(jpeg, (bytes, bytearray)):
            submitted = QImage.fromData(bytes(jpeg))
            if not submitted.isNull():
                self._captured_image = submitted
                if self._captured_frame is None:
                    self._captured_frame = bytes(jpeg)
                self.live_button.setVisible(True)
                self.live_button.setText("Live View" if self._show_capture else "Captured Frame")
        if self._show_capture and self._captured_image is not None:
            self.canvas.set_image(self._captured_image, captured=True)
            self.canvas.observations = self._observations
            self.canvas.update()
            self.state_label.setText("LOCAL CHECKS" if status == "local_only" else "CAPTURED")
        self.capture_button.setEnabled(self._camera_enabled and self._frame is not None and not self._show_capture)
        self.ask_button.setEnabled(self._captured_frame is not None and self._show_capture)
        summary = _plain(payload.get("summary") or "Inspection completed.")
        parts = [f'<h3 style="color:{C.ACCENT}">{summary}</h3>']
        for number, item in enumerate(self._observations, 1):
            label, detail = _plain(item.get("label"), 180), _plain(item.get("detail"))
            confidence = item.get("confidence")
            confidence_text = ""
            if isinstance(confidence, (int, float)) and math.isfinite(confidence) and 0 <= confidence <= 1:
                confidence_text = f" · model confidence {confidence:.0%}"
            evidence = _plain(item.get("evidence"), 900)
            evidence_line = f'<br><span style="color:{C.MUTED}">Visible evidence: {evidence}</span>' if evidence else ""
            cells = item.get("cells") or []
            where = ""
            if cells:
                from ..electronics_inspection import cell_span
                where = f' <span style="color:{C.ACCENT}">[{_plain(cell_span(cells), 20)}]</span>'
            parts.append(f'<p><b>{number}. {label}</b>{where}<span style="color:{C.MUTED}">{confidence_text}</span><br>{detail}{evidence_line}</p>')
        markings = payload.get("markings")
        if isinstance(markings, list) and markings:
            parts.append("<h4>" + ("Markings · Verify Visually" if payload.get("mode") in (None, "electronics")
                                   else "Text Found · Verify Visually") + "</h4><ul>")
            for marking in markings[:24]:
                if isinstance(marking, dict) and marking.get("text"):
                    parts.append(f'<li>{_plain(marking["text"], 180)}</li>')
            parts.append("</ul>")
        for key, heading in (("next_steps", "Next Checks"), ("limitations", "Limits of This View")):
            items = payload.get(key) or []
            if isinstance(items, list) and items:
                parts.append(f"<h4>{heading}</h4><ul>" + "".join(f"<li>{_plain(item, 700)}</li>" for item in items[:8]) + "</ul>")
        quality = payload.get("quality") or {}
        warnings = quality.get("warnings", []) if isinstance(quality, dict) else []
        self.quality_label.setText(" · ".join(str(item)[:180] for item in warnings[:3]) if warnings else "")
        self.report.setHtml("".join(parts))

    def on_scan_error(self, payload: Any) -> None:
        if not self._matches(payload):
            return
        self._busy = False
        self.report_heading.setText("Inspection Unavailable")
        retained = self._captured_image is not None
        self.state_label.setText("CAPTURE RETAINED" if retained else "SCAN UNAVAILABLE")
        if not retained:
            self._show_capture = False
            self.canvas.clear_image("Inspection unavailable")
        next_step = "Capture kept - ask again to retry." if retained else ""
        self.report.setHtml("<h3>Inspection could not finish.</h3><p>" + _plain(payload.get("error") or payload.get("message") or "The vision service is unavailable.") + "</p><p>" + next_step + "</p>")
        self.capture_button.setEnabled(self._camera_enabled and self._frame is not None and not self._show_capture)
        self.ask_button.setEnabled(self._captured_frame is not None and self._show_capture)

    def shutdown(self) -> None:
        self.stop_camera()
        self._scanner.stop(wait=False)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.stop()
        if self._owns_camera:
            self.stop_camera()
        else:
            # Frames stop arriving with the timer; the detector should not sit
            # on a thread waiting for work that cannot come.
            self._scanner.stop(wait=False)
        super().hideEvent(event)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        if self._camera_enabled:
            self._scanner.start()
            self._timer.start()
        super().showEvent(event)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        orientation = Qt.Orientation.Vertical if self.width() < 860 else Qt.Orientation.Horizontal
        if self.splitter.orientation() != orientation:
            self.splitter.setOrientation(orientation)
            self.splitter.setSizes([470, 230] if orientation == Qt.Orientation.Vertical else [700, 290])
        super().resizeEvent(event)
