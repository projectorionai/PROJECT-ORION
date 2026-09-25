"""Reusable widgets: metric bars, the mini orb, floating toggle, key dialog."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..constants import C


# ── control labelling ─────────────────────────────────────────────────────────

def describe_control(widget: Any, name: str, hint: str = "") -> Any:
    """Give a control a human name as well as its visual label. Returns it.

    ORION's decks use symbol-labelled buttons — ``|<``, ``⟳``, ``⇄``, ``‹`` —
    because they are compact and read well in a dense panel. The cost is that
    the *only* thing describing the control is a glyph:

      * assistive technology announces the character itself, so a screen reader
        says "less-than" where a person needs "previous move";
      * and a symbol that is obvious to whoever wrote the panel is a guess for
        anyone else.

    An audit found 119 buttons, 21 of them symbol-only, and **not one** of those
    21 carried an accessible name. Eight had no tooltip either.

    Setting both together in one call is the point: a tooltip helps a sighted
    user with a mouse and does nothing for a keyboard or screen-reader user,
    while an accessible name does the reverse. Doing only half is the easy
    mistake, so there is one function that does both.

    ``hint`` adds detail for the tooltip — the accessible name stays short,
    because it is spoken aloud.
    """
    label = (name or "").strip()
    if not label:
        return widget
    detail = (hint or "").strip()
    try:
        widget.setAccessibleName(label)
        if detail:
            widget.setAccessibleDescription(detail)
        widget.setToolTip(f"{label} — {detail}" if detail else label)
    except Exception:
        pass          # a control that cannot be described must still work
    return widget
from ..utils import clamp_channel


class _Particle:
    """Lightweight particle with __slots__ — avoids per-instance __dict__ overhead."""
    __slots__ = ("x", "y", "vx", "vy", "life", "size")

    def __init__(self, x: float, y: float, vx: float, vy: float, life: float, size: float) -> None:
        self.x    = x
        self.y    = y
        self.vx   = vx
        self.vy   = vy
        self.life = life
        self.size = size


class MetricBar(QWidget):
    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.label         = label
        self.value         = 0.0
        self.display_value = 0.0
        self.setMinimumHeight(58)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: float) -> None:
        self.value = max(0.0, min(100.0, float(value)))
        self.update()

    def paintEvent(self, event: Any) -> None:
        self.display_value += (self.value - self.display_value) * 0.18
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        painter.setPen(QPen(QColor(C.BORDER), 1))
        painter.setBrush(QColor(C.PANEL))
        painter.drawRoundedRect(rect, 6, 6)
        inner = rect.adjusted(10, 31, -10, -10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1a1116"))
        painter.drawRoundedRect(inner, 4, 4)
        fill_width = inner.width() * (self.display_value / 100.0)
        fill       = QRectF(inner.left(), inner.top(), fill_width, inner.height())
        gradient   = QLinearGradient(fill.topLeft(), fill.topRight())
        gradient.setColorAt(0.0, QColor(C.PRI_DIM))
        gradient.setColorAt(0.82, QColor(C.PRI))
        gradient.setColorAt(1.0, QColor(C.ACCENT))       # cyan leading edge
        painter.setBrush(gradient)
        painter.drawRoundedRect(fill, 4, 4)
        # Bright cyan tip cap for a high-tech "signal" feel.
        if fill_width > 3:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(C.ACCENT))
            tip = QRectF(fill.right() - 2.0, inner.top(), 2.0, inner.height())
            painter.drawRoundedRect(tip, 1, 1)
        painter.setPen(QColor(C.WHITE))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(10, 6, rect.width() - 20, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.label,
        )
        painter.setPen(QColor(C.ACCENT))
        painter.setFont(QFont("Cascadia Mono", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(10, 6, rect.width() - 20, 20),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{self.display_value:05.1f}%",
        )


class JobStatusPanel(QFrame):
    """Generic status row for any background job (Mark XX design-spec
    §3/§9). The Studio deck's former AudioCanvas drew two bars and a
    decorative sine-wave scan line — a custom-painted widget spent on what
    was really just two integers (raw/processed file counts), the exact
    case the design spec calls out as visual novelty over information
    density. This reuses MetricBar (already built, already on-brand,
    already animated) driven by a completion percentage, plus a status
    line — the same shape ANY background job needs (ingestion, audio
    processing, a Forge session), not a bespoke canvas per job type.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.bar = MetricBar(title)
        self.bar.setMinimumHeight(46)
        layout.addWidget(self.bar)
        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def set_counts(self, done: int, total: int, status: str = "") -> None:
        """done/total as whole counts (e.g. processed / (raw+processed)) —
        rendered as a single completion percentage rather than two
        separately-scaled bars, since "how much of the queue is done" is
        the one number that actually matters at a glance."""
        done = max(0, int(done))
        total = max(0, int(total))
        pct = (done / total * 100.0) if total else 0.0
        self.bar.set_value(pct)
        if status:
            self.status_label.setText(status)


class MetricGraph(QWidget):
    """Real-time scrolling graph of a 0–100 host metric.

    This is the *detailed* system-telemetry view shown on the TELEMETRY page;
    the compact MetricBar in the left panel remains the at-a-glance readout, so
    the two are complementary rather than duplicated.  Cheap by design: one
    fixed-length deque and a single antialiased path per repaint, and it only
    repaints when a new sample arrives (no free-running timer)."""

    def __init__(self, label: str, unit: str = "%",
                 parent: Optional[QWidget] = None, capacity: int = 120) -> None:
        super().__init__(parent)
        self.label         = label
        self.unit          = unit
        self.value         = 0.0
        self.display_value = 0.0
        self.history: deque[float] = deque([0.0] * capacity, maxlen=capacity)
        self.setMinimumHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: float) -> None:
        self.value = max(0.0, min(100.0, float(value)))
        self.history.append(self.value)
        self.update()

    def paintEvent(self, event: Any) -> None:
        self.display_value += (self.value - self.display_value) * 0.20
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        painter.setPen(QPen(QColor(C.BORDER), 1))
        painter.setBrush(QColor(C.PANEL))
        painter.drawRoundedRect(rect, 6, 6)

        plot = rect.adjusted(10, 30, -10, -10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#140c10"))
        painter.drawRoundedRect(plot, 4, 4)

        # Faint reference grid at 25/50/75%.
        painter.setPen(QPen(QColor(255, 255, 255, 18), 1))
        for frac in (0.25, 0.5, 0.75):
            y = plot.bottom() - plot.height() * frac
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        n = len(self.history)
        if n >= 2:
            dx  = plot.width() / (n - 1)
            pts = [
                QPointF(plot.left() + i * dx,
                        plot.bottom() - (v / 100.0) * plot.height())
                for i, v in enumerate(self.history)
            ]
            # Area fill under the trace.
            area = QPainterPath()
            area.moveTo(plot.left(), plot.bottom())
            for pt in pts:
                area.lineTo(pt)
            area.lineTo(plot.right(), plot.bottom())
            area.closeSubpath()
            grad = QLinearGradient(plot.topLeft(), plot.bottomLeft())
            grad.setColorAt(0.0, QColor(255, 26, 60, 120))
            grad.setColorAt(1.0, QColor(255, 26, 60, 8))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(grad)
            painter.drawPath(area)
            # The trace itself.
            line = QPainterPath()
            line.moveTo(pts[0])
            for pt in pts[1:]:
                line.lineTo(pt)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(C.PRI), 1.6))
            painter.drawPath(line)
            # Bright cyan leading edge.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(C.ACCENT))
            painter.drawEllipse(pts[-1], 2.6, 2.6)

        painter.setPen(QColor(C.WHITE))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(QRectF(10, 6, rect.width() - 20, 18),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         self.label)
        painter.setPen(QColor(C.ACCENT))
        painter.setFont(QFont("Cascadia Mono", 9, QFont.Weight.DemiBold))
        painter.drawText(QRectF(10, 6, rect.width() - 20, 18),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         f"{self.display_value:05.1f}{self.unit}")


class MiniOrb(QWidget):
    """
    64×64 floating orb that persists when the user navigates away from the
    face view.  Shares the same amplitude and state signals as the face via
    OrionBus; displays in the bottom-right corner of the shell.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(64, 64)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self.amplitude        = 0.0
        self.target_amplitude = 0.0
        self._pulse           = 0.0
        self.rotation         = 0.0
        self.state_name       = "STANDBY"

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_amplitude(self, value: float) -> None:
        self.target_amplitude = max(0.0, min(1.0, float(value)))

    def set_state(self, state: str) -> None:
        self.state_name = str(state or "STANDBY").upper()
        self.update()

    def _tick(self) -> None:
        # A hidden widget still gets its timer, and still asks Qt to repaint —
        # 30 wake-ups a second on the same thread as the audio callback, for
        # pixels nobody can see. Every other animated widget in the deck already
        # guards this way; this one was missed.
        if not self.isVisible():
            return
        self.amplitude  += (self.target_amplitude - self.amplitude) * 0.22
        self._pulse      = (self._pulse + 0.04) % (2 * math.pi)
        self.rotation    = (self.rotation + 1.2 + self.amplitude * 4.0) % 360.0
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(C.BG))

        cx      = 32.0
        cy      = 32.0
        amp     = self.amplitude
        pulse_n = (math.sin(self._pulse) + 1.0) * 0.5
        orb_r   = 14.0
        painter.setPen(Qt.PenStyle.NoPen)

        # Corona
        cr = 28 * (0.90 + 0.14 * pulse_n + 0.24 * amp)
        painter.setBrush(QColor(255, 26, 60, clamp_channel(28 + pulse_n * 20 + amp * 72)))
        painter.drawEllipse(QRectF(cx - cr, cy - cr, cr * 2, cr * 2))

        # Sphere body — radial gradient offset up-left for volume.
        body = QRadialGradient(cx - orb_r * 0.34, cy - orb_r * 0.38, orb_r * 1.7)
        body.setColorAt(0.0, QColor(255, 120, 140, 255))
        body.setColorAt(0.30, QColor(210, 26, 56, 250))
        body.setColorAt(0.75, QColor(120, 10, 26, 245))
        body.setColorAt(1.0, QColor(40, 4, 12, 255))
        painter.setBrush(body)
        painter.drawEllipse(QRectF(cx - orb_r, cy - orb_r, orb_r * 2, orb_r * 2))

        # Incandescent core.
        core_r = orb_r * (0.40 + 0.12 * pulse_n + 0.22 * amp)
        core = QRadialGradient(cx, cy, max(1.0, core_r))
        core.setColorAt(0.0, QColor(255, 240, 245, clamp_channel(210 + amp * 45)))
        core.setColorAt(0.5, QColor(255, 100, 120, 220))
        core.setColorAt(1.0, QColor(255, 100, 120, 0))
        painter.setBrush(core)
        painter.drawEllipse(QRectF(cx - core_r, cy - core_r, core_r * 2, core_r * 2))

        # Specular highlight.
        spec_r = orb_r * 0.22
        sx, sy = cx - orb_r * 0.34, cy - orb_r * 0.38
        painter.setBrush(QColor(255, 255, 255, clamp_channel(150 + amp * 80)))
        painter.drawEllipse(QRectF(sx - spec_r, sy - spec_r, spec_r * 2, spec_r * 2))

        # Fresnel rim (lower-right arc). It was cyan: a cool key light against
        # a warm orb is a real rendering technique, but it made a saturated
        # second hue the most prominent thing on ORION's identity element
        # after the crimson itself. A neutral rim reads just as cool against
        # red and belongs to the palette.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(*C.SILVER_RGB, clamp_channel(140 + amp * 90)), 1.4))
        painter.drawArc(QRectF(cx - orb_r, cy - orb_r, orb_r * 2, orb_r * 2),
                        int(-70 * 16), int(150 * 16))

        # Equatorial ring + a single orbiting node.
        ring_rx = orb_r * 1.20
        ring_ry = orb_r * 0.30
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.rotation * 0.6)
        painter.setPen(QPen(QColor(255, 26, 60, clamp_channel(120 + amp * 100)), 1.0))
        painter.drawEllipse(QRectF(-ring_rx, -ring_ry, ring_rx * 2, ring_ry * 2))
        node_a = math.radians(self.rotation * 2.4)
        nx, ny = math.cos(node_a) * ring_rx, math.sin(node_a) * ring_ry
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*C.SILVER_RGB, 220))
        painter.drawEllipse(QRectF(nx - 1.6, ny - 1.6, 3.2, 3.2))
        painter.restore()


class HolographicToggle(QPushButton):
    """Frameless always-on-top pill that shows/hides the core window."""

    def __init__(self, target: QWidget) -> None:
        super().__init__("ORION")
        self.target        = target
        self._drag_offset: Any = None
        self.setWindowTitle("O.R.I.O.N. Toggle")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(92, 42)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QPushButton {"
            "  background: rgba(15, 15, 20, 188);"
            f"  color: {C.WHITE};"
            f"  border: 1px solid {C.PRI};"
            "  border-radius: 20px;"
            "  font-family: 'Segoe UI'; font-weight: 800;"
            "}"
            "QPushButton:hover { background: rgba(255, 26, 60, 214); }"
        )
        self.clicked.connect(self.toggle_target)

    def toggle_target(self) -> None:
        if self.target.isVisible():
            self.target.hide()
        else:
            self.target.show()
            self.target.raise_()
            self.target.activateWindow()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)


class ApiKeyDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("O.R.I.O.N. Key Exchange")
        self.setModal(True)
        self.setMinimumWidth(460)
        # The application's own styling. This was the last of three dialogs
        # carrying a private copy of it — crimson outlines on the field and
        # the buttons, which is the accent spent on furniture.
        from .style import APP_STYLESHEET

        self.setStyleSheet(APP_STYLESHEET)
        layout = QVBoxLayout(self)
        title  = QLabel("Gemini Live authentication token required.")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        body   = QLabel("Enter the API key. It will be stored locally in config/api_keys.json.")
        body.setWordWrap(True)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("AIza...")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addWidget(self.key_edit)
        layout.addWidget(buttons)

    def key(self) -> str:
        return self.key_edit.text().strip()

    def _accept_if_valid(self) -> None:
        if not self.key():
            QMessageBox.warning(self, "Key Required", "Supply a Gemini API key to initialise O.R.I.O.N.")
            return
        self.accept()
