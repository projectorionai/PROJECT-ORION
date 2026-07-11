"""Reusable widgets: metric bars, the mini orb, floating toggle, key dialog."""

from __future__ import annotations

import math
from typing import Any, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..constants import C
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


class MiniOrb(QWidget):
    """
    64×64 floating orb that persists when the user navigates away from the
    main HUD view.  Shares the same amplitude and state signals as CentralHud
    via OrionBus; displays in the bottom-right corner of the shell.
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

        # Cyan Fresnel rim (lower-right arc).
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 229, 255, clamp_channel(140 + amp * 90)), 1.4))
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
        painter.setBrush(QColor(0, 229, 255, 220))
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
            "  color: #ffffff;"
            "  border: 1px solid #ff1a3c;"
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
        self.setStyleSheet(
            "QDialog { background: #050508; color: #ffffff; }"
            "QLabel { color: #ffffff; }"
            "QLineEdit {"
            "  background: #0f0f14; color: #ffffff;"
            "  border: 1px solid #991024; border-radius: 6px;"
            "  padding: 10px; selection-background-color: #ff1a3c;"
            "}"
            "QPushButton {"
            "  background: #991024; color: #ffffff;"
            "  border: 1px solid #ff1a3c; border-radius: 6px; padding: 8px 14px;"
            "}"
            "QPushButton:hover { background: #ff1a3c; }"
        )
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
