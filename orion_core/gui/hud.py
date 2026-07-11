"""CentralHud — ORION's avatar: a futuristic, sphere-shaded holographic orb.

Mark X.6 renderer. The orb is no longer a flat stack of ellipses: it is a
depth-shaded sphere wrapped in an electric-cyan Fresnel rim, breathing at its
core, encircled by a camera-iris aperture that dilates with the voice, and
orbited by satellite nodes on tilted gyroscopic planes. A parallax starfield
sits in the cached background, and holographic readouts frame the whole thing.

Rendering pipeline (two-pass, unchanged contract):
  Pass 1 — Static background cached into a QPixmap (grid + starfield + ambient
           glow + corner brackets). Re-rendered only on resize; zero per-frame
           cost.
  Pass 2 — Dynamic elements painted every tick, layered for genuine depth:
           orbital planes → back satellites → sphere orb (shading, core,
           aperture, rim) → front satellites → scanner → reticle → particles →
           holographic readouts → banner.

Public contract preserved: ``set_amplitude``, ``set_state``, ``set_banner``,
``timer``.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ..constants import C
from ..utils import clamp_channel
from .widgets import _Particle


def _rgba(rgb: tuple[int, int, int], alpha: float) -> QColor:
    """Build a QColor from an (r, g, b) tuple and a clamped alpha."""
    return QColor(rgb[0], rgb[1], rgb[2], clamp_channel(alpha))


class CentralHud(QWidget):
    """The living ORION orb — depth-shaded, holographic, voice-reactive."""

    _PARTICLE_HARD_CAP = 140
    _PARTICLE_TRIM_TO  = 100

    # Orbiting satellite planes: (radius×, vertical squash, plane tilt°, angular
    # speed, colour rgb, node size). Three planes at different tilts read as a
    # gyroscope caging the core.
    _SAT_PLANES = (
        (1.46, 0.30,  -26.0, 1.7, C.ACCENT_RGB, 3.2),
        (1.74, 0.24,   38.0, 1.1, C.PRI_RGB,    3.6),
        (1.28, 0.52,   84.0, 2.3, (255, 255, 255), 2.6),
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.rotation         = 0.0
        self.scan             = 0.0
        self.amplitude        = 0.0
        self.target_amplitude = 0.0
        self.face_amount      = 0.0
        self.face_target      = 0.0
        self.iris             = 0.30   # aperture openness (eased)
        self.state_name       = "INITIALISING"
        self.particles: list[_Particle] = []
        self.banner_text      = ""
        self.banner_alpha     = 0.0
        self.banner_priority  = 0
        self._pulse           = 0.0    # fast breathe oscillator (0 → 2π)
        self._pulse_slow      = 0.0    # slow drift oscillator
        self._scanline        = 0.0    # 0→1 CRT-style sweep position
        self._sat_angles      = [0.0, 2.1, 4.2]

        self.setMinimumSize(300, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # QPixmap cache for the static background — invalidated on resize.
        self._bg_pixmap: Optional[QPixmap] = None
        self._bg_size: tuple[int, int]     = (0, 0)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ── cache invalidation on resize ────────────────────────────────────────

    def resizeEvent(self, event: Any) -> None:
        self._bg_pixmap = None
        super().resizeEvent(event)

    # ── public interface ─────────────────────────────────────────────────────

    def set_amplitude(self, value: float) -> None:
        self.target_amplitude = max(0.0, min(1.0, float(value)))

    def set_banner(self, text: str, priority: int = 1) -> None:
        self.banner_text     = str(text or "").strip()[:140]
        self.banner_priority = max(0, min(5, int(priority or 1)))
        self.banner_alpha    = 255.0 if self.banner_text else 0.0
        self.update()

    def set_state(self, state: str) -> None:
        self.state_name  = str(state or "STANDBY").upper()
        active_states    = {"CONNECTING", "LISTENING", "PROCESSING", "SPEAKING"}
        self.face_target = 1.0 if self.state_name in active_states else 0.0
        target_interval  = 16 if self.face_target or self.amplitude > 0.04 else 33
        if self.timer.interval() != target_interval:
            self.timer.setInterval(target_interval)
        self.update()

    # ── animation tick ───────────────────────────────────────────────────────

    def _tick(self) -> None:
        self.rotation    = (self.rotation + 1.2 + self.amplitude * 4.0) % 360.0
        self.scan        = (self.scan + 2.5) % 360.0
        self.amplitude  += (self.target_amplitude - self.amplitude) * 0.22
        self.face_amount += (self.face_target - self.face_amount) * 0.12
        self._pulse      = (self._pulse + 0.05) % (2 * math.pi)
        self._pulse_slow = (self._pulse_slow + 0.012) % (2 * math.pi)
        self._scanline   = (self._scanline + 0.0055) % 1.0
        # Aperture dilates while speaking/listening and with loudness.
        iris_target = 0.30 + 0.55 * self.face_amount + 0.35 * self.amplitude
        self.iris   += (min(1.0, iris_target) - self.iris) * 0.14
        for index, plane in enumerate(self._SAT_PLANES):
            self._sat_angles[index] = (
                self._sat_angles[index] + math.radians(plane[3] + self.amplitude * 6.0)
            ) % (2 * math.pi)
        if self.amplitude > 0.08:
            self._spawn_particles()
        self._update_particles()
        if self.banner_alpha > 0:
            self.banner_alpha = max(0.0, self.banner_alpha - 1.4)
        self.update()

    # ── particle system ──────────────────────────────────────────────────────

    def _spawn_particles(self) -> None:
        count    = min(6, 1 + int(self.amplitude * 8))
        centre_x = self.width()  * 0.5
        centre_y = self.height() * 0.5
        for n in range(count):
            angle = math.radians((self.rotation * 2 + n * 51) % 360)
            speed = 0.8 + self.amplitude * 4.0
            self.particles.append(_Particle(
                x    = centre_x + math.cos(angle) * 18,
                y    = centre_y + math.sin(angle) * 18,
                vx   = math.cos(angle) * speed,
                vy   = math.sin(angle) * speed,
                life = 1.0,
                size = 1.5 + self.amplitude * 4.0,
            ))
        if len(self.particles) > self._PARTICLE_HARD_CAP:
            del self.particles[:len(self.particles) - self._PARTICLE_TRIM_TO]

    def _update_particles(self) -> None:
        live: list[_Particle] = []
        for p in self.particles:
            p.x    += p.vx
            p.y    += p.vy
            p.life -= 0.018
            p.vx   *= 0.992
            p.vy   *= 0.992
            if p.life > 0:
                live.append(p)
        self.particles = live

    # ── main paint event ─────────────────────────────────────────────────────

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width  = self.width()
        height = self.height()
        centre = QPointF(width / 2, height / 2)
        radius = min(width, height) * 0.34

        # ── PASS 1: static background (cached QPixmap) ───────────────────────
        current_size = (width, height)
        if self._bg_pixmap is None or self._bg_size != current_size:
            self._bg_pixmap = QPixmap(width, height)
            self._bg_pixmap.fill(QColor(C.BG))
            bg = QPainter(self._bg_pixmap)
            bg.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._paint_grid_static(bg, width, height)
            self._paint_starfield_static(bg, width, height)
            self._paint_ambient_static(bg, width, height)
            self._paint_brackets_static(bg, width, height)
            bg.end()
            self._bg_size = current_size

        painter.drawPixmap(0, 0, self._bg_pixmap)

        # ── PASS 2: dynamic elements (every frame), layered for depth ────────
        self._paint_arcs(painter, centre, radius)
        satellites = self._satellite_points(centre, radius)
        self._paint_orbital_planes(painter, centre, radius)
        self._paint_satellites(painter, [s for s in satellites if s["depth"] < 0])
        self._paint_sphere_orb(painter, centre, radius)
        self._paint_satellites(painter, [s for s in satellites if s["depth"] >= 0])
        self._paint_scanner(painter, centre, radius)
        self._paint_reticle(painter, centre, radius)
        self._paint_particles(painter)
        self._paint_scanline(painter, width, height)
        self._paint_readouts(painter, width, height, radius)
        self._paint_banner(painter, width)

    # ── static background helpers ────────────────────────────────────────────

    def _paint_grid_static(self, painter: QPainter, width: int, height: int) -> None:
        """Perspective-tinted background grid, cached — zero per-frame cost."""
        minor = QPen(QColor(153, 16, 36, 12), 1)
        major = QPen(QColor(153, 16, 36, 30), 1)
        spacing = 38
        for index, x in enumerate(range(0, width, spacing)):
            painter.setPen(major if index % 5 == 0 else minor)
            painter.drawLine(x, 0, x, height)
        for index, y in enumerate(range(0, height, spacing)):
            painter.setPen(major if index % 5 == 0 else minor)
            painter.drawLine(0, y, width, y)

    def _paint_starfield_static(self, painter: QPainter, width: int, height: int) -> None:
        """Deterministic parallax starfield — seeded so it never flickers."""
        rng = random.Random(20260704)  # fixed seed → stars never flicker
        painter.setPen(Qt.PenStyle.NoPen)
        for _ in range(int(width * height / 4200)):
            x = rng.random() * width
            y = rng.random() * height
            twinkle = rng.random()
            size = 0.6 + twinkle * 1.4
            if twinkle > 0.82:                       # rare cyan star
                painter.setBrush(_rgba(C.ACCENT_RGB, 60 + twinkle * 80))
            else:
                painter.setBrush(QColor(255, 255, 255, int(20 + twinkle * 70)))
            painter.drawEllipse(QRectF(x, y, size, size))

    def _paint_ambient_static(self, painter: QPainter, width: int, height: int) -> None:
        """Crimson core glow, cyan counter-glow and an edge vignette — cached."""
        cx, cy = width * 0.5, height * 0.5
        glow = QRadialGradient(cx, cy, min(width, height) * 0.52)
        glow.setColorAt(0.0, QColor(255, 26, 60, 30))
        glow.setColorAt(0.55, QColor(255, 26, 60, 12))
        glow.setColorAt(1.0, QColor(255, 26, 60, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawRect(0, 0, width, height)
        # Faint cyan rim-light bloom, offset — gives the scene a cold key light.
        cool = QRadialGradient(width * 0.66, height * 0.62, min(width, height) * 0.42)
        cool.setColorAt(0.0, QColor(0, 229, 255, 16))
        cool.setColorAt(1.0, QColor(0, 229, 255, 0))
        painter.setBrush(cool)
        painter.drawRect(0, 0, width, height)
        vignette = QRadialGradient(cx, cy, max(width, height) * 0.72)
        vignette.setColorAt(0.0, QColor(0, 0, 0, 0))
        vignette.setColorAt(0.70, QColor(0, 0, 0, 0))
        vignette.setColorAt(1.0, QColor(0, 0, 0, 180))
        painter.setBrush(vignette)
        painter.drawRect(0, 0, width, height)

    def _paint_brackets_static(self, painter: QPainter, width: int, height: int) -> None:
        """Corner targeting brackets with a cyan inner tick — cached."""
        margin = 22
        length = min(width, height) * 0.12
        corners = [
            ((margin, margin),                  (1, 1)),
            ((width - margin, margin),          (-1, 1)),
            ((margin, height - margin),         (1, -1)),
            ((width - margin, height - margin), (-1, -1)),
        ]
        for (px, py), (sx, sy) in corners:
            painter.setPen(QPen(QColor(255, 26, 60, 150), 2))
            painter.drawLine(QPointF(px, py), QPointF(px + sx * length, py))
            painter.drawLine(QPointF(px, py), QPointF(px, py + sy * length))
            painter.setPen(QPen(QColor(0, 229, 255, 130), 1))
            painter.drawLine(QPointF(px + sx * 6, py + sy * 6),
                             QPointF(px + sx * (length * 0.5), py + sy * 6))

    # ── dynamic painting helpers ─────────────────────────────────────────────

    def _paint_arcs(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        """Rotating arc layers — crimson body with a single cyan data-arc."""
        layers = [
            (radius * 1.00,  31, 108, 3, C.PRI_RGB),
            (radius * 0.78, -47,  64, 2, C.PRI_RGB),
            (radius * 1.22, 146,  78, 2, C.ACCENT_RGB),   # cyan accent arc
            (radius * 0.52, 212, 122, 2, C.PRI_RGB),
        ]
        for index, (r, offset, span, w, rgb) in enumerate(layers):
            alpha = 120 + self.amplitude * 100 - index * 12
            pen   = QPen(_rgba(rgb, alpha), w)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            rect  = QRectF(centre.x() - r, centre.y() - r, r * 2, r * 2)
            start = int((self.rotation * (1.0 if index % 2 == 0 else -0.7) + offset) * 16)
            painter.drawArc(rect, start, int(span * 16))
        painter.setPen(QPen(QColor(255, 255, 255, clamp_channel(58 + self.amplitude * 70)), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for r in (radius * 0.34, radius * 0.64, radius * 1.42):
            rect = QRectF(centre.x() - r, centre.y() - r, r * 2, r * 2)
            painter.drawEllipse(rect)

    def _satellite_points(self, centre: QPointF, radius: float) -> list[dict[str, Any]]:
        """Project each orbiting node onto the screen with a depth cue so the
        renderer can split them in-front-of / behind the orb."""
        cx, cy = centre.x(), centre.y()
        points: list[dict[str, Any]] = []
        for index, (r_scale, squash, tilt_deg, _speed, rgb, size) in enumerate(self._SAT_PLANES):
            angle  = self._sat_angles[index]
            r      = radius * r_scale * 0.52
            # Point on the (unrotated) tilted ellipse, then rotate the plane.
            lx = math.cos(angle) * r
            ly = math.sin(angle) * r * squash
            depth = math.sin(angle)                       # +front / -behind
            tilt = math.radians(tilt_deg)
            rx = lx * math.cos(tilt) - ly * math.sin(tilt)
            ry = lx * math.sin(tilt) + ly * math.cos(tilt)
            points.append({
                "x": cx + rx, "y": cy + ry, "depth": depth, "rgb": rgb,
                "size": size * (0.7 + 0.5 * (depth + 1) * 0.5),
            })
        return points

    def _paint_orbital_planes(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        """Faint tilted orbit ellipses — the gyroscope cage behind the orb."""
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for r_scale, squash, tilt_deg, _speed, rgb, _size in self._SAT_PLANES:
            r = radius * r_scale * 0.52
            painter.save()
            painter.translate(centre.x(), centre.y())
            painter.rotate(tilt_deg)
            painter.setPen(QPen(_rgba(rgb, 32 + self.amplitude * 46), 1))
            painter.drawEllipse(QRectF(-r, -r * squash, r * 2, r * 2 * squash))
            painter.restore()

    def _paint_satellites(self, painter: QPainter, nodes: list[dict[str, Any]]) -> None:
        """Draw orbiting nodes with a soft halo; dimmer when behind the orb."""
        painter.setPen(Qt.PenStyle.NoPen)
        for node in nodes:
            depth_f = (node["depth"] + 1) * 0.5          # 0 behind → 1 front
            alpha   = 70 + depth_f * 150
            x, y, s = node["x"], node["y"], node["size"]
            painter.setBrush(_rgba(node["rgb"], alpha * 0.28))
            painter.drawEllipse(QRectF(x - s * 2.4, y - s * 2.4, s * 4.8, s * 4.8))
            painter.setBrush(_rgba(node["rgb"], alpha))
            painter.drawEllipse(QRectF(x - s, y - s, s * 2, s * 2))
            painter.setBrush(QColor(255, 255, 255, clamp_channel(alpha)))
            painter.drawEllipse(QRectF(x - s * 0.4, y - s * 0.4, s * 0.8, s * 0.8))

    def _paint_sphere_orb(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        """The orb proper: sphere shading, incandescent core, camera-iris
        aperture and an electric-cyan Fresnel rim."""
        amp     = self.amplitude
        cx, cy  = centre.x(), centre.y()
        pulse_n = (math.sin(self._pulse) + 1.0) * 0.5
        orb_r   = radius * 0.52

        painter.setPen(Qt.PenStyle.NoPen)

        # 1. Diffuse corona (breathes + swells with the voice) — a soft glow
        #    via a radial gradient so loud speech blooms rather than turning
        #    the whole frame into a flat crimson disc.
        corona_r = radius * (0.90 + 0.12 * pulse_n + 0.18 * amp)
        corona = QRadialGradient(cx, cy, max(1.0, corona_r))
        corona.setColorAt(0.0, _rgba(C.PRI_RGB, 30 + pulse_n * 16 + amp * 60))
        corona.setColorAt(0.55, _rgba(C.PRI_RGB, 18 + amp * 34))
        corona.setColorAt(1.0, _rgba(C.PRI_RGB, 0))
        painter.setBrush(corona)
        painter.drawEllipse(QRectF(cx - corona_r, cy - corona_r, corona_r * 2, corona_r * 2))

        # 2. Sphere body — radial gradient offset up-left gives real volume.
        body = QRadialGradient(cx - orb_r * 0.34, cy - orb_r * 0.38, orb_r * 1.7)
        body.setColorAt(0.0, QColor(255, 120, 140, 255))
        body.setColorAt(0.28, QColor(210, 26, 56, 250))
        body.setColorAt(0.72, QColor(120, 10, 26, 245))
        body.setColorAt(1.0, QColor(40, 4, 12, 255))
        painter.setBrush(body)
        painter.drawEllipse(QRectF(cx - orb_r, cy - orb_r, orb_r * 2, orb_r * 2))

        # 3. Camera-iris aperture — six blades that dilate with the voice.
        self._paint_aperture(painter, cx, cy, orb_r * 0.86)

        # 4. Incandescent core behind the aperture opening.
        core_r = orb_r * (0.30 + 0.12 * pulse_n + 0.24 * amp) * (0.5 + 0.7 * self.iris)
        core = QRadialGradient(cx, cy, max(1.0, core_r))
        core.setColorAt(0.0, QColor(255, 240, 245, clamp_channel(210 + amp * 45)))
        core.setColorAt(0.4, _rgba(C.CORE_RGB, 220))
        core.setColorAt(1.0, _rgba(C.CORE_RGB, 0))
        painter.setBrush(core)
        painter.drawEllipse(QRectF(cx - core_r, cy - core_r, core_r * 2, core_r * 2))

        # 5. Fresnel rim — bright cyan lower-right, crimson elsewhere.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rim = QPen(_rgba(C.ACCENT_RGB, 150 + amp * 90), 2.0)
        painter.setPen(rim)
        painter.drawArc(QRectF(cx - orb_r, cy - orb_r, orb_r * 2, orb_r * 2),
                        int(-70 * 16), int(150 * 16))
        painter.setPen(QPen(_rgba(C.PRI_RGB, 120 + amp * 80), 1.4))
        painter.drawArc(QRectF(cx - orb_r, cy - orb_r, orb_r * 2, orb_r * 2),
                        int(110 * 16), int(210 * 16))

        # 6. Specular hotspot — the glassy highlight.
        spec_r = orb_r * 0.24
        sx, sy = cx - orb_r * 0.36, cy - orb_r * 0.40
        spec = QRadialGradient(sx, sy, spec_r)
        spec.setColorAt(0.0, QColor(255, 255, 255, clamp_channel(150 + amp * 90)))
        spec.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(spec)
        painter.drawEllipse(QRectF(sx - spec_r, sy - spec_r, spec_r * 2, spec_r * 2))

        # 7. Equatorial rotating ring with tick marks (3D squash).
        self._paint_equator(painter, cx, cy, orb_r, amp)

        # 8. Amplitude pulse rings during speech.
        if amp > 0.06:
            for factor, base in ((0.55, 200), (0.82, 110)):
                pr = orb_r * (1.0 + amp * factor)
                painter.setPen(QPen(_rgba(C.PRI_RGB, amp * base), 1.4))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QRectF(cx - pr, cy - pr, pr * 2, pr * 2))

        # 9. State label beneath the orb.
        label = (
            "SPEAKING"  if self.state_name == "SPEAKING"  else
            "LISTENING" if self.state_name == "LISTENING" else
            "ONLINE"    if self.face_target               else
            "STANDBY"
        )
        painter.setFont(QFont("Segoe UI Semibold", 10, QFont.Weight.DemiBold))
        painter.setPen(QColor(255, 255, 255, clamp_channel(150 + amp * 105)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawText(QRectF(cx - radius, cy + orb_r + 20, radius * 2, 26),
                         Qt.AlignmentFlag.AlignCenter, label)

    def _paint_aperture(self, painter: QPainter, cx: float, cy: float, r: float) -> None:
        """Six-blade camera iris. ``self.iris`` (0→1) sets how far it opens; a
        wider opening reveals more of the incandescent core."""
        opening   = 0.18 + 0.60 * self.iris
        inner_r   = r * opening
        blade_a   = 138 - int(self.iris * 70)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.rotation * 0.25)
        for n in range(6):
            a0 = math.radians(n * 60)
            a1 = math.radians(n * 60 + 60)
            blade = QPainterPath()
            blade.moveTo(math.cos(a0) * r, math.sin(a0) * r)
            blade.lineTo(math.cos(a1) * r, math.sin(a1) * r)
            blade.lineTo(math.cos(a1) * inner_r * 1.02, math.sin(a1) * inner_r * 1.02)
            blade.lineTo(math.cos(a0 + 0.16) * inner_r, math.sin(a0 + 0.16) * inner_r)
            blade.closeSubpath()
            painter.setBrush(QColor(18, 3, 8, blade_a))
            painter.drawPath(blade)
        # Bright edge where the blades meet the core opening.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_rgba(C.ACCENT_RGB, 90 + self.iris * 90), 1.0))
        painter.drawEllipse(QRectF(-inner_r, -inner_r, inner_r * 2, inner_r * 2))
        painter.restore()

    def _paint_equator(self, painter: QPainter, cx: float, cy: float,
                       orb_r: float, amp: float) -> None:
        ring_rx = orb_r * 1.20
        ring_ry = orb_r * 0.30
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.rotation * 0.6)
        painter.setPen(QPen(_rgba(C.PRI_RGB, 140 + amp * 100), 1.5))
        painter.drawEllipse(QRectF(-ring_rx, -ring_ry, ring_rx * 2, ring_ry * 2))
        painter.setPen(QPen(_rgba(C.CORE_RGB, 160 + amp * 80), 1.2))
        for tick_deg in range(0, 360, 30):
            t   = math.radians(tick_deg)
            tx  = math.cos(t) * ring_rx
            ty  = math.sin(t) * ring_ry
            nx  = tx / (ring_rx + 1e-9)
            ny  = ty / (ring_ry + 1e-9)
            ln  = orb_r * 0.10
            painter.drawLine(QPointF(tx, ty), QPointF(tx - nx * ln, ty - ny * ln))
        painter.restore()

    def _paint_scanner(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        angle = math.radians(self.scan)
        end   = QPointF(
            centre.x() + math.cos(angle) * radius * 1.48,
            centre.y() + math.sin(angle) * radius * 1.48,
        )
        pen = QPen(_rgba(C.ACCENT_RGB, 80 + self.amplitude * 130), 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(centre, end)
        wedge = QPainterPath()
        wedge.moveTo(centre)
        rect = QRectF(centre.x() - radius * 1.48, centre.y() - radius * 1.48,
                      radius * 2.96, radius * 2.96)
        wedge.arcTo(rect, -self.scan, -22)
        wedge.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_rgba(C.ACCENT_RGB, 14 + self.amplitude * 30))
        painter.drawPath(wedge)

    def _paint_reticle(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        """Subtle crosshair ticks at the cardinal points — targeting feel."""
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1))
        gap    = radius * 1.30
        length = radius * 0.12
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x0 = centre.x() + dx * gap
            y0 = centre.y() + dy * gap
            painter.drawLine(QPointF(x0, y0),
                             QPointF(x0 + dx * length, y0 + dy * length))

    def _paint_particles(self, painter: QPainter) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        for p in self.particles:
            painter.setBrush(_rgba(C.PRI_RGB, int(230 * p.life)))
            half = p.size * 0.5
            painter.drawEllipse(QRectF(p.x - half, p.y - half, p.size, p.size))

    def _paint_scanline(self, painter: QPainter, width: int, height: int) -> None:
        """A soft CRT-style horizontal sweep drifting down the display — a subtle
        holographic 'refresh' that sells the futuristic feel without distraction."""
        y = self._scanline * height
        band = 26.0
        grad = QLinearGradient(0.0, y - band, 0.0, y + band)
        grad.setColorAt(0.0, QColor(0, 229, 255, 0))
        grad.setColorAt(0.5, QColor(0, 229, 255, 20))
        grad.setColorAt(1.0, QColor(0, 229, 255, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(grad)
        painter.drawRect(QRectF(0.0, y - band, float(width), band * 2))
        painter.setPen(QPen(QColor(0, 229, 255, 38), 1))
        painter.drawLine(QPointF(0.0, y), QPointF(float(width), y))

    def _paint_readouts(self, painter: QPainter, width: int, height: int,
                        radius: float) -> None:
        """Holographic telemetry that frames the orb: a left status stack and a
        right amplitude column. Drawn in the cyan data-hue, always legible."""
        mono = QFont("Cascadia Mono", 8)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        painter.setFont(mono)

        # Left column — live status lines.
        lines = [
            f"STATE   {self.state_name[:11]}",
            f"ROT     {self.rotation:05.1f}°",
            f"SCAN    {self.scan:05.1f}°",
            f"IRIS    {int(self.iris * 100):3d}%",
        ]
        painter.setPen(_rgba(C.ACCENT_RGB, 150))
        x = 30
        y = height * 0.5 - (len(lines) * 15) / 2
        for line in lines:
            painter.drawText(QPointF(x, y), line)
            y += 15

        # Right column — amplitude ladder.
        bars   = 12
        active = int(self.amplitude * bars + 0.5)
        bx     = width - 42
        by     = height * 0.5 - (bars * 9) / 2
        for i in range(bars):
            lit = (bars - i) <= active
            painter.setPen(Qt.PenStyle.NoPen)
            if lit:
                painter.setBrush(_rgba(C.ACCENT_RGB if i > 3 else C.PRI_RGB, 210))
            else:
                painter.setBrush(QColor(255, 255, 255, 26))
            painter.drawRoundedRect(QRectF(bx, by + i * 9, 16, 6), 2, 2)
        painter.setPen(_rgba(C.ACCENT_RGB, 150))
        painter.drawText(QPointF(bx - 4, by - 8), "VOX")

    def _paint_banner(self, painter: QPainter, width: int) -> None:
        if not self.banner_text or self.banner_alpha <= 0:
            return
        alpha = clamp_channel(self.banner_alpha)
        rect  = QRectF(width * 0.15, 20, width * 0.70, 46)
        painter.setPen(QPen(_rgba(C.ACCENT_RGB, alpha), 1))
        painter.setBrush(QColor(12, 14, 20, clamp_channel(alpha * 0.86)))
        painter.drawRoundedRect(rect, 8, 8)
        painter.setFont(QFont("Segoe UI Semibold", 10, QFont.Weight.DemiBold))
        painter.setPen(QColor(255, 255, 255, alpha))
        painter.drawText(rect.adjusted(14, 0, -14, 0),
                         Qt.AlignmentFlag.AlignCenter, self.banner_text)
