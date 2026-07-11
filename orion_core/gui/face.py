"""
HologramFace — ORION's dissolving voxel visage.

This replaces the earlier cyberpunk skull with a face in the spirit of the
reference concept art: a human male face reconstructed from thousands of small
cubes (voxels) of electric blue, densest and brightest on the intact right of
the face and breaking apart into drifting cubes across the left and crown.

It is voice-reactive and alive, driven exactly like the orb:

    set_amplitude(0..1)  → the mouth opens in time with ORION's voice
    set_speaking(bool)   → brightens the whole face while he audibly speaks
    set_state("SPEAKING"|"LISTENING"|"PROCESSING"|"STANDBY"|…) → expression
    apply_emotion(name, params) → full emotional rendering (Mark X.7)

Mark X.7: expressions are no longer hardcoded here.  When the
EmotionStateManager broadcasts ``bus.emotion_changed`` (wired in the core
window), ``apply_emotion`` receives a complete parameter set — voxel density,
brow, eye geometry, particle velocity, glow, palette and animation speed —
and the face eases toward it.  Without an emotion engine attached the
legacy per-state expressions still apply, so the widget stands alone.

At rest the eyes are serene and nearly closed; while listening or speaking they
open into glowing cyan irises.  The face breathes, blinks and continuously
sheds and reforms voxels along its dissolving edge, so it never looks static.

Everything is drawn with QPainter — zero image assets.  The static skin (the
sculpted silhouette, its light-and-shadow, and the ragged dissolving rim) is
baked once per resize into a flat cell list; each frame only re-tints those
cells and paints the emissive eyes, mouth and a swarm of drifting cubes on top,
so the per-frame cost stays modest.

The whole palette derives from the module-level colour stops — swap them for
crimson and the face turns crimson without another edit.
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
    QRadialGradient,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ..constants import C
from ..utils import clamp_channel

# ── Voxel palette — deep navy shadow → bright cyan-white highlight ────────────
SKIN_DARK   = (9, 16, 34)
SKIN_MID    = (34, 84, 150)
SKIN_BRIGHT = (150, 214, 246)
EYE_RGB     = C.ACCENT_RGB          # electric cyan
HOT_RGB     = (210, 245, 255)


def _rgba(rgb: tuple[int, int, int], alpha: float) -> QColor:
    return QColor(rgb[0], rgb[1], rgb[2], clamp_channel(alpha))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _skin(bb: float) -> tuple[int, int, int]:
    """Three-stop tone ramp: navy shadow → mid blue → bright cyan highlight."""
    if bb <= 0.5:
        return _mix(SKIN_DARK, SKIN_MID, bb * 2.0)
    return _mix(SKIN_MID, SKIN_BRIGHT, (bb - 0.5) * 2.0)


# Face silhouette (v, half-width in u), crown → chin.
_FACE_PROFILE = (
    (-1.20, 0.06), (-1.04, 0.32), (-0.84, 0.58), (-0.58, 0.74),
    (-0.28, 0.83), (0.02, 0.86), (0.30, 0.82), (0.56, 0.72),
    (0.80, 0.58), (1.00, 0.40), (1.16, 0.20), (1.28, 0.05),
)


def _gauss(u: float, v: float, cu: float, cv: float, su: float, sv: float) -> float:
    du = (u - cu) / su
    dv = (v - cv) / sv
    e = du * du + dv * dv
    return math.exp(-e) if e < 9.0 else 0.0


def _face_width(v: float) -> float:
    pts = _FACE_PROFILE
    if v <= pts[0][0]:
        return pts[0][1]
    if v >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        v0, w0 = pts[i]
        v1, w1 = pts[i + 1]
        if v0 <= v <= v1:
            t = (v - v0) / (v1 - v0 + 1e-9)
            return w0 + (w1 - w0) * t
    return 0.0


def _skin_brightness(u: float, v: float) -> float:
    """
    Sculpted light-and-shadow — lit from the front with a rightward bias so the
    intact right of the face reads brightest (as in the reference), falling to
    shadow on the left before it dissolves.  The nose ridge, hollow eye sockets,
    brow, cheekbones and jaw shading are what make the voxels read as a face.
    """
    b = 0.30 + 0.44 * (0.5 + 0.42 * u)                       # front light, right bias
    b += 0.10 * (1.0 - min(1.0, (u / 0.9) ** 2))             # frontal volume
    b += 0.30 * _gauss(u, v, 0.0, 0.14, 0.09, 0.32)          # nose ridge
    b += 0.16 * _gauss(u, v, 0.0, 0.44, 0.10, 0.10)          # nose tip
    b -= 0.15 * _gauss(u, v, 0.0, 0.55, 0.17, 0.06)          # under-nose shadow
    b -= 0.36 * (_gauss(u, v, 0.34, -0.10, 0.18, 0.13)
                 + _gauss(u, v, -0.34, -0.10, 0.18, 0.13))   # eye sockets
    b += 0.13 * (_gauss(u, v, 0.34, -0.30, 0.22, 0.07)
                 + _gauss(u, v, -0.34, -0.30, 0.22, 0.07))   # brow highlight
    b -= 0.13 * (_gauss(u, v, 0.34, -0.19, 0.20, 0.05)
                 + _gauss(u, v, -0.34, -0.19, 0.20, 0.05))   # under-brow shadow
    b += 0.12 * (_gauss(u, v, 0.5, 0.18, 0.22, 0.18)
                 + _gauss(u, v, -0.5, 0.18, 0.22, 0.18))     # cheekbones
    b -= 0.10 * _gauss(u, v, 0.0, 0.72, 0.20, 0.05)          # under-lip shadow
    b += 0.10 * _gauss(u, v, 0.0, 0.90, 0.24, 0.12)          # chin highlight
    b -= 0.10 * _gauss(u, v, -0.5, -0.4, 0.3, 0.4)           # left temple falls to shadow
    return max(0.0, min(1.0, b))


def _dissolve_prob(u: float, v: float, edge: float) -> float:
    """How likely a voxel is missing (dissolved) — biased to the left and crown."""
    left = max(0.0, (-u - 0.05) / 0.95)
    top = max(0.0, (-v - 0.55) / 0.65)
    return min(0.96, 0.62 * (left ** 1.6) + 0.55 * (top ** 1.4) + edge)


class HologramFace(QWidget):
    """A voice-reactive dissolving voxel face. Same signals as CentralHud."""

    _PARTICLE_CAP = 150

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.amplitude        = 0.0
        self.target_amplitude = 0.0
        self.state_name       = "STANDBY"
        self.speaking         = False

        # Eased expression channels.
        self.jaw      = 0.0
        self.eye_open = 0.35
        self.brow     = 0.0
        self.glow     = 0.4
        self._eye_t   = 0.35
        self._brow_t  = 0.0
        self._glow_t  = 0.4

        self._pulse   = 0.0
        self._scan    = 0.0
        self._t       = 0.0
        self._gaze_x  = 0.0
        self._gaze_y  = 0.0
        self._blink   = 1.0
        self._blink_in = random.randint(60, 160)

        # ── Mark X.7 emotional rendering (targets eased every tick) ──────────
        self._emotion = ""                 # active emotion name ('' = legacy)
        self._density = 1.0                # baked voxel density (rebake on change)
        self._density_t = 1.0
        self._eye_w, self._eye_w_t = 1.0, 1.0     # eye width multiplier
        self._eye_h, self._eye_h_t = 1.0, 1.0     # eye height multiplier
        self._pvel, self._pvel_t = 1.0, 1.0       # particle velocity multiplier
        self._speed, self._speed_t = 1.0, 1.0     # animation-rate multiplier
        self._curve, self._curve_t = 0.05, 0.05   # mouth curve (-1..+1)
        self._tension, self._tension_t = 0.0, 0.0 # mouth tension (0..1)
        self._pulse_amt, self._pulse_amt_t = 0.0, 0.0   # glow pulse depth
        self._turb, self._turb_t = 0.1, 0.1       # particle turbulence
        self._direction = "drift"                 # drift | up | down | burst
        # Eased palette stops (dark, mid, bright, accent) — start at module hues.
        self._pal = {"dark": list(SKIN_DARK), "mid": list(SKIN_MID),
                     "bright": list(SKIN_BRIGHT), "accent": list(EYE_RGB)}
        self._pal_t = {k: list(v) for k, v in self._pal.items()}

        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Baked face, invalidated on resize.
        self._cells: list[tuple[float, float, float, float, bool]] = []
        self._emitters: list[tuple[float, float]] = []
        self.particles: list[list[float]] = []       # [x, y, vx, vy, life, size]
        self._cx = self._cy = 0.0
        self._u = 1.0
        self._cell = 6.0
        self._built_size: tuple[int, int] = (0, 0)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ── public interface (mirrors CentralHud / the old skull) ─────────────────

    def set_amplitude(self, value: float) -> None:
        self.target_amplitude = max(0.0, min(1.0, float(value)))

    def set_speaking(self, active: bool) -> None:
        self.speaking = bool(active)

    def set_state(self, state: str) -> None:
        self.state_name = str(state or "STANDBY").upper()
        if self._emotion:
            # Mark X.7: the EmotionStateManager owns expression — it already
            # folds pipeline state into its baseline, so only the state label
            # and jaw/gaze behaviour (which read state_name) update here.
            self.update()
            return
        if self.state_name == "LISTENING":
            self._eye_t, self._brow_t, self._glow_t = 1.0, 0.5, 0.8
        elif self.state_name == "PROCESSING":
            self._eye_t, self._brow_t, self._glow_t = 0.5, -0.6, 0.62
        elif self.state_name == "SPEAKING":
            self._eye_t, self._brow_t, self._glow_t = 0.85, 0.15, 1.0
        elif self.state_name in {"CONNECTING", "INITIALISING"}:
            self._eye_t, self._brow_t, self._glow_t = 0.45, 0.0, 0.5
        elif self.state_name == "PAUSED":
            self._eye_t, self._brow_t, self._glow_t = 0.12, -0.1, 0.35
        else:  # STANDBY — serene, eyes low
            self._eye_t, self._brow_t, self._glow_t = 0.3, 0.0, 0.4
        self.update()

    def apply_emotion(self, name: str, params: Any) -> None:
        """
        Mark X.7 — consume one EmotionStateManager parameter set.  Nothing is
        hardcoded here: whatever arrives is eased into over the next moments.
        Voxel density changes rebake the skin (quantised, so only a genuine
        density shift pays the few-millisecond rebuild).
        """
        if not isinstance(params, dict):
            return
        self._emotion = str(name or "")
        self._brow_t = float(params.get("brow", self._brow_t))
        self._glow_t = float(params.get("glow", self._glow_t))
        self._eye_w_t = float(params.get("eye_width", 1.0))
        self._eye_h_t = float(params.get("eye_height", 1.0))
        # eye_height doubles as the openness driver when the engine is active.
        self._eye_t = max(0.12, min(1.2, 0.75 * self._eye_h_t))
        self._pvel_t = float(params.get("particle_velocity", 1.0))
        self._speed_t = float(params.get("speed", 1.0))
        self._curve_t = float(params.get("mouth_curve", 0.05))
        self._tension_t = max(0.0, min(1.0, float(params.get("mouth_tension", 0.0))))
        self._pulse_amt_t = max(0.0, min(1.0, float(params.get("glow_pulse", 0.0))))
        self._turb_t = max(0.0, min(1.0, float(params.get("turbulence", 0.1))))
        self._direction = str(params.get("particle_direction", "drift"))
        for key, field in (("palette_dark", "dark"), ("palette_mid", "mid"),
                           ("palette_bright", "bright"), ("accent", "accent")):
            value = params.get(key)
            if isinstance(value, (tuple, list)) and len(value) == 3:
                self._pal_t[field] = [float(v) for v in value]
        density = round(float(params.get("voxel_density", 1.0)) * 20) / 20.0
        if abs(density - self._density_t) >= 0.05:
            self._density_t = density
            self._cells = []          # rebake on the next paint
        self.update()

    # ── animation ──────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        dt = max(0.008, self.timer.interval() / 1000.0)
        self._t += dt * self._speed
        self.amplitude += (self.target_amplitude - self.amplitude) * 0.35
        jaw_t = self.amplitude if (self.speaking or self.state_name == "SPEAKING") else 0.0
        self.jaw += (jaw_t - self.jaw) * 0.45
        self.eye_open += (self._eye_t - self.eye_open) * 0.12
        self.brow += (self._brow_t - self.brow) * 0.10
        self.glow += (self._glow_t - self.glow) * 0.08
        # Emotional rendering channels ease at their own gentle rates.
        self._eye_w += (self._eye_w_t - self._eye_w) * 0.10
        self._eye_h += (self._eye_h_t - self._eye_h) * 0.10
        self._pvel += (self._pvel_t - self._pvel) * 0.10
        self._speed += (self._speed_t - self._speed) * 0.08
        self._curve += (self._curve_t - self._curve) * 0.10
        self._tension += (self._tension_t - self._tension) * 0.10
        self._pulse_amt += (self._pulse_amt_t - self._pulse_amt) * 0.10
        self._turb += (self._turb_t - self._turb) * 0.10
        for field, target in self._pal_t.items():
            current = self._pal[field]
            for i in range(3):
                current[i] += (target[i] - current[i]) * 0.08
        self._pulse = (self._pulse + 0.055 * self._speed) % (2 * math.pi)
        self._scan = (self._scan + 0.9 * self._speed) % 100.0
        # Gaze wander for presence; centres while speaking.
        wander = 0.0 if self.state_name == "SPEAKING" else 0.4
        gx = wander * math.sin(self._t * 0.7)
        gy = wander * 0.6 * math.sin(self._t * 0.9 + 1.1)
        self._gaze_x += (gx - self._gaze_x) * 0.06
        self._gaze_y += (gy - self._gaze_y) * 0.06
        # Blink.
        self._blink_in -= 1
        if self._blink_in <= 0:
            self._blink = max(0.0, self._blink - 0.34)
            if self._blink <= 0.02:
                self._blink_in = random.randint(70, 210)
        else:
            self._blink = min(1.0, self._blink + 0.22)
        self._update_particles()
        self.update()

    def _update_particles(self) -> None:
        """
        Particle behaviour carries the emotional energy (spec): direction,
        speed, turbulence and dissolution rate all follow the active profile.
            drift — the resting shed toward the dissolving left
            up    — buoyant, energetic (happiness / excitement)
            down  — slow, sinking, fading early (sadness)
            burst — fast radial eruptions from the face (frustration / alert)
        """
        if self._emitters and random.random() < 0.9:
            vel = self._pvel
            direction = self._direction
            for _ in range(1 + int(self.amplitude * 4)):
                px, py = random.choice(self._emitters)
                if direction == "up":
                    vx = random.uniform(-0.4, 0.4) * vel
                    vy = (-0.8 - random.uniform(0.0, 1.0)) * vel
                elif direction == "down":
                    vx = random.uniform(-0.25, 0.25) * vel
                    vy = (0.5 + random.uniform(0.0, 0.7)) * vel
                elif direction == "burst":
                    dx, dy = px - self._cx, py - self._cy
                    dist = math.hypot(dx, dy) or 1.0
                    vx = (dx / dist) * (1.2 + random.uniform(0.0, 1.4)) * vel
                    vy = (dy / dist) * (1.2 + random.uniform(0.0, 1.4)) * vel
                else:  # drift — cubes shed toward the dissolving left
                    vx = (-0.4 - random.uniform(0.0, 1.2) - self.amplitude * 1.5) * vel
                    vy = (-0.3 + random.uniform(-0.5, 0.5)) * vel
                self.particles.append([px, py, vx, vy, 1.0,
                                       self._cell * random.uniform(0.5, 1.05)])
        if len(self.particles) > self._PARTICLE_CAP:
            del self.particles[: len(self.particles) - self._PARTICLE_CAP]
        # Dissolution: sinking sadness fades early; bursts burn out fast.
        fade = 0.02
        if self._direction == "down":
            fade = 0.035
        elif self._direction == "burst":
            fade = 0.03
        turb = self._turb * 0.35
        gravity = 0.012 if self._direction != "up" else -0.004
        live = []
        for q in self.particles:
            q[0] += q[2]
            q[1] += q[3]
            q[3] += gravity
            if turb > 0.0:
                q[2] += random.uniform(-turb, turb)
                q[3] += random.uniform(-turb, turb)
            q[2] *= 0.99
            q[4] -= fade
            q[5] *= 0.985
            if q[4] > 0 and q[5] > 0.6:
                live.append(q)
        self.particles = live

    # ── bake the face once per resize ─────────────────────────────────────────

    def _rebuild(self, w: int, h: int) -> None:
        self._cx = w * 0.5
        self._cy = h * 0.47
        self._u = u = min(w, h) * 0.42
        self._cell = cell = max(4.0, u * 0.062)
        du = cell / u
        cells: list[tuple[float, float, float, float, bool]] = []
        emitters: list[tuple[float, float]] = []
        v = -1.24
        row = 0
        while v <= 1.30:
            wv = _face_width(v)
            uu = -wv
            col = 0
            while uu <= wv + 1e-6:
                edge = 0.0
                if abs(uu) > wv - 1.8 * du:
                    edge = 0.55
                if v < -1.0:
                    edge = max(edge, 0.5)
                # Emotional voxel density: >1 condenses the skin (fewer holes),
                # <1 dissolves it further — the geometry itself carries mood.
                self._density = self._density_t
                prob = min(0.96, _dissolve_prob(uu, v, edge) / max(0.5, self._density))
                hsh = ((col * 92821 + row * 53987) % 1000) / 1000.0
                px = self._cx + uu * u
                py = self._cy + v * u
                if hsh < prob:
                    # Missing voxel — a source of drifting cubes.
                    if hsh < prob * 0.6:
                        emitters.append((px, py))
                    uu += du
                    col += 1
                    continue
                b = _skin_brightness(uu, v)
                is_edge = edge > 0.0 or prob > 0.25
                # Per-voxel character (spec: position, alpha, glow, lifespan):
                # a deterministic shimmer phase, and a pseudo-3D depth from the
                # facial bulge (nose nearest, rim furthest) that drives gaze
                # parallax so the head reads as a volume, not a decal.
                phase = hsh * 6.283
                z = _gauss(uu, v, 0.0, 0.15, 0.55, 0.75)
                cells.append((px, py, cell, b, is_edge, phase, z))
                if is_edge:
                    emitters.append((px, py))
                uu += du
                col += 1
            v += du
            row += 1
        self._cells = cells
        self._emitters = emitters
        self._built_size = (w, h)

    # ── paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(C.BG))

        if not self._cells or self._built_size != (w, h):
            self._rebuild(w, h)

        cx, cy, u = self._cx, self._cy, self._u
        self._paint_backdrop(p, w, h, cx, cy, u)
        self._paint_aura(p, cx, cy, u)
        self._paint_voxels(p)
        self._paint_particles(p)
        self._paint_eyes(p, cx, cy, u)
        self._paint_brows(p, cx, cy, u)
        self._paint_mouth(p, cx, cy, u)
        self._paint_frame(p, w, h)

    def _paint_backdrop(self, p, w, h, cx, cy, u) -> None:
        halo = QRadialGradient(cx + u * 0.15, cy, u * 2.1)
        halo.setColorAt(0.0, _rgba(SKIN_MID, 40 + self.glow * 34))
        halo.setColorAt(0.6, _rgba(SKIN_MID, 12))
        halo.setColorAt(1.0, _rgba(SKIN_MID, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawRect(0, 0, w, h)

    def _paint_aura(self, p, cx, cy, u) -> None:
        if self.amplitude > 0.06:
            for factor, base in ((1.12, 120), (1.32, 70)):
                r = u * factor
                p.setPen(QPen(_rgba(EYE_RGB, self.amplitude * base), 1.4))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

    def _paint_voxels(self, p) -> None:
        pulse_n = (math.sin(self._pulse) + 1.0) * 0.5
        breathe = 1.0 + 0.02 * pulse_n
        bob = math.sin(self._t * 1.05) * self._u * 0.01
        # Glow: steady base, plus the emotion's pulse depth (spec: low/medium/
        # high crossed with pulsing/steady — frustration throbs, calm holds).
        pulse_mod = 1.0 + self._pulse_amt * 0.18 * math.sin(self._t * 6.0)
        bscale = (0.6 + 0.7 * self.glow) * (0.92 + 0.08 * pulse_n) * pulse_mod
        amp = self.amplitude
        gap = self._cell * 0.16
        shimmer_t = self._t * 2.2
        # Gaze parallax: near voxels (nose) shift more than the rim → volume.
        par_x = self._gaze_x * self._u * 0.05
        par_y = self._gaze_y * self._u * 0.03

        # Emotion-eased palette stops (fall back to the module hues at start).
        dark = self._pal["dark"]
        mid = self._pal["mid"]
        bright = self._pal["bright"]

        p.save()
        p.translate(self._cx, self._cy + bob)
        p.scale(breathe, breathe)
        p.translate(-self._cx, -self._cy)
        p.setPen(Qt.PenStyle.NoPen)
        for px, py, sz, b, edge, phase, z in self._cells:
            bb = b * bscale + amp * (0.16 if edge else 0.05)
            if bb < 0.02:
                continue
            if bb > 1.0:
                bb = 1.0
            if bb <= 0.5:
                t = bb * 2.0
                r = dark[0] + (mid[0] - dark[0]) * t
                g = dark[1] + (mid[1] - dark[1]) * t
                bl = dark[2] + (mid[2] - dark[2]) * t
            else:
                t = (bb - 0.5) * 2.0
                r = mid[0] + (bright[0] - mid[0]) * t
                g = mid[1] + (bright[1] - mid[1]) * t
                bl = mid[2] + (bright[2] - mid[2]) * t
            # Per-voxel life: a gentle asynchronous shimmer.
            twinkle = 0.9 + 0.1 * math.sin(shimmer_t + phase)
            a = int((165 if edge else 255) * twinkle)
            s = sz - gap
            x = px + z * par_x
            y = py + z * par_y
            p.setBrush(QColor(clamp_channel(r * twinkle), clamp_channel(g * twinkle),
                              clamp_channel(bl * twinkle), clamp_channel(a)))
            p.drawRect(QRectF(x - s / 2, y - s / 2, s, s))
            # Lit top face — a lighter strip gives the cubes their extruded look.
            p.setBrush(QColor(clamp_channel(r + 45), clamp_channel(g + 40),
                              clamp_channel(bl + 30), clamp_channel(a)))
            p.drawRect(QRectF(x - s / 2, y - s / 2, s, s * 0.3))
        p.restore()

    def _paint_particles(self, p) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        for x, y, _vx, _vy, life, size in self.particles:
            a = clamp_channel(int(200 * life))
            p.setBrush(_rgba(SKIN_BRIGHT, a))
            p.drawRect(QRectF(x - size / 2, y - size / 2, size, size))
            p.setBrush(_rgba(HOT_RGB, a * 0.8))
            p.drawRect(QRectF(x - size / 2, y - size / 2, size, size * 0.34))

    def _eye_centre(self, cx, cy, u, side) -> tuple[float, float]:
        return cx + side * 0.34 * u, cy - 0.10 * u

    def _paint_eyes(self, p, cx, cy, u) -> None:
        openness = max(0.0, self.eye_open) * self._blink
        accent = tuple(int(v) for v in self._pal["accent"])
        for side in (-1, 1):
            ex, ey = self._eye_centre(cx, cy, u, side)
            half_w = 0.15 * u * self._eye_w
            if openness < 0.18:
                # Serene, near-closed — a soft glowing lid line with a gentle dip.
                lid = QPainterPath()
                lid.moveTo(ex - half_w, ey)
                lid.quadTo(ex, ey + 0.03 * u, ex + half_w, ey)
                p.setPen(QPen(_rgba(accent, 120 + self.glow * 90), max(1.6, u * 0.014)))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(lid)
                continue
            half_h = 0.11 * u * max(0.16, openness) * self._eye_h
            # Dark socket recess.
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(4, 8, 16, 235))
            p.drawEllipse(QRectF(ex - half_w, ey - half_h, half_w * 2, half_h * 2))
            # Glowing iris in the emotion accent, offset by gaze.
            ix = ex + self._gaze_x * u * 0.05
            iy = ey + self._gaze_y * u * 0.04
            ir = half_h * 0.95
            iris = QRadialGradient(ix, iy, ir * 2.0)
            iris.setColorAt(0.0, QColor(255, 255, 255, clamp_channel(200 + self.glow * 55)))
            iris.setColorAt(0.4, _rgba(accent, 235))
            iris.setColorAt(1.0, _rgba(accent, 0))
            p.setBrush(iris)
            p.drawEllipse(QRectF(ix - ir, iy - ir, ir * 2, ir * 2))

    def _paint_brows(self, p, cx, cy, u) -> None:
        tilt = self.brow
        bright = tuple(int(v) for v in self._pal["bright"])
        for side in (-1, 1):
            ex, ey = self._eye_centre(cx, cy, u, side)
            inner_y = ey - 0.17 * u + max(0.0, -tilt) * 0.06 * u - max(0.0, tilt) * 0.05 * u
            outer_y = ey - 0.20 * u - max(0.0, tilt) * 0.02 * u
            p.setPen(QPen(_rgba(bright, 120 + self.glow * 90), max(2.0, u * 0.03)))
            p.drawLine(QPointF(ex - side * 0.03 * u, inner_y),
                       QPointF(ex + side * 0.14 * u, outer_y))

    def _paint_mouth(self, p, cx, cy, u) -> None:
        my = cy + 0.58 * u
        # Tension compresses the mouth: narrower, flatter, tighter (spec:
        # curved / neutral / tense / compressed).
        tension = max(0.0, min(1.0, self._tension))
        hw = 0.22 * u * (1.0 - 0.3 * tension)
        opening = self.jaw * 0.15 * u * (1.0 - 0.5 * tension)
        accent = tuple(int(v) for v in self._pal["accent"])
        bright = tuple(int(v) for v in self._pal["bright"])
        # Emotional curve: positive lifts the corners (smile), negative drops
        # them; the mid-point moves the opposite way for a natural arc.
        curve = max(-1.0, min(1.0, self._curve))
        corner_y = my - curve * 0.045 * u
        top_mid = my - opening * 0.5 - u * 0.006 + curve * 0.02 * u
        bot_mid = my + opening * 0.5 + u * 0.02 + curve * 0.03 * u
        cavity = QPainterPath()
        cavity.moveTo(cx - hw, corner_y)
        cavity.quadTo(cx, top_mid, cx + hw, corner_y)
        cavity.quadTo(cx, bot_mid, cx - hw, corner_y)
        cavity.closeSubpath()
        if opening > u * 0.01:
            grad = QLinearGradient(cx, top_mid, cx, bot_mid)
            grad.setColorAt(0.0, _rgba(HOT_RGB, 60 + self.amplitude * 150))
            grad.setColorAt(0.6, _rgba(accent, 110 + self.amplitude * 110))
            grad.setColorAt(1.0, QColor(6, 12, 24, 235))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(grad)
            p.drawPath(cavity)
        else:
            p.setPen(QPen(_rgba(bright, 130 + self.glow * 70), max(1.4, u * 0.016)))
            p.setBrush(Qt.BrushStyle.NoBrush)
            lip = QPainterPath()
            lip.moveTo(cx - hw, corner_y)
            lip.quadTo(cx, my + (0.02 + 0.05 * curve) * u, cx + hw, corner_y)
            p.drawPath(lip)

    def _paint_frame(self, p, w, h) -> None:
        p.setPen(QPen(_rgba(EYE_RGB, 110), 1.6))
        m, L = 16, min(w, h) * 0.08
        for (px, py), (sx, sy) in (((m, m), (1, 1)), ((w - m, m), (-1, 1)),
                                   ((m, h - m), (1, -1)), ((w - m, h - m), (-1, -1))):
            p.drawLine(QPointF(px, py), QPointF(px + sx * L, py))
            p.drawLine(QPointF(px, py), QPointF(px, py + sy * L))
        y = (self._scan / 100.0) * h
        p.setPen(QPen(_rgba(EYE_RGB, 24), 1))
        p.drawLine(QPointF(0, y), QPointF(w, y))
        p.setFont(QFont("Cascadia Mono", 9))
        p.setPen(_rgba(EYE_RGB, 150))
        p.drawText(QRectF(0, h - 30, w, 22), Qt.AlignmentFlag.AlignHCenter,
                   f"ORION · {self.state_name}")
