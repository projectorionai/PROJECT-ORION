"""
HoloHead — ORION's holographic head, rasterised in software.

What this is
------------
An animated human head drawn entirely with QPainter. It speaks with real mouth
shapes, its brows follow the phrase, its eyes make saccades between fixation
points, it blinks, and it shows ORION's state before he has said a word: eyes
away while thinking, on you while listening, lids low while asleep.

Why software rather than GL
---------------------------
ORION already has three faces and each is blocked by its renderer, not its art:

  * ``gui/face3d.py`` needs QtWebEngine and cannot coexist with the GL renderer;
  * ``gui/native_face.py`` needs a GL context that ``app.py`` does not provide —
    it never calls ``setDefaultFormat``, so the app gets a compatibility profile
    in which whole passes draw nothing at all, silently;
  * ``gui/face.py`` works, but is a voxel silhouette rather than a head.

Drawing in software removes the entire class of problem. There is no driver to
disagree with, so a gaming rig, a 2013 laptop, a VM and a remote desktop session
all produce the same picture, and the only performance question is one we can
measure directly (see ``last_paint_ms``).

How it draws
------------
Painter's algorithm over a backface-culled triangle list. Backface culling
removes roughly half the mesh before anything is sorted, and the survivors are
bucketed by shade so that QPainter is handed a few dozen filled paths per frame
instead of a thousand individual polygons — ORION has form here: an earlier
face cost 441 ms of QPainter per second of wall clock and starved the qasync
loop, so batching is not a micro-optimisation, it is the difference between a
face and a stutter.

Lighting is a key light, a fill, and a rim term that traces the silhouette. The
rim is what stops the cranium reading as a dark mass behind a lit face: a head
lit only from the front has a terminator that follows the geometry's crease,
and the eye reads that as two objects rather than one.

Colour comes from the caller, so the head retints with ORION's theme rather
than owning a palette of its own.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np

from ..constants import C
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import (QColor, QImage, QPainter, QPainterPath,  # noqa: F401
                         QPolygonF)
from PyQt6.QtWidgets import QWidget

from ..viseme import NEUTRAL, VISEMES
from .head_mesh import JAW_MAX, JAW_PIVOT, get_head_mesh

# ── animation constants, all in SECONDS ──────────────────────────────────────
# Timing in seconds rather than frames is deliberate. Frame-counted constants
# mean three different faces at 60, 30 and 20 fps, and a mouth closure shorter
# than one frame can vanish entirely.
_BLINK_MIN, _BLINK_MAX = 2.4, 6.5
_BLINK_DURATION = 0.13
_SACCADE_SPEAKING = (0.5, 1.8)
_SACCADE_IDLE = (1.1, 3.4)
_SACCADE_THINKING = (1.6, 3.6)
_BREATH_PERIOD = 4.6

#: Shade buckets. Enough that banding is invisible on a gradient, few enough
#: that a frame is a few dozen fill calls rather than a thousand.
_SHADE_BUCKETS = 18
#: Depth slabs for the batching sort. Tuned by rendering, not by reasoning: at
#: five slabs a background triangle wins against a foreground one along the jaw
#: and punches a visible notch in the silhouette whenever the mouth opens. That
#: artifact looked exactly like a torn jaw rig and cost a detour into the
#: weights before a slab sweep showed it was the painter, not the mesh. Fourteen
#: is clean; forty is also clean and 70% slower.
_DEPTH_SLABS = 14

#: Largest head radius actually rasterised, in pixels. Above this the head is
#: drawn once at this size and stretched up — see _paint_scaled.
#:
#: Chosen by sweeping it and measuring both sides of the trade, at 30 Hz:
#:
#:     cap     480 px widget    800 px         1000 px
#:     190     315 ms/s 1.1x    302 ms/s 1.8x  319 ms/s 2.2x
#:     240     299 ms/s 1.0x    359 ms/s 1.4x  367 ms/s 1.8x
#:     none    301 ms/s 1.0x    450 ms/s 1.0x  574 ms/s 1.0x
#:
#: 190 is cheaper but a 2.2x upscale is visibly soft, and this face is the
#: thing the user actually looks at. 240 keeps the upscale under 2x everywhere
#: sane while still cutting the worst case by a third, and below ~640 px it
#: does not bind at all — the head renders natively at its true size.
MAX_RENDER_RADIUS = 240.0


#: How far the head dips on a stressed syllable, in radians — about three
#: degrees. Small on purpose: a nod you notice as a nod is a nod that is too
#: big, and this fires several times a sentence.
_NOD_DEPTH = 0.055

#: How long one dip takes, down and back up. Shorter than this reads as a
#: twitch; longer and it is still returning when the next stress lands.
_NOD_DURATION = 0.26

#: No second nod inside this. Without it, a sustained loud passage fires one
#: every frame and the head buzzes.
_NOD_REFRACTORY = 0.36

#: A syllable is stressed when it rises this far above the running level of
#: the speech around it. A fixed threshold cannot work — it would nod through
#: every word of a loud sentence and through none of a quiet one.
_STRESS_RATIO = 1.5

#: ...but it still has to be audible. Below this, the "rise" is noise.
_STRESS_FLOOR = 0.03

#: How quickly the running level follows the speech. Long enough to average
#: across a word, short enough to track a change of delivery.
_ENVELOPE_TAU = 0.45


def headless() -> bool:
    """Whether this process has no screen to draw on.

    True on the VPS node and under an offscreen Qt platform. The face is not
    merely invisible there — it is unreachable, so every frame it rasterises
    is CPU spent on a picture that cannot exist. The animation is suspended
    entirely rather than drawn into a buffer nobody reads.

    The sensory state behind it — amplitude, emotion, viseme, what ORION is
    doing — keeps flowing over the bus either way, which is what the phone
    and the web client actually render from.
    """
    import os

    if os.getenv("ORION_HEADLESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return os.getenv("QT_QPA_PLATFORM", "").strip().lower() in {
        "offscreen", "minimal", "vnc"}


def face_thread_enabled() -> bool:
    """Whether the head rasterises on its own thread. On unless told otherwise.

    Under qasync the Qt thread IS the asyncio event loop, so the ~13 ms this
    head spends per frame is 13 ms the audio callback does not get. Moving the
    paint to a worker gives that back without touching a single vertex.

    ``ORION_FACE_THREAD=0`` puts it back on the GUI thread, which is the first
    thing to try if the face ever misbehaves.
    """
    import os

    if headless():
        return False          # nothing to show it to
    return os.getenv("ORION_FACE_THREAD", "1").strip().lower() not in {
        "0", "false", "no", "off"}


def _ease(dt: float, tau: float) -> float:
    """Frame-rate independent approach factor for an exponential ease.

    A raw ``x += (target - x) * 0.1`` moves at a speed that depends on how
    often it is called, so the same code animates at two different rates on a
    60 Hz and a 30 Hz display. This converts a time constant into the right
    per-step factor for whatever dt actually elapsed.
    """
    if tau <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


class HoloHead:
    """The renderer. Deliberately not a QWidget, so it can be hosted by any
    painter-based surface — a panel, an overlay orb, a deck tile.

    Lifecycle:
        head = HoloHead()
        head.step(dt, amplitude, speaking=..., state=...)   # once per tick
        head.paint(painter, cx, cy, radius, primary, accent, background)
    """

    def __init__(self) -> None:
        mesh = get_head_mesh()
        self._v0: np.ndarray = mesh["verts"]
        self._n0: np.ndarray = mesh["normals"]
        self._faces: np.ndarray = mesh["faces"]
        self._jaw_w: np.ndarray = mesh["jaw"]
        self._lip_w: np.ndarray = mesh["lips"]
        self._lip_c: np.ndarray = mesh["lip_centre"]
        self._fade: np.ndarray = mesh["fade"]
        self._landmarks: dict[str, list[int]] = mesh["landmarks"]
        top, bottom = mesh["span"]
        #: Vertical extent in head-local units — a host uses this to size the
        #: head to a band without guessing.
        self.SPAN = top - bottom

        # Pose
        self._yaw = 0.0
        self._pitch = 0.0
        # Stress nodding. `_nod_t` runs the dip; `_loud` is the running level
        # a syllable has to rise above to count as stressed.
        self._nod_t = -1.0           # negative = not nodding
        self._loud = 0.0
        self._loud_ready = False     # no running level until speech primes it
        self._nod_block = 0.0        # refractory timer
        self._nod_offset = 0.0       # the dip this frame, for tests to read
        self._sway_t = random.random() * 10.0

        # Mouth
        self._open = 0.0
        self._wide = 0.0
        self._closure = 0.0
        self._target_open = 0.0
        self._target_wide = 0.0
        self._target_closure = 0.0

        # Eyes and brows
        self._lids = 1.0
        self._blink = 0.0
        self._blink_at = 0.0
        self._gaze = [0.0, 0.0]
        self._gaze_target = [0.0, 0.0]
        self._gaze_at = 0.0
        self._gaze_bias = [0.0, 0.0]
        self._bias_target = [0.0, 0.0]
        self._bias_at = 0.0
        self._brow = 0.0
        self._brow_bias = 0.0

        self._t = 0.0
        self._speaking = False
        self._state = ""
        #: Wall-clock cost of the last paint, in milliseconds. Exposed so the
        #: animation budget can see this face rather than infer it.
        self.last_paint_ms = 0.0

    # ── drive ────────────────────────────────────────────────────────────────

    def set_viseme(self, openness: float, width: float, closure: float = 0.0) -> None:
        """Target mouth posture, 0..1 openness, -1..1 width, 0..1 closure.

        Closure is separate from openness because it must win: /m/, /b/ and /p/
        are made with the lips pressed shut, and no amount of loudness should
        open them.
        """
        self._target_open = max(0.0, min(1.0, float(openness)))
        self._target_wide = max(-1.0, min(1.0, float(width)))
        self._target_closure = max(0.0, min(1.0, float(closure)))

    def glance(self, dx: float, dy: float, hold: float = 1.1) -> None:
        """Look somewhere for a moment — used to acknowledge new content."""
        self._gaze_target = [max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy))]
        self._gaze_at = self._t + max(0.1, hold)

    def step(self, dt: float, amplitude: float = 0.0, *, speaking: bool = False,
             state: str = "") -> None:
        """Advance the animation by ``dt`` seconds."""
        dt = max(0.0, min(0.25, float(dt)))       # a long stall must not lurch
        self._t += dt
        t = self._t
        self._speaking = bool(speaking)
        self._state = (state or "").upper()
        live = self._speaking and amplitude > 0.002

        thinking = self._state in ("THINKING", "PROCESSING")
        asleep = self._state in ("SLEEPING", "STANDBY", "OFFLINE")

        # ── mouth ────────────────────────────────────────────────────────────
        if live:
            # A viseme target set by the caller leads; amplitude keeps it honest
            # when no transcript is available.
            if self._target_open <= 0.0 and self._target_closure <= 0.0:
                self._target_open = min(1.0, amplitude * 1.7)
        else:
            self._target_open *= 0.0
            self._target_wide *= 0.0
            self._target_closure = 0.0

        # Closing is faster than opening: a plosive is a tap, and a mouth that
        # closes as slowly as it opens smears every consonant into the next.
        self._open += (self._target_open - self._open) * _ease(
            dt, 0.030 if self._target_open > self._open else 0.022)
        self._wide += (self._target_wide - self._wide) * _ease(dt, 0.045)
        self._closure += (self._target_closure - self._closure) * _ease(dt, 0.018)

        # ── state expression ─────────────────────────────────────────────────
        if thinking:
            # People look away to think, and hold it. Re-rolling the direction
            # slowly reads as thought; re-rolling it fast reads as scanning.
            if t >= self._bias_at:
                self._bias_target = [
                    random.choice((-1.0, 1.0)) * random.uniform(0.45, 0.8),
                    random.uniform(0.2, 0.5),
                ]
                self._bias_at = t + random.uniform(1.4, 3.0)
            brow_bias, lid_target = -0.26, 0.93
        elif asleep:
            self._bias_target = [0.0, -0.25]
            brow_bias, lid_target = -0.05, 0.20
        else:
            self._bias_target = [0.0, 0.0]
            self._bias_at = 0.0
            brow_bias = 0.10 if self._state == "LISTENING" else 0.0
            lid_target = 1.0

        for i in (0, 1):
            self._gaze_bias[i] += (self._bias_target[i] - self._gaze_bias[i]) * _ease(dt, 0.55)
        self._lids += (lid_target - self._lids) * _ease(dt, 0.35)
        self._brow_bias += (brow_bias - self._brow_bias) * _ease(dt, 0.5)

        # ── gaze ─────────────────────────────────────────────────────────────
        if t >= self._gaze_at:
            lo, hi = (_SACCADE_SPEAKING if live
                      else _SACCADE_THINKING if thinking else _SACCADE_IDLE)
            reach = 0.85 if live else (0.3 if thinking else 0.5)
            self._gaze_target = [random.uniform(-1.0, 1.0) * reach,
                                 random.uniform(-1.0, 1.0) * reach * 0.5]
            self._gaze_at = t + random.uniform(lo, hi)
        # Saccades are near-instant jumps, not drifts.
        for i in (0, 1):
            self._gaze[i] += (self._gaze_target[i] - self._gaze[i]) * _ease(dt, 0.035)

        # ── blink ────────────────────────────────────────────────────────────
        if asleep:
            self._blink = 0.0
        elif t >= self._blink_at:
            if thinking and random.random() < 0.6:
                # Concentration suppresses blinking; skip this one.
                self._blink_at = t + random.uniform(_BLINK_MIN, _BLINK_MAX)
            else:
                self._blink = 1.0
                self._blink_at = t + random.uniform(_BLINK_MIN, _BLINK_MAX)
        if self._blink > 0.0:
            self._blink = max(0.0, self._blink - dt / _BLINK_DURATION)

        # ── brows and head ───────────────────────────────────────────────────
        # Brows ride the phrase, not the syllable — following amplitude per
        # frame produces an eyebrow that twitches on every consonant.
        phrase = 0.5 + 0.5 * math.sin(t * 1.1)
        target_brow = (0.35 * phrase + 0.3 * min(1.0, amplitude * 2.0)) if live else 0.0
        self._brow += (target_brow + self._brow_bias - self._brow) * _ease(dt, 0.28)

        self._sway_t += dt
        self._yaw = math.sin(self._sway_t * 0.31) * 0.09 + self._gaze[0] * 0.12
        self._pitch = (math.sin(self._sway_t * 0.23) * 0.05
                       + self._gaze[1] * 0.07
                       - (0.10 if asleep else 0.0))
        self._pitch += self._nod(dt, amplitude, live)

    def _nod(self, dt: float, amplitude: float, live: bool) -> float:
        """The head's dip on a stressed syllable.

        Subtracting a fraction of the current amplitude — which is what this
        used to do — is not a nod. It ties head position to the waveform, so
        the head rides every consonant and returns the instant the sound does:
        a jitter locked to loudness rather than a gesture. A nod *lags* the
        syllable that caused it and outlives it, which is what makes it read
        as emphasis instead of vibration.

        So a stress is detected as a rise above the running level of the
        speech around it, and it fires one bounded dip that plays out on its
        own clock. A refractory period stops a loud passage firing one every
        frame.
        """
        if not live:
            # Let any dip in flight finish rather than snapping upright the
            # moment speech stops — the last nod of a sentence is a real one.
            self._loud = 0.0
            self._loud_ready = False
            self._nod_block = 0.0
        else:
            level = max(0.0, float(amplitude))
            if not self._loud_ready:
                # The first audible frame has nothing to be louder THAN. Left
                # comparing against zero it always reads as a huge rise, so
                # every utterance opened with a nod on its first syllable
                # whether or not that syllable was stressed. Prime the level
                # and judge from the second frame on.
                self._loud = level
                self._loud_ready = True
                stressed = False
            else:
                stressed = (level > _STRESS_FLOOR
                            and level > self._loud * _STRESS_RATIO)
            # The level follows the speech AFTER the comparison, so a syllable
            # is measured against what came before it and not against itself.
            self._loud += (level - self._loud) * _ease(dt, _ENVELOPE_TAU)
            self._nod_block = max(0.0, self._nod_block - dt)
            if stressed and self._nod_block <= 0.0 and self._nod_t < 0.0:
                self._nod_t = 0.0
                self._nod_block = _NOD_REFRACTORY

        if self._nod_t < 0.0:
            self._nod_offset = 0.0
            return 0.0
        self._nod_t += dt
        if self._nod_t >= _NOD_DURATION:
            self._nod_t = -1.0
            self._nod_offset = 0.0
            return 0.0
        # One smooth dip and return: down through the middle of the window,
        # back to rest at the end of it, with no discontinuity at either edge.
        self._nod_offset = -_NOD_DEPTH * math.sin(
            math.pi * self._nod_t / _NOD_DURATION)
        return self._nod_offset

    # ── geometry ─────────────────────────────────────────────────────────────

    def _posed(self) -> tuple[np.ndarray, np.ndarray]:
        """Apply the rigs and the head pose; return (vertices, normals)."""
        v = self._v0.copy()

        # Jaw: rotate the weighted vertices about the mandible hinge.
        angle = JAW_MAX * self._open * (1.0 - 0.85 * self._closure)
        if angle > 1e-5:
            w = self._jaw_w
            rel = v - JAW_PIVOT
            c, s = math.cos(angle), math.sin(angle)
            ry = rel[:, 1] * c - rel[:, 2] * s
            rz = rel[:, 1] * s + rel[:, 2] * c
            v[:, 1] = JAW_PIVOT[1] + rel[:, 1] * (1 - w) + ry * w
            v[:, 2] = JAW_PIVOT[2] + rel[:, 2] * (1 - w) + rz * w

        # Lips: spread/round about the lip centre, and press shut on a closure.
        lw = self._lip_w
        if lw.any():
            dx = v[:, 0] - self._lip_c[0]
            dy = v[:, 1] - self._lip_c[1]
            # Spread has to be large enough to tell /i/ from /u/ at HUD size.
            spread = 1.0 + 0.52 * self._wide
            v[:, 0] = self._lip_c[0] + dx * (1.0 + (spread - 1.0) * lw)
            # A closure pulls the lips together vertically; rounding pushes them
            # forward, which is what makes /u/ read as /u/ and not as a small /a/.
            v[:, 1] = self._lip_c[1] + dy * (1.0 - (0.55 * self._closure) * lw)
            v[:, 2] += lw * max(0.0, -self._wide) * 0.055

        # Brows.
        raise_by = self._brow * 0.030
        for key in ("brow_l", "brow_r"):
            idx = self._landmarks[key]
            v[idx, 1] += raise_by

        # Eyelids: the upper lid sweeps down over the eye.
        lid = 1.0 - self._lids * (1.0 - self._blink)
        if lid > 0.01:
            for key in ("eye_l", "eye_r"):
                idx = self._landmarks[key]
                pts = v[idx]
                middle = float(pts[:, 1].mean())
                upper = pts[:, 1] > middle
                v[np.asarray(idx)[upper], 1] = (
                    pts[upper, 1] * (1.0 - lid) + middle * lid)

        # Breathing — a small vertical rise, plus the sway already in the pose.
        v[:, 1] += math.sin(self._t * 2.0 * math.pi / _BREATH_PERIOD) * 0.006

        # Head pose.
        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        rot = ry @ rx
        return v @ rot.T, self._n0 @ rot.T

    # ── paint ────────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, cx: float, cy: float, radius: float,
              primary: QColor, accent: QColor, background: QColor) -> None:
        """Draw the head centred on (cx, cy), scaled so ``radius`` is half its
        height. Colours come from the caller so the head follows the theme."""
        started = time.perf_counter()
        try:
            self._paint(painter, cx, cy, radius, primary, accent, background)
        finally:
            self.last_paint_ms = (time.perf_counter() - started) * 1000.0

    def _paint(self, painter: QPainter, cx: float, cy: float, radius: float,
               primary: QColor, accent: QColor, background: QColor) -> None:
        verts, normals = self._posed()

        sx = verts[:, 0] * radius + cx
        sy = -verts[:, 1] * radius + cy
        depth = verts[:, 2]

        faces = self._faces
        # Backface cull on the projected winding: half the mesh leaves before
        # anything is sorted or filled.
        ax, ay = sx[faces[:, 0]], sy[faces[:, 0]]
        bx, by = sx[faces[:, 1]], sy[faces[:, 1]]
        gx, gy = sx[faces[:, 2]], sy[faces[:, 2]]
        area = (bx - ax) * (gy - ay) - (gx - ax) * (by - ay)
        visible = area < 0.0
        if not visible.any():
            visible = area > 0.0                  # winding flipped; take the other side
        faces = faces[visible]
        if faces.size == 0:
            return

        # Lambert key + fill + rim. The rim term is keyed on how edge-on the
        # surface is, which traces the silhouette and keeps the cranium reading
        # as part of the same head rather than a shadow behind it.
        fn = normals[faces].mean(axis=1)
        lengths = np.linalg.norm(fn, axis=1, keepdims=True)
        fn = fn / np.maximum(lengths, 1e-9)
        key = np.array([-0.42, 0.52, 0.74])
        key /= np.linalg.norm(key)
        lambert = np.clip(fn @ key, 0.0, 1.0)
        fill = np.clip(fn @ np.array([0.62, -0.20, 0.42]), 0.0, 1.0)
        rim = np.power(1.0 - np.clip(np.abs(fn[:, 2]), 0.0, 1.0), 2.6)
        shade = 0.13 + 0.62 * lambert + 0.16 * fill + 0.42 * rim
        shade *= self._fade[faces].mean(axis=1)

        if self._speaking:
            shade *= 1.12

        buckets = np.clip((shade * _SHADE_BUCKETS).astype(np.int32),
                          0, _SHADE_BUCKETS - 1)

        # Batching, and why it is shaped like this.
        #
        # Sorting purely by depth and coalescing runs of equal shade sounds
        # right and is useless: shade varies rapidly along a depth ordering, so
        # almost every run has length one and QPainter is handed ~1000 separate
        # fills. Measured at 18 ms a frame, which at 30 fps is most of the
        # budget and exactly the kind of load that has starved ORION's qasync
        # loop before.
        #
        # Grouping purely by shade would fix the batching and break the
        # picture: the back of the skull would paint over the nose.
        #
        # So depth is quantised into a few coarse slabs and the sort key is
        # (slab, shade). Slabs preserve occlusion at the only scale where this
        # mesh actually self-occludes — the nose against the cheek — while
        # letting every triangle of one shade inside a slab become a single
        # fill. Distinct fills drop from ~1000 to well under a hundred.
        face_depth = depth[faces].mean(axis=1)
        lo, hi = float(face_depth.min()), float(face_depth.max())
        span = max(hi - lo, 1e-9)
        slabs = np.clip(((face_depth - lo) / span * _DEPTH_SLABS).astype(np.int32),
                        0, _DEPTH_SLABS - 1)
        keys = slabs * _SHADE_BUCKETS + buckets
        order = np.argsort(keys, kind="stable")
        ordered_keys = keys[order]
        # Boundaries between runs of equal key — one fill per run.
        cuts = np.flatnonzero(np.diff(ordered_keys)) + 1
        runs = np.split(order, cuts)

        painter.setPen(Qt.PenStyle.NoPen)
        deep = _mix(background, primary, 0.22)
        denominator = float(_SHADE_BUCKETS - 1)

        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        for run in runs:
            if run.size == 0:
                continue
            path = QPainterPath()
            # moveTo/lineTo on plain floats rather than building a QPolygonF
            # from three QPointF: same picture, four calls instead of four
            # object constructions per triangle.
            for index in run:
                a, b, c = f0[index], f1[index], f2[index]
                path.moveTo(sx[a], sy[a])
                path.lineTo(sx[b], sy[b])
                path.lineTo(sx[c], sy[c])
                path.closeSubpath()
            painter.setBrush(_mix(deep, primary, int(buckets[run[0]]) / denominator))
            painter.drawPath(path)

        self._paint_features(painter, sx, sy, radius, accent)

    def _ring(self, sx: np.ndarray, sy: np.ndarray, key: str) -> QPolygonF:
        idx = self._landmarks[key]
        return QPolygonF([QPointF(sx[i], sy[i]) for i in idx])

    def _paint_features(self, painter: QPainter, sx: np.ndarray, sy: np.ndarray,
                        radius: float, accent: QColor) -> None:
        """Eyes and mouth line — the parts that must read at HUD size."""
        openness = 1.0 - (1.0 - self._lids) - self._blink
        if openness > 0.06:
            glow = QColor(accent)
            glow.setAlpha(int(200 * min(1.0, openness)))
            painter.setBrush(glow)
            painter.setPen(Qt.PenStyle.NoPen)
            for key in ("eye_l", "eye_r"):
                idx = self._landmarks[key]
                px = float(np.mean(sx[idx]))
                py = float(np.mean(sy[idx]))
                r = radius * 0.040 * openness
                # The iris tracks the gaze within the socket.
                px += self._gaze[0] * radius * 0.016
                py -= self._gaze[1] * radius * 0.012
                painter.drawEllipse(QPointF(px, py), r, r * 1.05)

        # The inner lip ring, filled dark when the mouth is open, so an open
        # mouth reads as an aperture rather than a lighter patch of skin.
        if self._open > 0.05 and self._closure < 0.6:
            dark = QColor(0, 0, 0)
            dark.setAlpha(int(190 * min(1.0, self._open)))
            painter.setBrush(dark)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(self._ring(sx, sy, "lips_inner"))


class HoloHeadPanel(QWidget):
    """Hosts a :class:`HoloHead` and speaks ORION's face protocol.

    ``core_window._to_face`` calls ``set_state``, ``set_amplitude``,
    ``set_speaking``, ``set_viseme`` and ``apply_emotion`` on whichever face is
    installed, guarding each with ``hasattr``. Implementing that surface is all
    a new face has to do to be swappable with the voxel one.
    """

    def __init__(self, parent: QWidget | None = None, *, fps: int = 30) -> None:
        super().__init__(parent)
        self._head = HoloHead()
        self._amplitude = 0.0
        self._speaking = False
        self._state = "IDLE"
        self._last = time.perf_counter()
        #: Offscreen buffer for the scaled path; kept between frames.
        self._buffer: QImage | None = None

        # ORION's own colours, not the renderer's. These were #39b6ff and
        # #9fe8ff — an electric cyan that only ever got replaced if the window
        # happened to call set_palette_colours, so any other path (a test, the
        # overlay orb, a renderer attached later) produced a cyan ORION in a
        # crimson shell.
        self._primary = QColor(C.PRI)
        self._accent = QColor(C.ACCENT)
        self._background = QColor(C.BG)

        # `timer` is PUBLIC and part of the face protocol, not an implementation
        # detail: overlay mode reaches in and retimes the face
        # (core_window.enter_overlay_mode does `self.face.timer.setInterval(33)`),
        # and both other faces expose it under this name. Naming it `_timer`
        # made entering overlay mode raise AttributeError.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._interval = max(1, int(1000 / max(1, fps)))
        self.timer.start(self._interval)

        # Spend frames only where they show. Under qasync the Qt thread IS the
        # asyncio event loop, so a fixed 30 Hz here is not 6 ms of paint next
        # to ORION's brain, it is 6 ms *inside* it — 180 ms of every second
        # during which the audio callback cannot be serviced and the Live
        # socket cannot be read. That is the measured cause of his voice
        # stuttering alongside the GUI, and the reason animation_budget exists.
        #
        # Most of those frames are identical: every channel below EASES toward
        # a target, so once it has arrived it is recomputed to the same number
        # forever. The rate follows the animation instead of the clock.
        from .animation_budget import AnimationBudget

        self._budget = AnimationBudget(self.timer, active_hz=float(fps),
                                       idle_hz=12.0)

        # The rasteriser. Built here, started on first show — a thread spun up
        # during construction would be running before the panel has a size to
        # draw at. The head is HANDED OVER: from here on the worker owns it,
        # and every call below is queued rather than made directly, because
        # two threads touching the same vertex arrays is a race that shows up
        # as a torn face once an hour and never reproduces.
        self._pipeline: Any = None
        self._threaded = face_thread_enabled()
        self._blit_generation = -1
        if headless():
            # The null sink. Stopping the timer means no step, no paint and
            # no wakeups — on a node with no display that is the whole cost
            # of the face, and it is not a cost worth paying for a picture
            # nobody can see.
            self.timer.stop()
        self.setMinimumSize(160, 200)

    # ORION's face protocol ---------------------------------------------------

    def _ensure_pipeline(self) -> Any:
        """Start the render thread, once, when there is something to draw."""
        if not self._threaded or self._pipeline is not None:
            return self._pipeline
        try:
            from .face_pipeline import FaceRenderPipeline

            pipeline = FaceRenderPipeline(self._head,
                                          fps=float(self.timer.interval() and
                                                    1000.0 / self.timer.interval()
                                                    or 30.0))
            pipeline.set_palette(self._primary, self._accent, self._background)
            pipeline.start()
            self._pipeline = pipeline
        except Exception:
            # Any failure here falls back to painting on the GUI thread, which
            # is slower but correct. A face that does not appear is worse than
            # a face that costs the loop.
            self._threaded = False
            self._pipeline = None
        return self._pipeline

    def showEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        super().showEvent(event)
        self._ensure_pipeline()

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        self.stop_pipeline()
        super().closeEvent(event)

    def stop_pipeline(self) -> None:
        """Stop the render thread and wait briefly for it.

        Joined rather than abandoned: a thread holding a QImage while Qt tears
        down its graphics stack is how "destroyed but pending" becomes a crash
        on exit.
        """
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def _to_head(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Apply a change to the head — queued if the worker owns it."""
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.call(name, *args, **kwargs)
            return
        method = getattr(self._head, name, None)
        if method is not None:
            method(*args, **kwargs)

    def set_state(self, state: Any) -> None:
        self._state = str(state or "").upper()

    def set_amplitude(self, value: Any) -> None:
        try:
            self._amplitude = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            self._amplitude = 0.0

    def set_speaking(self, active: Any) -> None:
        self._speaking = bool(active)
        if not self._speaking:
            self._to_head("set_viseme", 0.0, 0.0, 0.0)

    def set_viseme(self, openness: Any, width: Any = 0.0, closure: Any = 0.0) -> None:
        """Accepts either a viseme ID or a raw posture.

        Two callers, two shapes, and the panel has to take both. The bus
        contract is ``{"viseme": "PP", "weight": 1.0}`` — an identifier and a
        strength, which is what a SAPI event or a text schedule produces. The
        audio path instead measures openness and width from the spectrum
        directly and has no identifier to give. Rejecting either one would
        leave the mouth still, so a string is looked up and a number is used
        as-is.
        """
        if isinstance(openness, str):
            posture = VISEMES.get(openness, NEUTRAL)
            try:
                weight = max(0.0, min(1.0, float(width) if width else 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            self._to_head("set_viseme",
                posture.open * weight,
                (posture.width - 0.5) * 2.0 * weight,
                # "PP" is p/b/m — lips pressed shut. It is the one posture the
                # spectrum can never supply, so it must survive the round trip.
                weight if openness == "PP" else 0.0,
            )
            return
        try:
            self._to_head("set_viseme", float(openness), float(width),
                          float(closure))
        except (TypeError, ValueError):
            pass

    def glance(self, dx: Any = 0.0, dy: Any = -0.65, hold: Any = 1.2) -> None:
        """Look somewhere for a moment — used to acknowledge new content.

        Forwarded rather than inherited. `core_window._to_face` guards every
        call with hasattr on the PANEL, so a method that exists only on the
        renderer inside it is silently a no-op: the call succeeds, nothing
        happens, and nothing reports it. That is exactly how this arrived —
        HoloHead.glance was written, wired to the banner signal, and never once
        moved the eyes.
        """
        try:
            self._to_head("glance", float(dx), float(dy), float(hold))
        except (TypeError, ValueError):
            pass

    def apply_emotion(self, name: Any, params: Any = None) -> None:
        """Retint from an emotion payload. Geometry is driven by speech and
        state; an emotion only moves the palette, which keeps the two systems
        from fighting over the same brow."""
        if not isinstance(params, dict):
            return
        for key in ("primary", "colour", "color"):
            value = params.get(key)
            if value:
                colour = QColor(str(value))
                if colour.isValid():
                    self._primary = colour
                break
        glow = params.get("accent") or params.get("glow")
        if glow:
            colour = QColor(str(glow))
            if colour.isValid():
                self._accent = colour

    def set_palette_colours(self, primary: str | QColor, accent: str | QColor,
                            background: str | QColor | None = None) -> None:
        for target, value in (("_primary", primary), ("_accent", accent),
                              ("_background", background)):
            if value is None:
                continue
            colour = QColor(value) if not isinstance(value, QColor) else value
            if colour.isValid():
                setattr(self, target, colour)
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.set_palette(self._primary, self._accent, self._background)

    # ── rendering ────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        # A hidden or minimised window must not pay for animation. Resuming
        # picks up from where it left off rather than snapping, because the
        # clock is reset on the way back in.
        if not self.isVisible():
            self._last = time.perf_counter()
            return
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        head = self._head
        pipeline = self._pipeline
        if pipeline is not None:
            # The worker steps AND paints; this thread only says what the
            # world looks like and asks for a blit.
            pipeline.set_step_kwargs(amplitude=self._amplitude,
                                     speaking=self._speaking,
                                     state=self._state)
            pipeline.set_geometry(*self._draw_geometry())
        else:
            head.step(dt, self._amplitude, speaking=self._speaking,
                      state=self._state)

        # Only the channels that carry MEANING — speech, expression, state.
        # The always-advancing ambient ones (breath, sway, gaze saccades) are
        # deliberately excluded: including them would mean "moving" on every
        # single frame and the budget could never engage at all.
        self._budget.note((
            head._open, head._wide, head._closure,
            head._lids, head._blink, head._brow,
            self._amplitude, float(self._speaking),
            float(hash(self._state) & 0xFFFF),
        ))
        # Painted either way. Unlike the voxel face there is no frame here that
        # is truly identical to the last — breathing and sway advance
        # continuously — so the saving comes from redrawing an almost-still
        # picture LESS OFTEN, never from freezing it. A frozen face reads as a
        # hang; a calm one reads as calm.
        self.update()

    def _draw_geometry(self) -> tuple[int, int, float, float, float]:
        """Where and how big the head is drawn, at the current widget size.

        The same numbers the synchronous path computes, factored out so the
        worker draws the head in exactly the place the GUI would have.
        """
        w, h = max(8, self.width()), max(8, self.height())
        radius = min(w * 0.42, h / max(self._head.SPAN + 0.15, 1e-6))
        if radius > MAX_RENDER_RADIUS:
            # Above this the synchronous path renders small and stretches up;
            # the worker does the same, so cost stops growing with the window.
            scale = MAX_RENDER_RADIUS / radius
            bw, bh = max(8, int(w * scale)), max(8, int(h * scale))
            return (bw, bh, bw / 2.0, bh * 0.47, MAX_RENDER_RADIUS)
        return (w, h, w / 2.0, h * 0.47, radius)

    def paintEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        pipeline = self._pipeline
        if pipeline is not None:
            frame, generation = pipeline.frames.take()
            if frame is not None:
                painter = QPainter(self)
                try:
                    painter.setRenderHint(
                        QPainter.RenderHint.SmoothPixmapTransform, True)
                    painter.drawImage(self.rect(), frame)
                finally:
                    painter.end()
                self._blit_generation = generation
                return
            # No frame yet — fall through and paint one synchronously so the
            # first moments after launch are not an empty rectangle.

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            w, h = self.width(), self.height()
            if w < 8 or h < 8:
                painter.fillRect(self.rect(), self._background)
                return                     # too small to divide by safely
            radius = min(w * 0.42, h / max(self._head.SPAN + 0.15, 1e-6))
            if radius <= MAX_RENDER_RADIUS:
                painter.fillRect(self.rect(), self._background)
                self._head.paint(painter, w / 2.0, h * 0.47, radius,
                                 self._primary, self._accent, self._background)
                return
            self._paint_scaled(painter, w, h, radius)
        finally:
            painter.end()

    def _paint_scaled(self, painter: QPainter, w: int, h: int,
                      radius: float) -> None:
        """Render at a capped resolution and stretch the result up.

        Cost here is fill-bound, so painting at the widget's own size makes it
        track AREA. Measured, speaking, one frame:

            240 px   7.2 ms       640 px  12.9 ms
            340 px   8.3 ms       800 px  15.7 ms      1000 px  20.3 ms

        At 30 Hz the last of those is 608 ms of every second, spent on the Qt
        thread — which under qasync IS the asyncio event loop. That is worse
        than the 441 ms/sec that made ORION's voice stutter in the first place.
        The voxel face does not have this problem because it bakes a cell list
        on resize and only re-tints; this one rasterises a lit mesh every frame.

        So above a certain size the head is drawn once into an offscreen image
        and blitted. The picture is a soft holographic glow with no text and no
        fine detail, so a smooth upscale is indistinguishable at normal viewing
        distance — and the cost stops growing with the window.

        The buffer is kept between frames; reallocating a megapixel image 30
        times a second would hand back everything this saves.
        """
        scale = MAX_RENDER_RADIUS / radius
        bw = max(8, int(w * scale))
        bh = max(8, int(h * scale))
        buffer = self._buffer
        if buffer is None or buffer.width() != bw or buffer.height() != bh:
            buffer = QImage(bw, bh, QImage.Format.Format_RGB32)
            self._buffer = buffer
        buffer.fill(self._background)

        inner = QPainter(buffer)
        try:
            inner.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._head.paint(inner, bw / 2.0, bh * 0.47,
                             MAX_RENDER_RADIUS, self._primary, self._accent,
                             self._background)
        finally:
            inner.end()

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(self.rect(), buffer)

    @property
    def last_paint_ms(self) -> float:
        return self._head.last_paint_ms


__all__ = ["HoloHead", "HoloHeadPanel"]
