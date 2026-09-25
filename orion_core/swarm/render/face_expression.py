"""
face_expression.py — what ORION's face is doing, as shader uniforms
(Mark XXIII).

The face geometry is emitted once and never regenerated. Everything the
avatar DOES — blinking, speaking, breathing, looking, reacting — is expressed
as a small set of scalars fed to the face shader each frame, which displaces
points by their feature tag. That is the whole reason the migration off
WebEngine is cheap: animation costs a handful of uniform writes, not a
rebuild.

This module owns the behaviour and none of the drawing, so the entire
personality of the face is unit-testable without a GPU.

The design rule throughout is RESTRAINT. An avatar that blinks on a metronome,
snaps between expressions, or flaps its mouth at full travel on every syllable
reads as a cartoon. Believable AGI behaviour is mostly stillness with small,
well-timed, asymmetric motion — so blink intervals are irregular, expression
changes are eased rather than switched, and speech drives a smoothed envelope
rather than raw amplitude.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from ..cognition_state import Mode

# Blink timing. Humans blink every 2-10 s; the spread matters more than the
# average, because a fixed interval is instantly readable as machinery.
BLINK_MIN_GAP_S = 2.4
BLINK_MAX_GAP_S = 7.5
BLINK_DURATION_S = 0.13

# Speech envelope smoothing. Raw amplitude is far too jittery to drive a jaw:
# following it directly produces a chattering mouth. Attack is faster than
# release so onsets are crisp while the mouth closes naturally.
#
# These are FRACTION-PER-SECOND in _approach's exponential form, which is
# easy to misjudge: an earlier 0.12 release meant only 12% of the remaining
# distance closed per second — a time constant near eight seconds, leaving
# ORION's mouth hanging open long after he had stopped speaking. At these
# values the mouth opens in ~0.12 s and closes in ~0.22 s.
SPEECH_ATTACK = 0.9999
SPEECH_RELEASE = 0.99
# The jaw follows the mouth but slightly slower — that lag is most of what
# gives the face a sense of mass rather than reading as a puppet.
JAW_FOLLOW = 0.97
MOUTH_MAX_OPEN = 1.0

# Idle motion. Deliberately small — this is the difference between "alive"
# and "restless".
BREATH_RATE_HZ = 0.18
BREATH_DEPTH = 0.055
DRIFT_RATE_HZ = 0.07
DRIFT_DEPTH = 0.035


@dataclass(frozen=True, slots=True)
class Expression:
    """The complete uniform set for one frame.

    All values are normalised so the shader never needs to know about world
    units, and clamped so a bad input can never turn the face inside out.
    """

    blink: float          # 0 open .. 1 fully closed
    mouth_open: float     # 0 closed .. 1 wide
    jaw_drop: float       # follows mouth, lags slightly — mass
    brow_raise: float     # -1 furrowed .. +1 raised
    gaze_x: float         # -1 left .. +1 right
    gaze_y: float         # -1 down .. +1 up
    breath: float         # -1 .. +1, slow scale oscillation
    drift_x: float        # subtle head sway
    drift_y: float
    energy: float         # overall emissive intensity multiplier
    focus: float          # 0 relaxed .. 1 concentrated (narrows the eyes)

    def as_uniforms(self) -> dict[str, float]:
        return {
            "u_blink": self.blink,
            "u_mouth": self.mouth_open,
            "u_jaw": self.jaw_drop,
            "u_brow": self.brow_raise,
            "u_gaze_x": self.gaze_x,
            "u_gaze_y": self.gaze_y,
            "u_breath": self.breath,
            "u_drift_x": self.drift_x,
            "u_drift_y": self.drift_y,
            "u_energy": self.energy,
            "u_focus": self.focus,
        }


# Per-mode resting posture: (brow, focus, energy). Small numbers on purpose —
# ORION should read as composed and intelligent, not expressive.
_MODE_POSTURE: dict[Mode, tuple[float, float, float]] = {
    Mode.IDLE:        (0.00, 0.10, 0.85),
    Mode.LISTENING:   (0.18, 0.30, 1.00),   # brows slightly up: attentive
    Mode.THINKING:    (-0.22, 0.75, 1.15),  # drawn in, concentrated
    Mode.RESEARCHING: (-0.12, 0.65, 1.10),
    Mode.CODING:      (-0.18, 0.70, 1.08),
    Mode.SPEAKING:    (0.10, 0.25, 1.25),
    Mode.EXECUTING:   (-0.08, 0.60, 1.30),
    Mode.STANDBY:     (-0.05, 0.00, 0.40),  # eyes soft, glow well down
    Mode.ERROR:       (-0.35, 0.55, 1.20),
}

# How fast posture eases toward its target, as fraction-per-second. Slow
# enough that a mode change is a transition rather than a cut.
_POSTURE_EASE = 0.92


def _approach(current: float, target: float, dt: float, rate: float) -> float:
    """Frame-rate independent easing — the same exponential form the camera
    uses, so motion feels identical at 30 and 144 fps."""
    if dt <= 0.0:
        return current
    t = 1.0 - (1.0 - rate) ** dt
    return current + (target - current) * min(1.0, max(0.0, t))


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class FaceAnimator:
    """Turns real runtime state into a facial performance.

    Fed the speech engine's amplitude and ORION's cognition mode; produces an
    Expression per frame. Holds all timing state, so it is the single place
    that decides how the avatar behaves.
    """

    __slots__ = ("_blink", "_next_blink_at", "_blink_started_at", "_mouth",
                 "_jaw", "_brow", "_focus", "_energy", "_gaze_x", "_gaze_y",
                 "_gaze_target", "_next_gaze_at", "_elapsed", "_rng", "_mode",
                 "_spectrum")

    def __init__(self, seed: int = 0x4F52) -> None:
        # Seeded so a test run is reproducible; the irregularity that matters
        # is within a session, not between them.
        self._rng = random.Random(seed)
        self._elapsed = 0.0
        self._blink = 0.0
        self._blink_started_at = -1.0
        self._next_blink_at = self._rng.uniform(BLINK_MIN_GAP_S, BLINK_MAX_GAP_S)
        self._mouth = 0.0
        self._jaw = 0.0
        self._brow = 0.0
        self._focus = 0.0
        self._energy = 0.85
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._gaze_target = (0.0, 0.0)
        self._next_gaze_at = 1.5
        self._mode = Mode.IDLE
        # Latest voice spectral profile (low/mid/high energy ratios) — drives
        # ARTICULATION so the mouth forms syllables rather than just pulsing to
        # loudness. All-zero == no spectral info, and the mouth falls back to
        # pure amplitude, so this is backward-compatible.
        self._spectrum = (0.0, 0.0, 0.0)

    # ── input ────────────────────────────────────────────────────────────────

    def set_mode(self, mode: Mode) -> None:
        self._mode = mode if isinstance(mode, Mode) else Mode.IDLE

    def set_spectrum(self, bands: Any) -> None:
        """Feed the current voice spectral profile (a {low,mid,high} mapping).

        Vowels put energy in the mid band and open the mouth fully; sibilants
        ('s', 'f') sit in the high band and keep it tighter — that contrast is
        what turns a pulsing hole into something that looks like speech."""
        if not isinstance(bands, dict):
            return
        self._spectrum = (max(0.0, float(bands.get("low", 0.0) or 0.0)),
                          max(0.0, float(bands.get("mid", 0.0) or 0.0)),
                          max(0.0, float(bands.get("high", 0.0) or 0.0)))

    def _articulation(self) -> float:
        """A mouth-openness multiplier from the spectral shape — wide on vowels,
        tighter on sibilants. Returns 1.0 (neutral) when there's no spectrum."""
        low, mid, high = self._spectrum
        total = low + mid + high
        if total < 1e-4:
            return 1.0
        vowel = mid / total          # open vowels dominate the mid band
        sibilant = high / total      # 's'/'f'/'sh' dominate the high band
        return max(0.4, min(1.3, 0.72 + 0.85 * vowel - 0.45 * sibilant))

    def look_at(self, x: float, y: float) -> None:
        """Point the gaze deliberately — used by face tracking, so ORION
        actually looks at the person in front of him."""
        self._gaze_target = (_clamp(x), _clamp(y))
        # Push the idle saccade back so a deliberate look is not immediately
        # overridden by a wander.
        self._next_gaze_at = self._elapsed + 2.5

    # ── the frame ────────────────────────────────────────────────────────────

    def update(self, dt: float, amplitude: float = 0.0) -> Expression:
        """Advance by *dt* seconds against the current speech *amplitude*."""
        dt = max(0.0, min(0.25, float(dt)))
        self._elapsed += dt

        self._update_blink()
        self._update_gaze(dt)

        # Speech: a smoothed envelope, not raw amplitude. Attack faster than
        # release so onsets are crisp and the mouth closes naturally.
        level = _clamp(float(amplitude), 0.0, 1.0)
        # Articulation: the spectral shape decides how far a syllable opens the
        # mouth — a loud vowel opens it fully, a loud 's' barely at all.
        target = _clamp(level * MOUTH_MAX_OPEN * self._articulation(), 0.0, 1.0)
        rate = SPEECH_ATTACK if target > self._mouth else SPEECH_RELEASE
        self._mouth = _approach(self._mouth, target, dt, rate)
        # The jaw trails the mouth: giving it mass is most of what stops the
        # face reading as a puppet.
        self._jaw = _approach(self._jaw, self._mouth, dt, JAW_FOLLOW)

        brow_target, focus_target, energy_target = _MODE_POSTURE.get(
            self._mode, _MODE_POSTURE[Mode.IDLE])
        # Speaking lifts the brows a little on louder syllables — the single
        # cheapest cue that makes speech look intentional rather than mimed.
        brow_target += self._mouth * 0.18
        self._brow = _approach(self._brow, brow_target, dt, _POSTURE_EASE)
        self._focus = _approach(self._focus, focus_target, dt, _POSTURE_EASE)
        self._energy = _approach(self._energy, energy_target, dt, _POSTURE_EASE)

        breath = math.sin(self._elapsed * BREATH_RATE_HZ * math.tau) * BREATH_DEPTH
        # Two different rates so the sway never traces a repeating figure.
        drift_x = math.sin(self._elapsed * DRIFT_RATE_HZ * math.tau) * DRIFT_DEPTH
        drift_y = math.cos(self._elapsed * DRIFT_RATE_HZ * 0.73 * math.tau) * DRIFT_DEPTH * 0.6

        return Expression(
            blink=self._blink,
            mouth_open=_clamp(self._mouth, 0.0, 1.0),
            jaw_drop=_clamp(self._jaw, 0.0, 1.0),
            brow_raise=_clamp(self._brow),
            gaze_x=_clamp(self._gaze_x),
            gaze_y=_clamp(self._gaze_y),
            breath=breath,
            drift_x=drift_x,
            drift_y=drift_y,
            energy=max(0.0, self._energy),
            focus=_clamp(self._focus, 0.0, 1.0),
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _update_blink(self) -> None:
        """Irregular blinking. A fixed interval is instantly readable as
        machinery, so each gap is drawn fresh from a wide range."""
        if self._blink_started_at >= 0.0:
            progress = (self._elapsed - self._blink_started_at) / BLINK_DURATION_S
            if progress >= 1.0:
                self._blink = 0.0
                self._blink_started_at = -1.0
                self._next_blink_at = self._elapsed + self._rng.uniform(
                    BLINK_MIN_GAP_S, BLINK_MAX_GAP_S)
            else:
                # Close fast, open slightly slower — a symmetric blink looks
                # mechanical.
                self._blink = (math.sin(progress * math.pi) ** 0.7)
            return
        if self._elapsed >= self._next_blink_at:
            self._blink_started_at = self._elapsed
            self._blink = 0.0

    def _update_gaze(self, dt: float) -> None:
        """Small idle saccades toward nearby points, then settling.

        Eyes that never move look dead; eyes that sweep constantly look
        anxious. Short holds with small movements read as thought."""
        if self._elapsed >= self._next_gaze_at:
            self._gaze_target = (self._rng.uniform(-0.35, 0.35),
                                 self._rng.uniform(-0.22, 0.22))
            self._next_gaze_at = self._elapsed + self._rng.uniform(1.2, 4.0)
        self._gaze_x = _approach(self._gaze_x, self._gaze_target[0], dt, 0.97)
        self._gaze_y = _approach(self._gaze_y, self._gaze_target[1], dt, 0.97)
