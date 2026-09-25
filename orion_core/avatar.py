"""
ORION Avatar System — AvatarEngine / AvatarController / AvatarStateMachine /
AvatarAnimationManager.

The Mark XIV quantum-human rig (gui/face3d.py) is a superb *renderer*; this
module gives it a *nervous system*.  Everything here is pure Python with no Qt
imports, so the whole behaviour layer is unit-testable headless and reusable by
the phone PWA bridge:

    AvatarStateMachine    — eight behaviour states with eased cross-fades.
                            No abrupt changes: every transition blends over
                            ~0.6 s; NOTIFICATION and WARNING are transient
                            overlays that auto-decay back to the underlying
                            persistent state.
    AvatarAnimationManager— the "always alive" channels (Ultron principle:
                            nothing is ever still): breathing, a randomised
                            blink scheduler, amplitude-driven lip envelope
                            with attack/decay shaping, idle micro-sway from
                            mixed non-integer sine waves, and EMA-smoothed
                            head pose driven by webcam face tracking.
    AvatarEngine          — composes the two into one `tick(dt)` that yields
                            a complete render frame (state, morph, pose,
                            amplitude, emotion tint, label).
    AvatarController      — the thin binding that pushes engine frames into a
                            face widget (QuantumFace3D or the 2-D fallback)
                            through its existing public surface.  The caller
                            provides the timer; the controller never imports
                            Qt itself.

Face-tracking rules (from the Ultron hand-tracker study, applied to the head):
presence uses hysteresis so the avatar never flutters between tracked and
idle; targets are EMA-smoothed before they drive the rig; yaw is clamped to
±0.22 rad so the transparent head always stays readable (see face3d rendering
notes); and on any tracking-mode change the reference resets so the head never
jumps.  The follow is subtle by design — never aggressive, never unnatural.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable

__all__ = [
    "AVATAR_STATES", "AvatarAnimationManager", "AvatarController",
    "AvatarEngine", "AvatarStateMachine",
]

# ── the eight behaviour states ───────────────────────────────────────────────

AVATAR_STATES = (
    "idle", "listening", "thinking", "speaking",
    "researching", "executing", "notification", "warning",
)

# Transient overlay states revert to the underlying persistent state on expiry.
_TRANSIENT_HOLD = {"notification": 3.5, "warning": 6.0}

# How each ORION state renders on the Mark XIV rig (which understands
# IDLE / LISTENING / THINKING / SPEAKING) plus an emotion tint + label so all
# eight read distinctly on screen.
_RENDER_MAP: dict[str, dict[str, Any]] = {
    "idle":         {"js": "IDLE",      "label": "ORION"},
    "listening":    {"js": "LISTENING", "label": "ORION · LISTENING"},
    "thinking":     {"js": "THINKING",  "label": "ORION · THINKING"},
    "speaking":     {"js": "SPEAKING",  "label": "ORION · SPEAKING"},
    "researching":  {"js": "THINKING",  "label": "ORION · RESEARCHING",
                     "emotion": {"name": "focused", "intensity": 0.6}},
    "executing":    {"js": "THINKING",  "label": "ORION · EXECUTING",
                     "emotion": {"name": "determined", "intensity": 0.7}},
    "notification": {"js": "LISTENING", "label": "ORION · NOTICE",
                     "emotion": {"name": "alert", "intensity": 0.5}},
    "warning":      {"js": "THINKING",  "label": "ORION · WARNING",
                     "emotion": {"name": "alarmed", "intensity": 0.9}},
}

# Baseline energy (drives brightness/activity) per state — warning burns hot,
# idle breathes low.
_STATE_ENERGY = {
    "idle": 0.25, "listening": 0.45, "thinking": 0.6, "speaking": 0.7,
    "researching": 0.55, "executing": 0.65, "notification": 0.6, "warning": 0.9,
}

_TRANSITION_SECONDS = 0.6


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def canonical_state(raw: str) -> str:
    """Map any loose state string (bus, worker, tool) onto the eight states."""
    s = str(raw or "").strip().lower()
    if not s:
        return "idle"
    if s in AVATAR_STATES:
        return s
    for needle, state in (
        ("speak", "speaking"), ("talk", "speaking"),
        ("listen", "listening"), ("hear", "listening"),
        ("research", "researching"),
        ("think", "thinking"), ("process", "thinking"), ("reason", "thinking"),
        ("exec", "executing"), ("work", "executing"), ("tool", "executing"),
        ("warn", "warning"), ("alert", "warning"), ("fault", "warning"),
        ("notif", "notification"), ("notice", "notification"),
    ):
        if needle in s:
            return state
    return "idle"


class AvatarStateMachine:
    """Eight states, eased cross-fades, transient overlays that decay."""

    def __init__(self) -> None:
        self.state = "idle"           # transition target (the state being entered)
        self.previous = "idle"        # the state being left
        self.progress = 1.0           # 0 → 1 across the transition
        self._revert_to = "idle"      # persistent state under a transient overlay
        self._transient_left = 0.0
        self.history: list[str] = ["idle"]

    @property
    def settled(self) -> bool:
        return self.progress >= 1.0

    def request(self, state: str) -> bool:
        """Ask for a state; returns True when a transition begins."""
        state = canonical_state(state)
        if state == self.state:
            # Re-notifying refreshes a transient hold instead of re-blending.
            if state in _TRANSIENT_HOLD:
                self._transient_left = _TRANSIENT_HOLD[state]
            return False
        if state in _TRANSIENT_HOLD:
            # Remember what to fall back to (never another transient).
            self._revert_to = (self.state if self.state not in _TRANSIENT_HOLD
                               else self._revert_to)
            self._transient_left = _TRANSIENT_HOLD[state]
        elif self.state in _TRANSIENT_HOLD:
            # A persistent request during an overlay becomes the revert target
            # AND takes over now — the user outranks a notification.
            self._revert_to = state
            self._transient_left = 0.0
        self.previous = self.blended_dominant()
        self.state = state
        self.progress = 0.0
        self.history = (self.history + [state])[-16:]
        return True

    def tick(self, dt: float) -> None:
        if self.progress < 1.0:
            self.progress = min(1.0, self.progress + dt / _TRANSITION_SECONDS)
        if self.state in _TRANSIENT_HOLD and self.settled:
            self._transient_left -= dt
            if self._transient_left <= 0.0:
                self.request(self._revert_to)

    def weights(self) -> dict[str, float]:
        """Blend weights for the outgoing and incoming states (sum to 1)."""
        w = _smoothstep(self.progress)
        if w >= 1.0 or self.previous == self.state:
            return {self.state: 1.0}
        return {self.previous: 1.0 - w, self.state: w}

    def blended_dominant(self) -> str:
        weights = self.weights()
        return max(weights, key=lambda k: weights[k])


class AvatarAnimationManager:
    """Breathing, blinking, lip envelope, idle sway, smoothed head pose."""

    YAW_LIMIT = 0.22      # rad — beyond this the transparent head stops reading
    PITCH_LIMIT = 0.12

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.t = 0.0
        # Blink scheduler.
        self._next_blink = self.rng.uniform(2.5, 6.5)
        self._blink_left = 0.0
        self.blink = 0.0
        # Lip-sync envelope (attack fast, decay slower — speech feels natural).
        self.amplitude_target = 0.0
        self.mouth = 0.0
        # Spectral mouth-shape targets (Mark X.14): band-energy ratios from
        # AudioPlaybackThread._spectral_bands, so shape reflects what's being
        # said instead of only how loud it is. Same attack/decay smoothing as
        # amplitude, for the same reason (frame-to-frame jitter reads as
        # twitching, not speech).
        self.spectrum_target = (0.0, 0.0, 0.0)   # (low, mid, high)
        self.spectrum_low = 0.0
        self.spectrum_mid = 0.0
        self.spectrum_high = 0.0
        # Face-tracking pose with presence hysteresis (Ultron pinch rule).
        self._pose_target = (0.0, 0.0)
        self._seen_frames = 0
        self._lost_frames = 0
        self.tracking = False
        self.yaw = 0.0
        self.pitch = 0.0

    # — inputs —
    def set_amplitude(self, value: float) -> None:
        self.amplitude_target = max(0.0, min(1.0, float(value)))

    def set_spectrum(self, low: float, mid: float, high: float) -> None:
        clamp = lambda v: max(0.0, min(1.0, float(v)))
        self.spectrum_target = (clamp(low), clamp(mid), clamp(high))

    def observe_face(self, x: float, y: float, present: bool) -> None:
        """Feed one tracker sample: x/y are the face centre in [-1, 1]."""
        if present:
            self._seen_frames += 1
            self._lost_frames = 0
            if self._seen_frames >= 3 and not self.tracking:
                self.tracking = True     # engage only after 3 solid frames
            if self.tracking:
                # Mirror x so ORION turns TOWARD the user, scaled subtle.
                self._pose_target = (
                    max(-self.YAW_LIMIT, min(self.YAW_LIMIT,
                                             -float(x) * self.YAW_LIMIT)),
                    max(-self.PITCH_LIMIT, min(self.PITCH_LIMIT,
                                               float(y) * self.PITCH_LIMIT)),
                )
        else:
            self._lost_frames += 1
            self._seen_frames = 0
            if self._lost_frames >= 8 and self.tracking:
                self.tracking = False    # let go gently, drift home
                self._pose_target = (0.0, 0.0)

    # — the tick —
    def tick(self, dt: float, state: str) -> dict[str, float]:
        dt = max(0.0, min(0.25, float(dt)))
        self.t += dt

        # Breathing: calmer at idle, quicker under load or alarm.
        rate = {"warning": 2.6, "executing": 3.2, "speaking": 3.4}.get(state, 4.2)
        breath = 0.5 + 0.5 * math.sin(self.t * (2.0 * math.pi / rate))

        # Blink scheduler — suppressed mid-warning flash, doubled at idle rest.
        self._next_blink -= dt
        if self._blink_left > 0.0:
            self._blink_left -= dt
            self.blink = _smoothstep(min(1.0, self._blink_left / 0.12))
        elif self._next_blink <= 0.0:
            self._blink_left = 0.12
            self._next_blink = self.rng.uniform(2.5, 6.5)
        else:
            self.blink = 0.0

        # Lip envelope: attack ~80 ms, decay ~180 ms.
        tau = 0.08 if self.amplitude_target > self.mouth else 0.18
        self.mouth += (self.amplitude_target - self.mouth) * min(1.0, dt / tau)
        if state != "speaking":
            self.mouth *= max(0.0, 1.0 - dt / 0.15)   # close quickly off-mic

        # Spectral mouth shape: same attack/decay shape as the lip envelope.
        for attr, target in zip(
            ("spectrum_low", "spectrum_mid", "spectrum_high"), self.spectrum_target
        ):
            current = getattr(self, attr)
            band_tau = 0.08 if target > current else 0.18
            setattr(self, attr, current + (target - current) * min(1.0, dt / band_tau))
        if state != "speaking":
            decay = max(0.0, 1.0 - dt / 0.15)
            self.spectrum_low *= decay
            self.spectrum_mid *= decay
            self.spectrum_high *= decay

        # Idle micro-sway: mixed non-integer sines so the rhythm never repeats.
        sway_y = (math.sin(self.t * 0.41) * 0.6 + math.sin(self.t * 0.173) * 0.4) * 0.03
        sway_p = (math.sin(self.t * 0.31 + 1.7) * 0.5 + math.sin(self.t * 0.117) * 0.5) * 0.018

        # Pose: EMA toward the target (≈250 ms settle) + sway on top.
        alpha = min(1.0, dt / 0.25)
        self.yaw += (self._pose_target[0] - self.yaw) * alpha
        self.pitch += (self._pose_target[1] - self.pitch) * alpha

        energy = _STATE_ENERGY.get(state, 0.4)
        return {
            "breath": breath,
            "blink": self.blink,
            "mouth": self.mouth,
            "yaw": self.yaw + sway_y,
            "pitch": self.pitch + sway_p,
            "energy": min(1.0, energy + self.mouth * 0.3),
            "tracking": 1.0 if self.tracking else 0.0,
            "spectrum_low": self.spectrum_low,
            "spectrum_mid": self.spectrum_mid,
            "spectrum_high": self.spectrum_high,
        }


class AvatarEngine:
    """State machine + animation channels → one render frame per tick."""

    # Raw state strings that collapse the quantum human back to the orb —
    # mirrors QuantumFace3D._ORB_STATES so behaviour is unchanged when the
    # controller takes over the rig.
    _DORMANT_MARKERS = ("standby", "initialising", "initializing", "offline",
                        "shutting")

    def __init__(self, rng: random.Random | None = None) -> None:
        self.machine = AvatarStateMachine()
        self.animation = AvatarAnimationManager(rng)
        self.speaking = False
        self.dormant = True            # boots as the orb, like the rig itself
        self._morph = 0.0
        self._emotion_override: dict[str, Any] | None = None

    # — inputs (all loose-typed, bus-friendly) —
    def request_state(self, state: str) -> None:
        raw = str(state or "").lower()
        self.dormant = any(m in raw for m in self._DORMANT_MARKERS)
        self.machine.request(state)

    def set_speaking(self, active: bool) -> None:
        self.speaking = bool(active)
        if active:
            self.dormant = False
        self.machine.request("speaking" if active else "idle")

    def set_amplitude(self, value: float) -> None:
        self.animation.set_amplitude(value)
        if float(value) > 0.02 and not self.speaking:
            self.speaking = True
            self.machine.request("speaking")

    def set_spectrum(self, low: float, mid: float, high: float) -> None:
        self.animation.set_spectrum(low, mid, high)

    def notify(self, warning: bool = False) -> None:
        self.machine.request("warning" if warning else "notification")

    def apply_emotion(self, name: str, params: Any = None) -> None:
        self._emotion_override = ({"name": str(name), **params}
                                  if isinstance(params, dict)
                                  else {"name": str(name)})

    def observe_face(self, x: float, y: float, present: bool) -> None:
        self.animation.observe_face(x, y, present)

    # — the frame —
    def tick(self, dt: float) -> dict[str, Any]:
        self.machine.tick(dt)
        state = self.machine.blended_dominant()
        channels = self.animation.tick(dt, state)
        render = _RENDER_MAP[state]
        emotion = self._emotion_override or render.get("emotion")
        # Morph: how "human" the rig is (1 = full head, 0 = dormant orb).
        # Eased so materialisation/collapse always scrubs smoothly.
        target = 0.0 if self.dormant else 1.0
        self._morph += (target - self._morph) * min(1.0, dt / 0.5)
        morph = self._morph if abs(self._morph - target) > 0.005 else target
        self._morph = morph
        return {
            "state": state,
            "js_state": render["js"],
            "label": render["label"],
            "morph": round(morph, 3),
            "amplitude": round(channels["mouth"], 3),
            "yaw": round(channels["yaw"], 4),
            "pitch": round(channels["pitch"], 4),
            "breath": round(channels["breath"], 3),
            "blink": round(channels["blink"], 3),
            "energy": round(channels["energy"], 3),
            "tracking": channels["tracking"] >= 1.0,
            "settled": self.machine.settled,
            "spectrum_low": round(channels["spectrum_low"], 3),
            "spectrum_mid": round(channels["spectrum_mid"], 3),
            "spectrum_high": round(channels["spectrum_high"], 3),
        }


class AvatarController:
    """Push engine frames into a face widget through its public surface.

    The widget contract (QuantumFace3D and HologramFace both satisfy it, in
    whole or part — every call is guarded): ``set_state`` / ``set_amplitude``
    / ``apply_emotion`` and optionally ``set_pose``.  The caller owns the
    timer and calls :meth:`tick` at ~30 Hz while the face page is visible.
    """

    def __init__(self, engine: AvatarEngine | None = None,
                 face: Any = None,
                 on_frame: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.engine = engine or AvatarEngine()
        self.face = face
        self.on_frame = on_frame
        self._last = time.monotonic()
        self._pushed_state = ""
        self._pushed_label = ""
        self._pushed_morph = -1.0

    def attach(self, face: Any) -> None:
        self.face = face
        self._pushed_state = ""    # force a full re-push on next tick

    # Bus-facing slots (loose signatures so they connect to pyqtSignals).
    def on_state(self, state: str) -> None:
        self.engine.request_state(state)

    def on_speaking(self, active: bool) -> None:
        self.engine.set_speaking(active)

    def on_amplitude(self, value: float) -> None:
        self.engine.set_amplitude(value)

    def on_voice_spectrum(self, bands: Any) -> None:
        if isinstance(bands, dict):
            self.engine.set_spectrum(
                bands.get("low", 0.0), bands.get("mid", 0.0), bands.get("high", 0.0))

    def on_emotion(self, name: str, params: Any = None) -> None:
        self.engine.apply_emotion(name, params)

    def on_face_sample(self, payload: Any) -> None:
        if isinstance(payload, dict):
            self.engine.observe_face(
                float(payload.get("x") or 0.0), float(payload.get("y") or 0.0),
                bool(payload.get("present")))

    def tick(self, dt: float | None = None) -> dict[str, Any]:
        now = time.monotonic()
        if dt is None:
            dt = now - self._last
        self._last = now
        frame = self.engine.tick(dt)
        self._push(frame)
        if self.on_frame is not None:
            try:
                self.on_frame(frame)
            except Exception:
                pass
        return frame

    def _push(self, frame: dict[str, Any]) -> None:
        face = self.face
        if face is None:
            return
        try:
            if frame["js_state"] != self._pushed_state:
                self._pushed_state = frame["js_state"]
                if hasattr(face, "set_state"):
                    face.set_state(frame["js_state"])
            if hasattr(face, "set_amplitude"):
                face.set_amplitude(frame["amplitude"])
            if hasattr(face, "set_spectrum"):
                face.set_spectrum(
                    frame["spectrum_low"], frame["spectrum_mid"], frame["spectrum_high"])
            if hasattr(face, "set_pose"):
                face.set_pose(frame["yaw"], frame["pitch"])
            if (abs(frame["morph"] - self._pushed_morph) > 0.004
                    and hasattr(face, "set_morph")):
                self._pushed_morph = frame["morph"]
                face.set_morph(frame["morph"])
            if frame["label"] != self._pushed_label and hasattr(face, "set_label"):
                self._pushed_label = frame["label"]
                face.set_label(frame["label"])
        except Exception:
            pass   # a rendering fault must never take down the behaviour loop

    def status(self) -> dict[str, Any]:
        return {
            "state": self.engine.machine.state,
            "settled": self.engine.machine.settled,
            "tracking": self.engine.animation.tracking,
            "history": list(self.engine.machine.history[-8:]),
        }
