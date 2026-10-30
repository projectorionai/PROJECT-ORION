"""
ScreenFace — ORION as a robot display, where the emotion lives in the eyes.

Why this face exists
--------------------
Every face ORION wore before this one aimed at a person: a voxel silhouette,
a sculpted head, a quantum human in Three.js. The better each of them got, the
closer it came to a real face without arriving — and "almost a person" is the
exact place the uncanny valley lives. The owner found the realistic face
unsettling, and so would a visitor leaning over the desk.

A screen face never pretends. It is the Cozmo / EVE lineage: a dark glass
visor, two glowing eyes, a small mouth. It cannot be "nearly right" because it
is not trying to be a human at all, and it carries more emotion per pixel than
any of the rendered heads did — a happy ^ ^, a sleepy line with a floating z,
a curious tilt, all readable from across the room at 120 px.

Why it is cheap
---------------
Pure QPainter on a QImage. No numpy, no mesh, no QtWebEngine, no Chromium
process, no OpenGL context — so it is on screen the moment the window is, it
works with the network unplugged, and it renders identically on every machine
ORION runs on.

Most of the picture never changes: the background, the visor glass, the HUD
brackets, the scanlines and the rim. Those are baked into two cached images,
rebuilt only on resize or recolour, and a frame is one opaque blit, a handful
of eye/mouth paths and one translucent blit on top. The rest of the cost is
governed exactly like the other faces: it renders on the face pipeline's
worker thread (face_pipeline.py) so the qasync loop — which IS the Qt thread —
only ever blits, and the frame rate follows the animation (animation_budget)
rather than the clock.

How expressions work
--------------------
An expression is ~25 named scalars: eye size, width, roundness, three upper
lid cuts, the happy crescent, mouth curve, blush, thinking dots and so on.
ORION's *state* chooses a base set (listening is bigger, brighter and leans
in; thinking narrows and glances up), his *emotion* adds its deviation on top
(happy adds the crescent and the blush), and every channel eases toward its
target with its own time constant. Nothing ever snaps: a change of state is a
blend, a blink is a blink, and the whole face can be anywhere between two
expressions at once without a special case.

Mark XXXVII: the face got more to say. Every channel now eases through TWO
lags in series (an S-curve: no lurch when a feeling starts, none when it
settles), a feeling lands with a small overshoot (a quick widen on surprise, a
little sink on sadness), the lower lids rise into a squint when he is amused,
one brow lifts when he is curious, happy eyes twinkle, a thinking head bobs
softly and - every ten seconds or so - an idle face does something small (a
perk, a head tilt, a flicker of a smile) so it never looks frozen. All of it is
scalar arithmetic in step() and a couple of tiny polygons in paint(): the
frame cost is unchanged to within noise.

Colour comes from the caller — ``set_palette_colours`` — so the face is
ORION's crimson by default and whatever accent the owner picks in Preferences
otherwise. The only other colour on it is the silver highlight, which is
ORION's accent token, not a colour of this module's own.
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (QColor, QFont, QFontMetricsF, QImage, QLinearGradient,
                         QPainter, QPainterPath, QPen, QPolygonF,
                         QRadialGradient)
from PyQt6.QtWidgets import QWidget

from ..constants import C
from ..viseme import NEUTRAL as _REST_MOUTH
from ..viseme import VISEMES

# ── timing, all in SECONDS ───────────────────────────────────────────────────
# Seconds rather than frames for the reason holo_head.py gives: frame-counted
# constants animate at three different speeds at 12, 30 and 60 fps, and the
# budget moves this face between 12 and 30 all the time.

#: Natural blink spacing. Jittered uniformly inside the range, with an
#: occasional double blink — a metronome blink is what makes a face read as a
#: machine pretending, which is the one thing this face is not.
_BLINK_MIN, _BLINK_MAX = 3.0, 6.0
_DOUBLE_BLINK_CHANCE = 0.16
#: Closing is quicker than opening, as on a real lid: 70 ms down, a 30 ms
#: hold, 110 ms up. Long enough to be seen at 30 fps, short enough to be a
#: blink and not a wink.
_BLINK_CLOSE, _BLINK_HOLD, _BLINK_OPEN = 0.07, 0.03, 0.11

#: Saccade spacing (min, max) while idle, listening and speaking.
_SACCADE_IDLE = (1.2, 3.6)
_SACCADE_LISTENING = (1.8, 4.5)
_SACCADE_SPEAKING = (0.9, 2.4)
_SACCADE_THINKING = (1.6, 3.2)

_BREATH_PERIOD = 4.4

#: Idle variety: how long (min, max, seconds) between the small things an idle
#: or listening face does on its own, so it never looks frozen. Skipped under
#: Reduced motion - those are exactly the idle flourishes it asks to calm.
_VARIETY_EVERY = (6.0, 14.0)
_VARIETY_MIN_AMBIENT = 0.5
#: A feeling arriving overshoots for this long (seconds, exponential decay).
_ONSET_TAU = 0.2
#: The thinking head bob and the listening head-cock: period (s) and size (rad).
_BOB_PERIOD, _BOB_TILT = 3.4, 0.04
_COCK_PERIOD, _COCK_TILT = 6.1, 0.022

#: The power-on: a scanline draws across the glass, the eyes open out of it
#: and blink once. The very first frame already shows the visor, so the face
#: is never a blank rectangle while ORION boots.
BOOT_SECONDS = 1.35

#: How long ORION sits resting (standby) or idle before he dozes off and the
#: little z's start. Any state change, speech, emotion or glance wakes him.
_DOZE_AFTER = {"rest": 45.0, "idle": 120.0}

#: A viseme older than this has stopped arriving; the mouth falls back to the
#: amplitude envelope instead of freezing on the last shape.
_VISEME_FRESH = 0.22

#: Largest frame rasterised, in pixels. Above it the panel renders smaller and
#: the GUI's blit stretches it up. The picture is soft glow on dark glass, so a
#: modest upscale is invisible — but this face is drawn every frame, and a
#: 2250 x 1300 high-DPI frame is 3 MP of fill for nothing. Chosen so a 1500 px
#: wide window at 100 % renders natively and only high-DPI 4K pays a stretch.
MAX_RENDER_PIXELS = 1_400_000

#: Fixed-pitch families for the caption, as the HUD uses (hud_shell.MONO_FAMILIES,
#: repeated here so the face does not import the shell to draw one line).
_MONO_FAMILIES = ("JetBrains Mono", "Cascadia Mono", "Consolas",
                  "DejaVu Sans Mono", "Courier New")


def headless() -> bool:
    """Whether this process has no screen to draw on.

    The same rule holo_head.headless() applies, repeated rather than imported
    because importing holo_head pulls in numpy and the head mesh — exactly the
    weight this face exists to avoid at startup.
    """
    if os.getenv("ORION_HEADLESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return os.getenv("QT_QPA_PLATFORM", "").strip().lower() in {
        "offscreen", "minimal", "vnc"}


def face_thread_enabled() -> bool:
    """Whether the face rasterises on the pipeline's worker thread.

    On unless ``ORION_FACE_THREAD=0`` — the same switch the sculpted head
    honours, and the first thing to try if a face ever misbehaves.
    """
    if headless():
        return False
    return os.getenv("ORION_FACE_THREAD", "1").strip().lower() not in {
        "0", "false", "no", "off"}


def face_geometry(width: float, height: float) -> tuple[float, float, float]:
    """Where the visor sits in a ``width`` x ``height`` frame: (cx, cy, r).

    ``r`` is the visor's half-height; the visor is 2.9 r wide, and the HUD
    brackets and the caption need about 0.5 r of room around it. Bound by both
    dimensions so a tall overlay tile and a wide monitor both frame it.
    """
    w, h = max(8.0, float(width)), max(8.0, float(height))
    r = max(4.0, min(h * 0.33, w * 0.30))
    return w / 2.0, h * 0.47, r


# ── small helpers ────────────────────────────────────────────────────────────

def _ease(dt: float, tau: float) -> float:
    """Frame-rate independent approach factor (see holo_head._ease)."""
    if tau <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def _smooth(edge0: float, edge1: float, x: float) -> float:
    t = _clamp((x - edge0) / max(1e-9, edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _back_out(t: float) -> float:
    """Ease-out with a small overshoot — the eyes "pop" open on boot."""
    t = _clamp(t)
    c1 = 1.55
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


def _mix(a: QColor, b: QColor, t: float, alpha: int | None = None) -> QColor:
    t = _clamp(t)
    colour = QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )
    if alpha is not None:
        colour.setAlpha(int(_clamp(alpha, 0, 255)))
    return colour


def _alpha(colour: QColor, alpha: float) -> QColor:
    out = QColor(colour)
    out.setAlpha(int(_clamp(alpha, 0.0, 255.0)))
    return out


_WHITE = QColor(255, 255, 255)
_BLACK = QColor(0, 0, 0)


# ── the expression vocabulary ────────────────────────────────────────────────

#: Every channel the face can express, at its resting value. States set some
#: of these outright; emotions add their deviation from here.
NEUTRAL: dict[str, float] = {
    "size": 1.0,       # eye scale
    "w": 1.0,          # eye width multiplier
    "h": 1.0,          # eye height multiplier
    "round": 0.62,     # corner roundness, 0 squarish .. 1 capsule
    "open": 1.0,       # eyelid opening, 0 shut .. 1 open
    "lid_top": 0.0,    # both upper lids lowered (relaxed, focused)
    "lid_in": 0.0,     # inner corners lowered (cross, determined)
    "lid_out": 0.0,    # outer corners lowered (sad, worried)
    "smile": 0.0,      # the happy crescent, ^ ^
    "asym": 0.0,       # right eye bigger (+), left eye bigger (-)
    "squint_l": 0.0,   # the left lid alone (puzzled)
    "tilt": 0.0,       # head tilt, radians
    "mouth": 0.22,     # mouth curve, -1 frown .. 1 smile
    "mouth_w": 1.0,    # mouth width multiplier
    "mouth_o": 0.0,    # a held "o" (surprise), separate from speech
    "wave": 0.0,       # a wobbly mouth (unsure, worried)
    "zip": 0.0,        # muted: the mouth stitched shut
    "mouth_vis": 1.0,  # the mouth itself (thinking dots replace it)
    "dots": 0.0,       # three soft thinking dots
    "blush": 0.0,      # cheeks
    "heart": 0.0,      # heart eyes
    "glow": 1.0,       # brightness
    "lean": 0.0,       # leaning in to listen
    "shimmer": 0.0,    # the scanning sweep across the glass
    "squint": 0.0,     # lower lids pushed up by the cheeks (amused, happy)
    "brow": 0.0,       # one raised brow: right eye wider, left lid lowered (-: mirrored)
    "sparkle": 0.0,    # twinkling glints in the eyes (happy, excited, love)
    "sleep": 0.0,      # dozing: the floating z's
    "eq": 0.0,         # listening level ticks at the visor's edges
    "power": 1.0,      # 0 = switched off (the CRT collapse)
}

#: Per-channel easing time constants. Anything not listed uses _TAU_DEFAULT.
_TAU_DEFAULT = 0.14
_TAU = {"power": 0.22, "sleep": 0.55, "dots": 0.18, "shimmer": 0.3,
        "blush": 0.32, "heart": 0.18, "tilt": 0.32, "lean": 0.25,
        "glow": 0.25, "eq": 0.3, "mouth_vis": 0.16, "zip": 0.2,
        "squint": 0.22, "brow": 0.24, "sparkle": 0.3}

#: Channels that ease through ONE lag: "open" has its own timing (falling asleep
#: is slow, waking is not), the CRT power-off and the z's are meant to be
#: abrupt or slow in their own way. Everything else eases through two lags in
#: series - zero velocity at both ends of a change, an S-curve.
_SINGLE_LAG = frozenset({"open", "power", "sleep"})
#: Each of the two lags is this fraction of the channel's time constant, so the
#: pair settles in about the time the single lag did.
_LAG_SPLIT = 0.55

#: What each state family looks like. Absolute values for the channels it
#: sets; everything else stays at NEUTRAL.
MODES: dict[str, dict[str, float]] = {
    "idle": {},
    # Standby, the dormant avatar: relaxed, content, a little dimmer.
    "rest": {"lid_top": 0.24, "mouth": 0.32, "glow": 0.82},
    # Attentive, not intense — ORION is LISTENING most of the time he is on.
    "listening": {"size": 1.07, "glow": 1.12, "lean": 1.0, "eq": 1.0,
                  "round": 0.7, "mouth": 0.26},
    "speaking": {"glow": 1.06, "mouth": 0.18},
    "thinking": {"lid_top": 0.16, "w": 0.96, "dots": 1.0, "mouth_vis": 0.0,
                 "glow": 0.96},
    "connecting": {"shimmer": 1.0, "dots": 1.0, "mouth_vis": 0.0, "glow": 0.9,
                   "size": 0.96},
    "sleep": {"open": 0.0, "sleep": 1.0, "mouth": 0.1, "glow": 0.72,
              "round": 1.0, "mouth_w": 0.8},
    "offline": {"open": 0.34, "size": 0.9, "glow": 0.42, "mouth": 0.0,
                "lid_out": 0.08, "mouth_w": 0.8},
    "muted": {"zip": 1.0, "mouth": 0.0},
    "focus": {"lid_top": 0.1, "mouth": 0.02, "mouth_w": 0.72},
    "error": {"lid_out": 0.24, "size": 0.95, "wave": 1.0, "mouth": -0.08},
    "off": {"power": 0.0, "glow": 0.6},
}

#: Raw state words (bus, worker, avatar controller) -> state family.
_STATE_MODES: dict[str, str] = {
    "IDLE": "idle", "": "idle", "READY": "idle",
    "STANDBY": "rest", "RESTING": "rest", "DORMANT": "rest",
    "LISTENING": "listening", "HEARING": "listening",
    "SPEAKING": "speaking", "TALKING": "speaking",
    "THINKING": "thinking", "PROCESSING": "thinking", "WORKING": "thinking",
    "EXECUTING": "thinking", "RESEARCHING": "thinking", "REASONING": "thinking",
    "CONNECTING": "connecting", "RECONNECTING": "connecting",
    "INITIALISING": "connecting", "INITIALIZING": "connecting",
    "STARTING": "connecting", "BOOTING": "connecting", "LOADING": "connecting",
    "SLEEPING": "sleep", "SLEEP": "sleep", "PAUSED": "sleep",
    "OFFLINE": "offline", "DISCONNECTED": "offline",
    "MUTED": "muted", "FOCUS": "focus", "QUIET": "focus",
    "ERROR": "error", "FAULT": "error", "WARNING": "error", "FAILED": "error",
    "SHUTTING DOWN": "off", "SHUTTING_DOWN": "off", "SHUTDOWN": "off",
    "POWER OFF": "off",
}

#: Emotion presets — absolute values for the channels each one moves. Kept
#: small and soft on purpose: every negative feeling here is a pout, never a
#: glare. A cute face that can look menacing is not a cute face.
EMOTIONS: dict[str, dict[str, float]] = {
    "neutral": {},
    "happy": {"smile": 0.92, "blush": 1.0, "mouth": 0.9, "mouth_w": 1.25,
              "h": 0.94, "w": 1.06, "glow": 1.12, "squint": 0.14,
              "sparkle": 1.0},
    "excited": {"smile": 0.85, "blush": 0.85, "mouth": 1.0, "mouth_w": 1.35,
                "size": 1.08, "glow": 1.2, "squint": 0.1, "sparkle": 1.0},
    # Amused is the gentle squint: the cheeks lift the lower lids and the head
    # leans a hair to one side, a half-smile with a twinkle.
    "amused": {"smile": 0.6, "blush": 0.45, "mouth": 0.65, "mouth_w": 1.15,
               "glow": 1.06, "squint": 0.55, "sparkle": 0.5, "tilt": 0.035},
    "proud": {"smile": 0.4, "lid_top": 0.1, "mouth": 0.7, "mouth_w": 1.15,
              "glow": 1.08, "squint": 0.1, "sparkle": 0.45},
    "reassuring": {"smile": 0.32, "lid_top": 0.1, "mouth": 0.55,
                   "squint": 0.12, "tilt": 0.03},
    "content": {"smile": 0.2, "lid_top": 0.16, "mouth": 0.45, "squint": 0.2},
    "grateful": {"smile": 0.95, "blush": 0.9, "mouth": 0.75, "glow": 1.08,
                 "squint": 0.25, "sparkle": 0.8},
    "love": {"heart": 1.0, "blush": 1.0, "mouth": 0.75, "mouth_w": 1.1,
             "glow": 1.1, "squint": 0.12, "sparkle": 0.85},
    "shy": {"blush": 1.0, "smile": 0.35, "mouth": 0.35, "mouth_w": 0.8,
            "size": 0.95, "tilt": 0.04},
    "sad": {"lid_out": 0.36, "size": 0.92, "h": 0.92, "mouth": -0.6,
            "mouth_w": 0.82, "glow": 0.76},
    "disappointed": {"lid_out": 0.3, "lid_top": 0.1, "mouth": -0.42,
                     "mouth_w": 0.85, "glow": 0.82},
    "surprised": {"size": 1.2, "round": 1.0, "w": 1.04, "h": 1.08,
                  "mouth": 0.0, "mouth_o": 0.6, "glow": 1.15, "sparkle": 0.35},
    "alert": {"size": 1.1, "round": 0.85, "mouth": 0.0, "mouth_o": 0.25,
              "glow": 1.1},
    "alarmed": {"size": 1.16, "round": 0.95, "mouth": -0.1, "mouth_o": 0.42,
                "lid_out": 0.06, "glow": 1.12},
    # Curious: one brow up (the right eye wider, the left lid lowered a touch),
    # a head tilt and a hopeful glint.
    "curious": {"asym": 1.0, "tilt": 0.10, "mouth": 0.3, "mouth_w": 0.8,
                "glow": 1.05, "brow": 0.8, "sparkle": 0.25},
    # Concerned: worried brows (the outer lids down) and a sympathetic tilt.
    "concerned": {"lid_out": 0.3, "lid_top": 0.04, "size": 0.96,
                  "mouth": -0.28, "wave": 0.35, "brow": 0.3, "tilt": 0.05},
    "empathetic": {"lid_out": 0.22, "size": 0.98, "mouth": 0.12, "blush": 0.25,
                   "tilt": 0.04, "brow": 0.15},
    "angry": {"lid_in": 0.3, "size": 0.95, "mouth": -0.35, "mouth_w": 0.8},
    "determined": {"lid_in": 0.18, "lid_top": 0.08, "mouth": 0.05,
                   "mouth_w": 0.85},
    "thinking": {"lid_top": 0.15, "mouth": 0.0, "mouth_w": 0.8},
    "confused": {"squint_l": 0.34, "asym": -0.3, "tilt": -0.09, "wave": 0.8,
                 "mouth": -0.05},
    "uncertain": {"squint_l": 0.2, "tilt": -0.05, "size": 0.96, "mouth": -0.1,
                  "mouth_w": 0.8},
    "bored": {"lid_top": 0.36, "mouth": -0.04, "mouth_w": 0.7, "glow": 0.85},
    "listening": {"glow": 1.04},
    "speaking": {},
    # Sleepy is a state as much as a feeling; it is resolved to the sleep
    # family in step() so that it can never be mixed with an open-eyed state.
    "sleepy": {},
}

#: Every other word an emotion arrives as — EmotionStateManager's eleven-plus
#: names, the avatar controller's tints, sentiment labels and plain English —
#: mapped onto a preset. Unknown names rest at neutral rather than raising.
_EMOTION_ALIASES: dict[str, str] = {
    "joy": "happy", "joyful": "happy", "glad": "happy", "pleased": "happy",
    "delighted": "happy", "celebratory": "happy", "cheerful": "happy",
    "thrilled": "excited", "ecstatic": "excited", "eager": "excited",
    "laughing": "amused", "funny": "amused", "playful": "amused",
    "calm": "content", "relaxed": "content", "peaceful": "content",
    "confident": "neutral", "composed": "neutral",
    "thankful": "grateful", "thanks": "grateful", "appreciative": "grateful",
    "loving": "love", "affection": "love", "affectionate": "love",
    "adoring": "love", "heart": "love",
    "embarrassed": "shy", "bashful": "shy",
    "unhappy": "sad", "sorrow": "sad", "down": "sad", "melancholy": "sad",
    "gloomy": "sad", "lonely": "sad",
    "surprise": "surprised", "shocked": "surprised", "astonished": "surprised",
    "amazed": "surprised", "wow": "surprised", "startled": "surprised",
    "urgent": "alert", "notification": "alert", "notice": "alert",
    "warning": "alarmed", "critical": "alarmed", "scared": "alarmed",
    "afraid": "alarmed", "fearful": "alarmed",
    "inquisitive": "curious", "interested": "curious", "intrigued": "curious",
    "worried": "concerned", "anxious": "concerned", "nervous": "concerned",
    "empathetic_concern": "concerned", "sympathetic": "empathetic",
    "caring": "empathetic", "compassionate": "empathetic",
    "frustrated": "angry", "mad": "angry", "annoyed": "angry",
    "irritated": "angry", "cross": "angry",
    "focused": "thinking", "concentrating": "thinking", "analytical": "thinking",
    "pondering": "thinking", "contemplative": "thinking",
    "puzzled": "confused", "perplexed": "confused",
    "doubtful": "uncertain", "unsure": "uncertain", "hesitant": "uncertain",
    "tired": "sleepy", "drowsy": "sleepy", "sleeping": "sleepy",
    "asleep": "sleepy", "exhausted": "sleepy",
}


#: How a feeling ARRIVES: the sign and size of the overshoot it lands with
#: (+ widens the eyes for a moment, - sinks them). Anything not listed is a
#: faint +0.15. A surprise is a quick widen that settles; sadness arrives with
#: a small sink.
_ONSET: dict[str, float] = {
    "surprised": 1.0, "alarmed": 1.0, "alert": 0.8, "excited": 0.9,
    "happy": 0.45, "love": 0.4, "curious": 0.35, "grateful": 0.3, "proud": 0.3,
    "amused": 0.3, "confused": 0.2, "angry": -0.2, "concerned": -0.2,
    "disappointed": -0.4, "sad": -0.5, "bored": -0.3,
}

#: The small things an idle face does on its own: (seconds, peak offsets). The
#: offsets ride on the expression channels under a smooth bump, then vanish;
#: "side" ones are mirrored left/right at random. Nothing here is a feeling -
#: it is the little life a screen face has between thoughts.
VARIETY: dict[str, tuple[float, dict[str, float]]] = {
    "perk": (1.4, {"size": 0.07, "tilt": 0.06, "brow": 0.5, "sparkle": 0.4}),
    "tilt": (2.2, {"tilt": 0.075}),
    "smile": (1.8, {"smile": 0.28, "mouth": 0.3, "sparkle": 0.7, "blush": 0.25,
                    "squint": 0.15}),
    "squint": (1.6, {"lid_top": 0.12, "squint": 0.25, "tilt": 0.03}),
    "sparkle": (1.1, {"sparkle": 1.0}),
    "wonder": (1.8, {"brow": 0.6, "tilt": 0.05, "asym": 0.4}),
}
#: Channels whose variety offset is mirrored (a tilt to the left or the right).
_VARIETY_SIDED = frozenset({"tilt", "brow", "asym"})
#: With a feeling already on, only these are added (a smile on top of a frown
#: is a mixed message; a glint or a lean is not).
_VARIETY_WITH_EMOTION = ("tilt", "sparkle", "perk")


@dataclass(frozen=True)
class FaceDesign:
    """How one Screen Face design draws the same expression channels (Mark
    XXXIX). Every design wears every expression - the channels, easing and
    timing are shared - only the eyes', the mouth's and the cheeks' drawing
    differ. The choice is appearance.face_design()."""

    key: str
    eye: str = "pill"          # pill | blade | ring | slit | dot
    mouth: str = "curve"       # curve | bar | wave
    eye_w: float = 1.0         # eye box width multiplier
    eye_h: float = 1.0         # eye box height multiplier
    round_scale: float = 1.0   # multiplies the 'round' channel
    cute: float = 1.0          # blush, hearts, sparkles and the cartoon blush strokes
    glint: float = 1.0         # the silver highlight
    halo: float = 1.0          # how far each eye lights the glass
    outline: float = 0.0       # a crisp accent edge on each eye
    core: float = 0.0          # a bright inner line (blade/slit) or pupil (ring)


#: The designs. Sentinel - the default - is drawn for the crimson-and-silver
#: HUD: wide, low, square-shouldered eyes with a silver edge and a lit core,
#: a light-bar mouth that becomes a level meter when he speaks, nothing
#: cartoon. Companion is the original look, unchanged.
DESIGNS: dict[str, FaceDesign] = {
    "sentinel": FaceDesign("sentinel", eye="blade", mouth="bar", eye_w=1.22, eye_h=0.56,
                           round_scale=0.42, cute=0.12, glint=0.0, halo=0.8,
                           outline=0.6, core=0.85),
    "arc": FaceDesign("arc", eye="ring", mouth="bar", eye_w=1.02, eye_h=0.74,
                      round_scale=1.0, cute=0.2, glint=0.0, halo=1.0, outline=0.0, core=1.0),
    "visor": FaceDesign("visor", eye="slit", mouth="wave", eye_w=1.55, eye_h=0.3,
                        round_scale=0.35, cute=0.1, glint=0.0, halo=1.3, outline=0.0,
                        core=0.7),
    "minimal": FaceDesign("minimal", eye="dot", mouth="curve", eye_w=0.55, eye_h=0.42,
                          round_scale=1.6, cute=0.35, glint=0.5, halo=0.55),
    "companion": FaceDesign("companion"),
}
DEFAULT_DESIGN = "sentinel"


def emotion_preset(name: Any) -> str:
    """The preset key for an emotion name, or "neutral" if unknown."""
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in EMOTIONS:
        return key
    return _EMOTION_ALIASES.get(key, "neutral")


def state_mode(state: Any) -> str:
    """The state family for a raw state word; unknown states rest at idle."""
    word = str(state or "").strip().upper().replace("-", " ")
    if word in _STATE_MODES:
        return _STATE_MODES[word]
    word = word.replace("_", " ")
    if word in _STATE_MODES:
        return _STATE_MODES[word]
    for needle, mode in (("SPEAK", "speaking"), ("LISTEN", "listening"),
                         ("THINK", "thinking"), ("PROCESS", "thinking"),
                         ("CONNECT", "connecting"), ("SLEEP", "sleep"),
                         ("SHUT", "off"), ("ERROR", "error"),
                         ("OFFLINE", "offline")):
        if needle in word:
            return mode
    return "idle"


class ScreenFace:
    """The rig. Deliberately not a QWidget, so the face pipeline's worker can
    own it outright — the same contract HoloHead keeps:

        face = ScreenFace()
        face.step(dt, amplitude, speaking=..., state=...)     # once per frame
        face.paint(painter, cx, cy, radius, primary, accent, background)

    Every other method changes a target, and is applied by whichever thread
    owns the rig (the panel queues them through the pipeline).
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._t = 0.0
        self._boot = 0.0
        self._state = ""
        self._mode = "idle"
        self._speaking = False
        self._dormant = False
        self._idle_for = 0.0
        self._ambient = 1.0

        self._p: dict[str, float] = dict(NEUTRAL)
        #: The first lag of the two in series (see _SINGLE_LAG).
        self._p1: dict[str, float] = dict(NEUTRAL)
        #: What is painted: _p plus the transient offsets (the feeling's
        #: overshoot, the thinking bob, an idle flourish). Rebuilt every step.
        self._pe: dict[str, float] = dict(NEUTRAL)
        self._emotion = "neutral"
        self._intensity = 1.0
        self._pulse = 0.0
        # A feeling arriving: its overshoot, decaying. _onset is the envelope
        # (1 at the moment it lands), _onset_gain its sign and size.
        self._onset = 0.0
        self._onset_gain = 0.0
        # Idle variety: the current flourish, its age, who may play it.
        self._variety_at = 6.0 + self._rng.uniform(0.0, 4.0)
        self._variety: tuple[str, float, float, dict[str, float]] | None = None
        self._variety_gain = 0.0
        self._variety_last = ""
        #: How much transient motion is in play (0..~1), for the budget.
        self._fx_mag = 0.0

        # Blink: _blink is 0 open .. 1 shut; _blink_age < 0 means not blinking.
        self._blink = 0.0
        self._blink_age = -1.0
        # The first blink lands at the end of the power-on, as the "hello".
        self._blink_at = BOOT_SECONDS * 0.84

        # Gaze, in screen terms: x right, y DOWN, each roughly -1..1.
        self._gaze = [0.0, 0.0]
        self._gaze_target = [0.0, 0.0]
        self._gaze_at = 1.6
        self._glance = (0.0, 0.0)
        self._glance_until = -1.0
        self._micro = [0.0, 0.0]
        self._micro_at = 0.0
        self._pose = (0.0, 0.0)
        self._pose_at = -10.0

        # Mouth: current and target posture.
        self._open = 0.0
        self._wide = 0.0
        self._close = 0.0
        self._vis = (0.0, 0.0, 0.0)
        self._vis_at = -10.0
        self._spectrum = (0.0, 0.0, 0.0)

        # Loudness: the speech envelope (eyes lift a touch on stress) and the
        # listening level (rim pulse and edge ticks).
        self._speech = 0.0
        self._level = 0.0

        # Floating z's while dozing: [age, drift seed].
        self._zs: list[list[float]] = []
        self._z_at = 0.0

        self._label = ""
        self._label_alpha = 0.0
        #: How the channels are drawn (DESIGNS); set_design changes it.
        self._design: FaceDesign = DESIGNS[DEFAULT_DESIGN]

        # Cached layers, owned by whichever thread paints.
        self._layer_key: tuple | None = None
        self._under: QImage | None = None
        self._over: QImage | None = None
        self._over_at = (0, 0)
        self._squircle: QPainterPath | None = None
        self._font: QFont | None = None
        self._font_px = 0

        #: Wall-clock cost of the last paint, in milliseconds.
        self.last_paint_ms = 0.0
        #: The channels that carry meaning, published for the animation
        #: budget (read by the GUI thread; replaced whole, never mutated).
        self.channels: tuple[float, ...] = ()
        #: True while something is about to move or is mid-move in a way the
        #: idle frame rate would chop — a blink, the boot, the dots.
        self.anticipating = True

    # ── drive (targets only) ─────────────────────────────────────────────────

    def set_viseme(self, openness: float, width: float = 0.0,
                   closure: float = 0.0) -> None:
        """Target mouth posture: 0..1 openness, -1..1 width, 0..1 closure.

        Closure is separate because it must win: /m/ /b/ /p/ are lips pressed
        shut, and no loudness may open them (the same rule as HoloHead).
        """
        self._vis = (_clamp(float(openness)), _clamp(float(width), -1.0, 1.0),
                     _clamp(float(closure)))
        self._vis_at = self._t

    def glance(self, dx: float, dy: float, hold: float = 1.1) -> None:
        """Look somewhere for a moment. ``dy`` positive is UP, as everywhere
        else in ORION's face protocol (a banner glance is dy=-0.65: down)."""
        self._glance = (_clamp(float(dx), -1.0, 1.0), -_clamp(float(dy), -1.0, 1.0))
        self._glance_until = self._t + max(0.1, float(hold))
        self._idle_for = 0.0

    def apply_emotion(self, name: Any, params: Any = None) -> None:
        """Wear a named emotion. ``params`` (EmotionStateManager's profile)
        contributes its intensity and glow pulse; its palette is deliberately
        NOT applied — the owner's accent is the face's colour, always."""
        preset = emotion_preset(name)
        if preset != self._emotion:
            # A feeling LANDS: a quick overshoot that settles (the eyes pop on
            # surprise, sink a little on sadness). A return to neutral is quiet.
            self._onset = 1.0
            self._onset_gain = 0.0 if preset == "neutral" else _ONSET.get(preset, 0.15)
        self._emotion = preset
        intensity = 1.0
        pulse = 0.0
        if isinstance(params, dict):
            try:
                intensity = float(params.get("intensity", 1.0))
            except (TypeError, ValueError):
                intensity = 1.0
            try:
                pulse = float(params.get("glow_pulse", 0.0) or 0.0)
            except (TypeError, ValueError):
                pulse = 0.0
        self._intensity = _clamp(intensity, 0.2, 1.0)
        self._pulse = _clamp(pulse)
        self._idle_for = 0.0

    def set_label(self, text: Any) -> None:
        self._label = str(text or "").strip()[:60]

    def set_design(self, name: Any) -> str:
        """Wear another design (DESIGNS). Returns the design in effect; an
        unknown name changes nothing. The expression carries straight over -
        it is the same channels drawn another way - so a switch mid-sentence
        is a change of look, never a reset."""
        key = str(name or "").strip().lower()
        if key in DESIGNS:
            self._design = DESIGNS[key]
            self._idle_for = 0.0
        return self._design.key

    @property
    def design(self) -> str:
        return self._design.key

    def set_pose(self, yaw: float, pitch: float) -> None:
        """The avatar controller's face-tracking follow, as a gaze bias.

        ``yaw`` turns ORION toward the viewer's right when positive and
        ``pitch`` tilts him down; a screen face answers with its eyes, not its
        head, so the pose becomes where he looks — at you."""
        self._pose = (_clamp(float(yaw) / 0.22, -1.0, 1.0),
                      _clamp(float(pitch) / 0.22, -1.0, 1.0))
        self._pose_at = self._t

    def set_spectrum(self, low: float, mid: float, high: float) -> None:
        self._spectrum = (_clamp(float(low)), _clamp(float(mid)), _clamp(float(high)))

    def set_morph(self, value: float) -> None:
        """The avatar controller's materialisation channel (0 = dormant).

        A screen face has nothing to materialise, so it reads the channel for
        what it means: below one half ORION is dormant, and rests. Hysteresis
        keeps a value easing through the middle from flickering the look."""
        value = float(value)
        if value < 0.4:
            self._dormant = True
        elif value > 0.6:
            self._dormant = False

    def set_ambient(self, scale: float) -> None:
        """The Reduced-motion scale (appearance.MotionProfile.ambient):
        idle drift, saccade reach and breathing shrink; blinks, speech and
        expressions do not — they carry meaning."""
        self._ambient = _clamp(float(scale), 0.0, 1.0)

    def skip_boot(self) -> None:
        """Start already awake — for a surface nobody watches boot."""
        self._boot = BOOT_SECONDS + 1.0
        self._blink_at = min(self._blink_at, self._t + 2.0)

    def power_on(self) -> None:
        """Replay the power-on animation."""
        self._boot = 0.0
        self._blink_at = self._t + BOOT_SECONDS * 0.84

    # ── step ─────────────────────────────────────────────────────────────────

    def _base_mode(self, speaking: bool) -> str:
        """The state family before dozing: what ORION is doing right now."""
        mode = state_mode(self._state)
        if mode == "off":
            return "off"
        # Speech wins over everything but switching off: a mouth that freezes
        # mid-sentence because an emotion said "sleepy" is a broken face.
        if speaking or mode == "speaking":
            return "speaking"
        if self._emotion == "sleepy":
            return "sleep"
        if mode == "idle" and self._dormant:
            return "rest"
        return mode

    def _resolve_mode(self, base: str) -> str:
        doze = _DOZE_AFTER.get(base)
        if doze is not None and self._idle_for >= doze:
            return "sleep"
        return base

    def _targets(self, mode: str) -> dict[str, float]:
        target = dict(NEUTRAL)
        target.update(MODES.get(mode, {}))
        if mode not in ("sleep", "off", "offline"):
            emotion = EMOTIONS.get(self._emotion, {})
            k = self._intensity
            for key, value in emotion.items():
                target[key] += (value - NEUTRAL[key]) * k
        target["size"] = _clamp(target["size"], 0.6, 1.4)
        target["glow"] = _clamp(target["glow"], 0.3, 1.35)
        target["round"] = _clamp(target["round"])
        target["open"] = _clamp(target["open"], 0.0, 1.0)
        for key in ("lid_top", "lid_in", "lid_out", "squint_l", "squint"):
            target[key] = _clamp(target[key], 0.0, 0.8)
        target["brow"] = _clamp(target["brow"], -1.0, 1.0)
        target["sparkle"] = _clamp(target["sparkle"])
        target["mouth"] = _clamp(target["mouth"], -1.0, 1.0)
        return target

    def step(self, dt: float, amplitude: float = 0.0, *, speaking: bool = False,
             state: str = "") -> None:
        """Advance the animation by ``dt`` seconds."""
        dt = _clamp(float(dt), 0.0, 0.25)       # a long stall must not lurch
        self._t += dt
        self._boot += dt
        t = self._t
        try:
            amp = _clamp(float(amplitude))
        except (TypeError, ValueError):
            amp = 0.0

        raw = str(state or "").strip().upper()
        if raw != self._state:
            self._state = raw
            self._idle_for = 0.0
        vis_fresh = (t - self._vis_at) < _VISEME_FRESH
        self._speaking = bool(speaking) or vis_fresh
        base = self._base_mode(self._speaking)
        # Dozing is measured in quiet, not in time: a sound — your voice
        # included — resets it, so a dozing ORION wakes when you speak.
        if base in _DOZE_AFTER and amp < 0.03:
            self._idle_for += dt
        else:
            self._idle_for = 0.0
        mode = self._resolve_mode(base)
        if mode != self._mode:
            if self._mode == "sleep" and mode != "off":
                # Waking up: a blink straight away, like rubbing his eyes.
                self._blink_at = t + 0.25
            self._mode = mode
        talking = mode == "speaking"

        # ── expression channels ──────────────────────────────────────────────
        target = self._targets(mode)
        p = self._p
        p1 = self._p1
        # A startle (surprise, alarm) is QUICK: while its overshoot is still
        # fresh every channel eases at up to twice the speed.
        quick = 1.0 - 0.5 * self._onset if self._onset_gain >= 0.7 else 1.0
        for key, goal in target.items():
            tau = _TAU.get(key, _TAU_DEFAULT) * quick
            if key == "open":
                # Falling asleep is slow; waking up is not.
                tau = 0.6 if goal < p[key] and mode == "sleep" else 0.12
            if key in _SINGLE_LAG:
                p[key] += (goal - p[key]) * _ease(dt, tau)
                p1[key] = p[key]
            else:
                # Two lags in series: the first chases the goal, the second
                # chases the first. Zero velocity when a change begins and
                # when it ends - an S-curve, where one lag lurches at once.
                k = _ease(dt, tau * _LAG_SPLIT)
                p1[key] += (goal - p1[key]) * k
                p[key] += (p1[key] - p[key]) * k

        # ── loudness ─────────────────────────────────────────────────────────
        speech_goal = amp if talking else 0.0
        self._speech += (speech_goal - self._speech) * _ease(
            dt, 0.05 if speech_goal > self._speech else 0.18)
        level_goal = amp if mode == "listening" else 0.0
        self._level += (level_goal - self._level) * _ease(
            dt, 0.05 if level_goal > self._level else 0.35)

        # ── mouth ────────────────────────────────────────────────────────────
        if talking:
            loud = min(1.0, amp * 1.8)
            if vis_fresh:
                v_open, v_wide, v_close = self._vis
                open_goal = max(v_open, 0.3 * loud) * (1.0 - v_close)
                wide_goal, close_goal = v_wide, v_close
            else:
                low, _mid, high = self._spectrum
                open_goal = loud
                wide_goal = _clamp((high - low) * 0.9, -1.0, 1.0)
                close_goal = 0.0
        else:
            open_goal, wide_goal, close_goal = 0.0, 0.0, 0.0
        # Closing faster than opening: a plosive is a tap.
        self._open += (open_goal - self._open) * _ease(
            dt, 0.032 if open_goal > self._open else 0.024)
        self._wide += (wide_goal - self._wide) * _ease(dt, 0.05)
        self._close += (close_goal - self._close) * _ease(dt, 0.02)

        # ── gaze ─────────────────────────────────────────────────────────────
        reach_scale = 0.3 + 0.7 * self._ambient
        tracked = (t - self._pose_at) < 1.5
        if t < self._glance_until:
            goal = list(self._glance)
        elif mode == "thinking" or mode == "connecting":
            # Eyes up and to one side, held: thinking, not scanning.
            if t >= self._gaze_at:
                side = self._rng.choice((-1.0, 1.0))
                self._gaze_target = [side * self._rng.uniform(0.45, 0.7),
                                     -self._rng.uniform(0.45, 0.7)]
                self._gaze_at = t + self._rng.uniform(*_SACCADE_THINKING)
            goal = list(self._gaze_target)
        elif mode in ("sleep", "off", "offline"):
            goal = [0.0, 0.2]
        else:
            if t >= self._gaze_at:
                lo, hi, reach = {
                    "listening": (*_SACCADE_LISTENING, 0.28),
                    "speaking": (*_SACCADE_SPEAKING, 0.4),
                }.get(mode, (*_SACCADE_IDLE, 0.6))
                if self._rng.random() < 0.42:
                    # Back to you. Most of a friendly face's looking is at the
                    # person it is talking to.
                    self._gaze_target = [0.0, 0.0]
                else:
                    self._gaze_target = [
                        self._rng.uniform(-1.0, 1.0) * reach * reach_scale,
                        self._rng.uniform(-0.6, 0.6) * reach * reach_scale]
                self._gaze_at = t + self._rng.uniform(lo, hi)
            goal = list(self._gaze_target)
            if tracked:
                goal[0] = goal[0] * 0.4 + self._pose[0] * 0.6
                goal[1] = goal[1] * 0.4 + self._pose[1] * 0.5
        if t >= self._micro_at:
            jitter = 0.03 * self._ambient
            self._micro = [self._rng.uniform(-jitter, jitter),
                           self._rng.uniform(-jitter, jitter)]
            self._micro_at = t + self._rng.uniform(0.35, 0.9)
        for i in (0, 1):
            aim = goal[i] + self._micro[i]
            # Saccades are near-instant jumps, not drifts.
            self._gaze[i] += (aim - self._gaze[i]) * _ease(dt, 0.045)

        # ── blink ────────────────────────────────────────────────────────────
        booting = self._boot < BOOT_SECONDS
        if mode in ("sleep", "off"):
            self._blink = 0.0
            self._blink_age = -1.0
            self._blink_at = max(self._blink_at, t + 0.6)
        else:
            if self._blink_age < 0.0 and t >= self._blink_at:
                if mode == "thinking" and self._rng.random() < 0.45:
                    # Concentration suppresses blinking; skip this one.
                    self._blink_at = t + self._rng.uniform(_BLINK_MIN, _BLINK_MAX)
                else:
                    self._blink_age = 0.0
            if self._blink_age >= 0.0:
                self._blink_age += dt
                age = self._blink_age
                if age < _BLINK_CLOSE:
                    self._blink = _smooth(0.0, 1.0, age / _BLINK_CLOSE)
                elif age < _BLINK_CLOSE + _BLINK_HOLD:
                    self._blink = 1.0
                elif age < _BLINK_CLOSE + _BLINK_HOLD + _BLINK_OPEN:
                    self._blink = 1.0 - _smooth(
                        0.0, 1.0, (age - _BLINK_CLOSE - _BLINK_HOLD) / _BLINK_OPEN)
                else:
                    self._blink = 0.0
                    self._blink_age = -1.0
                    if self._rng.random() < _DOUBLE_BLINK_CHANCE:
                        self._blink_at = t + 0.09
                    else:
                        slow = 1.4 if mode == "rest" else 1.0
                        self._blink_at = t + slow * self._rng.uniform(
                            _BLINK_MIN, _BLINK_MAX)

        # ── z's ──────────────────────────────────────────────────────────────
        for z in self._zs:
            z[0] += dt
        self._zs = [z for z in self._zs if z[0] < 2.8]
        if p["sleep"] > 0.6 and t >= self._z_at and len(self._zs) < 3:
            self._zs.append([0.0, self._rng.uniform(-1.0, 1.0)])
            self._z_at = t + 1.25

        self._update_fx(dt, t, mode, booting)

        # ── caption ──────────────────────────────────────────────────────────
        label_goal = 1.0 if self._label else 0.0
        self._label_alpha += (label_goal - self._label_alpha) * _ease(dt, 0.3)

        # ── what the budget watches ──────────────────────────────────────────
        p_sum = (p["smile"] + p["lid_top"] + p["lid_in"] + p["lid_out"]
                 + p["size"] + p["heart"] + p["blush"] + p["tilt"])
        pe = self._pe
        self.channels = (
            round(self._blink, 3), round(p["open"], 3), round(self._open, 3),
            round(self._wide, 3), round(self._close, 3), round(p_sum, 3),
            round(p["power"], 3), round(self._level, 3), round(self._speech, 3),
            round(min(1.0, self._boot / BOOT_SECONDS), 3),
            round(pe["tilt"], 3), round(self._fx_mag, 3),
            round(p["squint"] + p["brow"] + p["sparkle"], 3),
        )
        self.anticipating = bool(
            booting or self._blink_age >= 0.0
            or (mode not in ("sleep", "off") and self._blink_at - t < 0.18)
            or p["dots"] > 0.05 or p["shimmer"] > 0.05
            or p["sparkle"] > 0.05 or self._fx_mag > 0.02
            or t < self._glance_until)

    # ── transients: arrival overshoot, head bob, idle variety ────────────────

    def _update_fx(self, dt: float, t: float, mode: str, booting: bool) -> None:
        """Rebuild the painted channels (_pe) from the eased ones (_p) plus the
        small moving parts that are not feelings:

          * the overshoot a feeling lands with (quick widen on surprise),
          * a soft head bob while he thinks, a slow head-cock while he listens,
          * an idle flourish every ten seconds or so - a perk, a tilt, a
            flicker of a smile - so a quiet face never looks frozen.

        Scalar arithmetic on ~28 floats: far below the cost of one eye path.
        """
        pe = self._pe
        pe.update(self._p)
        amb = self._ambient
        scale = 0.3 + 0.7 * amb              # Reduced motion shrinks the sway
        mag = 0.0

        # A feeling arriving.
        if self._onset > 1e-3:
            self._onset *= math.exp(-dt / _ONSET_TAU)
            env = self._onset * self._onset_gain
            pe["size"] += 0.13 * env
            pe["glow"] += 0.08 * env
            pe["h"] += 0.04 * env
            mag += abs(env)
        else:
            self._onset = 0.0

        # The head moves a little while he thinks and while he listens.
        think = _clamp(self._p["dots"]) * (1.0 - _clamp(self._p["shimmer"]))
        if think > 0.02:
            pe["tilt"] += _BOB_TILT * think * scale * math.sin(
                t * 2.0 * math.pi / _BOB_PERIOD)
            mag += 0.1 * think
        cock = _clamp(self._p["lean"])
        if cock > 0.02:
            pe["tilt"] += _COCK_TILT * cock * scale * math.sin(
                t * 2.0 * math.pi / _COCK_PERIOD)

        # Idle variety: only while he is idle or listening, awake, and not
        # under Reduced motion.
        allowed = (mode in ("idle", "listening") and not booting
                   and not self._speaking and amb >= _VARIETY_MIN_AMBIENT)
        self._variety_gain += ((1.0 if allowed else 0.0) - self._variety_gain) * _ease(dt, 0.15)
        if allowed and self._variety is None and t >= self._variety_at:
            self._start_variety(t, mode)
        if self._variety is not None:
            name, started, duration, offsets = self._variety
            u = (t - started) / duration
            if u >= 1.0:
                self._variety = None
                spacing = self._rng.uniform(*_VARIETY_EVERY)
                self._variety_at = t + spacing * (1.5 if mode == "listening" else 1.0)
            else:
                env = math.sin(math.pi * u) ** 2 * self._variety_gain
                for key, peak in offsets.items():
                    pe[key] += peak * env
                mag += env

        self._fx_mag = mag
        # Keep every painted channel inside the range the painters expect.
        for key in ("lid_top", "lid_in", "lid_out", "squint_l", "squint"):
            pe[key] = _clamp(pe[key], 0.0, 0.8)
        pe["smile"] = _clamp(pe["smile"], 0.0, 1.0)
        pe["blush"] = _clamp(pe["blush"], 0.0, 1.0)
        pe["sparkle"] = _clamp(pe["sparkle"])
        pe["brow"] = _clamp(pe["brow"], -1.0, 1.0)
        pe["mouth"] = _clamp(pe["mouth"], -1.0, 1.0)
        pe["size"] = _clamp(pe["size"], 0.6, 1.45)
        pe["glow"] = _clamp(pe["glow"], 0.3, 1.4)

    def _start_variety(self, t: float, mode: str) -> None:
        """Pick the next idle flourish. With a feeling already on, only the
        ones that cannot contradict it; mirrored at random."""
        names = list(VARIETY)
        if self._emotion != "neutral":
            names = [n for n in names if n in _VARIETY_WITH_EMOTION]
        if mode == "listening":
            names = [n for n in names if n != "squint"] or names
        # Never the same one twice running.
        names = [n for n in names if n != self._variety_last] or names
        name = self._rng.choice(names)
        self._variety_last = name
        duration, peaks = VARIETY[name]
        side = self._rng.choice((-1.0, 1.0))
        offsets = {key: (peak * side if key in _VARIETY_SIDED else peak)
                   for key, peak in peaks.items()}
        self._variety = (name, t, duration, offsets)

    # ── paint ────────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, cx: float, cy: float, radius: float,
              primary: QColor, accent: QColor, background: QColor) -> None:
        """Draw the face with its visor centred on (cx, cy) and ``radius`` as
        the visor's half-height. Colours come from the caller, every frame."""
        started = time.perf_counter()
        try:
            self._paint(painter, float(cx), float(cy), max(4.0, float(radius)),
                        QColor(primary), QColor(accent), QColor(background))
        finally:
            self.last_paint_ms = (time.perf_counter() - started) * 1000.0

    def _paint(self, painter: QPainter, cx: float, cy: float, r: float,
               primary: QColor, accent: QColor, background: QColor) -> None:
        device = painter.device()
        width = max(8, int(device.width())) if device is not None else int(cx * 2)
        height = max(8, int(device.height())) if device is not None else int(cy * 2)
        key = (width, height, round(cx, 1), round(cy, 1), round(r, 1),
               primary.rgba(), accent.rgba(), background.rgba())
        if key != self._layer_key:
            self._build_layers(width, height, cx, cy, r, primary, accent, background)
            self._layer_key = key

        p = self._pe
        boot = self._boot / BOOT_SECONDS
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._under is not None:
            painter.drawImage(0, 0, self._under)

        # Brightness: the expression's glow, a gentle lift on stressed speech,
        # an emotion's pulse, and the power level.
        bright = p["glow"] * (1.0 + 0.16 * self._speech)
        if self._pulse > 0.0:
            bright *= 1.0 + 0.1 * self._pulse * math.sin(self._t * 4.2)
        power = _clamp(p["power"])
        lit = self._lit_colour(primary, background, bright)

        self._paint_glass_effects(painter, cx, cy, r, primary, accent, boot)

        if power > 0.01:
            painter.save()
            try:
                self._paint_face(painter, cx, cy, r, lit, primary, accent,
                                 background, boot, power)
            finally:
                painter.restore()

        if self._over is not None:
            painter.drawImage(self._over_at[0], self._over_at[1], self._over)
        self._paint_rim_pulse(painter, primary, r)
        self._paint_label(painter, cx, cy, r, primary, accent, boot)

    @staticmethod
    def _lit_colour(primary: QColor, background: QColor, bright: float) -> QColor:
        """The eye colour at a brightness: toward the void below 1, toward
        incandescent white above it (never far — it must stay ORION's hue)."""
        if bright < 1.0:
            return _mix(background, primary, 0.25 + 0.75 * _clamp(bright))
        return _mix(primary, _WHITE, _clamp((bright - 1.0) * 0.9, 0.0, 0.35))

    # ── static layers ────────────────────────────────────────────────────────

    def _squircle_path(self, cx: float, cy: float, r: float, inset: float = 0.0) -> QPainterPath:
        """The visor: a superellipse, which reads as a screen rather than as a
        rounded rectangle or a pill."""
        a = 1.45 * r - inset
        b = r - inset
        n = 3.6
        e = 2.0 / n
        poly = QPolygonF()
        steps = 120
        for i in range(steps):
            th = 2.0 * math.pi * i / steps
            c, s = math.cos(th), math.sin(th)
            x = a * math.copysign(abs(c) ** e, c)
            y = b * math.copysign(abs(s) ** e, s)
            poly.append(QPointF(cx + x, cy + y))
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()
        return path

    def _build_layers(self, width: int, height: int, cx: float, cy: float,
                      r: float, primary: QColor, accent: QColor,
                      background: QColor) -> None:
        """Bake everything that does not move.

        UNDER is opaque and covers the frame, so drawing it is a copy rather
        than a blend: the void, the ambient glow, a faint HUD grid, the visor
        glass and the brackets. OVER is translucent and covers only the visor:
        scanlines, the glass sheen and the rim — on top of the eyes, so they
        read as light behind glass rather than shapes stuck on it.
        """
        visor = self._squircle_path(cx, cy, r)
        self._squircle = visor
        small = r < 60.0

        under = QImage(width, height, QImage.Format.Format_RGB32)
        under.fill(background)
        p = QPainter(under)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(Qt.PenStyle.NoPen)
            # Ambient: ORION's hue, very faint, pooled behind the visor.
            glow = QRadialGradient(cx, cy, r * 3.0)
            glow.setColorAt(0.0, _alpha(primary, 34))
            glow.setColorAt(0.45, _alpha(primary, 12))
            glow.setColorAt(1.0, _alpha(primary, 0))
            p.setBrush(glow)
            p.drawRect(QRectF(0, 0, width, height))

            # A faint HUD grid, fading out toward the edges.
            if not small:
                spacing = max(14.0, r * 0.2)
                minor = QPen(_alpha(primary, 10), 1.0)
                major = QPen(_alpha(primary, 20), 1.0)
                start_x = cx - math.floor(cx / spacing) * spacing
                start_y = cy - math.floor(cy / spacing) * spacing
                x = start_x
                while x < width:
                    index = int(round((x - cx) / spacing))
                    p.setPen(major if index % 5 == 0 else minor)
                    p.drawLine(QPointF(x, 0), QPointF(x, height))
                    x += spacing
                y = start_y
                while y < height:
                    index = int(round((y - cy) / spacing))
                    p.setPen(major if index % 5 == 0 else minor)
                    p.drawLine(QPointF(0, y), QPointF(width, y))
                    y += spacing
                vignette = QRadialGradient(cx, cy, max(width, height) * 0.62)
                vignette.setColorAt(0.0, _alpha(background, 0))
                vignette.setColorAt(0.45, _alpha(background, 60))
                vignette.setColorAt(1.0, _alpha(background, 245))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(vignette)
                p.drawRect(QRectF(0, 0, width, height))

            # Outer halo of the visor, so the glass sits IN light. Elliptical
            # to match the visor and faded all the way to nothing — a halo
            # clipped to a shape while still lit shows its edge as a plate.
            p.save()
            p.translate(cx, cy)
            p.scale(1.45, 1.0)
            halo = QRadialGradient(0.0, 0.0, r * 1.7)
            halo.setColorAt(0.0, _alpha(primary, 30))
            halo.setColorAt(0.6, _alpha(primary, 20))
            halo.setColorAt(1.0, _alpha(primary, 0))
            p.setBrush(halo)
            p.drawEllipse(QPointF(0.0, 0.0), r * 1.7, r * 1.7)
            p.restore()

            # The glass: near-black, faintly warmed by ORION's hue, lighter at
            # the top as if lit from above.
            glass = QLinearGradient(cx, cy - r, cx, cy + r)
            glass.setColorAt(0.0, _mix(background, primary, 0.075))
            glass.setColorAt(0.5, _mix(background, primary, 0.04))
            glass.setColorAt(1.0, _mix(background, _BLACK, 0.5))
            p.setBrush(glass)
            p.drawPath(visor)
            # The screen's own backlight, behind where the eyes sit.
            back = QRadialGradient(cx, cy - r * 0.05, r * 1.35)
            back.setColorAt(0.0, _alpha(primary, 30))
            back.setColorAt(0.6, _alpha(primary, 10))
            back.setColorAt(1.0, _alpha(primary, 0))
            p.setBrush(back)
            p.drawPath(visor)

            # HUD corner brackets, in the same grammar as the orb's
            # (gui/hud.py): ORION's hue, with a silver inner tick.
            margin = r * 0.13
            left, right = cx - 1.45 * r - margin, cx + 1.45 * r + margin
            top, bottom = cy - r - margin, cy + r + margin
            length = r * 0.24
            line = max(1.0, r * 0.014)
            for (px, py), (sx, sy) in (((left, top), (1, 1)), ((right, top), (-1, 1)),
                                       ((left, bottom), (1, -1)),
                                       ((right, bottom), (-1, -1))):
                pen = QPen(_alpha(primary, 165), line)
                pen.setCapStyle(Qt.PenCapStyle.SquareCap)
                p.setPen(pen)
                p.drawLine(QPointF(px, py), QPointF(px + sx * length, py))
                p.drawLine(QPointF(px, py), QPointF(px, py + sy * length))
                if not small:
                    p.setPen(QPen(_alpha(accent, 110), max(1.0, line * 0.6)))
                    inset = line * 3.2
                    p.drawLine(QPointF(px + sx * inset, py + sy * inset),
                               QPointF(px + sx * length * 0.45, py + sy * inset))
            # Side ticks at mid-height: a viewfinder's measure, not a decoration
            # the eye has to read.
            if not small:
                p.setPen(QPen(_alpha(accent, 55), max(1.0, line * 0.6)))
                for sx, x0 in ((-1, left), (1, right)):
                    for k in (-1, 0, 1):
                        y0 = cy + k * r * 0.12
                        reach = r * (0.07 if k == 0 else 0.04)
                        p.drawLine(QPointF(x0, y0), QPointF(x0 - sx * reach, y0))
        finally:
            p.end()
        self._under = under

        # OVER: only the visor's box plus room for the rim's glow.
        pad = r * 0.12 + 4.0
        ox = int(math.floor(cx - 1.45 * r - pad))
        oy = int(math.floor(cy - r - pad))
        ow = int(math.ceil(2.9 * r + 2 * pad)) + 1
        oh = int(math.ceil(2.0 * r + 2 * pad)) + 1
        over = QImage(max(1, ow), max(1, oh), QImage.Format.Format_ARGB32_Premultiplied)
        over.fill(Qt.GlobalColor.transparent)
        p = QPainter(over)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.translate(-ox, -oy)
            p.save()
            p.setClipPath(visor)
            # Scanlines: the texture that makes the eyes read as a display.
            pitch = max(2.0, r * 0.028)
            p.setPen(QPen(_alpha(_BLACK, 70 if not small else 40), max(1.0, pitch * 0.38)))
            y = cy - r
            while y < cy + r:
                p.drawLine(QPointF(cx - 1.5 * r, y), QPointF(cx + 1.5 * r, y))
                y += pitch
            # Glass sheen: a soft diagonal from the top-left, the one cue that
            # this is a surface with something behind it.
            sheen = QLinearGradient(cx - 1.45 * r, cy - r, cx - 0.2 * r, cy + 0.3 * r)
            sheen.setColorAt(0.0, _alpha(_WHITE, 26))
            sheen.setColorAt(0.5, _alpha(_WHITE, 6))
            sheen.setColorAt(1.0, _alpha(_WHITE, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(sheen)
            p.drawPath(visor)
            p.restore()

            # The rim: a soft glow, a crisp line in ORION's hue, and a silver
            # hairline just inside it — a bezel, not an outline.
            p.setBrush(Qt.BrushStyle.NoBrush)
            for width_k, alpha in ((0.09, 12), (0.05, 22), (0.026, 46)):
                p.setPen(QPen(_alpha(primary, alpha), max(1.0, r * width_k)))
                p.drawPath(visor)
            p.setPen(QPen(_mix(primary, _WHITE, 0.12, 225), max(1.1, r * 0.011)))
            p.drawPath(visor)
            if not small:
                p.setPen(QPen(_alpha(accent, 42), 1.0))
                p.drawPath(self._squircle_path(cx, cy, r, inset=max(2.5, r * 0.03)))
        finally:
            p.end()
        self._over = over
        self._over_at = (ox, oy)

    # ── dynamic layers ───────────────────────────────────────────────────────

    def _paint_glass_effects(self, painter: QPainter, cx: float, cy: float,
                             r: float, primary: QColor, accent: QColor,
                             boot: float) -> None:
        """Things that move on the glass itself rather than in the face: the
        boot scanline, the scanning shimmer and the listening ticks."""
        p = self._p
        # Power-on scanline: grows from the centre, then fades as the eyes
        # open out of it.
        if boot < 0.62:
            grow = _smooth(0.0, 0.3, boot)
            fade = 1.0 - _smooth(0.32, 0.6, boot)
            half = 1.3 * r * grow
            y = cy - r * 0.08
            thick = max(1.5, r * 0.02)
            for w_k, a in ((7.0, 30), (3.0, 70), (1.0, 230)):
                pen = QPen(_alpha(_mix(primary, _WHITE, 0.35), a * fade), thick * w_k)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QPointF(cx - half, y), QPointF(cx + half, y))

        if p["shimmer"] > 0.02 and self._squircle is not None:
            painter.save()
            painter.setClipPath(self._squircle)
            phase = (self._t * 0.55) % 1.0
            x = cx - 1.9 * r + phase * 3.8 * r
            band = QLinearGradient(x - r * 0.45, cy, x + r * 0.45, cy)
            band.setColorAt(0.0, _alpha(primary, 0))
            band.setColorAt(0.5, _alpha(_mix(primary, _WHITE, 0.3), 34 * p["shimmer"]))
            band.setColorAt(1.0, _alpha(primary, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(band)
            painter.drawRect(QRectF(x - r * 0.45, cy - r, r * 0.9, 2 * r))
            painter.restore()

        if p["eq"] > 0.02 and r >= 40.0:
            # Five short ticks inside each edge of the glass: dim while he
            # waits, lighting from the middle outward as you speak. It is the
            # face saying "I can hear you" without moving the face.
            level = _clamp(self._level * 1.6)
            dash = r * 0.075
            gap = r * 0.085
            thick = max(1.2, r * 0.018)
            for side in (-1, 1):
                x_in = cx + side * (1.45 * r - r * 0.16)
                for i in range(5):
                    lit_k = _clamp(level * 3.2 - abs(i - 2) * 1.1)
                    y = cy - r * 0.1 + (i - 2) * gap
                    colour = _mix(_mix(accent, primary, 0.5), primary, lit_k)
                    colour.setAlpha(int((40 + 200 * lit_k) * p["eq"]))
                    pen = QPen(colour, thick)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    length = dash * (0.6 + 0.4 * lit_k)
                    painter.drawLine(QPointF(x_in, y), QPointF(x_in - side * length, y))

    def _paint_rim_pulse(self, painter: QPainter, primary: QColor, r: float) -> None:
        """The rim brightens with what he hears while listening."""
        level = self._level * self._p["eq"]
        if level < 0.02 or self._squircle is None:
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_alpha(primary, 120 * _clamp(level * 1.5)),
                            max(1.5, r * 0.035)))
        painter.drawPath(self._squircle)

    def _paint_face(self, painter: QPainter, cx: float, cy: float, r: float,
                    lit: QColor, primary: QColor, accent: QColor,
                    background: QColor, boot: float, power: float) -> None:
        """The eyes, the mouth and everything that rides with them, in
        face-local coordinates: origin at the visor's centre, r = 1 unit."""
        p = self._pe
        ambient = self._ambient
        # Breathing: a slow bob and a breath of scale. Reduced motion shrinks
        # it rather than stopping it — a frozen face reads as a hang.
        breath = math.sin(self._t * 2.0 * math.pi / _BREATH_PERIOD)
        bob = breath * r * 0.016 * (0.25 + 0.75 * ambient)
        # Speech: the eyes lift a touch with stressed syllables.
        bob -= self._speech * r * 0.03
        lean = p["lean"]
        scale = (1.0 + 0.035 * lean + 0.006 * breath * ambient)
        gx, gy = self._gaze

        painter.translate(cx + gx * r * 0.05, cy + bob - lean * r * 0.02)
        if abs(p["tilt"]) > 1e-3:
            painter.rotate(math.degrees(p["tilt"]))
        painter.scale(scale, scale)
        if power < 0.999:
            painter.setOpacity(_clamp(power * 3.0))

        boot_eyes = _back_out(_clamp((boot - 0.28) / 0.34)) if boot < 0.62 else 1.0
        boot_visible = boot >= 0.27
        mouth_on = _smooth(0.55, 0.85, boot)

        # Cheeks under everything else (a cartoon design only, in full).
        blush = p["blush"] * self._design.cute
        if blush > 0.02 and boot_visible:
            self._paint_blush(painter, r, primary, blush, gx, gy)

        if boot_visible:
            for side in (-1, 1):
                self._paint_eye(painter, side, r, lit, primary, accent,
                                background, boot_eyes, power, gx, gy)

        if mouth_on > 0.01:
            painter.save()
            painter.setOpacity(painter.opacity() * mouth_on)
            painter.translate(gx * r * 0.08, gy * r * 0.05)
            self._paint_mouth(painter, r, lit, primary, background)
            painter.restore()

        if self._zs:
            self._paint_zs(painter, r, lit)

    # ── eyes ─────────────────────────────────────────────────────────────────

    def _eye_box(self, side: int, r: float, gx: float, gy: float,
                 boot_eyes: float, power: float) -> tuple[float, float, float, float, float]:
        """(centre x, centre y, width, full height, open height) of one eye."""
        p = self._pe
        size = p["size"]
        size *= 1.0 + 0.12 * p["asym"] * (1 if side > 0 else -0.7)
        # Perspective: the eye on the side he is looking toward grows, the
        # other shrinks — the cheapest possible cue that the face has depth.
        size *= 1.0 + 0.09 * gx * side
        design = self._design
        width = r * 0.46 * size * p["w"] * design.eye_w
        height = r * 0.64 * size * p["h"] * design.eye_h * (1.0 - 0.07 * abs(gy))
        ex = side * r * (0.52 + 0.04 * max(0.0, design.eye_w - 1.0)) + gx * r * 0.16
        ey = -r * 0.1 + gy * r * 0.1
        # A raised brow opens the eye on its side a little wider (the other
        # eye's lid comes down instead - see _lid_drops).
        brow = p["brow"]
        if brow and side == (1 if brow > 0 else -1):
            height *= 1.0 + 0.12 * abs(brow)
            ey -= r * 0.02 * abs(brow)
        # Lids: the eye shuts toward its centre, as a display would draw it.
        opening = p["open"] * (1.0 - self._blink)
        line = max(1.6, r * 0.05)
        opened = max(line, height * opening)
        # Boot: the eye grows out of the scanline.
        opened = max(line * 0.6, opened * boot_eyes)
        # Power-off: collapse to a line, then to a point (an old CRT going off).
        if power < 0.999:
            opened = max(line * 0.5, opened * _clamp((power - 0.45) / 0.55))
            width *= _clamp(power / 0.45)
        # The squint: the cheeks push the LOWER lid up. The top stays where it
        # is, so the eye gets shorter from the bottom and its centre rises.
        squint = p["squint"]
        if squint > 0.01 and opened > line * 1.5:
            ey -= 0.15 * squint * opened
            opened *= 1.0 - 0.3 * squint
        return ex, ey, width, height, opened

    def _lid_drops(self, side: int) -> tuple[float, float]:
        """How far (as a fraction of the open eye) the upper lid comes down at
        this eye's (left end, right end).

        Inner end down reads as cross or determined; outer end down as sad or
        worried. The happy crescent overrides lids: an arch with its top
        sliced off reads as a doorway, not a smile.
        """
        p = self._pe
        # The puzzled squint is one lid, slanted: lower at the nose than at
        # the temple, so it reads as a raised-brow "hm?" and not as a smaller
        # eye.
        squint = p["squint_l"] if side < 0 else 0.0
        asym_lid = 0.12 * p["asym"] if side < 0 else 0.0
        base = p["lid_top"] + max(0.0, asym_lid)
        # The raised brow's other half: the eye on the far side gets a lid a
        # little lower at the nose, a quizzical half-lid.
        brow = p["brow"]
        quiz = 0.0
        if brow and side == (-1 if brow > 0 else 1):
            quiz = 0.12 * abs(brow)
        soften = 1.0 - 0.9 * _clamp(p["smile"] * 1.4)
        inner = (base + quiz + p["lid_in"] + squint) * soften
        outer = (base + quiz * 0.35 + p["lid_out"] + squint * 0.35) * soften
        # Screen-left end first: for the left eye (side -1) that is the outer
        # corner, for the right eye the inner one.
        return (outer, inner) if side < 0 else (inner, outer)

    @staticmethod
    def _rounded_corner(path: QPainterPath, before: QPointF, corner: QPointF,
                        after: QPointF) -> None:
        """Round ``corner`` between the current point ``before`` and ``after``
        with a cubic that approximates a circular arc at right angles and
        stays soft at any other angle — so a slanted lid never makes a point."""
        k = 0.5523
        path.lineTo(before)
        path.cubicTo(QPointF(before.x() + (corner.x() - before.x()) * k,
                             before.y() + (corner.y() - before.y()) * k),
                     QPointF(after.x() + (corner.x() - after.x()) * k,
                             after.y() + (corner.y() - after.y()) * k),
                     after)

    def _eye_path(self, side: int, ex: float, ey: float, width: float,
                  height: float, opened: float) -> QPainterPath:
        p = self._pe
        top = ey - opened / 2.0
        bottom = ey + opened / 2.0
        x0, x1 = ex - width / 2.0, ex + width / 2.0
        rounding = 1.0 if self._design.eye == "ring" else _clamp(
            p["round"] * self._design.round_scale)
        radius = min(width, opened) / 2.0 * (0.55 + 0.45 * rounding)

        drop_l, drop_r = self._lid_drops(side)
        if max(drop_l, drop_r) <= 0.01 or opened <= height * 0.15:
            path = QPainterPath()
            path.addRoundedRect(QRectF(x0, top, width, opened), radius, radius)
        else:
            # The eye drawn with its upper edge where the lid is: every corner
            # stays rounded, so a sad or cross eye is still a soft shape.
            yl = top + opened * _clamp(drop_l, 0.0, 0.85)
            yr = top + opened * _clamp(drop_r, 0.0, 0.85)
            rt = min(radius, (bottom - max(yl, yr)) * 0.5, width * 0.5)
            rb = min(radius, (bottom - max(yl, yr)) * 0.5, width * 0.5)
            dx, dy = x0 - x1, yl - yr
            length = max(1e-6, math.hypot(dx, dy))
            ux, uy = dx / length, dy / length        # along the lid, right->left
            path = QPainterPath(QPointF(x0, yl + rt))
            path.lineTo(QPointF(x0, bottom - rb))
            self._rounded_corner(path, QPointF(x0, bottom - rb), QPointF(x0, bottom),
                                 QPointF(x0 + rb, bottom))
            self._rounded_corner(path, QPointF(x1 - rb, bottom), QPointF(x1, bottom),
                                 QPointF(x1, bottom - rb))
            self._rounded_corner(path, QPointF(x1, yr + rt), QPointF(x1, yr),
                                 QPointF(x1 + ux * rt, yr + uy * rt))
            self._rounded_corner(path, QPointF(x0 - ux * rt, yl - uy * rt),
                                 QPointF(x0, yl), QPointF(x0, yl + rt))
            path.closeSubpath()

        # The happy crescent: an ellipse taken out from below leaves an arch,
        # the ^ of ^ ^.
        # A small smile is a cheek pushing the bottom of the eye up — a wide,
        # shallow cut, no legs — and only a big one becomes the arch. A
        # narrow cut at a small smile leaves a horseshoe, which reads as a
        # doorway rather than a smile.
        smile = p["smile"]
        if smile > 0.02 and opened > height * 0.2:
            cut_top = top + opened * (1.0 - 0.82 * smile)
            half_w = width * (1.3 - 0.72 * _clamp(smile))
            half_h = opened * 0.8
            cut = QPainterPath()
            cut.addEllipse(QRectF(ex - half_w, cut_top, 2 * half_w, 2 * half_h))
            path = path.subtracted(cut)
        return path

    @staticmethod
    def _heart_path(ex: float, ey: float, size: float) -> QPainterPath:
        poly = QPolygonF()
        for i in range(48):
            th = 2.0 * math.pi * i / 48
            x = 16.0 * math.sin(th) ** 3
            y = (13.0 * math.cos(th) - 5.0 * math.cos(2 * th)
                 - 2.0 * math.cos(3 * th) - math.cos(4 * th))
            poly.append(QPointF(ex + x * size / 34.0, ey - y * size / 34.0 - size * 0.04))
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()
        return path

    def _paint_eye(self, painter: QPainter, side: int, r: float, lit: QColor,
                   primary: QColor, accent: QColor, background: QColor,
                   boot_eyes: float, power: float, gx: float, gy: float) -> None:
        p = self._pe
        design = self._design
        ex, ey, width, height, opened = self._eye_box(side, r, gx, gy, boot_eyes, power)
        if width < 0.5:
            return
        # Heart eyes are a cartoon's: a sleeker design shows love as warmth.
        heart = p["heart"] if design.cute >= 0.5 else 0.0
        path = self._eye_path(side, ex, ey, width, height, opened)
        if heart > 0.02 and opened > height * 0.3:
            heart_path = self._heart_path(ex, ey, width * 1.25 * (0.8 + 0.2 * heart))
        else:
            heart_path = None
            heart = 0.0

        # Halo: the eye is a light, so it lights the glass around it.
        span = max(width, opened) * 1.05 + r * 0.08
        halo = QRadialGradient(ex, ey, span)
        halo.setColorAt(0.0, _alpha(lit, 70 * design.halo))
        halo.setColorAt(0.5, _alpha(lit, 24 * design.halo))
        halo.setColorAt(1.0, _alpha(lit, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(QPointF(ex, ey), span, span)

        # Body: lit from above, like a glowing pill behind glass. A ring is a
        # dim lens with a bright rim and a lit pupil instead.
        body = QLinearGradient(ex, ey - height / 2.0, ex, ey + height / 2.0)
        body.setColorAt(0.0, _mix(lit, _WHITE, 0.32))
        body.setColorAt(0.45, lit)
        body.setColorAt(1.0, _mix(lit, background, 0.18))
        opacity = painter.opacity()
        if heart < 0.98:
            painter.setOpacity(opacity * (1.0 - heart))
            if design.eye == "ring":
                self._paint_ring(painter, path, ex, ey, width, opened, lit, background)
            else:
                painter.setBrush(body)
                painter.drawPath(path)
                self._paint_core_and_edge(painter, path, ex, ey, width, opened, lit, accent)
        if heart_path is not None:
            painter.setOpacity(opacity * heart)
            painter.setBrush(body)
            painter.drawPath(heart_path)
        painter.setOpacity(opacity)

        # The silver highlight: ORION's accent, small and high on the outer
        # side, so both eyes catch the same light. Hidden when the eye is a
        # line or an arch — a glint on a closed eye is the uncanny kind.
        openness = opened / max(1e-6, height)
        show = _clamp((openness - 0.45) / 0.3) * (1.0 - _clamp(p["smile"] * 1.6))
        show *= (1.0 - heart) * design.glint
        if show > 0.02 and width > 6.0:
            # Upper-left on BOTH eyes — one light source — and always below
            # wherever the lid is, so a sad eye keeps its glint.
            hx = ex - width * 0.2
            drop_l, drop_r = self._lid_drops(side)
            u = 0.3                                 # hx sits 30 % across
            lid = (drop_l * (1.0 - u) + drop_r * u)
            hy = ey - opened * 0.5 + opened * (0.13 + _clamp(lid, 0.0, 0.85))
            gw = width * 0.24
            gh = max(1.0, min(opened * 0.13, width * 0.16))
            glint = QPainterPath()
            glint.addRoundedRect(QRectF(hx - gw / 2.0, hy, gw, gh), gh * 0.5, gh * 0.5)
            dot = max(0.8, width * 0.05)
            glint.addEllipse(QPointF(hx + width * 0.16, hy + gh + dot * 1.6), dot, dot)
            if lid > 0.01 or p["smile"] > 0.02:
                glint = glint.intersected(path)
            painter.setBrush(_alpha(accent, 215 * show))
            painter.drawPath(glint)

        sparkle = p["sparkle"] * (1.0 - heart) * design.cute
        if sparkle > 0.04 and design.cute >= 0.3 and openness > 0.4 and width > 7.0:
            self._paint_sparkle(painter, side, ex, ey, width, opened, path, sparkle)

    def _paint_core_and_edge(self, painter: QPainter, path: QPainterPath, ex: float,
                             ey: float, width: float, opened: float, lit: QColor,
                             accent: QColor) -> None:
        """The HUD designs' detail: a bright core line through the eye (it
        follows the lids, being clipped to the eye) and a crisp silver edge."""
        design = self._design
        if design.core > 0.02 and opened > 3.0:
            painter.save()
            painter.setClipPath(path, Qt.ClipOperation.IntersectClip)
            band = max(1.0, opened * (0.16 if design.eye == "blade" else 0.3))
            core = QLinearGradient(ex - width / 2.0, ey, ex + width / 2.0, ey)
            bright = _mix(lit, _WHITE, 0.55)
            core.setColorAt(0.0, _alpha(bright, 0))
            core.setColorAt(0.5, _alpha(bright, 220 * design.core))
            core.setColorAt(1.0, _alpha(bright, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(core)
            painter.drawRect(QRectF(ex - width / 2.0, ey - band / 2.0 + opened * 0.06,
                                    width, band))
            painter.restore()
        if design.outline > 0.02:
            pen = QPen(_alpha(accent, 210 * design.outline), max(1.0, width * 0.035))
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen)

    def _paint_ring(self, painter: QPainter, path: QPainterPath, ex: float, ey: float,
                    width: float, opened: float, lit: QColor, background: QColor) -> None:
        """Arc's eye: a dim lens, a bright rim and a lit pupil that follows
        the gaze. Clipped to the eye's shape, so lids and the smile cut it
        exactly as they cut a pill."""
        painter.save()
        painter.setClipPath(path, Qt.ClipOperation.IntersectClip)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_mix(background, lit, 0.28))
        painter.drawPath(path)
        rim = max(1.5, width * 0.11)
        pen = QPen(_mix(lit, _WHITE, 0.12), rim)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        half = max(1.0, width / 2.0 - rim / 2.0)
        painter.drawEllipse(QPointF(ex, ey), half, half)
        painter.setPen(Qt.PenStyle.NoPen)
        gx, gy = self._gaze
        pupil = width * 0.2 * (0.85 + 0.25 * self._design.core)
        centre = QPointF(ex + gx * width * 0.12, ey + gy * width * 0.1)
        glow = QRadialGradient(centre, pupil * 1.9)
        glow.setColorAt(0.0, _alpha(_mix(lit, _WHITE, 0.6), 255))
        glow.setColorAt(0.5, _alpha(lit, 200))
        glow.setColorAt(1.0, _alpha(lit, 0))
        painter.setBrush(glow)
        painter.drawEllipse(centre, pupil * 1.9, pupil * 1.9)
        painter.restore()

    # ── eye sparkle ──────────────────────────────────────────────────────────

    @staticmethod
    def _star(cx: float, cy: float, size: float) -> QPolygonF:
        """A four-pointed twinkle: long thin points, a narrow waist."""
        waist = size * 0.24
        poly = QPolygonF()
        for x, y in ((0.0, -size), (waist, -waist), (size, 0.0), (waist, waist),
                     (0.0, size), (-waist, waist), (-size, 0.0), (-waist, -waist)):
            poly.append(QPointF(cx + x, cy + y))
        return poly

    def _paint_sparkle(self, painter: QPainter, side: int, ex: float, ey: float,
                       width: float, opened: float, path: QPainterPath,
                       amount: float) -> None:
        """Happy eyes twinkle: a bright four-point glint, and a smaller one on
        the other side that wakes a beat later. Each rides a slow sine so they
        swell and shrink out of step; a glint is only drawn where it lands ON
        the eye (an arched eye has a thin top, so it is tried a little lower
        and higher), so it can never float off the glass."""
        t = self._t
        top = ey - opened / 2.0
        big = 0.55 + 0.45 * math.sin(t * 3.1 + side * 1.3)
        small = 0.55 + 0.45 * math.sin(t * 2.3 + 2.1 - side * 0.9)
        # An occasional ping: the glint flares.
        ping = max(0.0, math.sin(t * 0.9 + side * 0.7) - 0.8) * 5.0
        placements = (
            (ex + width * 0.2, ey - opened * 0.18, width * 0.17 * (big + 0.5 * ping), 255.0),
            (ex - width * 0.2, ey + opened * 0.16, width * 0.1 * (small + 0.4 * ping), 215.0),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        for index, (x, y, size, alpha) in enumerate(placements):
            if size < 1.0:
                continue
            spot = QPointF(x, y)
            if not path.contains(spot):
                spot = QPointF(x, top + opened * (0.12 if index == 0 else 0.2))
                if not path.contains(spot):
                    continue
            painter.setBrush(_alpha(_WHITE, alpha * amount))
            painter.drawPolygon(self._star(spot.x(), spot.y(), size))

    # ── cheeks ───────────────────────────────────────────────────────────────

    def _paint_blush(self, painter: QPainter, r: float, primary: QColor,
                     amount: float, gx: float, gy: float) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        for side in (-1, 1):
            bx = side * r * 0.66 + gx * r * 0.12
            by = r * 0.31 + gy * r * 0.08
            rx, ry = r * 0.2, r * 0.09
            glow = QRadialGradient(bx, by, rx)
            glow.setColorAt(0.0, _alpha(primary, 120 * amount))
            glow.setColorAt(0.6, _alpha(primary, 50 * amount))
            glow.setColorAt(1.0, _alpha(primary, 0))
            painter.setBrush(glow)
            painter.save()
            painter.translate(bx, by)
            painter.scale(1.0, ry / rx)
            painter.translate(-bx, -by)
            painter.drawEllipse(QPointF(bx, by), rx, rx)
            painter.restore()
            # Three tiny strokes: the cartoon blush, which reads on a
            # one-colour display where a pink tint cannot. Cartoon designs only.
            if r >= 45.0 and self._design.cute >= 0.5:
                pen = QPen(_alpha(_mix(primary, _WHITE, 0.25), 150 * amount),
                           max(1.0, r * 0.016))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                for k in (-1, 0, 1):
                    x0 = bx + k * r * 0.06
                    painter.drawLine(QPointF(x0 + r * 0.018, by - r * 0.03),
                                     QPointF(x0 - r * 0.018, by + r * 0.03))
                painter.setPen(Qt.PenStyle.NoPen)

    # ── mouth ────────────────────────────────────────────────────────────────

    def _paint_mouth(self, painter: QPainter, r: float, lit: QColor,
                     primary: QColor, background: QColor) -> None:
        p = self._pe
        y = r * 0.44
        line = max(1.5, r * 0.036)

        # Thinking dots take the mouth's place: the face's "typing...".
        dots = p["dots"]
        if dots > 0.02:
            painter.setPen(Qt.PenStyle.NoPen)
            for i in range(3):
                wave = max(0.0, math.sin(self._t * 4.4 - i * 0.85))
                alpha = (95 + 160 * wave) * dots
                radius = r * (0.036 + 0.01 * wave)
                centre = QPointF((i - 1) * r * 0.13, y - wave * r * 0.04)
                halo = QRadialGradient(centre, radius * 2.6)
                halo.setColorAt(0.0, _alpha(lit, 60 * dots * (0.4 + 0.6 * wave)))
                halo.setColorAt(1.0, _alpha(lit, 0))
                painter.setBrush(halo)
                painter.drawEllipse(centre, radius * 2.6, radius * 2.6)
                painter.setBrush(_alpha(_mix(lit, _WHITE, 0.15 * wave), alpha))
                painter.drawEllipse(centre, radius, radius)

        visible = p["mouth_vis"]
        if visible < 0.02:
            return
        opacity = painter.opacity()
        painter.setOpacity(opacity * visible)
        try:
            speech_open = self._open * (1.0 - self._close)
            held_o = p["mouth_o"] * 0.55
            opening = max(speech_open, held_o)
            rounded = _clamp(max(-self._wide, 0.0) + p["mouth_o"] * 1.2)
            half = r * 0.15 * p["mouth_w"] * (1.0 + 0.4 * max(self._wide, 0.0))
            half *= 1.0 - 0.45 * rounded * _clamp(opening * 3.0)
            half *= 1.0 - 0.15 * self._close
            curve = p["mouth"] * r * 0.085 * (1.0 - 0.6 * self._close)
            depth = opening * r * 0.22 * (1.0 + 0.3 * rounded)

            glow_pen = QPen(_alpha(lit, 46), line * 2.8)
            glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            pen = QPen(_mix(lit, _WHITE, 0.08), line)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

            if p["zip"] > 0.5:
                # Muted: the mouth stitched shut. Flat, with three stitches.
                path = QPainterPath(QPointF(-half, y))
                path.lineTo(QPointF(half, y))
                for stroke_pen in (glow_pen, pen):
                    painter.setPen(stroke_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawPath(path)
                stitch = QPen(_mix(lit, _WHITE, 0.08), max(1.0, line * 0.6))
                stitch.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(stitch)
                for k in (-1, 0, 1):
                    x = k * half * 0.55
                    painter.drawLine(QPointF(x, y - r * 0.045), QPointF(x, y + r * 0.045))
                return

            style = self._design.mouth
            if style == "bar":
                # The meter keeps its width through rounded vowels: it is a
                # level display, not lips.
                width = max(half, r * 0.15 * p["mouth_w"]) * 1.25
                self._paint_bar_mouth(painter, r, y, width, curve, depth, line,
                                      lit, glow_pen, pen)
                return
            if style == "wave":
                self._paint_wave_mouth(painter, r, y, half * 1.25, curve, depth, glow_pen, pen)
                return

            if depth < line * 0.9:
                # Closed: one soft line with a curve — the resting smile.
                path = QPainterPath(QPointF(-half, y - curve * 0.5))
                wave = p["wave"]
                if wave > 0.05:
                    steps = 14
                    for i in range(1, steps + 1):
                        u = i / steps
                        x = -half + 2 * half * u
                        base = (4.0 * u * (1.0 - u)) * curve
                        wobble = math.sin(u * math.pi * 3.0) * r * 0.022 * wave
                        path.lineTo(QPointF(x, y - curve * 0.5 + base + wobble))
                else:
                    path.quadTo(QPointF(0.0, y + curve * 1.5), QPointF(half, y - curve * 0.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for stroke_pen in (glow_pen, pen):
                    painter.setPen(stroke_pen)
                    painter.drawPath(path)
                return

            # Open: a soft D — the upper edge keeps the expression's curve, the
            # lower one drops with the sound — becoming an oval for rounded
            # vowels. Filled dark so the opening reads as an opening. Cubics,
            # not quadratics: a quadratic lower lip makes a V, and a V-shaped
            # mouth reads as a tongue.
            top_y = y - curve * 0.5
            lift = depth * 0.55 * rounded           # an "o" rises at the top too
            sag = curve * 0.9
            path = QPainterPath(QPointF(-half, top_y))
            path.cubicTo(QPointF(-half * 0.55, top_y + sag - lift),
                         QPointF(half * 0.55, top_y + sag - lift),
                         QPointF(half, top_y))
            reach = depth * 1.33 + sag * 0.6
            path.cubicTo(QPointF(half * 1.02, top_y + reach),
                         QPointF(-half * 1.02, top_y + reach),
                         QPointF(-half, top_y))
            path.closeSubpath()
            painter.setPen(glow_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(pen)
            painter.setBrush(_mix(background, primary, 0.42))
            painter.drawPath(path)
        finally:
            painter.setOpacity(opacity)

    #: Segments in the light-bar mouth's level meter.
    _BAR_SEGMENTS = 9

    def _paint_bar_mouth(self, painter: QPainter, r: float, y: float, half: float,
                         curve: float, depth: float, line: float, lit: QColor,
                         glow_pen: QPen, pen: QPen) -> None:
        """Sentinel and Arc: a light bar with square brackets at its ends. At
        rest it bends a little with the feeling; speaking, it becomes a level
        meter - segments rising with the voice, the low, middle and high
        bands of the sound across it, tallest in the middle."""
        bend = curve * 0.6
        tick = r * 0.035
        if depth < line * 0.9:
            path = QPainterPath(QPointF(-half, y - bend * 0.5))
            path.quadTo(QPointF(0.0, y + bend * 1.5), QPointF(half, y - bend * 0.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for stroke_pen in (glow_pen, pen):
                painter.setPen(stroke_pen)
                painter.drawPath(path)
        else:
            low, mid, high = self._spectrum
            count = self._BAR_SEGMENTS
            step = 2.0 * half / count
            seg_w = step * 0.62
            painter.setPen(Qt.PenStyle.NoPen)
            for i in range(count):
                u = (i + 0.5) / count                       # 0..1 across
                centre_weight = 1.0 - abs(u - 0.5) * 1.3
                band = (low * max(0.0, 1.0 - 2.0 * u) + mid * (1.0 - abs(u - 0.5) * 2.0)
                        + high * max(0.0, 2.0 * u - 1.0))
                level = _clamp(0.35 + 0.65 * centre_weight + 0.35 * band
                               + 0.08 * math.sin(self._t * 23.0 + i * 1.7), 0.15, 1.3)
                h = max(line, depth * 0.9 * level)
                x = -half + i * step + (step - seg_w) / 2.0
                base = y - bend * 0.5 + 4.0 * u * (1.0 - u) * bend
                painter.setBrush(_alpha(lit, 60))
                painter.drawRoundedRect(QRectF(x - seg_w * 0.25, base - h / 2.0 - line * 0.5,
                                               seg_w * 1.5, h + line), seg_w * 0.5, seg_w * 0.5)
                painter.setBrush(_mix(lit, _WHITE, 0.1 + 0.25 * level))
                painter.drawRoundedRect(QRectF(x, base - h / 2.0, seg_w, h),
                                        seg_w * 0.4, seg_w * 0.4)
        bracket = QPen(_alpha(pen.color(), 150), max(1.0, line * 0.6))
        bracket.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(bracket)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for side in (-1, 1):
            x = side * (half + r * 0.05)
            edge = y - bend * 0.5
            painter.drawLine(QPointF(x, edge - tick), QPointF(x, edge + tick))
            painter.drawLine(QPointF(x, edge - tick), QPointF(x - side * tick * 0.6, edge - tick))
            painter.drawLine(QPointF(x, edge + tick), QPointF(x - side * tick * 0.6, edge + tick))

    def _paint_wave_mouth(self, painter: QPainter, r: float, y: float, half: float,
                          curve: float, depth: float, glow_pen: QPen, pen: QPen) -> None:
        """Visor: an oscilloscope trace. A shallow curve at rest; speaking, a
        travelling wave whose height follows the voice, pinned at both ends."""
        steps = 40
        amplitude = depth * 0.55
        low, mid, high = self._spectrum
        cycles = 2.0 + 1.5 * high + 0.5 * mid
        path = QPainterPath()
        for i in range(steps + 1):
            u = i / steps
            x = -half + 2.0 * half * u
            envelope = math.sin(math.pi * u)
            base = y - curve * 0.5 + 4.0 * u * (1.0 - u) * curve
            wave = amplitude * envelope * math.sin(2.0 * math.pi * cycles * u - self._t * 9.0)
            wave += amplitude * 0.35 * low * envelope * math.sin(2.0 * math.pi * u * 0.8)
            point = QPointF(x, base + wave)
            if i == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for stroke_pen in (glow_pen, pen):
            painter.setPen(stroke_pen)
            painter.drawPath(path)

    # ── z's ──────────────────────────────────────────────────────────────────

    def _paint_zs(self, painter: QPainter, r: float, lit: QColor) -> None:
        sleep = self._p["sleep"]
        if sleep < 0.05:
            return
        for age, seed in self._zs:
            u = age / 2.8
            size = r * (0.09 + 0.09 * u)
            x = r * 0.86 + u * r * 0.3 + math.sin(age * 2.1 + seed) * r * 0.04
            y = -r * 0.3 - u * r * 0.48
            alpha = 240 * sleep * _smooth(0.0, 0.15, u) * (1.0 - _smooth(0.7, 1.0, u))
            pen = QPen(_alpha(_mix(lit, _WHITE, 0.15), alpha), max(1.4, size * 0.16))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            half = size / 2.0
            z = QPainterPath(QPointF(x - half, y - half))
            z.lineTo(QPointF(x + half, y - half))
            z.lineTo(QPointF(x - half, y + half))
            z.lineTo(QPointF(x + half, y + half))
            painter.drawPath(z)

    # ── caption ──────────────────────────────────────────────────────────────

    def _paint_label(self, painter: QPainter, cx: float, cy: float, r: float,
                     primary: QColor, accent: QColor, boot: float) -> None:
        if self._label_alpha < 0.02 or not self._label or r < 50.0:
            return
        px = max(9, int(r * 0.085))
        if self._font is None or self._font_px != px:
            font = QFont()
            font.setFamilies(list(_MONO_FAMILIES))
            font.setStyleHint(QFont.StyleHint.Monospace)
            font.setPixelSize(px)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, px * 0.28)
            self._font = font
            self._font_px = px
        font = self._font
        metrics = QFontMetricsF(font)
        text = metrics.elidedText(self._label.upper(), Qt.TextElideMode.ElideRight,
                                  2.6 * r)
        y = cy + r + r * 0.34
        alpha = 175 * self._label_alpha * _smooth(0.6, 1.0, boot)
        painter.setFont(font)
        painter.setPen(_alpha(accent, alpha))
        rect = QRectF(cx - 1.45 * r, y - px, 2.9 * r, px * 2.0)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)
        text_w = metrics.horizontalAdvance(text)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_alpha(primary, 200 * self._label_alpha))
        painter.drawEllipse(QPointF(cx - text_w / 2.0 - px * 0.9, y), px * 0.18, px * 0.18)


#: Every screen face alive, held weakly, so ONE Reduced-motion listener can
#: serve them all. Per-panel listeners needed unregistering when a face was
#: retired, and the only place to do that was the ``destroyed`` signal — whose
#: Python slots PyQt can run from inside garbage collection, mid-teardown,
#: which crashed the process (access violation) rather than merely leaking.
_LIVE_PANELS: "weakref.WeakSet[Any]" = weakref.WeakSet()
_MOTION_HOOKED = False


def apply_design(name: Any) -> int:
    """Put every live Screen Face (the main window's, the dock's) in design
    *name* now. Returns how many faces changed. GUI thread only."""
    changed = 0
    for panel in list(_LIVE_PANELS):
        try:
            if panel.set_design(name):
                changed += 1
        except RuntimeError:
            continue                    # a panel mid-teardown
    return changed


def _on_motion_profile(profile: Any) -> None:
    for panel in list(_LIVE_PANELS):
        try:
            panel._to_head("set_ambient", float(profile.ambient))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            pass


class ScreenFacePanel(QWidget):
    """Hosts a :class:`ScreenFace` and speaks ORION's face protocol.

    The same shape as HoloHeadPanel, on purpose: ``core_window._to_face``
    calls set_state / set_amplitude / set_speaking / set_viseme / glance /
    apply_emotion on whichever face is installed, the AvatarController adds
    set_pose / set_spectrum / set_morph / set_label, and each is guarded with
    hasattr — so a method that exists only on the rig inside is silently a
    no-op. Every one of them is therefore defined HERE, on the panel.

    The rig is handed to the face pipeline's worker on first show and from
    then on is never touched from this thread: every call below is queued.
    """

    supports_orb_hands = False

    def __init__(self, parent: QWidget | None = None, *, fps: int = 30) -> None:
        super().__init__(parent)
        self._head = ScreenFace()
        self._amplitude = 0.0
        self._speaking = False
        self._state = "IDLE"
        self._last = time.perf_counter()
        self._fps = float(max(1, fps))
        #: Offscreen buffer for the capped synchronous path.
        self._buffer: QImage | None = None

        # ORION's colours from the first frame — see test_orion_is_red: a face
        # that only becomes crimson when Preferences is opened is not ORION's.
        self._primary = QColor(C.PRI)
        self._accent = QColor(C.ACCENT)
        self._background = QColor(C.BG)

        # `timer` is public and part of the face protocol (overlay mode retimes
        # it by name; see HoloHeadPanel).
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(max(1, int(1000 / self._fps)))

        from .animation_budget import AnimationBudget

        self._idle_hz = 12.0
        self._budget = AnimationBudget(self.timer, active_hz=self._fps,
                                       idle_hz=self._idle_hz)

        self._pipeline: Any = None
        self._threaded = face_thread_enabled()
        self._blit_generation = -1
        self._sent_kwargs: tuple | None = None
        self._sent_geometry: tuple | None = None
        if headless():
            # Nobody can watch it boot, and the timer is stopped below, so a
            # render here must show the settled face, not frame one.
            self._head.skip_boot()
            self.timer.stop()

        #: The last frame blitted, kept so a worker restarted on show has
        #: something to put up before its first frame lands — never a black
        #: flash between hiding and showing.
        self._last_frame: QImage | None = None
        self._follow_motion_profile()
        # The owner's chosen design (appearance.face_design), before the
        # worker owns the rig - so the first frame already wears it.
        try:
            from .. import appearance
            self._head.set_design(appearance.face_design())
        except Exception:
            pass
        self.setMinimumSize(120, 100)
        self.setAccessibleName("ORION's face")

    # ── reduced motion ───────────────────────────────────────────────────────

    def _follow_motion_profile(self) -> None:
        """Take the Reduced-motion scale now and whenever it changes.

        Through the module's one listener and a weak set of live panels, so a
        retired face is simply forgotten — nothing to unregister."""
        global _MOTION_HOOKED
        try:
            from .. import appearance
        except Exception:
            return
        try:
            self._head.set_ambient(appearance.motion_profile().ambient)
        except Exception:
            pass
        _LIVE_PANELS.add(self)
        if not _MOTION_HOOKED:
            appearance.on_motion_change(_on_motion_profile)
            _MOTION_HOOKED = True

    # ── pipeline lifecycle ───────────────────────────────────────────────────

    def _ensure_pipeline(self) -> Any:
        """Start the render thread, once, when there is something to draw.
        A pipeline whose last stop timed out is resumed, not replaced: its
        worker still owns the rig."""
        if not self._threaded:
            return self._pipeline
        if self._pipeline is not None:
            if self._pipeline.stopping:
                self._pipeline.start()
            return self._pipeline
        try:
            from .face_pipeline import FaceRenderPipeline

            pipeline = FaceRenderPipeline(self._head, fps=self._fps)
            pipeline.set_palette(self._primary, self._accent, self._background)
            pipeline.start()
            self._pipeline = pipeline
        except Exception:
            # A face that does not appear is worse than a face that costs the
            # loop: fall back to painting on this thread.
            self._threaded = False
            self._pipeline = None
        return self._pipeline

    def showEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        super().showEvent(event)
        self._ensure_pipeline()

    def hideEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        """Hidden faces do not render. This is also how a retired face's
        worker ends: core_window._retire_face reparents the widget to None,
        which hides it, before deleteLater — and a worker must never outlive
        the widget it draws for. Overlay mode hides the window, so the face
        rests there too, and resumes on show."""
        self.stop_pipeline()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        self.stop_pipeline()
        super().closeEvent(event)

    def stop_pipeline(self) -> None:
        """Stop the render thread and wait briefly for it (joined, never
        abandoned — see HoloHeadPanel.stop_pipeline)."""
        pipeline = self._pipeline
        if pipeline is None:
            return
        try:
            finished = pipeline.stop()
        except Exception:
            finished = not getattr(pipeline, "running", False)
        if finished is not False:
            self._pipeline = None
            self._blit_generation = -1     # the next worker counts from zero
        self._sent_kwargs = None
        self._sent_geometry = None

    def _to_head(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Apply a change to the rig — queued if the worker owns it."""
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.call(name, *args, **kwargs)
            return
        method = getattr(self._head, name, None)
        if method is not None:
            method(*args, **kwargs)

    # ── ORION's face protocol ────────────────────────────────────────────────

    def set_state(self, state: Any) -> None:
        self._state = str(state or "").upper()
        self._budget.wake()

    @property
    def state_name(self) -> str:
        """The state last set - CentralHud's name for it, so the compact
        window can be asked the same question whichever face it holds."""
        return self._state

    def set_amplitude(self, value: Any) -> None:
        try:
            self._amplitude = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            self._amplitude = 0.0

    def set_speaking(self, active: Any) -> None:
        self._speaking = bool(active)
        if not self._speaking:
            self._to_head("set_viseme", 0.0, 0.0, 0.0)
        self._budget.wake()

    def set_viseme(self, openness: Any, width: Any = 0.0, closure: Any = 0.0) -> None:
        """Accepts a viseme ID with a weight, or a raw (open, width, closure).

        Both shapes arrive (core_window._face_set_viseme): the bus contract is
        ``{"viseme": "PP", "weight": 1.0}``; the audio path measures openness
        and width from the spectrum and has no name to give.
        """
        if isinstance(openness, str):
            posture = VISEMES.get(openness, VISEMES.get(openness.lower(), _REST_MOUTH))
            try:
                weight = max(0.0, min(1.0, float(width) if width else 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            rounded = max(0.0, posture.round - 0.5) * 0.8
            self._to_head(
                "set_viseme",
                posture.open * weight,
                ((posture.width - 0.5) * 2.0 - rounded) * weight,
                # "PP" is p/b/m, lips pressed shut — the one posture the
                # spectrum can never supply, so it must survive the trip.
                weight if openness.upper() == "PP" else 0.0)
            self._budget.wake()
            return
        try:
            self._to_head("set_viseme", float(openness), float(width),
                          float(closure))
        except (TypeError, ValueError):
            return
        self._budget.wake()

    def glance(self, dx: Any = 0.0, dy: Any = -0.65, hold: Any = 1.2) -> None:
        """Look somewhere for a moment — used to acknowledge new content."""
        try:
            self._to_head("glance", float(dx), float(dy), float(hold))
        except (TypeError, ValueError):
            return
        self._budget.wake()

    def apply_emotion(self, name: Any, params: Any = None) -> None:
        """Wear an emotion. Shape, not colour: the palette stays the owner's."""
        self._to_head("apply_emotion", str(name or "neutral"),
                      dict(params) if isinstance(params, dict) else None)
        self._budget.wake()

    def set_label(self, text: Any) -> None:
        self._to_head("set_label", str(text or ""))

    def set_design(self, name: Any) -> str:
        """Wear another Screen Face design (DESIGNS) now. The rig's own
        set_design runs on whichever thread owns it; the key is returned
        from here so callers need not wait for the worker."""
        key = str(name or "").strip().lower()
        if key not in DESIGNS:
            return ""
        self._to_head("set_design", key)
        self._budget.wake()
        self.update()
        return key

    def set_pose(self, yaw: Any, pitch: Any) -> None:
        try:
            self._to_head("set_pose", float(yaw), float(pitch))
        except (TypeError, ValueError):
            pass

    def set_spectrum(self, low: Any, mid: Any, high: Any) -> None:
        try:
            self._to_head("set_spectrum", float(low), float(mid), float(high))
        except (TypeError, ValueError):
            pass

    def set_morph(self, value: Any) -> None:
        """The controller's dormancy channel: below one half, he rests."""
        try:
            self._to_head("set_morph", float(value))
        except (TypeError, ValueError):
            pass

    def set_orb_camera_enabled(self, enabled: Any) -> None:
        """Accept the shared face command; there is no orb on this face."""

    def set_orb_hands(self, sample: Any) -> None:
        """Accept shared orb hand samples; this face has no orb surface."""

    def set_palette_colours(self, primary: str | QColor, accent: str | QColor,
                            background: str | QColor | None = None) -> None:
        for target, value in (("_primary", primary), ("_accent", accent),
                              ("_background", background)):
            if value is None:
                continue
            colour = QColor(value) if not isinstance(value, QColor) else QColor(value)
            if colour.isValid():
                setattr(self, target, colour)
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.set_palette(self._primary, self._accent, self._background)
        self._budget.wake()
        self.update()

    # ── rendering ────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        pipeline = self._pipeline
        # Hidden or minimised: nothing is shown, so nothing is paid. The worker
        # is slowed to a crawl rather than stopped (a stop joins a thread), and
        # the clock resets so resuming picks up instead of lurching.
        if not self.isVisible():
            self._last = time.perf_counter()
            if pipeline is not None:
                pipeline.fps = 1.0
            return
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        head = self._head
        if pipeline is not None:
            kwargs = (round(self._amplitude, 3), self._speaking, self._state)
            if kwargs != self._sent_kwargs:
                # Only when something changed: every call wakes the worker,
                # and an unchanged world does not need a new frame early.
                pipeline.set_step_kwargs(amplitude=self._amplitude,
                                         speaking=self._speaking,
                                         state=self._state)
                self._sent_kwargs = kwargs
            geometry = self._draw_geometry()
            if geometry != self._sent_geometry:
                pipeline.set_geometry(*geometry)
                self._sent_geometry = geometry
        else:
            head.step(dt, self._amplitude, speaking=self._speaking,
                      state=self._state)

        # The channels that carry meaning — expression, mouth, blink, power,
        # loudness. Gaze and breathing are deliberately excluded: they always
        # move, and a budget that can never settle saves nothing.
        channels = head.channels + (self._amplitude, float(self._speaking),
                                    float(hash(self._state) & 0xFFFF))
        self._budget.note(channels)
        if head.anticipating:
            # A blink about to start, the boot, the thinking dots: full rate
            # BEFORE the motion, or its first frames are lost at the idle rate.
            self._budget.wake()
        if pipeline is not None:
            # The worker follows the budget too, or it would keep drawing 30
            # identical frames a second for a blit that happens 12 times.
            pipeline.fps = self._idle_hz if self._budget.idling else self._fps
            if pipeline.frames.generation == self._blit_generation:
                return          # nothing new to show
        self.update()

    def _draw_geometry(self) -> tuple[int, int, float, float, float]:
        """The frame the worker should render: the widget in DEVICE pixels, so
        a high-DPI screen gets a crisp face, capped at MAX_RENDER_PIXELS."""
        try:
            dpr = max(1.0, float(self.devicePixelRatioF()))
        except Exception:
            dpr = 1.0
        w = max(8.0, self.width() * dpr)
        h = max(8.0, self.height() * dpr)
        scale = min(1.0, math.sqrt(MAX_RENDER_PIXELS / (w * h)))
        bw, bh = max(8, int(w * scale)), max(8, int(h * scale))
        cx, cy, r = face_geometry(bw, bh)
        return (bw, bh, cx, cy, r)

    def paintEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        pipeline = self._pipeline
        if pipeline is not None:
            frame, generation = pipeline.frames.take()
            shown = frame if frame is not None else self._last_frame
            painter = QPainter(self)
            try:
                if shown is None:
                    painter.fillRect(self.rect(), self._background)
                else:
                    painter.setRenderHint(
                        QPainter.RenderHint.SmoothPixmapTransform, True)
                    painter.drawImage(QRectF(self.rect()), shown)
            finally:
                painter.end()
            if frame is not None:
                self._blit_generation = generation
                self._last_frame = frame
            # Until the first frame arrives, the background. The worker owns
            # the rig; painting it here would race it.
            return

        painter = QPainter(self)
        try:
            w, h = self.width(), self.height()
            if w < 8 or h < 8:
                painter.fillRect(self.rect(), self._background)
                return
            bw, bh, cx, cy, r = self._draw_geometry()
            buffer = self._buffer
            if buffer is None or buffer.width() != bw or buffer.height() != bh:
                buffer = QImage(bw, bh, QImage.Format.Format_RGB32)
                self._buffer = buffer
            buffer.fill(self._background)
            inner = QPainter(buffer)
            try:
                inner.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                self._head.paint(inner, cx, cy, r, self._primary, self._accent,
                                 self._background)
            finally:
                inner.end()
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(QRectF(self.rect()), buffer)
        finally:
            painter.end()

    @property
    def last_paint_ms(self) -> float:
        return self._head.last_paint_ms

    @property
    def render_thread(self) -> threading.Thread | None:
        """The worker drawing this face, if any — for diagnostics and tests."""
        pipeline = self._pipeline
        return getattr(pipeline, "_thread", None) if pipeline is not None else None


__all__ = ["BOOT_SECONDS", "EMOTIONS", "MAX_RENDER_PIXELS", "MODES", "NEUTRAL",
           "ScreenFace", "ScreenFacePanel", "emotion_preset", "face_geometry",
           "face_thread_enabled", "headless", "state_mode"]
