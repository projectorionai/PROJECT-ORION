"""
Application-wide constants for O.R.I.O.N. Mark VIII.

Everything configurable-but-static lives here: model identifiers, audio
geometry, wake words, filesystem anchors, the colour palette, and the
permanent voice profile.  No module-level side effects beyond path
resolution — importing this module is always safe.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# IDENTITY
# ──────────────────────────────────────────────────────────────────────────────

APP_NAME = "O.R.I.O.N. Mark X.5"
APP_SUBTITLE = "OPEN RESOLUTION INTELLIGENCE OVERT NETWORK"

# ──────────────────────────────────────────────────────────────────────────────
# LIVE MODEL CHANNEL
# ──────────────────────────────────────────────────────────────────────────────

LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
LIVE_MODEL_FALLBACKS = (
    "models/gemini-2.5-flash-native-audio-preview-12-2025",
    "models/gemini-2.5-flash-preview-native-audio-dialog",
    "models/gemini-2.0-flash-live-001",
)

# ──────────────────────────────────────────────────────────────────────────────
# AUDIO GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────

SEND_SAMPLE_RATE = 16_000
RECEIVE_SAMPLE_RATE = 24_000
CHANNELS = 1
CHUNK_SIZE = 512
VAD_SAMPLE_LIMIT = 256
MIC_QUEUE_LIMIT = 150

WAKE_WORDS = ("orion", "o'rion", "oh rye on", "oh ryan", "orien", "o rion")
WAKE_WINDOW_SECONDS = 45.0
BARGE_IN_CONFIDENCE = 0.80
VOICE_HANGOVER_SECONDS = 1.5

# Spoken pause / resume control.  Saying a pause word silences ORION and puts
# him in a listening-only PAUSED state; a resume word (or the wake word) brings
# him back — "zone back in".  The primary resume word is configurable.
PAUSE_WORDS = ("pause", "hold on", "hold that thought", "one moment", "stand by",
               "give me a moment", "zone out", "be quiet", "silence please")
RESUME_WORDS = ("resume", "continue", "carry on", "zone back in", "zone in",
                "unpause", "i'm back", "im back", "wake up", "you there")

# True voice interruption (Mark X.5).  These wake-word-prefixed commands are
# matched by a dedicated grammar-constrained listener that stays live even
# while ORION is speaking, so he can always be silenced or resumed by voice.
# Pause-family phrases HOLD playback (queue position preserved); resume-family
# phrases continue from the exact interruption point — nothing is regenerated.
INTERRUPT_PAUSE_PHRASES = (
    "orion pause", "orion stop", "orion wait", "orion hold on", "orion silence",
)
INTERRUPT_RESUME_PHRASES = (
    "orion continue", "orion resume",
)
# Cooldown between honoured interruption commands so one spoken phrase caught
# by both the partial and final recogniser passes never double-fires.
INTERRUPT_COOLDOWN_SECONDS = 1.5

# How long after the final playback chunk ORION is still considered to be
# speaking.  Generous enough to absorb device-buffer drain so the microphone
# never reopens on the tail of ORION's own sentence (echo / self-trigger).
PLAYBACK_TAIL_SECONDS = 0.60

# Deterministic buffering (Mark IX): accumulate a small prebuffer before the
# first write of a fresh utterance so a cold audio device cannot underrun and
# clip ORION's opening syllable.  Capped by PLAYBACK_PREBUFFER_MAX_WAIT so the
# prebuffer never adds perceptible latency to the start of speech.
PLAYBACK_PREBUFFER_CHUNKS = 3
PLAYBACK_PREBUFFER_MAX_WAIT = 0.12
# Soft high-water mark: queue depth beyond this is logged, never dropped —
# model audio is always played in full (dropping time-compresses speech).
PLAYBACK_QUEUE_HIGH_WATER = 400


# ──────────────────────────────────────────────────────────────────────────────
# VOICE PROFILE  — permanent, professional, male.  Never switched at runtime.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VoiceProfile:
    """
    The single authoritative voice identity for ORION.

    Frozen dataclass: nothing at runtime may mutate the voice.  Both speech
    paths (native Gemini audio and the local pyttsx3/SAPI fallback) read from
    this profile, so ORION sounds identical regardless of which channel is
    speaking.
    """

    # Gemini Live prebuilt voice — "Charon": deep, calm, professional male.
    gemini_voice_name: str = "Charon"
    # Local SAPI voice search order (first match wins) — professional UK/US
    # male voices only, so a fallback never flips ORION to a female voice.
    local_voice_patterns: tuple[str, ...] = (
        r"(?i)\bryan\b",            # Microsoft Ryan (en-GB, male)
        r"(?i)\bgeorge\b",          # Microsoft George (en-GB, male)
        r"(?i)en[-_ ]?(gb|uk).*male",
        r"(?i)\bdavid\b",           # Microsoft David (en-US, male)
        r"(?i)\bjames\b",
        r"(?i)\bmark\b",
        r"(?i)\bmale\b",
    )
    local_rate_wpm: int = 172        # measured, unhurried delivery
    local_volume: float = 1.0

    def describe(self) -> str:
        return (
            f"native voice '{self.gemini_voice_name}' (locked), "
            f"local fallback male profile at {self.local_rate_wpm} wpm"
        )


VOICE_PROFILE = VoiceProfile()

# Barge-in (interrupting ORION mid-sentence by voice) is disabled by default:
# ORION always finishes speaking before listening resumes.  Power users can
# re-enable the old behaviour with ORION_ALLOW_BARGE_IN=1.
ALLOW_BARGE_IN = os.getenv("ORION_ALLOW_BARGE_IN", "").strip().lower() in {
    "1", "true", "yes", "on",
}

# ──────────────────────────────────────────────────────────────────────────────
# STARTUP GREETINGS
# ──────────────────────────────────────────────────────────────────────────────

STARTUP_GREETINGS = (
    "Good {period}, sir. All systems are online and at your disposal.",
    "Good {period}, sir. ORION is fully operational — a pleasure to be back.",
    "Welcome back, sir. Diagnostics read green across the board.",
    "At your service, sir. All systems nominal and standing by.",
    "Good {period}, sir. The network is awake and awaiting your command.",
    "Systems restored, sir. Shall we get to work?",
    "Good {period}, sir. All channels secure; running at full capacity.",
    "Back online, sir. Everything is precisely where you left it.",
    "Good {period}, sir. Power at one hundred percent and holding steady.",
)

# ──────────────────────────────────────────────────────────────────────────────
# FILESYSTEM ANCHORS
# ──────────────────────────────────────────────────────────────────────────────

PACKAGE_DIR = Path(__file__).resolve().parent          # …/orion_core
BASE_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else PACKAGE_DIR.parent                             # project root
)
CONFIG_DIR = BASE_DIR / "config"
API_CONFIG_PATH = CONFIG_DIR / "api_keys.json"
CORE_DB_PATH = CONFIG_DIR / "orion_core.db"

# Paths the security layer refuses to mutate: the launcher and this package.
LAUNCHER_PATH = BASE_DIR / "orion.py"
CORE_SCRIPT_PATH = LAUNCHER_PATH                        # legacy alias


def is_protected_path(path: Path) -> bool:
    """True when *path* is ORION's own code and must never be written to."""
    try:
        resolved = path.resolve()
    except Exception:
        return True  # unresolvable paths are treated as hostile
    if resolved == LAUNCHER_PATH:
        return True
    return resolved == PACKAGE_DIR or PACKAGE_DIR in resolved.parents


# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (British English identifiers throughout)
# ──────────────────────────────────────────────────────────────────────────────

class C:
    # ── Crimson identity (unchanged core hues) ──────────────────────────────
    PRI      = "#ff1a3c"   # crimson primary
    PRI_DIM  = "#991024"   # crimson dim
    PRI_HI   = "#ff4d68"   # crimson bright (hover / highlight)
    CORE     = "#ff6478"   # incandescent orb core
    # ── Surfaces ────────────────────────────────────────────────────────────
    BG       = "#050508"   # deep void
    INK      = "#07070b"   # console / recess black
    PANEL    = "#0f0f14"   # panel surface
    PANEL_HI = "#16161f"   # raised panel / glass top-stop
    BORDER   = "#2a1118"   # subtle border
    # ── Futuristic accent (electric cyan — holographic rim + data lines) ────
    ACCENT      = "#00e5ff"  # electric cyan
    ACCENT_DIM  = "#0a7d8c"  # cyan dim
    ACCENT_DEEP = "#063b45"  # cyan shadow
    # ── Text ────────────────────────────────────────────────────────────────
    WHITE   = "#ffffff"
    MUTED   = "#a9a9b2"
    FAINT   = "#5c5c68"    # tertiary / axis labels
    # ── Status ──────────────────────────────────────────────────────────────
    GOOD    = "#3ddc84"    # nominal
    WARN    = "#ffb020"    # caution
    BAD     = "#ff4444"    # fault

    # ── Amber / gold (circuit-board face eyes + solder nodes) ───────────────
    AMBER     = "#ffb020"
    GOLD      = "#ffcd5a"

    # RGB tuples for painter code that blends alpha dynamically.
    PRI_RGB    = (255, 26, 60)
    CORE_RGB   = (255, 100, 120)
    ACCENT_RGB = (0, 229, 255)
    AMBER_RGB  = (255, 176, 32)
    GOLD_RGB   = (255, 205, 90)
