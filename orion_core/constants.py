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

#: The mark, in Roman numerals, in ONE place. Everything that shows a version
#: derives it from here — the shell's title bar, the swarm map's caption, the
#: Command Deck, the briefing, the standalone build stamp. It had drifted
#: before this was centralised: the window said Mark XXV while the swarm map
#: still said MARK XXI, which is the sort of thing nobody notices for a year.
APP_MARK_NUMBER = 32
APP_MARK = "Mark XXXII"

APP_NAME = f"O.R.I.O.N. {APP_MARK}"
APP_SUBTITLE = "OPEN RESOLUTION INTELLIGENCE OVERT NETWORK"
# Build stamp — shown in the header so a running instance can be told apart
# from a stale one at a glance (bump on each build so "am I on the new code?"
# is answerable without guessing).
APP_BUILD = "2026.09.25-xxxii"

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
PAUSE_WORDS = ("pause", "hold on", "hold that thought", "one moment",
               "stop listening", "go to sleep",
               "give me a moment", "zone out", "be quiet", "silence please")
RESUME_WORDS = ("resume", "continue", "carry on", "zone back in", "zone in",
                "start listening", "listen up",
                "unpause", "i'm back", "im back", "wake up", "you there")

# Spoken CANCEL control.  Unlike pause (a hold that resumes in place), a cancel
# word scraps the current request entirely and returns ORION to listening so the
# user can rephrase from scratch — the spoken twin of the STOP button.
CANCEL_WORDS = ("cancel that", "cancel", "never mind", "nevermind", "scrap that",
                "forget it", "forget that", "start over", "start again",
                "scratch that", "belay that")

# Presence check + standby (Mark X.10).  When the microphone input flatlines
# (a dead / muted / wrong device — NOT a merely quiet room, which still has a
# noise floor) while ORION is meant to be listening, he no longer silently
# hops between microphones every 30 seconds.  Instead he asks ONCE whether the
# user is still there; if there is no response within the grace window he drops
# to STANDBY and simply waits for a command, rather than talking to an empty
# room.  Device switching happens only on the user's explicit say-so (see the
# OUTPUT/INPUT switch phrases below).
PRESENCE_SILENCE_SECONDS = float(os.getenv("ORION_PRESENCE_SILENCE", "30") or 30.0)
PRESENCE_GRACE_SECONDS = float(os.getenv("ORION_PRESENCE_GRACE", "12") or 12.0)
PRESENCE_PROMPTS = (
    "Are you still there?",
    "Still with me? I've gone quiet on my end.",
    "I can't hear anything — are you still there?",
)

# Spoken/typed audio-device controls.  "I can't hear you" means the USER cannot
# hear ORION → rotate the OUTPUT (speaker/headphones).  "You can't hear me" or
# "switch microphone" means the INPUT is wrong → rotate the microphone.  These
# are the only triggers that change an audio device; the passive watchdog never
# switches devices on its own any more.
OUTPUT_SWITCH_WORDS = (
    "can't hear you", "cant hear you", "can not hear you", "cannot hear you",
    "i can't hear you", "i cant hear you", "you're muted", "youre muted",
    "no sound", "wrong speaker", "wrong speakers", "wrong output",
    "change the output", "change output", "switch output", "switch the output",
    "switch speaker", "switch speakers", "change speaker", "change speakers",
    "different speaker", "different output", "use another speaker",
)
INPUT_SWITCH_WORDS = (
    "you can't hear me", "you cant hear me", "you cannot hear me",
    "switch microphone", "change microphone", "switch the mic", "change the mic",
    "switch mic", "change mic", "wrong microphone", "wrong mic",
    "different microphone", "different mic", "use another microphone",
    "use another mic", "change the microphone",
)

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

# Device-buffer depth requested from PortAudio for ORION's voice, in seconds.
#
# The renderer is a Python thread and WILL be descheduled when the GUI is busy;
# measured on this machine, three competing Python threads stretched a 341 ms
# device write to 2311 ms.  The audio path cannot prevent that — it can only
# carry enough buffered audio to play through it.  A third of a second of extra
# latency is imperceptible in a spoken reply; speech breaking up is not.
#
# Override with ORION_AUDIO_LATENCY (seconds) on hardware that wants a
# different trade-off.
PLAYBACK_DEVICE_LATENCY_S = max(
    0.05, min(1.0, float(os.getenv("ORION_AUDIO_LATENCY", "0.30") or 0.30)))

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
# SAFEGUARDS — destructive-action confirmation, globe zoom, resource monitor
# ──────────────────────────────────────────────────────────────────────────────

# How long a pending destructive-action confirmation (OS shutdown/restart/
# logout/sleep/hibernate, file deletion, process termination) stays valid before
# it expires and must be re-requested.  A short window defeats stale-approval and
# replay attempts.
SYSTEM_CONFIRM_TTL_SECONDS = float(os.getenv("ORION_CONFIRM_TTL", "60") or 60.0)

# Bounded safeguards for max_zoom_in_globe (see orion_core.globe_zoom).
GLOBE_ZOOM_MAX_ATTEMPTS = int(os.getenv("ORION_GLOBE_ZOOM_MAX_ATTEMPTS", "40") or 40)
GLOBE_ZOOM_MAX_SECONDS = float(os.getenv("ORION_GLOBE_ZOOM_MAX_SECONDS", "20") or 20.0)
GLOBE_ZOOM_DELAY_S = float(os.getenv("ORION_GLOBE_ZOOM_DELAY_S", "0.05") or 0.05)

# Resource-pressure monitor thresholds (see orion_core.resource_monitor).  A
# reading must hold ABOVE a level for its duration window before ORION reacts, so
# a momentary spike from another application never interrupts an active task.
RESOURCE_ELEVATED_CPU = float(os.getenv("ORION_RES_CPU_ELEVATED", "85") or 85.0)
RESOURCE_CRITICAL_CPU = float(os.getenv("ORION_RES_CPU_CRITICAL", "95") or 95.0)
RESOURCE_ELEVATED_MEM = float(os.getenv("ORION_RES_MEM_ELEVATED", "85") or 85.0)
RESOURCE_CRITICAL_MEM = float(os.getenv("ORION_RES_MEM_CRITICAL", "95") or 95.0)
RESOURCE_ELEVATED_HOLD_S = float(os.getenv("ORION_RES_ELEVATED_HOLD_S", "20") or 20.0)
RESOURCE_CRITICAL_HOLD_S = float(os.getenv("ORION_RES_CRITICAL_HOLD_S", "10") or 10.0)


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
# re-enable it with ORION_ALLOW_BARGE_IN=1.  When enabled, a candidate
# interruption must (a) sustain qualifying VAD confidence for at least
# BARGE_IN_MIN_VOICED_MS, (b) clear BARGE_IN_MIN_AMPLITUDE (rejects quiet
# speaker bleed), and (c) decode to words that are NOT a match for what
# ORION is currently saying (see live_worker._is_own_echo) — a single VAD
# blip is not evidence of a real interruption; ORION's own voice echoing
# into the mic is real, grammatical speech too, so word-decoding alone
# can't discriminate it without comparing against his own utterance.
ALLOW_BARGE_IN = os.getenv("ORION_ALLOW_BARGE_IN", "").strip().lower() in {
    "1", "true", "yes", "on",
}
BARGE_IN_MIN_VOICED_MS = float(os.getenv("ORION_BARGE_IN_MIN_MS", "350"))
BARGE_IN_MIN_AMPLITUDE = float(os.getenv("ORION_BARGE_IN_MIN_AMPLITUDE", "0.03"))

# ──────────────────────────────────────────────────────────────────────────────
# STARTUP GREETINGS
# ──────────────────────────────────────────────────────────────────────────────

STARTUP_GREETINGS = (
    "Good {period}. All systems are online and at your disposal.",
    "Good {period}. ORION is fully operational — a pleasure to be back.",
    "Welcome back. Diagnostics read green across the board.",
    "At your service. All systems nominal and standing by.",
    "Good {period}. The network is awake and awaiting your command.",
    "Systems restored. Shall we get to work?",
    "Good {period}. All channels secure; running at full capacity.",
    "Back online. Everything is precisely where you left it.",
    "Good {period}. Power at one hundred percent and holding steady.",
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
#: Where files BUNDLED with the application live, which is not where writable
#: files live. PyInstaller 6 unpacks data into `_internal/` beside the
#: executable and exposes it as `sys._MEIPASS`; BASE_DIR is the executable's
#: own folder, so `BASE_DIR / "assets"` finds nothing in a frozen build.
#:
#: That distinction is silent when you get it wrong: the asset is simply
#: absent and whatever depended on it falls back. ORION's avatar would have
#: streamed three.js from a CDN in every standalone build while the vendored
#: copy sat unused inside the bundle.
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", "")) if getattr(
    sys, "frozen", False) and getattr(sys, "_MEIPASS", "") else BASE_DIR


def resource_path(*parts: str) -> Path:
    """A file that ships WITH ORION, wherever this build keeps them.

    Tries the frozen resource root first, then the source tree, so the same
    call works from a checkout and from the standalone.
    """
    for root in (RESOURCE_DIR, BASE_DIR, PACKAGE_DIR.parent):
        candidate = Path(root).joinpath(*parts)
        if candidate.exists():
            return candidate
    return Path(RESOURCE_DIR).joinpath(*parts)


#: Names the file that redirects ORION's writable data somewhere else.
#: It sits beside the application, holds one absolute path, and is written by
#: ``tools/move_config.py``.
CONFIG_POINTER_NAME = "config_location.txt"


def _resolve_config_dir() -> Path:
    """Where ORION keeps everything he writes.

    Historically this was always ``BASE_DIR / "config"`` — inside the project
    folder, which on this machine is inside OneDrive. That is a genuinely bad
    place for live SQLite: in WAL mode each store is three files (.db, .db-wal,
    .db-shm) and a sync client uploads and restores them independently. A .db
    recovered without its matching WAL is the database as it stood at the last
    checkpoint, and everything since is simply gone — with no error anywhere,
    because SQLite opened a perfectly valid file. That is the amnesia.

    Resolution order, most explicit first:

      1. ``ORION_CONFIG_DIR`` — an environment override, for one-off runs.
      2. A pointer file beside the application. Chosen over a hard-coded new
         default so the move is reversible by deleting one small text file,
         and so it applies per-installation: the frozen app can point at real
         data outside the sync folder while a source checkout — and the test
         suite — keep their own, which is the difference between running the
         tests and running them over your own memory.
      3. The legacy in-project directory.

    Never raises. A pointer that names somewhere unusable is ignored rather
    than allowed to stop ORION starting.
    """
    override = os.getenv("ORION_CONFIG_DIR", "").strip()
    if override:
        try:
            chosen = Path(override).expanduser()
            chosen.mkdir(parents=True, exist_ok=True)
            return chosen
        except Exception:
            pass
    try:
        pointer = BASE_DIR / CONFIG_POINTER_NAME
        if pointer.is_file():
            target = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
            if target.is_dir():
                return target
    except Exception:
        pass
    return BASE_DIR / "config"


CONFIG_DIR = _resolve_config_dir()


def _resolve_key_store() -> Path:
    """Where the API keys live — ONE store, outside the synced project folder.

    Copies of api_keys.json used to sit in the OneDrive project folder (and a
    stale one under config/config/ that _rescue_api_config would "restore"),
    which uploaded live keys to the cloud. They were removed; the installed
    app's store at ~/ORION/config is the only one. A source run with no store
    of its own now reads AND writes that one rather than creating a new copy
    in the project. Never under pytest: the suite must not see real keys.
    """
    local = CONFIG_DIR / "api_keys.json"
    if local.exists() or "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST"):
        return local
    if CONFIG_DIR != BASE_DIR / "config":
        return local                      # an explicit config dir owns its keys
    installed = Path.home() / "ORION" / "config" / "api_keys.json"
    return installed if installed.exists() else local


API_CONFIG_PATH = _resolve_key_store()
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
    """Near-monochrome surfaces; crimson is the only colour on screen.

    The neutrals are deliberately hue-free. The previous ramp was quietly
    blue (#050608, #0e1013, #242a31), which made crimson read faintly purple
    and the whole interface read as "a colour scheme". A true neutral lets one
    saturated hue sit cleanly on top of it, which is the mechanism this look
    depends on entirely.
    """

    # ── the one colour ──────────────────────────────────────────────────────
    # Crimson earns its place by being rare. It marks the live thing: the
    # active state, the focused control, the fault. Chrome never uses it.
    PRI      = "#ff2038"   # crimson — the only hue in the interface
    PRI_DIM  = "#8c1420"   # crimson, receded (pressed, disabled, trailing)
    PRI_HI   = "#ff5a6b"   # crimson, lifted (hover, peak, incandescence)
    CORE     = "#ff7a86"   # orb core — the brightest crimson anywhere

    # ── the neutral ramp ────────────────────────────────────────────────────
    # A single achromatic scale. Every surface is a step on it, and depth is
    # read from where a panel sits on the ramp plus the shadow beneath it —
    # not from a border drawn around it.
    BG       = "#060606"   # the void behind everything
    INK      = "#0a0a0a"   # recessed: consoles, wells, inputs
    PANEL    = "#161616"   # resting surface
    PANEL_HI = "#212121"   # raised surface — a panel above a panel
    BORDER   = "#2c2c2c"   # hairline, where a hard edge is genuinely needed

    # ── light ───────────────────────────────────────────────────────────────
    # Cold white, not silver-blue. Used for rim light, focus and data lines.
    ACCENT      = "#ededed"  # the lit edge
    ACCENT_DIM  = "#8a8a8a"  # secondary line work
    ACCENT_DEEP = "#3a3a3a"  # the shadowed side of an edge

    # ── text ────────────────────────────────────────────────────────────────
    WHITE   = "#fafafa"
    MUTED   = "#9a9a9a"
    FAINT   = "#5e5e5e"    # tertiary, axis labels, timestamps
    SILVER  = "#c8c8c8"    # body text

    # ── status, carried by position on the ramp ─────────────────────────────
    # Nominal recedes; degraded steps forward; a fault gets the one colour,
    # because if there is exactly one it belongs on the thing that most needs
    # to be looked at. HEALTH_GLYPHS (● ◐ ○) carries the same distinction in
    # shape, so none of this depends on seeing colour at all.
    GOOD    = "#c8c8c8"    # nominal — quiet, recedes
    WARN    = "#f0f0f0"    # degraded — brighter, steps forward
    # A fault is the accent pushed to full intensity, so on the default theme
    # it reads as the same one colour. It is deliberately NOT the same token
    # as PRI: live theming recolours the accent family by substituting hex
    # strings, and an identical string would make a fault chip turn green
    # along with everything else — indistinguishable from "live", which is
    # precisely the state it exists to be distinguished from.
    BAD     = "#ff0a24"    # fault — crimson at full intensity, never themed

    # ── amber / gold ────────────────────────────────────────────────────────
    # Kept for the circuit-board face's eyes and solder nodes, which are a
    # rendered OBJECT rather than interface chrome. Nothing in the UI uses
    # them: a second hue in the chrome is what this redesign removes.
    AMBER     = "#ffb020"
    GOLD      = "#ffcd5a"

    # RGB tuples for painter code that blends alpha dynamically.
    PRI_RGB    = (255, 32, 56)
    CORE_RGB   = (255, 122, 134)
    ACCENT_RGB = (237, 237, 237)
    SILVER_RGB = (200, 200, 200)
    AMBER_RGB  = (255, 176, 32)
    GOLD_RGB   = (255, 205, 90)


class ELEV:
    """Depth, as a scale rather than a decoration.

    Qt Style Sheets cannot blur or cast a shadow, so the old stylesheet faked
    depth with gradient pairs and bright/dim border tricks — which reads as a
    drawing of depth rather than depth. These are real values for
    QGraphicsDropShadowEffect, applied by ``gui.depth.lift()``.

    A panel's elevation says what it IS: resting content, something raised
    above it, something floating over everything. Three steps, because a scale
    with more than three is one nobody applies consistently.
    """

    #: (blur radius, y-offset, shadow alpha 0-255)
    #: Measured by rendering it: against a near-black field a drop shadow is
    #: almost invisible, because a shadow is a darkening and there is nothing
    #: left to darken. So depth here is carried by SURFACE BRIGHTNESS, and the
    #: shadow is reserved for things that genuinely float above the page —
    #: where it separates them from content rather than from the void.
    REST   = (0, 0, 0)          # flat on the surface — no shadow at all
    RAISED = (18, 4, 90)        # a panel above the field — barely, on purpose
    FLOAT  = (48, 12, 210)      # a menu or dialog, over CONTENT: this one reads

    #: Surface colour at each step. Light gathers on what is nearer.
    SURFACE = {0: C.INK, 1: C.PANEL, 2: C.PANEL_HI}

    #: Corner radius by elevation. Bigger radii read as softer and nearer,
    #: which is the whole grammar of this look.
    RADIUS = {0: 8, 1: 14, 2: 20}


class GLASS:
    """Translucency values for the layered surfaces.

    Real backdrop blur is a Windows compositor feature, not a Qt one — see
    ``gui.backdrop``. These alphas are what makes the layering read even when
    the compositor declines: a panel you can faintly see through sits ON
    something, while an opaque one is just a rectangle.
    """

    PANEL   = 0.72    # a resting panel over the window backdrop
    RAISED  = 0.82    # a raised panel — less transparent, reads as nearer
    FLOAT   = 0.94    # a floating surface — nearly solid, it owns the view
    HAIRLINE = 0.10   # the 1px light edge along a panel's top


# ──────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM TOKENS  (Mark XX redesign spec, §9 — High Impact)
#
# Before this: the OK/DEGRADED/DOWN status triad was independently defined
# three times with three different colour values (C.GOOD/WARN/BAD here,
# command_centre.py's own inline triad, diagnostics_centre.py's own third
# pair); 83 setContentsMargins/setSpacing calls across 15 gui files used
# hand-picked pixel tuples with no shared scale; font sizes were literal
# px values inside style.py's QSS string with no named scale; and the same
# "typewriter" thought-streaming effect was tuned to two different speeds
# (24ms and 22ms) in two different files for what is visually one effect.
# None of that was a matter of taste — it was an unfinished system. This
# section finishes it: one status triad, one spacing scale, one type scale,
# one motion vocabulary, everything else in the app migrates onto these.
# ──────────────────────────────────────────────────────────────────────────────

# Status triad — the single source for OK/DEGRADED/DOWN (or good/warn/bad)
# anywhere in the app. Reuses C.GOOD/WARN/BAD so there is exactly one
# definition of "what does green mean here" instead of three.
HEALTH_COLOURS: dict[str, str] = {"OK": C.GOOD, "DEGRADED": C.WARN, "DOWN": C.BAD}
HEALTH_GLYPHS: dict[str, str] = {"OK": "●", "DEGRADED": "◐", "DOWN": "○"}


class SPACE:
    """The four-step spacing scale — replaces hand-picked pixel tuples.

    Use SPACE.N for setContentsMargins()/setSpacing() instead of a literal;
    a panel's density should say something (this is denser, this is looser)
    rather than vary by which file happened to write it.
    """
    XS = 4    # inline gaps — icon to label, chip padding
    SM = 8    # related-control gaps — a button row, a form field group
    MD = 16   # panel internal padding — the default panel margin
    LG = 24   # between unrelated panels/sections on the same page


class TYPE:
    """Named type-scale sizes (px), replacing literal QSS font-size values.

    Three roles, not free variation per file: TITLE for page titles,
    HEADING for panel headers (uppercase, tracked — see style.py), BODY for
    prose/controls, DATA for the monospace convention already used
    instinctively for file paths/IDs/counters in three separate files
    before this token existed.
    """
    TITLE_PX   = 23
    HEADING_PX = 12
    BODY_PX    = 12
    DATA_PX    = 12
    MONO_FAMILY = '"Cascadia Mono", "Consolas", monospace'


class MOTION:
    """The animation-duration vocabulary — replaces scattered literals.

    FAST is new (hover/press feedback had no named duration before); PAGE
    and STREAM already existed as unnamed literals — PAGE was already
    correct (unified_dashboard.py's 220ms cross-fade) and just needed a
    name, STREAM unifies the two competing "typewriter" speeds (24ms in
    ops_deck.py, 22ms in command_centre.py) that were tuning the same
    visual effect independently. REFRESH_* names the three existing
    poll-interval tiers (mission_deck.py 2s, token_graph.py 15s,
    ops_deck.py 30s) — kept as-is, they're already sensibly staggered by
    data volatility, just not documented as a deliberate system before.
    """
    FAST_MS   = 120
    PAGE_MS   = 220
    STREAM_MS = 24
    REFRESH_FAST_MS = 2_000
    REFRESH_MED_MS  = 15_000
    REFRESH_SLOW_MS = 30_000
