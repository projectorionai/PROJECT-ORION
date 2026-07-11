"""
Emotion engine (Mark X.7) — state-driven emotional rendering.

Aligned to the ORION 3D Voxel Emotive Face specification: every emotion maps
to a complete rendering profile — voxel density, brow position, eye shape,
mouth shape and tension, colour palette, glow intensity (steady or pulsing)
and particle behaviour (speed, direction, dissolution, turbulence) — and the
face applies whatever arrives.  Nothing visual is hardcoded in the widgets.

Two co-operating pieces, both pure services (no widget is ever imported):

    SentimentAnalyser    — deterministic, offline classification of text into
                           {sentiment, confidence, intensity, reason}.  Zero
                           latency, zero tokens; identical for cloud, local
                           and offline replies — and for the USER's words,
                           so ORION can look concerned when you sound
                           stressed, not merely when he says so.

    EmotionStateManager  — the single authority on ORION's emotional state.
                           Listens to the bus (pipeline state + sentiment),
                           resolves one of eleven named emotions, scales its
                           expression by the sentiment's intensity, and
                           broadcasts ``bus.emotion_changed(name, params)``.

Dynamic behaviour rules honoured here: transitions are smooth and organic
(the face eases; the manager never flip-flops below the confidence floor);
sudden jumps are reserved for high urgency (urgent/critical bypasses the
hold-over); particle energy follows emotional energy; and the current
register is exposed to the ProviderRouter so emotion can colour ORION's
voice tone, pacing and word choice.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from .bus import OrionBus


# ──────────────────────────────────────────────────────────────────────────────
# EMOTION PROFILES — the rendering parameter sets.
# Palettes are (dark, mid, bright) skin stops + an iris/glow accent.
# Multipliers sit around 1.0, offsets around 0.0; the face applies them to its
# own baseline geometry and eases every change.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmotionProfile:
    name: str
    voxel_density: float = 1.0      # <1 dispersed … >1 condensed
    brow: float = 0.0               # -1 furrowed … +1 raised
    eye_width: float = 1.0          # widened / narrowed
    eye_height: float = 1.0         # openness multiplier (softened / tightened)
    mouth_curve: float = 0.0        # -1 downturned … +1 upturned
    mouth_tension: float = 0.0      # 0 relaxed … 1 tense / compressed
    glow: float = 0.55              # overall luminance 0..1
    glow_pulse: float = 0.0         # 0 steady … 1 strongly pulsing
    speed: float = 1.0              # animation-rate multiplier
    particle_velocity: float = 1.0  # drift-cube speed multiplier
    particle_direction: str = "drift"   # drift | up | down | burst
    turbulence: float = 0.1         # 0 laminar … 1 chaotic
    palette_dark: tuple[int, int, int] = (9, 16, 34)
    palette_mid: tuple[int, int, int] = (34, 84, 150)
    palette_bright: tuple[int, int, int] = (150, 214, 246)
    accent: tuple[int, int, int] = (0, 229, 255)

    def params(self) -> dict[str, Any]:
        return asdict(self)


_BLUE = dict(palette_dark=(9, 16, 34), palette_mid=(34, 84, 150),
             palette_bright=(150, 214, 246), accent=(0, 229, 255))
_BRIGHT_BLUE = dict(palette_dark=(10, 20, 44), palette_mid=(70, 140, 220),
                    palette_bright=(205, 240, 255), accent=(140, 250, 255))
_DEEP_BLUE = dict(palette_dark=(6, 12, 30), palette_mid=(24, 60, 120),
                  palette_bright=(110, 170, 230), accent=(0, 190, 235))
_AMBER = dict(palette_dark=(30, 18, 6), palette_mid=(140, 92, 28),
              palette_bright=(255, 196, 110), accent=(255, 176, 32))
_AMBER_RED = dict(palette_dark=(36, 10, 6), palette_mid=(165, 62, 26),
                  palette_bright=(255, 140, 80), accent=(255, 92, 40))
_RED = dict(palette_dark=(38, 8, 10), palette_mid=(160, 40, 36),
            palette_bright=(255, 116, 92), accent=(255, 64, 64))
_VIOLET = dict(palette_dark=(14, 12, 26), palette_mid=(58, 50, 96),
               palette_bright=(140, 126, 180), accent=(150, 130, 210))

# The six specification presets, extended to the full eleven-state set the
# pipeline uses (listening/speaking/alert/critical/excited stay coherent
# with their nearest preset family).
EMOTIONS: dict[str, EmotionProfile] = {
    # NEUTRAL — calm, balanced, attentive; blue/cyan, medium glow, slow drift.
    "neutral": EmotionProfile(
        "neutral", voxel_density=1.0, brow=0.0, eye_width=1.0, eye_height=0.75,
        mouth_curve=0.05, mouth_tension=0.0, glow=0.55, glow_pulse=0.0,
        speed=0.9, particle_velocity=0.8, particle_direction="drift",
        turbulence=0.1, **_BLUE),
    # HAPPINESS — slightly LOWER density (the form loosens), raised brows,
    # wider eyes, upturned mouth, bright blue/white, high glow, energetic
    # upward particle flow.
    "happy": EmotionProfile(
        "happy", voxel_density=0.92, brow=0.55, eye_width=1.12, eye_height=1.05,
        mouth_curve=0.6, mouth_tension=0.0, glow=0.95, glow_pulse=0.1,
        speed=1.15, particle_velocity=1.4, particle_direction="up",
        turbulence=0.3, **_BRIGHT_BLUE),
    "excited": EmotionProfile(
        "excited", voxel_density=0.88, brow=0.7, eye_width=1.18, eye_height=1.15,
        mouth_curve=0.7, mouth_tension=0.0, glow=1.0, glow_pulse=0.3,
        speed=1.35, particle_velocity=1.8, particle_direction="up",
        turbulence=0.45, **_BRIGHT_BLUE),
    # CONCERN / EMPATHY — medium density, slightly furrowed brows, softened
    # eyes, slightly downturned mouth, amber/soft orange, gentle slow drift.
    "concerned": EmotionProfile(
        "concerned", voxel_density=1.0, brow=-0.35, eye_width=1.0, eye_height=0.8,
        mouth_curve=-0.3, mouth_tension=0.1, glow=0.55, glow_pulse=0.0,
        speed=0.8, particle_velocity=0.6, particle_direction="drift",
        turbulence=0.05, **_AMBER),
    # FOCUS / THINKING — HIGH density (condensed attention), slightly lowered
    # brows, narrowed eyes, deep blue, controlled subtle particle movement.
    "thinking": EmotionProfile(
        "thinking", voxel_density=1.15, brow=-0.45, eye_width=0.92, eye_height=0.6,
        mouth_curve=-0.02, mouth_tension=0.15, glow=0.55, glow_pulse=0.0,
        speed=1.1, particle_velocity=0.7, particle_direction="drift",
        turbulence=0.05, **_DEEP_BLUE),
    # FRUSTRATION / ANGER — high condensed density, lowered furrowed brows,
    # narrowed eyes, tense mouth, amber/red, high PULSING glow, fast turbulent
    # outward bursts.
    "frustrated": EmotionProfile(
        "frustrated", voxel_density=1.25, brow=-0.8, eye_width=0.85, eye_height=0.55,
        mouth_curve=-0.45, mouth_tension=0.7, glow=0.85, glow_pulse=0.8,
        speed=1.3, particle_velocity=1.8, particle_direction="burst",
        turbulence=0.8, **_AMBER_RED),
    "critical": EmotionProfile(
        "critical", voxel_density=1.3, brow=-0.6, eye_width=1.1, eye_height=1.05,
        mouth_curve=-0.35, mouth_tension=0.6, glow=1.0, glow_pulse=1.0,
        speed=1.4, particle_velocity=2.0, particle_direction="burst",
        turbulence=0.9, **_RED),
    # SADNESS — lower density, inner-raised brows, drooped eyes, downturned
    # mouth, muted blue/violet, low glow, slow downward fading particles.
    "sad": EmotionProfile(
        "sad", voxel_density=0.85, brow=0.2, eye_width=0.9, eye_height=0.45,
        mouth_curve=-0.5, mouth_tension=0.0, glow=0.3, glow_pulse=0.0,
        speed=0.65, particle_velocity=0.45, particle_direction="down",
        turbulence=0.05, **_VIOLET),
    # Pipeline states.
    "listening": EmotionProfile(
        "listening", voxel_density=1.05, brow=0.35, eye_width=1.06, eye_height=1.0,
        mouth_curve=0.12, mouth_tension=0.0, glow=0.75, glow_pulse=0.0,
        speed=1.0, particle_velocity=0.9, particle_direction="up",
        turbulence=0.15, **_BLUE),
    "speaking": EmotionProfile(
        "speaking", voxel_density=1.05, brow=0.12, eye_width=1.0, eye_height=0.9,
        mouth_curve=0.1, mouth_tension=0.0, glow=1.0, glow_pulse=0.15,
        speed=1.1, particle_velocity=1.1, particle_direction="drift",
        turbulence=0.2, **_BLUE),
    "alert": EmotionProfile(
        "alert", voxel_density=1.1, brow=0.5, eye_width=1.15, eye_height=1.1,
        mouth_curve=-0.1, mouth_tension=0.3, glow=0.9, glow_pulse=0.4,
        speed=1.25, particle_velocity=1.4, particle_direction="burst",
        turbulence=0.4, **_AMBER),
}


# ──────────────────────────────────────────────────────────────────────────────
# SENTIMENT ANALYSER — offline, deterministic, instant.
# ──────────────────────────────────────────────────────────────────────────────

class SentimentAnalyser:
    """
    Classify text into {sentiment, confidence, intensity, reason} with a
    weighted lexical scorer.  Deliberately NOT a model call: it must cost
    nothing, work with the power off, and behave identically across every
    provider — and it runs over the user's words as well as ORION's, which
    is what makes empathetic concern possible.
    """

    _LEXICON: dict[str, tuple[tuple[str, float], ...]] = {
        "celebratory": (
            (r"\b(congratulations|well done|excellent|brilliant|superb|delighted|"
             r"succeeded|passed|shipped|complete[d]?|achieved|victory|milestone)\b", 2.0),
            (r"\b(great news|good news|pleased to report)\b", 2.5),
            (r"!", 0.4),
        ),
        "empathetic_concern": (
            (r"\b(sorry|regret|afraid|unfortunately|concern(?:ed|ing)?|difficult|"
             r"trouble|struggl\w+|sympath\w+|condolence\w*|stress(?:ed|ful)?|"
             r"worried|anxious|overwhelm\w+)\b", 2.0),
            (r"\b(take care|are you (all right|ok|okay))\b", 2.5),
        ),
        "urgent": (
            (r"\b(urgent(?:ly)?|immediately|critical|at once|right away|asap|"
             r"emergency|overdue|deadline|breach|failure|down|offline)\b", 2.2),
            (r"\b(must|now)\b", 0.6),
        ),
        "analytical": (
            (r"\b(analysis|therefore|however|evidence|data|compar\w+|because|"
             r"whereas|conclu\w+|indicates?|suggests?|metric\w*|percent|ratio)\b", 1.2),
            (r"\b(first(?:ly)?|second(?:ly)?|third(?:ly)?|in summary)\b", 1.0),
        ),
        "confident": (
            (r"\b(certainly|absolutely|without doubt|assured?|guarantee\w*|"
             r"precisely|exactly|confirmed|verified|of course)\b", 1.8),
            (r"\b(i shall|consider it done|already handled)\b", 2.2),
        ),
        "focused": (
            (r"\b(proceeding|executing|working on|in progress|step \d|next step|"
             r"on it|running|building|scanning|checking)\b", 1.6),
        ),
    }

    _COMPILED = {
        name: tuple((re.compile(pat, re.IGNORECASE), weight) for pat, weight in rules)
        for name, rules in _LEXICON.items()
    }

    @classmethod
    def analyse(cls, text: str, origin: str = "orion") -> dict[str, Any]:
        """Full spec payload: sentiment, confidence, intensity, reason."""
        text = str(text or "")[:4000]
        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0,
                    "intensity": 0.0, "reason": "empty text", "origin": origin}
        scores: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        for name, rules in cls._COMPILED.items():
            total = 0.0
            words: list[str] = []
            for pattern, weight in rules:
                hits = pattern.findall(text)
                if hits:
                    total += weight * min(len(hits), 4)   # saturate: 100
                    words.extend(h if isinstance(h, str) else h[0]  # marks
                                 for h in hits[:2])       # aren't 100× joy
            if total > 0.0:
                scores[name] = total
                matched[name] = [w for w in words if w and w != "!"][:3]
        if not scores:
            return {"sentiment": "neutral", "confidence": 0.3,
                    "intensity": 0.2, "reason": "no affective markers",
                    "origin": origin}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = (best_score - runner_up) / max(best_score, 1e-9)
        strength = min(1.0, best_score / 6.0)
        confidence = round(min(0.95, 0.35 + 0.4 * strength + 0.25 * margin), 2)
        intensity = round(min(1.0, best_score / 8.0), 2)
        keywords = ", ".join(dict.fromkeys(matched.get(best, []))) or "tone"
        who = "User" if origin == "user" else "Reply"
        return {"sentiment": best, "confidence": confidence,
                "intensity": intensity,
                "reason": f"{who} expressed {best.replace('_', ' ')} ({keywords})",
                "origin": origin}

    @classmethod
    def classify(cls, text: str) -> tuple[str, float]:
        """Compatibility shim: (sentiment, confidence)."""
        payload = cls.analyse(text)
        return (payload["sentiment"], payload["confidence"])

    @classmethod
    def broadcast(cls, bus: OrionBus, text: str, origin: str = "orion") -> dict[str, Any]:
        """Analyse *text* and emit both bus signals; never raises."""
        payload = cls.analyse(text, origin=origin)
        try:
            bus.sentiment_payload.emit(payload)
            bus.sentiment_changed.emit(payload["sentiment"],
                                       float(payload["confidence"]))
        except RuntimeError:
            pass  # Qt shutting down
        return payload


# Sentiment → emotion mapping.
_SENTIMENT_EMOTION: dict[str, str] = {
    "celebratory": "happy",
    "empathetic_concern": "concerned",
    "urgent": "alert",
    "analytical": "thinking",
    "confident": "neutral",     # confidence reads as composed, not excitable
    "focused": "thinking",
    "neutral": "neutral",
}

# Pipeline state → baseline emotion.
_STATE_EMOTION: dict[str, str] = {
    "LISTENING": "listening",
    "SPEAKING": "speaking",
    "PROCESSING": "thinking",
    "CONNECTING": "thinking",
    "INITIALISING": "neutral",
    "STANDBY": "neutral",
    "PAUSED": "sad",            # dormant, lowered — visually 'switched down'
    "SHUTTING DOWN": "sad",
}

# Fields scaled by sentiment intensity (deviation from the neutral resting
# value is what gets scaled, so low intensity = a hint, high = full force).
_INTENSITY_SCALED = {
    "voxel_density": 1.0, "brow": 0.0, "eye_width": 1.0, "eye_height": 0.75,
    "mouth_curve": 0.05, "mouth_tension": 0.0, "glow": 0.55, "glow_pulse": 0.0,
    "speed": 1.0, "particle_velocity": 1.0, "turbulence": 0.1,
}


class EmotionStateManager:
    """The single, bus-driven authority on ORION's emotional state."""

    SENTIMENT_HOLD_S = 10.0
    MIN_CONFIDENCE = 0.45
    EXCITED_AT = 0.8

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._baseline = "neutral"
        self._override: str = ""
        self._override_until = 0.0
        self._intensity = 1.0
        self._manual: str = ""
        self._current = ""
        self._current_intensity = -1.0
        self._last_payload: dict[str, Any] = {"sentiment": "neutral",
                                              "confidence": 0.0,
                                              "intensity": 0.0, "reason": ""}
        bus.state.connect(self._on_state)
        bus.sentiment_payload.connect(self._on_sentiment)
        if self.telemetry is not None:
            self.telemetry.health.register("emotion")
        self._emit("neutral", 1.0)

    # ── inputs (bus slots — GUI thread, O(1) work only) ───────────────────────

    def _on_state(self, state: str) -> None:
        self._baseline = _STATE_EMOTION.get(str(state or "").upper(), self._baseline)
        self._resolve()

    def _on_sentiment(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        self._last_payload = payload
        sentiment = str(payload.get("sentiment") or "").lower()
        confidence = float(payload.get("confidence") or 0.0)
        if confidence < self.MIN_CONFIDENCE:
            return
        emotion = _SENTIMENT_EMOTION.get(sentiment, "")
        if not emotion:
            return
        if sentiment == "celebratory" and confidence >= self.EXCITED_AT:
            emotion = "excited"
        if sentiment == "urgent" and confidence >= self.EXCITED_AT:
            emotion = "critical"
        self._override = emotion
        self._intensity = max(0.25, float(payload.get("intensity") or confidence))
        self._override_until = time.monotonic() + self.SENTIMENT_HOLD_S
        if self.telemetry is not None:
            self.telemetry.metrics.incr(f"emotion.sentiment.{sentiment}")
        self._resolve()

    # ── manual control (dispatcher tool / tests) ──────────────────────────────

    def set_emotion(self, name: str) -> str:
        """Pin an emotion explicitly; 'auto' returns control to the pipeline."""
        name = str(name or "").strip().lower()
        if name in ("auto", "", "none", "clear"):
            self._manual = ""
            self._resolve()
            return f"Emotional rendering returned to automatic (currently {self._current})."
        if name not in EMOTIONS:
            return ("Unknown emotion. Available: " + ", ".join(sorted(EMOTIONS))
                    + ", or 'auto'.")
        self._manual = name
        self._resolve()
        return f"Emotion pinned to {name}."

    # ── resolution ────────────────────────────────────────────────────────────

    def _resolve(self) -> None:
        if self._manual:
            self._emit(self._manual, 1.0)
            return
        if self._override and time.monotonic() < self._override_until:
            if self._baseline not in ("sad",):     # dormant states win
                self._emit(self._override, self._intensity)
                return
        self._override = ""
        self._emit(self._baseline, 1.0)

    def _emit(self, name: str, intensity: float) -> None:
        intensity = max(0.0, min(1.0, intensity))
        if name == self._current and abs(intensity - self._current_intensity) < 0.1:
            return
        self._current = name
        self._current_intensity = intensity
        profile = EMOTIONS.get(name, EMOTIONS["neutral"])
        params = self._scaled_params(profile, intensity)
        try:
            self.bus.emotion_changed.emit(name, params)
        except RuntimeError:
            return  # Qt shutting down
        try:
            reason = self._last_payload.get("reason", "")
            self.bus.log.emit(
                f"EMOTION: {name}"
                + (f" @ {intensity:.2f} — {reason}" if self._override else "")
            )
        except RuntimeError:
            pass
        if self.telemetry is not None:
            self.telemetry.health.beat("emotion", "OK", f"{name} @ {intensity:.2f}")
            self.telemetry.metrics.incr("emotion.transitions")

    @staticmethod
    def _scaled_params(profile: EmotionProfile, intensity: float) -> dict[str, Any]:
        """Blend the profile toward the neutral resting values by intensity,
        so a 0.5-intensity concern is a hint of amber, not full alarm."""
        params = profile.params()
        if intensity >= 0.99:
            return params
        blend = 0.45 + 0.55 * intensity     # never below a visible whisper
        for key, rest in _INTENSITY_SCALED.items():
            value = params.get(key)
            if isinstance(value, (int, float)):
                params[key] = rest + (float(value) - rest) * blend
        return params

    # ── introspection ─────────────────────────────────────────────────────────

    def current(self) -> str:
        return self._current

    def describe(self) -> dict[str, Any]:
        return {
            "current": self._current,
            "intensity": round(self._current_intensity, 2),
            "baseline": self._baseline,
            "override": self._override or None,
            "manual": self._manual or None,
            "last_sentiment": dict(self._last_payload),
            "available": sorted(EMOTIONS),
        }
