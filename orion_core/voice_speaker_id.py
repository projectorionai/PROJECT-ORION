"""
voice_speaker_id.py — real speaker recognition: distinguishing a SPECIFIC
enrolled individual by voice, not just classifying gender.

Mark XX architectural-audit pass, Track J — the audit's own wording: "voice
recognition capable of recognising me as a known speaker rather than simply
classifying voices by gender... remember my voice profile permanently
(with my consent)... support remembering trusted people individually."
voice_gender.py already does pitch-based gender classification; this is a
different, additive capability, not a replacement.

Uses resemblyzer's pretrained 256-dim speaker-embedding encoder (bundled
weights — no network fetch at runtime) rather than training anything:
enrolment is "compute this person's embedding centroid and remember it",
identification is "compute this utterance's embedding and find the closest
enrolled centroid". Consent is enforced at the boundary — enrol() refuses
without an explicit consent=True, mirroring send_draft's confirm=true for
any other consequential, hard-to-undo action. Every profile can be removed
on request; a biometric profile with no delete path is not something this
codebase should ship.

Lazy-loaded: importing resemblyzer/torch/librosa costs real time and RAM
(a model load), so nothing here pays that cost until a caller actually
enrols or identifies someone — the same discipline as vision.py's lazy
pytesseract import.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .constants import CONFIG_DIR
from .data import ToolResult
from .atomic_io import atomic_write_text

PROFILES_PATH = CONFIG_DIR / "voice_profiles.json"
# Cosine-similarity floor for a confident match. Resemblyzer's own
# documentation and common practice place same-speaker utterance pairs
# comfortably above 0.8 and different-speaker pairs below 0.75 for clean
# audio; this is a considered starting point, not an empirically tuned
# value against this specific user's voice — expect to revisit it once
# real enrolment/identification history exists.
MATCH_THRESHOLD = 0.75

_encoder = None  # module-level cache — one model load per process, not per instance


#: Where identification actually happens, which used to be one place only.
#:
#: ``speech_offline.transcribe_pcm`` reads every offline utterance, because an
#: upstream VAD has already delimited one. The Gemini Live path had no such
#: moment — it forwards every chunk so the server's VAD can find end-of-turn —
#: so a live session never compared anyone's voice against an enrolled
#: profile. ``live_presence`` now assembles the missing utterance inside the
#: capture gate and hands it over when the speech stops, on the same terms as
#: the gender tracker: voiced, not ORION's own output, past the echo guard.
#:
#: So both paths identify the speaker, read the tone, and honour the
#: only-my-voice gate. The live path decides part-way through an utterance
#: rather than at the end, because a stranger has to be cut off before the
#: turn completes rather than after it is answered — which means the opening
#: second of a stranger's speech does reach the model. Nobody can be
#: identified before they have spoken.

def record_enrollment_clip(seconds: float = 5.0, sample_rate: int = 16000) -> Any:
    """Blocking mic recording used ONLY for enrolment.

    Deliberately NOT routed through AudioGateThread/SileroVADGatekeeper —
    the hand-tuned real-time pipeline that serves live conversation
    (audio.py) — since modifying that thread for an occasional, deliberate
    action carries real regression risk to the live voice channel (the
    same boundary this codebase already draws around AudioPlaybackThread
    for the Gemini Live path). Enrolment is a rare, one-off, explicitly
    consented action: a direct, separate sounddevice.rec() call is a
    simpler and safer path for it, at the cost of live per-utterance
    speaker identification during normal conversation not being wired up
    in this pass — that would mean touching the real-time thread, and is
    intentionally left as a follow-up rather than done blind."""
    import sounddevice as sd
    from .audio_devices import resolve
    device = resolve("input")
    recording = sd.rec(
        int(seconds * sample_rate), samplerate=sample_rate,
        channels=1, dtype="float32", device=device)
    sd.wait()
    return recording.flatten()


#: Which encoder produced a stored profile. Written into every profile,
#: because the two preprocess audio differently and their embeddings are not
#: comparable: scoring one against the other raises nothing and returns a
#: number that means nothing, which is the worst way for this to fail.
BACKEND_ONNX = "voiceprint-onnx"
BACKEND_RESEMBLYZER = "resemblyzer"


def backend() -> str:
    """The encoder in use. The ONNX one first: it needs no torch, so it is
    present in the packaged application as well as in development, and a
    capability that exists only on the developer's machine is not one."""
    from . import voiceprint

    if voiceprint.available():
        return BACKEND_ONNX
    try:
        import resemblyzer  # noqa: F401
        return BACKEND_RESEMBLYZER
    except Exception:
        return ""


def available() -> bool:
    try:
        import numpy  # noqa: F401
    except Exception:
        return False
    return bool(backend())


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
    return _encoder


def _embed(wav: Any, source_sr: int):
    """Raw waveform (+ its sample rate) -> a 256-dim embedding.

    Raises ValueError if no speech survives silence-trimming (a blank or
    silent clip) — callers decide whether that is fatal or just "no match".
    """
    import numpy as np

    if backend() == BACKEND_ONNX:
        from . import voiceprint

        vector = voiceprint.embed(wav, source_sr)
        if vector is None:
            raise ValueError("no speech detected in the supplied audio")
        return vector

    from resemblyzer import preprocess_wav
    processed = preprocess_wav(np.asarray(wav, dtype=np.float32), source_sr=source_sr)
    if processed.size == 0:
        raise ValueError("no speech detected in the supplied audio")
    return _get_encoder().embed_utterance(processed)


def _cosine_similarity(a: Any, b: Any) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SpeakerIdentificationService:
    """Enrolled voice profiles + identification against them. Owns no
    microphone or audio capture — callers hand it raw waveform arrays,
    same separation of concerns as voice_elevenlabs.ElevenLabsVoice."""

    def __init__(self, bus: Any | None = None) -> None:
        self.bus = bus
        self._profiles: dict[str, dict[str, Any]] = self._load_profiles()

    # ── persistence ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_profiles() -> dict[str, dict[str, Any]]:
        try:
            if PROFILES_PATH.exists():
                data = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_profiles(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_text(PROFILES_PATH, json.dumps(self._profiles, indent=2), encoding="utf-8")
        except OSError:
            pass  # a read-only config directory must never break the call

    def _log(self, message: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(message)
            except Exception:
                pass

    # ── enrolment ────────────────────────────────────────────────────────────

    def enroll(self, name: str, samples: list, sample_rate: int,
              consent: bool = False) -> ToolResult:
        name = str(name or "").strip()
        if not name:
            return ToolResult("A name is required to enrol a voice profile.", ok=False)
        if not consent:
            return ToolResult(
                "Voice enrolment requires explicit consent — call again with "
                "consent=true once the person has agreed to have their voice "
                "remembered permanently.", ok=False)
        if not available():
            return ToolResult(
                "Voice recognition is not available (resemblyzer is not installed).", ok=False)
        samples = list(samples or [])
        if not samples:
            return ToolResult("No audio samples supplied for enrolment.", ok=False)
        try:
            import numpy as np
            embeddings = [_embed(sample, sample_rate) for sample in samples]
            centroid = np.mean(embeddings, axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0.0:
                centroid = centroid / norm
        except Exception as exc:
            return ToolResult(f"Could not process the audio samples: {exc}", ok=False)
        self._profiles[name] = {
            "embedding": centroid.tolist(),
            "backend": backend(),
            "enrolled_at": time.time(),
            "sample_count": len(samples),
        }
        self._save_profiles()
        self._log(f"VOICE-ID: enrolled '{name}' ({len(samples)} sample(s)).")
        return ToolResult(f"Voice profile for '{name}' saved from {len(samples)} sample(s).")

    # ── identification ───────────────────────────────────────────────────────

    def identify(self, audio: Any, sample_rate: int) -> tuple[str, float]:
        """Returns (name, score) for the closest enrolled profile once it
        clears MATCH_THRESHOLD, else ("", best_score_seen). Never raises —
        unprocessable audio (e.g. silence) yields ("", 0.0)."""
        if not available() or not self._profiles:
            return ("", 0.0)
        try:
            embedding = _embed(audio, sample_rate)
        except Exception:
            return ("", 0.0)
        best_name, best_score = "", 0.0
        for name, record in self._profiles.items():
            stored = str(record.get("backend") or BACKEND_RESEMBLYZER)
            if stored != backend():
                # Enrolled with a different encoder. Comparing them would
                # produce a plausible-looking number with no meaning, so the
                # profile is skipped and the person is simply not recognised
                # until they enrol again.
                continue
            score = _cosine_similarity(embedding, record.get("embedding") or [])
            if score > best_score:
                best_name, best_score = name, score
        if best_score >= MATCH_THRESHOLD:
            return (best_name, best_score)
        return ("", best_score)

    # ── management ───────────────────────────────────────────────────────────

    def list_profiles(self) -> list[str]:
        return sorted(self._profiles)

    def remove_profile(self, name: str) -> ToolResult:
        name = str(name or "").strip()
        if name not in self._profiles:
            return ToolResult(f"No voice profile for '{name}'.", ok=False)
        del self._profiles[name]
        self._save_profiles()
        self._log(f"VOICE-ID: removed profile '{name}'.")
        return ToolResult(f"Voice profile for '{name}' removed.")

    def describe(self) -> str:
        if not available():
            return "Voice recognition is not installed (resemblyzer/librosa missing)."
        if not self._profiles:
            return "No voice profiles enrolled yet."
        names = ", ".join(sorted(self._profiles))
        return f"{len(self._profiles)} voice profile(s) enrolled: {names}."
