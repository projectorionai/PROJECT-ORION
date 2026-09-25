"""
Offline speech I/O (Phase 2) — internet-free ears and voice.

VOICE (output) is already fully offline and permanent: ``SpeechSynthesiser``
(audio.py) drives pyttsx3/SAPI with the frozen male ``VOICE_PROFILE``.  This
module completes the loop on the input side and documents the TTS options.

    OfflineTranscriber — transcribe a PCM/utterance buffer with the best
                         available local engine, tried in order:
                             1. faster-whisper  (fastest, if installed)
                             2. openai-whisper  (installed on this host)
                             3. vosk            (streaming fallback)
                         Returns text with zero network calls.

Because the realtime pipeline (audio.py) streams to Gemini when online, this
transcriber is used for the *offline* voice loop and for on-demand
"transcribe this audio" tasks: buffer a VAD-gated segment, hand it here, get
text, route it to the LocalBrain / local LLM, and reply with the local voice.

TTS engine notes (all male, offline, no dependency beyond what ships):
    • Windows SAPI / pyttsx3  — default, zero install, used today.
    • Piper                    — high-quality neural voices; `pip install piper-tts`
                                 and drop a male .onnx voice; set ORION_PIPER_VOICE.
    • Coqui / XTTS             — highest quality, heavier; optional.
The active engine stays male and consistent via VOICE_PROFILE regardless.
"""

from __future__ import annotations

import asyncio
import gc
import os
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import Any, Optional

from .constants import SEND_SAMPLE_RATE


class OfflineTranscriber:
    """Local, network-free speech-to-text with graceful engine fallback."""

    def __init__(self, bus: Any | None = None, telemetry: Any | None = None,
                 model_size: str = "", defer: bool = False) -> None:
        """*defer* skips the model load until :meth:`ensure_ready`.

        Loading faster-whisper costs ~2.8 s (measured). Doing that inside
        __init__ on ORION's startup path meant nearly three seconds during
        which Qt could not pump a single message, which is most of why the
        window went "not responding". Nothing needs offline transcription in
        the first seconds of a session — the live channel handles speech — so
        startup constructs this deferred and warms it in the background.
        """
        self.bus = bus
        self.telemetry = telemetry
        self.model_size = model_size or os.getenv("ORION_WHISPER_MODEL", "base.en")
        self.engine = ""            # "faster-whisper" | "whisper" | "vosk" | ""
        self._model: Any = None
        self._vosk_rec: Any = None
        self._selected = False
        #: Set by app.py once the speaker-identification service exists. Left
        #: None here so this module keeps no dependency on it — transcription
        #: must work whether or not tone reading is wired up.
        self.presence: Any = None
        self._ignored_at = 0.0
        # Serialises load / transcribe / release: the idle release must never
        # pull the model out from under a transcription in flight.
        self._lock = threading.RLock()
        self._last_used = time.monotonic()
        if not defer:
            self.ensure_ready()

    #: Seconds unused before the model is released (ORION_STT_IDLE_S; 0 = never).
    IDLE_RELEASE_S = 600.0

    def ensure_ready(self) -> bool:
        """Load the engine if it has not been loaded yet. Idempotent.

        Called by any transcription on first use (and again after an idle
        release) — so an unloaded model can delay offline STT by a few
        seconds but can never lose it.
        """
        with self._lock:
            if self._selected:
                return bool(self.engine)
            self._selected = True
            self._select_engine()
            self._last_used = time.monotonic()
            return bool(self.engine)

    def release_if_idle(self, now: float | None = None,
                        limit_s: float | None = None) -> bool:
        """Drop the model when nothing has used it for IDLE_RELEASE_S.

        MEASURED (faster-whisper base.en, private memory): the libraries cost
        150 MB once imported and cannot be unloaded, the model and its buffers
        162 MB, of which releasing returns ~95 MB. The bigger saving is not
        loading at all: ORION's offline VOICE loop runs on Vosk, and this model
        serves only phone dictation and video analysis — so it is no longer
        warmed at startup, and a session that never needs it never pays.
        """
        try:
            limit = float(limit_s if limit_s is not None
                          else os.getenv("ORION_STT_IDLE_S", str(self.IDLE_RELEASE_S)))
        except ValueError:
            limit = self.IDLE_RELEASE_S
        if limit <= 0:
            return False
        if not self._lock.acquire(blocking=False):
            return False                       # busy transcribing: not idle
        try:
            now = time.monotonic() if now is None else now
            if not self.engine or now - self._last_used < limit:
                return False
            engine = self.engine
            self._model = None
            self._vosk_rec = None
            self.engine = ""
            self._selected = False
        finally:
            self._lock.release()
        gc.collect()
        self._log(f"SR: offline transcriber ({engine}) released after "
                  f"{limit / 60:.0f} min unused; it reloads on the next request.")
        return True

    async def idle_reaper(self, interval: float = 60.0) -> None:
        """Background loop: release the model once it has sat unused."""
        while True:
            await asyncio.sleep(interval)
            try:
                await asyncio.to_thread(self.release_if_idle)
            except Exception:
                pass

    # ── engine selection (one-time) ───────────────────────────────────────────

    def _select_engine(self) -> None:
        # 1. faster-whisper — best latency/accuracy trade-off.
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            self.engine = "faster-whisper"
            self._log(f"SR: offline transcriber = faster-whisper ({self.model_size}).")
            return
        except Exception:
            pass
        # 2. openai-whisper — present on this host.
        try:
            import whisper  # type: ignore
            self._model = whisper.load_model(self.model_size.replace(".en", ""))
            self.engine = "whisper"
            self._log(f"SR: offline transcriber = openai-whisper ({self.model_size}).")
            return
        except Exception:
            pass
        # 3. vosk — streaming fallback.
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore
            SetLogLevel(-1)
            model = Model(lang="en-us")
            self._vosk_rec = KaldiRecognizer(model, float(SEND_SAMPLE_RATE))
            self.engine = "vosk"
            self._log("SR: offline transcriber = vosk.")
            return
        except Exception:
            pass
        self._log("SR: no offline transcriber available "
                  "(pip install faster-whisper or openai-whisper for offline dictation).")

    @property
    def available(self) -> bool:
        """Whether offline transcription can be done.

        Loads the engine on first ask if construction deferred it — a caller
        must never see "unavailable" merely because the background warm-up has
        not reached it yet.
        """
        self.ensure_ready()
        return bool(self.engine)

    def _owner_is_speaking(self, pcm: bytes, sample_rate: int) -> bool:
        """Whether to go on and turn this utterance into words.

        The check is the only place ORION acts on WHO spoke rather than just
        noting it. It is off unless asked for, it fails open in every
        uncertain case, and it can never raise: an exception here would make
        him deaf, which is the exact failure it exists to avoid causing.
        """
        try:
            from . import voiceprint

            # judge() also records the verdict for the sensitive-action guard.
            verdict = voiceprint.judge(pcm, sample_rate)
            if verdict.is_owner or not voiceprint.store().only_owner:
                return True
        except Exception:
            return True
        # Logged, but not once per sentence: someone else talking nearby is
        # normal, and a line per utterance would bury everything else.
        now = time.monotonic()
        if now - getattr(self, "_ignored_at", 0.0) > 60.0:
            self._ignored_at = now
            self._log(f"SR: not your voice (similarity {verdict.score:.2f}) — "
                      f"ignored. Say \"answer anyone\" to turn this off.")
        return False

    @property
    def ready(self) -> bool:
        """Loaded ALREADY, without triggering a load. For status displays."""
        return bool(self.engine)

    # ── transcription ─────────────────────────────────────────────────────────

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = SEND_SAMPLE_RATE) -> str:
        """Transcribe 16-bit mono PCM bytes → text (no network).

        Also the seam where ORION reads WHO spoke and HOW they sounded (see
        voice_presence). This is the right place for it and the gate thread is
        not: here the utterance is already complete and already off the audio
        device thread, so speaker embedding and pitch tracking cost a
        transcription that was going to happen anyway rather than risking the
        real-time capture path.
        """
        if not pcm:
            return ""
        with self._lock:          # the idle release cannot unload mid-utterance
            self._last_used = time.monotonic()
            if not self.available:
                return ""
            if self.presence is not None:
                # Never allowed to affect the transcript: tone is a
                # nice-to-have, hearing the words is not.
                try:
                    self.presence.observe_pcm(pcm, sample_rate)
                except Exception:
                    pass
            if not self._owner_is_speaking(pcm, sample_rate):
                return ""
            return self._run_engine(pcm)

    def _run_engine(self, pcm: bytes) -> str:
        """The loaded engine over one utterance. Caller holds the lock."""
        try:
            if self.engine == "faster-whisper":
                import numpy as np
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                segments, _ = self._model.transcribe(audio, language="en", vad_filter=True)
                text = " ".join(seg.text for seg in segments).strip()
            elif self.engine == "whisper":
                import numpy as np
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                result = self._model.transcribe(audio, language="en", fp16=False)
                text = str(result.get("text", "")).strip()
            elif self.engine == "vosk":
                import json
                self._vosk_rec.AcceptWaveform(pcm)
                text = str(json.loads(self._vosk_rec.FinalResult() or "{}").get("text", "")).strip()
            else:
                text = ""
        except Exception as exc:
            self._log(f"SR: offline transcription fault - {str(exc).splitlines()[0][:90]}")
            return ""
        if text and self.telemetry is not None:
            self.telemetry.metrics.incr("sr.offline_transcriptions")
        return text

    def transcribe_wav(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.is_file():
            return ""
        try:
            with wave.open(str(p), "rb") as wf:
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            return self.transcribe_pcm(frames, sample_rate=rate)
        except Exception:
            return ""

    def status(self) -> dict[str, Any]:
        # Never loads: a status read used to trigger the whole model load.
        return {"engine": self.engine or ("not loaded" if not self._selected else "none"),
                "model": self.model_size, "available": self.ready,
                "loaded": bool(self.engine)}

    def _log(self, message: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(message)
            except RuntimeError:
                pass


#: The one transcriber the process shares (set by app.py). Every extra
#: instance is another ~300 MB model; the phone bridge used to build its own.
SHARED: "OfflineTranscriber | None" = None


def shared(bus: Any | None = None, telemetry: Any | None = None) -> OfflineTranscriber:
    """The process-wide transcriber, created deferred on first ask."""
    global SHARED
    if SHARED is None:
        SHARED = OfflineTranscriber(bus=bus, telemetry=telemetry, defer=True)
    return SHARED
