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

import os
import wave
from array import array
from pathlib import Path
from typing import Any, Optional

from .constants import SEND_SAMPLE_RATE


class OfflineTranscriber:
    """Local, network-free speech-to-text with graceful engine fallback."""

    def __init__(self, bus: Any | None = None, telemetry: Any | None = None,
                 model_size: str = "") -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.model_size = model_size or os.getenv("ORION_WHISPER_MODEL", "base.en")
        self.engine = ""            # "faster-whisper" | "whisper" | "vosk" | ""
        self._model: Any = None
        self._vosk_rec: Any = None
        self._select_engine()

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
        return bool(self.engine)

    # ── transcription ─────────────────────────────────────────────────────────

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = SEND_SAMPLE_RATE) -> str:
        """Transcribe 16-bit mono PCM bytes → text (no network)."""
        if not pcm or not self.available:
            return ""
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
        return {"engine": self.engine or "none", "model": self.model_size,
                "available": self.available}

    def _log(self, message: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(message)
            except RuntimeError:
                pass
