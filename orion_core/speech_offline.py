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
import importlib.util
import os
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import Any, Optional

from .constants import SEND_SAMPLE_RATE


def _whisper_checkpoints(model_size: str) -> list[Path]:
    """Already installed OpenAI Whisper checkpoints, exact model first."""
    configured = Path(os.path.expandvars(os.path.expanduser(model_size)))
    if configured.suffix.lower() == ".pt":
        return [configured]
    cache = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "whisper"
    names = [model_size]
    if model_size.endswith(".en"):
        names.append(model_size[:-3])  # retain the older multilingual fallback
    return [cache / f"{name}.pt" for name in names]


def _package_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _faster_whisper_cached(model_size: str) -> bool:
    """Check local model metadata without importing/loading a speech engine."""
    def complete(folder: Path) -> bool:
        return (all((folder / name).is_file() for name in
                    ("model.bin", "config.json", "tokenizer.json"))
                and any((folder / name).is_file() for name in
                        ("vocabulary.txt", "vocabulary.json")))

    configured = Path(os.path.expandvars(os.path.expanduser(model_size)))
    if configured.is_dir():
        return complete(configured)
    # The standard faster-whisper English/multilingual models are cached by
    # Hugging Face under this repository name. Explicit repo ids also work.
    aliases = {
        "large": "Systran/faster-whisper-large-v3",
        "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
        "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
        "distil-small.en": "Systran/faster-distil-whisper-small.en",
        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
        "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
        "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    }  # faster_whisper.utils._MODELS, mirrored without importing the engine
    repo = (model_size if "/" in model_size else
            aliases.get(model_size, "Systran/faster-whisper-" + model_size))
    default_home = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"
    hf_home = Path(os.getenv("HF_HOME", str(default_home)))
    cache = Path(os.getenv("HF_HUB_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))))
    root = cache / ("models--" + repo.replace("/", "--"))
    try:
        revision = (root / "refs" / "main").read_text(encoding="utf-8").strip()
        # A cached main reference is how local_files_only resolves a model;
        # an orphaned snapshot must not be reported as ready to load.
        if not revision or any(part in revision for part in ("/", "\\", "..")):
            return False
        return complete(root / "snapshots" / revision)
    except (OSError, UnicodeError, ValueError):
        return False


def _recites(text: str, prompt: str) -> bool:
    """True when *text* is mostly a run of the prompt's own phrases - three or
    more of them in order, making up most of what was "heard"."""
    phrases = [p.strip().lower() for p in prompt.rstrip(".").split(",") if p.strip()]
    said = " ".join(str(text).lower().replace(",", " ").split())
    hits = [p for p in phrases if p and p in said]
    if len(hits) < 3:
        return False
    return sum(len(p) for p in hits) >= 0.6 * len(said.replace(" ", ""))


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
        #: A size someone asked for is kept on the GPU too (see GPU_MODEL).
        self._size_chosen = bool(model_size or os.getenv("ORION_WHISPER_MODEL"))
        #: What the CPU loads: a GPU load switches model_size to GPU_MODEL, and
        #: a reload after an idle release that cannot use the GPU must not
        #: bring small.en onto the CPU, where it is ~3x slower than base.en.
        self._cpu_size = self.model_size
        self.engine = ""            # "faster-whisper" | "whisper" | "vosk" | ""
        self._model: Any = None
        self._vosk_rec: Any = None
        self._selected = False
        #: Set by app.py once the speaker-identification service exists. Left
        #: None here so this module keeps no dependency on it — transcription
        #: must work whether or not tone reading is wired up.
        self.presence: Any = None
        #: Set by app.py: the words to expect (listening_vocabulary) - names,
        #: places, apps. Whisper got no prompt before, so every proper noun
        #: was a guess (owner: "the transcription is not as accurate").
        self.vocabulary_source: Any = None
        self._prompt_cache: tuple[float, str] = (0.0, "")
        self._ignored_at = 0.0
        # Serialises load / transcribe / release: the idle release must never
        # pull the model out from under a transcription in flight.
        self._lock = threading.RLock()
        self._last_used = time.monotonic()
        self._cached_status: tuple[float, bool] | None = None
        if not defer:
            self.ensure_ready()

    #: Seconds unused before the model is released (ORION_STT_IDLE_S; 0 = never).
    IDLE_RELEASE_S = 600.0
    PRESSURE_IDLE_RELEASE_S = 60.0
    #: In the local and hybrid brains offline hearing IS the conversation, and
    #: a released model costs the next turn its ~2.8 s reload: kept this long.
    LOCAL_BRAIN_IDLE_RELEASE_S = 1800.0

    def warm(self) -> dict[str, Any]:
        """Load the engine AND run it once, so the first real utterance pays
        neither the load nor the first-inference allocation. Blocking: call
        it from a worker thread. Never raises."""
        started = time.perf_counter()
        try:
            if not self.ensure_ready():
                return {"ok": False, "engine": "", "why": "no offline transcriber"}
            loaded = time.perf_counter()
            with self._lock:
                self._last_used = time.monotonic()
                if self.engine in ("faster-whisper", "whisper"):
                    import numpy as np
                    # Half a second of faint noise: real decoding work, no words.
                    audio = (np.random.default_rng(7).standard_normal(SEND_SAMPLE_RATE // 2)
                             * 0.002).astype(np.float32)
                    if self.engine == "faster-whisper":
                        segments, _ = self._model.transcribe(audio, language="en",
                                                             vad_filter=False, beam_size=1)
                        list(segments)
                    else:
                        self._model.transcribe(audio, language="en", fp16=False)
            done = time.perf_counter()
            return {"ok": True, "engine": self.engine,
                    "load_ms": round((loaded - started) * 1000.0, 1),
                    "first_run_ms": round((done - loaded) * 1000.0, 1)}
        except Exception as exc:
            return {"ok": False, "engine": self.engine,
                    "why": str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__}

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

    @property
    def cached_available(self) -> bool:
        """Installed hearing, without loading a model or opening the network.

        Readiness/status callers must distinguish a deferred cached engine
        from missing offline hearing. Cache the small filesystem probe so GUI
        and voice-state reads do not repeatedly scan model metadata.
        """
        if self.engine:
            return True
        if self._selected:
            return False  # a real load already failed in this session
        now = time.monotonic()
        if self._cached_status is not None and now - self._cached_status[0] < 10.0:
            return self._cached_status[1]
        try:
            found = ((_package_present("faster_whisper") and _faster_whisper_cached(self.model_size))
                     or (_package_present("whisper")
                         and any(path.is_file() for path in _whisper_checkpoints(self.model_size))))
        except (OSError, UnicodeError, ValueError):
            found = False
        self._cached_status = (now, bool(found))
        return bool(found)

    def release_if_idle(self, now: float | None = None,
                        limit_s: float | None = None, *, collect: bool = True) -> bool:
        """Drop the model when nothing has used it for IDLE_RELEASE_S.

        MEASURED (faster-whisper base.en, private memory): the libraries cost
        150 MB once imported and cannot be unloaded, the model and its buffers
        162 MB, of which releasing returns ~95 MB. The bigger saving is not
        loading at all: desktop offline hearing, phone dictation and video
        analysis share this model. It is not warmed at startup, so a session
        that never needs offline transcription never pays.
        """
        try:
            limit = float(limit_s if limit_s is not None
                          else os.getenv("ORION_STT_IDLE_S", str(self._default_idle_s())))
        except ValueError:
            limit = self.IDLE_RELEASE_S
        if limit <= 0:
            return False
        if limit_s is None:
            from .resource_governor import GOVERNOR, Pressure

            if (GOVERNOR is not None and GOVERNOR.acting
                    and GOVERNOR.effective() >= Pressure.CONSTRAINED):
                # The governor's transition callback runs only once. If that
                # attempt found recent/busy speech, subsequent reaper ticks
                # must retain the shorter limit (CPU/RAM audit, 9 Oct 2026).
                # Explicit caller limits and the configured zero opt-out win.
                limit = min(limit, self.PRESSURE_IDLE_RELEASE_S)
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
        if collect:
            gc.collect()
        self._log(f"SR: offline transcriber ({engine}) released after "
                  f"{limit / 60:.0f} min unused; it reloads on the next request.")
        return True

    def _default_idle_s(self) -> float:
        """IDLE_RELEASE_S, or the longer local-brain window while the local or
        hybrid brain is in effect (memory pressure still shortens it)."""
        try:
            from . import brain_mode
            if brain_mode.effective_mode() in (brain_mode.HYBRID, brain_mode.LOCAL):
                return max(self.IDLE_RELEASE_S, self.LOCAL_BRAIN_IDLE_RELEASE_S)
        except Exception:
            pass
        return self.IDLE_RELEASE_S

    async def release_if_idle_async(self) -> bool:
        """Destroy the model off-loop, collect unrelated cycles on the loop."""
        released = await asyncio.to_thread(self.release_if_idle, collect=False)
        if released:
            # Process-wide GC can destroy unrelated Qt wrappers. The 9 Oct
            # affinity probe reproduced a GUI timer collected on a worker;
            # keep global collection on the owning loop, unlike model release.
            gc.collect()
        return released

    async def idle_reaper(self, interval: float = 60.0) -> None:
        """Background loop: release the model once it has sat unused."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.release_if_idle_async()
            except Exception:
                pass

    # ── engine selection (one-time) ───────────────────────────────────────────

    #: The model the GPU runs unless a size was chosen. MEASURED 29 Oct 2026 on
    #: the RTX 3060 Ti, a 9.5 s request with room noise mixed in: small.en made
    #: no word errors at 0 dB and 2.1% at -3 dB, base.en 6.2% and 9.4%, at
    #: 266-272 ms - less than base.en took on the GPU before timestamps were
    #: dropped (317 ms). The CPU keeps base.en, whose speed it needs.
    GPU_MODEL = "small.en"

    def _select_engine(self) -> None:
        # 1. faster-whisper — best latency/accuracy trade-off. On the NVIDIA
        # GPU when it can run there (accelerator.cuda_ready; 3x faster on this
        # PC), proved with one short run before it is trusted; else the CPU.
        try:
            from faster_whisper import WhisperModel  # type: ignore
            from . import accelerator
            if (accelerator.CUDA_POLICY.get("whisper") == accelerator.GPU and accelerator.cuda_ready()
                    and not accelerator.on_gui_thread()):
                chosen = getattr(self, "_size_chosen", True)
                sizes = [self.model_size] if chosen else [self.GPU_MODEL, self.model_size]
                failure = ""
                for size in dict.fromkeys(sizes):
                    try:
                        import numpy as np
                        model = WhisperModel(size, device="cuda", compute_type="float16",
                                             local_files_only=True)
                        segments, _ = model.transcribe(np.zeros(SEND_SAMPLE_RATE // 2, dtype=np.float32),
                                                       language="en", beam_size=1, vad_filter=False)
                        list(segments)
                        self._model = model
                        self.model_size = size
                        self.engine = "faster-whisper"
                        self.device = "cuda"
                        self._log(f"SR: offline transcriber = faster-whisper ({size}) on the GPU.")
                        return
                    except Exception as exc:
                        failure = str(exc).splitlines()[0][:120] if str(exc) else type(exc).__name__
                self._log(f"SR: the GPU could not run speech recognition ({failure}) - using the CPU.")
            if not getattr(self, "_size_chosen", True):
                self.model_size = getattr(self, "_cpu_size", self.model_size)
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8",
                                       cpu_threads=4, num_workers=1, local_files_only=True)
            self.engine = "faster-whisper"
            self.device = "cpu"
            self._log(f"SR: offline transcriber = faster-whisper ({self.model_size}).")
            return
        except Exception:
            pass
        # 2. openai-whisper — present on this host.
        try:
            import whisper  # type: ignore
            cached = next((path for path in _whisper_checkpoints(self.model_size) if path.is_file()), None)
            if cached is None:
                raise FileNotFoundError("offline Whisper model is not cached")
            self._model = whisper.load_model(str(cached))
            self.engine = "whisper"
            self._log(f"SR: offline transcriber = openai-whisper ({self.model_size}).")
            return
        except Exception:
            pass
        # 3. vosk — streaming fallback.
        try:
            import vosk  # type: ignore
            from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore
            from .vosk_models import cached_vosk_model
            SetLogLevel(-1)
            model = Model(cached_vosk_model(vosk))
            self._vosk_rec = KaldiRecognizer(model, float(SEND_SAMPLE_RATE))
            self._vosk_rec.SetWords(True)
            self.engine = "vosk"
            self._log("SR: offline transcriber = vosk.")
            return
        except Exception:
            pass
        self._log("SR: no offline transcriber available "
                  "(install and cache a Whisper model, or set ORION_VOSK_MODEL to an installed Vosk model).")

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

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = SEND_SAMPLE_RATE,
                       *, identity_checked: bool = False) -> str:
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
            if self.presence is not None and not identity_checked:
                # Never allowed to affect the transcript: tone is a
                # nice-to-have, hearing the words is not.
                try:
                    self.presence.observe_pcm(pcm, sample_rate)
                except Exception:
                    pass
            if not identity_checked and not self._owner_is_speaking(pcm, sample_rate):
                return ""
            return self._run_engine(pcm)

    #: How long the vocabulary prompt is reused before it is rebuilt.
    PROMPT_TTL_S = 600.0

    def _prompt(self) -> str:
        """Whisper's initial prompt: the words this person is likely to say."""
        source = self.vocabulary_source
        if source is None:
            return ""
        at, prompt = self._prompt_cache
        if prompt and time.monotonic() - at < self.PROMPT_TTL_S:
            return prompt
        try:
            from .listening_vocabulary import whisper_prompt
            prompt = whisper_prompt(source())
        except Exception:
            prompt = ""
        self._prompt_cache = (time.monotonic(), prompt)
        return prompt

    def _run_engine(self, pcm: bytes) -> str:
        """The loaded engine over one utterance. Caller holds the lock."""
        prompt = self._prompt()
        try:
            if self.engine == "faster-whisper":
                import numpy as np
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                # No timestamps: nothing reads them, and predicting them was
                # 42% of the decode (229 -> 133 ms on the GPU, same words).
                segments, _ = self._model.transcribe(audio, language="en", vad_filter=True,
                                                     beam_size=5, initial_prompt=prompt or None,
                                                     without_timestamps=True)
                text = " ".join(seg.text for seg in segments).strip()
            elif self.engine == "whisper":
                import numpy as np
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                result = self._model.transcribe(audio, language="en", fp16=False,
                                                initial_prompt=prompt or None)
                text = str(result.get("text", "")).strip()
            elif self.engine == "vosk":
                import json
                self._vosk_rec.AcceptWaveform(pcm)
                from .listening_vocabulary import vosk_final_text
                text = vosk_final_text(json.loads(self._vosk_rec.FinalResult() or "{}"))
            else:
                text = ""
        except Exception as exc:
            self._log(f"SR: offline transcription fault - {str(exc).splitlines()[0][:90]}")
            return ""
        if text and prompt and _recites(text, prompt):
            # Whisper, given near-silence and a prompt, can "hear" the prompt
            # itself - a list of names nobody said. Better nothing than that.
            text = ""
        if text and self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("sr.offline_transcriptions")
            except Exception:
                pass  # Metrics must never discard words already transcribed.
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
                "loaded": bool(self.engine), "cached_available": self.cached_available}

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
