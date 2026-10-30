"""A small local neural voice for what Charon has never said, with no internet.

Charon (voice_charon) speaks online through Gemini TTS and offline from his
cache. A sentence he has never said still needs SOME voice when the internet
is gone, and the Windows SAPI voice sounds nothing like him. Piper
(github.com/OHF-Voice/piper1-gpl) is a small neural voice that runs on the CPU
through onnxruntime - which ORION already ships - so for uncached text it sits
after Charon's cache and before SAPI, which stays the last resort.

Nothing is downloaded here. It switches itself on only when the owner has put
a voice in config/voices/piper (a <name>.onnx with its <name>.onnx.json) AND
an engine is present: the ``piper-tts`` package (in process, the model loaded
once; preferred) or a standalone piper.exe in that same folder (one process
per utterance). Otherwise ``available()`` is False and the synthesiser goes
straight to SAPI, exactly as before. Like Charon it is off in tests: only
``voice_charon.enable()`` (orion.py, a real run) switches it on.

MEASURED from the package index on 11 Oct 2026 (nothing installed):
piper-tts 1.8.0 ships one abi3 wheel for Windows x64 (cp39 and later, so
both 3.12 and 3.13), 34.1 MB - nearly all of it the espeak-ng phoneme data it
embeds - needing onnxruntime (already installed) and pathvalidate; licence
GPL-3.0-or-later. Medium-quality voices are 63.2 MB each (22,050 Hz).
Load time and real-time factor need a real model, so they are measured on
the owner's machine once one is there (no sound is played):

    python -m orion_core.voice_piper --measure
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: What ORION's renderer plays; Piper's own 22,050 or 16,000 Hz is resampled.
SAMPLE_RATE = 24000
CHUNK_BYTES = 9600            # 0.2 s at 24 kHz mono 16-bit
#: The standalone engine reads the whole utterance at once; never let it hang the voice.
EXE_TIMEOUT_S = 30.0
EXE_POLL_S = 0.1
EXE_REAP_S = 1.0
#: How often the folder is looked at again for a newly added voice.
RESCAN_S = 30.0

_ON = False


def enable() -> None:
    """Switch the local neural voice on (a real run; see voice_charon.enable)."""
    global _ON
    _ON = True


def enabled() -> bool:
    return _ON and os.getenv("ORION_PIPER_VOICE", "1").strip().lower() not in {"0", "false", "no", "off"}


def voices_dir() -> Path:
    from .constants import CONFIG_DIR
    return Path(CONFIG_DIR) / "voices" / "piper"


def find_model(directory: Path | None = None) -> Path | None:
    """The voice: ORION_PIPER_MODEL (a file name in the folder, or a path) when
    set, else the first <name>.onnx with its <name>.onnx.json, by name."""
    directory = Path(directory or voices_dir())
    candidates: list[Path] = []
    wanted = os.getenv("ORION_PIPER_MODEL", "").strip()
    if wanted:
        path = Path(wanted)
        candidates.append(path if path.is_absolute() else directory / path)
    try:
        candidates += sorted(directory.glob("*.onnx"))
    except OSError:
        pass
    for model in candidates:
        try:
            if model.suffix == ".onnx" and model.is_file() and Path(f"{model}.json").is_file():
                return model
        except OSError:
            continue
    return None


def package_present() -> bool:
    """The piper-tts package is importable - checked without importing it
    (importing it loads onnxruntime). Another, unrelated 'piper' has no
    voice.py."""
    try:
        spec = importlib.util.find_spec("piper")
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    if getattr(sys, "frozen", False):
        return True                       # bundled: no files to look at
    return any((Path(location) / "voice.py").is_file()
               for location in (spec.submodule_search_locations or ()))


def find_exe(directory: Path | None = None) -> Path | None:
    """A standalone piper.exe kept beside the voices (never one on PATH: the
    pip package's own 'piper' command takes different arguments)."""
    directory = Path(directory or voices_dir())
    for path in (directory / "piper.exe", directory / "piper" / "piper.exe"):
        if path.is_file():
            return path
    return None


def model_rate(model: Path) -> int:
    try:
        config = json.loads(Path(f"{model}.json").read_text(encoding="utf-8"))
        return int(config["audio"]["sample_rate"])
    except Exception:
        return 22050


def resample(pcm: bytes, rate: int) -> bytes:
    """16-bit mono PCM at *rate* -> SAMPLE_RATE (linear; it only goes up)."""
    if not pcm or rate == SAMPLE_RATE or rate <= 0:
        return pcm
    import numpy as np
    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    if not samples.size:
        return b""
    positions = np.linspace(0, samples.size - 1, int(round(samples.size * SAMPLE_RATE / rate)))
    return np.interp(positions, np.arange(samples.size), samples).astype("<i2").tobytes()


def _chunk_pcm(chunk: Any) -> bytes:
    """An AudioChunk's 16-bit PCM (piper-tts 1.3+), or raw bytes (older API)."""
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)
    return bytes(getattr(chunk, "audio_int16_bytes", b"") or b"")


class PiperLocalVoice:
    """Piper behind the cloud-voice interface (SAMPLE_RATE, available,
    stream_pcm), so the synthesiser streams it into ORION's one output stream."""

    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, *, directory: Path | None = None,
                 engine_loader: Callable[[Path], Any] | None = None,
                 exe_runner: Callable[[Path, str], bytes | None] | None = None) -> None:
        self._directory = directory
        self._loader = engine_loader
        self._runner = exe_runner
        self._engine: Any = None
        self._engine_model: Path | None = None
        self._lock = threading.Lock()
        self._warm_lock = threading.Lock()
        self._warm_thread: threading.Thread | None = None
        self._process: Any = None
        self._stopped = threading.Event()
        self._found: tuple[Path | None, str] = (None, "")
        self._found_at = -RESCAN_S
        #: load_ms (once), and for the last line: first_ms, total_ms, audio_s.
        self.timing: dict[str, float] = {}

    def _folder(self) -> Path:
        return Path(self._directory or voices_dir())

    def _discover(self) -> tuple[Path | None, str]:
        """(model, 'package' | 'exe' | ''). The loaded voice is kept for the
        session, so a second voice added later never changes him mid-talk."""
        now = time.monotonic()
        if now - self._found_at < RESCAN_S:
            return self._found
        folder = self._folder()
        model = self._engine_model if (self._engine_model and self._engine_model.is_file()) \
            else find_model(folder)
        kind = ""
        if model is not None:
            if self._loader is not None or package_present():
                kind = "package"
            elif self._runner is not None or find_exe(folder) is not None:
                kind = "exe"
        self._found, self._found_at = (model, kind), now
        return self._found

    def available(self) -> bool:
        if not enabled():                   # first: tests never look further
            return False
        model, kind = self._discover()
        return model is not None and bool(kind)

    def describe(self) -> str:
        model, kind = self._discover()
        if model is None:
            return f"no Piper voice in {self._folder()}"
        if not kind:
            return f"{model.name} is there, but neither the piper-tts package nor piper.exe"
        return f"{model.name} via {'piper-tts' if kind == 'package' else 'piper.exe'}"

    # ── the engine ───────────────────────────────────────────────────────────

    def _load(self, model: Path) -> Any:
        with self._lock:
            if self._engine is not None and self._engine_model == model:
                return self._engine
            started = time.perf_counter()
            if self._loader is not None:
                engine = self._loader(model)
            else:
                from piper import PiperVoice  # piper-tts 1.3+
                engine = PiperVoice.load(str(model))
            self._engine, self._engine_model = engine, model
            self.timing["load_ms"] = (time.perf_counter() - started) * 1000.0
            return engine

    def warm_async(self) -> bool:
        """Load the model on a worker thread, so the first offline line does
        not pay for it. True if a load was started."""
        with self._warm_lock:
            if (not self.available() or self._engine is not None
                    or (self._warm_thread is not None and self._warm_thread.is_alive())):
                return False
            model, kind = self._discover()
            if kind != "package" or model is None:
                return False

            def load() -> None:
                try:
                    self._load(model)
                except Exception:
                    pass

            self._warm_thread = threading.Thread(target=load, name="orion-piper-warm",
                                                 daemon=True)
            self._warm_thread.start()
            return True

    def stop(self) -> None:
        """Cancel only the standalone process this voice instance started."""
        self._stopped.set()
        process = self._process
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass

    def _run_exe(self, model: Path, text: str,
                 should_stop: Callable[[], bool] | None = None) -> bytes | None:
        if self._runner is not None:
            return self._runner(model, text)
        exe = find_exe(self._folder())
        if exe is None:
            return None
        process = subprocess.Popen(
            [str(exe), "--model", str(model), "--output-raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=str(exe.parent), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._process = process
        deadline = time.monotonic() + EXE_TIMEOUT_S
        payload: bytes | None = (" ".join(text.split()) + "\n").encode("utf-8")
        try:
            while True:
                if (self._stopped.is_set() or (should_stop is not None and should_stop())
                        or time.monotonic() >= deadline):
                    return None
                try:
                    out, _ = process.communicate(payload, timeout=EXE_POLL_S)
                    return (out or None) if process.returncode == 0 else None
                except subprocess.TimeoutExpired:
                    payload = None      # communicate retains the first call's input
        finally:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.communicate(timeout=EXE_REAP_S)
            except (OSError, subprocess.TimeoutExpired):
                pass
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
            self._process = None

    # ── speaking ─────────────────────────────────────────────────────────────

    def stream_pcm(self, text: str, emotion: str = "neutral", *,
                   on_chunk: Callable[[bytes], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> bytes | None:
        """Speak *text* (24 kHz PCM, a sentence at a time). None = no voice."""
        del emotion
        if not text.strip() or not self.available():
            return None
        model, kind = self._discover()
        if model is None:
            return None
        self._stopped.clear()
        started = time.perf_counter()
        out = bytearray()
        first: list[float] = []

        def emit(pcm: bytes) -> bool:
            if not first:
                first.append(time.perf_counter())
            out.extend(pcm)
            if on_chunk is not None:
                for start in range(0, len(pcm), CHUNK_BYTES):
                    if self._stopped.is_set() or (should_stop is not None and should_stop()):
                        return False
                    on_chunk(pcm[start:start + CHUNK_BYTES])
            return True

        try:
            if kind == "package":
                engine = self._load(model)
                for chunk in engine.synthesize(text):
                    if self._stopped.is_set() or (should_stop is not None and should_stop()):
                        break
                    rate = int(getattr(chunk, "sample_rate", 0) or model_rate(model))
                    if not emit(resample(_chunk_pcm(chunk), rate)):
                        break
            else:
                pcm = self._run_exe(model, text, should_stop)
                if pcm and not (should_stop is not None and should_stop()):
                    emit(resample(pcm, model_rate(model)))
        except Exception:
            if not out:
                return None
        done = time.perf_counter()
        self.timing.update(first_ms=((first[0] if first else done) - started) * 1000.0,
                           total_ms=(done - started) * 1000.0,
                           audio_s=len(out) / 2 / SAMPLE_RATE)
        return bytes(out) or None


#: Lines Charon's cache cannot hold - they change every time - for --measure.
MEASURE_LINES = (
    "It's Sunday the eleventh of October, seven forty-two in the evening.",
    "I couldn't find a file called budget forecast in your Documents folder.",
    "Notepad is open, and I've pasted your notes into it.",
    "Your next reminder is at half past nine tomorrow morning: call the dentist.",
    "The machine is healthy - CPU around twelve percent, memory at sixty-one percent.",
)


def measure(voice: PiperLocalVoice | None = None,
            lines: tuple[str, ...] = MEASURE_LINES) -> dict[str, Any]:
    """Size, load time, time to first audio and real-time factor of the
    installed voice. Synthesises only; plays nothing."""
    voice = voice or PiperLocalVoice()
    model, kind = voice._discover()
    if model is None or not kind:
        return {"ok": False, "why": voice.describe()}
    rows = []
    for text in lines:
        if voice.stream_pcm(text) is None:
            return {"ok": False, "why": f"no audio for {text!r}"}
        rows.append(dict(voice.timing))
    audio = sum(row["audio_s"] for row in rows)
    synth = sum(row["total_ms"] for row in rows[1:]) / 1000.0     # the first one loads
    return {
        "ok": True, "engine": kind, "model": model.name,
        "model_mb": round(model.stat().st_size / 1e6, 1),
        "load_ms": round(voice.timing.get("load_ms", 0.0)),
        "first_audio_ms_median": round(statistics.median(row["first_ms"] for row in rows[1:]))
        if len(rows) > 1 else None,
        "real_time_factor": round(synth / max(1e-6, sum(row["audio_s"] for row in rows[1:])), 3)
        if len(rows) > 1 else None,
        "audio_seconds": round(audio, 1),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    enable()
    voice = PiperLocalVoice()
    if "--measure" in argv:
        print(json.dumps(measure(voice), indent=2))
    else:
        print(voice.describe())
    return 0


__all__ = [
    "SAMPLE_RATE",
    "PiperLocalVoice",
    "enable",
    "enabled",
    "find_exe",
    "find_model",
    "measure",
    "model_rate",
    "package_present",
    "resample",
    "voices_dir",
]


if __name__ == "__main__":
    raise SystemExit(main())
