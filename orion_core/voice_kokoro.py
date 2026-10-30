"""ORION's offline voice: Kokoro, a British male neural voice that runs on this machine.

Charon (voice_charon) is ORION's voice: Gemini TTS online, and offline the
lines he has already said. Whatever he cannot say - no internet, a service
standing down, a spent quota, a sentence never cached - is spoken here, in
Kokoro v1.0 (an 82M-parameter speech model, Apache-2.0 weights, the ONNX export
of github.com/thewh1teagle/kokoro-onnx) through kokoro-onnx (MIT) on
onnxruntime, with espeak-ng (GPL-3.0, shipped by espeakng-loader) and
phonemizer (GPL-3.0) turning text into phonemes. It replaces the Windows SAPI
voice ("Microsoft David") in that role; SAPI stays behind it as the last resort.

The voice is bm_george (British English, calm - ORION's "sir") unless the owner
chooses another (choose_voice; the elevenlabs_voice tool's offline_voice
action; Preferences > Voice): any English voice in voices-v1.0.bin, women's
included since Mark XXXVII (MALE_VOICES is the original men-only list, kept).
The choice and the speed live in voice_settings (CONFIG_DIR/voices/
voice_settings.json, or ORION_VOICE_KOKORO_VOICE / _SPEED for one run); an
older CONFIG_DIR/voices/kokoro.json is still read when nothing is stored there.

The model is looked for in CONFIG_DIR/models/kokoro-v1.0, then in the
application's own assets (build_standalone.py bundles it into
_internal/assets/models/kokoro-v1.0). Nothing is downloaded here. The files are
checked by size before loading; verify(full=True) also checks the pinned
SHA-256 (the build does, before bundling them).

MEASURED on the owner's machine (Ryzen 5 5600X, 6 cores, onnxruntime 1.24.4,
CPU), 18 Oct 2026:

  load      0.6-0.7 s (1.4 s the first time in a process, imports and
            espeak-ng's start included); about 500 MB resident while loaded.
            Building the session holds Python's lock in two native calls, the
            longer 0.4-0.5 s with graph optimisation and weight pre-packing
            off against 0.8-0.9 s with the defaults - and synthesis is no
            slower without them (x3.3-3.5 real time either way). Every Python
            thread waits that out, the audio renderer included, whose device
            buffer is 0.3 s, so the session is never built while ORION is
            speaking: the synthesiser lets what is playing finish first, and
            the watcher loads it at a quiet moment while Charon cannot speak.
  speed     x3.1-3.5 real time on all six cores (x2.3 on three); inference
            releases the lock. stop() ends the piece being made in ~20 ms.
  first     a reply is made a piece at a time while the pieces before it play
  audio     (plan_chunks: sentences, split at a clause where needed). Over
            seven replies once loaded: first audio 0.32-1.38 s, median 0.59 s
            (the slowest are first sentences with no clause to split at), and
            no piece was late for its turn.
  DirectML  fails on this model ('/encoder/F0.1/pool/ConvTranspose': "The
            parameter is incorrect"), so Kokoro runs on the CPU.

Off in tests, like Charon and Piper: only voice_charon.enable() (orion.py, a
real run) switches it on.

    python -m orion_core.voice_kokoro --measure    synthesises; plays nothing
    python -m orion_core.voice_kokoro --verify     sizes and SHA-256 of the files
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: What ORION's renderer plays, and what Kokoro makes: mono 16-bit at 24 kHz.
SAMPLE_RATE = 24000
CHUNK_BYTES = 9600            # 0.2 s at 24 kHz mono 16-bit

DEFAULT_VOICE = "bm_george"
#: The male voices in voices-v1.0.bin: bm_ British English, am_ American English.
MALE_VOICES = ("bm_george", "bm_lewis", "bm_daniel", "bm_fable",
               "am_michael", "am_adam", "am_echo", "am_eric", "am_fenrir",
               "am_liam", "am_onyx", "am_puck", "am_santa")
#: The owner's own pace may be set in the settings file ("speed"), within this.
SPEED_LIMITS = (0.8, 1.25)

#: Silence after a sentence, and after a clause a sentence is split at
#: (Kokoro's own joins use 0.25 s and 0.1 s).
SENTENCE_PAUSE_S = 0.25
CLAUSE_PAUSE_S = 0.1
#: Kokoro's pace at speed 1.0 and how fast this machine makes it (x3.3 real
#: time), measured 18 Oct 2026, rounded down: estimates that keep each piece
#: short enough to be made before the speech ahead of it has played out. A
#: slower machine gets a slightly longer pause between pieces, never a cut.
SPOKEN_CHARS_PER_S = 15.0
MADE_CHARS_PER_S = 45.0
#: The first piece of a reply: ~0.6 s to make, and then he is speaking.
FIRST_PIECE_CHARS = 40
#: A later piece is never planned shorter than this, nor split into pieces
#: shorter than MIN_PIECE_CHARS.
MIN_BUDGET_CHARS = 24
MIN_PIECE_CHARS = 12

#: Kokoro speaks more quietly than Charon (RMS 0.087 for bm_george against
#: 0.07-0.16, mean 0.11, for his cached lines): this gain matches them, held
#: back on a sentence whose peak would pass PEAK_LIMIT.
OUTPUT_GAIN = 1.25
PEAK_LIMIT = 0.89

#: A loaded model this long unused, while Charon can speak again, is let go.
IDLE_RELEASE_S = 15 * 60.0
#: After a failed load, how long before another try (each costs the pause above).
LOAD_RETRY_S = 600.0
#: How often the folders and packages are looked at again.
RESCAN_S = 30.0
#: The watcher's beat, and how long the conversation must have been quiet first.
WATCH_POLL_S = 1.0
QUIET_SETTLE_S = 3.0


@dataclass(frozen=True)
class ModelFile:
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    name: str             # the folder it lives in
    release: str          # where the files were published
    files: dict[str, ModelFile]

    def url(self, file_name: str) -> str:
        return f"{self.release}/{file_name}"


MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"
MODEL = ModelSpec(
    name="kokoro-v1.0",
    release="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0",
    files={
        MODEL_FILE: ModelFile(
            325_532_387, "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"),
        VOICES_FILE: ModelFile(
            28_214_398, "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
    },
)

#: What the voice needs, found without importing (importing loads onnxruntime).
PACKAGES = ("kokoro_onnx", "onnxruntime", "espeakng_loader", "phonemizer")

_ON = False


def enable() -> None:
    """Switch the offline voice on (a real run; see voice_charon.enable)."""
    global _ON
    _ON = True


def enabled() -> bool:
    return _ON and os.getenv("ORION_KOKORO_VOICE", "1").strip().lower() not in {
        "0", "false", "no", "off"}


# ── the model files ──────────────────────────────────────────────────────────

def search_dirs(spec: ModelSpec | None = None) -> list[Path]:
    """CONFIG_DIR/models first (the owner's copy), then the application's own
    assets: a frozen build keeps bundled data in _internal (RESOURCE_DIR), a
    source checkout in its assets folder (BASE_DIR)."""
    from .constants import BASE_DIR, CONFIG_DIR, RESOURCE_DIR
    spec = spec or MODEL
    found: list[Path] = []
    for root in (Path(CONFIG_DIR) / "models", Path(RESOURCE_DIR) / "assets" / "models",
                 Path(BASE_DIR) / "assets" / "models"):
        folder = root / spec.name
        if folder not in found:
            found.append(folder)
    return found


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(folder: Path, spec: ModelSpec | None = None, *, full: bool = False) -> list[str]:
    """What is wrong with the model files in *folder* ([] = nothing). Sizes
    always; with *full* the SHA-256 too (reads all ~354 MB, about a second)."""
    problems: list[str] = []
    for name, item in (spec or MODEL).files.items():
        path = Path(folder) / name
        try:
            size = path.stat().st_size if path.is_file() else -1
        except OSError:
            size = -1
        if size < 0:
            problems.append(f"{name} is missing")
        elif size != item.size:
            problems.append(f"{name} is {size:,} bytes, not {item.size:,}")
        elif full and sha256_of(path) != item.sha256:
            problems.append(f"{name} does not match its pinned SHA-256")
    return problems


def model_dir(dirs: list[Path] | None = None, spec: ModelSpec | None = None) -> Path | None:
    """The first folder holding every model file at its pinned size."""
    for folder in (search_dirs(spec) if dirs is None else dirs):
        if not verify(Path(folder), spec):
            return Path(folder)
    return None


def packages_missing() -> list[str]:
    missing = []
    for name in PACKAGES:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


# ── the owner's choice of voice ──────────────────────────────────────────────

def settings_path() -> Path:
    from .constants import CONFIG_DIR
    return Path(CONFIG_DIR) / "voices" / "kokoro.json"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads(Path(path or settings_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def normalise_voice(name: Any) -> str | None:
    """'bm_george', 'George', 'BM-Lewis', 'michael', 'heart' -> a voice id; None
    if none. Any English voice in voices-v1.0.bin (voice_settings.KOKORO_VOICES):
    since Mark XXXVII the women's voices can be chosen too (the cute voice)."""
    from . import voice_settings
    return voice_settings.normalise_kokoro_voice(name)


def current_voice(path: Path | None = None) -> str:
    """The voice in use. With no *path*: voice_settings (ORION_VOICE_KOKORO_VOICE,
    then the stored choice), then an older kokoro.json, then the default. An
    explicit *path* is read as that file alone (the seam tests and tools use)."""
    if path is None:
        from . import voice_settings
        chosen = voice_settings.override("kokoro_voice") or voice_settings.stored("kokoro_voice")
        if chosen:
            return str(chosen)
    return normalise_voice(load_settings(path).get("voice")) or DEFAULT_VOICE


def base_speed(path: Path | None = None) -> float:
    """Kokoro's speed factor before the emotion's own (see current_voice for
    where it is read from)."""
    if path is None:
        from . import voice_settings
        chosen = voice_settings.override("kokoro_speed") or voice_settings.stored("kokoro_speed")
        if chosen:
            return float(chosen)
    try:
        speed = float(load_settings(path).get("speed", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return max(SPEED_LIMITS[0], min(SPEED_LIMITS[1], speed))


def language_for(voice: str) -> str:
    """espeak-ng's language for a voice: b... British English, else American."""
    return "en-gb" if str(voice).startswith("b") else "en-us"


def describe_voice(voice: str) -> str:
    accent = "British English" if language_for(voice) == "en-gb" else "American English"
    return f"{voice} ({accent}{', the default' if voice == DEFAULT_VOICE else ''})"


def choose_voice(name: Any, path: Path | None = None) -> tuple[str, bool]:
    """Make *name* ORION's offline voice, kept for later runs. (reply, ok)."""
    voice = normalise_voice(name)
    if voice is None:
        from . import voice_settings
        return (f"There is no offline voice called {str(name)!r}. "
                f"Choose from: {', '.join(voice_settings.KOKORO_VOICES)}.", False)
    if path is None:
        # The one store (voice_settings.json), shared with Preferences > Voice.
        from . import voice_settings
        if not voice_settings.set_value("kokoro_voice", voice):
            return "Could not save the offline voice.", False
    else:
        target = Path(path)
        settings = load_settings(target)
        settings["voice"] = voice
        try:
            from .atomic_io import atomic_write_text
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(target, json.dumps(settings, indent=2))
        except Exception as exc:
            return f"Could not save the offline voice: {exc}", False
    return (f"My offline voice is now {describe_voice(voice)}: I use it whenever "
            "Charon cannot speak, such as with no internet.", True)


def describe_choice(path: Path | None = None, dirs: list[Path] | None = None) -> str:
    voice = current_voice(path)
    line = (f"Offline voice: Kokoro {describe_voice(voice)}, used whenever Charon "
            "cannot speak (no internet, his service standing down).")
    if model_dir(dirs) is None:
        line += (" Its model is not installed here, so for now those lines use "
                 "the Windows voice.")
    elif packages_missing():
        line += " Its engine is not installed here, so for now those lines use the Windows voice."
    return line


def voices_text() -> str:
    from . import voice_settings
    groups = (("British men", "bm_"), ("British women", "bf_"),
              ("American men", "am_"), ("American women", "af_"))
    parts = [f"{label} " + ", ".join(v for v in voice_settings.KOKORO_VOICES if v.startswith(prefix))
             for label, prefix in groups]
    return ("Offline voices (Kokoro): " + "; ".join(parts)
            + f". The default is {DEFAULT_VOICE}.")


def delivery(emotion: str, speed: float = 1.0) -> tuple[float, float]:
    """(speed, volume) for *emotion*: voice_emotion.prosody_for's rate and
    volume (the same envelope the SAPI voice used), as Kokoro's speed factor
    and a gain."""
    try:
        from .voice_emotion import prosody_for
        rate, volume = prosody_for(emotion, 1000, 1.0)
        speed = speed * rate / 1000.0
    except Exception:
        volume = 1.0
    return max(0.5, min(2.0, round(speed, 3))), max(0.0, min(1.2, float(volume)))


# ── text -> pieces -> PCM ────────────────────────────────────────────────────

_SENTENCE = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")
#: Where a sentence may be split without breaking a phrase: after a clause's
#: punctuation or a dash, or before a conjunction.
_CLAUSE = re.compile(r"[,;:](?=\s)|\s[-–—](?=\s)")
_CONJUNCTION = re.compile(r"\s(?=(?:and|but|or|so|because|which|while|when|where|although|"
                          r"though|unless|until|then|if)\b)", re.IGNORECASE)
#: A full stop after these is not the end of a sentence.
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr", "vs",
                            "e.g", "i.e", "approx", "fig"})


def split_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    for line in str(text or "").splitlines():
        line = " ".join(line.split())
        start = 0
        for match in _SENTENCE.finditer(line):
            candidate = line[start:match.end()].strip()
            word = candidate.rsplit(None, 1)[-1].rstrip(".").lower() if candidate else ""
            if match.group(0) == "." and (word in _ABBREVIATIONS
                                          or (len(word) == 1 and word.isalpha())):
                continue
            if candidate:
                pieces.append(candidate)
            start = match.end()
        rest = line[start:].strip()
        if rest:
            pieces.append(rest)
    return pieces


def _split_at(text: str, budget: float) -> int | None:
    """Where to split *text* so its first part is at most *budget* characters:
    the last clause boundary within it, else the first one after it."""
    cuts = sorted({m.end() for m in _CLAUSE.finditer(text)}
                  | {m.start() for m in _CONJUNCTION.finditer(text)})
    cuts = [cut for cut in cuts if MIN_PIECE_CHARS <= cut <= len(text) - MIN_PIECE_CHARS]
    within = [cut for cut in cuts if cut <= budget]
    if within:
        return within[-1]
    return next((cut for cut in cuts if cut > budget), None)


def plan_chunks(text: str) -> list[tuple[str, float]]:
    """*text* as (piece, pause after it), synthesised one at a time while the
    pieces before play. Pieces are sentences, split at a clause where needed:
    the first one short, so speech starts soon, and each later one short
    enough to be made before the speech already made has played out. The
    last piece has no pause after it."""
    chunks: list[tuple[str, float]] = []
    lead = 0.0                  # seconds of speech made and not yet played
    for sentence in split_sentences(text):
        rest = sentence
        while rest:
            budget = (FIRST_PIECE_CHARS if not chunks
                      else max(MIN_BUDGET_CHARS, lead * MADE_CHARS_PER_S))
            cut = _split_at(rest, budget) if len(rest) > budget else None
            if cut:
                piece, rest = rest[:cut].strip(), rest[cut:].strip()
            else:
                piece, rest = rest, ""
            pause = CLAUSE_PAUSE_S if rest else SENTENCE_PAUSE_S
            chunks.append((piece, pause))
            lead = (max(0.0, lead - len(piece) / MADE_CHARS_PER_S)
                    + len(piece) / SPOKEN_CHARS_PER_S + pause)
    if chunks:
        chunks[-1] = (chunks[-1][0], 0.0)
    return chunks


def to_pcm(samples: Any, rate: int = SAMPLE_RATE, volume: float = 1.0) -> bytes:
    """Kokoro's float samples -> 16-bit PCM at SAMPLE_RATE, at Charon's level."""
    import numpy as np
    audio = np.asarray(samples, dtype=np.float32).ravel()
    if not audio.size:
        return b""
    if rate and rate != SAMPLE_RATE:
        positions = np.linspace(0, audio.size - 1, int(round(audio.size * SAMPLE_RATE / rate)))
        audio = np.interp(positions, np.arange(audio.size), audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    gain = min(OUTPUT_GAIN * max(0.0, float(volume)), PEAK_LIMIT / peak)
    return (np.clip(audio * gain, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


#: How Kokoro must be given his name. Its phonemiser reads "Orion" as
#: /ɒɹˈaɪən/ - a short, unstressed "o" that runs into "RYN" and is heard
#: as "ORIN" (owner, 26 Oct 2026); "Oh-ryan" is /ˈəʊɹˈaɪən/, the
#: "oh-RYE-un" his persona asks for. Speech only: nothing written changes.
_KOKORO_NAME = "Oh-ryan"
_ORION_WORD_RE = re.compile(r"\bOrion\b", re.IGNORECASE)


def spoken_name(text: str) -> str:
    """*text* as Kokoro should say it: every form of his name (dotted,
    all-caps, misspelt - utils.normalise_for_speech) as "Oh-ryan"."""
    from .utils import normalise_for_speech
    return _ORION_WORD_RE.sub(_KOKORO_NAME, normalise_for_speech(str(text or "")))


def _silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(max(0.0, seconds) * SAMPLE_RATE)


class _StoppableSession:
    """The onnxruntime session kokoro_onnx runs, with RunOptions this voice can
    terminate: an interruption ends the sentence being made at once instead of
    after it (up to ~1.5 s for a long one)."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.run_options: Any = None

    def run(self, output_names: Any, input_feed: Any, run_options: Any = None) -> Any:
        return self._session.run(output_names, input_feed, run_options or self.run_options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


# ── the voice ────────────────────────────────────────────────────────────────

class KokoroVoice:
    """Kokoro behind the cloud-voice interface (SAMPLE_RATE, available,
    stream_pcm), so the synthesiser streams it into ORION's one output stream.
    Thread-safe: one load, one synthesis at a time (espeak-ng is
    process-global), and nothing is loaded until it is needed."""

    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, *, dirs: list[Path] | None = None,
                 engine_loader: Callable[[Path], Any] | None = None,
                 settings: Path | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self._dirs = dirs
        self._loader = engine_loader
        self._settings = settings
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()            # the engine: load, release, busy
        self._speak_lock = threading.Lock()      # one synthesis at a time
        self._engine: Any = None
        self._session: Any = None                # the _StoppableSession, when real
        self._busy = 0
        self._last_used = 0.0
        self._failed_until = 0.0
        self._found: tuple[Path | None, str] = (None, "")
        self._found_at = -RESCAN_S
        self._stopped = threading.Event()
        self._closed = threading.Event()
        self._watcher: threading.Thread | None = None
        self._quiet_since: float | None = None
        self._quiet: Callable[[], bool] | None = None
        self._needed: Callable[[], bool] | None = None
        self._log: Callable[[str], Any] | None = None
        self.device = "CPU"
        self.last_error = ""
        #: load_ms (each load); for the last line: first_ms, total_ms, audio_s.
        self.timing: dict[str, float] = {}

    @property
    def voice(self) -> str:
        return current_voice(self._settings)

    def _discover(self) -> tuple[Path | None, str]:
        """(model folder, '') or (None, why not) - looked at again every RESCAN_S."""
        now = self._clock()
        if now - self._found_at < RESCAN_S:
            return self._found
        folder = model_dir(self._dirs)
        why = ""
        if folder is None:
            looked = self._dirs if self._dirs is not None else search_dirs()
            why = "the Kokoro model is in none of " + ", ".join(str(d) for d in looked)
        elif self._loader is None:
            missing = packages_missing()
            if missing:
                why = "not installed: " + ", ".join(missing)
        self._found, self._found_at = (None if why else folder, why), now
        return self._found

    def available(self) -> bool:
        if not enabled():                   # first: tests never look further
            return False
        if self._engine is not None:
            return True
        if self._clock() < self._failed_until:
            return False
        return self._discover()[0] is not None

    def loaded(self) -> bool:
        return self._engine is not None

    def describe(self) -> str:
        folder, why = self._discover()
        if self._engine is None and folder is None:
            return f"Kokoro is not available: {why}"
        state = "loaded" if self._engine is not None else "not loaded (loads when first needed)"
        return f"Kokoro {describe_voice(self.voice)} on the {self.device}, {state}; model in {folder}"

    # ── the engine ───────────────────────────────────────────────────────────

    def load(self) -> Any:
        """The engine, loaded first if need be. Blocking: ~0.6 s, up to 0.5 s
        of it holding Python's lock - call it from a worker thread, and never
        while audio is playing (see the module docstring)."""
        with self._lock:
            if self._engine is not None:
                return self._engine
            folder, why = self._discover()
            if folder is None:
                raise RuntimeError(why or "the Kokoro model is not installed")
            started = time.perf_counter()
            try:
                engine = self._loader(folder) if self._loader is not None else self._open(folder)
            except Exception as exc:
                self._failed_until = self._clock() + LOAD_RETRY_S
                self.last_error = (str(exc).splitlines() or [type(exc).__name__])[0][:200]
                raise
            self._engine = engine
            self._last_used = self._clock()
            self.last_error = ""
            self.timing["load_ms"] = (time.perf_counter() - started) * 1000.0
            return engine

    def _open(self, folder: Path) -> Any:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        options = ort.SessionOptions()
        # Measured (module docstring): with graph optimisation and weight
        # pre-packing off, the longer of the two calls that build the session
        # holds Python's lock 0.4-0.5 s instead of 0.8-0.9 s, and synthesis
        # is just as fast.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry("session.disable_prepacking", "1")
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        # Mark XXXX.4, measured: RSS 544 -> 344 MB and 33% less CPU for the same
        # 22 s of speech (2.7x real time instead of 2.9-3.2x - still ahead).
        options.enable_cpu_mem_arena = False
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = _StoppableSession(ort.InferenceSession(
            str(Path(folder) / MODEL_FILE), options, providers=["CPUExecutionProvider"]))
        engine = Kokoro.from_session(session, str(Path(folder) / VOICES_FILE))
        try:
            engine.tokenizer.phonemize("Ready.", "en-gb")   # espeak-ng's own start, ~0.25 s
        except Exception:
            pass
        self._session = session
        return engine

    def release(self) -> bool:
        """Let the model go (~500 MB); the next line loads it again."""
        with self._lock:
            if self._engine is None or self._busy:
                return False
            self._engine = None
            self._session = None
            return True

    def _arm_run(self) -> None:
        session = self._session
        if session is None:
            return
        try:
            import onnxruntime as ort
            options = ort.RunOptions()
            options.log_severity_level = 3
            session.run_options = options
        except Exception:
            pass

    def stop(self) -> None:
        """Cut what is being said: no further sentence, and the one being made
        ends now."""
        self._stopped.set()
        options = getattr(self._session, "run_options", None)
        if options is not None:
            try:
                options.terminate = True
            except Exception:
                pass

    # ── loading at a quiet moment, letting go when idle ──────────────────────

    def attach(self, *, quiet: Callable[[], bool] | None = None,
               needed: Callable[[], bool] | None = None,
               log: Callable[[str], Any] | None = None) -> None:
        """quiet(): nothing is being said or heard; needed(): Charon cannot
        speak now, so the next line would be Kokoro's."""
        if quiet is not None:
            self._quiet = quiet
        if needed is not None:
            self._needed = needed
        if log is not None:
            self._log = log

    def start_watch(self) -> bool:
        """A real run only: load at a quiet moment while needed() holds, and
        let the model go after IDLE_RELEASE_S unused once it does not."""
        if not enabled():
            return False
        if self._watcher is not None and self._watcher.is_alive():
            return False
        self._closed.clear()
        self._watcher = threading.Thread(target=self._watch, name="orion-kokoro-watch",
                                         daemon=True)
        self._watcher.start()
        return True

    def _say(self, message: str) -> None:
        if self._log is not None:
            try:
                self._log(message)
            except Exception:
                pass

    def _watch(self) -> None:
        while not self._closed.wait(WATCH_POLL_S):
            try:
                self.watch_once()
            except Exception:
                self._quiet_since = None

    def watch_once(self) -> None:
        """One beat of the watcher."""
        needed = bool(self._needed()) if self._needed is not None else False
        now = self._clock()
        if self._engine is None:
            if not needed or not self.available() or (self._quiet is not None and not self._quiet()):
                self._quiet_since = None
                return
            if self._quiet_since is None:
                self._quiet_since = now
            if now - self._quiet_since < QUIET_SETTLE_S:
                return
            self._quiet_since = None
            try:
                self.load()
                self._say(f"VOICE: offline voice Kokoro loaded at a quiet moment "
                          f"({self.timing.get('load_ms', 0.0) / 1000.0:.1f} s), "
                          "while Charon cannot speak.")
            except Exception:
                self._say(f"VOICE: offline voice Kokoro could not load - {self.last_error}")
        elif not needed and self._busy == 0 and now - self._last_used >= IDLE_RELEASE_S:
            if self.release():
                self._say(f"VOICE: offline voice Kokoro let go after "
                          f"{IDLE_RELEASE_S / 60:.0f} minutes unused (Charon can speak).")

    def close(self) -> None:
        """Shut down: stop speaking, stop the watcher, let the model go."""
        self._closed.set()
        self.stop()
        with self._lock:
            self._engine = None
            self._session = None

    # ── speaking ─────────────────────────────────────────────────────────────

    def stream_pcm(self, text: str, emotion: str = "neutral", *,
                   on_chunk: Callable[[bytes], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> bytes | None:
        """Speak *text* as 24 kHz mono 16-bit PCM, a sentence at a time, each
        handed to *on_chunk* in 0.2 s pieces as soon as it is made (the next
        is made while it plays). None = no voice."""
        chunks = plan_chunks(spoken_name(text))
        if not chunks or not self.available():
            return None
        with self._lock:
            self._busy += 1
        try:
            return self._speak(chunks, emotion, on_chunk, should_stop)
        finally:
            with self._lock:
                self._busy -= 1
                self._last_used = self._clock()

    def _speak(self, chunks: list[tuple[str, float]], emotion: str,
               on_chunk: Callable[[bytes], None] | None,
               should_stop: Callable[[], bool] | None) -> bytes | None:
        started = time.perf_counter()
        try:
            engine = self.load()
        except Exception:
            return None
        voice = self.voice
        voices = getattr(engine, "voices", None)
        try:
            if voices is not None and voice not in voices:
                voice = DEFAULT_VOICE
        except Exception:
            pass
        speed, volume = delivery(emotion, base_speed(self._settings))
        lang = language_for(voice)

        def stopping() -> bool:
            return self._stopped.is_set() or (should_stop is not None and should_stop())

        out = bytearray()
        first: list[float] = []
        with self._speak_lock:
            self._stopped.clear()
            self._arm_run()
            try:
                for piece, pause in chunks:
                    if stopping():
                        break
                    try:
                        samples, rate = engine.create(piece, voice=voice, speed=speed, lang=lang)
                    except ValueError:
                        continue            # nothing speakable in it ("...", a symbol)
                    if stopping():
                        break
                    pcm = to_pcm(samples, rate, volume)
                    if not pcm:
                        continue
                    if not first:
                        first.append(time.perf_counter())
                    pcm += _silence(pause)
                    out.extend(pcm)
                    if on_chunk is not None:
                        for start in range(0, len(pcm), CHUNK_BYTES):
                            if stopping():
                                break
                            on_chunk(pcm[start:start + CHUNK_BYTES])
            except Exception as exc:
                self.last_error = (str(exc).splitlines() or [type(exc).__name__])[0][:200]
                if not out:
                    return None
        done = time.perf_counter()
        self.timing.update(first_ms=((first[0] if first else done) - started) * 1000.0,
                           total_ms=(done - started) * 1000.0,
                           audio_s=len(out) / 2 / SAMPLE_RATE)
        return bytes(out) or None


_SHARED: KokoroVoice | None = None
_SHARED_LOCK = threading.Lock()


def shared_voice() -> KokoroVoice:
    """The one KokoroVoice of this process: one ~500 MB model, however many
    speakers want it."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = KokoroVoice()
        return _SHARED


# ── checks and measurement (no sound is played) ─────────────────────────────

def self_check(dirs: list[Path] | None = None, *, require_bundled: bool = False) -> str:
    """For the installed-app parity check: the packages import, espeak-ng turns
    text into phonemes, and the model files are where ORION looks for them, at
    their pinned sizes - with *require_bundled*, inside the application's own
    assets. Loads no model and plays nothing. Raises on any failure."""
    import espeakng_loader
    from kokoro_onnx.tokenizer import Tokenizer
    library = Path(espeakng_loader.get_library_path())
    if not library.is_file():
        raise RuntimeError(f"the espeak-ng library is missing: {library}")
    data = Path(espeakng_loader.get_data_path())
    phonemes = Tokenizer().phonemize("Good evening, sir.", "en-gb")
    if not phonemes:
        raise RuntimeError("espeak-ng produced no phonemes")
    if require_bundled and dirs is None:
        from .constants import RESOURCE_DIR
        dirs = [Path(RESOURCE_DIR) / "assets" / "models" / MODEL.name]
    folder = model_dir(dirs)
    if folder is None:
        looked = dirs if dirs is not None else search_dirs()
        problems = "; ".join(f"{d}: {', '.join(verify(Path(d)))}" for d in looked)
        raise RuntimeError(f"the Kokoro model files are not usable - {problems}")
    sizes = ", ".join(f"{name} {item.size:,} B" for name, item in MODEL.files.items())
    return (f"model {folder} ({sizes}); espeak-ng {library.name} + {data.name}; "
            f"{len(phonemes)} phonemes for a test line")


#: Lines Charon's cache cannot hold - they change every time - for --measure.
MEASURE_LINES = (
    "It's Sunday the eleventh of October, seven forty-two in the evening.",
    "I couldn't find a file called budget forecast in your Documents folder.",
    "Notepad is open, and I've pasted your notes into it.",
    "Your next reminder is at half past nine tomorrow morning: call the dentist.",
    "The machine is healthy - CPU around twelve percent, memory at sixty-one percent.",
)


def measure(voice: KokoroVoice | None = None,
            lines: tuple[str, ...] = MEASURE_LINES) -> dict[str, Any]:
    """Load time, time to the first audio and real-time factor of the offline
    voice. Synthesises only; plays nothing."""
    voice = voice or KokoroVoice()
    if not voice.available():
        return {"ok": False, "why": voice.describe()}
    rows = []
    for text in lines:
        if voice.stream_pcm(text, on_chunk=lambda chunk: None) is None:
            return {"ok": False, "why": f"no audio for {text!r}: {voice.last_error}"}
        rows.append(dict(voice.timing))
    warm = rows[1:] or rows
    return {
        "ok": True, "voice": voice.voice, "device": voice.device,
        "load_ms": round(voice.timing.get("load_ms", 0.0)),
        "cold_first_audio_ms": round(rows[0]["first_ms"]),
        "first_audio_ms_median": round(statistics.median(row["first_ms"] for row in warm)),
        "first_audio_ms_range": [round(min(row["first_ms"] for row in warm)),
                                 round(max(row["first_ms"] for row in warm))],
        "real_time_factor": round(sum(row["audio_s"] for row in warm)
                                  / max(1e-6, sum(row["total_ms"] for row in warm) / 1000.0), 2),
        "audio_seconds": round(sum(row["audio_s"] for row in rows), 1),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    enable()
    if "--verify" in argv:
        good = False
        for folder in search_dirs():
            if folder.is_dir():
                problems = verify(folder, full=True)
                print(f"{folder}: {'; '.join(problems) if problems else 'OK (sizes and SHA-256)'}")
                good = good or not problems
        return 0 if good else 1
    voice = KokoroVoice()
    if "--measure" in argv:
        print(json.dumps(measure(voice), indent=2))
    else:
        print(voice.describe())
    return 0


__all__ = [
    "DEFAULT_VOICE", "MALE_VOICES", "MODEL", "SAMPLE_RATE",
    "KokoroVoice", "ModelFile", "ModelSpec",
    "choose_voice", "current_voice", "delivery", "describe_choice", "enable", "enabled",
    "language_for", "measure", "model_dir", "normalise_voice", "packages_missing",
    "plan_chunks", "search_dirs", "self_check", "settings_path", "shared_voice",
    "split_sentences", "to_pcm", "verify", "voices_text",
]


if __name__ == "__main__":
    raise SystemExit(main())
