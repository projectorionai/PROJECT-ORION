"""
Where ORION's on-device models run: the graphics card or the CPU.

ORION runs several ONNX models on this machine - OCR, object detection,
sentence embeddings, the voiceprint encoder, the sound classifier and the
Kokoro voice. This module is the one place that decides where each runs and
builds its ONNX Runtime session:

    GPU   DirectML (the onnxruntime-directml build), when its provider is
          present, ORION_GPU is not 0, and POLICY below sends the model
          there. Such sessions get enable_mem_pattern=False and sequential
          execution, which the DirectML provider requires.
    CPU   everything else.

Keeping the feature comes first. A model whose GPU session cannot be built,
or whose GPU run fails, is rebuilt on the CPU - logged once - and stays there
for the rest of the run. The failed input is run again on the CPU, so the
caller gets its answer either way.

Sessions are built by the caller's thread, and building one holds Python's
lock (the GIL) for the whole native call - every Python thread waits it out.
MEASURED here (18 Oct 2026): the first DirectML session in a process holds it
~300 ms (device creation), later ones 130-290 ms, against 20-140 ms on the
CPU. So a GPU session is never built on the GUI thread (which is also ORION's
asyncio loop): there the model gets a CPU session, and the build is logged
so the caller can be moved to a worker.

One DirectML session must not be run by two threads at once (onnxruntime's
DirectML documentation), so runs on a GPU session are serialised here. CPU
sessions are left lock-free, as their callers rely on.

ORION_GPU=0 keeps every model on the CPU.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable

GPU = "gpu"
CPU = "cpu"
DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
SWITCH = "ORION_GPU"
_OFF = frozenset({"0", "false", "no", "off"})

#: Where each model runs when the GPU is usable. MEASURED on this PC (RTX 3060
#: Ti 8 GB, driver 32.0.16.1088, Ryzen 5 5600X, onnxruntime-directml 1.24.4,
#: 18 Oct 2026), steady state, CPU figures as the sites build them:
POLICY: dict[str, str] = {
    # Text detection: a 1280x720 camera frame 104-136 ms -> 16-27 ms; a
    # 1920x1080 screen 263-271 ms -> 68-83 ms. A new input size costs DirectML
    # more (48-85 ms) but stays under the CPU.
    "ocr_det": GPU,
    # Line recognition, batches of six: 54 ms -> 12-15 ms; a dense 1920x1080
    # screen 4.4 s -> 1.15 s. Whole reads with det+rec here, cls on the CPU:
    # camera frame 730-880 ms -> 251-255 ms, dense screen 4.9 s -> 1.42 s,
    # six detail crops 985 ms -> 753-779 ms; the same text read.
    "ocr_rec": GPU,
    # The angle classifier: 1-3 ms a batch on the CPU, 25-34 ms on DirectML
    # for every batch size after the first.
    "ocr_cls": CPU,
    # A YOLO model at a fixed 640x640, convolutions like ocr_det. No detector
    # is installed on this PC, so unmeasured here; a failure falls back.
    "object_detection": GPU,
    # One question: 5.4 ms on one CPU thread, 8.9 ms on DirectML - and on the
    # CPU it runs at the calling thread's priority (see semantic.py).
    "semantic_query": CPU,
    # Backfill, rows of 6-140 words: 32 rows 1.2 s -> 0.2 s, 128 rows
    # 3.6 s -> 0.6 s (DirectML, batches of 64). The vectors agree with the
    # CPU's as closely as two CPU batch sizes agree with each other (cosine
    # >= 0.993: the model's int8 quantisation is computed per batch).
    "semantic_batch": GPU,
    # 16.3 ms on the CPU against 13.7 ms: not worth a 300 ms build holding
    # the GIL beside live audio.
    "voiceprint": CPU,
    # YAMNet: DirectML rejects the model at build ("The parameter is
    # incorrect", 80070057); 6.5 ms per 5 s of sound on the CPU.
    "sound_sense": CPU,
    # Kokoro: DirectML fails at /encoder/F0.1/pool/ConvTranspose (80070057);
    # 3.2-3.5x real time on the CPU.
    "kokoro": CPU,
}

#: Models that run through CUDA rather than DirectML (CTranslate2).
CUDA_POLICY: dict[str, str] = {
    # Offline speech recognition (faster-whisper / CTranslate2, CUDA, not
    # DirectML). base.en on 9.5 s of speech, 28 Oct 2026: 824 ms on the CPU
    # (int8, 4 threads) -> 271-317 ms on the RTX 3060 Ti (float16), the same
    # words. The very first CUDA run on a PC compiles kernels (18 s, once;
    # cached on disk after: 593 ms), so it is loaded by the warm-up, off the
    # GUI thread. Needs cuBLAS: CUDA_RUNTIME below.
    "whisper": GPU,
}

#: NVIDIA's cuBLAS for CTranslate2 (it ships cuDNN itself), downloaded with
#: the owner's approval into his ORION folder rather than the app (28 Oct
#: 2026). ORION_GPU_RUNTIME points elsewhere.
CUDA_RUNTIME = Path(os.path.expandvars(os.getenv(
    "ORION_GPU_RUNTIME", r"%USERPROFILE%\ORION\gpu-runtime"))) / "nvidia" / "cublas" / "bin"
_cuda: bool | None = None


def cuda_ready() -> bool:
    """Whether CTranslate2 can run on the NVIDIA GPU here: a CUDA device and
    cuBLAS. Puts CUDA_RUNTIME on the DLL search path once. ORION_GPU=0: no."""
    global _cuda
    if switched_off():
        return False
    if _cuda is None:
        _cuda = False
        try:
            if (CUDA_RUNTIME / "cublas64_12.dll").is_file():
                add = getattr(os, "add_dll_directory", None)
                if callable(add):
                    add(str(CUDA_RUNTIME))
                os.environ["PATH"] = str(CUDA_RUNTIME) + os.pathsep + os.environ.get("PATH", "")
                import ctranslate2
                _cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            _cuda = False
    return _cuda

_lock = threading.Lock()
_providers: tuple[str, ...] | None = None
_fallen: dict[str, str] = {}            # model -> why it left the GPU
_noted: set[str] = set()                 # log lines already written once
_summarised = False
_log_sink: Callable[[str], Any] | None = None


# ── what is here ─────────────────────────────────────────────────────────────

def _onnxruntime() -> Any:
    import onnxruntime
    return onnxruntime


def switched_off() -> bool:
    """ORION_GPU=0 (or false/no/off)."""
    return os.getenv(SWITCH, "1").strip().lower() in _OFF


def available_providers() -> tuple[str, ...]:
    """The execution providers this onnxruntime build has (cached)."""
    global _providers
    if _providers is None:
        try:
            _providers = tuple(_onnxruntime().get_available_providers())
        except Exception:
            _providers = ()
    return _providers


def gpu_available() -> bool:
    return not switched_off() and DML_PROVIDER in available_providers()


def policy(model: str) -> str:
    """Where *model* runs when the GPU is usable; unknown models: the CPU."""
    return POLICY.get(model, CPU)


def wants_gpu(model: str) -> bool:
    """Whether a session built now for *model* would try the GPU."""
    return policy(model) == GPU and model not in _fallen and gpu_available()


def on_gui_thread() -> bool:
    """True on the thread running Qt's event loop - ORION's asyncio loop too."""
    if threading.current_thread() is not threading.main_thread():
        return False
    qtcore = sys.modules.get("PyQt6.QtCore")
    if qtcore is None:
        return False
    try:
        return qtcore.QCoreApplication.instance() is not None
    except Exception:
        return False


# ── building sessions ────────────────────────────────────────────────────────

def session_options(*, gpu: bool, threads: int | None = None, inter_threads: int | None = None,
                    configure: Callable[[Any], Any] | None = None) -> Any:
    """SessionOptions for one session. *configure* adds a site's own settings."""
    ort = _onnxruntime()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if threads is not None:
        options.intra_op_num_threads = int(threads)
    if inter_threads is not None:
        options.inter_op_num_threads = int(inter_threads)
    if not gpu:
        # Mark XXXX.4: onnxruntime's worker threads spin between runs by
        # default; ORION's models run in short bursts, so that spin is pure
        # CPU (-33% measured on the offline voice).
        set_entry = getattr(options, "add_session_config_entry", None)
        if callable(set_entry):
            set_entry("session.intra_op.allow_spinning", "0")
    if configure is not None:
        configure(options)
    if gpu:
        # Required by the DirectML provider (onnxruntime docs), so applied
        # after the site's settings.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def providers(gpu: bool) -> list[Any]:
    if gpu:
        return [(DML_PROVIDER, {"device_id": 0}), CPU_PROVIDER]
    return [CPU_PROVIDER]


def providers_for(model: str) -> list[Any]:
    """Providers for a library that builds its own session (GPU per policy)."""
    return providers(wants_gpu(model) and not on_gui_thread())


def _build(path: str, gpu: bool, threads: int | None, inter_threads: int | None,
           configure: Callable[[Any], Any] | None) -> Any:
    ort = _onnxruntime()
    options = session_options(gpu=gpu, threads=threads, inter_threads=inter_threads,
                              configure=configure)
    return ort.InferenceSession(path, sess_options=options, providers=providers(gpu))


def create_session(model: str, path: str | os.PathLike[str], *, threads: int | None = None,
                   inter_threads: int | None = None,
                   configure: Callable[[Any], Any] | None = None,
                   cpu: Callable[[], Any] | None = None) -> "AcceleratedSession":
    """A session for *model*, on the GPU when POLICY and this machine allow.

    *threads*/*inter_threads*: the CPU thread counts the site always used.
    *cpu*: returns an existing CPU session to fall back to instead of
    building another. Raises what onnxruntime raises if the model cannot be
    loaded on the CPU either."""
    path = str(Path(path))
    log_summary()

    def cpu_session() -> Any:
        return cpu() if cpu is not None else _build(path, False, threads, inter_threads, configure)

    def build() -> tuple[Any, bool]:
        """(session, on the GPU): at creation, and again after an idle release."""
        gpu = wants_gpu(model)
        gui = on_gui_thread()
        if gui:
            gpu = False
        started = time.perf_counter()
        session = None
        if gpu:
            try:
                session = _build(path, True, threads, inter_threads, configure)
            except Exception as exc:
                fell_back(model, "could not start on the GPU", exc)
                gpu = False
        if session is None:
            session = cpu_session()
        if gui:
            _note(f"gui:{model}", f"ACCEL: {model} session built on the GUI thread "
                  f"({(time.perf_counter() - started) * 1000:.0f} ms with every Python thread "
                  "waiting) - on the CPU there; it should be built in a worker.")
        return session, gpu

    session, gpu = build()
    accelerated = AcceleratedSession(model, session, gpu, cpu_session, rebuild=build)
    _register(accelerated)
    return accelerated


# ── letting idle sessions go ─────────────────────────────────────────────────
#
# Every session above lives as long as the object that built it - the OCR
# engine, the object detector, the sentence encoder's GPU batch session - so
# one screen read or one re-index kept its memory for the rest of the run: a
# DirectML session ~1 GB committed (semantic.py, GPU_BACKLOG_ROWS), RapidOCR's
# three models more. A session unused for IDLE_RELEASE_S is let go and rebuilt
# by its next run (on that caller's thread, by the same rules as at creation:
# never a GPU build on the GUI thread). Never mid-run: a run holds the session
# in use, and the reaper skips any session that is in use or being rebuilt.

#: Seconds unused before a session is let go (ORION_SESSION_IDLE_S; 0 = never).
IDLE_RELEASE_S = 300.0
IDLE_SWITCH = "ORION_SESSION_IDLE_S"
#: Small CPU models asked on the voice and recall paths: rebuilding one there
#: would hold Python's lock 20-140 ms in the middle of a turn, for a few MB.
#: Kept unless they run on the GPU (POLICY never puts them there). Kokoro and
#: Whisper release themselves (voice_kokoro.IDLE_RELEASE_S, speech_offline).
KEEP_RESIDENT: frozenset[str] = frozenset({"semantic_query", "voiceprint", "sound_sense"})

_live: "weakref.WeakSet[AcceleratedSession]" = weakref.WeakSet()
_reaper: threading.Thread | None = None
_process: Any = None


def idle_release_s() -> float:
    try:
        return max(0.0, float(os.getenv(IDLE_SWITCH, str(IDLE_RELEASE_S))))
    except ValueError:
        return IDLE_RELEASE_S


def _register(session: "AcceleratedSession") -> None:
    """Watch *session* for idleness; the reaper thread starts with the first."""
    global _reaper
    with _lock:
        _live.add(session)
        if _reaper is not None and _reaper.is_alive():
            return
        # Started under the lock: started after it, two sessions created at once
        # could both see no live reaper, and the second start() raised "threads
        # can only be started once" - that model was then off for the run.
        _reaper = threading.Thread(target=_reap_forever, name="orion-session-reaper",
                                   daemon=True)
        _reaper.start()


def _reap_forever() -> None:
    while True:
        idle = idle_release_s()
        time.sleep(max(15.0, min(60.0, idle / 4.0)) if idle else 60.0)
        try:
            release_idle()
        except Exception:
            pass


def _private_bytes() -> int:
    """This process's committed private memory, or 0 when it cannot be read."""
    global _process
    try:
        if _process is None:
            import psutil
            _process = psutil.Process()
        info = _process.memory_info()
        return int(getattr(info, "private", 0) or 0)
    except Exception:
        return 0


def release_idle(idle_s: float | None = None, now: float | None = None) -> list[str]:
    """Let every session unused for *idle_s* (default idle_release_s()) go;
    one log line per release. Returns the models released."""
    idle = idle_release_s() if idle_s is None else max(0.0, float(idle_s))
    if idle <= 0:
        return []
    with _lock:
        sessions = list(_live)
    released = []
    for session in sessions:
        before = _private_bytes()
        if not session.release_if_idle(idle, now):
            continue
        freed = max(0, before - _private_bytes()) if before else 0
        _emit(f"ACCEL: released idle {session.model} session (~{freed / 2**20:.0f} MB).")
        released.append(session.model)
    return released


class AcceleratedSession:
    """An onnxruntime session that moves itself from the GPU to the CPU, and
    lets its memory go when idle (release_idle), rebuilding on its next use.

    Usable wherever an ``InferenceSession`` is: ``run`` is wrapped, everything
    else (get_inputs, get_outputs, get_modelmeta, ...) is the live session's."""

    def __init__(self, model: str, session: Any, on_gpu: bool,
                 cpu_factory: Callable[[], Any],
                 rebuild: Callable[[], tuple[Any, bool]] | None = None) -> None:
        self.model = model
        self._session = session
        self._on_gpu = bool(on_gpu)
        self._cpu_factory = cpu_factory
        self._cpu: Any = None
        self._rebuild = rebuild
        self._lock = threading.Lock()          # the CPU rebuild
        self._run_lock = threading.Lock()      # one DirectML run at a time
        # The session itself: its swap, an idle release, a rebuild, and how
        # many runs are using it now.
        self._use_lock = threading.Lock()
        self._in_use = 0
        self._last_used = time.monotonic()
        self.releases = 0

    @property
    def on_gpu(self) -> bool:
        return self._on_gpu

    @property
    def device(self) -> str:
        return "GPU" if self._on_gpu else "CPU"

    @property
    def released(self) -> bool:
        """True while idle release has let the session go."""
        return self._session is None

    @property
    def session(self) -> Any:
        """The onnxruntime session in use now (rebuilt if it was released)."""
        with self._use_lock:
            return self._live_session()

    def _live_session(self) -> Any:
        """Caller holds _use_lock."""
        if self._session is None:
            if self._rebuild is None:
                raise RuntimeError(f"the {self.model} session was released")
            session, on_gpu = self._rebuild()
            self._session, self._on_gpu = session, bool(on_gpu)
            self._last_used = time.monotonic()
        return self._session

    def run(self, output_names: Any, input_feed: Any, run_options: Any = None) -> Any:
        with self._use_lock:
            session = self._live_session()
            on_gpu = self._on_gpu
            self._in_use += 1
        try:
            if not on_gpu:
                return session.run(output_names, input_feed, run_options)
            try:
                with self._run_lock:
                    return session.run(output_names, input_feed, run_options)
            except Exception as exc:
                cpu = self._cpu_session()
                # Raises if the CPU cannot run it either: then the input was the
                # problem, not the GPU, and the GPU session is kept.
                result = cpu.run(output_names, input_feed, run_options)
                with self._use_lock:
                    swapped = self._session is session and self._on_gpu
                    if swapped:
                        self._session, self._on_gpu = cpu, False
                if swapped:
                    fell_back(self.model, "failed on the GPU", exc)
                return result
        finally:
            with self._use_lock:
                self._in_use -= 1
                self._last_used = time.monotonic()

    def release_if_idle(self, idle_s: float, now: float | None = None) -> bool:
        """Let the session go if nothing has run it for *idle_s*. Never while a
        run is using it or a rebuild is under way; never one that cannot be
        rebuilt; KEEP_RESIDENT models only when they are on the GPU."""
        if self._rebuild is None or (self.model in KEEP_RESIDENT and not self._on_gpu):
            return False
        if not self._use_lock.acquire(blocking=False):
            return False                        # a run or a rebuild has it
        try:
            now = time.monotonic() if now is None else now
            if self._session is None or self._in_use or now - self._last_used < idle_s:
                return False
            self._session = None                # the last reference: freed here
            self._cpu = None
            self.releases += 1
            return True
        finally:
            self._use_lock.release()

    def _cpu_session(self) -> Any:
        with self._lock:
            if self._cpu is None:
                self._cpu = self._cpu_factory()
            return self._cpu

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or "_use_lock" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.session, name)

    def __repr__(self) -> str:
        return f"<AcceleratedSession {self.model} on the {self.device}>"


# ── what happened, in the log ────────────────────────────────────────────────

def set_log(sink: Callable[[str], Any] | None) -> None:
    """Where the ACCEL: lines go (default: session.log)."""
    global _log_sink
    _log_sink = sink


def _emit(line: str) -> None:
    sink = _log_sink
    try:
        if sink is not None:
            sink(line)
        else:
            from . import session_log
            session_log.write(line)
    except Exception:
        pass


def _note(key: str, line: str) -> bool:
    with _lock:
        if key in _noted:
            return False
        _noted.add(key)
    _emit(line)
    return True


def fell_back(model: str, what: str, exc: BaseException) -> None:
    """Record that *model* left the GPU, and say so once."""
    reason = (str(exc).strip().splitlines() or [type(exc).__name__])[-1]
    reason = " ".join(reason.split())[-160:]
    with _lock:
        _fallen.setdefault(model, f"{what}: {reason}")
    _note(f"fallback:{model}", f"ACCEL: {model} {what} ({reason}) - running on the CPU "
          "for the rest of this run; nothing is lost but speed.")


def _adapter_name() -> str:
    try:
        from . import gpu_stats
        name = str(gpu_stats.latest().get("name") or "")
    except Exception:
        return ""
    return "" if name.lower() in {"", "n/a"} else name


def describe() -> dict[str, Any]:
    """Where every known model runs now, and why the GPU is or is not used."""
    if switched_off():
        reason = f"{SWITCH}=0"
    elif DML_PROVIDER not in available_providers():
        reason = "onnxruntime has no DirectML provider"
    else:
        reason = ""
    usable = not reason
    with _lock:
        fallen = dict(_fallen)
    models = {}
    for model, where in POLICY.items():
        if usable and where == GPU and model not in fallen:
            models[model] = GPU
        elif model in fallen:
            models[model] = f"{CPU} ({fallen[model]})"
        else:
            models[model] = CPU
    return {"gpu": usable, "reason": reason, "adapter": _adapter_name() if usable else "",
            "models": models}


def summary() -> str:
    """One line: the device, and which models run on it."""
    state = describe()
    if not state["gpu"]:
        return f"ACCEL: CPU only - {state['reason']}; every model on the CPU."
    on_gpu = [m for m, where in state["models"].items() if where == GPU]
    on_cpu = [m for m, where in state["models"].items() if where != GPU]
    adapter = f" ({state['adapter']})" if state["adapter"] else ""
    return (f"ACCEL: GPU = DirectML{adapter} - GPU: {', '.join(on_gpu) or 'none'} | "
            f"CPU: {', '.join(on_cpu) or 'none'}.")


def log_summary() -> bool:
    """Write the summary line once per run (the first session build)."""
    global _summarised
    with _lock:
        if _summarised:
            return False
        _summarised = True
    _emit(summary())
    return True


def reset() -> None:
    """Forget the cached provider list, fallbacks and logged lines (tests, or
    after the GPU driver changed under a running ORION)."""
    global _providers, _summarised
    with _lock:
        _providers = None
        _fallen.clear()
        _noted.clear()
        _summarised = False
        _live.clear()


__all__ = [
    "AcceleratedSession", "CPU", "CPU_PROVIDER", "DML_PROVIDER", "GPU", "IDLE_RELEASE_S",
    "KEEP_RESIDENT", "POLICY", "SWITCH",
    "available_providers", "create_session", "describe", "fell_back", "gpu_available",
    "idle_release_s", "log_summary", "on_gui_thread", "policy", "providers", "providers_for",
    "release_idle", "reset",
    "session_options", "set_log", "summary", "switched_off", "wants_gpu",
]
