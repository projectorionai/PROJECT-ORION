"""
The GPU meters must never launch a process on the event loop.

History
-------
`CommandCentre._gpu` shelled out to nvidia-smi on the Qt thread, which under
qasync is also the asyncio event loop that paints the face and services the
audio callback. First its cache check was backwards (a machine with no NVIDIA
GPU spawned a doomed process once a second); once fixed it still ran
nvidia-smi every five seconds — MEASURED at ~80 ms per call, a visible hitch
each time, with a 2 s timeout at worst.

It now reads `gpu_stats.latest()`: NVML in-process when available (0.02 ms),
otherwise the last reading the background telemetry loop took. No subprocess
is ever launched by these tests.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import accelerator, gpu_stats  # noqa: E402


class _Probe:
    """The real method, on an object with nothing else attached."""

    from orion_core.gui.command_centre import CommandCentreWindow as _CC

    _gpu = _CC._gpu


@pytest.fixture
def no_processes(monkeypatch):
    launched = []
    monkeypatch.setattr(gpu_stats.subprocess, "run",
                        lambda cmd, **kw: launched.append(cmd))
    return launched


@pytest.fixture
def fresh_state(monkeypatch):
    monkeypatch.setattr(gpu_stats, "_state", {
        "init": True, "ok": True, "handle": None, "name": "", "backend": "nvidia-smi",
        "exe": "nvidia-smi", "at": 0.0, "cache": None})
    return gpu_stats._state


def test_the_gui_thread_never_launches_nvidia_smi(no_processes, fresh_state):
    for _ in range(20):
        assert _Probe()._gpu() == ("n/a", 0.0)
    assert no_processes == []


def test_the_gui_thread_shows_the_background_reading(no_processes, fresh_state):
    fresh_state["cache"] = {"available": True, "util": 41.0}
    assert _Probe()._gpu() == ("41%", 41.0)
    assert no_processes == []


def test_nvml_is_sampled_directly(monkeypatch, fresh_state):
    fresh_state["backend"] = "nvml"
    monkeypatch.setattr(gpu_stats, "sample", lambda: {"available": True, "util": 12.0})
    assert _Probe()._gpu() == ("12%", 12.0)


def test_a_machine_without_a_gpu_reads_na(no_processes, monkeypatch):
    monkeypatch.setattr(gpu_stats, "_state", {
        "init": True, "ok": False, "handle": None, "name": "", "backend": "",
        "exe": None, "at": 0.0, "cache": None})
    assert _Probe()._gpu() == ("n/a", 0.0)
    assert no_processes == []


def test_the_probe_never_raises(monkeypatch):
    def _explode():
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(gpu_stats, "latest", _explode)
    assert _Probe()._gpu() == ("n/a", 0.0)


def test_the_nvidia_smi_fallback_is_sampled_rarely(monkeypatch, fresh_state):
    calls = []

    class _Result:
        returncode = 0
        stdout = "RTX,41,1000,8000,50\n"
        stderr = ""

    monkeypatch.setattr(gpu_stats.subprocess, "run",
                        lambda cmd, **kw: (calls.append(cmd), _Result())[1])
    clock = [100.0]
    monkeypatch.setattr(gpu_stats.time, "monotonic", lambda: clock[0])
    for _ in range(4):
        gpu_stats.sample()
        clock[0] += 1.0
    assert len(calls) == 1, "nvidia-smi must not be relaunched every second"
    clock[0] += gpu_stats._SMI_INTERVAL_S
    gpu_stats.sample()
    assert len(calls) == 2


# ── accelerator.py: which models run on the GPU ─────────────────────────────
#
# accelerator decides GPU (DirectML) or CPU for every ONNX Runtime session
# ORION builds. A stand-in onnxruntime keeps these off the real GPU: what is
# checked is the decision, the options DirectML requires, and that a GPU that
# fails costs speed and never the feature.

class _FakeSession:
    def __init__(self, ort, path, options, providers):
        self.ort, self.path, self.options, self.providers = ort, path, options, providers
        self.gpu = isinstance(providers[0], tuple) and providers[0][0] == accelerator.DML_PROVIDER
        self.runs = 0

    def run(self, output_names, feed, run_options=None):
        self.runs += 1
        if feed.get("bad"):
            raise ValueError("input 'x' has the wrong shape")
        if self.gpu and self.ort.fail_run:
            raise RuntimeError("887A0005 The GPU device instance has been suspended.")
        return ["gpu" if self.gpu else "cpu"]

    def get_inputs(self):
        return [SimpleNamespace(name="x", shape=[1, 3])]


class _FakeOrt:
    """Just enough onnxruntime: a provider list, options, sessions that fail on cue."""

    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"
        ORT_PARALLEL = "parallel"

    class SessionOptions:
        pass

    def __init__(self):
        self.providers = [accelerator.DML_PROVIDER, accelerator.CPU_PROVIDER]
        self.fail_build = False
        self.fail_run = False
        self.sessions: list[_FakeSession] = []

    def get_available_providers(self):
        return list(self.providers)

    def InferenceSession(self, path, sess_options=None, providers=None):
        if self.fail_build and isinstance(providers[0], tuple):
            raise RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Exception during "
                               "initialization: 80070057 The parameter is incorrect.")
        session = _FakeSession(self, path, sess_options, providers)
        self.sessions.append(session)
        return session


@pytest.fixture
def fake_ort(monkeypatch):
    ort = _FakeOrt()
    lines: list[str] = []
    monkeypatch.setattr(accelerator, "_onnxruntime", lambda: ort)
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: False)
    monkeypatch.setattr(accelerator, "_adapter_name", lambda: "Test Adapter")
    monkeypatch.delenv(accelerator.SWITCH, raising=False)
    accelerator.reset()
    accelerator.set_log(lines.append)
    yield ort, lines
    accelerator.set_log(None)
    accelerator.reset()


def test_each_model_goes_where_it_was_measured_faster(fake_ort):
    ort, _lines = fake_ort
    assert {m for m in accelerator.POLICY if accelerator.wants_gpu(m)} == {
        "ocr_det", "ocr_rec", "object_detection", "semantic_batch"}
    # Kokoro and YAMNet fail under DirectML; the others measured faster on the CPU.
    for model in ("kokoro", "sound_sense", "voiceprint", "semantic_query", "ocr_cls", "unknown"):
        assert not accelerator.wants_gpu(model), model
    det = accelerator.create_session("ocr_det", "det.onnx")
    assert det.on_gpu and det.device == "GPU"
    assert det.session.options.enable_mem_pattern is False      # DirectML requires both
    assert det.session.options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert det.session.providers == [(accelerator.DML_PROVIDER, {"device_id": 0}),
                                     accelerator.CPU_PROVIDER]
    voice = accelerator.create_session("voiceprint", "encoder.onnx", threads=1, inter_threads=1)
    assert not voice.on_gpu and voice.session.providers == [accelerator.CPU_PROVIDER]
    assert voice.session.options.intra_op_num_threads == 1
    assert not hasattr(voice.session.options, "enable_mem_pattern"), \
        "a CPU session keeps onnxruntime's defaults"
    assert det.run(None, {}) == ["gpu"] and voice.run(None, {}) == ["cpu"]
    assert det.get_inputs()[0].name == "x", "everything but run is the live session's"


def test_orion_gpu_0_keeps_every_model_on_the_cpu(fake_ort, monkeypatch):
    _ort, lines = fake_ort
    monkeypatch.setenv("ORION_GPU", "0")
    assert not any(accelerator.wants_gpu(m) for m in accelerator.POLICY)
    session = accelerator.create_session("ocr_det", "det.onnx")
    assert not session.on_gpu and session.session.providers == [accelerator.CPU_PROVIDER]
    assert lines == ["ACCEL: CPU only - ORION_GPU=0; every model on the CPU."]
    assert set(accelerator.describe()["models"].values()) == {accelerator.CPU}


def test_without_the_directml_provider_everything_runs_on_the_cpu(fake_ort):
    ort, lines = fake_ort
    ort.providers = [accelerator.CPU_PROVIDER]           # plain onnxruntime
    assert not accelerator.create_session("ocr_rec", "rec.onnx").on_gpu
    assert lines == ["ACCEL: CPU only - onnxruntime has no DirectML provider; "
                     "every model on the CPU."]


def test_the_summary_says_what_runs_where_once(fake_ort):
    _ort, lines = fake_ort
    accelerator.create_session("ocr_det", "det.onnx")
    accelerator.create_session("voiceprint", "encoder.onnx")
    assert lines == ["ACCEL: GPU = DirectML (Test Adapter) - GPU: ocr_det, ocr_rec, "
                     "object_detection, semantic_batch | CPU: ocr_cls, semantic_query, "
                     "voiceprint, sound_sense, kokoro."]


def test_a_gpu_session_that_will_not_build_is_rebuilt_on_the_cpu(fake_ort):
    ort, lines = fake_ort
    ort.fail_build = True
    session = accelerator.create_session("ocr_det", "det.onnx")
    assert not session.on_gpu and session.run(None, {}) == ["cpu"]
    ort.fail_build = False
    assert not accelerator.create_session("ocr_det", "det.onnx").on_gpu, \
        "a model that failed stays on the CPU for the rest of the run"
    said = [line for line in lines if "could not start on the GPU" in line]
    assert len(said) == 1 and "ocr_det" in said[0] and "80070057" in said[0]
    assert accelerator.describe()["models"]["ocr_det"].startswith("cpu (could not start")
    assert accelerator.create_session("ocr_rec", "rec.onnx").on_gpu, "other models keep the GPU"


def test_a_failed_gpu_run_is_answered_on_the_cpu_and_stays_there(fake_ort):
    ort, lines = fake_ort
    session = accelerator.create_session("ocr_rec", "rec.onnx")
    assert session.on_gpu
    ort.fail_run = True
    assert session.run(None, {}) == ["cpu"], "the caller still gets its answer"
    assert not session.on_gpu and session.device == "CPU"
    assert session.run(None, {}) == ["cpu"]
    gpu, cpu = ort.sessions
    assert gpu.runs == 1 and cpu.runs == 2, "once it failed, the GPU is not tried again"
    assert len([line for line in lines if "failed on the GPU" in line]) == 1
    assert not accelerator.wants_gpu("ocr_rec")


def test_a_bad_input_is_not_blamed_on_the_gpu(fake_ort):
    _ort, lines = fake_ort
    session = accelerator.create_session("ocr_det", "det.onnx")
    with pytest.raises(ValueError):
        session.run(None, {"bad": True})
    assert session.on_gpu, "the CPU failed too, so the input was the problem"
    assert session.run(None, {}) == ["gpu"]
    assert not any("failed on the GPU" in line for line in lines)


def test_no_gpu_session_is_built_on_the_gui_thread(fake_ort, monkeypatch):
    _ort, lines = fake_ort
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: True)
    assert not accelerator.create_session("ocr_det", "det.onnx").on_gpu
    assert any("built on the GUI thread" in line for line in lines)
    assert accelerator.providers_for("ocr_det") == [accelerator.CPU_PROVIDER]


def test_the_gui_thread_is_the_main_thread_of_a_qt_application(monkeypatch):
    app = SimpleNamespace(QCoreApplication=SimpleNamespace(instance=lambda: object()))
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", app)
    assert accelerator.on_gui_thread()
    seen = []
    worker = threading.Thread(target=lambda: seen.append(accelerator.on_gui_thread()))
    worker.start()
    worker.join()
    assert seen == [False]
    none = SimpleNamespace(QCoreApplication=SimpleNamespace(instance=lambda: None))
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", none)
    assert not accelerator.on_gui_thread(), "no Qt application, no GUI thread"


def test_gpu_runs_are_one_at_a_time(fake_ort):
    """DirectML allows one Run per session at a time (onnxruntime's docs)."""
    _ort, _lines = fake_ort
    session = accelerator.create_session("semantic_batch", "bge.onnx")
    active, peak = [0], [0]

    def slow_run(output_names, feed, run_options=None):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        active[0] -= 1
        return ["gpu"]

    session.session.run = slow_run
    threads = [threading.Thread(target=session.run, args=(None, {})) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert peak[0] == 1


# ── speech recognition on the NVIDIA GPU (28 Oct 2026) ─────────────────────

def test_whisper_runs_on_the_gpu_when_cuda_works_and_falls_back_when_not(monkeypatch, tmp_path):
    """base.en: 824 ms on the CPU -> 271-317 ms on the RTX 3060 Ti, the same
    words. A GPU model that fails its first run is never trusted."""
    from orion_core import speech_offline

    made = []

    class Model:
        def __init__(self, size, device="cpu", compute_type="", **_kw):
            made.append((device, compute_type))
            self.device = device

        def transcribe(self, audio, **_kw):
            if self.device == "cuda" and broken["gpu"]:
                raise RuntimeError("Library cublas64_12.dll is not found")
            return iter(()), None

    broken = {"gpu": False}
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    monkeypatch.setattr(accelerator, "cuda_ready", lambda: True)
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: False)
    lines = []
    transcriber = speech_offline.OfflineTranscriber(model_size="base.en", defer=True)
    monkeypatch.setattr(transcriber, "_log", lines.append)
    transcriber._select_engine()
    assert made == [("cuda", "float16")] and transcriber.device == "cuda"
    assert "on the GPU" in lines[-1]

    made.clear()
    broken["gpu"] = True
    again = speech_offline.OfflineTranscriber(model_size="base.en", defer=True)
    monkeypatch.setattr(again, "_log", lines.append)
    again._select_engine()
    assert made == [("cuda", "float16"), ("cpu", "int8")] and again.device == "cpu"
    assert any("using the CPU" in line for line in lines)

    made.clear()
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: True)   # never on the GUI thread
    third = speech_offline.OfflineTranscriber(model_size="base.en", defer=True)
    third._select_engine()
    assert made == [("cpu", "int8")]


def test_the_gpu_hears_with_small_en_unless_a_size_was_chosen(monkeypatch):
    """29 Oct: small.en on the GPU made no word errors at 0 dB room noise
    (base.en 6.2%) in less time than base.en took there before (317 ms)."""
    from orion_core import speech_offline

    made, missing = [], set()

    class Model:
        def __init__(self, size, device="cpu", **_kw):
            if size in missing:
                raise RuntimeError(f"{size} is not in the local cache")
            made.append((size, device))

        def transcribe(self, audio, **kw):
            Model.options = kw
            return iter(()), None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    monkeypatch.setattr(accelerator, "cuda_ready", lambda: True)
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: False)
    monkeypatch.delenv("ORION_WHISPER_MODEL", raising=False)
    plain = speech_offline.OfflineTranscriber(defer=True)
    plain._log = lambda _line: None
    plain._select_engine()
    assert made == [("small.en", "cuda")] and plain.model_size == "small.en"
    plain._run_engine(b"\x00\x00" * 1600)
    assert Model.options["without_timestamps"] is True

    made.clear()
    missing.add("small.en")                       # not downloaded on this machine
    fallback = speech_offline.OfflineTranscriber(defer=True)
    fallback._log = lambda _line: None
    fallback._select_engine()
    assert made == [("base.en", "cuda")] and fallback.device == "cuda"

    made.clear()
    chosen = speech_offline.OfflineTranscriber(model_size="tiny.en", defer=True)
    chosen._log = lambda _line: None
    chosen._select_engine()
    assert made == [("tiny.en", "cuda")]


def test_cuda_is_not_used_without_its_runtime_or_with_orion_gpu_0(monkeypatch, tmp_path):
    monkeypatch.setattr(accelerator, "_cuda", None)
    monkeypatch.setattr(accelerator, "CUDA_RUNTIME", tmp_path / "missing")
    assert accelerator.cuda_ready() is False                 # no cuBLAS there
    monkeypatch.setattr(accelerator, "_cuda", True)
    monkeypatch.setenv("ORION_GPU", "0")
    assert accelerator.cuda_ready() is False                 # the switch wins



def test_a_cpu_reload_after_an_idle_release_goes_back_to_base_en(monkeypatch):
    """A GPU load switched the size to small.en; a reload that could not use
    the GPU loaded small.en on the CPU, about three times slower."""
    from orion_core import speech_offline
    made = []

    class Model:
        def __init__(self, size, device="cpu", **_kw):
            made.append((size, device))

        def transcribe(self, audio, **_kw):
            return iter(()), None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: False)
    monkeypatch.delenv("ORION_WHISPER_MODEL", raising=False)
    monkeypatch.setattr(accelerator, "cuda_ready", lambda: True)
    transcriber = speech_offline.OfflineTranscriber(defer=True)
    transcriber._log = lambda _line: None
    transcriber._select_engine()
    monkeypatch.setattr(accelerator, "cuda_ready", lambda: False)      # VRAM busy now
    transcriber._select_engine()
    assert made == [("small.en", "cuda"), ("base.en", "cpu")]
    assert transcriber.model_size == "base.en"
