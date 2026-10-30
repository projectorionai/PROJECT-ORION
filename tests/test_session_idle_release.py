"""Idle ONNX Runtime sessions let their memory go (accelerator.release_idle).

Every session lived as long as the object that built it, so one screen read or
one re-index kept its memory for the rest of the run - a DirectML session ~1 GB
committed (semantic.py), RapidOCR's three models more. A session unused for
IDLE_RELEASE_S is now released and rebuilt by its next run, never mid-run.

A stand-in onnxruntime keeps all of this off the real GPU.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import accelerator  # noqa: E402


class _FakeSession:
    def __init__(self, ort, path, options, providers):
        self.ort, self.path, self.providers = ort, path, providers
        self.gpu = isinstance(providers[0], tuple)
        self.runs = 0
        ort.alive += 1

    def __del__(self):
        self.ort.alive -= 1

    def run(self, output_names, feed, run_options=None):
        self.runs += 1
        gate = feed.get("gate")
        if gate is not None:
            feed["started"].set()
            gate.wait(5)
        return ["gpu" if self.gpu else "cpu"]

    def get_inputs(self):
        return [SimpleNamespace(name="x")]


class _FakeOrt:
    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    class SessionOptions:
        pass

    def __init__(self):
        self.providers = [accelerator.DML_PROVIDER, accelerator.CPU_PROVIDER]
        self.builds = 0
        self.alive = 0

    def get_available_providers(self):
        return list(self.providers)

    def InferenceSession(self, path, sess_options=None, providers=None):
        session = _FakeSession(self, path, sess_options, providers)
        self.builds += 1
        return session


@pytest.fixture
def fake_ort(monkeypatch):
    ort = _FakeOrt()
    lines: list[str] = []
    monkeypatch.setattr(accelerator, "_onnxruntime", lambda: ort)
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: False)
    monkeypatch.setattr(accelerator, "_adapter_name", lambda: "Test Adapter")
    monkeypatch.delenv(accelerator.SWITCH, raising=False)
    monkeypatch.delenv(accelerator.IDLE_SWITCH, raising=False)
    accelerator.reset()
    accelerator.set_log(lines.append)
    yield ort, lines
    accelerator.set_log(None)
    accelerator.reset()


def _later(seconds: float) -> float:
    return time.monotonic() + seconds


def test_an_idle_gpu_session_is_released_and_rebuilt_on_next_use(fake_ort):
    ort, lines = fake_ort
    det = accelerator.create_session("ocr_det", "det.onnx")
    assert det.run(None, {}) == ["gpu"] and ort.alive == 1
    assert accelerator.release_idle(60.0, now=_later(30)) == [], "not idle long enough"
    assert accelerator.release_idle(60.0, now=_later(61)) == ["ocr_det"]
    assert det.released and ort.alive == 0, "the last reference went: memory freed"
    assert det.on_gpu, "it still says where it will run"
    said = [line for line in lines if line.startswith("ACCEL: released idle")]
    assert len(said) == 1 and said[0].startswith("ACCEL: released idle ocr_det session (~")
    assert said[0].endswith(" MB).")
    assert det.run(None, {}) == ["gpu"], "rebuilt lazily, on the GPU again"
    assert not det.released and ort.builds == 2
    assert det.get_inputs()[0].name == "x"
    assert accelerator.release_idle(60.0, now=_later(10)) == [], "just used again"


def test_a_session_is_never_released_mid_run(fake_ort):
    ort, _lines = fake_ort
    rec = accelerator.create_session("ocr_rec", "rec.onnx")
    gate, started = threading.Event(), threading.Event()
    worker = threading.Thread(target=rec.run, args=(None, {"gate": gate, "started": started}))
    worker.start()
    assert started.wait(5)
    try:
        assert accelerator.release_idle(0.001, now=_later(3600)) == []
        assert not rec.released and ort.alive == 1
    finally:
        gate.set()
        worker.join(5)
    assert accelerator.release_idle(0.001, now=_later(3600)) == ["ocr_rec"]


def test_a_rebuild_in_progress_is_not_released(fake_ort):
    _ort, _lines = fake_ort
    obj = accelerator.create_session("object_detection", "yolo.onnx")
    with obj._use_lock:                  # what a run or a rebuild holds
        assert not obj.release_if_idle(0.0, now=_later(3600))
    assert obj.release_if_idle(0.0, now=_later(3600))


def test_small_cpu_models_on_the_voice_path_stay_resident(fake_ort):
    _ort, _lines = fake_ort
    for model in accelerator.KEEP_RESIDENT:
        accelerator.create_session(model, f"{model}.onnx")
    cls = accelerator.create_session("ocr_cls", "cls.onnx")      # OCR on the CPU
    released = accelerator.release_idle(1.0, now=_later(3600))
    assert released == ["ocr_cls"] and cls.released


def test_a_release_on_the_gui_thread_rebuilds_on_the_cpu(fake_ort, monkeypatch):
    _ort, lines = fake_ort
    batch = accelerator.create_session("semantic_batch", "bge.onnx")
    assert accelerator.release_idle(1.0, now=_later(10)) == ["semantic_batch"]
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: True)
    assert batch.run(None, {}) == ["cpu"], "never a GPU build on the GUI thread"
    assert any("built on the GUI thread" in line for line in lines)


def test_switched_off_with_zero(fake_ort, monkeypatch):
    _ort, _lines = fake_ort
    monkeypatch.setenv(accelerator.IDLE_SWITCH, "0")
    accelerator.create_session("ocr_det", "det.onnx")
    assert accelerator.idle_release_s() == 0.0
    assert accelerator.release_idle(now=_later(10 ** 6)) == []


def test_concurrent_runs_and_releases_always_answer(fake_ort):
    ort, _lines = fake_ort
    session = accelerator.create_session("ocr_rec", "rec.onnx")
    answers: list[str] = []
    stop = threading.Event()

    def reap():
        while not stop.is_set():
            accelerator.release_idle(0.001, now=_later(3600))

    reaper = threading.Thread(target=reap)
    reaper.start()
    try:
        for _ in range(300):
            answers.extend(session.run(None, {}))
    finally:
        stop.set()
        reaper.join(5)
    assert answers == ["gpu"] * 300
    assert ort.alive <= 1


def test_sessions_registered_at_once_start_one_reaper():
    """Review, 29 Oct 2026: the reaper was started outside the lock, so two
    sessions registered together could start the same thread twice."""
    import threading as _threading
    from orion_core import accelerator as acc

    class _Session:
        pass

    errors = []
    acc._reaper = None
    barrier = _threading.Barrier(8)

    def register():
        barrier.wait()
        try:
            acc._register(_Session())
        except RuntimeError as exc:
            errors.append(exc)

    workers = [_threading.Thread(target=register) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert errors == [] and acc._reaper is not None and acc._reaper.is_alive()
