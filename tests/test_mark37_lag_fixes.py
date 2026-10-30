"""Lag found in the owner's Mark 37.2 sessions (24-25 Oct 2026).

* 20-27 s after every launch the background indexers built the DirectML
  device beside the face; the GUI thread sat in Qt/native code for
  169+122+325+668 ms (25 Oct 11:50:36-38), 138+122+325 ms (11:58:04-05).
* Every shutdown imported world_feeds just to close it: 435-544 ms blocked.
* "Ollama not detected ... install from ollama.com" beside "a local model is
  installed but not running".
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orion_core import semantic

ROOT = Path(__file__).resolve().parents[1]


class _Session:
    def __init__(self, on_gpu: bool) -> None:
        self.on_gpu, self.rows = on_gpu, []

    def run(self, _names, feed, _options=None):
        rows, width = feed["input_ids"].shape
        self.rows.append(rows)
        return [np.ones((rows, width, semantic.MODEL.dim), np.float32)]


class _Tokens:
    def encode_batch(self, texts):
        return [SimpleNamespace(ids=[1, 2, 3], attention_mask=[1, 1, 1]) for _ in texts]


@pytest.fixture()
def encoder(monkeypatch, tmp_path):
    from orion_core import accelerator
    enc = semantic.SemanticEncoder()
    cpu, gpu = _Session(False), _Session(True)
    enc._session, enc._tokenizer = cpu, _Tokens()
    enc._input_names = {"input_ids", "attention_mask"}
    built = []
    monkeypatch.setattr(enc, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(accelerator, "wants_gpu", lambda model: model == "semantic_batch")
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: False)
    monkeypatch.setattr(accelerator, "create_session",
                        lambda model, path, **kw: built.append(model) or gpu)
    monkeypatch.setattr(semantic, "ENCODER", enc)
    return enc, cpu, gpu, built


def test_a_boot_top_up_never_builds_the_gpu_session(encoder):
    enc, cpu, gpu, built = encoder
    texts = [f"row {i}" for i in range(64)]
    semantic.encode_for_index(texts, backlog=64)
    assert built == [] and gpu.rows == [] and sum(cpu.rows) == 64


def test_a_real_backlog_still_uses_the_gpu(encoder):
    enc, cpu, gpu, built = encoder
    texts = [f"row {i}" for i in range(64)]
    semantic.encode_for_index(texts, backlog=enc.GPU_BACKLOG_ROWS)
    assert built == ["semantic_batch"] and gpu.rows == [64]


def test_both_indexers_go_through_the_backlog_rule():
    for name in ("memory.py", "knowledge_graph.py"):
        source = (ROOT / "orion_core" / name).read_text(encoding="utf-8")
        assert "semantic.encode_for_index(" in source, name
        assert "semantic.ENCODER.encode([text for _id, text in chunk])" not in source, name


def test_shutdown_does_not_import_world_feeds_to_close_it():
    tree = ast.parse((ROOT / "orion_core" / "app.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module == "world_feeds" and any(a.name == "shutdown_world_feeds"
                                                         for a in node.names)]
    assert imports == [], "teardown imported world_feeds (0.5 s on the loop) to close it"


def test_the_installed_app_ends_without_native_teardown(monkeypatch):
    """25 Oct 12:05:09: a clean shutdown, then an access violation in native
    finalisation (no Python frame). The frozen app now flushes and _exits."""
    from orion_core import app, session_log
    exits, flushed = [], []
    # Mark 38: the process is ended by TerminateProcess (no DLL unload code);
    # patched here, or this test would end the test run itself.
    monkeypatch.setattr(app, "_terminate_process", exits.append)
    monkeypatch.setattr(app.os, "_exit", lambda code: pytest.fail("os._exit reached"))
    monkeypatch.setattr(session_log, "flush", lambda *a, **k: flushed.append(True))
    monkeypatch.delattr(app.sys, "frozen", raising=False)
    app._exit_past_native_teardown()
    assert exits == []                        # a source run exits normally
    monkeypatch.setattr(app.sys, "frozen", True, raising=False)
    monkeypatch.setenv("ORION_NATIVE_TEARDOWN", "1")
    app._exit_past_native_teardown()
    assert exits == []                        # opt-out for diagnosis
    monkeypatch.delenv("ORION_NATIVE_TEARDOWN")
    app._exit_past_native_teardown()
    assert exits == [0] and flushed == [True]


def test_an_installed_but_stopped_ollama_is_not_told_to_install(monkeypatch):
    from orion_core import local_mind, local_models
    lines = []
    manager = local_models.OllamaManager.__new__(local_models.OllamaManager)
    manager._log = lines.append
    manager.probe = lambda timeout=None: False
    monkeypatch.setattr(local_mind, "ollama_binary", lambda: Path("C:/Ollama/ollama.exe"))
    assert manager.register(SimpleNamespace(providers={}, provider_order=[])) is False
    assert "installed but not running" in lines[-1] and "install from" not in lines[-1]
    monkeypatch.setattr(local_mind, "ollama_binary", lambda: None)
    manager.register(SimpleNamespace(providers={}, provider_order=[]))
    assert "install from ollama.com" in lines[-1]
