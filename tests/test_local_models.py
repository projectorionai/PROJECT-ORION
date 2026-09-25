"""
Tests for local_models.py — Ollama discovery/registration, and the
RAM-budget-aware model selection added for modest hardware (e.g. 16GB RAM /
8GB VRAM), where blindly picking the "strongest" pulled model regardless of
its size is exactly what turns the cloud-outage fallback into the reason
the machine starts swapping.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.local_models import OllamaManager, RECOMMENDED_PULLS
from orion_core.providers import OrionProviderSettings


class _Signal:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, message: str) -> None:
        self.messages.append(message)

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __init__(self) -> None:
        self.log = _Signal()

    def __getattr__(self, name):
        return _Signal()


class _FakeUrlResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _mock_tags(monkeypatch, models: list[dict]) -> None:
    body = json.dumps({"models": models}).encode("utf-8")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeUrlResponse(body))


def _tag(name: str, size_gb: float | None = None) -> dict:
    entry = {"name": name}
    if size_gb is not None:
        entry["size"] = int(size_gb * (1024 ** 3))
    return entry


# ── probe() ─────────────────────────────────────────────────────────────────

def test_probe_populates_models_and_sizes(monkeypatch):
    _mock_tags(monkeypatch, [_tag("llama3.1:latest", 4.9), _tag("mistral:7b", 4.1)])
    manager = OllamaManager()
    assert manager.probe() is True
    assert manager.models == ["llama3.1:latest", "mistral:7b"]
    assert manager.model_sizes["llama3.1:latest"] == int(4.9 * (1024 ** 3))
    assert manager.available is True


def test_probe_degrades_cleanly_when_ollama_unreachable(monkeypatch):
    def _raise(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    manager = OllamaManager()
    assert manager.probe() is False
    assert manager.available is False
    assert manager.models == []
    assert manager.model_sizes == {}


def test_probe_tolerates_entries_with_no_size(monkeypatch):
    _mock_tags(monkeypatch, [{"name": "custom:latest"}])
    manager = OllamaManager()
    assert manager.probe() is True
    assert manager.models == ["custom:latest"]
    assert "custom:latest" not in manager.model_sizes


# ── best_model(): RAM-budget-aware selection ─────────────────────────────────

def test_best_model_prefers_strongest_when_ram_unknown(monkeypatch):
    manager = OllamaManager()
    manager.models = ["mistral:7b", "qwen2.5:14b"]
    manager.model_sizes = {"mistral:7b": int(4.1 * 1024 ** 3), "qwen2.5:14b": int(9.0 * 1024 ** 3)}
    monkeypatch.setattr(OllamaManager, "_system_ram_gb", staticmethod(lambda: None))
    # Budget is unknowable (psutil missing) -> falls back to pure name
    # preference, same as before this feature existed.
    assert manager.best_model() == "qwen2.5:14b"


def test_best_model_skips_an_oversized_top_preference_model(monkeypatch):
    manager = OllamaManager()
    manager.models = ["qwen2.5:14b", "mistral:7b"]
    manager.model_sizes = {"qwen2.5:14b": int(9.0 * 1024 ** 3), "mistral:7b": int(4.1 * 1024 ** 3)}
    # 16GB machine -> ~5.6GB budget (35%): qwen2.5:14b (9GB) blows it,
    # mistral:7b (4.1GB) fits -> the smaller, lower-preference model wins.
    monkeypatch.setattr(OllamaManager, "_system_ram_gb", staticmethod(lambda: 16.0))
    assert manager.best_model() == "mistral:7b"


def test_best_model_falls_back_to_strongest_when_nothing_fits_budget(monkeypatch):
    manager = OllamaManager()
    manager.models = ["qwen2.5:14b"]
    manager.model_sizes = {"qwen2.5:14b": int(9.0 * 1024 ** 3)}
    monkeypatch.setattr(OllamaManager, "_system_ram_gb", staticmethod(lambda: 16.0))
    # A working (if oversized) local brain beats none.
    assert manager.best_model() == "qwen2.5:14b"


def test_best_model_with_no_pulled_models_is_blank():
    manager = OllamaManager()
    assert manager.best_model() == ""


# ── register() ────────────────────────────────────────────────────────────

def test_register_creates_a_fresh_local_ollama_profile(monkeypatch):
    _mock_tags(monkeypatch, [_tag("llama3.1:latest", 4.9)])
    bus = _StubBus()
    manager = OllamaManager(bus)
    settings = OrionProviderSettings(active_provider="none", provider_order=[], providers={})

    assert manager.register(settings) is True
    profile = settings.providers["local_ollama"]
    assert profile.model == "llama3.1:latest"
    assert profile.enabled is True
    assert "local_ollama" in settings.provider_order


def test_register_reports_no_server_detected(monkeypatch):
    def _raise(req, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    bus = _StubBus()
    manager = OllamaManager(bus)
    settings = OrionProviderSettings(active_provider="none", provider_order=[], providers={})

    assert manager.register(settings) is False
    assert any("not detected" in m for m in bus.log.messages)


def test_register_warns_when_the_chosen_model_is_oversized(monkeypatch):
    _mock_tags(monkeypatch, [_tag("qwen2.5:14b", 9.0)])
    bus = _StubBus()
    manager = OllamaManager(bus)
    monkeypatch.setattr(OllamaManager, "_system_ram_gb", staticmethod(lambda: 16.0))
    settings = OrionProviderSettings(active_provider="none", provider_order=[], providers={})

    manager.register(settings)

    assert any("larger than" in m and "llama3.2:3b" in m for m in bus.log.messages)


def test_register_does_not_warn_when_the_chosen_model_fits(monkeypatch):
    _mock_tags(monkeypatch, [_tag("llama3.2:3b", 2.0)])
    bus = _StubBus()
    manager = OllamaManager(bus)
    monkeypatch.setattr(OllamaManager, "_system_ram_gb", staticmethod(lambda: 16.0))
    settings = OrionProviderSettings(active_provider="none", provider_order=[], providers={})

    manager.register(settings)

    assert not any("larger than" in m for m in bus.log.messages)


# ── recommendations() ────────────────────────────────────────────────────────

def test_recommended_pulls_lead_with_lightweight_models():
    # First recommendation should be a small (<3GB) model, not the biggest
    # option — this is a fallback brain, not the user's primary model.
    first_tag, _why = RECOMMENDED_PULLS[0]
    assert "3b" in first_tag.lower() or "3.8b" in first_tag.lower()


def test_recommendations_excludes_already_pulled_families(monkeypatch):
    _mock_tags(monkeypatch, [_tag("qwen2.5:7b", 4.7)])
    manager = OllamaManager()
    manager.probe()
    recs = manager.recommendations()
    assert all(not tag.startswith("qwen2.5") for tag, _why in recs)
