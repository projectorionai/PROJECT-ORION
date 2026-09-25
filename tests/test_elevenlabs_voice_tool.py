"""
Tests for the elevenlabs_voice dispatcher tool (dispatch_desktop.py) — lets
a voice/text command check status, list the account's voices, or set the
permanent voice_id, mirroring the audio_devices tool's own pattern. The API
key itself is deliberately not settable through this tool (config/api_keys
.json directly, same convention as every other provider credential).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import voice_elevenlabs as ev
from orion_core.dispatcher import OrionDispatcher


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


def _isolate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ev, "API_CONFIG_PATH", tmp_path / "api_keys.json")
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "tts_cache")


def _dispatcher() -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _StubBus()
    return d


def test_status_reports_not_configured_by_default(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher()
    result = d.elevenlabs_voice_tool({"action": "status"})
    assert result.ok
    assert "not set" in result.text.lower()


def test_status_reports_configured_state(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test", voice_id="v123")
    d = _dispatcher()
    result = d.elevenlabs_voice_tool({"action": "status"})
    assert "configured" in result.text.lower()
    assert "v123" in result.text


def test_set_voice_id_persists_the_choice(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher()
    result = d.elevenlabs_voice_tool({"action": "set_voice_id", "voice_id": "v999"})
    assert result.ok
    assert ev.resolve_voice_id() == "v999"


def test_set_voice_id_with_no_id_supplied_fails_cleanly(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher()
    result = d.elevenlabs_voice_tool({"action": "set_voice_id"})
    assert not result.ok


def test_list_reports_no_voices_when_unconfigured(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher()
    result = d.elevenlabs_voice_tool({"action": "list"})
    assert not result.ok


def test_list_reports_voices_from_the_account(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    ev.set_voice(api_key="sk-test")
    monkeypatch.setattr(
        ev, "list_voices",
        lambda: [{"voice_id": "v1", "name": "Rachel"}, {"voice_id": "v2", "name": "Adam"}])
    d = _dispatcher()
    result = d.elevenlabs_voice_tool({"action": "list"})
    assert result.ok
    assert "Rachel" in result.text
    assert "v1" in result.text


def test_unknown_action_reports_the_supported_ones(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    d = _dispatcher()
    result = d.elevenlabs_voice_tool({"action": "explode"})
    assert not result.ok
    assert "status" in result.text and "list" in result.text


def test_tool_is_registered_in_the_handler_table():
    d = _dispatcher()
    assert d.handler_table()["elevenlabs_voice"] == d.elevenlabs_voice_tool
