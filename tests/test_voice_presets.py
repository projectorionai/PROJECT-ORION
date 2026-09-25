"""
Character voice presets (Mark XXII) — "make your voice more like Ultron."

The ElevenLabs config is redirected to a temp file so no test ever touches the
real config/api_keys.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import voice_elevenlabs as ev  # noqa: E402


@pytest.fixture()
def temp_config(tmp_path, monkeypatch):
    cfg = tmp_path / "api_keys.json"
    monkeypatch.setattr(ev, "API_CONFIG_PATH", cfg)
    return cfg


# ── the presets exist and are well-formed ────────────────────────────────────

def test_ultron_preset_exists_and_is_deep():
    assert "ultron" in ev.VOICE_PRESETS
    ultron = ev.VOICE_PRESETS["ultron"]
    assert ultron["voice_id"]                       # a real ElevenLabs voice id
    assert "menac" in ultron["description"].lower() or "deep" in ultron["description"].lower()
    s = ultron["settings"]
    assert 0.0 <= s["stability"] <= 1.0
    assert 0.0 <= s["style"] <= 1.0


def test_every_preset_has_a_voice_id_and_settings():
    for name, p in ev.VOICE_PRESETS.items():
        assert p["voice_id"], name
        assert set(p["settings"]) >= {"stability", "similarity_boost", "style"}


# ── applying a preset ────────────────────────────────────────────────────────

def test_apply_ultron_sets_voice_and_preset(temp_config):
    msg = ev.apply_preset("ultron")
    assert "Ultron" in msg
    assert ev.active_preset() == "ultron"
    assert ev.resolve_voice_id() == ev.VOICE_PRESETS["ultron"]["voice_id"]
    # persisted to the (temp) config
    data = json.loads(temp_config.read_text(encoding="utf-8"))
    assert data["elevenlabs"]["preset"] == "ultron"


def test_apply_is_case_insensitive(temp_config):
    ev.apply_preset("ULTRON")
    assert ev.active_preset() == "ultron"


def test_unknown_preset_is_reported_not_applied(temp_config):
    msg = ev.apply_preset("darth_vader")
    assert "No voice preset" in msg
    assert ev.active_preset() == ""


def test_apply_preserves_an_existing_api_key(temp_config):
    ev.set_voice(api_key="sk-test-key-123456")
    ev.apply_preset("ultron")
    assert ev.resolve_api_key() == "sk-test-key-123456"    # key untouched
    assert ev.resolve_voice_id() == ev.VOICE_PRESETS["ultron"]["voice_id"]


# ── the preset shapes delivery ───────────────────────────────────────────────

def test_active_preset_drives_delivery_settings(temp_config):
    ev.apply_preset("ultron")
    # the pure emotion mapping is unchanged...
    assert ev.voice_settings_for_emotion("neutral") == ev._EMOTION_VOICE_SETTINGS["neutral"]
    # ...but the DELIVERY resolver applies the preset
    settings = ev.resolve_delivery_settings("neutral")
    assert settings["stability"] == ev.VOICE_PRESETS["ultron"]["settings"]["stability"]
    assert settings["style"] == ev.VOICE_PRESETS["ultron"]["settings"]["style"]


def test_emotion_nudges_within_the_preset(temp_config):
    ev.apply_preset("ultron")
    base = ev.VOICE_PRESETS["ultron"]["settings"]["stability"]
    urgent = ev.resolve_delivery_settings("urgent")["stability"]
    calm = ev.resolve_delivery_settings("calm")["stability"]
    assert urgent < base < calm                     # urgent looser, calm steadier


def test_without_a_preset_the_emotion_table_is_used(temp_config):
    # no preset applied → resolver falls back to the plain emotion settings
    assert ev.active_preset() == ""
    assert ev.resolve_delivery_settings("neutral") == \
        ev._EMOTION_VOICE_SETTINGS["neutral"]


# ── the tool + schema ────────────────────────────────────────────────────────

def test_tool_and_schema_expose_presets():
    import inspect
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    src = inspect.getsource(DesktopDispatchMixin.elevenlabs_voice_tool)
    assert "apply_preset" in src
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "elevenlabs_voice")
    assert "ultron" in tool["description"].lower()
    assert "preset" in tool["parameters"]["properties"]
