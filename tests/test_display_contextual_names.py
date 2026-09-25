"""
Tests for DisplayTopologyManager's contextual monitor naming (Mark XX
architectural-audit pass, Track I) — "Coding Monitor"/"Gaming Monitor" style
labels instead of a raw OS device string or bare index. User-set names are
persisted to config/display_names.json; anything unnamed falls back to an
objective geometry-derived role rather than a guessed semantic label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.display import DisplayTopologyManager, Monitor


def _manager(tmp_path, monkeypatch) -> DisplayTopologyManager:
    import orion_core.display as dm
    monkeypatch.setattr(dm, "_DISPLAY_NAMES_PATH", tmp_path / "display_names.json")
    monkeypatch.setattr(dm, "CONFIG_DIR", tmp_path)
    # Avoid touching real hardware enumeration in a headless test run — a
    # deterministic single-monitor fallback is exactly what refresh() does
    # itself when no real enumeration succeeds, so no monkeypatch needed
    # for _enumerate_windows/_enumerate_screeninfo on a CI box with no
    # physical monitors; on a dev box with real monitors this still works
    # identically since contextual_name only depends on geometry + names.
    return DisplayTopologyManager()


def _primary_monitor(width=1920, height=1080) -> Monitor:
    return Monitor(index=0, x=0, y=0, width=width, height=height,
                   is_primary=True, scale=1.0, name="\\\\.\\DISPLAY1")


def _secondary_monitor(width=1920, height=1080) -> Monitor:
    return Monitor(index=1, x=1920, y=0, width=width, height=height,
                   is_primary=False, scale=1.0, name="\\\\.\\DISPLAY2")


# ── geometry-derived fallback ────────────────────────────────────────────────

def test_primary_monitor_gets_a_primary_label(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    assert manager.contextual_name(_primary_monitor()) == "Primary Monitor"


def test_non_primary_landscape_monitor_gets_secondary_label(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    assert manager.contextual_name(_secondary_monitor()) == "Secondary Display"


def test_a_taller_than_wide_monitor_is_vertical_regardless_of_primary(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    portrait = Monitor(index=1, x=1920, y=0, width=1080, height=1920,
                        is_primary=False, scale=1.0, name="\\\\.\\DISPLAY2")
    assert manager.contextual_name(portrait) == "Vertical Monitor"


# ── user-set custom names ───────────────────────────────────────────────────

def test_set_custom_name_requires_both_fields(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    result = manager.set_custom_name("", "Coding Monitor")
    assert not result.ok
    result = manager.set_custom_name("\\\\.\\DISPLAY1", "")
    assert not result.ok


def test_set_custom_name_persists_and_is_used(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    result = manager.set_custom_name("\\\\.\\DISPLAY1", "Coding Monitor")
    assert result.ok
    assert manager.contextual_name(_primary_monitor()) == "Coding Monitor"


def test_custom_name_overrides_the_vertical_geometry_fallback(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    manager.set_custom_name("\\\\.\\DISPLAY2", "Gaming Monitor")
    portrait = Monitor(index=1, x=1920, y=0, width=1080, height=1920,
                        is_primary=False, scale=1.0, name="\\\\.\\DISPLAY2")
    assert manager.contextual_name(portrait) == "Gaming Monitor"


def test_custom_names_survive_a_new_manager_instance(tmp_path, monkeypatch):
    first = _manager(tmp_path, monkeypatch)
    first.set_custom_name("\\\\.\\DISPLAY1", "Coding Monitor")
    second = _manager(tmp_path, monkeypatch)
    assert second.contextual_name(_primary_monitor()) == "Coding Monitor"


def test_custom_names_file_is_valid_json(tmp_path, monkeypatch):
    import orion_core.display as dm
    manager = _manager(tmp_path, monkeypatch)
    manager.set_custom_name("\\\\.\\DISPLAY1", "Coding Monitor")
    data = json.loads(dm._DISPLAY_NAMES_PATH.read_text(encoding="utf-8"))
    assert data == {"\\\\.\\DISPLAY1": "Coding Monitor"}


def test_a_monitor_with_no_custom_name_is_unaffected_by_others(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    manager.set_custom_name("\\\\.\\DISPLAY1", "Coding Monitor")
    assert manager.contextual_name(_secondary_monitor()) == "Secondary Display"


# ── describe()/summary() surface the contextual name ────────────────────────

def test_describe_includes_contextual_name_per_monitor(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    manager.set_custom_name(manager.topology().monitors[0].name, "Coding Monitor")
    described = manager.describe()
    assert described["monitors"][0]["contextual_name"] == "Coding Monitor"


def test_summary_includes_the_contextual_name(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    device = manager.topology().monitors[0].name
    manager.set_custom_name(device, "Coding Monitor")
    assert "Coding Monitor" in manager.summary()
