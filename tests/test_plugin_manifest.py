"""Tests for the plugin manifest system (Mark X.12 §2.6)."""

from __future__ import annotations

import json

import pytest

from orion_core.plugin_manifest import (
    CAPABILITY_TIERS,
    ManifestError,
    PluginManifest,
    discover_manifests,
    load_manifest,
    plan_load,
    unmet_requirements,
)


def _write_manifest(directory, name, **overrides):
    data = {
        "name": name,
        "description": f"{name} does a thing",
        "module": f"{name}_tool.py",
        "tier": "confirm",
        "requires": [],
        "parameters": {"type": "OBJECT", "properties": {}},
    }
    data.update(overrides)
    path = directory / f"{name}.plugin.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_module(directory, name):
    (directory / f"{name}_tool.py").write_text(
        "def get_tool_schema():\n    return {'name': '%s'}\n"
        "def run(**kw):\n    return 'ok'\n" % name,
        encoding="utf-8",
    )


# ── parsing + validation ───────────────────────────────────────────────────────

def test_valid_manifest_parses(tmp_path):
    path = _write_manifest(tmp_path, "spotify", requires=["os"], tier="allow")
    m = load_manifest(path)
    assert m.name == "spotify"
    assert m.tier == "allow" and m.tier in CAPABILITY_TIERS
    assert m.requires == ("os",)
    assert m.module_path == tmp_path / "spotify_tool.py"


def test_tier_defaults_to_confirm_when_absent(tmp_path):
    data = {"name": "x", "module": "x_tool.py"}
    m = PluginManifest.from_dict(data)
    assert m.tier == "confirm"


@pytest.mark.parametrize("bad,err", [
    ({"module": "x_tool.py"}, "name"),
    ({"name": "x"}, "module"),
    ({"name": "x", "module": "x.txt"}, "module"),
    ({"name": "x", "module": "x_tool.py", "tier": "root"}, "tier"),
    ({"name": "x", "module": "x_tool.py", "requires": "os"}, "requires"),
])
def test_invalid_manifests_raise(bad, err):
    with pytest.raises(ManifestError) as exc:
        PluginManifest.from_dict(bad)
    assert err in str(exc.value)


def test_invalid_json_is_a_manifest_error(tmp_path):
    path = tmp_path / "broken.plugin.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(path)


# ── dependency checking ────────────────────────────────────────────────────────

def test_unmet_requirements_detects_missing_and_ignores_present():
    m = PluginManifest.from_dict({
        "name": "x", "module": "x_tool.py",
        "requires": ["os", "json", "totally_not_a_real_module_xyz"],
    })
    assert unmet_requirements(m) == ["totally_not_a_real_module_xyz"]


# ── discovery + planning ───────────────────────────────────────────────────────

def test_discover_collects_valid_and_reports_broken(tmp_path):
    _write_manifest(tmp_path, "good")
    (tmp_path / "bad.plugin.json").write_text("{oops", encoding="utf-8")
    manifests, errors = discover_manifests(tmp_path)
    assert [m.name for m in manifests] == ["good"]
    assert "bad.plugin.json" in errors


def test_plan_load_separates_loadable_from_skipped(tmp_path):
    # loadable: manifest valid, module present, deps met
    _write_manifest(tmp_path, "ready", requires=["os"])
    _write_module(tmp_path, "ready")
    # skipped: module file missing
    _write_manifest(tmp_path, "nomodule")
    # skipped: dependency missing
    _write_manifest(tmp_path, "needsdep", requires=["totally_not_a_real_module_xyz"])
    _write_module(tmp_path, "needsdep")

    plan = plan_load(tmp_path)

    assert [m.name for m in plan.loadable] == ["ready"]
    assert "module file" in plan.skipped["nomodule"]
    assert "missing dependencies" in plan.skipped["needsdep"]


def test_plan_load_on_empty_or_missing_dir_is_harmless(tmp_path):
    plan = plan_load(tmp_path / "does-not-exist")
    assert plan.loadable == [] and plan.skipped == {}


# ── load_plugins() — the integration step (Mark X.14) ─────────────────────────

def _write_contract_passing_module(directory, name):
    """A module whose schema/run() actually pass ReflectiveModuleLoader's
    contract check — _write_module's bare schema does not (no description,
    no parameters), which is fine for the parsing tests above but not for
    exercising the real load-and-activate pipeline."""
    (directory / f"{name}_tool.py").write_text(
        "def get_tool_schema():\n"
        "    return {'name': '%s', 'description': 'does a thing',\n"
        "            'parameters': {'type': 'object', 'properties': {}, 'required': []}}\n\n"
        "def run(**kwargs):\n"
        "    return 'ok'\n" % name,
        encoding="utf-8",
    )


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)

    def connect(self, *_a, **_k):
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig

    def lines(self):
        return " ".join(str(part) for payload in self.log.emitted for part in payload)


def test_load_plugins_loads_and_activates_and_registers_its_tier(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from orion_core import remote_capability
    from orion_core.dynamic_loader import ReflectiveModuleLoader
    from orion_core.plugin_manifest import load_plugins
    from orion_core.remote_capability import Tier, classify

    # _PLUGIN_TIERS is module-global state — isolate it so this test can't
    # leak a registration into any other test in the session.
    monkeypatch.setattr(remote_capability, "_PLUGIN_TIERS", {})

    _write_manifest(tmp_path, "spotify_control", tier="allow")
    _write_contract_passing_module(tmp_path, "spotify_control")

    bus = _StubBus()
    loader = ReflectiveModuleLoader(bus)
    loader.custom_tools_dir = tmp_path
    dispatcher = SimpleNamespace()  # a plain object() rejects arbitrary attrs

    plan = asyncio.run(load_plugins(loader, dispatcher, tmp_path, bus))

    assert [m.name for m in plan.loadable] == ["spotify_control"]
    assert "spotify_control_tool" in loader.loaded_modules  # keyed by module stem
    assert dispatcher._tool_handlers["spotify_control"] is not None  # keyed by schema name
    assert "loaded and live" in bus.lines()
    # The manifest declared tier=allow — the remote gate must respect it,
    # not fall back to the CONFIRM fail-safe for an unclassified tool.
    assert classify("spotify_control") is Tier.ALLOW


def test_load_plugins_logs_skipped_plugins_without_crashing(tmp_path):
    import asyncio
    from orion_core.dynamic_loader import ReflectiveModuleLoader
    from orion_core.plugin_manifest import load_plugins

    _write_manifest(tmp_path, "needsdep", requires=["totally_not_a_real_module_xyz"])
    _write_contract_passing_module(tmp_path, "needsdep")

    bus = _StubBus()
    loader = ReflectiveModuleLoader(bus)
    loader.custom_tools_dir = tmp_path

    plan = asyncio.run(load_plugins(loader, object(), tmp_path, bus))

    assert plan.loadable == []
    assert "needsdep" not in loader.loaded_modules
    assert "skipped" in bus.lines() and "needsdep" in bus.lines()


def test_load_plugins_does_not_activate_a_contract_breaking_module(tmp_path, monkeypatch):
    import asyncio
    from orion_core import remote_capability
    from orion_core.dynamic_loader import ReflectiveModuleLoader
    from orion_core.plugin_manifest import load_plugins
    from orion_core.remote_capability import Tier, classify

    monkeypatch.setattr(remote_capability, "_PLUGIN_TIERS", {})
    _write_manifest(tmp_path, "broken", tier="allow")
    (tmp_path / "broken_tool.py").write_text(
        "def get_tool_schema():\n"
        "    return {'name': 'broken', 'description': 'x',\n"
        "            'parameters': {'type': 'object',\n"
        "                           'properties': {'url': {'type': 'string'}},\n"
        "                           'required': ['url']}}\n\n"
        "def run(path):\n"
        "    return path\n",
        encoding="utf-8",
    )

    bus = _StubBus()
    loader = ReflectiveModuleLoader(bus)
    loader.custom_tools_dir = tmp_path

    plan = asyncio.run(load_plugins(loader, object(), tmp_path, bus))

    assert [m.name for m in plan.loadable] == ["broken"]  # planning saw it as loadable
    assert "broken_tool" not in loader.loaded_modules      # but it broke the contract
    # A tool that never activated must never be granted a remote tier.
    assert classify("broken") is Tier.CONFIRM
