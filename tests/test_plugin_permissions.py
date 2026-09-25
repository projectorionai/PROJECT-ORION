"""
Plugin capability disclosure + versioning (Mark XXVI improvements #2/#3).

A dropped-in or forged plugin is untrusted code. It can now DECLARE which coarse
capabilities it needs (network/filesystem/subprocess/input/screen/audio), and the
registry statically compares that declaration against what the module actually
does — so an undeclared capability is visible BEFORE the plugin is ever imported.
Versioning lets an installed plugin be compared against a candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import plugin_registry as pr  # noqa: E402
from orion_core.plugin_manifest import (  # noqa: E402
    PERMISSIONS,
    ManifestError,
    PluginManifest,
)

_CONTRACT = (
    "def get_tool_schema():\n"
    "    return {'name': 'x', 'description': 'd', 'parameters': {}}\n"
    "def run(**kwargs):\n"
    "    return 'ok'\n"
)


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    return pr.PluginRegistry(bus=None, directory=tmp_path / "custom_tools")


def _write(registry, filename, body):
    registry.directory.mkdir(parents=True, exist_ok=True)
    path = registry.directory / filename
    path.write_text(body, encoding="utf-8")
    return path


# ── manifest fields (backward compatible) ────────────────────────────────────

def test_version_and_permissions_default_so_old_manifests_still_load():
    m = PluginManifest.from_dict({"name": "x", "module": "x_tool.py"})
    assert m.version == "0.0.0"
    assert m.permissions == ()


def test_declared_permissions_parse():
    m = PluginManifest.from_dict({
        "name": "x", "module": "x_tool.py",
        "version": "2.1.0", "permissions": ["network", "FileSystem"]})
    assert m.version == "2.1.0"
    assert set(m.permissions) == {"network", "filesystem"}


def test_an_unknown_permission_is_rejected():
    with pytest.raises(ManifestError):
        PluginManifest.from_dict({"name": "x", "module": "x_tool.py",
                                  "permissions": ["mind_control"]})


def test_permissions_must_be_a_list():
    with pytest.raises(ManifestError):
        PluginManifest.from_dict({"name": "x", "module": "x_tool.py",
                                  "permissions": "network"})


def test_the_vocabulary_is_the_documented_set():
    assert PERMISSIONS == frozenset({"network", "filesystem", "subprocess",
                                     "input", "screen", "audio"})


# ── static capability detection (never imports the module) ───────────────────

def test_network_use_is_detected(registry):
    mod = _write(registry, "net_tool.py", "import requests\n" + _CONTRACT)
    assert "network" in pr.detect_capabilities(mod)


def test_subprocess_use_is_detected(registry):
    mod = _write(registry, "proc_tool.py", "import subprocess\n" + _CONTRACT)
    assert "subprocess" in pr.detect_capabilities(mod)


def test_input_and_screen_use_are_detected(registry):
    mod = _write(registry, "ui_tool.py", "import pyautogui\nimport mss\n" + _CONTRACT)
    caps = pr.detect_capabilities(mod)
    assert "input" in caps and "screen" in caps


def test_a_pure_plugin_needs_nothing(registry):
    mod = _write(registry, "pure_tool.py",
                 "def get_tool_schema():\n    return {}\n"
                 "def run(**k):\n    return str(2 + 2)\n")
    assert pr.detect_capabilities(mod) == set()


def test_detection_never_imports_the_module(registry, tmp_path):
    marker = tmp_path / "RAN"
    body = ("from pathlib import Path\n"
            "Path(" + repr(str(marker)) + ").write_text('x')\n"
            "import requests\n") + _CONTRACT
    mod = _write(registry, "sneaky_tool.py", body)
    pr.detect_capabilities(mod)
    assert not marker.exists()


def test_a_syntactically_broken_module_reports_nothing_rather_than_raising(registry):
    mod = _write(registry, "bad_tool.py", "def run(:\n")
    assert pr.detect_capabilities(mod) == set()


# ── the audit: declared vs actual ────────────────────────────────────────────

def test_audit_flags_an_undeclared_capability(registry):
    mod = _write(registry, "leaky_tool.py", "import requests\n" + _CONTRACT)
    report = pr.audit_permissions(mod, ["filesystem"])
    assert "network" in report["undeclared"]
    assert "filesystem" in report["unused"]


def test_audit_is_clean_when_the_declaration_matches(registry):
    mod = _write(registry, "honest_tool.py", "import requests\n" + _CONTRACT)
    assert pr.audit_permissions(mod, ["network"])["undeclared"] == []


def test_the_record_carries_the_disclosure(registry):
    registry.create("demo", "A demo.", "allow")
    # rewrite the scaffold so it genuinely uses the network, undeclared
    _write(registry, "demo_tool.py", "import requests\n" + _CONTRACT)
    record = registry.get("demo")
    assert "network" in record.undeclared
    # a disclosure gap is NOT a health fault — it must not flip `healthy`
    assert record.healthy is True
    assert "network" in record.to_dict()["undeclared"]


def test_the_audit_report_names_offenders(registry):
    _write(registry, "leaky_tool.py", "import subprocess\n" + _CONTRACT)
    registry.backfill_manifests()
    text = registry.audit().text
    assert "leaky" in text and "undeclared" in text.lower()


def test_auditing_an_unknown_plugin_is_graceful(registry):
    assert registry.audit("nope").ok is False


# ── versioning ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("left,right,expected", [
    ("1.0.0", "1.0.0", 0),
    ("1.2.0", "1.10.0", -1),
    ("2.0", "1.9.9", 1),
    ("v2", "2.0", 0),
    ("", "0.0.1", -1),
    ("junk", "junk", 0),
])
def test_version_comparison(left, right, expected):
    assert pr.compare_versions(left, right) == expected


def test_check_update_detects_a_newer_candidate(registry):
    registry.create("demo", "d", "allow")
    result = registry.check_update("demo", "9.9.9")
    assert result.ok and "newer" in result.text.lower()


def test_check_update_detects_no_change(registry):
    registry.create("demo", "d", "allow")
    record = registry.get("demo")
    assert "already" in registry.check_update("demo", record.version).text.lower()


def test_check_update_of_an_unknown_plugin_is_graceful(registry):
    assert registry.check_update("nope", "1.0.0").ok is False


# ── tool wiring ──────────────────────────────────────────────────────────────

def test_the_plugin_tool_advertises_audit_and_update():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "plugin")
    assert "audit" in tool["description"]
    assert "update" in tool["description"]
    assert "version" in tool["parameters"]["properties"]


def test_the_dispatcher_routes_the_new_actions():
    src = (Path(__file__).resolve().parents[1] / "orion_core"
           / "dispatch_productivity.py").read_text(encoding="utf-8")
    assert "self.plugins.audit(" in src
    assert "self.plugins.check_update(" in src


# ── tier vs capability coherence (improvement #12) ──────────────────────────

def _manifest(registry, name, tier, body_import=""):
    import json
    _write(registry, f"{name}_tool.py", f"{body_import}\n" + _CONTRACT)
    (registry.directory / f"{name}.plugin.json").write_text(
        json.dumps({"name": name, "module": f"{name}_tool.py",
                    "tier": tier, "description": "x"}), encoding="utf-8")


def test_an_unattended_plugin_that_spawns_processes_is_flagged(registry):
    _manifest(registry, "runner", "allow", "import subprocess")
    risky = registry.risky_grants()
    assert risky and risky[0][0] == "runner"
    assert "subprocess" in risky[0][2]


def test_a_confirm_tier_plugin_is_not_flagged(registry):
    _manifest(registry, "runner", "confirm", "import subprocess")
    assert registry.risky_grants() == []


def test_an_unattended_but_harmless_plugin_is_not_flagged(registry):
    _manifest(registry, "pure", "allow", "")
    assert registry.risky_grants() == []


def test_declared_capabilities_count_too_not_just_undeclared(registry):
    import json
    _write(registry, "net_tool.py", "import requests\n" + _CONTRACT)
    (registry.directory / "net.plugin.json").write_text(
        json.dumps({"name": "net", "module": "net_tool.py", "tier": "allow",
                    "permissions": ["network"], "description": "x"}),
        encoding="utf-8")
    risky = registry.risky_grants()
    assert risky and "network" in risky[0][2], (
        "declaring a risky capability does not make an unattended tier safe")


def test_doctor_surfaces_the_coherence_warning(registry):
    _manifest(registry, "runner", "allow", "import subprocess")
    text = registry.doctor().text
    assert "UNATTENDED" in text and "runner" in text


# ── load-time disclosure (improvement #16) ──────────────────────────────────

class _LogBus:
    def __init__(self):
        self.messages = []

        class _Sig:
            def emit(_s, msg):
                self.messages.append(str(msg))
        self.log = _Sig()


def test_loading_a_plugin_logs_its_undeclared_capabilities(registry):
    from orion_core.plugin_manifest import PluginManifest, _disclose
    module = _write(registry, "runner_tool.py", "import subprocess\n" + _CONTRACT)
    manifest = PluginManifest.from_dict(
        {"name": "runner", "module": "runner_tool.py", "tier": "confirm",
         "description": "x"},
        source_path=registry.directory / "runner.plugin.json")
    bus = _LogBus()
    _disclose(manifest, module, bus)
    assert any("UNDECLARED" in m and "subprocess" in m for m in bus.messages)


def test_an_unattended_risky_plugin_is_called_out_at_load(registry):
    from orion_core.plugin_manifest import PluginManifest, _disclose
    module = _write(registry, "runner_tool.py", "import subprocess\n" + _CONTRACT)
    manifest = PluginManifest.from_dict(
        {"name": "runner", "module": "runner_tool.py", "tier": "allow",
         "permissions": ["subprocess"], "description": "x"},
        source_path=registry.directory / "runner.plugin.json")
    bus = _LogBus()
    _disclose(manifest, module, bus)
    assert any("UNATTENDED" in m for m in bus.messages)


def test_a_clean_plugin_logs_nothing_at_load(registry):
    from orion_core.plugin_manifest import PluginManifest, _disclose
    module = _write(registry, "pure_tool.py", _CONTRACT)
    manifest = PluginManifest.from_dict(
        {"name": "pure", "module": "pure_tool.py", "tier": "allow",
         "description": "x"},
        source_path=registry.directory / "pure.plugin.json")
    bus = _LogBus()
    _disclose(manifest, module, bus)
    assert bus.messages == [], "a harmless plugin must not add log noise"


def test_the_load_path_calls_the_disclosure():
    src = (Path(__file__).resolve().parents[1] / "orion_core"
           / "plugin_manifest.py").read_text(encoding="utf-8")
    assert "_disclose(manifest, module_path, bus)" in src


# ── ordinary English is not a filesystem write ───────────────────────────────
#
# Found while auditing a new plugin: a weather tool that opens no files and
# imports no filesystem module was reported as using `filesystem`. The cause
# was `place.replace(" ", "")` in a postcode helper — `replace` was in the
# filesystem call list because `Path.replace` renames a file.
#
# Over-reporting is the safe direction for a disclosure check, which is why it
# went unnoticed. It stops being safe when the false positives are `replace`,
# `remove` and `run` on strings and lists: a check that cries wolf on normal
# code is one people learn to wave through, and this one gates what a plugin
# is allowed to touch.

def test_a_string_replace_is_not_a_file_rename(registry):
    mod = _write(registry, "text_tool.py",
                 'def get_tool_schema():\n    return {}\n'
                 'def run(**k):\n    return "a b".replace(" ", "")\n')
    assert "filesystem" not in pr.detect_capabilities(mod)


def test_removing_from_a_list_is_not_deleting_a_file(registry):
    mod = _write(registry, "list_tool.py",
                 'def get_tool_schema():\n    return {}\n'
                 'def run(**k):\n    xs = [1]\n    xs.remove(1)\n    return str(xs)\n')
    assert "filesystem" not in pr.detect_capabilities(mod)


def test_calling_run_on_an_object_is_not_a_subprocess(registry):
    mod = _write(registry, "job_tool.py",
                 'def get_tool_schema():\n    return {}\n'
                 'def run(**k):\n    return k["job"].run()\n')
    assert "subprocess" not in pr.detect_capabilities(mod)


# The other half: narrowing this must not let real use through.

def test_the_builtin_open_still_counts(registry):
    mod = _write(registry, "read_tool.py",
                 'def get_tool_schema():\n    return {}\n'
                 'def run(**k):\n    return open("x").read()\n')
    assert "filesystem" in pr.detect_capabilities(mod)


def test_os_and_shutil_calls_still_count(registry):
    for name, body in (("rm_tool.py", 'import os\nos.remove("x")\n'),
                       ("tree_tool.py", 'import shutil\nshutil.rmtree("x")\n')):
        mod = _write(registry, name,
                     body + 'def get_tool_schema():\n    return {}\n'
                            'def run(**k):\n    return ""\n')
        assert "filesystem" in pr.detect_capabilities(mod), name


def test_a_call_on_something_named_like_a_path_still_counts(registry):
    """`dest.rename(...)` and `log_path.replace(...)` are the real thing even
    though the method names are ordinary words."""
    mod = _write(registry, "move_tool.py",
                 'def get_tool_schema():\n    return {}\n'
                 'def run(**k):\n    dest.rename("y")\n    log_path.replace("z")\n')
    assert "filesystem" in pr.detect_capabilities(mod)


def test_a_path_constructor_receiver_still_counts(registry):
    mod = _write(registry, "ctor_tool.py",
                 'def get_tool_schema():\n    return {}\n'
                 'def run(**k):\n    Path("a").replace("b")\n')
    assert "filesystem" in pr.detect_capabilities(mod)


def test_unambiguous_path_writes_never_needed_corroboration(registry):
    """Nothing but a path object has write_bytes, so it counts wherever it
    appears — including on a receiver this detector has never heard of."""
    mod = _write(registry, "write_tool.py",
                 'def get_tool_schema():\n    return {}\n'
                 'def run(**k):\n    whatever.write_bytes(b"x")\n')
    assert "filesystem" in pr.detect_capabilities(mod)
