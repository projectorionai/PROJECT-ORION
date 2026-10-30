"""
Plugin registry (Mark XXVI) — the ecosystem layer around the plugin runtime.

741 lines of lifecycle (scan/enable/disable/create/install/remove/doctor) shipped
with no tests at all. This is the safety net: the registry must never raise to the
caller, must never import a candidate module to inspect it, and disabling must
survive a restart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import plugin_registry as pr  # noqa: E402

_CONTRACT = (
    "def get_tool_schema():\n"
    "    return {'name': 'x', 'description': 'd', 'parameters': {}}\n"
    "def run(**kwargs):\n"
    "    return 'ok'\n"
)


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """A registry rooted in a temp dir with temp state — never the real config."""
    monkeypatch.setattr(pr, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    return pr.PluginRegistry(bus=None, directory=tmp_path / "custom_tools")


def _write(registry, filename, body):
    registry.directory.mkdir(parents=True, exist_ok=True)
    path = registry.directory / filename
    path.write_text(body, encoding="utf-8")
    return path


def _plugin(reg, name="demo", description="A demo plugin.", tier="allow"):
    return reg.create(name, description, tier)


# ── scaffolding ───────────────────────────────────────────────────────────────

def test_create_scaffolds_a_working_plugin(registry):
    result = _plugin(registry)
    assert result.ok
    module = registry.directory / "demo_tool.py"
    manifest = registry.directory / "demo.plugin.json"
    assert module.is_file() and manifest.is_file()
    # The scaffold must satisfy the contract it will be loaded against.
    assert pr.validate_module(module) == []


def test_a_scaffold_is_visible_to_the_scan(registry):
    _plugin(registry)
    assert "demo" in {r.name for r in registry.scan()}


def test_create_refuses_a_duplicate(registry):
    _plugin(registry)
    again = _plugin(registry)
    assert not again.ok and "already exists" in again.text


def test_create_refuses_an_empty_name(registry):
    assert not registry.create("").ok


# ── enable / disable persists ────────────────────────────────────────────────

def test_disable_persists_across_a_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    directory = tmp_path / "custom_tools"
    first = pr.PluginRegistry(bus=None, directory=directory)
    first.create("demo", "d", "allow")
    assert first.set_enabled("demo", False).ok
    assert first.is_enabled("demo") is False

    # A fresh registry (i.e. a restart) must still see it disabled.
    second = pr.PluginRegistry(bus=None, directory=directory)
    assert second.is_enabled("demo") is False
    record = second.get("demo")
    assert record is not None and record.status() == "disabled"


def test_re_enabling_works(registry):
    _plugin(registry)
    registry.set_enabled("demo", False)
    assert registry.set_enabled("demo", True).ok
    assert registry.is_enabled("demo") is True


# ── static validation never executes the candidate ───────────────────────────

def test_validation_flags_a_module_missing_the_contract(registry):
    bad = _write(registry, "broken_tool.py", "x = 1\n")
    assert pr.validate_module(bad), "a module with no get_tool_schema/run must be flagged"


def test_inspecting_a_hostile_module_does_not_execute_it(registry, tmp_path):
    """Static (ast) introspection is the rule — merely listing an untrusted
    plugin must not run its code."""
    marker = tmp_path / "EXECUTED"
    body = (
        "from pathlib import Path\n"
        "Path(" + repr(str(marker)) + ").write_text('boom')\n"
    ) + _CONTRACT
    hostile = _write(registry, "hostile_tool.py", body)
    pr.validate_module(hostile)
    pr.extract_description(hostile)
    pr.extract_parameters(hostile)
    registry.scan()
    assert not marker.exists(), "inspection executed the plugin's code"


def test_extract_description_reads_the_docstring(registry):
    mod = _write(registry, "doc_tool.py", '"""Turns the lights on."""\n' + _CONTRACT)
    assert "lights" in pr.extract_description(mod).lower()


# ── health / records ─────────────────────────────────────────────────────────

def test_a_missing_module_reads_as_missing(registry):
    _plugin(registry)
    (registry.directory / "demo_tool.py").unlink()
    record = registry.get("demo")
    assert record is not None and record.status() == "missing"
    assert record.healthy is False


def test_call_and_error_health_is_tracked(registry):
    _plugin(registry)
    registry.record_call("demo")
    registry.record_call("demo")
    registry.mark_loaded("demo", ok=True)
    record = registry.get("demo")
    assert record.calls == 2 and record.loaded is True

    registry.mark_loaded("demo", ok=False, error="kaboom")
    assert registry.get("demo").status() == "error"


def test_record_to_dict_is_serialisable(registry):
    _plugin(registry)
    d = registry.get("demo").to_dict()
    assert d["name"] == "demo" and "status" in d and isinstance(d["requires"], list)


# ── orphans + backfill (the forged tools) ────────────────────────────────────

def test_an_orphan_module_without_a_manifest_is_still_seen(registry):
    _write(registry, "forged_tool.py", '"""Forged."""\n' + _CONTRACT)
    record = registry.get("forged")
    assert record is not None
    assert record.has_manifest is False
    assert "forged" in registry.manifestless()


def test_backfill_gives_an_orphan_a_manifest_and_a_tier(registry):
    _write(registry, "forged_tool.py", '"""Forged."""\n' + _CONTRACT)
    written = registry.backfill_manifests()
    assert "forged" in written
    assert (registry.directory / "forged.plugin.json").is_file()
    record = registry.get("forged")
    assert record.has_manifest is True and record.tier


# ── install / remove ─────────────────────────────────────────────────────────

def test_install_copies_a_plugin_in(registry, tmp_path):
    src = tmp_path / "external_tool.py"
    src.write_text('"""External."""\n' + _CONTRACT, encoding="utf-8")
    result = registry.install(str(src))
    assert result.ok, result.text
    assert (registry.directory / "external_tool.py").is_file()
    assert registry.get("external") is not None


def test_install_rejects_a_missing_path(registry):
    assert not registry.install(str(registry.directory / "nope_tool.py")).ok


def test_install_rejects_an_empty_source(registry):
    assert not registry.install("").ok


def test_enable_state_write_failure_does_not_claim_success(registry, monkeypatch):
    _plugin(registry)
    assert registry.is_enabled("demo")
    monkeypatch.setattr(pr.os, "replace", lambda *args: (_ for _ in ()).throw(PermissionError("locked")))
    result = registry.set_enabled("demo", False)
    assert not result.ok
    assert registry.is_enabled("demo")
    assert not list(pr.PLUGIN_STATE_PATH.parent.glob("*.tmp"))


def test_remove_deletes_the_plugin(registry):
    _plugin(registry)
    assert registry.remove("demo").ok
    assert registry.get("demo") is None


def test_remove_of_an_unknown_plugin_is_graceful(registry):
    result = registry.remove("no_such_plugin")
    assert result.ok is False and result.text


# ── scaffold kinds: a REACTIVE plugin (Mark XXVI improvement #8) ─────────────

def test_creating_a_reactive_plugin_generates_the_hook_contract(registry):
    result = registry.create("lamp", "Flashes a lamp.", "allow", kind="event")
    assert result.ok
    module = (registry.directory / "lamp_tool.py").read_text(encoding="utf-8")
    assert "def on_event" in module, "a reactive plugin must get an on_event handler"
    assert "def run" in module and "def get_tool_schema" in module
    assert pr.validate_module(registry.directory / "lamp_tool.py") == []


def test_a_reactive_plugin_declares_default_events(registry):
    import json
    registry.create("lamp", "Flashes a lamp.", "allow", kind="event")
    manifest = json.loads(
        (registry.directory / "lamp.plugin.json").read_text(encoding="utf-8"))
    assert manifest["events"], "a reactive plugin with no events would never fire"


def test_explicit_events_win(registry):
    import json
    registry.create("watch", "Watches.", "allow", kind="event",
                    events=["speaking", "banner"])
    manifest = json.loads(
        (registry.directory / "watch.plugin.json").read_text(encoding="utf-8"))
    assert set(manifest["events"]) == {"speaking", "banner"}


def test_a_plain_tool_plugin_has_no_hooks(registry):
    registry.create("plain", "Just a tool.", "allow")
    module = (registry.directory / "plain_tool.py").read_text(encoding="utf-8")
    assert "def on_event" not in module


# ── doctor: the one-stop view (improvement #9) ──────────────────────────────

def test_doctor_reports_undeclared_capabilities(registry):
    _write(registry, "leaky_tool.py", '"""Leaky."""\nimport subprocess\n' + _CONTRACT)
    registry.backfill_manifests()
    text = registry.doctor().text
    assert "never declared" in text and "leaky" in text


def test_doctor_reports_event_hooks_when_a_bridge_is_supplied(registry):
    from orion_core.plugin_events import PluginEventBridge

    class _P:
        def on_event(self, event, payload):
            pass

    bridge = PluginEventBridge(log=lambda _m: None)
    bridge._handlers["lamp"] = _P().on_event
    bridge._bound["lamp"] = ["state"]
    text = registry.doctor(events=bridge).text
    assert "hooked to ORION events" in text and "lamp" in text


def test_doctor_without_a_bridge_still_works(registry):
    registry.create("demo", "d", "allow")
    assert registry.doctor().ok


def test_doctor_flags_a_muted_plugin(registry):
    from orion_core.plugin_events import HookStats, PluginEventBridge
    bridge = PluginEventBridge(log=lambda _m: None)
    bridge._bound["bad"] = ["state"]
    bridge.stats["bad"] = HookStats(muted=True, events=("state",))
    assert "MUTED" in registry.doctor(events=bridge).text


# ── portable bundles (Mark XXVI improvement #10) ────────────────────────────

def test_export_then_install_round_trips_a_plugin(registry, tmp_path, monkeypatch):
    registry.create("lamp", "Flashes a lamp.", "allow", kind="event")
    result = registry.export_bundle("lamp")
    assert result.ok, result.text
    bundle = registry.directory / "bundles" / "lamp.orionplugin"
    assert bundle.is_file()

    monkeypatch.setattr(pr, "PLUGIN_STATE_PATH", tmp_path / "other.json")
    fresh = pr.PluginRegistry(bus=None, directory=tmp_path / "elsewhere")
    assert fresh.install(str(bundle)).ok
    record = fresh.get("lamp")
    assert record is not None
    assert record.version == "1.0.0" and record.tier == "allow"
    assert record.has_manifest is True


def test_export_to_an_explicit_destination(registry, tmp_path):
    registry.create("demo", "d", "allow")
    target = tmp_path / "share" / "demo.orionplugin"
    assert registry.export_bundle("demo", str(target)).ok
    assert target.is_file()


def test_exporting_an_unknown_plugin_is_graceful(registry):
    assert registry.export_bundle("nope").ok is False


def test_a_bundle_carries_a_manifest_even_for_an_orphan(registry):
    import zipfile
    _write(registry, "forged_tool.py", '"""Forged."""\n' + _CONTRACT)
    assert registry.export_bundle("forged").ok
    bundle = registry.directory / "bundles" / "forged.orionplugin"
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert any(n.endswith(".plugin.json") for n in names), (
        "an exported bundle must not lose the capability tier")


def test_a_bundle_cannot_path_traverse_out_of_the_plugin_directory(registry, tmp_path):
    """A malicious bundle must not be able to write outside custom_tools."""
    import zipfile
    evil = tmp_path / "evil.orionplugin"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../../escaped_tool.py", _CONTRACT)
        zf.writestr("nested/deep_tool.py", _CONTRACT)
        zf.writestr("good_tool.py", _CONTRACT)
    result = registry.install(str(evil))
    assert result.ok
    assert (registry.directory / "good_tool.py").is_file()
    assert not (tmp_path / "escaped_tool.py").exists()
    assert not (registry.directory / "nested").exists()
    assert "ignored" in result.text


def test_a_bundle_with_no_module_is_rejected(registry, tmp_path):
    import zipfile
    empty = tmp_path / "empty.orionplugin"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    assert registry.install(str(empty)).ok is False


def test_a_corrupt_bundle_is_rejected_not_fatal(registry, tmp_path):
    broken = tmp_path / "broken.orionplugin"
    broken.write_text("this is not a zip", encoding="utf-8")
    result = registry.install(str(broken))
    assert result.ok is False and "bundle" in result.text.lower()


def test_the_tool_exposes_export():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "plugin")
    assert "export" in tool["description"] and "orionplugin" in tool["description"]
    src = (Path(__file__).resolve().parents[1] / "orion_core"
           / "dispatch_productivity.py").read_text(encoding="utf-8")
    assert "self.plugins.export_bundle" in src


# ── describe + call accounting (improvement #13) ────────────────────────────

def test_describe_shows_version_and_capability_disclosure(registry):
    registry.create("demo", "A demo.", "allow")
    _write(registry, "demo_tool.py", "import subprocess\n" + _CONTRACT)
    text = registry.describe("demo").text
    assert "version" in text and "1.0.0" in text
    assert "can touch" in text
    assert "UNDECLARED" in text and "subprocess" in text


def test_describe_of_an_unknown_plugin_is_graceful(registry):
    assert registry.describe("nope").ok is False


@pytest.mark.parametrize("raises", [False, True], ids=["failure_result", "handler_exception"])
def test_plugin_calls_and_failures_are_recorded_by_the_dispatcher(registry, monkeypatch, raises):
    """The dispatcher must count plugin invocations and record failures against
    the plugin's health — otherwise 'calls' in doctor/describe is always zero."""
    import asyncio
    from types import SimpleNamespace
    from orion_core import dispatcher
    from orion_core.data import ToolResult

    assert registry.create("demo", "Temporary call-accounting fixture.", "allow").ok
    before = registry.get("demo")
    assert before is not None
    calls = []

    def handler():
        calls.append("demo")
        if raises:
            raise RuntimeError("synthetic plugin failure")
        return ToolResult("synthetic plugin failure", ok=False)

    # The synthetic activated handler represents the temporary registry source;
    # no real extension is imported or executed by this witness.
    handler._orion_plugin_source_sha256 = before.source_sha256
    host = SimpleNamespace(handler_table=lambda: {}, _tool_handlers={"demo": handler},
                           plugins=registry, TOOL_DECLARATIONS=[])
    monkeypatch.setattr(dispatcher, "_note_gap", lambda *_: None)
    result = asyncio.run(dispatcher.OrionDispatcher.dispatch(host, "demo", {}))

    after = registry.get("demo")
    assert after is not None
    assert not result.ok and calls == ["demo"]
    assert after.calls == before.calls + 1
    assert after.loaded is True
    assert after.last_error == "synthetic plugin failure"
    assert result.execution["state"] == "returned"
    assert result.execution["executor_ok"] is False


# ── the contract the LOADER enforces (regression) ───────────────────────────

def test_a_gemini_style_uppercase_schema_is_caught_statically(registry):
    """The loader requires parameters.type == 'object' (lowercase). A plugin
    declaring 'OBJECT' passed every static check, was reported healthy, and then
    silently refused to activate. The registry must not promise that plugin."""
    bad = _write(registry, "upper_tool.py",
                 "def get_tool_schema():\n"
                 "    return {'name': 'x', 'parameters': {'type': 'OBJECT',\n"
                 "            'properties': {}}}\n"
                 "def run(**k):\n    return 'ok'\n")
    problems = pr.validate_module(bad)
    assert problems and any("object" in p for p in problems)


def test_a_lowercase_schema_is_accepted(registry):
    good = _write(registry, "lower_tool.py",
                  "def get_tool_schema():\n"
                  "    return {'name': 'x', 'description': 'A useful tool.',\n"
                  "            'parameters': {'type': 'object',\n"
                  "            'properties': {}}}\n"
                  "def run(**k):\n    return 'ok'\n")
    assert pr.validate_module(good) == []


@pytest.mark.parametrize("schema, error", [
    ({"name": "demo", "parameters": {}}, "description"),
    ({"name": "demo", "description": "   ", "parameters": {}}, "description"),
    ({"name": "demo", "description": "Describe a file."}, "parameters"),
    ({"name": "demo", "description": "Describe a file.",
      "parameters": {"required": ["path"]}}, "not in properties"),
    ({"name": "demo", "description": "Describe a file.",
      "parameters": {"properties": []}}, "properties"),
    ({"name": "demo", "description": "Describe a file.",
      "parameters": {"required": [{}]}}, "entries must be strings"),
])
def test_literal_plugin_schema_problems_are_visible_before_loading(registry, schema, error):
    path = _write(registry, "demo_tool.py",
                  "def get_tool_schema():\n    return " + repr(schema) + "\n"
                  "def run(**kwargs):\n    return 'ok'\n")
    problems = pr.validate_module(path)
    assert any(error in problem for problem in problems), problems
    record = registry.get("demo")
    assert record is not None and not record.healthy
    assert error in registry.describe("demo").text


@pytest.mark.parametrize("run, properties, error", [
    ("def run(path):\n    return path\n", {}, "requires argument 'path'"),
    ("def run(path):\n    return path\n", {"path": {"type": "string"}},
     "does not mark it as required"),
    ("def run(*, path):\n    return path\n", {"path": {"type": "string"}},
     "does not mark it as required"),
    ("def run():\n    return 'ok'\n", {"path": {"type": "string"}},
     "does not accept it"),
    ("def run(path, /):\n    return path\n", {"path": {"type": "string"}},
     "positional-only"),
    ("async def run(*, path):\n    return path\n", {}, "requires argument 'path'"),
])
def test_literal_plugin_arguments_match_the_handler_before_loading(
        registry, run, properties, error):
    schema = {"name": "demo", "description": "Describe a file.",
              "parameters": {"type": "object", "properties": properties}}
    path = _write(registry, "demo_tool.py",
                  "def get_tool_schema():\n    return " + repr(schema) + "\n" + run)
    assert any(error in problem for problem in pr.validate_module(path))


@pytest.mark.parametrize("required, handler", [
    (["path"], "def run(path):\n    return path\n"),
    ([], "def run(path='local'):\n    return path\n"),
])
def test_literal_plugin_schema_requiredness_accepts_callable_controls(
        registry, required, handler):
    schema = {"name": "demo", "description": "Describe a file.",
              "parameters": {"type": "object", "properties": {
                  "path": {"type": "string"}}, "required": required}}
    path = _write(registry, "demo_tool.py",
                  "def get_tool_schema():\n    return " + repr(schema) + "\n" + handler)
    assert pr.validate_module(path) == []


def test_literal_plugin_contract_inspection_does_not_evaluate_defaults(registry, tmp_path):
    marker = tmp_path / "DEFAULT_WAS_EXECUTED"
    path = _write(registry, "defaults_tool.py",
                  "from pathlib import Path\n" + _CONTRACT.replace(
                      "def run(**kwargs):",
                      "def run(query=Path(" + repr(str(marker)) + ").write_text('bad')):"))
    assert pr.validate_module(path) == []
    assert not marker.exists()


def test_non_schema_object_literals_do_not_make_a_plugin_unhealthy(registry):
    path = _write(registry, "data_tool.py", "DATA = {'type': 'OBJECT'}\n" + _CONTRACT)
    assert pr.validate_module(path) == []


def test_computed_plugin_schema_is_left_to_the_runtime_contract(registry):
    path = _write(registry, "computed_tool.py",
                  "def get_tool_schema():\n    return build_schema()\n"
                  "def run(**kwargs):\n    return 'ok'\n")
    assert pr.validate_module(path) == []


@pytest.mark.parametrize("replacement", [
    "if enabled:\n    get_tool_schema = build_schema\n",
    "from unavailable_schema_provider import get_tool_schema\n",
    "import unavailable_schema_provider as get_tool_schema\n",
    "if enabled:\n    def get_tool_schema():\n        return build_schema()\n",
    "class get_tool_schema:\n    pass\n",
    "if enabled:\n    from unavailable_handler_provider import run\n",
])
def test_rebound_plugin_exports_are_left_to_the_runtime_contract(registry, replacement):
    invalid_stub = _CONTRACT.replace("'description': 'd', ", "")
    path = _write(registry, "rebound_tool.py", invalid_stub + replacement)
    assert pr.validate_module(path) == []


@pytest.mark.parametrize("local_scope", [
    "def helper():\n    get_tool_schema = build_schema\n",
    "class Helper:\n    def run(self):\n        return 'local method'\n",
])
def test_local_names_do_not_bypass_literal_plugin_validation(registry, local_scope):
    invalid_stub = _CONTRACT.replace("'description': 'd', ", "")
    path = _write(registry, "local_tool.py", invalid_stub + local_scope)
    assert any("description" in problem for problem in pr.validate_module(path))


def test_a_literal_contract_failure_cannot_replace_an_installed_plugin(registry, tmp_path):
    assert registry.create("demo", "Describe local files.", "confirm").ok
    original = {path.name: path.read_bytes() for path in registry.directory.iterdir()}
    state = pr.PLUGIN_STATE_PATH.read_bytes()
    external = tmp_path / "external"
    external.mkdir()
    candidate = external / "demo_tool.py"
    candidate.write_text(
        "def get_tool_schema():\n    return {'name': 'demo', 'parameters': {}}\n"
        "def run(**kwargs):\n    return 'replacement'\n", encoding="utf-8")
    result = registry.install(str(candidate))
    assert not result.ok and "description" in result.text
    assert {path.name: path.read_bytes() for path in registry.directory.iterdir()} == original
    assert pr.PLUGIN_STATE_PATH.read_bytes() == state


def test_a_scaffolded_plugin_passes_the_real_loader_contract(registry):
    """create() promises the plugin 'already conforms to the contract' — it must
    be true against the contract the loader actually applies."""
    from orion_core.forge_contract import schema_problems
    registry.create("demo", "A demo.", "allow")
    module = registry.directory / "demo_tool.py"
    namespace: dict = {}
    exec(compile(module.read_text(encoding="utf-8"), str(module), "exec"), namespace)
    schema = namespace["get_tool_schema"]()
    problems, _properties = schema_problems(schema)
    assert problems == [], f"a scaffolded plugin would not load: {problems}"


@pytest.mark.parametrize("kind", ["tool", "event"])
def test_scaffold_preserves_a_detailed_purpose_as_text(registry, kind):
    import json

    purpose = ('Inspect "quoted" files beneath C:\\Users\\Example.\n'
               'Report findings and do not change files.\n'
               'Literal docstring delimiter: """; control: \x00; symbol: \u03a9.')
    result = registry.create("purposeful", purpose, "confirm", kind=kind)
    assert result.ok, result.text
    module = registry.directory / "purposeful_tool.py"
    assert pr.validate_module(module) == []
    namespace = {}
    exec(compile(module.read_text(encoding="utf-8"), str(module), "exec"), namespace)
    assert namespace["get_tool_schema"]()["description"] == purpose
    assert namespace["__doc__"].strip().startswith(purpose)
    manifest = json.loads((registry.directory / "purposeful.plugin.json").read_text(encoding="utf-8"))
    assert manifest["description"] == purpose


def test_a_reactive_scaffold_also_passes_the_loader_contract(registry):
    from orion_core.forge_contract import schema_problems
    registry.create("lamp", "Reacts.", "allow", kind="event")
    module = registry.directory / "lamp_tool.py"
    namespace: dict = {}
    exec(compile(module.read_text(encoding="utf-8"), str(module), "exec"), namespace)
    problems, _ = schema_problems(namespace["get_tool_schema"]())
    assert problems == []


def _ship(folder, name, body="def run(**k):\n    return 'v1'\n"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}_tool.py").write_text(body, encoding="utf-8")
    (folder / f"{name}.plugin.json").write_text('{"name": "%s"}' % name, encoding="utf-8")


def test_shipped_plugins_are_delivered_once_and_deletion_is_respected(tmp_path):
    source, target = tmp_path / "bundle", tmp_path / "live"
    _ship(source, "weather")
    assert pr.seed_shipped(target, source) == ["weather"]
    assert (target / "weather_tool.py").read_text(encoding="utf-8").endswith("'v1'\n")
    assert pr.seed_shipped(target, source) == []
    (target / "weather_tool.py").unlink()
    (target / "weather.plugin.json").unlink()
    _ship(source, "weather", "def run(**k):\n    return 'v2'\n")
    assert pr.seed_shipped(target, source) == []
    assert not (target / "weather_tool.py").exists()


def test_shipped_upgrade_replaces_untouched_copies_but_not_edits(tmp_path):
    source, target = tmp_path / "bundle", tmp_path / "live"
    _ship(source, "hue")
    _ship(source, "push")
    pr.seed_shipped(target, source)
    (target / "push_tool.py").write_text("def run(**k):\n    return 'mine'\n", encoding="utf-8")
    _ship(source, "hue", "def run(**k):\n    return 'v2'\n")
    _ship(source, "push", "def run(**k):\n    return 'v2'\n")
    assert pr.seed_shipped(target, source) == ["hue"]
    assert "'v2'" in (target / "hue_tool.py").read_text(encoding="utf-8")
    assert "'mine'" in (target / "push_tool.py").read_text(encoding="utf-8")


def test_an_older_unledgered_copy_is_backed_up_then_replaced(tmp_path):
    source, target = tmp_path / "bundle", tmp_path / "live"
    _ship(source, "error_handler", "def run(**k):\n    return 'reviewed'\n")
    _ship(target, "error_handler", "def run(**k):\n    return 'july'\n")
    assert pr.seed_shipped(target, source) == ["error_handler"]
    assert "'reviewed'" in (target / "error_handler_tool.py").read_text(encoding="utf-8")
    kept = list((target / "_superseded").glob("error_handler_tool.py.*"))
    assert kept and "'july'" in kept[0].read_text(encoding="utf-8")


def test_a_quarantined_plugin_gets_a_newly_shipped_fix(tmp_path):
    source, target = tmp_path / "bundle", tmp_path / "live"
    _ship(source, "json_validator")
    pr.seed_shipped(target, source)
    (target / "_quarantine").mkdir()
    (target / "json_validator_tool.py").rename(
        target / "_quarantine" / "json_validator_tool.py.broken")
    assert pr.seed_shipped(target, source) == []
    _ship(source, "json_validator", "def run(**k):\n    return 'fixed'\n")
    assert pr.seed_shipped(target, source) == ["json_validator"]


def test_seeding_is_a_no_op_without_a_bundle(tmp_path):
    assert pr.seed_shipped(tmp_path / "live", tmp_path / "absent") == []
    assert pr.seed_shipped(tmp_path / "live") == []


def test_static_audits_are_cached_by_content_and_redone_on_any_edit(tmp_path, monkeypatch):
    """28 Oct 2026: the Plugins page rescans every few seconds; re-parsing
    every unchanged plugin held the interpreter lock for 50-370 ms."""
    module = tmp_path / "probe_tool.py"
    module.write_text("def get_tool_schema():\n    return {}\n\ndef run(args):\n    return 1\n",
                      encoding="utf-8")
    calls: list[str] = []
    real = pr.validate_module_uncached
    monkeypatch.setattr(pr, "validate_module_uncached",
                        lambda path: calls.append(path.name) or real(path))
    pr._AUDITS.clear()
    first = pr.validate_module(module)
    assert pr.validate_module(module) == first
    assert calls == ["probe_tool.py"]                         # parsed once
    module.write_text("def run(args):\n    return 1\n", encoding="utf-8")
    assert "missing a top-level get_tool_schema() function" in pr.validate_module(module)
    assert len(calls) == 2                                    # an edit is audited afresh
    problems = pr.validate_module(module)
    problems.append("caller's own note")
    assert "caller's own note" not in pr.validate_module(module)   # callers get a copy
    assert pr.validate_module(tmp_path / "missing_tool.py")

