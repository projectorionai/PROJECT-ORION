"""
Tests for the follow-up hardening after Forge Mark II.

Five gaps, each one a thing that was silently wrong rather than loudly broken:

  * tools forged BEFORE the contract gate existed loaded fine and only failed
    when the dispatcher first called them → the loader now applies the same gate;
  * quarantined tools vanished with no record of what they were or why they
    failed → a quarantine index, and a forge health report that surfaces it;
  * the improvement heartbeat was throttled for a Forge that used to fail;
  * self-repair was purely reactive — behaviour that is wrong but never raises
    could not be reported at all → request_repair locates it from a description;
  * console output died on Windows the first time a log line carried a tick.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dynamic_loader import ReflectiveModuleLoader
from orion_core.forge import ForgeOrchestrationManager, ImprovementHeartbeat
from orion_core.forge_contract import (
    contract_problems,
    schema_problems,
    signature_problems,
)
from orion_core.forge_lessons import ForgeLessonStore
from orion_core.sandbox import CONFORMANCE_SCRIPT_NAME, SandboxVerificationHarness, probe_source


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
        return " ".join(
            str(part) for payload in self.log.emitted for part in payload)


# ── the contract rules themselves ────────────────────────────────────────────


def _schema(properties=None, required=None, **overrides):
    schema = {
        "name": "demo",
        "description": "a demo tool",
        "parameters": {
            "type": "object",
            "properties": properties if properties is not None else {},
            "required": required if required is not None else [],
        },
    }
    schema.update(overrides)
    return schema


def test_kwargs_tool_accepts_anything():
    assert contract_problems(_schema({"url": {"type": "string"}}),
                             lambda **kwargs: "ok") == []


def test_declared_property_run_cannot_accept_is_rejected():
    def run(path=""):
        return path

    problems = contract_problems(_schema({"url": {"type": "string"}}), run)
    assert any("does not accept" in p for p in problems)


def test_required_run_argument_missing_from_schema_is_rejected():
    def run(url):
        return url

    problems = contract_problems(_schema({}), run)
    assert any("does not declare it" in p for p in problems)


@pytest.mark.parametrize("source", [
    "def run(path):\n    return path\n",
    "def run(*, path):\n    return path\n",
    "def run(path, **kwargs):\n    return path\n",
])
@pytest.mark.parametrize("required", [None, []])
def test_schema_optional_argument_cannot_be_required_by_run(source, required):
    namespace = {}
    exec(compile(source, "<test>", "exec"), namespace)
    schema = _schema({"path": {"type": "string"}})
    if required is None:
        del schema["parameters"]["required"]
    problems = contract_problems(schema, namespace["run"])
    assert any("does not mark it as required" in p for p in problems), problems


def test_signature_properties_only_api_keeps_legacy_behavior():
    def run(path):
        return path

    properties = {"path": {"type": "string"}}
    assert signature_problems(run, properties) == []
    assert signature_problems(run, properties, required=["path"]) == []
    assert signature_problems(run, properties, required=[])


@pytest.mark.parametrize("source, required", [
    ("def run(path):\n    return path\n", ["path"]),
    ("def run(*, path):\n    return path\n", ["path"]),
    ("def run(path='local'):\n    return path\n", []),
    ("def run(path='local'):\n    return path\n", ["path"]),
    ("def run(**kwargs):\n    return kwargs\n", []),
])
def test_schema_requiredness_accepts_callable_handler_controls(source, required):
    namespace = {}
    exec(compile(source, "<test>", "exec"), namespace)
    schema = _schema({"path": {"type": "string"}}, required)
    assert contract_problems(schema, namespace["run"]) == []


@pytest.mark.parametrize("entry", [{}, [], None, 1, True])
def test_malformed_required_entries_are_validation_problems(entry):
    schema = _schema({"path": {"type": "string"}}, [entry])
    problems, _ = schema_problems(schema)
    assert any("required" in p and "strings" in p for p in problems), problems
    assert contract_problems(schema, lambda **kwargs: kwargs) == problems


def test_positional_only_parameter_is_rejected():
    source = "def run(url, /):\n    return url\n"
    namespace: dict = {}
    exec(compile(source, "<test>", "exec"), namespace)
    problems = contract_problems(_schema({"url": {"type": "string"}}), namespace["run"])
    assert any("positional-only" in p for p in problems)


def test_required_key_absent_from_properties_is_rejected():
    problems, _ = schema_problems(_schema({}, ["url"]))
    assert any("listed as required" in p for p in problems)


def test_schema_shape_faults_are_all_reported():
    assert any("must return a dict" in p for p in schema_problems("nope")[0])
    assert any("non-empty 'name'" in p for p in schema_problems(_schema(name=""))[0])
    assert any("'parameters' object" in p
               for p in schema_problems({"name": "d", "description": "d"})[0])


def test_uninspectable_callable_is_not_rejected():
    """Some callables cannot be introspected at all; that is not a contract
    fault, and it must not crash the check."""
    class _Opaque:
        def __call__(self, *args, **kwargs):
            return "ok"

        @property
        def __signature__(self):
            raise ValueError("no signature available")

    assert signature_problems(_Opaque(), {"url": {"type": "string"}}) == []


def test_the_sandbox_probe_ships_the_same_rules():
    """The probe and the loader must never drift — the probe is built FROM the
    rules module, so its source necessarily contains them."""
    source = probe_source()
    assert "def contract_problems" in source
    assert 'if __name__ == "__main__":' in source
    compile(source, CONFORMANCE_SCRIPT_NAME, "exec")     # it must be valid Python
    # It runs inside the isolated staging directory, so it must not import
    # anything from ORION's own package — that isolation is the whole point.
    code_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert code_lines, "the probe should import something"
    assert not any("orion_core" in line.lower() for line in code_lines), code_lines


def test_sandbox_probe_rejects_optional_handler_argument_without_calling_it(
        tmp_path, monkeypatch, capsys):
    from orion_core.forge_fixtures import fixture_source
    (tmp_path / "orion_forge_fixtures.py").write_text(fixture_source(), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    schema = _schema({"path": {"type": "string"}})
    (tmp_path / "optional_tool.py").write_text(
        "def get_tool_schema():\n    return " + repr(schema)
        + "\ndef run(path):\n    raise AssertionError('handler was called')\n",
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [CONFORMANCE_SCRIPT_NAME, "optional"])
    with pytest.raises(SystemExit) as stopped:
        exec(compile(probe_source(), CONFORMANCE_SCRIPT_NAME, "exec"),
             {"__name__": "__main__"})
    assert stopped.value.code == 1
    assert "does not mark it as required" in capsys.readouterr().err


# ── the loader gate ──────────────────────────────────────────────────────────


_CONTRACT_BREAKER = (
    "def get_tool_schema():\n"
    "    return {'name': 'mismatch', 'description': 'broken',\n"
    "            'parameters': {'type': 'object',\n"
    "                           'properties': {'url': {'type': 'string'}},\n"
    "                           'required': ['url']}}\n\n"
    "def run(path):\n"
    "    return path\n"
)


def test_loader_refuses_a_tool_whose_schema_and_run_disagree(tmp_path):
    """A tool forged before the contract gate existed loads perfectly and then
    raises TypeError the first time the dispatcher calls it."""
    loader = ReflectiveModuleLoader(_StubBus())
    path = tmp_path / "mismatch_tool.py"
    path.write_text(_CONTRACT_BREAKER, encoding="utf-8")

    outcome = asyncio.run(loader.load_and_register(path))

    assert outcome.succeeded is False
    assert any("CONTRACT:" in line for line in outcome.error_log)
    assert "mismatch_tool" in loader.contract_failures
    # It is NOT quarantined: the file is valid Python and worth re-forging.
    assert path.exists()
    assert not (tmp_path / "_quarantine").exists()


def test_loader_still_activates_a_conformant_tool(tmp_path):
    loader = ReflectiveModuleLoader(_StubBus())
    path = tmp_path / "good_tool.py"
    path.write_text(
        "def get_tool_schema():\n"
        "    return {'name': 'good', 'description': 'good',\n"
        "            'parameters': {'type': 'object', 'properties': {}}}\n\n"
        "def run(**kwargs):\n"
        "    return 'ok'\n",
        encoding="utf-8")

    outcome = asyncio.run(loader.load_and_register(path))

    assert outcome.succeeded is True
    assert loader.contract_failures == {}


def test_loader_refuses_schema_optional_argument_required_by_handler(tmp_path):
    loader = ReflectiveModuleLoader(_StubBus())
    path = tmp_path / "optional_tool.py"
    schema = _schema({"path": {"type": "string"}})
    path.write_text("def get_tool_schema():\n    return " + repr(schema)
                    + "\ndef run(path):\n    return path\n", encoding="utf-8")
    outcome = asyncio.run(loader.load_and_register(path))
    assert not outcome.succeeded
    assert any("does not mark it as required" in p for p in outcome.error_log)
    assert path.exists()


# ── quarantine visibility ────────────────────────────────────────────────────


def test_quarantine_records_the_reason(tmp_path):
    loader = ReflectiveModuleLoader(_StubBus())
    loader.custom_tools_dir = tmp_path
    bad = tmp_path / "netprobe_tool.py"
    bad.write_text("code\n", encoding="utf-8")

    asyncio.run(loader.load_and_register(bad))

    index = json.loads((tmp_path / "_quarantine" / "index.json").read_text(encoding="utf-8"))
    assert "netprobe_tool.py.broken" in index
    assert index["netprobe_tool.py.broken"]["tool"] == "netprobe_tool"
    assert index["netprobe_tool.py.broken"]["reason"]

    rows = loader.quarantined()
    assert len(rows) == 1
    assert rows[0]["tool"] == "netprobe_tool"
    assert rows[0]["reason"] != "reason not recorded"


def test_quarantined_files_without_an_index_are_still_listed(tmp_path):
    """Artefacts quarantined before the index existed must not stay invisible."""
    loader = ReflectiveModuleLoader(_StubBus())
    loader.custom_tools_dir = tmp_path
    qdir = tmp_path / "_quarantine"
    qdir.mkdir()
    (qdir / "ancient_tool.py.broken").write_text("code\n", encoding="utf-8")

    rows = loader.quarantined()
    assert len(rows) == 1
    assert rows[0]["tool"] == "ancient_tool"
    assert rows[0]["reason"] == "reason not recorded"


def test_quarantined_is_empty_when_there_is_no_quarantine(tmp_path):
    loader = ReflectiveModuleLoader(_StubBus())
    loader.custom_tools_dir = tmp_path
    assert loader.quarantined() == []


# ── the forge health report ──────────────────────────────────────────────────


def _forge_with_loader(loader, lessons=None):
    forge = ForgeOrchestrationManager(_StubBus(), lessons=lessons)
    forge.loader = loader
    return forge


def test_health_reports_contract_failures_and_quarantine(tmp_path):
    loader = ReflectiveModuleLoader(_StubBus())
    loader.custom_tools_dir = tmp_path
    (tmp_path / "mismatch_tool.py").write_text(_CONTRACT_BREAKER, encoding="utf-8")
    asyncio.run(loader.load_and_register(tmp_path / "mismatch_tool.py"))
    bad = tmp_path / "broken_tool.py"
    bad.write_text("code\n", encoding="utf-8")
    asyncio.run(loader.load_and_register(bad))

    result = _forge_with_loader(loader).health()

    assert not result.ok
    assert "break the tool contract" in result.text
    assert "mismatch_tool" in result.text
    assert "quarantined" in result.text
    assert "broken_tool" in result.text


def test_health_is_clean_when_nothing_is_held_back(tmp_path):
    loader = ReflectiveModuleLoader(_StubBus())
    loader.custom_tools_dir = tmp_path
    result = _forge_with_loader(loader).health()
    assert result.ok
    assert "Nothing is held back" in result.text


def test_health_includes_the_lesson_corpus(tmp_path):
    from orion_core.forge_diagnosis import diagnose
    store = ForgeLessonStore(tmp_path / "lessons.jsonl")
    log = ["ValueError: boom"]
    store.record_failure("demo", diagnose(log, "demo"), log)
    loader = ReflectiveModuleLoader(_StubBus())
    loader.custom_tools_dir = tmp_path
    assert "Lesson corpus: 1 recorded" in _forge_with_loader(loader, store).health().text


def test_snapshot_matches_health_for_contract_failures_and_quarantine(tmp_path):
    """snapshot() and health() must never drift — same fixture as
    test_health_reports_contract_failures_and_quarantine, checked as
    structured data instead of text."""
    loader = ReflectiveModuleLoader(_StubBus())
    loader.custom_tools_dir = tmp_path
    (tmp_path / "mismatch_tool.py").write_text(_CONTRACT_BREAKER, encoding="utf-8")
    asyncio.run(loader.load_and_register(tmp_path / "mismatch_tool.py"))
    bad = tmp_path / "broken_tool.py"
    bad.write_text("code\n", encoding="utf-8")
    asyncio.run(loader.load_and_register(bad))

    snap = _forge_with_loader(loader).snapshot()

    names = {row["name"] for row in snap["contract_failures"]}
    assert "mismatch_tool" in names
    quarantined_names = {row["tool"] for row in snap["quarantine"]}
    assert "broken_tool" in quarantined_names


def test_snapshot_is_clean_when_nothing_is_held_back(tmp_path):
    loader = ReflectiveModuleLoader(_StubBus())
    loader.custom_tools_dir = tmp_path
    snap = _forge_with_loader(loader).snapshot()
    assert snap["contract_failures"] == []
    assert snap["quarantine"] == []
    assert snap["live"] == 0


def test_snapshot_includes_lesson_corpus_stats(tmp_path):
    from orion_core.forge_diagnosis import diagnose
    store = ForgeLessonStore(tmp_path / "lessons.jsonl")
    log = ["ValueError: boom"]
    store.record_failure("demo", diagnose(log, "demo"), log)
    loader = ReflectiveModuleLoader(_StubBus())
    loader.custom_tools_dir = tmp_path
    snap = _forge_with_loader(loader, store).snapshot()
    assert snap["lessons"]["lessons"] == 1


def test_reload_announces_tools_it_held_back(tmp_path, monkeypatch):
    import orion_core.forge as forge_mod
    tools_dir = tmp_path / "config" / "custom_tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "mismatch_tool.py").write_text(_CONTRACT_BREAKER, encoding="utf-8")
    monkeypatch.setattr(forge_mod, "CONFIG_DIR", tmp_path / "config")

    bus = _StubBus()
    forge = ForgeOrchestrationManager(bus)
    forge.loader = ReflectiveModuleLoader(_StubBus())
    forge.dispatcher = object()

    assert asyncio.run(forge.reload_persisted_tools()) == 0
    assert "held back" in bus.lines()


# ── the heartbeat is less timid now the Forge is reliable ────────────────────


def test_heartbeat_cadence_was_raised():
    assert ImprovementHeartbeat.INTERVAL_S <= 25 * 60
    assert ImprovementHeartbeat.MAX_AUTO_FORGES_PER_SESSION >= 5


# ── proactive self-repair ────────────────────────────────────────────────────


class _Telemetry:
    class _Log:
        def recent(self, limit=25):
            return []

        def info(self, *_a, **_k):
            pass

    class _Metrics:
        def incr(self, *_a, **_k):
            pass

    class _Health:
        def register(self, *_a, **_k):
            pass

        def beat(self, *_a, **_k):
            pass

    def __init__(self):
        self.log = self._Log()
        self.metrics = self._Metrics()
        self.health = self._Health()


class _Router:
    def has_text_fallback(self):
        return False


def _agent():
    from orion_core.selfrepair import SelfRepairAgent
    return SelfRepairAgent(_StubBus(), _Telemetry(), _Router())


def test_locate_source_finds_the_module_named_in_the_report():
    agent = _agent()
    path, line, rationale = agent.locate_source(
        "the camera vision analysis keeps missing objects in the frame")
    assert Path(path).name == "vision.py", rationale
    assert line >= 1


def test_locate_source_prefers_the_filename_match():
    agent = _agent()
    path, _line, _why = agent.locate_source("the forge lessons are not recorded")
    assert Path(path).stem.startswith("forge")


def test_explicit_module_filename_outranks_incidental_scene_prose(tmp_path, monkeypatch):
    from orion_core import selfrepair
    package = tmp_path / "synthetic_package"
    package.mkdir()
    (package / "vision.py").write_text("def analyse():\n    pass\n", encoding="utf-8")
    (package / "scene_analysis.py").write_text(
        '# camera vision analysis missing objects in the frame\n' * 100,
        encoding="utf-8")
    monkeypatch.setattr(selfrepair, "PACKAGE_DIR", package)
    path, line, _why = _agent().locate_source("vision.py camera vision analysis missing objects in the frame")
    assert Path(path).name == "vision.py" and line >= 1


def test_locate_source_reports_when_nothing_matches():
    agent = _agent()
    path, line, rationale = agent.locate_source("the the and for")
    assert path == "" and line == 0
    assert "no searchable terms" in rationale or "no module matched" in rationale


def test_request_repair_creates_an_actionable_located_incident():
    agent = _agent()
    incident = agent.request_repair(
        "the camera vision analysis keeps missing objects in the frame")
    assert incident.reported is True
    assert incident.actionable is True
    assert incident.error_type == "ReportedDefect"
    assert Path(incident.file).name == "vision.py"
    assert agent.latest() is incident


def test_request_repair_honours_an_explicit_file():
    agent = _agent()
    incident = agent.request_repair("something is off here", file="orion_core/memory.py")
    assert incident.file == "orion_core/memory.py"


def test_request_repair_without_a_match_is_not_actionable():
    agent = _agent()
    incident = agent.request_repair("the and for with that")
    assert incident.file == ""
    assert incident.actionable is False


def test_reported_incident_gets_a_wider_source_window():
    """A description has no traceback narrowing the location, so the model must
    be shown far more of the file than a stack frame would justify."""
    agent = _agent()
    incident = agent.request_repair(
        "the camera vision analysis keeps missing objects in the frame")
    narrow = agent._read_source_window(incident.file, incident.line, 30)
    wide = agent._read_source_window(incident.file, incident.line, 120)
    assert len(wide) > len(narrow)
    result = asyncio.run(agent.propose_fix(incident.id))
    # No text provider configured → it still produces the capture for the live
    # model rather than failing outright.
    assert result.ok


# ── console safety ───────────────────────────────────────────────────────────


def test_launcher_forces_utf8_streams():
    """ORION's log lines carry '✓' and '✗'; on a Windows console that is a
    UnicodeEncodeError unless the streams are reconfigured first."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "orion.py").read_text(encoding="utf-8")

    # It must run at import time, before any dependency gate can print.
    call_at = source.index("\n_force_utf8_console()")
    assert call_at < source.index("def _headless_requested")

    prelude = source[: call_at + len("\n_force_utf8_console()")]
    namespace: dict = {}
    exec(compile(prelude, "orion.py", "exec"), namespace)
    # Running it twice must be harmless — a redirected or already-wrapped
    # stream must never be the thing that crashes the launcher.
    namespace["_force_utf8_console"]()
    assert sys.stdout.encoding.lower().replace("-", "") in {"utf8", "cp1252", "ascii"}


def test_forge_routes_a_covered_request_before_generation():
    from types import SimpleNamespace

    manager = ForgeOrchestrationManager.__new__(ForgeOrchestrationManager)
    manager.bus = _StubBus()
    manager.dispatcher = SimpleNamespace(
        handler_table=lambda: {"find_files": object()},
        TOOL_DECLARATIONS=[{"name": "find_files", "description": "Find files by size or name"}],
    )

    def unexpected_generation(*_args):
        pytest.fail("a known built-in capability must not spend a generation")

    manager.code_generator = unexpected_generation
    result = asyncio.run(manager.forge_tool(
        "bulk_space_inventory", "Find the biggest files on the PC."))

    assert not result.ok
    assert result.receipt["covered_by"] == "find_files"
    assert result.receipt["triage"]["arguments"] == {"sort": "largest"}
    assert "find_files(sort='largest')" in result.text


def test_sandbox_stages_usable_fixtures_for_mechanically_repaired_tests(tmp_path, monkeypatch):
    from orion_core.forge import ForgeSession
    from orion_core.forge_fixtures import FIXTURE_MODULE_NAME
    from orion_core.sandbox import SandboxVerificationHarness

    session = ForgeSession(
        session_id="fixture-wiring", tool_name="echo_identifier",
        tool_code=(
            'def get_tool_schema():\n'
            '    return {"name": "echo_identifier", "description": "Echo an identifier",\n'
            '            "parameters": {"type": "object", "properties": {"identifier": {"type": "integer"}},\n'
            '                           "required": ["identifier"]}}\n'
            'def run(identifier):\n'
            '    return str(identifier)\n'),
        test_code=(
            'with tmp_workdir({"nested/note.txt": "local fixture"}) as folder:\n'
            '    assert (folder / "nested/note.txt").read_text(encoding="utf-8") == "local fixture"\n'
            '    assert run(identifier="9007199254740993") == "9007199254740993"\n'),
    )
    notes = ForgeOrchestrationManager._mechanical_fixes(session)
    assert notes
    harness = SandboxVerificationHarness.__new__(SandboxVerificationHarness)
    harness.staging_dir = tmp_path
    harness._isolation_mode = "process"
    harness._write_artefacts(session.tool_name, session.tool_code, session.test_code)

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    # Exercise only these synthetic staged files in this interpreter; no child
    # process, package installation or real generated tool is involved.
    monkeypatch.delitem(sys.modules, FIXTURE_MODULE_NAME, raising=False)
    monkeypatch.delitem(sys.modules, "echo_identifier_tool", raising=False)
    try:
        exec(compile(session.test_code, str(tmp_path / "echo_identifier_test.py"), "exec"), {})
    finally:
        sys.modules.pop(FIXTURE_MODULE_NAME, None)
        sys.modules.pop("echo_identifier_tool", None)
    assert not list(tmp_path.glob("fx_*"))


def test_duplicate_plugin_is_held_before_its_import_and_its_file_is_preserved(tmp_path, monkeypatch):
    from orion_core import dynamic_loader

    monkeypatch.setattr(dynamic_loader, "_bundled_plugins", lambda: frozenset())
    (tmp_path / ".shipped.json").write_text("{}", encoding="utf-8")
    source = b'raise AssertionError("the duplicate must never be imported")\n'
    path = tmp_path / "gesture_control_manager_tool.py"
    path.write_bytes(source)
    loader = ReflectiveModuleLoader.__new__(ReflectiveModuleLoader)
    loader.native_names = lambda: {"gesture_control"}
    loader.bus = _StubBus()

    outcome = asyncio.run(loader.load_and_register(path))

    assert not outcome.succeeded
    assert "duplicates ORION's built-in tool 'gesture_control'" in outcome.error_log[0]
    assert path.read_bytes() == source
    assert not (tmp_path / "_quarantine").exists()


def test_duplicate_schema_name_is_refused_at_activation_even_with_a_distinct_filename(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from orion_core import dynamic_loader
    from orion_core.dynamic_loader import ModuleLoadOutcome

    monkeypatch.setattr(dynamic_loader, "_bundled_plugins", lambda: frozenset())
    (tmp_path / ".shipped.json").write_text("{}", encoding="utf-8")
    loader = ReflectiveModuleLoader.__new__(ReflectiveModuleLoader)
    loader.loaded_modules = {}
    schema = _schema(name="gesture_control_manager")
    outcome = ModuleLoadOutcome(
        True, "distinct_tool", module_path=tmp_path / "distinct_tool.py",
        tool_schema=schema, handler_fn=lambda: "should not be activated")
    dispatcher = SimpleNamespace(
        handler_table=lambda: {"gesture_control": object()},
        _tool_handlers={}, _mcp_routes={}, TOOL_DECLARATIONS=[])

    result = asyncio.run(loader.activate_tool(outcome, dispatcher))

    assert not result.ok and result.receipt["state"] == "duplicate"
    assert dispatcher._tool_handlers == {} and dispatcher.TOOL_DECLARATIONS == []


def test_reviewed_shipped_adapter_is_exempt_from_the_duplicate_guard(tmp_path, monkeypatch):
    from orion_core import dynamic_loader

    monkeypatch.setattr(dynamic_loader, "_bundled_plugins", lambda: frozenset())
    (tmp_path / ".shipped.json").write_text(
        json.dumps({"gesture_control_manager": "reviewed delivery"}), encoding="utf-8")
    loader = ReflectiveModuleLoader.__new__(ReflectiveModuleLoader)
    assert loader.duplicate_reason(
        "gesture_control_manager", tmp_path / "gesture_control_manager_tool.py",
        {"gesture_control"}) == ""


def _example_plan():
    return {
        "inputs": {"n": {"type": "integer"}}, "required": ["n"],
        "output": "A string reporting n=<the supplied integer>", "dependencies": [],
        "examples": [{"arguments": {"n": n}, "expect": f"n={n}"} for n in (1, 2, 3)],
    }


_EXAMPLE_MODULE = '''
def get_tool_schema():
    return {"name": "number_report", "description": "Report the supplied number",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}},
                           "required": ["n"]}}
def run(n):
    return f"n={n}"
'''


@pytest.mark.parametrize("bad", ["count", "output", "expect", "inputs", "arguments", "path",
                                 "argument_path", "argument_traversal", "all_errors", "oversize"])
def test_planning_contract_rejects_unjudgeable_or_unsafe_examples(bad):
    from orion_core.forge_fixtures import validate_plan

    plan = _example_plan()
    if bad == "count":
        plan["examples"].pop()
    elif bad == "output":
        plan["output"] = ""
    elif bad == "expect":
        plan["examples"][0]["expect"] = ""
    elif bad == "inputs":
        plan["inputs"]["n"]["type"] = "python-object"
    elif bad == "arguments":
        plan["examples"][0]["arguments"] = {"unknown": 1}
    elif bad == "path":
        plan["examples"][0]["files"] = {"../outside.txt": "must not write"}
    elif bad == "argument_path":
        plan["examples"][0]["arguments"]["n"] = "C:/Users/synthetic/private.txt"
    elif bad == "argument_traversal":
        plan["examples"][0]["arguments"]["n"] = "../../private.txt"
    elif bad == "all_errors":
        for example in plan["examples"]:
            example["expect_error"] = True
    elif bad == "oversize":
        plan["description"] = "x" * 32769
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_planned_examples_judge_answers_and_hold_the_input_shape(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from orion_core.forge_fixtures import run_examples, validate_plan

    monkeypatch.chdir(tmp_path)
    plan = validate_plan(_example_plan())
    schema = {"parameters": {"type": "object", "properties": {"n": {"type": "integer"}},
                             "required": ["n"]}}
    calls = []

    def run(n):
        calls.append(n)
        return f"n={n}"

    module = SimpleNamespace(get_tool_schema=lambda: schema, run=run)
    run_examples(module, plan["examples"], plan=plan)
    assert calls == [1, 2, 3]

    module.run = lambda **_kwargs: "a plausible but wrong answer"
    with pytest.raises(AssertionError, match="expected the result"):
        run_examples(module, plan["examples"], plan=plan)
    schema["parameters"]["properties"]["n"]["type"] = "string"
    with pytest.raises(AssertionError, match="planned type"):
        run_examples(module, plan["examples"], plan=plan)
    assert not list(tmp_path.glob("fx_*"))


def test_planned_file_examples_get_only_their_private_work_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from orion_core.forge_fixtures import run_examples

    monkeypatch.chdir(tmp_path)
    plan = {
        "inputs": {"path": {"type": "string"}}, "required": ["path"],
        "output": "The file's text", "dependencies": [],
        "examples": [{"arguments": {"path": "{workdir}/nested/note.txt"},
                      "files": {"nested/note.txt": text}, "expect": text}
                     for text in ("one", "two", "three")],
    }
    module = SimpleNamespace(
        get_tool_schema=lambda: {"parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                 "required": ["path"]}},
        run=lambda path: Path(path).read_text(encoding="utf-8"))
    run_examples(module, plan["examples"], plan=plan)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [None, "answer", "timeout", "memory"])
def test_sandbox_checks_examples_before_the_models_test(tmp_path, monkeypatch, failure):
    from orion_core.sandbox import EXAMPLES_SCRIPT_NAME, SandboxVerificationHarness

    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "0")
    harness = SandboxVerificationHarness.__new__(SandboxVerificationHarness)
    harness.bus = _StubBus()
    harness.staging_dir = tmp_path
    harness._isolation_mode = "process"
    harness._typecheck_policy = "off"
    harness._plan_contract = _example_plan()
    stages = []

    def run_script(script, timeout, *_args):
        stages.append(script)
        assert 0 < timeout <= 90
        if script == CONFORMANCE_SCRIPT_NAME:
            schema = {"name": "number_report", "description": "Number report",
                      "parameters": {"type": "object", "properties": {"n": {"type": "integer"}},
                                     "required": ["n"]}}
            return json.dumps(schema) + "\nconformance OK", "", 0, False, False
        if script == EXAMPLES_SCRIPT_NAME and failure:
            return "", "AssertionError: expected n=1", 1, failure == "timeout", failure == "memory"
        if script == EXAMPLES_SCRIPT_NAME:
            from orion_core.forge_fixtures import plan_fingerprint
            receipt = {"count": 3, "plan": plan_fingerprint(_example_plan())}
            return "ORION_EXAMPLES_COMPLETE:" + json.dumps(receipt), "", 0, False, False
        return "all expected answers checked", "", 0, False, False

    harness._run_script = run_script
    outcome = asyncio.run(harness._verify_staged(_EXAMPLE_MODULE, "assert True", "number_report"))
    assert stages[:2] == [CONFORMANCE_SCRIPT_NAME, EXAMPLES_SCRIPT_NAME]
    if failure:
        assert not outcome.passed and not outcome.examples_verified
        assert outcome.stage == "examples" and len(stages) == 2
        assert outcome.timed_out is (failure == "timeout")
        assert outcome.memory_exceeded is (failure == "memory")
    else:
        assert outcome.passed and outcome.examples_verified
        assert stages[2] == "number_report_test.py"
        assert outcome.to_dict()["plan"] == _example_plan()


def _isolated_planned_manager(generator, verification):
    from types import SimpleNamespace

    manager = ForgeOrchestrationManager.__new__(ForgeOrchestrationManager)
    manager.bus = _StubBus()
    manager.dispatcher = None
    manager.code_generator = generator
    manager.sessions = {}
    manager.max_attempts = 1
    manager.history = manager.lessons = manager.repair = None
    manager.llm_repairer = manager.llm_fixer = None
    manager.pending_retries = []
    manager.sandbox = SimpleNamespace(verify_tool=verification)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("a candidate without successful planned examples cannot be installed or activated")

    manager.resolver = SimpleNamespace(resolve_and_install=forbidden)
    manager._activate_verified = forbidden
    return manager


def test_real_brain_plans_then_builds_and_failed_examples_prevent_acceptance(monkeypatch):
    from orion_core.forge import LlmForgeBrain
    from orion_core.sandbox import VerificationOutcome

    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "0")

    class Router:
        def __init__(self):
            self.prompts = []

        async def generate_text(self, prompt, **_kwargs):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return object(), json.dumps(_example_plan())
            return object(), f"=== TOOL ===\n```python\n{_EXAMPLE_MODULE}```\n=== TEST ===\n```python\nassert True\n```"

    router = Router()
    brain = LlmForgeBrain(_StubBus(), router)
    plans = []

    async def failed(_source, _test, name, *, plan):
        plans.append(plan)
        return VerificationOutcome(False, name, stage="examples", error_log=["AssertionError: wrong answer"])

    manager = _isolated_planned_manager(brain.generate, failed)
    result = asyncio.run(manager.forge_tool("number_report", "Report a supplied integer"))

    assert not result.ok and len(router.prompts) == 2
    assert "before writing any code" in router.prompts[0]
    assert "FIXED PLAN CONTRACT" in router.prompts[1]
    assert plans == [_example_plan()]
    session = next(iter(manager.sessions.values()))
    assert session.plan_contract == _example_plan() and not session.activation_ok
    assert not session.examples_verified


@pytest.mark.parametrize("false_proof", [False, True])
def test_a_verifiers_generic_pass_cannot_skip_planned_example_acceptance(monkeypatch, false_proof):
    from orion_core.forge_artefacts import ForgeArtefacts
    from orion_core.sandbox import VerificationOutcome

    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "0")

    async def unproved(_source, _test, name, **_kwargs):
        return VerificationOutcome(True, name, examples_verified=false_proof,
                                   plan=None)  # an absent/mismatched plan cannot prove these examples

    manager = _isolated_planned_manager(
        lambda *_args: ForgeArtefacts(_EXAMPLE_MODULE, "assert True", plan=_example_plan()), unproved)
    result = asyncio.run(manager.forge_tool("number_report", "Report a supplied integer"))
    assert not result.ok and "three planned examples" in result.text
    session = next(iter(manager.sessions.values()))
    assert not session.sandbox_ok and not session.activation_ok


def test_invalid_plan_stops_before_the_brain_generates_source(monkeypatch):
    from orion_core.forge import LlmForgeBrain

    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "0")
    calls = []

    class Router:
        async def generate_text(self, prompt, **_kwargs):
            calls.append(prompt)
            plan = _example_plan()
            plan["examples"] = []
            return object(), json.dumps(plan)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("invalid plans must not reach verification")

    brain = LlmForgeBrain(_StubBus(), Router())
    manager = _isolated_planned_manager(brain.generate, forbidden)
    result = asyncio.run(manager.forge_tool("number_report", "Report a supplied integer"))
    assert not result.ok and "exactly three" in result.text
    assert len(calls) == 1


def test_planner_timeout_is_bounded_and_does_not_generate_source(monkeypatch):
    from orion_core.forge import LlmForgeBrain

    timeouts = []

    async def expire(awaitable, timeout):
        timeouts.append(timeout)
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", expire)
    brain = LlmForgeBrain(_StubBus(), object())
    with pytest.raises(RuntimeError, match="45 seconds; no code was generated"):
        asyncio.run(brain.plan("number_report", "Report a supplied integer"))
    assert timeouts == [45.0]


def test_customer_review_receipt_keeps_the_detached_plan_and_example_proof(tmp_path):
    from orion_core.forge_isolation import save_review_candidate
    from orion_core.sandbox import VerificationOutcome

    outcome = VerificationOutcome(
        True, "number_report", isolation="container", typecheck="passed",
        plan=_example_plan(), examples_verified=True)
    detached = outcome.to_dict()
    detached["plan"]["examples"][0]["expect"] = "changed elsewhere"
    assert outcome.plan["examples"][0]["expect"] == "n=1"

    receipt = save_review_candidate(
        tmp_path / "candidates", "number_report", _EXAMPLE_MODULE, "assert True", outcome.to_dict())
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert saved["verification"]["examples_verified"] is True
    assert saved["verification"]["plan"] == _example_plan()
    assert saved["activated"] is False and saved["dependencies_installed"] is False


@pytest.mark.parametrize("where,exit_source", [
    ("import", "raise SystemExit(0)"), ("run", "raise SystemExit(0)"),
    ("import", "import os; os._exit(0)"), ("run", "import os; os._exit(0)"),
])
def test_zero_exit_before_verifier_completion_never_counts_as_proof(tmp_path, monkeypatch, where, exit_source):
    monkeypatch.setenv("ORION_FORGE_TYPECHECK", "off")
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    source = _EXAMPLE_MODULE
    if where == "import":
        source = exit_source + "\n" + source
    else:
        source = source.replace('return f"n={n}"', exit_source)
    outcome = asyncio.run(harness.verify_tool(
        source, "assert True", "number_report", plan=_example_plan()))
    assert not outcome.passed and not outcome.examples_verified
    assert outcome.stage == ("conformance" if where == "import" else "examples")


def test_real_planned_verifier_requires_all_three_examples_and_preserves_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_FORGE_TYPECHECK", "off")
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    outcome = asyncio.run(harness.verify_tool(
        _EXAMPLE_MODULE, 'print(\'{"name":"false_schema"}\')', "number_report", plan=_example_plan()))
    assert outcome.passed and outcome.examples_verified, outcome.error_log
    assert outcome.schema["name"] == "number_report"


@pytest.mark.parametrize("receipt", [None, {"count": 2, "plan": "bad"}, {"count": 3, "plan": "bad"}])
def test_exit_zero_without_exact_example_completion_receipt_is_refused(tmp_path, monkeypatch, receipt):
    from orion_core.sandbox import EXAMPLES_SCRIPT_NAME
    monkeypatch.setenv("ORION_FORGE_TYPECHECK", "off")
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    harness._plan_contract = _example_plan()
    seen = []

    def run_script(script, *_args):
        seen.append(script)
        if script == CONFORMANCE_SCRIPT_NAME:
            schema = {"name": "number_report", "description": "Number report",
                      "parameters": {"type": "object", "properties": {"n": {"type": "integer"}},
                                     "required": ["n"]}}
            return json.dumps(schema) + "\nconformance OK", "", 0, False, False
        if script == EXAMPLES_SCRIPT_NAME:
            output = "" if receipt is None else "ORION_EXAMPLES_COMPLETE:" + json.dumps(receipt)
            return output, "", 0, False, False
        pytest.fail("model tests must not run without fixed example proof")

    harness._run_script = run_script
    outcome = asyncio.run(harness._verify_staged(_EXAMPLE_MODULE, "assert True", "number_report"))
    assert not outcome.passed and not outcome.examples_verified
    assert outcome.stage == "examples"
    assert len(seen) == 2


@pytest.mark.parametrize("unsafe", ["file:///C:/private.txt", "file:///etc/passwd", "ftp://host/data",
                                   "~/private.txt", "%USERPROFILE%/private.txt", "$HOME/private.txt"])
def test_fixed_plan_refuses_non_private_url_and_expanded_paths(unsafe):
    from orion_core.forge_fixtures import validate_plan
    plan = _example_plan()
    plan["examples"][0]["arguments"]["n"] = unsafe
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_negative_example_requires_failure_rather_than_merely_allowing_it():
    from types import SimpleNamespace
    from orion_core.forge_fixtures import run_examples
    module = SimpleNamespace(get_tool_schema=lambda: {"name": "demo", "parameters": {
        "properties": {}, "required": []}}, run=lambda: "Everything succeeded")
    with pytest.raises(AssertionError, match="requires a failure"):
        run_examples(module, [{"arguments": {}, "expect": "succeeded", "expect_error": True}])


@pytest.mark.parametrize("requirements,planned", [
    (["requests"], []), (["requests>=2"], ["requests==2"]),
    (["other_package"], ["requests"]),
])
def test_frozen_plan_refuses_unplanned_packages_before_verifying_or_installing(
        monkeypatch, requirements, planned):
    from orion_core.forge_artefacts import ForgeArtefacts
    monkeypatch.setenv("ORION_FORGE_CUSTOMER_MODE", "0")
    plan = _example_plan()
    plan["dependencies"] = planned

    async def forbidden(*_args, **_kwargs):
        pytest.fail("unplanned dependencies must fail before verification or installation")

    manager = _isolated_planned_manager(
        lambda *_args: ForgeArtefacts(_EXAMPLE_MODULE, "assert True", requirements=requirements, plan=plan),
        forbidden)
    result = asyncio.run(manager.forge_tool("number_report", "Report an integer"))
    assert not result.ok and "Unplanned Forge dependency" in result.text


def test_missing_planned_dependency_keeps_the_frozen_version_constraint():
    from orion_core.forge import ForgeSession
    session = ForgeSession("session", "number_report")
    session.plan_contract = _example_plan()
    session.plan_contract["dependencies"] = ["requests==2.32.5"]
    assert ForgeOrchestrationManager._planned_missing_requirement(session, "requests") == "requests==2.32.5"
    with pytest.raises(ValueError, match="Unplanned Forge dependency"):
        ForgeOrchestrationManager._planned_missing_requirement(session, "other")


def test_conformance_probe_blocks_network_even_during_module_import(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_FORGE_TYPECHECK", "off")
    monkeypatch.setenv("ORION_FORGE_SANDBOX", "process")
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    source = "import socket\nsocket.create_connection(('127.0.0.1', 9))\n" + _EXAMPLE_MODULE
    outcome = asyncio.run(harness.verify_tool(source, "assert True", "number_report"))
    assert not outcome.passed and outcome.stage == "conformance"
    assert "NetworkBlocked" in "\n".join(outcome.error_log)
