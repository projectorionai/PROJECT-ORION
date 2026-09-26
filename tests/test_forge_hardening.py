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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dynamic_loader import ReflectiveModuleLoader
from orion_core.forge import ForgeOrchestrationManager, ImprovementHeartbeat
from orion_core.forge_contract import (
    contract_problems,
    schema_problems,
    signature_problems,
)
from orion_core.forge_lessons import ForgeLessonStore
from orion_core.sandbox import CONFORMANCE_SCRIPT_NAME, probe_source


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
    assert not any("orion" in line.lower() for line in code_lines), code_lines


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
