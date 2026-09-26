"""
Tests for Forge Mark II — the reliability overhaul.

Covers the four structural defects that made forge sessions fail:

  * source travelled inside JSON strings, so one unescaped newline in a 200-line
    module killed the session → artefacts now travel in fenced blocks;
  * every failure got the same generic repair prompt → failures are classified,
    and each class carries its own directive;
  * only the MODULE was ever repaired, so a wrong TEST was unfixable → repairs
    are aimed by diagnosis at the artefact actually at fault;
  * nothing was remembered between sessions → failures accumulate as lessons
    that are injected back into the next generation.

Plus the sandbox's new contract gate, which catches a schema that disagrees
with run() before any test runs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.forge import ForgeOrchestrationManager, LlmForgeBrain, blocks_turn
from orion_core.forge_artefacts import (
    ForgeArtefacts,
    extract_artefacts,
    parse_requirements,
    strip_fences,
    synthesise_test,
)
from orion_core.forge_diagnosis import (
    FailureClass,
    RepairTarget,
    diagnose,
    error_signature,
    failing_artefact,
)
from orion_core.forge_lessons import ForgeLessonStore
from orion_core.sandbox import SandboxVerificationHarness, budget_for


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


_MODULE = (
    "def get_tool_schema():\n"
    "    return {'name': 'demo', 'description': 'demo tool',\n"
    "            'parameters': {'type': 'object', 'properties': {}, 'required': []}}\n\n"
    "def run(**kwargs):\n"
    "    return 'ok'\n"
)


# ── artefact transport: fences, not JSON escaping ────────────────────────────


def test_extracts_labelled_fenced_blocks():
    raw = (
        "=== TOOL ===\n```python\n" + _MODULE + "```\n"
        "=== TEST ===\n```python\nassert True\nprint('ok')\n```\n"
        "=== REQUIREMENTS ===\n```\nhttpx\nbeautifulsoup4\n```\n"
    )
    artefacts = extract_artefacts(raw)
    assert artefacts.transport == "blocks"
    assert "def get_tool_schema" in artefacts.tool_code
    assert "assert True" in artefacts.test_code
    assert artefacts.requirements == ["httpx", "beautifulsoup4"]


def test_source_with_raw_newlines_and_quotes_survives():
    """The exact payload that broke the JSON transport: embedded quotes,
    backslashes and blank lines inside the module body."""
    body = (
        'def get_tool_schema():\n'
        '    return {"name": "q", "description": "quote \\"test\\"",\n'
        '            "parameters": {"type": "object", "properties": {}}}\n'
        '\n'
        'def run(**kwargs):\n'
        '    path = "C:\\\\Users\\\\test"\n'
        '    return f"read {path}"\n'
    )
    raw = f"=== TOOL ===\n```python\n{body}```\n=== TEST ===\n```python\nassert 1\n```\n"
    artefacts = extract_artefacts(raw)
    assert artefacts.tool_code.strip() == body.strip()


def test_marker_punctuation_variants_all_parse():
    for header_tool, header_test in (
        ("### TOOL", "### TEST"),
        ("## Tool Module", "## Test Code"),
        ("**TOOL**", "**TEST**"),
        ("TOOL:", "TEST:"),
    ):
        raw = (
            f"{header_tool}\n```python\n{_MODULE}```\n"
            f"{header_test}\n```python\nassert True\n```\n"
        )
        artefacts = extract_artefacts(raw)
        assert "get_tool_schema" in artefacts.tool_code, header_tool
        assert "assert True" in artefacts.test_code, header_test


def test_unlabelled_blocks_are_assigned_by_shape():
    """Two bare fences, no headers: the one exporting the contract is the
    module and the one full of assertions is the test — even in this order."""
    raw = (
        "Here is the test first.\n```python\nimport demo_tool\nassert demo_tool.run()\n```\n"
        "And the module:\n```python\n" + _MODULE + "```\n"
    )
    artefacts = extract_artefacts(raw)
    assert "get_tool_schema" in artefacts.tool_code
    assert "assert demo_tool.run()" in artefacts.test_code


def test_json_transport_still_works_as_a_fallback():
    raw = '{"tool_code": "def get_tool_schema():\\n    pass\\ndef run():\\n    pass\\n",' \
          ' "test_code": "assert True", "requirements": ["httpx"]}'
    artefacts = extract_artefacts(raw)
    assert artefacts.transport == "json"
    assert "get_tool_schema" in artefacts.tool_code
    assert artefacts.requirements == ["httpx"]


def test_bare_source_with_no_fences_is_recovered():
    artefacts = extract_artefacts(_MODULE)
    assert artefacts.transport == "bare"
    assert "def run" in artefacts.tool_code


def test_unusable_response_yields_empty_artefacts():
    artefacts = extract_artefacts("I'm sorry, I can't help with that.")
    assert not artefacts.complete
    assert artefacts.tool_code == ""


def test_requirement_noise_is_discarded():
    assert parse_requirements("```\nNone\n```") == []
    assert parse_requirements("- httpx\n- pandas>=2.0\n") == ["httpx", "pandas>=2.0"]
    assert parse_requirements("httpx, rich") == ["httpx", "rich"]
    assert parse_requirements('["httpx"]') == ["httpx"]
    assert parse_requirements("standard library only") == []


def test_strip_fences_handles_double_wrapping():
    assert strip_fences("```python\nx = 1\n```") == "x = 1"
    assert strip_fences("x = 1") == "x = 1"


def test_synthesised_test_is_valid_python():
    import ast
    ast.parse(synthesise_test("demo"))


# ── failure classification ───────────────────────────────────────────────────


def test_missing_dependency_is_environmental_not_a_code_fault():
    d = diagnose(["ModuleNotFoundError: No module named 'httpx'"], "demo")
    assert d.failure_class is FailureClass.MISSING_DEPENDENCY
    assert d.target is RepairTarget.DEPENDENCY
    assert d.missing_module == "httpx"
    assert d.consumes_attempt is False


def test_dependency_that_stays_missing_after_install_becomes_a_code_fault():
    d = diagnose(["No module named 'ghost'"], "demo", already_installed={"ghost"})
    assert d.target is RepairTarget.BOTH
    assert d.consumes_attempt is True


def test_self_import_blames_the_test_not_pip():
    d = diagnose(
        ["ModuleNotFoundError: No module named 'enhanced_research_module'"],
        "EnhancedResearchModule",
    )
    assert d.failure_class is FailureClass.SELF_IMPORT
    assert d.target is RepairTarget.TEST
    assert d.missing_module == ""


def test_assertion_failure_targets_both_so_the_test_can_be_blamed():
    """The defect Mark II exists to fix: a wrong assertion used to be
    unfixable because only the module was ever sent for repair."""
    d = diagnose(
        ['File "demo_test.py", line 4, in <module>', "AssertionError: expected 'Hello'"],
        "demo",
    )
    assert d.failure_class is FailureClass.ASSERTION_FAILED
    assert d.repairs_test and d.repairs_module


def test_contract_violation_from_the_conformance_probe():
    d = diagnose(
        ["CONTRACT: the schema declares parameter 'url', but run() does not accept it"],
        "demo",
    )
    assert d.failure_class is FailureClass.CONTRACT_VIOLATION
    assert d.target is RepairTarget.MODULE


def test_signature_mismatch_targets_both():
    d = diagnose(["TypeError: run() got an unexpected keyword argument 'url'"], "demo")
    assert d.failure_class is FailureClass.CONTRACT_VIOLATION
    assert d.target is RepairTarget.BOTH


def test_timeout_is_not_diagnosed_as_a_code_bug():
    d = diagnose(["Subprocess timeout after 15s"], "demo", timed_out=True)
    assert d.failure_class is FailureClass.TIMEOUT
    assert "unbounded" in d.directive


def test_network_failure_blames_the_test():
    d = diagnose(
        ["urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>"],
        "demo",
    )
    assert d.failure_class is FailureClass.NETWORK
    assert d.target is RepairTarget.TEST


def test_syntax_error_is_attributed_to_the_file_that_failed_to_parse():
    d = diagnose(
        ['File "demo_test.py", line 3', "    assert (", "SyntaxError: invalid syntax"],
        "demo",
    )
    assert d.failure_class is FailureClass.SYNTAX_ERROR
    assert d.target is RepairTarget.TEST


def test_failing_artefact_reads_the_deepest_frame():
    log = ['File "demo_tool.py", line 9, in run', "ValueError: bad input"]
    assert failing_artefact(log, "demo") is RepairTarget.MODULE
    assert failing_artefact(['File "demo_test.py", line 2'], "demo") is RepairTarget.TEST
    assert failing_artefact(["no frames here"], "demo") is None


def test_error_signature_is_stable_across_paths_and_line_numbers():
    a = error_signature(['File "C:/tmp/x/demo_test.py", line 4', "AssertionError: nope"])
    b = error_signature(['File "/var/y/demo_test.py", line 41', "AssertionError: nope"])
    assert a == b and a


# ── the lesson corpus ────────────────────────────────────────────────────────


def test_lessons_deduplicate_and_count_recurrences(tmp_path):
    store = ForgeLessonStore(tmp_path / "lessons.jsonl")
    d = diagnose(["AssertionError: expected 'Hello'"], "demo")
    store.record_failure("demo", d, ["AssertionError: expected 'Hello'"])
    store.record_failure("other", d, ["AssertionError: expected 'Hello'"])
    lessons = store.lessons()
    assert len(lessons) == 1
    assert lessons[0].hits == 2
    assert set(lessons[0].tool_names) == {"demo", "other"}


def test_lessons_survive_a_reload(tmp_path):
    path = tmp_path / "lessons.jsonl"
    d = diagnose(["ValueError: boom"], "demo")
    ForgeLessonStore(path).record_failure("demo", d, ["ValueError: boom"])
    assert ForgeLessonStore(path).lessons()[0].failure_class == "runtime_error"


def test_resolution_damps_a_lesson_without_deleting_it(tmp_path):
    store = ForgeLessonStore(tmp_path / "lessons.jsonl")
    log = ["ValueError: boom"]
    store.record_failure("demo", diagnose(log, "demo"), log)
    before = store.lessons()[0].weight()
    store.record_resolution(log, note="widened the input guard")
    after = store.lessons()[0]
    assert after.weight() < before
    assert after.resolved_count == 1
    assert "widened the input guard" in after.lesson


def test_guidance_block_is_empty_when_nothing_was_learned(tmp_path):
    assert ForgeLessonStore(tmp_path / "lessons.jsonl").guidance_block("anything") == ""


def test_guidance_block_carries_lessons_into_the_prompt(tmp_path):
    store = ForgeLessonStore(tmp_path / "lessons.jsonl")
    log = ["AssertionError: expected exact string"]
    store.record_failure("demo", diagnose(log, "demo"), log)
    block = store.guidance_block("build a text formatter")
    assert "LESSONS FROM PREVIOUS FORGE FAILURES" in block
    assert "assertion_failed" in block


def test_lesson_store_never_raises_on_a_bad_path(tmp_path):
    # A directory where the file should be: writes must be contained, silently.
    bad = tmp_path / "blocked"
    bad.mkdir()
    store = ForgeLessonStore(bad)
    assert store.record_failure("demo", diagnose(["ValueError: x"], "demo"), ["ValueError: x"])
    assert store.guidance_block("x") == "" or True     # must simply not raise


# ── the sandbox contract gate ────────────────────────────────────────────────


def test_budget_adapts_to_imports():
    assert budget_for("import json") == 15.0
    assert budget_for("import requests") > 15.0
    # Costs are a maximum, not a sum.
    assert budget_for("import requests\nimport numpy") == budget_for("import requests")


def test_conformance_gate_rejects_schema_run_mismatch(tmp_path):
    """A schema advertising a parameter run() cannot accept used to pass
    verification and only explode when the dispatcher first called it."""
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    module = (
        "def get_tool_schema():\n"
        "    return {'name': 'mismatch', 'description': 'broken',\n"
        "            'parameters': {'type': 'object',\n"
        "                           'properties': {'url': {'type': 'string'}},\n"
        "                           'required': ['url']}}\n\n"
        "def run(path):\n"
        "    return path\n"
    )
    outcome = asyncio.run(harness.verify_tool(module, "print('never runs')\n", "mismatch"))
    assert not outcome.passed
    assert outcome.stage == "conformance"
    joined = "\n".join(outcome.error_log)
    assert "CONTRACT:" in joined
    # And the diagnosis layer recognises it as a contract fault.
    assert diagnose(outcome.error_log, "mismatch").failure_class is (
        FailureClass.CONTRACT_VIOLATION)


def test_conformance_gate_passes_a_well_formed_tool(tmp_path):
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    test = (
        "import demo_tool\n"
        "assert demo_tool.run() == 'ok'\n"
        "print('behaviour OK')\n"
    )
    outcome = asyncio.run(harness.verify_tool(_MODULE, test, "demo"))
    assert outcome.passed, outcome.error_log
    assert outcome.stage == "test"
    assert outcome.schema is not None and outcome.schema["name"] == "demo"


def test_unicode_output_does_not_fail_verification(tmp_path):
    """On Windows the child used to inherit a cp1252 stdout, so a test printing
    a tick died with UnicodeEncodeError — a harness defect the old pipeline
    reported to the model as a bug in its code."""
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    outcome = asyncio.run(harness.verify_tool(_MODULE, "print('✓ passed — ok')\n", "demo"))
    assert outcome.passed, outcome.error_log


def test_timeout_is_reported_as_a_timeout(tmp_path):
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    outcome = asyncio.run(
        harness.verify_tool(_MODULE, "import time\ntime.sleep(30)\n", "demo",
                            timeout_seconds=2.0))
    assert not outcome.passed
    assert outcome.timed_out
    assert diagnose(outcome.error_log, "demo",
                    timed_out=outcome.timed_out).failure_class is FailureClass.TIMEOUT


def test_runaway_memory_allocation_is_killed_and_reported(tmp_path, monkeypatch):
    """A forged test that crosses the memory limit must be killed well before
    the wall-clock timeout, and reported distinctly from a plain timeout so
    the diagnosis isn't 'give it more time' when the real issue is memory.

    The RSS sampler itself is monkeypatched to report an over-limit reading
    for a real (but otherwise idle) subprocess — live OS memory accounting
    for a genuinely memory-hungry child proved unreliable to assert against
    in this environment (verified independently via psutil and PowerShell's
    Get-Process: WorkingSet stayed flat for a process provably still
    allocating), so this isolates the kill-on-exceeded DECISION from
    environment-dependent memory measurement, the same way
    test_timeout_is_reported_as_a_timeout isolates the timeout decision with
    a real, deterministic sleep."""
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    monkeypatch.setattr(harness, "_rss_bytes", lambda pid: 999 * 1024 * 1024)
    outcome = asyncio.run(
        harness.verify_tool(_MODULE, "import time\ntime.sleep(30)\n", "demo",
                            timeout_seconds=20.0))
    assert not outcome.passed
    assert outcome.memory_exceeded
    assert not outcome.timed_out
    assert outcome.duration_ms < 5000  # killed on the next poll, not the 20s timeout
    assert any("memory limit" in line.lower() for line in outcome.error_log)


# ── the orchestration loop ───────────────────────────────────────────────────


class _Outcome:
    def __init__(self, passed, errors=(), timed_out=False):
        self.passed = passed
        self.error_log = list(errors)
        self.timed_out = timed_out


class _DepOutcome:
    def __init__(self, ok=True):
        self.succeeded = ok
        self.installed_packages = []
        self.error_log = []


class _LoadOutcome:
    def __init__(self, ok=True):
        self.succeeded = ok
        self.error_log = []


def _forge(sandbox_results, *, repairer=None, lessons=None, tmp_path=None,
           monkeypatch=None):
    forge = ForgeOrchestrationManager(
        _StubBus(),
        code_generator=lambda name, plan: (_MODULE, "assert True\n", []),
        llm_repairer=repairer,
        lessons=lessons,
    )
    results = list(sandbox_results)

    class _Sandbox:
        async def verify_tool(self, code, test, name):
            return results.pop(0)

    class _Resolver:
        def __init__(self):
            self.installed = []

        async def resolve_and_install(self, requirements):
            self.installed.append(list(requirements))
            return _DepOutcome(True)

    class _Loader:
        async def load_and_register(self, path):
            return _LoadOutcome(True)

        async def activate_tool(self, outcome, dispatcher):
            from orion_core.data import ToolResult
            return ToolResult("active", ok=True)

    forge.sandbox = _Sandbox()
    forge.resolver = _Resolver()
    forge.loader = _Loader()
    forge.dispatcher = object()
    if tmp_path is not None and monkeypatch is not None:
        import orion_core.forge as forge_mod
        (tmp_path / "orion_core").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(forge_mod, "CONFIG_DIR", tmp_path / "config")
    return forge


def test_a_broken_test_is_repaired_instead_of_the_module(tmp_path, monkeypatch):
    """The headline fix: an over-specific assertion is a TEST defect, and the
    repair must rewrite the test while leaving the verified module alone."""
    seen: list = []

    async def repairer(tool_code, test_code, diagnosis, error_log):
        seen.append(diagnosis)
        return ForgeArtefacts(tool_code="", test_code="assert True  # relaxed\n")

    forge = _forge(
        [_Outcome(False, ['File "demo_test.py", line 2', "AssertionError: expected 'Hi'"]),
         _Outcome(True)],
        repairer=repairer, tmp_path=tmp_path, monkeypatch=monkeypatch,
    )
    result = asyncio.run(forge.forge_tool("demo", "a demo tool"))
    assert result.ok, result.text
    session = list(forge.sessions.values())[0]
    assert seen and seen[0].failure_class is FailureClass.ASSERTION_FAILED
    assert session.test_code == "assert True  # relaxed\n"     # the test changed
    assert session.tool_code == _MODULE                        # the module did not
    assert session.sandbox_attempts == 2   # reports the attempt that SUCCEEDED


def test_repair_aimed_at_the_test_cannot_clobber_the_module(tmp_path, monkeypatch):
    async def overreaching_repairer(tool_code, test_code, diagnosis, error_log):
        # A repairer that returns BOTH files for a test-only diagnosis must not
        # be allowed to swap out a module that just passed its contract check.
        return ForgeArtefacts(tool_code="# junk\n", test_code="assert True\n")

    forge = _forge(
        [_Outcome(False, ["ModuleNotFoundError: No module named 'demo'"]),
         _Outcome(True)],
        repairer=overreaching_repairer, tmp_path=tmp_path, monkeypatch=monkeypatch,
    )
    assert asyncio.run(forge.forge_tool("demo", "a demo tool")).ok
    assert list(forge.sessions.values())[0].tool_code == _MODULE


def test_repairer_may_return_a_plain_pair(tmp_path, monkeypatch):
    async def pair_repairer(tool_code, test_code, diagnosis, error_log):
        return (_MODULE.replace("'ok'", "'fixed module'"), "assert True\n")

    forge = _forge(
        [_Outcome(False, ["ValueError: boom"]), _Outcome(True)],
        repairer=pair_repairer, tmp_path=tmp_path, monkeypatch=monkeypatch,
    )
    assert asyncio.run(forge.forge_tool("demo", "a demo tool")).ok
    assert "fixed module" in list(forge.sessions.values())[0].tool_code


def test_repairer_receives_the_diagnosis_and_directive():
    captured: list = []

    async def repairer(tool_code, test_code, diagnosis, error_log):
        captured.append(error_log)
        return ForgeArtefacts(tool_code=_MODULE, test_code="assert True\n")

    forge = _forge([_Outcome(False, ["ValueError: boom"])] * 6, repairer=repairer)
    asyncio.run(forge.forge_tool("demo", "a demo tool"))
    joined = "\n".join(captured[0])
    assert "DIAGNOSIS: runtime_error" in joined
    assert "REPAIR TARGET:" in joined
    assert "DIRECTIVE:" in joined


def test_session_stops_immediately_when_no_repairer_is_configured():
    forge = _forge([_Outcome(False, ["ValueError: boom"])] * 6)
    result = asyncio.run(forge.forge_tool("demo", "a demo tool"))
    assert not result.ok
    assert list(forge.sessions.values())[0].sandbox_attempts == 1


def test_failure_report_leads_with_the_diagnosis_not_raw_stderr():
    forge = _forge([_Outcome(False, ["Exit code: 1", "AssertionError: nope"])] * 6)
    result = asyncio.run(forge.forge_tool("demo", "a demo tool"))
    assert not result.ok
    assert "What went wrong:" in result.text
    assert "assertion_failed" in result.text


def test_lessons_are_recorded_on_failure_and_resolved_on_success(tmp_path, monkeypatch):
    store = ForgeLessonStore(tmp_path / "lessons.jsonl")

    async def repairer(tool_code, test_code, diagnosis, error_log):
        return ForgeArtefacts(tool_code=_MODULE, test_code="assert True\n")

    forge = _forge(
        [_Outcome(False, ["AssertionError: expected 'Hi'"]), _Outcome(True)],
        repairer=repairer, lessons=store, tmp_path=tmp_path, monkeypatch=monkeypatch,
    )
    assert asyncio.run(forge.forge_tool("demo", "a demo tool")).ok
    lessons = store.lessons()
    assert len(lessons) == 1
    assert lessons[0].hits == 1
    assert lessons[0].resolved_count == 1      # the repair worked; damp the warning


def test_missing_test_code_is_synthesised(tmp_path, monkeypatch):
    forge = _forge([_Outcome(True)], tmp_path=tmp_path, monkeypatch=monkeypatch)
    forge.code_generator = lambda name, plan: (_MODULE, "", [])
    assert asyncio.run(forge.forge_tool("demo", "a demo tool")).ok
    assert "importlib" in list(forge.sessions.values())[0].test_code


def test_generator_may_return_artefacts_directly(tmp_path, monkeypatch):
    forge = _forge([_Outcome(True)], tmp_path=tmp_path, monkeypatch=monkeypatch)
    forge.code_generator = lambda name, plan: ForgeArtefacts(
        tool_code=_MODULE, test_code="assert True\n", requirements=["httpx"],
        transport="blocks")
    assert asyncio.run(forge.forge_tool("demo", "a demo tool")).ok
    session = list(forge.sessions.values())[0]
    assert session.requirements == ["httpx"]
    assert session.transport == "blocks"


# ── the brain ────────────────────────────────────────────────────────────────


class _BlockRouter:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def generate_text(self, prompt, instruction="", **_kw):
        self.prompts.append(prompt)
        return object(), self.responses.pop(0)


def test_blocks_turn_retries_once_on_an_unusable_response():
    router = _BlockRouter(
        "Sorry, I cannot do that.",
        f"=== TOOL ===\n```python\n{_MODULE}```\n=== TEST ===\n```python\nassert 1\n```\n",
    )
    artefacts = asyncio.run(blocks_turn(router, "build it"))
    assert artefacts.complete
    assert len(router.prompts) == 2
    assert "DID NOT CONTAIN A USABLE MODULE" in router.prompts[1]


def test_generate_injects_lessons_into_the_prompt(tmp_path):
    store = ForgeLessonStore(tmp_path / "lessons.jsonl")
    log = ["AssertionError: expected exact string"]
    store.record_failure("old_tool", diagnose(log, "old_tool"), log)
    router = _BlockRouter(
        f"=== TOOL ===\n```python\n{_MODULE}```\n=== TEST ===\n```python\nassert 1\n```\n")
    brain = LlmForgeBrain(_StubBus(), router, lessons=store)
    asyncio.run(brain.generate("demo", "a demo tool"))
    assert "LESSONS FROM PREVIOUS FORGE FAILURES" in router.prompts[0]


def test_repair_asks_only_for_the_sections_it_needs():
    router = _BlockRouter("=== TEST ===\n```python\nassert True\n```\n")
    brain = LlmForgeBrain(_StubBus(), router)
    d = diagnose(["ModuleNotFoundError: No module named 'demo'"], "demo")
    artefacts = asyncio.run(brain.repair(_MODULE, "old test", d, ["boom"]))
    prompt = router.prompts[0]
    assert "=== TEST ===" in prompt
    assert "=== TOOL ===" not in prompt.split("Return ONLY")[-1]
    # An untouched section falls back to what was already there.
    assert artefacts.tool_code == _MODULE


def test_generate_synthesises_a_missing_test():
    router = _BlockRouter(f"=== TOOL ===\n```python\n{_MODULE}```\n")
    brain = LlmForgeBrain(_StubBus(), router)
    artefacts = asyncio.run(brain.generate("demo", "a demo tool"))
    assert "importlib" in artefacts.test_code


def test_generate_raises_when_no_module_is_returned():
    router = _BlockRouter("nothing useful", "still nothing")
    brain = LlmForgeBrain(_StubBus(), router)
    try:
        asyncio.run(brain.generate("demo", "a demo tool"))
    except RuntimeError as exc:
        assert "no tool module" in str(exc)
    else:                                          # pragma: no cover
        raise AssertionError("expected a RuntimeError")
