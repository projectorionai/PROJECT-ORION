"""Analysis trials are guarded at every execution and repair boundary."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from orion_core.dynamic_loader import ReflectiveModuleLoader
from orion_core.forge import ForgeOrchestrationManager, ForgeSession
from orion_core.forge_artefacts import ForgeArtefacts
from orion_core.forge_diagnosis import diagnose
from orion_core.forge_trial_policy import MAX_SOURCE_CHARACTERS, generation_guidance
from orion_core.sandbox import SandboxVerificationHarness, VerificationOutcome
from orion_core.security import SecuritySanitiser, SecurityViolation

TRIAL = "trial_analysis"
SOURCE = """import json
from statistics import mean
def get_tool_schema():
    return {"name": "trial_analysis", "description": "Calculate supplied figures",
            "parameters": {"type": "object", "properties": {
                "values": {"type": "array", "items": {"type": "number"}}}}}
def run(values=None, **kwargs):
    values = [] if values is None else values
    if not isinstance(values, list) or not all(isinstance(v, (int, float)) for v in values):
        raise ValueError("Supply a list of numbers")
    return json.dumps({"count": len(values), "mean": mean(values) if values else 0})
"""
TEST = """import json
from trial_analysis_tool import run
import pytest
def test_figures():
    assert json.loads(run(values=[1, 2, 3])) == {"count": 3, "mean": 2}
    assert json.loads(run(values=[]))["count"] == 0
    with pytest.raises(ValueError):
        run(values="invalid")
"""


class Signal:
    def emit(self, *args):
        pass


BUS = SimpleNamespace(log=Signal())


def guard(source, *, test=False):
    return SecuritySanitiser.guard_forged_source(
        source, "regression.trial", trial_name=TRIAL, trial_test=test
    )


def manager(tmp_path, monkeypatch, artefacts=None):
    from orion_core import forge

    monkeypatch.setattr(forge, "_custom_tools_dir", lambda: tmp_path)
    return ForgeOrchestrationManager(
        BUS, code_generator=lambda *_: artefacts or (SOURCE, TEST, [])
    )


def test_pure_analysis_and_acceptance_tests_are_permitted():
    assert guard(SOURCE) == SOURCE
    assert guard(TEST, test=True) == TEST
    assert "Approved standard-library exports" in generation_guidance(TRIAL)


def test_admitted_analysis_passes_the_real_verifier(tmp_path):
    guard(SOURCE)
    guard(TEST, test=True)
    harness = SandboxVerificationHarness(BUS)
    harness.staging_dir = tmp_path
    result = asyncio.run(harness.verify_tool(SOURCE, TEST, TRIAL))
    assert result.passed, result.error_log
    assert result.stage == "test"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "module",
    [
        "requests",
        "httpx",
        "urllib.request",
        "socket",
        "subprocess",
        "os",
        "pathlib",
        "io",
        "ctypes",
        "pyautogui",
        "pynput",
        "cv2",
        "mss",
        "orion_core.security",
        "config",
        "another_trial_tool",
        "asyncio",
        "pickle",
        "marshal",
        "importlib",
        "sys",
    ],
)
def test_io_device_core_and_dynamic_imports_are_rejected(module):
    with pytest.raises(SecurityViolation):
        guard(f"import {module}\n" + SOURCE)


@pytest.mark.parametrize(
    "prefix",
    [
        "from json import codecs\n",
        "from statistics import sys\n",
        "from math import *\n",
        "from .math import sqrt\n",
        "import json\nj = json\n",
        "import json\njson.encoder\n",
        "import json\njson.__dict__\n",
        "from json import dumps\ndumps.__globals__\n",
        "f = open\n",
        "__builtins__['open']('private.key')\n",
        "globals()['ev' + 'al']('1')\n",
        "getattr(1, '__class__')\n",
        "compile('1', 'x', 'eval')\n",
        "eval('1')\n",
        "exec('1')\n",
        "import math as os\n",
        "class Example:\n    pass\n",
        "def __getattr__(name):\n    return name\n",
        "template = '{0.__globals__}'\ntemplate.format(run)\n",
        "template = '{func.__globals__}'\ntemplate.format_map({'func': run})\n",
        "(lambda: 1)()\n",
        "functions = [run]\nfunctions[0]()\n",
        "callback = run\ncallback()\n",
        "stream.write('change')\n",
        "path.read_bytes()\n",
        "path.write_text('change')\n",
        "import json\nprint(json)\n",
        "import json\njson.dumps.json\n",
    ],
)
def test_reflection_namespace_and_callable_bypasses_are_rejected(prefix):
    with pytest.raises(SecurityViolation):
        guard(prefix + SOURCE)


@pytest.mark.parametrize(
    "test_source",
    [
        "from another_trial_tool import run\nassert run()\n",
        "from trial_analysis_tool import os\nassert True\n",
        "import pytest\npytest.main(['x'])\n",
        "import unittest\nunittest.main(module='os')\n",
        "import pathlib\npathlib.Path('settings.json').unlink()\n",
    ],
)
def test_acceptance_tests_cannot_import_capabilities(test_source):
    with pytest.raises(SecurityViolation):
        guard(test_source, test=True)


@pytest.mark.parametrize(
    "source",
    [
        "",
        "def broken(:",
        "#" * (MAX_SOURCE_CHARACTERS + 1),
        "x = [" + ",".join("0" for _ in range(12000)) + "]",
    ],
    ids=["missing", "syntax", "source_budget", "ast_budget"],
)
def test_missing_invalid_and_oversize_sources_fail_closed(source):
    with pytest.raises(SecurityViolation):
        guard(source)


def test_ordinary_forged_tool_policy_is_preserved():
    ordinary = "import pathlib\npathlib.Path('scratch.txt').unlink(missing_ok=True)\n"
    assert SecuritySanitiser.guard_forged_source(ordinary) == ordinary
    assert (
        SecuritySanitiser.guard_forged_source(ordinary, trial_name="ordinary")
        == ordinary
    )


@pytest.mark.parametrize(
    "artefacts",
    [
        ("import socket\n" + SOURCE, TEST, []),
        (SOURCE, "import os\nassert True", []),
        (SOURCE, TEST, ["requests"]),
        (SOURCE, "", []),
    ],
)
def test_generation_gate_runs_before_verification_or_install(
    tmp_path, monkeypatch, artefacts
):
    obj = manager(tmp_path, monkeypatch, artefacts)
    seen = []

    async def forbidden(*args):
        seen.append(args)
        raise AssertionError("unsafe generation reached an execution boundary")

    obj.sandbox.verify_tool = forbidden
    obj.resolver.resolve_and_install = forbidden
    result = asyncio.run(obj.forge_tool(TRIAL, "analyse supplied figures"))
    assert not result.ok
    assert seen == []
    assert not (tmp_path / (TRIAL + "_tool.py")).exists()


@pytest.mark.parametrize("repair_kind", ["module", "test", "requirements", "legacy"])
def test_repairs_are_guarded_immediately(tmp_path, monkeypatch, repair_kind):
    obj = manager(tmp_path, monkeypatch)
    session = ForgeSession("repair", TRIAL, "analyse", tool_code=SOURCE, test_code=TEST)
    if repair_kind == "test":
        error = [
            'File "trial_analysis_test.py", line 1',
            "AssertionError: test incorrect",
        ]
        obj.llm_repairer = lambda *_: ForgeArtefacts(
            tool_code=SOURCE, test_code="import subprocess\nassert True"
        )
    else:
        error = [
            'File "trial_analysis_tool.py", line 1',
            "ValueError: module incorrect",
        ]
        if repair_kind == "legacy":
            obj.llm_fixer = lambda *_: "import socket\n" + SOURCE
        else:
            obj.llm_repairer = lambda *_: ForgeArtefacts(
                tool_code="import pathlib\n" + SOURCE
                if repair_kind == "module"
                else SOURCE,
                test_code=TEST,
                requirements=["httpx"] if repair_kind == "requirements" else [],
            )
    with pytest.raises(SecurityViolation):
        asyncio.run(obj._apply_repair(session, diagnose(error, TRIAL), error))


def test_bad_repair_never_reaches_second_verification(tmp_path, monkeypatch):
    obj = manager(tmp_path, monkeypatch)
    seen = []

    async def verify(source, *_):
        seen.append(source)
        return VerificationOutcome(False, TRIAL, error_log=["ValueError: bad module"])

    obj.sandbox.verify_tool = verify
    obj.llm_fixer = lambda *_: "import socket\n" + SOURCE
    result = asyncio.run(obj.forge_tool(TRIAL, "analyse"))
    assert not result.ok
    assert seen == [SOURCE]


def test_missing_dependency_diagnosis_never_installs_for_trial(tmp_path, monkeypatch):
    obj = manager(tmp_path, monkeypatch)
    installed = []

    async def verify(*_):
        return VerificationOutcome(
            False, TRIAL, error_log=["ModuleNotFoundError: No module named 'requests'"]
        )

    async def install(args):
        installed.extend(args)
        return SimpleNamespace(succeeded=True, installed_packages=args)

    obj.sandbox.verify_tool = verify
    obj.resolver.resolve_and_install = install
    result = asyncio.run(obj.forge_tool(TRIAL, "analyse"))
    assert not result.ok
    assert installed == []


def test_mechanical_repair_is_guarded_before_retry(tmp_path, monkeypatch):
    from orion_core import forge

    obj = manager(tmp_path, monkeypatch)
    seen = []

    async def verify(*args):
        seen.append(args)
        return VerificationOutcome(
            False,
            TRIAL,
            error_log=[
                "CONTRACT: the schema declares parameter 'values', but run() does not accept it"
            ],
        )

    obj.sandbox.verify_tool = verify
    monkeypatch.setattr(
        forge,
        "autofix_contract",
        lambda *_: SimpleNamespace(
            changed=True,
            complete=True,
            code="import os\n" + SOURCE,
            summary=lambda: "repair",
        ),
    )
    result = asyncio.run(obj.forge_tool(TRIAL, "analyse"))
    assert not result.ok
    assert len(seen) == 1


def test_prepublication_guard_preserves_existing_file(tmp_path, monkeypatch):
    obj = manager(tmp_path, monkeypatch)
    path = tmp_path / (TRIAL + "_tool.py")
    path.write_text(SOURCE, encoding="utf-8")
    session = ForgeSession(
        "activation", TRIAL, "analyse", tool_code="import os\n" + SOURCE, test_code=TEST
    )
    with pytest.raises(SecurityViolation):
        asyncio.run(obj._activate_verified(session, path))
    assert path.read_text(encoding="utf-8") == SOURCE


def test_persisted_trial_is_rejected_before_import_side_effect(tmp_path):
    loader = ReflectiveModuleLoader(BUS)
    path = tmp_path / (TRIAL + "_tool.py")
    marker = tmp_path / "side_effect"
    path.write_text(
        f"open({str(marker)!r}, 'w').write('changed')\n" + SOURCE, encoding="utf-8"
    )
    previous = SimpleNamespace(marker="stable")
    sys.modules[path.stem] = previous
    try:
        with pytest.raises(SecurityViolation):
            loader._load_module_file(path)
        assert not marker.exists()
        assert sys.modules[path.stem] is previous
    finally:
        sys.modules.pop(path.stem, None)


def test_pure_persisted_trial_loads_and_calculates(tmp_path):
    loader = ReflectiveModuleLoader(BUS)
    path = tmp_path / (TRIAL + "_tool.py")
    path.write_text(SOURCE, encoding="utf-8")
    try:
        module = loader._load_module_file(path)
        import json

        assert json.loads(module.run(values=[2, 4])) == {"count": 2, "mean": 3}
    finally:
        sys.modules.pop(path.stem, None)


def test_persisted_trial_cannot_replace_another_tool(tmp_path):
    loader = ReflectiveModuleLoader(BUS)
    path = tmp_path / (TRIAL + "_tool.py")
    path.write_text(SOURCE.replace('"name": "trial_analysis"', '"name": "sentinel"'),
                    encoding="utf-8")
    try:
        outcome = asyncio.run(loader.load_and_register(path))
        assert not outcome.succeeded
        assert "reserved" in " ".join(outcome.error_log)
    finally:
        sys.modules.pop(path.stem, None)


def test_loader_compiles_the_guarded_snapshot(tmp_path, monkeypatch):
    loader = ReflectiveModuleLoader(BUS)
    path = tmp_path / (TRIAL + "_tool.py")
    path.write_text(SOURCE, encoding="utf-8")
    original = SecuritySanitiser.guard_forged_source

    def guard_then_change(source, context, **kwargs):
        result = original(source, context, **kwargs)
        path.write_text("import os\n" + SOURCE, encoding="utf-8")
        return result

    monkeypatch.setattr(SecuritySanitiser, "guard_forged_source", guard_then_change)
    try:
        assert loader._load_module_file(path).run(values=[2, 4])
        assert path.read_text(encoding="utf-8").startswith("import os")
    finally:
        sys.modules.pop(path.stem, None)


def test_acceptance_test_requires_assertions():
    with pytest.raises(SecurityViolation):
        guard("from trial_analysis_tool import run\nprint(run(values=[1]))\n", test=True)


def test_quarantined_trial_is_guarded_without_probe_or_install(tmp_path):
    loader = ReflectiveModuleLoader(BUS)
    path = tmp_path / (TRIAL + "_tool.py.broken")
    marker = tmp_path / "side_effect"
    path.write_text(
        f"open({str(marker)!r}, 'w').write('changed')\n" + SOURCE, encoding="utf-8"
    )
    assert not asyncio.run(loader._can_rehabilitate(path))
    assert not marker.exists()
