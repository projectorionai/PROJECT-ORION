"""Source verification must never treat the GUI executable as Python's CLI."""

import asyncio
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from orion_core import selfrepair


def test_installed_selfrepair_tests_never_launch_the_gui_as_python(tmp_path, monkeypatch):
    agent = selfrepair.SelfRepairAgent.__new__(selfrepair.SelfRepairAgent)
    agent.telemetry = None
    calls = []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "ORION.exe"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    result = asyncio.run(agent.run_tests(str(tmp_path)))
    assert not result.ok
    assert "Source verification is unavailable" in result.text
    assert "no tests were run" in result.text
    assert calls == []


def test_installed_patched_tests_return_no_verdict_without_copying_or_launching(tmp_path, monkeypatch):
    agent = selfrepair.SelfRepairAgent.__new__(selfrepair.SelfRepairAgent)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    def no_discovery(*_args):
        raise AssertionError("source test discovery should not run inside the bundle")

    monkeypatch.setattr(agent, "_relevant_tests", no_discovery)
    passed, detail = agent._run_patched_tests(tmp_path / "module.py", "pass\n")
    assert passed is None
    assert "has not been tested" in detail


def test_source_suite_preserves_real_python_commands_and_failed_test_result(tmp_path, monkeypatch):
    agent = selfrepair.SelfRepairAgent.__new__(selfrepair.SelfRepairAgent)
    agent.telemetry = None
    (tmp_path / "tests").mkdir()
    package = tmp_path / "orion_core"
    package.mkdir()
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(selfrepair, "PACKAGE_DIR", package)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0 if len(calls) == 1 else 1,
                               stdout="" if len(calls) == 1 else "1 failed", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    result = asyncio.run(agent.run_tests(str(tmp_path)))
    assert calls[0][0] == [sys.executable, "-m", "compileall", "-q", str(package)]
    assert calls[1][0] == [sys.executable, "-m", "pytest", "-q", str(tmp_path / "tests")]
    assert Path(calls[1][1]["cwd"]) == tmp_path
    assert not result.ok
    assert "compileall: exit 0" in result.text and "pytest: exit 1" in result.text
