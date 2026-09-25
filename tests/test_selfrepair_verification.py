"""Verification results follow exit codes, including pytest collection errors."""
from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from orion_core.selfrepair import SelfRepairAgent


@pytest.mark.parametrize("compile_code,pytest_code", [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (2, 0),
])
def test_verification_uses_exit_codes(tmp_path, monkeypatch, compile_code, pytest_code):
    (tmp_path / "tests").mkdir()
    codes = iter([compile_code, pytest_code])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=next(codes),
                                        stdout="example mentions exit 1 and exit 0",
                                        stderr="diagnostic text"))
    agent = SelfRepairAgent.__new__(SelfRepairAgent)
    beats = []
    agent.telemetry = SimpleNamespace(health=SimpleNamespace(beat=lambda *args: beats.append(args)))
    result = asyncio.run(agent.run_tests(str(tmp_path)))
    expected = compile_code == 0 and pytest_code == 0
    assert result.ok is expected
    assert beats[-1][1] == ("OK" if expected else "DEGRADED")
    assert "diagnostic text" in result.text


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("pytest", 300)])
def test_verification_does_not_pass_when_tests_cannot_finish(tmp_path, monkeypatch, failure):
    (tmp_path / "tests").mkdir()
    def run(command, **kwargs):
        if "pytest" in command:
            raise failure
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    agent = SelfRepairAgent.__new__(SelfRepairAgent)
    agent.telemetry = None
    assert not asyncio.run(agent.run_tests(str(tmp_path))).ok


def test_compile_only_check_is_labelled_when_no_tests_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="", stderr=""))
    agent = SelfRepairAgent.__new__(SelfRepairAgent)
    agent.telemetry = None
    result = asyncio.run(agent.run_tests(str(tmp_path)))
    assert result.ok
    assert "compile smoke only" in result.text
