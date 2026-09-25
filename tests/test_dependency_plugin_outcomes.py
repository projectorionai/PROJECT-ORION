"""Dependency plugins must report observed state and use the right interpreter."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "config/custom_tools" / (name + "_tool.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_check_observes_versions_and_missing_packages(monkeypatch):
    plugin = load("dependency_resolver")
    def version(name):
        if name == "missing":
            raise plugin.metadata.PackageNotFoundError(name)
        return "1.0"
    monkeypatch.setattr(plugin.metadata, "version", version)
    result = plugin.run("sample", ["example>=2", "missing", "other>=1"])
    assert not result.ok
    assert [row["status"] for row in result.evidence] == ["version mismatch", "missing", "satisfied"]
    assert plugin.run("sample", ["example==1"]).ok
    assert not plugin.run("sample", "not a list").ok


@pytest.mark.parametrize("exit_code", [0, 1])
def test_installer_uses_current_python_one_transaction_and_timeout(tmp_path, monkeypatch, exit_code):
    plugin = load("dependency_manager")
    (tmp_path / "requirements.txt").write_text("# comment\nexample>=1\nother==2\n")
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=exit_code, stderr="sensitive output")
    monkeypatch.setattr(plugin.subprocess, "run", run)
    assert plugin.run(str(tmp_path)).ok
    assert not calls
    result = plugin.run(str(tmp_path), install=True)
    assert result.ok is (exit_code == 0)
    assert len(calls) == 1
    assert calls[0][0][:4] == [sys.executable, "-m", "pip", "install"]
    assert "-r" in calls[0][0] and calls[0][1]["timeout"] == 120
    assert "sensitive output" not in result.text


def test_installer_handles_timeout_and_frozen_runtime(tmp_path, monkeypatch):
    plugin = load("dependency_manager")
    (tmp_path / "requirements.txt").write_text("example\n")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("pip", 120)
    monkeypatch.setattr(plugin.subprocess, "run", timeout)
    assert not plugin.run(str(tmp_path), True).ok
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    result = plugin.run(str(tmp_path), True)
    assert not result.ok and "source Python" in result.text
