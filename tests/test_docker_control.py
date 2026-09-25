"""
Tests for docker_control.py — CLI-shelling Docker integration for the
Development workspace's Docker panel. Nothing here touches a real Docker
install: subprocess.run and shutil.which are monkeypatched throughout, so
these tests are hermetic and fast.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import docker_control as dc


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_is_available_false_when_docker_not_on_path(monkeypatch):
    monkeypatch.setattr(dc.shutil, "which", lambda name: None)
    assert dc.is_available() is False


def test_is_available_true_when_docker_on_path(monkeypatch):
    monkeypatch.setattr(dc.shutil, "which", lambda name: "C:/docker/docker.exe")
    assert dc.is_available() is True


def test_list_containers_returns_empty_without_docker(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: False)
    assert dc.list_containers() == []


def test_list_containers_parses_json_lines(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    payload = (
        '{"ID": "abc123", "Names": "web", "Status": "Up 2 hours"}\n'
        '{"ID": "def456", "Names": "db", "Status": "Exited (0)"}\n'
    )
    monkeypatch.setattr(dc, "_run", lambda args, timeout=dc._TIMEOUT: _Completed(0, payload))
    containers = dc.list_containers()
    assert len(containers) == 2
    assert containers[0]["Names"] == "web"
    assert containers[1]["ID"] == "def456"


def test_list_containers_skips_malformed_json_lines(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    payload = '{"ID": "abc123"}\nnot json\n'
    monkeypatch.setattr(dc, "_run", lambda args, timeout=dc._TIMEOUT: _Completed(0, payload))
    containers = dc.list_containers()
    assert len(containers) == 1


def test_list_containers_returns_empty_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    monkeypatch.setattr(dc, "_run", lambda args, timeout=dc._TIMEOUT: _Completed(1, "", "boom"))
    assert dc.list_containers() == []


def test_list_containers_returns_empty_on_timeout(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)

    def _boom(args, timeout=dc._TIMEOUT):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=timeout)

    monkeypatch.setattr(dc, "_run", _boom)
    assert dc.list_containers() == []


def test_list_images_parses_json_lines(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    payload = '{"Repository": "nginx", "Tag": "latest", "Size": "142MB"}\n'
    monkeypatch.setattr(dc, "_run", lambda args, timeout=dc._TIMEOUT: _Completed(0, payload))
    images = dc.list_images()
    assert images == [{"Repository": "nginx", "Tag": "latest", "Size": "142MB"}]


def test_lifecycle_actions_require_a_name(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    result = dc.start_container("   ")
    assert not result.ok


def test_lifecycle_actions_report_docker_unavailable(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: False)
    result = dc.stop_container("web")
    assert not result.ok
    assert "not installed" in result.text.lower()


def test_start_container_success(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    calls = []

    def _run(args, timeout=dc._TIMEOUT):
        calls.append(args)
        return _Completed(0, "web\n")

    monkeypatch.setattr(dc, "_run", _run)
    result = dc.start_container("web")
    assert result.ok
    assert calls == [["start", "web"]]


def test_stop_container_failure_surfaces_stderr(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    monkeypatch.setattr(
        dc, "_run", lambda args, timeout=dc._TIMEOUT: _Completed(1, "", "no such container"))
    result = dc.stop_container("ghost")
    assert not result.ok
    assert "no such container" in result.text


def test_restart_and_remove_call_the_right_docker_subcommand(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(
        dc, "_run", lambda args, timeout=dc._TIMEOUT: calls.append(args) or _Completed(0))
    dc.restart_container("web")
    dc.remove_container("web")
    assert calls == [["restart", "web"], ["rm", "web"]]


def test_lifecycle_action_times_out_cleanly(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)

    def _boom(args, timeout=dc._TIMEOUT):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=timeout)

    monkeypatch.setattr(dc, "_run", _boom)
    result = dc.start_container("web")
    assert not result.ok
    assert "timed out" in result.text.lower()


def test_container_logs_requires_a_name(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    result = dc.container_logs("")
    assert not result.ok


def test_container_logs_returns_trimmed_output(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    monkeypatch.setattr(
        dc, "_run", lambda args, timeout=dc._TIMEOUT: _Completed(0, "line one\nline two\n"))
    result = dc.container_logs("web", tail=50)
    assert result.ok
    assert "line one" in result.text


def test_container_logs_clamps_tail_range(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(
        dc, "_run", lambda args, timeout=dc._TIMEOUT: calls.append(args) or _Completed(0, "x"))
    dc.container_logs("web", tail=999999)
    assert calls[0] == ["logs", "--tail", "2000", "web"]


def test_daemon_running_false_without_docker(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: False)
    assert dc.daemon_running() is False


def test_daemon_running_true_when_info_succeeds(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    monkeypatch.setattr(dc, "_run", lambda args, timeout=8: _Completed(0, "24.0.5"))
    assert dc.daemon_running() is True


def test_describe_reports_not_installed(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: False)
    assert "not found" in dc.describe().lower()


def test_describe_reports_daemon_not_responding(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    monkeypatch.setattr(dc, "daemon_running", lambda: False)
    assert "daemon" in dc.describe().lower()


def test_describe_reports_ready(monkeypatch):
    monkeypatch.setattr(dc, "is_available", lambda: True)
    monkeypatch.setattr(dc, "daemon_running", lambda: True)
    assert "available" in dc.describe().lower()
