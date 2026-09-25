"""
docker_control.py — real Docker CLI integration (list/start/stop/restart/
remove containers, list images, tail logs).

Shells out to the `docker` CLI rather than depending on the `docker` Python
SDK: the CLI is what a Docker Desktop / Engine install already provides, so
this needs no extra pip dependency, and every call goes through the same
allow-listed-subprocess discipline as dev_workbench's run_command (fixed
argument list, no shell=True, bounded timeout). Degrades cleanly when
Docker isn't installed or the daemon isn't running — same "detect via
shutil.which, never assume" pattern as security_recon.py's nmap/scapy
checks.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from .data import ToolResult

_TIMEOUT = 20


def is_available() -> bool:
    return shutil.which("docker") is not None


def _run(args: list[str], timeout: int = _TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, shell=False,
    )


def daemon_running() -> bool:
    if not is_available():
        return False
    try:
        completed = _run(["info", "--format", "{{.ServerVersion}}"], timeout=8)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def _parse_json_lines(stdout: str) -> list[dict]:
    items: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def list_containers(show_all: bool = True) -> list[dict]:
    if not is_available():
        return []
    args = ["ps", "--format", "{{json .}}"]
    if show_all:
        args.insert(1, "-a")
    try:
        completed = _run(args)
    except (subprocess.TimeoutExpired, OSError):
        return []
    if completed.returncode != 0:
        return []
    return _parse_json_lines(completed.stdout)


def list_images() -> list[dict]:
    if not is_available():
        return []
    try:
        completed = _run(["images", "--format", "{{json .}}"])
    except (subprocess.TimeoutExpired, OSError):
        return []
    if completed.returncode != 0:
        return []
    return _parse_json_lines(completed.stdout)


def _lifecycle(action: str, name_or_id: str) -> ToolResult:
    name_or_id = name_or_id.strip()
    if not name_or_id:
        return ToolResult(f"No container name/ID supplied for {action}.", ok=False)
    if not is_available():
        return ToolResult("Docker is not installed (or not on PATH).", ok=False)
    try:
        completed = _run([action, name_or_id])
    except subprocess.TimeoutExpired:
        return ToolResult(f"docker {action} timed out.", ok=False)
    except OSError as exc:
        return ToolResult(f"Could not run docker {action}: {exc}", ok=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[:400]
        return ToolResult(f"docker {action} {name_or_id} failed: {detail}", ok=False)
    return ToolResult(f"docker {action} {name_or_id}: OK")


def start_container(name_or_id: str) -> ToolResult:
    return _lifecycle("start", name_or_id)


def stop_container(name_or_id: str) -> ToolResult:
    return _lifecycle("stop", name_or_id)


def restart_container(name_or_id: str) -> ToolResult:
    return _lifecycle("restart", name_or_id)


def remove_container(name_or_id: str) -> ToolResult:
    return _lifecycle("rm", name_or_id)


def container_logs(name_or_id: str, tail: int = 100) -> ToolResult:
    name_or_id = name_or_id.strip()
    if not name_or_id:
        return ToolResult("No container name/ID supplied.", ok=False)
    if not is_available():
        return ToolResult("Docker is not installed (or not on PATH).", ok=False)
    tail = max(1, min(2000, tail))
    try:
        completed = _run(["logs", "--tail", str(tail), name_or_id])
    except subprocess.TimeoutExpired:
        return ToolResult("docker logs timed out.", ok=False)
    except OSError as exc:
        return ToolResult(f"Could not run docker logs: {exc}", ok=False)
    text = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0 and not text.strip():
        return ToolResult(f"docker logs {name_or_id} failed.", ok=False)
    return ToolResult(text.strip()[-8000:] or "(no log output)")


def describe() -> str:
    if not is_available():
        return "Docker CLI not found on PATH — install Docker Desktop/Engine to enable this panel."
    if not daemon_running():
        return "Docker CLI found, but the daemon is not responding — is Docker Desktop running?"
    return "Docker is available and the daemon is responding."
