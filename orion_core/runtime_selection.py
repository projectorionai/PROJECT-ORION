"""Choose a working source runtime and avoid launching an outdated bundle."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def bundle_is_current(base: Path, executable: Path) -> bool:
    if not executable.is_file():
        return False
    try:
        built = executable.stat().st_mtime_ns
        sources = [base / "orion.py", *base.joinpath("orion_core").rglob("*.py")]
        return not any(path.is_file() and path.stat().st_mtime_ns > built for path in sources)
    except OSError:
        return False


def interpreter_can_start(executable: Path) -> bool:
    console = executable.with_name("python.exe") if executable.name.lower() == "pythonw.exe" else executable
    if not console.is_file():
        return False
    probe = ("import importlib.util; "
             "assert all(importlib.util.find_spec(m) for m in "
             "('PyQt6.QtWidgets','PyQt6.QtWebEngineWidgets','aiohttp','qasync',"
             "'sounddevice','google.genai','numpy','psutil'))")
    try:
        result = subprocess.run([str(console), "-c", probe], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=12,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def choose_interpreter(base: Path) -> str:
    candidates: list[Path] = []
    if os.getenv("ORION_PYTHON"):
        candidates.append(Path(os.environ["ORION_PYTHON"]))
    for folder in (".neural-debug-env", ".venv"):
        candidates.extend(base / folder / "Scripts" / name for name in ("pythonw.exe", "python.exe"))
    if not getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable))
    for name in ("pythonw", "python"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))
    for candidate in dict.fromkeys(candidates):
        if interpreter_can_start(candidate):
            return str(candidate)
    raise RuntimeError("No working ORION Python environment was found. Run deploy/start_orion.ps1 to see the dependency check.")


def launch_environment(base: Path) -> dict[str, str]:
    env = dict(os.environ, PYTHONUTF8="1")
    try:
        settings = json.loads((base / "config" / "remote_launch.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return env
    if not isinstance(settings, dict) or settings.get("enabled") is not True:
        return env
    exposure = settings.get("exposure")
    if exposure not in {"serve", "funnel", "quick-tunnel"}:
        return env
    # Saved preferences cannot turn the loopback listener into a public bind.
    for name, value in {"ORION_REMOTE_ACCESS": "1", "ORION_REMOTE_HOST": "127.0.0.1",
                        "ORION_REMOTE_PORT": "8000", "ORION_REMOTE_TLS": "0",
                        "ORION_REMOTE_EXPOSURE": exposure, "ORION_REMOTE_PAIR_ON_START": "1"}.items():
        env.setdefault(name, value)
    return env
