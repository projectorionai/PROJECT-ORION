"""The Python the Forge verifies with and installs into.

From source that is simply ``sys.executable``. In the installed app it is not:
``sys.executable`` is ORION.exe, so the sandbox refused outright ("A frozen
ORION executable is not a Python verifier") and pip ran ``ORION.exe -m pip``.
MEASURED 8-10 Oct 2026: 15 Forge sessions in the installed app, all failed that
way, each then spending four model calls "repairing" code that had never run.

So the installed app verifies with ITSELF, through an explicit mode:
``ORION.exe --forge-verify <script> [args]`` runs one sandbox script with the
bundled interpreter. That is the exact runtime a forged tool will live in (the
same Python and the same packages), which source mode could only approximate.
The sandbox's own limits (time, memory, cancellation) still wrap the process.

Packages a forged tool needs are installed by a real Python found on this PC,
into ``config/forge_packages``, as wheels for the bundle's Python version, and
that folder is put on the path of the app and of every verification. With no
Python on the PC the Forge says so plainly instead of failing obscurely.
"""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Sequence

FLAG = "--forge-verify"

_external: list[str | None] = []          # cached discovery: [path or None]


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def packages_dir() -> Path:
    from .constants import CONFIG_DIR
    return Path(CONFIG_DIR) / "forge_packages"


def add_packages_to_path() -> bool:
    """Let the installed app import what the Forge installed. True if added."""
    folder = packages_dir()
    if not frozen() or not folder.is_dir():
        return False
    entry = str(folder)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return True


def verifier_command(script: str, args: Sequence[str] = ()) -> list[str]:
    """The command that runs one sandbox script."""
    if frozen():
        return [sys.executable, FLAG, str(script), *map(str, args)]
    return [sys.executable, str(script), *map(str, args)]


def run_script(argv: Sequence[str]) -> int:
    """Entry for ``ORION.exe --forge-verify script [args]``: run it as __main__."""
    from .mcp_bundled import _ensure_stdio
    _ensure_stdio()
    if not argv:
        print("forge-verify: no script given", file=sys.stderr)
        return 2
    script = os.path.abspath(str(argv[0]))
    add_packages_to_path()
    sys.argv = [script, *map(str, argv[1:])]
    sys.path.insert(0, os.path.dirname(script))
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
    return 0


def _probe(command: list[str]) -> str | None:
    """The real interpreter path behind *command*, if it is Python 3.10+."""
    try:
        done = subprocess.run(
            [*command, "-c", "import sys; print(sys.version_info >= (3, 10)); print(sys.executable)"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (done.stdout or "").split()
    if done.returncode == 0 and len(lines) >= 2 and lines[0] == "True":
        return " ".join(lines[1:])
    return None


def external_python() -> str | None:
    """A real Python on this PC, for pip in the installed app (cached)."""
    if _external:
        return _external[0]
    found: str | None = None
    wanted = os.getenv("ORION_FORGE_PYTHON", "").strip()
    attempts: list[list[str]] = []
    if wanted:
        attempts.append([wanted])
    launcher = shutil.which("py")
    if launcher:
        attempts.append([launcher, f"-{sys.version_info[0]}.{sys.version_info[1]}"])
        attempts.append([launcher, "-3"])
    for name in ("python", "python3"):
        path = shutil.which(name)
        if path:
            attempts.append([path])
    for command in attempts:
        found = _probe(command)
        if found:
            break
    _external.append(found)
    return found


def pip_install_command(packages: Sequence[str]) -> list[str] | None:
    """How to install *packages* for forged tools; None when nothing can."""
    if not frozen():
        return [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    python = external_python()
    if python is None:
        return None
    major, minor = sys.version_info[0], sys.version_info[1]
    return [python, "-m", "pip", "install", "--quiet", "--upgrade",
            "--target", str(packages_dir()),
            # Wheels for the BUNDLE's interpreter, whatever Python runs pip.
            "--python-version", f"{major}.{minor}", "--implementation", "cp",
            "--only-binary=:all:", *packages]


__all__ = ["FLAG", "verifier_command", "run_script", "pip_install_command",
           "external_python", "packages_dir", "add_packages_to_path", "frozen"]
