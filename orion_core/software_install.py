"""Installing software, with the owner's approval (Mark XXXX.4).

On 27 Oct the owner asked ORION to install a push-notification app and he had
to say he could not: nothing in him installed anything. This is that ability,
through winget - Windows' own package manager - and only ever on the owner's
click:

* ``search`` reads the winget community repository (no terms accepted).
* ``install`` never installs. It shows the owner the exact package - name, id,
  version, publisher's repository - in ORION's confirmation dialog, bound to a
  single-use token (system_guard.SOFTWARE_INSTALL). Only the owner's click
  releases it; accepting the package's licence is part of what that click
  approves, and the dialog says so. A package that needs administrator rights
  raises Windows' own UAC prompt as well.
* From the phone or a background job it is refused outright
  (remote_capability FORBID, execution_policy's foreground rule).

The package id is checked against winget's own id shape and passed as one
argument - never through a shell.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

#: winget ids: Publisher.Package with letters, digits, dots, dashes, underscores, plus.
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]{1,127}$")
SEARCH_TIMEOUT_S = 60.0
INSTALL_TIMEOUT_S = 30 * 60.0
MAX_RESULTS = 8
SOURCE = "winget"

_results: list[dict[str, Any]] = []
_lock = threading.Lock()


@dataclass(frozen=True)
class Package:
    name: str
    id: str
    version: str

    def line(self) -> str:
        return f"{self.name} ({self.id}{', ' + self.version if self.version else ''})"


def winget() -> str | None:
    return shutil.which("winget")


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    exe = winget()
    if exe is None:
        raise FileNotFoundError("winget (the Windows Package Manager) isn't on this PC")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, creationflags=flags)


def parse_table(output: str) -> list[Package]:
    """winget's fixed-width table (Name, Id, Version, ...) as packages."""
    lines = [line.rstrip("\r") for line in str(output or "").splitlines()]
    header_at = next((i for i, line in enumerate(lines)
                      if re.match(r"^\s*Name\s+Id\s+Version", line)), None)
    if header_at is None:
        return []
    header = lines[header_at]
    starts = [header.index(word) for word in ("Name", "Id", "Version")]
    after = [m.start() for m in re.finditer(r"\S+", header) if m.start() > starts[2]]
    end_version = after[0] if after else None
    found = []
    for line in lines[header_at + 1:]:
        if not line.strip() or set(line.strip()) <= {"-"}:
            continue
        name = line[:starts[1]].strip()
        package_id = line[starts[1]:starts[2]].strip()
        version = (line[starts[2]:end_version] if end_version else line[starts[2]:]).strip()
        if ID_RE.match(package_id):
            found.append(Package(name, package_id, version))
    return found


def search(query: str) -> list[Package]:
    """Packages in the winget repository that match *query* (blocking)."""
    query = str(query or "").strip()[:80]
    if not query:
        return []
    done = _run(["search", "--query", query, "--source", SOURCE, "--disable-interactivity"],
                SEARCH_TIMEOUT_S)
    return parse_table(done.stdout)[:MAX_RESULTS]


def resolve(target: str) -> tuple[Package | None, list[Package]]:
    """(the one package *target* names, the candidates). An exact id, or a
    search with a single match / an exact name match, names one."""
    target = str(target or "").strip()
    found = search(target)
    exact = [p for p in found if p.id.lower() == target.lower() or p.name.lower() == target.lower()]
    if len(exact) == 1:
        return exact[0], found
    if len(found) == 1:
        return found[0], found
    return None, found


def install(package_id: str) -> tuple[bool, str]:
    """Install one package by exact id (blocking; only after approval)."""
    if not ID_RE.match(str(package_id or "")):
        return False, "That isn't a valid package id."
    try:
        done = _run(["install", "--id", package_id, "--exact", "--source", SOURCE, "--silent",
                     "--accept-package-agreements", "--accept-source-agreements",
                     "--disable-interactivity"], INSTALL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, f"{package_id} was still installing after {INSTALL_TIMEOUT_S / 60:.0f} minutes."
    except (OSError, FileNotFoundError) as exc:
        return False, f"winget could not run: {exc}"
    text = " ".join((done.stdout or "").split())[-400:]
    ok = done.returncode == 0 or "Successfully installed" in (done.stdout or "")
    if "No newer package versions" in (done.stdout or "") or "already installed" in text.lower():
        return True, f"{package_id} is already installed and up to date."
    if ok:
        return True, f"Installed {package_id}."
    return False, f"winget could not install {package_id} (exit {done.returncode}): {text[-200:]}"


def note_result(package: str, ok: bool, text: str) -> None:
    with _lock:
        _results.append({"at": time.time(), "package": package, "ok": ok, "text": text})
        del _results[:-10]


def recent_results() -> list[dict[str, Any]]:
    with _lock:
        return list(_results)


__all__ = ["ID_RE", "Package", "install", "note_result", "parse_table", "recent_results",
           "resolve", "search", "winget"]
