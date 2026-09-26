"""
Thinking that costs nothing.

ORION reflects every five minutes, for as long as he is running. That is by
design — the inner monologue is most of what makes him feel present rather
than transactional — and the router has always *preferred* a local model for
it. But preferring is not requiring: when no local model is running it falls
through to a paid API, and a thought every five minutes on a paid model is a
bill that arrives without anyone having asked for anything.

Measured on this machine: Ollama installed at
``%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe`` with ``llama3.1`` already
downloaded, and not running. So every thought ORION had was billed, while a
perfectly good model sat on the disk doing nothing.

Three things, in order of how much they help
--------------------------------------------
1. **Use a local model if one is running.** Free in tokens. ORION does NOT
   start it himself: a local model is cheap in money and expensive in CPU —
   it will sit on several cores and make a laptop hot and loud, which is a
   worse problem than the one it solves. ``ORION_START_OLLAMA=1`` if the
   machine can afford it.
2. **Never fall through.** With local-only thinking on — the default — a
   thought that cannot be had locally is simply not had. So with no local
   model running, ORION does not muse in the background at all: no tokens,
   no CPU. He still answers everything you ask exactly as before. Silence
   costs nothing and a missing reflection is invisible; a surprise bill is
   not.
3. **Reading thoughts is always free.** They are already written to a journal.
   Looking at them should never have cost anything and now demonstrably does
   not, because the viewer reads the file.

What this deliberately does not do
----------------------------------
It does not make ORION's *conversation* local. When you ask him something he
uses the best model available, because that is the part you are actually
waiting on. This is about the thinking he does when nobody asked.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

#: Where Ollama listens. Not configurable here on purpose — this is the
#: address its own default install uses, and a different one is a sign the
#: user has a setup this module should not be second-guessing.
OLLAMA_HOST = "http://127.0.0.1:11434"

#: How long to wait for a freshly started Ollama to answer. Generous because
#: the first start loads a model from disk, and the alternative to waiting is
#: deciding it failed and paying for the thought instead.
START_TIMEOUT_S = 25.0

#: Places Ollama installs itself on Windows when it is not on PATH.
_WINDOWS_PATHS = (
    r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe",
    r"%LOCALAPPDATA%\Ollama\ollama.exe",
    r"%PROGRAMFILES%\Ollama\ollama.exe",
)


def local_only_thoughts() -> bool:
    """Whether ORION may spend money to think. Off by default.

    Set ``ORION_PAID_THOUGHTS=1`` to allow the inner monologue to reach a paid
    provider when no local model is available. It is deliberately awkward to
    turn on: the failure it guards against is silent and recurring, which is
    the worst combination for a cost.
    """
    return os.getenv("ORION_PAID_THOUGHTS", "").strip().lower() not in {
        "1", "true", "yes", "on"}


def ollama_binary() -> Path | None:
    """Where Ollama is, or None. Checks PATH first, then the usual places."""
    found = shutil.which("ollama")
    if found:
        return Path(found)
    if os.name == "nt":
        for template in _WINDOWS_PATHS:
            candidate = Path(os.path.expandvars(template))
            if candidate.is_file():
                return candidate
    return None


def ollama_running(timeout: float = 2.0) -> bool:
    """Whether something is answering on Ollama's port."""
    try:
        import urllib.request

        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags",
                                    timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def local_models(timeout: float = 3.0) -> list[str]:
    """Model names Ollama is currently serving."""
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags",
                                    timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    return [str(entry.get("name") or "") for entry in
            (payload.get("models") or []) if entry.get("name")]


def models_on_disk() -> list[str]:
    """Models that have been downloaded, whether or not Ollama is running.

    Read from the manifest tree rather than by asking Ollama, because the
    whole point is to know there is something worth starting it FOR.
    """
    root = Path(os.path.expanduser("~")) / ".ollama" / "models" / "manifests"
    if not root.is_dir():
        return []
    found: list[str] = []
    for manifest in root.rglob("*"):
        if manifest.is_file():
            # …/manifests/registry.ollama.ai/library/llama3.1/latest
            found.append(f"{manifest.parent.name}:{manifest.name}")
    return sorted(set(found))


def start_ollama(wait: float = START_TIMEOUT_S) -> tuple[bool, str]:
    """Start Ollama if it is installed and not already running.

    Returns (running, explanation). Never raises, and never blocks longer than
    *wait* — a slow local model must not hold up ORION's own start-up.
    """
    if ollama_running():
        return True, "already running"

    binary = ollama_binary()
    if binary is None:
        return False, ("Ollama is not installed. Without it every thought "
                       "ORION has goes to a paid provider.")
    if not models_on_disk():
        return False, (f"Ollama is installed but no model has been "
                       f"downloaded. Run: {binary.name} pull llama3.1")

    try:
        creation = 0
        if os.name == "nt":
            creation = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        | getattr(subprocess, "DETACHED_PROCESS", 0))
        subprocess.Popen(
            [str(binary), "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, creationflags=creation, close_fds=True)
    except Exception as exc:
        return False, f"Ollama could not be started ({exc})."

    deadline = time.monotonic() + max(1.0, wait)
    while time.monotonic() < deadline:
        if ollama_running(timeout=1.0):
            return True, "started"
        time.sleep(0.5)
    return False, f"Ollama did not answer within {wait:.0f}s of being started."


def autostart_allowed() -> bool:
    """Whether ORION may start Ollama himself. Off by default.

    A local model is free in tokens and expensive in CPU — it will sit on
    several cores and make the machine hot and loud, which on a laptop is a
    worse problem than the one it solves. So ORION never starts it uninvited,
    even though doing so would save money.

    ``ORION_START_OLLAMA=1`` if the machine can afford it.
    """
    return os.getenv("ORION_START_OLLAMA", "").strip().lower() in {
        "1", "true", "yes", "on"}


def ensure_local_mind(bus: Any = None) -> tuple[bool, str]:
    """Report what free thinking is available, and start it only if allowed.

    Called during start-up, off the critical path. Reported either way,
    because "ORION is thinking for free", "ORION is not thinking at all" and
    "ORION is about to bill you every five minutes" are three different
    situations that must not look the same in the log.
    """
    try:
        if ollama_running():
            running, detail = True, "already running"
        elif autostart_allowed():
            running, detail = start_ollama()
        else:
            # Deliberately not started. It is free in tokens and costly in
            # CPU, and an assistant that quietly pins several cores on a
            # laptop has solved the wrong problem.
            running = False
            detail = ("a local model is installed but not running, and ORION "
                      "does not start it uninvited because it is heavy on the "
                      "CPU (ORION_START_OLLAMA=1 to allow it).")
    except Exception as exc:                # pragma: no cover - defensive
        running, detail = False, f"the check itself failed ({exc})"

    if running:
        names = local_models()
        message = (f"local mind ready — {', '.join(names[:3]) or 'a model'} "
                   f"({detail}). Thinking costs nothing.")
    elif local_only_thoughts():
        message = (f"background thinking is OFF — {detail} No tokens are "
                   f"spent and no CPU is used; ORION still answers you "
                   f"normally, he just does not muse to himself.")
    else:
        message = (f"no local model: {detail} ORION_PAID_THOUGHTS is set, so "
                   f"reflections WILL be billed.")

    if bus is not None:
        try:
            bus.log.emit(f"[Mind] {message}")
        except Exception:
            pass
    return running, message


__all__ = [
    "OLLAMA_HOST", "START_TIMEOUT_S",
    "autostart_allowed", "ensure_local_mind", "local_models",
    "local_only_thoughts",
    "models_on_disk", "ollama_binary", "ollama_running", "start_ollama",
]
