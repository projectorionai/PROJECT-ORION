"""One OS lease for every trainer writing a grandmaster checkpoint directory."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any


class TrainingLease:
    """A held file descriptor, released by the OS even after a worker crashes."""

    def __init__(self, handle):
        self.handle = handle

    @classmethod
    def acquire(cls, directory: Path, *, create: bool = True):
        directory = Path(directory)
        lock = directory / "improvement.lock"
        if not create and not lock.is_file():
            return cls(None)
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        handle = lock.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return cls(handle)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return None
            raise

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _continuous_process(pid: Any, directory: Path) -> bool:
    """Verify the worker and its directory before treating a PID as an owner."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        import psutil
        process = psutil.Process(pid)
        if not process.is_running():
            return False
        command = process.cmdline()
        index = next(i for i in range(len(command) - 1)
                     if command[i] == "-m" and command[i + 1] == "tools.improve_chess")
        arguments = command[index + 2:]
        chosen = None
        for i, arg in enumerate(arguments):
            if arg == "--directory" and i + 1 < len(arguments):
                chosen = arguments[i + 1]
            elif arg.startswith("--directory="):
                chosen = arg.split("=", 1)[1]
        if chosen:
            candidate = Path(chosen)
            if not candidate.is_absolute():
                candidate = Path(process.cwd()) / candidate
        else:
            # The shipped CLI defaults to the owning checkout's config.
            candidate = Path(process.cwd()) / "config" / "grandmaster"
        return candidate.resolve() == Path(directory).resolve()
    except (OSError, ValueError, StopIteration):
        return False
    except Exception:
        # Access denied or an exiting process cannot authorise a stop request.
        return False


def external_training_status(directory: Path) -> dict | None:
    """Read a live owner only while its OS lease is demonstrably held."""
    directory = Path(directory)
    probe = TrainingLease.acquire(directory, create=False)
    if probe is not None:
        probe.close()
        return None
    try:
        state = json.loads((directory / "improvement.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if isinstance(state, dict) and _continuous_process(state.get("pid"), directory):
        phase = state.get("phase")
        descriptions = {
            "collect": "collecting complete self-play games",
            "train": "fitting a candidate network",
            "arena": "checking a candidate against the incumbent",
        }
        progress = ""
        if phase == "arena" and isinstance(state.get("arena"), dict):
            scores = state["arena"].get("scores")
            if isinstance(scores, list):
                progress = f" ({len(scores)}/80 colour pairs)"
        stopping = (directory / "improvement.stop").is_file()
        return {
            "status": "stopping" if stopping else "running",
            "reason": ("The continuous trainer is finishing and saving its current work."
                       if stopping else "Continuous training is " + descriptions.get(phase, "running")
                       + progress + ". It owns this checkpoint directory."),
            "external": True, "owner": "continuous", "pid": state["pid"],
            "phase_detail": phase,
        }
    return {
        "status": "busy", "external": True, "owner": "unknown",
        "reason": "Another trainer owns this checkpoint directory. Its identity could not be verified; a second writer will not start.",
    }
