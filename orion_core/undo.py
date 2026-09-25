"""
One shared undo stack for everything ORION does that changes state.

Why this exists
---------------
A voice assistant mishears. When it does, "sorry" is not a remedy: the file has
already been overwritten and the folder has already been created. Before this
module the only recovery was to fix it by hand, and ORION's ``file_controller``
could write, append, create and delete with no way back at all.

The alternative — asking "are you sure?" before everything — is worse. Every
confirmation costs a round trip to the model and back, and an assistant that
checks with you before creating a folder is one you stop talking to. So: act
immediately, remember how to reverse it, and let the user say "undo".
Confirmation is then reserved for the few actions that genuinely cannot be
reversed, which is a decision about REVERSIBILITY rather than about how
alarming the word sounds.

How an action opts in
---------------------
Callers do not need to know anything about this file's internals. They capture
the "before" state and hand back a zero-argument callable::

    from .undo import push_undo

    previous = path.read_text() if path.exists() else None
    path.write_text(new_text)
    push_undo(f"wrote {path.name}", lambda: _restore(path, previous))

Only the action itself can know that the reverse of "move A to B" is "move B to
A", which is why this cannot be centralised any further. What IS central is the
stack, the ordering, the thread safety and the depth limit.

Three things it deliberately does not do
----------------------------------------
* **It does not guess.** Where the previous state cannot be read, nothing is
  registered. An undo that restores a guess is worse than no undo, because the
  user believes the original is back.
* **It does not hoard.** Undoing a write means holding the old contents in
  memory, so oversized files are excluded and say so rather than ORION quietly
  keeping a 200 MB log alive for the session.
* **It does not delete the user's files to tidy up.** The reverse of "create a
  folder" is removing it only while it is still empty; if something has been
  put inside it since, the folder stays.

Cost
----
Pushing is a list append behind a lock: microseconds. Nothing registered here
runs unless the user actually asks for it, so this module cannot make ORION
slower.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

#: How many reversible operations to keep. Ten is roughly "this conversation" —
#: far enough back to catch a mistake noticed a few commands later, short enough
#: that a closure holding a file's old contents cannot pile up in memory.
MAX_DEPTH = 10

#: Largest file whose contents are held for an undo. Above this the operation
#: still happens and simply is not reversible, which is stated at the time
#: rather than discovered later.
MAX_CAPTURE_BYTES = 1_048_576


@dataclass
class _Entry:
    label: str
    undo: Callable[[], str]
    at: float = field(default_factory=time.monotonic)


_stack: list[_Entry] = []
_lock = threading.Lock()


def push_undo(label: str, undo_fn: Callable[[], str]) -> None:
    """Record that *label* just happened and ``undo_fn()`` reverses it.

    Called from tool handlers, which run in executor threads — hence the lock.
    Never raises: a failed undo registration must not take down the action that
    actually succeeded.
    """
    if not callable(undo_fn):
        return
    try:
        with _lock:
            _stack.append(_Entry(label=str(label)[:160], undo=undo_fn))
            # Drop the oldest rather than refusing the newest: the recent past
            # is what people ask to undo.
            while len(_stack) > MAX_DEPTH:
                _stack.pop(0)
    except Exception:
        pass


def can_undo() -> bool:
    with _lock:
        return bool(_stack)


def peek() -> str:
    """Label of the operation ``undo_last()`` would reverse, or ""."""
    with _lock:
        return _stack[-1].label if _stack else ""


def history() -> list[str]:
    """Most recent first — for the tool's list mode and any UI panel."""
    with _lock:
        return [entry.label for entry in reversed(_stack)]


def depth() -> int:
    with _lock:
        return len(_stack)


def undo_last() -> tuple[bool, str]:
    """Reverse the most recent reversible operation. Returns (ok, message).

    The entry is popped BEFORE it runs, so a failing undo cannot be retried
    forever against a world that has already moved on — the file someone is
    trying to restore may have been deleted by something else since.
    """
    with _lock:
        entry = _stack.pop() if _stack else None

    if entry is None:
        return False, ("There is nothing to undo. I only track things I changed "
                       "myself — files I wrote, created or removed, and settings "
                       "I adjusted.")
    try:
        detail = entry.undo() or ""
    except Exception as exc:
        return False, f"Could not undo '{entry.label}': {exc}"
    return True, f"Undone: {entry.label}." + (f" {detail}" if detail else "")


def clear() -> None:
    """Forget the stack, so closures holding old file contents do not outlive
    the session. Called at shutdown."""
    with _lock:
        _stack.clear()


__all__ = ["MAX_CAPTURE_BYTES", "MAX_DEPTH", "can_undo", "clear", "depth",
           "history", "peek", "push_undo", "undo_last"]
