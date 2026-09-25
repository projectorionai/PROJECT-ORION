"""
Whose voice released this action — the owner check for sensitive tools.

Tools that act on the world (send a message, place a call, delete, pay) are
released by a second call carrying ``confirm``, ``consent`` or ``submit`` once
the user has agreed. Spoken, that agreement came from whoever was in the room.

With the guard on, a release that arrives through the VOICE channel is refused
when the last utterance was confidently not the owner's voice. Everything else
passes, on purpose:

* typed and on-screen requests — the keyboard is the owner's;
* an uncertain verdict, too little speech, or no recent verdict at all — the
  voiceprint check fails open everywhere else, and a false refusal here would
  teach the owner to switch the guard off;
* ordinary calls that release nothing.

This module holds no model and imports nothing heavy: the dispatcher consults
it on every tool call. Verdicts are written by the audio path, which computes
them only while a voice gate is switched on.
"""

from __future__ import annotations

import contextlib
import time
from contextvars import ContextVar
from typing import Any, Callable, Iterator

#: True while a tool call issued by the voice channel is being dispatched.
VOICE_TURN: ContextVar[bool] = ContextVar("orion_voice_turn", default=False)

#: How long an utterance's verdict speaks for the turn it started. A Live
#: tool call follows its utterance within seconds; a verdict older than this
#: belongs to an earlier conversation.
RECENT_S = 30.0

#: Arguments whose truth releases an action a first call only proposed.
RELEASE_ARGS = ("confirm", "consent", "submit")

_last: tuple[float, Any] | None = None


def note_verdict(verdict: Any, now: float | None = None) -> None:
    """Record the speaker verdict for the utterance just heard."""
    global _last
    _last = (time.monotonic() if now is None else now, verdict)


def recent_verdict(now: float | None = None, max_age_s: float = RECENT_S) -> Any | None:
    """The latest verdict, or None when nothing was judged recently."""
    record = _last
    if record is None:
        return None
    at, verdict = record
    current = time.monotonic() if now is None else now
    return verdict if current - at <= max_age_s else None


def clear() -> None:
    global _last
    _last = None


@contextlib.contextmanager
def voice_turn() -> Iterator[None]:
    """Mark the enclosed dispatch as coming from the voice channel. Tasks
    created inside inherit the mark (contextvars are copied at creation)."""
    token = VOICE_TURN.set(True)
    try:
        yield
    finally:
        VOICE_TURN.reset(token)


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    return isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}


def releases_action(args: dict[str, Any] | None) -> bool:
    """Whether this call releases something a first call only proposed."""
    return any(_truthy((args or {}).get(key)) for key in RELEASE_ARGS)


def refusal(name: str, args: dict[str, Any] | None,
            guard_on: Callable[[], bool]) -> str | None:
    """Why this call may not proceed, or None when it may.

    ``guard_on`` is consulted last, and only for a voice-channel release, so
    the setting is read from disk only when it can matter.
    """
    if not VOICE_TURN.get() or not releases_action(args):
        return None
    try:
        if not guard_on():
            return None
    except Exception:
        return None
    verdict = recent_verdict()
    if verdict is None or getattr(verdict, "is_owner", True):
        return None
    if not getattr(verdict, "confident", False):
        return None
    score = float(getattr(verdict, "score", 0.0) or 0.0)
    return (f"'{name}' needs the owner's own voice to go ahead, and the last "
            f"thing said did not sound like them (similarity {score:.2f}). "
            "Nothing was done. Ask the owner to confirm it, by voice or typed.")


__all__ = [
    "RECENT_S", "RELEASE_ARGS", "VOICE_TURN",
    "clear", "note_verdict", "recent_verdict", "refusal", "releases_action", "voice_turn",
]
