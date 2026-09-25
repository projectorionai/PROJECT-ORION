"""
WorkflowPatternDetector — notices when the same short run of tool calls keeps
repeating, so ORION can offer to save it as a named workflow.

The observation this exists to act on: a user who runs the same three tools
in the same order a dozen times has a workflow, they just haven't written it
down. WorkflowEngine has been able to run named tool chains for a long while;
nothing ever noticed that a chain was worth naming.

Two deliberate constraints:

  * **Suggests, never acts.** This module produces observations. Turning one
    into a real workflow goes through the ordinary `workflow`/`automation`
    tool the user or model already uses, with the user's say-so. Nothing here
    writes to the workflow library.

  * **Records argument KEYS, never values.** A signature is "this call had a
    `query` and a `path`", not what the query or path were. That is enough to
    tell "search then read" apart from "search then write", which is all the
    n-gram matching needs — and it means a rolling log of the user's recent
    activity never holds their actual search terms, file paths or message
    bodies. The cost is that a detected pattern is a naming hint rather than a
    replayable script: real arguments get supplied at define-time.

Pure, synchronous, in-memory: no I/O, no bus, no model calls, bounded deque.
Cheap enough to feed from the dispatcher's hot path without thinking about it.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAXLEN = 300


@dataclass(frozen=True)
class PatternSequence:
    """A run of tool calls seen repeatedly in the rolling log."""

    steps: tuple[tuple[str, str], ...]   # (tool, args-key-signature) pairs
    occurrences: int
    last_seen: float = field(default=0.0)

    @property
    def tool_names(self) -> list[str]:
        return [tool for tool, _signature in self.steps]

    def describe(self) -> str:
        # ASCII arrow deliberately: this string reaches bus.log.emit, which
        # can end up on a Windows console still running cp1252, where a "→"
        # raises UnicodeEncodeError rather than degrading.
        return " -> ".join(self.tool_names)

    def suggested_name(self) -> str:
        return "_".join(self.tool_names)[:40]


def args_signature(args: dict[str, Any] | None) -> str:
    """The argument KEYS of a call, sorted and joined — never the values.

    See the module docstring: this is what keeps a rolling activity log from
    holding the user's actual queries, paths and message bodies.
    """
    if not isinstance(args, dict):
        return ""
    return ",".join(sorted(str(key) for key in args))


class WorkflowPatternDetector:
    """A bounded rolling log of tool calls, and n-gram repetition over it."""

    def __init__(self, maxlen: int = DEFAULT_MAXLEN) -> None:
        self._log: deque[tuple[str, str, float]] = deque(maxlen=max(1, int(maxlen)))

    # ── recording ────────────────────────────────────────────────────────────

    def record(self, tool: str, args: dict[str, Any] | None = None,
               ok: bool = True) -> None:
        """Note one dispatched tool call. Failed calls are ignored — a
        sequence that errored out is not a workflow worth suggesting."""
        tool = str(tool or "").strip()
        if not tool or not ok:
            return
        self._log.append((tool, args_signature(args), time.time()))

    def __len__(self) -> int:
        return len(self._log)

    def clear(self) -> None:
        self._log.clear()

    # ── detection ────────────────────────────────────────────────────────────

    def detect_repeated_sequences(
        self,
        min_length: int = 2,
        max_length: int = 4,
        min_occurrences: int = 3,
    ) -> list[PatternSequence]:
        """Contiguous (tool, signature) runs that recur at least
        *min_occurrences* times, longest and most frequent first.

        A shorter run wholly contained in a longer kept one is dropped: if
        "A → B → C" is the real pattern, reporting "A → B" alongside it is
        noise, not a second finding.
        """
        min_length = max(1, int(min_length))
        max_length = max(min_length, int(max_length))
        min_occurrences = max(2, int(min_occurrences))

        entries = list(self._log)
        pairs = [(tool, signature) for tool, signature, _at in entries]
        timestamps = [at for _tool, _signature, at in entries]

        found: list[PatternSequence] = []
        for length in range(min_length, max_length + 1):
            if len(pairs) < length * min_occurrences:
                continue
            counts: dict[tuple, int] = {}
            latest: dict[tuple, float] = {}
            for start in range(len(pairs) - length + 1):
                window = tuple(pairs[start:start + length])
                counts[window] = counts.get(window, 0) + 1
                latest[window] = max(latest.get(window, 0.0),
                                     timestamps[start + length - 1])
            for window, count in counts.items():
                if count >= min_occurrences:
                    found.append(PatternSequence(
                        steps=window, occurrences=count, last_seen=latest[window]))

        # Most-repeated first, then longest. Frequency has to lead: a user
        # cycling A→B→C four times also produces three-off rotations like
        # B→C→A→B, and ranking by length alone would surface that rotation
        # instead of the actual cycle.
        found.sort(key=lambda seq: (-seq.occurrences, -len(seq.steps)))
        kept: list[PatternSequence] = []
        for candidate in found:
            if not any(_overlaps(candidate, existing) for existing in kept):
                kept.append(candidate)
        return kept


def _contains(long: tuple, short: tuple) -> bool:
    """Whether *short* appears as a contiguous run inside *long*."""
    if len(short) >= len(long):
        return False
    return any(long[i:i + len(short)] == short
               for i in range(len(long) - len(short) + 1))


def _overlaps(candidate: PatternSequence, kept: PatternSequence) -> bool:
    """Whether *candidate* is really the same finding as one already kept.

    Either it sits inside the kept run (or wraps around it), or it is built
    from the same tools (or a subset of them) — a rotation, or the seam where
    one repetition of the cycle joins the next. All of those are the same
    underlying habit reported twice, which reads as noise rather than a
    second discovery.

    Because candidates arrive most-frequent-first, this only ever suppresses
    the *rarer* of two overlapping findings: a genuinely distinct habit that
    happens to reuse some of the same tools still surfaces on its own merits
    if it recurs more often than the run it overlaps.
    """
    if _contains(kept.steps, candidate.steps) or _contains(candidate.steps, kept.steps):
        return True
    return set(candidate.tool_names) <= set(kept.tool_names)
