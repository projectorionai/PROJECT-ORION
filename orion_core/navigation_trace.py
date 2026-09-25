"""
navigation_trace.py — a rolling, in-memory record of every UI-targeting
attempt (Mark XXI, Track D7).

Built FIRST, deliberately, ahead of every other navigation-accuracy fix in
this pass: "ORION's navigation isn't perfect" was a feeling with no
evidence behind it. This turns every click_text/click_element attempt into
a queryable record — query, which strategy found it (or didn't), candidates
considered, final coordinates, verification outcome — so the fixes that
follow (fuzzy label matching, an OCR fallback, bidirectional scroll,
informative failures) can be measured against real before/after data
instead of a vibe, and so a live failure can be diagnosed from Diagnostics
rather than guessed at.

Deliberately dependency-light: no Qt, no bus requirement (though one may be
attached for a live diagnostics feed), safe to construct before anything
else exists.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NavigationAttempt:
    at: float
    query: str
    strategy: str                          # "uia" | "ocr" | "not_found"
    found: bool
    candidates: list[str] = field(default_factory=list)   # near-miss names, on failure
    coordinates: tuple[int, int] | None = None
    verified: bool | None = None
    change_ratio: float | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        when = time.strftime("%H:%M:%S", time.localtime(self.at))
        status = "found" if self.found else "NOT FOUND"
        verdict = "" if self.verified is None else (" verified" if self.verified else " unverified")
        return f"{when}  [{self.strategy}] '{self.query}' — {status}{verdict}"


class NavigationTrace:
    """Bounded rolling log of navigation attempts, for Diagnostics and for
    composing informative failure messages (Track D5)."""

    def __init__(self, bus: Any | None = None, maxlen: int = 300) -> None:
        self.bus = bus
        self._log: deque[NavigationAttempt] = deque(maxlen=maxlen)

    def record(self, **fields: Any) -> NavigationAttempt:
        attempt = NavigationAttempt(at=time.time(), **fields)
        self._log.append(attempt)
        if self.bus is not None:
            try:
                self.bus.dashboard_event.emit("navigation", attempt.to_dict())
            except Exception:
                pass
        return attempt

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return [a.to_dict() for a in list(self._log)[-limit:]]

    def failures(self, limit: int = 20) -> list[dict[str, Any]]:
        misses = [a.to_dict() for a in self._log if not a.found]
        return misses[-limit:]

    def summary(self) -> str:
        total = len(self._log)
        if not total:
            return "No navigation attempts recorded yet this session."
        found = sum(1 for a in self._log if a.found)
        verified = sum(1 for a in self._log if a.verified)
        by_strategy: dict[str, int] = {}
        for a in self._log:
            by_strategy[a.strategy] = by_strategy.get(a.strategy, 0) + 1
        strat_line = ", ".join(f"{k}: {v}" for k, v in sorted(by_strategy.items()))
        lines = [
            f"{total} navigation attempt(s) this session — {found} found "
            f"({found / total * 100:.0f}%), {verified} verified.",
            f"By strategy: {strat_line}",
        ]
        misses = self.failures(limit=5)
        if misses:
            lines.append("Recent failures:")
            for m in misses:
                lines.append(f"  - '{m['query']}' ({m['strategy']})"
                             + (f" — nearest: {', '.join(m['candidates'][:3])}" if m["candidates"] else ""))
        return "\n".join(lines)

    def report(self, limit: int = 15) -> str:
        if not self._log:
            return "No navigation attempts recorded yet this session."
        lines = [self.summary(), "", "Recent attempts:"]
        for a in list(self._log)[-limit:]:
            lines.append(f"  {a.line()}")
        return "\n".join(lines)
