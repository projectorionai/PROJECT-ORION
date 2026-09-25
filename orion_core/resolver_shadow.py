"""
Shadow evaluation for the tool resolver — earning the right to switch it on.

ORION declares 136 tools, about 31,600 tokens of schema. The tool resolver
(``tool_resolver.py``) exists to pre-filter that surface to the handful a turn
actually needs. It is written, optimised and tested — and it is switched off,
wired to nothing, delivering exactly zero value.

The reason is sound: filtering the tool list risks hiding a tool the model
needed, and a capability that silently disappears is far worse than a large
schema. Nobody could justify enabling it without evidence, and there was no way
to get evidence without enabling it.

This module breaks that circle the way a production ML system would — shadow
mode. On every real turn the resolver runs *alongside* the live path, is told
which tool the model actually chose, and records whether its selection would
have contained that tool. Nothing is filtered. Nothing changes. Evidence
accumulates.

**Recall is the only metric that decides safety.** Not precision, not the token
saving, not the average number of tools kept — those are the *benefit*, and a
benefit is worthless if the cost is ORION losing a capability mid-sentence. A
miss is a turn where the model reached for a tool the resolver would have taken
away. One miss is a defect; the token saving does not buy it back.

So the verdict this module gives is deliberately hard to satisfy:

  * a minimum number of observed turns, because 100% of four turns is not
    evidence of anything;
  * perfect recall over them;
  * and every miss recorded with its query, so a failure is diagnosable rather
    than merely counted.

Anything short of that reports NOT YET and says exactly what is missing. It
never reports a percentage without the sample size beside it — a recall figure
without an N is the kind of number that gets a feature switched on by mistake.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .constants import CONFIG_DIR
from .db import apply_pragmas

#: Turns needed before a verdict is offered at all. Chosen so the sample spans
#: genuinely different requests rather than one repeated session habit.
MIN_SAMPLE = 200

#: Queries are truncated before storage — this is engineering telemetry about
#: tool routing, not a transcript, and it should not quietly become one.
MAX_QUERY_CHARS = 120

# Beside the rest of ORION's data. This was Path("config") — RELATIVE, so it
# followed the working directory: under the standalone app that is
# dist/ORION/config, not the real config folder, and the evidence gathered for
# switching the resolver on was written somewhere nothing else reads.
DEFAULT_PATH = CONFIG_DIR / "resolver_shadow.db"


@dataclass(frozen=True)
class ShadowOutcome:
    """What the resolver would have done on one real turn."""
    query: str
    tool: str
    kept: bool
    rank: int | None          # position in the selection, None if filtered out
    selected: int             # how many tools the resolver would have exposed
    total: int                # how many exist

    @property
    def saving(self) -> float:
        """Fraction of the tool surface the resolver would have removed."""
        if self.total <= 0:
            return 0.0
        return 1.0 - (self.selected / self.total)


class ShadowEvaluator:
    """Runs the resolver in parallel with the live path and scores it."""

    def __init__(self, path: str | Path | None = None, *,
                 min_sample: int = MIN_SAMPLE) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.min_sample = int(min_sample)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                at       REAL NOT NULL,
                query    TEXT NOT NULL,
                tool     TEXT NOT NULL,
                kept     INTEGER NOT NULL,
                rank     INTEGER,
                selected INTEGER NOT NULL,
                total    INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_shadow_kept ON observations(kept);
            CREATE INDEX IF NOT EXISTS idx_shadow_tool ON observations(tool);
            """
        )
        self._db.commit()

    # ── observing ────────────────────────────────────────────────────────────

    def observe(self, query: str, tool: str,
                declarations: Sequence[dict[str, Any]] | None = None,
                state: Any = None) -> ShadowOutcome | None:
        """Score one real turn. Returns None if there was nothing to score.

        Runs the resolver with ``enabled=True`` regardless of the feature flag —
        the whole point is to measure what it WOULD do while it is switched off.
        """
        query = (query or "").strip()
        tool = (tool or "").strip()
        if not query or not tool:
            return None
        try:
            from . import tool_resolver as tr
            if declarations is None:
                from .dispatch_schema import TOOL_DECLARATIONS
                declarations = TOOL_DECLARATIONS
            declarations = list(declarations)
            if not any(d.get("name") == tool for d in declarations):
                # A forged, MCP or plugin tool the resolver was never shown.
                # Scoring it would report a miss the resolver cannot be blamed
                # for, and a metric that punishes the wrong thing gets ignored.
                return None
            # Scored WITH what ORION has learned about how this household phrases
            # things (intent_brain) once that network is built; lexical until then.
            from . import intent_brain
            learned = intent_brain.loaded()
            scorer = ((lambda q, d: intent_brain.combined_scores(q, d, learned))
                      if learned is not None else tr.hybrid_scores)
            selected = tr.resolve(query, state, declarations, enabled=True, scorer=scorer)
        except Exception:
            return None          # shadow evaluation must never disturb a turn

        names = [d.get("name") for d in selected]
        kept = tool in names
        outcome = ShadowOutcome(
            query=query[:MAX_QUERY_CHARS], tool=tool, kept=kept,
            rank=(names.index(tool) if kept else None),
            selected=len(selected), total=len(declarations),
        )
        self._record(outcome)
        return outcome

    def _record(self, outcome: ShadowOutcome) -> None:
        self._db.execute(
            "INSERT INTO observations (at, query, tool, kept, rank, selected, total)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), outcome.query, outcome.tool, 1 if outcome.kept else 0,
             outcome.rank, outcome.selected, outcome.total),
        )
        self._db.commit()

    # ── scoring ──────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT COUNT(*) AS n, SUM(kept) AS kept, AVG(selected) AS sel,"
            " AVG(total) AS tot, AVG(rank) AS rank FROM observations"
        ).fetchone()
        n = row["n"] or 0
        kept = row["kept"] or 0
        selected = row["sel"] or 0.0
        total = row["tot"] or 0.0
        return {
            "turns": n,
            "kept": kept,
            "missed": n - kept,
            "recall": (kept / n) if n else 0.0,
            "mean_selected": selected,
            "mean_total": total,
            "mean_saving": (1.0 - selected / total) if total else 0.0,
            "mean_rank": row["rank"],
        }

    def misses(self, limit: int = 20) -> list[dict[str, Any]]:
        """The turns where the model reached for a tool the resolver would have
        removed. This is the list that matters; everything else is a summary."""
        rows = self._db.execute(
            "SELECT query, tool, selected, total, at FROM observations"
            " WHERE kept = 0 ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def missed_tools(self) -> list[tuple[str, int]]:
        """Which tools get filtered out, worst first — a repeated offender is a
        scoring bug in one tool's description, not a reason to abandon the idea."""
        rows = self._db.execute(
            "SELECT tool, COUNT(*) AS n FROM observations WHERE kept = 0"
            " GROUP BY tool ORDER BY n DESC"
        ).fetchall()
        return [(r["tool"], r["n"]) for r in rows]

    # ── the verdict ──────────────────────────────────────────────────────────

    def ready(self) -> bool:
        """True only on a large enough sample with perfect recall."""
        s = self.stats()
        return s["turns"] >= self.min_sample and s["missed"] == 0

    def verdict(self) -> str:
        s = self.stats()
        if s["turns"] < self.min_sample:
            return (f"NOT YET — {s['turns']} of {self.min_sample} turns observed. "
                    "Recall on a small sample is not evidence.")
        if s["missed"]:
            worst = self.missed_tools()[:3]
            detail = ", ".join(f"{t} x{n}" for t, n in worst)
            return (f"NOT SAFE — {s['missed']} of {s['turns']} turns would have "
                    f"lost the tool the model used ({detail}). Fix the scoring "
                    "for those tools before enabling.")
        return (f"SAFE TO ENABLE — {s['turns']} turns, perfect recall, and the "
                f"resolver would have exposed {s['mean_selected']:.0f} tools "
                f"instead of {s['mean_total']:.0f} "
                f"({s['mean_saving'] * 100:.0f}% less schema). "
                "Set ORION_TOOL_RESOLVER=1.")

    def report(self) -> str:
        s = self.stats()
        if not s["turns"]:
            return ("Tool resolver shadow mode: no turns observed yet. It runs "
                    "alongside the live path and changes nothing; evidence "
                    "accumulates as ORION is used.")
        lines = [
            "Tool resolver — shadow evaluation (nothing is being filtered):",
            f"  turns observed   {s['turns']}",
            f"  recall           {s['recall'] * 100:.1f}%  "
            f"({s['kept']} kept, {s['missed']} missed)",
            f"  would expose     {s['mean_selected']:.1f} of {s['mean_total']:.0f} tools"
            f"  ({s['mean_saving'] * 100:.0f}% less schema)",
        ]
        if s["mean_rank"] is not None:
            lines.append(f"  mean rank of the tool used   {s['mean_rank']:.1f}")
        missed = self.misses(5)
        if missed:
            lines.append("  turns that would have LOST the tool:")
            for m in missed:
                lines.append(f"    {m['tool']:<20} {m['query'][:56]!r}")
        lines.append("  verdict: " + self.verdict())
        return "\n".join(lines)

    def clear(self) -> None:
        """Discard the evidence — after changing the resolver's scoring, the
        old observations describe a different algorithm and would flatter it."""
        self._db.execute("DELETE FROM observations")
        self._db.commit()

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


__all__ = ["DEFAULT_PATH", "MAX_QUERY_CHARS", "MIN_SAMPLE", "ShadowEvaluator",
           "ShadowOutcome"]
