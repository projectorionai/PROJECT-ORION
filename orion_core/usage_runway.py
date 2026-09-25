"""
How much longer ORION can keep going today.

  "there must be an estimated time of usage remaining too in ORION"

The token ledger already records every call — input, output, model, provider,
timestamp. What it could not answer is the only question a person actually asks
about it: *am I going to run out, and when?*

That needs two things the ledger does not have. A **budget** — what the limit
actually is, which the provider knows and the ledger does not — and a **burn
rate** measured over a window short enough to reflect what ORION is doing now
rather than what he did at breakfast.

Why the estimate is deliberately conservative
---------------------------------------------
A runway estimate that is too optimistic is worse than none, because it is
believed until the moment it fails. So the rate is taken from the RECENT window
rather than the daily mean (a quiet morning must not hide a heavy afternoon),
and the answer is phrased in the vocabulary of an estimate — "about two hours
at this rate" — never as a countdown, which implies a precision that does not
exist.

Where the limits come from
--------------------------
The table below is a starting point, not a truth. Providers change tiers
without notice, and a limit that is wrong in the generous direction is exactly
the failure this module exists to prevent — so a limit LEARNED from a real 429
or 402 always overrides the table, and the report says which it used.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

STATE_PATH = CONFIG_DIR / "usage_budgets.json"

#: Minutes of history used to judge the CURRENT rate. Short enough to reflect
#: what he is doing now; long enough that one big call does not dominate.
RATE_WINDOW_MINUTES = 45

#: Published free-tier daily token allowances, as of writing. Starting points
#: only — see the module docstring on why a learned limit always wins.
DEFAULT_DAILY_TOKENS: dict[str, int] = {
    "groq": 500_000,
    "openrouter": 200_000,
    "gemini": 1_000_000,
    "google": 1_000_000,
    "cerebras": 1_000_000,
    "together": 250_000,
}


@dataclass
class Budget:
    provider: str
    daily_tokens: int
    source: str = "default table"      # "default table" | "learned" | "user"

    def as_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "daily_tokens": self.daily_tokens,
                "source": self.source}


@dataclass
class Runway:
    provider: str
    used: int
    budget: int
    tokens_per_minute: float
    source: str = ""

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def fraction_used(self) -> float:
        return min(1.0, self.used / self.budget) if self.budget > 0 else 0.0

    @property
    def minutes_left(self) -> float | None:
        """None when idle — an unknown answer, not an infinite one."""
        if self.tokens_per_minute <= 0.01:
            return None
        if self.remaining <= 0:
            return 0.0
        return self.remaining / self.tokens_per_minute

    def describe(self) -> str:
        if self.budget <= 0:
            return f"{self.provider}: no budget known, {self.used:,} tokens used today."
        percent = f"{self.fraction_used * 100:.0f}%"
        head = (f"{self.provider}: {self.used:,} of {self.budget:,} tokens used "
                f"today ({percent})")
        minutes = self.minutes_left
        if self.remaining <= 0:
            return head + " — the daily allowance is gone."
        if minutes is None:
            return head + " — nothing running, so there's no rate to project from."
        return head + f", {_human_duration(minutes)} left at the current rate."


def _human_duration(minutes: float) -> str:
    """Vague on purpose. 'about 2 hours' is honest; '1h 47m' is not."""
    if minutes < 5:
        return "a few minutes"
    if minutes < 55:
        return f"about {int(round(minutes / 5) * 5)} minutes"
    hours = minutes / 60.0
    if hours < 1.75:
        return "about an hour"
    if hours < 10:
        return f"about {round(hours)} hours"
    return "the rest of the day"


class UsageRunway:
    """Budgets, burn rate, and how long that leaves."""

    def __init__(self, ledger: Any, path: Path | None = None) -> None:
        self.ledger = ledger
        self.path = Path(path) if path else STATE_PATH
        self._budgets: dict[str, Budget] = {}
        self._load()

    # ── budgets ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        for provider, entry in (raw or {}).items():
            if isinstance(entry, dict) and entry.get("daily_tokens"):
                self._budgets[provider] = Budget(
                    provider, int(entry["daily_tokens"]),
                    str(entry.get("source") or "user"))

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path,
                json.dumps({p: b.as_dict() for p, b in self._budgets.items()},
                           indent=2), encoding="utf-8")
        except OSError:
            pass

    def budget_for(self, provider: str) -> Budget:
        key = str(provider or "").lower().strip()
        if key in self._budgets:
            return self._budgets[key]
        for name, tokens in DEFAULT_DAILY_TOKENS.items():
            if name in key:
                return Budget(key, tokens, "default table")
        return Budget(key, 0, "unknown")

    def set_budget(self, provider: str, daily_tokens: int,
                   source: str = "user") -> Budget:
        key = str(provider or "").lower().strip()
        budget = Budget(key, max(0, int(daily_tokens)), source)
        self._budgets[key] = budget
        self._save()
        return budget

    def learn_from_limit(self, provider: str, stated_limit: int) -> Budget | None:
        """Record a limit a provider stated in a 429/402.

        Always believed over the table: the provider knows its own tier and the
        table is a guess that goes stale. A limit that is wrong in the generous
        direction is the exact failure this module exists to prevent.
        """
        if stated_limit <= 0:
            return None
        return self.set_budget(provider, stated_limit, "learned from the provider")

    # ── rate ─────────────────────────────────────────────────────────────────

    def tokens_per_minute(self, provider: str = "") -> float:
        """Burn rate over the recent window.

        The recent window rather than the daily mean: a quiet morning would
        otherwise hide a heavy afternoon and the estimate would be optimistic
        exactly when it matters.
        """
        rows = self._timeseries(provider)
        if not rows:
            return 0.0
        cutoff = time.time() - RATE_WINDOW_MINUTES * 60
        recent = [r for r in rows if float(r.get("at", 0) or 0) >= cutoff]
        if not recent:
            return 0.0
        total = sum(int(r.get("tokens", 0) or 0) for r in recent)
        return total / float(RATE_WINDOW_MINUTES)

    def _timeseries(self, provider: str = "") -> list[dict[str, Any]]:
        getter = getattr(self.ledger, "timeseries", None)
        if getter is None:
            return []
        try:
            filters = {"provider": provider} if provider else None
            return list(getter(bucket="hour", filters=filters) or [])
        except Exception:
            return []

    def used_today(self, provider: str = "") -> int:
        summary = getattr(self.ledger, "summary", None)
        if summary is None:
            return 0
        try:
            filters = {"provider": provider} if provider else None
            data = summary(filters) or {}
        except Exception:
            return 0
        return int(data.get("total_tokens")
                   or (int(data.get("input_tokens") or 0)
                       + int(data.get("output_tokens") or 0)))

    # ── the answer ───────────────────────────────────────────────────────────

    def runway(self, provider: str) -> Runway:
        budget = self.budget_for(provider)
        return Runway(provider=provider, used=self.used_today(provider),
                      budget=budget.daily_tokens,
                      tokens_per_minute=self.tokens_per_minute(provider),
                      source=budget.source)

    def report(self, providers: list[str] | None = None) -> str:
        names = providers or sorted(
            set(DEFAULT_DAILY_TOKENS) & set(self._known_providers())) or \
            list(self._known_providers())
        if not names:
            return ("I haven't recorded any model usage yet, so there's nothing "
                    "to project from.")
        lines = [self.runway(name).describe() for name in sorted(names)]
        tightest = min((self.runway(n) for n in names),
                       key=lambda r: r.minutes_left if r.minutes_left is not None
                       else float("inf"))
        head = ""
        if tightest.minutes_left is not None and tightest.minutes_left < 60:
            head = (f"Heads up — {tightest.provider} runs out in about "
                    f"{_human_duration(tightest.minutes_left)}.\n")
        return head + "\n".join(lines)

    def _known_providers(self) -> list[str]:
        getter = getattr(self.ledger, "by_dimension", None)
        if getter is None:
            return []
        try:
            rows = getter("provider") or []
        except Exception:
            return []
        return [str(r.get("provider") or r.get("value") or "") for r in rows
                if r.get("provider") or r.get("value")]


__all__ = ["DEFAULT_DAILY_TOKENS", "RATE_WINDOW_MINUTES", "STATE_PATH",
           "Budget", "Runway", "UsageRunway"]
