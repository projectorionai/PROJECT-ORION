"""Bounded strength controls; nominal Elo is a handicap, not a measured rating."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    seconds: float = 1.5
    depth: int = 64
    nodes: int = 2_000_000
    elo: int = 2604
    multipv: bool = False
    exploration: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.seconds) or not 0.005 <= self.seconds <= 30:
            raise ValueError("Search time outside 5 ms–30 s")
        if not 1 <= self.depth <= 64 or not 1 <= self.nodes <= 10_000_000:
            raise ValueError("Invalid search limits")
        if not 400 <= self.elo <= 3000 or not 0 <= self.exploration <= 1:
            raise ValueError("Invalid strength controls")


def handicap(elo: int, seconds: float) -> Limits:
    elo = max(400, min(3000, int(elo)))
    seconds = max(0.005, min(30.0, float(seconds)))
    if elo >= 2604:
        return Limits(seconds=seconds, elo=elo)
    fraction = (elo - 400) / 2204
    # Work budget, independent of machine speed. No busy-wait NPS throttling.
    nodes = max(64, int((500 + 150_000 * fraction ** 3) * seconds))
    return Limits(seconds, max(1, int(1 + 11 * fraction)), nodes, elo, True)


def select_move(scores: list[tuple[str, int]], limits: Limits, rng: random.Random) -> tuple[str, int]:
    if not scores:
        raise ValueError("No legal candidates")
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    if limits.elo >= 2604 and not limits.exploration:
        return scores[0]
    temperature = max(12.0, (2604 - limits.elo) * 0.16)
    # Keep proven mates distinct from cp. Prefer a forced win whenever known.
    if scores[0][1] > 29000:
        scores = [s for s in scores if s[1] > 29000]
    elif scores[0][1] > -29000:
        scores = [s for s in scores if s[1] > -29000]
    logits = [math.exp(max(-60.0, (s - scores[0][1]) / temperature)) for _, s in scores]
    total = sum(logits)
    probabilities = [p / total for p in logits]
    if limits.exploration:
        # Dirichlet(alpha=0.3), at the root only, never perturb value targets.
        noise = [rng.gammavariate(0.3, 1.0) for _ in scores]
        total = sum(noise)
        eps = limits.exploration
        probabilities = [(1 - eps) * p + eps * n / total for p, n in zip(probabilities, noise)]
    return rng.choices(scores, weights=probabilities, k=1)[0]


def time_budget(remaining_ms: int, increment_ms: int, cap: float = 1.5) -> tuple[float, float]:
    """(search seconds, absolute move budget); leave network and scheduling reserve."""
    remaining = max(0.0, remaining_ms / 1000)
    usable = max(0.0, remaining - min(1.0, remaining * 0.2))
    total = min(usable, max(0.005, remaining / 35 + max(0, increment_ms) / 1500))
    return max(0.005, min(cap, total * 0.55)), total


def human_delay(total: float, elapsed: float, swing_cp: int, out_of_book: bool,
                rng: random.Random) -> float:
    desired = rng.uniform(0.15, 0.65) + min(1.5, abs(swing_cp) / 350)
    desired += 0.4 if out_of_book else 0.0
    return max(0.0, min(desired, total - elapsed - 0.15))
