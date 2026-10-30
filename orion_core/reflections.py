"""Bounded presentation of generated reflections and recorded observations.

These entries summarise supplied application context. They are not a provider's
hidden reasoning, and a recorded action request is not proof of completion.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any


def concise_text(value: Any, limit: int = 640) -> str:
    """Keep plain text readable and bounded, including untrusted event fields."""
    if not isinstance(value, str):
        return ""
    text = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", value[:16_000]).split())
    if len(text) <= limit:
        return text
    return text[:limit - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "…"


@dataclass(frozen=True)
class ReflectionEntry:
    event_id: str
    at: str
    observed_at: str
    kind: str
    text: str
    context: str
    evidence: tuple[str, ...]
    origin: str

    @property
    def heading(self) -> str:
        label = "Activity note" if self.origin == "observation" else "Reflection"
        if self.kind == "decision" and self.origin != "observation":
            label = "Action reflection"
        return f"{self.at} · {label}" if self.at else label


def reflection_entry(payload: Any) -> ReflectionEntry | None:
    if not isinstance(payload, dict):
        return None
    text = concise_text(payload.get("text"))
    if not text:
        return None
    context = payload.get("context")
    context_parts: list[str] = []
    if isinstance(context, dict):
        for field, label in (("focus", "Focus"), ("active_project", "Project")):
            value = concise_text(context.get(field), 120)
            if value:
                context_parts.append(f"{label}: {value}")
    evidence = payload.get("evidence")
    details = tuple(filter(None, (
        concise_text(item, 180) for item in evidence[:4]
    ))) if isinstance(evidence, (list, tuple)) else ()
    at = str(payload.get("at") or "")
    at = at if re.fullmatch(r"\d{2}:\d{2}:\d{2}", at) else ""
    observed_at = concise_text(payload.get("observed_at"), 40)
    if observed_at:
        try:
            datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError:
            observed_at = ""
    event_id = payload.get("id")
    # Only scalar IDs are used. An event without an ID is deduplicated by its
    # timestamp and content, so repeated bus delivery cannot duplicate a row.
    if isinstance(event_id, (int, str)) and not isinstance(event_id, bool):
        key = str(event_id)[:100]
    else:
        key = f"{observed_at or at}:{text}"
    return ReflectionEntry(
        event_id=key,
        at=at,
        observed_at=observed_at,
        kind="decision" if payload.get("kind") == "decision" else "reflection",
        text=text,
        context=" · ".join(context_parts),
        evidence=details,
        origin="observation" if payload.get("origin") == "observation" else "generated_reflection",
    )


class ReflectionHistory:
    """Newest-first history; deduplication storage is bounded with the rows."""

    def __init__(self, limit: int = 50) -> None:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ValueError("Reflection history must hold between 1 and 200 entries")
        self.entries: deque[ReflectionEntry] = deque(maxlen=limit)

    def add(self, payload: Any) -> bool:
        entry = reflection_entry(payload)
        if entry is None or any(item.event_id == entry.event_id for item in self.entries):
            return False
        self.entries.appendleft(entry)
        return True

    def clear(self) -> None:
        self.entries.clear()
