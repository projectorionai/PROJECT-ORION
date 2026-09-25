"""
Forge lesson store (Mark II) — the Forge's institutional memory.

Every forge session used to start blind.  A tool could fail three times in a
row for exactly the reason a tool failed last week, and nothing in the pipeline
remembered.  ``ImprovementHeartbeat`` writes a journal, but nothing ever read
it back into a generation prompt, so ORION re-made the same mistakes forever.

This store closes that loop.  Failures are recorded against a normalised error
signature so recurrences *count* rather than accumulate as noise, successful
repairs mark the matching lesson resolved, and ``guidance_block`` renders the
most valuable lessons as a short prompt fragment injected into the next
generation.  The forge that has failed on unescaped Windows paths five times
gets told about it before it writes line one.

Storage is a JSONL file under ``config/self_improvement/`` — append-only, human
readable, and safe to delete (the store rebuilds empty).  All writes are
failure-contained: a lesson that cannot be persisted must never break a forge.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from .constants import CONFIG_DIR
from .forge_diagnosis import Diagnosis, error_signature
from .utils import first_line, utc_stamp

# Module-level so tests can redirect it without touching the user's config.
FORGE_LESSONS_PATH = CONFIG_DIR / "self_improvement" / "forge_lessons.jsonl"

# Never let the corpus grow unbounded — the prompt only ever uses a handful,
# and an enormous file makes every startup read slower for no benefit.
MAX_LESSONS = 200


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{4,}", str(value or "").lower()))


@dataclass
class Lesson:
    """One thing the Forge learned the hard way."""

    signature: str
    failure_class: str
    lesson: str
    tool_names: list[str] = field(default_factory=list)
    hits: int = 1
    resolved_count: int = 0
    first_seen: str = field(default_factory=utc_stamp)
    last_seen: str = field(default_factory=utc_stamp)

    @property
    def unresolved(self) -> int:
        """How often this failure occurred without a repair ever fixing it."""
        return max(0, self.hits - self.resolved_count)

    def weight(self) -> float:
        """Ranking score: recurring, still-unfixed failures matter most."""
        return self.hits + self.unresolved * 2.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lesson":
        return cls(
            signature=str(data.get("signature") or ""),
            failure_class=str(data.get("failure_class") or "unknown"),
            lesson=str(data.get("lesson") or ""),
            tool_names=[str(t)[:60] for t in (data.get("tool_names") or [])][:8],
            hits=max(1, int(data.get("hits") or 1)),
            resolved_count=max(0, int(data.get("resolved_count") or 0)),
            first_seen=str(data.get("first_seen") or utc_stamp()),
            last_seen=str(data.get("last_seen") or utc_stamp()),
        )


class ForgeLessonStore:
    """Deduplicated, weighted memory of why forge sessions fail."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else FORGE_LESSONS_PATH
        self._lock = RLock()
        self._lessons: dict[str, Lesson] = {}
        self._loaded = False

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Read the corpus once, tolerating a partially written tail."""
        if self._loaded:
            return
        self._loaded = True
        try:
            if not self.path.exists():
                return
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    lesson = Lesson.from_dict(json.loads(line))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                if lesson.signature:
                    self._lessons[lesson.signature] = lesson
        except OSError:
            pass

    def _flush(self) -> None:
        """Rewrite the corpus atomically, newest-and-heaviest kept."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            keep = sorted(self._lessons.values(),
                          key=lambda item: (-item.weight(), item.last_seen))[:MAX_LESSONS]
            payload = "\n".join(
                json.dumps(asdict(lesson), ensure_ascii=False) for lesson in keep
            )
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
            tmp.replace(self.path)
            self._lessons = {lesson.signature: lesson for lesson in keep}
        except OSError:
            pass

    # ── recording ────────────────────────────────────────────────────────────

    def record_failure(
        self,
        tool_name: str,
        diagnosis: Diagnosis,
        error_log: Iterable[str] | str | None = None,
    ) -> Lesson | None:
        """Record that *tool_name* failed with *diagnosis*.

        Returns the stored lesson, or None when the failure carried nothing
        identifiable enough to learn from.  Never raises.
        """
        try:
            signature = error_signature(
                list(error_log) if isinstance(error_log, (list, tuple)) else error_log)
            if not signature:
                signature = f"{diagnosis.failure_class.value}:{diagnosis.detail}"[:160]
            if not signature.strip():
                return None
            with self._lock:
                self._load()
                lesson = self._lessons.get(signature)
                if lesson is None:
                    lesson = Lesson(
                        signature=signature,
                        failure_class=diagnosis.failure_class.value,
                        lesson=self._phrase(diagnosis, signature),
                    )
                    self._lessons[signature] = lesson
                else:
                    lesson.hits += 1
                    lesson.last_seen = utc_stamp()
                name = str(tool_name or "")[:60]
                if name and name not in lesson.tool_names:
                    lesson.tool_names = (lesson.tool_names + [name])[-8:]
                self._flush()
                return lesson
        except Exception:
            return None

    def record_resolution(
        self,
        error_log: Iterable[str] | str | None,
        note: str = "",
    ) -> None:
        """Mark the lesson matching *error_log* as having been repaired.

        A failure the Forge reliably recovers from is far less worth warning the
        model about than one that keeps killing sessions, so resolutions damp a
        lesson's ranking weight rather than deleting it.
        """
        try:
            signature = error_signature(
                list(error_log) if isinstance(error_log, (list, tuple)) else error_log)
            if not signature:
                return
            with self._lock:
                self._load()
                lesson = self._lessons.get(signature)
                if lesson is None:
                    return
                lesson.resolved_count += 1
                lesson.last_seen = utc_stamp()
                if note and note not in lesson.lesson:
                    lesson.lesson = f"{lesson.lesson} Fixed by: {first_line(note, 90)}"[:400]
                self._flush()
        except Exception:
            return

    @staticmethod
    def _phrase(diagnosis: Diagnosis, signature: str) -> str:
        """Turn a diagnosis into a sentence worth showing the next generation."""
        return (
            f"A previous tool failed with {diagnosis.failure_class.value} "
            f"({signature}). {diagnosis.detail}."
        )[:400]

    # ── retrieval ────────────────────────────────────────────────────────────

    def lessons(self, limit: int = 8) -> list[Lesson]:
        """The heaviest lessons — recurring and still unfixed rank highest."""
        with self._lock:
            self._load()
            ranked = sorted(self._lessons.values(),
                            key=lambda item: -item.weight())
        return ranked[: max(0, int(limit or 0))]

    def guidance_block(self, plan: str = "", limit: int = 5) -> str:
        """A prompt fragment carrying the lessons most relevant to *plan*.

        Lessons whose text overlaps the capability plan are surfaced first, so a
        plan mentioning HTTP gets the network lessons rather than the generic
        top five.  Returns an empty string when there is nothing worth saying —
        the caller appends it unconditionally.
        """
        candidates = self.lessons(limit=MAX_LESSONS)
        if not candidates:
            return ""
        plan_tokens = _tokens(plan)

        def _rank(lesson: Lesson) -> float:
            overlap = len(plan_tokens & _tokens(lesson.lesson)) if plan_tokens else 0
            return lesson.weight() + overlap * 3.0

        top = sorted(candidates, key=_rank, reverse=True)[: max(1, int(limit or 5))]
        lines = [
            f"- ({lesson.failure_class}, seen {lesson.hits}×) {lesson.lesson}"
            for lesson in top if lesson.lesson.strip()
        ]
        if not lines:
            return ""
        return (
            "LESSONS FROM PREVIOUS FORGE FAILURES (avoid repeating these):\n"
            + "\n".join(lines)
        )

    def describe(self) -> dict[str, Any]:
        with self._lock:
            self._load()
            total = len(self._lessons)
            unresolved = sum(1 for item in self._lessons.values() if item.unresolved)
        return {"lessons": total, "unresolved": unresolved, "path": str(self.path)}


__all__ = ["FORGE_LESSONS_PATH", "ForgeLessonStore", "Lesson"]
