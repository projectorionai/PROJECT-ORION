"""
Conversation Memory engine (Phase 4) — long-horizon recall of what was said,
decided and researched, working fully offline.

Built on the MemoryAgent's episodic store (SQLite FTS5), it adds:

    ConversationSummariser   — turns a span of episodes into a durable summary
                               (LLM-written when a model is available, else a
                               deterministic extractive summary offline).
    CompressionEngine        — periodically rolls old raw turns into compact
                               summaries stored in the LONG_TERM tier, keeping
                               the searchable history rich but small.
    ContextRetrieval         — pulls the most relevant past turns + summaries
                               for the current query, as grounding context.
    SessionRecall            — answers time-scoped questions like
                               "what did we discuss three weeks ago about
                               TikTok Shop?" by resolving the time window and
                               searching within it.

All of it degrades gracefully: with the internet down and Ollama present the
summaries are model-written; with nothing but the local store it still recalls
and extractively summarises.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .bus import OrionBus
from .data import ToolResult
from .memory import MemoryAgent, MemoryTier
from .utils import first_line

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "your", "was", "were",
    "have", "has", "had", "our", "are", "but", "not", "what", "when", "where",
    "which", "would", "could", "should", "about", "there", "their", "them",
    "then", "than", "into", "over", "just", "like", "some", "very", "will",
    "can", "did", "does", "yes", "okay", "orion", "sir", "let", "get", "got",
}

# Relative-time phrases → (unit, amount) for window resolution.
_TIME_RE = re.compile(
    r"\b(\d+)?\s*(day|week|month|year)s?\s+ago\b"
    r"|\b(yesterday|today|last week|last month|this week|this month|recently)\b",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "couple": 2, "few": 3, "several": 4,
}
_NUM_WORD_RE = re.compile(
    r"\b(" + "|".join(_WORD_NUMBERS) + r")\s+(day|week|month|year)", re.IGNORECASE)


def _normalise_time_words(text: str) -> str:
    """Turn 'three weeks ago' / 'a couple of months' into '3 weeks ago' etc."""
    text = re.sub(r"(?i)\bcouple of\b", "couple", text)
    return _NUM_WORD_RE.sub(
        lambda m: f"{_WORD_NUMBERS[m.group(1).lower()]} {m.group(2)}", text)


class ConversationSummariser:
    def __init__(self, memory: MemoryAgent, router: Any | None = None) -> None:
        self.memory = memory
        self.router = router

    async def summarise(self, episodes: list[dict[str, str]], topic: str = "") -> str:
        if not episodes:
            return ""
        transcript = "\n".join(
            f"{e.get('role', '?')}: {e.get('content', '')[:280]}" for e in episodes)[:6000]
        # Model-written summary when any provider (local or cloud) is available.
        if self.router is not None and self.router.has_text_fallback():
            try:
                prompt = (
                    "Summarise the following conversation excerpt into 3-5 concise "
                    "bullet points capturing decisions, ideas and action items"
                    + (f" about {topic}" if topic else "") + ":\n\n" + transcript
                )
                _profile, text = await self.router.generate_text(prompt)
                if text.strip():
                    return text.strip()
            except Exception:
                pass
        return self._extractive(episodes, topic)

    def _extractive(self, episodes: list[dict[str, str]], topic: str) -> str:
        """Deterministic offline summary: top sentences by keyword salience."""
        text = " ".join(e.get("content", "") for e in episodes)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        words = [w.lower() for w in re.findall(r"[A-Za-z']+", text) if len(w) > 3]
        freq = Counter(w for w in words if w not in _STOPWORDS)
        topic_terms = set(re.findall(r"[a-z]+", topic.lower()))
        scored: list[tuple[float, str]] = []
        for s in sentences:
            s_words = [w.lower() for w in re.findall(r"[A-Za-z']+", s)]
            if len(s_words) < 4:
                continue
            score = sum(freq.get(w, 0) for w in s_words) / (len(s_words) ** 0.5)
            if topic_terms & set(s_words):
                score *= 2.0
            scored.append((score, s.strip()))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:5]]
        return "\n".join(f"- {s}" for s in top) if top else "- (no substantive content)"


class CompressionEngine:
    """Roll ageing raw turns into compact LONG_TERM summaries."""

    def __init__(self, memory: MemoryAgent, summariser: ConversationSummariser,
                 bus: OrionBus) -> None:
        self.memory = memory
        self.summariser = summariser
        self.bus = bus

    async def compress_older_than(self, days: int = 14, topic: str = "history") -> ToolResult:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
        episodes = self.memory.recall_episodes("", limit=500)
        old = [e for e in episodes if self._before(e.get("created_at", ""), cutoff)]
        if len(old) < 10:
            return ToolResult(f"Only {len(old)} old turn(s); nothing worth compressing yet.")
        summary = await self.summariser.summarise(old, topic=topic)
        key = f"compressed_{cutoff.strftime('%Y%m%d')}"
        self.memory.remember(MemoryTier.LONG_TERM, key, summary)
        self.bus.log.emit(f"MEM: compressed {len(old)} old turns into a durable summary.")
        return ToolResult(f"Compressed {len(old)} older turns into long-term memory:\n{summary}")

    @staticmethod
    def _before(stamp: str, cutoff: datetime) -> bool:
        try:
            dt = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            return dt < cutoff
        except Exception:
            return False


class ContextRetrieval:
    """Assemble grounding context (past turns + summaries) for a query."""

    def __init__(self, memory: MemoryAgent) -> None:
        self.memory = memory

    def context_for(self, query: str, limit: int = 6, max_chars: int = 1600) -> str:
        episodes = self.memory.recall_episodes(query, limit=limit)
        summaries = [r for r in self.memory.recall(MemoryTier.LONG_TERM, query, limit=3)]
        if not episodes and not summaries:
            return ""
        lines = ["[CONVERSATION CONTEXT — prior discussion relevant to this]"]
        for s in summaries:
            lines.append(f"(summary) {s.get('value', '')[:300]}")
        for e in episodes:
            lines.append(f"{e.get('created_at', '')[:10]} {e.get('role', '?')}: "
                         f"{e.get('content', '')[:200]}")
        return "\n".join(lines)[:max_chars]


class SessionRecall:
    """Time-scoped recall: 'what did we discuss N weeks ago about X'."""

    def __init__(self, memory: MemoryAgent, summariser: ConversationSummariser) -> None:
        self.memory = memory
        self.summariser = summariser

    def resolve_window(self, text: str) -> tuple[Optional[datetime], Optional[datetime], str]:
        """Parse a relative-time phrase into a UTC window and a label."""
        now = datetime.now(timezone.utc)
        text = _normalise_time_words(text or "")
        m = _TIME_RE.search(text)
        if not m:
            return None, None, ""
        if m.group(2):  # "N unit ago"
            amount = int(m.group(1) or 1)
            unit = m.group(2).lower()
            days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * amount
            centre = now - timedelta(days=days)
            # Forgiving window so recall isn't brittle about the exact day.
            span = {"day": 2, "week": 7, "month": 20, "year": 60}[unit]
            return centre - timedelta(days=span), centre + timedelta(days=span), f"{amount} {unit}(s) ago"
        phrase = (m.group(3) or "").lower()
        table = {
            "yesterday": (1, 1), "today": (0, 1), "last week": (7, 7),
            "this week": (0, 7), "last month": (30, 15), "this month": (0, 30),
            "recently": (0, 10),
        }
        back, span = table.get(phrase, (0, 10))
        centre = now - timedelta(days=back)
        return centre - timedelta(days=span), now if back == 0 else centre + timedelta(days=span), phrase

    async def recall(self, question: str) -> ToolResult:
        start, end, label = self.resolve_window(question)
        # Extract the topic (strip the time phrase, number words and filler).
        topic = _TIME_RE.sub(" ", _normalise_time_words(question))
        topic = re.sub(
            r"(?i)\b(what|did|we|discuss|discussed|talk|talked|about|regarding|"
            r"the|our|for|of|was|were|ago|\d+|day|days|week|weeks|month|months|year|years)\b",
            " ", topic)
        topic = re.sub(r"\s+", " ", topic).strip(" ?.")
        episodes = self.memory.recall_episodes(topic or question, limit=60)
        if start is not None and end is not None:
            episodes = [e for e in episodes if self._in_window(e.get("created_at", ""), start, end)]
        if not episodes and start is None:
            # No time scope and no keyword hit — the user almost certainly
            # means the conversation happening right now, so never dead-end:
            # fall back to the most recent turns.
            recent = self.memory.recall_episodes("", limit=30)
            if recent:
                summary = await self.summariser.summarise(list(reversed(recent)), topic=topic)
                return ToolResult(
                    "Nothing in the archive matched that exactly, so here is the "
                    f"current conversation instead:\n{summary}\n"
                    "(The 'transcript' tool exports this session verbatim.)")
        if not episodes:
            when = f" from {label}" if label else ""
            return ToolResult(
                f"I have no recorded discussion{when}"
                + (f" about '{topic}'" if topic else "")
                + ". (The 'transcript' tool can export or search this session verbatim.)")
        summary = await self.summariser.summarise(episodes, topic=topic)
        when = f" (around {label})" if label else ""
        head = f"Here's what we discussed{when}" + (f" about '{topic}'" if topic else "") + ":"
        return ToolResult(f"{head}\n{summary}")

    @staticmethod
    def _in_window(stamp: str, start: datetime, end: datetime) -> bool:
        try:
            dt = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            return start <= dt <= end
        except Exception:
            return True  # undated → don't exclude


class ConversationMemoryEngine:
    """Façade tying summariser, compression, retrieval and recall together."""

    def __init__(self, bus: OrionBus, memory: MemoryAgent, router: Any | None = None,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.telemetry = telemetry
        self.summariser = ConversationSummariser(memory, router)
        self.compression = CompressionEngine(memory, self.summariser, bus)
        self.retrieval = ContextRetrieval(memory)
        self.recall_engine = SessionRecall(memory, self.summariser)

    async def recall(self, question: str) -> ToolResult:
        return await self.recall_engine.recall(question)

    async def summarise_recent(self, turns: int = 40) -> ToolResult:
        episodes = self.memory.recall_episodes("", limit=turns)
        if not episodes:
            return ToolResult("No conversation to summarise yet.")
        summary = await self.summariser.summarise(list(reversed(episodes)))
        return ToolResult(f"Recent conversation summary:\n{summary}")

    async def compress(self, days: int = 14) -> ToolResult:
        return await self.compression.compress_older_than(days=days)

    def context_for(self, query: str) -> str:
        return self.retrieval.context_for(query)
