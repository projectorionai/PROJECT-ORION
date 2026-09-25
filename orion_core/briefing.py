"""
Real-Time Intelligence Engine (Mark X.7, Phase 2) — the executive briefing.

A redesign of the morning briefing into a private intelligence report:

    BREAKING ONLY      — every story carries its RSS ``pubDate``; anything
                         older than the freshness window is discarded before
                         ranking, and Google News queries carry ``when:1h``.
    NEVER REPEATED     — a dedicated rolling SQLite signature cache
                         (``config/news_cache.db``) with TTL expiry replaces
                         the old LIKE-scans through general memory (which,
                         worse, were never even wired in — the memory handle
                         was omitted at the composition root, so every
                         briefing repeated stories).
    FINGERPRINTED      — sha-256 over the normalised title + URL.
    SEMANTICALLY DEDUPED — the same story from three outlets collapses to the
                         highest-priority source via title token overlap.
    PRIORITY RANKED    — score = source weight + topic weight + exponential
                         recency decay (six-hour half-life); the report leads
                         with what matters most right now.

Sources: Google News RSS per topic cluster (plus site-scoped Reuters and AP,
whose native feeds are discontinued) and direct feeds from the FT, Bloomberg,
TechCrunch, The Verge, Ars Technica and MIT Technology Review.  Every fetch
runs concurrently and every feed may fail alone — a dead API never sinks the
briefing.  Markets, crypto, calendar, tasks and priority email complete the
picture exactly as before.

The public contract is unchanged: ``compose_source_material()``,
``delivery_instruction()``, ``articles``, ``greeting_period()`` and the
constructor signature all remain, so the worker, dispatcher and dashboards
need no modification.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import math
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any
from urllib.parse import quote_plus

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from .lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")

from .bus import OrionBus
from .constants import CONFIG_DIR
from .memory import MemoryAgent
from .notion import NotionService
from .outlook import OutlookService
from .utils import first_line, spoken_time, spoken_year, utc_stamp
from . import background
from .db import apply_pragmas
from .atomic_io import atomic_write_text

NEWS_CACHE_PATH = CONFIG_DIR / "news_cache.db"
# "Was the user already briefed today" — a tiny JSON sidecar, same pattern as
# ProactiveReportingService's config/reporting.json schedule bookkeeping, so
# a restart never re-offers a briefing already given a few hours ago.
BRIEFING_STATE_PATH = CONFIG_DIR / "briefing_state.json"

_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were as at by from into "
    "after over under new says said say will would can could has have had its "
    "his her their our your this that these those be been but not it he she "
    "they we you i up out about amid than more most".split()
)

_RFC822 = ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT",
           "%d %b %Y %H:%M:%S %z")


def _parse_pubdate(raw: str) -> datetime | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    for fmt in _RFC822:
        try:
            parsed = datetime.strptime(raw, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            continue
    return None


def _title_tokens(title: str) -> frozenset[str]:
    return frozenset(
        t for t in re.findall(r"[a-z0-9]{3,}", str(title or "").lower())
        if t not in _STOPWORDS
    )


class NewsSignatureCache:
    """Rolling SQLite cache of consumed stories with TTL expiry (Phase 2)."""

    TTL_DAYS = 7.0

    def __init__(self) -> None:
        self._lock = RLock()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(NEWS_CACHE_PATH, check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._conn)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS seen (
                signature TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                url TEXT NOT NULL DEFAULT '',
                topic TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                consumed_at TEXT NOT NULL,
                expires_at REAL NOT NULL
            )""")
        self._conn.commit()

    def purge_expired(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM seen WHERE expires_at < ?", (time.time(),))
            self._conn.commit()
            return cursor.rowcount

    def seen(self, signature: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM seen WHERE signature = ? AND expires_at >= ?",
                (signature, time.time())).fetchone()
            return row is not None

    def recent_titles(self, limit: int = 300) -> list[str]:
        """Titles still inside the TTL — for cross-briefing semantic dedup."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT title FROM seen WHERE expires_at >= ? "
                "ORDER BY consumed_at DESC LIMIT ?",
                (time.time(), max(1, limit))).fetchall()
        return [str(r[0]) for r in rows]

    def commit(self, article: dict[str, str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO seen"
                "(signature, title, url, topic, source, consumed_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (article.get("signature", ""), article.get("title", ""),
                 article.get("url", ""), article.get("topic", ""),
                 article.get("source", ""), utc_stamp(),
                 time.time() + self.TTL_DAYS * 86400.0))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class MorningBriefingService:
    """Aggregates breaking news, markets, schedule and mail into one report."""

    # Topic clusters → (label, Google News query, topic weight).  The time
    # window (``when:Nh``) is appended at fetch time — see FRESHNESS_LADDER.
    TOPICS: tuple[tuple[str, str, float], ...] = (
        ("AI & AGI", "artificial intelligence OR AGI OR \"large language model\"", 1.0),
        ("AI labs", "OpenAI OR Anthropic OR \"Google DeepMind\" OR Microsoft AI", 0.95),
        ("Neural interfaces", "Neuralink OR \"brain computer interface\" OR neurotechnology", 0.95),
        ("Robotics", "robotics OR humanoid robot", 0.85),
        ("Semiconductors", "Nvidia OR semiconductor OR TSMC OR chips", 0.85),
        ("Science & space", "space mission OR physics OR fusion energy OR quantum computing", 0.9),
        ("Engineering", "engineering breakthrough OR materials science OR battery technology", 0.85),
        ("Markets & economy", "stock market OR inflation OR \"central bank\" OR economy", 0.8),
        ("Geopolitics", "geopolitics OR sanctions OR summit OR election", 0.7),
        ("Crypto", "bitcoin OR ethereum OR solana OR cryptocurrency", 0.65),
        # Wire services via site scoping — their native RSS is discontinued.
        ("Reuters wire", "site:reuters.com technology OR economy OR AI", 1.0),
        ("AP wire", "site:apnews.com technology OR economy OR AI", 1.0),
    )

    # Direct publisher feeds: (label, url, source name, source weight).
    DIRECT_FEEDS: tuple[tuple[str, str, str, float], ...] = (
        ("Markets & economy", "https://www.ft.com/rss/home", "Financial Times", 1.0),
        ("Markets & economy", "https://feeds.bloomberg.com/markets/news.rss", "Bloomberg", 1.0),
        ("AI & AGI", "https://www.technologyreview.com/feed/", "MIT Technology Review", 0.9),
        ("Science & space", "https://www.nasa.gov/rss/dyn/breaking_news.rss", "NASA", 0.9),
        ("Science & space", "https://phys.org/rss-feed/", "Phys.org", 0.8),
        ("Engineering", "https://spectrum.ieee.org/feeds/feed.rss", "IEEE Spectrum", 0.85),
        ("Technology", "https://techcrunch.com/feed/", "TechCrunch", 0.8),
        ("Technology", "https://www.theverge.com/rss/index.xml", "The Verge", 0.75),
        ("Technology", "http://feeds.arstechnica.com/arstechnica/index", "Ars Technica", 0.75),
    )

    # Whole markets, not one company. Nvidia used to sit in this list beside
    # three indices, so a single stock got equal billing with the S&P every
    # morning and the briefing sounded like it was about Nvidia. An index is
    # the market; a ticker is a position, and ORION is not told to hold one.
    # UK first because that is where the user is, then the US majors.
    MARKET_INDICES = (
        ("^FTSE", "the FTSE"),
        ("^GSPC", "the S and P"),
        ("^IXIC", "the Nasdaq"),
        ("^DJI", "the Dow"),
    )

    # DYNAMIC BY CONSTRUCTION: start with breaking-only, but if the last hour
    # is quiet WIDEN the window rather than deliver an empty report — an empty
    # report is what used to make the model improvise from remembered
    # headlines, which read as "the same news every session".  Every rung
    # still deduplicates against the 7-day seen-cache, so a story is never
    # delivered twice even at the 24-hour rung.
    FRESHNESS_LADDER = (1.5, 6.0, 24.0)   # hours, tried in order
    # A morning briefing is an overnight catch-up, not "what happened since I
    # last glanced at this" — the 1.5h rung is almost always satisfied by
    # MIN_STORIES across ~21 feeds regardless of time of day, which is exactly
    # why every briefing used to read the same ("general... past hour").
    # Starting wider here is what actually makes morning distinct.
    MORNING_FRESHNESS_LADDER = (6.0, 24.0)
    MIN_STORIES = 6               # widen the window until this many are found
    RECENCY_HALF_LIFE_H = 0.75    # scoring half-life — heavily favour the newest
    SEMANTIC_OVERLAP = 0.55       # Jaccard threshold: same story, other outlet
    MAX_STORIES = 14              # overall report cap
    MAX_PER_TOPIC = 3

    def __init__(
        self,
        bus: OrionBus,
        notion: NotionService,
        outlook: OutlookService,
        memory: MemoryAgent | None = None,
    ) -> None:
        self.bus = bus
        self.notion = notion
        self.outlook = outlook
        self.memory = memory          # retained for compatibility; dedup now
        self.cache = NewsSignatureCache()   # lives in its own SQLite cache
        # The exact stories read out, so open_news can open the right source.
        self.articles: list[dict[str, str]] = []
        self._briefing_state = self._load_briefing_state()

    # ── "already briefed today" bookkeeping ────────────────────────────────────

    @staticmethod
    def _load_briefing_state() -> dict[str, str]:
        try:
            data = json.loads(BRIEFING_STATE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_briefing_state(self) -> None:
        try:
            BRIEFING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(BRIEFING_STATE_PATH,
                json.dumps(self._briefing_state, indent=2), encoding="utf-8")
        except OSError:
            pass

    def already_briefed_today(self, moment: datetime | None = None) -> bool:
        """True once a briefing has actually been DELIVERED today (local
        date) — checked before offering another, so a restart or a new
        session the same day never re-asks for no reason."""
        from .time_service import TIME
        today = (moment or TIME.now()).strftime("%Y-%m-%d")
        return self._briefing_state.get("last_briefed_date") == today

    def mark_briefed(self, moment: datetime | None = None) -> None:
        """Record that a briefing was just delivered.  Called at the point of
        actual delivery, not merely when one is offered/declined.

        Records the TIME as well as the date. The date alone answers "has he
        had one today", which is enough to avoid re-offering — but not enough
        to say anything useful about it. "You had your briefing in the past
        hour" and "you had it first thing this morning" are different
        sentences, and only a timestamp can tell them apart.
        """
        from .time_service import TIME
        stamp = moment or TIME.now()
        self._briefing_state["last_briefed_date"] = stamp.strftime("%Y-%m-%d")
        self._briefing_state["last_briefed_at"] = stamp.isoformat(timespec="seconds")
        self._save_briefing_state()

    def last_briefed_at(self) -> datetime | None:
        """When the last briefing was actually delivered, if known.

        Returns None for state written before timestamps were recorded, so a
        caller can tell "never briefed" from "briefed, time unknown" — the
        second must not produce a confidently wrong "in the past hour".
        """
        raw = self._briefing_state.get("last_briefed_at")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            return None

    def time_since_briefing(self, moment: datetime | None = None) -> timedelta | None:
        """How long since the last briefing, or None if that is not known."""
        last = self.last_briefed_at()
        if last is None:
            return None
        # Compare in absolute time. Stamps are written as naive LOCAL time, and
        # naive subtraction is an hour out across a clock change — in a phrase
        # that is minute-accurate by request. It also raised TypeError the
        # moment either side was timezone-aware, which is what TimeService.now()
        # returns. astimezone() on a naive value applies the local offset that
        # was in force at THAT moment, so both halves are right.
        def _absolute(value: datetime) -> datetime:
            return value if value.tzinfo is not None else value.astimezone()

        delta = _absolute(moment or datetime.now()) - _absolute(last)
        # A clock change or an edited state file can put this in the future;
        # "in -2 hours" is worse than saying nothing.
        return delta if delta >= timedelta(0) else None

    def briefing_recency_phrase(self, moment: datetime | None = None) -> str:
        """How ORION should refer to the last briefing out loud, or "".

        MINUTE-ACCURATE by request: an earlier version rounded to vague bands
        ("in the past hour", "a couple of hours ago") on the reasoning that
        vague reads as human — but the model then paraphrased those bands into
        something wrong, so a 45-minute-old briefing was announced as "a few
        hours ago". Exact wording removes both the vagueness and the room to
        drift; the worker also tells the model to say this phrase verbatim.

        Still phrased as speech, not a log line — "45 minutes ago", "1 hour and
        20 minutes ago" — never a clock time.
        """
        delta = self.time_since_briefing(moment)
        if delta is None:
            return ""
        total = int(delta.total_seconds())
        if total < 60:
            return "just now"
        minutes = total // 60
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes // 60
        rem = minutes % 60
        hour_part = f"{hours} hour{'s' if hours != 1 else ''}"
        if rem == 0:
            return f"{hour_part} ago"
        return (f"{hour_part} and {rem} minute{'s' if rem != 1 else ''} ago")

    #: A briefing this recent still counts, even across midnight.
    RECENT_BRIEFING_HOURS = 6.0

    def briefed_recently(self, moment: datetime | None = None,
                         hours: float | None = None) -> bool:
        """True if a briefing was delivered within the last few hours.

        Companion to :meth:`already_briefed_today`, which is a CALENDAR test
        and therefore resets at midnight. A briefing at 23:40 stops counting at
        00:01, so ORION would offer a full fresh briefing twenty minutes after
        giving one — and "you've had your briefing in the past hour" is plainly
        the truer thing to say there than "good morning, would you like your
        briefing?".

        Kept separate rather than folded into already_briefed_today() because
        that method's date semantics are relied on elsewhere and are correct
        for what they do; this is an additional reason to skip, not a
        redefinition of the existing one.
        """
        delta = self.time_since_briefing(moment)
        if delta is None:
            return False
        limit = self.RECENT_BRIEFING_HOURS if hours is None else hours
        return delta <= timedelta(hours=limit)

    async def has_fresh_news_since_last_briefing(self) -> bool:
        """Narrowest-rung, READ-ONLY check (never commits to the seen-cache,
        unlike _news_report) for genuinely new, unseen news right now.

        Used to decide whether to re-offer a briefing already given today —
        "is there better news" cannot be answered cheaper than actually
        gathering candidates, so this reuses the same pipeline up to (but not
        including) selection/commit, at just the breaking-only rung."""
        timeout = ClientTimeout(total=12.0, connect=5.0)
        try:
            async with ClientSession(timeout=timeout) as session:
                candidates = await self._gather_candidates(session, self.FRESHNESS_LADDER[0])
            fresh = self._deduplicate(self._filter_fresh(candidates, self.FRESHNESS_LADDER[0]))
            return len(fresh) > 0
        except Exception:
            return False

    @staticmethod
    def greeting_period(moment: datetime | None = None) -> str:
        """Return morning, afternoon or evening from the shared time bands.

        Delegates to TimeService so a briefing header can never name a
        different part of the day than the greeting or the system prompt —
        this copy used a 17:00 evening boundary, local_brain used 18:00."""
        from .time_service import TIME, greeting_for
        return greeting_for(moment) if moment is not None else TIME.greeting_period()

    # ── composition ───────────────────────────────────────────────────────────

    async def compose_source_material(self, period: str = "general") -> str:
        """Build the raw intelligence report; every feed concurrent, every
        failure isolated, every story fresh, deduplicated and ranked.

        period="morning" is an overnight catch-up, not a generic "what's new
        right now" — it widens the news floor past the 1.5h rung (see
        MORNING_FRESHNESS_LADDER) and frames the header accordingly, instead
        of running byte-for-byte the same composition regardless of when or
        why it was asked for."""
        # The shared clock, not the machine's: TimeService reads ORION's own
        # zone, and on a machine set to another one (a UTC cloud node) the
        # header said "afternoon" and the wrong local time while the greeting
        # said "evening".
        from .time_service import TIME
        now = TIME.now()
        is_morning = period.strip().lower() == "morning"
        # Times and years in words — if the digits never reach the voice
        # channel, they can never be misread aloud (22:27 → "20:27").
        if is_morning:
            header = (
                f"Good {self.greeting_period(now)} — here is what happened overnight "
                f"and what is ahead today. It is {now.strftime('%A %d %B')} "
                f"{spoken_year(now.year)}, and the local time is "
                f"{spoken_time(now.hour, now.minute)}."
            )
        else:
            # Named for the part of the day it is. "Intelligence briefing"
            # left the model to supply the word itself, and it reached for the
            # tool's name: "here is your morning briefing" at half past six in
            # the evening.
            header = (
                f"Your {self.greeting_period(now)} briefing — the freshest developments "
                f"you have not yet heard. It is {now.strftime('%A %d %B')} "
                f"{spoken_year(now.year)}, and the local time is "
                f"{spoken_time(now.hour, now.minute)}."
            )
        await asyncio.to_thread(self.cache.purge_expired)
        timeout = ClientTimeout(total=18.0, connect=5.0)
        ladder = self.MORNING_FRESHNESS_LADDER if is_morning else None
        async with ClientSession(timeout=timeout) as session:
            news_task = asyncio.create_task(self._news_report(session, ladder))
            side_tasks = [
                background.spawn(self._market_section(session)),
                background.spawn(self._crypto_section(session)),
                background.spawn(self._calendar_section()),
                background.spawn(self._tasks_section()),
                background.spawn(self._email_section()),
            ]
            results = await asyncio.gather(news_task, *side_tasks,
                                           return_exceptions=True)
        lines = [header]
        for section in results:
            if isinstance(section, BaseException) or not section:
                continue
            lines.append(str(section))
        briefing = "\n".join(lines)
        self.bus.dashboard_event.emit("briefing", briefing)
        self.mark_briefed(now)
        return briefing

    # ── the news engine ───────────────────────────────────────────────────────

    async def _news_report(self, session: ClientSession,
                           ladder: tuple[float, ...] | None = None) -> str:
        # Climb the freshness ladder: breaking-only first, wider only if the
        # wires are quiet — but ALWAYS unseen stories (the signature cache
        # guarantees this briefing differs from every previous one).
        chosen: list[dict[str, str]] = []
        candidates: list[dict[str, Any]] = []
        for window_h in (ladder or self.FRESHNESS_LADDER):
            candidates = await self._gather_candidates(session, window_h)
            fresh = self._filter_fresh(candidates, window_h)
            ranked = self._rank(self._deduplicate(fresh))
            chosen = self._select(ranked)
            if len(chosen) >= self.MIN_STORIES:
                break
            self.bus.log.emit(
                f"NEWS: only {len(chosen)} unseen story(ies) inside {window_h:g}h — widening the window.")
        if not chosen:
            return ("News: nothing has broken that you have not already heard — "
                    "the wires are genuinely quiet since your last briefing.")
        # Commit what will actually be read out, cache-side (off the loop).
        await asyncio.to_thread(lambda: [self.cache.commit(a) for a in chosen])
        self.articles[:] = chosen
        # To the content panel as well as the log. A headline read aloud cannot
        # be clicked, and by the third story the first has scrolled out of the
        # log entirely.
        try:
            self.bus.content_results.emit(list(chosen))
        except Exception:
            pass
        self.bus.log.emit(
            f"NEWS: {len(chosen)} fresh story(ies) selected from "
            f"{len(candidates)} candidates - say \"open the story about …\"."
        )
        for index, article in enumerate(chosen, 1):
            self.bus.log.emit(
                f"NEWS[{index}]: ({article['topic']}) {article['title']}"
                f" — {article['source']}, {article['age_label']}")
        # Group by topic for the report body.
        sections: dict[str, list[str]] = {}
        for a in chosen:
            fragment = f"{a['title']} ({a['source']}, {a['age_label']})."
            sections.setdefault(a["topic"], []).append(fragment)
        lines = []
        for topic, fragments in sections.items():
            lines.append(f"{topic}: " + " ".join(fragments))
        return "\n".join(lines)

    async def _gather_candidates(self, session: ClientSession,
                                 window_hours: float = 1.0) -> list[dict[str, Any]]:
        when = f" when:{max(1, int(round(window_hours)))}h"
        tasks = [
            self._fetch_rss(
                session,
                "https://news.google.com/rss/search"
                f"?q={quote_plus(query + when)}&hl=en-GB&gl=GB&ceid=GB:en",
                topic=label, source="", source_weight=0.6, topic_weight=weight,
            )
            for label, query, weight in self.TOPICS
        ] + [
            self._fetch_rss(session, url, topic=label, source=source,
                            source_weight=weight, topic_weight=0.8)
            for label, url, source, weight in self.DIRECT_FEEDS
        ]
        batches = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[dict[str, Any]] = []
        for batch in batches:
            if isinstance(batch, list):
                out.extend(batch)
        return out

    async def _fetch_rss(
        self, session: ClientSession, url: str, topic: str,
        source: str, source_weight: float, topic_weight: float,
    ) -> list[dict[str, Any]]:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (ORION intelligence engine)"}
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise RuntimeError(f"feed returned {response.status}")
                raw = await response.text()
        except Exception as exc:
            self.bus.log.emit(f"NEWS: feed skipped ({topic}) - {first_line(exc, 70)}")
            return []
        stories: list[dict[str, Any]] = []
        for block in re.findall(r"<item>(.*?)</item>", raw, re.S)[:12]:
            title_m = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", block, re.S)
            link_m = re.search(r"<link>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</link>", block, re.S)
            date_m = re.search(r"<pubDate>(.*?)</pubDate>", block, re.S)
            src_m = re.search(r"<source[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</source>", block, re.S)
            if not title_m:
                continue
            title = html.unescape(re.sub(r"\s+", " ", title_m.group(1))).strip()
            # Google News suffixes " - Publisher"; recover it as the source.
            story_source = source or (html.unescape(src_m.group(1)).strip() if src_m else "")
            if not source and " - " in title:
                title, _, suffix = title.rpartition(" - ")
                story_source = story_source or suffix.strip()
            if len(title) < 18:
                continue
            link = html.unescape(link_m.group(1).strip()) if link_m else ""
            published = _parse_pubdate(date_m.group(1) if date_m else "")
            stories.append({
                "title": title[:200],
                "url": link,
                "topic": topic,
                "source": (story_source or "wire")[:60],
                "published": published,
                "source_weight": source_weight if not source else source_weight,
                "topic_weight": topic_weight,
                "signature": hashlib.sha256(
                    (re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
                     + "|" + link.lower()).encode("utf-8")).hexdigest()[:32],
                "tokens": _title_tokens(title),
            })
        return stories

    # ── freshness, dedup, ranking ─────────────────────────────────────────────

    def _filter_fresh(self, candidates: list[dict[str, Any]],
                      window_hours: float) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        fresh = []
        for c in candidates:
            published = c.get("published")
            if published is not None and published < cutoff:
                continue     # stale — breaking news only
            fresh.append(c)
        return fresh

    def _deduplicate(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Signature dedup against the TTL cache, then semantic dedup within
        the batch AND against recently-consumed titles (same story, new outlet)."""
        recent = [_title_tokens(t) for t in self.cache.recent_titles()]
        kept: list[dict[str, Any]] = []
        for c in sorted(candidates, key=lambda x: -(x["source_weight"])):
            if self.cache.seen(c["signature"]):
                continue
            tokens = c["tokens"]
            if not tokens:
                continue
            duplicate = False
            for other in kept:
                union = tokens | other["tokens"]
                if union and len(tokens & other["tokens"]) / len(union) >= self.SEMANTIC_OVERLAP:
                    duplicate = True
                    break
            if not duplicate:
                for seen_tokens in recent:
                    union = tokens | seen_tokens
                    if union and len(tokens & seen_tokens) / len(union) >= self.SEMANTIC_OVERLAP:
                        duplicate = True
                        break
            if not duplicate:
                kept.append(c)
        return kept

    def _rank(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        for c in candidates:
            published = c.get("published")
            if published is not None:
                age_h = max(0.0, (now - published).total_seconds() / 3600.0)
                recency = math.exp(-age_h * math.log(2) / self.RECENCY_HALF_LIFE_H)
                c["age_label"] = (f"{int(age_h)}h ago" if age_h >= 1.0 else "within the hour")
            else:
                recency = 0.4          # undated: assume merely recent
                c["age_label"] = "today"
            c["score"] = (c["source_weight"] * 0.8
                          + c["topic_weight"] * 0.8
                          + recency * 1.4)
        return sorted(candidates, key=lambda x: -x["score"])

    def _select(self, ranked: list[dict[str, Any]]) -> list[dict[str, str]]:
        per_topic: dict[str, int] = {}
        chosen: list[dict[str, str]] = []
        for c in ranked:
            if len(chosen) >= self.MAX_STORIES:
                break
            topic = c["topic"]
            if per_topic.get(topic, 0) >= self.MAX_PER_TOPIC:
                continue
            per_topic[topic] = per_topic.get(topic, 0) + 1
            chosen.append({
                "topic": topic, "title": c["title"], "url": c["url"],
                "source": c["source"], "signature": c["signature"],
                "age_label": c["age_label"], "guid": "",
            })
        return chosen

    # ── markets / crypto / personal feeds (proven; retained) ─────────────────

    async def _market_section(self, session: ClientSession) -> str:
        parts: list[str] = []
        headers = {"User-Agent": "Mozilla/5.0 (ORION briefing)"}
        for symbol, label in self.MARKET_INDICES:
            try:
                url = (
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(symbol)}"
                    "?range=1d&interval=1d"
                )
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        continue
                    data = await response.json(content_type=None)
                meta = (
                    ((data.get("chart") or {}).get("result") or [{}])[0].get("meta") or {}
                )
                price = meta.get("regularMarketPrice")
                previous = meta.get("chartPreviousClose") or meta.get("previousClose")
                if price is None:
                    continue
                # Spoken, the absolute level is noise — nobody needs to hear
                # that the S&P is "at five thousand nine hundred". The move is
                # the whole message, so lead with it and keep the level for
                # the case where there is no previous close to compare.
                if previous:
                    change = (float(price) - float(previous)) / float(previous) * 100.0
                    if abs(change) < 0.05:
                        # "up 0.0 percent" is not something anyone says, and it
                        # is the wrong claim besides — that is a market that
                        # did not move.
                        fragment = f"{label} flat"
                    else:
                        direction = "up" if change > 0 else "down"
                        fragment = f"{label} {direction} {abs(change):.1f} percent"
                else:
                    fragment = f"{label} at {float(price):,.0f}"
                parts.append(fragment)
            except Exception:
                continue
        if not parts:
            return "Markets: live quotes unavailable."
        return "Markets: " + ", ".join(parts) + "."

    async def _crypto_section(self, session: ClientSession) -> str:
        try:
            url = (
                "https://api.coingecko.com/api/v3/simple/price"
                "?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true"
            )
            async with session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"crypto feed returned {response.status}")
                data = await response.json()
            parts: list[str] = []
            for coin_id, coin_label in (("bitcoin", "Bitcoin"), ("ethereum", "Ethereum"),
                                        ("solana", "Solana")):
                coin = data.get(coin_id) or {}
                price = coin.get("usd")
                change = coin.get("usd_24h_change")
                if price is None:
                    continue
                fragment = f"{coin_label} at {float(price):,.0f} dollars"
                if change is not None:
                    fragment += f" ({float(change):+.1f} percent over twenty-four hours)"
                parts.append(fragment)
            if not parts:
                raise RuntimeError("no crypto prices returned")
            return "Cryptocurrency: " + "; ".join(parts) + "."
        except Exception:
            return "Cryptocurrency pricing unavailable."

    async def _calendar_section(self) -> str:
        if not self.notion.available:
            return ""
        try:
            result = await self.notion.upcoming_events(days=2, limit=8)
            return "Calendar: " + result.text if result.ok else ""
        except Exception as exc:
            return f"Calendar feed unavailable ({first_line(exc, 80)})."

    async def _tasks_section(self) -> str:
        if not self.notion.available:
            return ""
        try:
            result = await self.notion.list_tasks(limit=6)
            return "Tasks: " + result.text if result.ok else ""
        except Exception as exc:
            return f"Task feed unavailable ({first_line(exc, 80)})."

    async def _email_section(self) -> str:
        if not self.outlook.available:
            return ""
        try:
            result = await self.outlook.priority_emails(limit=5)
            return "Email: " + result.text if result.ok else ""
        except Exception as exc:
            return f"Email feed unavailable ({first_line(exc, 80)})."

    # ── delivery instruction ──────────────────────────────────────────────────

    @staticmethod
    def delivery_instruction(greeting: str, briefing: str) -> str:
        """The synthesis instruction handed to whichever model delivers it.

        Pass an empty ``greeting`` when the user has ALREADY been greeted this
        session (the startup offer) — the model then opens with a short
        transition instead of greeting twice, which read as ORION repeating
        himself."""
        from .time_service import TIME

        period = TIME.greeting_period()
        if greeting.strip():
            opener = f'Open with this exact greeting, word for word: "{greeting}" '
        else:
            opener = (
                "You have already greeted the user moments ago — do NOT greet "
                "them again, do not say 'good morning/afternoon/evening', and do "
                "not restate the date, time or weather. Open with one short "
                f"transition such as 'Right then — your {period} briefing.' "
            )
        return (
            opener
            + f"It is the {period} now: if you name the briefing, call it your "
            f"{period} briefing or simply your briefing — never a 'morning "
            "briefing' unless it is actually the morning. "
            + "Then deliver a private intelligence briefing in the manner of a "
            "trusted executive aide. Every story below is fresh and previously "
            "unheard — synthesise, never recite: lead with the single most "
            "significant development and say why it matters, connect stories "
            "that touch, and add one or two measured observations of your own. "
            "HARD RULE: use ONLY the stories in the material below. Do not add, "
            "substitute or embellish with any headline you remember from an "
            "earlier session or from training — remembered news is stale news. "
            "If the material lists nothing for a topic, say the wires are quiet "
            "on it rather than inventing coverage. "
            "The topic labels and story ages are for your orientation; weave "
            "them in naturally. If calendar entries, tasks or priority email "
            "appear, close with a short 'for your attention today'. Keep it "
            "conversational and calm at a steady, unhurried pace, and TIGHT — "
            "about seventy-five seconds, six or seven stories at most. It is far "
            "better to cover fewer stories and FINISH than to run long: always "
            "complete your closing sentence and never stop mid-thought. End by "
            "offering to open any story or explore one in depth:\n\n" + briefing
        )
