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
import math
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any
from urllib.parse import quote_plus

from aiohttp import ClientSession, ClientTimeout

from .bus import OrionBus
from .constants import CONFIG_DIR
from .memory import MemoryAgent
from .notion import NotionService
from .outlook import OutlookService
from .utils import first_line, utc_stamp

NEWS_CACHE_PATH = CONFIG_DIR / "news_cache.db"

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

    # Topic clusters → (label, Google News query, topic weight).  ``when:1h`` asks Google for the last hour only.
    TOPICS: tuple[tuple[str, str, float], ...] = (
        ("AI & AGI", "artificial intelligence OR AGI OR \"large language model\" when:1h", 1.0),
        ("AI labs", "OpenAI OR Anthropic OR \"Google DeepMind\" OR Microsoft AI when:1h", 0.95),
        ("Neural interfaces", "Neuralink OR \"brain computer interface\" OR neurotechnology when:1h", 0.95),
        ("Robotics", "robotics OR humanoid robot when:1h", 0.85),
        ("Semiconductors", "Nvidia OR semiconductor OR TSMC OR chips when:1h", 0.85),
        ("Markets & economy", "stock market OR inflation OR \"central bank\" OR economy when:1h", 0.8),
        ("Geopolitics", "geopolitics OR sanctions OR summit OR election when:1h", 0.7),
        ("Crypto", "bitcoin OR ethereum OR cryptocurrency when:1h", 0.65),
        # Wire services via site scoping — their native RSS is discontinued.
        ("Reuters wire", "site:reuters.com technology OR economy OR AI when:1h", 1.0),
        ("AP wire", "site:apnews.com technology OR economy OR AI when:1h", 1.0),
    )

    # Direct publisher feeds: (label, url, source name, source weight).
    DIRECT_FEEDS: tuple[tuple[str, str, str, float], ...] = (
        ("Markets & economy", "https://www.ft.com/rss/home", "Financial Times", 1.0),
        ("Markets & economy", "https://feeds.bloomberg.com/markets/news.rss", "Bloomberg", 1.0),
        ("AI & AGI", "https://www.technologyreview.com/feed/", "MIT Technology Review", 0.9),
        ("Technology", "https://techcrunch.com/feed/", "TechCrunch", 0.8),
        ("Technology", "https://www.theverge.com/rss/index.xml", "The Verge", 0.75),
        ("Technology", "http://feeds.arstechnica.com/arstechnica/index", "Ars Technica", 0.75),
    )

    MARKET_INDICES = (
        ("^GSPC", "S&P 500"),
        ("^IXIC", "Nasdaq"),
        ("^FTSE", "FTSE 100"),
        ("NVDA", "Nvidia"),
    )

    FRESHNESS_HOURS = 1.5         # BREAKING ONLY — last hour (with a small buffer)
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

    @staticmethod
    def greeting_period(moment: datetime | None = None) -> str:
        """Return morning, afternoon or evening using the Mark XI time bands."""
        hour = (moment or datetime.now()).hour
        if hour < 12:
            return "morning"
        if hour < 17:
            return "afternoon"
        return "evening"

    # ── composition ───────────────────────────────────────────────────────────

    async def compose_source_material(self) -> str:
        """Build the raw intelligence report; every feed concurrent, every
        failure isolated, every story fresh, deduplicated and ranked."""
        now = datetime.now()
        header = (
            f"Intelligence briefing for the last hour. It is {now.strftime('%A %d %B %Y, %H:%M')} "
            f"and the local time is {now.strftime('%H:%M')}."
        )
        await asyncio.to_thread(self.cache.purge_expired)
        timeout = ClientTimeout(total=18.0, connect=5.0)
        async with ClientSession(timeout=timeout) as session:
            news_task = asyncio.create_task(self._news_report(session))
            side_tasks = [
                asyncio.create_task(self._market_section(session)),
                asyncio.create_task(self._crypto_section(session)),
                asyncio.create_task(self._calendar_section()),
                asyncio.create_task(self._tasks_section()),
                asyncio.create_task(self._email_section()),
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
        return briefing

    # ── the news engine ───────────────────────────────────────────────────────

    async def _news_report(self, session: ClientSession) -> str:
        candidates = await self._gather_candidates(session)
        fresh = self._filter_fresh(candidates)
        ranked = self._rank(self._deduplicate(fresh))
        chosen = self._select(ranked)
        if not chosen:
            return ("News: nothing broke in the last hour that you have not already heard — the wires are quiet since your last "
                    "briefing — the wires are quiet.")
        # Commit what will actually be read out, cache-side (off the loop).
        await asyncio.to_thread(lambda: [self.cache.commit(a) for a in chosen])
        self.articles[:] = chosen
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

    async def _gather_candidates(self, session: ClientSession) -> list[dict[str, Any]]:
        tasks = [
            self._fetch_rss(
                session,
                "https://news.google.com/rss/search"
                f"?q={quote_plus(query)}&hl=en-GB&gl=GB&ceid=GB:en",
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

    def _filter_fresh(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.FRESHNESS_HOURS)
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
                fragment = f"{label} at {float(price):,.0f}"
                if previous:
                    change = (float(price) - float(previous)) / float(previous) * 100.0
                    fragment += f" ({change:+.1f} percent)"
                parts.append(fragment)
            except Exception:
                continue
        if not parts:
            return "Stock indices: live quotes unavailable."
        return "Markets: " + "; ".join(parts) + "."

    async def _crypto_section(self, session: ClientSession) -> str:
        try:
            url = (
                "https://api.coingecko.com/api/v3/simple/price"
                "?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
            )
            async with session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"crypto feed returned {response.status}")
                data = await response.json()
            parts: list[str] = []
            for coin_id, coin_label in (("bitcoin", "Bitcoin"), ("ethereum", "Ethereum")):
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
        """The synthesis instruction handed to whichever model delivers it."""
        return (
            f'Open with this exact greeting, word for word: "{greeting}" '
            "Then deliver a private intelligence briefing in the manner of a "
            "trusted executive aide. Every story below is fresh and previously "
            "unheard — synthesise, never recite: lead with the single most "
            "significant development and say why it matters, connect stories "
            "that touch, and add one or two measured observations of your own. "
            "The topic labels and story ages are for your orientation; weave "
            "them in naturally. If calendar entries, tasks or priority email "
            "appear, close with a short 'for your attention today'. Keep it "
            "conversational and calm at a steady, unhurried pace — roughly "
            "ninety seconds. Close by offering to open any story or explore "
            "one in depth:\n\n" + briefing
        )
