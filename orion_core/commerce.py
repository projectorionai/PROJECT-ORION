"""
Entrepreneurial Intelligence suite (Phases 5-7, 9, 11).

Offline-first agents that turn ORION into a business partner:

    ProductOpportunityScore  — a deterministic 0-100 score over seven metrics
                               (virality, demand, competition, profit margin,
                               shipping complexity, return risk, seasonal
                               dependence), inferable from a plain description.

    DropshippingResearchAgent — product research, validation, competitor and
                               saturation analysis; every product is scored.

    TikTokShopAgent          — trend/product/creator analysis + report
                               generation over supplied or stored data.

    InstagramCommerceAgent   — product discovery, influencer + store analysis,
                               weekly opportunity rankings.

    FounderKnowledgeAgent    — structured, searchable profiles of leading
                               operators (strategies, systems, frameworks) for
                               educational analysis.

    BusinessAdvisorAgent     — brand/product/marketing/growth strategy that
                               understands the user's brand (ExampleStore — home
                               products) and grounds advice in the knowledge
                               packs and prior product research.

Everything scores and reports deterministically offline; when a model (local
Ollama or cloud) is available, advice is additionally LLM-written, grounded in
the same local context.  Research persists to memory (``product_research``)
so it is recallable across sessions and in MODE B.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Optional
from urllib.parse import quote

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .memory import MemoryAgent, MemoryTier
from .utils import first_line, utc_stamp
from .db import apply_pragmas
from . import user_profile

COMMERCE_SIGNAL_DB = CONFIG_DIR / "commerce_signals.db"
_WIKIMEDIA_PAGEVIEWS = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
)
_SIGNAL_USER_AGENT = "ORION-AI-OS/10.x (personal assistant; contact via desktop app)"


# ──────────────────────────────────────────────────────────────────────────────
# PRODUCT OPPORTUNITY SCORE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProductMetrics:
    virality: float = 5.0            # 0-10, higher better
    demand: float = 5.0              # 0-10, higher better
    competition: float = 5.0         # 0-10, higher = MORE competition (worse)
    profit_margin: float = 5.0       # 0-10, higher better
    shipping_complexity: float = 5.0 # 0-10, higher = harder (worse)
    return_risk: float = 5.0         # 0-10, higher = riskier (worse)
    seasonal_dependence: float = 5.0 # 0-10, higher = more seasonal (worse)

    def clamp(self) -> "ProductMetrics":
        for f in self.__dataclass_fields__:
            setattr(self, f, max(0.0, min(10.0, float(getattr(self, f)))))
        return self


class ProductOpportunityScore:
    """Weighted score; 'bad' metrics are inverted so higher output = better."""

    WEIGHTS = {
        "virality": 0.22, "demand": 0.22, "profit_margin": 0.20,
        "competition": 0.15, "shipping_complexity": 0.08,
        "return_risk": 0.07, "seasonal_dependence": 0.06,
    }
    INVERTED = {"competition", "shipping_complexity", "return_risk", "seasonal_dependence"}

    @classmethod
    def score(cls, m: ProductMetrics) -> float:
        m.clamp()
        total = 0.0
        for field_name, weight in cls.WEIGHTS.items():
            value = getattr(m, field_name)
            norm = (10.0 - value) if field_name in cls.INVERTED else value
            total += weight * (norm / 10.0)
        return round(total * 100.0, 1)

    @staticmethod
    def verdict(score: float) -> str:
        if score >= 75:
            return "STRONG — high-priority test candidate"
        if score >= 60:
            return "PROMISING — worth a small validation test"
        if score >= 45:
            return "MARGINAL — only with a strong angle or offer"
        return "AVOID — poor risk/reward at current read"

    # ── heuristic inference from a description (offline) ──────────────────────

    @classmethod
    def infer(cls, description: str, given: dict[str, Any] | None = None) -> ProductMetrics:
        d = description.lower()
        m = ProductMetrics()

        def has(*words: str) -> bool:
            return any(w in d for w in words)

        # Virality / wow factor
        m.virality = 7.5 if has("viral", "trending", "tiktok", "wow", "satisfying", "gadget") else 4.5
        # Demand
        m.demand = 7.0 if has("problem", "everyday", "home", "kitchen", "pain", "must-have", "organis") else 5.0
        # Competition
        m.competition = 7.5 if has("saturated", "everyone selling", "common", "cheap", "generic") else 4.5
        # Profit margin
        m.profit_margin = 7.0 if has("premium", "high margin", "markup", "unique", "branded") else 5.0
        if has("cheap", "low cost", "commodity"):
            m.profit_margin = 3.5
        # Shipping complexity
        m.shipping_complexity = 7.5 if has("fragile", "heavy", "bulky", "large", "glass", "liquid") else 3.5
        # Return risk
        m.return_risk = 7.0 if has("sizing", "electronic", "battery", "fragile", "clothing", "fit") else 3.5
        # Seasonal
        m.seasonal_dependence = 8.0 if has("christmas", "summer", "winter", "halloween", "seasonal", "holiday") else 3.0
        if given:
            for k, v in given.items():
                if k in m.__dataclass_fields__ and isinstance(v, (int, float)):
                    setattr(m, k, float(v))
        return m.clamp()


# ──────────────────────────────────────────────────────────────────────────────
# COMMERCE DEMAND SIGNAL (Improvement Pass, Priority 3.2)
# ──────────────────────────────────────────────────────────────────────────────

class CommerceSignalProvider:
    """One real, free demand signal that *grounds* the offline scoring.

    Wikimedia pageviews are a public, key-free proxy for consumer interest in a
    product/topic.  A rising view trend and high absolute traffic are genuine
    evidence a product is being searched for.  The read is interpreted into a
    small, bounded signal (``interest`` 0-10, ``momentum`` 0-10 centred on 5,
    ``trend``) and cached in SQLite so we never hammer the API.

    The whole path is defensive: an unknown article, a network error, thin data
    or a throttle all resolve to ``None`` so the deterministic offline score
    (MODE B) is *always* preserved — the signal only ever adds confidence, it
    never blocks advice.  ``fetch`` and ``clock`` are injected so the behaviour
    is fully testable offline.
    """

    def __init__(self, db_path: Path | str, *, ttl_hours: float = 24.0,
                 min_interval_s: float = 1.05,
                 fetch: Callable[[str], Any] | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.ttl_s = max(0.0, float(ttl_hours)) * 3600.0
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._fetch = fetch or self._default_fetch
        self._clock = clock or time.time
        self._lock = RLock()
        self._last_fetch: float | None = None
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._conn)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS signal ("
            "  term TEXT PRIMARY KEY,"
            "  payload TEXT NOT NULL,"
            "  cached_at REAL NOT NULL)")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ── public read ───────────────────────────────────────────────────────────

    async def interest(self, term: str) -> dict[str, Any] | None:
        """Real demand signal for *term*, or ``None`` (→ offline fallback)."""
        key = self._normalise(term)
        if not key:
            return None
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        # Politeness throttle — a cache miss during the cool-down degrades to
        # offline rather than spamming the API.
        now = self._clock()
        if (self._last_fetch is not None
                and (now - self._last_fetch) < self.min_interval_s):
            return None
        self._last_fetch = now
        try:
            payload = await self._fetch(self._build_url(key))
        except Exception:
            return None
        if not payload:
            return None
        signal = self._interpret(key, payload)
        if signal is None:
            return None
        self._cache_put(key, signal)
        return signal

    # ── interpretation ────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(term: str) -> str:
        return re.sub(r"\s+", " ", str(term or "")).strip().lower()

    @staticmethod
    def _build_url(key: str) -> str:
        article = quote((key[:1].upper() + key[1:]).replace(" ", "_"), safe="_")
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=60)
        return _WIKIMEDIA_PAGEVIEWS.format(
            article=article, start=start.strftime("%Y%m%d"),
            end=end.strftime("%Y%m%d"))

    @staticmethod
    def _interpret(key: str, payload: Any) -> dict[str, Any] | None:
        items = payload.get("items") if isinstance(payload, dict) else None
        if not items:
            return None
        views = [float(it["views"]) for it in items
                 if isinstance(it, dict) and isinstance(it.get("views"), (int, float))]
        if len(views) < 8:
            return None
        half = len(views) // 2
        first = sum(views[:half]) / max(1, half)
        second = sum(views[half:]) / max(1, len(views) - half)
        # Momentum: log-ratio of the two halves squashed to (0, 10) about 5.
        # +1 smoothing keeps zero-view days well-behaved.
        ratio = (second + 1.0) / (first + 1.0)
        momentum = round(max(0.0, min(10.0, 5.0 + 5.0 * math.tanh(math.log(ratio)))), 2)
        avg = sum(views) / len(views)
        interest = round(max(0.0, min(10.0, 2.0 * math.log10(avg + 1.0))), 2)
        trend = ("rising" if momentum >= 5.5
                 else "falling" if momentum <= 4.5 else "steady")
        return {"term": key, "interest": interest, "momentum": momentum,
                "trend": trend, "samples": len(views),
                "source": "wikimedia-pageviews"}

    # ── cache ─────────────────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, cached_at FROM signal WHERE term = ?",
                (key,)).fetchone()
        if row is None:
            return None
        if self._clock() - row[1] > self.ttl_s:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def _cache_put(self, key: str, signal: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO signal(term, payload, cached_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(signal), self._clock()))
            self._conn.commit()

    # ── default network fetch (real Wikimedia API) ────────────────────────────

    async def _default_fetch(self, url: str) -> Any | None:
        try:
            from aiohttp import ClientSession, ClientTimeout
        except Exception:
            return None
        try:
            timeout = ClientTimeout(total=10.0, connect=5.0)
            headers = {"User-Agent": _SIGNAL_USER_AGENT}
            async with ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json(content_type=None)
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# BASE
# ──────────────────────────────────────────────────────────────────────────────

class _CommerceBase:
    """Shared grounding + optional LLM advice for the commerce agents."""

    persona = "SPECIALIST MODE — E-commerce strategist."

    def __init__(self, bus: OrionBus, memory: MemoryAgent, router: Any,
                 knowledge: Any | None = None, telemetry: Any | None = None,
                 signals: "CommerceSignalProvider | None" = None) -> None:
        self.bus = bus
        self.memory = memory
        self.router = router
        self.knowledge = knowledge
        self.telemetry = telemetry
        self.signals = signals

    async def _demand_signal(self, term: str) -> dict[str, Any] | None:
        """Best-effort real demand read; never raises, ``None`` when offline."""
        if self.signals is None:
            return None
        try:
            return await self.signals.interest(term)
        except Exception:
            return None

    @staticmethod
    def _demand_note(sig: dict[str, Any], term: str) -> str:
        return (f"REAL DEMAND SIGNAL ({sig['source']}) — '{term}': "
                f"interest {sig['interest']:.0f}/10, trend {sig['trend']} "
                f"(momentum {sig['momentum']:.1f}/10).")

    async def _advise(self, prompt: str, ground_query: str = "") -> str:
        """Ground in knowledge packs, then answer via provider (local/cloud)."""
        context = ""
        if self.knowledge is not None:
            context = self.knowledge.context_for(ground_query or prompt, limit=3)
        full = (context + "\n\n" + prompt) if context else prompt
        if self.router is not None and self.router.has_text_fallback():
            try:
                _profile, text = await self.router.generate_text(full, system_extra=self.persona)
                return text
            except Exception as exc:
                return f"(Model unavailable: {first_line(exc)})\n" + (context or "")
        return context or "(No model or knowledge available for deeper analysis.)"


# ──────────────────────────────────────────────────────────────────────────────
# DROPSHIPPING RESEARCH AGENT (Phase 5)
# ──────────────────────────────────────────────────────────────────────────────

class DropshippingResearchAgent(_CommerceBase):
    persona = (
        "SPECIALIST MODE — Dropshipping research analyst. Evaluate products with "
        "discipline: virality, demand, competition/saturation, margin, shipping, "
        "returns and seasonality. Be decisive and honest; recommend test or pass."
    )

    async def score_product(self, name: str, description: str = "",
                            metrics: dict[str, Any] | None = None) -> ToolResult:
        given = dict(metrics or {})
        # Ground the two demand-driven metrics in a real signal *only* when the
        # caller hasn't pinned both — caller-supplied metrics always win, and an
        # unavailable signal leaves the deterministic offline read untouched.
        sig: dict[str, Any] | None = None
        if not {"demand", "virality"} <= given.keys():
            sig = await self._demand_signal(name)
            if sig:
                given.setdefault("demand", sig["interest"])
                given.setdefault("virality", sig["momentum"])
        m = ProductOpportunityScore.infer(description or name, given)
        score = ProductOpportunityScore.score(m)
        verdict = ProductOpportunityScore.verdict(score)
        breakdown = " | ".join(
            f"{k.replace('_', ' ')}: {getattr(m, k):.0f}/10" for k in m.__dataclass_fields__)
        # Persist to product research memory (recallable offline, cross-session).
        self.memory.matrix.save(
            "product_research", self.memory._project_slug(name),
            f"score {score} ({verdict.split(' — ')[0]}) | {description[:160]}", silent=True)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("commerce.products_scored")
        self.bus.dashboard_event.emit("product_scored",
                                      {"name": name, "score": score, "verdict": verdict})
        grounding = ""
        if sig:
            grounding = (
                f"\nGrounded in real demand signal ({sig['source']}): "
                f"interest {sig['interest']:.0f}/10, trend {sig['trend']} "
                f"(momentum {sig['momentum']:.1f}/10).")
        return ToolResult(
            f"Product Opportunity Score for '{name}': {score}/100\n"
            f"Verdict: {verdict}\n"
            f"Metrics — {breakdown}\n"
            "(0-10 metrics; competition/shipping/returns/seasonality are risks — "
            "lower is better and already inverted in the score.)"
            + grounding
        )

    async def validate(self, name: str, description: str = "") -> ToolResult:
        score = await self.score_product(name, description)
        advice = await self._advise(
            f"Give a concise validation plan for testing the dropshipping product "
            f"'{name}' ({description}). Include: how to gauge demand and saturation, "
            f"a cheap test (organic vs paid), target margin maths, and clear "
            f"go/no-go criteria.",
            ground_query="dropshipping product validation saturation margin")
        return ToolResult(f"{score.text}\n\nValidation plan:\n{advice}")

    async def analyse_competition(self, niche: str) -> ToolResult:
        advice = await self._advise(
            f"Analyse the competitive landscape and market saturation for the niche "
            f"'{niche}' in dropshipping. Cover typical sellers, differentiation "
            f"angles, and whether it's early, heating, or saturated.",
            ground_query="competition saturation niche dropshipping")
        return ToolResult(f"Competition & saturation — {niche}:\n{advice}")

    def research_log(self, limit: int = 20) -> list[dict[str, str]]:
        rows = self.memory.records(limit=200)
        return [r for r in rows if r.get("category") == "product_research"][:limit]


# ──────────────────────────────────────────────────────────────────────────────
# TIKTOK SHOP AGENT (Phase 6)
# ──────────────────────────────────────────────────────────────────────────────

class TikTokShopAgent(_CommerceBase):
    persona = (
        "SPECIALIST MODE — TikTok Shop intelligence. You understand virality "
        "signals, UGC creative, affiliate/creator seeding and category timing. "
        "Be practical and current in your thinking."
    )

    async def trend_report(self, niche: str = "") -> ToolResult:
        advice = await self._advise(
            f"Produce a TikTok Shop trend report{' for ' + niche if niche else ''}. "
            "Cover: how to spot products heating up (virality signals + velocity), "
            "which categories are timely, and the creative angle that would sell now.",
            ground_query="tiktok shop virality signals trending category")
        return ToolResult(f"TikTok Shop trend report{(' — ' + niche) if niche else ''}:\n{advice}")

    async def product_report(self, product: str) -> ToolResult:
        advice = await self._advise(
            f"Assess the product '{product}' for TikTok Shop: virality potential, "
            "the hook/angle for UGC, affiliate-creator seeding approach, and risks.",
            ground_query="tiktok shop product virality UGC creator")
        return ToolResult(f"TikTok Shop product assessment — {product}:\n{advice}")

    def score_virality(self, signals: dict[str, Any]) -> ToolResult:
        """Deterministic virality velocity from supplied signals (offline)."""
        views = float(signals.get("avg_views", 0))
        creators = float(signals.get("creators_posting", 0))
        comment_intent = float(signals.get("buy_intent_comments", 0))  # 0-10
        video_velocity = float(signals.get("new_videos_per_day", 0))
        v = (
            min(1.0, views / 200_000) * 30
            + min(1.0, creators / 30) * 25
            + min(1.0, comment_intent / 10) * 20
            + min(1.0, video_velocity / 40) * 25
        )
        v = round(v, 1)
        band = ("EXPLODING" if v >= 75 else "HEATING" if v >= 55
                else "EMERGING" if v >= 35 else "FLAT")
        return ToolResult(f"Virality velocity: {v}/100 — {band}. "
                          f"(views {views:.0f}, creators {creators:.0f}, "
                          f"buy-intent {comment_intent:.0f}/10, {video_velocity:.0f} new videos/day)")


# ──────────────────────────────────────────────────────────────────────────────
# INSTAGRAM COMMERCE AGENT (Phase 7)
# ──────────────────────────────────────────────────────────────────────────────

class InstagramCommerceAgent(_CommerceBase):
    persona = (
        "SPECIALIST MODE — Instagram commerce intelligence. You understand reels "
        "virality, influencer partnerships, store/profile analysis and product "
        "discovery on Instagram."
    )

    async def discover(self, niche: str) -> ToolResult:
        advice = await self._advise(
            f"Suggest a product-discovery approach on Instagram for the niche "
            f"'{niche}': where to look (reels, hashtags, shops), what signals a "
            "rising product, and how to shortlist opportunities.",
            ground_query="instagram product discovery reels viral niche")
        return ToolResult(f"Instagram discovery — {niche}:\n{advice}")

    async def influencer_strategy(self, brand: str = "") -> ToolResult:
        brand = brand or user_profile.brand()
        advice = await self._advise(
            f"Recommend an influencer partnership strategy for {brand} on Instagram: "
            "tiers (nano/micro/mid), outreach, seeding vs paid, and how to measure ROI.",
            ground_query="influencer partnership instagram seeding ROI")
        return ToolResult(f"Influencer strategy — {brand}:\n{advice}")

    async def weekly_report(self, niche: str = "home products") -> ToolResult:
        advice = await self._advise(
            f"Draft a weekly Instagram commerce opportunity report for '{niche}': "
            "trend movements, opportunity rankings, and 3 concrete actions this week.",
            ground_query="instagram weekly opportunity report trend")
        return ToolResult(f"Instagram weekly report — {niche} ({datetime.now():%d %b %Y}):\n{advice}")


# ──────────────────────────────────────────────────────────────────────────────
# FOUNDER KNOWLEDGE AGENT (Phase 9)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FounderProfile:
    name: str
    domain: str
    strategies: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)


def _seed_founders() -> list[FounderProfile]:
    return [
        FounderProfile(
            "Alex Hormozi", "Offers & acquisition",
            ["Make a 'Grand Slam Offer' so good people feel stupid saying no",
             "Maximise value equation: dream outcome × perceived likelihood ÷ time delay ÷ effort"],
            ["Value Equation", "Core Four lead generation", "$100M Offers / Leads"],
            ["The offer beats the traffic; lower risk and raise value before spending more on ads"]),
        FounderProfile(
            "Gary Vaynerchuk", "Attention & organic content",
            ["Document, don't create — high-volume native content per platform",
             "Jab, jab, jab, right hook: give value repeatedly before asking"],
            ["Content pyramid", "Day trading attention"],
            ["Distribution and attention are the modern moat; be where attention is cheap"]),
        FounderProfile(
            "Sara Blakely (Spanx)", "Bootstrapped consumer product",
            ["Solve a personal problem; prototype cheaply; sell before scaling",
             "PR and story-led marketing over paid ads early"],
            ["Bootstrapping", "Founder-story marketing"],
            ["Constraints breed creativity; own the narrative and the product truth"]),
        FounderProfile(
            "Jeff Bezos (Amazon)", "Operations & long-term thinking",
            ["Customer obsession over competitor obsession",
             "Day 1 mentality; high-velocity, reversible decisions"],
            ["Working backwards (PR/FAQ)", "Two-pizza teams", "Flywheel"],
            ["Optimise for long-term free cash flow; lower prices and selection compound"]),
    ]


class FounderKnowledgeAgent(_CommerceBase):
    persona = "SPECIALIST MODE — analyst of entrepreneurial strategy and systems."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._founders = {f.name.lower(): f for f in _seed_founders()}

    def seed_into_memory(self) -> None:
        for f in self._founders.values():
            body = (f"{f.name} ({f.domain}). Strategies: {'; '.join(f.strategies)}. "
                    f"Frameworks: {', '.join(f.frameworks)}. Lessons: {'; '.join(f.lessons)}")
            self.memory.matrix.save("founder_kb", self.memory._project_slug(f.name), body, silent=True)

    def profile(self, name: str) -> ToolResult:
        key = name.lower().strip()
        match = next((f for k, f in self._founders.items() if key in k), None)
        if match is None:
            return ToolResult(
                f"No stored profile for '{name}'. Known: "
                + ", ".join(f.name for f in self._founders.values()) + ".")
        return ToolResult(
            f"{match.name} — {match.domain}\n"
            f"Strategies:\n" + "\n".join(f"- {s}" for s in match.strategies) + "\n"
            f"Frameworks: {', '.join(match.frameworks)}\n"
            f"Key lessons:\n" + "\n".join(f"- {l}" for l in match.lessons))

    async def learn_from(self, name: str, question: str = "") -> ToolResult:
        prof = self.profile(name)
        if not prof.ok:
            return prof
        advice = await self._advise(
            f"Using this founder profile, {question or 'extract the transferable playbook a small e-commerce founder could apply'}:\n{prof.text}",
            ground_query=f"{name} strategy framework")
        return ToolResult(f"{prof.text}\n\nApplied analysis:\n{advice}")

    def list_founders(self) -> list[str]:
        return [f.name for f in self._founders.values()]


# ──────────────────────────────────────────────────────────────────────────────
# BUSINESS ADVISOR AGENT (Phase 11)
# ──────────────────────────────────────────────────────────────────────────────

class BusinessAdvisorAgent(_CommerceBase):
    persona = (
        "SPECIALIST MODE — personal business advisor for the user's brand "
        "described in the supplied brand context. You give strategy on brand, product, marketing, store "
        "optimisation, conversion, paid and organic growth. Be specific, "
        "prioritised and honest; tie advice to the configured brand and niche."
    )

    @property
    def BRAND_CONTEXT(self) -> str:
        return (f"[BRAND: {user_profile.brand()} — {user_profile.brand_niche()}. "
                "Channels: TikTok Shop, Instagram commerce, dropshipping. "
                "Goal: profitable growth.]")

    async def advise(self, topic: str) -> ToolResult:
        # Surface a real demand read up front so even the deterministic MODE B
        # answer is grounded in evidence, not just the model's priors.
        sig = await self._demand_signal(topic)
        advice = await self._advise(
            f"{self.BRAND_CONTEXT}\nAdvise on: {topic}. Give 3-6 prioritised, "
            "specific recommendations with the reasoning and the first action.",
            ground_query=topic)
        body = f"Business advice — {topic}:\n{advice}"
        if sig:
            body = f"{self._demand_note(sig, topic)}\n\n{body}"
        return ToolResult(body)

    async def brand_strategy(self) -> ToolResult:
        return await self.advise(f"{user_profile.brand()} brand strategy and positioning in {user_profile.brand_niche()}")

    async def growth_plan(self, horizon: str = "next 90 days") -> ToolResult:
        return await self.advise(f"a growth plan for {user_profile.brand()} for the {horizon} across product, content and paid")

    async def store_optimisation(self) -> ToolResult:
        return await self.advise(f"conversion-rate optimisation for the {user_profile.brand()} store (offer, page, trust, AOV)")


# ──────────────────────────────────────────────────────────────────────────────
# PRODUCT RESEARCH AGENT (Mark X.5)
# ──────────────────────────────────────────────────────────────────────────────

class ProductResearchAgent(DropshippingResearchAgent):
    """
    The Entrepreneurial Intelligence Division's discovery arm: product
    discovery, market validation, saturation analysis and opportunity scoring.
    Builds on the dropshipping analyst (scoring/validation/saturation are
    shared machinery) and adds structured discovery whose findings persist to
    long-term memory for cross-session recall.
    """

    persona = (
        "SPECIALIST MODE — Product research analyst. You discover and validate "
        "e-commerce product opportunities with discipline: demand evidence, "
        "saturation read, margin maths and a clear test-or-pass verdict."
    )

    async def discover(self, niche: str, count: int = 5) -> ToolResult:
        count = max(1, min(10, int(count or 5)))
        advice = await self._advise(
            f"Propose {count} concrete product opportunities in the niche "
            f"'{niche}'. For each: the product, who buys it and why now, an "
            "estimated saturation read (early/heating/saturated), and the "
            "first cheap validation step.",
            ground_query=f"product discovery {niche} demand saturation")
        self.memory.matrix.save(
            "product_research", self.memory._project_slug(f"discovery {niche}"),
            f"discovery run {datetime.now():%Y-%m-%d} | {advice[:220]}", silent=True)
        self.bus.dashboard_event.emit(
            "product_research", {"niche": niche, "kind": "discovery"})
        return ToolResult(f"Product discovery — {niche}:\n{advice}")

    async def market_validation(self, product: str, description: str = "") -> ToolResult:
        """Score + validation plan in one pass (the spec's validation surface)."""
        return await self.validate(product, description)


# ──────────────────────────────────────────────────────────────────────────────
# COMPETITOR INTELLIGENCE AGENT (Mark X.5)
# ──────────────────────────────────────────────────────────────────────────────

class CompetitorIntelligenceAgent(_CommerceBase):
    persona = (
        "SPECIALIST MODE — Competitor intelligence analyst. You dissect rival "
        "stores, offers and funnels: positioning, pricing architecture, offer "
        "stack, traffic plays, checkout friction and retention hooks. Be "
        "specific and evidence-led; end with what to copy, beat or avoid."
    )

    async def store_analysis(self, store: str) -> ToolResult:
        advice = await self._advise(
            f"Analyse the competitor store '{store}': positioning, product "
            "range architecture, pricing, trust signals, and the two clearest "
            "weaknesses a challenger brand could exploit.",
            ground_query="competitor store analysis positioning pricing")
        self._persist("store", store, advice)
        return ToolResult(f"Store analysis — {store}:\n{advice}")

    async def offer_analysis(self, competitor: str, offer: str = "") -> ToolResult:
        advice = await self._advise(
            f"Break down the offer{' ' + offer if offer else ''} from "
            f"'{competitor}': value stack, price anchoring, risk reversal, "
            "urgency mechanics, and how to construct a superior offer against it.",
            ground_query="offer analysis value stack risk reversal")
        self._persist("offer", competitor, advice)
        return ToolResult(f"Offer analysis — {competitor}:\n{advice}")

    async def funnel_analysis(self, competitor: str) -> ToolResult:
        advice = await self._advise(
            f"Map the likely sales funnel of '{competitor}' from first touch "
            "to post-purchase: traffic sources, lander style, upsell points, "
            "email/SMS flows, and where prospects leak out.",
            ground_query="funnel analysis traffic lander upsell retention")
        self._persist("funnel", competitor, advice)
        return ToolResult(f"Funnel analysis — {competitor}:\n{advice}")

    def _persist(self, kind: str, subject: str, advice: str) -> None:
        # Findings live in long-term memory so MODE B can recall them.
        self.memory.matrix.save(
            "competitor_intel", self.memory._project_slug(f"{kind} {subject}"),
            f"{kind} analysis {datetime.now():%Y-%m-%d} | {advice[:220]}", silent=True)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("commerce.competitor_analyses")


# ──────────────────────────────────────────────────────────────────────────────
# BRAND GROWTH AGENT (Mark X.5)
# ──────────────────────────────────────────────────────────────────────────────

class BrandGrowthAgent(_CommerceBase):
    persona = (
        "SPECIALIST MODE — Brand growth strategist for the user's configured brand. You own "
        "strategy, conversion optimisation, product positioning and retention "
        "systems. Prioritised, specific, honest — every recommendation names "
        "its first action."
    )

    #: Publishable default; config/profile.json carries the real brand.
    BRAND = user_profile.brand()

    async def strategy(self, focus: str = "") -> ToolResult:
        advice = await self._advise(
            f"Set out the {self.BRAND} growth strategy"
            + (f" with focus on {focus}" if focus else "")
            + ": positioning in the home-products market, channel priorities, "
              "and the 3 moves with the highest expected return this quarter.",
            ground_query=f"{self.BRAND} brand strategy home products")
        self._persist("strategy", focus or "general", advice)
        return ToolResult(f"{self.BRAND} strategy:\n{advice}")

    async def conversion_optimisation(self, page: str = "store") -> ToolResult:
        advice = await self._advise(
            f"Give a conversion-optimisation plan for the {self.BRAND} {page}: "
            "offer clarity, social proof, page speed, AOV levers (bundles, "
            "thresholds), and checkout friction — ranked by expected lift.",
            ground_query="conversion rate optimisation offer AOV checkout")
        self._persist("conversion", page, advice)
        return ToolResult(f"Conversion optimisation — {page}:\n{advice}")

    async def positioning(self, product: str) -> ToolResult:
        advice = await self._advise(
            f"Position the product '{product}' for {self.BRAND}: the customer, "
            "the problem framing, the angle that differentiates it from generic "
            "listings, and the one-line promise for ads and the product page.",
            ground_query=f"product positioning angle {product}")
        self._persist("positioning", product, advice)
        return ToolResult(f"Positioning — {product}:\n{advice}")

    async def retention_systems(self) -> ToolResult:
        advice = await self._advise(
            f"Design retention systems for {self.BRAND}: post-purchase email/SMS "
            "flows, replenishment or cross-sell logic for home products, loyalty "
            "mechanics, and the metrics that prove each loop works.",
            ground_query="retention email flows loyalty repeat purchase")
        self._persist("retention", "systems", advice)
        return ToolResult(f"{self.BRAND} retention systems:\n{advice}")

    def _persist(self, kind: str, subject: str, advice: str) -> None:
        self.memory.matrix.save(
            "brand_growth", self.memory._project_slug(f"{kind} {subject}"),
            f"{kind} {datetime.now():%Y-%m-%d} | {advice[:220]}", silent=True)
        if self.telemetry is not None:
            self.telemetry.metrics.incr("commerce.brand_growth_runs")


# ──────────────────────────────────────────────────────────────────────────────
# SUITE CONTAINER
# ──────────────────────────────────────────────────────────────────────────────

class CommerceSuite:
    """Bundles the entrepreneurial agents behind one object for the dispatcher."""

    def __init__(self, bus: OrionBus, memory: MemoryAgent, router: Any,
                 knowledge: Any | None = None, telemetry: Any | None = None,
                 signals: "CommerceSignalProvider | None" = None) -> None:
        # One shared demand-signal provider grounds every agent's advice; built
        # by default so the grounding is on out of the box, injectable for tests.
        if signals is None:
            try:
                signals = CommerceSignalProvider(COMMERCE_SIGNAL_DB)
            except Exception:
                signals = None
        self.signals = signals
        common = (bus, memory, router)
        kw = {"knowledge": knowledge, "telemetry": telemetry, "signals": signals}
        self.dropship = DropshippingResearchAgent(*common, **kw)
        self.tiktok = TikTokShopAgent(*common, **kw)
        self.instagram = InstagramCommerceAgent(*common, **kw)
        self.founder = FounderKnowledgeAgent(*common, **kw)
        self.advisor = BusinessAdvisorAgent(*common, **kw)
        # Mark X.5 — Entrepreneurial Intelligence Division additions.
        self.product = ProductResearchAgent(*common, **kw)
        self.competitor = CompetitorIntelligenceAgent(*common, **kw)
        self.growth = BrandGrowthAgent(*common, **kw)
        # Seed founder profiles into memory once (idempotent save).
        try:
            self.founder.seed_into_memory()
        except Exception:
            pass
