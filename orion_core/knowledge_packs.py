"""
Knowledge Pack system (Phase 3) — installable, offline-consultable expertise.

A Knowledge Pack is a curated, versioned bundle of domain entries ORION can
search while completely offline.  Packs live as JSON on disk
(``config/knowledge_packs/<id>.json``) and are indexed into the MemoryAgent's
KNOWLEDGE tier (FTS5) under a ``pack_<id>`` category, so retrieval is fast and
uses the same search the rest of ORION already relies on.

The user can install, update, remove and expand packs.  Ten substantive
built-in packs ship with ORION and are seeded once (idempotent).  Retrieval is
grounded, deterministic and needs no internet — and doubles as context for the
local LLM when it answers business questions in MODE B.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .memory import MemoryAgent
from .utils import first_line, utc_stamp

PACKS_DIR = CONFIG_DIR / "knowledge_packs"


@dataclass
class KnowledgeEntry:
    topic: str
    content: str
    tags: list[str] = field(default_factory=list)


@dataclass
class KnowledgePack:
    id: str
    title: str
    description: str
    version: str = "1.0.0"
    entries: list[KnowledgeEntry] = field(default_factory=list)
    installed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "description": self.description,
            "version": self.version, "installed_at": self.installed_at,
            "entries": [{"topic": e.topic, "content": e.content, "tags": e.tags}
                        for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgePack":
        entries = [KnowledgeEntry(topic=e.get("topic", ""), content=e.get("content", ""),
                                  tags=list(e.get("tags", []))) for e in data.get("entries", [])]
        return cls(
            id=str(data.get("id", "")), title=str(data.get("title", "")),
            description=str(data.get("description", "")), version=str(data.get("version", "1.0.0")),
            entries=entries, installed_at=str(data.get("installed_at", "")),
        )


class KnowledgePackManager:
    """Install / update / remove / expand / search curated knowledge packs."""

    SEED_MARKER = "knowledge_packs_seeded_v1"

    def __init__(self, bus: OrionBus, memory: MemoryAgent, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.telemetry = telemetry
        PACKS_DIR.mkdir(parents=True, exist_ok=True)
        self._packs: dict[str, KnowledgePack] = {}
        self._load_from_disk()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        for path in PACKS_DIR.glob("*.json"):
            try:
                pack = KnowledgePack.from_dict(json.loads(path.read_text(encoding="utf-8")))
                if pack.id:
                    self._packs[pack.id] = pack
            except Exception as exc:
                self.bus.log.emit(f"PACK: could not load {path.name} - {first_line(exc)}")

    def seed_builtin(self) -> None:
        """Install the built-in packs once (marker-guarded, idempotent)."""
        marker = self.memory.recall(  # cheap check via KNOWLEDGE tier
            "knowledge", self.SEED_MARKER, limit=1)
        already = any(r.get("key_ref") == self.SEED_MARKER for r in marker)
        for pack in _BUILTIN_PACKS():
            if pack.id not in self._packs:
                self.install(pack, silent=True)
        if not already:
            self.memory.remember_knowledge(self.SEED_MARKER, datetime.now().isoformat())
        self.bus.log.emit(f"PACK: {len(self._packs)} knowledge pack(s) available offline.")

    def install(self, pack: KnowledgePack | dict[str, Any], silent: bool = False) -> ToolResult:
        if isinstance(pack, dict):
            pack = KnowledgePack.from_dict(pack)
        if not pack.id:
            return ToolResult("Pack has no id.", ok=False)
        pack.installed_at = utc_stamp()
        self._packs[pack.id] = pack
        self._persist(pack)
        self._index(pack)
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("packs.installed", float(len(self._packs)))
        if not silent:
            self.bus.log.emit(f"PACK: installed '{pack.title}' ({len(pack.entries)} entries).")
        return ToolResult(f"Installed knowledge pack '{pack.title}' with {len(pack.entries)} entries.")

    def update(self, pack: KnowledgePack | dict[str, Any]) -> ToolResult:
        if isinstance(pack, dict):
            pack = KnowledgePack.from_dict(pack)
        existing = self._packs.get(pack.id)
        if existing is None:
            return self.install(pack)
        return self.install(pack)  # install overwrites + re-indexes

    def expand(self, pack_id: str, entries: list[dict[str, Any]]) -> ToolResult:
        pack = self._packs.get(pack_id)
        if pack is None:
            return ToolResult(f"No installed pack '{pack_id}'.", ok=False)
        added = 0
        for e in entries:
            topic = str(e.get("topic") or "").strip()
            content = str(e.get("content") or "").strip()
            if topic and content:
                pack.entries.append(KnowledgeEntry(topic, content, list(e.get("tags", []))))
                added += 1
        self._persist(pack)
        self._index(pack)
        return ToolResult(f"Expanded '{pack.title}' with {added} new entry(ies).")

    def remove(self, pack_id: str) -> ToolResult:
        pack = self._packs.pop(pack_id, None)
        if pack is None:
            return ToolResult(f"No installed pack '{pack_id}'.", ok=False)
        try:
            (PACKS_DIR / f"{pack_id}.json").unlink(missing_ok=True)
        except Exception:
            pass
        return ToolResult(f"Removed knowledge pack '{pack.title}'. "
                          "(Indexed entries remain searchable until the store is rebuilt.)")

    # ── retrieval ─────────────────────────────────────────────────────────────

    def list_packs(self) -> list[dict[str, Any]]:
        return [{"id": p.id, "title": p.title, "version": p.version,
                 "entries": len(p.entries), "description": p.description}
                for p in sorted(self._packs.values(), key=lambda p: p.title)]

    def search(self, query: str, limit: int = 5, pack_id: str = "") -> list[dict[str, str]]:
        """Rank entries across installed packs by keyword overlap (offline)."""
        query = str(query or "").strip().lower()
        if not query:
            return []
        terms = set(re.findall(r"[a-z0-9]+", query))
        scored: list[tuple[float, str, KnowledgeEntry]] = []
        for pid, pack in self._packs.items():
            if pack_id and pid != pack_id:
                continue
            for entry in pack.entries:
                haystack = f"{entry.topic} {entry.content} {' '.join(entry.tags)}".lower()
                hay_terms = set(re.findall(r"[a-z0-9]+", haystack))
                overlap = len(terms & hay_terms)
                if query in haystack:
                    overlap += 3
                if entry.topic.lower() in query or query in entry.topic.lower():
                    overlap += 4
                if overlap:
                    scored.append((overlap, pack.title, entry))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [{"pack": title, "topic": e.topic, "content": e.content}
                for _, title, e in scored[:limit]]

    def consult(self, query: str, limit: int = 4) -> ToolResult:
        hits = self.search(query, limit=limit)
        if not hits:
            return ToolResult(
                f"No knowledge-pack entry matches '{query}'. Installed packs: "
                + ", ".join(p["title"] for p in self.list_packs()) + ".")
        lines = [f"Knowledge on '{query}':"]
        for h in hits:
            lines.append(f"\n[{h['pack']} — {h['topic']}]\n{h['content']}")
        return ToolResult("\n".join(lines))

    def context_for(self, query: str, limit: int = 3, max_chars: int = 1800) -> str:
        """Compact grounding block for the local/cloud LLM to reason over."""
        hits = self.search(query, limit=limit)
        if not hits:
            return ""
        block = ["[KNOWLEDGE PACK CONTEXT — ground your answer in this]"]
        for h in hits:
            block.append(f"- {h['topic']} ({h['pack']}): {h['content']}")
        return "\n".join(block)[:max_chars]

    # ── persistence + indexing ────────────────────────────────────────────────

    def _persist(self, pack: KnowledgePack) -> None:
        try:
            (PACKS_DIR / f"{pack.id}.json").write_text(
                json.dumps(pack.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            self.bus.log.emit(f"PACK: persist failed for {pack.id} - {first_line(exc)}")

    def _index(self, pack: KnowledgePack) -> None:
        """Index each entry into the KNOWLEDGE tier for FTS retrieval."""
        for i, entry in enumerate(pack.entries):
            try:
                self.memory.matrix.save(
                    f"pack_{pack.id}", f"{i:03d}_{self._slug(entry.topic)}",
                    f"{entry.topic}: {entry.content}", silent=True)
            except Exception:
                continue

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", str(text).strip().lower()).strip("_")[:48] or "entry"


# ──────────────────────────────────────────────────────────────────────────────
# BUILT-IN PACKS  — substantive, curated, offline-first
# ──────────────────────────────────────────────────────────────────────────────

def _pack(pid: str, title: str, desc: str, entries: list[tuple[str, str]]) -> KnowledgePack:
    return KnowledgePack(id=pid, title=title, description=desc, version="1.0.0",
                         entries=[KnowledgeEntry(t, c) for t, c in entries])


def _BUILTIN_PACKS() -> list[KnowledgePack]:
    return [
        _pack("entrepreneurship", "Entrepreneurship Pack",
              "Foundational frameworks for starting and growing ventures.", [
            ("Lean startup loop", "Build–Measure–Learn: ship a minimum viable product, measure real user behaviour, learn, iterate. Validate demand before scaling spend. Prefer pre-selling and landing-page smoke tests over building the full product."),
            ("Problem-solution fit", "Confirm a painful, frequent, expensive problem for a specific segment before product-market fit. Interview 20+ target customers; look for pull (they ask to buy) not politeness."),
            ("Unit economics", "Track CAC (cost to acquire a customer), LTV (lifetime value), contribution margin, and payback period. A healthy LTV:CAC is >=3:1 with payback under 3 months for e-commerce."),
            ("Moats", "Durable advantage comes from brand, network effects, switching costs, proprietary data, or economies of scale. In dropshipping the moat is usually brand + operations + creative velocity, not the product itself."),
        ]),
        _pack("dropshipping", "Dropshipping Pack",
              "Product research, validation and operations for dropshipping.", [
            ("Winning product criteria", "Solves a clear problem or has strong wow factor; hard to find in local shops; 2.5-4x markup possible; light and durable (cheap shipping, low breakage); broad or passionate audience; not restricted by ad platforms."),
            ("Validation before scale", "Test with a small ad budget or organic video first. Look for saturation on TikTok/Meta, existing sellers' review counts, and Google Trends direction. Kill fast if CTR and CVR are weak."),
            ("Margin maths", "Selling price - product cost - shipping - transaction fees - ad cost per sale = profit. Aim for product cost <= 30% of selling price so ad spend has room. Break-even ROAS = 1 / margin fraction."),
            ("Fulfilment risk", "Long shipping times and returns erode trust. Prefer suppliers with tracked 7-12 day shipping, or local/agent fulfilment for scaling winners. Track return rate as a core health metric."),
        ]),
        _pack("tiktok_shop", "TikTok Shop Pack",
              "Trend spotting, creative and selling on TikTok Shop.", [
            ("Virality signals", "A product is heating up when multiple mid-size creators post organically, videos cross 100k views fast, comments ask 'where to buy', and duet/stitch volume rises. Velocity of new videos matters more than any single viral hit."),
            ("Content that sells", "Native, un-polished UGC beats ads. Hooks in the first 1.5s; show the problem then the product resolving it; demonstrate, don't describe. Post 3-5+ variations daily; let the algorithm find the winner."),
            ("Affiliate + Shop", "Enable TikTok Shop affiliate so creators sell for you on commission. Seed product to 20-50 micro-creators; a single creator break-out can carry a product. Track GMV per creator."),
            ("Category timing", "Ride emerging niches early (before saturation): home gadgets, beauty tools, viral snacks, organisation, pet. Once a product is everywhere, margins and CTR collapse — rotate."),
        ]),
        _pack("marketing", "Marketing Pack",
              "Positioning, channels, funnels and growth.", [
            ("Positioning", "Own a specific position in the customer's mind: who it's for, the one problem it solves, why it's different. Confused buyers don't buy. Lead with the transformation, not the features."),
            ("AIDA + funnel", "Attention, Interest, Desire, Action. Map content to funnel stage: top (reach/hook), middle (value/proof), bottom (offer/urgency). Retarget warm audiences — they convert cheapest."),
            ("Creative testing", "Ads are won at the creative level, not the settings. Test hooks, angles and formats broadly; scale winners; kill losers within 2-3 days. 80% of results come from ~20% of creatives."),
            ("Offer > everything", "A strong offer (bundle, guarantee, urgency, bonus) beats clever targeting. Reduce risk (money-back), increase value (free gift), add urgency (limited) — the offer is the biggest lever on conversion."),
        ]),
        _pack("copywriting", "Copywriting Pack",
              "Persuasive writing for ads, pages and email.", [
            ("Hooks", "The first line does 80% of the work. Use curiosity, a bold claim, a relatable pain, or a pattern interrupt. If the hook fails, nothing else is read."),
            ("PAS framework", "Problem — agitate the pain — solution. Name the reader's problem precisely, twist the knife on the consequences, then present your product as the relief."),
            ("Features vs benefits", "Sell the outcome. 'Non-slip base' → 'stays put so it never spills on your counter'. Every feature should answer 'so what does that do for me?'"),
            ("Clarity + specifics", "Specific beats vague: numbers, timeframes, concrete images. Short sentences. One idea per line. Read it aloud — if you stumble, rewrite."),
        ]),
        _pack("sales_psychology", "Sales Psychology Pack",
              "Behavioural principles behind buying decisions.", [
            ("Cialdini's principles", "Reciprocity, commitment/consistency, social proof, authority, liking, scarcity, unity. Reviews and UGC = social proof; limited stock = scarcity; expert/founder story = authority."),
            ("Loss aversion", "People feel losses ~2x as strongly as equivalent gains. Frame around what they'll miss, not only what they'll gain. Guarantees remove the felt risk of loss and lift conversion."),
            ("Decision friction", "Every extra field, click, or unclear price loses buyers. Reduce choices (paradox of choice), default the best option, and make the next step obvious."),
            ("Anchoring", "The first number sets the reference. Show a higher 'compare at' price, or a premium tier, so the target offer feels like a deal."),
        ]),
        _pack("business", "Business Pack",
              "Operations, finance and strategy for small businesses.", [
            ("Cash flow is king", "Profit is opinion, cash is fact. Track runway, receivables and inventory tied-up cash. Many growing e-commerce brands die from cash starvation while 'profitable' on paper."),
            ("Pricing strategy", "Price on value and positioning, not cost-plus. Test price points; premium pricing can increase perceived quality and margin. Bundles raise average order value."),
            ("Systems + SOPs", "Document repeatable processes so the business runs without you. What gets measured and documented can be delegated and scaled. Dashboards over gut feel."),
            ("Focus", "One product line, one channel, one avatar until it works. Diversification before traction dilutes energy. Do fewer things, better, first."),
        ]),
        _pack("coding", "Coding Pack",
              "Pragmatic software engineering practice.", [
            ("Simplicity", "Prefer the simplest solution that works. Optimise for readability and change; clever code is a liability. YAGNI — don't build what you don't yet need."),
            ("Debugging method", "Reproduce reliably, read the actual error, form one hypothesis, test it, change one thing at a time. Bisect to isolate. The bug is usually where you're most sure it isn't."),
            ("Testing", "Test behaviour, not implementation. Cover the highest-impact paths and edge cases first. A failing test that reproduces a bug is the fastest way to fix it and prevent regression."),
            ("Architecture", "Low coupling, high cohesion. Depend on abstractions. Keep the dependency graph acyclic and the blast radius of change small. Name things for what they mean."),
        ]),
        _pack("ai", "Artificial Intelligence Pack",
              "Practical LLM and AI product knowledge.", [
            ("LLM basics", "Large language models predict the next token from context. Quality depends on the prompt, the context provided (retrieval), and the model. They don't 'know' facts reliably — ground them with data."),
            ("RAG", "Retrieval-Augmented Generation: fetch relevant documents and put them in the prompt so the model answers from your data, reducing hallucination. ORION uses this via its knowledge packs and memory."),
            ("Local vs cloud", "Local models (Ollama: Qwen, Llama, DeepSeek, Mistral) run offline, free and private, at lower peak capability. Cloud models are stronger for hard reasoning. Route by task complexity and connectivity."),
            ("Prompt engineering", "Be specific about role, task, format and constraints. Give examples (few-shot). Ask for step-by-step reasoning on hard tasks. Provide grounding context rather than relying on model memory."),
        ]),
        _pack("personal_development", "Personal Development Pack",
              "Focus, habits and execution for founders.", [
            ("Habits", "Systems beat goals. Make the good behaviour obvious, easy, and satisfying; shrink it until it's too small to fail, then let it grow. Environment design beats willpower."),
            ("Deep work", "Protect long, uninterrupted blocks for high-value work. Batch shallow tasks. Single-task — context switching is a silent productivity tax. Energy management matters as much as time."),
            ("Bias to action", "Speed of execution compounds. Ship, measure, adjust. Perfectionism is procrastination in disguise. Most decisions are reversible — make them fast."),
            ("Resilience", "Reframe failure as data. Detach identity from outcomes. Consistent effort over long horizons, plus quick recovery from setbacks, is what separates operators who win."),
        ]),
    ]
