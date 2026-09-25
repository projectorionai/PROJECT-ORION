"""
ToolResolver (Mark XXVI, Phase 2) — shrink the 132-tool surface to the handful a
single turn actually needs, deterministically and offline.

Flat-routing every tool in one prompt costs context and erodes instruction
adherence. This module computes, for a given query and runtime state, a *subset*
of the tool declarations to expose to the reasoner. It is deliberately:

  * **pure** — ``resolve(query, state, declarations) -> list[declaration]`` is a
    free function with no I/O, no Qt, and no import-time side effects, so it is
    trivially unit-testable and safe to call from the qasync loop;
  * **provably regression-free** — with ``enabled=False`` (the default, and what
    the ``ORION_TOOL_RESOLVER`` flag gates) it returns the input declarations
    unchanged, so wiring it in is a no-op until the flag is set;
  * **offline-first** — the base scorer is a dependency-free lexical (IDF)
    ranker. ``hybrid_scores`` adds sentence meaning (semantic.py) behind the
    same ``Scorer`` seam when the local encoder is warm, and IS the lexical
    scorer when it is not. Measured on 40 new everyday requests for tools the
    golden set never covered: recall within 25 tools 97.5% -> 100%, within
    10 tools 92.5% -> 97.5% (docs/ORION_TOOL_ROUTING.md).

It is a *pre-filter*, never a gate: the full ``handler_table`` and
``TOOL_DECLARATIONS`` remain intact and individually dispatchable. A safety floor
(``ALWAYS_INCLUDE``) guarantees the tools needed to recover or re-orient are never
filtered out.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

#: Hard ceiling on the resolved set — the point of the exercise.
MAX_TOOLS = 25

#: Never filtered out: the tools that let ORION re-orient or reason regardless of
#: what the query looked like. A resolver that hid these could strand a turn.
ALWAYS_INCLUDE: tuple[str, ...] = ("catch_up", "reason", "capabilities")

#: Tools that cannot function without the internet AND have no offline value, so
#: they are dropped in MODE B / when offline. Kept deliberately conservative:
#: anything that can report a useful offline result stays visible. Notably
#: ``security_recon`` is NOT here — 'authorize_target' is one of its actions, so
#: hiding it when no target is authorised would make authorising impossible.
CLOUD_ONLY: frozenset[str] = frozenset({"web_search", "open_news"})

# Context boost weights (added on top of the lexical score).
BOOST_PAGE = 3.0        # a tool in the foreground deck page's domain
BOOST_FOCUS = 4.0       # study/focus while a focus block is running
BOOST_RECENT = 1.5      # a tool used earlier this turn-sequence
#: Tie-breaker only — see the exactness bonus in lexical_scores.
NAME_EXACTNESS_BONUS = 0.5

def vocabulary_for(tool: str) -> str:
    """Curated invoking phrases for a tool, or "" — never raises.

    Indirected through a function rather than imported as a dict so a caller
    (or a test) can measure the resolver with the expansion switched off, and so
    a missing module degrades to the old behaviour instead of breaking routing.
    """
    try:
        from .tool_vocabulary import terms_for
        return terms_for(tool)
    except Exception:
        return ""


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of to in on at for and or but is are was were be it its this that "
    "with from into about over under as by you your i me my we our so if then "
    "run get set show tell make do use one".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


# ── runtime state ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResolverState:
    """The context that shapes resolution. All optional; an empty state resolves
    purely on the query."""

    mode: str = "A"                       # "A" cloud-enhanced, "B" offline
    online: bool = True
    active_page: str = ""                 # foreground deck page (domain boost)
    focus_active: bool = False            # a focus block is running
    auth_targets: bool = False            # any authorised security target
    recent_tools: tuple[str, ...] = ()    # tools used earlier this exchange


# ── domain classification (page boosts today; facade grouping tomorrow) ───────

#: name -> domain. The same grouping the Phase-2 capability facades will use, so
#: this map is the single source of truth for "which domain owns this tool".
_TOOL_DOMAIN: dict[str, str] = {}


def _register(domain: str, *names: str) -> None:
    for n in names:
        _TOOL_DOMAIN[n] = domain


_register("learning", "study", "focus", "language_tutor", "expand_mind",
          "neuro_knowledge", "programming_knowledge", "cyber_knowledge",
          "cyber_curriculum", "knowledge_pack", "literature_vault")
_register("memory", "save_memory", "query_intelligence", "recall_conversation",
          "conversation_recall", "rewind", "second_brain", "ingest", "learn",
          "transcript", "awareness", "companion", "catch_up")
_register("research", "research", "standing_questions", "proactive_check",
          "proactive_report", "read_documents")
_register("vision", "vision_analyse", "vision_verify", "capture_screen",
          "screen_read", "perception")
_register("desktop", "open_app", "close_app", "window_control", "media_control",
          "desktop_control", "peripherals", "gesture_control", "clipboard_operate",
          "process_governor", "process_file", "file_controller", "find_files",
          "display_info", "system_notify", "interface_control", "workspace_control",
          "cursor_overlay", "organise_files")
_register("web", "browser_control", "web_control", "web_automation",
          "social_media", "muscle_memory", "web_search", "fetch_url", "open_news",
          "flight_search")
_register("engineering", "dev_workbench", "codebase_copilot", "debugger", "docker",
          "forge", "self_repair", "diagnostics", "code_changes", "self_changes",
          "cleanup_review")
_register("automation", "execute_plan", "autoplan", "workflow", "workflow_patterns",
          "job", "mission", "skill", "protocol", "agent_dispatch", "reason",
          "strategy", "perception")
_register("business", "product_research", "tiktok_intel", "instagram_intel",
          "commerce_hub", "competitor_intel", "brand_growth", "business_advisor",
          "founder_knowledge", "creator_intel", "campaign_pipeline",
          "community_share", "commerce_hub")
_register("security", "security_recon", "security_watch", "breach_check",
          "antivirus", "privacy_guard", "sentinel", "pentest_lab")
_register("creative", "audio_studio", "draft_report", "document_export",
          "entertainment", "gaming")
_register("comms", "messaging", "outlook_mail", "notion_workspace", "phone_action")
_register("geo", "globe", "max_zoom_in_globe", "geo", "aviation")
_register("system", "shutdown_orion", "restart_orion", "system_startup",
          "patch_notes", "capabilities", "token_usage", "resource_status",
          "ai_mode", "backup", "reminder", "morning_briefing", "briefing",
          "executive", "momentum", "mcp")
_register("voice", "voice_tone", "speaker_id", "voice_speaker_id",
          "elevenlabs_voice", "audio_devices", "emotion", "avatar")
_register("games", "chess")
_register("finance", "finance")
_register("wellbeing", "wellbeing")

#: deck page -> the domain whose tools it should boost while it is foreground.
_PAGE_DOMAIN: dict[str, str] = {
    "COGNITION": "learning", "MEMORY": "memory", "RESEARCH": "research",
    "LIBRARY": "memory", "SECURITY": "security", "DEVELOPMENT": "engineering",
    "MARKETING": "business", "TOOLKIT": "business",
    "WORKBENCH": "vision", "GLOBE": "geo", "CHESS": "games",
    "AUTOMATION": "automation", "DIAGNOSTICS": "system", "LOG": "system",
    "TELEMETRY": "system", "OPS": "system", "WIDGETS": "comms",
    "MISSION": "automation",
}


def classify_tool(name: str) -> str:
    """The domain a tool belongs to, or 'general' if unclassified."""
    return _TOOL_DOMAIN.get(name, "general")


# ── scoring ───────────────────────────────────────────────────────────────────

Scorer = Callable[[str, Sequence[dict[str, Any]]], dict[str, float]]


class _LexicalIndex:
    """A pre-built inverted index over the tool declarations.

    The old scorer re-tokenised all 136 declarations and rebuilt the document
    frequency table on EVERY call — 4.34 ms per turn, on the qasync loop, to
    recompute something that only changes when a plugin adds a tool. This is the
    same arithmetic, done once: term -> [(tool, weight)] postings plus a cached
    IDF, so scoring costs one dict lookup per query term instead of a full scan.

    Identical output to the original by construction: the postings carry the
    same 2.0 name-token weighting, terms are de-duplicated per document exactly
    as the old set-membership test did, and IDF uses the same formula.
    """

    __slots__ = ("postings", "idf", "names", "pure_name", "fingerprint")

    def __init__(self, declarations: Sequence[dict[str, Any]]) -> None:
        docs: dict[str, set[str]] = {}
        name_terms: dict[str, set[str]] = {}
        df: dict[str, int] = {}
        for d in declarations:
            name = d["name"]
            own = set(_tokens(name.replace("_", " ")))
            # Curated invoking vocabulary is weighted like the tool's own name,
            # because that is what it is: the words that mean "this tool". See
            # tool_vocabulary — without it, recall was 76.5%, because terms like
            # "weather" and "quiz" appear nowhere in the schema at all.
            own |= set(_tokens(vocabulary_for(name)))
            terms = own | set(_tokens(d.get("description", "")))
            docs[name] = terms
            name_terms[name] = own
        # df counted over the same per-declaration term sets as before
        for d in declarations:
            for t in docs[d["name"]]:
                df[t] = df.get(t, 0) + 1

        n = max(1, len(declarations))
        postings: dict[str, list[tuple[str, float]]] = {}
        for name, terms in docs.items():
            own = name_terms[name]
            for t in terms:
                postings.setdefault(t, []).append((name, 2.0 if t in own else 1.0))

        self.postings = {t: tuple(v) for t, v in postings.items()}
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.names = tuple(d["name"] for d in declarations)
        # PURE name tokens, vocabulary excluded — used only for the exactness
        # bonus below, where the question is how much of the tool's own NAME the
        # query accounted for.
        self.pure_name = {d["name"]: frozenset(_tokens(d["name"].replace("_", " ")))
                          for d in declarations}
        self.fingerprint = _fingerprint(declarations)


def _fingerprint(declarations: Sequence[dict[str, Any]]) -> int:
    """Cheap identity for a declaration set — names plus count.

    Descriptions are not hashed: they are static per tool, and hashing 136 of
    them every turn would reintroduce a slice of the cost this index removes.
    A plugin that adds, removes or renames a tool changes this.
    """
    return hash((len(declarations), tuple(d.get("name", "") for d in declarations)))


_INDEX: _LexicalIndex | None = None


def _index_for(declarations: Sequence[dict[str, Any]]) -> _LexicalIndex:
    global _INDEX
    if _INDEX is None or _INDEX.fingerprint != _fingerprint(declarations):
        _INDEX = _LexicalIndex(declarations)
    return _INDEX


def reset_index() -> None:
    """Drop the cached index — call after the tool schema changes shape."""
    global _INDEX
    _INDEX = None


def lexical_scores(query: str, declarations: Sequence[dict[str, Any]]) -> dict[str, float]:
    """IDF-weighted term-overlap of the query against each tool's name+description.

    Dependency-free and deterministic — the guaranteed offline scorer. A tool's
    NAME tokens are weighted extra: a query that literally names the capability
    should win decisively. Backed by a cached inverted index (see _LexicalIndex).
    """
    q_terms = set(_tokens(query))
    if not q_terms:
        return {d["name"]: 0.0 for d in declarations}

    index = _index_for(declarations)
    scores: dict[str, float] = dict.fromkeys(index.names, 0.0)
    touched: set[str] = set()
    for t in q_terms:
        posting = index.postings.get(t)
        if not posting:
            continue
        idf = index.idf[t]
        for name, weight in posting:
            scores[name] += idf * weight
            touched.add(name)

    # Exactness bonus. "research" matches the whole of the research tool's name
    # but only half of product_research's, yet both scored identically and the
    # tie fell to dictionary order. This is ordinary length normalisation: a
    # query accounting for ALL of a tool's name is a better match for it than
    # one accounting for half. Capped well below a single IDF term (~3-5) so it
    # breaks ties without ever outweighing real evidence, and applied only to
    # tools that already scored, so it costs nothing on the common path.
    for name in touched:
        own = index.pure_name.get(name)
        if not own:
            continue
        matched = len(q_terms & own)
        if matched:
            scores[name] += NAME_EXACTNESS_BONUS * (matched / len(own))
    return scores


# ── the meaning scorer (semantic.py) ─────────────────────────────────────────
#
# Words find the tool whose schema or vocabulary shares the request's words.
# Meaning finds the one whose PURPOSE matches, when the words differ: "kill
# spotify" (close_app), "wake me up at 7" (reminder), "take that last change
# back" (undo). Tool documents are embedded once per schema.

#: Lexical units added per standard deviation a tool's meaning stands above
#: the other tools for this request (only above-average tools gain).
SEMANTIC_BOOST = 1.0

#: tool document text -> its vector. Per document, not per schema: a newly
#: connected MCP server or a forged tool embeds only itself (tens of ms), not
#: all ~140 tools again (~2 s) on whatever thread happened to ask.
_TOOL_VECTORS: dict[str, Any] = {}
#: schema fingerprint -> (names, stacked matrix), so a turn does not restack.
_TOOL_MATRIX: dict[int, tuple[list[str], Any]] = {}


def _tool_document(decl: dict[str, Any]) -> str:
    name = str(decl.get("name", ""))
    description = " ".join(str(decl.get("description", "")).split())[:500]
    return f"{name.replace('_', ' ')}. {description} {vocabulary_for(name)}".strip()


def warm_semantic(declarations: Sequence[dict[str, Any]] | None = None) -> int:
    """Embed every tool now (call from a worker thread). Returns how many."""
    if declarations is None:
        from .dispatch_schema import TOOL_DECLARATIONS
        declarations = TOOL_DECLARATIONS
    return len(_tool_matrix(list(declarations))[0]) if semantic_ready() else 0


def semantic_ready() -> bool:
    from . import semantic
    return semantic.ENCODER.ready()


def _tool_matrix(declarations: list[dict[str, Any]]) -> tuple[list[str], Any]:
    from . import semantic
    import numpy as np

    key = _fingerprint(declarations)
    cached = _TOOL_MATRIX.get(key)
    if cached is not None:
        return cached
    documents = [_tool_document(d) for d in declarations]
    missing = [doc for doc in dict.fromkeys(documents) if doc not in _TOOL_VECTORS]
    if missing:
        vectors = semantic.ENCODER.encode(missing)
        if vectors is None:
            return [], None
        _TOOL_VECTORS.update(zip(missing, vectors))
    names = [d["name"] for d in declarations]
    matrix = np.vstack([_TOOL_VECTORS[doc] for doc in documents]) if documents else None
    _TOOL_MATRIX.clear()                 # one live schema at a time
    _TOOL_MATRIX[key] = (names, matrix)
    return names, matrix


def semantic_scores(query: str, declarations: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Cosine of *query* against each tool's purpose; {} when the sentence
    encoder is not warm, so callers fall back to words alone."""
    from . import semantic

    if not semantic.ENCODER.ready() or not str(query or "").strip():
        return {}
    names, matrix = _tool_matrix(list(declarations))
    question = semantic.ENCODER.encode_one(query)
    if matrix is None or question is None:
        return {}
    sims = matrix @ question
    return {name: float(sims[i]) for i, name in enumerate(names)}


def hybrid_scores(query: str, declarations: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Words plus meaning. Exactly lexical_scores when the encoder is cold."""
    scores = dict(lexical_scores(query, declarations))
    meaning = semantic_scores(query, declarations)
    if not meaning:
        return scores
    values = list(meaning.values())
    mean = sum(values) / len(values)
    spread = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 or 1e-6
    for name, cosine in meaning.items():
        z = (cosine - mean) / spread
        if z > 0:
            scores[name] = scores.get(name, 0.0) + SEMANTIC_BOOST * z
    return scores


# ── the resolver ──────────────────────────────────────────────────────────────

def _passes_hard_filters(decl: dict[str, Any], state: ResolverState) -> bool:
    name = decl.get("name", "")
    if (state.mode.upper() == "B" or not state.online) and name in CLOUD_ONLY:
        return False
    return True


def resolve(
    query: str,
    state: ResolverState | None,
    declarations: Sequence[dict[str, Any]],
    *,
    enabled: bool = True,
    max_tools: int = MAX_TOOLS,
    scorer: Scorer = lexical_scores,
) -> list[dict[str, Any]]:
    """Return the subset of *declarations* to expose for this turn.

    With ``enabled=False`` the input is returned unchanged (list copy) — the
    provable no-op that makes wiring this in safe. Otherwise: hard-filter by
    mode, score by relevance, add context boosts, guarantee the safety floor, and
    cap at ``max_tools``, preserving a stable, deterministic order.
    """
    declarations = list(declarations)
    if not enabled:
        return declarations

    state = state or ResolverState()
    by_name = {d["name"]: d for d in declarations}
    candidates = [d for d in declarations if _passes_hard_filters(d, state)]

    scores = dict(scorer(query, candidates))
    page_domain = _PAGE_DOMAIN.get(state.active_page.upper(), "")
    for d in candidates:
        name = d["name"]
        if page_domain and classify_tool(name) == page_domain:
            scores[name] = scores.get(name, 0.0) + BOOST_PAGE
        if state.focus_active and name in {"focus", "study"}:
            scores[name] = scores.get(name, 0.0) + BOOST_FOCUS
        if name in state.recent_tools:
            scores[name] = scores.get(name, 0.0) + BOOST_RECENT

    # Safety floor first (guaranteed present, in declared order), then the
    # highest-scoring candidates. Ties break by name for determinism.
    ordered: list[dict[str, Any]] = [by_name[n] for n in ALWAYS_INCLUDE if n in by_name]
    ranked = sorted(candidates, key=lambda d: (-scores.get(d["name"], 0.0), d["name"]))

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in ordered + ranked:
        name = d["name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(d)
        if len(out) >= max_tools:
            break
    return out


def resolver_enabled() -> bool:
    """Whether the resolver is active. Off by default — flipping it on is a
    deliberate, follow-up step once the golden set and a live pass are green."""
    return os.getenv("ORION_TOOL_RESOLVER", "").strip().lower() in {"1", "true", "on", "yes"}


__all__ = [
    "MAX_TOOLS", "ALWAYS_INCLUDE", "CLOUD_ONLY", "ResolverState", "classify_tool",
    "lexical_scores", "resolve", "resolver_enabled",
]
