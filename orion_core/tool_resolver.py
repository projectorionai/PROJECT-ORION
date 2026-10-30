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
import sys
import threading
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
BOOST_PAGE = 5.0        # a tool in the foreground deck page's domain
BOOST_FOCUS = 4.0       # study/focus while a focus block is running
BOOST_RECENT = 1.5      # a tool used earlier this turn-sequence
#: Tie-breaker only — see the exactness bonus in lexical_scores.
NAME_EXACTNESS_BONUS = 0.5

# Text and voice discovery use slightly different menus. Retain a few exact
# schemas without retaining an unbounded history of plugin/MCP descriptions.
_SCHEMA_CACHE_ENTRIES = 4
_LEXICAL_CACHE_BYTES = 4 * 1024 * 1024
_VECTOR_CACHE_BYTES = 4 * 1024 * 1024
_MATRIX_CACHE_BYTES = 4 * 1024 * 1024
_CACHE_LOCK = threading.RLock()


class _RoutingCache(dict):
    """A mapping with precomputed entry costs; cache hits never size a tree."""

    def __init__(self) -> None:
        super().__init__()
        self.costs: dict[Any, int] = {}

    def clear(self) -> None:
        super().clear()
        self.costs.clear()


def _cache_size(value: Any, seen: set[int] | None = None) -> int:
    """Conservative retained bytes, including keys, containers and array data."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_cache_size(k, seen) + _cache_size(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(_cache_size(v, seen) for v in value)
    else:
        # Owning NumPy arrays may already include their buffer in getsizeof;
        # counting nbytes again is conservative and also covers array views.
        size += int(getattr(value, "nbytes", 0))
        base = getattr(value, "base", None)
        if base is not None:
            size += _cache_size(base, seen)
        for slot in getattr(type(value), "__slots__", ()):
            if hasattr(value, slot):
                size += _cache_size(getattr(value, slot), seen)
    return size


def _cached(cache: dict, key: Any) -> Any:
    with _CACHE_LOCK:
        value = cache.pop(key, None)
        if value is not None:
            cache[key] = value
        return value


def _cache_costs(cache: dict) -> dict:
    # Plain mappings remain supported for lightweight existing test seams.
    return (cache.costs if isinstance(cache, _RoutingCache)
            else {key: _cache_size((key, value)) for key, value in cache.items()})


def _evict(cache: dict, costs: dict, budget: int, entries: int) -> None:
    global _INDEX
    metadata = sys.getsizeof(costs) + len(costs) * sys.getsizeof(0)
    total = sys.getsizeof(cache) + metadata + sum(costs.values())
    while cache and (len(cache) > entries or total > budget):
        oldest = next(iter(cache))
        total -= costs.pop(oldest)
        del cache[oldest]
    if cache is _INDEX_CACHE and not any(value is _INDEX for value in cache.values()):
        _INDEX = None  # the legacy latest-index view must not pin an evicted tree


def _retain(cache: dict, key: Any, value: Any, *, budget: int, entries: int,
            valid: Callable[[], bool] | None = None) -> bool:
    """Bounded LRU admission; oversized work still returns to its own caller."""
    size = _cache_size((key, value))
    if size > budget:
        return False
    with _CACHE_LOCK:
        if valid is not None and not valid():
            return False
        costs = _cache_costs(cache)
        cache.pop(key, None)
        cache[key] = value
        costs[key] = size
        _evict(cache, costs, budget, entries)
        return key in cache


def _retain_vectors(cache: dict, values: Iterable[tuple[str, Any]], *,
                    valid: Callable[[], bool]) -> None:
    """Admit an embedding batch without repeatedly traversing the whole cache."""
    prepared = []
    for key, value in values:
        # Encoder rows are views of its entire batch. Retaining one such
        # row would otherwise pin an unaccounted, potentially huge buffer.
        owned = value.copy()
        owned.flags.writeable = False
        size = _cache_size((key, owned))
        if size <= _VECTOR_CACHE_BYTES:
            prepared.append((key, owned, size))
    with _CACHE_LOCK:
        if not valid():
            return
        costs = _cache_costs(cache)
        for key, owned, size in prepared:
            cache.pop(key, None)
            cache[key] = owned
            costs[key] = size
        _evict(cache, costs, _VECTOR_CACHE_BYTES, 8192)

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
#: Words that carry no tool's meaning. The second group are the question and
#: helper words of everyday speech: while they counted, "how much cash have I
#: got LEFT" ranked token_usage first, because "how much ... left" is how its
#: vocabulary describes usage. Who/where/when/why/like stay: they mean
#: something to voice ID, file search, reminders, diagnostics ("why can't you
#: do that" has no other word to go on) and the voice tool.
_STOP = frozenset(
    "the a an of to in on at for and or but is are was were be it its this that "
    "with from into about over under as by you your i me my we our so if then "
    "run get set show tell make do use one "
    "how much many what whats which can could would will have has had got any "
    "some there here please give just want need does did done been being very "
    "really lets".split())


def _fold(term: str) -> str:
    """A plural and its singular are one term ("emails" is "email",
    "helicopters" is "helicopter"). Applied to queries and tool documents
    alike, so a fold that is not a real word ("focus" -> "focu") still
    matches itself."""
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _tokens(text: str) -> list[str]:
    return [_fold(t) for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


# ── requests that ask for more than one thing ────────────────────────────────
#
# "Find my CV and email it to Hannah" needs two tools, but ranked as one
# sentence the second need competed with the first for the same few places:
# on the planner's seven multi-part goals the six-result find_tool list held
# every part's tool for 4 (encoder cold) or 6 (warm) of them. Each part is
# now also ranked on its own and its best tools are placed just below the
# whole request's best, so the first-ranked tool never changes.

_PART_SPLIT = re.compile(
    r"[;.?!]\s+|,?\s+\b(?:and then|then|after that|afterwards|and also|as well as|and|plus)\b\s+",
    re.IGNORECASE)
#: At most this many parts are ranked separately (each costs one ranking).
MAX_PARTS = 4
#: Part n's best tool scores (1 - n * PART_STEP) of the whole request's best.
PART_STEP = 0.01


#: Words that never make a piece of a request an ask of its own. Narrower
#: than _STOP on purpose: "run diagnostics" and "what is on my screen" are
#: asks, though _STOP leaves each of them one word.
_FUNCTION_WORDS = frozenset(
    "the a an my your our me us we i it its this that these those some any of to in on at "
    "for with is are be".split())


def request_parts(text: str) -> list[str]:
    """The separate asks in *text* ("X and then Y", "X, and Y", "X. Y"), or []
    for a single ask. A piece with fewer than two words that are not function
    words is not an ask of its own ("pros and cons", "rock and roll" stay
    whole)."""
    pieces = (piece.strip(" ,") for piece in _PART_SPLIT.split(str(text or "")[:600]))
    parts = [piece for piece in pieces
             if sum(1 for word in _TOKEN.findall(piece.lower())
                    if len(word) > 1 and word not in _FUNCTION_WORDS) >= 2]
    return parts[:MAX_PARTS] if len(parts) >= 2 else []


#: A part counts as answered when a plan uses one of its this-many best tools.
COVERAGE_DEPTH = 5


def uncovered_parts(request: str, tools: Iterable[str], declarations: Sequence[dict[str, Any]],
                    scorer: "Scorer | None" = None, depth: int = COVERAGE_DEPTH) -> list[str]:
    """The parts of a multi-part *request* that none of *tools* (a plan's
    steps) answers: none of them is among that part's *depth* best tools.
    A part nothing scores for is not reported - there is no evidence
    either way. [] for a single ask."""
    used = {str(tool) for tool in tools}
    score = scorer or lexical_scores
    missing = []
    for part in request_parts(request):
        scores = score(part, declarations)
        best = [name for name in sorted(scores, key=lambda name: (-scores[name], name))[:depth]
                if scores[name] > 0]
        if best and not used.intersection(best):
            missing.append(part)
    return missing


def merge_part_scores(whole: dict[str, float],
                      parts: Sequence[dict[str, float]]) -> dict[str, float]:
    """*whole* with each part's best tool lifted to (1 - n * PART_STEP) of
    the whole's best (part n, from 1); every other score is left alone, so
    the whole ranking moves down by at most one place per part. Scaling each
    part's whole ranking up instead pushed right tools out of the top ten
    (measured, encoder cold)."""
    top = max(whole.values(), default=0.0)
    if top <= 0 or not parts:
        return whole
    merged = dict(whole)
    for index, scores in enumerate(parts, start=1):
        if not scores:
            continue
        best = min(scores, key=lambda name: (-scores[name], name))
        if scores[best] > 0:
            merged[best] = max(merged.get(best, 0.0), top * (1.0 - index * PART_STEP))
    return merged


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
          "transcript", "awareness", "companion", "catch_up", "context_capsule")
_register("research", "research", "standing_questions", "proactive_check",
          "proactive_report", "read_documents", "experiment_notebook")
_register("vision", "vision_analyse", "vision_verify", "capture_screen",
          "screen_read", "perception")
_register("desktop", "open_app", "close_app", "window_control", "media_control",
          "desktop_control", "peripherals", "gesture_control", "clipboard_operate",
          "process_governor", "process_file", "file_controller", "find_files",
          "display_info", "system_notify", "interface_control", "workspace_control",
          "cursor_overlay", "organise_files", "display", "ui_theme", "ui_reactor",
          "ui_orbit", "ui_chrome", "ui_effect", "ui_screen", "ui_reset", "undo")
_register("web", "browser_control", "web_control", "web_automation",
          "social_media", "muscle_memory", "web_search", "fetch_url", "open_news",
          "flight_search", "navigation_trace")
_register("engineering", "dev_workbench", "codebase_copilot", "debugger", "docker",
          "forge", "self_repair", "diagnostics", "code_changes", "self_changes",
          "cleanup_review")
_register("automation", "execute_plan", "autoplan", "workflow", "workflow_patterns",
          "job", "mission", "skill", "protocol", "agent_dispatch", "reason",
          "strategy", "decision", "perception", "project_rehearsal")
_register("business", "product_research", "tiktok_intel", "instagram_intel",
          "commerce_hub", "competitor_intel", "brand_growth", "business_advisor",
          "founder_knowledge", "creator_intel", "campaign_pipeline",
          "community_share", "commerce_hub")
_register("security", "security_recon", "security_watch", "breach_check",
          "antivirus", "privacy_guard", "sentinel", "pentest_lab")
_register("creative", "audio_studio", "draft_report", "document_export",
          "entertainment", "gaming")
_register("comms", "messaging", "outlook_mail", "notion_workspace", "phone_action",
          "telephony")
_register("geo", "globe", "max_zoom_in_globe", "geo", "aviation")
_register("system", "shutdown_orion", "restart_orion", "system_startup",
          "patch_notes", "capabilities", "token_usage", "resource_status",
          "ai_mode", "backup", "reminder", "morning_briefing", "briefing",
          "executive", "momentum", "mcp")
_register("system", "plugin", "find_tool", "use_tool")
_register("voice", "voice_tone", "speaker_id", "voice_speaker_id",
          "elevenlabs_voice", "audio_devices", "emotion", "avatar", "sound_sense")
_register("voice", "sing", "conversation_recording")
_register("games", "chess")
_register("finance", "finance")
_register("wellbeing", "wellbeing")
_register("goals", "goals")

#: deck page -> the domain whose tools it should boost while it is foreground.
_PAGE_DOMAIN: dict[str, str] = {
    "COGNITION": "learning", "MEMORY": "memory", "RESEARCH": "research",
    "LIBRARY": "memory", "SECURITY": "security", "DEVELOPMENT": "engineering",
    "MARKETING": "business", "TOOLKIT": "business",
    "WORKBENCH": "vision", "GLOBE": "geo", "CHESS": "games",
    "AUTOMATION": "automation", "DIAGNOSTICS": "system", "LOG": "system",
    "TELEMETRY": "system", "OPS": "system", "WIDGETS": "comms",
    "MISSION": "automation", "GOALS": "goals",
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

    def __init__(self, declarations: Sequence[dict[str, Any]], fingerprint: tuple | None = None) -> None:
        fingerprint = _fingerprint(declarations) if fingerprint is None else fingerprint
        docs: dict[str, set[str]] = {}
        name_terms: dict[str, set[str]] = {}
        df: dict[str, int] = {}
        for name, description, vocabulary in fingerprint:
            own = set(_tokens(name.replace("_", " ")))
            # Curated invoking vocabulary is weighted like the tool's own name,
            # because that is what it is: the words that mean "this tool". See
            # tool_vocabulary — without it, recall was 76.5%, because terms like
            # "weather" and "quiz" appear nowhere in the schema at all.
            own |= set(_tokens(vocabulary))
            terms = own | set(_tokens(description))
            docs[name] = terms
            name_terms[name] = own
        # df counted over the same per-declaration term sets as before
        for name, _description, _vocabulary in fingerprint:
            for t in docs[name]:
                df[t] = df.get(t, 0) + 1

        n = max(1, len(fingerprint))
        postings: dict[str, list[tuple[str, float]]] = {}
        for name, terms in docs.items():
            own = name_terms[name]
            for t in terms:
                postings.setdefault(t, []).append((name, 2.0 if t in own else 1.0))

        self.postings = {t: tuple(v) for t, v in postings.items()}
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.names = tuple(name for name, *_ in fingerprint)
        # PURE name tokens, vocabulary excluded — used only for the exactness
        # bonus below, where the question is how much of the tool's own NAME the
        # query accounted for.
        self.pure_name = {name: frozenset(_tokens(name.replace("_", " ")))
                          for name, *_ in fingerprint}
        self.fingerprint = fingerprint


def _fingerprint(declarations: Sequence[dict[str, Any]]) -> tuple:
    """Exact routing inputs: a same-named MCP tool can change its purpose."""
    return tuple((str(d.get("name", "")), str(d.get("description") or ""),
                  vocabulary_for(str(d.get("name", "")))) for d in declarations)


_INDEX: _LexicalIndex | None = None
_INDEX_CACHE: dict[tuple, _LexicalIndex] = _RoutingCache()


def _index_for(declarations: Sequence[dict[str, Any]]) -> _LexicalIndex:
    global _INDEX
    key = _fingerprint(declarations)
    with _CACHE_LOCK:
        found = _cached(_INDEX_CACHE, key)
        if found is not None:
            _INDEX = found
            return found
    made = _LexicalIndex(declarations, fingerprint=key)
    with _CACHE_LOCK:
        # Another worker may have built the exact same immutable inputs.
        found = _cached(_INDEX_CACHE, key)
        if found is not None:
            _INDEX = found
            return found
    kept = _retain(_INDEX_CACHE, key, made, budget=_LEXICAL_CACHE_BYTES,
                   entries=_SCHEMA_CACHE_ENTRIES)
    with _CACHE_LOCK:
        _INDEX = _INDEX_CACHE.get(key) if kept else None
    return made


def reset_index() -> None:
    """Drop the cached index — call after the tool schema changes shape."""
    global _INDEX
    with _CACHE_LOCK:
        _INDEX = None
        _INDEX_CACHE.clear()
        _TOOL_MATRIX.clear()
        _EXAMPLE_MATRIX.clear()


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
#: the other tools for this request (only above-average tools gain). 1.5 was
#: chosen on the development half of a 376-request everyday set once filler
#: words stopped counting as evidence: with the intent network on, 1.0 -> 1.5
#: took that half's top-3 from 87.2% to 88.8% (docs/ORION_TOOL_ROUTING.md).
SEMANTIC_BOOST = 1.5

#: tool document text -> its vector. Per document, not per schema: a newly
#: connected MCP server or a forged tool embeds only itself (tens of ms), not
#: all ~140 tools again (~2 s) on whatever thread happened to ask.
_TOOL_VECTORS: dict[str, Any] = _RoutingCache()
#: schema fingerprint -> (names, stacked matrix), so a turn does not restack.
_TOOL_MATRIX: dict[tuple, tuple[list[str], Any]] = _RoutingCache()
_SEMANTIC_OWNER: tuple | None = None
_SEMANTIC_EPOCH = 0


def _encoder_identity(encoder: Any) -> tuple:
    # Keep strong references: an allocator reusing an old id cannot revive an
    # unrelated model's vectors. A replaced native session is a new generation.
    return (encoder, getattr(encoder, "_session", None),
            getattr(encoder, "_tokenizer", None), getattr(encoder, "spec", None))


def _same_encoder(left: tuple | None, right: tuple) -> bool:
    return left is not None and all(a is b for a, b in zip(left, right))


def _semantic_context() -> tuple[Any, int]:
    from . import semantic
    global _SEMANTIC_OWNER, _SEMANTIC_EPOCH
    encoder = semantic.ENCODER
    identity = _encoder_identity(encoder)
    with _CACHE_LOCK:
        if not _same_encoder(_SEMANTIC_OWNER, identity):
            _TOOL_VECTORS.clear()
            _EXAMPLE_VECTORS.clear()
            _TOOL_MATRIX.clear()
            _EXAMPLE_MATRIX.clear()
            _SEMANTIC_OWNER = identity
            _SEMANTIC_EPOCH += 1
        return encoder, _SEMANTIC_EPOCH


def _semantic_current(context: tuple[Any, int]) -> bool:
    from . import semantic
    return (context[1] == _SEMANTIC_EPOCH
            and _same_encoder(_SEMANTIC_OWNER, _encoder_identity(semantic.ENCODER)))


def _tool_document(decl: dict[str, Any]) -> str:
    name = str(decl.get("name", ""))
    description = " ".join(str(decl.get("description", "")).split())[:500]
    return f"{name.replace('_', ' ')}. {description} {vocabulary_for(name)}".strip()


def warm_semantic(declarations: Sequence[dict[str, Any]] | None = None) -> int:
    """Embed every tool now (call from a worker thread). Returns how many."""
    if declarations is None:
        from .dispatch_schema import TOOL_DECLARATIONS
        declarations = TOOL_DECLARATIONS
    try:                                  # the typing-slip speller (~90 ms), here off the loop
        from .spelling import speller
        speller()
    except Exception:
        pass
    if not semantic_ready():
        return 0
    declarations = list(declarations)
    _example_matrix(declarations)         # the example requests, also off the loop
    return len(_tool_matrix(declarations)[0])


def semantic_ready() -> bool:
    # Synchronize even a cold replacement so the old native model is released.
    return _semantic_context()[0].ready()


# ── vectors kept between starts ──────────────────────────────────────────────
#
# Every launch re-embedded the same ~1,000 tool documents and example requests.
# Measured 23 Oct on a settled install: 4.7 s on the orion-semantic-warm thread
# plus ONNX's own worker threads - most of the ~13 CPU-seconds ORION burned in
# the 30 s after reporting ready, competing with the owner's first request.
# Those texts only change when the code or the model does, so their vectors
# are kept on disk keyed by a digest of the model files and the exact text: an
# edited description re-embeds itself and nothing else.

#: Where they are kept. None = CONFIG_DIR/cache/routing_vectors.npz, except
#: under pytest, where nothing is persisted unless a test sets a path here.
VECTOR_CACHE_PATH: Any = None
#: Rows kept at most (~1.5 KB each); the oldest go first.
_VECTOR_CACHE_MAX = 6000
_DISK: dict[str, Any] | None = None
_DISK_LOCK = threading.Lock()


def _vector_cache_path() -> Any:
    if VECTOR_CACHE_PATH is not None:
        return VECTOR_CACHE_PATH
    if "pytest" in sys.modules:
        return None
    from .constants import CONFIG_DIR
    return CONFIG_DIR / "cache" / "routing_vectors.npz"


def _disk_key(encoder: Any, text: str) -> str:
    import hashlib
    spec = getattr(encoder, "spec", None)
    files = getattr(spec, "files", None) or {}
    pins = ",".join(sorted(str(getattr(f, "sha256", ""))[:16] for f in files.values()))
    return hashlib.sha1(f"{getattr(spec, 'name', '?')}\0{pins}\0{text}".encode(
        "utf-8", "replace")).hexdigest()


def _disk_lookup(encoder: Any, texts: Sequence[str], load: bool = True) -> dict[str, Any]:
    """Stored vectors for whichever of *texts* have one. Never raises. With
    *load* False an unread file is not read now (a caller on the event loop)."""
    global _DISK
    path = _vector_cache_path()
    dim = getattr(getattr(encoder, "spec", None), "dim", None)
    if path is None or dim is None:
        return {}      # only a real model (named, pinned, sized) may reuse vectors
    with _DISK_LOCK:
        if _DISK is None:
            if not load:
                return {}
            _DISK = {}
            try:
                import numpy as np
                with np.load(path) as data:
                    _DISK = {str(k): row for k, row in zip(data["keys"], data["vectors"])}
            except Exception:
                _DISK = {}
        found = {}
        for text in texts:
            row = _DISK.get(_disk_key(encoder, text))
            if row is not None and row.shape[0] == dim:
                found[text] = row
        return found


def _disk_store(encoder: Any, pairs: Iterable[tuple[str, Any]]) -> None:
    """Add freshly encoded rows and rewrite the file atomically. Never raises."""
    global _DISK
    path = _vector_cache_path()
    if path is None or getattr(getattr(encoder, "spec", None), "dim", None) is None:
        return
    try:
        import numpy as np
        with _DISK_LOCK:
            if _DISK is None:
                _DISK = {}
            for text, vector in pairs:
                key = _disk_key(encoder, text)
                _DISK.pop(key, None)
                _DISK[key] = np.array(vector, dtype=np.float32)
            while len(_DISK) > _VECTOR_CACHE_MAX:
                _DISK.pop(next(iter(_DISK)))
            keys = np.array(list(_DISK), dtype="U40")
            rows = np.vstack(list(_DISK.values())) if _DISK else None
        if rows is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".partial.npz")
        np.savez(temporary, keys=keys, vectors=rows)
        os.replace(temporary, path)
    except Exception:
        pass


def _tool_matrix(declarations: list[dict[str, Any]], build: bool = True,
                 _context: tuple[Any, int] | None = None) -> tuple[list[str], Any]:
    """(names, stacked document vectors). With *build* False, tools whose
    documents are not embedded yet are not embedded here: ([], None)."""
    import numpy as np

    context = _semantic_context() if _context is None else _context
    encoder, epoch = context
    key = (epoch, _fingerprint(declarations))
    cached = _cached(_TOOL_MATRIX, key)
    if cached is not None:
        return cached
    documents = [_tool_document(d) for d in declarations]
    with _CACHE_LOCK:
        available = {doc: _cached(_TOOL_VECTORS, doc) for doc in dict.fromkeys(documents)
                     if _semantic_current(context)}
    missing = [doc for doc in dict.fromkeys(documents) if available.get(doc) is None]
    if missing:
        stored = _disk_lookup(encoder, missing, load=build)
        if stored:
            available.update(stored)
            _retain_vectors(_TOOL_VECTORS, stored.items(),
                            valid=lambda: _semantic_current(context))
            missing = [doc for doc in missing if doc not in stored]
    if missing and not build:
        return [], None
    if missing:
        vectors = encoder.encode(missing)
        if vectors is None:
            return [], None
        # Local rows remain usable even when their cache budget refuses them.
        available.update(zip(missing, vectors))
        _retain_vectors(_TOOL_VECTORS, zip(missing, vectors),
                        valid=lambda: _semantic_current(context))
        _disk_store(encoder, zip(missing, vectors))
    names = [d["name"] for d in declarations]
    matrix = np.vstack([available[doc] for doc in documents]) if documents else None
    _retain(_TOOL_MATRIX, key, (names, matrix), budget=_MATRIX_CACHE_BYTES,
            entries=_SCHEMA_CACHE_ENTRIES, valid=lambda: _semantic_current(context))
    return names, matrix


# ── nearest example request ──────────────────────────────────────────────────
#
# A tool's document is one vector for everything it does; an indirect request
# ("my laptop is crawling") is far from the whole of resource_status's
# description but close to one of the everyday requests it is known by
# (intent_situations.py). So a tool also scores the cosine of its NEAREST
# example request, and keeps whichever is higher. Measured on the development
# half of the routing set, encoder warm: top-1 79.4% -> 81.9% (3 seeds).

#: A request quoted in a tool's description ('play some jazz').
_QUOTED_REQUEST = re.compile(r"'([^']{4,90})'")
#: example text -> its vector (per text, like _TOOL_VECTORS).
_EXAMPLE_VECTORS: dict[str, Any] = _RoutingCache()
#: schema fingerprint -> (tool index per group, group starts, stacked matrix).
_EXAMPLE_MATRIX: dict[tuple, tuple[Any, Any, Any]] = _RoutingCache()


def _tool_examples(declarations: list[dict[str, Any]]) -> list[list[str]]:
    """Per declaration: the everyday requests it is known by and the requests
    its description quotes."""
    try:
        from .intent_situations import phrases
        known = phrases()
    except Exception:
        known = {}
    out = []
    for decl in declarations:
        name = str(decl.get("name", ""))
        text = " ".join(str(decl.get("description", "")).split())
        quoted = [q for q in _QUOTED_REQUEST.findall(text) if len(q.split()) >= 2]
        out.append(list(dict.fromkeys([*known.get(name, ()), *quoted])))
    return out


def _example_matrix(declarations: list[dict[str, Any]], build: bool = True,
                    _context: tuple[Any, int] | None = None
                    ) -> tuple[Any, Any, Any] | None:
    """(declaration index of each group, start row of each group, matrix) for
    the tools that have examples; None when not built and *build* is False."""
    import numpy as np

    context = _semantic_context() if _context is None else _context
    encoder, epoch = context
    key = (epoch, _fingerprint(declarations))
    cached = _cached(_EXAMPLE_MATRIX, key)
    if cached is not None or not build:
        return cached
    per_tool = _tool_examples(declarations)
    texts = list(dict.fromkeys(p for ps in per_tool for p in ps))
    with _CACHE_LOCK:
        available = {text: _cached(_EXAMPLE_VECTORS, text) for text in texts
                     if _semantic_current(context)}
    missing = [text for text in texts if available.get(text) is None]
    if missing:
        stored = _disk_lookup(encoder, missing)
        if stored:
            available.update(stored)
            _retain_vectors(_EXAMPLE_VECTORS, stored.items(),
                            valid=lambda: _semantic_current(context))
            missing = [text for text in missing if text not in stored]
    if missing:
        vectors = encoder.encode(missing)
        if vectors is None:
            return None
        available.update(zip(missing, vectors))
        _retain_vectors(_EXAMPLE_VECTORS, zip(missing, vectors),
                        valid=lambda: _semantic_current(context))
        _disk_store(encoder, zip(missing, vectors))
    owners, starts, rows = [], [], []
    for index, examples in enumerate(per_tool):
        if examples:
            owners.append(index)
            starts.append(len(rows))
            rows.extend(available[p] for p in examples)
    if not rows:
        return None
    made = (np.array(owners), np.array(starts), np.vstack(rows))
    _retain(_EXAMPLE_MATRIX, key, made, budget=_MATRIX_CACHE_BYTES,
            entries=_SCHEMA_CACHE_ENTRIES, valid=lambda: _semantic_current(context))
    return made


_WARMING = threading.Lock()


def _warm_in_background(declarations: list[dict[str, Any]]) -> None:
    """Embed a schema's tool documents and example requests on a worker,
    once at a time."""
    if not _WARMING.acquire(blocking=False):
        return

    def warm() -> None:
        try:
            _tool_matrix(declarations)
            _example_matrix(declarations)
        except Exception:
            pass
        finally:
            _WARMING.release()
    threading.Thread(target=warm, name="orion-tool-embed", daemon=True).start()


def semantic_scores(query: str, declarations: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Cosine of *query* against each tool's purpose - or its nearest example
    request, when that is closer; {} when the sentence encoder is not warm,
    so callers fall back to words alone."""
    import numpy as np

    context = _semantic_context()
    encoder = context[0]
    if not encoder.ready() or not str(query or "").strip():
        return {}
    declarations = list(declarations)
    # Embedding tool documents is a worker's job (warm_semantic). On the GUI
    # thread - which under qasync is also the event loop - a schema that has
    # just grown (an MCP server connecting adds a batch of tools) was embedded
    # inline: an onnxruntime run() of a quarter of a second that froze the
    # face. That turn is ranked on words while a worker catches up.
    on_loop = threading.current_thread() is threading.main_thread()
    names, matrix = _tool_matrix(declarations, build=not on_loop, _context=context)
    if matrix is None:
        if on_loop:
            _warm_in_background(declarations)
        return {}
    question = encoder.encode_one(query)
    if question is None:
        return {}
    sims = matrix @ question
    # The examples are embedded by warm_semantic (a worker); a question on
    # the main thread before that uses the documents alone, never waits.
    examples = _example_matrix(declarations,
                               build=threading.current_thread() is not threading.main_thread(),
                               _context=context)
    if examples is not None:
        owners, starts, stacked = examples
        nearest = np.maximum.reduceat(stacked @ question, starts)
        sims = sims.copy()
        sims[owners] = np.maximum(sims[owners], nearest)
    return {name: float(sims[i]) for i, name in enumerate(names)}


def hybrid_scores(query: str, declarations: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Words plus meaning. Exactly lexical_scores when the encoder is cold.
    Typing slips are put right first (spelling.py): "colse spotify" scores
    as "close spotify"."""
    from .spelling import correct
    query = correct(query)
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
