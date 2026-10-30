"""
Capability triage before forging (Forge Mark III, W3.3) and duplicate detection.

Why this exists
---------------
Measured on the owner's machine (self_improvement/journal.jsonl and
evolution.sqlite3, 9-10 Oct 2026): "Can you find the biggest files currently on
the PC" was refused twice by find_files, recorded as an unmet request, and the
self-improvement review then forged ``largest_files_scanner``,
``largest_files_finder`` and ``disk_space_analyzer`` — 15 verification runs,
every one failing (the installed app had no verifier at the time). The real
fix was one line in find_files, which now ranks the whole PC by size with
``sort="largest"``. Worse, the generated disk_space_analyzer refused any folder
outside its own working directory, so even a passing build could never have
answered the question.

The same pattern filled config/custom_tools: three gesture-control tools beside
the built-in ``gesture_control``, two fitness trackers, two re-forgers of
quarantined tools beside ``forge(action='health')``. None has a recorded call.

So before any model is asked to write code, the request is checked against
what ORION can already do:

  * ROUTES — known needs with a known answer ("biggest files" →
    ``find_files(sort='largest')``), each valid only while that tool exists;
  * built-in tools whose name and description cover the request → route;
  * installed plugins that already do it → duplicate.

Only a genuinely new capability reaches the Forge. Pure text analysis — no
model call, no I/O — so it is cheap, deterministic and testable.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

#: Words that say what KIND of tool something is, not what it does. Shared in
#: spirit with ImprovementHeartbeat._GENERIC_NAME_WORDS.
_GENERIC = frozenset({
    "manager", "tracker", "generator", "monitor", "tool", "tools", "helper",
    "agent", "engine", "system", "service", "assistant", "analyzer", "analyser",
    "handler", "builder", "creator", "enhancer", "personal", "smart", "auto",
    "insight", "insights", "recognizer", "recogniser", "detector", "interface",
    "utility", "module", "orion", "advanced", "simple", "basic", "custom",
    "dynamic", "intelligent", "automated", "automatic", "finder", "scanner",
    "checker", "reporter", "selector", "adapter", "debugger", "optimizer",
    "optimiser",
})
_STOP = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "onto", "your",
    "their", "them", "they", "user", "users", "use", "uses", "using", "used",
    "will", "can", "could", "should", "would", "may", "must", "not", "any", "all",
    "each", "every", "its", "it's", "are", "was", "were", "been", "being", "has",
    "have", "had", "via", "per", "such", "than", "then", "when", "which", "who",
    "what", "where", "how", "also", "only", "just", "like", "more", "most", "less",
    "new", "create", "build", "make", "provide", "provides", "allow", "allows",
    "tool", "tools", "orion", "o.r.i.o.n", "returns", "return", "given", "based",
    "optional", "optionally", "default", "defaults", "specified", "specific",
    "list", "lists", "data", "information", "capability", "capabilities",
    "function", "functionality", "support", "supports", "handle", "handles",
    "safely", "clean", "cleanly", "readable", "string", "result", "results",
})
_WORD_RE = re.compile(r"[a-z][a-z0-9']{2,}")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "er", "ies", "es", "ed", "ly", "s")


def stem(word: str) -> str:
    word = word.lower().strip("'")
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    return word[:6]


def content_stems(text: str) -> set[str]:
    words = _WORD_RE.findall(str(text or "").lower().replace("_", " "))
    return {stem(w) for w in words if w not in _STOP}


def name_concepts(name: str) -> set[str]:
    """What a snake_case tool name is ABOUT, ignoring words for its kind."""
    return {stem(part) for part in re.split(r"[^a-z0-9]+", str(name).lower())
            if len(part) >= 4 and part not in _GENERIC and part not in _STOP}


# ── known needs with a known answer ──────────────────────────────────────────

@dataclass(frozen=True)
class CapabilityRoute:
    tool: str
    arguments: dict
    pattern: str
    reason: str
    exclude: str = ""

    def matches(self, text: str) -> bool:
        if self.exclude and re.search(self.exclude, text):
            return False
        # "Does not schedule reminders" / "without installing packages" names
        # the very thing a tool does NOT do — two shipped plugins read as
        # reminder and package tools until negations were skipped.
        for found in re.finditer(self.pattern, text):
            before = text[max(0, found.start() - 28):found.start()]
            if not _NEGATION_RE.search(before):
                return True
        return False


_NEGATION_RE = re.compile(r"\b(?:not|no|never|without|nor|rather than|instead of)\b[^.;,]*$")


_FILES = r"(?:files?|items?|downloads?|videos?|documents?)"
ROUTES: tuple[CapabilityRoute, ...] = (
    CapabilityRoute(
        "find_files", {"action": "usage"},
        r"\b(?:folders?|director(?:y|ies))\b.{0,40}\b(?:sizes?|usage|taking up|biggest|largest|space)\b"
        r"|\b(?:biggest|largest)\s+(?:folders?|director(?:y|ies))\b",
        "find_files(action='usage') lists the largest folders inside a folder",
        exclude=r"\b(?:duplicate|identical|hash)\w*\b"),
    CapabilityRoute(
        "find_files", {"sort": "largest"},
        r"\b(?:large|larg(?:er|est)|big+(?:er|est)?|huge|heav(?:y|iest))\b.{0,30}\b" + _FILES + r"\b"
        r"|\b" + _FILES + r"\b.{0,50}\b(?:by size|largest|biggest|taking up|sorted by size|size order)\b"
        r"|\bdisk[ -](?:space|usage)\b.{0,40}\b(?:analy[sz]\w*|scan\w*|hog\w*|report\w*|find\w*|clean\w*|manag\w*)"
        r"|\b(?:analy[sz]\w*|scan\w*|manag\w*)\b.{0,20}\bdisk[ -](?:space|usage)\b"
        r"|\b(?:free|clear)(?: up| out)? (?:some )?(?:disk |drive )?space\b",
        "find_files(sort='largest') ranks the whole PC — or one folder with folder=... — "
        "by size, using Everything or the Windows index, with no name or type needed",
        exclude=r"\b(?:duplicate|identical|hash|compress|upload)\w*\b"),
    CapabilityRoute(
        "sentinel", {"action": "status"},
        r"\b(?:free|remaining|available|left)\b.{0,15}\b(?:disk|drive|storage)\b.{0,10}\b(?:space|capacity)\b"
        r"|\bdisk (?:space|capacity) (?:left|remaining|free)\b|\bbattery (?:level|status|health)\b",
        "sentinel(action='status') reports CPU, RAM, disk and battery"),
    CapabilityRoute(
        "process_governor", {"action": "top"},
        r"\b(?:cpu|ram|memory|gpu)\b.{0,40}\b(?:hog\w*|top|biggest|heav\w*|using the most|consum\w*)\b"
        r"|\bprocess(?:es)?\b.{0,30}\b(?:monitor\w*|usage|top|kill\w*|terminat\w*|list\w*)\b",
        "process_governor(action='top') lists the biggest CPU/RAM users and can stop one"),
    CapabilityRoute(
        "resource_status", {},
        r"\b(?:system|host|pc|computer)\s+(?:resources?|health|performance|load)\b"
        r"|\bresource\s+(?:monitor\w*|optimi[sz]\w*|usage|pressure|track\w*)\b"
        r"|\bmemory\s+(?:prun\w*|pressure|threshold\w*|optimi[sz]\w*)\b",
        "resource_status reports CPU/RAM/swap pressure with a recommendation; "
        "ORION already sheds idle components under memory pressure"),
    CapabilityRoute(
        "geo", {"action": "weather"}, r"\bweather\b|\bforecast\b",
        "geo(action='weather') gives live conditions and the forecast anywhere"),
    CapabilityRoute(
        "gesture_control", {"action": "start"}, r"\bgestures?\b",
        "gesture_control is ORION's built-in webcam hand-gesture control"),
    CapabilityRoute(
        "focus", {}, r"\bpomodoro\b|\bfocus (?:session|block|timer)s?\b",
        "focus runs timed deep-work blocks with breaks"),
    CapabilityRoute(
        "reminder", {}, r"\bremind\w*\b|\balarms?\b|\bcount ?down\b|\btimers?\b",
        "reminder sets spoken reminders, alarms and timers",
        exclude=r"\bpomodoro\b|\bfocus\b"),
    CapabilityRoute(
        "calculate", {},
        r"\bunit conver\w*|\bconvert\w* (?:between )?units\b|\barithmetic\b|\bpercentages?\b"
        r"|\bcurrency conver\w*",
        "calculate does exact arithmetic, percentages and unit conversions"),
    CapabilityRoute(
        "screen_read", {"action": "read"},
        r"\bread\w* (?:my |the )?screen\b|\btext (?:from|on) (?:my |the )?screen\b"
        r"|\b(?:ocr|text extract\w*)\b.{0,40}\bscreen(?:shot)?s?\b",
        "screen_read captures the screen and reads it (model vision with OCR fallback)"),
    CapabilityRoute(
        "process_file", {},
        r"\b(?:ocr|extract\w* text|text extract\w*)\b.{0,40}\b(?:images?|photos?|pictures?|scans?|pdfs?)\b"
        r"|\b(?:images?|photos?|pdfs?)\b.{0,30}\b(?:ocr|text extract\w*)\b",
        "process_file reads a local image or PDF (vision, OCR) by path"),
    CapabilityRoute(
        "capture_screen", {}, r"\b(?:take|capture)\w* (?:a )?screenshots?\b",
        "capture_screen takes a screenshot"),
    CapabilityRoute(
        "clipboard_operate", {}, r"\bclipboard\b", "clipboard_operate reads and copies text"),
    CapabilityRoute(
        "peripherals", {}, r"\b(?:master |system )?volume (?:control|level)\b|\bscreen brightness\b",
        "peripherals sets volume and brightness"),
    CapabilityRoute(
        "messaging", {},
        r"\bsend\w* (?:a )?(?:message|text|whatsapp|telegram|discord)\b",
        "messaging sends texts over WhatsApp/Telegram/Discord and e-mail"),
    CapabilityRoute(
        "organise_files", {"action": "preview"},
        r"\b(?:organi[sz]\w*|tidy|sort\w*) (?:up )?(?:my |the )?(?:files|downloads|desktop|documents|folders?)\b",
        "organise_files previews and applies a folder tidy-up with undo"),
    CapabilityRoute(
        "web_search", {}, r"\b(?:search the web|web search|google search|internet search)\b",
        "web_search searches the web"),
    CapabilityRoute(
        "fetch_url", {},
        r"\b(?:fetch|read|scrape|download)\w* (?:the )?(?:contents? of )?(?:a |the )?(?:web ?pages?|urls?|websites?)\b",
        "fetch_url reads a web page as clean text"),
    CapabilityRoute(
        "forge", {"action": "health"},
        r"\bquarantin\w*\b.{0,40}\b(?:tools?|plugins?|modules?)\b|\bre-?forg\w*\b",
        "forge(action='health') lists quarantined and held-back tools; forge rebuilds them"),
    CapabilityRoute(
        "plugin", {"action": "deps"},
        r"\b(?:install|resolv|manag|identif)\w* (?:missing |the )?(?:python )?(?:dependenc\w*|packages?)\b",
        "the Forge installs a tool's missing packages itself; plugin(action='deps') does it for a plugin"),
    CapabilityRoute(
        "self_repair", {"action": "status"},
        r"\bself[- ]?repair\w*\b|\berror handl\w*\b|\bincidents?\b.{0,20}\b(?:track|record|log)\w*",
        "self_repair captures incidents and repairs; diagnostics explains failures"),
    CapabilityRoute(
        "voice_tone", {}, r"\bemotion\w*\b.{0,40}\b(?:recogni\w*|detect\w*|voice|speech|tone)\b",
        "voice_tone reads how the user sounds (energy, pitch, pace) from their voice"),
    CapabilityRoute(
        "wellbeing", {}, r"\b(?:mood|emotion\w*|wellbeing|energy)\b.{0,40}\b(?:track\w*|log\w*|monitor\w*|insight\w*|trend\w*)\b",
        "wellbeing logs energy, sleep and mood and reports trends"),
)


# ── the verdict ──────────────────────────────────────────────────────────────

@dataclass
class TriageVerdict:
    """What to do with a request for a new tool."""

    decision: str = "forge"            # forge | route | duplicate
    tool: str = ""                     # the existing tool that covers it
    arguments: dict = field(default_factory=dict)
    reason: str = ""
    score: float = 0.0
    basis: str = ""                    # route | builtin | plugin

    @property
    def covered(self) -> bool:
        return self.decision != "forge"

    def call(self) -> str:
        if not self.tool:
            return ""
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.tool}({args})"

    def message(self, requested: str = "") -> str:
        if not self.covered:
            return ""
        what = requested.replace("_", " ") if requested else "that"
        lead = ("No new tool needed: " if self.decision == "route"
                else "Not forging a duplicate: ")
        return (f"{lead}{self.call()} already does {what}. {self.reason}. "
                f"Use it now. If it genuinely cannot do what is needed, say exactly what "
                f"it lacks: a plan for only that missing part is judged on its own.")

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision, "tool": self.tool, "arguments": dict(self.arguments),
                "reason": self.reason, "score": round(self.score, 3), "basis": self.basis}


def _as_tools(tools: Iterable[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in tools or ():
        if isinstance(item, dict):
            name = str(item.get("name") or "")
            if name:
                out.append({"name": name, "description": str(item.get("description") or ""),
                            "origin": str(item.get("origin") or "builtin")})
        elif isinstance(item, str) and item:
            out.append({"name": item, "description": "", "origin": "builtin"})
    return out


def similarity(name: str, plan: str, other_name: str, other_description: str) -> tuple[int, float, int]:
    """(name concepts shared, overlap coefficient of content, shared content stems)."""
    shared_names = len(name_concepts(name) & name_concepts(other_name))
    mine = content_stems(f"{name} {plan}")
    theirs = content_stems(f"{other_name} {other_description}")
    shared = mine & theirs
    coefficient = len(shared) / max(1, min(len(mine), len(theirs)))
    return shared_names, coefficient, len(shared)


def _covers(shared_names: int, coefficient: float, shared: int) -> float:
    """A score > 0 when another tool plausibly does the same job.

    Calibrated on the owner's 44 plugins and the 18-request benchmark: two
    shared name concepts ('gesture'+'control') is the same tool under another
    name; one shared concept needs most of the description to agree as well.
    """
    if shared_names >= 2:
        return 0.9 + min(0.1, coefficient / 10)
    if shared_names == 1 and coefficient >= 0.45 and shared >= 4:
        return 0.6 + coefficient / 4
    if coefficient >= 0.7 and shared >= 6:
        return 0.5 + coefficient / 4
    return 0.0


def triage(tool_name: str, plan: str, tools: Iterable[Any] = ()) -> TriageVerdict:
    """Decide whether *tool_name*/*plan* needs a new tool at all.

    *tools* is the live catalogue: dicts with name, description and origin
    ('builtin', 'plugin' or 'mcp'), or bare names. A route is offered only
    when its tool is in the catalogue; with an empty catalogue the answer is
    always "forge" (nothing can be shown to cover it).
    """
    catalogue = _as_tools(tools)
    if not catalogue or str(tool_name).startswith("trial_"):
        return TriageVerdict()
    names = {item["name"]: item for item in catalogue}
    text = f"{str(tool_name).replace('_', ' ')} {plan}".lower()
    for route in ROUTES:
        if route.tool in names and route.matches(text):
            return TriageVerdict("route", route.tool, dict(route.arguments), route.reason,
                                 1.0, "route")
    best: TriageVerdict = TriageVerdict()
    for item in catalogue:
        if item["name"] == tool_name:
            continue
        score = _covers(*similarity(tool_name, plan, item["name"], item["description"]))
        if score > best.score:
            origin = item["origin"]
            best = TriageVerdict(
                "duplicate" if origin == "plugin" else "route", item["name"], {},
                f"'{item['name']}' ({origin}): {item['description'][:140].strip() or 'same name'}",
                score, origin)
    return best if best.score > 0 else TriageVerdict()


def catalogue_from_dispatcher(dispatcher: Any) -> list[dict[str, str]]:
    """Every tool the live dispatcher can route to, labelled by origin."""
    if dispatcher is None:
        return []
    native: set[str] = set()
    table = getattr(dispatcher, "handler_table", None)
    if callable(table):
        try:
            native = set(table())
        except Exception:
            native = set()
    declarations = getattr(dispatcher, "TOOL_DECLARATIONS", None)
    if not isinstance(declarations, list):
        declarations = []
    forged = set(getattr(dispatcher, "_tool_handlers", {}) or {})
    mcp = set(getattr(dispatcher, "_mcp_routes", {}) or {})
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in declarations:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        seen.add(name)
        origin = ("builtin" if name in native else "plugin" if name in forged
                  else "mcp" if name in mcp else "builtin")
        out.append({"name": name, "description": str(item.get("description") or ""),
                    "origin": origin})
    for name in sorted(native - seen):
        out.append({"name": name, "description": "", "origin": "builtin"})
    return out


# ── the installed-tool audit (duplicates, junk, unused) ──────────────────────

#: ORION internals a forged tool cannot reach — the security guard forbids
#: importing orion_core, so code that talks about them is simulating them.
_INTERNALS_RE = re.compile(r"\b(?:QuantumFace3D|set_viseme|OrionBus|dispatcher|orion_core|"
                           r"thought model|LocalMind|ForgeOrchestrationManager)\b")


@dataclass
class AuditRow:
    name: str
    origin: str                    # shipped | forged
    description: str = ""
    calls: int = 0
    last_used: str = ""
    duplicate_of: str = ""
    duplicate_kind: str = ""       # builtin | plugin | route
    problems: list[str] = field(default_factory=list)
    verdict: str = "keep"          # keep | review | retire-candidate
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s|\s—\s", str(text or "").strip(), maxsplit=1)[0]


def _static_problems(name: str, description: str, source: str, *, forged: bool) -> list[str]:
    """Signs a forged tool cannot do what it says. Shipped plugins are
    reviewed adapters (the jv_* files are 12-line delegates into
    orion_core.jv_plugins by design), so only a parse failure counts there."""
    problems: list[str] = []
    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        return [f"does not parse ({exc.msg})"]
    if not forged:
        return problems
    imports = {alias.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names}
    imports |= {(node.module or "").split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)}
    code_only = "\n".join(line for line in (source or "").splitlines()
                          if not line.lstrip().startswith("#"))
    if _INTERNALS_RE.search(code_only) and "orion_core" not in imports:
        problems.append("talks to ORION internals it cannot import (it can only simulate them)")
    concepts = name_concepts(name)
    if concepts and description and not (concepts & content_stems(description)):
        problems.append("its description does not match its name")
    if len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]) <= 2 and \
            len((source or "").splitlines()) < 25:
        problems.append("very small body — check it does real work")
    return problems


def audit_tools(plugins: Iterable[dict[str, Any]], catalogue: Iterable[Any],
                usage: dict[str, dict[str, Any]] | None = None) -> list[AuditRow]:
    """One row per installed plugin: duplicate, broken-looking or unused.

    *plugins*: dicts with name, description, origin ('shipped'/'forged') and
    source (module text). *catalogue*: the built-in tools (see triage). *usage*:
    tool → {calls, last}. Never deletes or disables anything — it recommends.
    """
    plugin_rows = [dict(p) for p in plugins or ()]
    builtins_only = [t for t in _as_tools(catalogue) if t["origin"] == "builtin"]
    usage = usage or {}
    rows: list[AuditRow] = []
    for plugin in plugin_rows:
        name = str(plugin.get("name") or "")
        description = str(plugin.get("description") or "")
        # The first sentence says what a tool IS; later sentences list
        # examples ("'remind me on my phone'") that read as other jobs.
        summary = _first_sentence(description)
        stats = usage.get(name) or {}
        forged = str(plugin.get("origin") or "forged") == "forged"
        row = AuditRow(name=name, origin="forged" if forged else "shipped",
                       description=description[:160], calls=int(stats.get("calls") or 0),
                       last_used=str(stats.get("last") or ""))
        verdict = triage(name, summary, builtins_only)
        if verdict.covered:
            row.duplicate_of, row.duplicate_kind = verdict.tool, verdict.basis
        elif forged:
            # Shipped plugins are distinct reviewed adapters by construction;
            # only a forged one can be a second copy of another plugin.
            peers = [{"name": p["name"], "description": _first_sentence(p.get("description", "")),
                      "origin": "plugin"} for p in plugin_rows if p.get("name") != name]
            verdict = triage(name, summary, peers)
            if verdict.covered:
                row.duplicate_of, row.duplicate_kind = verdict.tool, "plugin"
        problems = _static_problems(name, description, str(plugin.get("source") or ""),
                                    forged=forged)
        if plugin.get("held"):
            problems.insert(0, f"not live: {plugin['held']}")
        row.problems = problems
        unused = row.calls == 0
        never = "never called" if unused else f"{row.calls} call(s)"
        if not forged:
            # Reviewed and shipped: messaging, for one, delivers through the
            # Discord/Telegram plugins, so an overlap is not a duplicate.
            row.verdict = "review" if problems else "keep"
            row.note = "; ".join(filter(None, [
                "shipped/reviewed", f"overlaps built-in {row.duplicate_of}" if row.duplicate_of else "",
                "; ".join(problems[:2]), never]))
        elif row.duplicate_of and unused:
            row.verdict = "retire-candidate"
            kind = "plugin" if row.duplicate_kind == "plugin" else "built-in"
            row.note = f"duplicates {kind} {row.duplicate_of}; never called" + \
                (f"; {problems[0]}" if problems else "")
        elif problems and unused:
            row.verdict = "retire-candidate"
            row.note = "; ".join(problems[:2]) + "; never called"
        elif row.duplicate_of or problems:
            row.verdict = "review"
            row.note = "; ".join(filter(None, [
                f"overlaps {row.duplicate_of}" if row.duplicate_of else "", "; ".join(problems[:2]), never]))
        else:
            row.note = f"forged, not reviewed; {never}"
            row.verdict = "review" if unused else "keep"
        rows.append(row)
    return rows


def render_audit(rows: Iterable[AuditRow]) -> str:
    rows = list(rows)
    if not rows:
        return "No installed plugins to audit."
    order = {"retire-candidate": 0, "review": 1, "keep": 2}
    rows.sort(key=lambda r: (order.get(r.verdict, 3), r.name))
    counts = {v: sum(1 for r in rows if r.verdict == v) for v in order}
    lines = [f"Installed tool audit: {len(rows)} plugin(s) — {counts['retire-candidate']} "
             f"retire candidate(s), {counts['review']} to review, {counts['keep']} keep. "
             "Nothing was changed; disabling or removing is the owner's decision."]
    for row in rows:
        dup = f" (≈ {row.duplicate_of})" if row.duplicate_of else ""
        lines.append(f"  [{row.verdict}] {row.name}{dup} — {row.note}")
    return "\n".join(lines)


__all__ = ["CapabilityRoute", "ROUTES", "TriageVerdict", "triage", "similarity",
           "catalogue_from_dispatcher", "AuditRow", "audit_tools", "render_audit",
           "name_concepts", "content_stems"]
