"""
The reasoning engine — deliberation, adversarial critique and verification.

ORION could always *answer*. What he could not do was **think about it twice**.
A hard question and a trivial one took the same path: one prompt, one provider
call, one reply, no lookups, no self-examination, and no check that anything he
said was true. The specialist workforce in ``agents.py`` looked like reasoning
but was a persona wrapper — a regex picked a system message and the model
answered from the shape of the question.

This module is the missing layer, and it has four parts:

    Budget tiers      Thinking is not free, so how much of it to spend is an
                      explicit, inspectable decision rather than an accident.
                      ``assess_complexity`` reads the question and picks
                      REFLEX, STANDARD, DELIBERATE or EXHAUSTIVE; the caller
                      may override, and an environment cap can hold the whole
                      system down to a cheaper ceiling.

    Tool-using        A specialist may look things up mid-thought — memory,
    specialists       past conversations, the knowledge graph, the evidence
                      store, files, the live web — over a read-only instrument
                      tray (``AgentToolbelt``). Answers become grounded in this
                      machine and this user instead of plausible-sounding.

    Panel + critique  Above STANDARD, several specialists draft *independently
                      and concurrently*, then a red team attacks every draft:
                      what is wrong, what is missing, what is the strongest
                      counter-argument, where do the drafts contradict each
                      other. A chair then synthesises one answer that carries
                      only what survived. Independence before criticism is the
                      point — a panel that sees each other's work first
                      converges instead of covering the space.

    Verification      The synthesised answer is broken back into checkable
                      claims and each is put to the ``EvidenceEngine``:
                      supported, contradicted, or simply unknown to us. A
                      contradiction is surfaced, never quietly averaged away.

Two invariants worth keeping: every instrument is read-only and class-gated
(see ``AgentToolbelt``), so a reasoning pass can never act on the user's
machine; and every stage degrades rather than fails — no provider, no evidence
store, no research agent, a critic that returns nonsense — each of those costs
depth, not the answer.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Sequence

from .agents import AgentFinding, AgentManager, AgentToolbelt, BaseAgent, ToolInvocation
from .bus import OrionBus
from .concurrency import max_parallel, parallelism_enabled
from .data import ToolResult
from .security import SecuritySanitiser
from .utils import first_line


# ──────────────────────────────────────────────────────────────────────────────
# REASONING BUDGET TIERS
# ──────────────────────────────────────────────────────────────────────────────

class ReasoningTier(str, Enum):
    """How much thought a question is worth."""

    REFLEX = "reflex"            # one specialist, no lookups, no review
    STANDARD = "standard"        # one specialist that may look things up
    DELIBERATE = "deliberate"    # panel + red team + verification
    EXHAUSTIVE = "exhaustive"    # panel + red team + revision + audited synthesis


TIER_ORDER: tuple[ReasoningTier, ...] = (
    ReasoningTier.REFLEX,
    ReasoningTier.STANDARD,
    ReasoningTier.DELIBERATE,
    ReasoningTier.EXHAUSTIVE,
)


@dataclass(frozen=True)
class ReasoningBudget:
    """The concrete spend a tier authorises.

    Every number here is a ceiling, never a quota: a tier that permits four
    lookups spends none when the specialist does not want them, and a panel of
    three seats one when only one specialist is relevant.
    """

    tier: ReasoningTier
    panel: int          # specialists drafting independently
    lookups: int        # instrument readings available to each specialist
    critiques: int      # 0 none · 1 critique the drafts · 2 also audit the synthesis
    revise: bool        # specialists rewrite in light of the critique
    verify: bool        # check the answer's claims against the evidence store
    record: bool        # write instrument-grounded claims back to that store
    summary: str


BUDGETS: dict[ReasoningTier, ReasoningBudget] = {
    ReasoningTier.REFLEX: ReasoningBudget(
        ReasoningTier.REFLEX, panel=1, lookups=0, critiques=0,
        revise=False, verify=False, record=False,
        summary="answered directly"),
    ReasoningTier.STANDARD: ReasoningBudget(
        ReasoningTier.STANDARD, panel=1, lookups=3, critiques=0,
        revise=False, verify=False, record=False,
        summary="one specialist, free to look things up"),
    ReasoningTier.DELIBERATE: ReasoningBudget(
        ReasoningTier.DELIBERATE, panel=3, lookups=3, critiques=1,
        revise=False, verify=True, record=True,
        summary="panel, red-team critique, verified"),
    ReasoningTier.EXHAUSTIVE: ReasoningBudget(
        ReasoningTier.EXHAUSTIVE, panel=4, lookups=5, critiques=2,
        revise=True, verify=True, record=True,
        summary="panel, critique, revision, audited synthesis, verified"),
}


# (pattern, weight, what the signal means) — the whole complexity model, in one
# readable table. Deterministic on purpose: asking a model how hard a question
# is costs a model call before any thinking has happened, and answers
# inconsistently. Regex is free, testable and never rate-limited.
_SIGNALS: tuple[tuple[re.Pattern[str], int, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), weight, label) for pattern, weight, label in (
        # The user asking outright for effort outranks everything else.
        (r"\bthink (?:hard|harder|carefully|deeply|it through|about it)\b", 5,
         "an explicit request to think"),
        (r"\b(?:deep[- ]dive|in depth|thorough(?:ly)?|rigorous|exhaustive|"
         r"properly|from first principles)\b", 4, "a request for depth"),
        (r"\b(?:are you sure|double[- ]check|sanity[- ]check|verify|challenge "
         r"(?:me|this|that))\b", 4, "a request to check the answer"),
        # Decisions with consequences.
        (r"\b(?:should i|should we|worth it|which (?:one|option|approach)|"
         r"what would you (?:do|choose)|help me (?:decide|choose))\b", 3,
         "a decision to make"),
        (r"\b(?:vs\.?|versus|compare|comparison|trade[- ]?offs?|pros and cons|"
         r"better than|instead of)\b", 2, "a comparison"),
        (r"\b(?:strateg|plan|roadmap|architect|design a|approach for|"
         r"how (?:should|would|do) (?:i|we))\b", 2, "open-ended design"),
        (r"\b(?:why|what if|implications?|consequences?|root cause|"
         r"second[- ]order)\b", 2, "causal or hypothetical analysis"),
        # Stakes.
        (r"\b(?:risk|risky|safe|safety|legal|liabilit|security|irreversible|"
         r"can't undo|money|invest|budget|cost|expensive|contract)\b", 2,
         "consequences worth being careful about"),
        (r"\b(?:diagnos|debug|troubleshoot|failing|broken|regression|"
         r"intermittent|race condition)\b", 2, "a diagnosis from evidence"),
        # Constraint density.
        (r"\b(?:must|cannot|without|constraint|requirement|deadline|"
         r"limited to|no more than|at least)\b", 1, "stated constraints"),
        (r"\b(?:research|evidence|source|data|study|citation|prove)\b", 1,
         "a demand for evidence"),
    )
)

# Short, closed, factual or conversational — the question is done before a
# panel could convene. Checked before the signal table so "why is the sky blue"
# does not summon a red team.
_TRIVIAL = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|ok(?:ay)?|yes|no|good (?:morning|"
    r"afternoon|evening|night)|what(?:'s| is) the time|what time is it|"
    r"how are you)\b", re.IGNORECASE)


def cap_tier(tier: ReasoningTier) -> ReasoningTier:
    """Hold *tier* under the ``ORION_MAX_REASONING_TIER`` ceiling.

    One environment variable pins the entire system to a cheaper mode — useful
    on a metered connection, on battery, or when isolating whether a wrong
    answer came from the deliberation machinery or from the model underneath.
    """
    raw = os.getenv("ORION_MAX_REASONING_TIER", "").strip().lower()
    if not raw:
        return tier
    try:
        ceiling = ReasoningTier(raw)
    except ValueError:
        return tier
    return tier if TIER_ORDER.index(tier) <= TIER_ORDER.index(ceiling) else ceiling


def assess_complexity(question: str) -> tuple[ReasoningTier, str]:
    """Pick a tier for *question*, with the reason in plain words.

    The rationale is returned rather than logged because the tier is a decision
    ORION made on the user's behalf about how much to spend, and decisions made
    on someone's behalf should be legible to them.
    """
    text = str(question or "").strip()
    if not text:
        return ReasoningTier.REFLEX, "nothing was asked"
    words = len(text.split())
    if _TRIVIAL.match(text) or (words <= 5 and "?" not in text):
        return cap_tier(ReasoningTier.REFLEX), "a short, closed question"

    score = 0
    reasons: list[str] = []
    for pattern, weight, label in _SIGNALS:
        if pattern.search(text):
            score += weight
            reasons.append(label)
    if words >= 45:
        score += 2
        reasons.append("a long, multi-part question")
    elif words >= 22:
        score += 1
        reasons.append("a substantial question")
    if text.count("?") >= 2:
        score += 1
        reasons.append("several questions at once")

    if score >= 8:
        tier = ReasoningTier.EXHAUSTIVE
    elif score >= 4:
        tier = ReasoningTier.DELIBERATE
    elif score >= 1:
        tier = ReasoningTier.STANDARD
    else:
        tier = ReasoningTier.REFLEX
    capped = cap_tier(tier)
    rationale = "; ".join(list(dict.fromkeys(reasons))[:3])
    if capped is not tier:
        rationale = f"{rationale or 'no strong signal'} (held at {capped.value} by policy)"
    return capped, rationale or "no strong signal"


def resolve_tier(question: str, requested: str | ReasoningTier | None = None
                 ) -> tuple[ReasoningBudget, str]:
    """Budget for *question*, honouring an explicit request when it is valid."""
    if requested:
        raw = requested.value if isinstance(requested, ReasoningTier) else str(requested).strip().lower()
        aliases = {"quick": "reflex", "fast": "reflex", "normal": "standard",
                   "deep": "deliberate", "max": "exhaustive", "maximum": "exhaustive",
                   "auto": "", "": ""}
        raw = aliases.get(raw, raw)
        if raw:
            try:
                tier = cap_tier(ReasoningTier(raw))
                return BUDGETS[tier], "requested"
            except ValueError:
                pass
    tier, rationale = assess_complexity(question)
    return BUDGETS[tier], rationale


# ──────────────────────────────────────────────────────────────────────────────
# THE PANEL'S OTHER TWO VOICES
# ──────────────────────────────────────────────────────────────────────────────

# Both of these replace the system message entirely (``instruction=``) rather
# than extending it: the critic and the chair are not ORION talking to the
# user, they are internal passes with a strict output shape, and the assistant
# persona plus tool map would be tokens spent on nothing.

CRITIC_INSTRUCTION = (
    "You are the RED TEAM reviewing draft answers before they reach anyone. "
    "Your job is to attack them, not to summarise or improve them.\n"
    "For each draft, output exactly:\n"
    "DRAFT <n> — <specialist>\n"
    "WRONG: claims that are false, unsupported, invented or subtly misleading — "
    "quote the exact words. Write 'none found' if there are none.\n"
    "MISSING: what a careful expert would have covered and this draft does not.\n"
    "COUNTER: the single strongest argument against the draft's conclusion.\n"
    "SCORE: an integer 0-10 for how much of the draft survives scrutiny.\n"
    "Then one final block:\n"
    "DISAGREEMENT: where the drafts contradict each other and which side is "
    "better supported — or 'the drafts agree' if they do.\n"
    "Be specific and quote. Invented figures, invented APIs, invented citations "
    "and confident claims about the user's own machine or files that no lookup "
    "supports are the failures that matter most. Never manufacture a flaw to "
    "look rigorous: 'none found' is a valid and useful finding."
)

CHAIR_INSTRUCTION = (
    "You are the CHAIR. You receive a question, independent specialist drafts, "
    "and a red-team critique of those drafts. Produce ONE answer for the user.\n"
    "Rules: carry forward only what survived the critique; drop or explicitly "
    "hedge anything the red team showed to be unsupported; where the drafts "
    "disagreed, resolve it and give the reason in one line; keep every concrete "
    "specific — names, numbers, code, steps — that survived, because specifics "
    "are what makes the answer worth having; state the load-bearing assumptions "
    "and the one thing most worth doing next.\n"
    "Never mention the panel, the drafts, the critique, the specialists or this "
    "process: the user asked a question, not for a meeting report. Write as "
    "ORION — calm, precise British English — and be honest about what is still "
    "uncertain."
)

AUDIT_INSTRUCTION = (
    "You are the FINAL AUDITOR. Read the question and the proposed answer. "
    "Repair the answer in place and return the corrected answer only — no "
    "commentary, no preamble, no list of changes.\n"
    "Fix: claims that are stated more confidently than the reasoning supports; "
    "invented specifics (figures, prices, APIs, citations, file paths); internal "
    "contradictions; instructions that would not actually work; anything that "
    "answers a different question than the one asked. If the answer is already "
    "sound, return it unchanged. Keep its structure, its specifics and its "
    "British English."
)


@dataclass
class Critique:
    """The red team's reading of the panel."""

    text: str = ""
    scores: dict[str, int] = field(default_factory=dict)
    disagreement: str = ""

    @property
    def applied(self) -> bool:
        return bool(self.text.strip())

    def weakest(self) -> str:
        if not self.scores:
            return ""
        return min(self.scores.items(), key=lambda pair: pair[1])[0]


@dataclass
class VerificationReport:
    """What the evidence store had to say about the answer's claims."""

    checked: int = 0
    supported: list[dict[str, Any]] = field(default_factory=list)
    contradicted: list[dict[str, Any]] = field(default_factory=list)
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    ran: bool = False

    def summary(self) -> str:
        if not self.ran:
            return "not run"
        if not self.checked:
            return "no checkable claims"
        return (f"{self.checked} claim(s) — {len(self.supported)} supported, "
                f"{len(self.contradicted)} contradicted, {len(self.unsupported)} unknown")

    def note(self) -> str:
        """The user-facing warning, emitted only when there is a conflict.

        Silence about unsupported claims is deliberate. An empty evidence store
        makes *everything* unsupported, so reporting it would train the user to
        ignore the whole section. A contradiction, by contrast, means ORION
        holds a recorded claim that disagrees with what he has just said, and
        that is always worth interrupting for.
        """
        if not self.contradicted:
            return ""
        lines = ["⚠ Evidence check — this conflicts with what I have on record:"]
        for item in self.contradicted[:3]:
            conflict = item.get("conflict") or [{}]
            lines.append(f"  • I said: {first_line(item.get('claim', ''), 140)}")
            lines.append(f"    On record ({conflict[0].get('source', 'unknown')}): "
                         f"{first_line(conflict[0].get('claim', ''), 140)}")
        return "\n".join(lines)


@dataclass
class ReasoningOutcome:
    """Everything one reasoning pass produced — answer plus its provenance."""

    question: str
    answer: str
    budget: ReasoningBudget
    rationale: str = ""
    findings: list[AgentFinding] = field(default_factory=list)
    critique: Critique = field(default_factory=Critique)
    verification: VerificationReport = field(default_factory=VerificationReport)
    elapsed_s: float = 0.0
    ok: bool = True

    @property
    def lookups(self) -> int:
        return sum(len(f.tools) for f in self.findings)

    def footer(self) -> str:
        parts = [f"{self.budget.tier.value} tier"]
        if self.findings:
            parts.append("panel: " + ", ".join(f.title for f in self.findings))
        if self.lookups:
            parts.append(f"{self.lookups} lookup(s)")
        if self.critique.applied:
            parts.append("red-team critique applied")
        if self.verification.ran:
            parts.append(f"verification: {self.verification.summary()}")
        parts.append(f"{self.elapsed_s:.1f}s")
        return "— reasoning: " + " · ".join(parts)

    def to_tool_result(self) -> ToolResult:
        blocks = [self.answer.strip()]
        note = self.verification.note()
        if note:
            blocks.append(note)
        blocks.append(self.footer())
        return ToolResult("\n\n".join(b for b in blocks if b), ok=self.ok)


# ──────────────────────────────────────────────────────────────────────────────
# CLAIM EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_COPULA = re.compile(r"\b(?:is|are|was|were|has|have|had|will|costs?|"
                     r"requires?|supports?|means?|causes?)\b", re.IGNORECASE)
_PROPER = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][A-Za-z0-9.+#-]{2,}\b")
_HEDGED = re.compile(r"\b(?:might|maybe|perhaps|i think|probably|possibly|"
                     r"consider|try|you could|it depends)\b", re.IGNORECASE)


def extract_claims(text: str, limit: int = 10) -> list[str]:
    """Sentences from *text* that assert something checkable.

    Only assertions are worth verifying. Questions, hedges, imperatives
    ("consider X", "try Y") and headings assert nothing, so putting them to the
    evidence store would produce a page of 'unknown' verdicts that says nothing
    about the answer's quality. A sentence qualifies when it carries a figure,
    a proper noun or a copula — the three shapes a factual claim takes.
    """
    body = re.sub(r"```.*?```", " ", str(text or ""), flags=re.DOTALL)   # code is not a claim
    claims: list[str] = []
    for raw_line in body.splitlines():
        line = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s*", "", raw_line).strip()
        if not line or line.startswith("#") or line.startswith("|"):
            continue
        for sentence in _SENTENCE_SPLIT.split(line):
            sentence = re.sub(r"[*_`]", "", sentence).strip()
            words = sentence.split()
            if not (4 <= len(words) <= 60) or sentence.endswith("?"):
                continue
            if _HEDGED.search(sentence):
                continue
            if not (re.search(r"\d", sentence) or _PROPER.search(sentence)
                    or _COPULA.search(sentence)):
                continue
            claims.append(sentence.rstrip(".").strip() + ".")
            if len(claims) >= max(1, limit):
                return claims
    return claims


# ──────────────────────────────────────────────────────────────────────────────
# THE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class ReasoningEngine:
    """Deliberation, critique and verification over the specialist workforce."""

    #: Direct dispatcher tools a specialist may fire unmodified. Both are pure
    #: searches with no write action, so there is no argument that could turn
    #: them into a mutation. Everything else is exposed as a pinned instrument
    #: below rather than trusted to be called safely.
    DIRECT_TOOLS: dict[str, str] = {
        "query_intelligence":
            'Search ORION\'s long-term memory — saved facts, notes, preferences, '
            'projects. ARGS: {"query": "text"}',
        "recall_conversation":
            'Search past conversations for what was said before and when. '
            'ARGS: {"query": "text"}',
    }

    def __init__(
        self,
        bus: OrionBus,
        router: Any,
        agents: AgentManager,
        *,
        dispatch: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        evidence: Any | None = None,
        research: Any | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.bus = bus
        self.router = router
        self.agents = agents
        self.dispatch = dispatch
        self.evidence = evidence
        self.research = research
        self.telemetry = telemetry
        self.last: ReasoningOutcome | None = None

    # ── public API ────────────────────────────────────────────────────────────

    async def reason(
        self,
        question: str,
        *,
        tier: str | ReasoningTier | None = None,
        context: str = "",
        agent: str = "",
    ) -> ReasoningOutcome:
        """Think about *question* to the depth its difficulty warrants."""
        question = SecuritySanitiser.guard_text(str(question or "").strip(), "reasoning.question")
        budget, rationale = resolve_tier(question, tier)
        if not question:
            return ReasoningOutcome(question, "Nothing was asked.", budget,
                                    rationale=rationale, ok=False)
        started = time.perf_counter()
        self.bus.log.emit(
            f"REASON: {budget.tier.value} tier — {rationale} ({budget.summary}).")

        panel = self._seat_panel(question, budget, agent)
        if not panel:
            return ReasoningOutcome(
                question, "No specialist is registered to answer this.", budget,
                rationale=rationale, ok=False)

        findings = await self._draft(question, context, panel, budget)
        critique = await self._critique(question, findings) if budget.critiques else Critique()
        if budget.revise and critique.applied:
            findings = await self._revise(question, context, panel, findings, critique, budget)

        answer = await self._synthesise(question, findings, critique, budget)
        if budget.critiques >= 2 and answer.strip():
            answer = await self._audit(question, answer)

        verification = VerificationReport()
        if budget.verify and self.evidence is not None:
            verification = self._verify(question, answer)
        if budget.record and self.evidence is not None:
            self._record(question, findings)

        outcome = ReasoningOutcome(
            question=question, answer=answer or "I could not form an answer.",
            budget=budget, rationale=rationale, findings=findings,
            critique=critique, verification=verification,
            elapsed_s=time.perf_counter() - started, ok=bool(answer.strip()),
        )
        self.last = outcome
        self._publish(outcome)
        return outcome

    def snapshot(self) -> dict[str, Any]:
        """Last pass, for the dashboard."""
        if self.last is None:
            return {"ran": False}
        return {
            "ran": True,
            "question": first_line(self.last.question, 120),
            "tier": self.last.budget.tier.value,
            "rationale": self.last.rationale,
            "panel": [f.title for f in self.last.findings],
            "lookups": self.last.lookups,
            "critique": self.last.critique.text[:600],
            "verification": self.last.verification.summary(),
            "elapsed_s": round(self.last.elapsed_s, 2),
        }

    # ── stage 1: independent drafts ───────────────────────────────────────────

    def _seat_panel(self, question: str, budget: ReasoningBudget,
                    agent: str) -> list[BaseAgent]:
        if agent.strip() and agent.strip().lower() not in {"auto", "any"}:
            chosen = self.agents.get(agent)
            return [chosen] if chosen is not None else []
        return self.agents.panel(question, budget.panel)

    def toolbelt(self, budget: ReasoningBudget) -> AgentToolbelt:
        """A fresh instrument tray with this tier's lookup budget.

        Each specialist gets its own tray so one lookup-happy member cannot
        starve the rest, and so the budget means the same thing regardless of
        panel size.
        """
        return AgentToolbelt(
            dispatch=self.dispatch,
            allowed=dict(self.DIRECT_TOOLS),
            extras=self._instruments(),
            budget=budget.lookups,
            bus=self.bus,
        )

    def _instruments(self) -> dict[str, tuple[str, Callable[[dict[str, Any]], Awaitable[str]]]]:
        """Purpose-shaped instruments over tools that also have write actions.

        ``second_brain`` can ingest, ``find_files`` can open what it finds,
        ``research`` can start a background run. Rather than trusting a model to
        stay on the read-only branch of a multi-action tool, each instrument
        here pins the action and exposes exactly one argument, so the unsafe
        branches are unreachable rather than merely discouraged.
        """
        instruments: dict[str, tuple[str, Callable[[dict[str, Any]], Awaitable[str]]]] = {}

        if self.dispatch is not None:
            async def _find_files(args: dict[str, Any]) -> str:
                result = await self.dispatch(
                    "find_files", {"query": str(args.get("query") or args.get("name") or ""),
                                   "open": False})
                return str(getattr(result, "text", result))

            async def _knowledge_recall(args: dict[str, Any]) -> str:
                result = await self.dispatch(
                    "second_brain", {"action": "recall",
                                     "query": str(args.get("query") or args.get("topic") or "")})
                return str(getattr(result, "text", result))

            async def _situation(_args: dict[str, Any]) -> str:
                result = await self.dispatch("awareness", {"action": "situation"})
                return str(getattr(result, "text", result))

            instruments["find_files"] = (
                'Find files and folders on the user\'s machine by name (it only '
                'looks; it never opens them). ARGS: {"query": "name fragment"}',
                _find_files)
            instruments["knowledge_recall"] = (
                'Ask ORION\'s local knowledge graph what it knows about a topic, '
                'entity or event. ARGS: {"query": "topic"}', _knowledge_recall)
            instruments["situation"] = (
                'The current situation: active projects, priorities, deadlines and '
                'focus. ARGS: {}', _situation)

        if self.evidence is not None:
            async def _evidence_lookup(args: dict[str, Any]) -> str:
                query = str(args.get("query") or args.get("claim") or "")
                rows = await asyncio.to_thread(self.evidence.search_claims, query, 6)
                if not rows:
                    return "No recorded evidence matches that."
                return "\n".join(
                    f"- {r['claim']} (source: {r['source']}, confidence {r['confidence']})"
                    for r in rows)

            instruments["evidence_lookup"] = (
                'Search ORION\'s evidence store for recorded claims with their '
                'sources and confidence. ARGS: {"query": "topic or claim"}',
                _evidence_lookup)

        if self.research is not None and hasattr(self.research, "gather_sources"):
            async def _web_lookup(args: dict[str, Any]) -> str:
                query = str(args.get("query") or args.get("topic") or "")
                if not query.strip():
                    return "No query supplied."
                sources = await self.research.gather_sources(query)
                if not sources:
                    return "Nothing came back — offline, or no source covers this."
                return "\n".join(
                    f"- {s.get('title', '')}: {str(s.get('snippet', ''))[:300]}"
                    f"{' [' + s['url'] + ']' if s.get('url') else ''}"
                    for s in sources[:6])

            instruments["web_lookup"] = (
                'Look a topic up on the live web (encyclopaedia summary plus recent '
                'news headlines). Silent — nothing opens on screen. '
                'ARGS: {"query": "topic"}', _web_lookup)

        return instruments

    async def _draft(self, question: str, context: str, panel: Sequence[BaseAgent],
                     budget: ReasoningBudget) -> list[AgentFinding]:
        """Every seated specialist answers independently, at the same time.

        Independence is the whole value of a panel: members who can see each
        other's drafts converge on the first plausible framing instead of
        covering the space. They therefore share nothing but the question.
        """
        brief = ""
        if len(panel) > 1:
            brief = (
                "You are one member of a panel answering this independently. "
                "Answer from YOUR specialism and say what only you would notice; "
                "do not hedge toward a consensus you cannot see, and do not try "
                "to cover the other specialisms. Be concise and concrete."
            )
        if len(panel) == 1 or not parallelism_enabled():
            return [await member.investigate(question, context, self.toolbelt(budget), brief)
                    for member in panel]
        semaphore = asyncio.Semaphore(max_parallel())

        async def _one(member: BaseAgent) -> AgentFinding:
            async with semaphore:
                return await member.investigate(
                    question, context, self.toolbelt(budget), brief)

        results = await asyncio.gather(*(_one(m) for m in panel), return_exceptions=True)
        findings: list[AgentFinding] = []
        for member, result in zip(panel, results):
            if isinstance(result, BaseException):
                # One specialist failing must not take the panel with it.
                self.bus.log.emit(
                    f"REASON: {member.title} failed - {first_line(result, 120)}")
                findings.append(AgentFinding(member.name, member.title,
                                             "", ok=False))
                continue
            findings.append(result)
        return [f for f in findings if f.answer.strip()] or findings

    # ── stage 2: adversarial critique ─────────────────────────────────────────

    async def _critique(self, question: str, findings: Sequence[AgentFinding]) -> Critique:
        usable = [f for f in findings if f.answer.strip()]
        if not usable:
            return Critique()
        drafts = "\n\n".join(
            f"DRAFT {i} — {f.title}\n{f.answer.strip()[:2400]}"
            for i, f in enumerate(usable, 1))
        prompt = f"QUESTION:\n{question}\n\n{drafts}"
        try:
            _profile, text = await self.router.generate_text(
                prompt, instruction=CRITIC_INSTRUCTION, task="reason.critic")
        except Exception as exc:
            self.bus.log.emit(f"REASON: red team unavailable - {first_line(exc, 120)}")
            return Critique()
        critique = Critique(text=text.strip())
        for index, score in enumerate(re.findall(r"SCORE:\s*(\d{1,2})", text)):
            if index < len(usable):
                critique.scores[usable[index].title] = max(0, min(10, int(score)))
        match = re.search(r"DISAGREEMENT:\s*(.+)", text, re.DOTALL)
        if match:
            critique.disagreement = match.group(1).strip()[:800]
        self.bus.log.emit(
            "REASON: red team reported — "
            + (", ".join(f"{title} {score}/10" for title, score in critique.scores.items())
               or "no scores parsed") + ".")
        return critique

    async def _revise(self, question: str, context: str, panel: Sequence[BaseAgent],
                      findings: Sequence[AgentFinding], critique: Critique,
                      budget: ReasoningBudget) -> list[AgentFinding]:
        """Each specialist answers again, having read the attack on its draft."""
        by_name = {f.agent: f for f in findings}
        brief = (
            "Your first answer has been attacked by a red team. Their critique "
            "follows. Concede what is genuinely wrong, defend with reasons what "
            "you still believe, and fill the gaps they identified. Return your "
            "improved answer only — no commentary on the critique itself.\n\n"
            f"RED TEAM:\n{critique.text[:3000]}"
        )
        revised: list[AgentFinding] = []
        for member in panel:
            previous = by_name.get(member.name)
            if previous is None or not previous.answer.strip():
                continue
            request = f"{question}\n\nYOUR FIRST ANSWER:\n{previous.answer.strip()[:2400]}"
            try:
                finding = await member.investigate(
                    request, context, self.toolbelt(budget), brief)
            except Exception as exc:
                self.bus.log.emit(f"REASON: revision failed for {member.title} "
                                  f"- {first_line(exc, 120)}")
                revised.append(previous)
                continue
            if finding.answer.strip():
                # Readings from both rounds are what the answer actually rests on.
                finding.tools = list(previous.tools) + list(finding.tools)
                revised.append(finding)
            else:
                revised.append(previous)
        return revised or list(findings)

    # ── stage 3: synthesis and audit ──────────────────────────────────────────

    async def _synthesise(self, question: str, findings: Sequence[AgentFinding],
                          critique: Critique, budget: ReasoningBudget) -> str:
        usable = [f for f in findings if f.answer.strip()]
        if not usable:
            return ""
        if len(usable) == 1 and not critique.applied:
            return usable[0].answer.strip()
        drafts = "\n\n".join(
            f"DRAFT {i} — {f.title}\n{f.answer.strip()[:2400]}"
            for i, f in enumerate(usable, 1))
        prompt = (f"QUESTION:\n{question}\n\n{drafts}\n\n"
                  f"RED-TEAM CRITIQUE:\n{critique.text[:3000] or '(none)'}")
        try:
            _profile, text = await self.router.generate_text(
                prompt, instruction=CHAIR_INSTRUCTION, task="reason.chair")
            return text.strip()
        except Exception as exc:
            self.bus.log.emit(f"REASON: chair unavailable - {first_line(exc, 120)}")
            # Falling back to the best-scored draft beats returning nothing.
            best = max(usable, key=lambda f: critique.scores.get(f.title, 5))
            return best.answer.strip()

    async def _audit(self, question: str, answer: str) -> str:
        try:
            _profile, text = await self.router.generate_text(
                f"QUESTION:\n{question}\n\nPROPOSED ANSWER:\n{answer[:6000]}",
                instruction=AUDIT_INSTRUCTION, task="reason.audit")
        except Exception as exc:
            self.bus.log.emit(f"REASON: audit skipped - {first_line(exc, 120)}")
            return answer
        repaired = text.strip()
        # An auditor that returns a fragment has misunderstood the brief; the
        # unaudited answer is better than a truncated one.
        return repaired if len(repaired) >= len(answer) * 0.5 else answer

    # ── stage 4: verification against the evidence store ──────────────────────

    def _verify(self, question: str, answer: str) -> VerificationReport:
        report = VerificationReport(ran=True)
        topic = _topic_of(question)
        for claim in extract_claims(answer, limit=12):
            try:
                assessment = self.evidence.assess_claim(claim, topic=topic)
            except Exception as exc:
                self.bus.log.emit(f"REASON: verification fault - {first_line(exc, 120)}")
                break
            report.checked += 1
            bucket = {"supported": report.supported,
                      "contradicted": report.contradicted}.get(
                          assessment["verdict"], report.unsupported)
            bucket.append(assessment)
        if report.contradicted:
            self.bus.log.emit(
                f"REASON: verification found {len(report.contradicted)} "
                "contradiction(s) with recorded evidence.")
        return report

    def _record(self, question: str, findings: Sequence[AgentFinding]) -> None:
        """Write back what the *instruments* saw — never what the model said.

        This is the line that keeps the evidence store worth consulting. A
        claim recorded here came from a lookup and carries the instrument as its
        source; recording the panel's own prose would let a confident invention
        become tomorrow's corroborating evidence, and the store would slowly
        launder hallucination into fact.
        """
        topic = _topic_of(question)
        readings: list[ToolInvocation] = [
            reading for finding in findings for reading in finding.tools if reading.ok
        ]
        recorded = 0
        for reading in readings[:6]:
            for claim in extract_claims(reading.observation, limit=3):
                try:
                    if self.evidence.record_claim(
                            topic, claim, source=f"instrument:{reading.tool}"):
                        recorded += 1
                except Exception:
                    return
        if recorded and self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("reasoning.claims_recorded", recorded)
            except Exception:
                pass

    # ── reporting ─────────────────────────────────────────────────────────────

    def _publish(self, outcome: ReasoningOutcome) -> None:
        self.bus.log.emit(
            f"REASON: {outcome.budget.tier.value} complete in "
            f"{outcome.elapsed_s:.1f}s — {len(outcome.findings)} specialist(s), "
            f"{outcome.lookups} lookup(s), verification {outcome.verification.summary()}.")
        try:
            self.bus.dashboard_event.emit("reasoning", self.snapshot())
        except Exception:
            pass
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("reasoning.runs")
                self.telemetry.metrics.incr(f"reasoning.tier.{outcome.budget.tier.value}")
                self.telemetry.metrics.observe("reasoning.ms", outcome.elapsed_s * 1000.0)
            except Exception:
                pass


def _topic_of(question: str) -> str:
    """A stable topic key for the evidence store, from the question's content words."""
    words = [w for w in re.findall(r"[A-Za-z0-9']+", str(question or "").lower())
             if len(w) > 3][:6]
    return " ".join(words) or "general"


__all__ = [
    "AUDIT_INSTRUCTION",
    "BUDGETS",
    "CHAIR_INSTRUCTION",
    "CRITIC_INSTRUCTION",
    "Critique",
    "ReasoningBudget",
    "ReasoningEngine",
    "ReasoningOutcome",
    "ReasoningTier",
    "TIER_ORDER",
    "VerificationReport",
    "assess_complexity",
    "cap_tier",
    "extract_claims",
    "resolve_tier",
]
