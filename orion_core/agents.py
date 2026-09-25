"""
Agent framework — the Mark VIII specialist workforce.

    BaseAgent        — persona + routing keywords over the ProviderRouter.
    AgentToolbelt    — the read-only instrument tray a specialist may reach for
                       while it thinks (Track C).
    AgentManager     — registry and dynamic router: scores an incoming
                       request against every agent's keyword profile and
                       dispatches to the best match (or an explicit choice).
    DesktopAgent     — functional agent owning host control: launching
                       applications, window management, media keys, browsers,
                       development tools, and quick access to Outlook/Notion.
    Specialists      — Digital Marketing, Coding, Research & Analysis,
                       Neuroscience, AI/ML and Cyber Security: domain personas
                       that answer through the best available text provider.
                       (Design & Art, Fashion and Entertainment were retired in
                       Mark XXXI — chat personas with no instruments behind
                       them, which is not what a specialist here is for.)

Specialists answer in one of two modes.  ``handle`` is the original stateless
pass: persona in, provider out.  ``investigate`` (Track C) is the tool-using
pass — the specialist may look things up, read the observation and think again
before it commits to an answer, so it argues from what is actually true of this
machine and this user rather than from the shape of the question.

When no text provider is configured either mode returns the specialist *brief*
instead, so the live multimodal model can adopt the persona and answer
directly — the user always gets a specialist-grade response.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import webbrowser  # noqa: F401  (kept: see .browser.open_url)
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import urlparse

import psutil

from .bus import OrionBus
from .concurrency import ToolClass, classify, max_parallel, parallelism_enabled
from .data import ToolResult
from .providers import ProviderRouter
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line, fold_title, utc_stamp
from .browser import open_url


# ──────────────────────────────────────────────────────────────────────────────
# BASE AGENT
# ──────────────────────────────────────────────────────────────────────────────

# The operating method every specialist follows before answering — the single
# biggest lever on answer quality. Injected into each system instruction so
# even a small local model reasons like a senior practitioner.
OPERATING_METHOD = (
    "OPERATING METHOD (follow silently, do not narrate the steps):\n"
    "1. Restate the real objective in one line — what outcome does the user "
    "actually want, not just the literal words?\n"
    "2. Note any assumptions you must make (budget, audience, platform, "
    "skill level, constraints) and state the load-bearing ones explicitly.\n"
    "3. Reason from first principles and your domain expertise before writing "
    "the answer; when debugging or diagnosing, reason from the evidence given, "
    "never guess.\n"
    "4. Give a specific, actionable answer — concrete examples, names, numbers, "
    "code or steps — not generic advice. Prefer the shortest answer that is "
    "genuinely complete.\n"
    "5. Close with the single most valuable next action, and flag any risk or "
    "trade-off the user should know about.\n"
    "Be honest about uncertainty and never invent facts, APIs, prices or "
    "citations. Keep ORION's calm, professional British register."
)


# The tool protocol a specialist follows while investigating.  Deliberately
# line-oriented rather than JSON-wrapped: Track A's forge taught us that source
# and prose travelling inside JSON string values die on one bad escape.  Args
# are small enough that a single JSON object per call is safe, and the parser
# below accepts three spellings of the same intent anyway.
TOOL_PROTOCOL = (
    "INSTRUMENTS. You may look something up before you answer, instead of "
    "guessing. To use an instrument, reply with NOTHING but call lines:\n"
    "    TOOL: <name>\n"
    "    ARGS: {\"key\": \"value\"}\n"
    "You may issue several calls at once (they run simultaneously). You will "
    "then be shown the results and asked again. When you have what you need — "
    "or when no instrument can help — reply with your final answer as ordinary "
    "prose, with no TOOL line anywhere in it.\n"
    "Every instrument is read-only: they observe, they never change anything. "
    "Use them when the answer depends on this user's actual files, memory, "
    "history or machine, or on facts you should not invent. Do not narrate the "
    "lookups in your final answer; just use what you learned, and say plainly "
    "when a lookup returned nothing."
)

_TOOL_LINE = re.compile(r"^\s*TOOL\s*[:=]\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)
_FENCED_TOOL = re.compile(r"```(?:tool|json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ToolInvocation:
    """One instrument reading taken during an investigation."""

    tool: str
    args: dict[str, Any]
    ok: bool
    observation: str

    def as_line(self, width: int = 900) -> str:
        head = f"{self.tool}({json.dumps(self.args, default=str)[:160]})"
        body = self.observation.strip()[:width] or "(no result)"
        return f"[{head}] {'' if self.ok else 'FAILED — '}{body}"


@dataclass
class AgentFinding:
    """What one specialist concluded, and what it looked at to get there."""

    agent: str
    title: str
    answer: str
    provider: str = ""
    tools: list[ToolInvocation] = field(default_factory=list)
    ok: bool = True

    @property
    def grounded(self) -> bool:
        """True when at least one instrument reading fed the answer."""
        return any(t.ok for t in self.tools)


def parse_tool_calls(text: str) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """Split a specialist's reply into (tool calls, prose).

    Accepts three spellings so a model that half-remembers the protocol is
    still understood: ``TOOL:``/``ARGS:`` line pairs, a fenced block holding
    ``{"tool": …, "args": {…}}``, and a bare JSON object of the same shape.
    Returns an empty call list when the reply is a final answer, which is the
    loop's termination condition — a model that simply answers is never forced
    into a tool call it did not want.
    """
    raw = str(text or "")
    calls: list[tuple[str, dict[str, Any]]] = []

    for match in _FENCED_TOOL.finditer(raw):
        payload = _coerce_call(match.group(1))
        if payload is not None:
            calls.append(payload)
    if calls:
        return calls, _FENCED_TOOL.sub("", raw).strip()

    for match in _TOOL_LINE.finditer(raw):
        name = match.group(1)
        # ARGS may sit on the next line, or be omitted entirely for a no-arg tool.
        tail = raw[match.end():]
        args: dict[str, Any] = {}
        args_match = re.match(r"\s*ARGS\s*[:=]\s*(\{.*?\})\s*(?:\n|$)", tail, re.DOTALL)
        if args_match:
            args = _loads_or_empty(args_match.group(1))
        calls.append((name, args))
    if calls:
        # Prose accompanying a call is scratch reasoning, not an answer.
        return calls, ""

    stripped = raw.strip()
    if stripped.startswith("{"):
        payload = _coerce_call(stripped)
        if payload is not None:
            return [payload], ""
    return [], stripped


def _answer_text(prose: str, reply: str, readings: Sequence[ToolInvocation]) -> str:
    """The user-facing answer from a reply that may still be asking for tools.

    A specialist can run out of budget mid-request, or keep asking for readings
    until the loop closes on it.  Echoing the raw reply then leaks ``TOOL:``
    protocol lines into the user's answer, so the protocol is stripped; if
    nothing but protocol was said, ORION reports the readings honestly rather
    than pretending to have concluded something.
    """
    if prose.strip():
        return prose.strip()
    stripped = re.sub(r"^\s*(?:TOOL|ARGS)\s*[:=].*$", "", reply, flags=re.MULTILINE)
    stripped = _FENCED_TOOL.sub("", stripped).strip()
    if stripped:
        return stripped
    if readings:
        return ("I could not settle on a conclusion. What I found:\n"
                + "\n".join(r.as_line(300) for r in readings))
    return "I could not form an answer to that."


def _coerce_call(blob: str) -> tuple[str, dict[str, Any]] | None:
    data = _loads_or_empty(blob)
    name = str(data.get("tool") or data.get("name") or "").strip()
    if not name:
        return None
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    return name, dict(args or {})


def _loads_or_empty(blob: str) -> dict[str, Any]:
    try:
        data = json.loads(blob)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


class AgentToolbelt:
    """
    The read-only instrument tray a specialist may reach for while it thinks.

    Two guarantees make this safe to hand to a language model:

    * **Allowlist.** Only tools named in *allowed* (or supplied as *extras*)
      exist at all; anything else is refused with the catalogue attached.
    * **Class gate.** Every dispatcher tool is additionally checked against
      :func:`concurrency.classify` and refused unless it is PARALLEL — the
      class reserved for calls that observe without writing.  So even if a
      side-effecting tool is added to the allowlist by mistake, a specialist
      still cannot fire it.  A reasoning pass can never move a window, send a
      message or delete a file; the worst it can do is read too much.

    The budget is a hard count of readings across the whole investigation, not
    per step, so a specialist cannot spend the turn in a lookup spiral.
    """

    def __init__(
        self,
        dispatch: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        allowed: Mapping[str, str] | None = None,
        extras: Mapping[str, tuple[str, Callable[[dict[str, Any]], Awaitable[str]]]] | None = None,
        budget: int = 4,
        bus: OrionBus | None = None,
    ) -> None:
        self._dispatch = dispatch
        self._allowed = dict(allowed or {})
        self._extras = dict(extras or {})
        self.budget = max(0, int(budget))
        self.spent = 0
        self.bus = bus

    # ── introspection ─────────────────────────────────────────────────────────

    def available(self) -> bool:
        return self.budget > 0 and bool(self._extras or (self._dispatch and self._allowed))

    def remaining(self) -> int:
        return max(0, self.budget - self.spent)

    def names(self) -> list[str]:
        return sorted({*self._extras, *self._allowed})

    def catalogue(self) -> str:
        """The instrument list as the specialist sees it."""
        lines = [f"  {name} — {self._describe(name)}" for name in self.names()]
        return "AVAILABLE INSTRUMENTS:\n" + "\n".join(lines)

    def _describe(self, name: str) -> str:
        if name in self._extras:
            return self._extras[name][0]
        return self._allowed.get(name, "")

    # ── execution ─────────────────────────────────────────────────────────────

    async def run(self, tool: str, args: dict[str, Any]) -> ToolInvocation:
        name = str(tool or "").strip()
        args = dict(args or {})
        if self.remaining() <= 0:
            return ToolInvocation(name, args, False,
                                  "Instrument budget spent — answer from what you already have.")
        self.spent += 1
        if name in self._extras:
            try:
                observation = await self._extras[name][1](args)
                return ToolInvocation(name, args, True, str(observation))
            except Exception as exc:
                return ToolInvocation(name, args, False, first_line(exc, 200))
        if name not in self._allowed or self._dispatch is None:
            return ToolInvocation(
                name, args, False,
                f"No such instrument. Available: {', '.join(self.names()) or 'none'}.")
        if classify(name, args) is not ToolClass.PARALLEL:
            # Belt and braces: the allowlist and the concurrency table must
            # agree, and when they do not the safe reading wins.
            return ToolInvocation(
                name, args, False,
                "That instrument is not read-only, so it is unavailable while reasoning.")
        try:
            result = await self._dispatch(name, args)
        except Exception as exc:
            return ToolInvocation(name, args, False, first_line(exc, 200))
        ok = bool(getattr(result, "ok", True))
        return ToolInvocation(name, args, ok, str(getattr(result, "text", result)))

    async def run_many(self, calls: Sequence[tuple[str, dict[str, Any]]]) -> list[ToolInvocation]:
        """Take several readings at once — they are read-only by construction.

        Honours the same ceiling and the same kill switch as the tool executor
        in the live worker, so ``ORION_PARALLEL_TOOLS=0`` makes reasoning
        strictly sequential too.
        """
        wanted = list(calls)[: self.remaining()]
        if not wanted:
            return []
        if len(wanted) == 1 or not parallelism_enabled():
            return [await self.run(name, args) for name, args in wanted]
        semaphore = asyncio.Semaphore(max_parallel())

        async def _guarded(name: str, args: dict[str, Any]) -> ToolInvocation:
            async with semaphore:
                return await self.run(name, args)

        return list(await asyncio.gather(*(_guarded(n, a) for n, a in wanted)))


class BaseAgent:
    """
    A specialist persona routed through the provider layer.

    Subclasses (or instances) define:
        name        — registry key, e.g. "coding"
        title       — human label for the dashboard
        expertise   — one-line domain summary (dashboard + routing rationale)
        primary     — high-signal regex fragments (weight 3 in scoring)
        keywords    — supporting regex fragments (weight 1 in scoring)
        persona     — system-message extension establishing the specialism

    Scoring is *weighted*: a request that names the domain outright (a primary
    hit) outranks one that merely brushes past a supporting term, so routing is
    far less trigger-happy than a flat keyword count.
    """

    name: str = "general"
    title: str = "General Aide"
    expertise: str = "general assistance"
    primary: Sequence[str] = ()
    keywords: Sequence[str] = ()
    persona: str = ""

    _PRIMARY_WEIGHT = 3
    _SECONDARY_WEIGHT = 1

    def __init__(self, router: ProviderRouter, bus: OrionBus) -> None:
        self.router = router
        self.bus = bus
        self._primary_res = [re.compile(k, re.IGNORECASE) for k in self.primary]
        self._keyword_res = [re.compile(k, re.IGNORECASE) for k in self.keywords]

    def score(self, request: str) -> int:
        """Weighted relevance score: primary signals dominate secondary ones."""
        primary = sum(1 for p in self._primary_res if p.search(request))
        secondary = sum(1 for p in self._keyword_res if p.search(request))
        return primary * self._PRIMARY_WEIGHT + secondary * self._SECONDARY_WEIGHT

    def system_extra(self) -> str:
        """Full specialist instruction = persona + shared operating method."""
        return f"{self.persona}\n\n{OPERATING_METHOD}"

    async def handle(self, request: str, context: str = "") -> ToolResult:
        """Answer *request* in this agent's specialist capacity."""
        request = SecuritySanitiser.guard_text(str(request or "").strip(), f"agent.{self.name}")
        if not request:
            return ToolResult("No request supplied to the specialist agent.", ok=False)
        prompt = request if not context.strip() else f"{request}\n\nContext:\n{context.strip()[:2000]}"
        instruction = self.system_extra()
        if not self.router.has_text_fallback():
            # No dedicated provider — hand the full brief back so the live model
            # can adopt the specialism itself. Never leave the user empty-handed.
            return ToolResult(
                f"[Specialist brief — adopt this role and answer directly]\n"
                f"{instruction}\n\nRequest: {prompt}"
            )
        try:
            profile, response = await self.router.generate_text(
                prompt, system_extra=instruction, task=self.name)
            self.bus.agent_activity.emit(self.title, first_line(request, 90))
            return ToolResult(f"[{self.title} via {profile.name}]\n{response}")
        except Exception as exc:
            return ToolResult(
                f"The {self.title} could not reach a provider: {first_line(exc)}", ok=False
            )

    # ── tool-using investigation (Track C) ────────────────────────────────────

    MAX_INVESTIGATION_STEPS = 3

    async def investigate(
        self,
        request: str,
        context: str = "",
        toolbelt: AgentToolbelt | None = None,
        brief: str = "",
    ) -> AgentFinding:
        """Answer *request*, reaching for instruments when the answer needs them.

        A bounded think → look → think loop.  Each pass the specialist either
        asks for readings or delivers prose; prose ends the loop.  With no
        toolbelt (or no budget left in it) this degrades to exactly one
        provider call — the same behaviour as :meth:`handle`, so the loop costs
        nothing when there is nothing to look up.
        """
        request = SecuritySanitiser.guard_text(str(request or "").strip(), f"agent.{self.name}")
        if not request:
            return AgentFinding(self.name, self.title, "No request supplied.", ok=False)

        instruction = self.system_extra()
        if brief.strip():
            instruction = f"{instruction}\n\n{brief.strip()}"
        usable = toolbelt is not None and toolbelt.available()
        if usable and toolbelt is not None:
            instruction = f"{instruction}\n\n{TOOL_PROTOCOL}\n{toolbelt.catalogue()}"

        if not self.router.has_text_fallback():
            # No dedicated provider — hand the brief back so the live model can
            # adopt the specialism itself, exactly as ``handle`` does.
            return AgentFinding(
                self.name, self.title,
                f"[Specialist brief — adopt this role and answer directly]\n"
                f"{instruction}\n\nRequest: {request}",
            )

        base = request if not context.strip() else f"{request}\n\nContext:\n{context.strip()[:2000]}"
        readings: list[ToolInvocation] = []
        provider_name = ""
        prompt = base
        answer = ""
        for _step in range(max(1, self.MAX_INVESTIGATION_STEPS)):
            try:
                profile, reply = await self.router.generate_text(
                    prompt, system_extra=instruction, task=f"reason.{self.name}")
                provider_name = getattr(profile, "name", "") or provider_name
            except Exception as exc:
                return AgentFinding(
                    self.name, self.title,
                    f"The {self.title} could not reach a provider: {first_line(exc)}",
                    provider=provider_name, tools=readings, ok=False)
            calls, prose = parse_tool_calls(reply)
            if not calls or toolbelt is None or toolbelt.remaining() <= 0:
                answer = _answer_text(prose, reply, readings)
                break
            fresh = await toolbelt.run_many(calls)
            readings.extend(fresh)
            if self.bus is not None and fresh:
                self.bus.log.emit(
                    f"REASON: {self.title} consulted "
                    + ", ".join(sorted({r.tool for r in fresh})) + ".")
            prompt = (
                f"{base}\n\nINSTRUMENT READINGS SO FAR:\n"
                + "\n".join(r.as_line() for r in readings)
                + "\n\nEither request further readings, or give your final answer now."
            )
        else:
            # Loop exhausted without prose: force a close on what we have.
            answer = await self._forced_close(base, readings, instruction)

        if self.bus is not None:
            self.bus.agent_activity.emit(self.title, first_line(request, 90))
        return AgentFinding(self.name, self.title, answer.strip(),
                            provider=provider_name, tools=readings, ok=bool(answer.strip()))

    async def _forced_close(self, base: str, readings: list[ToolInvocation],
                            instruction: str) -> str:
        """Final pass with the instruments withdrawn — no more lookups, answer now."""
        prompt = (
            f"{base}\n\nINSTRUMENT READINGS:\n"
            + ("\n".join(r.as_line() for r in readings) or "(none)")
            + "\n\nNo further lookups are possible. Give your final answer now, "
            "stating plainly what remains unverified."
        )
        try:
            _profile, reply = await self.router.generate_text(
                prompt, system_extra=instruction, task=f"reason.{self.name}")
            _calls, prose = parse_tool_calls(reply)
            return _answer_text(prose, reply, readings)
        except Exception as exc:
            return f"The {self.title} could not close its answer: {first_line(exc)}"


# ──────────────────────────────────────────────────────────────────────────────
# SPECIALIST PERSONAS
# ──────────────────────────────────────────────────────────────────────────────

class DigitalMarketingAgent(BaseAgent):
    name = "marketing"
    title = "Digital Marketing Agent"
    expertise = "brand growth, SEO/SEM, content, funnels, paid acquisition"
    primary = (
        r"\bmarket(?:ing)?\b", r"\bseo\b", r"\bsem\b", r"\bcampaign\b",
        r"\bfunnel\b", r"\bconversion\b", r"\bgo[- ]?to[- ]?market\b",
        r"\bcontent strategy\b", r"\bpaid ads?\b", r"\bppc\b", r"\bgrowth\b",
    )
    keywords = (
        r"\bbrand(?:ing)?\b", r"\bsocial media\b", r"\bengagement\b",
        r"\baudience\b", r"\bcopywriting\b", r"\badvert", r"\bnewsletter\b",
        r"\bemail\b", r"\binstagram\b", r"\btiktok\b", r"\blinkedin\b",
        r"\bctr\b", r"\bcac\b", r"\broas\b", r"\bcta\b", r"\blead\b",
        r"\bretention\b", r"\banalytics\b", r"\binfluencer\b",
    )
    persona = (
        "SPECIALIST MODE — Digital Marketing. You are ORION's growth strategist: "
        "expert in brand positioning, SEO/SEM, content strategy, social and "
        "video growth, funnel and landing-page CRO, lifecycle email, paid "
        "acquisition and marketing analytics. Think in the funnel (awareness → "
        "consideration → conversion → retention) and in unit economics (CAC, "
        "LTV, ROAS, payback). Recommend platform-specific tactics with example "
        "hooks/angles, the metric each move should shift, a rough timeline and a "
        "cheap test to validate before spending. State budget and audience "
        "assumptions up front. Distinguish quick wins from compounding bets."
    )


class CodingAgent(BaseAgent):
    name = "coding"
    title = "Coding Agent"
    expertise = "software engineering, debugging, architecture, performance"
    primary = (
        r"\bcode\b", r"\bcoding\b", r"\bdebug", r"\brefactor\b", r"\bbug\b",
        r"\bstack ?trace\b", r"\btraceback\b", r"\bexception\b", r"\bcompile",
        r"\balgorithm\b", r"\bfunction\b", r"\bclass\b", r"\bunit test\b",
    )
    keywords = (
        r"\bpython\b", r"\bjavascript\b", r"\btypescript\b", r"\brust\b",
        r"\bgo(?:lang)?\b", r"\bc\+\+\b", r"\bjava\b", r"\bpytest\b", r"\bapi\b",
        r"\bscript\b", r"\bregex\b", r"\brepo(?:sitory)?\b", r"\bgit\b",
        r"\bdatabase\b", r"\bsql\b", r"\basync\b", r"\bthread\b", r"\bdocker\b",
        r"\bendpoint\b", r"\bframework\b", r"\bperformance\b", r"\bmemory leak\b",
    )
    persona = (
        "SPECIALIST MODE — Software Engineering. You are ORION's principal "
        "engineer: expert across Python, JavaScript/TypeScript, Rust, Go, "
        "systems design, concurrency, databases, testing, performance and "
        "architecture. When debugging, anchor every hypothesis to the actual "
        "error text, stack trace or observed behaviour — reproduce the failure "
        "in your head before proposing a fix, and give the root cause, not just "
        "a patch. Prefer minimal, correct, idiomatic solutions; show only the "
        "code that clarifies; state Big-O and trade-offs where they matter; call "
        "out edge cases, race conditions and security pitfalls. Never invent "
        "APIs, flags or library behaviour — if unsure, say how to verify."
    )


class ResearchAnalysisAgent(BaseAgent):
    name = "research"
    title = "Research & Analysis Agent"
    expertise = "research synthesis, comparison, evaluation, structured analysis"
    primary = (
        r"\bresearch\b", r"\banalys[ei]", r"\bevaluat", r"\bcompare\b",
        r"\bcomparison\b", r"\bpros and cons\b", r"\btrade[- ]?offs?\b",
        r"\bsummar(?:y|ise|ize)\b", r"\bexplain\b", r"\bassess", r"\bbreak ?down\b",
    )
    keywords = (
        r"\bwhy\b", r"\bhow does\b", r"\bwhat is\b", r"\bimplications?\b",
        r"\bevidence\b", r"\bdata\b", r"\bstudy\b", r"\breport\b",
        r"\boptions?\b", r"\bdecision\b", r"\brecommendation\b", r"\bversus\b",
        r"\bvs\.?\b", r"\bshould i\b", r"\bimpact\b", r"\bframework\b",
    )
    persona = (
        "SPECIALIST MODE — Research & Analysis. You are ORION's analyst: you turn "
        "a messy question into a clear, structured answer. Separate what is known "
        "from what is inferred and what is unknown; weigh evidence rather than "
        "asserting; and when comparing options, use explicit criteria and, where "
        "it helps, a compact comparison table with a clear recommendation and the "
        "reasoning behind it. Surface second-order effects, risks and the "
        "strongest counter-argument to your own conclusion. Distinguish fact from "
        "opinion, cite the *type* of source you would check, and never fabricate "
        "specific figures, quotes or citations — flag what needs verifying."
    )


class NeuroscienceAgent(BaseAgent):
    name = "neuroscience"
    title = "Neuroscience & Cognition Agent"
    expertise = "neuroscience, neural engineering, cognitive & behavioural psychology"
    primary = (
        r"\bneuro(?:science|nal|engineering)?\b", r"\bbrain\b", r"\bcognit",
        r"\bpsycholog", r"\bneuron\b", r"\bsynap", r"\bbci\b",
        r"\bbrain[- ]?computer interface\b", r"\bEEG\b", r"\bfMRI\b",
        r"\bneuroplasticity\b", r"\bneurotransmitter\b",
        r"\bdopamine\b", r"\bserotonin\b", r"\bhippocampus\b", r"\bamygdala\b",
        r"\bneural (?:engineering|network|interface|circuit)\b",
    )
    keywords = (
        r"\bdopamine\b", r"\bserotonin\b", r"\bcortex\b", r"\bhippocampus\b",
        r"\bamygdala\b", r"\bmemory\b", r"\battention\b", r"\bperception\b",
        r"\bconditioning\b", r"\bbehaviou?r\b", r"\bstimulus\b", r"\breinforc",
        r"\bcognitive load\b", r"\bworking memory\b", r"\bplasticity\b",
        r"\baction potential\b", r"\bconnectom", r"\bpsychophysic",
    )
    persona = (
        "SPECIALIST MODE — Neuroscience, Neural Engineering & Psychology. You are "
        "ORION's neuroscientist, tuned to the user's own field of study. Explain "
        "mechanisms from molecules to behaviour — ion channels and neurotransmitters, "
        "circuits and systems, cognition and behaviour — and connect them across "
        "those levels rather than staying at one. When it is a study or a claim, "
        "weigh the evidence quality (sample, design, effect size, replication) and "
        "separate established finding from hypothesis. Teach like a patient graduate "
        "supervisor: define terms the first time, give a concrete example or analogy, "
        "then the precise version. For neural engineering / BCI questions, cover the "
        "signal (EEG/EMG/spikes), the transducer, decoding, and the biocompatibility "
        "and ethics constraints. Never invent citations, statistics or study results "
        "— name the type of source to check and flag what is contested."
    )


class AIMLAgent(BaseAgent):
    name = "aiml"
    title = "AI / Machine-Learning Agent"
    expertise = "machine learning, deep learning, LLMs, model training and evaluation"
    primary = (
        r"\bmachine learning\b", r"\bdeep learning\b", r"\bneural net", r"\bml\b",
        r"\bLLM\b", r"\btransformer\b", r"\bfine[- ]?tun", r"\bembedding\b",
        r"\btraining (?:data|loop|run)\b", r"\binference\b", r"\bmodel\b",
        r"\bgradient\b", r"\bbackprop", r"\bRAG\b",
    )
    keywords = (
        r"\bpytorch\b", r"\btensorflow\b", r"\bhugging ?face\b", r"\btoken(?:iz|is)",
        r"\bdataset\b", r"\boverfit", r"\bregular(?:iz|is)ation\b", r"\bdropout\b",
        r"\bhyperparameter\b", r"\bepoch\b", r"\bloss (?:function|curve)\b",
        r"\bprompt\b", r"\bdiffusion\b", r"\bconvolution\b", r"\battention\b",
        r"\bquant(?:iz|is)ation\b", r"\bGPU\b", r"\bbenchmark\b", r"\bagent(?:ic)?\b",
    )
    persona = (
        "SPECIALIST MODE — AI / Machine Learning. You are ORION's ML engineer and "
        "researcher: fluent in classical ML, deep learning, transformers and LLMs, "
        "training and evaluation, and the practical engineering of systems like "
        "ORION himself. Reason from first principles — what the model is actually "
        "learning, what the loss rewards, where the data or objective is the real "
        "problem. Give concrete, runnable specifics (architecture, hyperparameters, "
        "the metric that matters and its failure modes) over hand-waving, and always "
        "separate what is empirically established from what is folklore. Flag when a "
        "simpler baseline would beat a fancy model, call out data leakage, eval "
        "contamination and overfitting, and be honest about cost, latency and "
        "hardware limits. Never invent benchmark numbers or paper results."
    )


class CyberSecurityAgent(BaseAgent):
    name = "security"
    title = "Cybersecurity Agent"
    expertise = "defensive security, threat modelling, secure design, recognition of attacks"
    primary = (
        r"\bcyber\s?security\b", r"\bsecurity\b", r"\bvulnerabilit", r"\bexploit\b",
        r"\bmalware\b", r"\bphishing\b", r"\bthreat model", r"\bpen ?test",
        r"\bencryption\b", r"\bfirewall\b", r"\bzero[- ]?day\b", r"\bCVE\b",
        r"\bkeylogger\b", r"\brootkit\b", r"\bransomware\b", r"\bSQL injection\b",
        r"\bXSS\b", r"\bprivilege escalation\b",
    )
    keywords = (
        r"\bhack", r"\battack\b", r"\bpayload\b", r"\bransomware\b", r"\brootkit\b",
        r"\bkeylogger\b", r"\bSQL injection\b", r"\bXSS\b", r"\bauthenticat",
        r"\bhardening\b", r"\bleast privilege\b", r"\bpassword\b", r"\bMFA\b",
        r"\bnmap\b", r"\bport scan\b", r"\bprivilege escalation\b", r"\bsandbox\b",
    )
    persona = (
        "SPECIALIST MODE — Cybersecurity (defensive & educational). You are ORION's "
        "security analyst. Teach recognition FIRST: for any technique, lead with the "
        "key identifiers that let the user spot it in code or behaviour, then how it "
        "works and, above all, how to DEFEND against it. Stay strictly educational "
        "and lawful — you help understand, harden and detect, never weaponise. "
        "Malware-class topics are concept-and-recognition only (no working attack "
        "code); hands-on practice is pointed at lawful ranges the user owns or is "
        "authorised on. Think like a threat modeller: assets, trust boundaries, "
        "attacker capabilities, and the cheapest control that removes the risk. "
        "Prefer defence-in-depth and least privilege; explain the trade-offs. This "
        "matches ORION's built-in cyber curriculum and its default-deny posture."
    )


# ──────────────────────────────────────────────────────────────────────────────
# DESKTOP AGENT  — functional host control
# ──────────────────────────────────────────────────────────────────────────────

class DesktopAgent:
    """
    ORION's hands on the host machine.

    Owns application launching (Start Menu index + allowlist), window
    management, media keys, browser opening and development-tool shortcuts.
    The dispatcher delegates its desktop tools here, so this logic exists in
    exactly one place.
    """

    SAFE_APPS = {
        "notepad":    "notepad.exe",
        "calculator": "calc.exe",
        "calc":       "calc.exe",
        "paint":      "mspaint.exe",
        "cmd":        "cmd.exe",
        "terminal":   "wt.exe",
        "powershell": "powershell.exe",
        "edge":       "msedge.exe",
        "microsoft edge": "msedge.exe",
        "chrome":     "chrome.exe",
        "google chrome": "chrome.exe",
        "firefox":    "firefox.exe",
        "explorer":   "explorer.exe",
        "file explorer": "explorer.exe",
        "outlook":    "outlook.exe",
        "microsoft outlook": "outlook.exe",
        "word":       "winword.exe",
        "microsoft word": "winword.exe",
        "excel":      "excel.exe",
        "microsoft excel": "excel.exe",
        "powerpoint": "powerpnt.exe",
        "microsoft powerpoint": "powerpnt.exe",
        "vscode":     "code",
        "vs code":    "code",
        "visual studio code": "code",
    }

    # Well-known web destinations ORION may open by name.
    WEB_APPS = {
        "notion":  "https://www.notion.so",
        "gmail":   "https://mail.google.com",
        "youtube": "https://www.youtube.com",
        "github":  "https://github.com",
        "outlook web": "https://outlook.office.com/mail/",
    }

    _MEDIA_KEYS = {
        "play_pause": 0xB3, "play": 0xB3, "pause": 0xB3, "toggle": 0xB3,
        "next": 0xB0, "next_track": 0xB0,
        "previous": 0xB1, "previous_track": 0xB1, "prev": 0xB1,
        "stop": 0xB2,
        "volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD,
    }

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        # Installed-application index discovered from the Start Menu — built on
        # a worker thread so startup never blocks.
        self._app_index: dict[str, str] = {}
        Thread(target=self._build_app_index, name="orion-app-indexer", daemon=True).start()

    # ── application launching ─────────────────────────────────────────────────

    def open_app(self, app_name: str) -> ToolResult:
        app_name = SecuritySanitiser.guard_text(str(app_name or ""), "open_app.app_name")
        if not app_name:
            return ToolResult("No application name supplied.", ok=False)
        if self._is_url(app_name):
            open_url(app_name)
            return ToolResult(f"Opened URL: {app_name}")
        key = app_name.strip().lower()
        web_target = self.WEB_APPS.get(key)
        if web_target:
            open_url(web_target)
            return ToolResult(f"Opened {app_name} in the browser: {web_target}")
        executable = self.SAFE_APPS.get(key, "")
        resolved = shutil.which(executable) if executable else None
        if resolved:
            subprocess.Popen([resolved], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
            return ToolResult(f"Opened application: {app_name}.")

        # ── common sense ─────────────────────────────────────────────────────
        # "Word" is WINWORD.EXE; "please open up microsoft word for me" is also
        # Word. The resolver normalises the phrasing and searches the App Paths
        # registry, the Start Menu and PATH rather than requiring the exact
        # binary name (see app_resolver.py).
        from .app_resolver import resolve_app

        match = resolve_app(app_name)
        if match.ok and match.confident:
            target = match.candidate.target
            try:
                self._launch(target)
            except Exception as exc:
                return ToolResult(
                    f"I found {match.candidate.name} for '{app_name}' but could "
                    f"not start it: {exc}", ok=False)
            # Say WHAT was opened, not just that something was. If the guess is
            # wrong the user can see it immediately instead of wondering why
            # the wrong window appeared.
            said = match.candidate.name
            note = "" if said.lower() == key else f" (matched '{app_name}' to {said})"
            self.bus.log.emit(f"APP: opened {said} <- '{app_name}' "
                              f"[{match.candidate.source}, {match.score:.2f}]")
            return ToolResult(f"Opened {said}{note}.")

        # Not an application. It may be a DOCUMENT: "open my CV", "open the
        # essay" arrive here as often as through find_files, and used to end
        # in "I could not find an application matching 'my CV'".
        from . import file_search

        if os.path.exists(os.path.expandvars(os.path.expanduser(app_name))):
            target = os.path.expandvars(os.path.expanduser(app_name))
            problem = file_search.open_path(target)
            return (ToolResult(f"Opened {target}.") if not problem
                    else ToolResult(f"Could not open {target}: {problem}", ok=False))
        document = file_search.find_document(app_name)
        if document:
            problem = file_search.open_path(document)
            if not problem:
                self.bus.log.emit(f"APP: opened document {document} <- '{app_name}'")
                return ToolResult(f"Opened the document {document}.")

        # Nothing confident. Try the literal string — it may be a URI scheme
        # the shell understands — then be honest and offer real names.
        try:
            os.startfile(app_name)  # type: ignore[attr-defined]
            return ToolResult(f"Open request issued: {app_name}.")
        except Exception as exc:
            hints = match.alternatives or self._suggest_apps(key)
            hint_text = (f" The closest things I have are: {', '.join(hints)}."
                         if hints else "")
            return ToolResult(
                f"I could not find an application matching '{app_name}'.{hint_text}",
                ok=False)

    @staticmethod
    def _launch(target: str) -> None:
        """Start *target*, whichever kind of thing it is.

        A shortcut, a URI scheme (``ms-settings:``) and a management console
        (``devmgmt.msc``) all need the shell; a plain executable is better
        started directly so no console window flashes up.
        """
        lowered = str(target).lower()
        if lowered.endswith(".exe") and os.path.sep in str(target):
            subprocess.Popen([target], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, shell=False)
            return
        os.startfile(target)  # type: ignore[attr-defined]

    def _suggest_apps(self, key: str) -> list[str]:
        from .app_resolver import CATALOGUE
        try:
            return CATALOGUE.suggest(key)
        except Exception:
            return []

    def _build_app_index(self) -> None:
        """Discover installed applications from Start Menu shortcuts (Windows)."""
        index: dict[str, str] = {}
        roots = [
            Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(os.environ.get("APPDATA", ""))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        ]
        for root in roots:
            if not root.is_dir():
                continue
            try:
                for shortcut in root.rglob("*.lnk"):
                    index.setdefault(shortcut.stem.lower(), str(shortcut))
            except Exception:
                continue
        self._app_index = index
        if index:
            self.bus.log.emit(f"SYS: application index built - {len(index)} installed apps discovered.")

    def close_app(self, label: str) -> ToolResult:
        label = str(label or "").strip()
        if not label:
            return ToolResult("No application name supplied.", ok=False)
        SecuritySanitiser.guard_text(label, "close_app.name")
        tokens = self._process_match_tokens(label)
        terminated: list[str] = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = fold_title(proc.info.get("name") or "")
                if name and any(token in name for token in tokens):
                    self._terminate_process(proc, terminated)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if terminated:
            return ToolResult("Terminated: " + ", ".join(terminated))
        # No process matched the label — the app may live under a different
        # executable name; fall back to a polite close of matching windows.
        window_result = self.window_control("close", label)
        if window_result.ok:
            return window_result
        return ToolResult(f"No running process or window matches '{label}'.", ok=False)

    def _process_match_tokens(self, label: str) -> set[str]:
        """Match candidates for a spoken app name: the folded label, the label
        without its vendor prefix, and the executable it maps to ("Microsoft
        Edge" must find msedge.exe)."""
        folded = fold_title(label)
        candidates = {folded}
        for prefix in ("microsoft ", "google ", "windows "):
            if folded.startswith(prefix):
                candidates.add(folded[len(prefix):].strip())
        tokens = set(candidates)
        for candidate in candidates:
            executable = self.SAFE_APPS.get(candidate)
            if executable:
                tokens.add(executable.lower())
                tokens.add(Path(executable).stem.lower())
        return {token for token in tokens if token}

    def _terminate_process(self, proc: psutil.Process, terminated: list[str]) -> None:
        if proc.pid == os.getpid():
            raise SecurityViolation(
                "blocked unsafe process operation: refusing to terminate O.R.I.O.N."
            )
        name = proc.name()
        SecuritySanitiser.guard_text(name, "process.name")
        proc.terminate()
        terminated.append(f"{name}:{proc.pid}")

    # ── window management ─────────────────────────────────────────────────────

    def window_control(self, action: str, title: str = "") -> ToolResult:
        if sys.platform != "win32":
            return ToolResult("Window control is only available on Windows.", ok=False)
        action = str(action or "list").lower().strip()
        title = fold_title(title)
        import ctypes
        import ctypes.wintypes as wintypes
        user32 = ctypes.windll.user32
        windows: list[tuple[int, str]] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _collect(hwnd: Any, lparam: Any) -> bool:
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    windows.append((hwnd, buffer.value))
            return True

        user32.EnumWindows(_collect, 0)
        if action in {"list", "inventory"}:
            return ToolResult(
                "\n".join(name for _, name in windows[:60]) or "No visible windows."
            )
        if not title:
            return ToolResult("No window title supplied.", ok=False)
        # Fold both sides: real titles carry invisible Unicode (Edge's
        # "Microsoft​ Edge" hides a zero-width space) that breaks naive matching.
        matches = [(h, t) for h, t in windows if title in fold_title(t)]
        if not matches:
            return ToolResult(f"No visible window matches '{title}'.", ok=False)
        if action == "close":
            for hwnd, _ in matches:
                user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE — polite close
            closed = "; ".join(t for _, t in matches[:10])
            return ToolResult(
                f"Close request sent to {len(matches)} window(s): {closed}"
            )
        hwnd, matched = matches[0]
        if action in {"focus", "activate", "switch", "switch_to"}:
            user32.ShowWindow(hwnd, 9)   # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return ToolResult(f"Focused window: {matched}")
        if action in {"minimise", "minimize"}:
            user32.ShowWindow(hwnd, 6)   # SW_MINIMIZE
            return ToolResult(f"Minimised window: {matched}")
        if action in {"maximise", "maximize"}:
            user32.ShowWindow(hwnd, 3)   # SW_MAXIMIZE
            return ToolResult(f"Maximised window: {matched}")
        return ToolResult(f"Unsupported window action: {action}", ok=False)

    # ── media keys ────────────────────────────────────────────────────────────

    def media_control(self, action: str, steps: int = 2) -> ToolResult:
        if sys.platform != "win32":
            return ToolResult("Media control is only available on Windows.", ok=False)
        action = str(action or "play_pause").lower().strip()
        key = self._MEDIA_KEYS.get(action)
        if key is None:
            return ToolResult(
                f"Unsupported media action: {action}. "
                f"Supported: {', '.join(sorted(set(self._MEDIA_KEYS)))}.",
                ok=False,
            )
        import ctypes
        repeats = 1
        if action in {"volume_up", "volume_down"}:
            repeats = max(1, min(10, int(steps or 2)))
        for _ in range(repeats):
            ctypes.windll.user32.keybd_event(key, 0, 0, 0)
            ctypes.windll.user32.keybd_event(key, 0, 2, 0)  # KEYEVENTF_KEYUP
        return ToolResult(f"Media command issued: {action}.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_url(self, value: str) -> bool:
        parsed = urlparse(value if "://" in value else "")
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class AgentManager:
    """
    Registry and dynamic router for the specialist workforce.

    ``dispatch(request, agent="auto")`` scores the request against every
    registered agent's keyword profile; ties resolve in registration order.
    Requests with no specialist signal return None from ``route`` and the
    caller answers in the general ORION persona instead.
    """

    # Below this weighted score, the signal is too weak to prefer a specialist
    # over ORION's general capacity (a lone supporting keyword, e.g. "email").
    ROUTE_THRESHOLD = 3

    def __init__(self, router: ProviderRouter, bus: OrionBus) -> None:
        self.bus = bus
        self.router = router
        self._agents: dict[str, BaseAgent] = {}
        # Late-attached once the reasoning engine exists (app.py) — gives
        # ordinary agent_dispatch calls the same read-only instrument tray
        # Track C's `reason` tool already built, instead of duplicating it.
        self._toolbelt_factory: Callable[[], "AgentToolbelt"] | None = None
        # Per-specialist call count + last activity, surfaced by describe() so
        # the Agent Monitor panel shows something beyond a static roster.
        self._activity: dict[str, dict[str, Any]] = {}
        for agent_cls in (
            DigitalMarketingAgent,
            CodingAgent,
            ResearchAnalysisAgent,
            NeuroscienceAgent,
            AIMLAgent,
            CyberSecurityAgent,
        ):
            self.register(agent_cls(router, bus))
        # Mark XXI, Track E3: additional specialists declared in
        # config/agents/*.json — no code change needed to add one. Loaded
        # last so a manifest can never shadow a built-in (register_declarative
        # refuses any name already taken).
        self.load_declarative_agents()

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent

    def register_declarative(self, manifest: dict[str, Any]) -> bool:
        """Register one manifest-built agent (Track E3). Refuses to shadow
        an already-registered agent (built-in or otherwise) — a
        misconfigured or duplicate manifest can never silently replace a
        real specialist. Returns whether it was registered."""
        from .agent_registry import build_agent
        name = str(manifest.get("name") or "").strip()
        if not name or name in self._agents:
            return False
        self.register(build_agent(manifest, self.router, self.bus))
        return True

    def load_declarative_agents(self) -> int:
        """Load every valid config/agents/*.json manifest and register it.
        Returns how many were newly registered."""
        from . import agent_registry
        registered = 0
        for manifest in agent_registry.load_all(self.bus):
            if self.register_declarative(manifest):
                registered += 1
            else:
                try:
                    self.bus.log.emit(
                        f"AGENTS: skipped manifest '{manifest.get('name')}' — "
                        "that name is already registered.")
                except Exception:
                    pass
        return registered

    def attach_toolbelt_factory(self, factory: Callable[[], "AgentToolbelt"]) -> None:
        self._toolbelt_factory = factory

    def agent_names(self) -> list[str]:
        return list(self._agents)

    def available_instruments(self) -> list[str]:
        """The instrument names a specialist could reach for right now — pure
        introspection (constructing a toolbelt costs nothing until .run() is
        called), so a GUI can show "what can this agent look up" without
        spending any of the investigation budget. Empty when no factory is
        attached yet (matches investigate()'s own degrade-to-no-toolbelt
        behaviour)."""
        if self._toolbelt_factory is None:
            return []
        try:
            return self._toolbelt_factory().names()
        except Exception:
            return []

    def describe(self) -> list[dict[str, Any]]:
        """Dashboard feed: one row per registered specialist."""
        rows = []
        for agent in self._agents.values():
            activity = self._activity.get(agent.name, {})
            rows.append({
                "name": agent.name, "title": agent.title, "focus": agent.expertise,
                "calls": activity.get("calls", 0),
                "last_active": activity.get("last_active", ""),
            })
        return rows

    def route_with_confidence(self, request: str) -> tuple[BaseAgent | None, int]:
        """Best specialist and its weighted score (0 when nothing scores)."""
        request = str(request or "")
        best: BaseAgent | None = None
        best_score = 0
        for agent in self._agents.values():        # ties resolve in reg. order
            score = agent.score(request)
            if score > best_score:
                best, best_score = agent, score
        return best, best_score

    def route(self, request: str) -> BaseAgent | None:
        """Pick the best specialist, or None when the signal is below threshold."""
        agent, score = self.route_with_confidence(request)
        return agent if score >= self.ROUTE_THRESHOLD else None

    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(str(name or "").strip().lower())

    ANCHOR_AGENT = "research"

    def panel(self, request: str, size: int = 3) -> list[BaseAgent]:
        """The *size* most relevant specialists, for a deliberation panel.

        The analyst is always seated.  A panel of one domain expert agreeing
        with itself is not deliberation — the generalist is what forces the
        question to be examined from outside the domain that claimed it, and it
        is also the only sensible panel for a question no specialist matches.
        Below-threshold specialists are never seated: an irrelevant persona
        contributes noise the chair then has to discount.
        """
        size = max(1, int(size))
        ranked = sorted(
            ((agent, agent.score(str(request or ""))) for agent in self._agents.values()),
            key=lambda pair: -pair[1],
        )
        chosen: list[BaseAgent] = [
            agent for agent, score in ranked if score >= self.ROUTE_THRESHOLD
        ][:size]
        anchor = self._agents.get(self.ANCHOR_AGENT)
        if anchor is not None and anchor not in chosen and (size > 1 or not chosen):
            # Displace the weakest seated specialist rather than growing the panel.
            chosen = chosen[: size - 1] + [anchor] if len(chosen) >= size else chosen + [anchor]
        return chosen[:size]

    async def dispatch(self, request: str, agent_name: str = "auto",
                       context: str = "") -> ToolResult:
        """Route *request* to a named or auto-selected specialist."""
        request = str(request or "").strip()
        if not request:
            return ToolResult("No request supplied for agent dispatch.", ok=False)
        agent: BaseAgent | None
        agent_name = str(agent_name or "auto").strip().lower()
        if agent_name in {"", "auto", "any"}:
            agent = self.route(request)
            if agent is None:
                return ToolResult(
                    "No specialist agent matches this request; answer it in the "
                    "general ORION capacity.",
                )
        else:
            agent = self._agents.get(agent_name)
            if agent is None:
                return ToolResult(
                    f"Unknown agent '{agent_name}'. Registered specialists: "
                    + ", ".join(self.agent_names()) + ".",
                    ok=False,
                )
        self.bus.log.emit(f"AGENT: routing request to the {agent.title}.")
        # A fresh toolbelt per call (the factory returns a new instance with a
        # reset budget each time) — investigate() degrades to exactly today's
        # handle()-shaped single provider call when the toolbelt is None or
        # the specialist never asks for a lookup, so this costs nothing when
        # unused and gives real capability when it is.
        toolbelt = self._toolbelt_factory() if self._toolbelt_factory is not None else None
        finding = await agent.investigate(request, context=context, toolbelt=toolbelt)
        self._activity[agent.name] = {
            "calls": self._activity.get(agent.name, {}).get("calls", 0) + 1,
            "last_request": first_line(request, 90),
            "last_active": utc_stamp(),
        }
        # Mark XX design-spec §6: every reading the specialist took is
        # already recorded as a ToolInvocation on finding.tools — it used
        # to be computed and then discarded right here, leaving only prose.
        # Now it rides along as structured evidence a GUI surface can choose
        # to render (e.g. a "Sources consulted (N)" strip), at zero extra
        # cost since the investigation already happened.
        evidence = [
            {"tool": t.tool, "ok": t.ok, "summary": t.as_line(width=200)}
            for t in finding.tools
        ] or None
        return ToolResult(
            f"[{finding.title}]\n{finding.answer}", ok=finding.ok, evidence=evidence)
