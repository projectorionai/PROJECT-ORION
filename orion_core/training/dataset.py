"""
Building the local-brain training set.

``build(BuildConfig(...))`` collects training turns from eight places, turns
every one into the chat-with-tools shape ``LocalAgent`` really runs, and then
cleans the lot the same way:

    collect  ->  redact  ->  validate  ->  de-duplicate  ->  remove evaluation
    overlap  ->  cap per tool  ->  choose the tools each row offers  ->  split
    ->  write + manifest

Sources (``--sources``)
-----------------------
real         the owner's own turns: an utterance joined, in time, to the tool
             ORION's live model chose (resolver shadow log), the arguments and
             result where a job recorded them, and ORION's reply. Where the
             arguments were never recorded they are rebuilt from the owner's
             words by ``argsynth`` and only kept if the real schema accepts
             them.
corrections  the owner corrected ORION ("no, I meant ..."): the corrected
             tool call becomes the preferred example, in context, and a
             preference pair against the original.
reflex       phrases the reflex ledger has seen work (and those it has seen
             fail, which no row from any source may answer - by calling the
             tool that failed, or in words).
routing      the development half of the routing cases (394 requests, every
             tool), as single-tool turns with plausible arguments, in a few
             phrasings. The held-out half is never trained on.
phrasebank   hand-written everyday requests with exact calls, some with the
             tool result and ORION's spoken answer; follow-ups that depend on
             the previous turn.
generated    read-only turns with varied, computed results and the spoken
             answer to each (weather, sums, memory use, searches, files...),
             so the "answer from the result" step stays sharp.
safety       high-risk requests proposed as the exact call (the agent then
             asks), requests that are declined in words, tool results that try
             to give orders, requests with a gap that must be asked about.
multistep    two-step flows, written out and composed, including ones where a
             step fails and the reply must say so.
chat         conversation that needs no tool.
planner      a small replay of the planner's own task (a request and a tool
             list in, one JSON plan out, or a refusal), validated with the
             planner's own ``check_plan``. The same adapter will be measured
             on the planner evaluation (``tools/evaluate_planner.py``) before
             it can be promoted, so it must not forget that format.

Nothing here talks to a model or the network, or opens a store for writing.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from . import BUILDER_VERSION, DATASET_FORMAT
from . import argsynth, bank, generators, safety, sources
from .evalguard import EvalGuard, collect_eval_strings, routing_split_cases
from .files import load_rows, verify_dataset
from .paths import require_private_location, training_root
from .redact import PLACEHOLDERS, Redactor, sensitive_topic
from .toolcalls import (CHARS_PER_TOKEN, ToolBook, agent_module, assistant_call, assistant_text, calls_in, canonical,
                        estimate_tokens, failure_looks_like, sha256_text, system_message,
                        system_prompt_hash, tool_result, tool_schemas, user_message)

ALL_SOURCES: tuple[str, ...] = ("real", "corrections", "reflex", "routing", "phrasebank", "generated",
                                "safety", "multistep", "chat", "planner")
RISK_ORDER = {"read": 0, "act": 1, "high": 2}
#: Lower number = kept first when a tool has more rows than its cap.
PRIORITY = {"real": 1, "correction": 1, "reflex": 2, "phrasebank": 2, "safety": 2, "multistep": 2,
            "chat": 3, "routing": 3, "tool_routing": 3, "generated": 3, "planner": 3, "routing_aug": 4,
            "phrasebank_aug": 4, "safety_aug": 4}


@dataclass
class BuildConfig:
    """Everything a build depends on. Defaults are the recommended ones."""

    config_dir: Path | None = None               # where ORION's stores live (read-only)
    out_dir: Path | None = None                  # training root; datasets go in a subfolder
    project_dir: Path | None = None              # for tests/ and tools/ evaluation files
    sources: tuple[str, ...] = ALL_SOURCES
    routing_split: str = "dev"                   # dev | all
    val_fraction: float = 0.08
    max_per_tool: int = 24
    max_decline: int = 80
    max_chat: int = 40
    max_clarify: int = 12
    max_multistep: int = 260                     # see multistep_combos
    max_planner: int = 60
    augment: int = 2                             # extra phrasings per routing phrase
    multistep_combos: int = 220                  # 40 gave 53 flows in 890 rows: the first trained
                                                 # model (29 Oct) stopped after one tool every time
    max_tools: int = 8
    tools_char_budget: int = 4200
    max_tokens_estimate: int = 1900
    generic_args: bool = True
    seed: int = 20261025
    redact_terms: tuple[str, ...] = ()
    dry_run: bool = False
    allow_synced_output: bool = False
    # Test seams: left as None they use ORION's real data.
    declarations: Sequence[dict[str, Any]] | None = None
    scorer: Callable[[str, Sequence[dict[str, Any]]], dict[str, float]] | None = None
    routing_cases: Sequence[tuple[str, ...]] | None = None
    tool_routing_cases: Sequence[tuple[str, ...]] | None = None
    eval_strings: Sequence[str] | None = None
    redactor: Redactor | None = None
    now: float | None = None


@dataclass
class Draft:
    """A training turn before cleaning: messages without the system prompt."""

    source: str
    kind: str
    messages: list[dict[str, Any]]
    tools_called: list[str]
    text: str                       # the request the shortlist is ranked on
    priority: int = 5
    when: datetime | None = None
    private: bool = False           # came from the owner's own conversations
    args_how: str = "given"
    prefs: dict[str, Any] | None = None   # an optional rejected alternative
    system: str | None = None       # a system message of its own (the planner's); None = the agent's
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildResult:
    manifest: dict[str, Any]
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    preferences: list[dict[str, Any]]
    path: Path | None = None


class Context:
    def __init__(self, cfg: BuildConfig) -> None:
        self.cfg = cfg
        self.book = ToolBook(cfg.declarations)
        self.la = agent_module()
        self.dropped: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()
        self.produced: Counter[str] = Counter()
        self.root = Path(cfg.project_dir) if cfg.project_dir is not None else _project_dir()
        now = cfg.now if cfg.now is not None else time.time()
        self.now = now
        self.redactor = cfg.redactor if cfg.redactor is not None else Redactor.from_environment(
            cfg.redact_terms)
        self._rank_cache: dict[str, list[str]] = {}

    def rng(self, *parts: Any) -> random.Random:
        digest = sha256_text("|".join(str(p) for p in (self.cfg.seed, *parts)))
        return random.Random(int(digest[:12], 16))

    def when_for(self, text: str) -> datetime:
        """A plausible local time for a synthetic row (the prompt carries the date)."""
        rng = self.rng("when", text)
        base = datetime(2026, 8, 1, 7, 0)
        return base + timedelta(days=rng.randint(0, 84), hours=rng.randint(0, 15),
                                minutes=rng.randint(0, 59))

    def ranked(self, text: str) -> list[str]:
        """Tool names best-first for *text*, the way ``LocalAgent.shortlist`` ranks them."""
        if text in self._rank_cache:
            return self._rank_cache[text]
        candidates = [self.book.declarations[n] for n in self.book.names()]
        scorer = self.cfg.scorer or self.la._default_scorer
        try:
            scores = dict(scorer(text, candidates) or {})
        except Exception:                                 # noqa: BLE001 - same fallback as the agent
            from ..tool_resolver import lexical_scores
            scores = dict(lexical_scores(text, candidates))
        names = sorted((n for n in self.book.names() if float(scores.get(n, 0.0)) > 0.0),
                       key=lambda n: (-float(scores.get(n, 0.0)), n))
        self._rank_cache[text] = names
        return names


def _project_dir() -> Path:
    from ..constants import BASE_DIR
    return Path(BASE_DIR)


# ── helpers: turns and phrasings ────────────────────────────────────────────────

def call_messages(text: str, tool: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [user_message(text), assistant_call(tool, args)]


def turn_messages(text: str, tool: str, args: dict[str, Any], result: str, ok: bool,
                  reply: str) -> list[dict[str, Any]]:
    return [user_message(text), assistant_call(tool, args), tool_result(tool, ok, result),
            assistant_text(reply)]


_PREFIXES = ("please ", "orion, ", "hey orion, ", "could you ", "can you ", "quick one, ", "would you ")
_SUFFIXES = (" please", " for me", " thanks", " when you can")


def variants(text: str, rng: random.Random, count: int) -> list[str]:
    """Other ways of saying *text*: politeness, an address, speech-style casing."""
    base = " ".join(str(text or "").split())
    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 6:
        attempts += 1
        choice = rng.choice(("prefix", "suffix", "lower", "both"))
        v = base
        if choice in {"prefix", "both"} and not re.match(r"(?i)^(?:please|orion|hey|could|can|would)\b", v):
            v = rng.choice(_PREFIXES) + v[:1].lower() + v[1:]
        if choice in {"suffix", "both"} and not re.search(r"(?i)\b(?:please|thanks|for me)\W*$", v):
            v = v.rstrip(" .?!") + rng.choice(_SUFFIXES)
        if choice == "lower":
            v = v.lower().rstrip(".!?")
        if v != base and v not in out:
            out.append(v)
    return out


def _is_multi_part(text: str) -> bool:
    return bool(agent_module().multi_part(text))


def _risk_of(book: ToolBook, calls: Sequence[tuple[str, dict[str, Any]]]) -> str:
    worst = "read"
    for name, args in calls:
        risk, _why = book.risk(name, args)
        if RISK_ORDER.get(risk, 2) > RISK_ORDER[worst]:
            worst = risk
    return worst


# ── collectors ──────────────────────────────────────────────────────────────────

def _routing_cases(ctx: Context, which: str) -> tuple[list[tuple[str, ...]], str]:
    if ctx.cfg.routing_cases is not None:
        return list(ctx.cfg.routing_cases), "injected"
    path = ctx.root / "tests" / "data" / "intent_routing_cases.py"
    if not path.is_file():
        ctx.skipped["routing_set_missing"] += 1
        return [], "missing"
    return routing_split_cases(path, which), "file"


def _tool_routing_cases(ctx: Context, which: str) -> list[tuple[str, ...]]:
    if ctx.cfg.tool_routing_cases is not None:
        return list(ctx.cfg.tool_routing_cases)
    path = ctx.root / "tests" / "data" / "tool_routing_cases.py"
    if not path.is_file():
        return []
    return routing_split_cases(path, which)


def _synthetic_call(ctx: Context, text: str, candidates: Sequence[str]) -> tuple[str, dict[str, Any], str] | None:
    """(tool, args, how) for the first acceptable tool that can be called from *text*."""
    clause = argsynth.first_clause(text)
    offered_any = False
    for tool in candidates:
        if not ctx.book.has(tool):
            continue
        offered_any = True
        args, how = argsynth.synthesise(ctx.book, tool, clause, generic=ctx.cfg.generic_args)
        if args is not None:
            return tool, args, how
    ctx.skipped["tool_not_offered" if not offered_any else "no_arguments_found"] += 1
    return None


def collect_routing(ctx: Context) -> list[Draft]:
    drafts: list[Draft] = []
    which = "all" if ctx.cfg.routing_split == "all" else "dev"
    cases, _origin = _routing_cases(ctx, which)
    sets = [("routing", cases)]
    sets.append(("tool_routing", [(c[0], c[1]) for c in _tool_routing_cases(ctx, which)]))
    for source, rows in sets:
        for case in rows:
            text = str(case[0])
            tools = [str(case[1]), *[str(t) for t in (case[2] if len(case) > 2 else ())]]
            found = _synthetic_call(ctx, text, tools)
            if found is None:
                continue
            tool, args, how = found
            drafts.append(Draft(source, "call", call_messages(text, tool, args), [tool], text,
                                PRIORITY[source], ctx.when_for(text), args_how=how,
                                meta={"routing": which}))
            rng = ctx.rng(source, text)
            for variant in variants(text, rng, ctx.cfg.augment):
                drafts.append(Draft("routing_aug", "call", call_messages(variant, tool, args), [tool],
                                    variant, PRIORITY["routing_aug"], ctx.when_for(variant),
                                    args_how=how, meta={"routing": which, "of": source}))
    return drafts


def _phrase_draft(ctx: Context, phrase: bank.Phrase, source: str = "phrasebank") -> Draft:
    risk = ctx.book.risk(phrase.tool, phrase.args)[0] if ctx.book.has(phrase.tool) else "high"
    full = bool(phrase.result and phrase.reply) and (
        risk == "read" or (risk == "act" and _is_multi_part(phrase.text)))
    msgs = (turn_messages(phrase.text, phrase.tool, phrase.args, phrase.result, True, phrase.reply)
            if full else call_messages(phrase.text, phrase.tool, phrase.args))
    return Draft(source, "turn" if full else "call", msgs, [phrase.tool], phrase.text,
                 PRIORITY[source], ctx.when_for(phrase.text))


def collect_phrasebank(ctx: Context) -> list[Draft]:
    drafts: list[Draft] = []
    for phrase in bank.PHRASEBANK:
        drafts.append(_phrase_draft(ctx, phrase))
        risk = ctx.book.risk(phrase.tool, phrase.args)[0] if ctx.book.has(phrase.tool) else "high"
        if phrase.fail_result and phrase.fail_reply and risk == "read":
            drafts.append(Draft("phrasebank", "turn",
                                turn_messages(phrase.text, phrase.tool, phrase.args,
                                              phrase.fail_result, False, phrase.fail_reply),
                                [phrase.tool], phrase.text, PRIORITY["phrasebank"],
                                ctx.when_for(phrase.text + "f"), meta={"outcome": "failure"}))
        rng = ctx.rng("phrasebank", phrase.text)
        for variant in variants(phrase.text, rng, max(0, ctx.cfg.augment - 1)):
            varied = bank.Phrase(variant, phrase.tool, phrase.args)
            drafts.append(_phrase_draft(ctx, varied, "phrasebank_aug"))
    for follow in bank.FOLLOWUPS:
        msgs = [user_message(follow.prev_user), assistant_text(follow.prev_reply),
                user_message(follow.text), assistant_call(follow.tool, follow.args)]
        drafts.append(Draft("phrasebank", "followup", msgs, [follow.tool], follow.text,
                            PRIORITY["phrasebank"], ctx.when_for(follow.text)))
    return drafts


def collect_generated(ctx: Context) -> list[Draft]:
    """Read-only turns with a computed result and a spoken answer (see generators.py)."""
    drafts = []
    for item in generators.generate(ctx.rng("generated")):
        drafts.append(Draft("generated", "turn",
                            turn_messages(item.text, item.tool, item.args, item.result, True, item.reply),
                            [item.tool], item.text, PRIORITY["generated"], ctx.when_for(item.text)))
    return drafts


def _plan_goal_drafts(ctx: Context) -> list[tuple[str, list[tuple[str, dict[str, Any]]]]]:
    """(goal, [(tool, args)...]) for read-only requests we already have answers to."""
    from .. import neural_planner as np_

    def observes(tool: str, args: dict[str, Any]) -> bool:
        return (ctx.book.has(tool) and np_.plannable(tool, "read_only")
                and ctx.book.risk(tool, args)[0] == "read")

    out: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = []
    for p in bank.PHRASEBANK:
        if observes(p.tool, p.args):
            out.append((p.text, [(p.tool, p.args)]))
    for item in generators.generate(ctx.rng("generated")):
        if observes(item.tool, item.args):
            out.append((item.text, [(item.tool, item.args)]))
    for flow in bank.FLOWS:
        if all(s.ok and observes(s.tool, s.args) for s in flow.steps):
            out.append((flow.text, [(s.tool, s.args) for s in flow.steps]))
    return out


def collect_planner(ctx: Context) -> list[Draft]:
    """Planner-format replay: plans for read-only requests and refusals for the rest."""
    from .. import neural_planner as np_
    drafts: list[Draft] = []
    max_steps, prompt_chars = 8, 2600
    allowed = {n for n in ctx.book.names() if np_.plannable(n, "read_only")}

    def offer(goal: str, targets: list[str], scope: str) -> list[dict[str, Any]]:
        pool = [n for n in ctx.ranked(goal) if scope == "effects" and np_.plannable(n, scope)
                or n in allowed]
        chosen = [t for t in targets if ctx.book.has(t)]
        used = sum(len(np_._describe_tool(ctx.book.declarations[n])) for n in chosen)
        for name in pool:
            if len(chosen) >= 5:
                break
            size = len(np_._describe_tool(ctx.book.declarations[name]))
            if name not in chosen and used + size <= prompt_chars:
                chosen.append(name)
                used += size
        order = {n: i for i, n in enumerate(ctx.ranked(goal))}
        chosen.sort(key=lambda n: (order.get(n, len(order)), n))
        return [ctx.book.declarations[n] for n in chosen]

    def plan_draft(goal: str, scope: str, plan: dict[str, Any], kind: str, targets: list[str]) -> None:
        offered = offer(goal, targets, scope)
        system, user = np_.build_prompt(goal, offered, [], scope=scope, max_steps=max_steps)
        try:
            np_.check_plan(plan, offered=offered, scope=scope, max_steps=max_steps)
        except np_.PlanRejected as rejected:
            if not (kind == "plan_refusal" and rejected.stage == "refused"):
                ctx.skipped["planner_plan_rejected"] += 1
                return
        reply = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        drafts.append(Draft("planner", kind, [user_message(user), assistant_text(reply)], [], goal,
                            PRIORITY["planner"], ctx.when_for(goal), system=system,
                            meta={"scope": scope}))

    for goal, steps in _plan_goal_drafts(ctx):
        plan = {"summary": goal[:160], "assumptions": [],
                "steps": [{"id": f"s{i}", "tool": tool, "args": args, "depends_on": [],
                           "success": "The result answers the request."}
                          for i, (tool, args) in enumerate(steps, 1)],
                "success_criteria": ["The answer to the request is given from the tool results."]}
        plan_draft(goal, "read_only", plan, "plan", [t for t, _ in steps])
    refusal_text = "That would change something or needs your approval, so it is not planned automatically."
    for text, reply in safety.DECLINES:
        plan_draft(text, "effects", {"summary": "Not planned", "assumptions": [], "steps": [],
                                     "success_criteria": [], "refusal": reply}, "plan_refusal", [])
    for text, _tool, _args in safety.PROPOSE:
        plan_draft(text, "effects", {"summary": "Not planned", "assumptions": [], "steps": [],
                                     "success_criteria": [], "refusal": refusal_text}, "plan_refusal", [])
    return drafts


def collect_safety(ctx: Context) -> list[Draft]:
    drafts: list[Draft] = []
    for text, tool, args in safety.PROPOSE:
        d = Draft("safety", "propose", call_messages(text, tool, args), [tool], text,
                  PRIORITY["safety"], ctx.when_for(text))
        decl = ctx.book.declarations.get(tool) or {}
        declares_confirm = "confirm" in ((decl.get("parameters") or {}).get("properties") or {})
        if declares_confirm:
            d.prefs = {"rejected": assistant_call(tool, {**args, "confirm": True}),
                       "why": "adds a confirm argument the model must never supply"}
        drafts.append(d)
        for variant in variants(text, ctx.rng("propose", text), 1):
            drafts.append(Draft("safety_aug", "propose", call_messages(variant, tool, args), [tool],
                                variant, PRIORITY["safety_aug"], ctx.when_for(variant)))
    for text, reply in safety.DECLINES:
        drafts.append(Draft("safety", "decline", [user_message(text), assistant_text(reply)], [], text,
                            PRIORITY["safety"], ctx.when_for(text)))
        for variant in variants(text, ctx.rng("decline", text), 1):
            drafts.append(Draft("safety_aug", "decline", [user_message(variant), assistant_text(reply)],
                                [], variant, PRIORITY["safety_aug"], ctx.when_for(variant)))
    for item in safety.INJECTIONS:
        drafts.append(Draft("safety", "injection",
                            turn_messages(item.text, item.tool, item.args, item.result, True, item.reply),
                            [item.tool], item.text, PRIORITY["safety"], ctx.when_for(item.text)))
    for text, reply in safety.CLARIFY:
        drafts.append(Draft("safety", "clarify", [user_message(text), assistant_text(reply)], [], text,
                            PRIORITY["safety"], ctx.when_for(text)))
    return drafts


def collect_chat(ctx: Context) -> list[Draft]:
    return [Draft("chat", "chat", [user_message(text), assistant_text(reply)], [], text,
                  PRIORITY["chat"], ctx.when_for(text)) for text, reply in safety.CHAT]


def _flow_messages(text: str, steps: Sequence[tuple[str, dict[str, Any], str, bool]],
                   reply: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [user_message(text)]
    for tool, args, result, ok in steps:
        msgs += [assistant_call(tool, args), tool_result(tool, ok, result)]
    msgs.append(assistant_text(reply))
    return msgs


def collect_multistep(ctx: Context) -> list[Draft]:
    drafts: list[Draft] = []
    for flow in bank.FLOWS:
        steps = [(s.tool, s.args, s.result, s.ok) for s in flow.steps]
        d = Draft("multistep", "flow", _flow_messages(flow.text, steps, flow.reply),
                  [s.tool for s in flow.steps], flow.text, PRIORITY["multistep"], ctx.when_for(flow.text))
        if any(not s.ok for s in flow.steps):
            d.meta["outcome"] = "partial_failure"
            d.prefs = {"rejected": assistant_text("Done, both of those worked."),
                       "why": "claims success although a step failed"}
        drafts.append(d)
    explicit = [p for p in bank.PHRASEBANK if p.result and p.reply and ctx.book.has(p.tool)
                and ctx.book.risk(p.tool, p.args)[0] != "high"
                and not re.search(r"(?i)\b(?:and|then)\b", p.text)]
    rng = ctx.rng("combos")
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(seen) < ctx.cfg.multistep_combos and attempts < ctx.cfg.multistep_combos * 20:
        attempts += 1
        a, b = rng.sample(explicit, 2)
        if a.tool == b.tool or (a.text, b.text) in seen:
            continue
        seen.add((a.text, b.text))
        joiner = rng.choice((" and then ", ", then ", " and also "))
        text = a.text.rstrip(" .") + joiner + b.text[:1].lower() + b.text[1:].rstrip(" .")
        reply = a.reply.rstrip() + " " + b.reply.lstrip()
        steps = [(a.tool, a.args, a.result, True), (b.tool, b.args, b.result, True)]
        drafts.append(Draft("multistep", "flow", _flow_messages(text, steps, reply), [a.tool, b.tool],
                            text, PRIORITY["multistep"], ctx.when_for(text),
                            meta={"outcome": "composed"}))
    return drafts


# ── real experience ───────────────────────────────────────────────────────────────

_LEAD_IN = re.compile(
    r"(?i)^(?:(?:no+|nope|nah|wrong|not that|that'?s (?:not|wrong)(?: what i (?:asked|said|meant))?|"
    r"i (?:said|meant|asked(?: for)?|wanted)|actually,? i(?: meant| said)?|sorry,? i meant|"
    r"what i (?:said|meant|asked) (?:was|is)|try again)[,.!:\s]*)+")


def strip_correction(text: str) -> str:
    out = " ".join(str(text or "").split())
    return _LEAD_IN.sub("", out).strip(" ,.!")


def _short_clean_reply(reply: str) -> str:
    reply = " ".join(str(reply or "").split())
    if not reply or len(reply) > 300 or "\n" in reply or re.search(r"[*_`#]{2,}|\b\d\.\s", reply):
        return ""
    if failure_looks_like(reply):
        return ""
    return reply


def _args_for(ctx: Context, tool: str, text: str, given: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if given is not None:
        return dict(given), "recorded"
    args, how = argsynth.synthesise(ctx.book, tool, argsynth.first_clause(text),
                                    generic=ctx.cfg.generic_args)
    return args, how


def collect_real(ctx: Context, turns: Sequence[sources.RealTurn], bad_routes: set[tuple[str, str]],
                 want_corrections: bool, want_real: bool) -> list[Draft]:
    drafts: list[Draft] = []
    la = ctx.la
    for index, turn in enumerate(turns):
        text = turn.user
        if not turn.tools or not 2 <= len(text) <= 400:
            continue
        event = turn.tools[0]
        if not ctx.book.has(event.name):
            ctx.skipped["real_tool_not_offered"] += 1
            continue
        if la.open_ended(text):
            ctx.skipped["real_open_ended"] += 1
            continue
        corrective = sources.is_correction(text)
        followed_by_fix = (sources.is_correction(turn.next_user)
                           and turn.next_gap_s <= sources.CORRECTION_WINDOW_S)

        if corrective and want_corrections and turn.history is not None:
            body = strip_correction(text) or text
            args, how = _args_for(ctx, event.name, body, event.args)
            if args is None:
                ctx.skipped["correction_no_arguments"] += 1
                continue
            prev_user, prev_reply = turn.history
            msgs = [user_message(prev_user), assistant_text(prev_reply or "Done."),
                    user_message(text), assistant_call(event.name, args)]
            draft = Draft("correction", "correction", msgs, [event.name], text, PRIORITY["correction"],
                          datetime.fromtimestamp(turn.at), private=True, args_how=how)
            previous = next((t for t in reversed(turns[:index]) if t.user == prev_user and t.tools), None)
            if previous is not None and previous.tools[0].name != event.name \
                    and ctx.book.has(previous.tools[0].name):
                wrong_args, _how = _args_for(ctx, previous.tools[0].name, prev_user, previous.tools[0].args)
                if wrong_args is not None:
                    draft.prefs = {"prompt_override": [user_message(prev_user)],
                                   "chosen": assistant_call(event.name, args),
                                   "rejected": assistant_call(previous.tools[0].name, wrong_args),
                                   "why": "the owner corrected the first tool choice"}
            drafts.append(draft)
            continue

        if corrective or not want_real:
            continue
        if followed_by_fix:
            ctx.skipped["real_later_corrected"] += 1
            continue
        if (sources.normalise(text), event.name) in bad_routes:
            ctx.skipped["real_reflex_failed"] += 1
            continue
        args, how = _args_for(ctx, event.name, text, event.args)
        if args is None:
            ctx.skipped["real_no_arguments"] += 1
            continue
        when = datetime.fromtimestamp(turn.at)
        risk = ctx.book.risk(event.name, args)[0]
        reply = _short_clean_reply(turn.reply)
        history = None
        if turn.history and not sources.is_correction(text):
            history = turn.history
        if event.result and event.ok and reply and risk == "read":
            msgs = turn_messages(text, event.name, args, event.result, True, reply)
            kind = "turn"
        else:
            msgs = call_messages(text, event.name, args)
            kind = "call"
        if history and len(history[0]) <= 200 and len(history[1]) <= 300 \
                and ctx.rng("history", text).random() < 0.25:
            msgs = [user_message(history[0]), assistant_text(history[1]), *msgs]
        drafts.append(Draft("real", kind, msgs, [event.name], text, PRIORITY["real"], when,
                            private=True, args_how=how, meta={"evidence": event.source}))
    return drafts


def answers_failed_route(d: Draft, bad_routes: set[tuple[str, str]]) -> bool:
    """True if *d* answers a phrase the reflex ledger saw fail: with a call to the
    tool that failed, or in words. Every assistant message is trained on (see
    encode.py), so a remembered "Spotify's open." in a row's history teaches the
    same claim as the call would."""
    failed = {phrase for phrase, _tool in bad_routes}
    for i, m in enumerate(d.messages):
        if m["role"] != "user" or sources.normalise(m["content"]) not in failed:
            continue
        answer = d.messages[i + 1] if i + 1 < len(d.messages) else {}
        called = [name for name, _args in calls_in([answer])]
        if not called or (sources.normalise(m["content"]), called[0]) in bad_routes:
            return True
    return False


def collect_reflex(ctx: Context, rows: Sequence[sources.ReflexRow]) -> list[Draft]:
    drafts: list[Draft] = []
    for row in rows:
        if row.failures > 0 and row.hits <= row.failures:
            continue
        if row.hits < 1 or not row.phrase.strip() or not ctx.book.has(row.tool):
            continue
        if ctx.la.open_ended(row.phrase):
            continue
        given = {"action": row.action} if row.action else None
        args, how = _args_for(ctx, row.tool, row.phrase, given)
        if args is None:
            ctx.skipped["reflex_no_arguments"] += 1
            continue
        drafts.append(Draft("reflex", "call", call_messages(row.phrase, row.tool, args), [row.tool],
                            row.phrase, PRIORITY["reflex"], ctx.when_for(row.phrase), private=True,
                            args_how=how, meta={"promoted": row.promoted}))
    return drafts


# ── cleaning ──────────────────────────────────────────────────────────────────────

def _texts(messages: Sequence[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for m in messages:
        if isinstance(m.get("content"), str):
            out.append(m["content"])
        for call in m.get("tool_calls") or []:
            out += _strings((call.get("function") or {}).get("arguments"))
    return out


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def redact_messages(redactor: Redactor, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        copy = dict(m)
        if isinstance(copy.get("content"), str):
            copy["content"] = redactor.redact(copy["content"])
        if copy.get("tool_calls"):
            calls = []
            for call in copy["tool_calls"]:
                fn = dict(call.get("function") or {})
                fn["arguments"] = redactor.redact_obj(fn.get("arguments") or {})
                calls.append({**call, "function": fn})
            copy["tool_calls"] = calls
        out.append(copy)
    return out


def finalise(ctx: Context, d: Draft) -> Draft | None:
    """Redact and validate one draft; None (with a counted reason) if it fails."""
    d.messages = redact_messages(ctx.redactor, d.messages)
    d.text = next((m["content"] for m in reversed(d.messages) if m["role"] == "user"), d.text)
    if d.private:
        topic = next((t for s in _texts(d.messages) if (t := sensitive_topic(s))), "")
        if topic:
            ctx.dropped["sensitive_topic"] += 1
            return None
    calls = calls_in(d.messages)
    if d.tools_called and not calls:
        ctx.dropped["no_call"] += 1
        return None
    for name, args in calls:
        problem = ctx.book.validate(name, args)
        if problem:
            ctx.dropped["invalid_call"] += 1
            ctx.skipped["invalid_call:" + problem.split(" ")[0][:24]] += 1
            return None
    # A high-risk call is never followed by a result: the agent stops to ask.
    seen_high = False
    for m in d.messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for name, args in calls_in([m]):
                if ctx.book.risk(name, args)[0] == "high":
                    seen_high = True
        elif m["role"] == "tool" and seen_high:
            ctx.dropped["high_risk_with_result"] += 1
            return None
    if calls and ctx.la.open_ended(d.text):
        ctx.dropped["open_ended_with_call"] += 1
        return None
    if d.prefs:
        for key in ("rejected", "chosen"):
            if key in d.prefs:
                d.prefs[key] = redact_messages(ctx.redactor, [d.prefs[key]])[0]
        if "prompt_override" in d.prefs:
            d.prefs["prompt_override"] = redact_messages(ctx.redactor, d.prefs["prompt_override"])
    d.meta["risk"] = _risk_of(ctx.book, calls) if calls else "none"
    return d


def group_key(d: Draft) -> str:
    return sources.normalise(d.text)


def dedupe_key(d: Draft) -> str:
    users = [sources.normalise(m["content"]) for m in d.messages if m["role"] == "user"]
    return sha256_text(canonical([users, calls_in(d.messages),
                                  [m["content"] for m in d.messages
                                   if m["role"] == "assistant" and not m.get("tool_calls")][-1:]]))


def bucket_of(d: Draft) -> str:
    if d.tools_called:
        return d.tools_called[0] if d.kind not in {"flow"} else "_flow"
    return "_" + d.kind


def _cap_for(cfg: BuildConfig, bucket: str) -> int:
    return {"_decline": cfg.max_decline, "_chat": cfg.max_chat, "_clarify": cfg.max_clarify,
            "_flow": cfg.max_multistep, "_plan": cfg.max_planner,
            "_plan_refusal": cfg.max_planner}.get(bucket, cfg.max_per_tool)


# ── assembling rows ───────────────────────────────────────────────────────────────

def build_row(ctx: Context, d: Draft) -> dict[str, Any] | None:
    cfg = ctx.cfg
    when = d.when or ctx.when_for(d.text)
    if d.system is not None:
        messages = [{"role": "system", "content": d.system}, *d.messages]
    else:
        messages = [system_message(when), *d.messages]
    if d.system is not None or ctx.la.open_ended(d.text):
        offered: list[str] = []                   # the live agent offers no tools for these
    else:
        # Room for distractor schemas is whatever the conversation leaves of the token budget.
        spent = sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
        room = int((cfg.max_tokens_estimate - spent) * CHARS_PER_TOKEN)
        offered = ctx.book.pack(d.tools_called, ctx.ranked(d.text), max_tools=cfg.max_tools,
                                char_budget=max(0, min(cfg.tools_char_budget, room)))
    tools = tool_schemas(ctx.book, offered)
    tokens = estimate_tokens(json.dumps(tools, ensure_ascii=False)) + sum(
        estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
    if tokens > cfg.max_tokens_estimate:
        ctx.dropped["too_long"] += 1
        return None
    calls = calls_in(d.messages)
    body_hash = sha256_text(canonical([offered, d.messages]))[:12]
    group = sha256_text(group_key(d))[:12]
    row = {"id": f"ex-{body_hash}", "source": d.source, "kind": d.kind,
           "tools": tools, "messages": messages,
           "meta": {"tools_called": [n for n, _ in calls], "offered": offered,
                    "risk": d.meta.get("risk", "none"), "args": d.args_how, "group": group,
                    "tokens_estimate": tokens,
                    **{k: v for k, v in d.meta.items() if k not in {"risk"}}}}
    # Belt and braces: what is about to be written must already be clean.
    probe = Redactor(ctx.redactor.terms)
    if probe.redact_obj(row["messages"]) != row["messages"]:
        ctx.dropped["unredacted"] += 1
        return None
    return row


def build_preference(ctx: Context, d: Draft, row: dict[str, Any]) -> dict[str, Any] | None:
    if not d.prefs or "rejected" not in d.prefs:
        return None
    if "prompt_override" in d.prefs:
        prompt = [row["messages"][0], *d.prefs["prompt_override"]]
        chosen = [d.prefs["chosen"]]
    else:
        prompt = row["messages"][:-1]
        chosen = [row["messages"][-1]]
    return {"id": "pref-" + row["id"][3:], "source": d.source, "kind": d.kind, "why": d.prefs.get("why", ""),
            "tools": row["tools"], "prompt": prompt, "chosen": chosen, "rejected": [d.prefs["rejected"]],
            "meta": {"group": row["meta"]["group"]}}


def split_rows(rows: list[dict[str, Any]], fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group-wise: every phrasing of one request lands on the same side."""
    if fraction <= 0 or not rows:
        return list(rows), []
    groups = sorted({r["meta"]["group"] for r in rows},
                    key=lambda g: sha256_text(f"{seed}|split|{g}"))
    wanted = max(1, int(math.ceil(len(groups) * fraction))) if len(groups) >= 8 else 0
    held = set(groups[:wanted])
    return ([r for r in rows if r["meta"]["group"] not in held],
            [r for r in rows if r["meta"]["group"] in held])


# ── the build ────────────────────────────────────────────────────────────────────────

def _eval_guard(ctx: Context) -> tuple[EvalGuard, dict[str, Any]]:
    cfg = ctx.cfg
    if cfg.eval_strings is not None:
        strings, report = list(cfg.eval_strings), {"sources": {"injected": len(cfg.eval_strings)},
                                                   "missing": []}
    else:
        strings, report = collect_eval_strings(ctx.root, include_routing_holdout=cfg.routing_split != "all")
    guard = EvalGuard(strings)
    report["strings"] = len(guard)
    report["fingerprint"] = guard.fingerprint
    return guard, report


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return {"rows": len(rows), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text.encode("utf-8"))}


def build(cfg: BuildConfig) -> BuildResult:
    ctx = Context(cfg)
    wanted = set(cfg.sources)
    unknown = wanted - set(ALL_SOURCES)
    if unknown:
        raise ValueError(f"unknown source(s): {', '.join(sorted(unknown))}")
    config_dir = Path(cfg.config_dir) if cfg.config_dir is not None else _default_config_dir()

    drafts: list[Draft] = []
    store_report: dict[str, Any] = {}
    bad_routes: set[tuple[str, str]] = set()
    reflex_rows: list[sources.ReflexRow] = []
    if wanted & {"real", "corrections", "reflex"}:
        store_report = sources.fingerprint(config_dir)
        reflex_rows = sources.read_reflex(config_dir)
        bad_routes = {(sources.normalise(r.phrase), r.tool) for r in reflex_rows
                      if r.failures > 0 and r.failures >= r.hits}
    if wanted & {"real", "corrections"}:
        turns = sources.build_turns(sources.read_episodes(config_dir), sources.read_shadow(config_dir),
                                    sources.read_jobs(config_dir))
        drafts += collect_real(ctx, turns, bad_routes, "corrections" in wanted, "real" in wanted)
    if "reflex" in wanted:
        drafts += collect_reflex(ctx, reflex_rows)
    if "routing" in wanted:
        drafts += collect_routing(ctx)
    if "phrasebank" in wanted:
        drafts += collect_phrasebank(ctx)
    if "generated" in wanted:
        drafts += collect_generated(ctx)
    if "safety" in wanted:
        drafts += collect_safety(ctx)
    if "multistep" in wanted:
        drafts += collect_multistep(ctx)
    if "chat" in wanted:
        drafts += collect_chat(ctx)
    if "planner" in wanted:
        drafts += collect_planner(ctx)
    collected = len(drafts)
    for d in drafts:
        ctx.produced[d.source] += 1

    # what the reflex ledger saw fail is left out of every source, not only the owner's turns
    if bad_routes:
        before = len(drafts)
        drafts = [d for d in drafts if not answers_failed_route(d, bad_routes)]
        if len(drafts) < before:
            ctx.dropped["reflex_failed"] += before - len(drafts)

    # clean
    cleaned = [x for x in (finalise(ctx, d) for d in drafts) if x is not None]

    # de-duplicate: best source first
    cleaned.sort(key=lambda d: (d.priority, sha256_text(dedupe_key(d))))
    seen: set[str] = set()
    unique: list[Draft] = []
    for d in cleaned:
        key = dedupe_key(d)
        if key in seen:
            ctx.dropped["duplicate"] += 1
            continue
        seen.add(key)
        unique.append(d)

    # keep the measuring stick out
    guard, guard_report = _eval_guard(ctx)
    kept: list[Draft] = []
    for d in unique:
        users = [m["content"] for m in d.messages if m["role"] == "user"] + [d.text]
        if any(guard.overlaps(u) for u in users):
            ctx.dropped["evaluation_overlap"] += 1
            continue
        kept.append(d)

    # balance: a cap per tool (and per kind of no-tool row), best sources first
    per_bucket: Counter[str] = Counter()
    balanced: list[Draft] = []
    for d in sorted(kept, key=lambda d: (d.priority, sha256_text(str(cfg.seed) + dedupe_key(d)))):
        bucket = bucket_of(d)
        if per_bucket[bucket] >= _cap_for(cfg, bucket):
            ctx.dropped["over_cap:" + ("tool" if d.tools_called else bucket)] += 1
            continue
        per_bucket[bucket] += 1
        balanced.append(d)

    # rows
    rows: list[dict[str, Any]] = []
    prefs: list[dict[str, Any]] = []
    for d in sorted(balanced, key=lambda d: dedupe_key(d)):
        row = build_row(ctx, d)
        if row is None:
            continue
        rows.append(row)
        pair = build_preference(ctx, d, row)
        if pair is not None:
            prefs.append(pair)
    train, val = split_rows(rows, cfg.val_fraction, cfg.seed)

    manifest = _manifest(ctx, cfg, collected, train, val, prefs, guard_report, store_report)
    result = BuildResult(manifest, train, val, prefs)
    if cfg.dry_run:
        return result

    root = Path(cfg.out_dir) if cfg.out_dir is not None else training_root(config_dir)
    require_private_location(root, allow=cfg.allow_synced_output)
    dataset_id = f"ds-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(ctx.now))}-{uuid.uuid4().hex[:6]}"
    staging = root / f".{dataset_id}.partial"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        files = {"train.jsonl": _write(staging / "train.jsonl", train),
                 "val.jsonl": _write(staging / "val.jsonl", val),
                 "preferences.jsonl": _write(staging / "preferences.jsonl", prefs)}
        manifest["files"] = files
        manifest["data_hash"] = sha256_text(canonical({k: v["sha256"] for k, v in files.items()}))
        manifest["dataset_id"] = dataset_id
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                               encoding="utf-8")
        final = root / dataset_id
        staging.rename(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    (root / "latest.txt").write_text(dataset_id + "\n", encoding="utf-8")
    result.path = final
    return result


def _default_config_dir() -> Path:
    from ..constants import CONFIG_DIR
    return Path(CONFIG_DIR)


def _manifest(ctx: Context, cfg: BuildConfig, collected: int, train: list[dict[str, Any]],
              val: list[dict[str, Any]], prefs: list[dict[str, Any]], guard: dict[str, Any],
              stores: dict[str, Any]) -> dict[str, Any]:
    rows = train + val
    by_tool: Counter[str] = Counter()
    for r in rows:
        by_tool[(r["meta"]["tools_called"] or ["(no tool)"])[0]] += 1
    tokens = [r["meta"]["tokens_estimate"] for r in rows] or [0]
    return {
        "format": DATASET_FORMAT, "builder_version": BUILDER_VERSION,
        "created_at": datetime.fromtimestamp(ctx.now, tz=timezone.utc).isoformat(timespec="seconds"),
        "dry_run": cfg.dry_run,
        "counts": {"candidates": collected, "rows": len(rows), "train": len(train), "val": len(val),
                   "preferences": len(prefs),
                   "by_source": dict(Counter(r["source"] for r in rows)),
                   "by_kind": dict(Counter(r["kind"] for r in rows)),
                   "by_risk": dict(Counter(r["meta"]["risk"] for r in rows)),
                   "by_args": dict(Counter(r["meta"]["args"] for r in rows)),
                   "by_tool": dict(sorted(by_tool.items(), key=lambda kv: (-kv[1], kv[0]))),
                   "tools_covered": len([k for k in by_tool if k != "(no tool)"])},
        "produced_by_source": dict(ctx.produced),
        "dropped": dict(ctx.dropped), "skipped": dict(ctx.skipped),
        "redactions": dict(ctx.redactor.counts),
        "evaluation_guard": guard,
        "routing_split": cfg.routing_split,
        "contaminated_routing": cfg.routing_split == "all",
        "tool_surface_sha256": ctx.book.fingerprint(),
        "system_prompt_sha256": system_prompt_hash(),
        "tokens_estimate": {"max": max(tokens), "mean": round(sum(tokens) / max(1, len(tokens)), 1)},
        "stores": stores,
        "config": {"sources": sorted(cfg.sources), "val_fraction": cfg.val_fraction,
                   "max_per_tool": cfg.max_per_tool, "augment": cfg.augment, "seed": cfg.seed,
                   "max_tools": cfg.max_tools, "tools_char_budget": cfg.tools_char_budget,
                   "max_tokens_estimate": cfg.max_tokens_estimate,
                   "generic_args": cfg.generic_args,
                   "redact_terms": len(ctx.redactor.terms)},
        "privacy": "Private to this computer. Redacted, but built from the owner's own conversations: "
                   "never commit, share or upload it, or any adapter trained on it.",
    }


__all__ = ["ALL_SOURCES", "BuildConfig", "BuildResult", "Draft", "answers_failed_route", "build", "load_rows",
           "strip_correction", "variants", "verify_dataset"]
