"""
Reflex fast-path (Mark XXVI) — instant local understanding of clear commands.

Almost every spoken/typed turn currently pays a full model round-trip, even for
unambiguous imperatives ("start a 25 minute focus block", "what's my runway",
"quiz me"). This layer recognises a small, curated set of HIGH-PRECISION command
forms locally and maps them straight to a tool call — so ORION acts in tens of
milliseconds instead of a network turn, and never mis-selects a tool for the ones
that matter most.

Design rules that keep it safe:

  * **Precision over recall.** A rule fires only on an unambiguous imperative /
    interrogative shape (anchored, specific verbs). Anything conversational,
    hedged, or open-ended returns None and flows to the model unchanged. The
    tests enforce a corpus of "must NOT match" conversational lines.
  * **Only safe, known-argument tools.** Read-mostly or trivially-reversible
    intents (focus/study/catch_up/finance/diagnostics/token_usage). No
    destructive, no ambiguous-argument, no open-domain commands.
  * **Pure.** ``match_reflex(text) -> ReflexMatch | None`` is a free function with
    no I/O; the worker does the dispatching. Fully unit-testable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ReflexMatch:
    tool: str
    args: dict
    why: str            # which reflex fired (for the log / trace)


def _focus_start(m: re.Match) -> tuple[str, dict, str]:
    args: dict = {"action": "start"}
    if m.group("min"):
        args["minutes"] = int(m.group("min"))
    preset = (m.group("preset") or "").lower()
    for key in ("pomodoro", "long", "sprint"):
        if key in preset:
            args["preset"] = key
            break
    else:
        if "deep" in preset:
            args["preset"] = "deep"
    label = (m.group("label") or "").strip()
    args["label"] = label or "focus"
    return "focus", args, "focus.start"


# Each rule: (compiled anchored pattern, builder(match) -> (tool, args, why)).
# Patterns are deliberately strict — they describe a COMMAND, not a conversation.
# Every rule accepts any of . ? ! as its terminator: the punctuation on a spoken
# command is chosen by the transcriber, not the speaker, so treating "log my
# mood?" as a different utterance from "log my mood." would drop real commands.
_RULES: list[tuple[re.Pattern, Callable[[re.Match], tuple[str, dict, str]]]] = [
    # focus — "start a [25 minute] {pomodoro|deep work|focus|long|sprint} [block] [on X]"
    (re.compile(
        r"^(?:start|begin|kick off)\s+(?:a\s+|an\s+)?"
        r"(?:(?P<min>\d{1,3})\s*(?:min|mins|minute|minutes|m)\s+)?"
        r"(?P<preset>pomodoro|deep[- ]?work(?:\s+(?:block|session))?|"
        r"deep\s+(?:block|session)|long\s+(?:block|session)|"
        r"focus(?:\s+(?:block|session))?|sprint)"
        r"(?:\s+(?:on|for|about)\s+(?P<label>.+?))?[?.!]?$", re.I), _focus_start),
    (re.compile(r"^(?:how (?:long|much time)\s+(?:is\s+)?(?:left|remaining)"
                r"(?:\s+(?:on\s+)?(?:my\s+)?focus)?|focus status)[?.!]?$", re.I),
     lambda m: ("focus", {"action": "status"}, "focus.status")),
    (re.compile(r"^(?:end|finish|stop)\s+(?:my\s+|the\s+)?focus(?:\s+(?:block|session))?[?.!]?$", re.I),
     lambda m: ("focus", {"action": "done"}, "focus.done")),

    # study — "quiz me", "review my cards", "what's due"
    (re.compile(r"^(?:quiz me|test me|review(?:\s+my)?\s+(?:cards|flashcards)|"
                r"start (?:a\s+)?review)[?.!]?$", re.I),
     lambda m: ("study", {"action": "review"}, "study.review")),
    (re.compile(r"^(?:what'?s\s+due(?:\s+for\s+review)?|study (?:stats|progress)|"
                r"how many (?:cards|flashcards) (?:are\s+)?due)[?.!]?$", re.I),
     lambda m: ("study", {"action": "stats"}, "study.stats")),

    # daily brief
    (re.compile(r"^(?:catch me up|situation report|brief me|"
                r"where do (?:things|we) stand|what'?s (?:new|the situation))[?.!]?$", re.I),
     lambda m: ("catch_up", {"action": "report"}, "catch_up")),

    # finance — runway / balance
    (re.compile(r"^(?:what'?s\s+)?(?:my\s+)?runway(?:\s+(?:like|now))?[?.!]?$", re.I),
     lambda m: ("finance", {"action": "runway"}, "finance.runway")),
    (re.compile(r"^how (?:much|long) (?:is\s+)?(?:my\s+)?runway[?.!]?$", re.I),
     lambda m: ("finance", {"action": "runway"}, "finance.runway")),
    (re.compile(r"^(?:what'?s\s+)?(?:my\s+)?(?:cash|balance|liquid cash)(?:\s+(?:left|now))?[?.!]?$", re.I),
     lambda m: ("finance", {"action": "balance"}, "finance.balance")),

    # capability health — "what can you do", "why can't you…"
    (re.compile(r"^(?:what can you (?:actually\s+)?do|what are you capable of|"
                r"capabilit(?:y|ies)(?:\s+health)?|run (?:a\s+)?capability check)[?.!]?$", re.I),
     lambda m: ("diagnostics", {"action": "capabilities"}, "diagnostics.capabilities")),

    # token usage
    (re.compile(r"^(?:token usage|how many tokens(?:\s+(?:have\s+)?i\s+used)?|"
                r"how much (?:usage|is left|do i have left))[?.!]?$", re.I),
     lambda m: ("token_usage", {}, "token_usage")),

    # ── Mark XXVI expansion ──────────────────────────────────────────────────
    # Each of these previously cost a full model round-trip (1-3 s) to reach a
    # tool call that was never in doubt. The patterns stay strictly anchored:
    # a reflex that fires on a conversational sentence is far worse than one
    # that misses, because the model fall-through is always correct and only
    # slower.

    # wellbeing
    (re.compile(r"^(?:how(?:'?s| is| am i)\s+(?:my\s+)?(?:wellbeing|sleeping|sleep|mood)"
                r"(?:\s+(?:lately|recently|been|doing|going))?|"
                r"wellbeing (?:trend|report|summary))[?.!]?$", re.I),
     lambda m: ("wellbeing", {"action": "trend"}, "wellbeing.trend")),
    (re.compile(r"^(?:wellbeing (?:check ?in|today)|"
                r"(?:log|record) (?:my |a )?(?:check ?in|mood))[?.!]?$", re.I),
     lambda m: ("wellbeing", {"action": "checkin"}, "wellbeing.checkin")),

    # decisions — the calibration loop
    (re.compile(r"^(?:(?:what|which) decisions (?:are\s+)?due|decisions due|"
                r"any decisions to (?:resolve|score))[?.!]?$", re.I),
     lambda m: ("decision", {"action": "due"}, "decision.due")),
    (re.compile(r"^(?:(?:my |show (?:me )?(?:my )?)?calibration(?:\s+(?:score|curve|report))?|"
                r"how (?:well )?calibrated am i|(?:my )?brier score)[?.!]?$", re.I),
     lambda m: ("decision", {"action": "calibration"}, "decision.calibration")),

    # plugins
    (re.compile(r"^(?:(?:list|show) (?:my |the )?plugins|what plugins"
                r"(?:\s+(?:do you have|are (?:installed|there)))?)[?.!]?$", re.I),
     lambda m: ("plugin", {"action": "list"}, "plugin.list")),
    (re.compile(r"^(?:plugin (?:doctor|health|check)|"
                r"(?:are|is) (?:my |the )?plugins? (?:ok|healthy|working))[?.!]?$", re.I),
     lambda m: ("plugin", {"action": "doctor"}, "plugin.doctor")),

    # responsiveness — the new latency view
    (re.compile(r"^(?:why (?:were|are) you (?:so\s+)?(?:slow|laggy|lagging|stuttering)|"
                r"(?:are you|were you) lagging|"
                r"(?:show (?:me )?(?:your )?)?(?:latency|performance) (?:report|stats)|"
                r"latency report)[?.!]?$", re.I),
     lambda m: ("diagnostics", {"action": "latency"}, "diagnostics.latency")),

    # appearance — "go into your orb form". Asked for when somebody else has
    # just walked in, so it must not wait on a model round-trip.
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+|please\s+)?(?:"
                r"(?:go|switch|change|turn|morph|transform|shift)(?:\s+back)?\s+(?:in\s*)?to\s+(?:your\s+)?(?:the\s+)?(?P<a>orb|sphere|face|human)(?:\s+(?:form|mode|shape|version))?"
                r"|(?:use|be|become|wear)\s+(?:your\s+|an?\s+|the\s+)?(?P<d>orb|sphere|face|human)(?:\s+(?:form|mode|shape))?"
                r"|(?P<b>orb|sphere|face|human)\s+(?:form|mode)"
                r"|(?:show|give)\s+me\s+(?:your\s+)?(?P<c>orb|sphere|face)(?:\s+(?:form|mode))?"
                r")(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("interface_control",
                {"action": "face_form",
                 "target": (m.group("a") or m.group("b") or m.group("c")
                            or m.group("d") or "").lower()},
                "interface.face_form")),

    # camera — "show me the camera" is a request to SEE, not to be told.
    # It must be matched BEFORE the page rule below, which would otherwise
    # swallow "show me the camera lab" as ordinary navigation. Note what is
    # deliberately absent: anything resembling "what can you see" or "what's
    # in front of me", which are questions for vision_analyse and belong to
    # the model, not to a regex.
    (re.compile(r"^(?:(?:can you\s+)?(?:show|bring|pull|put|open)"
                r"(?:\s+me)?(?:\s+up)?\s+(?:the\s+|my\s+)?(?:live\s+)?"
                r"(?:camera|webcam)(?:\s+(?:lab|feed|view))?|"
                r"camera\s+lab|let me see the camera)[?.!]?$", re.I),
     lambda m: ("interface_control", {"action": "camera"}, "interface.camera")),

    # deck navigation — the trailing "page"/"deck"/"tab" is what keeps this
    # anchored. Without it "show me the memory" is a request for a memory,
    # not for a page, and a reflex firing on a conversational sentence is far
    # worse than one that misses. The tool validates the name against the real
    # page list, so a misheard target is reported rather than silently ignored.
    (re.compile(r"^(?:open|go to|show me|switch to|take me to|bring up)\s+"
                r"(?:the\s+)?(?P<name>[a-z0-9][a-z0-9 ]{1,28}?)\s+"
                r"(?:page|deck|tab)[?.!]?$", re.I),
     lambda m: ("interface_control",
                {"action": "page", "target": m.group("name").strip()},
                "interface.page")),

    # self-diagnostic
    (re.compile(r"^(?:run (?:a )?(?:self[- ])?diagnostics?(?:\s+check)?|"
                r"(?:self[- ])?diagnostics?|system check|check yourself)[?.!]?$", re.I),
     lambda m: ("diagnostics", {"action": "full"}, "diagnostics.full")),
]


def match_reflex(text: str) -> ReflexMatch | None:
    """The first high-precision reflex the text matches, or None (→ model)."""
    t = (text or "").strip()
    if not t or len(t) > 90:            # a long utterance is a conversation, not a command
        return None
    for pattern, build in _RULES:
        m = pattern.match(t)
        if m:
            tool, args, why = build(m)
            return ReflexMatch(tool, args, why)
    return None


def reflex_enabled() -> bool:
    """On by default (the point is instant response); disable with ORION_REFLEX=0.
    Precision is enforced by tests, and the model fall-through is always intact."""
    return os.getenv("ORION_REFLEX", "1").strip().lower() not in {"0", "false", "off", "no"}


__all__ = ["ReflexMatch", "match_reflex", "reflex_enabled"]
