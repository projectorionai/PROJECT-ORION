"""Hold ORION's spoken claims to what his tools actually did.

Recover a promised investigation only from an explicit user instruction, and
(Mark XXXVII) notice any other completion claim - "I've sent it", "it's no
longer listed", "the reminder is set" - that no successful tool backs.

This module makes no calls and does no I/O. The Live worker supplies the
completed turn, the request identity, and receipts of its real tool calls.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

START_ACTIONS = frozenset({
    "start", "conduct", "research", "paper", "write_paper", "deep", "deep_research",
    "study", "thesis", "report", "dossier", "investigate", "unattended", "background",
    "browse", "live", "read", "properly", "human",
})
_GENERIC = frozenset({
    "a", "an", "the", "all", "these", "those", "this", "that", "them", "it", "topics", "topic",
    "subjects", "subject", "everything", "more", "depth", "deeper", "detail", "detailed", "further",
    "recent", "latest", "briefing", "in", "on", "into", "about", "for", "please", "research",
    "investigation", "dossier", "report", "news", "now", "currently",
})


def is_research_start(name: str, args: Any) -> bool:
    """Tool discovery/status/cancellation are not evidence of an investigation."""
    params = args
    for _depth in range(5):
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (ValueError, TypeError):
                return False
        if not isinstance(params, dict):
            return False
        if name == "antigravity_research":
            return True
        if name == "research":
            return str(params.get("action") or "start").strip().lower() in START_ACTIONS
        if name == "use_tool":
            name = str(params.get("name") or params.get("tool") or "")
            params = params.get("arguments", params.get("args", {}))
        elif name == "job" and str(params.get("action") or "list").lower() in {
                "run", "start", "background", "detach"}:
            name = str(params.get("tool") or params.get("tool_name") or "")
            params = params.get("args") or params.get("tool_args") or {}
        else:
            return False
    return False


def _meaningful(topic: str) -> bool:
    words = re.findall(r"[^\W_]+", topic.casefold())
    return bool(words) and any(word not in _GENERIC and len(word) >= 2 for word in words)


def _clean_topic(topic: str) -> str:
    topic = " ".join(topic.strip(" \t\r\n.,:;!?'\"").split())
    topic = re.sub(r"^(?:on|into|about)\s+", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"\s+(?:in (?:more )?depth|in (?:more )?detail|deeply|further|please)$", "", topic,
                   flags=re.IGNORECASE)
    return topic if 2 <= len(topic) <= 600 and _meaningful(topic) else ""


def user_research_request(user: str) -> tuple[bool, str]:
    """Return consent and a concrete topic; consent can be deictic, never inferred."""
    text = " ".join(str(user or "").replace("’", "'").split()).strip()
    if re.search(r"\b(?:don't|do not|never|stop|cancel|not yet|without)\b", text, re.IGNORECASE):
        return False, ""
    if re.search(r"\b(?:later|tomorrow|tonight|next week|next month)\b", text, re.IGNORECASE):
        return False, ""             # permission to schedule is not permission to start now
    prefix = (r"^(?:orion[, ]+)?(?:please\s+)?"
              r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
              r"i\s+(?:want|need|would like)\s+you\s+to\s+)?")
    match = re.match(prefix + r"(?:research|investigate|look into|conduct research (?:on|into)|"
                     r"do (?:deep |in-depth )?research (?:on|into))\s+(.+)", text, re.IGNORECASE)
    if match:
        return True, _clean_topic(match.group(1))
    match = re.match(prefix + r"(?:explore|examine)\s+(.+)", text, re.IGNORECASE)
    if match and re.search(r"\b(?:in (?:more )?depth|in (?:more )?detail|deeply|further)\b",
                           match.group(1), re.IGNORECASE):
        return True, _clean_topic(match.group(1))
    return False, ""


def promised_research(reply: str) -> tuple[bool, str]:
    """Recognise a narrow affirmative promise; extract its named topic if present."""
    text = " ".join(str(reply or "").replace("’", "'").split())
    if re.search(r"\b(?:haven't|have not|had not|didn't|did not|not yet|couldn't|cannot|can't)\b",
                 text, re.IGNORECASE):
        return False, ""
    underway = re.search(
        r"\b(?:research|investigation|study)\s+(?:into|on|about)\s+(.{2,600}?)\s+"
        r"(?:is\s+|has\s+)?(?:(?:now|already)\s+)?(?:underway|in progress|started|begun)\b",
        text, re.IGNORECASE)
    if underway:
        return True, _clean_topic(underway.group(1))
    ongoing = re.search(
        r"\bi(?:'m| am| will|'ll)\s+(?:now\s+)?(?:researching|investigating|research|investigate)\s+"
        r"([^.!?]{2,600})", text, re.IGNORECASE)
    if ongoing:
        return True, _clean_topic(re.sub(r"\s+now$", "", ongoing.group(1), flags=re.IGNORECASE))
    doing = re.search(r"\bi(?:'m| am)\s+(?:now\s+)?doing\s+(?:deep\s+)?research\b",
                      text, re.IGNORECASE)
    if doing:
        topic = re.match(r"\s+(?:on|into|about)\s+([^.!?]{2,600})", text[doing.end():], re.IGNORECASE)
        return True, _clean_topic(topic.group(1)) if topic else ""
    promise = re.search(
        r"\bi(?:'ll| will| am|'m)\s+(?:now\s+)?(?:starting|beginning|start|begin|conduct|"
        r"conducting|carry out|carrying out|compile|compiling|prepare|preparing)\s+"
        r"(?:an?\s+)?(?:(?:extensive|detailed|deep|in-depth|comprehensive)\s+)?"
        r"(?:research|investigation|dossier|report|study)\b", text, re.IGNORECASE)
    if promise:
        topic = re.match(r"\s+(?:on|into|about)\s+([^.!?]{2,600})", text[promise.end():], re.IGNORECASE)
        return True, _clean_topic(topic.group(1)) if topic else ""
    return False, ""


@dataclass(frozen=True)
class ResearchRecovery:
    topic: str = ""
    correction: str = ""


# ── completion claims and the receipts that back them ─────────────────────────

def effective_tool(name: Any, args: Any) -> tuple[str, Any]:
    """The tool a call really ran, through use_tool and job envelopes."""
    tool, params = str(name or ""), args
    for _depth in range(3):
        if isinstance(params, str):
            try:
                params = json.loads(params) if params.strip() else {}
            except (ValueError, TypeError):
                params = {}
        if not isinstance(params, dict):
            return tool, {}
        if tool == "use_tool":
            tool = str(params.get("name") or params.get("tool") or "") or tool
            params = params.get("arguments", params.get("args", {}))
        elif tool == "job" and str(params.get("action") or "").lower() in {
                "run", "start", "background", "detach"}:
            tool = str(params.get("tool") or params.get("tool_name") or "") or tool
            params = params.get("args") or params.get("tool_args") or {}
        else:
            return tool, params
    return tool, params if isinstance(params, dict) else {}


@dataclass
class ToolReceipt:
    """One real tool call of a request: running until it ends, then its outcome."""

    name: str
    action: str = ""
    research: bool = False
    ok: bool | None = None
    result: str = ""
    at: float = field(default_factory=time.monotonic)

    @property
    def running(self) -> bool:
        return self.ok is None

    @property
    def reads_only(self) -> bool:
        """A look (list, status, search), which changes nothing it could back."""
        return bool(_READ_ACTIONS.match(self.action.strip().lower()))


#: Tools that can do anything (multi-step executors, the desktop, plugins,
#: MCP servers): a success from one of them cannot be ruled out as the
#: receipt for any claim, so it satisfies all of them.
_GENERIC_TOOLS = re.compile(
    r"plan|workflow|agent|macro|routine|protocol|automat|computer_use|desktop_control|"
    r"copilot|execute|skill|chain|plugin|custom|mcp__|forge|swarm", re.IGNORECASE)
#: Actions that only look. 17 Oct 11:13: awareness was asked to list the
#: tasks (fine) and to clear the completed ones (rejected), and ORION said
#: "they are no longer listed" - the successful LOOK is no receipt for that.
#: What a tool does when it is called with no action.
_DEFAULT_ACTIONS = {"file_controller": "list", "notion_workspace": "list_tasks", "job": "list"}

_READ_ACTIONS = re.compile(
    r"(?:list|status|get|show|read|search|find|query|check|view|describe|summar|overview|"
    r"info|lookup|recall|due|upcoming|today)", re.IGNORECASE)
#: Tools that relay what the world says (a briefing, the news, a search):
#: "an investigation is now underway" read from a headline is not a claim.
_READER_TOOLS = re.compile(
    r"brief|news|search|read|summar|recall|query|weather|lookup|fetch|wiki|knowledge|explain",
    re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(?:i|i've|i'll|i'm|me|my|your|you)\b", re.IGNORECASE)

#: Not a claim at all: negated, conditional, offered or asked.
_NOT_A_CLAIM = re.compile(
    r"\b(?:not|never|unable|failed|couldn't|could not|cannot|can't|haven't|have not|hasn't|"
    r"has not|didn't|did not|wasn't|was not|won't|isn't|aren't|if|whether|would you|shall i|"
    r"should i|want me to|do you want|would you like|let me know|once|try|trying|attempt|"
    r"going to|about to|unless|hopefully|might|may)\b", re.IGNORECASE)

_DONE_PREFIX = r"(?:i(?:'ve| have)\s+(?:just\s+|now\s+|already\s+|successfully\s+)?)"


@dataclass(frozen=True)
class ClaimDomain:
    """One kind of action ORION can claim, and the tools that could do it.

    A claim needs an action verb AND a noun of the domain in the same
    sentence (or one of the few phrases that carry both, like "no longer
    listed"), which keeps "Done." and "I've sent it" alone - too vague to
    hold anyone to - out of it."""

    name: str
    tools: re.Pattern[str]
    verb: re.Pattern[str] | None = None
    noun: re.Pattern[str] | None = None
    phrase: re.Pattern[str] | None = None

    def claimed_in(self, sentence: str) -> bool:
        if self.phrase is not None and self.phrase.search(sentence):
            return True
        return bool(self.verb is not None and self.noun is not None
                    and self.verb.search(sentence) and self.noun.search(sentence))

    def backed_by(self, receipt: ToolReceipt) -> bool:
        """Whether *receipt* is the kind of call that could have done this."""
        if self.name == "research":
            return receipt.research or bool(self.tools.search(receipt.name))
        if receipt.reads_only:
            return False
        return bool(self.tools.search(receipt.name) or self.tools.search(receipt.action)
                    or _GENERIC_TOOLS.search(receipt.name))


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


CLAIM_DOMAINS: tuple[ClaimDomain, ...] = (
    ClaimDomain(
        "research",
        tools=_rx(r"research|investig|dossier|analy|study|deep|browse|search|job"),
        phrase=_rx(r"\b(?:research|investigation|study|analysis|dossier)\b[^.!?]{0,300}?\b"
                   r"(?:is|has)\s+(?:now\s+|already\s+)?(?:underway|under way|in progress|started|"
                   r"begun|running)\b|\b" + _DONE_PREFIX + r"(?:started|begun|launched|kicked off)\s+"
                   r"(?:the\s+|an?\s+|your\s+)?(?:research|investigation|study|analysis|dossier)\b")),
    ClaimDomain(
        "message",
        tools=_rx(r"send|messag|mail|discord|telegram|whatsapp|sms|notif|push|social|post|tweet|"
                  r"slack|teams|ifttt|webhook|phone|call|contact"),
        phrase=_rx(r"\b" + _DONE_PREFIX + r"(?:texted|messaged|emailed|e-mailed)\b"),
        verb=_rx(r"\b(?:" + _DONE_PREFIX + r"(?:sent|posted|forwarded|replied)|(?:has|have)\s+been\s+"
                 r"(?:sent|posted|delivered)|(?:is|was|were)\s+(?:now\s+)?(?:sent|posted|delivered))\b"),
        noun=_rx(r"\b(?:message|email|e-mail|text|dm|reply|whatsapp|discord|telegram|sms|"
                 r"notification|tweet|post|slack)s?\b")),
    ClaimDomain(
        "open",
        tools=_rx(r"open|launch|app|browser|url|web|window|desktop|file|folder|explorer|play|media|"
                  r"spotify|youtube|program|navigate|screen|click|computer|shell|terminal|run|start|focus"),
        verb=_rx(r"\b(?:" + _DONE_PREFIX + r"(?:opened|launched|loaded|pulled up|brought up)|"
                 r"(?:is|are)\s+now\s+open|(?:has|have)\s+been\s+(?:opened|launched))\b"),
        noun=_rx(r"\b(?:app|application|program|browser|tab|window|website|site|page|link|url|file|"
                 r"folder|document|notepad|calculator|spotify|youtube|chrome|edge|firefox|explorer|"
                 r"settings|terminal|steam|word|excel|outlook|editor)s?\b")),
    ClaimDomain(
        "reminder",
        tools=_rx(r"remind|alarm|timer|schedul|calendar|event|task|todo|to_do|agenda|awareness|"
                  r"routine|cron|job|planner|diary|appoint|meeting|productivity"),
        phrase=_rx(r"\bi(?:'ll| will)\s+remind\s+you\b|\b(?:reminder|alarm|timer)\s+(?:is|has been)\s+"
                   r"(?:now\s+)?set\b"),
        verb=_rx(r"\b(?:" + _DONE_PREFIX + r"(?:set|created|scheduled|added|booked|put)|"
                 r"(?:is|are|has been|have been)\s+(?:now\s+)?(?:set|scheduled|booked|added|created))\b"),
        noun=_rx(r"\b(?:reminder|alarm|timer|appointment|event|meeting|calendar|diary|task|to-do|"
                 r"todo|agenda)s?\b")),
    ClaimDomain(
        "removal",
        tools=_rx(r"task|todo|to_do|list|clear|remov|delet|remind|calendar|event|file|note|alarm|"
                  r"timer|awareness|agenda|complete|cancel|forget|memory|archive|productivity|clean"),
        phrase=_rx(r"\bno longer (?:listed|on (?:the|your) (?:list|calendar|schedule|agenda))\b"),
        verb=_rx(r"\b(?:" + _DONE_PREFIX + r"(?:removed|deleted|cleared|cancelled|canceled|ticked off|"
                 r"crossed off|archived)|(?:has|have)\s+been\s+(?:removed|deleted|cleared|cancelled|"
                 r"canceled|archived)|(?:is|are)\s+now\s+(?:gone|cleared|removed|deleted))\b"),
        noun=_rx(r"\b(?:task|item|list|reminder|todo|to-do|event|file|entry|note|appointment|"
                 r"meeting|alarm|timer|history|memory)s?\b")),
    ClaimDomain(
        "save",
        tools=_rx(r"save|write|file|note|memory|remember|store|document|record|journal|clipboard|"
                  r"brain|knowledge|ingest|docs|word|excel|log|list|task|todo|copy|desktop|editor"),
        verb=_rx(r"\b(?:" + _DONE_PREFIX + r"(?:saved|stored|written|wrote|recorded|logged|filed|"
                 r"copied)|(?:has|have)\s+been\s+(?:saved|stored|written|recorded|logged|copied))\b"),
        noun=_rx(r"\b(?:file|note|notes|document|doc|memory|record|draft|journal|clipboard|"
                 r"folder|log|spreadsheet)\b")),
    ClaimDomain(
        "call",
        tools=_rx(r"call|phone|dial|telephon|twilio|sip|voip|contact"),
        phrase=_rx(r"\b" + _DONE_PREFIX + r"(?:phoned|rung|dialled|dialed)\b|\bi(?:'m| am)\s+(?:now\s+)?"
                   r"(?:calling|ringing|dialling|dialing)\s+\w")),
    ClaimDomain(
        "media",
        tools=_rx(r"media|music|play|spotify|youtube|volume|audio|sound|mute|song|track|player|"
                  r"pause|skip|radio|podcast|speaker"),
        phrase=_rx(r"\bnow playing\b"),
        verb=_rx(r"\b" + _DONE_PREFIX + r"(?:paused|stopped|skipped|resumed|played|muted|unmuted|"
                 r"turned (?:up|down)|started)\b"),
        noun=_rx(r"\b(?:music|song|track|video|playlist|volume|audio|sound|spotify|youtube|"
                 r"podcast|album|radio|playback)s?\b")),
)


def claimed_completions(reply: str, *, skip: tuple[str, ...] = ()) -> list[tuple[ClaimDomain, str]]:
    """(domain, sentence) for each completion ORION claimed in *reply*."""
    text = " ".join(str(reply or "").replace("’", "'").split())
    found: list[tuple[ClaimDomain, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence or sentence.endswith("?") or _NOT_A_CLAIM.search(sentence):
            continue
        for domain in CLAIM_DOMAINS:
            if domain.name in skip or any(d is domain for d, _ in found):
                continue
            if domain.claimed_in(sentence):
                found.append((domain, sentence))
    return found


@dataclass(frozen=True)
class ClaimFinding:
    """A claim with no successful receipt, and the note that corrects it."""

    domain: str
    claim: str
    reason: str

    def note(self) -> str:
        return (f"[CONTEXT UPDATE: in your last reply you said \"{self.claim[:160]}\", but "
                f"{self.reason}. It has NOT been done. Say so plainly to the user at the next "
                "natural moment and offer to do it properly; never claim an action without a "
                "successful tool result.]")


class ResearchCommitments:
    """Bounded request identities; observing queued calls precedes recovery.

    Also keeps each request's tool receipts, which audit_claims holds a
    finished reply's completion claims against."""

    #: Receipts of earlier requests younger than this may back a claim that
    #: answers "did you send it?" about work an earlier turn really did.
    RECENT_RECEIPT_S = 180.0

    def __init__(self) -> None:
        self._attempted: set[int] = set()
        self._recovered: set[int] = set()
        self._last_decided = 0
        self._dismissed = 0
        self._audited: set[int] = set()
        self._receipts: dict[int, list[ToolReceipt]] = {}

    def observe(self, request_id: int, name: str, args: Any) -> None:
        if request_id > 0 and is_research_start(name, args):
            self._attempted.add(request_id)
            if len(self._attempted) > 128:
                self._attempted = set(sorted(self._attempted)[-64:])

    def begin(self, request_id: int, name: str, args: Any) -> ToolReceipt:
        """Record a tool call that has STARTED for *request_id*."""
        tool, params = effective_tool(name, args)
        action = params.get("action") or params.get("operation") or "" if isinstance(params, dict) else ""
        # A tool called without an action runs its default - file_controller
        # LISTS - and an empty action counted as proof that a file was opened.
        action = action or _DEFAULT_ACTIONS.get(tool, "")
        receipt = ToolReceipt(tool, action=str(action), research=is_research_start(name, args))
        if request_id > 0:
            self._receipts.setdefault(request_id, []).append(receipt)
            if len(self._receipts) > 64:
                for old in sorted(self._receipts)[:-32]:
                    del self._receipts[old]
        return receipt

    @staticmethod
    def finish(receipt: ToolReceipt, *, ok: bool, result: str = "") -> None:
        receipt.ok = bool(ok)
        receipt.result = " ".join(str(result or "").split())[:160]

    def receipts(self, request_id: int) -> list[ToolReceipt]:
        return list(self._receipts.get(request_id, ()))

    def audit_claims(self, request_id: int, reply: str, *,
                     skip: tuple[str, ...] = ()) -> list[ClaimFinding]:
        """The claims in *reply* that no successful (or still running) tool
        of the right kind backs. Once per request; never for a cancelled one.

        Receipts are THIS request's. A recent earlier request's success
        backs a claim only when this turn has no failure of that kind - so
        "yes, I've sent it" about the last turn's message stands, while
        "it's no longer listed" right after the removal was refused does not."""
        if request_id <= 0 or request_id <= self._dismissed or request_id in self._audited:
            return []
        self._audited.add(request_id)
        if len(self._audited) > 128:
            self._audited = set(sorted(self._audited)[-64:])
        now = time.monotonic()
        mine = self._receipts.get(request_id, [])
        recent = [r for rid, rows in self._receipts.items()
                  if rid != request_id and rid >= request_id - 2
                  for r in rows if now - r.at <= self.RECENT_RECEIPT_S]
        relayed = any(r.ok and _READER_TOOLS.search(r.name) for r in mine)
        findings: list[ClaimFinding] = []
        for domain, sentence in claimed_completions(reply, skip=skip):
            if relayed and not _FIRST_PERSON.search(sentence):
                continue                     # reported content, not ORION's own claim
            relevant = [r for r in mine if domain.backed_by(r)]
            if any(r.ok is not False for r in relevant):
                continue                     # it worked, or it is still running
            if not relevant and any(r.ok and domain.backed_by(r) for r in recent):
                continue
            if relevant:
                failed = relevant[-1]
                reason = f"{failed.name} did not succeed" + (f" ({failed.result})" if failed.result else "")
            elif mine:
                reason = (f"no {domain.name} action was taken this turn (only "
                          f"{', '.join(sorted({r.name for r in mine}))[:80]} ran)")
            else:
                reason = "no tool was used this turn"
            findings.append(ClaimFinding(domain.name, sentence, reason))
        return findings

    def dismiss(self, request_id: int) -> None:
        self._last_decided = max(self._last_decided, request_id)
        self._dismissed = max(self._dismissed, request_id)

    def recovered(self, request_id: int) -> bool:
        return request_id in self._recovered

    def decide(self, request_id: int, user: str, reply: str, *, briefing: str = "") -> ResearchRecovery | None:
        if request_id <= self._last_decided or request_id in self._attempted:
            return None
        authorised, requested_topic = user_research_request(user)
        promised, claimed_topic = promised_research(reply)
        if not authorised or not promised:
            return None
        self._last_decided = request_id
        topic = requested_topic or _clean_topic(briefing) or claimed_topic
        if not topic:
            return ResearchRecovery(correction=(
                "I said research was underway, but no investigation had started. "
                "I need the specific topics before I can begin."))
        self._recovered.add(request_id)
        if len(self._recovered) > 128:
            self._recovered = set(sorted(self._recovered)[-64:])
        return ResearchRecovery(topic=topic)
