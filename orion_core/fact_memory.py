"""
Learning about the user from what they say — without being told to remember.

Until now a personal fact reached ORION's long-term memory only if, in the
middle of answering, the model decided to call the memory tool. Mention in
passing that your sister Emma has moved to Leeds, that you cannot eat gluten,
that you start the new job on the 3rd — and unless the model happened to think
"I should save that", it lived only in the raw conversation log, where recall
had to dig it out again by luck.

This listens to every user turn and keeps what lasts:

    gate        a cheap local test (regular expressions, microseconds) for the
                shape of a durable first-person fact — "my sister is…",
                "I'm allergic to…", "I work at…", "call me…". Requests,
                questions and passing states ("I'm tired") do not pass. Only
                what passes goes further, so the model is not consulted on
                every "open Chrome".
    extract     the model rewrites what passed as short third-person facts
                ("The user's sister, Emma, lives in Leeds."), in the
                background, never on a reply's path. Offline, the sentence
                itself is kept — recall by meaning reads it just as well.
    guard       nothing that looks like a credential, key or card number
                (spillage_guard) or that NAMES one — password, PIN, sort
                code, recovery phrase — is ever stored, whatever the model
                returns.
    de-dup      a fact that MEANS what an existing one says (cosine >= 0.90)
                updates that memory instead of adding a near-copy beside it;
                one saying it in almost the same words (>= 0.97) is skipped.
    change      facts change. The model is shown today's date (so "next
                Friday" is stored as a date, not a phrase that goes stale) and
                the stored memories nearest to what was said, and marks the
                one a new fact REPLACES — "I've moved to Manchester" updates
                "lives in Northgate" in place (logged "MEMORY: updated — …")
                instead of leaving the two side by side, contradicting.

Measured on utterances written AFTER the gate was tuned: 14/15 lasting facts
passed (the miss, "I'm lactose intolerant", then fixed) and 1/15 requests did
("I prefer the second option" — one wasted background call, nothing stored).

Every learned fact is announced on the log ("MEMORY: learned — …"), so
nothing is remembered silently. ORION_AUTO_REMEMBER=0 turns it off.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from .utils import first_line

#: Durable first-person statements. Each alternative needs a relationship or
#: attribute AND a verb that states it, so "open my email" or "check my
#: calendar" (a possessive in a request) does not pass.
_GATE = re.compile(
    r"""(?ix)
    \bmy\s+(?:[a-z]+\s+){0,2}?
        (?:name|birthday|partner|girlfriend|boyfriend|wife|husband|fianc[ée]e?|
           mum|mom|mother|dad|father|parents?|sister|brother|son|daughter|kids?|
           children|grandm\w+|grandad|grandfather|nan|aunt|uncle|cousin|
           dog|cat|pet|boss|manager|supervisor|tutor|lecturer|best\s+friend|
           friend|flatmate|housemate|landlord|neighbour|doctor|dentist|gp|
           flat|house|address|postcode|job|role|course|degree|module|
           university|uni|college|school|team|car|bike|
           favourite\s+\w+|favorite\s+\w+|allerg\w*|diet|blood\s+type|
           shift|rent|salary|deadline|exam|interview)
        (?:\s+[a-z]+){0,2}?                 # a name between: "my sister Emma has"
        (?:'s|\s+is|\s+was|\s+are|\s+were|\s+has|\s+have|\s+lives|\s+works|
           \s+called|\s+named|\s+just|\s+starts?|\s+finish\w*|\s+moved|\s+got)\b
    |\bi(?:'m|\s+am)\s+(?:(?:[a-z]+\s+)?(?:allergic|intolerant)|vegan|vegetarian|pescatarian|
        coeliac|celiac|diabetic|asthmatic|colou?r[- ]blind|left[- ]handed|
        pregnant|married|engaged|single|
        (?:a|an)\s+(?!bit\b|little\b|lot\b)\w+|from\s+\w+|based\s+in|
        studying|training\s+(?:as|to)|learning\s+\w+|working\s+(?:at|as|on|for)|
        moving\s+to|starting\s+(?:at|a|my|work|uni))
    |\bi\s+(?:live|work|study|was\s+born|grew\s+up|support|
        (?:hate|love|can'?t\s+stand|prefer)(?!\s+(?:this|that|it|these|those|you)\b)|
        don'?t\s+(?:eat|drink|like)|never\s+(?:eat|drink)|
        drive|own|train\s+at|go\s+to|volunteer|play|speak|
        have\s+(?:a|an|two|three)\s+(?!question\b|problem\b|idea\b|feeling\b)\w+)\b
    |\bi(?:'ve|\s+have)\s+(?:just\s+|now\s+|finally\s+)?(?:moved|started|finished|
        left|changed(?!\s+my\s+mind)|joined|quit|bought|sold|passed|failed|graduated|
        got\s+(?:a\s+new|engaged|married|promoted|divorced)|
        been\s+(?:diagnosed|promoted|accepted|offered))\b
    |\b(?:call\s+me|my\s+name'?s)\b
    |\bremember\s+(?:that|this)\b
    """)

#: Sentences that pass the gate but state nothing lasting.
_NOT_A_FACT = re.compile(
    r"(?i)^\s*(?:can|could|would|will|should|do|does|did|is|are|what|when|where|"
    r"who|why|how|which)\b|\?\s*$|\bif\s+i\b|\bwould\s+i\b|\bpretend\b|\bimagine\b|"
    r"\bi(?:'m|\s+am)\s+(?:tired|bored|hungry|busy|back|here|done|off|going\s+to\s+bed)\b")

#: At most one model extraction this often; the rest wait in the queue.
MIN_INTERVAL_S = 8.0
QUEUE_LIMIT = 20
#: Same meaning as an existing memory: update it rather than add a copy.
DUPLICATE_COSINE = 0.90
#: ...and this close is the same memory in other words: nothing to learn.
SAME_COSINE = 0.97
#: Categories a learned fact may be written to, and where checks for copies.
CATEGORY = "personal"
_DEDUP_CATEGORIES = ("personal", "long_term", "identity", "preferences", "relationship")

_EXTRACT_INSTRUCTION = (
    "You extract durable personal facts from one message a user said to their "
    "assistant. Keep only what is worth remembering for weeks or months: "
    "identity, relationships and names, health and dietary needs, preferences, "
    "routines, work and study, possessions, important dates, places. Ignore "
    "requests, questions, passing states (tired, busy), opinions about the task "
    "at hand, and anything hypothetical. Never output passwords, card numbers, "
    "codes or other secrets. Write each fact as ONE short sentence in the third "
    "person beginning 'The user' (e.g. 'The user's sister, Emma, lives in "
    "Leeds.'). Turn relative dates ('next Friday', 'in two weeks') into "
    "calendar dates using today's date, given with the message. If a fact "
    "UPDATES or CONTRADICTS one of the existing memories listed with the "
    "message (the user moved, changed job, a date changed), set \"replaces\" "
    "to that memory's key; otherwise null. Reply with JSON only: {\"facts\": "
    "[{\"key\": \"short_snake_case\", \"fact\": \"...\", \"replaces\": null}]} "
    "— or {\"facts\": []} when nothing lasting was said."
)


def enabled() -> bool:
    return os.getenv("ORION_AUTO_REMEMBER", "1").strip().lower() not in {
        "0", "false", "no", "off"}


def looks_like_a_fact(text: str) -> bool:
    """The local gate: a durable first-person statement, not a request."""
    text = str(text or "").strip()
    words = len(text.split())
    if words < 3 or words > 80:
        return False
    if _NOT_A_FACT.search(text):
        return False
    return bool(_GATE.search(text))


def _slug(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return (slug or "fact")[:limit]


#: Words that mean a credential is being stated, whatever its shape. The
#: spillage guard recognises secrets by FORM (key prefixes, card numbers);
#: "my password is hunter2" has no form to recognise, only these words.
_SECRET_WORDS = re.compile(
    r"(?i)\b(?:password|passcode|pass\s*word|pin(?:\s*(?:code|number))?|"
    r"security\s+(?:code|question|answer)|cvv|cvc|sort\s+code|"
    r"account\s+number|card\s+number|recovery\s+(?:code|phrase)|seed\s+phrase|"
    r"2fa|otp|one[- ]time\s+code|api\s+key|secret\s+key|private\s+key|token)\b")


def _contains_secret(text: str) -> bool:
    if _SECRET_WORDS.search(str(text or "")):
        return True
    try:
        from .spillage_guard import SpillageGuard

        return bool(SpillageGuard().scan(text).findings)
    except Exception:
        return False


def parse_facts(raw: str) -> list[tuple[str, str]]:
    """(key, fact) pairs from the model's JSON, tolerant of fences and prose."""
    return [(key, fact) for key, fact, _replaces in parse_fact_updates(raw)]


def parse_fact_updates(raw: str) -> list[tuple[str, str, str]]:
    """(key, fact, replaces) triples; *replaces* is "" when the fact is new."""
    text = str(raw or "")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:
        return []
    out: list[tuple[str, str, str]] = []
    for item in (data.get("facts") or []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        fact = " ".join(str(item.get("fact") or "").split())
        if not (8 <= len(fact) <= 300):
            continue
        replaces = str(item.get("replaces") or "").strip()
        if replaces.lower() in {"null", "none"}:
            replaces = ""
        out.append((_slug(item.get("key") or fact), fact, replaces))
    return out[:5]


class FactHarvester:
    """Listens to user turns; keeps the lasting facts in them."""

    def __init__(self, memory: Any, router: Any = None, bus: Any = None) -> None:
        self.memory = memory
        self.router = router
        self.bus = bus
        self._queue: asyncio.Queue[str] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._last_call = 0.0
        self.learned = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Begin consuming on *loop* (the running one by default)."""
        if self._task is not None or not enabled():
            return
        self._loop = loop or asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=QUEUE_LIMIT)
        self._task = self._loop.create_task(self._run(), name="orion-fact-harvester")

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    # ── intake (any thread, never blocks) ───────────────────────────────────

    def observe(self, text: str) -> bool:
        """Queue *text* if it looks like a lasting fact. True when queued."""
        if self._queue is None or self._loop is None or not looks_like_a_fact(text):
            return False
        cleaned = str(text).strip()[:600]

        def _put() -> None:
            try:
                self._queue.put_nowait(cleaned)
            except asyncio.QueueFull:
                pass            # a burst of facts: keep the earliest, drop the rest

        try:
            self._loop.call_soon_threadsafe(_put)
        except RuntimeError:
            return False        # the loop has closed (shutdown)
        return True

    # ── the work ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            text = await self._queue.get()
            wait = MIN_INTERVAL_S - (time.monotonic() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                await self.learn(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log(f"MEMORY: could not learn from that - {first_line(exc, 100)}")

    async def learn(self, text: str) -> list[str]:
        """Extract, guard, de-duplicate and store. Returns the facts kept."""
        if _contains_secret(text):
            return []
        related = await asyncio.to_thread(self._related, text)
        facts = await self._extract(text, related)
        offered = {key: category for key, category, _value in related}
        kept: list[str] = []
        for key, fact, replaces in facts:
            if _contains_secret(fact):
                continue
            if replaces and replaces in offered:
                # The model says this supersedes a memory it was shown ("I've
                # moved to Manchester" over "lives in Northgate"): update that
                # one in place, so the two never sit side by side disagreeing.
                category, key, known = offered[replaces], replaces, False
            else:
                category, key, known = await asyncio.to_thread(self._home_for, key, fact)
            if known:
                continue        # already remembered, in so many words
            try:
                self.memory.save(category, key, fact, silent=True)
            except Exception:
                continue
            kept.append(fact)
            self.learned += 1
            self._log(f"MEMORY: {'updated' if replaces in offered else 'learned'} — {fact}")
        return kept

    #: Existing memories closer than this to what was said are shown to the
    #: model, so it can say which one a new fact replaces. Low on purpose:
    #: "I've moved to Manchester" scored 0.622 against "lives in Northgate"
    #: and a job change 0.507 against the old job — no better than unrelated
    #: facts — so similarity only shortlists and the model judges.
    RELATED_FLOOR = 0.45
    RELATED_LIMIT = 6

    def _related(self, text: str) -> list[tuple[str, str, str]]:
        """(key, category, value) of stored personal memories near *text*."""
        matrix = getattr(self.memory, "matrix", self.memory)
        finder = getattr(matrix, "relevant", None)
        if finder is None:
            return []
        try:
            rows = finder(text, limit=self.RELATED_LIMIT, floor=self.RELATED_FLOOR)
        except Exception:
            return []
        return [(str(r["key_ref"]), str(r["category"]), str(r["value"]))
                for r in rows if r.get("category") in _DEDUP_CATEGORIES]

    async def _extract(self, text: str, related: list[tuple[str, str, str]] | None = None
                       ) -> list[tuple[str, str, str]]:
        router = self.router
        if router is not None:
            self._last_call = time.monotonic()
            from datetime import date

            context = f"Today is {date.today():%A %d %B %Y}.\n"
            if related:
                context += "Existing memories (key: fact):\n" + "\n".join(
                    f"- {key}: {value}" for key, _category, value in related) + "\n"
            try:
                _profile, raw = await router.generate_text(
                    f"{context}Message: {text}", instruction=_EXTRACT_INSTRUCTION,
                    task="memory", max_tokens=500)
                return parse_fact_updates(raw)
            except Exception:
                pass            # no provider: keep the sentence itself (below)
        # Offline, or the model failed: the user's own sentence is the fact.
        # Meaning-based recall reads "My sister Emma just moved to Leeds" as
        # well as any rewrite of it.
        return [(_slug(" ".join(text.split()[:8])), text, "")]

    def _home_for(self, key: str, fact: str) -> tuple[str, str, bool]:
        """Where to store *fact*: (category, key, already_known).

        An existing memory that MEANS the same (cosine >= 0.90) is updated in
        place rather than joined by a near-copy; one that says it in almost
        the same words (>= 0.97, or identically) is left alone entirely.
        """
        matrix = getattr(self.memory, "matrix", self.memory)
        finder = getattr(matrix, "relevant", None)
        if finder is not None:
            try:
                for row in finder(fact, limit=3, floor=DUPLICATE_COSINE):
                    if row.get("category") in _DEDUP_CATEGORIES:
                        same = (float(row.get("similarity", 0.0)) >= SAME_COSINE
                                or str(row.get("value", "")).strip() == fact.strip())
                        return row["category"], row["key_ref"], same
            except Exception:
                pass
        return CATEGORY, f"learned_{key}", False

    def _log(self, message: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(message)
            except Exception:
                pass


__all__ = ["DUPLICATE_COSINE", "FactHarvester", "enabled", "looks_like_a_fact", "parse_fact_updates",
           "parse_facts"]
