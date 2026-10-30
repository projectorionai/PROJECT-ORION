"""
LocalBrain — ORION's fully offline conversational engine.

When every cloud provider is exhausted (quota, rate limits, no network, no
keys) ORION must not become a mute tool-runner.  The LocalBrain keeps him
genuinely conversational and useful with **zero API calls**: it understands
intent with rules, answers daily questions from local facts (time, date,
system, weather already fetched), discusses neuroscience from the resident
knowledge base, recalls and stores memory, does arithmetic, and — crucially —
routes "help me do X" straight to the local tool dispatcher (open apps, search,
screenshots, files, workspace, desktop control, Notion/Outlook when present).

It speaks in ORION's calm, Alfred-Pennyworth register.  Everything it returns is
a natural spoken reply; task intents also trigger the corresponding tool and
report the real result.  If a local LLM (LM Studio / Ollama) *is* configured it
is used for open-ended chat, but nothing here depends on it.

This is the safety net that makes "he can still talk to me about daily things
and help with tasks" true even with the tokens gone.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from typing import Any, Optional

from .bus import OrionBus
from .data import ToolResult
from .identity import CREATOR_QUESTION_RE, creator_answer
from .knowledge import NeuroKnowledgeBase
from .remote_capability import RemoteToolGate
from .time_service import TIME
from .utils import first_line


# ── correction detection (Mark X.12 §2.3) ──────────────────────────────────────
# A casual "no, that's wrong" / "actually, it's X" should correct the last
# fact ORION gave, not be treated as a fresh turn. This pure classifier decides
# whether an utterance is such a correction and extracts the substantive
# replacement; the caller only acts on it when a correctable answer was just
# given, which is what keeps a stray "no" from ever firing.

_WRONGNESS = re.compile(
    r"\b(that'?s|that is|this is|you'?re|you are)\s+"
    r"(wrong|incorrect|not\s+right|not\s+correct|not\s+true|mistaken|false)\b",
    re.IGNORECASE,
)
_REPLACEMENT = re.compile(
    r"\b(?:actually|i\s+mean|no,?\s+it'?s|it'?s\s+actually|it\s+is\s+actually|"
    r"should\s+be|the\s+correct\s+\w+\s+is|no,?\s+that'?s\s+not)\b",
    re.IGNORECASE,
)
_LEAD_STRIP = (
    re.compile(r"^\s*(?:no|nope|nah)\b[,.!]?\s*", re.IGNORECASE),
    re.compile(r"^\s*that'?s\s+(?:wrong|incorrect|not\s+\w+)[,.!]?\s*", re.IGNORECASE),
    re.compile(r"^\s*actually\b[,.!]?\s*", re.IGNORECASE),
)


def detect_correction(text: str) -> Optional[str]:
    """Return the correction content if *text* disputes a prior answer, else None.

    Fires on an explicit wrongness assertion ("that's wrong", "you're not
    right") or a corrective replacement lead-in ("no, actually it's X",
    "the correct answer is Y"). Returns the substantive replacement when one is
    present, otherwise the disputing phrase itself so the record still flags the
    prior answer. A bare "no" / "no thanks" is deliberately *not* a correction.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    if not (_WRONGNESS.search(raw) or _REPLACEMENT.search(raw)):
        return None
    correction = raw
    for pattern in _LEAD_STRIP:
        correction = pattern.sub("", correction, count=1)
    correction = correction.strip()
    return correction or raw


def _request_body(raw: str) -> Optional[str]:
    """A direct request after polite address, or None for a quoted/negated mention.

    This only narrows deterministic rules. Unrecognised prose remains available
    to the conversational model; an action verb somewhere in prose is not
    permission to perform that action.
    """
    body = raw.strip()
    prefix = re.compile(
        r"^(?:(?:hello|hi|hey|thanks|thank you|cheers|orion|please)[,!: ]+"
        r"|(?:can|could|would|will)\s+you\s+(?:please\s+)?"
        r"|i(?:'d| would)?\s+like\s+(?:you\s+)?to\s+"
        r"|i\s+want\s+(?:you\s+)?to\s+"
        r"|help\s+me(?:\s+to)?\s+(?=(?:open|close|launch|start|find|save|read|set|add)\b))",
        re.IGNORECASE)
    for _ in range(6):
        match = prefix.match(body)
        if match is None:
            break
        remainder = body[match.end():].strip()
        if not remainder or remainder.lower().strip(" ,.!?") in {"orion", "there", "very much", "so much", "a lot", "again"}:
            break  # preserve a standalone greeting such as "hey Orion"
        body = remainder
    if not body or body.startswith(('"', "'", "`", "“", "‘", ">")):
        return None
    if re.match(
            r"^(?:do\s+not(?!\s+forget\b)|don['’]?t(?!\s+forget\b)|never|not|avoid|"
            r"refrain\s+from|without)\b", body, re.IGNORECASE):
        return None
    if re.match(
            r"^(?:explain|teach|understand|say|repeat|quote|translate)\b"
            r"|^(?:describe|show|tell)(?:\s+me)?\s+(?:how|why)\b"
            r"|^(?:list|show)(?:\s+me)?\s+(?:the\s+)?(?:steps|instructions|commands|examples)\b"
            r"|^how\s+(?:do|can|could|would|to)\b"
            r"|^what\s+(?:is|are)\s+(?:the\s+)?(?:command|instructions|meaning|example)\b"
            r"|^what\s+can\s+you\s+do\s+(?:to|if|when)\b"
            r"|^(?:i|we|he|she|they|someone|somebody)\s+(?:said|says|asked|told|wrote|mentioned)\b"
            r"|^(?:my|our|the)\s+(?:friend|colleague|teacher|assistant|user|owner)\s+(?:said|says|asked|told|wrote|mentioned)\b"
            r"|^the\s+(?:example|instruction|command|message|quote)\s+(?:says|was|is)\b",
            body, re.IGNORECASE):
        return None
    return body


class LocalBrain:
    """Rule-based, no-API conversational fallback with local tool routing."""

    def __init__(
        self,
        bus: OrionBus,
        memory: Any,
        dispatcher: Any,
        knowledge: NeuroKnowledgeBase,
        router: Any | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.dispatcher = dispatcher
        self.knowledge = knowledge
        self.router = router
        self.telemetry = telemetry
        # §2.3: the last fact ORION gave from KNOWLEDGE/memory, so a follow-up
        # "no, that's wrong" can correct *that* rather than start a new turn.
        self._last_fact_topic = ""
        self._last_fact_answer = ""

    # ── public entry point ────────────────────────────────────────────────────

    async def respond(self, text: str, *, allow_llm: bool = True) -> Optional[str]:
        """Return a spoken reply; may perform a local tool action en route."""
        raw = str(text or "").strip()
        if not raw:
            return "I'm here." if allow_llm else None
        lowered = raw.lower().strip(" .!?")
        def count_turn() -> None:
            if self.telemetry is None:
                return
            try:
                self.telemetry.metrics.incr("local_brain.turns")
            except Exception:
                pass

        if allow_llm:
            count_turn()

        def known(reply: str) -> str:
            if not allow_llm:
                count_turn()
            return reply

        body = _request_body(raw)
        if body is None:
            if not allow_llm:
                return None
            reply = await self._try_local_llm(raw)
            return reply if reply is not None else self._chit_chat(lowered, raw)
        raw = body
        lowered = raw.lower().strip(" .!?")

        # §2.3: a casual "no, that's wrong" / "actually it's X" corrects the last
        # fact ORION gave (from KNOWLEDGE/memory) instead of being a fresh turn.
        corrected = await self._maybe_apply_correction(raw)
        if corrected is not None:
            return known(corrected)

        # Ordered intent resolution — first match wins.  Answers sourced from
        # memory/knowledge (recall, neuroscience) are remembered as correctable.
        _correctable = (self._intent_recall, self._intent_neuro)
        for handler in (
            # First: "hey, who made you?" must not stop at the greeting.
            self._intent_creator,
            self._intent_greeting,
            self._intent_wellbeing,
            self._intent_thanks,
            self._intent_farewell,
            self._intent_identity,
            self._intent_capabilities,
            self._intent_time,
            self._intent_date,
            self._intent_weather,
            self._intent_system,
            self._intent_math,
            self._intent_remember,
            self._intent_recall,
            self._intent_neuro,
        ):
            reply = handler(lowered, raw)
            if reply is not None:
                if handler in _correctable:
                    self._last_fact_topic = raw
                    self._last_fact_answer = reply
                return known(reply)

        # Task intents run a real local tool and report the outcome.
        task_reply = await self._intent_task(lowered, raw)
        if task_reply is not None:
            return known(task_reply)

        if not allow_llm:
            return None

        # A local LLM, if configured, handles open-ended chat.
        llm_reply = await self._try_local_llm(raw)
        if llm_reply is not None:
            return llm_reply

        return self._chit_chat(lowered, raw)

    async def try_known_response(self, text: str) -> Optional[str]:
        """Rules/local tools only; an unknown request stays available to a model."""
        return await self.respond(text, allow_llm=False)

    async def _maybe_apply_correction(self, raw: str) -> Optional[str]:
        """If ORION just gave a fact and the user disputes it, record an
        authoritative correction via the LearningService and confirm — instead
        of treating "no, that's wrong" as an ordinary turn. Returns the spoken
        acknowledgement, or None when this isn't a correction of a live fact."""
        if not self._last_fact_answer:
            return None
        if not re.match(
                r"^(?:(?:no|nope|nah)\b[, ]*)?(?:actually\b|that'?s\b|that is\b|this is\b|"
                r"you'?re\b|you are\b|i mean\b|it'?s actually\b|it is actually\b|"
                r"should be\b|the correct\b|i (?:think|believe) (?:you|that|this)\b)",
                raw, re.IGNORECASE):
            return None
        correction = detect_correction(raw)
        if not correction:
            return None
        topic, self._last_fact_topic = self._last_fact_topic, ""
        prior, self._last_fact_answer = self._last_fact_answer, ""
        learning = getattr(self.dispatcher, "learning", None) if self.dispatcher else None
        if learning is None:
            return None
        # Give the correction the disputed answer as context so the stored
        # record is meaningful even when the user only said "that's wrong".
        detail = f"{correction} (correcting the earlier answer: {first_line(prior, 160)})"
        try:
            result = await learning.correct(topic, detail)
        except Exception as exc:
            self.bus.log.emit(f"LOCAL_BRAIN: correction failed - {first_line(exc, 100)}")
            return None
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("local_brain.corrections")
            except Exception:
                pass
        return result.text

    async def try_task(self, text: str) -> Optional[str]:
        """Run *only* the actionable-command router (open/search/read/mail/…)
        and report the outcome, or return None if the message isn't a task.

        The remote uplink uses this so a spoken command actually *does* the
        thing (through whatever dispatcher this brain was built with — gated,
        for remote origins), while non-command chat still goes to the model.
        """
        raw = str(text or "").strip()
        if not raw:
            return None
        return await self._intent_task(raw.lower().strip(" .!?"), raw)

    # ── daily-conversation intents ────────────────────────────────────────────

    def _intent_greeting(self, low: str, raw: str) -> Optional[str]:
        if re.fullmatch(
                r"(?:(?:hello|hi|hey|greetings)(?:[, ]+(?:there|orion))?"
                r"|good (?:morning|afternoon|evening)(?:[, ]+orion)?)", low.strip(" ,.!?")):
            # Shared clock: this used an 18:00 evening boundary while the
            # briefing and the live worker used 17:00, so the offline brain
            # could greet you "good afternoon" in the same minute the HUD
            # called it evening.
            period = TIME.greeting_period()
            return random.choice([
                f"Good {period}. How may I help?",
                f"Good {period}. I'm right here.",
                "At your service.",
            ])
        return None

    def _intent_wellbeing(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bhow are you\b|\bhow's it going\b|\byou (ok|okay|doing)\b", low):
            return random.choice([
                "Running smoothly — all systems nominal. And yourself?",
                "In fine form, thank you for asking. How are you keeping?",
                "Quite well. Ready when you are.",
            ])
        if re.search(r"\bi'?m (tired|exhausted|stressed|sad|down|unwell|ill)\b", low):
            return ("I'm sorry to hear that. Do take a moment — I can hold "
                    "your tasks, dim the pace, or simply keep you company.")
        return None

    def _intent_thanks(self, low: str, raw: str) -> Optional[str]:
        if re.fullmatch(r"(?:thank you|thanks|cheers|(?:i )?appreciate (?:it|that))(?: very much| so much| a lot| again|[, ]+orion)?", low):
            return random.choice(["A pleasure.", "Always.", "Think nothing of it."])
        return None

    def _intent_farewell(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\b(goodbye|good night|see you|that'?s all|bye)\b", low):
            return random.choice([
                "Very good. I'll be here if you need me.",
                "Good night.",
                "Standing by.",
            ])
        return None

    def _intent_creator(self, low: str, raw: str) -> Optional[str]:
        if CREATOR_QUESTION_RE.search(low):
            return creator_answer()
        return None

    def _intent_identity(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bwho are you\b|\bwhat are you\b|\byour name\b", low):
            return ("I am ORION — your personal operating system and aide. At the "
                    "moment I'm running on my local faculties, without the cloud, "
                    "yet still very much at your disposal.")
        return None

    def _intent_capabilities(self, low: str, raw: str) -> Optional[str]:
        if (re.fullmatch(
                r"(?:(?:hello|hi|hey)[, ]+)?(?:what can you do(?: for me)?"
                r"|(?:(?:what are|tell me about|describe|show me) )?your (?:capabilities|features))",
                low.strip(" ,.!?"))
                or low in {"help", "help me", "help me with"}):
            return ("Even offline, I can open and manage applications, search "
                    "your files, capture and read the screen, take notes and recall "
                    "them, manage your workspace, run desktop actions, discuss "
                    "neuroscience and engineering, do quick calculations, and keep "
                    "you company. Simply ask.")
        return None

    def _intent_time(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bwhat('?s| is) the time\b|\bthe time\b|\bwhat time\b", low):
            stamp = TIME.now().strftime("%I:%M %p").lstrip("0")
            return f"It's {stamp}."
        return None

    def _intent_date(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bwhat('?s| is) (the |today'?s )?date\b|\bwhat day\b|\btoday'?s date\b", low):
            return f"Today is {TIME.now().strftime('%A, %d %B %Y')}."
        return None

    def _intent_weather(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bweather\b|\bforecast\b|\btemperature outside\b", low):
            cached = self._recall_value("weather")
            if cached:
                return f"From the last reading: {cached}."
            return ("I can't reach the weather service without the network, "
                    "but the moment we're back online I'll refresh it for you.")
        return None

    def _intent_system(self, low: str, raw: str) -> Optional[str]:
        if (re.fullmatch(
                r"(?:(?:what(?:'s| is| are)|show|check|read|tell me|report)\s+)?"
                r"(?:(?:the|your|my)\s+)?(?:current\s+)?(?:cpu|memory|ram|system|resources)"
                r"(?:\s+(?:usage|status|health|load|utilisation|utilization))?", low)
                or re.fullmatch(r"how(?:'s| is) (?:the|my) (?:pc|computer|machine)(?: doing)?", low)):
            try:
                import psutil
                # interval=None samples against the PREVIOUS call instead of
                # sleeping. interval=0.2 blocked the qasync loop — the thread
                # that draws the face and services audio — for 200 ms every time
                # the user asked how the machine was doing. The first reading
                # after a cold start can be 0.0, so fall back to a one-shot
                # sample in that case rather than reporting a false zero.
                cpu = psutil.cpu_percent(interval=None)
                if not cpu:
                    cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                return f"The machine is healthy — CPU around {cpu:.0f} percent, memory at {ram:.0f} percent."
            except Exception:
                return "I couldn't read the system counters just now."
        return None

    # ── arithmetic ────────────────────────────────────────────────────────────

    def _intent_math(self, low: str, raw: str) -> Optional[str]:
        """Sums and unit conversions, exactly (calculator.py).

        The old version deleted every character that was not a digit or an
        operator, so "what's 12 stone in kilos" answered "That's 12." and any
        "what is ..." with a number in it got a confident wrong answer; and
        its powers were unbounded ("9^9^9" would hang this brain).
        """
        asked = re.search(r"(?:what('?s| is)|calculate|compute|work out|convert|how many)\s+(.+)", low)
        bare = re.fullmatch(r"[0-9\.\s\+\-\*/x%\^\(\),]+", low)
        if not (asked or bare) or not re.search(r"\d", low) and "how many" not in low:
            return None
        from .calculator import calculate, convert
        try:
            converted = convert(raw)
        except Exception:
            converted = None
        if converted:
            return converted
        if not re.search(r"\d", low):
            return None
        result = calculate(raw)
        if not result.ok:
            return None
        return f"That's {result.text.rsplit(' = ', 1)[-1]}."

    # ── memory ────────────────────────────────────────────────────────────────

    def _intent_remember(self, low: str, raw: str) -> Optional[str]:
        body = _request_body(raw)
        if body is None:
            return None
        m = re.match(r"^(?:remember|note|make a note|don'?t forget|do not forget)\s+(?:that\s+)?(.+)", body, re.IGNORECASE)
        if not m:
            return None
        fact = m.group(1).strip(" .")
        if not fact:
            return None
        key = re.sub(r"[^a-z0-9]+", "_", fact.lower())[:40] or "note"
        try:
            self.memory.remember("long_term", key, fact)
            return f"Noted: {fact}."
        except Exception as exc:
            return f"I tried to note that but hit a snag: {first_line(exc, 60)}."

    def _intent_recall(self, low: str, raw: str) -> Optional[str]:
        # Personal-fact questions ("what is my brand called", "what's my goal")
        # must be answered deterministically from memory, never invented by the
        # local model.
        personal = re.search(
            r"what(?:'?s| is| are| was| were)?\s+my\s+([a-z0-9 ]+?)(?:\s+(?:called|named))?\s*\??$",
            low,
        )
        if personal:
            topic = personal.group(1).strip()
            value = self._recall_value(f"my {topic}") or self._recall_value(topic)
            if value:
                return f"Your {topic}: {value}."
            return f"I don't have your {topic} on record yet. Tell me and I'll remember it."
        m = re.search(r"(?:what do you (?:know|remember) about|recall|remind me (?:about|of))\s+(.+)", low)
        if not m:
            return None
        query = m.group(1).strip(" ?.")
        try:
            rows = self.memory.query(query, limit=3)
        except Exception:
            rows = []
        if rows:
            return "Here's what I have: " + "; ".join(r["value"] for r in rows[:3]) + "."
        # Fall through to the knowledge base for domain topics (if present).
        neuro = self.knowledge.answer(query) if self.knowledge is not None else None
        if neuro:
            return neuro
        return f"I don't have anything stored about {query}."

    def _recall_value(self, query: str) -> str:
        try:
            rows = self.memory.query(query, limit=1)
            return rows[0]["value"] if rows else ""
        except Exception:
            return ""

    # ── neuroscience ──────────────────────────────────────────────────────────

    def _intent_neuro(self, low: str, raw: str) -> Optional[str]:
        if self.knowledge is None or not self.knowledge.is_neuro_query(low):
            return None
        answer = self.knowledge.answer(raw)
        if answer:
            return answer
        return ("That's within my field, though I don't have a specific note "
                "on it offline. Ask me about neurons, synapses, brain–computer "
                "interfaces, the Utah array, Neuralink, spike sorting or decoding, "
                "and I can go deep.")

    # ── task routing (real local tools) ───────────────────────────────────────

    async def _intent_task(self, low: str, raw: str) -> Optional[str]:
        body = _request_body(raw)
        if body is None:
            return None
        raw, low = body, body.lower().strip(" .!?")
        phone_reply = await self._intent_phone_task(low, raw)
        if phone_reply is not None:
            return phone_reply
        # Explicit electronics inspection remains useful offline: the camera
        # and local quality/OCR checks work without a cloud conversation model.
        if re.match(r"^(?:scan|inspect|analyse|analyze|examine|look at)\s+(?:(?:this|the|my|a|these)\s+)?(?:pcb|circuit\s+board|electronics|electronic\s+(?:board|device|components?))\b", low):
            result = await self._run_tool("vision_analyse", {"action": "pcb", "prompt": raw})
            if result.ok and result.evidence:
                report = result.evidence[0]
                return str(report.get("summary") or result.text)
            return self._spoken(result, None)
        # Reminders, timers and alarms. Offline there was no route to them at
        # all: "remind me in 20 minutes" got small talk, and "start a timer"
        # fell to the open-an-app rule below and opened the Clock app.
        if re.search(r"\b(?:remind me|reminders?|timers?|countdown|alarms?|wake me)\b", low):
            if (re.match(r"^(?:list|show|check|read|any)\b", low)
                    or re.match(r"^(?:what|which) (?:are|is) (?:my|the)\b", low)
                    or re.fullmatch(r"help me with (?:a list of )?(?:my )?(?:reminders?|timers?|alarms?)", low)):
                result = await self._run_tool("reminder", {"action": "list"})
            elif re.match(r"^(?:cancel|stop|clear|delete|remove)\s+(?:(?:all|the|my|active)\s+)*(?:reminders?|timers?|alarms?)\b", low):
                result = await self._run_tool("reminder", {"action": "cancel"})
            elif (re.match(r"^(?:remind me|wake me)\b", low)
                  or re.match(r"^(?:set|create|add|schedule|start)\s+(?:(?:a|an|the|my)\s+)?(?:(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:second|minute|hour)s?\s+)?(?:reminder|timer|alarm|countdown)\b", low)):
                result = await self._run_tool("reminder", {"action": "add", "phrase": raw})
            else:
                result = None
            # The tool's own sentence ("Timer set - 5 minutes (until 14:05)")
            # is the confirmation; a preamble would replace it with "Done."
            if result is not None:
                return self._spoken(result, None)
        # Open an application or website.
        m = re.match(r"^(?:open|launch|start|run)\s+(.+)", raw, re.IGNORECASE)
        if m:
            target = m.group(1).strip(" .")
            result = await self._run_tool("open_app", {"app_name": target})
            return self._spoken(result, f"Opening {target}.")
        # Close an application.
        m = re.match(r"^(?:close|quit|exit)\s+(.+)", raw, re.IGNORECASE)
        if m:
            target = m.group(1).strip(" .")
            result = await self._run_tool("close_app", {"app_name": target})
            return self._spoken(result, f"Closing {target}.")
        # Web search.
        m = re.match(r"^(?:search(?: for| the web for)?|google|look up)\s+(.+)", raw, re.IGNORECASE)
        if m:
            query = m.group(1).strip(" .")
            result = await self._run_tool("web_search", {"query": query})
            return self._spoken(result, f"Searching the web for {query}.")
        # Screen awareness.
        if re.search(r"\bwhat('?s| is) on (my |the )?screen\b|\blook at (my |the )?screen\b|\bread (my |the )?screen\b", low):
            result = await self._run_tool("vision_analyse", {"action": "describe"})
            return self._spoken(result, "Let me look.")
        # Find files.
        m = re.match(r"^find (?:my |the )?(?:file|files|folder|document)s?\s+(?:called |named |for )?(.+)", raw, re.IGNORECASE)
        if m:
            query = m.group(1).strip(" .")
            result = await self._run_tool("find_files", {"query": query})
            return self._spoken(result, f"Searching for {query}.")
        # Workspace snapshot / resume.
        if re.match(r"^(save|snapshot) (my )?workspace\b", low):
            result = await self._run_tool("workspace_control", {"action": "save"})
            return self._spoken(result, "Saving your workspace.")
        if re.search(r"\bwhere (did we|were we|was i) (leave off|left off|up to)\b|\bresume (my )?work\b", low):
            result = await self._run_tool("workspace_control", {"action": "resume_context"})
            return self._spoken(result, None)
        # Notion tasks.
        if re.search(r"\b(my )?(tasks|to.?do|task list)\b", low):
            result = await self._run_tool("notion_workspace", {"action": "list_tasks"})
            return self._spoken(result, None)
        # Email.
        if re.search(r"\b(my )?(email|emails|inbox|mail)\b", low):
            result = await self._run_tool("outlook_mail", {"action": "priority"})
            return self._spoken(result, None)
        return None

    async def _intent_phone_task(self, low: str, raw: str) -> Optional[str]:
        """Route phone-specific offline commands through the remote gate only."""
        gate = self.dispatcher
        if not isinstance(gate, RemoteToolGate):
            return None
        device_id = gate.request_device()
        if not device_id:
            return None

        phone_screen = re.search(
            r"\b(?:phone|mobile)\s+(?:screen|display|viewport)\b|"
            r"\b(?:screen|display)\s+(?:on|of)\s+(?:my\s+)?phone\b|"
            r"\b(?:what(?:'s| is)(?:\s+on)?|look at|read|describe|see)\s+(?:my|the)\s+screen\b",
            low)
        if phone_screen:
            action = ("latest_frame" if re.search(r"\b(?:screenshot|screen\s*shot|frame|capture image)\b", low)
                      else "capture_dom")
            result = await self._run_tool("phone_device", {"action": action})
            if action == "capture_dom" and result.ok and not result.media:
                return self._phone_dom_summary(result.text)
            return self._spoken(result, None)

        target = r"([+0-9][+0-9() -]{2,39}|[^\W\d_][^\d_]{1,79}?)"
        sms = re.match(rf"^(?:text|sms|send\s+(?:an?\s+)?sms\s+to)\s+{target}\s+(?:saying|that\s+says|message|:)\s+(.+)$", raw, re.IGNORECASE)
        if sms:
            result = await self._run_tool("phone_device", {
                "action": "send_sms", "number": sms.group(1).strip(), "body": sms.group(2).strip(),
            })
            return self._spoken(result, None)
        call = re.match(rf"^(?:call|dial)\s+{target}(?:\s+(?:on|from)\s+(?:my\s+)?phone)?\s*$", raw, re.IGNORECASE)
        if call:
            result = await self._run_tool("phone_device", {
                "action": "make_call", "number": call.group(1).strip(),
            })
            return self._spoken(result, None)

        if "phone" not in low and "mobile" not in low:
            return None

        app = re.match(r"^(?:open|launch|start)\s+(.+?)\s+(?:on\s+)?(?:my\s+)?(?:phone|mobile)\b", raw, re.IGNORECASE)
        if app:
            label = app.group(1).strip(" .").lower()
            aliases = {
                "settings": "com.android.settings",
                "chrome": "com.android.chrome",
                "gmail": "com.google.android.gm",
                "google maps": "com.google.android.apps.maps",
                "maps": "com.google.android.apps.maps",
                "whatsapp": "com.whatsapp",
                "youtube": "com.google.android.youtube",
            }
            package = aliases.get(label, label)
            result = await self._run_tool("phone_device", {"action": "open_app", "package": package})
            return self._spoken(result, None)

        gesture = re.match(r"^(tap|click|press|scroll)\s+(?:(up|down|left|right)\s+)?(?:node\s+)?(0(?:/[0-9]{1,2}){0,12})\b", low)
        if gesture:
            bridge = gate.mobile_viewport
            dom = bridge.latest_dom(device_id) if bridge is not None else None
            if dom is None:
                return "Please read the phone screen first so I have a fresh UI snapshot."
            action = "scroll" if gesture.group(1) == "scroll" else "click"
            args = {"action": action, "snapshotId": dom.snapshot.get("snapshotId"),
                    "nodeId": gesture.group(3)}
            if action == "scroll":
                if gesture.group(2) is None:
                    return "Which direction should I scroll: up, down, left, or right?"
                args["direction"] = gesture.group(2)
            result = await self._run_tool("phone_device", args)
            return self._spoken(result, None)
        named_tap = re.match(
            r"^(?:tap|click|press)\s+(?:the\s+)?(.+?)\s+on\s+(?:my\s+)?(?:phone|mobile)\s*$",
            raw, re.IGNORECASE)
        if named_tap:
            bridge = gate.mobile_viewport
            dom = bridge.latest_dom(device_id) if bridge is not None else None
            if dom is None:
                return "Please read the phone screen first so I have a fresh UI snapshot."
            label = named_tap.group(1).strip().casefold()
            matches: list[str] = []

            def find(node: Any, clickable_parent: str = "") -> None:
                if not isinstance(node, dict):
                    return
                node_id = str(node.get("id") or "")
                clickable = node_id if node.get("clickable") else clickable_parent
                text = str(node.get("text") or node.get("contentDescription") or "").strip()
                if text.casefold() == label and clickable:
                    matches.append(clickable)
                for child in node.get("children", []):
                    find(child, clickable)

            find(dom.snapshot.get("root"))
            matches = list(dict.fromkeys(matches))
            if len(matches) != 1:
                return ("I found more than one matching control. Name a node ID from the phone screen."
                        if matches else "I cannot find that control on the current phone screen.")
            result = await self._run_tool("phone_device", {
                "action": "click", "snapshotId": dom.snapshot.get("snapshotId"),
                "nodeId": matches[0],
            })
            return self._spoken(result, None)
        return None

    @staticmethod
    def _phone_dom_summary(raw: str) -> str:
        try:
            snapshot = json.loads(raw)
        except (TypeError, ValueError):
            return "I captured the phone's visible controls."
        if not isinstance(snapshot, dict):
            return "I captured the phone's visible controls."
        labels: list[str] = []

        def visit(node: Any) -> None:
            if not isinstance(node, dict) or len(labels) >= 15:
                return
            label = str(node.get("text") or node.get("contentDescription") or "").strip()
            if label:
                labels.append(f"{node.get('id', '?')}: {label[:70]}")
            for child in node.get("children", []):
                visit(child)

        visit(snapshot.get("root"))
        package = str(snapshot.get("package") or "the phone")[:80]
        if labels:
            return f"I can read {package}: {'; '.join(labels)}"[:900]
        return f"I captured {package}, but found no visible text labels."

    async def _run_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            return await self.dispatcher.dispatch(name, args)
        except Exception as exc:
            return ToolResult(f"{name} failed: {first_line(exc, 80)}", ok=False)

    def _spoken(self, result: ToolResult, preamble: Optional[str]) -> str:
        if result.media and result.media.get("confirm"):
            return result.text
        line = (result.text or "").splitlines()[0] if result.text else ""
        if not result.ok:
            return f"I couldn't complete that. {line}"
        if preamble and len(line) < 60:
            return preamble
        return f"{preamble + ' ' if preamble else ''}{line}".strip() or "Done."

    # ── optional local LLM (LM Studio / Ollama) ───────────────────────────────

    async def _try_local_llm(self, raw: str) -> Optional[str]:
        if self.router is None:
            return None
        try:
            profiles = getattr(self.router, "local_text_profiles", None)
            local = (profiles() if callable(profiles) else
                     [p for p in self.router.text_profiles() if getattr(p, "is_local", False)])
            if not local:
                return None
            generate = getattr(self.router, "generate_local_text", None)
            if callable(generate):
                _profile, reply = await generate(raw, instruction="", task="offline-chat")
            else:
                _profile, reply = await self.router.generate_text_offline(raw)
            return str(reply or "").strip() or None
        except Exception:
            return None

    # ── final fallback chit-chat ──────────────────────────────────────────────

    def _chit_chat(self, low: str, raw: str) -> str:
        if low.endswith("?") or raw.strip().endswith("?") or re.match(r"^(what|why|how|when|where|who|is|are|can|could|would|should)\b", low):
            return ("I'm on my local faculties just now, so I can't reach the "
                    "wider world for that — but ask me the time, your tasks, to open "
                    "something, to search your files, or anything in neuroscience, "
                    "and I'll handle it directly.")
        # Said word for word every time before (one line five times in the
        # owner's last 500 replies); a fresh one each time now (voice_variety).
        from .voice_variety import pick
        return pick("local_idle")
