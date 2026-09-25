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

import ast
import operator
import random
import re
from datetime import datetime
from typing import Any, Optional

from .bus import OrionBus
from .data import ToolResult
from .knowledge import NeuroKnowledgeBase
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

    async def respond(self, text: str) -> str:
        """Return a spoken reply; may perform a local tool action en route."""
        raw = str(text or "").strip()
        if not raw:
            return "I'm here."
        lowered = raw.lower().strip(" .!?")
        if self.telemetry is not None:
            self.telemetry.metrics.incr("local_brain.turns")

        # §2.3: a casual "no, that's wrong" / "actually it's X" corrects the last
        # fact ORION gave (from KNOWLEDGE/memory) instead of being a fresh turn.
        corrected = await self._maybe_apply_correction(raw)
        if corrected is not None:
            return corrected

        # Ordered intent resolution — first match wins.  Answers sourced from
        # memory/knowledge (recall, neuroscience) are remembered as correctable.
        _correctable = (self._intent_recall, self._intent_neuro)
        for handler in (
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
                return reply

        # Task intents run a real local tool and report the outcome.
        task_reply = await self._intent_task(lowered, raw)
        if task_reply is not None:
            return task_reply

        # A local LLM, if configured, handles open-ended chat.
        llm_reply = await self._try_local_llm(raw)
        if llm_reply is not None:
            return llm_reply

        return self._chit_chat(lowered, raw)

    async def _maybe_apply_correction(self, raw: str) -> Optional[str]:
        """If ORION just gave a fact and the user disputes it, record an
        authoritative correction via the LearningService and confirm — instead
        of treating "no, that's wrong" as an ordinary turn. Returns the spoken
        acknowledgement, or None when this isn't a correction of a live fact."""
        if not self._last_fact_answer:
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
            self.telemetry.metrics.incr("local_brain.corrections")
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
        if re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b", low):
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
        if re.search(r"\b(thank you|thanks|cheers|appreciate)\b", low):
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

    def _intent_identity(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bwho are you\b|\bwhat are you\b|\byour name\b", low):
            return ("I am ORION — your personal operating system and aide. At the "
                    "moment I'm running on my local faculties, without the cloud, "
                    "yet still very much at your disposal.")
        return None

    def _intent_capabilities(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bwhat can you do\b|\bhelp me with\b|\byour (capabilities|features)\b", low):
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
        if re.search(r"\b(cpu|memory|ram|system|how('?s| is) the (pc|computer|machine)|resources)\b", low):
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

    _MATH_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.FloorDiv: operator.floordiv,
    }

    def _intent_math(self, low: str, raw: str) -> Optional[str]:
        m = re.search(r"(?:what('?s| is)|calculate|compute|work out)\s+(.+)", low)
        expr = m.group(2) if m else (low if re.fullmatch(r"[0-9\.\s\+\-\*/x%\^\(\)]+", low) else "")
        if not expr:
            return None
        expr = expr.replace("x", "*").replace("plus", "+").replace("minus", "-")
        expr = expr.replace("times", "*").replace("divided by", "/").replace("^", "**")
        expr = re.sub(r"[^0-9\.\+\-\*/%\(\)\s]", "", expr).strip()
        if not expr or not re.search(r"[0-9]", expr):
            return None
        try:
            value = self._safe_eval(ast.parse(expr, mode="eval").body)
        except Exception:
            return None
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return f"That's {value}."

    def _safe_eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in self._MATH_OPS:
            return self._MATH_OPS[type(node.op)](self._safe_eval(node.left), self._safe_eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._MATH_OPS:
            return self._MATH_OPS[type(node.op)](self._safe_eval(node.operand))
        raise ValueError("unsupported expression")

    # ── memory ────────────────────────────────────────────────────────────────

    def _intent_remember(self, low: str, raw: str) -> Optional[str]:
        m = re.search(r"(?:remember|note|make a note|don'?t forget)\s+(?:that\s+)?(.+)", raw, re.IGNORECASE)
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
        # Explicit electronics inspection remains useful offline: the camera
        # and local quality/OCR checks work without a cloud conversation model.
        if re.search(r"\b(?:scan|inspect|analyse|analyze|examine|look at)\s+(?:(?:this|the|my|a|these)\s+)?(?:pcb|circuit\s+board|electronics|electronic\s+(?:board|device|components?))\b", low):
            result = await self._run_tool("vision_analyse", {"action": "pcb", "prompt": raw})
            if result.ok and result.evidence:
                report = result.evidence[0]
                return str(report.get("summary") or result.text)
            return self._spoken(result, None)
        # Open an application or website.
        m = re.search(r"\b(?:open|launch|start|run)\s+(.+)", raw, re.IGNORECASE)
        if m:
            target = m.group(1).strip(" .")
            result = await self._run_tool("open_app", {"app_name": target})
            return self._spoken(result, f"Opening {target}.")
        # Close an application.
        m = re.search(r"\b(?:close|quit|exit)\s+(.+)", raw, re.IGNORECASE)
        if m:
            target = m.group(1).strip(" .")
            result = await self._run_tool("close_app", {"app_name": target})
            return self._spoken(result, f"Closing {target}.")
        # Web search.
        m = re.search(r"\b(?:search (?:for|the web for)?|google|look up)\s+(.+)", raw, re.IGNORECASE)
        if m:
            query = m.group(1).strip(" .")
            result = await self._run_tool("web_search", {"query": query})
            return self._spoken(result, f"Searching the web for {query}.")
        # Screen awareness.
        if re.search(r"\bwhat('?s| is) on (my |the )?screen\b|\blook at (my |the )?screen\b|\bread (my |the )?screen\b", low):
            result = await self._run_tool("vision_analyse", {"action": "describe"})
            return self._spoken(result, "Let me look.")
        # Find files.
        m = re.search(r"\bfind (?:my |the )?(?:file|files|folder|document)s?\s+(?:called |named |for )?(.+)", raw, re.IGNORECASE)
        if m:
            query = m.group(1).strip(" .")
            result = await self._run_tool("find_files", {"query": query})
            return self._spoken(result, f"Searching for {query}.")
        # Workspace snapshot / resume.
        if re.search(r"\b(save|snapshot) (my )?workspace\b", low):
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

    async def _run_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            return await self.dispatcher.dispatch(name, args)
        except Exception as exc:
            return ToolResult(f"{name} failed: {first_line(exc, 80)}", ok=False)

    def _spoken(self, result: ToolResult, preamble: Optional[str]) -> str:
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
        local = [p for p in self.router.text_profiles()
                 if p.base_url.startswith(("http://127.0.0.1", "http://localhost"))]
        if not local:
            return None
        try:
            _profile, reply = await self.router.generate_text(raw)
            return reply
        except Exception:
            return None

    # ── final fallback chit-chat ──────────────────────────────────────────────

    def _chit_chat(self, low: str, raw: str) -> str:
        if low.endswith("?") or raw.strip().endswith("?") or re.match(r"^(what|why|how|when|where|who|is|are|can|could|would|should)\b", low):
            return ("I'm on my local faculties just now, so I can't reach the "
                    "wider world for that — but ask me the time, your tasks, to open "
                    "something, to search your files, or anything in neuroscience, "
                    "and I'll handle it directly.")
        return random.choice([
            "Understood. What would you like me to do?",
            "I'm with you. Shall I open something, take a note, or look "
            "something up locally?",
            "Very good. How can I be useful right now?",
        ])
