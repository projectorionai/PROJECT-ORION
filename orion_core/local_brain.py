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
from .utils import first_line


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

    # ── public entry point ────────────────────────────────────────────────────

    async def respond(self, text: str) -> str:
        """Return a spoken reply; may perform a local tool action en route."""
        raw = str(text or "").strip()
        if not raw:
            return "I'm here, sir."
        lowered = raw.lower().strip(" .!?")
        if self.telemetry is not None:
            self.telemetry.metrics.incr("local_brain.turns")

        # Ordered intent resolution — first match wins.
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

    # ── daily-conversation intents ────────────────────────────────────────────

    def _intent_greeting(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b", low):
            hour = datetime.now().hour
            period = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
            return random.choice([
                f"Good {period}, sir. How may I help?",
                f"Good {period}, sir. I'm right here.",
                "At your service, sir.",
            ])
        return None

    def _intent_wellbeing(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bhow are you\b|\bhow's it going\b|\byou (ok|okay|doing)\b", low):
            return random.choice([
                "Running smoothly, sir — all systems nominal. And yourself?",
                "In fine form, thank you for asking. How are you keeping?",
                "Quite well, sir. Ready when you are.",
            ])
        if re.search(r"\bi'?m (tired|exhausted|stressed|sad|down|unwell|ill)\b", low):
            return ("I'm sorry to hear that, sir. Do take a moment — I can hold "
                    "your tasks, dim the pace, or simply keep you company.")
        return None

    def _intent_thanks(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\b(thank you|thanks|cheers|appreciate)\b", low):
            return random.choice(["A pleasure, sir.", "Always, sir.", "Think nothing of it."])
        return None

    def _intent_farewell(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\b(goodbye|good night|see you|that'?s all|bye)\b", low):
            return random.choice([
                "Very good, sir. I'll be here if you need me.",
                "Good night, sir.",
                "Standing by, sir.",
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
            return ("Even offline, sir, I can open and manage applications, search "
                    "your files, capture and read the screen, take notes and recall "
                    "them, manage your workspace, run desktop actions, discuss "
                    "neuroscience and engineering, do quick calculations, and keep "
                    "you company. Simply ask.")
        return None

    def _intent_time(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bwhat('?s| is) the time\b|\bthe time\b|\bwhat time\b", low):
            stamp = datetime.now().strftime("%I:%M %p").lstrip("0")
            return f"It's {stamp}, sir."
        return None

    def _intent_date(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bwhat('?s| is) (the |today'?s )?date\b|\bwhat day\b|\btoday'?s date\b", low):
            return f"Today is {datetime.now().strftime('%A, %d %B %Y')}, sir."
        return None

    def _intent_weather(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\bweather\b|\bforecast\b|\btemperature outside\b", low):
            cached = self._recall_value("weather")
            if cached:
                return f"From the last reading, sir: {cached}."
            return ("I can't reach the weather service without the network, sir, "
                    "but the moment we're back online I'll refresh it for you.")
        return None

    def _intent_system(self, low: str, raw: str) -> Optional[str]:
        if re.search(r"\b(cpu|memory|ram|system|how('?s| is) the (pc|computer|machine)|resources)\b", low):
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=0.2)
                ram = psutil.virtual_memory().percent
                return f"The machine is healthy, sir — CPU around {cpu:.0f} percent, memory at {ram:.0f} percent."
            except Exception:
                return "I couldn't read the system counters just now, sir."
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
        return f"That's {value}, sir."

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
            return f"Noted, sir: {fact}."
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
                return f"Your {topic}, sir: {value}."
            return f"I don't have your {topic} on record yet, sir. Tell me and I'll remember it."
        m = re.search(r"(?:what do you (?:know|remember) about|recall|remind me (?:about|of))\s+(.+)", low)
        if not m:
            return None
        query = m.group(1).strip(" ?.")
        try:
            rows = self.memory.query(query, limit=3)
        except Exception:
            rows = []
        if rows:
            return "Here's what I have, sir: " + "; ".join(r["value"] for r in rows[:3]) + "."
        # Fall through to the knowledge base for domain topics (if present).
        neuro = self.knowledge.answer(query) if self.knowledge is not None else None
        if neuro:
            return neuro
        return f"I don't have anything stored about {query}, sir."

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
        return ("That's within my field, sir, though I don't have a specific note "
                "on it offline. Ask me about neurons, synapses, brain–computer "
                "interfaces, the Utah array, Neuralink, spike sorting or decoding, "
                "and I can go deep.")

    # ── task routing (real local tools) ───────────────────────────────────────

    async def _intent_task(self, low: str, raw: str) -> Optional[str]:
        # Open an application or website.
        m = re.search(r"\b(?:open|launch|start|run)\s+(.+)", raw, re.IGNORECASE)
        if m:
            target = m.group(1).strip(" .")
            result = await self._run_tool("open_app", {"app_name": target})
            return self._spoken(result, f"Opening {target}, sir.")
        # Close an application.
        m = re.search(r"\b(?:close|quit|exit)\s+(.+)", raw, re.IGNORECASE)
        if m:
            target = m.group(1).strip(" .")
            result = await self._run_tool("close_app", {"app_name": target})
            return self._spoken(result, f"Closing {target}, sir.")
        # Web search.
        m = re.search(r"\b(?:search (?:for|the web for)?|google|look up)\s+(.+)", raw, re.IGNORECASE)
        if m:
            query = m.group(1).strip(" .")
            result = await self._run_tool("web_search", {"query": query})
            return self._spoken(result, f"Searching the web for {query}, sir.")
        # Screen awareness.
        if re.search(r"\bwhat('?s| is) on (my |the )?screen\b|\blook at (my |the )?screen\b|\bread (my |the )?screen\b", low):
            result = await self._run_tool("vision_analyse", {"action": "describe"})
            return self._spoken(result, "Let me look, sir.")
        # Find files.
        m = re.search(r"\bfind (?:my |the )?(?:file|files|folder|document)s?\s+(?:called |named |for )?(.+)", raw, re.IGNORECASE)
        if m:
            query = m.group(1).strip(" .")
            result = await self._run_tool("find_files", {"query": query})
            return self._spoken(result, f"Searching for {query}, sir.")
        # Workspace snapshot / resume.
        if re.search(r"\b(save|snapshot) (my )?workspace\b", low):
            result = await self._run_tool("workspace_control", {"action": "save"})
            return self._spoken(result, "Saving your workspace, sir.")
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
            return f"I couldn't complete that, sir. {line}"
        if preamble and len(line) < 60:
            return preamble
        return f"{preamble + ' ' if preamble else ''}{line}".strip() or "Done, sir."

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
            return ("I'm on my local faculties just now, sir, so I can't reach the "
                    "wider world for that — but ask me the time, your tasks, to open "
                    "something, to search your files, or anything in neuroscience, "
                    "and I'll handle it directly.")
        return random.choice([
            "Understood, sir. What would you like me to do?",
            "I'm with you, sir. Shall I open something, take a note, or look "
            "something up locally?",
            "Very good, sir. How can I be useful right now?",
        ])
