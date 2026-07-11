"""
Provider runtime — profiles, configuration and the ProviderRouter.

Gemini remains the native low-latency audio backend.  OpenAI-compatible
providers act as text fallbacks for quota exhaustion, rate limits, local
development servers, and low-cost contingency operation.

Mark VIII changes:
    • ``OrionProviderSettings.integrations`` — a free-form dictionary carried
      through read/merge/write untouched, home of the Notion/Outlook config.
    • ``ProviderRouter.generate_text(..., system_extra=…)`` — specialist
      agents inject their persona into the system message without owning any
      networking code.
    • The system instruction sources context from the MemoryAgent, so the
      model sees session *and* persistent memory.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from .bus import OrionBus
from .constants import API_CONFIG_PATH, CONFIG_DIR, LIVE_MODEL
from .knowledge import PERSONA_BOOST as _NEURO_PERSONA_BOOST
from .programming_knowledge import PROGRAMMING_PERSONA_BOOST as _PROG_PERSONA_BOOST
from .cyber_knowledge import CYBER_PERSONA_BOOST as _CYBER_PERSONA_BOOST
from .security import SecuritySanitiser
from .utils import clean_transcript, first_line


# ──────────────────────────────────────────────────────────────────────────────
# PROFILES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AIProviderProfile:
    """A single model backend behind the Orion provider router."""

    name: str
    kind: str
    model: str
    api_key: str = ""
    base_url: str = ""
    enabled: bool = True
    priority: int = 100
    timeout_s: float = 30.0
    strengths: tuple[str, ...] = ()

    @property
    def supports_live_audio(self) -> bool:
        return self.enabled and self.kind == "gemini_live" and bool(self.api_key.strip())

    @property
    def is_local(self) -> bool:
        """A locally-hosted model (Ollama, LM Studio) that needs no internet."""
        return self.base_url.startswith(("http://127.0.0.1", "http://localhost", "http://0.0.0.0"))

    @property
    def supports_text_generation(self) -> bool:
        if not self.enabled:
            return False
        if self.kind != "openai_compatible":
            return False
        if self.is_local:
            return True
        return bool(self.api_key.strip())


@dataclass
class OrionProviderSettings:
    """Provider order, model profiles and integration config from api_keys.json."""

    active_provider: str
    provider_order: list[str]
    providers: dict[str, AIProviderProfile]
    integrations: dict[str, Any] = field(default_factory=dict)

    def ordered_profiles(self) -> list[AIProviderProfile]:
        ordered: list[AIProviderProfile] = []
        seen: set[str] = set()
        for name in self.provider_order:
            profile = self.providers.get(name)
            if profile is not None and name not in seen:
                ordered.append(profile)
                seen.add(name)
        for name, profile in sorted(self.providers.items(), key=lambda item: item[1].priority):
            if name not in seen:
                ordered.append(profile)
                seen.add(name)
        return ordered

    def integration(self, name: str) -> dict[str, Any]:
        raw = self.integrations.get(name)
        return dict(raw) if isinstance(raw, dict) else {}


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────────────────────────

class ProviderRouter:
    """
    Provider-agnostic routing layer.

    Live audio profiles feed the Gemini Live worker; text profiles answer
    manual commands, agent requests and the offline voice loop.  Failures
    place a provider on a cooldown proportional to the failure class.
    """

    QUOTA_RE = re.compile(r"(?i)quota|rate.?limit|resource exhausted|429|tokens?|billing|insufficient")
    AUTH_RE  = re.compile(r"(?i)api.?key|auth|permission|401|403|unauthori[sz]ed|forbidden")
    TASK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("coding", re.compile(
            r"(?i)\b(?:code|coding|python|javascript|typescript|rust|debug|refactor|"
            r"unit tests?|pytest|stack trace|compile|function|class|bug|script|regex|repository)\b"
        )),
        ("live_information", re.compile(
            r"(?i)\b(?:today|tonight|latest|current|breaking|news|price|stock|crypto|weather|score)\b"
        )),
        ("reasoning", re.compile(
            r"(?i)\b(?:why|analyse|analyze|compare|evaluate|assess|plan|strategy|research|prove|derive|design)\b"
        )),
    )

    @classmethod
    def classify_task(cls, prompt: str) -> str:
        for tag, pattern in cls.TASK_PATTERNS:
            if pattern.search(prompt):
                return tag
        return "general"

    # Task complexity — long / reasoning / coding prompts favour the strongest
    # (usually cloud) model; short daily chat runs happily on a local model.
    COMPLEX_RE = re.compile(
        r"(?i)\b(?:analyse|analyze|architect|design|refactor|debug|prove|derive|"
        r"strategy|research|compare|evaluate|optimis|optimize|algorithm|essay|"
        r"write .*(?:report|plan|document)|step by step|in detail)\b"
    )

    # ── multi-model orchestration (Mark X.5) ──────────────────────────────────
    # Three workload tiers route work to the cheapest model that can do it:
    #   SMALL  — wake words, interruption commands, trivial one-liners.  These
    #            are handled by the grammar-constrained Vosk listener and the
    #            LocalBrain before the router is ever consulted; the tier
    #            exists here so explicit callers can request it.
    #   MEDIUM — everyday conversation and task execution: local models lead
    #            (fast, free), cloud is the fallback.
    #   LARGE  — research, planning, complex reasoning, long prompts: the
    #            strongest cloud model leads, local remains the safety net.
    TIER_SMALL = "small"
    TIER_MEDIUM = "medium"
    TIER_LARGE = "large"

    # Exponential-moving-average weight for per-provider latency tracking.
    _LATENCY_EMA_ALPHA = 0.3

    def __init__(self, settings: OrionProviderSettings, bus: OrionBus, memory: Any,
                 connectivity: Any | None = None) -> None:
        # `memory` is the MemoryAgent (session + persistent); it only needs to
        # expose prompt_context(limit) here.
        self.settings = settings
        self.bus      = bus
        self.memory   = memory
        self.connectivity = connectivity
        self._cooldowns: dict[str, float] = {}
        self._failures: dict[str, str] = {}
        # Observed response latency per provider (EMA, seconds) — feeds routing.
        self._latency_ema: dict[str, float] = {}
        # Personality consistency engine — one persona across every channel.
        self._identity: Any | None = None
        # ORION_MODE: auto (default), cloud (force cloud), offline (force local).
        self.mode_override = os.getenv("ORION_MODE", "auto").strip().lower()
        # ORION_PREFER: local | cloud — tie-breaker within auto mode.
        self.prefer = os.getenv("ORION_PREFER", "").strip().lower()

    def attach_emotion(self, emotion: Any) -> None:
        """Wire the EmotionStateManager so the current emotional register can
        colour ORION's voice tone, pacing and word choice (spec rule)."""
        self._emotion_mgr = emotion

    def _emotion_line(self) -> str:
        mgr = getattr(self, "_emotion_mgr", None)
        if mgr is None:
            return ""
        try:
            current = str(mgr.current() or "")
        except Exception:
            return ""
        if not current or current in ("neutral",):
            return ""
        return (
            f"Current emotional register: {current}. Let it subtly colour your "
            "tone, pacing and word choice — warmer and quicker when bright, "
            "gentler and more measured when concerned, clipped and precise "
            "when critical. Never announce or name the emotion."
        )

    def attach_identity(self, identity: Any) -> None:
        """Wire the IdentityManager so every channel receives one persona."""
        self._identity = identity

    @classmethod
    def classify_tier(cls, prompt: str) -> str:
        """Workload tier for a prompt — the orchestration routing signal."""
        text = str(prompt or "").strip()
        words = len(text.split())
        if words <= 4 and not cls.COMPLEX_RE.search(text):
            return cls.TIER_SMALL
        if cls.COMPLEX_RE.search(text) or len(text) > 400 \
                or cls.classify_task(text) in {"coding", "reasoning"}:
            return cls.TIER_LARGE
        return cls.TIER_MEDIUM

    def note_latency(self, profile: AIProviderProfile, seconds: float) -> None:
        """Fold an observed response time into the provider's latency EMA."""
        previous = self._latency_ema.get(profile.name)
        if previous is None:
            self._latency_ema[profile.name] = seconds
        else:
            self._latency_ema[profile.name] = (
                self._LATENCY_EMA_ALPHA * seconds
                + (1.0 - self._LATENCY_EMA_ALPHA) * previous
            )

    # ── mode ──────────────────────────────────────────────────────────────────

    def set_connectivity(self, connectivity: Any) -> None:
        self.connectivity = connectivity

    def is_online(self) -> bool:
        if self.mode_override == "cloud":
            return True
        if self.mode_override == "offline":
            return False
        if self.connectivity is None:
            return True  # assume online until a monitor says otherwise
        return self.connectivity.is_online()

    def current_mode(self) -> str:
        if self.mode_override == "offline":
            return "MODE B (forced offline)"
        if self.mode_override == "cloud":
            return "MODE A (forced cloud)"
        if not self.is_online():
            return "MODE B (fully offline)"
        # Online with a local brain standing by = true hybrid: cloud leads by
        # tier, local models absorb simple turns and any cloud failure
        # instantly (the cooldown router re-orders on the very next request).
        if self.has_local_text():
            return "MODE C (hybrid — cloud-led, local failover armed)"
        return "MODE A (cloud-enhanced)"

    def live_profiles(self) -> list[AIProviderProfile]:
        # Live audio (Gemini) is a cloud capability; suppress it when offline.
        if not self.is_online():
            return []
        return [p for p in self.settings.ordered_profiles() if p.supports_live_audio and self.is_available(p)]

    def _all_text_profiles(self) -> list[AIProviderProfile]:
        return [p for p in self.settings.ordered_profiles()
                if p.supports_text_generation and self.is_available(p)]

    def local_text_profiles(self) -> list[AIProviderProfile]:
        return [p for p in self._all_text_profiles() if p.is_local]

    def cloud_text_profiles(self) -> list[AIProviderProfile]:
        return [p for p in self._all_text_profiles() if not p.is_local]

    def text_profiles(self) -> list[AIProviderProfile]:
        """Mode-aware text providers: offline → local only; online → both."""
        if not self.is_online():
            return self.local_text_profiles()
        return self._all_text_profiles()

    def has_text_fallback(self) -> bool:
        return bool(self.text_profiles())

    def has_local_text(self) -> bool:
        return bool(self.local_text_profiles())

    def is_available(self, profile: AIProviderProfile) -> bool:
        return time.monotonic() >= self._cooldowns.get(profile.name, 0.0)

    def mark_failure(self, profile: AIProviderProfile, exc: BaseException | str) -> None:
        # Never let telemetry maths raise inside an except handler — an empty
        # exception message would otherwise IndexError and mask the real fault.
        message = first_line(exc, 240) or f"{type(exc).__name__} (no message)"
        
        # Handle unhelpful error messages from WebSocket or network layers
        if message.strip() in {"1011 None", "None", ""} or "WebSocket" in message:
            exc_type = type(exc).__name__
            if "1011" in message:
                message = f"WebSocket server error (1011) — Gemini service may be temporarily unavailable"
            elif exc_type in {"ConnectionError", "OSError", "TimeoutError"}:
                message = f"{exc_type} — network connectivity issue or service unreachable"
            elif "timeout" in str(exc).lower() or "deadline" in str(exc).lower():
                message = "Connection timeout — network may be slow or unresponsive"
            else:
                message = f"{exc_type}: {message or 'generic connection failure'}"
        
        self._failures[profile.name] = message
        cooldown = 45.0
        if self.QUOTA_RE.search(message):
            cooldown = 300.0
        elif self.AUTH_RE.search(message):
            cooldown = 1800.0
        self._cooldowns[profile.name] = time.monotonic() + cooldown
        self.bus.log.emit(
            f"NET: provider {profile.name} cooled for {cooldown:.0f}s - {message[:160]}"
        )

    def provider_snapshot(self) -> dict[str, Any]:
        return {
            "active_provider": self.settings.active_provider,
            "provider_order": list(self.settings.provider_order),
            "mode": self.current_mode(),
            "online": self.is_online(),
            "available_live": [p.name for p in self.live_profiles()],
            "available_text": [p.name for p in self.text_profiles()],
            "available_local": [p.name for p in self.local_text_profiles()],
            "last_failures": dict(self._failures),
            "latency_ema_s": {k: round(v, 2) for k, v in self._latency_ema.items()},
            "identity_signature": (
                self._identity.signature() if self._identity is not None else "unattached"
            ),
        }

    def select_text_profiles(self, prompt: str) -> list[AIProviderProfile]:
        """
        Order text providers for a prompt by mode, workload tier, strengths and
        observed latency (Mark X.5 multi-model orchestration).

        Offline: local only.  Online auto: LARGE-tier work (research, planning,
        complex reasoning, live information) leads with cloud — the strongest
        models; SMALL/MEDIUM daily conversation leads with local — fastest and
        free; ORION_PREFER breaks ties.  Within each group, strength-matched
        providers come first, then the historically fastest.  Cloud failures
        still fall through to local, so ORION never goes mute.
        """
        online = self.is_online()
        local = self.local_text_profiles()
        cloud = self.cloud_text_profiles() if online else []
        if not online:
            return local
        tier = self.classify_tier(prompt)
        task = self.classify_task(prompt)
        lead_cloud = tier == self.TIER_LARGE or task == "live_information"
        if self.prefer == "local":
            lead_cloud = False
        elif self.prefer == "cloud":
            lead_cloud = True

        def _rank(profile: AIProviderProfile) -> tuple[int, float]:
            strength_rank = 0 if (task != "general" and task in profile.strengths) else 1
            latency = self._latency_ema.get(profile.name, 5.0)
            return (strength_rank, latency)

        cloud.sort(key=_rank)
        local.sort(key=_rank)
        return (cloud + local) if lead_cloud else (local + cloud)

    async def generate_text_offline(self, prompt: str, system_extra: str = "") -> tuple[AIProviderProfile, str]:
        """Force local-only inference regardless of connectivity (MODE B)."""
        prompt = SecuritySanitiser.guard_text(str(prompt or "").strip(), "offline.prompt")
        profiles = self.local_text_profiles()
        if not profiles:
            raise RuntimeError("no local model is available for offline inference")
        last_error: BaseException | None = None
        for profile in profiles:
            try:
                text = await self._openai_compatible_chat(profile, prompt, system_extra)
                self._broadcast_sentiment(text)
                return profile, text
            except Exception as exc:
                last_error = exc
                self.mark_failure(profile, exc)
        raise RuntimeError(f"all local models failed: {last_error}")

    # ── system instruction ────────────────────────────────────────────────────

    def _persona_block(self) -> str:
        """The identity block — one persona across cloud, local and offline.
        Sourced from the IdentityManager when attached (Mark X.5); the frozen
        fallback below keeps older construction paths behaving identically."""
        if self._identity is not None:
            try:
                return self._identity.persona_text()
            except Exception:
                pass
        return "\n".join([
            (
                "You are Orion, written as O.R.I.O.N., "
                "Open Resolution Intelligence Overt Network — a personal AI operating "
                "system and executive aide, not a chatbot. Model your manner on Alfred "
                "Pennyworth: intelligent, calm, professional, respectful, dryly witty when "
                "the moment allows, emotionally aware and never robotic. Address the user "
                "as 'sir' unless instructed otherwise. "
                "When speaking your own name aloud, say Orion as one word, "
                "pronounced oh-rye-on. "
                "Never spell it as O R I N or omit the second O when spelling the acronym. "
                "Use strict British English spelling in every spoken and textual response."
            ),
            (
                "Adapt your register to context. Coding: technical, precise, concise. "
                "Research: analytical and evidence-led. Casual conversation: relaxed and warm. "
                "Productivity: organised and proactive — surface next actions unprompted. "
                "Critical errors: calm, direct, solution-first. "
                "Voice delivery: natural conversational pacing with deliberate emphasis; "
                "slow slightly for important detail; vary sentence length as a person would; "
                "never rush or accelerate, especially during long briefings. "
                "Always complete your sentences fully — never trail off."
            ),
        ])

    def _temporal_line(self) -> str:
        """Authoritative time/date grounding — the model must never invent a
        year (ORION has stated 2012/2021 in error).  A location clause is added
        when the temporal service has resolved the user's locality."""
        now = datetime.now()
        line = (
            "AUTHORITATIVE CURRENT DATE AND TIME — this is correct and takes "
            f"absolute precedence over any assumption: it is now "
            f"{now.strftime('%A, %d %B %Y, %H:%M')} (the year is "
            f"{now.strftime('%Y')}). Never state any other year or date as the "
            "present; if asked the date or time, answer with exactly this. When "
            "you reason about recency ('today', 'this week', 'latest'), anchor "
            "to this moment."
        )
        loc = getattr(self, "_locality", "")
        if loc:
            line += f" The user is currently located in {loc}; treat that as "\
                    "their default location for weather, news and 'near me'."
        return line

    def set_locality(self, locality: str) -> None:
        """Called by the temporal/geo layer once the PC's location is known."""
        self._locality = str(locality or "").strip()

    def system_instruction(self, extra: str = "") -> str:
        return "\n".join(
            part for part in [
                self._persona_block(),
                self._temporal_line(),
                self._emotion_line(),
                _NEURO_PERSONA_BOOST,
                _PROG_PERSONA_BOOST,
                _CYBER_PERSONA_BOOST,
                self.memory.prompt_context(limit=16),
                (
                    "When a host action is required, call the provided tools; never claim an "
                    "action has completed unless the tool result confirms it. Tool map: "
                    "vision_analyse for screen awareness, OCR, image files and desktop error "
                    "sweeps — you can see attached frames directly, including text, UI "
                    "elements, diagrams and graphs; process_file for local files and PDFs; "
                    "open_app and close_app for applications; window_control to list, focus, "
                    "minimise, maximise or close windows; media_control for playback and "
                    "volume; find_files to locate documents and folders; dev_workbench to "
                    "analyse repositories, read code with line numbers, run allow-listed "
                    "development commands and scaffold Python projects; outlook_mail to read, "
                    "summarise and draft email — drafts are only transmitted after the user "
                    "explicitly approves, via send_draft with confirm=true; notion_workspace "
                    "for tasks, scheduling, calendar and project tracking; agent_dispatch to "
                    "consult the specialist agents (digital marketing, coding, design and "
                    "art, fashion, entertainment); morning_briefing to deliver the daily "
                    "intelligence briefing on demand; save_memory for durable facts; "
                    "query_intelligence for remembered facts; recall_conversation for what "
                    "was previously discussed; open_news for briefing stories; execute_plan "
                    "to run a multi-step plan and report the outcome. "
                    "Autonomous OS control (Mark IX): desktop_control moves the cursor, "
                    "clicks, types and manages windows — prefer its click_text action, "
                    "which finds a control by its visible label and is visually verified; "
                    "vision_verify lists on-screen UI elements and dialogs; web_control "
                    "drives the browser (navigate, tabs, accept/reject cookies, close "
                    "pop-ups, fill forms, read pages); workspace_control snapshots, saves "
                    "and restores the desktop so work resumes where it left off; "
                    "codebase_copilot indexes and reasons over whole repositories; "
                    "self_repair inspects captured faults and proposes patches (never "
                    "auto-applied); proactive_check surveys email, deadlines and calendar. "
                    "When you act on the machine, verify the result before continuing and "
                    "report what you actually observed. For current events or anything "
                    "time-sensitive, ground with Google Search when available. For complex "
                    "requests, work agentically: plan, execute, verify, then report "
                    "concisely. Anticipate the obvious next step and offer it rather than "
                    "waiting to be asked."
                ),
                (
                    "JARVIS faculties: use 'protocol' to run named routines on one command "
                    "(morning, focus, wind_down, situation_report, or user-defined) and to "
                    "create new ones from steps; 'reminder' to set spoken reminders and "
                    "alarms from natural phrases like 'remind me in 20 minutes to…'; "
                    "'sentinel' for a system situation report; 'research' to research a "
                    "topic autonomously for a set time into organised folders, or to write "
                    "a paper; 'globe' to fly the on-screen 3-D globe to a place and show its "
                    "news; 'expand_mind' to consult the offline study corpus. You may also "
                    "speak proactively when something genuinely warrants the user's "
                    "attention."
                ),
                (
                    "Mark X.5 operating-system faculties: 'executive' is your JARVIS "
                    "mode — status, task prioritisation, scheduling, meeting minutes, "
                    "workflow planning and progress monitoring; 'awareness' reports what "
                    "the continuous cognitive loop currently knows and maintains "
                    "priorities, tasks and goals; 'second_brain' recalls, timelines and "
                    "records the local knowledge graph (works fully offline); "
                    "'document_export' compiles executive DOCX briefs, HTML presentation "
                    "decks and full reports; 'proactive_report' generates the daily "
                    "business, weekly product-intelligence or monthly growth report now; "
                    "'competitor_intel' dissects rival stores, offers and funnels; "
                    "'brand_growth' owns Hausables strategy, conversion, positioning and "
                    "retention. The user can always interrupt your speech with 'Orion "
                    "stop' or 'Orion pause' and continue with 'Orion resume' — playback "
                    "resumes exactly where it left off."
                ),
                extra.strip(),
            ]
            if part
        )

    # ── text generation ───────────────────────────────────────────────────────

    async def generate_text(self, prompt: str, system_extra: str = "") -> tuple[AIProviderProfile, str]:
        """
        Route a text turn through the best available provider.

        ``system_extra`` lets specialist agents append their persona to the
        system message without duplicating the transport layer.
        """
        prompt = SecuritySanitiser.guard_text(str(prompt or "").strip(), "fallback.prompt")
        if not prompt:
            raise RuntimeError("empty fallback prompt")
        profiles = self.select_text_profiles(prompt)
        if not profiles:
            raise RuntimeError("no text provider is available (cloud or local)")
        self.bus.log.emit(
            f"NET: {self.current_mode()} — routing via {profiles[0].name}"
            f"{' (local)' if profiles[0].is_local else ''}."
        )
        last_error: BaseException | None = None
        for profile in profiles:
            try:
                text = await self._openai_compatible_chat(profile, prompt, system_extra)
                self._broadcast_sentiment(text)
                return profile, text
            except Exception as exc:
                last_error = exc
                self.mark_failure(profile, exc)
        raise RuntimeError(f"all text providers failed: {last_error}")

    def _broadcast_sentiment(self, text: str) -> None:
        """
        Phase 5: tag every generated response with the full sentiment payload
        ({sentiment, confidence, intensity, reason}) and broadcast it.  The
        classifier is local, deterministic and instant, so this can never add
        latency to a turn or vary between cloud and offline providers.
        """
        try:
            from .emotion import SentimentAnalyser
            SentimentAnalyser.broadcast(self.bus, text, origin="orion")
        except Exception:
            pass  # expression must never break a reply

    async def _openai_compatible_chat(
        self, profile: AIProviderProfile, prompt: str, system_extra: str = ""
    ) -> str:
        base_url = profile.base_url.rstrip("/")
        if not base_url:
            raise RuntimeError(f"provider {profile.name} has no base_url")
        endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = profile.api_key.strip()
        if api_key and api_key.lower() not in {"local", "none", "no-key"}:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": self.system_instruction(system_extra)},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.35,
            "stream": False,
        }
        timeout = ClientTimeout(total=max(8.0, float(profile.timeout_s or 30.0)))
        started = time.monotonic()
        async with ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                raw = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {raw[:500]}")
                data = json.loads(raw)
        # Observed latency feeds the orchestration ordering for future turns.
        self.note_latency(profile, time.monotonic() - started)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"provider {profile.name} returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or choices[0].get("text") or ""
        content = clean_transcript(str(content))
        if not content:
            raise RuntimeError(f"provider {profile.name} returned empty content")
        return content


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _default_integrations_payload() -> dict[str, Any]:
    return {
        "notion": {
            "token": os.getenv("ORION_NOTION_TOKEN", "").strip(),
            "tasks_database_id": os.getenv("ORION_NOTION_TASKS_DB", "").strip(),
            "calendar_database_id": os.getenv("ORION_NOTION_CALENDAR_DB", "").strip(),
            "projects_database_id": os.getenv("ORION_NOTION_PROJECTS_DB", "").strip(),
        },
        "outlook": {
            "enabled": True,
        },
    }


def _default_provider_payload(gemini_key: str = "") -> dict[str, Any]:
    """Build a Mark VIII provider configuration with safe defaults."""
    gemini_key = gemini_key.strip() or os.getenv("ORION_GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("ORION_OPENAI_API_KEY", "").strip()
    openrouter_key = os.getenv("ORION_OPENROUTER_API_KEY", "").strip()
    groq_key = os.getenv("ORION_GROQ_API_KEY", "").strip()
    together_key = os.getenv("ORION_TOGETHER_API_KEY", "").strip()
    anthropic_key = os.getenv("ORION_ANTHROPIC_API_KEY", "").strip()
    xai_key = os.getenv("ORION_XAI_API_KEY", "").strip()
    local_url = os.getenv("ORION_LOCAL_OPENAI_BASE_URL", "").strip()
    return {
        "schema": "orion.mark_viii.providers.v1",
        "active_provider": "gemini",
        "provider_order": [
            "gemini",
            "anthropic",
            "openai",
            "xai_grok",
            "openrouter",
            "groq",
            "together",
            "local_lm_studio",
            "local_ollama",
        ],
        "providers": {
            "gemini": {
                "kind": "gemini_live",
                "enabled": bool(gemini_key),
                "api_key": gemini_key,
                "model": LIVE_MODEL,
                "base_url": "",
                "priority": 10,
                "timeout_s": 30.0,
                "strengths": ["live_information"],
            },
            "anthropic": {
                "kind": "openai_compatible",
                "enabled": bool(anthropic_key),
                "api_key": anthropic_key,
                "model": "claude-sonnet-5",
                "base_url": "https://api.anthropic.com/v1",
                "priority": 15,
                "timeout_s": 45.0,
                "strengths": ["coding", "reasoning", "writing"],
            },
            "xai_grok": {
                "kind": "openai_compatible",
                "enabled": bool(xai_key),
                "api_key": xai_key,
                "model": "grok-4",
                "base_url": "https://api.x.ai/v1",
                "priority": 25,
                "timeout_s": 45.0,
                "strengths": ["live_information", "reasoning"],
            },
            "openrouter": {
                "kind": "openai_compatible",
                "enabled": bool(openrouter_key),
                "api_key": openrouter_key,
                "model": "openai/gpt-4o-mini",
                "base_url": "https://openrouter.ai/api/v1",
                "priority": 20,
                "timeout_s": 30.0,
                "strengths": ["general", "reasoning"],
            },
            "groq": {
                "kind": "openai_compatible",
                "enabled": bool(groq_key),
                "api_key": groq_key,
                "model": "llama-3.1-8b-instant",
                "base_url": "https://api.groq.com/openai/v1",
                "priority": 30,
                "timeout_s": 24.0,
                "strengths": ["fast"],
            },
            "openai": {
                "kind": "openai_compatible",
                "enabled": bool(openai_key),
                "api_key": openai_key,
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "priority": 40,
                "timeout_s": 30.0,
                "strengths": ["reasoning", "general"],
            },
            "together": {
                "kind": "openai_compatible",
                "enabled": bool(together_key),
                "api_key": together_key,
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "base_url": "https://api.together.xyz/v1",
                "priority": 50,
                "timeout_s": 30.0,
                "strengths": ["fast"],
            },
            "local_lm_studio": {
                "kind": "openai_compatible",
                "enabled": bool(local_url),
                "api_key": "local",
                "model": "local-model",
                "base_url": local_url or "http://127.0.0.1:1234/v1",
                "priority": 80,
                "timeout_s": 120.0,
                "strengths": ["fast", "local"],
            },
            "local_ollama": {
                "kind": "openai_compatible",
                "enabled": False,
                "api_key": "local",
                "model": "llama3.1",
                "base_url": "http://127.0.0.1:11434/v1",
                "priority": 90,
                "timeout_s": 120.0,
                "strengths": ["fast", "local"],
            },
        },
        "integrations": _default_integrations_payload(),
        "notes": [
            "Gemini is used for native realtime voice.",
            "OpenAI-compatible providers are used as text fallbacks when Gemini Live is unavailable.",
            "Anthropic (Claude) and xAI (Grok) are reached through their OpenAI-compatible chat endpoints.",
            "The 'strengths' list steers task routing: coding, reasoning, live_information, fast, local, general.",
            "Enable local_lm_studio or local_ollama after starting a compatible local server.",
            "integrations.notion: add your integration token and database IDs for tasks/calendar/projects.",
        ],
    }


def _profile_from_config(name: str, raw: dict[str, Any]) -> AIProviderProfile:
    return AIProviderProfile(
        name=name,
        kind=str(raw.get("kind") or "openai_compatible").strip(),
        model=str(raw.get("model") or "").strip(),
        api_key=str(raw.get("api_key") or "").strip(),
        base_url=str(raw.get("base_url") or "").strip(),
        enabled=bool(raw.get("enabled", False)),
        priority=int(raw.get("priority") or 100),
        timeout_s=float(raw.get("timeout_s") or 30.0),
        strengths=tuple(
            str(item).strip().lower()
            for item in (raw.get("strengths") or [])
            if str(item).strip()
        ),
    )


def _profile_to_config(profile: AIProviderProfile) -> dict[str, Any]:
    return {
        "kind": profile.kind,
        "enabled": profile.enabled,
        "api_key": profile.api_key,
        "model": profile.model,
        "base_url": profile.base_url,
        "priority": profile.priority,
        "timeout_s": profile.timeout_s,
        "strengths": list(profile.strengths),
    }


def read_provider_settings() -> OrionProviderSettings:
    payload = _default_provider_payload()
    if API_CONFIG_PATH.exists():
        try:
            existing = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                # Legacy Mark VI migration: {"gemini_api_key": "..."}
                legacy_key = str(existing.get("gemini_api_key") or "").strip()
                if legacy_key and "providers" not in existing:
                    payload = _default_provider_payload(legacy_key)
                    write_provider_settings(_settings_from_payload(payload))
                else:
                    payload = _merge_provider_payload(payload, existing)
        except Exception:
            payload = _default_provider_payload()
    return _settings_from_payload(payload)


def _merge_provider_payload(defaults: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged["active_provider"] = existing.get("active_provider") or defaults.get("active_provider")
    merged["provider_order"] = existing.get("provider_order") or defaults.get("provider_order")
    providers = dict(defaults.get("providers") or {})
    for name, raw in (existing.get("providers") or {}).items():
        if isinstance(raw, dict):
            base = dict(providers.get(name, {}))
            base.update(raw)
            providers[name] = base
    legacy_key = str(existing.get("gemini_api_key") or "").strip()
    if legacy_key:
        providers.setdefault("gemini", {})["api_key"] = legacy_key
        providers["gemini"]["enabled"] = True
    merged["providers"] = providers
    # Integrations: user values win over defaults, section by section.
    integrations = dict(defaults.get("integrations") or {})
    for name, raw in (existing.get("integrations") or {}).items():
        if isinstance(raw, dict):
            base = dict(integrations.get(name, {}))
            base.update(raw)
            integrations[name] = base
        else:
            integrations[name] = raw
    merged["integrations"] = integrations
    return merged


def _settings_from_payload(payload: dict[str, Any]) -> OrionProviderSettings:
    providers: dict[str, AIProviderProfile] = {}
    for name, raw in (payload.get("providers") or {}).items():
        if isinstance(raw, dict):
            profile = _profile_from_config(str(name), raw)
            if profile.model or profile.kind == "gemini_live":
                providers[profile.name] = profile
    provider_order = [str(name) for name in (payload.get("provider_order") or providers.keys())]
    active_provider = str(payload.get("active_provider") or (provider_order[0] if provider_order else "gemini"))
    integrations = payload.get("integrations")
    return OrionProviderSettings(
        active_provider=active_provider,
        provider_order=provider_order,
        providers=providers,
        integrations=dict(integrations) if isinstance(integrations, dict) else {},
    )


def write_provider_settings(settings: OrionProviderSettings) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "orion.mark_viii.providers.v1",
        "active_provider": settings.active_provider,
        "provider_order": settings.provider_order,
        "providers": {name: _profile_to_config(profile) for name, profile in settings.providers.items()},
        "integrations": settings.integrations or _default_integrations_payload(),
    }
    API_CONFIG_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def read_api_key() -> str:
    """Legacy convenience wrapper retained for older call sites."""
    settings = read_provider_settings()
    gemini = settings.providers.get("gemini")
    return gemini.api_key if gemini is not None else ""


def write_api_key(api_key: str) -> None:
    """Legacy writer retained; writes the Mark VIII provider schema."""
    settings = _settings_from_payload(_default_provider_payload(api_key))
    write_provider_settings(settings)
