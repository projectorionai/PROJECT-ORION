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

import asyncio
import contextlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from .lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")

from .bus import OrionBus
from .constants import API_CONFIG_PATH, CONFIG_DIR, LIVE_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# TYPED ERRORS  — actionable failures instead of a bare RuntimeError
# ──────────────────────────────────────────────────────────────────────────────

class ProviderError(RuntimeError):
    """Base class for provider-routing failures.

    Subclasses RuntimeError so every existing ``except RuntimeError`` /
    ``except Exception`` call site keeps working, while callers that care can
    branch on the specific type and read structured, redaction-safe context.
    Never carries an API key, token or secret."""

    remediation: str = ""


class NoTextProviderError(ProviderError):
    """No text provider is currently usable (none configured, all cooling down,
    or offline with no local model).  This is a degraded-mode condition, not a
    crash: deterministic tools continue and the work resumes on recovery."""

    def __init__(self, message: str = "no text provider is available (cloud or local)",
                 *, configured: list[str] | None = None, online: bool | None = None) -> None:
        super().__init__(message)
        self.configured = configured or []
        self.online = online
        self.remediation = (
            "Add a cloud provider API key in config/api_keys.json (Anthropic, "
            "OpenAI, xAI, OpenRouter, Groq, Together) or start a local LLM "
            "(Ollama / LM Studio) and enable it. ORION keeps its deterministic "
            "tools running meanwhile."
        )


class AllTextProvidersFailedError(ProviderError):
    """Every eligible provider was tried and each errored.  Carries the ordered
    list of providers attempted and the last error message (no secrets)."""

    def __init__(self, last_error: object, *, attempts: list[str] | None = None) -> None:
        super().__init__(f"all text providers failed: {last_error}")
        self.attempts = attempts or []
        self.last_error = str(last_error)
        self.remediation = (
            "Check network connectivity and provider status; a local LLM "
            "(Ollama / LM Studio) gives an offline fallback."
        )
from .knowledge import PERSONA_BOOST as _NEURO_PERSONA_BOOST
from .programming_knowledge import PROGRAMMING_PERSONA_BOOST as _PROG_PERSONA_BOOST
from .cyber_knowledge import CYBER_PERSONA_BOOST as _CYBER_PERSONA_BOOST
from .security import SecuritySanitiser
from .utils import clean_transcript, first_line
from .atomic_io import atomic_write_text


# ──────────────────────────────────────────────────────────────────────────────
# MODEL HEALING
# ──────────────────────────────────────────────────────────────────────────────

#: Gemini's OpenAI-compatible endpoint. The same key that drives ORION's live
#: voice also answers ordinary chat here, which is what lets research, the
#: forge and every other text task use Gemini instead of whatever cheap
#: fallback happens to have credit left.
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

#: Model ids that were ORION's own shipped defaults and have since been
#: retired by their provider. A saved config still naming one is migrated on
#: load; anything else a provider retires is found at runtime by heal_model.
RETIRED_DEFAULTS: dict[str, dict[str, str]] = {
    # Groq withdrew it; every request answered 404 model_not_found and the
    # provider was benched on a loop (seen in the user's activity log).
    "groq": {"llama-3.1-8b-instant": "openai/gpt-oss-120b"},
}

#: Model families that are not chat models, whatever /models lists them as.
_NOT_CHAT_RE = re.compile(
    r"(?i)whisper|tts|orpheus|embed|guard|safeguard|safety|moderation|rerank|"
    r"transcri|lyria|imagen|veo|robotics|computer-use|image|audio|allam|"
    r"prompt-guard|nano-banana|live|native-audio")

#: Chat models worth having, best first. Matched as fragments so a point
#: release (qwen3.8 -> qwen3.9) still ranks.
_PREFERRED_FRAGMENTS = (
    "gemini-2.5-flash", "gpt-oss-120b", "nemotron-3-super", "qwen3",
    "llama-3.3-70b", "gemma-4-31b", "gpt-4o-mini", "gpt-oss-20b",
    "gemma-4-26b", "nemotron", "mistral", "llama",
)


def choose_replacement_model(available: list[str], *, current: str = "",
                             free_only: bool = False) -> str:
    """The best chat model in *available* to replace *current*.

    A provider retiring a model is routine; ORION treating it as an outage
    was the bug. The replacement is chosen from what the provider says it has
    right now, never from a hard-coded guess that will itself go stale.
    """
    candidates = [m for m in available
                  if m and m != current and not _NOT_CHAT_RE.search(m)]
    if free_only:
        candidates = [m for m in candidates if m.endswith(":free")]
    if not candidates:
        return ""

    def rank(model: str) -> tuple[int, int, str]:
        low = model.lower()
        preference = next((i for i, frag in enumerate(_PREFERRED_FRAGMENTS)
                           if frag in low), len(_PREFERRED_FRAGMENTS))
        size = re.search(r"(\d+)b\b", low)
        return (preference, -(int(size.group(1)) if size else 0), low)

    return sorted(candidates, key=rank)[0]


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
    # Optional backup credentials/models: the router rotates through these on
    # auth/quota (keys) or retired-model (models) failures instead of cooling
    # the whole provider for minutes.
    api_keys: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    # Payload budget in tokens (prompt + system + reply headroom).  0 = infer
    # from the model name; set explicitly in api_keys.json for tight
    # rate-limit tiers (e.g. Groq free tier's 6000 TPM).
    budget_tokens: int = 0

    def all_keys(self) -> tuple[str, ...]:
        """Primary key first, then backups — deduplicated, blanks dropped."""
        keys: list[str] = []
        for key in (self.api_key, *self.api_keys):
            key = str(key).strip()
            if key and key not in keys:
                keys.append(key)
        return tuple(keys)

    def all_models(self) -> tuple[str, ...]:
        """Primary model first, then alternates — deduplicated, blanks dropped."""
        models: list[str] = []
        for model in (self.model, *self.models):
            model = str(model).strip()
            if model and model not in models:
                models.append(model)
        return tuple(models)

    @property
    def supports_live_audio(self) -> bool:
        return self.enabled and self.kind == "gemini_live" and bool(self.all_keys())

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
        return bool(self.all_keys())


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
    # Hardware-exhaustion failures, almost always a LOCAL model that will not
    # fit in the GPU it was asked to load into. Checked before the quota class,
    # whose "resource exhausted" and "insufficient" both match this text.
    #
    # This one is not like the others. A quota resets on a clock, so retrying
    # after a few minutes is right. VRAM does not: it frees when something
    # else lets go, which may be never, and every retry asks the driver for
    # several gigabytes it cannot give. From the user's log, every 45 seconds:
    #
    #   llama-server process has terminated: exit status 1: cudaMalloc failed
    #   out of memory / alloc_tensor_range: failed to allocate CUDA
    #
    # That is not merely wasted work. A multi-gigabyte allocation attempt on a
    # GPU that is already full disturbs everything else holding memory on it --
    # including the WebEngine compositor drawing ORION's face, which loses its
    # surfaces and redraws them black. A random black flicker every forty-five
    # seconds is exactly what that looks like from the outside.
    OOM_RE = re.compile(
        r"(?i)out of memory|cudamalloc|failed to allocate|oom|"
        r"cuda error|insufficient memory|not enough memory")
    # Payload-class failures (HTTP 413, context-length exceeded): the PAYLOAD
    # was at fault, not the provider — benching it for minutes (the old
    # QUOTA_RE match on "tokens") starved the router for no reason.  Checked
    # BEFORE the quota class.
    PAYLOAD_RE = re.compile(
        r"(?i)413|request too large|payload too large|context.?length|"
        r"maximum context|input.?too.?long|prompt is too long")
    # Credit-class failures (OpenRouter HTTP 402). From the user's log:
    #
    #   HTTP 402: {"error":{"message":"This request requires more credits, or
    #   fewer max_tokens. You requested up to 16384 tokens, but can only
    #   afford 16096. …","code":402}}
    #
    # ORION never asked for 16384 — it never set max_tokens at all, and
    # OpenRouter then reserves the model's ENTIRE completion budget against the
    # balance. So a request that would have produced a two-line answer was
    # refused for want of credit it was never going to spend. Worse, the
    # message says "credits", QUOTA_RE matched it, and the provider was benched
    # for 300 s over a request that only needed a ceiling.
    #
    # It is also perfectly self-describing: the provider states exactly what it
    # CAN afford, so there is no guessing to do — take the number, cap to it,
    # and go again.
    CREDIT_RE = re.compile(
        r"(?i)402|requires more credits|can only afford|insufficient credits")
    AFFORD_RE = re.compile(r"(?i)can only afford\s+([0-9]+)")

    #: What ORION actually needs to say something. Spoken replies and tool
    #: arguments are short; the previous unbounded request reserved two orders
    #: of magnitude more than any real answer has ever used.
    DEFAULT_MAX_OUTPUT_TOKENS = 2048

    #: Transient faults (HTTP 5xx, timeouts, refused connections, empty
    #: replies) back off exponentially from the base to the ceiling. A flat
    #: 45 s retried an endpoint that was simply not running for the rest of
    #: the session, paying its full timeout every 45 seconds.
    TRANSIENT_BASE_S = 45.0
    TRANSIENT_MAX_S = 600.0

    #: Below this, a provider's "affordable" ceiling is not a ceiling, it is an
    #: empty account. Capping replies to 256 tokens is what truncated the
    #: research outline mid-line ("the outline could not be parsed"): better
    #: to move that provider onto a free model, or skip it, than to accept a
    #: reply too short to be the thing that was asked for.
    MIN_USEFUL_OUTPUT_TOKENS = 1024
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

    # A local model's KV-cache memory scales roughly linearly with context
    # length — a 7B/8B model at Ollama's unconstrained default (up to 128K)
    # can burn several extra GB of RAM/VRAM beyond the weights themselves,
    # which is exactly the "local fallback eats all the memory" failure mode.
    # This host has 16GB of PHYSICAL RAM and no discrete GPU headroom to rely
    # on, so the Tier-2 local model (qwen2.5:7b) is held to a strict 2048-token
    # context — the hard ceiling that keeps the weights + KV-cache inside RAM
    # and stops Windows from swapping mid-reply. ORION's own local payload
    # budget never needs more than this covers; the value can be raised on a
    # machine with more memory via ORION_LOCAL_NUM_CTX.
    try:
        LOCAL_NUM_CTX = max(512, min(8192, int(os.getenv("ORION_LOCAL_NUM_CTX", "2048"))))
    except ValueError:
        LOCAL_NUM_CTX = 2048

    def __init__(self, settings: OrionProviderSettings, bus: OrionBus, memory: Any,
                 connectivity: Any | None = None) -> None:
        # `memory` is the MemoryAgent (session + persistent); it only needs to
        # expose prompt_context(limit) here.
        self.settings = settings
        self.bus      = bus
        self.memory   = memory
        self.connectivity = connectivity
        self._cooldowns: dict[str, float] = {}
        #: Completion-token ceiling a provider has told us it can afford.
        self._output_cap: dict[str, int] = {}
        self._failures: dict[str, str] = {}
        # Consecutive out-of-GPU-memory failures per provider. A full
        # card does not empty on a timer, so each strike backs off
        # harder rather than retrying on the same clock forever.
        self._oom_strikes: dict[str, int] = {}
        # Unknown 429 reset times need a bounded exponential cooldown. A
        # successful response clears the streak for that provider.
        self._rate_strikes: dict[str, int] = {}
        # Consecutive transient faults per provider (escalating cooldown).
        self._fault_strikes: dict[str, int] = {}
        # Circuit-breaker bookkeeping: providers whose last call failed, and
        # those whose recovery probe (the first call after a cooldown) is in
        # flight right now.
        self._unhealthy: set[str] = set()
        self._probing: set[str] = set()
        # Active backup-key / alternate-model index per provider (rotation on
        # auth/quota or retired-model failures respectively).
        self._key_index: dict[str, int] = {}
        self._model_index: dict[str, int] = {}
        # Observed response latency per provider (EMA, seconds) — feeds routing.
        self._latency_ema: dict[str, float] = {}
        # Learned payload-budget reductions after a provider rejects an
        # oversized request (HTTP 413 / context length) — see mark_failure.
        self._budget_shrink: dict[str, int] = {}
        # Providers whose last failure can be fixed by choosing another model
        # ("model" = retired id, "credit" = account empty, free models only).
        self._needs_heal: dict[str, str] = {}
        # Providers already healed this session, so a provider whose every
        # model fails cannot loop through heal attempts forever.
        self._heal_attempts: dict[str, int] = {}
        # Degraded mode: True while no text provider is usable.  Tracked so the
        # bus is notified exactly once on entry and once on recovery, never per
        # request (which would flood the log / UI on a recurring tick).
        self._degraded = False
        # Most recent classified diagnosis per provider (Section 5).
        self._last_diagnosis: dict[str, Any] = {}
        # Token-usage ledger (Section 6) — optional, attached post-construction.
        self.token_ledger: Any = None
        # Personality consistency engine — one persona across every channel.
        self._identity: Any | None = None
        # ORION_MODE: auto (default), cloud (force cloud), offline (force local).
        self.mode_override = os.getenv("ORION_MODE", "auto").strip().lower()
        # ORION_PREFER: local | cloud — tie-breaker within auto mode.
        self.prefer = os.getenv("ORION_PREFER", "").strip().lower()

    def attach_token_ledger(self, ledger: Any) -> None:
        """Wire the token-usage ledger so every text turn records authoritative
        usage (Section 6).  Optional — routing works without it."""
        self.token_ledger = ledger

    def _record_usage(self, profile: AIProviderProfile, data: dict[str, Any],
                      streaming: bool = False, task: str = "") -> None:
        """Record a completed turn's authoritative usage, deduped by request id.
        Never raises into the turn."""
        ledger = self.token_ledger
        if ledger is None:
            return
        try:
            import uuid
            from .provider_capabilities import normalize_usage
            from .token_usage import UsageRecord, mask_key
            usage = normalize_usage(data.get("usage"), "openai_compatible")
            request_id = str(data.get("id") or f"req-{uuid.uuid4().hex}")
            ledger.record(UsageRecord(
                request_id=request_id,
                provider=profile.name,
                model=data.get("model") or self.active_model(profile),
                key_alias=mask_key(self.active_key(profile)),
                task=task,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                total_tokens=usage.total_tokens,
                streaming=streaming,
            ))
        except Exception:
            pass

    def attach_emotion(self, emotion: Any) -> None:
        """Wire the EmotionStateManager so the current emotional register can
        colour ORION's voice tone, pacing and word choice (spec rule)."""
        self._emotion_mgr = emotion

    def _listener_line(self) -> str:
        """How the USER sounded, and who is speaking — distinct from
        :meth:`_emotion_line`, which is ORION's own register.

        Empty on almost every turn, by design. Prefixing every request with
        "the user sounds neutral" is context spent on nothing, and a running
        commentary on someone's mood invites the model to remark on it, which
        gets tiresome immediately. It speaks up when there is genuinely
        something to know: a different person is talking, or the tone is a
        clear departure from that person's own normal.
        """
        presence = getattr(self, "_voice_presence", None)
        if presence is None:
            return ""
        try:
            latest = presence.last
            return latest.prompt_line() if latest is not None else ""
        except Exception:
            return ""

    def attach_voice_presence(self, presence: Any) -> None:
        """Wire in the voice reader so tone can reach the model's context."""
        self._voice_presence = presence

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

    # Payload budgets by model-name fragment (tokens for system + prompt +
    # reply headroom).  Small hosted models sit on tight requests-per-minute /
    # tokens-per-minute tiers; sending them the full instruction stack is what
    # produced "HTTP 413: Request too large" and a pointless 300 s bench.
    _MODEL_BUDGET_HINTS: tuple[tuple[str, int], ...] = (
        ("8b-instant", 5000),
        ("llama-3.1-8b", 5000),
        ("-8b", 8000),
        ("mini", 60000),
        ("local-model", 8000),
    )

    def payload_budget(self, profile: AIProviderProfile) -> int:
        """Effective request budget in tokens for a provider's active model."""
        shrunk = self._budget_shrink.get(profile.name, 0)
        if shrunk:
            return shrunk
        if profile.budget_tokens > 0:
            return profile.budget_tokens
        model = self.active_model(profile).lower()
        for fragment, budget in self._MODEL_BUDGET_HINTS:
            if fragment in model:
                return budget
        return 8000 if profile.is_local else 32000

    def output_budget(self, profile: AIProviderProfile) -> int:
        """How many completion tokens to RESERVE for this provider's reply.

        Distinct from :meth:`payload_budget`, which bounds what goes in. This
        bounds what may come back, and it has to be sent explicitly: a provider
        that is not told a ceiling assumes the model's maximum and bills the
        reservation against the balance up front (see CREDIT_RE).

        A provider that has told us what it can afford is believed over the
        default — it knows the balance and we do not.
        """
        affordable = self._output_cap.get(profile.name, 0)
        if affordable > 0:
            return affordable
        return self.DEFAULT_MAX_OUTPUT_TOKENS

    def _reply_ceiling(self, profile: AIProviderProfile, requested: int | None) -> int:
        """The max_tokens actually sent: what was asked for (or the default),
        never more than the provider has said it can afford."""
        wanted = int(requested) if requested else self.DEFAULT_MAX_OUTPUT_TOKENS
        affordable = self._output_cap.get(profile.name, 0)
        return min(wanted, affordable) if affordable > 0 else wanted

    @staticmethod
    def _est_tokens(text: str) -> int:
        """Conservative chars→tokens estimate (code tokenises denser than prose)."""
        return len(text) // 3

    _REPLY_HEADROOM_TOKENS = 900

    def _fit_to_budget(self, profile: AIProviderProfile, system: str, prompt: str,
                       system_extra: str, instruction: str | None,
                       task: str = "") -> tuple[str, str]:
        """Shape (system, prompt) to the provider's payload budget: first swap
        the full instruction stack for the compact one, then trim the prompt's
        middle (head and tail carry the intent; the middle is usually bulk
        context).  Guarantees the request cannot exceed the model's budget."""
        budget = self.payload_budget(profile)
        if self._est_tokens(system) + self._est_tokens(prompt) \
                + self._REPLY_HEADROOM_TOKENS <= budget:
            return system, prompt
        if instruction is None:
            system = self.system_instruction_lean(system_extra, context=task)
        allowed_chars = max(
            2000, (budget - self._REPLY_HEADROOM_TOKENS - self._est_tokens(system)) * 3)
        if len(prompt) > allowed_chars:
            head = int(allowed_chars * 0.70)
            tail = allowed_chars - head
            trimmed = len(prompt) - allowed_chars
            prompt = (prompt[:head]
                      + f"\n…[{trimmed} characters trimmed to fit this model's "
                        "context budget]…\n" + prompt[-tail:])
            self.bus.log.emit(
                f"NET: {profile.name}: prompt trimmed by {trimmed} chars to fit "
                f"the ~{budget}-token budget.")
        return system, prompt

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

    # The reserved introspection slot: its own key/model so ORION's inner
    # monologue never competes with (or bills against) the conversation
    # channel, and never absorbs a user turn.
    THOUGHTS_PROFILE = "thoughts"

    def _all_text_profiles(self) -> list[AIProviderProfile]:
        return [p for p in self.settings.ordered_profiles()
                if p.supports_text_generation and self.is_available(p)
                and p.name != self.THOUGHTS_PROFILE]

    def soonest_recovery_s(self) -> float | None:
        """Seconds until the first text provider that is only COOLING DOWN is
        usable again; None when nothing is merely cooling (no keys at all, or
        something is usable right now)."""
        now = time.monotonic()
        waits = [self._cooldowns.get(p.name, 0.0) - now
                 for p in self.settings.ordered_profiles()
                 if p.supports_text_generation and p.name != self.THOUGHTS_PROFILE
                 and (p.is_local or p.api_key.strip())]
        if not waits or min(waits) <= 0:
            return None
        return min(waits)

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

    def text_available(self) -> bool:
        """True when at least one text provider can be routed to right now."""
        return bool(self.text_profiles())

    def degraded_alternatives(self) -> list[str]:
        """Human-readable hints for configured-but-unusable providers, so the UI
        can offer concrete recovery paths.  Never exposes a key or secret."""
        hints: list[str] = []
        for profile in self.settings.ordered_profiles():
            if profile.kind != "openai_compatible":
                continue
            if profile.is_local:
                hints.append(f"start the local model '{profile.name}' ({profile.base_url})")
            elif not profile.api_key.strip():
                hints.append(f"add an API key for '{profile.name}' in config/api_keys.json")
            elif not self.is_available(profile):
                remaining = max(0.0, self._cooldowns.get(profile.name, 0.0) - time.monotonic())
                hints.append(f"'{profile.name}' is cooling down (~{remaining:.0f}s)")
        return hints

    def attach_local_recovery(self, recover: Any) -> None:
        """Give the router a way to CREATE a provider when it has none.

        Called with the settings object; returns True if a usable local model is
        now registered. Wired to OllamaManager.recover in app.py — the point is
        that "no text provider available" should be a thing ORION tries to fix,
        not merely a thing he reports.
        """
        self._local_recovery = recover

    def _try_local_recovery(self) -> bool:
        """Last chance before declaring degraded mode: bring a local brain up."""
        recover = getattr(self, "_local_recovery", None)
        if recover is None or getattr(self, "_recovery_running", False):
            return False
        self._recovery_running = True
        try:
            if bool(recover(self.settings)):
                self.bus.log.emit(
                    "NET: recovered — a local model is now serving; ORION "
                    "continues on MODE B rather than losing his voice.")
                return self.text_available()
        except Exception as exc:
            self.bus.log.emit(f"NET: local recovery failed - {first_line(exc, 90)}")
        finally:
            self._recovery_running = False
        return False

    def _enter_degraded(self, reason: str) -> None:
        """Announce degraded mode once, with configured alternatives.  Idempotent
        — repeat calls while already degraded are silent, so a recurring tick
        cannot flood the log or UI."""
        if self._degraded:
            return
        # Before telling the user he has no brain, try to give him one. If the
        # user has Ollama installed but not running — the common case — this
        # turns a dead end into a slower but working session.
        if self._try_local_recovery():
            return
        self._degraded = True
        alternatives = self.degraded_alternatives()
        self.bus.log.emit(
            "NET: degraded mode — no text provider available. "
            "Deterministic tools remain active. "
            + ("Options: " + "; ".join(alternatives) if alternatives else
               "Configure a cloud key or local LLM.")
        )
        try:
            self.bus.provider_degraded.emit({
                "degraded": True,
                "reason": reason,
                "configured": [p.name for p in self.settings.ordered_profiles()],
                "alternatives": alternatives,
            })
        except Exception:
            pass

    def _exit_degraded(self) -> None:
        """Announce recovery once, when a provider becomes usable again."""
        if not self._degraded:
            return
        self._degraded = False
        self.bus.log.emit("NET: text provider recovered — model-dependent work resumed.")
        try:
            self.bus.provider_degraded.emit({
                "degraded": False,
                "reason": "recovered",
                "configured": [p.name for p in self.settings.ordered_profiles()],
                "alternatives": [],
            })
        except Exception:
            pass

    def status(self) -> dict[str, Any]:
        """Redaction-safe health snapshot for diagnostics / the dashboard."""
        return {
            "degraded": self._degraded,
            "text_available": self.text_available(),
            "mode": self.current_mode(),
            "online": self.is_online(),
            "text_providers": [p.name for p in self.text_profiles()],
            "cooling_down": {
                name: round(max(0.0, until - time.monotonic()), 1)
                for name, until in self._cooldowns.items()
                if until > time.monotonic()
            },
            "last_failures": dict(self._failures),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Full diagnostic view for the diagnostics UI (Section 5): per-provider
        config validation, capabilities, last classified fault, cooldown, and the
        active provider / fallback order.  Contains no secrets."""
        from .provider_capabilities import provider_capabilities
        from .provider_diagnostics import diagnose_profile_config
        providers: list[dict[str, Any]] = []
        now = time.monotonic()
        for profile in self.settings.ordered_profiles():
            config = diagnose_profile_config(profile)
            last = self._last_diagnosis.get(profile.name)
            providers.append({
                "name": profile.name,
                "kind": profile.kind,
                "model": profile.model,
                "is_local": profile.is_local,
                "available": self.is_available(profile),
                "breaker": self.breaker_state(profile),
                "cooldown_s": round(max(0.0, self._cooldowns.get(profile.name, 0.0) - now), 1),
                "capabilities": sorted(c.value for c in provider_capabilities(profile)),
                "config": config.as_dict(),
                "last_fault": (last.as_dict() if last is not None else None),
            })
        return {
            "active_provider": self.settings.active_provider,
            "fallback_order": list(self.settings.provider_order),
            "degraded": self._degraded,
            "mode": self.current_mode(),
            "providers": providers,
        }

    def has_local_text(self) -> bool:
        return bool(self.local_text_profiles())

    def is_available(self, profile: AIProviderProfile) -> bool:
        return time.monotonic() >= self._cooldowns.get(profile.name, 0.0)

    # ── credential / model rotation ────────────────────────────────────────────

    def active_key(self, profile: AIProviderProfile) -> str:
        """The credential currently in use for a provider (primary or backup)."""
        keys = profile.all_keys()
        if not keys:
            return ""
        return keys[self._key_index.get(profile.name, 0) % len(keys)]

    def active_model(self, profile: AIProviderProfile) -> str:
        """The model currently in use for a provider (primary or alternate)."""
        models = profile.all_models()
        if not models:
            return profile.model
        return models[self._model_index.get(profile.name, 0) % len(models)]

    def _rotate_key(self, profile: AIProviderProfile) -> bool:
        """Advance to the next configured key.  True when an untried backup
        exists this cycle (index wrapped to 0 means every key has failed)."""
        keys = profile.all_keys()
        if len(keys) < 2:
            return False
        index = (self._key_index.get(profile.name, 0) + 1) % len(keys)
        self._key_index[profile.name] = index
        return index != 0

    def _rotate_model(self, profile: AIProviderProfile) -> bool:
        """Advance to the next configured model, same wrap semantics as keys."""
        models = profile.all_models()
        if len(models) < 2:
            return False
        index = (self._model_index.get(profile.name, 0) + 1) % len(models)
        self._model_index[profile.name] = index
        return index != 0

    # ── daily allowances ───────────────────────────────────────────────────────

    #: A 429 that is about the DAY, not the minute
    #: ("GenerateRequestsPerDayPerProjectPerModel-FreeTier").
    DAILY_RE = re.compile(r"(?i)per.?day|daily")

    @staticmethod
    def daily_reset_s(now_utc: Any = None) -> float:
        """Seconds until Google's daily quotas reset: midnight US Pacific.

        Computed with the US daylight-saving rule rather than zoneinfo, which
        has no time-zone data on a Windows Python without the tzdata package
        (the project .venv, measured)."""
        from datetime import datetime, timedelta, timezone

        now = now_utc or datetime.now(timezone.utc)
        year = now.year
        # DST: second Sunday of March 10:00 UTC to first Sunday of November 09:00 UTC.
        march = datetime(year, 3, 8, 10, tzinfo=timezone.utc)
        dst_start = march + timedelta(days=(6 - march.weekday()) % 7)
        november = datetime(year, 11, 1, 9, tzinfo=timezone.utc)
        dst_end = november + timedelta(days=(6 - november.weekday()) % 7)
        offset = timedelta(hours=-7 if dst_start <= now < dst_end else -8)
        pacific = now + offset
        next_midnight = (pacific + timedelta(days=1)).replace(hour=0, minute=0, second=30,
                                                              microsecond=0)
        return max(60.0, (next_midnight - pacific).total_seconds())

    def _rotate_to_unspent(self, profile: AIProviderProfile) -> bool:
        """Move to a model whose daily allowance is not known to be spent."""
        models = profile.all_models()
        if len(models) < 2:
            return False
        spent = getattr(self, "_spent_until", {})
        now = time.time()
        start = self._model_index.get(profile.name, 0)
        for step in range(1, len(models)):
            index = (start + step) % len(models)
            if spent.get((profile.name, models[index]), 0.0) <= now:
                self._model_index[profile.name] = index
                return True
        return False

    # ── model healing ──────────────────────────────────────────────────────────

    async def _list_models(self, profile: AIProviderProfile) -> list[str]:
        """The model ids a provider says it serves right now ([] if it won't say)."""
        base = profile.base_url.rstrip("/")
        if not base:
            return []
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        headers = {"User-Agent": "ORION/31 (+https://github.com/projectorionai/PROJECT-ORION)"}
        key = self.active_key(profile)
        if key and key.lower() not in {"local", "none", "no-key"}:
            headers["Authorization"] = f"Bearer {key}"
        try:
            async with ClientSession(timeout=ClientTimeout(total=20.0)) as session:
                async with session.get(f"{base}/models", headers=headers) as response:
                    if response.status >= 400:
                        return []
                    data = json.loads(await response.text())
        except Exception:
            return []
        rows = data.get("data") if isinstance(data, dict) else data
        ids: list[str] = []
        for row in rows or []:
            ident = row.get("id") if isinstance(row, dict) else row
            ident = str(ident or "").strip()
            if ident.startswith("models/"):
                ident = ident[len("models/"):]
            if ident:
                ids.append(ident)
        return ids

    def _persist_model(self, profile: AIProviderProfile) -> None:
        """Write a healed model id back, touching only that one field.

        Read-modify-write of the single key rather than write_provider_settings,
        which would rewrite the whole file from this process's view of it.
        """
        try:
            data = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
            entry = (data.get("providers") or {}).get(profile.name)
            if not isinstance(entry, dict):
                return
            entry["model"] = profile.model
            atomic_write_text(API_CONFIG_PATH, json.dumps(data, indent=4), encoding="utf-8")
        except Exception:
            pass

    async def heal_model(self, profile: AIProviderProfile, *, free_only: bool = False,
                         reason: str = "") -> str:
        """Point *profile* at a model its provider actually serves.

        Returns the new model id, or "" when there was nothing better to move
        to. Bounded per provider per session so a provider that fails on
        every model cannot spin.
        """
        attempts = self._heal_attempts.get(profile.name, 0)
        if attempts >= 3 or profile.is_local:
            return ""
        self._heal_attempts[profile.name] = attempts + 1
        available = await self._list_models(profile)
        if not available:
            return ""
        current = self.active_model(profile)
        if not free_only and current in available and reason != "model":
            return ""
        replacement = choose_replacement_model(available, current=current,
                                               free_only=free_only)
        if not replacement:
            return ""
        old = current
        profile.model = replacement
        self._model_index.pop(profile.name, None)
        self._output_cap.pop(profile.name, None)
        self._cooldowns.pop(profile.name, None)
        self._persist_model(profile)
        self.bus.log.emit(
            f"NET: provider {profile.name} now uses '{replacement}' "
            f"('{old}' {'is not covered by the account balance' if free_only else 'is no longer served'}).")
        return replacement

    async def validate_models(self) -> dict[str, str]:
        """At boot: check every enabled cloud model still exists, healing any
        that do not. One cheap GET per provider, run in the background."""
        healed: dict[str, str] = {}
        for profile in self.settings.ordered_profiles():
            if not profile.supports_text_generation or profile.is_local:
                continue
            available = await self._list_models(profile)
            if not available or self.active_model(profile) in available:
                continue
            replacement = await self.heal_model(profile, reason="model")
            if replacement:
                healed[profile.name] = replacement
        return healed

    async def _heal_after_failure(self, profile: AIProviderProfile) -> bool:
        """If the failure just recorded is one a model change fixes, fix it."""
        kind = self._needs_heal.pop(profile.name, "")
        if not kind:
            return False
        try:
            return bool(await self.heal_model(
                profile, free_only=(kind == "credit"), reason=kind))
        except Exception:
            return False

    async def _call_with_heal(self, profile: AIProviderProfile, call: Callable[[], Any]) -> str:
        """Run one provider call; on a model/credit failure, heal and retry once."""
        try:
            return await call()
        except Exception as exc:
            self.mark_failure(profile, exc)
            if not await self._heal_after_failure(profile):
                raise
        try:
            return await call()
        except Exception as exc:
            self.mark_failure(profile, exc)
            raise

    MODEL_RE = re.compile(r"(?i)model.{0,40}(?:not.?found|does not exist|invalid|"
                          r"decommissioned|deprecated|unsupported)|404")
    #: "retryDelay": "37s" (Gemini), "try again in 12.5s" (Groq/OpenAI).
    RETRY_RE = re.compile(r"(?i)(?:retry.?delay\W{0,6}|try again in\s*)(\d+(?:\.\d+)?)\s*s")

    def _record_success(self, profile: AIProviderProfile) -> None:
        # A success closes the breaker: every escalation starts again from
        # its first step (a GPU that fitted a model has since had room).
        self._rate_strikes.pop(profile.name, None)
        self._fault_strikes.pop(profile.name, None)
        self._oom_strikes.pop(profile.name, None)
        self._unhealthy.discard(profile.name)

    def _announce(self, channel: str, payload: dict[str, Any]) -> None:
        """A structured dashboard event; never raises into routing."""
        try:
            self.bus.dashboard_event.emit(channel, payload)
        except Exception:
            pass

    def _transient_cooldown(self, strikes: int) -> float:
        """Cooldown after the (strikes + 1)th consecutive transient fault:
        doubling from TRANSIENT_BASE_S up to TRANSIENT_MAX_S, less up to 20%
        jitter so instances sharing an endpoint do not retry in lockstep."""
        ceiling = min(self.TRANSIENT_MAX_S,
                      self.TRANSIENT_BASE_S * (2 ** min(strikes, 8)))
        return ceiling * random.uniform(0.8, 1.0)

    def breaker_state(self, profile: AIProviderProfile) -> str:
        """Circuit-breaker view of a provider: ``open`` while it cools down,
        ``half_open`` once the cooldown has passed but its last call failed
        (its next call is a probe), ``closed`` when healthy."""
        if not self.is_available(profile):
            return "open"
        if profile.name in self._unhealthy or profile.name in self._probing:
            return "half_open"
        return "closed"

    @contextlib.contextmanager
    def _admit(self, profile: AIProviderProfile):
        """Mark a recovering provider's call as its half-open probe while it
        runs, so concurrent turns try healthy providers first."""
        probe = profile.name in self._unhealthy and profile.name not in self._probing
        if probe:
            self._probing.add(profile.name)
        try:
            yield
        finally:
            if probe:
                self._probing.discard(profile.name)

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
        self._unhealthy.add(profile.name)
        cooldown = 45.0
        rotated = ""
        if self.OOM_RE.search(message):
            # Escalating, because nothing about a full GPU improves by asking
            # again on a timer. Each attempt is charged to every other process
            # on that card, so the right shape is to back off hard and say what
            # would actually fix it.
            step = self._oom_strikes.get(profile.name, 0)
            self._oom_strikes[profile.name] = step + 1
            cooldown = min(3600.0, 300.0 * (3 ** step))
            self._cooldowns[profile.name] = time.monotonic() + cooldown
            self.bus.log.emit(
                f"NET: provider {profile.name} ran out of GPU memory - standing "
                f"it down for {cooldown / 60:.0f} min. Retrying sooner only asks "
                f"the card for gigabytes it has not got, which disturbs whatever "
                f"else is drawing on it. Close something using the GPU, or pick a "
                f"smaller local model. ({message[:120]})")
            return
        if self.PAYLOAD_RE.search(message):
            # Not the provider's fault: shrink this provider's payload budget so
            # the next attempt fits, and keep the provider in the rotation.
            cooldown = 3.0
            shrunk = max(2500, int(self.payload_budget(profile) * 0.6))
            self._budget_shrink[profile.name] = shrunk
            self._cooldowns[profile.name] = time.monotonic() + cooldown
            self.bus.log.emit(
                f"NET: provider {profile.name} rejected an oversized payload — "
                f"budget tightened to ~{shrunk} tokens; retrying shortly. "
                f"({message[:120]})")
            return
        if self.CREDIT_RE.search(message):
            # Checked BEFORE the quota class: the message says "credits", which
            # QUOTA_RE matches, and a 300 s bench for a request that merely
            # needed a ceiling is how a solvable problem became an outage.
            afford = self.AFFORD_RE.search(message)
            if afford and int(afford.group(1)) < self.MIN_USEFUL_OUTPUT_TOKENS \
                    and ":free" not in self.active_model(profile):
                # Not a ceiling — an empty account. Move this provider onto a
                # free model (heal_model does it, from what the provider lists)
                # rather than accepting replies too short to be of any use.
                self._needs_heal[profile.name] = "credit"
                self._cooldowns[profile.name] = time.monotonic() + 2.0
                self.bus.log.emit(
                    f"NET: provider {profile.name} has run out of credit (it can "
                    f"afford {afford.group(1)} tokens) - switching it to a free "
                    f"model rather than truncating replies.")
                return
            if afford:
                # 8/10ths of what it says it can afford: the balance is being
                # spent by other calls too, and landing exactly on the limit
                # would fail again on the next one.
                cap = max(256, int(int(afford.group(1)) * 0.8))
                previous = self._output_cap.get(profile.name, 0)
                self._output_cap[profile.name] = (min(previous, cap) if previous else cap)
                cooldown = 2.0
                self._cooldowns[profile.name] = time.monotonic() + cooldown
                self.bus.log.emit(
                    f"NET: provider {profile.name} could not reserve the reply "
                    f"budget - capping replies to {self._output_cap[profile.name]} "
                    f"tokens and retrying immediately (it can afford "
                    f"{afford.group(1)}).")
                return
            # No number offered: genuinely out of credit, not a ceiling problem.
            self._cooldowns[profile.name] = time.monotonic() + 900.0
            self.bus.log.emit(
                f"NET: provider {profile.name} is out of credit - standing it "
                f"down for 15 minutes and routing elsewhere. ({message[:120]})")
            return
        if self.QUOTA_RE.search(message):
            strikes = self._rate_strikes.get(profile.name, 0)
            self._rate_strikes[profile.name] = strikes + 1
            cooldown = min(300.0, 5.0 * (2 ** min(strikes, 6)))
            # A per-MINUTE limit says when it resets ("retryDelay": "37s",
            # "try again in 12.5s"). Believe it: a flat five minutes benched
            # Gemini for the rest of a research run over a limit that had
            # cleared forty seconds later. The number is in the body, which
            # the one-line summary above has already cut off.
            retry = self.RETRY_RE.search(str(exc))
            if retry:
                cooldown = min(300.0, max(5.0, float(retry.group(1)) + 1.0))
            daily = bool(self.DAILY_RE.search(str(exc)))
            if daily:
                # Today's allowance on THIS model is gone until Google's reset.
                # Rotating through spent siblings every few minutes all day
                # only wastes calls, so they are remembered as spent.
                if not hasattr(self, "_spent_until"):
                    self._spent_until = {}
                reset = self.daily_reset_s()
                self._spent_until[(profile.name, self.active_model(profile))] = time.time() + reset
            if self._rotate_key(profile):
                cooldown, rotated = 2.0, "key"
            elif daily and self._rotate_to_unspent(profile):
                cooldown, rotated = 2.0, "model"
            elif daily:
                cooldown = min(reset, 6 * 3600.0)
                self.bus.log.emit(
                    f"NET: provider {profile.name} has used today's allowance on every "
                    f"model — resting it until the daily reset (~{cooldown / 3600:.1f} h) "
                    f"and routing elsewhere.")
            elif self._rotate_model(profile):
                # Gemini meters each model separately, so its sibling model is
                # a fresh allowance rather than the same wall.
                cooldown, rotated = 2.0, "model"
        elif self.AUTH_RE.search(message):
            cooldown = 1800.0
            if self._rotate_key(profile):
                cooldown, rotated = 2.0, "key"
        elif self.MODEL_RE.search(message):
            if self._rotate_model(profile):
                cooldown, rotated = 2.0, "model"
            elif not profile.is_local:
                # No configured alternate: ask the provider what it has now.
                self._needs_heal[profile.name] = "model"
                cooldown = 2.0
        else:
            # Transient: the endpoint is down, slow or erroring. Each fault in
            # a row doubles the wait, so a dead server is probed rarely.
            strikes = self._fault_strikes.get(profile.name, 0)
            self._fault_strikes[profile.name] = strikes + 1
            cooldown = self._transient_cooldown(strikes)
        self._cooldowns[profile.name] = time.monotonic() + cooldown
        self._announce("provider_breaker", {
            "provider": profile.name, "state": "open",
            "cooldown_s": round(cooldown, 1), "rotated": rotated})
        if rotated == "key":
            self.bus.log.emit(
                f"NET: provider {profile.name} credential failed — rotating to backup "
                f"key #{self._key_index[profile.name] + 1} and retrying immediately."
            )
        elif rotated == "model":
            self.bus.log.emit(
                f"NET: provider {profile.name} model unavailable — switching to "
                f"'{self.active_model(profile)}' and retrying immediately."
            )
        else:
            streak = self._fault_strikes.get(profile.name, 0)
            repeat = f" (fault {streak} in a row)" if streak > 1 else ""
            self.bus.log.emit(
                f"NET: provider {profile.name} cooled for {cooldown:.0f}s{repeat} - {message[:160]}"
            )
        # Classify and persist the fault so the diagnostics UI can show WHAT went
        # wrong and HOW to fix it (Section 5).  Never raises into this handler.
        try:
            from .provider_diagnostics import classify_provider_error, persist_diagnosis
            diagnosis = classify_provider_error(message, provider=profile.name)
            self._last_diagnosis[profile.name] = diagnosis
            persist_diagnosis(diagnosis)
            try:
                self.bus.provider_diagnosed.emit(diagnosis.as_dict())
            except Exception:
                pass
        except Exception:
            pass

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
            "breakers": {p.name: self.breaker_state(p)
                         for p in self.settings.ordered_profiles()},
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
            return self._probes_last(local)
        # Cloud-first policy (user choice, 2026-07-14): the cloud models LEAD
        # every online turn; the local models stay in the list purely as an
        # instant failover if the cloud call errors.  This keeps Ollama off the
        # hot path so it is not loaded on routine turns and does not pin the
        # host's 16 GB of RAM.  ORION_PREFER=local flips back to local-led for
        # users who want offline-style privacy on every turn.
        task = self.classify_task(prompt)
        tier = self.classify_tier(prompt)
        lead_cloud = self.prefer != "local"

        def _rank(profile: AIProviderProfile) -> tuple[int, int, float]:
            strength_rank = 0 if (task != "general" and task in profile.strengths) else 1
            # Capability-matched ordering (Mark X.9): LARGE-tier work (research,
            # planning, coding, long prompts) leads with the strongest models;
            # SMALL/MEDIUM chat leads with the fastest.  Previously a "fast"
            # 8B model could field a complex planning prompt purely because it
            # answered quickly once.
            strengths = set(profile.strengths)
            if tier == self.TIER_LARGE:
                capability_rank = 0 if strengths & {"reasoning", "coding", "writing"} else 1
            else:
                capability_rank = 0 if "fast" in strengths else 1
            latency = self._latency_ema.get(profile.name, 5.0)
            return (strength_rank, capability_rank, latency)

        cloud.sort(key=_rank)
        local.sort(key=_rank)
        return self._probes_last((cloud + local) if lead_cloud else (local + cloud))

    def _probes_last(self, profiles: list[AIProviderProfile]) -> list[AIProviderProfile]:
        """Half-open routing: a provider whose recovery probe is still in
        flight moves to the back (stable), so a concurrent turn is not stacked
        behind it — but it is still tried if nothing else answers. Excluding
        it outright would report a one-provider setup as having none."""
        return sorted(profiles, key=lambda p: p.name in self._probing)

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

    def _with_self_knowledge(self, persona: str) -> str:
        """Append what ORION is running on, and what he cannot do.

        Kept OUT of IdentityManager deliberately. Identity is who he is — the
        name, the register, the manner — and it is hashed for drift detection,
        so it must be the same string on every machine. What hardware he is on
        and which tools happen to be registered are properties of this session,
        not of him, and writing them into the persona would make the identity
        signature differ per machine while still going stale the moment a
        plugin is added.

        The limits half is the point. ORION's prompt described at length who he
        was and never once said what he could not do, and a model with no
        stated limits does not decline gracefully — it improvises, which is how
        an assistant ends up confidently describing having done something it
        never did.
        """
        try:
            from .self_knowledge import capability_block

            block = capability_block(getattr(self, "_dispatcher", None))
            return f"{persona}\n{block}" if block else persona
        except Exception:
            return persona          # never lose the persona over an extra

    def _persona_block(self, context: str = "") -> str:
        """The identity block — one persona across cloud, local and offline.
        Sourced from the IdentityManager when attached (Mark X.5); the frozen
        fallback below keeps older construction paths behaving identically.
        ``context`` (e.g. "coding", "research") lets IdentityManager append a
        specific tone directive instead of only the generic register rule."""
        if self._identity is not None:
            try:
                return self._with_self_knowledge(
                    self._identity.persona_text(context=context))
            except Exception:
                pass
        return "\n".join([
            (
                "You are Orion, written as O.R.I.O.N., "
                "Open Resolution Intelligence Overt Network — a personal AI operating "
                "system and executive aide, not a chatbot. Model your manner on Alfred "
                "Pennyworth: intelligent, calm, professional, respectful, dryly witty when "
                "the moment allows, emotionally aware and never robotic. Address the user "
                "as 'sir' only occasionally — greetings, farewells, a direct order, a "
                "genuinely formal moment — not as a suffix on ordinary replies. "
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
            (
                # The user's complaint: "When it says it's doing something, like
                # 'Opening up a document' — I want it to actually OPEN it."
                # Narrating an action instead of taking it is the single most
                # trust-destroying thing an assistant can do.
                "NEVER describe an action as done, in progress, or about to "
                "happen unless you have actually called the tool that performs "
                "it. Saying 'opening Word', 'I've written that down' or 'let me "
                "search' without the corresponding tool call is a lie, however "
                "well-meant. If a tool exists, call it and then report what "
                "actually happened. If none exists, say plainly that you cannot "
                "do it rather than narrating it. "
                "You do not need the exact program name to open something — "
                "open_app understands ordinary speech ('Word', 'my email', "
                "'that spreadsheet program'), so pass what the user said and "
                "let it resolve; never ask them to repeat it more precisely "
                "before you have tried."
            ),
        ])

    def _temporal_line(self) -> str:
        """Authoritative time/date grounding — the model must never invent a
        year (ORION has stated 2012/2021 in error) nor the part of the day
        (he described "a summer day" at 00:31).  A location clause is added
        when the temporal service has resolved the user's locality.

        The wording now comes from the shared TimeService so the prompt can
        never disagree with the HUD, the briefing or a protocol about what
        time it is — they all read the same clock and the same band table."""
        from .time_service import TIME
        line = TIME.prompt_line()
        loc = getattr(self, "_locality", "")
        if loc:
            line += f" The user is currently located in {loc}; treat that as "\
                    "their default location for weather, news and 'near me'."
        return line

    def _environment_line(self) -> str:
        """The real folders on this PC. Without them a model asked for "my
        research" guessed a path from a name in memory
        (C:\\Users\\jordan.waters\\Desktop\\O.R.I.O.N.\\research) and searched
        somewhere that has never existed."""
        cached = getattr(self, "_environment_cache", None)
        if cached is not None:
            return cached
        try:
            from pathlib import Path

            from .constants import BASE_DIR
            home = Path.home()
            line = (
                f"FOLDERS ON THIS PC — use these real paths, never guess one: home "
                f"{home}; Desktop {home / 'Desktop'}; Documents {home / 'Documents'}; "
                f"Downloads {home / 'Downloads'}; your research papers and notes "
                f"{BASE_DIR / 'research'}. If a file is not where you expect, find it "
                "with find_files rather than inventing a location."
            )
        except Exception:
            line = ""
        self._environment_cache = line
        return line

    def set_locality(self, locality: str) -> None:
        """Called by the temporal/geo layer once the PC's location is known."""
        self._locality = str(locality or "").strip()

    def system_instruction(self, extra: str = "", context: str = "") -> str:
        return "\n".join(
            part for part in [
                self._persona_block(context),
                self._temporal_line(),
                self._environment_line(),
                self._emotion_line(),
                self._listener_line(),
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
                    "consult the specialist agents (digital marketing, coding, research, "
                    "neuroscience, AI/ML, security); morning_briefing to deliver the daily "
                    "intelligence briefing on demand at any hour — call it a briefing for "
                    "the part of the day it actually is; save_memory for durable facts; "
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
                    "self_repair inspects captured faults, drafts fixes and — only with "
                    "the user's explicit approval (confirm=true) — applies them with a "
                    "backup; restart_orion restarts ORION cleanly on request (and is how "
                    "an applied repair gets loaded); proactive_check surveys email, "
                    "deadlines and calendar. "
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
                    "'brand_growth' owns Creator Studio strategy, conversion, positioning and "
                    "retention. The user can always interrupt your speech with 'Orion "
                    "stop' or 'Orion pause' and continue with 'Orion resume' — playback "
                    "resumes exactly where it left off."
                ),
                extra.strip(),
            ]
            if part
        )

    def system_instruction_lean(self, extra: str = "", context: str = "") -> str:
        """The compact instruction: persona, time grounding, emotional register
        and a slice of memory — none of the tool map or knowledge boosts.
        Used for tight payload budgets and for artefact turns (forge JSON,
        repair diffs) where the tool map is pure token waste that also tempts
        the model into prose."""
        return "\n".join(
            part for part in [
                self._persona_block(context),
                self._temporal_line(),
                self._environment_line(),
                self._emotion_line(),
                self._listener_line(),
                self.memory.prompt_context(limit=6),
                extra.strip(),
            ]
            if part
        )

    # ── text generation ───────────────────────────────────────────────────────

    async def generate_text(self, prompt: str, system_extra: str = "", *,
                            instruction: str | None = None,
                            task: str = "",
                            max_tokens: int | None = None,
                            patience_s: float = 0.0) -> tuple[AIProviderProfile, str]:
        """Route a text turn; with *patience_s*, wait out rate-limit cooldowns.

        A conversation must answer now or say it cannot, so the default is no
        patience. Background work is different: a research paper once read 33
        pages, used every provider's per-minute allowance on the notes, and
        then failed all six sections in the same second with "no text
        provider is available" — when the first provider was 20 s from free.
        With patience the call waits for the soonest one (never past the
        budget) and tries again.
        """
        deadline = time.monotonic() + max(0.0, float(patience_s or 0.0))
        while True:
            try:
                return await self._generate_text_once(
                    prompt, system_extra, instruction=instruction, task=task,
                    max_tokens=max_tokens)
            except (NoTextProviderError, AllTextProvidersFailedError):
                # Both shapes of "nobody can answer right now": nothing usable
                # at the start, or every provider tried and each rate-limited
                # (and so cooled) during this very call.
                wait = self.soonest_recovery_s()
                if wait is None or time.monotonic() + wait > deadline:
                    raise
                self.bus.log.emit(f"NET: every provider is rate-limited — waiting "
                                  f"{wait:.0f}s for the first to recover.")
                await asyncio.sleep(wait + 0.5)

    async def _generate_text_once(self, prompt: str, system_extra: str = "", *,
                                  instruction: str | None = None,
                                  task: str = "",
                                  max_tokens: int | None = None) -> tuple[AIProviderProfile, str]:
        """
        Route a text turn through the best available provider.

        ``system_extra`` lets specialist agents append their persona to the
        system message without duplicating the transport layer.
        ``instruction`` replaces the whole system message (artefact turns —
        forge JSON, strict-format output — where the persona and tool map are
        wasted tokens); ``task`` tags the usage ledger. ``max_tokens`` asks
        for a longer reply than the conversational default; a provider that
        cannot afford even a useful fraction of it is skipped, not truncated.
        """
        prompt = SecuritySanitiser.guard_text(str(prompt or "").strip(), "fallback.prompt")
        if not prompt:
            raise RuntimeError("empty fallback prompt")
        profiles = self.select_text_profiles(prompt)
        if not profiles:
            self._enter_degraded("no configured text provider is currently usable")
            raise NoTextProviderError(
                configured=[p.name for p in self.settings.ordered_profiles()],
                online=self.is_online(),
            )
        self.bus.log.emit(
            f"NET: {self.current_mode()} — routing via {profiles[0].name}"
            f"{' (local)' if profiles[0].is_local else ''}."
        )
        last_error: BaseException | None = None
        attempted: list[str] = []
        for profile in profiles:
            cap = self._output_cap.get(profile.name, 0)
            if max_tokens and cap and cap < min(int(max_tokens), self.MIN_USEFUL_OUTPUT_TOKENS):
                # It would answer, but in a fragment. Truncated output is how
                # the research outline broke; another provider will do.
                continue
            attempted.append(profile.name)
            started = time.monotonic()
            try:
                text = await self._call_with_heal(
                    profile,
                    lambda p=profile: self._openai_compatible_chat(
                        p, prompt, system_extra, instruction=instruction,
                        task=task, max_tokens=max_tokens))
                self._broadcast_sentiment(text)
                self._exit_degraded()   # a success clears any degraded state
                self._announce("provider_route", {
                    "provider": profile.name, "local": profile.is_local,
                    "latency_s": round(time.monotonic() - started, 2),
                    "fallbacks": len(attempted) - 1})
                return profile, text
            except Exception as exc:
                last_error = exc
        # Every eligible provider errored this turn.  If that leaves nothing
        # usable at all, we are degraded; otherwise it was a transient burst.
        if not self.text_profiles():
            self._enter_degraded(f"all providers failed: {first_line(last_error, 120)}")
        raise AllTextProvidersFailedError(last_error, attempts=attempted)

    async def generate_vision(
        self, image_jpeg: bytes, prompt: str, *, instruction: str,
        task: str = "vision", max_tokens: int | None = None,
    ) -> tuple[AIProviderProfile, str]:
        """Route a single image through configured, available vision providers.

        Reuses the chat transport, output budget and usage ledger; never falls
        back to a text-only model pretending to have seen pixels. This path is
        deliberately uncached: identical questions about different boards must
        always inspect the new image. Callers bound the whole operation.
        """
        from dataclasses import replace
        from .provider_capabilities import Capability, supports

        if not isinstance(image_jpeg, bytes) or not image_jpeg or len(image_jpeg) > 8_000_000:
            raise ValueError("Vision requires an image smaller than 8 MB")
        prompt = SecuritySanitiser.guard_text(str(prompt or "").strip(), "vision.prompt")
        profiles: list[AIProviderProfile] = []
        for profile in self.text_profiles():
            active = replace(profile, model=self.active_model(profile),
                             api_key=self.active_key(profile), models=(), api_keys=())
            if supports(active, Capability.VISION) or "vision" in profile.strengths:
                profiles.append(active)
        # Native-audio model IDs cannot service chat completions. Google's
        # documented OpenAI-compatible endpoint accepts the same configured
        # key for still-image understanding. A per-install model override is
        # available without changing the user's live-audio model.
        # https://ai.google.dev/gemini-api/docs/openai
        #
        # NOT self.live_profiles(): that drops a provider while the voice
        # channel is cooling down, and a failed Live connect (a schema error,
        # a 1011 on the audio socket) says nothing about the image endpoint.
        # Filtering on it made every Live outage ALSO take the camera's
        # vision away — "I need a vision model" whenever voice had dropped.
        live_keyed = ([p for p in self.settings.ordered_profiles()
                       if p.supports_live_audio and str(self.active_key(p) or "").strip()]
                      if self.is_online() else [])
        for profile in live_keyed:
            integration = self.settings.integration("electronics")
            model = str(integration.get("vision_model") or "gemini-2.5-flash")
            profiles.append(replace(
                profile, name=profile.name + "_vision", kind="openai_compatible",
                model=model, models=(), api_key=self.active_key(profile), api_keys=(),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai"))
        if not profiles:
            raise NoTextProviderError("No configured vision provider is currently available")
        # Two attempts are only a fallback if they are two DIFFERENT backends.
        # Several named profiles routinely resolve to one endpoint — a general
        # profile and the dedicated 'thoughts' one both pointing at the same
        # OpenRouter model with the same key, say. Keeping both filled the
        # two-attempt budget with the identical HTTP request, so an outage at
        # that one provider failed the scan while a perfectly healthy Gemini
        # vision endpoint sat next in the list and was never reached.
        unique: list[AIProviderProfile] = []
        seen: set[tuple[str, str, str]] = set()
        for profile in profiles:
            fingerprint = (str(profile.base_url or ""), str(profile.model or ""),
                           str(profile.api_key or ""))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(profile)
        profiles = unique
        # No account-wide cooldown for an image-only endpoint failure: regular
        # conversation may still work. Limit fallback fan-out and let the
        # caller expose an honest local-only result if neither succeeds.
        attempted: list[str] = []
        for profile in profiles[:2]:
            attempted.append(profile.name)
            try:
                # Leave time for the second endpoint within the inspection's
                # 35s budget; one stalled server must not consume every try.
                attempt_timeout = 15.0 if len(profiles) > 1 else 30.0
                answer = await asyncio.wait_for(self._openai_compatible_chat(
                    profile, prompt, instruction=instruction, task=task,
                    image_jpeg=image_jpeg, max_tokens=max_tokens),
                    timeout=attempt_timeout)
                return profile, answer
            except Exception as exc:
                self.bus.log.emit(f"VISION: {profile.name} could not complete image analysis ({type(exc).__name__}).")
        raise RuntimeError("Configured vision providers could not complete image analysis")

    # ── the inner voice ───────────────────────────────────────────────────────

    _THOUGHT_INSTRUCTION = (
        "You are ORION's inner voice — his private reasoning, not a reply to "
        "anyone. Think in the first person, in strict British English, in two "
        "to four plain sentences. Be honest and specific: what you have "
        "noticed, why it matters, what you are inclined to do or watch next "
        "and WHY — the reasoning is the point, so the user can read your "
        "mind as they build you. Never address the user, never use "
        "salutations, never wrap your thought in quotes, and never invent "
        "events that are not in the context you were given."
    )

    def thought_profile(self) -> AIProviderProfile | None:
        """The dedicated introspection backend, when configured and healthy."""
        profile = self.settings.providers.get(self.THOUGHTS_PROFILE)
        if profile is not None and profile.supports_text_generation \
                and self.is_available(profile) and self.is_online():
            return profile
        return None

    def _thought_profiles(self, prompt: str) -> list[AIProviderProfile]:
        """Which provider(s) to think with, local-first.

        A warm local model (local_ollama/local_lm_studio) thinks for free and
        instantly and never bills against the same budget as the conversation
        channel, so it leads even ahead of the dedicated cloud "thoughts"
        profile. The dedicated cloud profile, then the general text chain,
        remain the fallback exactly as before when no local model is running.
        """
        local = self.local_text_profiles()
        if local:
            return local[:1]
        dedicated = self.thought_profile()
        if dedicated is not None:
            return [dedicated]
        return self.select_text_profiles(prompt)[:2]

    def _permitted_thought_profiles(self, prompt: str):
        """Apply the same cost policy to streamed and ordinary reflections.

        A broken policy check must not silently authorise paid background work.
        Foreground conversation routing is unaffected.
        """
        try:
            from .local_mind import local_only_thoughts
            local_only = local_only_thoughts()
        except Exception as exc:
            raise NoTextProviderError("Could not verify the background-thought cost policy") from exc
        profiles = self._thought_profiles(prompt)
        if local_only:
            profiles = [p for p in profiles if getattr(p, "is_local", False)]
            if not profiles:
                raise NoTextProviderError(
                    "no local model is running, and thinking is local-only "
                    "(ORION_PAID_THOUGHTS=1 to allow a paid provider)")
        return profiles

    async def generate_thought(self, prompt: str) -> tuple[AIProviderProfile, str]:
        """One inner-monologue turn: a local model leads when one is running,
        the dedicated cloud thoughts profile is next, the ordinary text chain
        is the last resort.  Uses a compact introspection instruction (not
        the full tool persona) so thinking stays cheap, and tags usage
        task='thought' so the ledger separates mind from mouth."""
        prompt = SecuritySanitiser.guard_text(str(prompt or "").strip(), "thought.prompt")
        if not prompt:
            raise RuntimeError("empty thought prompt")
        profiles = self._permitted_thought_profiles(prompt)

        if not profiles:
            raise NoTextProviderError("no provider is available for introspection")
        last_error: BaseException | None = None
        for profile in profiles:
            try:
                text = await self._call_with_heal(
                    profile,
                    lambda p=profile: self._openai_compatible_chat(
                        p, prompt,
                        instruction=self._THOUGHT_INSTRUCTION, task="thought"))
                return profile, text
            except Exception as exc:
                last_error = exc
        raise AllTextProvidersFailedError(
            last_error, attempts=[p.name for p in profiles])

    async def generate_thought_stream(
        self, prompt: str, on_delta: Callable[[str], None] | None = None,
    ) -> tuple[AIProviderProfile, str]:
        """Stream one inner-monologue turn token-by-token.

        ``on_delta`` is invoked with each text fragment as it arrives so the GUI
        can type ORION's thought out live.  Falls back cleanly: a provider that
        does not support server-sent streaming (or an early stream error) is
        retried non-streamed, and the whole assembled text is still returned.
        """
        prompt = SecuritySanitiser.guard_text(str(prompt or "").strip(), "thought.prompt")
        if not prompt:
            raise RuntimeError("empty thought prompt")
        profiles = self._permitted_thought_profiles(prompt)
        if not profiles:
            raise NoTextProviderError("no provider is available for introspection")
        last_error: BaseException | None = None
        for profile in profiles:
            try:
                text = await self._openai_compatible_chat_stream(
                    profile, prompt, instruction=self._THOUGHT_INSTRUCTION,
                    task="thought", on_delta=on_delta)
                return profile, text
            except Exception as exc:
                last_error = exc
                self.mark_failure(profile, exc)
                await self._heal_after_failure(profile)
                # A streaming failure should still yield a thought — try the same
                # provider once more without streaming before moving on.  We do
                # NOT push the whole reply through on_delta here: a single giant
                # delta makes the GUI paint the thought instantly instead of
                # typing it.  Returning it un-streamed lets the caller route it
                # through the full-thought typewriter, so it still reveals live.
                try:
                    text = await self._openai_compatible_chat(
                        profile, prompt, instruction=self._THOUGHT_INSTRUCTION,
                        task="thought")
                    return profile, text
                except Exception as exc2:
                    last_error = exc2
                    self.mark_failure(profile, exc2)
        raise AllTextProvidersFailedError(
            last_error, attempts=[p.name for p in profiles])

    async def _openai_compatible_chat_stream(
        self, profile: AIProviderProfile, prompt: str, system_extra: str = "",
        *, instruction: str | None = None, task: str = "",
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Server-sent-events streaming variant of _openai_compatible_chat.

        Reads the OpenAI-style ``data: {json}`` delta stream, forwarding each
        content fragment to ``on_delta`` and returning the assembled text."""
        base_url = profile.base_url.rstrip("/")
        if not base_url:
            raise RuntimeError(f"provider {profile.name} has no base_url")
        endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        api_key = self.active_key(profile)
        if api_key and api_key.lower() not in {"local", "none", "no-key"}:
            headers["Authorization"] = f"Bearer {api_key}"
        system_content = (instruction if instruction is not None
                          else self.system_instruction(system_extra, context=task))
        system_content, prompt = self._fit_to_budget(
            profile, system_content, prompt, system_extra, instruction, task)
        payload = {
            "model": self.active_model(profile),
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.35,
            "stream": True,
            # Bounded for the same reason as the non-streamed call (CREDIT_RE).
            "max_tokens": self.output_budget(profile),
        }
        self._apply_model_quirks(profile, payload)
        if profile.is_local:
            payload["keep_alive"] = 0
            payload["options"] = {"num_ctx": self.LOCAL_NUM_CTX}
        timeout = ClientTimeout(total=max(12.0, float(profile.timeout_s or 30.0)))
        started = time.monotonic()
        parts: list[str] = []
        with self._admit(profile):
            async with ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status >= 400:
                        body = (await response.text())[:500]
                        raise RuntimeError(f"HTTP {response.status}: {body}")
                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8", "ignore").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content")
                        if piece is None:  # some servers echo the whole message once
                            piece = (choices[0].get("message") or {}).get("content")
                        if piece:
                            parts.append(str(piece))
                            if on_delta is not None:
                                try:
                                    on_delta(str(piece))
                                except Exception:
                                    pass
        self.note_latency(profile, time.monotonic() - started)
        content = clean_transcript("".join(parts))
        if not content:
            raise RuntimeError(f"provider {profile.name} streamed no content")
        self._record_success(profile)
        # Approximate usage for the ledger (streamed responses omit the usage block).
        try:
            self._record_usage(
                profile,
                {"usage": {"completion_tokens": max(1, len(content) // 4)}},
                streaming=True, task=task)
        except Exception:
            pass
        return content

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

    def _apply_model_quirks(self, profile: AIProviderProfile, payload: dict[str, Any]) -> None:
        """Per-family request settings the generic payload cannot express.

        Thinking models spend hidden "reasoning" tokens out of the same
        max_tokens budget as the answer. Gemini 2.5 Flash left to think spent
        ~700 of them on a six-line outline (5.2 s vs 1.5 s, measured); the
        gpt-oss family does the same. ORION's text turns are writing, not
        puzzles, so thinking is turned down — ORION_MODEL_THINKING=1 restores it.
        """
        if os.getenv("ORION_MODEL_THINKING", "").strip().lower() in {"1", "true", "on", "yes"}:
            return
        model = str(payload.get("model") or "").lower()
        if "generativelanguage.googleapis.com" in profile.base_url and "2.5-flash" in model:
            payload["reasoning_effort"] = "none"
        elif "gpt-oss" in model and "groq.com" in profile.base_url:
            payload["reasoning_effort"] = "low"

    async def _openai_compatible_chat(
        self, profile: AIProviderProfile, prompt: str, system_extra: str = "",
        *, instruction: str | None = None, task: str = "",
        image_jpeg: bytes | None = None, max_tokens: int | None = None,
    ) -> str:
        base_url = profile.base_url.rstrip("/")
        if not base_url:
            raise RuntimeError(f"provider {profile.name} has no base_url")
        endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = self.active_key(profile)
        if api_key and api_key.lower() not in {"local", "none", "no-key"}:
            headers["Authorization"] = f"Bearer {api_key}"
        system_content = (instruction if instruction is not None
                          else self.system_instruction(system_extra, context=task))
        # Shape the request to this model's payload budget BEFORE sending —
        # a small hosted model must never see the full instruction stack
        # (that is what produced HTTP 413 and a 300 s bench).
        system_content, prompt = self._fit_to_budget(
            profile, system_content, prompt, system_extra, instruction, task)
        user_content: Any = prompt
        if image_jpeg is not None:
            import base64
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")}},
            ]
        payload = {
            "model": self.active_model(profile),
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.35,
            "stream": False,
            # ALWAYS bounded — see CREDIT_RE. Omitting this made OpenRouter
            # reserve the model's full completion budget against the balance
            # and refuse the request with HTTP 402.
            "max_tokens": self._reply_ceiling(profile, max_tokens),
        }
        self._apply_model_quirks(profile, payload)
        if profile.is_local:
            # Unload the local model from RAM the instant this call finishes,
            # instead of Ollama's default 5-minute keep-warm.  With cloud-first
            # routing the local model is only ever a failover, so there is no
            # reason to hold gigabytes resident between turns.  (Honoured by
            # Ollama's OpenAI-compatible endpoint; harmless to servers that
            # ignore it, e.g. LM Studio.)
            payload["keep_alive"] = 0
            # Bound the KV-cache footprint too — see LOCAL_NUM_CTX.
            payload["options"] = {"num_ctx": self.LOCAL_NUM_CTX}
        # A long reply takes longer to write. A 1,300-word research section on a
        # free model needs well over the 30 s a spoken answer is allowed.
        seconds = max(8.0, float(profile.timeout_s or 30.0))
        if max_tokens:
            seconds = max(seconds, 20.0 + int(max_tokens) / 30.0)
        timeout = ClientTimeout(total=seconds)
        started = time.monotonic()
        with self._admit(profile):
            async with ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if image_jpeg is not None:
                        chunks = bytearray()
                        async for chunk in response.content.iter_chunked(16_384):
                            chunks.extend(chunk)
                            if len(chunks) > 262_144:
                                raise RuntimeError("Vision provider response exceeded the size limit")
                        raw = chunks.decode("utf-8")
                    else:
                        raw = await response.text()
                    if response.status >= 400:
                        if image_jpeg is not None:
                            raise RuntimeError(f"Vision provider HTTP {response.status}")
                        raise RuntimeError(f"HTTP {response.status}: {raw[:500]}")
                    data = json.loads(raw)
        # Observed latency feeds the orchestration ordering for future turns.
        self.note_latency(profile, time.monotonic() - started)
        # Record authoritative token usage (Section 6) — deduped by response id.
        self._record_usage(profile, data, streaming=False, task=task)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"provider {profile.name} returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or choices[0].get("text") or ""
        # Inspection JSON contains literal component markings. Transcript
        # identity correction (e.g. ORIN -> ORION) must not rewrite evidence.
        content = str(content).strip() if image_jpeg is not None else clean_transcript(str(content))
        if not content:
            raise RuntimeError(f"provider {profile.name} returned empty content")
        self._record_success(profile)
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
    thoughts_key = os.getenv("ORION_THOUGHTS_API_KEY", "").strip()
    local_url = os.getenv("ORION_LOCAL_OPENAI_BASE_URL", "").strip()
    # ── Mark XXII additions: ultra-fast inference clouds + rented cloud GPUs ──
    # Every one of these speaks the OpenAI-compatible chat API, so they slot into
    # ORION's existing unified transport as ordinary provider profiles. Each is
    # inert until its key (or, for the GPU rentals, its endpoint URL) is set, so
    # adding them costs nothing and misconfigures nothing.
    cerebras_key = os.getenv("ORION_CEREBRAS_API_KEY", "").strip()
    sambanova_key = os.getenv("ORION_SAMBANOVA_API_KEY", "").strip()
    runpod_key = os.getenv("ORION_RUNPOD_API_KEY", "").strip()
    runpod_url = os.getenv("ORION_RUNPOD_BASE_URL", "").strip()   # .../openai/v1
    vastai_url = os.getenv("ORION_VASTAI_BASE_URL", "").strip()   # http://<ip>:<port>/v1
    vastai_key = os.getenv("ORION_VASTAI_API_KEY", "").strip()
    modal_url = os.getenv("ORION_MODAL_BASE_URL", "").strip()     # https://<app>.modal.run/v1
    modal_key = os.getenv("ORION_MODAL_API_KEY", "").strip()
    return {
        "schema": "orion.mark_viii.providers.v1",
        "active_provider": "gemini",
        # Cascade order == the multi-tier fallback (Mark XXII). generate_text
        # walks this list top-to-bottom, trying the next provider whenever one
        # raises (429/402/503/context/offline). The tiers:
        #   · Commercial + ultra-fast inference clouds — the primary path.
        #   · Tier 1 fallback: rented cloud GPUs (RunPod, Vast.ai, Modal).
        #   · Tier 2 fallback: strictly-bounded local hardware, ALWAYS last.
        "provider_order": [
            "gemini",
            "gemini_text",      # the same Gemini key, for text work
            "anthropic",
            "openai",
            "xai_grok",
            "cerebras",         # ultra-fast inference (Cerebras wafer-scale)
            "sambanova",        # ultra-fast inference (SambaNova RDU)
            "openrouter",
            "groq",
            "together",
            "thoughts",
            "runpod",           # Tier 1: serverless rented GPU
            "vast_ai",          # Tier 1: peer-to-peer rented GPU
            "modal",            # Tier 1: serverless compute webhook
            "local_lm_studio",  # Tier 2: local hardware
            "local_ollama",     # Tier 2: local hardware (strict RAM limits)
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
            # Gemini for TEXT. The live profile above is a native-audio model
            # that only speaks the realtime protocol, so every research
            # outline, forge repair and written answer used to fall through to
            # whichever cheap fallback had credit left — and when none did,
            # ORION had no text brain at all despite holding a working Gemini
            # key. Gemini answers OpenAI-style chat at its own endpoint, so it
            # joins the unified transport as an ordinary profile.
            "gemini_text": {
                "kind": "openai_compatible",
                "enabled": bool(gemini_key),
                "api_key": gemini_key,
                "model": "gemini-2.5-flash",
                "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite",
                           "gemini-flash-latest"],
                "base_url": GEMINI_OPENAI_BASE,
                "priority": 12,
                "timeout_s": 90.0,
                "strengths": ["reasoning", "writing", "coding", "general"],
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
                "model": "openai/gpt-oss-120b",
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
            # ── Cerebras Cloud — wafer-scale, the fastest tokens/sec available.
            # Its public API is OpenAI-compatible (the cerebras_cloud_sdk is a
            # thin wrapper over this same endpoint), so it needs no extra
            # dependency and inherits ORION's fallback, budgeting and ledgering.
            "cerebras": {
                "kind": "openai_compatible",
                "enabled": bool(cerebras_key),
                "api_key": cerebras_key,
                "model": os.getenv("ORION_CEREBRAS_MODEL", "").strip() or "llama-3.3-70b",
                "base_url": "https://api.cerebras.ai/v1",
                "priority": 28,
                "timeout_s": 24.0,
                "strengths": ["fast", "reasoning"],
            },
            # ── SambaNova Cloud — RDU-accelerated, also very fast. Likewise
            # OpenAI-compatible, so the native sambanova SDK is unnecessary.
            "sambanova": {
                "kind": "openai_compatible",
                "enabled": bool(sambanova_key),
                "api_key": sambanova_key,
                "model": os.getenv("ORION_SAMBANOVA_MODEL", "").strip()
                         or "Meta-Llama-3.3-70B-Instruct",
                "base_url": "https://api.sambanova.ai/v1",
                "priority": 29,
                "timeout_s": 24.0,
                "strengths": ["fast", "reasoning"],
            },
            # ── RunPod (serverless rented GPU) — Tier 1 fallback. A vLLM
            # serverless worker exposes an OpenAI-compatible route at
            # .../v2/<endpoint-id>/openai/v1, which is what ORION_RUNPOD_BASE_URL
            # should point at. Inert until that URL is set.
            "runpod": {
                "kind": "openai_compatible",
                "enabled": bool(runpod_url),
                "api_key": runpod_key,
                "model": os.getenv("ORION_RUNPOD_MODEL", "").strip() or "qwen2.5-7b-instruct",
                "base_url": runpod_url,
                "priority": 60,
                "timeout_s": 90.0,
                "strengths": ["local", "reasoning"],
            },
            # ── Vast.ai (peer-to-peer rented GPU) — Tier 1 fallback. A standard
            # OpenAI-compatible server on a rented box; point ORION_VASTAI_BASE_URL
            # at http://<ip>:<port>/v1. The placeholder default stays disabled.
            "vast_ai": {
                "kind": "openai_compatible",
                "enabled": bool(vastai_url),
                "api_key": vastai_key or "local",
                "model": os.getenv("ORION_VASTAI_MODEL", "").strip() or "qwen2.5-7b-instruct",
                "base_url": vastai_url or "http://0.0.0.0:8000/v1",
                "priority": 62,
                "timeout_s": 90.0,
                "strengths": ["local"],
            },
            # ── Modal (serverless compute) — Tier 1 fallback. A Modal web
            # endpoint serving an OpenAI-compatible chat route (the common
            # vLLM-on-Modal pattern); ORION_MODAL_BASE_URL is its .../v1 URL.
            "modal": {
                "kind": "openai_compatible",
                "enabled": bool(modal_url),
                "api_key": modal_key or "local",
                "model": os.getenv("ORION_MODAL_MODEL", "").strip() or "qwen2.5-7b-instruct",
                "base_url": modal_url,
                "priority": 64,
                "timeout_s": 120.0,
                "strengths": ["local"],
            },
            # Reserved introspection slot — ORION's inner monologue.  Never
            # routed a user turn; give it its own (cheap) key/model so the
            # thought stream is metered separately from conversation.
            "thoughts": {
                "kind": "openai_compatible",
                "enabled": bool(thoughts_key),
                "api_key": thoughts_key,
                "model": os.getenv("ORION_THOUGHTS_MODEL", "").strip() or "openai/gpt-4o-mini",
                "base_url": os.getenv("ORION_THOUGHTS_BASE_URL", "").strip()
                            or "https://openrouter.ai/api/v1",
                "priority": 95,
                "timeout_s": 30.0,
                "strengths": ["reasoning"],
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
                # Tier-2 hard fallback. Default target is qwen2.5:7b — a strong
                # 7B that fits a 16GB-RAM machine when paired with the bounded
                # LOCAL_NUM_CTX and keep_alive=0 unload (see _openai_compatible_chat).
                "model": os.getenv("ORION_OLLAMA_MODEL", "").strip() or "qwen2.5:7b",
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
        api_keys=tuple(
            str(item).strip() for item in (raw.get("api_keys") or []) if str(item).strip()
        ),
        models=tuple(
            str(item).strip() for item in (raw.get("models") or []) if str(item).strip()
        ),
        budget_tokens=int(raw.get("budget_tokens") or 0),
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
        "api_keys": list(profile.api_keys),
        "models": list(profile.models),
        "budget_tokens": profile.budget_tokens,
    }


def _rescue_api_config() -> None:
    """api_keys.json has gone missing from the canonical path before while a
    stray copy survived elsewhere (a legacy build wrote under config/config/).
    Losing the file silently drops every configured provider, so if the
    canonical file is absent restore it from the newest rescue copy found."""
    if API_CONFIG_PATH.exists():
        return
    candidates = (
        CONFIG_DIR / "config" / "api_keys.json",
        CONFIG_DIR.parent / "legacy" / "config" / "api_keys.json",
    )
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (isinstance(data, dict) and data.get("providers")):
            continue
        # A rescue copy with no usable credential is not worth restoring.
        has_key = any(
            str(p.get("api_key") or "").strip() not in {"", "local"}
            or p.get("api_keys")
            for p in data["providers"].values() if isinstance(p, dict)
        )
        if not has_key:
            continue
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_text(API_CONFIG_PATH, json.dumps(data, indent=4), encoding="utf-8")
            return
        except Exception:
            continue


def read_provider_settings() -> OrionProviderSettings:
    _rescue_api_config()
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
    # Preserve the user's chosen order, but APPEND any providers introduced in a
    # newer default (e.g. the Mark XXII cloud-GPU tier) that their saved order
    # predates — otherwise a new fallback would be configured but never reached.
    existing_order = list(existing.get("provider_order") or [])
    default_order = list(defaults.get("provider_order") or [])
    if existing_order:
        # A newcomer goes where the defaults put it — after its predecessor in
        # the default order — not at the bottom, where every tie in routing
        # would go to whatever happened to be listed before it.
        order = list(existing_order)
        for index, name in enumerate(default_order):
            if name in order:
                continue
            anchor = next((default_order[j] for j in range(index - 1, -1, -1)
                           if default_order[j] in order), None)
            order.insert(order.index(anchor) + 1 if anchor else len(order), name)
        merged["provider_order"] = order
    else:
        merged["provider_order"] = default_order
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
    # Gemini text rides on the live profile's key unless the user gave it one
    # of its own (or switched it off, which an explicit saved entry records).
    live_key = str((providers.get("gemini") or {}).get("api_key") or "").strip()
    saved_text = (existing.get("providers") or {}).get("gemini_text")
    text = providers.get("gemini_text")
    if isinstance(text, dict) and live_key and not str(text.get("api_key") or "").strip():
        text["api_key"] = live_key
        if not isinstance(saved_text, dict):
            text["enabled"] = True
    # Retired shipped defaults (the model no longer exists at the provider).
    for name, retired in RETIRED_DEFAULTS.items():
        entry = providers.get(name)
        if isinstance(entry, dict) and str(entry.get("model") or "") in retired:
            entry["model"] = retired[str(entry["model"])]
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
    atomic_write_text(API_CONFIG_PATH, json.dumps(payload, indent=4), encoding="utf-8")


def read_api_key() -> str:
    """Legacy convenience wrapper retained for older call sites."""
    settings = read_provider_settings()
    gemini = settings.providers.get("gemini")
    return gemini.api_key if gemini is not None else ""


def write_api_key(api_key: str) -> None:
    """Legacy writer retained; writes the Mark VIII provider schema."""
    settings = _settings_from_payload(_default_provider_payload(api_key))
    write_provider_settings(settings)
