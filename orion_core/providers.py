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
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from urllib.parse import urlsplit

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from .lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
ClientTimeout = lazy_attr("aiohttp", "ClientTimeout")

from .bus import OrionBus
from .constants import API_CONFIG_PATH, CONFIG_DIR, LIVE_MODEL
from . import provider_transport as _transport
from .request_trace import mark as _trace_mark


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
            "OpenAI, xAI, OpenRouter, Groq, Mistral, Hugging Face, NVIDIA) or start a local LLM "
            "(Ollama / LM Studio) and enable it. ORION keeps its deterministic "
            "tools running meanwhile."
        )


class ThoughtGenerationUnavailable(NoTextProviderError):
    """Background generation is idle under the user's local-only cost policy.

    An expected, quiet state rather than an activity-log diagnostic.
    """


class AllTextProvidersFailedError(ProviderError):
    """Every eligible provider was tried and each errored.  Carries the ordered
    list of providers attempted and the last error message (no secrets)."""

    def __init__(self, last_error: object, *, attempts: list[str] | None = None) -> None:
        safe_error = redacted_error_line(last_error, 240)
        super().__init__(f"all text providers failed: {safe_error}")
        self.attempts = attempts or []
        self.last_error = safe_error
        self.remediation = (
            "Check network connectivity and provider status; a local LLM "
            "(Ollama / LM Studio) gives an offline fallback."
        )


class ProviderDeadlineError(AllTextProvidersFailedError):
    """The request's total time budget ran out before any provider answered.

    A subclass of AllTextProvidersFailedError because, to the caller, it is
    the same situation — nobody answered in time — and every existing handler
    for that should treat it the same way. It is never charged to a provider:
    running out of the caller's time says nothing about the provider's health.
    """

    def __init__(self, last_error: object = "the time budget for this request ran out",
                 *, attempts: list[str] | None = None) -> None:
        super().__init__(last_error, attempts=attempts)
        self.remediation = (
            "The request's time budget ran out. Interactive turns are bounded "
            "by ORION_INTERACTIVE_BUDGET_S, background work by "
            "ORION_RESEARCH_BUDGET_S.")


class StreamInterruptedError(ProviderError):
    """A provider failed after part of its answer had already been streamed,
    and the caller gave no ``on_restart`` to start the answer over. Raised
    rather than splicing another provider's answer onto the visible start."""

    def __init__(self, provider: str, cause: object = "") -> None:
        super().__init__(
            f"{provider} failed part-way through a streamed answer: "
            f"{redacted_error_line(cause, 160)}")
        self.provider = provider


class ProviderClosedError(ProviderError):
    """The router has been shut down; no new provider call may start."""


class ProviderConfigurationError(ProviderError):
    """Retained provider policy is unavailable or changed; never reset it."""


class ProviderText(str):
    """String-compatible reply retaining the provider's completion signal."""

    def __new__(cls, text: str, done_reason: str = "", raw_text: str | None = None):
        value = super().__new__(cls, text)
        value.done_reason = str(done_reason).lower()
        value.raw_text = text if raw_text is None else raw_text
        return value


@dataclass
class _CallScope:
    """What one generate_text call carries down to every attempt it makes.

    Held in a context variable rather than threaded through signatures: the
    per-provider attempt methods are the seam that tests and callers replace,
    and their signatures are relied on as they are.
    """

    cls: str = _transport.INTERACTIVE
    deadline: float | None = None           # time.monotonic() value
    first_token_marked: bool = False
    #: Held back from the current attempt for the providers after it.
    reserve_s: float = 0.0

    def remaining(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.monotonic()

    def attempt_remaining(self) -> float | None:
        """What the current provider attempt may use: the budget left, less
        the time held back for the providers waiting behind it."""
        left = self.remaining()
        return None if left is None else left - max(0.0, self.reserve_s)


_CALL: ContextVar[_CallScope | None] = ContextVar("orion_provider_call", default=None)


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(os.getenv(name, "") or default)))
    except ValueError:
        return default


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.getenv(name, "") or default)))
    except ValueError:
        return default


def _ollama_model_key(model: str) -> str:
    """Ollama names an untagged model ``name:latest`` in /api/ps."""
    model = str(model or "").strip()
    return model if ":" in model else f"{model}:latest"


def _keep_alive_seconds(value: int | float | str | None, cap: int = 3600) -> int:
    """Seconds from an Ollama keep-alive ("10m", "30s", "1h", 600, "0").
    Unparseable or negative values mean 0 (release at once); capped at an hour."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, min(cap, int(value)))
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*", str(value).lower())
    if not match:
        return 0
    amount = float(match.group(1))
    scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, None: 1}[match.group(2)]
    return max(0, min(cap, int(amount * scale)))


from .knowledge import PERSONA_BOOST as _NEURO_PERSONA_BOOST
from .programming_knowledge import PROGRAMMING_PERSONA_BOOST as _PROG_PERSONA_BOOST
from .cyber_knowledge import CYBER_PERSONA_BOOST as _CYBER_PERSONA_BOOST
from .cyber_knowledge import SECURITY_TEACHING_DIRECTIVE as _SECURITY_TEACHING_DIRECTIVE
from .learning_style import LEARNING_STYLE_DIRECTIVE as _LEARNING_STYLE_DIRECTIVE
from .identity import creator_rule
from .security import SecuritySanitiser
from .utils import clean_transcript, first_line
from . import key_seal
from .atomic_io import atomic_write_text
from .provider_diagnostics import redacted_error_line


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

# Text models only.  Each identifier is an ordinary generateContent model with
# function calling; Live, TTS, image, agent and embedding endpoints use other
# request shapes.  The smaller profile leads everyday turns, while the second
# profile leads complex work.  Google meters their model quotas separately.
_GEMINI_TEXT_MODELS = (
    "gemini-3.5-flash-lite", "gemini-2.5-flash-lite",
    "gemini-2.5-flash", "gemini-3.1-flash-lite",
)
_GEMINI_REASONING_MODELS = (
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash",
)
_FORMER_GEMINI_TEXT_MODELS = (
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest",
)
_PREVIOUS_GEMINI_TEXT_MODELS = (
    "gemini-2.5-flash", "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite", "gemini-flash-latest",
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
        from urllib.parse import urlsplit

        try:
            endpoint = urlsplit(self.base_url)
            return (endpoint.scheme == "http" and endpoint.username is None
                    and endpoint.password is None
                    and endpoint.hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"})
        except ValueError:
            return False

    @property
    def supports_text_generation(self) -> bool:
        if not self.enabled:
            return False
        if self.kind not in {"openai_compatible", "gemma_native"}:
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
    configuration_error: str = ""
    _source_observed: bool = field(default=False, repr=False, compare=False)
    _source_path: str = field(default="", repr=False, compare=False)
    _source_sha256: str | None = field(default=None, repr=False, compare=False)

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
    # Ollama's native API words its refusal to load as "model requires more
    # system memory (5.3 GiB) than is available (4.1 GiB)", which none of the
    # other phrasings match — it was classed as a transient fault and retried
    # every 45 seconds, which is the very loop this class exists to stop.
    OOM_RE = re.compile(
        r"(?i)out of memory|cudamalloc|failed to allocate|oom|"
        r"cuda error|insufficient memory|not enough memory|"
        r"requires more (?:system )?memory")
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
    # length — a 7B/8B model at an unconstrained context (up to 128K) can burn
    # several extra GB of RAM/VRAM beyond the weights themselves, which is
    # exactly the "local fallback eats all the memory" failure mode.
    #
    # The effective context policy. This used to say 2048 and was never in
    # force: it travelled through Ollama's /v1 endpoint, which drops the
    # field. Measured on this machine (Ollama 0.32.6, llama3.1 8B Q4_K_M,
    # RTX 3060 Ti 8 GB): a /v1 request carrying num_ctx=2048 loaded the model
    # at context 4096, 5.27 GB, all of it in VRAM. 2048 cannot simply be
    # switched on now that it would be obeyed: the lean instruction alone is
    # ~6.4k characters, and a 2048 context would cut ORION's own persona off
    # the front of every local prompt. 4096 is what was really running, so
    # memory use is unchanged; what changes is that the payload budget for a
    # native-Ollama profile is now held inside it (see _chat_attempt), so the
    # prompt always fits the context it is actually given. Raise it on a
    # machine with more VRAM via ORION_LOCAL_NUM_CTX.
    LOCAL_NUM_CTX = _env_int("ORION_LOCAL_NUM_CTX", 4096, 512, 32768)

    # ── one total deadline per request ────────────────────────────────────────
    # Each attempt had its own timeout and nothing bounded the sum: three
    # providers at 30 s plus a local model at 120 s could hold a conversation
    # turn for over three minutes. generate_text now carries one deadline
    # through admission, connection, generation, the healing retry, cooldown
    # waits and every fallback.
    INTERACTIVE_BUDGET_S = _env_float("ORION_INTERACTIVE_BUDGET_S", 90.0, 5.0, 3600.0)
    RESEARCH_BUDGET_S = _env_float("ORION_RESEARCH_BUDGET_S", 900.0, 30.0, 7200.0)
    #: An attempt is not started with less than this left, nor with less than
    #: the provider's own recent response time (it would only be cut off).
    MIN_ATTEMPT_S = 2.0
    #: While another provider waits behind it, an attempt may not spend the
    #: whole budget: this much (never more than a third of what is left) is
    #: held back for the next. 26 Oct 2026: gemini_reasoning, its own timeout
    #: equal to the 90 s budget, held a specialist agent's request until the
    #: budget ran out, and the request failed with no other provider asked.
    FALLBACK_RESERVE_S = _env_float("ORION_FALLBACK_RESERVE_S", 30.0, 0.0, 600.0)
    #: Tasks that are background work whatever budget their caller passed.
    #: Those callers (research, the forge, the inner voice, fact harvesting)
    #: predate the budget argument; naming them here keeps them behind the
    #: conversation for a provider's slots without editing each one.
    BACKGROUND_TASKS = frozenset({"research", "forge", "thought", "memory"})

    # ── shared admission ──────────────────────────────────────────────────────
    # Conversation, research, the forge and the inner voice share one router,
    # and each had its own limits (research alone runs six sections at once),
    # so together they could exceed anything a provider or the GPU accepts.
    #: Concurrent requests per cloud profile, and how many of those background
    #: work may hold — the rest stay free for a conversation turn.
    CLOUD_CONCURRENCY = _env_int("ORION_PROVIDER_CONCURRENCY", 6, 1, 64)
    CLOUD_BACKGROUND_CONCURRENCY = _env_int("ORION_PROVIDER_BACKGROUND_CONCURRENCY", 4, 1, 64)
    #: Concurrent generations per LOCAL server: one GPU, one at a time.
    LOCAL_CONCURRENCY = _env_int("ORION_LOCAL_CONCURRENCY", 1, 1, 8)

    # ── local residency ───────────────────────────────────────────────────────
    # Unloading after every call (the old intent) makes every local turn a
    # cold one: measured once here, a cold llama3.1 8B turn took 17.3 s end
    # to end for a five-token reply, nearly all of it loading the model
    # (Ollama 0.32.6, RTX 3060 Ti). Keeping it for a couple of minutes makes
    # a local conversation warm after its first turn; memory pressure
    # (set_memory_pressure / unload_local_models) takes it back early.
    LOCAL_KEEP_ALIVE_S = _env_int("ORION_LOCAL_KEEP_ALIVE_S", 120, 0, 3600)
    #: Background work keeps the model only briefly, and never shortens a
    #: window a conversation opened.
    LOCAL_BACKGROUND_KEEP_ALIVE_S = _env_int("ORION_LOCAL_BACKGROUND_KEEP_ALIVE_S", 30, 0, 3600)
    #: How long a closed or firewalled local port may keep ORION waiting to
    #: find out it is closed (it hangs here rather than refusing).
    LOCAL_PROBE_TIMEOUT_S = 1.5
    #: A server that answered "not Ollama" is asked again after this long.
    NOT_OLLAMA_TTL_S = 600.0

    def __init__(self, settings: OrionProviderSettings, bus: OrionBus, memory: Any,
                 connectivity: Any | None = None) -> None:
        # `memory` is the MemoryAgent (session + persistent); it only needs to
        # expose prompt_context(limit) here.
        self.settings = settings
        self.bus      = bus
        self.memory   = memory
        if getattr(settings, "configuration_error", ""):
            bus.log.emit("NET: " + settings.configuration_error)
        # Attached by the composition root after built-in skills are loaded.
        # Prompt reads use its in-memory registry, never a directory scan.
        self.skills: Any | None = None
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
        # Mark XXXX.4: the alternate that answered after a server error, kept
        # for STICKY_MODEL_S so the next session does not pay the failing
        # model's timeout first (sticky_models.json; never the saved config).
        self._sticky_models: dict[str, dict[str, Any]] = _read_sticky_models()
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
        # Brain mode "local" (brain_mode.py): no AI API at all. Unlike
        # ORION_MODE=offline this leaves connectivity alone, so web tools keep
        # working; only cloud models (text, vision and Live) are withheld.
        self.local_only = False
        # Transport state: pooled sessions (one per event loop), per-provider
        # admission gates, which local servers are Ollama, and what ORION has
        # asked Ollama to keep resident.
        self._pool = _transport.SessionPool()
        self._gates: dict[str, _transport.EndpointGate] = {}
        #: root URL -> (is_ollama, version, checked_at monotonic)
        self._ollama_roots: dict[str, tuple[bool, str, float]] = {}
        #: (root, model) -> requests running on that model right now.
        self._inflight: dict[tuple[str, str], int] = {}
        #: (root, model) -> monotonic time ORION asked Ollama to keep it until.
        self._resident_until: dict[tuple[str, str], float] = {}
        #: Unloads asked for while the model was busy; done when it goes idle.
        self._unload_pending: set[tuple[str, str]] = set()
        self._memory_pressure = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._provider_ready = False
        # Catalogue validation is shared by profiles using the same account
        # and endpoint. Store a digest rather than another copy of the key.
        self._model_catalogues: dict[tuple[str, str, str], tuple[float, tuple[str, ...]]] = {}
        self._model_catalogue_locks: dict[tuple[Any, tuple[str, str, str]], asyncio.Lock] = {}

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
        lines = []
        try:
            from .voice_enrolment import turns
            note = turns().listener_note()
            if note:
                lines.append(note)
        except Exception:
            pass
        presence = getattr(self, "_voice_presence", None)
        if presence is not None:
            try:
                latest = presence.last
                note = latest.prompt_line() if latest is not None else ""
                if note:
                    lines.append(note)
            except Exception:
                pass
        return "\n".join(lines)

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
                       task: str = "", cap: int | None = None) -> tuple[str, str]:
        """Shape (system, prompt) to the provider's payload budget: first swap
        the full instruction stack for the compact one, then trim the prompt's
        middle (head and tail carry the intent; the middle is usually bulk
        context).  Guarantees the request cannot exceed the model's budget.
        ``cap`` is a hard context the server will really apply (a native
        Ollama num_ctx): the budget never exceeds it, or the server would cut
        the start of the prompt — the persona — to make it fit."""
        budget = self.payload_budget(profile)
        if cap:
            budget = min(budget, int(cap))
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

    def set_local_only(self, active: bool) -> None:
        """Brain mode 'local': withhold every cloud model (text, vision, Live)
        while leaving connectivity — and so the internet tools — alone."""
        self.local_only = bool(active)

    def cloud_allowed(self) -> bool:
        """Whether a cloud AI service may be called at all right now."""
        return not self.local_only and self.mode_override != "offline"

    def current_mode(self) -> str:
        if self.mode_override == "offline":
            return "MODE B (forced offline)"
        if getattr(self, "local_only", False):
            return "MODE L (local brain only — no AI API)"
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
        # Live audio (Gemini) is a cloud capability; suppress it when offline
        # and in the local-only brain mode.
        if not self.is_online() or getattr(self, "local_only", False):
            return []
        return [p for p in self.settings.ordered_profiles() if p.supports_live_audio and self.is_available(p)]

    # The reserved introspection slot: its own key/model so ORION's inner
    # monologue never competes with (or bills against) the conversation
    # channel, and never absorbs a user turn.
    THOUGHTS_PROFILE = "thoughts"

    def _all_text_profiles(self) -> list[AIProviderProfile]:
        local_only = getattr(self, "local_only", False)
        return [p for p in self.settings.ordered_profiles()
                if p.supports_text_generation and self.is_available(p)
                and p.name != self.THOUGHTS_PROFILE
                and (p.is_local or not local_only)]

    async def wait_for_text(self, max_wait_s: float = 90.0) -> bool:
        """Wait, within a bound, for a text provider that is only cooling down.

        Background work (the forge, self-improvement) used to fail outright
        the moment every provider was rate limited, even when the soonest one
        was back in 20 s, and then burned the next attempt the same way. True
        once a provider is usable; False at once when nothing is merely
        cooling (no keys, nothing local), or when the bound runs out.
        """
        deadline = time.monotonic() + max(0.0, float(max_wait_s))
        while True:
            if self.text_available():
                return True
            wait = self.soonest_recovery_s()
            remaining = deadline - time.monotonic()
            if wait is None or remaining <= 0:
                return False
            await asyncio.sleep(max(0.2, min(wait + 0.25, remaining)))

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
            if profile.kind not in {"openai_compatible", "gemma_native"}:
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
        """Last chance before declaring degraded mode: bring a local brain up.

        Starting Ollama means launching it and polling until it answers: up to
        six seconds, then a two-second registration probe. That used to run
        right here, on the event loop, whenever every cloud provider was rate
        limited, freezing voice, face and GUI for as long as eight seconds.
        With a loop running it now happens on a worker thread; degraded mode is
        announced at once and lifted the moment the local model is serving.
        """
        recover = getattr(self, "_local_recovery", None)
        if recover is None or getattr(self, "_recovery_running", False):
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            self._recovery_running = True

            def _finished(future: Any) -> None:
                self._recovery_running = False
                try:
                    started = bool(future.result())
                except Exception as exc:
                    self.bus.log.emit(
                        f"NET: local recovery failed - {redacted_error_line(exc, 90)}")
                    return
                if started and self.text_available():
                    self._degraded = False
                    self.bus.log.emit(
                        "NET: recovered — a local model is now serving; ORION "
                        "continues on MODE B rather than losing his voice.")

            loop.run_in_executor(None, recover, self.settings).add_done_callback(
                lambda future: loop.call_soon_threadsafe(_finished, future))
            return False
        self._recovery_running = True
        try:
            if bool(recover(self.settings)):
                self.bus.log.emit(
                    "NET: recovered — a local model is now serving; ORION "
                    "continues on MODE B rather than losing his voice.")
                return self.text_available()
        except Exception as exc:
            self.bus.log.emit(f"NET: local recovery failed - {redacted_error_line(exc, 90)}")
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
        if profile.name not in self._model_index:
            remembered = self._sticky_models.get(profile.name) or {}
            if (remembered.get("model") in models
                    and float(remembered.get("until") or 0) > time.time()):
                self._model_index[profile.name] = models.index(remembered["model"])
        return models[self._model_index.get(profile.name, 0) % len(models)]

    def _remember_model(self, profile: AIProviderProfile) -> None:
        """Keep the alternate that just answered for STICKY_MODEL_S."""
        self._sticky_models[profile.name] = {"model": self.active_model(profile),
                                             "until": round(time.time() + STICKY_MODEL_S)}
        try:
            STICKY_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(STICKY_MODELS_PATH, json.dumps(self._sticky_models, indent=2),
                              encoding="utf-8")
        except OSError:
            pass

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
        """Account-scoped model ids, cached for five minutes after success.

        Routine and reasoning profiles often use the same Gemini account.
        Boot validation and healing previously fetched that catalogue again
        for each profile, including twice for every retired model. Concurrent
        callers now share one read; each retains its own deadline.
        """
        base = profile.base_url.rstrip("/")
        if not base:
            return []
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        key = self.active_key(profile)
        catalogue_key = (base, profile.kind, hashlib.sha256(key.encode("utf-8")).hexdigest())

        def cached() -> list[str] | None:
            entry = self._model_catalogues.get(catalogue_key)
            if entry is not None and time.monotonic() - entry[0] < 300.0:
                return list(entry[1])
            return None

        ready = cached()
        if ready is not None:
            return ready
        # Healing runs inside a request's deadline; it must not outlive it.
        scope = _CALL.get()
        remaining = scope.remaining() if scope is not None else None
        seconds = 20.0 if remaining is None else min(20.0, remaining - self.MIN_ATTEMPT_S)
        if seconds <= 0.5:
            return []
        loop = asyncio.get_running_loop()
        for old in list(self._model_catalogue_locks):
            if old[0].is_closed():
                self._model_catalogue_locks.pop(old, None)
        lock_key = (loop, catalogue_key)
        lock = self._model_catalogue_locks.setdefault(lock_key, asyncio.Lock())

        async def read() -> list[str]:
            async with lock:
                ready = cached()
                if ready is not None:
                    return ready
                headers = {"User-Agent": "ORION/33 (+https://github.com/projectorionai/PROJECT-ORION)"}
                if key and key.lower() not in {"local", "none", "no-key"}:
                    headers["x-goog-api-key" if profile.kind == "gemma_native"
                            else "Authorization"] = key if profile.kind == "gemma_native" else f"Bearer {key}"
                session = await self._pool.get()
                async with session.get(f"{base}/models", headers=headers,
                                       timeout=ClientTimeout(total=seconds),
                                       allow_redirects=False) as response:
                    if response.status >= 300:
                        return []
                    data = json.loads(await _transport._read_bounded(response, 1_048_576))
                rows = (data.get("models") if profile.kind == "gemma_native"
                        else data.get("data")) if isinstance(data, dict) else data
                if not isinstance(rows, list):
                    return []
                ids: list[str] = []
                for row in rows:
                    ident = (row.get("name") if profile.kind == "gemma_native" else row.get("id")) \
                        if isinstance(row, dict) else row
                    ident = str(ident or "").strip()
                    if ident.startswith("models/"):
                        ident = ident[len("models/"):]
                    if ident and (profile.kind != "gemma_native" or ident in {
                            "gemma-4-26b-a4b-it", "gemma-4-31b-it"}):
                        ids.append(ident)
                if ids:
                    self._model_catalogues[catalogue_key] = (time.monotonic(), tuple(ids))
                return ids

        try:
            # Include waiting for another catalogue reader in this caller's
            # budget. Cancelling a waiter leaves the active reader alone.
            return await asyncio.wait_for(read(), timeout=seconds)
        except Exception:
            return []

    def _persist_model(self, profile: AIProviderProfile, *, expected_model: str | None = None) -> bool:
        """Write a healed model id back, touching only that one field.

        Read-modify-write of the single key rather than write_provider_settings,
        which would rewrite the whole file from this process's view of it.
        """
        try:
            with _PROVIDER_CONFIG_LOCK:
                data, observed = _read_provider_document(API_CONFIG_PATH)
                entry = (data.get("providers") or {}).get(profile.name) if data is not None else None
                if (not isinstance(entry, dict)
                        or expected_model is not None and entry.get("model") != expected_model):
                    return False
                entry["model"] = profile.model
                published = _publish_provider_document(data, observed)
                settings = getattr(self, "settings", None)
                if (settings is not None and settings._source_observed
                        and settings._source_path == str(API_CONFIG_PATH.absolute())
                        and settings._source_sha256 == observed):
                    settings._source_sha256 = published
                return True
        except Exception:
            return False

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
        saved_model = profile.model
        profile.model = replacement
        self._model_index.pop(profile.name, None)
        self._output_cap.pop(profile.name, None)
        self._cooldowns.pop(profile.name, None)
        saved = self._persist_model(profile, expected_model=saved_model)
        self.bus.log.emit(
            f"NET: provider {profile.name} now uses '{replacement}' "
            f"('{old}' {'is not covered by the account balance' if free_only else 'is no longer served'})."
            + ("" if saved else " Session only; saved provider settings could not be updated."))
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

    #: Failures that say nothing about the provider's health: the caller ran
    #: out of time, or ORION is shutting down. Never charged to a provider.
    _NOT_THE_PROVIDERS_FAULT = (ProviderDeadlineError, ProviderClosedError)

    async def _call_with_heal(self, profile: AIProviderProfile, call: Callable[[], Any],
                              *, may_retry: Callable[[], bool] | None = None) -> str:
        """Run one provider call; on a model/credit failure, heal and retry once.

        The retry sends the same request to a healed model, which is safe for
        a chat completion (nothing happens twice but the bill). It is refused
        when ``may_retry`` says the first attempt already showed the caller
        part of an answer, and when the request's deadline cannot fit it.
        """
        try:
            return await call()
        except self._NOT_THE_PROVIDERS_FAULT:
            raise
        except Exception as exc:
            # Google can briefly return 500/503 while both Gemini and hosted
            # Gemma remain healthy. One bounded retry avoids opening every
            # fallback's breaker on a single overloaded backend moment. Never
            # replay a streamed answer the caller has already seen.
            scope = _CALL.get()
            remaining = scope.remaining() if scope is not None else None
            retry_delay = 0.35
            google_endpoint = urlsplit(profile.base_url)
            if (google_endpoint.scheme == "https"
                    and google_endpoint.hostname == "generativelanguage.googleapis.com"
                    and re.search(r"\bHTTP\s+(?:500|502|503|504)\b", str(exc), re.I)
                    and (may_retry is None or may_retry())
                    and (remaining is None or remaining >= retry_delay + self.MIN_ATTEMPT_S)):
                failing = self.active_model(profile)
                status = re.search(r"\bHTTP\s+(\d{3})\b", str(exc), re.I)
                rotated = self._rotate_model(profile)
                detail = (f"trying configured alternate '{self.active_model(profile)}' once"
                          if rotated else "retrying once")
                self.bus.log.emit(
                    f"NET: {profile.name} returned a temporary server error "
                    f"(HTTP {status.group(1) if status else '5xx'} on '{failing}'); {detail}.")
                await asyncio.sleep(retry_delay)
                try:
                    answer = await call()
                    if rotated:
                        await asyncio.to_thread(self._remember_model, profile)
                    return answer
                except self._NOT_THE_PROVIDERS_FAULT:
                    raise
                except Exception as retried:
                    exc = retried
            self.mark_failure(profile, exc)
            if may_retry is not None and not may_retry():
                raise exc
            if not await self._heal_after_failure(profile):
                raise exc
            scope = _CALL.get()
            remaining = scope.remaining() if scope is not None else None
            if remaining is not None and remaining < self.MIN_ATTEMPT_S:
                raise exc
        try:
            return await call()
        except self._NOT_THE_PROVIDERS_FAULT:
            raise
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
        message = redacted_error_line(exc, 240) or f"{type(exc).__name__} (no message)"
        
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
            "admission": self.admission_snapshot(),
            "local_residency": self.residency_snapshot(),
        }

    def admission_snapshot(self) -> dict[str, dict[str, int]]:
        """Per-provider slots in use and queued, for diagnostics."""
        return {key: gate.snapshot() for key, gate in list(self._gates.items())}

    def residency_snapshot(self) -> dict[str, Any]:
        """What ORION has asked a local server to keep loaded, and for how
        long; which models are busy right now. No prompts, no keys."""
        now = time.monotonic()
        return {
            "memory_pressure": self._memory_pressure,
            "keep_alive_s": 0 if self._memory_pressure else self.LOCAL_KEEP_ALIVE_S,
            "resident": {f"{root} {model}": round(max(0.0, until - now), 1)
                         for (root, model), until in list(self._resident_until.items())
                         if until > now},
            "busy": {f"{root} {model}": count
                     for (root, model), count in list(self._inflight.items()) if count},
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
            except self._NOT_THE_PROVIDERS_FAULT:
                raise
            except Exception as exc:
                last_error = exc
                self.mark_failure(profile, exc)
        raise RuntimeError(f"all local models failed: {redacted_error_line(last_error)}")

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
                "system and executive aide, not a chatbot. Model your manner on JARVIS "
                "from Iron Man: a refined British AI butler - calm, precise, respectful, "
                "bone-dry wit when the moment allows, emotionally aware and never robotic "
                "or cute. Address the user "
                "as 'sir' only occasionally — greetings, farewells, a direct order, a "
                "genuinely formal moment — not as a suffix on ordinary replies. "
                "When speaking your own name aloud, say Orion as one word, "
                "pronounced oh-rye-on. "
                "Never spell it as O R I N or omit the second O when spelling the acronym. "
                "Use strict British English spelling in every spoken and textual response."
            ),
            creator_rule(),
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
            _SECURITY_TEACHING_DIRECTIVE,
            _LEARNING_STYLE_DIRECTIVE,
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

            from .constants import BASE_DIR, CONFIG_DIR
            from .app_resolver import known_folder
            home = Path.home()
            locations = [f"home {home}"]
            for name in ("desktop", "documents", "downloads"):
                folder = known_folder(name, location=True)
                if folder is not None:
                    locations.append(f"{name.title()} {folder}")
            line = (
                "FOLDERS ON THIS PC — use resolved locations, never guess one: "
                + "; ".join(locations)
                + f"; research reports and dossiers {CONFIG_DIR / 'research'}; "
                f"legacy research notes and papers {BASE_DIR / 'research'}. "
                "Research folders are created when output is saved; a configured "
                "location does not prove a file exists. Use the saved artifact path "
                "returned by the tool, or find_files, rather than inventing a location."
            )
        except Exception:
            line = ""
        self._environment_cache = line
        return line

    def set_locality(self, locality: str) -> None:
        """Called by the temporal/geo layer once the PC's location is known."""
        self._locality = str(locality or "").strip()

    def _skill_context(self, limit: int = 3000) -> str:
        manager = getattr(self, "skills", None)
        if manager is None:
            return ""
        try:
            return str(manager.prompt_context(limit=limit) or "")[:limit]
        except Exception:
            # An optional skill must not prevent the core voice/text channel
            # from coming up. Package failures are reported by skill management.
            return ""

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
                self._skill_context(),
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
                    "Background work: you are not told when it changes, so never say a "
                    "background job or agent is still running, finished, or what it found "
                    "without checking first with job(action='list') or job(action='result'). "
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
                    "Loosely worded requests are normal — act on the most reasonable "
                    "reading instead of asking for exact names. Pass the user's own words "
                    "to the tool: open_app understands kinds of application ('a browser', "
                    "'something to write code in', 'my music', 'my downloads') and opens "
                    "their default or most-used one, saying which; close_app and "
                    "window_control understand 'the browser' too; media_control 'play' "
                    "starts their music app when nothing is open. Ask a question only when "
                    "two readings would do clearly different things, and then offer the "
                    "likely options rather than asking open-endedly."
                ),
                (
                    "Common sense: when the user describes a situation instead of giving "
                    "an instruction ('the screen is blinding me', 'everything is lagging', "
                    "'I keep forgetting my tablets', 'it's freezing in here'), work out what "
                    "would actually help and do it if it is quick and reversible — dim the "
                    "screen, check what is loading the machine, offer a reminder — or offer "
                    "it in one sentence if it is not. Answer the question that was meant: "
                    "'do I need a coat?' is about the weather where they are, now; 'how are "
                    "the markets?' means the major indices, not one company. Use where and "
                    "when they are: British units (°C, miles, pounds), their local time and "
                    "today's date. If a fact about the user may be in memory, check memory "
                    "before saying you don't know. Say what you are unsure of rather than "
                    "guessing, and never report a result a tool did not return. Match the "
                    "length of the answer to the question — a quick question gets a quick "
                    "answer, by voice especially."
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
                self._skill_context(limit=1250),
                extra.strip(),
            ]
            if part
        )

    # ── text generation ───────────────────────────────────────────────────────

    def _budget_class(self, budget: str) -> str:
        return (_transport.INTERACTIVE if str(budget or "").lower() == "interactive"
                else _transport.BACKGROUND)

    def _default_deadline_s(self, budget: str, max_tokens: int | None,
                            patience_s: float) -> float:
        """The total budget a call gets when it does not name one.

        Never shorter than a single attempt at the requested length was
        allowed before a total existed (a long repair or research section
        must not be cut short by a default meant for spoken replies), plus
        any patience the caller asked for explicitly.
        """
        base = (self.INTERACTIVE_BUDGET_S
                if self._budget_class(budget) == _transport.INTERACTIVE
                else self.RESEARCH_BUDGET_S)
        if max_tokens:
            base = max(base, 20.0 + int(max_tokens) / 30.0 + 10.0)
        return base + max(0.0, float(patience_s or 0.0))

    @contextlib.contextmanager
    def _call_scope(self, cls: str, deadline: float | None):
        token = _CALL.set(_CallScope(cls=cls, deadline=deadline))
        foreground_timer = None
        if cls == _transport.INTERACTIVE:
            from . import admission
            def foreground_pulse():
                nonlocal foreground_timer
                admission.ADMISSION.note_foreground("Language model request")
                try:
                    foreground_timer = asyncio.get_running_loop().call_later(1, foreground_pulse)
                except RuntimeError:
                    pass
            foreground_pulse()
        try:
            yield
        finally:
            if foreground_timer is not None:
                foreground_timer.cancel()
            _CALL.reset(token)

    async def generate_text(self, prompt: str, system_extra: str = "", *,
                            instruction: str | None = None,
                            task: str = "",
                            max_tokens: int | None = None,
                            patience_s: float = 0.0,
                            deadline_s: float | None = None,
                            budget: str = "interactive",
                            on_delta: Callable[[str], None] | None = None,
                            on_restart: Callable[[str], None] | None = None,
                            complete_reply: bool = True,
                            ) -> tuple[AIProviderProfile, str]:
        """Route a text turn; with *patience_s*, wait out rate-limit cooldowns.

        A conversation must answer now or say it cannot, so the default is no
        patience. Background work is different: a research paper once read 33
        pages, used every provider's per-minute allowance on the notes, and
        then failed all six sections in the same second with "no text
        provider is available" — when the first provider was 20 s from free.
        With patience the call waits for the soonest one (never past the
        budget) and tries again.

        ``deadline_s`` bounds the WHOLE call — admission, connection,
        generation, the healing retry, cooldown waits and every fallback —
        and defaults by ``budget``: "interactive" (a person is waiting) or
        "research" / "background". Patience, the tasks in BACKGROUND_TASKS and
        any non-interactive budget queue behind conversation for a provider's
        slots. Running out raises ProviderDeadlineError.

        Streaming: with ``on_delta``, each fragment of the answer is passed to
        it as it arrives. If a provider fails after showing at least one
        fragment, ``on_restart(reason)`` is called once and the next provider
        answers from the beginning; without ``on_restart`` the call raises
        StreamInterruptedError instead. A second provider's answer is never
        spliced onto the first's start. The returned text is the complete
        answer of the provider that finished (after the same transcript
        clean-up as an unstreamed reply, so it can differ slightly from the
        joined fragments — show the returned text as the final answer).
        """
        if budget == "interactive" and (patience_s or task in self.BACKGROUND_TASKS):
            budget = "research"
        # Explicit requests for depth need room to finish. Strict-format calls
        # own their validation and are not automatically extended here.
        comprehensive = instruction is None and bool(re.search(
            r"\b(comprehensive|exhaustive|thorough|in.depth|detailed|step.by.step)\b",
            str(prompt), re.I))
        if comprehensive and max_tokens is None:
            max_tokens = 4096
        cls = self._budget_class(budget)
        total = (float(deadline_s) if deadline_s is not None
                 else self._default_deadline_s(budget, max_tokens, patience_s))
        if comprehensive and deadline_s is None:
            total = max(total, 180.0)
        now = time.monotonic()
        deadline = now + max(0.0, total)
        patience_until = min(deadline, now + max(0.0, float(patience_s or 0.0)))
        with self._call_scope(cls, deadline):
            while True:
                try:
                    profile, text = await self._generate_text_once(
                        prompt, system_extra, instruction=instruction, task=task,
                        max_tokens=max_tokens, on_delta=on_delta, on_restart=on_restart)
                    if complete_reply and instruction is None:
                        text = await self._complete_reply(
                            profile, text, prompt, system_extra, task=task,
                            max_tokens=max_tokens, on_delta=on_delta)
                    return profile, str(text)
                except (ProviderDeadlineError, StreamInterruptedError):
                    raise
                except (NoTextProviderError, AllTextProvidersFailedError):
                    # Both shapes of "nobody can answer right now": nothing usable
                    # at the start, or every provider tried and each rate-limited
                    # (and so cooled) during this very call.
                    wait = self.soonest_recovery_s()
                    now = time.monotonic()
                    if wait is None or now + wait > patience_until \
                            or now + wait + self.MIN_ATTEMPT_S > deadline:
                        raise
                    self.bus.log.emit(f"NET: every provider is rate-limited — waiting "
                                      f"{wait:.0f}s for the first to recover.")
                    await asyncio.sleep(min(wait + 0.5, deadline - now - self.MIN_ATTEMPT_S))

    async def propose_gemma_functions(
        self, prompt: str, declarations: list[dict[str, Any]], *,
        instruction: str | None = None,
    ) -> tuple[AIProviderProfile, list[dict[str, Any]], str]:
        """Get native Gemma function *proposals* without executing them.

        A caller must pass any accepted proposal through ORION's existing
        review and dispatch boundary. The router returns only a name and
        arguments; neither the model nor this method may report an action as
        complete before the dispatcher produces a verified receipt.
        """
        from .gemini_schema import sanitise_declarations

        clean, dropped = sanitise_declarations(declarations)
        if dropped or not clean or len(clean) > 32:
            raise ValueError("Supply 1–32 valid Gemma function declarations")
        prompt = SecuritySanitiser.guard_text(str(prompt or "").strip(),
                                              "gemma.function_prompt")
        if not prompt:
            raise ValueError("Gemma needs a prompt")
        profiles = [profile for profile in self.text_profiles()
                    if profile.kind == "gemma_native"]
        if not profiles:
            raise NoTextProviderError("No native Gemma provider is available")
        system = instruction or (
            "You are ORION. Propose a declared function when appropriate. "
            "Do not claim a proposal has run. Use British English.")
        last_error: BaseException | None = None
        with self._call_scope(_transport.INTERACTIVE, time.monotonic() + 120.0):
            for profile in profiles:
                try:
                    reply = await self._call_with_heal(
                        profile,
                        lambda p=profile: self._chat_attempt(
                            p, prompt, instruction=system, task="function_proposal",
                            function_declarations=clean),
                    )
                    return profile, list(getattr(reply, "tool_calls", [])), str(reply)
                except self._NOT_THE_PROVIDERS_FAULT:
                    raise
                except Exception as exc:
                    last_error = exc
        raise AllTextProvidersFailedError(
            last_error, attempts=[profile.name for profile in profiles])

    async def _complete_reply(
        self, profile: AIProviderProfile, text: str, prompt: str, system_extra: str,
        *, task: str, max_tokens: int | None,
        on_delta: Callable[[str], None] | None,
    ) -> str:
        """Resume output-limit stops on the same model without replaying tools.

        Five continuations and the original request deadline bound provider
        work. A failed continuation keeps all received text and clearly states
        that the answer is incomplete; it cannot silently masquerade as done.
        """
        reason = getattr(text, "done_reason", "")
        if reason not in {"length", "max_tokens", "max_output_tokens"}:
            return text
        raw = getattr(text, "raw_text", str(text))
        from dataclasses import replace
        profile = replace(profile, model=getattr(text, "model", self.active_model(profile)), models=())
        problem = "the response limit was reached"
        for number in range(5):
            scope = _CALL.get()
            remaining = scope.remaining() if scope else None
            if len(raw) >= 48_000 or (remaining is not None and remaining < self.MIN_ATTEMPT_S):
                problem = "the response time or size limit was reached"
                break
            self.bus.log.emit(f"NET: {profile.name}: continuing the answer ({number + 1}/5).")
            pieces: list[str] = []
            def forward(piece: str) -> None:
                pieces.append(piece)
                _transport._emit(on_delta, piece)
            try:
                # Do not heal, restart or change model after visible text: a
                # different answer must never be spliced into this one.
                part = await self._chat_attempt(
                    profile, prompt, system_extra, task=task,
                    max_tokens=max_tokens, continuation=raw,
                    stream=on_delta is not None,
                    on_delta=forward if on_delta is not None else None)
            except (ProviderClosedError, asyncio.CancelledError):
                raise
            except Exception as exc:
                if pieces:
                    raw += "".join(pieces)
                problem = "the provider could not finish the continuation"
                self.bus.log.emit(f"NET: {profile.name}: continuation stopped ({type(exc).__name__}).")
                break
            raw += getattr(part, "raw_text", str(part))
            if getattr(part, "done_reason", "") not in {"length", "max_tokens", "max_output_tokens"}:
                return clean_transcript(raw)
        warning = f"\n\n[This answer is incomplete: {problem}. Ask me to continue from the last sentence.]"
        _transport._emit(on_delta, warning)
        return clean_transcript(raw) + warning

    async def generate_local_text(self, prompt: str, *, instruction: str,
                                  task: str = "local",
                                  model: str = "",
                                  max_tokens: int | None = None,
                                  deadline_s: float | None = None,
                                  budget: str = "interactive",
                                  response_format: dict[str, Any] | None = None,
                                  temperature: float | None = None,
                                  ) -> tuple[AIProviderProfile, str]:
        """A turn answered by a model on THIS machine, or not at all.

        generate_text ranks providers by strength and falls through to the
        cloud, which is right for conversation and wrong for work that must
        never be billed or leave the machine (the optional neural planner).
        This walks the local profiles only, under the same call scope,
        endpoint gate, GPU lease and total deadline as any local request.

        ``model`` asks the first local server for a named model (a promoted
        planner tag) instead of the profile's own. Attempts are booked under
        "<profile>:<task>", so a long planning reply does not raise the
        latency estimate that decides whether a spoken turn may go local.
        There is no healing retry and no cloud fallback: failure raises.
        """
        from dataclasses import replace as _replace_profile
        prompt = SecuritySanitiser.guard_text(str(prompt or "").strip(), "local.prompt")
        if not prompt:
            raise RuntimeError("empty local prompt")
        if self._closed:
            raise ProviderClosedError("the provider router has been shut down")
        local = self.local_text_profiles()
        if not local:
            raise NoTextProviderError(
                "no local text model is available",
                configured=[p.name for p in self.settings.ordered_profiles()],
                online=self.is_online())
        suffix = str(task or "local").strip() or "local"
        profiles = [
            _replace_profile(p, name=f"{p.name}:{suffix}",
                             **({"model": str(model), "models": ()} if model else {}))
            for p in (local[:1] if model else local)]
        cls = self._budget_class(budget)
        total = (float(deadline_s) if deadline_s is not None
                 else self._default_deadline_s(budget, max_tokens, 0.0))
        deadline = time.monotonic() + max(0.0, total)
        last_error: BaseException | None = None
        attempted: list[str] = []
        with self._call_scope(cls, deadline):
            for profile in profiles:
                attempted.append(profile.name)
                try:
                    text = await self._chat_attempt(
                        profile, prompt, "", instruction=instruction, task=suffix,
                        max_tokens=max_tokens, response_format=response_format,
                        temperature=temperature)
                    return profile, text
                except (ProviderDeadlineError, ProviderClosedError):
                    raise
                except Exception as exc:
                    last_error = exc
        raise AllTextProvidersFailedError(last_error, attempts=attempted)

    #: The context the local agent's tool turns are loaded with: a shortlist
    #: of tool schemas, a few rounds of results and the reply share it.
    LOCAL_AGENT_NUM_CTX = _env_int("ORION_LOCAL_AGENT_NUM_CTX", 8192, 2048, 32768)

    async def local_tool_chat(self, messages: list[dict[str, Any]],
                              tools: list[dict[str, Any]], *,
                              model: str = "",
                              deadline_s: float = 30.0,
                              keep_alive: int | str | None = None,
                              num_ctx: int | None = None,
                              temperature: float = 0.2,
                              max_tokens: int = 512,
                              budget: str = "interactive",
                              task: str = "local_agent",
                              on_delta: Any = None) -> tuple[AIProviderProfile, Any]:
        """One tool-offering turn on THIS machine's Ollama, or an exception.

        The local agent's model call (local_agent.py). Ollama's native
        ``/api/chat`` with a ``tools`` list; the returned ChatResult carries
        ``tool_calls`` as proposals the agent checks and dispatches itself.
        Never a cloud model, never the /v1 door (which cannot carry tools,
        context or keep-alive), and never a healing retry. The same endpoint
        gate, GPU lease, residency bookkeeping (so shutdown and memory
        pressure release it) and one total deadline as every local request.
        ``keep_alive`` (e.g. "10m") keeps the model warm while hybrid or local
        mode is using it; memory pressure still forces 0. ``on_delta`` receives
        each fragment of the reply's text as it streams in (the local voice
        speaks the answer while it is still being written); cancelling the
        call closes the stream, which stops the generation.
        """
        from dataclasses import replace as _replace_profile
        if self._closed:
            raise ProviderClosedError("the provider router has been shut down")
        local = self.local_text_profiles()
        if not local:
            raise NoTextProviderError(
                "no local model is available",
                configured=[p.name for p in self.settings.ordered_profiles()],
                online=self.is_online())
        profile = local[0]
        if not self._may_be_ollama(profile):
            raise NoTextProviderError("the local agent needs Ollama's native API "
                                      "(ORION_OLLAMA_NATIVE is off)")
        root = _transport.ollama_root(profile.base_url)
        cls = self._budget_class(budget)
        deadline = time.monotonic() + max(0.0, float(deadline_s))
        with self._call_scope(cls, deadline):
            scope = _CALL.get() or _CallScope(cls=cls, deadline=deadline)
            if not await self._is_ollama(root, scope):
                raise NoTextProviderError(
                    f"the local server at {root} is not Ollama; the local agent needs Ollama")
            transport = _transport.OllamaToolTransport(root)
            chosen = str(model or "").strip() or self.active_model(profile)
            residency = (transport.root, _ollama_model_key(chosen))
            if self._memory_pressure:
                keep_value, keep_s = 0, 0
            elif keep_alive is not None:
                keep_s = _keep_alive_seconds(keep_alive)
                held = self._resident_until.get(residency, 0.0) - time.monotonic()
                if held > 0:
                    keep_s = int(math.ceil(max(keep_s, held)))
                keep_value = f"{keep_s}s" if keep_s > 0 else 0
            else:
                keep_value, keep_s = self._keep_alive_for(residency, cls)
            attempt = _replace_profile(profile, name=f"{profile.name}:{task}",
                                       model=chosen, models=())
            request = _transport.ToolChatRequest(
                model=chosen, messages=[dict(m) for m in messages], tools=list(tools or []),
                max_tokens=max(16, int(max_tokens)), api_key=self.active_key(profile),
                temperature=float(temperature),
                num_ctx=int(num_ctx or self.LOCAL_AGENT_NUM_CTX), keep_alive=keep_value,
                on_delta=on_delta if callable(on_delta) else None)
            gate = self._gate_for(profile, transport)
            remaining = scope.remaining()
            wait_limit = None if remaining is None else max(0.0, remaining - self.MIN_ATTEMPT_S)
            queue_started = time.monotonic()
            try:
                await gate.acquire(scope.cls, timeout=wait_limit)
            except (asyncio.TimeoutError, TimeoutError):
                raise ProviderDeadlineError(
                    f"{attempt.name} had no free slot before the time budget ran out",
                    attempts=[attempt.name]) from None
            queue_ms = (time.monotonic() - queue_started) * 1000.0
            trace: dict[str, Any] = {}
            result: Any = None
            ok = False
            try:
                remaining = scope.remaining()
                async with self._gpu_lease(scope.cls,
                                           timeout=None if remaining is None
                                           else max(0.0, remaining - self.MIN_ATTEMPT_S),
                                           provider=attempt.name):
                    self._inflight[residency] = self._inflight.get(residency, 0) + 1
                    try:
                        result = await self._send_attempt(attempt, transport, request, scope,
                                                          max_tokens, trace)
                    finally:
                        self._finish_inflight(residency)
                ok = True
            finally:
                gate.release(scope.cls)
                _trace_mark("provider_attempt", provider=attempt.name, transport="ollama_tools",
                            ok=ok, queue_ms=queue_ms, connect_ms=trace.get("connect_ms"),
                            first_token_ms=getattr(result, "first_token_ms", None),
                            load_ms=getattr(result, "load_ms", None))
            if keep_s > 0:
                self._resident_until[residency] = time.monotonic() + keep_s
            else:
                self._resident_until.pop(residency, None)
            self.note_latency(attempt, float(getattr(result, "total_ms", 0.0)) / 1000.0)
            self._record_usage(attempt, {"id": None, "model": chosen,
                                         "usage": dict(getattr(result, "usage", {}) or {})},
                               streaming=True, task=task)
            self._record_success(profile)
            return profile, result

    def _fallback_reserve_s(self, remaining: float | None) -> float:
        """Time to hold back for later providers, with *remaining* left."""
        if remaining is None:
            return 0.0
        return max(0.0, min(self.FALLBACK_RESERVE_S, remaining / 3.0))

    def _attempt_floor_s(self, profile: AIProviderProfile) -> float:
        """Time an attempt needs to be worth starting: the provider's recent
        response time when known, never less than MIN_ATTEMPT_S."""
        return max(self.MIN_ATTEMPT_S, self._latency_ema.get(profile.name, 0.0))

    async def _generate_text_once(self, prompt: str, system_extra: str = "", *,
                                  instruction: str | None = None,
                                  task: str = "",
                                  max_tokens: int | None = None,
                                  on_delta: Callable[[str], None] | None = None,
                                  on_restart: Callable[[str], None] | None = None,
                                  ) -> tuple[AIProviderProfile, str]:
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
        if self._closed:
            raise ProviderClosedError("the provider router has been shut down")
        scope = _CALL.get()
        remaining = scope.remaining() if scope is not None else None
        if remaining is not None and remaining <= 0:
            raise ProviderDeadlineError(attempts=[])
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
        try:
            return await self._try_text_profiles(
                profiles, scope, prompt, system_extra, instruction=instruction,
                task=task, max_tokens=max_tokens, on_delta=on_delta, on_restart=on_restart)
        finally:
            if scope is not None:
                scope.reserve_s = 0.0

    async def _try_text_profiles(self, profiles: list[AIProviderProfile],
                                 scope: "_CallScope | None",
                                 prompt: str, system_extra: str, *,
                                 instruction: str | None, task: str,
                                 max_tokens: int | None,
                                 on_delta: Callable[[str], None] | None,
                                 on_restart: Callable[[str], None] | None,
                                 ) -> tuple[AIProviderProfile, str]:
        """Each provider in turn until one answers (_generate_text_once)."""
        last_error: BaseException | None = None
        attempted: list[str] = []
        out_of_time = False
        for index, profile in enumerate(profiles):
            cap = self._output_cap.get(profile.name, 0)
            if max_tokens and cap and cap < min(int(max_tokens), self.MIN_USEFUL_OUTPUT_TOKENS):
                # It would answer, but in a fragment. Truncated output is how
                # the research outline broke; another provider will do.
                continue
            remaining = scope.remaining() if scope is not None else None
            if remaining is not None and remaining < self._attempt_floor_s(profile):
                # Starting it would only be cut off by the deadline; a faster
                # provider further down the list may still fit.
                out_of_time = True
                continue
            if scope is not None:
                # Another provider waits behind this one: keep it some time.
                reserve = (self._fallback_reserve_s(remaining)
                           if index < len(profiles) - 1 else 0.0)
                if remaining is not None and remaining - reserve < self._attempt_floor_s(profile):
                    reserve = 0.0
                scope.reserve_s = reserve
            attempted.append(profile.name)
            started = time.monotonic()
            try:
                if on_delta is None:
                    text = await self._call_with_heal(
                        profile,
                        lambda p=profile: self._openai_compatible_chat(
                            p, prompt, system_extra, instruction=instruction,
                            task=task, max_tokens=max_tokens))
                else:
                    text = await self._streamed_attempt(
                        profile, prompt, system_extra, instruction=instruction,
                        task=task, max_tokens=max_tokens,
                        on_delta=on_delta, on_restart=on_restart)
                self._broadcast_sentiment(text)
                self._exit_degraded()   # a success clears any degraded state
                self._announce("provider_route", {
                    "provider": profile.name, "local": profile.is_local,
                    "latency_s": round(time.monotonic() - started, 2),
                    "fallbacks": len(attempted) - 1})
                return profile, text
            except StreamInterruptedError:
                raise
            except ProviderClosedError:
                raise
            except ProviderDeadlineError as exc:
                last_error, out_of_time = exc, True
                break
            except Exception as exc:
                last_error = exc
        if out_of_time:
            raise ProviderDeadlineError(
                last_error or "the time budget for this request ran out", attempts=attempted)
        # Every eligible provider errored this turn.  If that leaves nothing
        # usable at all, we are degraded; otherwise it was a transient burst.
        if not self.text_profiles():
            self._enter_degraded(f"all providers failed: {redacted_error_line(last_error, 120)}")
        raise AllTextProvidersFailedError(last_error, attempts=attempted)

    async def _streamed_attempt(self, profile: AIProviderProfile, prompt: str,
                                system_extra: str, *, instruction: str | None,
                                task: str, max_tokens: int | None,
                                on_delta: Callable[[str], None],
                                on_restart: Callable[[str], None] | None) -> str:
        """One provider, streamed to the caller, under the restart contract."""
        shown = False
        scope = _CALL.get()

        def forward(piece: str) -> None:
            nonlocal shown
            if not shown and scope is not None and not scope.first_token_marked:
                scope.first_token_marked = True
                _trace_mark("first_token", provider=profile.name)
            shown = True
            try:
                on_delta(piece)
            except Exception:
                pass

        try:
            return await self._call_with_heal(
                profile,
                lambda: self._openai_compatible_chat_stream(
                    profile, prompt, system_extra, instruction=instruction,
                    task=task, on_delta=forward, max_tokens=max_tokens),
                # A retry after the caller has seen part of an answer would
                # splice a second answer onto it.
                may_retry=lambda: not shown)
        except self._NOT_THE_PROVIDERS_FAULT:
            # Nothing follows a deadline or a shutdown, so there is nothing to
            # restart into: the caller gets the error and keeps no answer.
            raise
        except Exception as exc:
            if not shown:
                raise
            if on_restart is None:
                raise StreamInterruptedError(profile.name, exc) from exc
            self._restart(on_restart, profile, "provider_failed")
            raise

    def _restart(self, on_restart: Callable[[str], None], profile: AIProviderProfile,
                 why: str) -> None:
        """Tell the caller to discard what it has shown. If it cannot be told,
        continuing would splice two answers, so the call fails instead."""
        _trace_mark("stream_restart", provider=profile.name, reason=why)
        try:
            on_restart(f"{why}:{profile.name}")
        except Exception as exc:
            raise StreamInterruptedError(profile.name, exc) from exc

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
                      if self.is_online() and not getattr(self, "local_only", False) else [])
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
                # The same bound as a deadline, so waiting for the provider's
                # slot and the HTTP timeout both live inside it too.
                with self._call_scope(_transport.INTERACTIVE,
                                      time.monotonic() + attempt_timeout):
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
        "Write a concise application reflection for ORION in strict British "
        "English, using one to three plain sentences. Summarise a recorded "
        "observation, its practical relevance and a useful next check, grounded "
        "only in the supplied context. This is a status summary, not a transcript "
        "of a model's hidden reasoning. Do not invent motives, emotions, events "
        "or completed actions. Distinguish an observation from a suggestion and "
        "state uncertainty where the evidence is incomplete. Omit salutations, "
        "quotation marks and repeated generic advice."
    )

    def thought_profile(self) -> AIProviderProfile | None:
        """The dedicated introspection backend, when configured and healthy."""
        profile = self.settings.providers.get(self.THOUGHTS_PROFILE)
        if profile is not None and profile.supports_text_generation \
                and self.is_available(profile) and self.is_online() \
                and (profile.is_local or not getattr(self, "local_only", False)):
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
                raise ThoughtGenerationUnavailable(
                    "no local model is running, and thinking is local-only "
                    "(ORION_PAID_THOUGHTS=1 to allow a paid provider)")
        return profiles

    async def generate_thought(self, prompt: str) -> tuple[AIProviderProfile, str]:
        """One inner-monologue turn: a local model leads when one is running,
        the dedicated cloud thoughts profile is next, the ordinary text chain
        is the last resort.  Uses a compact introspection instruction (not
        the full tool persona) so thinking stays cheap, and tags usage
        task='thought' so the ledger separates mind from mouth. Background
        work: it queues behind conversation for a provider's slots."""
        with self._call_scope(_transport.BACKGROUND, time.monotonic()
                              + self._default_deadline_s("background", None, 0.0)):
            return await self._generate_thought(prompt)

    async def _generate_thought(self, prompt: str) -> tuple[AIProviderProfile, str]:
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
            except self._NOT_THE_PROVIDERS_FAULT:
                raise
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
        Background work, like generate_thought.
        """
        with self._call_scope(_transport.BACKGROUND, time.monotonic()
                              + self._default_deadline_s("background", None, 0.0)):
            return await self._generate_thought_stream(prompt, on_delta)

    async def _generate_thought_stream(
        self, prompt: str, on_delta: Callable[[str], None] | None = None,
    ) -> tuple[AIProviderProfile, str]:
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
            except self._NOT_THE_PROVIDERS_FAULT:
                raise
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
                except self._NOT_THE_PROVIDERS_FAULT:
                    raise
                except Exception as exc2:
                    last_error = exc2
                    self.mark_failure(profile, exc2)
        raise AllTextProvidersFailedError(
            last_error, attempts=[p.name for p in profiles])

    async def _openai_compatible_chat_stream(
        self, profile: AIProviderProfile, prompt: str, system_extra: str = "",
        *, instruction: str | None = None, task: str = "",
        on_delta: Callable[[str], None] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """One provider attempt, streamed: each fragment goes to ``on_delta``
        as it arrives and the assembled text is returned.

        The name predates the Ollama transport and is kept because callers
        and tests replace this method as the streaming seam; the wire format
        is chosen per server (SSE for OpenAI-compatible, NDJSON for Ollama).
        A stream that ends without its end marker, or reports an error
        mid-way, is a failure, never a short answer."""
        return await self._chat_attempt(
            profile, prompt, system_extra, instruction=instruction, task=task,
            max_tokens=max_tokens, stream=True, on_delta=on_delta)

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
        try:
            host = (urlsplit(profile.base_url).hostname or "").lower()
        except ValueError:
            host = ""
        if host == "generativelanguage.googleapis.com":
            if model in {"gemini-2.5-flash", "gemini-2.5-flash-lite"}:
                payload["reasoning_effort"] = "none"
            elif model in {"gemini-3.1-flash-lite", "gemini-3.5-flash-lite"}:
                payload["reasoning_effort"] = "minimal"
            elif model in {"gemini-3-flash-preview", "gemini-3.5-flash",
                           "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"}:
                # Gemini 3 cannot disable thinking. "low" is supported by
                # these Flash models; "none" would reject the request (400).
                payload["reasoning_effort"] = "low"
        elif "gpt-oss" in model and host == "api.groq.com":
            payload["reasoning_effort"] = "low"
        else:
            # These fields are provider-specific, not generic model settings.
            # Restrict them to the documented API and recognised model family.
            if host == "api.deepseek.com" and model in {"deepseek-flash", "deepseek-v4-pro"}:
                payload["thinking"] = {"type": "disabled"}
            elif host == "api.cerebras.ai" and model == "gpt-oss-120b":
                payload["reasoning_effort"] = "low"

    async def _openai_compatible_chat(
        self, profile: AIProviderProfile, prompt: str, system_extra: str = "",
        *, instruction: str | None = None, task: str = "",
        image_jpeg: bytes | None = None, max_tokens: int | None = None,
    ) -> str:
        """One provider attempt, whole answer at once.

        The name predates the Ollama transport and is kept because callers
        and tests replace this method as the per-attempt seam; the wire format
        is chosen per server by _chat_attempt."""
        return await self._chat_attempt(
            profile, prompt, system_extra, instruction=instruction, task=task,
            image_jpeg=image_jpeg, max_tokens=max_tokens)

    # ── transport, admission and residency ────────────────────────────────────

    def _may_be_ollama(self, profile: AIProviderProfile) -> bool:
        """Worth asking whether this server is Ollama: it is on this machine,
        or it is the profile OllamaManager registers. ORION_OLLAMA_NATIVE=0
        keeps every server on /v1 (the rollback switch)."""
        if os.getenv("ORION_OLLAMA_NATIVE", "1").strip().lower() in {"0", "false", "no", "off"}:
            return False
        return profile.is_local or profile.name == "local_ollama"

    async def _is_ollama(self, root: str, scope: _CallScope | None = None) -> bool:
        """Whether *root* is an Ollama server, asked once and remembered.

        Raises ConnectionError when nothing answers, so a dead local server
        fails the attempt in about a second rather than after its full
        two-minute request timeout.
        """
        cached = self._ollama_roots.get(root)
        now = time.monotonic()
        if cached is not None and (cached[0] or now - cached[2] < self.NOT_OLLAMA_TTL_S):
            return cached[0]
        remaining = scope.remaining() if scope is not None else None
        timeout = (self.LOCAL_PROBE_TIMEOUT_S if remaining is None
                   else max(0.1, min(self.LOCAL_PROBE_TIMEOUT_S, remaining)))
        session = await self._pool.get()
        version = await _transport.probe_ollama(session, root, timeout=timeout)
        self._ollama_roots[root] = (version is not None, version or "", now)
        if version is not None and cached is None:
            self.bus.log.emit(
                f"NET: {root} is Ollama {version} - using its native API "
                f"(context {self.LOCAL_NUM_CTX} tokens, kept loaded "
                f"{self.LOCAL_KEEP_ALIVE_S}s after a conversation turn).")
        return version is not None

    async def _transport_for(self, profile: AIProviderProfile,
                             scope: _CallScope | None = None) -> Any:
        base_url = profile.base_url.rstrip("/")
        if profile.kind == "gemma_native":
            from .gemma_native import GemmaNativeTransport
            return GemmaNativeTransport(base_url)
        if self._may_be_ollama(profile):
            root = _transport.ollama_root(base_url)
            if await self._is_ollama(root, scope):
                return _transport.OllamaChatTransport(root)
        # Everything else — cloud, LM Studio, llama.cpp, vLLM — speaks /v1 and
        # never sees an Ollama-only field.
        return _transport.OpenAIChatTransport(base_url)

    def _gate_for(self, profile: AIProviderProfile, transport: Any) -> _transport.EndpointGate:
        """The admission gate every consumer of this provider shares. Local
        servers are gated per server (one GPU), cloud providers per profile."""
        if profile.is_local:
            root = getattr(transport, "root", "") or _transport.ollama_root(profile.base_url)
            key, limit, background = (f"local:{root}", self.LOCAL_CONCURRENCY,
                                      self.LOCAL_CONCURRENCY)
        else:
            key, limit, background = (f"profile:{profile.name}", self.CLOUD_CONCURRENCY,
                                      min(self.CLOUD_CONCURRENCY,
                                          self.CLOUD_BACKGROUND_CONCURRENCY))
        gate = self._gates.get(key)
        if gate is None:
            gate = self._gates[key] = _transport.EndpointGate(limit, background)
        return gate

    @contextlib.asynccontextmanager
    async def _gpu_lease(self, cls: str, timeout: float | None = None, provider: str = ""):
        """Hold the shared GPU lease throughout local generation. Training
        relinquishes this lease only after its CUDA process has exited.

        The wait is bounded by what is left of the request's own deadline,
        and a GPU that is saturated (or admission shutting down) surfaces as
        a deadline, not as a failure of the local provider: the circuit
        breaker must not learn "Ollama is broken" from "the GPU was busy".
        """
        from . import admission
        ticket = None
        try:
            ticket = admission.ADMISSION.acquire_lease("gpu", cls, label="Local language model")
            await ticket.wait(timeout)
        except admission.Busy as busy:
            if ticket is not None:
                ticket.release()
            raise ProviderDeadlineError(f"the GPU is busy: {busy.reason}",
                                        attempts=[provider] if provider else []) from None
        try:
            yield ticket
        finally:
            ticket.release()

    def _keep_alive_for(self, key: tuple[str, str], cls: str) -> tuple[int | str, int]:
        """(value sent to Ollama, seconds) for a request on *key*."""
        if self._memory_pressure:
            return 0, 0
        want = (self.LOCAL_KEEP_ALIVE_S if cls == _transport.INTERACTIVE
                else self.LOCAL_BACKGROUND_KEEP_ALIVE_S)
        held = self._resident_until.get(key, 0.0) - time.monotonic()
        # Each request resets Ollama's timer to ITS keep_alive, so a short
        # background request after a conversation turn would otherwise cut
        # the conversation's window short.
        seconds = int(math.ceil(max(want, held))) if held > 0 else int(want)
        return (f"{seconds}s" if seconds > 0 else 0), seconds

    def _finish_inflight(self, key: tuple[str, str]) -> None:
        count = self._inflight.get(key, 1) - 1
        if count > 0:
            self._inflight[key] = count
            return
        self._inflight.pop(key, None)
        if key in self._unload_pending and not self._closed:
            # Asked for while this model was answering; it is free now.
            self._unload_pending.discard(key)
            self._spawn(self._unload_one(key[0], key[1], reason="memory pressure"))

    def _spawn(self, coro: Any) -> None:
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
            return
        # Held until done: a task nothing references can be collected mid-way.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _prompt_work(self, build: Callable[[], Any], scope: _CallScope) -> Any:
        """Prepare grounded context off the live loop within the turn budget."""
        started = time.monotonic()
        remaining = scope.remaining()
        try:
            work = asyncio.to_thread(build)
            if remaining is None:
                return await work
            if remaining <= 0:
                work.close()
                raise ProviderDeadlineError("the time budget ended before context preparation")
            return await asyncio.wait_for(work, timeout=remaining)
        except (asyncio.TimeoutError, TimeoutError):
            raise ProviderDeadlineError("context preparation exceeded the time budget") from None
        finally:
            _trace_mark("prompt_context", ms=(time.monotonic() - started) * 1000.0)

    async def _chat_attempt(
        self, profile: AIProviderProfile, prompt: str, system_extra: str = "",
        *, instruction: str | None = None, task: str = "",
        image_jpeg: bytes | None = None, max_tokens: int | None = None,
        stream: bool = False, on_delta: Callable[[str], None] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        continuation: str = "",
        function_declarations: list[dict[str, Any]] | None = None,
    ) -> str:
        """One request to one provider: transport, admission, timeouts inside
        the call's deadline, residency bookkeeping and measurement.

        ``response_format`` (a JSON schema) reaches native Ollama only; other
        servers never see it. ``temperature`` None keeps the default."""
        if self._closed:
            raise ProviderClosedError("the provider router has been shut down")
        base_url = profile.base_url.rstrip("/")
        if not base_url:
            raise RuntimeError(f"provider {profile.name} has no base_url")
        scope = _CALL.get() or _CallScope()
        attempt_started = time.monotonic()
        transport = await self._transport_for(profile, scope)
        native = transport.kind == "ollama"
        def prepare_prompt() -> tuple[str, str]:
            system_content = (instruction if instruction is not None
                              else self.system_instruction(system_extra, context=task))
            return self._fit_to_budget(
                profile, system_content, prompt, system_extra, instruction, task,
                cap=self.LOCAL_NUM_CTX if native else None)
        # Shape the request to this model's payload budget BEFORE sending —
        # a small hosted model must never see the full instruction stack
        # (that is what produced HTTP 413 and a 300 s bench). A native Ollama
        # request is also held inside the context it is really loaded with.
        if instruction is None:
            # Persistent recall and persona grounding may touch SQLite/files.
            # They must not stop audio or avatar frames on the live loop.
            system_content, prompt = await self._prompt_work(prepare_prompt, scope)
        else:
            system_content, prompt = prepare_prompt()
        if continuation:
            # Fit the preceding tail alongside the question and output reserve.
            # This also prevents native Ollama from evicting the instruction.
            context = min(self.payload_budget(profile), self.LOCAL_NUM_CTX) if native else self.payload_budget(profile)
            reserve = min(self._reply_ceiling(profile, max_tokens), max(256, context // 4))
            if self._est_tokens(system_content + prompt) + reserve + 256 >= context:
                system_content = await self._prompt_work(
                    lambda: self.system_instruction_lean(system_extra, context=task), scope)
                prompt = prompt[:max(256, (context // 4) * 3)]
            room = max(128, (context - reserve - self._est_tokens(system_content + prompt) - 256) * 3)
            continuation = continuation[-room:]
        model = self.active_model(profile)
        request = _transport.ChatRequest(
            model=model, system=system_content, prompt=prompt,
            # ALWAYS bounded — see CREDIT_RE.
            max_tokens=self._reply_ceiling(profile, max_tokens),
            api_key=self.active_key(profile), image_jpeg=image_jpeg,
            stream=stream, on_delta=on_delta,
            continuation=continuation,
            # Inspection replies are untrusted JSON; bound what is read.
            max_response_bytes=262_144 if image_jpeg is not None else None)
        if temperature is not None:
            request.temperature = float(temperature)
        if function_declarations is not None:
            if profile.kind != "gemma_native":
                raise ValueError("Native Gemma functions require a Gemma profile")
            request.extra["function_declarations"] = function_declarations
        if native and response_format is not None:
            request.response_format = response_format
        residency: tuple[str, str] | None = None
        keep_s = 0
        if native:
            # Prompt and reply share the context: a reply allowed to run past
            # it would push the start of the prompt out mid-answer.
            room = (self.LOCAL_NUM_CTX - self._est_tokens(system_content)
                    - self._est_tokens(prompt) - self._est_tokens(continuation) -
                    (self._est_tokens(_transport.CONTINUE_REPLY) if continuation else 0))
            request.num_ctx = self.LOCAL_NUM_CTX
            request.max_tokens = min(request.max_tokens, max(256, room))
            residency = (transport.root, _ollama_model_key(model))
            request.keep_alive, keep_s = self._keep_alive_for(residency, scope.cls)
        else:
            quirks: dict[str, Any] = {"model": model}
            self._apply_model_quirks(profile, quirks)
            quirks.pop("model", None)
            request.extra.update(quirks)

        gate = self._gate_for(profile, transport)
        remaining = scope.remaining()
        wait_limit = None if remaining is None else max(0.0, remaining - self.MIN_ATTEMPT_S)
        queue_started = time.monotonic()
        try:
            await gate.acquire(scope.cls, timeout=wait_limit)
        except (asyncio.TimeoutError, TimeoutError):
            _trace_mark("provider_attempt", provider=profile.name, transport=transport.kind,
                        ok=False, queue_ms=(time.monotonic() - queue_started) * 1000.0,
                        ms=(time.monotonic() - attempt_started) * 1000.0)
            raise ProviderDeadlineError(
                f"{profile.name} had no free slot before the time budget ran out",
                attempts=[profile.name]) from None
        queue_ms = (time.monotonic() - queue_started) * 1000.0
        trace: dict[str, Any] = {}
        result: Any = None
        ok = False
        try:
            remaining = scope.remaining()
            lease = (self._gpu_lease(scope.cls,
                                     timeout=None if remaining is None
                                     else max(0.0, remaining - self.MIN_ATTEMPT_S),
                                     provider=profile.name)
                     if profile.is_local else contextlib.nullcontext())
            async with lease:
                if residency is not None:
                    self._inflight[residency] = self._inflight.get(residency, 0) + 1
                try:
                    result = await self._send_attempt(
                        profile, transport, request, scope, max_tokens, trace)
                finally:
                    if residency is not None:
                        self._finish_inflight(residency)
            # Inspection JSON contains literal component markings. Transcript
            # identity correction (e.g. ORIN -> ORION) must not rewrite evidence.
            content = (str(result.text).strip() if image_jpeg is not None
                       else clean_transcript(str(result.text)))
            if not content and not (function_declarations is not None and result.tool_calls):
                raise RuntimeError(
                    f"provider {profile.name} "
                    f"{'streamed no content' if stream else 'returned empty content'}")
            ok = True
        finally:
            gate.release(scope.cls)
            _trace_mark(
                "provider_attempt", provider=profile.name, transport=transport.kind, ok=ok,
                queue_ms=queue_ms, connect_ms=trace.get("connect_ms"),
                first_token_ms=getattr(result, "first_token_ms", None),
                load_ms=getattr(result, "load_ms", None),
                ms=(time.monotonic() - attempt_started) * 1000.0)
        # Observed latency feeds the orchestration ordering for future turns.
        self.note_latency(profile, result.total_ms / 1000.0)
        if residency is not None:
            if keep_s > 0:
                self._resident_until[residency] = time.monotonic() + keep_s
            else:
                self._resident_until.pop(residency, None)
        usage = dict(result.usage)
        if not usage and result.streamed:
            # Most SSE streams omit the usage block; approximate for the ledger.
            usage = {"completion_tokens": max(1, len(content) // 4)}
        # Record authoritative token usage (Section 6) — deduped by response id.
        self._record_usage(profile, {"id": result.response_id or None,
                                     "model": result.model or None, "usage": usage},
                           streaming=stream, task=task)
        self._record_success(profile)
        if not self._provider_ready and image_jpeg is None:
            self._provider_ready = True
            try:
                from .startup_budget import milestone
                milestone("provider_ready")
            except Exception:
                pass
        answer = ProviderText(content, result.done_reason, str(result.text))
        answer.model = model
        answer.tool_calls = list(result.tool_calls)
        return answer

    async def _send_attempt(self, profile: AIProviderProfile, transport: Any,
                            request: Any, scope: _CallScope, max_tokens: int | None,
                            trace: dict[str, Any]) -> Any:
        """The HTTP exchange, with this attempt's timeout cut to the deadline."""
        aiohttp = _transport._aiohttp()
        # A long reply takes longer to write. A 1,300-word research section on
        # a free model needs well over the 30 s a spoken answer is allowed.
        limit_s = max(12.0 if request.stream else 8.0, float(profile.timeout_s or 30.0))
        if max_tokens:
            limit_s = max(limit_s, 20.0 + int(max_tokens) / 30.0)
        # Less any time held back for the providers after this one.
        remaining = scope.attempt_remaining()
        clipped = remaining is not None and remaining < limit_s
        seconds = max(0.1, min(limit_s, remaining)) if remaining is not None else limit_s
        # Loopback connects either succeed at once or hang (this machine
        # drops, rather than refuses, connections to a closed local port).
        connect_s = min(seconds, 2.0 if profile.is_local else 15.0)
        timeout = ClientTimeout(total=seconds, sock_connect=connect_s)
        try:
            session = await self._pool.get()
            with self._admit(profile):
                return await transport.chat(session, request, timeout=timeout, trace=trace)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            if self._closed:
                raise ProviderClosedError("the provider router has been shut down") from exc
            if isinstance(exc, getattr(aiohttp, "ConnectionTimeoutError", ())):
                self._forget_server(transport)
                raise TimeoutError(
                    f"provider {profile.name} could not connect within {connect_s:.0f}s") from exc
            if clipped:
                left = scope.remaining()
                if scope.reserve_s > 0 and left is not None and left >= self.MIN_ATTEMPT_S:
                    # Cut short to keep time for the next provider: an
                    # ordinary failure of this one, so the next is asked.
                    raise TimeoutError(
                        f"provider {profile.name} gave no reply within {seconds:.0f}s "
                        f"(the rest of the time is kept for another provider)") from exc
                raise ProviderDeadlineError(
                    f"{profile.name} was still answering when the time budget ran out",
                    attempts=[profile.name]) from exc
            raise TimeoutError(
                f"provider {profile.name} gave no reply within {seconds:.0f}s") from exc
        except Exception as exc:
            if self._closed:
                raise ProviderClosedError("the provider router has been shut down") from exc
            if isinstance(exc, (OSError, aiohttp.ClientConnectionError)):
                self._forget_server(transport)
            raise

    def _forget_server(self, transport: Any) -> None:
        """A local server that stopped answering may come back as something
        else; ask it again next time rather than trusting the old answer."""
        root = getattr(transport, "root", None)
        if root:
            self._ollama_roots.pop(root, None)

    # ── residency control (memory pressure, shutdown) ─────────────────────────

    def set_memory_pressure(self, active: bool) -> None:
        """While set, local models are released as soon as each reply is done
        (keep_alive 0). Cleared by the resource governor on recovery."""
        self._memory_pressure = bool(active)

    async def relieve_memory_pressure(self, reason: str = "memory pressure") -> list[str]:
        """What the resource governor calls when memory runs short: stop
        keeping local models warm, and release the ones held now."""
        self.set_memory_pressure(True)
        return await self.unload_local_models(reason)

    def _local_ollama_models(self) -> dict[str, set[str]]:
        """root -> the models ORION uses there. Only these are ever unloaded:
        another program's model on the same server is not ORION's to evict."""
        found: dict[str, set[str]] = {}
        for profile in self.settings.ordered_profiles():
            if not profile.supports_text_generation or not self._may_be_ollama(profile):
                continue
            root = _transport.ollama_root(profile.base_url)
            found.setdefault(root, set()).update(
                _ollama_model_key(m) for m in profile.all_models())
        for root, model in list(self._resident_until):
            found.setdefault(root, set()).add(model)
        return found

    async def unload_local_models(self, reason: str = "memory pressure") -> list[str]:
        """Release ORION's local Ollama models from memory now.

        A model that is answering a request is not unloaded under it: the
        unload is remembered and done the moment that request finishes.
        Returns the models released.
        """
        if self._closed:
            return []
        if reason == "shutdown" and not (
            self._resident_until or self._inflight or self._unload_pending
        ):
            # Configuring Ollama is not evidence that this process loaded a
            # model. Avoid a discovery round trip on every cloud-only exit.
            return []
        released: list[str] = []
        for root, models in self._local_ollama_models().items():
            try:
                if not await self._is_ollama(root):
                    continue
                session = await self._pool.get()
                loaded = await _transport.OllamaChatTransport(root).loaded_models(session)
            except Exception:
                continue
            for row in loaded:
                name = str(row.get("name") or row.get("model") or "")
                model = _ollama_model_key(name)
                if model not in models:
                    continue
                if await self._unload_one(root, model):
                    released.append(name)
        if released:
            self.bus.log.emit(
                f"NET: released local model{'s' if len(released) > 1 else ''} "
                f"{', '.join(released)} from memory ({reason}).")
        return released

    async def _unload_one(self, root: str, model: str, reason: str = "") -> bool:
        key = (root, model)
        if self._inflight.get(key):
            self._unload_pending.add(key)
            return False
        try:
            session = await self._pool.get()
            await _transport.OllamaChatTransport(root).unload(session, model)
        except Exception as exc:
            self.bus.log.emit(
                f"NET: could not release local model {model} - {redacted_error_line(exc, 90)}")
            return False
        self._resident_until.pop(key, None)
        if reason:
            self.bus.log.emit(f"NET: released local model {model} from memory ({reason}).")
        return True

    async def aclose(self) -> None:
        """Close pooled connections and stop background unloads. Idempotent.
        No provider call can start afterwards (ProviderClosedError)."""
        if self._closed:
            return
        self._closed = True
        self._model_catalogues.clear()
        self._model_catalogue_locks.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        tasks = [t for t in list(self._tasks) if not t.done()]
        for task in tasks:
            task.cancel()
        mine = [t for t in tasks if loop is not None and t.get_loop() is loop]
        if mine:
            await asyncio.gather(*mine, return_exceptions=True)
        await self._pool.aclose()


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
    mistral_key = os.getenv("ORION_MISTRAL_API_KEY", "").strip()
    huggingface_key = os.getenv("ORION_HUGGINGFACE_API_KEY", "").strip()
    nvidia_key = os.getenv("ORION_NVIDIA_API_KEY", "").strip()
    deepseek_key = os.getenv("ORION_DEEPSEEK_API_KEY", "").strip()
    cloudflare_key = os.getenv("ORION_CLOUDFLARE_API_TOKEN", "").strip()
    cloudflare_account = os.getenv("ORION_CLOUDFLARE_ACCOUNT_ID", "").strip()
    cloudflare_ready = bool(cloudflare_key and re.fullmatch(r"[0-9a-fA-F]{32}", cloudflare_account))
    cloudflare_url = (
        f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account}/ai/v1"
        if cloudflare_ready else ""
    )
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
            "gemini_text",      # quick, high-throughput text work
            "gemini_reasoning", # longer, more demanding text work
            "gemma_4_31b",      # native Gemini API, not OpenAI chat
            "gemma_4_26b",      # smaller hosted Gemma 4 fallback
            "cloudflare",       # low-cost Workers AI with a daily free allowance
            "openai",
            "xai_grok",
            "cerebras",         # ultra-fast inference (Cerebras wafer-scale)
            "sambanova",        # ultra-fast inference (SambaNova RDU)
            "openrouter",
            "groq",
            "mistral",
            "huggingface",
            "nvidia",
            "together",
            "deepseek",
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
                "model": _GEMINI_TEXT_MODELS[0],
                "models": list(_GEMINI_TEXT_MODELS),
                "base_url": GEMINI_OPENAI_BASE,
                "priority": 12,
                "timeout_s": 90.0,
                "strengths": ["fast", "general"],
            },
            "gemini_reasoning": {
                "kind": "openai_compatible",
                "enabled": bool(gemini_key),
                "api_key": gemini_key,
                "model": _GEMINI_REASONING_MODELS[0],
                "models": list(_GEMINI_REASONING_MODELS),
                "base_url": GEMINI_OPENAI_BASE,
                "priority": 13,
                "timeout_s": 90.0,
                "strengths": ["reasoning", "writing", "coding"],
            },
            "gemma_4_31b": {
                "kind": "gemma_native",
                "enabled": bool(gemini_key),
                "api_key": gemini_key,
                "model": "gemma-4-31b-it",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "priority": 14,
                "timeout_s": 90.0,
                "strengths": ["reasoning", "writing", "coding"],
            },
            "gemma_4_26b": {
                "kind": "gemma_native",
                "enabled": bool(gemini_key),
                "api_key": gemini_key,
                "model": "gemma-4-26b-a4b-it",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "priority": 15,
                "timeout_s": 90.0,
                "strengths": ["fast", "general"],
            },
            "cloudflare": {
                "kind": "openai_compatible",
                "enabled": cloudflare_ready,
                "api_key": cloudflare_key,
                "model": os.getenv("ORION_CLOUDFLARE_MODEL", "").strip()
                         or "@cf/google/gemma-4-26b-a4b-it",
                "base_url": cloudflare_url,
                "priority": 16,
                "timeout_s": 45.0,
                "strengths": ["fast", "general", "reasoning"],
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
                "model": os.getenv("ORION_TOGETHER_MODEL", "").strip() or "Qwen/Qwen3.5-9B",
                "base_url": "https://api.together.ai/v1",
                "priority": 50,
                "timeout_s": 30.0,
                "strengths": ["fast"],
            },
            # Free-mode access is rate-limited by the Mistral workspace.
            "mistral": {
                "kind": "openai_compatible",
                "enabled": bool(mistral_key),
                "api_key": mistral_key,
                "model": os.getenv("ORION_MISTRAL_MODEL", "").strip() or "mistral-small-latest",
                "base_url": "https://api.mistral.ai/v1",
                "priority": 34,
                "timeout_s": 45.0,
                "strengths": ["general", "writing", "reasoning"],
            },
            # Hugging Face routes chat completions to an inference provider.
            # Its :cheapest policy selects the lowest listed output-token rate;
            # the account's monthly free credit is finite.
            "huggingface": {
                "kind": "openai_compatible",
                "enabled": bool(huggingface_key),
                "api_key": huggingface_key,
                "model": os.getenv("ORION_HUGGINGFACE_MODEL", "").strip()
                         or "openai/gpt-oss-120b:cheapest",
                "base_url": "https://router.huggingface.co/v1",
                "priority": 35,
                "timeout_s": 45.0,
                "strengths": ["general", "reasoning"],
            },
            # NVIDIA's hosted NIM endpoints provide development access.
            "nvidia": {
                "kind": "openai_compatible",
                "enabled": bool(nvidia_key),
                "api_key": nvidia_key,
                "model": os.getenv("ORION_NVIDIA_MODEL", "").strip()
                         or "meta/llama-3.3-70b-instruct",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "priority": 36,
                "timeout_s": 45.0,
                "strengths": ["reasoning", "writing"],
            },
            # Paid usage is opt-in. This gives ORION a low-cost hosted fallback
            # once its free allowances have run out.
            "deepseek": {
                "kind": "openai_compatible",
                "enabled": bool(deepseek_key),
                "api_key": deepseek_key,
                "model": os.getenv("ORION_DEEPSEEK_MODEL", "").strip() or "deepseek-flash",
                "base_url": "https://api.deepseek.com",
                "priority": 55,
                "timeout_s": 60.0,
                "strengths": ["reasoning", "coding", "general"],
            },
            # ── Cerebras Cloud — hosted inference with a compatible API.
            # Its public API is OpenAI-compatible (the cerebras_cloud_sdk is a
            # thin wrapper over this same endpoint), so it needs no extra
            # dependency and inherits ORION's fallback, budgeting and ledgering.
            "cerebras": {
                "kind": "openai_compatible",
                "enabled": bool(cerebras_key),
                "api_key": cerebras_key,
                "model": os.getenv("ORION_CEREBRAS_MODEL", "").strip() or "gpt-oss-120b",
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
                # LOCAL_NUM_CTX and the residency policy (LOCAL_KEEP_ALIVE_S,
                # released early under memory pressure), both applied through
                # Ollama's native API (see _chat_attempt).
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
            "xAI (Grok) is reached through its OpenAI-compatible chat endpoint.",
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


_PROVIDER_CONFIG_LOCK = threading.RLock()
_PROVIDER_CONFIG_MAX_BYTES = 8 * 1024 * 1024
_PROVIDER_CONFIG_SCHEMA = "orion.mark_viii.providers.v1"
_PROVIDER_CONFIG_REPAIR = (
    "Provider settings are unavailable. Restore or repair api_keys.json; "
    "the saved file has been preserved and no default providers were enabled."
)


def _read_provider_document(path: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Read bounded, unambiguous saved policy; None is a genuinely missing file."""
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            raw = stream.read(_PROVIDER_CONFIG_MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except FileNotFoundError:
        # A file disappearing after open is a changed read, not first run.
        if "before" in locals():
            raise ProviderConfigurationError(_PROVIDER_CONFIG_REPAIR) from None
        return None, None
    except (OSError, UnicodeError) as exc:
        raise ProviderConfigurationError(_PROVIDER_CONFIG_REPAIR) from exc
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if (len(raw) > _PROVIDER_CONFIG_MAX_BYTES or len(raw) != before.st_size
            or identity(before) != identity(after) or identity(after) != identity(current)):
        raise ProviderConfigurationError(_PROVIDER_CONFIG_REPAIR)

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate policy field")
            result[key] = value
        return result

    try:
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite value")))
        if not isinstance(data, dict) or data.get("schema", _PROVIDER_CONFIG_SCHEMA) != _PROVIDER_CONFIG_SCHEMA:
            raise ValueError("unsupported provider policy")
        for key in ("providers", "integrations"):
            if key in data and not isinstance(data[key], dict):
                raise ValueError("invalid policy mapping")
        if "provider_order" in data and (not isinstance(data["provider_order"], list)
                or any(not isinstance(item, str) for item in data["provider_order"])):
            raise ValueError("invalid provider order")
        for raw_profile in data.get("providers", {}).values():
            if not isinstance(raw_profile, dict):
                raise ValueError("invalid provider record")
            if "enabled" in raw_profile and type(raw_profile["enabled"]) is not bool:
                raise ValueError("invalid provider enablement")
            for key in ("kind", "model", "api_key", "base_url"):
                if key in raw_profile and not isinstance(raw_profile[key], str):
                    raise ValueError("invalid provider text field")
            for key in ("strengths", "api_keys", "models"):
                if key in raw_profile and (not isinstance(raw_profile[key], list)
                        or any(not isinstance(item, str) for item in raw_profile[key])):
                    raise ValueError("invalid provider list")
            for key in ("priority", "budget_tokens", "timeout_s"):
                if key in raw_profile and not math.isfinite(float(raw_profile[key])):
                    raise ValueError("nonfinite provider number")
        # Mark XXXX.3: credentials on disk are sealed to the Windows account
        # (key_seal.py); everything above this line sees them as stored.
        return key_seal.open_tree(data), hashlib.sha256(raw).hexdigest()
    except (ValueError, UnicodeError, RecursionError, TypeError) as exc:
        raise ProviderConfigurationError(_PROVIDER_CONFIG_REPAIR) from exc


def _publish_provider_document(payload: dict[str, Any], observed: str | None) -> str:
    """Process-serialized observed-change guard, not an interprocess CAS."""
    saved, current = _read_provider_document(API_CONFIG_PATH)
    if current != observed:
        raise ProviderConfigurationError("Provider settings changed on disk. Reload them before saving.")
    if saved == payload:
        return current
    try:
        on_disk = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        on_disk = None
    rendered = json.dumps(key_seal.seal_tree(key_seal.keep_unopenable(payload, on_disk)),
                          indent=4)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(API_CONFIG_PATH, rendered, encoding="utf-8")
    published, fingerprint = _read_provider_document(API_CONFIG_PATH)
    if published != payload:
        raise ProviderConfigurationError("Provider settings changed while saving. Reload them before saving again.")
    return fingerprint


def _rescue_api_config() -> None:
    """api_keys.json has gone missing from the canonical path before while a
    stray copy survived elsewhere (a legacy build wrote under config/config/).
    Losing the file silently drops every configured provider, so if the
    canonical file is absent restore it from the newest rescue copy found."""
    canonical, _ = _read_provider_document(API_CONFIG_PATH)
    if canonical is not None:
        return
    candidates = (
        CONFIG_DIR / "config" / "api_keys.json",
        CONFIG_DIR.parent / "legacy" / "config" / "api_keys.json",
    )
    for candidate in candidates:
        try:
            data, _ = _read_provider_document(candidate)
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
            with _PROVIDER_CONFIG_LOCK:
                _publish_provider_document(data, None)
            return
        except Exception:
            continue


#: How long an alternate that answered after a server error stays first.
STICKY_MODEL_S = 6 * 3600
STICKY_MODELS_PATH = CONFIG_DIR / "diagnostics" / "sticky_models.json"


def _read_sticky_models() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(STICKY_MODELS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    now = time.time()
    return {name: entry for name, entry in (data.items() if isinstance(data, dict) else ())
            if isinstance(entry, dict) and float(entry.get("until") or 0) > now}


def _seal_plain_keys() -> None:
    """A key pasted into api_keys.json in plain text is sealed on the next
    read (key_seal.py). A file that cannot be parsed is left for the read to
    report; a sealing failure leaves the file exactly as it was."""
    try:
        key_seal.seal_file_in_place(API_CONFIG_PATH, indent=4)
    except Exception:
        pass


def read_provider_settings() -> OrionProviderSettings:
    with _PROVIDER_CONFIG_LOCK:
        try:
            _rescue_api_config()
            _seal_plain_keys()
            existing, observed = _read_provider_document(API_CONFIG_PATH)
            legacy_key = str((existing or {}).get("gemini_api_key") or "").strip()
            payload = _default_provider_payload(legacy_key) if legacy_key else _default_provider_payload()
            if existing is not None:
                payload = _merge_provider_payload(payload, existing)
            settings = _settings_from_payload(payload)
        except (ProviderConfigurationError, ValueError, TypeError, OverflowError, AttributeError):
            # An uncertain saved policy is not permission to enable defaults,
            # open a replacement-key dialog or write over retained credentials.
            return OrionProviderSettings("", [], {}, configuration_error=_PROVIDER_CONFIG_REPAIR)
        settings._source_observed = True
        settings._source_path = str(API_CONFIG_PATH.absolute())
        settings._source_sha256 = observed
        return settings


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
    # Startup persists disabled defaults for every hosted provider. A key
    # added to the launch environment later must activate a profile that has
    # no saved credential. A saved credential (including an explicitly
    # disabled profile) remains authoritative.
    for name in (
        "gemini", "gemini_text", "gemini_reasoning", "gemma_4_31b", "gemma_4_26b",
        "cloudflare", "openai", "xai_grok", "cerebras",
        "sambanova", "openrouter", "groq", "mistral", "huggingface",
        "nvidia", "together", "deepseek", "thoughts",
    ):
        default_profile = (defaults.get("providers") or {}).get(name)
        saved_profile = (existing.get("providers") or {}).get(name)
        if (isinstance(default_profile, dict) and default_profile.get("enabled")
                and isinstance(saved_profile, dict)
                and not str(saved_profile.get("api_key") or "").strip()):
            providers[name]["api_key"] = default_profile["api_key"]
            providers[name]["enabled"] = True
            if not str(saved_profile.get("base_url") or "").strip():
                providers[name]["base_url"] = default_profile["base_url"]
    # Migrate only shipped defaults.  A user's chosen primary, alternate list
    # or strengths must remain their choice.  Flash-Lite leads routine work;
    # the separately routed reasoning profile leads demanding turns.
    text_saved = (existing.get("providers") or {}).get("gemini_text")
    shipped_text = (isinstance(text_saved, dict)
                    and tuple(text_saved.get("models") or ()) in {
                        _FORMER_GEMINI_TEXT_MODELS, _PREVIOUS_GEMINI_TEXT_MODELS,
                        _GEMINI_TEXT_MODELS,
                    }
                    and str(text_saved.get("model") or "") in {
                        "gemini-2.5-flash", _GEMINI_TEXT_MODELS[0],
                    })
    if shipped_text:
        providers["gemini_text"]["models"] = list(_GEMINI_TEXT_MODELS)
        if str(text_saved.get("model") or "") == "gemini-2.5-flash":
            providers["gemini_text"]["model"] = _GEMINI_TEXT_MODELS[0]
        if tuple(text_saved.get("strengths") or ()) == (
                "reasoning", "writing", "coding", "general"):
            providers["gemini_text"]["strengths"] = ["fast", "general"]
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
    # A new reasoning profile shares the already-configured Gemini key.  An
    # explicitly saved profile, including a disabled one, is never rewritten.
    saved_reasoning = (existing.get("providers") or {}).get("gemini_reasoning")
    reasoning = providers.get("gemini_reasoning")
    if (isinstance(reasoning, dict) and not isinstance(saved_reasoning, dict)
            and isinstance(text, dict)):
        reasoning["api_key"] = str(text.get("api_key") or live_key)
        reasoning["enabled"] = (bool(text.get("enabled"))
                                and bool(reasoning["api_key"])
                                and (not isinstance(saved_text, dict) or shipped_text))
    # Hosted Gemma uses the same Gemini key through a different native API.
    # A custom text profile remains authoritative until the owner explicitly
    # enables the new Gemma profiles; saved Gemma choices also remain intact.
    for gemma_name in ("gemma_4_31b", "gemma_4_26b"):
        saved_gemma = (existing.get("providers") or {}).get(gemma_name)
        gemma = providers.get(gemma_name)
        if not isinstance(gemma, dict):
            continue
        if not str(gemma.get("api_key") or "").strip():
            gemma["api_key"] = str((text or {}).get("api_key") or live_key)
        if not isinstance(saved_gemma, dict):
            gemma["enabled"] = (bool((text or {}).get("enabled"))
                                and bool(gemma["api_key"])
                                and (not isinstance(saved_text, dict) or shipped_text))
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
    if settings.configuration_error:
        raise ProviderConfigurationError(_PROVIDER_CONFIG_REPAIR)
    with _PROVIDER_CONFIG_LOCK:
        existing, observed = _read_provider_document(API_CONFIG_PATH)
        if settings._source_observed and (settings._source_path != str(API_CONFIG_PATH.absolute())
                or settings._source_sha256 != observed):
            raise ProviderConfigurationError("Provider settings changed on disk. Reload them before saving.")
        # Retain extensions we do not interpret, including provider-specific
        # options used by other integrations. Only known fields are replaced.
        payload = dict(existing or {})
        profiles = dict(payload.get("providers") or {})
        for name, profile in settings.providers.items():
            raw = dict(profiles.get(name) or {})
            raw.update(_profile_to_config(profile))
            profiles[name] = raw
        payload.update(schema=_PROVIDER_CONFIG_SCHEMA, active_provider=settings.active_provider,
                       provider_order=settings.provider_order, providers=profiles,
                       integrations=settings.integrations)
        settings._source_sha256 = _publish_provider_document(payload, observed)
        settings._source_path = str(API_CONFIG_PATH.absolute())
        settings._source_observed = True


def read_api_key() -> str:
    """Legacy convenience wrapper retained for older call sites."""
    settings = read_provider_settings()
    gemini = settings.providers.get("gemini")
    return gemini.api_key if gemini is not None else ""


def write_api_key(api_key: str) -> None:
    """Set the live key without resetting other saved providers/integrations."""
    settings = read_provider_settings()
    if settings.configuration_error:
        raise ProviderConfigurationError(_PROVIDER_CONFIG_REPAIR)
    if settings._source_sha256 is None:
        first_run = _settings_from_payload(_default_provider_payload(api_key))
        first_run._source_observed = settings._source_observed
        first_run._source_path = settings._source_path
        first_run._source_sha256 = settings._source_sha256
        write_provider_settings(first_run)
        return
    live = settings.providers.get("gemini")
    previous = live.api_key if live is not None else ""
    if live is None:
        live = _settings_from_payload(_default_provider_payload(api_key)).providers.get("gemini")
        if live is None:
            raise ProviderConfigurationError("The Gemini provider profile is unavailable; repair api_keys.json before saving a key.")
        settings.providers["gemini"] = live
    live.api_key, live.enabled = str(api_key).strip(), True
    for name in ("gemini_text", "gemini_reasoning", "gemma_4_31b", "gemma_4_26b"):
        profile = settings.providers.get(name)
        if profile is not None and (not profile.api_key or previous and profile.api_key == previous):
            profile.api_key = live.api_key
    write_provider_settings(settings)
