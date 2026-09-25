"""
Capability-based provider interfaces and response normalisation (Section 5).

ORION must not be coupled to one provider.  This module expresses what a
provider can do as a set of capabilities, validates a capability before a task
is assigned to a provider, and normalises the parts of a response that core
logic actually consumes — token usage, stop reason and tool calls — so
provider-specific SDK types never leak upward.

Pure and deterministic; provider-specific SDK code stays in adapters that call
these normalisers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Capability(str, Enum):
    TEXT_GENERATION = "text_generation"
    STREAMING_TEXT = "streaming_text"
    TOOL_CALLING = "tool_calling"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    SPEECH_RECOGNITION = "speech_recognition"
    SPEECH_SYNTHESIS = "speech_synthesis"
    REALTIME_AUDIO = "realtime_audio"
    IMAGE_GENERATION = "image_generation"


class StopReason(str, Enum):
    STOP = "stop"                    # natural completion
    LENGTH = "length"               # hit max tokens / context
    TOOL_CALLS = "tool_calls"       # model wants a tool
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


# Model-name hints that indicate specialised capabilities on an otherwise
# OpenAI-compatible endpoint.
_EMBED_RE = re.compile(r"(?i)embed|text-embedding")
_ASR_RE = re.compile(r"(?i)whisper|transcrib|speech-to-text|stt")
_TTS_RE = re.compile(r"(?i)\btts\b|text-to-speech|speech-01|voice")
_IMAGE_RE = re.compile(r"(?i)dall-?e|imagen|stable-?diffusion|flux|image")
_VISION_RE = re.compile(r"(?i)vision|gpt-4o|gpt-4\.1|claude-3|claude-sonnet|claude-opus|gemini|llava|pixtral|grok")


def provider_capabilities(profile: Any) -> set[Capability]:
    """Derive the capability set of a provider profile from its kind and model.

    Conservative: a capability is only asserted when the kind/model supports it,
    so ``validate_capability`` never routes a task to a provider that cannot
    perform it."""
    caps: set[Capability] = set()
    kind = str(getattr(profile, "kind", "") or "").lower()
    model = str(getattr(profile, "model", "") or "")
    enabled = bool(getattr(profile, "enabled", True))
    has_key = bool(str(getattr(profile, "api_key", "") or "").strip())
    is_local = bool(getattr(profile, "is_local", False))

    if not enabled:
        return caps

    if kind == "gemini_live":
        if has_key:
            caps |= {
                Capability.REALTIME_AUDIO, Capability.TEXT_GENERATION,
                Capability.STREAMING_TEXT, Capability.TOOL_CALLING,
                Capability.VISION, Capability.SPEECH_RECOGNITION,
                Capability.SPEECH_SYNTHESIS,
            }
        return caps

    if kind == "openai_compatible":
        usable = is_local or has_key
        if not usable:
            return caps
        if _EMBED_RE.search(model):
            caps.add(Capability.EMBEDDINGS)
            return caps
        if _ASR_RE.search(model):
            caps.add(Capability.SPEECH_RECOGNITION)
            return caps
        if _TTS_RE.search(model):
            caps.add(Capability.SPEECH_SYNTHESIS)
            return caps
        if _IMAGE_RE.search(model) and not _VISION_RE.search(model):
            caps.add(Capability.IMAGE_GENERATION)
            return caps
        # Chat-completions family.
        caps |= {Capability.TEXT_GENERATION, Capability.STREAMING_TEXT,
                 Capability.TOOL_CALLING}
        if _VISION_RE.search(model):
            caps.add(Capability.VISION)
        return caps

    return caps


def supports(profile: Any, capability: Capability) -> bool:
    return capability in provider_capabilities(profile)


def validate_capability(profile: Any, capability: Capability) -> tuple[bool, str]:
    """Check a provider before assigning a task.  Returns (ok, reason)."""
    caps = provider_capabilities(profile)
    if not caps:
        return False, f"provider '{getattr(profile, 'name', '?')}' is disabled or unconfigured"
    if capability in caps:
        return True, "capability available"
    return False, (f"provider '{getattr(profile, 'name', '?')}' ({getattr(profile, 'model', '?')}) "
                   f"does not support {capability.value}")


def select_for_capability(profiles: list[Any], capability: Capability) -> list[Any]:
    """Filter/preserve-order profiles that provide a capability."""
    return [p for p in profiles if supports(p, capability)]


# ── normalisation ────────────────────────────────────────────────────────────

@dataclass
class NormalizedUsage:
    """Authoritative token usage in one provider-neutral shape.  Fields left at
    ``None`` were not reported by the provider (never fabricated)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class NormalizedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


def _get(obj: Any, *names: str) -> Any:
    """Read an attribute or dict key by any of several names."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_usage(raw: Any, provider_kind: str = "openai_compatible") -> NormalizedUsage:
    """Normalise a provider's usage block.

    Handles OpenAI-style (``usage.prompt_tokens`` / ``completion_tokens`` /
    ``prompt_tokens_details.cached_tokens`` / ``completion_tokens_details.
    reasoning_tokens``) and Gemini-style (``usage_metadata.prompt_token_count``
    / ``candidates_token_count`` / ``cached_content_token_count`` /
    ``thoughts_token_count``)."""
    if raw is None:
        return NormalizedUsage()
    kind = (provider_kind or "").lower()

    if kind.startswith("gemini") or _get(raw, "prompt_token_count") is not None:
        cached = _get(raw, "cached_content_token_count")
        reasoning = _get(raw, "thoughts_token_count")
        return NormalizedUsage(
            input_tokens=_int_or_none(_get(raw, "prompt_token_count")),
            output_tokens=_int_or_none(_get(raw, "candidates_token_count")),
            cached_input_tokens=_int_or_none(cached),
            reasoning_tokens=_int_or_none(reasoning),
            total_tokens=_int_or_none(_get(raw, "total_token_count")),
        )

    # OpenAI-compatible.
    prompt_details = _get(raw, "prompt_tokens_details") or {}
    completion_details = _get(raw, "completion_tokens_details") or {}
    cached = _get(prompt_details, "cached_tokens")
    reasoning = _get(completion_details, "reasoning_tokens")
    return NormalizedUsage(
        input_tokens=_int_or_none(_get(raw, "prompt_tokens", "input_tokens")),
        output_tokens=_int_or_none(_get(raw, "completion_tokens", "output_tokens")),
        cached_input_tokens=_int_or_none(cached),
        reasoning_tokens=_int_or_none(reasoning),
        total_tokens=_int_or_none(_get(raw, "total_tokens")),
    )


_STOP_MAP = {
    # OpenAI finish_reason
    "stop": StopReason.STOP,
    "length": StopReason.LENGTH,
    "max_tokens": StopReason.LENGTH,
    "tool_calls": StopReason.TOOL_CALLS,
    "function_call": StopReason.TOOL_CALLS,
    "content_filter": StopReason.CONTENT_FILTER,
    # Gemini FinishReason
    "max_tokens".upper(): StopReason.LENGTH,
    "stop".upper(): StopReason.STOP,
    "safety": StopReason.CONTENT_FILTER,
    "recitation": StopReason.CONTENT_FILTER,
    "end_turn": StopReason.STOP,
    "tool_use": StopReason.TOOL_CALLS,
}


def normalize_stop_reason(raw: Any) -> StopReason:
    if raw is None:
        return StopReason.UNKNOWN
    key = str(getattr(raw, "name", raw)).strip().lower()
    return _STOP_MAP.get(key, _STOP_MAP.get(key.upper(), StopReason.UNKNOWN))


def normalize_tool_calls(raw: Any) -> list[NormalizedToolCall]:
    """Normalise tool/function calls from OpenAI (``tool_calls``/``function``) or
    Gemini (``function_calls``/``args``) shapes."""
    import json
    out: list[NormalizedToolCall] = []
    if not raw:
        return out
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    for item in items:
        fn = _get(item, "function") or item
        name = _get(fn, "name") or _get(item, "name") or ""
        args = _get(fn, "arguments") or _get(item, "args") or _get(item, "arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_value": args}
        out.append(NormalizedToolCall(
            name=str(name), arguments=args, call_id=_get(item, "id", "call_id")))
    return out
