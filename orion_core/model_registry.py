"""
Versioned model-metadata registry (Section 6).

Authoritative, documented context-window sizes for models ORION may route to.
This is deliberately conservative: only widely-published limits are listed, and
anything not listed returns ``None`` so the dashboard shows "Not reported by
provider" rather than a fabricated number.  NEVER add a guessed value here.

Context window is a PER-REQUEST input limit — it is NOT account quota, a rate
limit, or purchased billing credit.  Those are tracked separately in the usage
ledger and must never be conflated with this value.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bump when entries change so an estimate can record which registry it used.
REGISTRY_VERSION = "2026-07-15"


@dataclass(frozen=True)
class ModelMetadata:
    model: str
    context_window: int | None      # tokens; None = not documented here
    max_output_tokens: int | None = None
    # Indicative price per 1,000,000 tokens (USD); None = not recorded.
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None


# Keys are matched by longest-prefix so versioned ids (e.g. "gpt-4o-mini-2024")
# resolve to their family entry.  Values reflect each provider's published
# per-request context limits at REGISTRY_VERSION.
_REGISTRY: dict[str, ModelMetadata] = {
    # OpenAI
    "gpt-4o-mini": ModelMetadata("gpt-4o-mini", 128_000, 16_384, 0.15, 0.60),
    "gpt-4o": ModelMetadata("gpt-4o", 128_000, 16_384, 2.50, 10.00),
    "gpt-4.1": ModelMetadata("gpt-4.1", 1_000_000, 32_768, 2.00, 8.00),
    "o3": ModelMetadata("o3", 200_000, 100_000, None, None),
    # Anthropic (Claude)
    "claude-sonnet": ModelMetadata("claude-sonnet", 200_000, 64_000, 3.00, 15.00),
    "claude-opus": ModelMetadata("claude-opus", 200_000, 32_000, 15.00, 75.00),
    "claude-haiku": ModelMetadata("claude-haiku", 200_000, 32_000, 0.80, 4.00),
    "claude-3": ModelMetadata("claude-3", 200_000, 4_096, None, None),
    # Google Gemini
    "gemini-2.5": ModelMetadata("gemini-2.5", 1_048_576, 65_536, None, None),
    "gemini-2.0": ModelMetadata("gemini-2.0", 1_048_576, 8_192, None, None),
    "gemini-1.5-pro": ModelMetadata("gemini-1.5-pro", 2_097_152, 8_192, None, None),
    "gemini-1.5": ModelMetadata("gemini-1.5", 1_048_576, 8_192, None, None),
    # xAI
    "grok-4": ModelMetadata("grok-4", 256_000, None, None, None),
    "grok": ModelMetadata("grok", 131_072, None, None, None),
    # Meta Llama (typical served context)
    "llama-3.1": ModelMetadata("llama-3.1", 131_072, None, None, None),
    "llama3.1": ModelMetadata("llama3.1", 131_072, None, None, None),
    "meta-llama-3.1": ModelMetadata("meta-llama-3.1", 131_072, None, None, None),
}


def _lookup(model: str) -> ModelMetadata | None:
    m = str(model or "").strip().lower()
    if not m:
        return None
    # Strip a leading "models/" (Gemini) and any provider path prefix.
    if "/" in m:
        m = m.split("/")[-1]
    # Longest-prefix match so a versioned id resolves to its family.
    best: ModelMetadata | None = None
    best_len = -1
    for key, meta in _REGISTRY.items():
        if m.startswith(key) and len(key) > best_len:
            best, best_len = meta, len(key)
    if best is not None:
        return best
    # Also match when the family key is contained (e.g. "…-claude-sonnet-…").
    for key, meta in _REGISTRY.items():
        if key in m and len(key) > best_len:
            best, best_len = meta, len(key)
    return best


def context_window(model: str) -> int | None:
    """Authoritative per-request context window, or ``None`` when not documented."""
    meta = _lookup(model)
    return meta.context_window if meta is not None else None


def model_metadata(model: str) -> ModelMetadata | None:
    return _lookup(model)


def context_utilisation(model: str, tokens_in_context: int | None) -> float | None:
    """Percent of the context window occupied, or ``None`` when either the limit
    is undocumented or the occupancy is unknown.  Never guesses the limit."""
    window = context_window(model)
    if window is None or window <= 0 or tokens_in_context is None:
        return None
    return round(100.0 * float(tokens_in_context) / float(window), 2)


def remaining_context(model: str, tokens_in_context: int | None) -> int | None:
    window = context_window(model)
    if window is None or tokens_in_context is None:
        return None
    return max(0, window - int(tokens_in_context))


def estimated_cost_usd(model: str, input_tokens: int | None,
                       output_tokens: int | None) -> tuple[float | None, str]:
    """Indicative cost from the registry's published prices.  Returns
    (cost_or_None, source) where source is 'estimated' or 'unavailable' — the UI
    must label estimated costs and never present them as billed amounts."""
    meta = _lookup(model)
    if meta is None or meta.input_price_per_mtok is None or meta.output_price_per_mtok is None:
        return None, "unavailable"
    inp = (input_tokens or 0) / 1_000_000.0 * meta.input_price_per_mtok
    out = (output_tokens or 0) / 1_000_000.0 * meta.output_price_per_mtok
    return round(inp + out, 6), "estimated"
