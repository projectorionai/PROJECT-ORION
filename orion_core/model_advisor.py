"""
Which model to use — the free ones, and the cheap ones.

  "I want mostly all the free API models but also cheap ones such as DeepSeek,
   the ones which cost less $ per million tokens ... give me the most cheapest
   and best models ... that also do not lag out my computer."

model_registry.py records prices for cost ESTIMATION of a model already in use.
This is the other direction: a curated, opinionated catalogue of the models
worth reaching for, so ORION can answer "what's the cheapest good model for
this" and "what can I run for free".

Two axes the request cares about:

  * **Cost** — free tiers first (Groq, Gemini Flash, Cerebras, OpenRouter's free
    models), then the genuinely cheap paid ones (DeepSeek above all), ranked by
    a blended $/million-token figure.
  * **Local load** — a cloud API uses none of your CPU or GPU; a LOCAL model
    (Ollama) uses a lot. So for "don't lag my computer" the honest answer is
    almost always a cloud model, and where a local one is wanted the catalogue
    says how heavy it is and whether it needs a GPU.

Prices move, and providers change tiers without notice. Every figure here is
INDICATIVE, dated, and labelled as such — the catalogue is a shortlist to
verify, never a bill. A learned limit from a real 429/402 (see usage_runway)
always beats a number written here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#: When these indicative prices were last set. Shown to the user so a stale
#: figure is obviously stale rather than silently trusted.
PRICES_AS_OF = "January 2026"


class Host(str, Enum):
    CLOUD = "cloud"        # an API — uses none of your machine
    LOCAL = "local"        # runs on your hardware (Ollama) — uses CPU/GPU


class Load(str, Enum):
    """How hard a LOCAL model leans on the machine. Cloud models are NONE."""

    NONE = "none"          # cloud
    LIGHT = "light"        # a small local model, CPU-friendly
    HEAVY = "heavy"        # a large local model, wants a GPU


@dataclass
class Model:
    name: str
    provider: str
    access: str                       # how to reach it
    host: Host
    input_per_mtok: float             # USD per million input tokens (0 = free tier)
    output_per_mtok: float
    free_tier: bool
    context: int                      # tokens
    strengths: tuple[str, ...] = ()
    local_load: Load = Load.NONE
    note: str = ""

    @property
    def blended_per_mtok(self) -> float:
        """A single comparable figure: input and output weighted 3:1, the usual
        real-world ratio for chat/agent work."""
        return (self.input_per_mtok * 3 + self.output_per_mtok) / 4

    def cost_line(self) -> str:
        if self.free_tier and self.input_per_mtok == 0:
            price = "FREE tier"
        else:
            price = f"${self.input_per_mtok:g} in / ${self.output_per_mtok:g} out per M"
        load = "" if self.host is Host.CLOUD else f", {self.local_load.value} local load"
        return f"{price}{load}"

    def describe(self) -> str:
        return (f"{self.name} ({self.provider}) — {self.cost_line()}; "
                f"{self.context // 1000}k context; "
                f"{', '.join(self.strengths) or 'general'}."
                + (f" {self.note}" if self.note else ""))


#: The shortlist. Ordered roughly best-value-first within each group. Prices are
#: indicative (PRICES_AS_OF) and should be checked before relying on them.
CATALOGUE: list[Model] = [
    # ── free API tiers — no cost, no local load ──────────────────────────────
    Model("llama-3.3-70b", "Groq", "Groq API (free tier)", Host.CLOUD, 0, 0,
          True, 128_000, ("fast", "coding", "general"),
          note="Groq's free tier is fast but rate-limited per minute/day."),
    Model("gemini-2.0-flash", "Google", "Gemini API (free tier)", Host.CLOUD,
          0, 0, True, 1_048_576, ("huge context", "vision", "fast"),
          note="Generous free tier; also ORION's live-voice model."),
    Model("qwen-2.5-72b", "Groq / OpenRouter", "free tiers", Host.CLOUD, 0, 0,
          True, 32_000, ("coding", "multilingual")),
    Model("llama-3.1-8b-instant", "Groq", "Groq API (free tier)", Host.CLOUD,
          0, 0, True, 128_000, ("very fast", "cheap bulk"),
          note="Ideal for high-volume, low-stakes calls."),
    Model("gpt-oss / free OpenRouter models", "OpenRouter", "models tagged :free",
          Host.CLOUD, 0, 0, True, 32_000, ("varied", "no cost"),
          note="OpenRouter rotates free models; availability varies."),
    Model("cerebras-llama", "Cerebras", "Cerebras API (free tier)", Host.CLOUD,
          0, 0, True, 8_192, ("extremely fast inference",)),
    # ── cheap paid — the DeepSeek tier the user named ────────────────────────
    Model("deepseek-v3", "DeepSeek", "DeepSeek API", Host.CLOUD, 0.14, 0.28,
          False, 64_000, ("cheapest strong general", "coding", "reasoning"),
          note="The value leader — near-frontier quality at a fraction of the price. "
               "Off-peak pricing is cheaper still."),
    Model("deepseek-r1", "DeepSeek", "DeepSeek API", Host.CLOUD, 0.55, 2.19,
          False, 64_000, ("deep reasoning", "maths", "cheap for its class"),
          note="A reasoning model at a fraction of o1's price."),
    Model("gemini-2.0-flash (paid)", "Google", "Gemini API", Host.CLOUD, 0.10,
          0.40, False, 1_048_576, ("huge context", "vision", "cheap"),
          note="Beyond the free tier, still very cheap for the context it gives."),
    Model("gpt-4o-mini", "OpenAI", "OpenAI API / OpenRouter", Host.CLOUD, 0.15,
          0.60, False, 128_000, ("reliable", "tool use", "vision")),
    Model("claude-haiku", "Anthropic", "Anthropic API / OpenRouter", Host.CLOUD,
          0.80, 4.0, False, 200_000, ("careful", "long context", "writing"),
          note="Pricier than DeepSeek but strong at instruction-following."),
    Model("mistral-small", "Mistral / OpenRouter", "API", Host.CLOUD, 0.20, 0.60,
          False, 32_000, ("multilingual", "European data option")),
    # ── local — zero cost, but uses YOUR hardware ────────────────────────────
    Model("llama-3.1-8b (local)", "Ollama", "ollama pull llama3.1:8b", Host.LOCAL,
          0, 0, True, 128_000, ("offline", "private"), Load.LIGHT,
          note="Runs on CPU; comfortable on 16GB RAM. No network, no cost, some lag."),
    Model("qwen-2.5-7b (local)", "Ollama", "ollama pull qwen2.5:7b", Host.LOCAL,
          0, 0, True, 32_000, ("offline", "coding", "multilingual"), Load.LIGHT),
    Model("deepseek-r1-14b (local)", "Ollama", "ollama pull deepseek-r1:14b",
          Host.LOCAL, 0, 0, True, 64_000, ("offline reasoning",), Load.HEAVY,
          note="Wants a GPU (8GB+ VRAM); heavy on CPU-only, will lag."),
]


#: Task → the models to reach for, best first. Names are prefixes matched
#: against the catalogue, so "deepseek" picks whichever DeepSeek fits.
RECOMMENDATIONS: dict[str, tuple[str, ...]] = {
    "free": ("gemini-2.0-flash", "llama-3.3-70b", "qwen-2.5-72b",
             "llama-3.1-8b-instant"),
    "cheapest": ("deepseek-v3", "gemini-2.0-flash (paid)", "gpt-4o-mini",
                 "mistral-small"),
    "coding": ("deepseek-v3", "llama-3.3-70b", "qwen-2.5-72b"),
    "reasoning": ("deepseek-r1", "deepseek-v3", "gemini-2.0-flash"),
    "vision": ("gemini-2.0-flash", "gpt-4o-mini"),
    "long_context": ("gemini-2.0-flash", "claude-haiku"),
    "bulk": ("llama-3.1-8b-instant", "gemini-2.0-flash", "deepseek-v3"),
    "offline": ("llama-3.1-8b (local)", "qwen-2.5-7b (local)"),
    "fastest": ("cerebras-llama", "llama-3.1-8b-instant", "llama-3.3-70b"),
}

_TASK_ALIASES = {
    "cheap": "cheapest", "value": "cheapest", "budget": "cheapest",
    "code": "coding", "programming": "coding", "python": "coding",
    "think": "reasoning", "maths": "reasoning", "math": "reasoning",
    "image": "vision", "images": "vision", "video": "vision",
    "context": "long_context", "long": "long_context",
    "local": "offline", "private": "offline", "no internet": "offline",
    "fast": "fastest", "speed": "fastest", "quick": "fastest",
    "high volume": "bulk", "cheap bulk": "bulk",
}


def _find(prefix: str) -> Model | None:
    prefix = prefix.lower()
    for model in CATALOGUE:
        if model.name.lower().startswith(prefix) or prefix in model.name.lower():
            return model
    return None


def cheapest(limit: int = 6, include_local: bool = False) -> list[Model]:
    """Models ranked by blended cost, free tiers first."""
    pool = [m for m in CATALOGUE if include_local or m.host is Host.CLOUD]
    return sorted(pool, key=lambda m: (not m.free_tier, m.blended_per_mtok))[:limit]


def recommend(task: str = "cheapest") -> list[Model]:
    """The models to reach for, for a task."""
    key = str(task or "cheapest").lower().strip()
    key = _TASK_ALIASES.get(key, key)
    if key not in RECOMMENDATIONS:
        # Loose match, else fall back to plain value ranking.
        for alias, canon in _TASK_ALIASES.items():
            if alias in key:
                key = canon
                break
        else:
            key = "cheapest" if key not in RECOMMENDATIONS else key
    names = RECOMMENDATIONS.get(key, RECOMMENDATIONS["cheapest"])
    out: list[Model] = []
    for name in names:
        model = _find(name)
        if model is not None and model not in out:
            out.append(model)
    return out


def advise(task: str = "") -> str:
    """A spoken/printed recommendation for the user."""
    if not task or task.lower().strip() in {"cheapest", "cheap", "value", ""}:
        picks = cheapest(6)
        head = ("Cheapest strong models right now "
                f"(indicative prices, {PRICES_AS_OF} — verify before relying):")
    else:
        picks = recommend(task)
        head = f"For {task}, I'd reach for (best first):"
    if not picks:
        return "I don't have a recommendation for that."
    lines = [head]
    for i, model in enumerate(picks, 1):
        lines.append(f"  {i}. {model.describe()}")
    # The load answer the request also asked for.
    if any(m.host is Host.LOCAL for m in picks):
        lines.append("\nNote: local models use your CPU/GPU and can lag the "
                     "machine; the cloud options above use neither.")
    else:
        lines.append("\nAll cloud — none of these use your CPU or GPU, so they "
                     "won't lag the machine. DeepSeek is the value pick if you "
                     "go paid.")
    return "\n".join(lines)


def catalogue() -> list[dict]:
    return [{"name": m.name, "provider": m.provider, "free": m.free_tier,
             "blended_per_mtok": round(m.blended_per_mtok, 3),
             "host": m.host.value, "context": m.context} for m in CATALOGUE]


__all__ = [
    "CATALOGUE", "PRICES_AS_OF", "RECOMMENDATIONS", "Host", "Load", "Model",
    "advise", "catalogue", "cheapest", "recommend",
]
