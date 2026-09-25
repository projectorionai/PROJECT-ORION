"""
The cost governor (CAP-02) — spend the free tier first, and don't cook the CPU.

Two separate questions the user kept asking, answered by one object:

    "I want the cheapest and best models ... the estimated time of usage
     remaining ... and if CPU usage is high, can we divert to GPU?"

**How much runway is left, and should we switch?** ``usage_runway`` already
knows the budget, the burn rate and how long that leaves; ``model_advisor``
already knows which models are free and which are cheap. Neither of them decides
anything. This does: it reads the pressure on the active provider and, when the
allowance is running tight, names the free or cheap model to fall back to so a
long afternoon doesn't hit a wall mid-sentence.

**Is the machine the bottleneck?** A local model runs on your hardware, and the
user's specific worry was the CPU. When CPU load is high and the work is (or
would be) local, the governor says so and points at the GPU — because on this
machine that is the difference between a fluid reply and a stuttering one. When
there's no GPU to divert to, it says *that* instead, and suggests a free cloud
tier so the CPU stops being the limit.

Everything here is deterministic and injectable — the runway, the CPU reading
and the GPU check are all passed in — so the decision logic is tested without a
real ledger, a real CPU spike or a real graphics card.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from . import model_advisor


class Pressure(str, Enum):
    """How close the daily allowance is to running out."""
    EASY = "easy"            # plenty left
    WATCH = "watch"          # over half gone
    TIGHT = "tight"          # nearly gone, or little time left at this rate
    EXHAUSTED = "exhausted"  # the allowance is spent

    @property
    def label(self) -> str:
        return {
            Pressure.EASY: "plenty of headroom",
            Pressure.WATCH: "over half used",
            Pressure.TIGHT: "running low",
            Pressure.EXHAUSTED: "allowance spent",
        }[self]


# Thresholds on the fraction of the daily allowance consumed.
_WATCH_AT = 0.50
_TIGHT_AT = 0.80
_EXHAUSTED_AT = 0.97
# If, at the current burn rate, this little wall-clock time remains, treat the
# provider as TIGHT regardless of the fraction — a burst can empty a big budget
# fast, and the point of the governor is to move BEFORE the wall.
_TIGHT_MINUTES = 30.0

# Above this CPU load, a local model should be leaning on the GPU, not the CPU.
_CPU_HOT = 85.0


def _default_gpu_probe() -> bool:
    """Cheap, honest GPU check: is an NVIDIA driver present? ``nvidia-smi`` on
    PATH is the least-intrusive positive signal on Windows. Injectable, so a
    test never depends on the host actually having a card."""
    return shutil.which("nvidia-smi") is not None


@dataclass
class GovernorDecision:
    provider: str
    pressure: Pressure
    used: int
    budget: int
    fraction_used: float
    minutes_left: float | None
    should_switch: bool
    suggested_model: str          # "" when no switch is advised
    suggested_reason: str
    cpu_percent: float
    gpu_available: bool
    gpu_advice: str
    message: str                  # the spoken/printed summary

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "pressure": self.pressure.value,
            "used": self.used,
            "budget": self.budget,
            "fraction_used": round(self.fraction_used, 3),
            "minutes_left": self.minutes_left,
            "should_switch": self.should_switch,
            "suggested_model": self.suggested_model,
            "cpu_percent": self.cpu_percent,
            "gpu_available": self.gpu_available,
        }


class CostGovernor:
    """Decide, from live usage and machine load, what to run next."""

    def __init__(
        self,
        runway: Any,
        cpu_percent: Callable[[], float] | None = None,
        gpu_available: Callable[[], bool] | None = None,
    ) -> None:
        self.runway = runway
        self._cpu_percent = cpu_percent or _read_cpu_percent
        self._gpu_available = gpu_available or _default_gpu_probe

    # ── the decision ──────────────────────────────────────────────────────────

    def assess(self, provider: str, *, local: bool = False) -> GovernorDecision:
        """Read the runway for *provider* and decide whether to switch models
        and whether the CPU should hand off to the GPU.

        ``local`` says the current/candidate work runs on the machine — which is
        what makes the CPU→GPU advice relevant.
        """
        rw = self.runway.runway(provider)
        fraction = rw.fraction_used
        minutes = rw.minutes_left

        pressure = self._pressure(fraction, minutes, rw.remaining)
        should_switch = pressure in (Pressure.TIGHT, Pressure.EXHAUSTED)

        suggested_model, suggested_reason = "", ""
        if should_switch:
            suggested_model, suggested_reason = self._fallback(pressure)

        cpu = float(self._cpu_percent() or 0.0)
        gpu = bool(self._gpu_available())
        gpu_advice = self._gpu_advice(cpu, gpu, local)

        message = self._compose(provider, rw, pressure, should_switch,
                                 suggested_model, suggested_reason, gpu_advice)

        return GovernorDecision(
            provider=provider, pressure=pressure, used=rw.used, budget=rw.budget,
            fraction_used=fraction, minutes_left=minutes,
            should_switch=should_switch, suggested_model=suggested_model,
            suggested_reason=suggested_reason, cpu_percent=cpu,
            gpu_available=gpu, gpu_advice=gpu_advice, message=message)

    # ── pieces ────────────────────────────────────────────────────────────────

    @staticmethod
    def _pressure(fraction: float, minutes: float | None, remaining: int) -> Pressure:
        if remaining <= 0 or fraction >= _EXHAUSTED_AT:
            return Pressure.EXHAUSTED
        if fraction >= _TIGHT_AT or (minutes is not None and minutes <= _TIGHT_MINUTES):
            return Pressure.TIGHT
        if fraction >= _WATCH_AT:
            return Pressure.WATCH
        return Pressure.EASY

    @staticmethod
    def _fallback(pressure: Pressure) -> tuple[str, str]:
        """Which model to move to under pressure.

        EXHAUSTED → a FREE model, because the point is to stop spending. TIGHT →
        the cheapest capable paid model (DeepSeek), which stretches the runway
        without dropping to a free tier's rate limits mid-task.
        """
        if pressure is Pressure.EXHAUSTED:
            picks = model_advisor.recommend("free")
            if picks:
                return picks[0].name, "it's free — no spend against a spent budget"
        cheap = [m for m in model_advisor.cheapest(12) if not m.free_tier]
        if cheap:
            m = cheap[0]
            return m.name, f"cheapest capable option (~${m.blended_per_mtok:.2f}/M blended)"
        picks = model_advisor.recommend("free")
        return (picks[0].name, "free fallback") if picks else ("", "")

    @staticmethod
    def _gpu_advice(cpu: float, gpu: bool, local: bool) -> str:
        if not local:
            # Cloud work uses neither the CPU nor the GPU meaningfully.
            if cpu >= _CPU_HOT:
                return (f"CPU is at {cpu:.0f}%, but the model runs in the cloud, "
                        "so that load is something else on the machine — not me.")
            return ""
        if cpu < _CPU_HOT:
            return f"CPU at {cpu:.0f}% — comfortable for a local model."
        if gpu:
            return (f"CPU is hot ({cpu:.0f}%). A local model should offload to the "
                    "GPU — raise the GPU layers (Ollama num_gpu / a CUDA build) so "
                    "the graphics card does the work and the CPU frees up.")
        return (f"CPU is hot ({cpu:.0f}%) and I don't see an NVIDIA GPU to divert "
                "to. Better to use a free cloud model for this so the CPU stops "
                "being the bottleneck.")

    @staticmethod
    def _compose(provider, rw, pressure, should_switch, model, reason, gpu_advice) -> str:
        parts = [rw.describe()]
        parts.append(f"Pressure: {pressure.label}.")
        if should_switch and model:
            verb = "Switch" if pressure is Pressure.EXHAUSTED else "Consider switching"
            parts.append(f"{verb} to {model} — {reason}.")
        elif not should_switch:
            parts.append("No need to switch models yet.")
        if gpu_advice:
            parts.append(gpu_advice)
        return " ".join(parts)


# ── live probes (only used when nothing is injected) ─────────────────────────

def _read_cpu_percent() -> float:
    try:
        import psutil
        return float(psutil.cpu_percent(interval=0.0))
    except Exception:
        return 0.0


__all__ = ["Pressure", "GovernorDecision", "CostGovernor"]
