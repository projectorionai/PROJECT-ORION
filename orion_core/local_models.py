"""
Local model management (Phase 1) — native Ollama discovery + registration.

ORION reaches local models (Qwen, Llama, DeepSeek, Mistral, Gemma, …) through
Ollama's OpenAI-compatible endpoint at ``/v1/chat/completions``.  This module:

    • detects a running Ollama server (default 127.0.0.1:11434);
    • lists the models the user has actually pulled;
    • picks the best default model for ORION and enables/updates the
      ``local_ollama`` provider profile so the router can use it immediately;
    • recommends models worth pulling for a stronger offline brain.

It touches the network only against localhost, so it is safe and instant even
with the internet down — the whole point of MODE B.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

# Preference order when several models are pulled — newer/stronger first.
# Matched as substrings against the pulled model tags.
_MODEL_PREFERENCE = (
    "qwen2.5", "qwen2", "qwen", "llama3.2", "llama3.1", "llama3", "llama",
    "deepseek-r1", "deepseek", "mistral", "mixtral", "gemma2", "gemma", "phi",
)

# Curated pull suggestions surfaced to the user for a better offline brain.
RECOMMENDED_PULLS = (
    ("qwen2.5:7b", "Excellent general reasoning + instruction following (4.7 GB)"),
    ("llama3.1:8b", "Strong all-rounder from Meta (4.9 GB)"),
    ("deepseek-r1:8b", "Chain-of-thought reasoning specialist (5.2 GB)"),
    ("mistral:7b", "Fast, lean, good for quick daily chat (4.1 GB)"),
    ("gemma2:9b", "Google's capable compact model (5.4 GB)"),
)


class OllamaManager:
    """Discovers and wires up a local Ollama server."""

    def __init__(self, bus: Any | None = None, telemetry: Any | None = None,
                 host: str = "") -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.host = (host or os.getenv("ORION_OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.available = False
        self.models: list[str] = []

    # ── discovery ─────────────────────────────────────────────────────────────

    def probe(self, timeout: float = 2.0) -> bool:
        """Query /api/tags; populate the pulled-model list. True if reachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            self.models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            self.available = True
            return True
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            self.available = False
            self.models = []
            return False

    def best_model(self) -> str:
        """The strongest pulled model per the preference order (or first pulled)."""
        if not self.models:
            return ""
        for pref in _MODEL_PREFERENCE:
            for tag in self.models:
                if pref in tag.lower():
                    return tag
        return self.models[0]

    # ── registration into provider settings ──────────────────────────────────

    def register(self, settings: Any) -> bool:
        """
        Enable/refresh the ``local_ollama`` provider with the best pulled model.
        Returns True if a usable local model was wired up.  Safe to call every
        startup — idempotent.
        """
        if not self.probe():
            self._log("SYS: Ollama not detected; local LLM inference offline "
                      "(install from ollama.com and `ollama pull qwen2.5` for MODE B).")
            return False
        model = self.best_model()
        if not model:
            self._log("SYS: Ollama is running but no models are pulled. "
                      "Run `ollama pull qwen2.5` to give ORION an offline brain.")
            return False
        profile = settings.providers.get("local_ollama")
        base_url = f"{self.host}/v1"
        if profile is None:
            # Build a fresh profile object matching the dataclass shape.
            from .providers import AIProviderProfile
            profile = AIProviderProfile(
                name="local_ollama", kind="openai_compatible", model=model,
                api_key="local", base_url=base_url, enabled=True,
                priority=70, timeout_s=120.0, strengths=("local", "fast", "offline"),
            )
            settings.providers["local_ollama"] = profile
            if "local_ollama" not in settings.provider_order:
                settings.provider_order.append("local_ollama")
        else:
            profile.enabled = True
            profile.model = model
            profile.base_url = base_url
            if "offline" not in profile.strengths:
                profile.strengths = tuple(sorted(set(profile.strengths) | {"local", "offline"}))
        self._log(f"SYS: Ollama online — {len(self.models)} model(s); "
                  f"ORION's offline brain = '{model}'.")
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("local.models", float(len(self.models)))
            self.telemetry.health.beat("ollama", "OK", model)
        return True

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "host": self.host,
            "models": list(self.models),
            "best_model": self.best_model(),
        }

    def recommendations(self) -> list[tuple[str, str]]:
        pulled = {t.split(":")[0].lower() for t in self.models}
        return [(tag, why) for tag, why in RECOMMENDED_PULLS
                if tag.split(":")[0].lower() not in pulled]

    def _log(self, message: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(message)
            except RuntimeError:
                pass


class AIModeInfo:
    """Reports ORION's active intelligence mode (cloud vs offline) and models."""

    def __init__(self, router: Any, connectivity: Any | None = None,
                 ollama: Any | None = None, offline_stt: Any | None = None) -> None:
        self.router = router
        self.connectivity = connectivity
        self.ollama = ollama
        self.offline_stt = offline_stt

    def report(self) -> Any:
        from .data import ToolResult
        snap = self.router.provider_snapshot()
        lines = [
            f"Intelligence mode: {snap.get('mode')}",
            f"Internet: {'online' if snap.get('online') else 'OFFLINE'}",
            f"Cloud live voice: {', '.join(snap.get('available_live') or []) or 'none'}",
            f"Text models available: {', '.join(snap.get('available_text') or []) or 'none'}",
            f"Local models (offline-capable): {', '.join(snap.get('available_local') or []) or 'none'}",
        ]
        if self.ollama is not None:
            st = self.ollama.status()
            lines.append(f"Ollama: {'up' if st['available'] else 'down'}"
                         + (f" — brain '{st['best_model']}'" if st['best_model'] else ""))
            recs = self.ollama.recommendations()
            if recs and not st["models"]:
                lines.append("Pull a model for a stronger offline brain: "
                             + ", ".join(t for t, _ in recs[:3]))
        if self.offline_stt is not None:
            lines.append(f"Offline dictation: {self.offline_stt.status().get('engine', 'none')}")
        return ToolResult("\n".join(lines))
