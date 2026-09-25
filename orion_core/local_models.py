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
# Lightweight options lead — this is a FALLBACK brain (cloud-first routing
# means it only ever runs when the cloud is unreachable), and on a modest
# box (e.g. 16GB RAM / 8GB VRAM) a 7-9B model plus its KV-cache can leave
# little headroom once ORION itself and the rest of Windows are accounted
# for. The bigger models stay listed for machines that can spare it.
RECOMMENDED_PULLS = (
    ("llama3.2:3b", "Light, fast fallback brain — comfortable on 8GB VRAM (2.0 GB)"),
    ("qwen2.5:3b", "Light, strong for its size, good instruction following (1.9 GB)"),
    ("phi3.5:3.8b", "Compact Microsoft model, quick and capable (2.2 GB)"),
    ("qwen2.5:7b", "Excellent general reasoning + instruction following (4.7 GB)"),
    ("llama3.1:8b", "Strong all-rounder from Meta (4.9 GB)"),
    ("mistral:7b", "Fast, lean, good for quick daily chat (4.1 GB)"),
    ("deepseek-r1:8b", "Chain-of-thought reasoning specialist (5.2 GB)"),
    ("gemma2:9b", "Google's capable compact model (5.4 GB)"),
)

# Conservative fraction of TOTAL system RAM a local fallback model's weights
# should stay under — the rest of that budget goes to ORION itself, the OS
# and whatever else is running. A model comfortably under this fits without
# the fallback path becoming the reason the machine starts swapping.
_SAFE_MODEL_FRACTION_OF_RAM = 0.35


class OllamaManager:
    """Discovers and wires up a local Ollama server."""

    def __init__(self, bus: Any | None = None, telemetry: Any | None = None,
                 host: str = "") -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.host = (host or os.getenv("ORION_OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.available = False
        self.models: list[str] = []
        self.model_sizes: dict[str, int] = {}   # tag -> size in bytes, from /api/tags

    # ── discovery ─────────────────────────────────────────────────────────────

    #: How long to wait for Ollama at STARTUP.
    #:
    #: Measured on this machine: a raw socket connect to 127.0.0.1:11434 with
    #: nothing listening does not get refused — it hangs and times out. (A
    #: closed local port normally refuses instantly; something here, most
    #: likely a firewall rule or a WSL/Hyper-V port reservation, is dropping
    #: the packets rather than rejecting them.) A hanging connect cannot be
    #: made fast, so the only lever is how long ORION is willing to wait, and
    #: two seconds of frozen GUI to discover a service is absent is a bad
    #: trade. Ollama on loopback answers in single-digit milliseconds when it
    #: IS running, so a third of a second is generous for the case that matters
    #: and cheap for the case that does not.
    STARTUP_PROBE_TIMEOUT = 0.35

    def probe(self, timeout: float = 2.0) -> bool:
        """Query /api/tags; populate the pulled-model list. True if reachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            entries = [m for m in data.get("models", []) if m.get("name")]
            self.models = [m["name"] for m in entries]
            self.model_sizes = {m["name"]: int(m["size"]) for m in entries if m.get("size")}
            self.available = True
            return True
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            self.available = False
            self.models = []
            self.model_sizes = {}
            return False

    #: How long Ollama keeps a model resident after the last request.
    #:
    #: Ollama's own default is 5 minutes, and it holds the FULL weights plus the
    #: KV-cache for that whole time — which is why a 7B model quietly sits on
    #: ~5 GB of RAM long after ORION has finished with it. That is the "why does
    #: Ollama take up so much RAM" answer, and it is a setting rather than a
    #: fact of life. ORION uses local inference as a FALLBACK, in bursts, so
    #: paying five minutes of resident memory for a thirty-second conversation
    #: is the wrong trade: it reloads in a couple of seconds when next needed.
    #:
    #: Override with ORION_OLLAMA_KEEP_ALIVE if you would rather trade RAM for
    #: reload latency.
    KEEP_ALIVE = os.getenv("ORION_OLLAMA_KEEP_ALIVE", "30s")

    def keep_alive_env(self) -> dict[str, str]:
        """Environment for a server ORION starts itself."""
        env = dict(os.environ)
        env["OLLAMA_KEEP_ALIVE"] = self.KEEP_ALIVE
        # One model resident at a time: ORION only ever needs the one, and the
        # default lets several accumulate.
        env.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
        return env

    def autostart(self, wait: float = 6.0) -> bool:
        """Try to bring Ollama up ourselves. Returns True if it is now serving.

        Called when every cloud provider has failed: if the user has Ollama
        installed but not running, ORION starting it is the difference between
        "no text provider available" and simply carrying on more slowly.
        Never raises; never blocks longer than *wait*.
        """
        import shutil
        import subprocess
        import time as _time

        if self.probe(0.35):
            return True
        binary = shutil.which("ollama")
        if not binary:
            return False
        try:
            self._log("SYS: no cloud provider reachable — starting Ollama for a "
                      "local fallback brain.")
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(
                [binary, "serve"], env=self.keep_alive_env(),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=creation)
        except Exception as exc:
            self._log(f"SYS: could not start Ollama - {exc}")
            return False
        deadline = _time.monotonic() + max(1.0, wait)
        while _time.monotonic() < deadline:
            if self.probe(0.5):
                self._log("SYS: Ollama is up — local inference available.")
                return True
            _time.sleep(0.4)
        self._log("SYS: Ollama did not come up in time; staying on deterministic tools.")
        return False

    def recover(self, settings: Any) -> bool:
        """Full local-brain recovery: start Ollama if needed, then register it."""
        if not self.autostart():
            return False
        return self.register(settings, timeout=2.0)

    @staticmethod
    def _system_ram_gb() -> float | None:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            return None

    def _safe_model_budget_bytes(self) -> int | None:
        """How large a fallback model's weights can be before it starts
        eating into what ORION and the rest of the machine need — None
        when total RAM can't be read (psutil missing), so callers degrade
        to size-blind selection rather than guessing."""
        ram_gb = self._system_ram_gb()
        if ram_gb is None:
            return None
        return int(ram_gb * _SAFE_MODEL_FRACTION_OF_RAM * (1024 ** 3))

    def best_model(self) -> str:
        """The strongest pulled model that fits this machine's RAM budget,
        per the preference order. Falls back to the strongest pulled model
        regardless of size if none fit (a working local brain beats none),
        and to size-blind preference alone when size data isn't available —
        e.g. psutil missing, or /api/tags omitted sizes."""
        if not self.models:
            return ""
        budget = self._safe_model_budget_bytes()
        if budget is not None and self.model_sizes:
            for pref in _MODEL_PREFERENCE:
                for tag in self.models:
                    if pref in tag.lower() and self.model_sizes.get(tag, 0) <= budget:
                        return tag
        for pref in _MODEL_PREFERENCE:
            for tag in self.models:
                if pref in tag.lower():
                    return tag
        return self.models[0]

    # ── registration into provider settings ──────────────────────────────────

    def register(self, settings: Any, timeout: float | None = None) -> bool:
        """
        Enable/refresh the ``local_ollama`` provider with the best pulled model.
        Returns True if a usable local model was wired up.  Safe to call every
        startup — idempotent.

        *timeout* bounds the reachability probe. Startup passes
        ``STARTUP_PROBE_TIMEOUT``; a later, deliberate re-check (the user asking
        "is Ollama up?") can afford to wait longer and should pass its own.
        """
        if not self.probe(self.STARTUP_PROBE_TIMEOUT if timeout is None else timeout):
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
        self._warn_if_oversized(model)
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("local.models", float(len(self.models)))
            self.telemetry.health.beat("ollama", "OK", model)
        return True

    def _warn_if_oversized(self, model: str) -> None:
        """A one-line, actionable heads-up (not a block — a working fallback
        beats none) when the chosen model is larger than this machine can
        comfortably spare for a cloud-outage safety net."""
        budget = self._safe_model_budget_bytes()
        size = self.model_sizes.get(model)
        if budget is None or not size or size <= budget:
            return
        self._log(
            f"SYS: local brain '{model}' is ~{size / (1024 ** 3):.1f}GB — "
            f"larger than the ~{budget / (1024 ** 3):.1f}GB this machine can "
            "comfortably spare for a fallback model alongside ORION and "
            "everything else running. Consider `ollama pull llama3.2:3b` "
            "(~2GB) for a lighter fallback."
        )

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
