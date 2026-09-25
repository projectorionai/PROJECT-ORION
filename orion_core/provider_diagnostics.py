"""
Provider diagnostic subsystem (Section 5).

Classifies a provider fault into a specific, actionable category and produces a
remediation message — turning an opaque failure into "here is exactly what is
wrong and how to fix it".  Diagnostic events are persisted through the project's
existing config directory (a small JSONL journal) so the history survives
restarts, and are surfaced to the diagnostics UI.

Self-diagnosis NEVER makes a destructive system change — it only inspects and
reports.  No secrets are read, logged or included in any message.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

try:
    from .constants import CONFIG_DIR
except Exception:  # pragma: no cover
    CONFIG_DIR = Path(".")

DIAGNOSTICS_JOURNAL = CONFIG_DIR / "diagnostics" / "provider_events.jsonl"


class DiagnosticCategory(str, Enum):
    OK = "ok"
    MISSING_CONFIG = "missing_configuration"
    INVALID_CREDENTIALS = "invalid_credentials"
    UNREACHABLE_ENDPOINT = "unreachable_endpoint"
    DNS_FAILURE = "dns_network_failure"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    UNSUPPORTED_MODEL = "unsupported_model"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    CONTEXT_OVERFLOW = "context_window_overflow"
    LOCAL_MODEL_DOWN = "local_model_not_running"
    SDK_FAILURE = "provider_sdk_failure"
    RECONNECT_LOOP = "repeated_reconnect_loop"
    UNKNOWN = "unknown"


_REMEDIATION = {
    DiagnosticCategory.MISSING_CONFIG: "Add the provider's API key (or a local base_url) in config/api_keys.json.",
    DiagnosticCategory.INVALID_CREDENTIALS: "The API key was rejected — replace it with a valid key in config/api_keys.json.",
    DiagnosticCategory.UNREACHABLE_ENDPOINT: "The endpoint refused the connection — check the base_url and that the service is running.",
    DiagnosticCategory.DNS_FAILURE: "Name resolution/network failed — check internet connectivity and DNS.",
    DiagnosticCategory.TIMEOUT: "The request timed out — the network may be slow or the provider overloaded; retry or raise the timeout.",
    DiagnosticCategory.RATE_LIMIT: "You are being rate-limited — back off, then retry; consider spreading load across keys/providers.",
    DiagnosticCategory.QUOTA_EXHAUSTED: "Account quota/credits are exhausted — top up, or switch to another provider or a local model.",
    DiagnosticCategory.UNSUPPORTED_MODEL: "The requested model is unavailable — pick a supported model id for this provider.",
    DiagnosticCategory.UNSUPPORTED_CAPABILITY: "This provider/model cannot perform the requested capability — route the task to one that can.",
    DiagnosticCategory.CONTEXT_OVERFLOW: "The prompt exceeds the model's context window — shorten the input or use a larger-context model.",
    DiagnosticCategory.LOCAL_MODEL_DOWN: "The local LLM is not reachable — start Ollama/LM Studio (or correct its base_url/port).",
    DiagnosticCategory.SDK_FAILURE: "The provider SDK raised an unexpected error — check versions and logs.",
    DiagnosticCategory.RECONNECT_LOOP: "The channel is reconnecting repeatedly — rotate credential/provider or fall back to local.",
    DiagnosticCategory.UNKNOWN: "Unclassified provider error — inspect the log detail.",
    DiagnosticCategory.OK: "No fault detected.",
}


@dataclass
class Diagnosis:
    category: DiagnosticCategory
    detail: str                       # short, redaction-safe
    remediation: str
    provider: str = ""
    at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "detail": self.detail,
            "remediation": self.remediation,
            "provider": self.provider,
            "at": self.at,
        }


# Ordered (pattern, category) — first match wins, so specific patterns precede
# generic ones.
_RULES: tuple[tuple[re.Pattern[str], DiagnosticCategory], ...] = (
    (re.compile(r"(?i)no (?:text )?provider|not configured|no api.?key|missing.*key"), DiagnosticCategory.MISSING_CONFIG),
    (re.compile(r"(?i)invalid api.?key|api key not valid|api key expired|unauthenti|unauthori[sz]|permission denied|401|403|forbidden"), DiagnosticCategory.INVALID_CREDENTIALS),
    (re.compile(r"(?i)quota|insufficient_quota|billing|credit|payment required|402"), DiagnosticCategory.QUOTA_EXHAUSTED),
    (re.compile(r"(?i)rate.?limit|resource exhausted|429|too many requests|overloaded"), DiagnosticCategory.RATE_LIMIT),
    (re.compile(r"(?i)context length|maximum context|context window|too many tokens|reduce the length|token.*exceed"), DiagnosticCategory.CONTEXT_OVERFLOW),
    (re.compile(r"(?i)model .*not found|no such model|unknown model|invalid model|unsupported model|does not exist|model_not_found|404"), DiagnosticCategory.UNSUPPORTED_MODEL),
    (re.compile(r"(?i)does not support|unsupported (?:capability|operation)|not implemented"), DiagnosticCategory.UNSUPPORTED_CAPABILITY),
    (re.compile(r"(?i)11434|1234|localhost|127\.0\.0\.1|0\.0\.0\.0"), DiagnosticCategory.LOCAL_MODEL_DOWN),
    (re.compile(r"(?i)refused|connection reset|ECONNREFUSED|cannot connect|connection error"), DiagnosticCategory.UNREACHABLE_ENDPOINT),
    (re.compile(r"(?i)getaddrinfo|name or service not known|nodename nor servname|dns|temporary failure in name resolution|network is unreachable"), DiagnosticCategory.DNS_FAILURE),
    (re.compile(r"(?i)timeout|timed out|deadline"), DiagnosticCategory.TIMEOUT),
    (re.compile(r"(?i)reconnect|go_away|repeated"), DiagnosticCategory.RECONNECT_LOOP),
)


def _first_line(text: str, limit: int = 200) -> str:
    line = str(text or "").strip().split("\n", 1)[0]
    return line[:limit]


def classify_provider_error(error: Any, provider: str = "",
                            clock: Callable[[], float] | None = None) -> Diagnosis:
    """Map an exception or message to a :class:`DiagnosticCategory`.  Order of
    rules matters — a "cannot connect to 127.0.0.1:11434" reads as the local
    model being down, not a generic unreachable endpoint."""
    now = (clock or time.time)()
    text = _first_line(error if isinstance(error, str) else str(error))
    # Local-endpoint connection failures should read as "local model down" even
    # though they also match the generic refused/unreachable pattern.
    is_conn = bool(re.search(r"(?i)refused|cannot connect|connection (?:error|reset)|timed out|timeout", text))
    is_local = bool(re.search(r"(?i)11434|:1234|localhost|127\.0\.0\.1|0\.0\.0\.0", text))
    if is_conn and is_local:
        cat = DiagnosticCategory.LOCAL_MODEL_DOWN
    else:
        cat = DiagnosticCategory.UNKNOWN
        for pattern, category in _RULES:
            if pattern.search(text):
                cat = category
                break
    return Diagnosis(cat, text, _REMEDIATION[cat], provider=provider, at=now)


def diagnose_profile_config(profile: Any) -> Diagnosis:
    """Static (no-network) configuration check for a single provider profile."""
    now = time.time()
    name = str(getattr(profile, "name", "?"))
    kind = str(getattr(profile, "kind", "") or "")
    enabled = bool(getattr(profile, "enabled", True))
    is_local = bool(getattr(profile, "is_local", False))
    has_key = bool(str(getattr(profile, "api_key", "") or "").strip())
    base_url = str(getattr(profile, "base_url", "") or "")
    model = str(getattr(profile, "model", "") or "")
    if not enabled:
        return Diagnosis(DiagnosticCategory.MISSING_CONFIG, f"{name} is disabled",
                         "Enable the provider in config/api_keys.json.", name, now)
    if kind == "openai_compatible":
        if not base_url:
            return Diagnosis(DiagnosticCategory.MISSING_CONFIG, f"{name} has no base_url",
                             _REMEDIATION[DiagnosticCategory.MISSING_CONFIG], name, now)
        if not is_local and not has_key:
            return Diagnosis(DiagnosticCategory.MISSING_CONFIG, f"{name} has no API key",
                             _REMEDIATION[DiagnosticCategory.MISSING_CONFIG], name, now)
    if kind == "gemini_live" and not has_key:
        return Diagnosis(DiagnosticCategory.MISSING_CONFIG, f"{name} has no API key",
                         _REMEDIATION[DiagnosticCategory.MISSING_CONFIG], name, now)
    if not model and kind != "gemini_live":
        return Diagnosis(DiagnosticCategory.UNSUPPORTED_MODEL, f"{name} has no model set",
                         _REMEDIATION[DiagnosticCategory.UNSUPPORTED_MODEL], name, now)
    return Diagnosis(DiagnosticCategory.OK, f"{name} configuration looks valid",
                     _REMEDIATION[DiagnosticCategory.OK], name, now)


def persist_diagnosis(diagnosis: Diagnosis, journal: Path | None = None) -> None:
    """Append a diagnostic event to the JSONL journal (existing-persistence
    mechanism).  Never raises into the caller; contains no secrets."""
    path = journal or DIAGNOSTICS_JOURNAL
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(diagnosis.as_dict()) + "\n")
    except Exception:
        pass


def recent_diagnoses(limit: int = 20, journal: Path | None = None) -> list[dict[str, Any]]:
    path = journal or DIAGNOSTICS_JOURNAL
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
    except Exception:
        return []
