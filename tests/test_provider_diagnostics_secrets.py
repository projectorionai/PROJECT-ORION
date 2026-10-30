"""Provider faults must not turn credentials into dashboard or journal data."""

from __future__ import annotations

import tempfile
from pathlib import Path

from orion_core.provider_diagnostics import (
    Diagnosis,
    DiagnosticCategory,
    classify_provider_error,
    persist_diagnosis,
    recent_diagnoses,
)
from orion_core.providers import (
    AIProviderProfile,
    AllTextProvidersFailedError,
    OrionProviderSettings,
    ProviderRouter,
)


def _synthetic_key(label):
    # Construct test data at runtime so distributable source has no key literals.
    return "sk-" + label + "-1234567890"


def test_error_detail_redacts_keys_before_truncation():
    secret = _synthetic_key("real-secret-value")
    detail = classify_provider_error(
        "HTTP 401: " + "x" * 160 + f"?api_key={secret}"
    )
    assert detail.category is DiagnosticCategory.INVALID_CREDENTIALS
    assert secret not in detail.detail
    assert "sk-real-secret" not in detail.detail
    assert "[REDACTED]" in detail.detail


def test_error_detail_redacts_authorization_and_json_fields():
    token = "eyJhbGciOiJIUzI1NiJ9.payload.signature"
    secret = _synthetic_key("json-secret")
    header = classify_provider_error(f"HTTP 403 Authorization: Bearer {token}")
    body = classify_provider_error(f'HTTP 401: {{"api_key":"{secret}"}}')
    assert token not in header.detail
    assert secret not in body.detail
    assert "[REDACTED]" in header.detail and "[REDACTED]" in body.detail


def test_persistence_sanitizes_direct_diagnoses_and_old_journal_reads():
    secret = _synthetic_key("persisted-secret")
    diagnosis = Diagnosis(
        DiagnosticCategory.INVALID_CREDENTIALS,
        f"api_key={secret}",
        f"Replace token={secret}",
        provider=f"provider-{secret}",
    )
    with tempfile.TemporaryDirectory() as directory:
        journal = Path(directory) / "diagnostics.jsonl"
        persist_diagnosis(diagnosis, journal=journal)
        assert secret not in journal.read_text(encoding="utf-8")
        # Existing journals may predate the write-time redaction. Reading
        # them for the UI must still hide credentials.
        journal.write_text(
            '{"category":"invalid_credentials","detail":"token=' + secret + '",'
            '"remediation":"replace it","provider":"p","at":0}\n',
            encoding="utf-8",
        )
        rows = recent_diagnoses(journal=journal)
        assert len(rows) == 1
        assert secret not in rows[0]["detail"]


def test_router_logs_and_raised_failures_do_not_echo_provider_keys(monkeypatch):
    secret = _synthetic_key("provider-key")
    error = RuntimeError(f"HTTP 401: https://api.example/v1?api_key={secret}")

    class Signal:
        def __init__(self):
            self.events = []

        def emit(self, *args):
            self.events.append(args)

    class Bus:
        def __init__(self):
            self.log = Signal()
            self.dashboard_event = Signal()
            self.provider_diagnosed = Signal()

    profile = AIProviderProfile(
        name="example", kind="openai_compatible", model="chat",
        api_key="configured-secret", base_url="https://api.example/v1",
    )
    settings = OrionProviderSettings("example", ["example"], {"example": profile})
    bus = Bus()
    router = ProviderRouter(settings, bus, memory=object())
    monkeypatch.setattr("orion_core.provider_diagnostics.persist_diagnosis", lambda *_: None)
    router.mark_failure(profile, error)

    assert secret not in router._failures["example"]
    assert secret not in str(bus.log.events)
    assert secret not in str(bus.dashboard_event.events)
    assert secret not in str(router.provider_snapshot())
    raised = AllTextProvidersFailedError(error, attempts=["example"])
    assert secret not in str(raised)
    assert secret not in raised.last_error
