"""
Tests for SecurityReconService — real cybersecurity/pentesting execution
tooling for the user's own authorized targets/labs/CTFs.

Hermetic: these never touch a real network, never shell out to a real nmap
binary, and never require Npcap. The scope gate (cyber_curriculum.
gate_security_action, default-deny) is exercised for real; nmap/scapy calls
themselves are mocked or simply never reached because an unauthorized
target refuses before either library would be touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import authorized_targets as at_module
from orion_core.data import ToolResult
from orion_core.security_recon import SecurityReconService


class _RecordingSignal:
    def __init__(self) -> None:
        self.emitted: list = []

    def emit(self, *args) -> None:
        self.emitted.append(args[0] if len(args) == 1 else args)

    def connect(self, *a, **k) -> None:
        pass


class _FakeBus:
    def __init__(self) -> None:
        self.log = _RecordingSignal()


def _service(tmp_path, monkeypatch) -> SecurityReconService:
    # Isolate the authorized-targets store so tests never touch the real
    # config/authorized_targets.json.
    monkeypatch.setattr(at_module, "AUTHORIZED_TARGETS_PATH", tmp_path / "targets.json")
    return SecurityReconService(_FakeBus())


# ── scope management ──────────────────────────────────────────────────────────

def test_authorize_revoke_list_round_trip(tmp_path, monkeypatch):
    import asyncio
    service = _service(tmp_path, monkeypatch)
    result = service.authorize_target("192.168.1.50")
    assert result.ok
    assert "192.168.1.50" in result.text

    listed = service.list_authorized()
    assert "192.168.1.50" in listed.text

    revoked = service.revoke_target("192.168.1.50")
    assert revoked.ok
    listed_after = service.list_authorized()
    assert "192.168.1.50" not in listed_after.text


def test_list_authorized_reports_none_when_empty(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    result = service.list_authorized()
    assert "no targets" in result.text.lower()


def test_authorize_requires_a_target(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    result = service.authorize_target("")
    assert not result.ok


# ── scan_host: the scope gate ───────────────────────────────────────────────────

def test_scan_host_refuses_unauthorized_target(tmp_path, monkeypatch):
    import asyncio
    service = _service(tmp_path, monkeypatch)
    # A public-looking hostname, never authorized — must refuse, and must
    # never even probe for the nmap binary.
    probed = []
    monkeypatch.setattr(
        "shutil.which", lambda name: probed.append(name) or None
    )
    result = asyncio.run(service.scan_host("example.com"))
    assert not result.ok
    assert "denied" in result.text.lower() or "refused" in result.text.lower()
    assert probed == []  # never reached the nmap-availability check


def test_scan_host_loopback_is_always_in_scope_but_reports_missing_nmap(tmp_path, monkeypatch):
    import asyncio
    service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr("shutil.which", lambda name: None)  # simulate nmap absent
    result = asyncio.run(service.scan_host("127.0.0.1"))
    assert not result.ok
    assert "nmap" in result.text.lower()
    assert "nmap.org" in result.text.lower()


def test_scan_host_authorized_target_with_nmap_present_calls_scanner(tmp_path, monkeypatch):
    import asyncio
    service = _service(tmp_path, monkeypatch)
    service.authorize_target("10.0.0.9")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nmap")

    class _FakeScanner:
        def scan(self, target, ports):
            assert target == "10.0.0.9"
            return {"scan": {"10.0.0.9": {
                "status": {"state": "up"},
                "tcp": {22: {"state": "open", "name": "ssh", "product": "OpenSSH"}},
            }}}

    fake_nmap = type(sys)("nmap")
    fake_nmap.PortScanner = _FakeScanner
    monkeypatch.setitem(sys.modules, "nmap", fake_nmap)

    result = asyncio.run(service.scan_host("10.0.0.9"))
    assert result.ok
    assert "22" in result.text and "ssh" in result.text.lower()


def test_format_scan_reports_no_results_for_empty_scan():
    text = SecurityReconService._format_scan("10.0.0.9", {"scan": {}})
    assert "no results" in text.lower()


# ── packet_capture / craft_packet: same scope gate ──────────────────────────────

def test_packet_capture_refuses_unauthorized_target(tmp_path, monkeypatch):
    import asyncio
    service = _service(tmp_path, monkeypatch)
    result = asyncio.run(service.packet_capture("example.com"))
    assert not result.ok
    assert "refused" in result.text.lower() or "denied" in result.text.lower()


def test_craft_packet_refuses_unauthorized_target(tmp_path, monkeypatch):
    import asyncio
    service = _service(tmp_path, monkeypatch)
    result = asyncio.run(service.craft_packet("example.com"))
    assert not result.ok
    assert "refused" in result.text.lower() or "denied" in result.text.lower()


def test_craft_packet_requires_a_target(tmp_path, monkeypatch):
    import asyncio
    service = _service(tmp_path, monkeypatch)
    result = asyncio.run(service.craft_packet(""))
    assert not result.ok


# ── cve_lookup: no authorization needed (public, read-only) ───────────────────

class _FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeGetContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def get(self, url, params=None, timeout=None):
        return _FakeGetContext(self._response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_cve_lookup_needs_no_authorization_even_for_unauthorized_scope(tmp_path, monkeypatch):
    """cve_lookup must work with an EMPTY authorized-targets store — it's a
    public read, not a target-touching action."""
    import asyncio
    import aiohttp
    service = _service(tmp_path, monkeypatch)  # nothing authorized
    payload = {
        "vulnerabilities": [{
            "cve": {
                "id": "CVE-2024-12345",
                "descriptions": [{"lang": "en", "value": "A serious issue."}],
                "metrics": {"cvssMetricV31": [{"cvssData": {"baseSeverity": "HIGH"}}]},
            }
        }]
    }
    monkeypatch.setattr(
        aiohttp, "ClientSession",
        lambda *a, **k: _FakeSession(_FakeResponse(200, payload)),
    )
    result = asyncio.run(service.cve_lookup("CVE-2024-12345"))
    assert result.ok
    assert "CVE-2024-12345" in result.text
    assert "HIGH" in result.text


def test_cve_lookup_reports_no_results(tmp_path, monkeypatch):
    import asyncio
    import aiohttp
    service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        aiohttp, "ClientSession",
        lambda *a, **k: _FakeSession(_FakeResponse(200, {"vulnerabilities": []})),
    )
    result = asyncio.run(service.cve_lookup("totally-made-up-query"))
    assert "no cve results" in result.text.lower()


def test_cve_lookup_requires_a_query(tmp_path, monkeypatch):
    import asyncio
    service = _service(tmp_path, monkeypatch)
    result = asyncio.run(service.cve_lookup(""))
    assert not result.ok


# ── cve_lookup: per-process TTL cache ─────────────────────────────────────────
#
# Published CVE records don't change minute to minute, so repeating the same
# lookup within a session is pure latency (and burns NVD's public rate limit).

class _CountingSession(_FakeSession):
    """_FakeSession that records how many round-trips were actually made."""

    def __init__(self, response: _FakeResponse, counter: list) -> None:
        super().__init__(response)
        self._counter = counter

    def get(self, url, params=None, timeout=None):
        self._counter.append(params)
        return super().get(url, params=params, timeout=timeout)


_CVE_PAYLOAD = {
    "vulnerabilities": [{
        "cve": {
            "id": "CVE-2024-12345",
            "descriptions": [{"lang": "en", "value": "A serious issue."}],
            "metrics": {"cvssMetricV31": [{"cvssData": {"baseSeverity": "HIGH"}}]},
        }
    }]
}


def _counting_service(tmp_path, monkeypatch, status: int = 200, payload=None):
    import aiohttp
    service = _service(tmp_path, monkeypatch)
    calls: list = []
    monkeypatch.setattr(
        aiohttp, "ClientSession",
        lambda *a, **k: _CountingSession(
            _FakeResponse(status, _CVE_PAYLOAD if payload is None else payload), calls),
    )
    return service, calls


def test_cve_lookup_caches_repeated_queries_within_the_ttl(tmp_path, monkeypatch):
    import asyncio
    service, calls = _counting_service(tmp_path, monkeypatch)
    first = asyncio.run(service.cve_lookup("CVE-2024-12345"))
    second = asyncio.run(service.cve_lookup("CVE-2024-12345"))
    assert len(calls) == 1, "the second lookup must be served from cache"
    assert first.text == second.text
    assert second.ok


def test_cve_lookup_cache_key_ignores_case_and_surrounding_space(tmp_path, monkeypatch):
    import asyncio
    service, calls = _counting_service(tmp_path, monkeypatch)
    asyncio.run(service.cve_lookup("CVE-2024-12345"))
    asyncio.run(service.cve_lookup("  cve-2024-12345  "))
    assert len(calls) == 1


def test_cve_lookup_refetches_once_the_ttl_has_expired(tmp_path, monkeypatch):
    import asyncio
    service, calls = _counting_service(tmp_path, monkeypatch)
    asyncio.run(service.cve_lookup("CVE-2024-12345"))
    monkeypatch.setattr(service, "_CVE_CACHE_TTL_S", 0.0)
    asyncio.run(service.cve_lookup("CVE-2024-12345"))
    assert len(calls) == 2


def test_cve_lookup_caches_a_genuine_no_results_answer(tmp_path, monkeypatch):
    # "No results" is a real answer from NVD, not a failure — worth caching.
    import asyncio
    service, calls = _counting_service(
        tmp_path, monkeypatch, payload={"vulnerabilities": []})
    asyncio.run(service.cve_lookup("nothing-matches-this"))
    asyncio.run(service.cve_lookup("nothing-matches-this"))
    assert len(calls) == 1


def test_cve_lookup_never_caches_an_http_failure(tmp_path, monkeypatch):
    import asyncio
    service, calls = _counting_service(tmp_path, monkeypatch, status=500)
    first = asyncio.run(service.cve_lookup("CVE-2024-12345"))
    assert not first.ok
    asyncio.run(service.cve_lookup("CVE-2024-12345"))
    assert len(calls) == 2, "a failed lookup must not get stuck in the cache"


def test_different_queries_are_cached_separately(tmp_path, monkeypatch):
    import asyncio
    service, calls = _counting_service(tmp_path, monkeypatch)
    asyncio.run(service.cve_lookup("CVE-2024-12345"))
    asyncio.run(service.cve_lookup("CVE-2024-99999"))
    assert len(calls) == 2


def test_extract_severity_falls_back_across_cvss_versions():
    assert SecurityReconService._extract_severity(
        {"cvssMetricV31": [{"cvssData": {"baseSeverity": "CRITICAL"}}]}
    ) == "CRITICAL"
    assert SecurityReconService._extract_severity({}) == ""
