"""
The panels that read UNKNOWN / NO DATA on screen.

Two separate causes, one of which explained two symptoms:

  * the Memory probe looked for a ``conn`` attribute MemoryAgent has never
    exposed, so it always returned UNKNOWN — and because Memory is a critical
    probe, that also dragged the NAV panel's "System Health" to UNKNOWN;
  * nothing ever registered a Security probe, so the SECURITY ring said
    NO DATA while the sentinel was running perfectly well.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orion_core.health_model import ONLINE, UNKNOWN, HealthModel


class _Sig:
    def emit(self, *a):
        pass

    def connect(self, *a):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


# ── Memory ────────────────────────────────────────────────────────────────────

def test_the_memory_probe_uses_the_real_public_api():
    """MemoryAgent has no conn/_conn — the old probe could only say UNKNOWN."""
    from orion_core.memory import MemoryAgent

    assert not hasattr(MemoryAgent, "conn")
    assert not hasattr(MemoryAgent, "_conn")
    assert hasattr(MemoryAgent, "tiers_snapshot"), (
        "the probe relies on this being the public way to ask the store a "
        "question")


def test_a_working_memory_store_reads_online():
    health = HealthModel(_Bus())
    memory = SimpleNamespace(tiers_snapshot=lambda: {"short": 12, "long": 40})
    health.register_defaults(memory=memory)
    entry = health.snapshot()["Memory"]
    assert entry["status"] == ONLINE
    assert "52 records" in entry["detail"]


def test_a_broken_memory_store_reads_offline_not_unknown():
    def boom():
        raise RuntimeError("database is locked")

    health = HealthModel(_Bus())
    health.register_defaults(memory=SimpleNamespace(tiers_snapshot=boom))
    entry = health.snapshot()["Memory"]
    assert entry["status"] == "OFFLINE"
    assert "database is locked" in entry["detail"]


def test_an_empty_but_responding_store_is_still_online():
    health = HealthModel(_Bus())
    health.register_defaults(memory=SimpleNamespace(tiers_snapshot=lambda: {}))
    assert health.snapshot()["Memory"]["status"] == ONLINE


def test_a_healthy_memory_no_longer_drags_system_health_to_unknown():
    """The NAV panel read 'System Health UNKNOWN' purely because of this."""
    health = HealthModel(_Bus())
    health.register_defaults(memory=SimpleNamespace(tiers_snapshot=lambda: {"a": 1}))
    assert health.overall() == ONLINE


# ── Security ──────────────────────────────────────────────────────────────────

def test_security_is_registered_when_a_sentinel_is_supplied():
    health = HealthModel(_Bus())
    health.register_defaults(security=SimpleNamespace(enabled=True, alerts=[]))
    assert "Security" in health.registered(), (
        "the SECURITY ring said NO DATA because no probe existed")


def test_a_monitoring_sentinel_with_no_alerts_is_online():
    health = HealthModel(_Bus())
    health.register_defaults(security=SimpleNamespace(enabled=True, alerts=[]))
    assert health.snapshot()["Security"]["status"] == ONLINE


def test_open_alerts_read_degraded():
    health = HealthModel(_Bus())
    health.register_defaults(
        security=SimpleNamespace(enabled=True, alerts=["a", "b"]))
    entry = health.snapshot()["Security"]
    assert entry["status"] == "DEGRADED"
    assert "2 alert" in entry["detail"]


def test_monitoring_switched_off_reads_degraded_not_online():
    health = HealthModel(_Bus())
    health.register_defaults(security=SimpleNamespace(enabled=False, alerts=[]))
    entry = health.snapshot()["Security"]
    assert entry["status"] == "DEGRADED"
    assert "switched off" in entry["detail"]


def test_no_sentinel_means_absent_not_green():
    health = HealthModel(_Bus())
    health.register_defaults()
    assert "Security" not in health.registered()


# ── the rings point at probes that exist ──────────────────────────────────────
