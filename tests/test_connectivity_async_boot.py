"""Connectivity startup must not stall the GUI or headless event loop."""

from __future__ import annotations

import asyncio

from orion_core.connectivity import ConnectivityMonitor


def test_constructor_defers_network_probe_inside_event_loop(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ConnectivityMonitor, "_probe", lambda self: calls.append("probe") or True)

    async def create() -> ConnectivityMonitor:
        return ConnectivityMonitor()

    monitor = asyncio.run(create())
    assert calls == []
    assert monitor.mode() == "CHECKING CONNECTIVITY"


def test_constructor_probes_when_no_event_loop_is_running(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ConnectivityMonitor, "_probe", lambda self: calls.append("probe") or True)
    monitor = ConnectivityMonitor()
    assert calls == ["probe"]
    assert monitor.is_online()


def test_one_failed_probe_does_not_switch_orion_offline(monkeypatch):
    """7 Oct 21:32: one round of timed-out connects left the voice with "no
    live audio provider" for five seconds, with no provider at fault."""
    answers = iter([True, False, True, False, False])
    monkeypatch.setattr(ConnectivityMonitor, "_probe", lambda self: next(answers))
    monitor = ConnectivityMonitor()                 # online
    assert monitor.is_online(force=True)            # one miss: still online
    assert monitor.is_online(force=True)            # recovered
    assert monitor.is_online(force=True)            # a new miss
    assert not monitor.is_online(force=True)        # second in a row: offline


def test_a_miss_is_rechecked_soon_not_trusted_for_the_cache_period(monkeypatch):
    monkeypatch.setattr(ConnectivityMonitor, "_probe", lambda self: True)
    monitor = ConnectivityMonitor()
    monitor._update(False)
    assert monitor._online and monitor._checked_at == 0.0


def test_forced_offline_never_probes_even_when_a_refresh_is_requested(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("forced offline mode attempted a network probe")

    monkeypatch.setenv("ORION_MODE", "offline")
    monkeypatch.setattr(ConnectivityMonitor, "_probe", forbidden)
    monitor = ConnectivityMonitor()
    assert not monitor.is_online(force=True)
    assert not monitor.cached_online()
    assert monitor.mode() == "MODE B (fully offline)"

    async def one_monitor_iteration():
        running = ConnectivityMonitor()

        async def finish(_delay):
            running.stop()

        monkeypatch.setattr("orion_core.connectivity.asyncio.sleep", finish)
        await running.run()
        return running

    assert not asyncio.run(one_monitor_iteration()).cached_online()


def test_cached_connectivity_cannot_launch_a_probe_from_the_speech_thread(monkeypatch):
    monkeypatch.delenv("ORION_MODE", raising=False)
    monkeypatch.setattr(ConnectivityMonitor, "_probe", lambda self: True)
    monitor = ConnectivityMonitor()
    monitor._checked_at = 0.0

    def forbidden(*args, **kwargs):
        raise AssertionError("cached speech predicate attempted a refresh")

    monkeypatch.setattr(monitor, "_probe", forbidden)
    assert monitor.cached_online()
    monitor._update(False)
    monitor._update(False)
    assert not monitor.cached_online()
