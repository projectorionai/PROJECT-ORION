"""
Tests for the uplink's best-effort firewall opener — the fix for a phone on the
same Wi-Fi seeing 'refused to connect'.  subprocess is fully mocked; the real
Windows Firewall is never touched.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.remote as remote
from orion_core.remote import RemoteGateway


class _Sig:
    def emit(self, *a):
        pass


class _Bus:
    def __getattr__(self, n):
        s = _Sig()
        object.__setattr__(self, n, s)
        return s


def _gateway(tmp_path):
    gateway = RemoteGateway(object(), object(), _Bus(), config_dir=tmp_path)
    gateway.host = "0.0.0.0"  # Explicit direct LAN mode needs inbound rules.
    return gateway


def _fake_run_factory(calls, *, show_stdout="", show_rc=0, add_rc=0):
    def _fake(args, **kw):
        calls.append(list(args))

        class _R:
            pass

        r = _R()
        if "show" in args:
            r.returncode, r.stdout, r.stderr = show_rc, show_stdout, ""
        else:
            r.returncode, r.stdout, r.stderr = add_rc, "", ""
        return r

    return _fake


def test_rule_added_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_on_windows", lambda: True)
    monkeypatch.delenv("ORION_REMOTE_FIREWALL", raising=False)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        remote.subprocess, "run",
        _fake_run_factory(calls, show_stdout="No rules match the specified criteria."))

    gw = _gateway(tmp_path)
    asyncio.run(gw._ensure_firewall_access())

    add = [c for c in calls if "add" in c]
    assert len(add) == 2, "LAN and Tailscale rules are expected when absent"
    lan_rule, tailscale_rule = add
    assert any(f"localport={gw.port}" in tok for tok in lan_rule)
    assert "profile=private,domain" in lan_rule
    assert "profile=public" in tailscale_rule
    assert "remoteip=100.64.0.0/10,fd7a:115c:a1e0::/48" in tailscale_rule


def test_rule_not_re_added_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_on_windows", lambda: True)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        remote.subprocess, "run",
        _fake_run_factory(calls, show_stdout=(
            "Rule Name: ORION Remote Uplink\nEnabled: Yes\n"
            "Rule Name: ORION Remote Uplink (Tailscale)\nEnabled: Yes")))

    gw = _gateway(tmp_path)
    asyncio.run(gw._ensure_firewall_access())

    assert not any("add" in c for c in calls), "must not duplicate an existing rule"


def test_skipped_off_windows(tmp_path, monkeypatch):
    # Build the gateway with the real os.name (Path needs it), then simulate a
    # non-Windows host only for the duration of the method call.
    gw = _gateway(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(remote.subprocess, "run", _fake_run_factory(calls))
    monkeypatch.setattr(remote, "_on_windows", lambda: False)

    asyncio.run(gw._ensure_firewall_access())

    assert calls == []


def test_disable_flag_respected(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_on_windows", lambda: True)
    monkeypatch.setenv("ORION_REMOTE_FIREWALL", "0")
    calls: list[list[str]] = []
    monkeypatch.setattr(remote.subprocess, "run", _fake_run_factory(calls))

    gw = _gateway(tmp_path)
    asyncio.run(gw._ensure_firewall_access())

    assert calls == []


def test_loopback_serve_does_not_open_an_inbound_port(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_on_windows", lambda: True)
    calls: list[list[str]] = []
    monkeypatch.setattr(remote.subprocess, "run", _fake_run_factory(calls))
    gateway = _gateway(tmp_path)
    gateway.host = "127.0.0.1"

    asyncio.run(gateway._ensure_firewall_access())

    assert calls == []
