"""
Tests for reachable-endpoint discovery (#12 — frictionless connect).

The detection seams (Tailscale status, LAN IPs) are injected, so these run with
no real network, sockets, or the tailscale CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.remote_endpoints import (
    detect_endpoints,
    parse_tailscale_self,
    tailscale_exe,
    tailscale_status_json,
)


def test_parse_tailscale_self_running():
    status = {
        "BackendState": "Running",
        "Self": {"DNSName": "orion-pc.tail1234.ts.net.",
                 "TailscaleIPs": ["100.101.102.103", "fd7a::1"]},
    }
    got = parse_tailscale_self(status)
    assert got["dns"] == "orion-pc.tail1234.ts.net"      # trailing dot stripped
    assert got["ips"] == ["100.101.102.103", "fd7a::1"]


def test_parse_tailscale_self_stopped_yields_nothing():
    stopped = {"BackendState": "Stopped",
               "Self": {"DNSName": "x.ts.net", "TailscaleIPs": ["100.1.1.1"]}}
    assert parse_tailscale_self(stopped) == {"dns": "", "ips": []}
    assert parse_tailscale_self({}) == {"dns": "", "ips": []}


def test_detect_orders_tailscale_first_then_lan():
    eps = detect_endpoints(
        8765, https=True,
        tailscale={"dns": "orion-pc.ts.net", "ips": ["100.1.2.3"]},
        lan=["192.168.1.20", "10.0.0.5"],
    )
    kinds = [e["kind"] for e in eps]
    assert kinds == ["tailscale", "tailscale-ip", "lan", "lan"]
    assert eps[0]["url"] == "https://orion-pc.ts.net:8765/"
    assert eps[0]["priority"] <= eps[-1]["priority"]     # sorted by priority


def test_detect_http_scheme_and_dedup():
    eps = detect_endpoints(
        80, https=False,
        tailscale={"dns": "", "ips": []},
        lan=["192.168.0.2", "192.168.0.2"],   # duplicate collapses
    )
    assert len(eps) == 1
    assert eps[0]["url"] == "http://192.168.0.2:80/"
    assert eps[0]["kind"] == "lan"


def test_detect_empty_when_nothing_reachable():
    assert detect_endpoints(8765, tailscale={"dns": "", "ips": []}, lan=[]) == []


def test_tailscale_status_json_absent_cli_is_empty(monkeypatch):
    import orion_core.remote_endpoints as mod
    monkeypatch.setattr(mod, "tailscale_exe", lambda: None)
    assert tailscale_status_json() == {}


def test_tailscale_status_json_parses_runner_output(monkeypatch):
    import orion_core.remote_endpoints as mod
    monkeypatch.setattr(mod, "tailscale_exe", lambda: "/usr/bin/tailscale")

    class _Proc:
        stdout = '{"BackendState": "Running", "Self": {"DNSName": "n.ts.net."}}'

    got = tailscale_status_json(runner=lambda *a, **k: _Proc())
    assert got["BackendState"] == "Running"


def test_tailscale_exe_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "tailscale"
    fake.write_text("x")
    monkeypatch.setenv("ORION_TAILSCALE_PATH", str(fake))
    assert tailscale_exe() == str(fake)
