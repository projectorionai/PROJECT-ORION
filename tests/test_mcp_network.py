"""The local network MCP reports only the configured gateway and tailnet node."""

from __future__ import annotations

import json

from orion_core import mcp_network


def test_tailnet_status_has_only_local_identity(monkeypatch):
    monkeypatch.setattr(mcp_network, "_tailscale_status", lambda: {
        "BackendState": "Running",
        "Self": {"DNSName": "orion.tail1234.ts.net.",
                 "TailscaleIPs": ["100.101.102.103", "fd7a::1"],
                 "AuthKey": "must-not-leak"},
        "Peer": {"peer": {"DNSName": "private-peer.ts.net"}},
    })
    result = mcp_network.sample("tailnet_status", {})
    assert result == {"state": "Running", "magicdns": "orion.tail1234.ts.net",
                      "ipv4": "100.101.102.103"}
    assert "must-not-leak" not in json.dumps(result)
    assert "private-peer" not in json.dumps(result)


def test_gateway_status_probes_only_configured_listener(monkeypatch):
    calls = []

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def connect(address, timeout):
        calls.append((address, timeout))
        return _Connection()

    monkeypatch.setattr(mcp_network.socket, "create_connection", connect)
    monkeypatch.setenv("ORION_REMOTE_HOST", "100.101.102.103")
    monkeypatch.setenv("ORION_REMOTE_PORT", "443")
    monkeypatch.setenv("ORION_REMOTE_TLS", "1")
    monkeypatch.setenv("ORION_REMOTE_EXPOSURE", "tailnet-direct")
    assert mcp_network.sample("phone_gateway_status", {}) == {
        "mode": "tailnet-direct", "host": "100.101.102.103", "port": 443,
        "tcp_listening": True, "tls_requested": True,
    }
    assert calls == [(('100.101.102.103', 443), 0.75)]


def test_network_mcp_rejects_arbitrary_probe_targets():
    reply = mcp_network.respond({"jsonrpc": "2.0", "id": 7,
                                 "method": "tools/call",
                                 "params": {"name": "phone_gateway_status",
                                            "arguments": {"host": "example.org"}}})
    assert reply["result"]["isError"] is True
