"""A cloud reverse proxy advertises only its named HTTPS origin."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from orion_core.remote import RemoteGateway
from orion_core.remote_endpoints import public_https_endpoint


def test_public_https_endpoint_is_phone_ready():
    endpoint = public_https_endpoint("https://Orion-Home.duckdns.org/")
    assert endpoint == {
        "kind": "reverse-proxy", "host": "orion-home.duckdns.org",
        "port": 443, "https": True,
        "url": "https://orion-home.duckdns.org:443/", "priority": 10,
    }


@pytest.mark.parametrize("url", [
    "http://orion.duckdns.org", "https://orion.duckdns.org:8765",
    "https://user:password@orion.duckdns.org", "https://orion.duckdns.org/private",
    "https://orion.duckdns.org?token=abc", "https://127.0.0.1",
    "https://orion..duckdns.org", "https://orion.duckdns.org#fragment",
])
def test_public_https_endpoint_rejects_unsafe_origins(url):
    assert public_https_endpoint(url) is None


def test_reverse_proxy_never_advertises_lan_or_tailnet(monkeypatch):
    monkeypatch.setenv("ORION_REMOTE_EXPOSURE", "reverse-proxy")
    gateway = RemoteGateway.__new__(RemoteGateway)
    gateway._reverse_proxy_mode = True
    gateway._public_endpoint = public_https_endpoint("https://orion.duckdns.org")
    assert asyncio.run(gateway._endpoints()) == [gateway._public_endpoint]


def test_reverse_proxy_gateway_reports_external_https(monkeypatch, tmp_path):
    monkeypatch.setenv("ORION_REMOTE_EXPOSURE", "reverse-proxy")
    monkeypatch.setenv("ORION_REMOTE_PUBLIC_URL", "https://orion.duckdns.org")
    monkeypatch.setenv("ORION_REMOTE_HOST", "127.0.0.1")
    monkeypatch.setenv("ORION_REMOTE_PORT", "8765")
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda _message: None))
    gateway = RemoteGateway(None, None, bus, config_dir=tmp_path)
    assert gateway.port == 8765
    assert gateway._serve_mode is False
    assert gateway._public_https is True
    assert asyncio.run(gateway._endpoints())[0]["url"] == "https://orion.duckdns.org:443/"


def test_reverse_proxy_refuses_non_loopback_bind(monkeypatch):
    monkeypatch.setenv("ORION_REMOTE_EXPOSURE", "reverse-proxy")
    monkeypatch.setenv("ORION_REMOTE_PUBLIC_URL", "https://orion.duckdns.org")
    monkeypatch.setenv("ORION_REMOTE_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        RemoteGateway(None, None, None)
