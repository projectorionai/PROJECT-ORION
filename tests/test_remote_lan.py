"""
The home Wi-Fi link (remote_lan.py): the companion app reaches the PC on the
LAN with no Tailscale, and security is no weaker than the Tailscale path.

The real Windows Firewall and the real LAN address are never touched: netsh is
faked, and the listener binds loopback with the private-peer rule widened for
that one socket where a test says so.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import ssl
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.remote as remote
import orion_core.remote_lan as remote_lan
from orion_core.remote import RemoteGateway
from orion_core.remote_lan import (FIREWALL_RULE, LanLink, display_code,
                                   firewall_rule_args, is_private_peer,
                                   lan_endpoint, lan_port, lan_wanted)


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)

    def connect(self, handler):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Memory:
    def log_episode(self, role, content):
        pass

    def prompt_context(self, limit=18):
        return ""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# ── policy ────────────────────────────────────────────────────────────────────

def test_lan_link_is_on_for_serve_and_tailnet_but_never_for_proxies(monkeypatch):
    monkeypatch.setenv("ORION_REMOTE_LAN", "1")
    monkeypatch.delenv("ORION_REMOTE_LAN_PORT", raising=False)
    assert lan_wanted("127.0.0.1", "", 8000)              # Tailscale Serve loopback
    assert lan_wanted("127.0.0.1", "quick-tunnel", 8000)
    assert lan_wanted("100.101.2.3", "tailnet-direct", 443)
    assert not lan_wanted("127.0.0.1", "reverse-proxy", 8000)
    assert not lan_wanted("0.0.0.0", "platform-proxy", 8000)
    # The legacy direct bind is already on the LAN: no second listener.
    assert not lan_wanted("0.0.0.0", "", 8765)
    assert not lan_wanted("192.168.1.20", "", 8000)
    # The same port as the main listener cannot be bound twice.
    assert not lan_wanted("127.0.0.1", "", 8765)
    monkeypatch.setenv("ORION_REMOTE_LAN", "0")
    assert not lan_wanted("127.0.0.1", "", 8000)


def test_lan_port_is_validated(monkeypatch):
    monkeypatch.delenv("ORION_REMOTE_LAN_PORT", raising=False)
    assert lan_port() == 8765
    monkeypatch.setenv("ORION_REMOTE_LAN_PORT", "9443")
    assert lan_port() == 9443
    for bad in ("80", "70000", "x"):
        monkeypatch.setenv("ORION_REMOTE_LAN_PORT", bad)
        with pytest.raises(ValueError):
            lan_port()


@pytest.mark.parametrize("address,private", [
    ("192.168.1.20", True), ("10.0.0.7", True), ("172.16.4.1", True),
    ("172.31.255.1", True), ("169.254.3.3", True), ("::ffff:192.168.0.9", True),
    ("fd12:3456::1", True), ("fe80::1%12", True),
    ("172.32.0.1", False), ("100.64.0.1", False), ("100.100.100.100", False),
    ("8.8.8.8", False), ("127.0.0.1", False), ("::1", False),
    ("2001:db8::1", False), ("", False), ("not-an-ip", False), (None, False),
])
def test_only_local_network_peers_count_as_private(address, private):
    assert is_private_peer(address) is private


def test_certificate_code_and_endpoint_carry_the_exact_pin():
    sha = hashlib.sha256(b"certificate").hexdigest()
    assert display_code(sha) == "-".join(sha.upper()[i:i + 4] for i in range(0, 16, 4))
    endpoint = lan_endpoint("192.168.1.20", 8765, sha.upper())
    assert endpoint == {"kind": "lan", "host": "192.168.1.20", "port": 8765, "https": True,
                        "url": "https://192.168.1.20:8765/", "priority": 5,
                        "cert_sha256": sha}


# ── Windows Firewall: Private profile only, local subnet only ─────────────────

def test_firewall_rule_is_private_profile_and_local_subnet_only():
    args = firewall_rule_args(8765)
    assert "profile=private" in args
    assert "remoteip=localsubnet" in args
    assert "localport=8765" in args and "dir=in" in args and "protocol=TCP" in args
    assert not any("public" in arg.lower() and arg.startswith("profile") for arg in args)
    assert not any(arg.startswith("profile=") and arg != "profile=private" for arg in args)


class _Run:
    def __init__(self, show_stdout="", show_rc=0, add_rc=0):
        self.calls: list[list[str]] = []
        self.show_stdout, self.show_rc, self.add_rc = show_stdout, show_rc, add_rc

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        result = type("R", (), {})()
        if "show" in args:
            result.returncode, result.stdout = self.show_rc, self.show_stdout
        else:
            result.returncode, result.stdout = self.add_rc, ""
        result.stderr = ""
        return result


def _link(tmp_path, *, windows=True):
    gateway = RemoteGateway(None, _Memory(), _Bus(), config_dir=tmp_path)
    link = LanLink(gateway, lan_ip=lambda: "192.168.1.20", on_windows=lambda: windows)
    link.port = 8765
    return gateway, link


def test_firewall_rule_added_once_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REMOTE_FIREWALL", raising=False)
    run = _Run(show_stdout="No rules match the specified criteria.", show_rc=1)
    monkeypatch.setattr(remote_lan.subprocess, "run", run)
    _gateway, link = _link(tmp_path)
    asyncio.run(link.ensure_firewall())
    adds = [call for call in run.calls if "add" in call]
    assert adds == [firewall_rule_args(8765)]


def test_existing_private_rule_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REMOTE_FIREWALL", raising=False)
    run = _Run(show_stdout=f"Rule Name: {FIREWALL_RULE}\nProfiles: Private\nEnabled: Yes")
    monkeypatch.setattr(remote_lan.subprocess, "run", run)
    gateway, link = _link(tmp_path)
    asyncio.run(link.ensure_firewall())
    assert not any("add" in call for call in run.calls)
    assert not any("Public" in str(line) for line in gateway.bus.log.emitted)


def test_a_rule_widened_to_public_networks_is_reported_not_trusted(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REMOTE_FIREWALL", raising=False)
    run = _Run(show_stdout=f"Rule Name: {FIREWALL_RULE}\nProfiles: Private,Public\n")
    monkeypatch.setattr(remote_lan.subprocess, "run", run)
    gateway, link = _link(tmp_path)
    asyncio.run(link.ensure_firewall())
    assert not any("add" in call for call in run.calls)
    assert any("Public" in str(line) for line in gateway.bus.log.emitted)


def test_firewall_untouched_off_windows_or_when_disabled(tmp_path, monkeypatch):
    run = _Run()
    monkeypatch.setattr(remote_lan.subprocess, "run", run)
    _gateway, link = _link(tmp_path, windows=False)
    asyncio.run(link.ensure_firewall())
    monkeypatch.setenv("ORION_REMOTE_FIREWALL", "0")
    _gateway, link = _link(tmp_path, windows=True)
    asyncio.run(link.ensure_firewall())
    assert run.calls == []


def test_refused_rule_logs_the_exact_private_only_command(tmp_path, monkeypatch):
    monkeypatch.delenv("ORION_REMOTE_FIREWALL", raising=False)
    run = _Run(show_rc=1, add_rc=1)
    monkeypatch.setattr(remote_lan.subprocess, "run", run)
    gateway, link = _link(tmp_path)
    asyncio.run(link.ensure_firewall())
    text = " ".join(str(line) for line in gateway.bus.log.emitted)
    assert "profile=private remoteip=localsubnet" in text
    assert "profile=public" not in text.lower()


# ── the listener itself: TLS with ORION's certificate, pin in the endpoints ───

def _lan_env(monkeypatch, lan_port_value):
    monkeypatch.setenv("ORION_REMOTE_LAN", "1")
    monkeypatch.setenv("ORION_REMOTE_LAN_PORT", str(lan_port_value))
    monkeypatch.setenv("ORION_REMOTE_FIREWALL", "0")
    monkeypatch.delenv("ORION_REMOTE_TLS", raising=False)
    monkeypatch.delenv("ORION_REMOTE_EXPOSURE", raising=False)
    # Never ask Tailscale or the real interfaces during a test.
    monkeypatch.setattr(remote, "detect_endpoints", lambda *a, **k: [])


def _started_gateway(tmp_path, monkeypatch, *, accept_loopback: bool):
    lan_port_value = _free_port()
    _lan_env(monkeypatch, lan_port_value)
    gateway = RemoteGateway(None, _Memory(), _Bus(), config_dir=tmp_path)
    gateway.host, gateway.port = "127.0.0.1", _free_port()
    # Stand loopback in for the PC's LAN address. The bind check and the
    # request guard both use is_private_peer; widen it ONLY for 127.0.0.1
    # when the test says so, so the refusal path stays the real rule.
    real = remote_lan.is_private_peer
    monkeypatch.setattr(remote_lan, "is_private_peer",
                        lambda value: (accept_loopback and str(value) == "127.0.0.1")
                        or real(value))
    gateway.lan._lan_ip = lambda: "127.0.0.1"
    return gateway, lan_port_value


def _client_context():
    """Accept the self-signed certificate so the test can READ it - the
    phone instead accepts only the pinned fingerprint."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def test_lan_listener_serves_tls_and_endpoints_pin_the_certificate_it_presents(
        tmp_path, monkeypatch):
    import aiohttp

    async def scenario():
        gateway, lan_port_value = _started_gateway(tmp_path, monkeypatch, accept_loopback=True)
        await gateway.start()
        try:
            assert gateway.lan.active
            # The certificate the LAN socket actually presents...
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", lan_port_value, ssl=_client_context())
            der = writer.get_extra_info("ssl_object").getpeercert(binary_form=True)
            writer.close()
            presented = hashlib.sha256(der).hexdigest()
            # ...is exactly the pin a paired phone learns with its token.
            code = gateway.auth.begin_pairing()
            async with aiohttp.ClientSession() as session:
                url = f"https://127.0.0.1:{lan_port_value}"
                pair = await session.post(f"{url}/v1/auth/pair", ssl=_client_context(),
                                          json={"code": code, "device_name": "phone"})
                assert pair.status == 200
                paired = await pair.json()
                token = await session.post(f"{url}/v1/auth/token", ssl=_client_context(),
                                           json={"device_id": paired["device_id"],
                                                 "refresh_token": paired["refresh_token"]})
                body = await token.json()
                # Plain HTTP on the LAN port is refused: TLS or nothing.
                with pytest.raises(aiohttp.ClientError):
                    await session.get(f"http://127.0.0.1:{lan_port_value}/api/health")
            for endpoints in (paired["endpoints"], body["endpoints"]):
                first = endpoints[0]
                assert first["kind"] == "lan" and first["https"] is True
                assert first["cert_sha256"] == presented
                assert first["url"] == f"https://127.0.0.1:{lan_port_value}/"
            assert body["face"]["type"] == "face" and body["face"]["design"]
            assert display_code(presented) in gateway.lan_pairing_hint()
        finally:
            await asyncio.wait_for(gateway.stop(), timeout=10)
        assert not gateway.lan.active
    asyncio.run(scenario())


def test_lan_listener_refuses_non_private_peers_but_main_uplink_is_unchanged(
        tmp_path, monkeypatch):
    import aiohttp

    async def scenario():
        # Loopback is NOT private: the LAN socket must refuse it with 403,
        # exactly as it would a public address that got through a firewall.
        gateway, lan_port_value = _started_gateway(tmp_path, monkeypatch, accept_loopback=False)
        # Bind anyway (the bind check is not the guard under test).
        gateway.lan._bindable = lambda ip: ip == "127.0.0.1"
        await gateway.start()
        try:
            assert gateway.lan.active
            async with aiohttp.ClientSession() as session:
                lan = await session.get(f"https://127.0.0.1:{lan_port_value}/api/health",
                                        ssl=_client_context())
                assert lan.status == 403
                main = await session.get(f"http://127.0.0.1:{gateway.port}/api/health")
                assert main.status == 200
        finally:
            await asyncio.wait_for(gateway.stop(), timeout=10)
    asyncio.run(scenario())


def test_lan_link_stays_off_without_a_private_address(tmp_path, monkeypatch):
    async def scenario():
        gateway, _port = _started_gateway(tmp_path, monkeypatch, accept_loopback=False)
        gateway.lan._lan_ip = lambda: "127.0.0.1"         # no LAN: loopback only
        await gateway.start()
        try:
            assert not gateway.lan.active
            assert gateway.lan.endpoint() is None
            assert gateway.lan_pairing_hint() == ""
            assert all(e.get("kind") != "lan" for e in await gateway._endpoints())
        finally:
            await asyncio.wait_for(gateway.stop(), timeout=10)
    asyncio.run(scenario())


def test_events_accept_a_bearer_header_so_the_token_stays_out_of_the_url(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    async def scenario():
        gateway = RemoteGateway(None, _Memory(), _Bus(), config_dir=tmp_path)
        code = gateway.auth.begin_pairing()
        device_id, refresh = gateway.auth.complete_pairing(code, "phone")
        token, _ = gateway.auth.issue_access(device_id, refresh)
        client = TestClient(TestServer(gateway._build_app()))
        await client.start_server()
        try:
            res = await client.get("/api/events", headers={"Authorization": "Bearer nope"})
            assert res.status == 401
            res = await client.get("/api/events", headers={"Authorization": f"Bearer {token}"})
            assert res.status == 200
            line = await asyncio.wait_for(res.content.readline(), timeout=5)
            assert json.loads(line.decode().removeprefix("data: "))["type"] == "state"
            res.close()
        finally:
            await client.close()
    asyncio.run(scenario())


# ── review, 29 Oct 2026 ─────────────────────────────────────────────────────


def test_the_home_address_comes_from_a_real_adapter_not_a_vpn_or_wsl():
    from orion_core.remote_lan import home_lan_ip
    adapters = {"vEthernet (WSL)": ["172.20.48.1"], "NordLynx VPN": ["10.5.0.2"],
                "Wi-Fi": ["192.168.1.40"], "Ethernet 2": ["192.168.56.1"]}
    up = {"vEthernet (WSL)", "NordLynx VPN", "Wi-Fi"}
    pick = lambda order: home_lan_ip(lambda: order, lambda: adapters, lambda: up)
    assert pick(["10.5.0.2", "172.20.48.1", "192.168.1.40"]) == "192.168.1.40"
    assert pick(["172.20.48.1", "192.168.56.1"]) == ""          # Wi-Fi gone, Ethernet 2 down
    assert pick([]) == ""


def test_a_failed_bind_unregisters_its_site_and_logs_once():
    import asyncio as _asyncio
    import types as _types
    from orion_core.remote_lan import LanLink

    logs, stopped = [], []

    class _Site:
        def __init__(self, *a, **k):
            pass

        async def start(self):
            raise OSError("address in use")

        async def stop(self):
            stopped.append(1)

    gateway = _types.SimpleNamespace(bus=_types.SimpleNamespace(log=_types.SimpleNamespace(emit=logs.append)),
                                     _endpoints_cache=None)
    link = LanLink(gateway, lan_ip=lambda: "192.168.1.40")
    link.port = 8765
    web = _types.SimpleNamespace(TCPSite=_Site)

    async def twice():
        assert await link._bind(object(), web) is False
        assert await link._bind(object(), web) is False
    _asyncio.run(twice())
    assert len(stopped) == 2 and sum("could not listen" in line for line in logs) == 1


def test_the_firewall_port_is_read_as_a_number_in_any_language():
    from orion_core.remote_lan import _rule_has_port
    assert _rule_has_port("Lokaler Port:                         8765\n", 8765)
    assert not _rule_has_port("LocalPort:                            8765\n", 9001)
