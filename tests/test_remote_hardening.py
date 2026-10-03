"""
Remote gateway hardening (audit 2026-10-03).

* The HMAC key that signs every access token, and the uplink's TLS private
  key, were written with the process umask — world-readable (0644) on a
  headless Linux node, beside 0600 siblings. Whoever can read the key can mint
  an access token for any paired device.
* Rate limiting keyed on the socket peer put every phone behind the deploy/
  Caddy proxy (or Tailscale serve) into ONE bucket: ten bad pairing attempts
  from anywhere locked the owner out of token refresh for a minute.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.remote import RemoteAuthManager, RemoteGateway  # noqa: E402

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


class _Signal:
    def emit(self, *payload):
        pass


class _Bus:
    def __getattr__(self, name):
        return _Signal()


class _Request:
    def __init__(self, remote: str, forwarded: str = ""):
        self.remote = remote
        self.headers = {"X-Forwarded-For": forwarded} if forwarded else {}


def _gateway(tmp_path) -> RemoteGateway:
    return RemoteGateway(None, None, _Bus(), config_dir=tmp_path)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@posix_only
def test_a_new_token_secret_is_owner_only(tmp_path):
    RemoteAuthManager(config_dir=tmp_path)
    assert _mode(tmp_path / "remote_secret.key") == 0o600


@posix_only
def test_an_existing_world_readable_secret_is_tightened_and_kept(tmp_path):
    secret = tmp_path / "remote_secret.key"
    secret.write_text("cd" * 32, encoding="utf-8")
    secret.chmod(0o644)
    auth = RemoteAuthManager(config_dir=tmp_path)
    assert _mode(secret) == 0o600
    # Tightening must not rotate the key: paired phones keep working.
    assert auth._secret == bytes.fromhex("cd" * 32)


@posix_only
def test_the_tls_private_key_is_owner_only(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    monkeypatch.setenv("ORION_REMOTE_TLS", "1")
    gateway = _gateway(tmp_path)
    assert gateway._maybe_ssl_context() is not None
    assert _mode(tmp_path / "remote_tls" / "key.pem") == 0o600


def test_a_direct_client_cannot_choose_its_rate_limit_bucket(tmp_path):
    gateway = _gateway(tmp_path)
    spoofed = _Request("192.168.1.50", forwarded="10.9.9.9")
    assert gateway._client_key(spoofed) == "192.168.1.50"


def test_clients_behind_a_local_proxy_get_their_own_buckets(tmp_path):
    gateway = _gateway(tmp_path)
    phone = _Request("127.0.0.1", forwarded="203.0.113.7")
    attacker = _Request("127.0.0.1", forwarded="198.51.100.66")
    assert gateway._client_key(phone) == "203.0.113.7"
    assert gateway._client_key(attacker) == "198.51.100.66"


def test_the_proxys_own_hop_wins_over_a_client_supplied_chain(tmp_path):
    gateway = _gateway(tmp_path)
    # The client prepended a fake address; the proxy appended the real one.
    chained = _Request("::1", forwarded="10.0.0.1, 203.0.113.7")
    assert gateway._client_key(chained) == "203.0.113.7"


def test_a_listed_remote_proxy_is_trusted(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_TRUSTED_PROXIES", "10.0.0.2")
    gateway = _gateway(tmp_path)
    assert gateway._client_key(_Request("10.0.0.2", forwarded="203.0.113.9")) == "203.0.113.9"
    assert gateway._client_key(_Request("10.0.0.3", forwarded="203.0.113.9")) == "10.0.0.3"


def test_a_bad_pairing_storm_through_the_proxy_does_not_lock_out_the_owner(tmp_path):
    gateway = _gateway(tmp_path)
    attacker = gateway._client_key(_Request("127.0.0.1", forwarded="198.51.100.66"))
    for _ in range(12):
        gateway._auth_rate_ok(attacker)
    assert not gateway._auth_rate_ok(attacker)
    owner = gateway._client_key(_Request("127.0.0.1", forwarded="203.0.113.7"))
    assert gateway._auth_rate_ok(owner)
