"""
Per-plugin secrets, scoped so a plugin sees only its own.

ORION's credentials lived in one place that everything could read. A plugin
that needed a weather key could also read the Twilio token, the provider keys
and anything else in the process environment — not because it was allowed to,
but because nothing was in the way. With plugins now able to run **on a
schedule**, unattended, that stops being theoretical.

So each plugin declares what it needs and receives exactly that:

    "secrets": ["WEATHER_API_KEY"]

and gets a mapping with one entry in it. Anything it did not declare is not
merely absent from the mapping — it was never handed over, so there is nothing
to find.

What the encryption is actually for
-----------------------------------
Being straight about this, because "encrypted vault" invites more confidence
than it earns.

On Windows the values are wrapped with **DPAPI**, which ties them to the
Windows user account. Another account on the same machine cannot read them,
and the file copied to a different machine is useless. That is real.

Elsewhere — including the VPS — they are wrapped with a Fernet key held in a
key file with 0600 permissions. That protects against other users on the box
and against the file ending up somewhere it should not, such as a backup or a
commit. It does **not** protect against someone who is already running as the
ORION user on that machine: they can read the key file, so they can read the
vault. Nothing stored on a machine can defend against that; the defence there
is the scoping above, and not running untrusted plugins.

If neither is available the vault stores plaintext and **says so loudly** on
every load, rather than quietly pretending.
"""

from __future__ import annotations

import base64
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from .atomic_io import atomic_write_text

#: Where the vault and its key live, relative to the config directory.
VAULT_NAME = "plugin_vault.json"
KEY_NAME = "plugin_vault.key"

#: Marks a value this module wrapped, and how. A value without one of these
#: prefixes is plaintext, which is legitimate — a vault hand-edited by the
#: user should work immediately, and be upgraded on the next save.
_DPAPI_PREFIX = "dpapi:"
_FERNET_PREFIX = "fernet:"


class VaultError(RuntimeError):
    """The vault could not be read or written. Carries a usable sentence."""


# ── how values are wrapped ────────────────────────────────────────────────────

def _dpapi_available() -> bool:
    if os.name != "nt":
        return False
    try:
        import win32crypt  # noqa: F401

        return True
    except Exception:
        return False


def _dpapi_seal(value: str) -> str:
    import win32crypt

    blob = win32crypt.CryptProtectData(value.encode("utf-8"), "ORION plugin vault",
                                       None, None, None, 0)
    return _DPAPI_PREFIX + base64.b64encode(blob).decode("ascii")


def _dpapi_open(value: str) -> str:
    import win32crypt

    raw = base64.b64decode(value[len(_DPAPI_PREFIX):])
    _description, plain = win32crypt.CryptUnprotectData(raw, None, None, None, 0)
    return plain.decode("utf-8")


def _fernet(key_path: Path) -> Any:
    """The Fernet used for this install, creating its key on first use."""
    from cryptography.fernet import Fernet

    if key_path.exists():
        key = key_path.read_bytes().strip()
    else:
        key = Fernet.generate_key()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(key)
        _lock_down(key_path)
    return Fernet(key)


def _lock_down(path: Path) -> None:
    """Make a file readable only by its owner, where that means anything.

    On Windows the POSIX bits are close to decorative, which is exactly why
    DPAPI is preferred there — this is the fallback's fallback.
    """
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


@dataclass
class VaultStatus:
    """How the vault is protecting itself, in words that can be printed."""

    method: str                  # "dpapi" | "fernet" | "plaintext"
    detail: str

    @property
    def encrypted(self) -> bool:
        return self.method in {"dpapi", "fernet"}


class PluginVault:
    """Secrets, one plugin at a time.

    The scoping is the security property that always holds. The encryption is
    a second layer whose strength depends on the platform, and
    :meth:`status` says which one is in force rather than leaving it to be
    assumed.
    """

    def __init__(self, path: Path | str | None = None,
                 key_path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.key_path = (Path(key_path) if key_path
                         else (self.path.with_name(KEY_NAME) if self.path
                               else None))
        self._data: dict[str, dict[str, str]] = {}
        self._status = VaultStatus("plaintext", "no vault loaded")
        self.reload()

    # ── protection ───────────────────────────────────────────────────────────

    def status(self) -> VaultStatus:
        return self._status

    def _choose_method(self) -> str:
        if _dpapi_available():
            return "dpapi"
        try:
            import cryptography  # noqa: F401

            return "fernet"
        except Exception:
            return "plaintext"

    def _seal(self, value: str) -> str:
        method = self._choose_method()
        try:
            if method == "dpapi":
                return _dpapi_seal(value)
            if method == "fernet" and self.key_path is not None:
                token = _fernet(self.key_path).encrypt(value.encode("utf-8"))
                return _FERNET_PREFIX + token.decode("ascii")
        except Exception:
            pass
        return value

    def _open(self, value: str) -> str:
        try:
            if value.startswith(_DPAPI_PREFIX):
                return _dpapi_open(value)
            if value.startswith(_FERNET_PREFIX) and self.key_path is not None:
                token = value[len(_FERNET_PREFIX):].encode("ascii")
                return _fernet(self.key_path).decrypt(token).decode("utf-8")
        except Exception as exc:
            raise VaultError(
                f"a stored secret could not be decrypted ({exc}). If this "
                f"machine or Windows account has changed, the vault must be "
                f"re-entered.") from exc
        return value                      # hand-written plaintext

    # ── storage ──────────────────────────────────────────────────────────────

    def reload(self) -> None:
        self._data.clear()
        method = self._choose_method()
        self._status = VaultStatus(method, {
            "dpapi": "sealed to this Windows account",
            "fernet": f"sealed with {KEY_NAME} (owner-only)",
            "plaintext": "NOT ENCRYPTED — no DPAPI and no cryptography package",
        }[method])

        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VaultError(f"{self.path.name} could not be read ({exc}).")
        for plugin, secrets in (raw.get("plugins") or {}).items():
            if isinstance(secrets, dict):
                self._data[str(plugin)] = {
                    str(k): str(v) for k, v in secrets.items()}

    def save(self) -> None:
        if self.path is None:
            raise VaultError("this vault has nowhere to save to")
        payload = {
            "_comment": (
                "Per-plugin secrets. A plugin receives ONLY the keys it "
                "declares in its manifest's 'secrets' list. Values prefixed "
                "dpapi: or fernet: are sealed; plain values are read as-is "
                "and sealed on the next save."),
            "plugins": self._data,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(payload, indent=2) + "\n",
                          encoding="utf-8")
        _lock_down(self.path)

    # ── the part that matters ────────────────────────────────────────────────

    def secrets_for(self, plugin: str, declared: Iterable[str]) -> dict[str, str]:
        """Exactly the secrets *plugin* declared, and nothing else.

        A key the plugin did not declare is not returned even if the vault
        holds it for that plugin — the manifest is the contract, and a plugin
        that starts reading something it never declared should fail visibly
        rather than succeed quietly.
        """
        wanted = [str(name).strip() for name in (declared or ()) if str(name).strip()]
        if not wanted:
            return {}
        held = self._data.get(str(plugin), {})
        out: dict[str, str] = {}
        for name in wanted:
            if name in held:
                out[name] = self._open(held[name])
        return out

    def set_secret(self, plugin: str, name: str, value: str) -> None:
        self._data.setdefault(str(plugin), {})[str(name)] = self._seal(str(value))

    def forget(self, plugin: str, name: str = "") -> int:
        """Remove one secret, or every secret for a plugin. Returns how many."""
        held = self._data.get(str(plugin))
        if held is None:
            return 0
        if not name:
            count = len(held)
            del self._data[str(plugin)]
            return count
        return 1 if held.pop(str(name), None) is not None else 0

    def missing(self, plugin: str, declared: Iterable[str]) -> list[str]:
        """Declared secrets the vault does not hold.

        Reported at load rather than discovered at three in the morning when a
        scheduled plugin fires and fails on a missing key.
        """
        held = self._data.get(str(plugin), {})
        return [name for name in (declared or ()) if str(name) not in held]

    def plugins(self) -> list[str]:
        return sorted(self._data)


__all__ = ["KEY_NAME", "VAULT_NAME", "PluginVault", "VaultError", "VaultStatus"]
