"""
Per-plugin secret mappings and platform-dependent encryption at rest.

The lookup API returns the keys declared for that plugin:

    "secrets": ["WEATHER_API_KEY"]

This filters the supplied mapping. It does not isolate arbitrary Python code
running inside ORION from its environment, objects or readable files. Run
only trusted plugins; separate process isolation is not provided here.

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
vault. The filtered lookup API does not prevent code running as that user
from opening those files itself.

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
from threading import RLock
from typing import Any, Iterable
from .atomic_io import atomic_write_text

_KEY_LOCK = RLock()

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


def _fernet(key_path: Path, *, create: bool = True) -> Any:
    """Read the existing key; only a sealing operation may create one."""
    from cryptography.fernet import Fernet

    with _KEY_LOCK:
        if key_path.exists():
            key = key_path.read_bytes().strip()
        else:
            if not create:
                raise VaultError(f"The decryption key {key_path.name} is missing; restore the original key.")
            key = Fernet.generate_key()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with key_path.open("xb") as stream:
                    stream.write(key)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                key = key_path.read_bytes().strip()
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
    plaintext_entries: int = 0

    @property
    def encrypted(self) -> bool:
        return self.method in {"dpapi", "fernet"} and not self.plaintext_entries


class PluginVault:
    """Secrets, one plugin at a time.

    The API filters requested keys; it is not an in-process code sandbox.
    Encryption protects stored values according to the platform/user boundary.
    """

    def __init__(self, path: Path | str | None = None,
                 key_path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.key_path = (Path(key_path) if key_path
                         else (self.path.with_name(KEY_NAME) if self.path
                               else None))
        self._data: dict[str, dict[str, str]] = {}
        self._lock = RLock()
        self._load_error = ""
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
            if method != "plaintext":
                raise VaultError("Encryption requires a vault key location")
        except Exception as exc:
            raise VaultError(f"Could not seal the secret using {method}; no plaintext fallback was used.") from exc
        return value  # Explicit no-crypto mode, reported by status().

    def _open(self, value: str) -> str:
        try:
            if value.startswith(_DPAPI_PREFIX):
                return _dpapi_open(value)
            if value.startswith(_FERNET_PREFIX) and self.key_path is not None:
                token = value[len(_FERNET_PREFIX):].encode("ascii")
                return _fernet(self.key_path, create=False).decrypt(token).decode("utf-8")
            if value.startswith(_FERNET_PREFIX):
                raise VaultError("The vault has no decryption key location")
        except Exception as exc:
            raise VaultError(
                f"a stored secret could not be decrypted ({exc}). If this "
                f"machine or Windows account has changed, the vault must be "
                f"re-entered.") from exc
        return value                      # hand-written plaintext

    # ── storage ──────────────────────────────────────────────────────────────

    def reload(self) -> None:
        with self._lock:
            self._reload_locked()

    def _reload_locked(self) -> None:
        candidate: dict[str, dict[str, str]] = {}
        try:
            if self.path is not None:
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    raw = {}
                if not isinstance(raw, dict) or not isinstance(raw.get("plugins", {}), dict):
                    raise ValueError("vault and plugin containers must be objects")
                for plugin, secrets in raw.get("plugins", {}).items():
                    if not isinstance(plugin, str) or not plugin.strip() or not isinstance(secrets, dict):
                        raise ValueError("each plugin must have a non-empty name and a secret object")
                    if any(not isinstance(key, str) or not key.strip() or not isinstance(value, str)
                           for key, value in secrets.items()):
                        raise ValueError("secret names and values must be strings")
                    candidate[plugin] = dict(secrets)
        except (OSError, ValueError, RecursionError) as exc:
            self._load_error = f"{self.path.name if self.path else 'vault'} could not be read ({exc})."
            raise VaultError(self._load_error) from exc
        self._data = candidate
        self._load_error = ""
        self._update_status()

    def _update_status(self) -> None:
        method = self._choose_method()
        plain = sum(not value.startswith((_DPAPI_PREFIX, _FERNET_PREFIX))
                    for secrets in self._data.values() for value in secrets.values())
        detail = {
            "dpapi": "DPAPI sealing available for this Windows account",
            "fernet": f"Fernet sealing available with {KEY_NAME} (owner-only where supported)",
            "plaintext": "NOT ENCRYPTED — no DPAPI and no cryptography package",
        }[method]
        if plain:
            detail += (f"; {plain} plaintext value(s) imported, " +
                       ("remain unsealed in this mode" if method == "plaintext"
                        else "sealed on the next successful save"))
        self._status = VaultStatus(method, detail, plain)

    def _writable(self) -> None:
        if self._load_error:
            raise VaultError(self._load_error + " Saved bytes are preserved; repair the file and reload before saving.")

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        self._writable()
        if self.path is None:
            raise VaultError("this vault has nowhere to save to")
        candidate = {
            plugin: {name: value if value.startswith((_DPAPI_PREFIX, _FERNET_PREFIX)) else self._seal(value)
                     for name, value in secrets.items()}
            for plugin, secrets in self._data.items()}
        payload = {
            "_comment": (
                "Per-plugin secrets. A plugin receives ONLY the keys it "
                "declares in its manifest's 'secrets' list. Values prefixed "
                "dpapi: or fernet: are sealed; plain values are read as-is "
                "and sealed on the next save."),
            "plugins": candidate,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path, json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            raise VaultError(f"{self.path.name} could not be saved; previous file retained.") from exc
        self._data = candidate
        self._update_status()
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
        with self._lock:
            held = dict(self._data.get(str(plugin), {}))
        out: dict[str, str] = {}
        for name in wanted:
            if name in held:
                out[name] = self._open(held[name])
        return out

    def set_secret(self, plugin: str, name: str, value: str) -> None:
        with self._lock:
            self._writable()
            sealed = self._seal(str(value))
            self._data.setdefault(str(plugin), {})[str(name)] = sealed
            self._update_status()

    def forget(self, plugin: str, name: str = "") -> int:
        """Remove one secret, or every secret for a plugin. Returns how many."""
        with self._lock:
            self._writable()
            return self._forget_locked(plugin, name)

    def _forget_locked(self, plugin: str, name: str) -> int:
        held = self._data.get(str(plugin))
        if held is None:
            return 0
        if not name:
            count = len(held)
            del self._data[str(plugin)]
            self._update_status()
            return count
        count = 1 if held.pop(str(name), None) is not None else 0
        self._update_status()
        return count

    def missing(self, plugin: str, declared: Iterable[str]) -> list[str]:
        """Declared secrets the vault does not hold.

        Reported at load rather than discovered at three in the morning when a
        scheduled plugin fires and fails on a missing key.
        """
        with self._lock:
            held = set(self._data.get(str(plugin), {}))
        return [name for name in (declared or ()) if str(name) not in held]

    def plugins(self) -> list[str]:
        with self._lock:
            return sorted(self._data)


__all__ = ["KEY_NAME", "VAULT_NAME", "PluginVault", "VaultError", "VaultStatus"]
