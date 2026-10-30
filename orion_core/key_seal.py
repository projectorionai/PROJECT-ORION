"""ORION's keys, sealed to the owner's Windows account (Mark XXXX.3).

``config/api_keys.json`` held every provider key in plain text: anyone who
copied the config folder - a backup, a synced drive, a stolen laptop's disk -
had them. Each credential in it is now sealed with Windows DPAPI
(CryptProtectData, the same protection Windows gives saved Wi-Fi and browser
passwords): only this Windows account on this computer can open it.

* Values only, in place: the file stays readable JSON, so the owner can still
  see which providers are set up and paste a NEW key in plain text - it is
  sealed the next time ORION reads the file.
* Sealed values read ``"dpapi:<base64>"``. Empty values and the local
  providers' ``"local"`` placeholder are left as they are.
* A value that cannot be opened (the file came from another account or PC)
  reads as empty: that provider is simply not configured, and the owner
  re-enters the key. It is never written back over.

ctypes, not pywin32: the frozen build must not depend on win32crypt being
bundled. Off Windows nothing is sealed and sealed values read as empty.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path
from typing import Any

PREFIX = "dpapi:"
#: Extra entropy: another program running as the owner cannot open the value
#: with a plain CryptUnprotectData call; it must also know this.
_ENTROPY = b"ORION key store v1"
_DESCRIPTION = "ORION key"
#: Field names whose string values are credentials.
SECRET_FIELDS = frozenset({
    "api_key", "api_keys", "token", "bot_token", "access_token", "refresh_token",
    "client_secret", "app_secret", "secret", "password", "auth_token",
    "webhook_url", "app_password", "lichess_bot_token", "gemini_api_key", "key",
})
#: Values that name a mode, not a credential.
_PLACEHOLDERS = frozenset({"", "local", "none", "ollama", "lm-studio", "lmstudio"})

_cache: dict[str, str] = {}
_cache_lock = threading.Lock()


# ── DPAPI ────────────────────────────────────────────────────────────────────

def available() -> bool:
    return os.name == "nt"


def _blob(data: bytes):
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buffer = ctypes.create_string_buffer(data, len(data))
    return DATA_BLOB, DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _crypt(data: bytes, *, protect: bool) -> bytes:
    import ctypes

    blob_type, source, _keep = _blob(data)
    _etype, entropy, _keep_entropy = _blob(_ENTROPY)
    out = blob_type()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    flags = 0x1                                   # CRYPTPROTECT_UI_FORBIDDEN
    if protect:
        ok = crypt32.CryptProtectData(ctypes.byref(source), _DESCRIPTION, ctypes.byref(entropy),
                                      None, None, flags, ctypes.byref(out))
    else:
        ok = crypt32.CryptUnprotectData(ctypes.byref(source), None, ctypes.byref(entropy),
                                        None, None, flags, ctypes.byref(out))
    if not ok:
        raise OSError(ctypes.GetLastError(), "DPAPI refused")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def is_sealed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def seal(value: str) -> str:
    """*value* sealed for this account; already-sealed and placeholder values
    are returned unchanged. Off Windows the value is returned as it is."""
    if not isinstance(value, str) or is_sealed(value) or value.strip().lower() in _PLACEHOLDERS:
        return value
    if not available():
        return value
    sealed = PREFIX + base64.b64encode(_crypt(value.encode("utf-8"), protect=True)).decode("ascii")
    with _cache_lock:
        _cache[sealed] = value
    return sealed


def open_value(value: Any) -> Any:
    """The plain value of a sealed one; anything else unchanged. A value this
    account cannot open reads as ""."""
    if not is_sealed(value):
        return value
    with _cache_lock:
        cached = _cache.get(value)
    if cached is not None:
        return cached
    if not available():
        return ""
    try:
        plain = _crypt(base64.b64decode(value[len(PREFIX):]), protect=False).decode("utf-8")
    except Exception:
        return ""
    with _cache_lock:
        if len(_cache) > 512:
            _cache.clear()
        _cache[value] = plain
    return plain


# ── whole documents ──────────────────────────────────────────────────────────

def _walk(node: Any, change, secret: bool = False) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, change, str(k).lower() in SECRET_FIELDS) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, change, secret) for v in node]
    if isinstance(node, str) and secret:
        return change(node)
    return node


def open_tree(data: Any) -> Any:
    """A copy of *data* with every sealed value opened."""
    return _walk(data, open_value)


def seal_tree(data: Any) -> Any:
    """A copy of *data* with every credential field sealed."""
    return _walk(data, seal)


def needs_sealing(data: Any) -> bool:
    """True when a credential field still holds a plain value."""
    found = []

    def check(value: str) -> str:
        if not is_sealed(value) and value.strip().lower() not in _PLACEHOLDERS:
            found.append(1)
        return value

    _walk(data, check)
    return bool(found) and available()


def unopenable(data: Any) -> int:
    """How many sealed values this account cannot open."""
    count = []

    def check(value: str) -> str:
        if is_sealed(value) and open_value(value) == "":
            count.append(1)
        return value

    _walk(data, check)
    return len(count)


def keep_unopenable(data: Any, raw: Any) -> Any:
    """*data* with every credential that opened to "" because this account
    cannot open it put back as the sealed value from *raw* (the file as it is
    on disk). Without this the next save wrote "" over a key sealed by another
    account - the one thing this module promises never happens."""
    if isinstance(data, dict) and isinstance(raw, dict):
        out = {}
        for key, value in data.items():
            before = raw.get(key)
            if (str(key).lower() in SECRET_FIELDS and value == "" and is_sealed(before)
                    and open_value(before) == ""):
                out[key] = before
            else:
                out[key] = keep_unopenable(value, before)
        return out
    if isinstance(data, list) and isinstance(raw, list) and len(data) == len(raw):
        return [b if (v == "" and is_sealed(b) and open_value(b) == "") else keep_unopenable(v, b)
                for v, b in zip(data, raw)]
    return data


def _raw(path: Path | str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def load_json(path: Path | str) -> dict[str, Any]:
    """A JSON key file with its sealed values opened; {} when missing or bad."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return open_tree(data) if isinstance(data, dict) else {}


def save_json(path: Path | str, data: dict[str, Any], *, indent: int = 2) -> None:
    """Write a key file with its credentials sealed."""
    from .atomic_io import atomic_write_text

    data = keep_unopenable(data, _raw(path))
    atomic_write_text(Path(path), json.dumps(seal_tree(data), indent=indent), encoding="utf-8")


def seal_file_in_place(path: Path | str, *, indent: int = 2) -> bool:
    """Seal a key file's plain credentials on disk. True when it rewrote it.

    Values that cannot be opened stay as they are (sealed by another account):
    the plain read of the file is what is re-sealed, never an opened copy."""
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("duplicate field")      # left for the reader to report
        return dict(pairs)

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict) or not needs_sealing(raw):
        return False
    from .atomic_io import atomic_write_text

    atomic_write_text(Path(path), json.dumps(seal_tree(raw), indent=indent), encoding="utf-8")
    return True


__all__ = ["PREFIX", "SECRET_FIELDS", "available", "is_sealed", "keep_unopenable", "load_json",
           "needs_sealing",
           "open_tree", "open_value", "save_json", "seal", "seal_file_in_place", "seal_tree",
           "unopenable"]
