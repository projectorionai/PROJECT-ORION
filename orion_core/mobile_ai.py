"""Explicitly approved Gemini provisioning for a paired standalone phone.

No credential is available until the host owner authorises that device in the
private provisioning file. The credential is returned only to authenticated
native HTTPS requests and must be sealed by Android Keystore before use.
"""
from __future__ import annotations

import json
from pathlib import Path


def approved(config_dir: Path, device_id: str) -> bool:
    try:
        source = config_dir / "mobile_ai_permissions.json"
        if source.stat().st_size > 16384:
            return False
        policy = json.loads(source.read_text(encoding="utf-8"))
        return (isinstance(policy, dict) and policy.get("version") == 1
                and isinstance(policy.get("devices"), list) and device_id in policy["devices"])
    except (OSError, ValueError):
        return False


def profile(router) -> dict:
    providers = getattr(getattr(router, "settings", None), "providers", {})
    provider = providers.get("gemini")
    if provider is None or not getattr(provider, "enabled", False):
        raise ValueError("Enable the Gemini provider on the PC to provision this phone.")
    active = getattr(router, "active_key", None)
    key = active(provider) if callable(active) else getattr(provider, "api_key", "")
    if not isinstance(key, str) or not key:
        raise ValueError("The PC has no Gemini credential to provision.")
    return {"ok": True, "api_key": key, "model": "gemini-3.5-flash-lite",
            "voice_model": "gemini-3.8-flash-tts", "voice": "Charon",
            "provider": "gemini", "requires_encrypted_storage": True}
