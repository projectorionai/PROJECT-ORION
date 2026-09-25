"""
Telephony MCP (Mark XXVI) — ORION can place real phone calls / SMS via Twilio.

Adds a disabled-by-default Twilio server to the MCP template AND the live config,
without disturbing the existing servers. Enabling it (with the user's credentials)
is what lets ORION dial directly, distinct from phone_action's tap-to-confirm
dialer hand-off.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.mcp_host import _default_config, load_config  # noqa: E402


def test_the_template_offers_a_telephony_server():
    servers = _default_config()["servers"]
    assert "twilio" in servers, "the telephony MCP is missing from the template"
    tw = servers["twilio"]
    assert tw["enabled"] is False, "telephony must ship DISABLED (real calls cost money)"
    assert tw["command"] == "npx"
    assert any("twilio" in str(a).lower() for a in tw["args"])
    for key in ("TWILIO_ACCOUNT_SID", "TWILIO_API_KEY", "TWILIO_API_SECRET"):
        assert key in tw["env"], f"{key} placeholder missing"


def test_the_description_warns_it_is_a_real_outward_action():
    desc = _default_config()["servers"]["twilio"]["description"].lower()
    assert "call" in desc and ("real" in desc or "confirm" in desc)


def test_credentials_ship_empty_never_hardcoded():
    for v in _default_config()["servers"]["twilio"]["env"].values():
        assert v == "", "telephony credentials must not be baked into source"


def test_the_live_config_has_telephony_and_keeps_the_others():
    cfg = load_config()
    servers = cfg.get("servers", {})
    assert "twilio" in servers, "telephony not added to the live config"
    # additive — the pre-existing servers must survive
    for existing in ("gmail", "google_calendar", "filesystem"):
        assert existing in servers, f"{existing} was lost when adding telephony"


def test_the_live_config_file_is_valid_json():
    # A clean public checkout deliberately has no private live config.
    from orion_core.mcp_host import MCP_CONFIG_PATH
    if MCP_CONFIG_PATH.is_file():
        data = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        data = json.loads(json.dumps(_default_config()))
    assert "twilio" in data["servers"]
    # Whether the user has switched telephony on is their business; the
    # SHIPPED default is what must be off (test above). What must hold for
    # the live file is that an enabled Twilio with no credentials is never
    # launched — it waits for them instead.
    from orion_core.mcp_host import autofill_credentials, missing_credentials
    twilio = data["servers"]["twilio"]
    if twilio.get("enabled") and not any((twilio.get("env") or {}).values()):
        assert missing_credentials(autofill_credentials("twilio", twilio))
