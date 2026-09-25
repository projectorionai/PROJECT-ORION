"""Fire an IFTTT applet, so ORION can reach things nobody wrote a plugin for.

IFTTT's Webhooks service turns one HTTP request into whatever applet you have
built on the other side — a smart plug, a spreadsheet row, a car preconditioner,
anything with an IFTTT integration. That makes it the useful catch-all: the
long tail of devices ORION will never have a dedicated tool for.

The honest limits, because they decide whether this is worth setting up:

* The free tier allows a small number of applets. Enough for a few, not for
  a house.
* It is fire-and-forget. IFTTT acknowledges the request, not the outcome, so
  "sent" here means the webhook was accepted — not that the light came on.
  A tool that claimed otherwise would be lying, so this says "asked" rather
  than "done".
* Your webhook key is a bearer token for every applet you own. ``config/``
  only.
"""

from __future__ import annotations

from orion_core.data import ToolResult

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TIMEOUT = 15
CONFIG_NAME = "ifttt.json"
ENDPOINT = "https://maker.ifttt.com/trigger/{event}/with/key/{key}"


def _config() -> dict:
    data: dict = {}
    try:
        path = Path(__file__).resolve().parent.parent / CONFIG_NAME
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
    except Exception:
        data = {}
    key = os.getenv("IFTTT_KEY", "").strip()
    if key:
        data["key"] = key
    return data


def _run_impl(**kwargs) -> ToolResult:
    """Execute the plugin. Returns a short line for ORION to speak."""
    action = str(kwargs.get("action") or "trigger").strip().lower()
    if action not in {"trigger", "status", "check"}:
        return ToolResult("Unsupported IFTTT action. Use trigger or status.", ok=False)
    config = _config()
    key = str(config.get("key") or "").strip()

    if action in {"status", "check"}:
        if not key:
            return ToolResult("IFTTT is not set up. Get your key from "
                    "ifttt.com/maker_webhooks -> Documentation, then put it "
                    "in config/ifttt.json as {\"key\": \"...\"} or set "
                    "IFTTT_KEY.", ok=False)
        known = ", ".join(config.get("events") or []) or "none recorded"
        return ToolResult(f"IFTTT is configured locally. Recorded applets: {known}. Remote access has not been checked.", ok=True)

    if not key:
        return ToolResult("No IFTTT key, so there is nothing to trigger. Get one from "
                "ifttt.com/maker_webhooks and set IFTTT_KEY.", ok=False)

    event = str(kwargs.get("event") or kwargs.get("name") or "").strip()
    if not event:
        return ToolResult("Which applet? Give me the event name you set in the IFTTT "
                "Webhooks trigger, e.g. event='kettle_on'.", ok=False)

    payload = {}
    for index, key_name in enumerate(("value1", "value2", "value3"), start=1):
        value = kwargs.get(key_name) or kwargs.get(f"value{index}")
        if value is not None and str(value).strip():
            payload[key_name] = str(value)

    request = urllib.request.Request(
        ENDPOINT.format(event=urllib.parse.quote(event), key=key),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ORION/1.0"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return ToolResult("IFTTT rejected the key (401). Check it has not been reset.", ok=False)
        return ToolResult(f"IFTTT refused the trigger ({exc.code} {exc.reason}).", ok=False)
    except urllib.error.URLError as exc:
        return ToolResult(f"IFTTT could not be reached ({exc.reason}).", ok=False)
    except Exception as exc:
        return ToolResult(f"The IFTTT trigger failed ({exc}).", ok=False)

    if "errors" in body.lower():
        return ToolResult(f"IFTTT reported a problem: {body[:160]}", ok=False)
    # "Asked", not "done": IFTTT acknowledges the request, never the outcome.
    # Whether the applet on the other side actually fired is not something
    # this can know, and claiming otherwise would be a lie told confidently.
    return ToolResult(f"Asked IFTTT to run {event!r}. It acknowledges the request "
            f"rather than the result, so I cannot tell you whether the applet "
            f"itself succeeded.", ok=True)


def get_tool_schema():
    """Loader contract; inspection does not contact the external service."""
    return {'name': 'ifttt_webhook',
     'description': 'Trigger an IFTTT applet by name, reaching devices and services ORION has no '
                    'dedicated tool for. Fire-and-forget: IFTTT acknowledges the request, not the '
                    'outcome, so this cannot confirm the applet actually ran.',
     'parameters': {'type': 'object',
                    'properties': {'action': {'type': 'string',
                                              'description': 'trigger (default) or status.'},
                                   'event': {'type': 'string',
                                             'description': 'The event name from the IFTTT '
                                                            "Webhooks trigger, e.g. 'kettle_on'."},
                                   'value1': {'type': 'string',
                                              'description': 'Optional value passed to the '
                                                             'applet.'},
                                   'value2': {'type': 'string',
                                              'description': 'Optional second value.'},
                                   'value3': {'type': 'string',
                                              'description': 'Optional third value.'}},
                    'required': []}}


def run(**kwargs) -> ToolResult:
    """Report failures explicitly, including invalid input and malformed replies."""
    try:
        return _run_impl(**kwargs)
    except Exception as exc:
        return ToolResult(f"Plugin action failed ({type(exc).__name__}); completion was not verified.", ok=False)
