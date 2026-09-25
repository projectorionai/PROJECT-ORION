"""Control the house: lights, switches, scenes, anything Home Assistant exposes.

Home Assistant's REST API is the whole integration — there is no SDK to install
and no cloud account in the path. You need two things: the address of your
instance, and a long-lived access token from your own profile page.

That token is the keys to the house, so it is read from the environment or from
``config/`` and never from this file. Anyone holding it can unlock whatever
your Home Assistant can unlock.

Three verbs, deliberately
-------------------------
``states`` reads, ``call`` acts, and ``entities`` lists what exists. The lister
matters more than it looks: entity ids are not guessable (``light.kitchen`` and
``light.kitchen_ceiling`` are different things, and yours may be called
``light.0x00158d0004a1b2c3``), and a tool that can only act is a tool you
cannot use without already knowing the answer.
"""

from __future__ import annotations

from orion_core.data import ToolResult

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

TIMEOUT = 15
CONFIG_NAME = "homeassistant.json"

#: Domains whose services change the physical world. Everything here is worth
#: confirming before ORION does it on his own initiative — the manifest sets
#: the tier, this list is what the description warns about.
PHYSICAL = {"lock", "cover", "alarm_control_panel", "climate", "vacuum",
            "water_heater", "humidifier", "valve"}


def _config() -> dict:
    """``config/homeassistant.json``, with the environment winning.

    Shape:
        {"url": "http://homeassistant.local:8123", "token": "ey..."}
    """
    data: dict = {}
    try:
        path = Path(__file__).resolve().parent.parent / CONFIG_NAME
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
    except Exception:
        data = {}
    for key, env in (("url", "HASS_URL"), ("token", "HASS_TOKEN")):
        value = os.getenv(env, "").strip()
        if value:
            data[key] = value
    return data


def _request(config: dict, path: str, payload: dict | None = None) -> Any:
    url = str(config.get("url") or "").rstrip("/") + path
    headers = {"Authorization": f"Bearer {config.get('token')}",
               "Content-Type": "application/json",
               "User-Agent": "ORION/1.0"}
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=body, headers=headers,
        method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        text = response.read().decode("utf-8")
    return json.loads(text) if text.strip() else []


def _friendly(state: dict) -> str:
    attributes = state.get("attributes") or {}
    name = attributes.get("friendly_name") or state.get("entity_id")
    return f"{name} ({state.get('entity_id')}): {state.get('state')}"


def _run_impl(**kwargs) -> ToolResult:
    """Execute the plugin. Returns a short line for ORION to speak."""
    action = str(kwargs.get("action") or "call").strip().lower()
    config = _config()

    if not config.get("url") or not config.get("token"):
        return ToolResult("Home Assistant is not configured. Set HASS_URL and HASS_TOKEN, "
                "or put {\"url\": \"http://homeassistant.local:8123\", "
                "\"token\": \"...\"} in config/homeassistant.json. The token is "
                "a long-lived access token from your Home Assistant profile "
                "page — it can do anything you can, so treat it as a key to "
                "the house.", ok=False)

    try:
        if action in {"status", "check"}:
            _request(config, "/api/")
            return ToolResult(f"Home Assistant is reachable at {config['url']}.", ok=True)

        if action in {"entities", "list"}:
            match = str(kwargs.get("match") or "").strip().lower()
            states = _request(config, "/api/states")
            rows = [_friendly(s) for s in states
                    if not match or match in json.dumps(s).lower()]
            if not rows:
                return ToolResult(f"Nothing matched {match!r}." if match
                        else "Home Assistant reported no entities.", ok=True)
            shown = rows[:40]
            more = len(rows) - len(shown)
            return ToolResult("\n".join(shown)
                    + (f"\n…and {more} more." if more > 0 else ""), ok=True)

        if action in {"state", "states", "get"}:
            entity = str(kwargs.get("entity_id") or "").strip()
            if not entity:
                return ToolResult("Which entity? Use action='entities' to see what exists.", ok=False)
            state = _request(config,
                             f"/api/states/{urllib.parse.quote(entity)}")
            return ToolResult(_friendly(state), ok=True)

        if action in {"call", "service", "do"}:
            domain = str(kwargs.get("domain") or "").strip()
            service = str(kwargs.get("service") or "").strip()
            entity = str(kwargs.get("entity_id") or "").strip()
            if not domain or not service:
                return ToolResult("Both a domain and a service are needed, e.g. "
                        "domain='light', service='turn_on', "
                        "entity_id='light.kitchen'.", ok=False)
            payload: dict = {}
            if entity:
                payload["entity_id"] = entity
            extra = kwargs.get("data")
            if isinstance(extra, dict):
                payload.update(extra)
            elif isinstance(extra, str) and extra.strip():
                try:
                    parsed = json.loads(extra)
                    if isinstance(parsed, dict):
                        payload.update(parsed)
                    else:
                        return ToolResult("Service data must be a JSON object.", ok=False)
                except ValueError:
                    return ToolResult(f"'data' is not valid JSON: {extra!r}", ok=False)
            changed = _request(
                config, f"/api/services/{domain}/{service}", payload)
            target = entity or "the configured target"
            count = len(changed) if isinstance(changed, list) else 0
            note = (" This one moves something physical." if domain in PHYSICAL
                    else "")
            return ToolResult(f"Called {domain}.{service} on {target}"
                    + (f"; {count} entity/entities changed." if count
                       else ", which reported no change.") + note, ok=True)

        return ToolResult(f"Unsupported action {action!r}. Use call, state, entities or "
                "status.", ok=False)

    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return ToolResult("Home Assistant rejected the token (401). Long-lived "
                    "tokens can be revoked from your profile page.", ok=False)
        if exc.code == 404:
            return ToolResult("Home Assistant returned 404 — check the entity id or "
                    "service name.", ok=False)
        return ToolResult(f"Home Assistant answered {exc.code} ({exc.reason}).", ok=False)
    except urllib.error.URLError as exc:
        return ToolResult(f"Home Assistant could not be reached at {config['url']} "
                f"({exc.reason}).", ok=False)
    except Exception as exc:
        return ToolResult(f"The Home Assistant call failed ({exc}).", ok=False)


def get_tool_schema():
    """Loader contract; inspection does not contact the external service."""
    return {'name': 'homeassistant_call_service',
     'description': 'Control Home Assistant over its REST API — lights, switches, scenes, locks, '
                    'anything it exposes. Needs HASS_URL and a long-lived access token. '
                    "'entities' lists what exists, 'state' reads one, 'call' acts.",
     'parameters': {'type': 'object',
                    'properties': {'action': {'type': 'string',
                                              'description': 'call (default), state, entities or '
                                                             'status.'},
                                   'domain': {'type': 'string',
                                              'description': "Service domain for 'call', e.g. "
                                                             'light, switch, scene, lock.'},
                                   'service': {'type': 'string',
                                               'description': "Service name for 'call', e.g. "
                                                              'turn_on, turn_off, toggle.'},
                                   'entity_id': {'type': 'string',
                                                 'description': 'The entity to act on or read, '
                                                                'e.g. light.kitchen.'},
                                   'match': {'type': 'string',
                                             'description': "Filter for 'entities', matched "
                                                            'against the whole record.'},
                                   'data': {'type': 'string',
                                            'description': 'Extra service data as a JSON object, '
                                                           'e.g. {"brightness_pct": 40}.'}},
                    'required': []}}


def run(**kwargs) -> ToolResult:
    """Report failures explicitly, including invalid input and malformed replies."""
    try:
        return _run_impl(**kwargs)
    except Exception as exc:
        return ToolResult(f"Plugin action failed ({type(exc).__name__}); completion was not verified.", ok=False)
