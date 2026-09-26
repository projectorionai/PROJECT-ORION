"""Philips Hue, over the local network.

The bridge sits on your LAN and answers HTTP directly, so nothing here touches
Philips' cloud, needs an account, or stops working when their service does.
It also means ORION can turn the lights on with no internet connection at all.

Two things to know about pairing. The bridge will not issue a username until
somebody presses the physical button on top of it — which is the whole security
model, and a good one: you have to be in the room. And the "username" it hands
back is a long random string, not a name; it is a bearer token and belongs in
``config/`` like any other.

If you already run Home Assistant this is redundant — ``homeassistant_call_service``
reaches the same bulbs. This is for people who do not.
"""

from __future__ import annotations

from orion_core.data import ToolResult

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT = 10
CONFIG_NAME = "philips_hue.json"
DISCOVERY = "https://discovery.meethue.com/"

#: Hue takes 0-254, people say percentages.
def _to_brightness(percent: float) -> int:
    return max(1, min(254, int(round(float(percent) / 100.0 * 254))))


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
    for key, env in (("bridge", "HUE_BRIDGE"), ("username", "HUE_USERNAME")):
        value = os.getenv(env, "").strip()
        if value:
            data[key] = value
    return data


def _save(data: dict) -> None:
    path = Path(__file__).resolve().parent.parent / CONFIG_NAME
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if json.loads(path.read_text(encoding="utf-8")) != data:
        raise OSError("Pairing configuration could not be verified")


def _request(url: str, payload: dict | None = None,
             method: str = "GET") -> object:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json", "User-Agent": "ORION/1.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        text = response.read().decode("utf-8")
    return json.loads(text) if text.strip() else {}


def _discover() -> str:
    """Ask Philips' discovery service where the bridge is on this network.

    The only part of this plugin that leaves the LAN, and only to learn a
    local IP address. Give ``bridge`` in the config to skip it entirely.
    """
    try:
        found = _request(DISCOVERY)
    except Exception:
        return ""
    if isinstance(found, list) and found:
        return str(found[0].get("internalipaddress") or "")
    return ""


def _base(config: dict) -> str:
    return f"http://{config['bridge']}/api/{config['username']}"


def _lights(config: dict) -> dict:
    lights = _request(f"{_base(config)}/lights")
    if not isinstance(lights, dict) or "error" in lights:
        raise ValueError("The bridge did not return a valid light inventory")
    return lights


def _find(lights: dict, name: str) -> list[str]:
    """Light ids whose name matches. Substring, because nobody says "Kitchen
    Ceiling 2" when they mean the kitchen."""
    wanted = name.strip().lower()
    if not wanted or wanted in {"all", "everything", "every light"}:
        return list(lights)
    return [i for i, light in lights.items()
            if wanted in str(light.get("name") or "").lower()]


def _run_impl(**kwargs) -> ToolResult:
    """Execute the plugin. Returns a short line for ORION to speak."""
    action = str(kwargs.get("action") or "on").strip().lower()
    config = _config()

    # ── pairing ──────────────────────────────────────────────────────────────
    if action in {"pair", "link", "setup"}:
        bridge = str(kwargs.get("bridge") or config.get("bridge")
                     or _discover()).strip()
        if not bridge:
            return ToolResult("I could not find a Hue bridge. Give me its IP address, "
                    "e.g. action='pair', bridge='192.168.1.42'.", ok=False)
        try:
            reply = _request(f"http://{bridge}/api", {"devicetype": "orion#desktop"},
                             method="POST")
        except Exception as exc:
            return ToolResult(f"The bridge at {bridge} did not answer ({exc}).", ok=False)
        first = reply[0] if isinstance(reply, list) and reply else {}
        if "error" in first:
            description = str(first["error"].get("description") or "")
            if "link button" in description.lower():
                return ToolResult("Press the round button on top of the Hue bridge, "
                        "then ask me to pair again within thirty seconds. "
                        "That button is the whole security model — you have "
                        "to be in the room.", ok=False)
            return ToolResult(f"The bridge refused: {description}", ok=False)
        username = str((first.get("success") or {}).get("username") or "")
        if not username:
            return ToolResult("The bridge answered but gave me no username.", ok=False)
        _save({"bridge": bridge, "username": username})
        return ToolResult(f"Paired with the Hue bridge at {bridge}. The key is in "
                f"config/{CONFIG_NAME} — treat it as a password.", ok=True)

    if not config.get("bridge") or not config.get("username"):
        return ToolResult("Hue is not paired yet. Press the button on the bridge, then "
                "ask me to pair. No account and no cloud — it is all on your "
                "own network.", ok=False)

    try:
        lights = _lights(config)
        if action in {"list", "lights", "status"}:
            if not lights:
                return ToolResult("The bridge reports no lights.", ok=True)
            rows = [f"{light.get('name')}: "
                    f"{'on' if (light.get('state') or {}).get('on') else 'off'}"
                    for light in lights.values()]
            return ToolResult("\n".join(rows), ok=True)

        name = str(kwargs.get("light") or kwargs.get("name") or "all")
        targets = _find(lights, name)
        if not targets:
            available = ", ".join(str(l.get("name")) for l in lights.values())
            return ToolResult(f"No light matches {name!r}. I can see: {available}.", ok=False)

        state: dict = {}
        if action in {"on", "turn_on"}:
            state["on"] = True
        elif action in {"off", "turn_off"}:
            state["on"] = False
        elif action in {"toggle"}:
            first = lights[targets[0]].get("state") or {}
            state["on"] = not bool(first.get("on"))
        elif action in {"brightness", "dim", "bright", "set"}:
            state["on"] = True
            state["bri"] = _to_brightness(kwargs.get("brightness", 60))
        else:
            return ToolResult(f"Unsupported action {action!r}. Use on, off, toggle, "
                    "brightness, list, or pair.", ok=False)

        if "brightness" in kwargs and "bri" not in state:
            state["on"] = True
            state["bri"] = _to_brightness(kwargs["brightness"])

        for light_id in targets:
            reply = _request(f"{_base(config)}/lights/{light_id}/state", state,
                             method="PUT")
            if (not isinstance(reply, list) or not reply or
                    any(not isinstance(item, dict) or "error" in item or
                        not item.get("success") for item in reply)):
                return ToolResult("The Hue bridge did not acknowledge every change; some lights may have changed.", ok=False)
        observed = _lights(config)
        if any(any((observed.get(light_id, {}).get("state") or {}).get(key) != value
                   for key, value in state.items()) for light_id in targets):
            return ToolResult("The Hue bridge accepted the request, but its reported state does not yet confirm every change.", ok=False)
        which = (lights[targets[0]].get("name") if len(targets) == 1
                 else f"{len(targets)} lights")
        if "bri" in state:
            return ToolResult(f"The Hue bridge confirms {which} at brightness {round(state['bri'] / 254 * 100)}%.", ok=True)
        return ToolResult(f"The Hue bridge confirms {which} {'on' if state.get('on') else 'off'}.", ok=True)

    except urllib.error.URLError as exc:
        return ToolResult(f"The Hue bridge at {config['bridge']} could not be reached "
                f"({exc.reason}).", ok=False)
    except Exception as exc:
        return ToolResult(f"The Hue request failed ({exc}).", ok=False)


def get_tool_schema():
    """Loader contract; inspection does not contact the external service."""
    return {'name': 'philips_hue',
     'description': 'Control Philips Hue lights over the local network — on, off, toggle, '
                    'brightness. No account and no cloud: the bridge answers directly on your '
                    'LAN, so it works with no internet. Pair once by pressing the button on the '
                    'bridge. Redundant if you already run Home Assistant.',
     'parameters': {'type': 'object',
                    'properties': {'action': {'type': 'string',
                                              'description': 'on (default), off, toggle, '
                                                             'brightness, list, or pair.'},
                                   'light': {'type': 'string',
                                             'description': 'Light or room name, matched loosely. '
                                                            "'all' for everything."},
                                   'brightness': {'type': 'number',
                                                  'description': '0-100 percent, for the '
                                                                 'brightness action.'},
                                   'bridge': {'type': 'string',
                                              'description': 'Bridge IP address, for pairing when '
                                                             'discovery fails.'}},
                    'required': []}}


def run(**kwargs) -> ToolResult:
    """Report failures explicitly, including invalid input and malformed replies."""
    try:
        return _run_impl(**kwargs)
    except Exception as exc:
        return ToolResult(f"Plugin action failed ({type(exc).__name__}); completion was not verified.", ok=False)
