"""
ORION's smart home - Home Assistant and Philips Hue, built in (Mark XXXVII.5).

Two forged plugins (homeassistant_call_service, philips_hue) could each flip a
light, separately, with no shared idea of what is in the house. This is one
model of it: every light, switch, fan, scene and lock either backend knows,
found by the names people use ("the bedroom lamp", "all the lights"), with
routines on top ("goodnight", "morning", "movie", "away") and a Home page on
the Command Deck.

Configuration is where the plugins kept it, so nothing has to be entered again:

    config/homeassistant.json   {"url": "http://homeassistant.local:8123",
                                 "token": "<long-lived access token>"}
    config/philips_hue.json     {"bridge": "192.168.1.42", "username": "<key>"}
    config/smart_home.json      {"routines": {"goodnight": [{"do": "off",
                                 "target": "all lights"}, ...]}}

Environment variables win (HASS_URL, HASS_TOKEN, HUE_BRIDGE, HUE_USERNAME),
then a secret stored for the old plugin in the plugin vault, then the file.

Safety: nothing here unlocks a lock, opens a garage door or a cover, or
disarms an alarm - a misheard sentence must never open the house. Those stay
in the Home Assistant app. Locking and closing are fine.

Blocking network calls: run them off the event loop (the tool does).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from .data import ToolResult

TIMEOUT_S = 8.0
MAX_RESPONSE = 2 * 1024 * 1024

#: Home Assistant domains ORION shows and controls.
HA_DOMAINS = ("light", "switch", "fan", "scene", "script", "lock", "cover",
              "climate", "media_player", "input_boolean")
#: What a misheard sentence must never do.
REFUSED = {("lock", "unlock"), ("lock", "open"), ("cover", "on"), ("cover", "open"),
           ("alarm_control_panel", "off")}
#: Named colours (Hue hue 0-65535, saturation 0-254; Home Assistant takes the name).
COLOURS = {
    "red": (0, 254), "orange": (5461, 254), "yellow": (10923, 254), "green": (21845, 254),
    "cyan": (32768, 254), "blue": (43690, 254), "purple": (49152, 254),
    "pink": (56100, 200), "magenta": (54613, 254), "white": (8418, 30),
    "warm white": (8418, 140), "warm": (8418, 140), "cool white": (39000, 40),
}
#: Routines when config/smart_home.json names none. Each step: do (on, off,
#: brightness, colour, scene), target, optional value.
DEFAULT_ROUTINES: dict[str, list[dict[str, Any]]] = {
    "goodnight": [{"do": "off", "target": "all lights"}],
    "morning": [{"do": "brightness", "target": "all lights", "value": 80}],
    "movie": [{"do": "brightness", "target": "all lights", "value": 15},
              {"do": "colour", "target": "all lights", "value": "warm"}],
    "away": [{"do": "off", "target": "all lights"}, {"do": "off", "target": "all switches"}],
}

SETUP = ("No smart home is connected yet. Home Assistant: put {\"url\": ..., \"token\": ...} in "
         "config/homeassistant.json (a long-lived token from your Home Assistant profile page). "
         "Philips Hue: press the bridge's button, then say 'pair my Hue bridge' (or give its IP "
         "address).")


@dataclass
class Device:
    id: str
    name: str
    kind: str                       # light, switch, fan, scene, lock, cover, climate...
    backend: str                    # "ha" or "hue"
    state: str = ""                 # on / off / locked / unavailable / ...
    brightness: int | None = None   # percent
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def on(self) -> bool:
        return self.state in {"on", "open", "playing", "heat", "cool", "auto"}


# ── configuration ────────────────────────────────────────────────────────────

def _file(name: str) -> dict[str, Any]:
    from .constants import CONFIG_DIR
    try:
        data = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _vault(plugin: str, env: str) -> str:
    try:
        from .jv_plugins.runtime import secret
        return str(secret(plugin, env) or "").strip()
    except Exception:
        return ""


def ha_config() -> dict[str, Any]:
    data = _file("homeassistant.json")
    url = os.getenv("HASS_URL", "").strip() or str(data.get("url") or "").strip()
    token = (os.getenv("HASS_TOKEN", "").strip() or _vault("homeassistant_call_service", "HASS_TOKEN")
             or str(data.get("token") or "").strip())
    return {"url": url.rstrip("/"), "token": token}


def hue_config() -> dict[str, Any]:
    data = _file("philips_hue.json")
    bridge = os.getenv("HUE_BRIDGE", "").strip() or str(data.get("bridge") or "").strip()
    username = (os.getenv("HUE_USERNAME", "").strip() or _vault("philips_hue", "HUE_USERNAME")
                or str(data.get("username") or "").strip())
    return {"bridge": bridge, "username": username}


def routines() -> dict[str, list[dict[str, Any]]]:
    custom = _file("smart_home.json").get("routines")
    merged = {k: list(v) for k, v in DEFAULT_ROUTINES.items()}
    if isinstance(custom, dict):
        for name, steps in custom.items():
            if isinstance(steps, list):
                merged[str(name).lower()] = [s for s in steps if isinstance(s, dict)]
    return merged


def backends() -> list[str]:
    out = []
    ha = ha_config()
    if ha["url"] and ha["token"]:
        out.append("ha")
    hue = hue_config()
    if hue["bridge"] and hue["username"]:
        out.append("hue")
    return out


# ── HTTP ─────────────────────────────────────────────────────────────────────

class HomeError(RuntimeError):
    pass


def _request(url: str, *, method: str = "GET", payload: Any = None,
             headers: dict[str, str] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = response.read(MAX_RESPONSE)
    except urllib.error.HTTPError as exc:
        raise HomeError("the token was refused (401)" if exc.code == 401
                        else f"it answered {exc.code}") from None
    except urllib.error.URLError as exc:
        raise HomeError(f"it could not be reached ({exc.reason})") from None
    try:
        return json.loads(body.decode("utf-8", "replace") or "null")
    except ValueError:
        raise HomeError("it sent something that was not JSON") from None


def _ha(path: str, payload: Any = None) -> Any:
    config = ha_config()
    return _request(config["url"] + path, method="POST" if payload is not None else "GET",
                    payload=payload, headers={"Authorization": f"Bearer {config['token']}"})


def _hue(path: str, payload: Any = None, method: str = "GET") -> Any:
    config = hue_config()
    return _request(f"http://{config['bridge']}/api/{config['username']}{path}",
                    method=method, payload=payload)


# ── the house ────────────────────────────────────────────────────────────────

def _ha_devices() -> list[Device]:
    out = []
    for state in _ha("/api/states") or []:
        entity = str(state.get("entity_id") or "")
        domain = entity.split(".", 1)[0]
        if domain not in HA_DOMAINS:
            continue
        attributes = state.get("attributes") or {}
        bri = attributes.get("brightness")
        out.append(Device(entity, str(attributes.get("friendly_name") or entity), domain, "ha",
                          str(state.get("state") or ""),
                          round(int(bri) / 255 * 100) if isinstance(bri, (int, float)) else None))
    return out


def _hue_devices() -> list[Device]:
    out = []
    for light_id, light in (_hue("/lights") or {}).items():
        state = light.get("state") or {}
        reachable = state.get("reachable", True)
        out.append(Device(f"light:{light_id}", str(light.get("name") or f"Light {light_id}"),
                          "light", "hue",
                          "unavailable" if not reachable else ("on" if state.get("on") else "off"),
                          round(int(state.get("bri", 0)) / 254 * 100) if state.get("on") else None))
    for group_id, group in (_hue("/groups") or {}).items():
        if group.get("type") in {"Room", "Zone"}:
            state = group.get("state") or {}
            out.append(Device(f"group:{group_id}", str(group.get("name") or f"Room {group_id}"),
                              "room", "hue", "on" if state.get("any_on") else "off"))
    for scene_id, scene in (_hue("/scenes") or {}).items():
        out.append(Device(f"scene:{scene_id}", str(scene.get("name") or scene_id), "scene", "hue",
                          extra={"group": scene.get("group") or ""}))
    return out


def devices() -> tuple[list[Device], list[str]]:
    """Everything the connected backends know, and any backend that failed."""
    found: list[Device] = []
    problems: list[str] = []
    for name, reader, label in (("ha", _ha_devices, "Home Assistant"),
                                ("hue", _hue_devices, "the Hue bridge")):
        if name not in backends():
            continue
        try:
            found.extend(reader())
        except HomeError as exc:
            problems.append(f"{label}: {exc}")
    return found, problems


_FILLER = frozenset({"the", "my", "a", "all", "of", "in", "please", "light", "lights", "lamp",
                     "lamps", "switch", "switches", "room"})


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in _FILLER}


def resolve(target: str, house: list[Device], kinds: tuple[str, ...] = ()) -> list[Device]:
    """The devices *target* names: "all lights", "the bedroom", "desk lamp".
    Exact name first, then every word of the target in the name, then any."""
    text = " ".join(str(target or "").lower().split())
    pool = [d for d in house if not kinds or d.kind in kinds]
    if re.fullmatch(r"(?:all|every|everything|all the)(?: the)?(?: lights?| lamps?)?|everything|"
                    r"the house|the whole house|whole house", text):
        lights = [d for d in pool if d.kind == "light"]
        return lights or [d for d in pool if d.kind in {"light", "room"}]
    if re.fullmatch(r"(?:all|every)(?: the)? switch(?:es)?", text):
        return [d for d in pool if d.kind == "switch"]
    exact = [d for d in pool if d.name.lower() == text or d.id.lower() == text]
    if exact:
        return exact
    wanted = _words(text)
    if not wanted:
        return []
    every = [d for d in pool if wanted <= _words(d.name)]
    if every:
        rooms = [d for d in every if d.kind == "room"]
        return rooms[:1] or every          # a room is one switch for all its lights
    return [d for d in pool if wanted & _words(d.name)]


# ── acting ───────────────────────────────────────────────────────────────────

def _act(device: Device, do: str, value: Any = None) -> None:
    if ((device.kind, do) in REFUSED
            or (device.kind in {"lock", "alarm_control_panel"}
                and do in {"off", "unlock", "open", "toggle"})
            or (device.kind == "cover" and do in {"on", "open", "toggle", "brightness"})):
        raise HomeError(f"I won't {('unlock' if device.kind == 'lock' else 'open')} "
                        f"{device.name} by voice - use the Home Assistant app for that")
    if device.backend == "ha":
        domain = device.kind
        if do == "scene" or domain in {"scene", "script"}:
            _ha(f"/api/services/{domain}/turn_on", {"entity_id": device.id})
            return
        if domain == "lock" and do in {"on", "lock"}:
            _ha("/api/services/lock/lock", {"entity_id": device.id})
            return
        if domain == "cover" and do in {"off", "close"}:
            _ha("/api/services/cover/close_cover", {"entity_id": device.id})
            return
        payload: dict[str, Any] = {"entity_id": device.id}
        service = {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}.get(do, "turn_on")
        if do == "brightness":
            payload["brightness_pct"] = max(1, min(100, int(value)))
        elif do == "colour":
            payload["color_name"] = str(value).replace("warm white", "navajowhite").replace(
                "warm", "navajowhite").replace("cool white", "aliceblue")
        _ha(f"/api/services/{domain}/{service}", payload)
        return
    kind, _, hue_id = device.id.partition(":")
    if kind == "scene":
        group = device.extra.get("group") or "0"
        _hue(f"/groups/{group}/action", {"scene": hue_id}, method="PUT")
        return
    state: dict[str, Any]
    if do == "off":
        state = {"on": False}
    elif do == "toggle":
        state = {"on": not device.on}
    elif do == "brightness":
        state = {"on": True, "bri": max(1, min(254, round(int(value) / 100 * 254)))}
    elif do == "colour":
        hue, sat = COLOURS[str(value)]
        state = {"on": True, "hue": hue, "sat": sat}
    else:
        state = {"on": True}
    path = f"/groups/{hue_id}/action" if kind == "group" else f"/lights/{hue_id}/state"
    reply = _hue(path, state, method="PUT")
    if isinstance(reply, list) and any("error" in item for item in reply if isinstance(item, dict)):
        error = next(item["error"] for item in reply if isinstance(item, dict) and "error" in item)
        raise HomeError(str(error.get("description") or "the bridge refused it"))


def _verb(do: str, value: Any) -> str:
    return {"on": "on", "off": "off", "toggle": "toggled",
            "brightness": f"at {value}%", "colour": f"{value}", "scene": "set"}.get(do, do)


def apply(do: str, target: str, value: Any = None, *, house: list[Device] | None = None) -> ToolResult:
    """Do one thing to whatever *target* names. Blocking; never raises."""
    if not backends():
        return ToolResult(SETUP, ok=False, receipt={"status": "not_configured"})
    do = str(do or "").lower()
    if do in {"color", "colour"}:
        do = "colour"
        value = " ".join(str(value or "").lower().split())
        if value not in COLOURS:
            return ToolResult(f"I know these colours: {', '.join(COLOURS)}.", ok=False)
    if do == "brightness":
        try:
            value = max(0, min(100, int(float(str(value).rstrip("%")))))
        except (TypeError, ValueError):
            return ToolResult("Give the brightness as a percentage, e.g. 40.", ok=False)
        if value == 0:
            do = "off"
    if house is None:
        house, problems = devices()
        if not house and problems:
            return ToolResult("The smart home did not answer: " + "; ".join(problems), ok=False)
    kinds = ("scene",) if do == "scene" else ()
    found = resolve(target, house, kinds)
    if not found and do != "scene":
        found = resolve(target, house, ("scene",)) if do == "on" else []
        if found:
            do = "scene"
    if not found:
        names = ", ".join(sorted({d.name for d in house if d.kind != "scene"})[:12]) or "nothing yet"
        return ToolResult(f"I can't find {target!r} in your smart home. I know: {names}.", ok=False)
    done, failed = [], []
    for device in found[:24]:
        try:
            _act(device, do, value)
            done.append(device.name)
        except HomeError as exc:
            failed.append(f"{device.name} ({exc})")
    if done and not failed:
        what = done[0] if len(done) == 1 else f"{len(done)} devices"
        return ToolResult(f"{what[0].upper() + what[1:]} {_verb(do, value)}.",
                          evidence=[{"smart_home": do, "devices": done}])
    if done:
        return ToolResult(f"Done for {', '.join(done)}; not for {', '.join(failed)}.", ok=False)
    return ToolResult("Nothing changed: " + "; ".join(failed), ok=False)


def run_routine(name: str) -> ToolResult:
    key = " ".join(str(name or "").lower().split()).removesuffix(" routine").removeprefix("my ")
    key = {"good night": "goodnight", "night": "goodnight", "bedtime": "goodnight",
           "wake up": "morning", "good morning": "morning", "film": "movie",
           "cinema": "movie", "leaving": "away", "out": "away"}.get(key, key)
    steps = routines().get(key)
    if steps is None:
        return ToolResult(f"I have no {name!r} routine. I know: {', '.join(routines())}.", ok=False)
    if not backends():
        return ToolResult(SETUP, ok=False, receipt={"status": "not_configured"})
    house, problems = devices()
    if not house:
        return ToolResult("The smart home did not answer: " + "; ".join(problems or ["no devices"]),
                          ok=False)
    outcomes = [apply(str(step.get("do") or "on"), str(step.get("target") or "all lights"),
                      step.get("value"), house=house) for step in steps]
    ok = all(r.ok for r in outcomes)
    detail = " ".join(r.text for r in outcomes)
    return ToolResult((f"{key.title()} routine done. " if ok else f"{key.title()} routine: ") + detail,
                      ok=ok)


def pair_hue(bridge: str = "") -> ToolResult:
    """Ask the bridge for a key (its button must have been pressed in the last
    30 s) and keep it in config/philips_hue.json."""
    address = str(bridge or hue_config()["bridge"] or "").strip()
    if not address:
        try:
            found = _request("https://discovery.meethue.com/")
            address = str((found or [{}])[0].get("internalipaddress") or "")
        except (HomeError, IndexError, AttributeError):
            address = ""
    if not re.fullmatch(r"[0-9a-zA-Z.\-:]+", address or ""):
        return ToolResult("I couldn't find a Hue bridge. Tell me its IP address (the Hue app "
                          "shows it under Settings > Bridges).", ok=False)
    try:
        reply = _request(f"http://{address}/api", method="POST",
                         payload={"devicetype": "orion#desktop"})
    except HomeError as exc:
        return ToolResult(f"The bridge at {address} did not answer: {exc}.", ok=False)
    first = reply[0] if isinstance(reply, list) and reply else {}
    if "error" in first:
        if (first["error"] or {}).get("type") == 101:
            return ToolResult("Press the round button on top of the Hue bridge, then ask me to pair "
                              "again within 30 seconds.", ok=False)
        return ToolResult(f"The bridge refused: {(first['error'] or {}).get('description')}.", ok=False)
    username = str((first.get("success") or {}).get("username") or "")
    if not username:
        return ToolResult("The bridge answered but gave me no key.", ok=False)
    from .atomic_io import atomic_write_text
    from .constants import CONFIG_DIR
    path = CONFIG_DIR / "philips_hue.json"
    data = _file("philips_hue.json")
    data.update(bridge=address, username=username)
    atomic_write_text(path, json.dumps(data, indent=2), encoding="utf-8")
    return ToolResult(f"Paired with the Hue bridge at {address}. Your lights are on the Home page.")


def describe() -> tuple[str, list[dict[str, Any]]]:
    """(a spoken summary, the devices as plain dicts for the Home page)."""
    if not backends():
        return SETUP, []
    house, problems = devices()
    rows = [asdict(d) | {"on": d.on} for d in house]
    lights = [d for d in house if d.kind == "light"]
    lit = [d.name for d in lights if d.on]
    parts = [f"{len(lights)} light{'s' if len(lights) != 1 else ''}"
             + (f", {len(lit)} on ({', '.join(lit[:5])})" if lit else ", all off")]
    others = {k: sum(1 for d in house if d.kind == k)
              for k in ("switch", "fan", "lock", "scene", "room", "climate")}
    parts += [f"{n} {k}{'es' if k == 'switch' and n != 1 else ('s' if n != 1 else '')}"
              for k, n in others.items() if n]
    text = "Your smart home: " + "; ".join(parts) + "."
    if problems:
        text += " Not answering: " + "; ".join(problems) + "."
    return text, rows


__all__ = ["COLOURS", "DEFAULT_ROUTINES", "Device", "HomeError", "SETUP", "apply", "backends",
           "describe", "devices", "pair_hue", "resolve", "routines", "run_routine"]
