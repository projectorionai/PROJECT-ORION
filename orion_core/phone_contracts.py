"""Fixed, bounded native phone commands shared by routing and the uplink.

Commands carry data to the paired device. They never accept shell commands,
arbitrary Android intents, guessed accessibility nodes or another device id.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

PACKAGE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+\Z")
EMAIL = re.compile(r"[^\s@,;<>]+@[^\s@,;<>]+\.[^\s@,;<>]+\Z")
SCREENS = frozenset({"app", "accessibility", "wifi", "bluetooth", "notifications",
                     "location", "sound", "display", "privacy", "battery"})

# action -> (native method, required fields, optional fields)
COMMANDS = {
    "compose_email": ("device.compose_email", {"to", "subject", "body"}, {"package"}),
    "send_email": ("device.send_email", {"to", "subject", "body"}, {"package"}),
    "open_url": ("device.open_url", {"url"}, set()),
    "map": ("device.map", {"query"}, set()),
    "navigate": ("device.navigate", {"query"}, set()),
    "share": ("device.share", {"text"}, set()),
    "open_settings": ("device.open_settings", {"screen"}, set()),
    "set_alarm": ("device.set_alarm", {"hour", "minutes"}, {"label"}),
    "set_timer": ("device.set_timer", {"seconds"}, {"label"}),
    "create_contact": ("device.create_contact", {"name"}, {"number", "email"}),
    "create_calendar_event": ("device.create_calendar_event", {"title", "start", "end"}, {"location", "description"}),
    "show_trending_videos": ("device.show_trending_videos", set(), set()),
    "list_apps": ("device.list_apps", set(), set()),
    "find_contact": ("device.find_contact", {"name"}, set()),
    "back": ("device.back", set(), set()),
    "home": ("device.home", set(), set()),
}
METHODS = frozenset(spec[0] for spec in COMMANDS.values())


def units(value: str) -> int:
    try:
        return len(value.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        return 2**31


def text(value, maximum: int, *, empty: bool = False, multiline: bool = False) -> bool:
    return (isinstance(value, str) and units(value) <= maximum
            and (empty or bool(value.strip())) and "\x00" not in value
            and (multiline or not any(char in value for char in "\r\n")))


def contact_name(value) -> bool:
    return (text(value, 80) and units(value.strip()) >= 2
            and any(char.isalpha() for char in value)
            and all(char.isalpha() or char in " .'-" for char in value))


def email(value) -> bool:
    return text(value, 254) and EMAIL.fullmatch(value) is not None


def integer(value, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def valid_command(method: str, params: dict) -> bool:
    if not isinstance(params, dict):
        return False
    spec = next((row for row in COMMANDS.values() if row[0] == method), None)
    if spec is None:
        return False
    _, required, optional = spec
    if not required <= params.keys() <= required | optional:
        return False
    if "label" in params and not text(params["label"], 200, empty=True):
        return False
    if method in {"device.compose_email", "device.send_email"}:
        return ((email(params["to"]) or contact_name(params["to"]))
                and text(params["subject"], 200, empty=True)
                and text(params["body"], 16000, empty=method == "device.compose_email", multiline=True)
                and ("package" not in params or
                     (text(params["package"], 150) and PACKAGE.fullmatch(params["package"]) is not None)))
    if method == "device.open_url":
        if not text(params["url"], 2048):
            return False
        try:
            url = urlsplit(params["url"])
            return bool(url.scheme in {"https", "http"} and url.hostname
                        and url.username is None and url.password is None)
        except ValueError:
            return False
    if method in {"device.map", "device.navigate"}:
        return text(params["query"], 1000)
    if method == "device.share":
        return text(params["text"], 16000, multiline=True)
    if method == "device.open_settings":
        return isinstance(params["screen"], str) and params["screen"] in SCREENS
    if method == "device.set_alarm":
        return integer(params["hour"], 0, 23) and integer(params["minutes"], 0, 59)
    if method == "device.set_timer":
        return integer(params["seconds"], 1, 86400)
    if method == "device.create_contact":
        return (text(params["name"], 80) and bool({"number", "email"} & params.keys())
                and ("number" not in params or
                     (text(params["number"], 40) and re.fullmatch(r"\+?[0-9() -]{3,40}", params["number"]) is not None))
                and ("email" not in params or email(params["email"])))
    if method == "device.create_calendar_event":
        return (text(params["title"], 200)
                and integer(params["start"], 1, 253402300799000)
                and integer(params["end"], 1, 253402300799000)
                and params["end"] > params["start"]
                and ("location" not in params or text(params["location"], 1000, empty=True))
                and ("description" not in params or text(params["description"], 4000, empty=True, multiline=True)))
    if method == "device.find_contact":
        return contact_name(params["name"])
    return not params


def translate(action: str, args: dict) -> tuple[str, dict] | None:
    spec = COMMANDS.get(action) if isinstance(action, str) else None
    if spec is None or not isinstance(args, dict):
        return None
    method, required, optional = spec
    fields = {key: value for key, value in args.items() if key != "action"}
    if set(args) != {"action", *fields} or not valid_command(method, fields):
        return None
    return method, fields
