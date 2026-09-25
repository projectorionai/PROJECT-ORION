"""Get a notification onto your phone, by whichever route you have set up.

Three services, one tool. ntfy, Gotify and Pushover all do the same job and
all have a free route to it, so they are backends here rather than three
plugins — a tool list with three entries that each mean "tell me something" is
a tool list the model picks from badly, and ORION already has a great many
tools competing for the same attention.

    ntfy      free, no account at all. Choose a topic, subscribe in the app.
    Gotify    free, self-hosted. You already run the server, so you already
              have the URL and a token.
    Pushover  one-off app purchase, then the API is free. Most reliable
              delivery of the three on iOS.

Whichever is configured is used. Configure more than one and ORION prefers
the one you name, or the first that is ready.

What NOT to send
----------------
On ntfy's public server a topic is only as private as its name: anyone who
learns it can read what is published there and publish to it themselves.
Gotify self-hosted and Pushover are both properly authenticated. So the honest
default is to treat a push as a nudge rather than as a private message, and
to say so rather than letting someone discover it.
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
CONFIG_NAME = "messaging.json"
NTFY_DEFAULT = "https://ntfy.sh"
PUSHOVER_API = "https://api.pushover.net/1/messages.json"

#: Named rather than numbered: "priority 4" means nothing to someone asking
#: ORION to send something urgently. Each service numbers them differently.
PRIORITY = {
    "min": {"ntfy": 1, "gotify": 0, "pushover": -2},
    "low": {"ntfy": 2, "gotify": 2, "pushover": -1},
    "default": {"ntfy": 3, "gotify": 5, "pushover": 0},
    "normal": {"ntfy": 3, "gotify": 5, "pushover": 0},
    "high": {"ntfy": 4, "gotify": 7, "pushover": 1},
    "urgent": {"ntfy": 5, "gotify": 8, "pushover": 1},
}

SERVICES = ("ntfy", "gotify", "pushover")


def _config() -> dict:
    """``config/messaging.json``, with environment variables winning.

    Shape::

        {"ntfy":     {"topic": "...", "server": "...", "token": ""},
         "gotify":   {"url": "https://...", "token": "..."},
         "pushover": {"token": "app-token", "user": "user-key"}}

    Never in this file. A plugin that ships with its own topic publishes
    everyone's notifications to the same place.
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

    for service, keys in (("ntfy", (("topic", "NTFY_TOPIC"),
                                    ("server", "NTFY_SERVER"),
                                    ("token", "NTFY_TOKEN"))),
                          ("gotify", (("url", "GOTIFY_URL"),
                                      ("token", "GOTIFY_TOKEN"))),
                          ("pushover", (("token", "PUSHOVER_TOKEN"),
                                        ("user", "PUSHOVER_USER")))):
        section = dict(data.get(service) or {})
        for key, env in keys:
            value = os.getenv(env, "").strip()
            if value:
                section[key] = value
        data[service] = section
    return data


def _ready(service: str, config: dict) -> bool:
    section = config.get(service) or {}
    if service == "ntfy":
        return bool(section.get("topic"))
    if service == "gotify":
        return bool(section.get("url") and section.get("token"))
    if service == "pushover":
        return bool(section.get("token") and section.get("user"))
    return False


def _post(url: str, data: bytes, headers: dict) -> tuple[bool, str]:
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            if response.status not in (200, 201):
                return False, f"the service answered {response.status}"
            body = response.read()
            if body:
                reply = json.loads(body)
                if isinstance(reply, dict) and (reply.get("errors") or reply.get("status") == 0):
                    return False, "the service rejected the notification"
    except urllib.error.HTTPError as exc:
        return False, f"refused ({exc.code} {exc.reason})"
    except urllib.error.URLError as exc:
        return False, f"could not be reached ({exc.reason})"
    except Exception as exc:
        return False, f"failed ({exc})"
    return True, ""


def _send_ntfy(section: dict, message: str, title: str, level: str,
               tags: str) -> tuple[bool, str]:
    server = str(section.get("server") or NTFY_DEFAULT).rstrip("/")
    topic = str(section.get("topic") or "")
    headers = {"Content-Type": "text/plain; charset=utf-8",
               "User-Agent": "ORION/1.0"}
    if title:
        # ntfy reads its metadata from HTTP headers, which cannot carry
        # arbitrary Unicode. A non-Latin title joins the body rather than
        # failing the whole request.
        try:
            title.encode("latin-1")
            headers["Title"] = title
        except UnicodeEncodeError:
            message = f"{title}\n\n{message}"
    if level in PRIORITY:
        headers["Priority"] = str(PRIORITY[level]["ntfy"])
    if tags:
        headers["Tags"] = tags
    if section.get("token"):
        headers["Authorization"] = f"Bearer {section['token']}"
    ok, why = _post(f"{server}/{urllib.parse.quote(topic)}",
                    message.encode("utf-8"), headers)
    return ok, why or "ntfy accepted the notification; phone delivery is not confirmed."


def _send_gotify(section: dict, message: str, title: str, level: str,
                 tags: str) -> tuple[bool, str]:
    url = str(section.get("url") or "").rstrip("/")
    payload = {"message": message, "title": title or "ORION"}
    if level in PRIORITY:
        payload["priority"] = PRIORITY[level]["gotify"]
    ok, why = _post(
        f"{url}/message?token={urllib.parse.quote(str(section['token']))}",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json", "User-Agent": "ORION/1.0"})
    return ok, why or "Gotify accepted the notification; phone delivery is not confirmed."


def _send_pushover(section: dict, message: str, title: str, level: str,
                   tags: str) -> tuple[bool, str]:
    fields = {"token": section["token"], "user": section["user"],
              "message": message}
    if title:
        fields["title"] = title
    if level in PRIORITY:
        fields["priority"] = PRIORITY[level]["pushover"]
    ok, why = _post(PUSHOVER_API, urllib.parse.urlencode(fields).encode("utf-8"),
                    {"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent": "ORION/1.0"})
    return ok, why or "Pushover accepted the notification; phone delivery is not confirmed."


_SENDERS = {"ntfy": _send_ntfy, "gotify": _send_gotify,
            "pushover": _send_pushover}


def _run_impl(**kwargs) -> ToolResult:
    """Execute the plugin. Returns a short line for ORION to speak."""
    action = str(kwargs.get("action") or "send").strip().lower()
    if action not in {"send", "status", "check"}:
        return ToolResult("Unsupported push action. Use send or status.", ok=False)
    config = _config()
    ready = [s for s in SERVICES if _ready(s, config)]

    if action in {"status", "check"}:
        if not ready:
            return ToolResult("No push service is set up. The quickest is ntfy: install "
                    "the app, subscribe to an unguessable topic name, and put "
                    "it under \"ntfy\" in config/messaging.json. Gotify and "
                    "Pushover work too.", ok=False)
        notes = []
        for service in ready:
            if service == "ntfy" and not (config["ntfy"].get("token")):
                notes.append("ntfy (anyone who knows the topic can read it)")
            else:
                notes.append(service)
        return ToolResult("Push is configured locally via " + ", ".join(notes) + ". Remote delivery has not been checked.", ok=True)

    if not ready:
        return ToolResult("Nowhere to send this — no push service is configured. Set "
                "NTFY_TOPIC, or GOTIFY_URL + GOTIFY_TOKEN, or PUSHOVER_TOKEN "
                "+ PUSHOVER_USER.", ok=False)

    message = str(kwargs.get("message") or kwargs.get("text") or "").strip()
    if not message:
        return ToolResult("Nothing to send — give me the message.", ok=False)

    wanted = str(kwargs.get("service") or "").strip().lower()
    if wanted and wanted not in SERVICES:
        return ToolResult(f"I don't know a push service called {wanted!r}.", ok=False)
    if wanted and wanted not in ready:
        return ToolResult(f"{wanted} is not configured. Ready: "
                + ", ".join(ready) + ".", ok=False)

    service = wanted or ready[0]
    ok, detail = _SENDERS[service](
        config[service], message,
        str(kwargs.get("title") or "").strip(),
        str(kwargs.get("priority") or "").strip().lower(),
        str(kwargs.get("tags") or "").strip())
    return ToolResult(detail, ok=ok)


def get_tool_schema():
    """Loader contract; inspection does not contact the external service."""
    return {'name': 'push_notify',
     'description': "Send a notification to the user's phone via ntfy, Gotify or Pushover — "
                    "whichever is configured. Use when they ask ORION to 'send that to my phone', "
                    "'notify me', or 'remind me on my phone'. ntfy needs no account at all.",
     'parameters': {'type': 'object',
                    'properties': {'action': {'type': 'string',
                                              'description': 'send (default) or status.'},
                                   'message': {'type': 'string',
                                               'description': 'What the notification should say.'},
                                   'title': {'type': 'string', 'description': 'Optional heading.'},
                                   'priority': {'type': 'string',
                                                'description': 'min, low, default, high or '
                                                               'urgent.'},
                                   'tags': {'type': 'string',
                                            'description': 'Comma-separated ntfy tags, e.g. '
                                                           "'warning,skull'."},
                                   'service': {'type': 'string',
                                               'description': 'Force one of ntfy, gotify or '
                                                              'pushover. Omit to use whichever is '
                                                              'set up.'}},
                    'required': []}}


def run(**kwargs) -> ToolResult:
    """Report failures explicitly, including invalid input and malformed replies."""
    try:
        return _run_impl(**kwargs)
    except Exception as exc:
        return ToolResult(f"Plugin action failed ({type(exc).__name__}); completion was not verified.", ok=False)
