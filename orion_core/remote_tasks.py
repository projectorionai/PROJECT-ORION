"""Route phone instructions into reviewed tools instead of conversational claims."""

from __future__ import annotations

import json
import inspect
import re
from typing import Any


_QUOTES = re.compile(r'"[^"]*"|“[^”]*”|‘[^’]*’|(?<!\w)\'[^\']*\'(?!\w)', re.S)
_PHONE_INFO = frozenset({"query_intelligence", "recall_conversation", "web_search", "capabilities"})
_MCP_READ_ACTIONS = frozenset({"list", "servers"})
_UNKNOWN_RECIPIENTS = frozenset({"someone", "somebody", "anyone", "them", "him", "her", "everyone", "all contacts"})


def _outside_quotes(text: str) -> str:
    """Keep offsets whilst excluding quoted message content from command detection."""
    return _QUOTES.sub(lambda match: " " * len(match.group()), text)


def _instruction(message: str) -> str:
    text = str(message or "").strip()
    for _ in range(3):
        text = re.sub(r"^(?:(?:hey|hello)\s+)?orion\b[,!:\s]*|^please\s+|"
                      r"^(?:can|could|would|will)\s+you\s+|"
                      r"^i\s+(?:want|need)\s+(?:you\s+)?to\s+", "", text, flags=re.I)
    return text


def requested_target(message: str, *, default: str = "phone") -> str:
    """Phone turns stay on the phone unless a host is explicitly named."""
    original = _instruction(message)
    text = _outside_quotes(original)
    # An unquoted e-mail subject or message is also content, not a target.
    content = re.search(r"\b(?:subject(?:\s+line)?|body|saying|that\s+says|message|share)\b", text, re.I)
    if content:
        quoted = [match for match in _QUOTES.finditer(original) if match.start() > content.start()]
        tail = text[quoted[-1].end():] if quoted else ""
        text = text[:content.start()]
        if re.match(r"^[\s,.]*(?:on|from|using|via|not|rather\s+than|instead\s+of)\b", tail, re.I):
            text += " " + tail
    host_word = r"(?:pc|desktop|computer|host|windows(?:\s+pc)?)"
    phone_word = r"(?:phone|mobile|android|handset)"
    host = bool(re.search(rf"\b(?:on|from|using|via|at|control|operate)\s+(?:(?:my|the|our)\s+)?{host_word}\b|"
                          rf"\b(?:my|the|our)\s+{host_word}\b|"
                          rf"\b{host_word}\s+(?:screen|apps?|files?|metrics|stats|telemetry|keyboard|mouse|notepad|terminal|e-?mail)\b|"
                          rf"^\s*{host_word}\s*:", text, re.I))
    phone = bool(re.search(rf"\b(?:on|from|using|via|at|control|operate)\s+(?:(?:my|the|our)\s+)?{phone_word}\b|"
                           rf"\b(?:my|the|our)\s+{phone_word}\b|^\s*{phone_word}\s*:", text, re.I))
    # Negated destinations must not silently reverse the requested device.
    if host and re.search(r"\b(?:not|rather\s+than|instead\s+of)\s+(?:on\s+)?(?:my\s+|the\s+)?(?:pc|desktop|computer|host|windows)\b", text, re.I):
        host = False
    if phone and re.search(r"\b(?:not|rather\s+than|instead\s+of)\s+(?:on\s+)?(?:my\s+|the\s+)?(?:phone|mobile|android|handset)\b", text, re.I):
        phone = False
    if host and phone:
        raise ValueError("Should this instruction run on your phone or your PC? Please choose one device.")
    if host:
        return "desktop"
    if phone:
        return "phone"
    return "desktop" if default.lower() in {"desktop", "pc", "computer", "host"} else "phone"


def is_action_request(message: str) -> bool:
    text = _instruction(message)
    text = re.sub(r"^(?:pc|desktop|computer|host|phone|mobile|android)\s*:\s*", "", text, flags=re.I)
    return bool(re.match(
        r"(?:open|launch|close|quit|write|type|click|tap|press|scroll|create|save|send|"
        r"call|dial|email|e-mail|text|sms|draft|compose|navigate|map|share|go|copy|paste|move|rename|delete|organise|schedule|"
        r"download|upload|turn|set|search|find|read|capture|take|run|start|stop|"
        r"play|pause|resume|enable|disable|show|check|list|get|lock|unlock|mute|unmute|"
        r"display|clear|change|recolour|reshape)\b", text, re.I))


def notepad_plan(message: str) -> list[dict[str, Any]] | None:
    """An exact quoted Notepad request needs no model to choose its tools."""
    if not re.search(r"\bnotepad\b", message, re.I) or not re.search(r"\b(?:write|type)\b", message, re.I):
        return None
    match = re.search(r'["“](.+?)["”]', message, re.S)
    if match is None:
        return None
    return [
        {"tool": "open_app", "args": {"app_name": "notepad"}},
        {"tool": "desktop_control", "args": {"action": "edit_text", "text": match[1],
                                              "title": "Notepad", "replace": False}},
    ]


def _value(raw: str) -> str:
    raw = raw.strip()
    pairs = {'"': '"', "“": "”", "‘": "’", "'": "'"}
    if raw[:1] in pairs:
        if len(raw) < 2 or raw[-1] != pairs[raw[0]]:
            raise ValueError("Please put the exact message text inside matching quotation marks.")
        return raw[1:-1]
    return raw


def _recipient(raw: str, *, email: bool = False) -> str:
    value = _value(re.sub(r"\s+(?:with|with\s+the|with\s+a)\s*$", "", raw, flags=re.I).strip(" ,"))
    if not value or value.casefold() in _UNKNOWN_RECIPIENTS:
        raise ValueError("Who should receive the e-mail? Give an exact contact name or e-mail address."
                         if email else "Which exact contact name or phone number should I use?")
    return value


def _marked_fields(text: str) -> tuple[str, dict[str, str]]:
    visible = _outside_quotes(text)
    markers = list(re.finditer(r"\b(?:(?:and|with)\s+(?:the\s+)?)?(subject(?:\s+line)?|body|message|saying|that\s+says)\b", visible, re.I))
    if not markers:
        return text, {}
    fields: dict[str, str] = {}
    for index, match in enumerate(markers):
        field = "subject" if match.group(1).lower().startswith("subject") else "body"
        if field in fields:
            raise ValueError(f"Please provide one exact {field}; I found more than one.")
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        remainder = text[match.end():end]
        remainder = re.sub(r"^\s*[:=]?\s*", "", remainder)
        fields[field] = _value(remainder.strip().rstrip(","))
    return text[:markers[0].start()], fields


def _phone_step(action: str, **args: Any) -> list[dict[str, Any]]:
    return [{"tool": "phone_device", "args": {"action": action, **args}}]


def phone_plan(message: str) -> list[dict[str, Any]] | None:
    """Exact native requests need neither a model nor guessed account details.

    A recognised but incomplete request raises a specific question. Unknown or
    compound requests continue through the constrained planner.
    """
    if requested_target(message) != "phone":
        return None
    text = _instruction(message)
    visible = _outside_quotes(text)
    if re.search(r"\b(?:and\s+(?:then\s+)?|then\s+)(?:open|send|email|call|text|share|navigate|write|type|tap|click|start|set|create|check|read|find|show|stop|clear|go)\b", visible, re.I):
        return None
    # Remove a trailing destination only outside the quoted payload.
    destination = re.search(r"\b(?:on|from|using|via)\s+(?:my\s+|the\s+)?(?:phone|mobile|android|handset)\s*[.!?]?\s*$", visible, re.I)
    if destination:
        content = re.search(r"\b(?:body|message|saying|that\s+says|share)\b", visible[:destination.start()], re.I)
        quoted_content = bool(content and any(match.start() >= content.end() and match.end() <= destination.start()
                                              for match in _QUOTES.finditer(text)))
        if content is None or quoted_content:
            text = text[:destination.start()].rstrip()
    email = re.match(r"^(?:(send|draft|compose)\s+(?:an?\s+)?(?:e-?mail)\s+(?:to\s+)?|(?:e-?mail)\s+(?:to\s+)?)(.*)$", text, re.I | re.S)
    if email:
        recipient, fields = _marked_fields(email.group(2))
        recipient = _recipient(recipient, email=True)
        missing = [label for label in ("subject", "body") if not fields.get(label)]
        if missing:
            label = " and ".join("message text" if value == "body" else "subject" for value in missing)
            raise ValueError(f"What exact {label} should I use for the e-mail to {recipient}?")
        action = "compose_email" if (email.group(1) or "").lower() in {"draft", "compose"} else "send_email"
        return _phone_step(action, to=recipient, subject=fields["subject"], body=fields["body"])
    sms = re.match(r"^(?:text|sms|send\s+(?:an?\s+)?(?:text(?:\s+message)?|sms)\s+(?:to\s+)?)(.*)$", text, re.I | re.S)
    if sms:
        recipient, fields = _marked_fields(sms.group(1).lstrip())
        # A colon is a useful exact separator for a number or contact label.
        if not fields and ":" in recipient:
            recipient, body = recipient.split(":", 1)
            fields["body"] = _value(body)
        recipient = _recipient(recipient)
        if not fields.get("body"):
            raise ValueError(f"What exact text should I send to {recipient}?")
        return _phone_step("send_sms", number=recipient, body=fields["body"])
    call = re.match(r"^(?:call|dial)\b\s*(.*)$", text, re.I)
    if call:
        return _phone_step("make_call", number=_recipient(call.group(1)))
    navigation = re.match(r"^(?:navigate(?:\s+to)?\b\s*|(?:show|open)\s+(?:a\s+)?map\s+(?:of|for)\s+|map\s+)(.*)$", text, re.I | re.S)
    if navigation:
        query = _value(navigation.group(1))
        if not query:
            raise ValueError("Which exact address or place should I use?")
        return _phone_step("navigate" if text.lower().startswith("navigate") else "map", query=query)
    opened = re.match(r"^(?:open|launch|start)\s+(?:the\s+)?(.+)$", text, re.I | re.S)
    if opened:
        supplied = _value(opened.group(1)).strip()
        if re.match(r"https?://", supplied, re.I):
            return _phone_step("open_url", url=supplied)
        if supplied.lower() in {"settings", "phone settings", "android settings"}:
            return _phone_step("open_settings", screen="app")
        screen = re.fullmatch(r"(accessibility|wi-?fi|bluetooth|notifications|location|sound|display|privacy|battery)\s+settings", supplied, re.I)
        if screen:
            return _phone_step("open_settings", screen=screen.group(1).lower().replace("-", ""))
        if re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", supplied):
            return _phone_step("open_app", package=supplied)
        if supplied.casefold() in {"app", "an app", "application", "an application"}:
            raise ValueError("Which installed phone app should I open? Give its exact name.")
        if supplied.lower().endswith(" app"):
            supplied = supplied[:-4].rstrip()
        return _phone_step("open_app", name=supplied)
    share = re.match(r"^share\b\s*(.*)$", text, re.I | re.S)
    if share:
        payload = _value(share.group(1))
        if not payload:
            raise ValueError("What exact text would you like to share?")
        return _phone_step("share", text=payload)
    if re.fullmatch(r"(?:list|show)\s+(?:my\s+|the\s+)?(?:installed\s+)?apps[.!]?", text, re.I):
        return _phone_step("list_apps")
    contact = re.match(r"^(?:find|look\s+up)\s+(?:the\s+)?contact\b\s*(.*)$", text, re.I)
    if contact:
        return _phone_step("find_contact", name=_recipient(contact.group(1)))
    timer = re.fullmatch(r"(?:set|start|create)\s+(?:a\s+)?timer\s+(?:for\s+)?([0-9]+)\s*(seconds?|minutes?|hours?)[.!]?", text, re.I)
    if timer:
        seconds = int(timer.group(1)) * {"second": 1, "minute": 60, "hour": 3600}[timer.group(2).lower().rstrip("s")]
        if not 1 <= seconds <= 86400:
            raise ValueError("Please choose a timer duration between one second and 24 hours.")
        return _phone_step("set_timer", seconds=seconds)
    if re.match(r"^(?:set|start|create)\s+(?:a\s+)?timer\b", text, re.I):
        raise ValueError("How long should the timer run? Give an exact number of seconds, minutes or hours.")
    alarm = re.fullmatch(r"(?:set|create)\s+(?:an?\s+)?alarm\s+(?:for|at)\s+([0-9]{1,2}):([0-9]{2})(?:\s*([ap]m))?[.!]?", text, re.I)
    if alarm:
        hour, minutes = int(alarm.group(1)), int(alarm.group(2))
        period = (alarm.group(3) or "").lower()
        if minutes >= 60 or period and not 1 <= hour <= 12 or not period and hour >= 24:
            raise ValueError("Please give a valid alarm time, such as 07:30 or 7:30 am.")
        if period:
            hour = hour % 12 + (12 if period == "pm" else 0)
        return _phone_step("set_alarm", hour=hour, minutes=minutes)
    if re.match(r"^(?:set|create)\s+(?:an?\s+)?alarm\b", text, re.I):
        raise ValueError("What exact alarm time should I use? Give a time such as 07:30 or 7:30 am.")
    if re.fullmatch(r"(?:go\s+)?back[.!]?", text, re.I):
        return _phone_step("back")
    if re.fullmatch(r"(?:go\s+)?(?:home|to\s+(?:the\s+)?home\s+screen)[.!]?", text, re.I):
        return _phone_step("home")
    return None


def _planner_declarations(gate: Any, message: str, target: str) -> list[dict[str, Any]]:
    declarations = [item for item in gate.TOOL_DECLARATIONS if item.get("name")]
    if target == "desktop":
        return [item for item in declarations if item["name"] not in {"phone_device", "phone_action"}]
    selected = {"phone_device"}
    low = _outside_quotes(message).lower()
    if re.search(r"\b(?:search|research|find|look\s+up|web|weather|where)\b", low):
        selected |= _PHONE_INFO
    if re.search(r"\b(?:capabilities|features|permissions)\b", low):
        selected.add("capabilities")
    if re.search(r"\bmcp\b", low):
        selected.add("mcp")
    return [item for item in declarations if item["name"] in selected]


def _validate_planned_steps(steps: Any, declarations: list[dict[str, Any]], *,
                            message: str, context: str, target: str) -> list[dict[str, Any]]:
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        raise ValueError("I could not form a valid tool request. Nothing has run.")
    allowed = {item["name"] for item in declarations}
    visible = _outside_quotes(message).casefold()
    for step in steps:
        if (not isinstance(step, dict) or set(step) != {"tool", "args"}
                or step.get("tool") not in allowed or not isinstance(step.get("args"), dict)):
            raise ValueError("The plan selected a tool outside the requested device. Nothing has run.")
        if target != "phone":
            continue
        args = step["args"]
        if step["tool"] == "mcp" and args.get("action") not in _MCP_READ_ACTIONS:
            raise ValueError("Phone requests can inspect MCP services here. Specify the PC explicitly for a host MCP action.")
        if step["tool"] != "phone_device":
            continue
        action = args.get("action")
        if action == "send_email" and re.search(r"\b(?:draft|compose)\b", visible):
            raise ValueError("You asked for an e-mail draft. I have not sent it.")
        required_intent = {
            "send_email": r"\be-?mail\b", "compose_email": r"\be-?mail\b",
            "send_sms": r"\b(?:sms|text)\b", "make_call": r"\b(?:call|dial)\b",
            "open_app": r"\b(?:open|launch|start)\b", "open_url": r"\b(?:open|browse|navigate)\b",
            "navigate": r"\b(?:navigate|directions)\b", "map": r"\b(?:map|where)\b",
            "share": r"\bshare\b", "set_alarm": r"\balarm\b", "set_timer": r"\btimer\b",
            "create_contact": r"\bcontact\b", "create_calendar_event": r"\b(?:calendar|event|meeting)\b",
            "set_text": r"\b(?:write|type|fill|replace|enter)\b", "click": r"\b(?:tap|click|press)\b",
            "scroll": r"\bscroll\b", "back": r"\bback\b", "home": r"\bhome\b",
            "open_settings": r"\b(?:settings|wi-?fi|bluetooth|permissions|privacy|battery)\b",
        }.get(action)
        intent = re.split(r"\b(?:subject(?:\s+line)?|body|saying|that\s+says|message)\b", visible, maxsplit=1, flags=re.I)[0]
        if required_intent and not re.search(required_intent, intent, re.I):
            raise ValueError("That phone action was not part of your instruction. Nothing has run.")
        for field in ("package", "name", "number", "to", "url", "query"):
            value = args.get(field)
            if value is not None and (not isinstance(value, str) or value.casefold() not in message.casefold()):
                raise ValueError(f"Please provide the exact {field.replace('_', ' ')}. Nothing has run.")
        for field in ("body", "subject", "text"):
            value = args.get(field)
            if value is not None and (not isinstance(value, str) or value not in message):
                raise ValueError(f"Please provide the exact {field}. Nothing has run.")
        for field in ("snapshotId", "nodeId"):
            value = args.get(field)
            if value is not None and (not isinstance(value, str) or value not in context and value not in message):
                raise ValueError("Please read your phone screen first so I can use its current controls. Nothing has run.")
    return steps


async def plan_request(router: Any, gate: Any, message: str, context: str, *,
                       target: str = "paired phone") -> list[dict[str, Any]]:
    destination = requested_target(message, default=target)
    direct = notepad_plan(message) if destination == "desktop" else phone_plan(message)
    if direct is not None:
        return direct
    if router is None:
        raise ValueError("The task planner is offline. Nothing has run. Give one exact phone command with its required details, or name the PC for a desktop task.")
    declarations = _planner_declarations(gate, message, destination)
    index = [{"tool": item["name"], "description": item.get("description", "")[:220],
              "parameters": item.get("parameters", {})}
             for item in declarations if item.get("name")]
    instructions = (
        f"Plan an instruction for ORION's {destination}. Return ONLY a JSON object with "
        "steps: an array of 1–6 objects containing tool and args. Use only the tool "
        "names and parameter names listed below. Do not answer with promises or claim "
        "execution. The requested device executes this plan after the user reviews it. "
        "Keep the exact requested text. Phone is the default destination: only an explicit "
        "PC, desktop, computer or host request permits desktop tools. Phone actions use "
        "phone_device and require the native phone's approval and verified result. "
        "E-mail drafts use compose_email; sending uses send_email with the exact recipient, "
        "subject and body. Do not use a PC inbox or send an e-mail from the PC for a phone request. "
        "Open an app by its supplied installed label (name), never invent its package. "
        "Observe the current screen before interacting; do not invent future snapshot or node IDs. "
        "MCP actions on a phone turn are limited to list and servers. "
        "Never add an action the user did not ask for. "
        "If essential details are missing, return {\"steps\":[],\"question\":\"...\"}. "
        "Do not guess node IDs, credentials, recipients, paths or account details.\n\n"
        + context + "\n\nTOOLS:\n" + json.dumps(index, ensure_ascii=False, separators=(",", ":"))
    )
    options = {"system_extra": instructions}
    if "complete_reply" in inspect.signature(router.generate_text).parameters:
        options["complete_reply"] = False
    _profile, reply = await router.generate_text(message, **options)
    raw = str(reply).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    result = json.loads(raw)
    steps = result.get("steps") if isinstance(result, dict) else None
    if steps == []:
        question = result.get("question")
        raise ValueError(str(question or "Please provide the missing task details.")[:300])
    return _validate_planned_steps(steps, declarations, message=message, context=context, target=destination)
