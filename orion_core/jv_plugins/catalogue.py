"""The installed adapters' public, side-effect-free tool declarations."""

from __future__ import annotations

import copy


def _field(kind: str, description: str, choices: list[str] | None = None) -> dict:
    result = {"type": kind, "description": description}
    if kind == "object":
        # Gemini receives a JSON string for free-form mappings. The runner
        # decodes it against this original schema before invoking a handler.
        result["additionalProperties"] = True
    if choices is not None:
        result["enum"] = choices
    return result


def _action(*choices: str) -> dict:
    return _field("string", "The action to perform.", list(choices))


_CATALOGUE = {
    "jv_document_review": ("Review supplied document text or render structured findings with location, issue and suggestion.", {
        "title": _field("string", "Document title."), "text": _field("string", "Document text to review; maximum 60,000 characters."),
        "summary": _field("string", "Existing review summary, when supplying findings."),
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            k: _field("string", k) for k in ("severity", "location", "issue", "suggestion")}}, "description": "Structured review findings."},
        "unclear": {"type": "array", "items": {"type": "string"}, "description": "Points requiring clarification."}}, [], ["network"], "confirm", []),
    "jv_quiz": ("Run a persistent quiz one question at a time. Grade exact choice answers and explicitly mark alternative/open answers for judgement.", {
        "action": _action("start", "answer", "grade", "status", "results", "stop"),
        "topic": _field("string", "Subject or title."),
        "questions": {"type": "array", "description": "One to twenty questions for start.", "items": {"type": "object", "properties": {
            "type": _field("string", "Question format.", ["multiple_choice", "true_false", "fill_blank", "open"]),
            "question": _field("string", "Question wording."), "answer": _field("string", "Reference answer."),
            "explanation": _field("string", "Explanation after answering."),
            "options": {"type": "array", "items": {"type": "string"}, "description": "Choice labels."}},
            "required": ["question", "answer"]}},
        "answer": _field("string", "The user's answer to the current question."),
        "index": _field("integer", "One-based answered question index, for manual grading."),
        "correct": _field("boolean", "Manual judgement of an answered open question.")}, ["action"], ["filesystem"], "allow", []),
    "jv_pomodoro": ("Start or stop a repeating focus/break timer. Every fourth break is long; stats count actually completed blocks. Timers stop when ORION closes.", {
        "action": _action("start", "stop", "status", "stats"), "task": _field("string", "Focus intention."),
        "work_minutes": _field("number", "Work block, 1–180 minutes; default 25."),
        "break_minutes": _field("number", "Short break, 1–60 minutes; default 5."),
        "long_break_minutes": _field("number", "Every fourth break, 1–90 minutes; default 15.")},
        ["action"], ["filesystem", "audio"], "allow", []),
    "jv_water_reminder": ("Log water intake and explicitly start cancellable reminders towards the user's chosen daily goal; no medical target is inferred.", {
        "action": _action("start", "stop", "log", "reset", "status"),
        "interval_minutes": _field("number", "5–240 minutes between reminders; default 60."),
        "goal_ml": _field("number", "User's daily goal in ml, 250–5,000."),
        "amount_ml": _field("number", "Measured amount drunk, 1–2,000 ml; default 250.")},
        ["action"], ["filesystem", "audio"], "allow", []),
    "jv_excel_writer": ("Create and verify an XLSX workbook with tables, permitted formulas and bar/line charts. Use provided data; unknown figures remain blank.", {
        "request": _field("string", "Workbook brief and supplied source data, if not using a structured specification."),
        "filename": _field("string", "New workbook filename; existing files are never overwritten."),
        "workbook": _field("object", "Structured {sheets:[{name, columns:[...], rows:[[...]], chart:{type:bar|line,title}}]}. Formula cells are {formula:'=SUM(B2:B8)'}. Charts are optional.")},
        [], ["filesystem", "network"], "confirm", ["openpyxl"]),
    "jv_calorie_counter": ("Estimate food nutrition from a user-requested webcam photo or image file through ORION's vision router. Estimates state portion uncertainty and are never measurements.", {
        "query": _field("string", "User's nutrition question."), "path": _field("string", "Optional image file; omit for the selected webcam.")},
        [], ["network", "screen", "filesystem"], "confirm", ["PIL"]),
    "jv_game_updater": ("Inspect, install or update games with the supplied Steam/Epic integration. Launcher/account support is required. Completion is reported from client evidence, not inferred from a click. Unattended PC shutdown is disabled.", {
        "action": _action("list", "download_status", "install", "update", "schedule", "cancel_schedule", "schedule_status"),
        "platform": _field("string", "Gaming client.", ["steam", "epic", "both"]),
        "game_name": _field("string", "User-chosen game title."), "app_id": _field("string", "Steam AppID supplied by the user."),
        "hour": _field("integer", "Daily schedule hour, 0–23."), "minute": _field("integer", "Daily schedule minute, 0–59.")},
        ["action"], ["filesystem", "subprocess", "input", "network"], "confirm", []),
    "jv_chat_takeover": ("Draft replies from a named conversation window; optionally monitor for up to thirty minutes. Sending requires an explicit platform, recipient and final text. No background auto-sending.", {
        "action": _action("draft", "start", "stop", "status", "send"), "window": _field("string", "Conversation window title to read."),
        "query": _field("string", "User's purpose and desired tone."), "duration_minutes": _field("number", "Monitor for 1–30 minutes; default 5."),
        "platform": _field("string", "Platform for send.", ["telegram", "whatsapp", "discord"]),
        "contact": _field("string", "Explicit recipient."), "message": _field("string", "Approved final message text.")},
        ["action"], ["network", "screen", "input"], "confirm", []),
    "jv_telegram_remote": ("Explicit Telegram polling for permitted private chats only. Incoming /tool JSON commands queue for local review; approve uses native dispatch and its safeguards. Never starts during discovery.", {
        "action": _action("status", "start", "stop", "send", "pending", "approve", "reject"),
        "chat_id": _field("string", "Permitted private chat id for send."), "message": _field("string", "Final outgoing message text."),
        "id": _field("string", "Exact queued request id for local approve or reject.")},
        ["action"], ["network", "filesystem"], "confirm", ["aiohttp"]),
    "jv_cloud_agent": ("Connect to a deployed ORION HTTPS node, pair with a fresh code, inspect jobs or delegate a specific supported tool. This does not deploy a server or enable legacy JARVIS protocols.", {
        "action": _action("connect", "disconnect", "status", "delegate", "jobs", "result", "remind", "cancel_reminder", "briefing_now"),
        "url": _field("string", "ORION cloud HTTPS origin, for connect."), "pairing_code": _field("string", "Fresh cloud-node pairing code."),
        "tool": _field("string", "Exact supported ORION tool for delegate."), "arguments": _field("object", "Arguments for the delegated tool."),
        "id": _field("string", "Cloud task/reminder id."), "text": _field("string", "Reminder text."),
        "minutes": _field("number", "Reminder delay."), "at": _field("string", "Absolute reminder time.")},
        ["action"], ["network", "filesystem"], "confirm", ["aiohttp"]),
    "jv_reminder": ("Use ORION's persistent spoken reminders and alarms, replacing the legacy cloud reminder imports.", {
        "action": _action("add", "list", "cancel"), "text": _field("string", "Reminder text."),
        "minutes": _field("number", "Delay in minutes."), "at": _field("string", "Absolute time."),
        "phrase": _field("string", "Natural-language reminder request."), "id": _field("integer", "Reminder id to cancel.")},
        ["action"], ["filesystem", "audio"], "allow", []),
    "jv_forge_trial": ("Propose and review one bounded Forge experiment, run its contract/tests, inspect its result or roll its current generation back. No recursive unattended changes to ORION's core.", {
        "action": _action("propose", "list", "status", "run", "rollback"), "name": _field("string", "Short proposed trial name."),
        "plan": _field("string", "Purpose, measurable acceptance checks and known constraints."),
        "id": _field("string", "Recorded trial id for run/status/rollback.")},
        ["action"], ["network", "filesystem", "subprocess"], "confirm", []),
}


def names() -> tuple[str, ...]:
    return tuple(_CATALOGUE)


def schema(name: str) -> dict:
    description, properties, required, _, _, _ = _CATALOGUE[name]
    return {"name": name, "description": description, "parameters": {
        "type": "object", "properties": copy.deepcopy(properties), "required": list(required)}}


def manifest(name: str) -> dict:
    description, _, _, permissions, tier, requires = _CATALOGUE[name]
    return {"name": name, "description": description, "module": name + "_tool.py",
            "version": "33.0.0", "tier": tier, "permissions": permissions,
            "requires": requires, "parameters": schema(name)["parameters"],
            "secrets": ({"jv_telegram_remote": ["TELEGRAM_BOT_TOKEN"],
                          "jv_cloud_agent": ["ORION_CLOUD_REFRESH_TOKEN"]}.get(name, []))}
