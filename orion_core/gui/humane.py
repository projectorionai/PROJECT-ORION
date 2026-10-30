"""
The activity feed, said like a person would say it.

The Core Window's activity log printed ORION's engineering log as it was
written for an engineer:

    NET: gemini live channel synchronised.
    AUDIO: microphone pipeline initialised — listening on [1] Microphone (2- HD Pro Webcam C9.
    PERF: event loop blocked 250 ms by close() at sounddevice.py:1168
    MCP[orion_world]: handshake failed - server closed the connection (...)
    TOOLS: geo → weather (412 ms)

with the owner's own conversation buried among it. The owner's verdict on
the interface was "stale and boring"; this is most of why. The full log is
still on the Command Deck's LOG page and in the journals - nothing is lost -
but the window he looks at most now reads as a conversation:

  * what he said and what ORION said, first and clearly;
  * the events a person would care about, in a sentence ("Connected - ready
    to talk", "Lost the connection for a moment - reconnecting…");
  * engineering chatter (timings, boot steps, debug traces, sub-systems
    talking to themselves) left to the full log.

Pure: no Qt. ``humanise(line)`` returns a ``Said`` or None (leave it out).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Said:
    """One line of the human feed."""

    kind: str          # "you" | "orion" | "event" | "trouble"
    text: str
    label: str = ""    # a short name for an event's source ("Reminder", "Phone")


#: Prefixes that are conversation.
_YOU = re.compile(r"^YOU(?:\s*\(([^)]*)\))?\s*:\s*(.*)$", re.S)
_ORION = re.compile(r"^ORION(?:\s*\(([^)]*)\))?\s*:\s*(.*)$", re.S)
_PREFIX = re.compile(r"^([A-Z][A-Z0-9_]{1,19})(?:\[([^\]]*)\])?\s*:\s*(.*)$", re.S)

#: Sub-systems whose lines are engineering, not news: they stay in the full log.
QUIET = frozenset({
    "PERF", "DEBUG", "BOOT", "GUI", "DECK", "TRACE", "SR", "IMPROVE", "REPAIR",
    "SERVER", "PLUGIN", "SYNC", "TEMPORAL", "JOB", "EXEC", "REASON", "STRATEGY",
    "TELEMETRY", "LATENCY", "CACHE", "INDEX", "SEMANTIC", "LIBRARY", "GRAPH",
    "MCP", "FORGE", "EVOLVE", "PACK", "LIT", "STUDIO", "WORKFLOW", "SECURITY_RECON",
    "PROVIDER", "ROUTER", "TOKENS", "BRAIN", "INTENT", "SHADOW", "REFLEX",
    "PERCEPTION", "GESTURE", "DIAG", "HEALTH", "CONSOLE", "SANDBOX", "PRESENCE",
})

#: How the sources a person cares about are named in the feed.
LABELS = {
    "REMINDER": "Reminder", "MESSAGING": "Messages", "RESEARCH": "Research",
    "PLAN": "Plan", "FILE": "Files", "APP": "Apps", "VISION": "Vision",
    "GLOBE": "Globe", "CHESS": "Chess", "REMOTE": "Phone", "SENTINEL": "Heads-up",
    "SECURITY": "Security", "SEC": "Security", "TOOL": "Done", "TOOLS": "Done",
    "MEM": "Memory", "MEMORY": "Memory", "LEARN": "Learned", "VOICE": "Voice",
    "PERIPHERALS": "Devices", "REPORT": "Report", "SYS": "", "NET": "", "AUDIO": "",
}

#: Things that went wrong, said as such.
_TROUBLE = re.compile(r"\b(fail(ed|ure)?|error|could not|couldn't|unavailable|refused|"
                      r"denied|timed out|not responding)\b", re.IGNORECASE)


def _sentence(text: str) -> str:
    text = " ".join(str(text or "").split()).strip(" -—")
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _net(text: str) -> Said | None:
    low = text.lower()
    if "synchronised" in low or "connected" in low and "dis" not in low:
        return Said("event", "Connected - ready to talk.")
    if "dropped" in low or "reconnect" in low or "resuming" in low:
        return Said("trouble", "Lost the connection for a moment - reconnecting…")
    if "offline" in low:
        return Said("trouble", "I'm offline just now - working locally.")
    return None                                # initialising, error codes, traces


_MIC = re.compile(r"listening on\s*(?:\[\d+\]\s*)?(.+?)\.?$", re.IGNORECASE)


def _audio(text: str) -> Said | None:
    low = text.lower()
    if "microphone pipeline initialised" in low:
        found = _MIC.search(text)
        device = found.group(1).strip() if found else ""
        device = re.sub(r"\(\d+-\s*", "(", device).rstrip(". ")   # "(2- HD Pro Webcam"
        if "(" in device and ")" not in device:
            # Windows cuts device names at 31 characters, mid-bracket:
            # "Microphone (HD Pro Webcam C9" - say the part that names it.
            device = device.split("(", 1)[1].strip() + "…"
        if not device:
            return Said("event", "Microphone ready.")
        return Said("event", f"Listening through {device}" + ("" if device.endswith("…") else "."))
    if "no microphone" in low or "would not open" in low or "could not open" in low:
        return Said("trouble", "I can't hear through the microphone right now.")
    if "switched" in low or "fell back" in low or "default microphone" in low:
        return Said("event", _sentence(text))
    return None


#: What ORION is doing when he reaches for a tool, said as an activity.
TOOL_PHRASES = {
    "geo": "Checking the map and the weather", "weather": "Checking the weather",
    "web_search": "Searching the web", "research": "Researching",
    "open_app": "Opening an app", "close_app": "Closing an app",
    "media_control": "Seeing to the music", "peripherals": "Adjusting your devices",
    "reminder": "Setting a reminder", "calculate": "Working it out",
    "save_memory": "Making a note", "query_intelligence": "Checking what I remember",
    "recall_conversation": "Looking back over our conversation",
    "conversation_recall": "Looking back over our conversation",
    "find_files": "Looking for your files", "file_controller": "Working with your files",
    "organise_files": "Tidying your files", "vision_analyse": "Taking a look",
    "capture_screen": "Taking a screenshot", "outlook_mail": "Checking your email",
    "notion_workspace": "Checking your calendar and tasks",
    "morning_briefing": "Putting your briefing together", "open_news": "Opening the news",
    "desktop_control": "Working on your desktop", "web_control": "Working in the browser",
    "browser_control": "Working in the browser", "window_control": "Arranging your windows",
    "messaging": "Sending a message", "telephony": "Placing a call", "chess": "Playing chess",
    "globe": "Turning the globe", "flight_search": "Looking for flights",
    "finance": "Looking at your finances", "focus": "Setting up focus time",
    "diagnostics": "Checking on myself", "find_tool": "Finding the right tool",
    "use_tool": "Using a specialist tool", "process_file": "Reading the file",
    "read_documents": "Reading your documents", "translate": "Translating",
}

_TOOL_LINE = re.compile(r"^(?P<name>[a-z_][a-z0-9_]*)\s+(?P<what>requested|failed)\b\s*[-:]?\s*(?P<why>.*)$",
                        re.IGNORECASE | re.S)


def _tool(text: str) -> Said | None:
    match = _TOOL_LINE.match(text.strip())
    if not match:
        return None                                       # cancellations, channel notes: full log
    name = match.group("name").lower()
    phrase = TOOL_PHRASES.get(name, "Using " + name.replace("_", " "))
    if match.group("what").lower() == "failed":
        why = _sentence(match.group("why"))[:120]
        return Said("trouble", f"{phrase} didn't work{': ' + why if why else '.'}")
    return Said("event", f"{phrase}…")


def humanise(line: str) -> Said | None:
    """The feed's version of one log line, or None to leave it out."""
    text = str(line or "").strip()
    if not text:
        return None
    match = _YOU.match(text)
    if match:
        said = _sentence(match.group(2))
        if not said or said.lower() in {"[wake]", "wake"}:
            return None
        return Said("you", said)
    match = _ORION.match(text)
    if match:
        said = _sentence(match.group(2))
        return Said("orion", said) if said else None
    match = _PREFIX.match(text)
    if not match:
        return Said("event", _sentence(text))           # not a log line: shown as it is
    source, _detail, body = match.group(1), match.group(2), match.group(3)
    if source in QUIET:
        return None
    if source == "NET":
        return _net(body)
    if source == "AUDIO":
        return _audio(body)
    if source in {"TOOL", "TOOLS"}:
        return _tool(body)
    if source in {"MEM", "MEMORY"}:
        noted = re.match(r"synchronised\s+\S+?/(\S+)", body.strip(), re.IGNORECASE)
        if noted:
            what = noted.group(1).replace("_", " ").strip()
            return Said("event", f"Noted something about your {what}.", "Memory")
    label = LABELS.get(source)
    if label is None:
        return None                                       # an unknown sub-system: the full log
    body = _sentence(body)
    if not body:
        return None
    kind = "trouble" if _TROUBLE.search(body) else "event"
    return Said(kind, body, label)


__all__ = ["LABELS", "QUIET", "Said", "heading_case", "humanise"]


# ── headings in sentence case ────────────────────────────────────────────────

#: Words that stay in capitals when a heading is put into sentence case.
ACRONYMS = frozenset({
    "ORION", "AI", "CPU", "GPU", "RAM", "PC", "OS", "QR", "URL", "PDF", "MCP", "API",
    "UK", "US", "EU", "TV", "VPN", "GB", "MB", "KB", "FPS", "IP", "SMS", "PCB", "UI",
    "ID", "LLM", "ML", "NFC", "USB", "CCTV", "DNS", "HTTP", "HTTPS", "SQL", "ETA", "FAQ",
    "NASA", "USGS", "NOAA", "ISS", "GPS", "BBC", "FTSE", "EV", "XP", "MK", "II", "III", "KPI", "ROI", "ROAS",
})

#: Labels whose text is a heading (style.py).
HEADING_NAMES = frozenset({"panelHeading", "titleLabel", "sidebarCaption", "sectionTitle",
                           "cardTitle", "pageTitle", "workspaceTitle", "workbenchTitle",
                           "summaryKey"})

_HEADING_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def sentence_case(text: str) -> str:
    """'WAITING FOR YOUR APPROVAL' -> 'Waiting for your approval';
    'ORION AI CORE' -> 'ORION AI core'. Text that is not shouted is left as it is."""
    letters = [c for c in str(text or "") if c.isalpha()]
    if len(letters) < 2 or not all(c.isupper() for c in letters):
        return text

    first = [True]

    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.upper() in ACRONYMS:
            first[0] = False
            return word.upper()
        out = word.lower()
        if first[0]:
            out = out[:1].upper() + out[1:]
            first[0] = False
        return out
    return _HEADING_WORD.sub(fix, text)


def heading_case(text: str) -> str:
    """A heading as a NAME, in Title Case: 'DEEP FOCUS' and 'Deep focus' both
    become 'Deep Focus', 'STUDY & RECALL' 'Study & Recall', 'ORION AI CORE'
    'ORION AI Core'. (Mark XXXV put shouted headings in sentence case; the owner
    then asked for Title Case on every name - Mark XXXVII.)"""
    from .labels import title_case
    return title_case(sentence_case(text))


def apply_heading_voice(root) -> int:
    """Title Case every heading label in *root* (a window or a page);
    returns how many it changed."""
    try:
        from PyQt6.QtWidgets import QLabel
        labels = list(root.findChildren(QLabel))
    except Exception:
        return 0
    if isinstance(root, QLabel):
        labels.append(root)
    changed = 0
    for label in labels:
        try:
            if label.objectName() in HEADING_NAMES:
                text = label.text()
                better = heading_case(text)
                if better != text:
                    label.setText(better)
                    changed += 1
        except Exception:
            pass
    return changed


def install_heading_voice(app) -> object | None:
    """Put every heading label into Title Case.

    Headings across the Command Deck were literal capitals, letter-spaced
    ("RECENT ACTIONS", "LIVE CHANNEL"), then sentence case ("Deep focus"); the
    owner wants names in Title Case ("Deep Focus"). The literals are Title
    Case at rest; this catches the ones built at run time - every window as
    it gains focus and every Command Deck page as it is shown
    (UnifiedDashboard._select). Returns the hook (keep a reference) or None
    without Qt.

    28 Oct 2026: this was an event filter on the whole application, which
    ran Python for every Qt event in the process (paints, timers, mouse
    moves) to act on one kind; it was 31% of the time a heavy deck page
    took to open."""
    try:
        from PyQt6.QtCore import QObject, QTimer
    except Exception:
        return None

    class _HeadingVoice(QObject):
        def __init__(self, parent) -> None:
            super().__init__(parent)
            self.last = None

        def focus_moved(self, _old, new) -> None:
            try:
                window = new.window() if new is not None else None
            except Exception:
                return
            if window is not None and window is not self.last:
                self.last = window
                apply_heading_voice(window)

        def sweep(self) -> None:
            for window in app.topLevelWidgets():
                apply_heading_voice(window)

    voice = _HeadingVoice(app)
    app.focusChanged.connect(voice.focus_moved)
    QTimer.singleShot(0, voice.sweep)
    return voice
