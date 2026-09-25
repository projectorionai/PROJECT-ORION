"""
app_context.py — turns a foreground window title into a specific,
human-meaningful description of what the user is doing right now.

Mark XX architectural-audit pass, Track I: workspace.py already parsed a
VS-Code-specific title shape to recover a repo name, but nothing generalised
that into "what does having THIS app open actually mean" — the audit's own
example was "if VS Code is open, ORION should understand the user is
programming; if Photoshop is open, image editing; if Chess.com is open,
playing chess." This is that generalisation: an ordered table of known
app/site name fragments, most specific first, and a real fallback rather
than silence for anything unmatched — an unrecognised title still yields
"using <app>" from its own text instead of nothing.
"""

from __future__ import annotations

import re

# Ordered (pattern, context) pairs — first match wins, so a specific pattern
# must precede a more general one that could also match the same title
# (e.g. "chrome" would swallow "chess.com" if it came first).
_APP_CONTEXTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"visual studio code|\bvs ?code\b", re.I), "programming"),
    (re.compile(r"pycharm|intellij|clion|rider\b|webstorm|android studio", re.I), "programming"),
    (re.compile(r"sublime text|notepad\+\+|\bvim\b|neovim", re.I), "editing code or text"),
    (re.compile(r"photoshop", re.I), "editing an image"),
    (re.compile(r"illustrator", re.I), "vector illustration"),
    (re.compile(r"premiere pro|davinci resolve|final cut", re.I), "editing video"),
    (re.compile(r"figma|sketch\b", re.I), "designing an interface"),
    (re.compile(r"blender", re.I), "3D modelling"),
    (re.compile(r"chess\.com|lichess", re.I), "playing chess"),
    (re.compile(r"steam\b|epic games|battle\.net", re.I), "gaming"),
    (re.compile(r"\bexcel\b", re.I), "working with a spreadsheet"),
    (re.compile(r"\bword\b", re.I), "writing a document"),
    (re.compile(r"powerpoint", re.I), "building a presentation"),
    (re.compile(r"outlook", re.I), "managing email"),
    (re.compile(r"slack|\bteams\b|discord", re.I), "messaging"),
    (re.compile(r"spotify", re.I), "listening to music"),
    (re.compile(r"youtube", re.I), "watching a video"),
    (re.compile(r"netflix|disney\+|prime video", re.I), "watching something"),
    (re.compile(r"github\.com|gitlab\.com", re.I), "browsing a code repository"),
    (re.compile(r"stack overflow", re.I), "researching a coding problem"),
    (re.compile(r"terminal|powershell|cmd\.exe|command prompt|windows terminal", re.I), "working in a terminal"),
    (re.compile(r"chrome|\bedge\b|firefox|\bopera\b|\bbrave\b", re.I), "browsing the web"),
]


def describe_context(title: str) -> str:
    """A short, natural-language description of what a foreground window
    title implies the user is doing, or '' when the title is blank.

    Never returns a refusal for a non-blank, unrecognised title — degrades
    to "using <app>" from the title's own trailing segment instead, per the
    audit's instruction to attempt an alternative before giving up."""
    title = str(title or "").strip()
    if not title:
        return ""
    for pattern, context in _APP_CONTEXTS:
        if pattern.search(title):
            return context
    parts = [p.strip() for p in re.split(r"\s[-—]\s", title) if p.strip()]
    app = parts[-1] if parts else title
    return f"using {app[:60]}"
