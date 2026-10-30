"""
House style for the words ORION puts on screen.

Names are written in Title Case: window titles, pages, sections, tabs, group
boxes, menu entries, buttons, column headers. Sentences are not: status lines,
empty states, help and tooltips keep ordinary sentence case, because a full
sentence in Title Case reads as shouting.

``title_case`` exists so the few places that BUILD a name at run time (a tool
called ``web_search`` becoming a palette entry, a page key becoming a tab) get
exactly the same casing as the labels that are written out by hand, rather than
Python's ``str.title()`` which capitalises "Of", "To" and "Wi-Fi"'s tail, and
flattens ORION to Orion.
"""

from __future__ import annotations

import re

#: Short words left lower-case inside a name (but never first or last).
_MINOR = frozenset({
    "a", "an", "the", "and", "or", "nor", "but", "of", "to", "in", "on",
    "for", "with", "at", "by", "from", "as", "vs", "via",
})

#: Words that are always written in capitals. A tool called ``ai_mode`` is
#: "AI Mode" in a menu, not "Ai Mode".
_ACRONYMS = {
    "ai": "AI", "mcp": "MCP", "ui": "UI", "url": "URL", "id": "ID", "gpu": "GPU",
    "cpu": "CPU", "pdf": "PDF", "ocr": "OCR", "api": "API", "tts": "TTS", "qr": "QR",
    "pcb": "PCB", "kpi": "KPI", "kpis": "KPIs", "roi": "ROI", "roas": "ROAS", "usb": "USB", "nnue": "NNUE", "elo": "ELO", "ram": "RAM",
}

_WORD = re.compile(r"[^\s]+")


def _cap(piece: str) -> str:
    """Capitalise the first letter of *piece*, leaving the rest untouched."""
    for index, char in enumerate(piece):
        if char.isalpha():
            return piece[:index] + char.upper() + piece[index + 1:]
        if char.isdigit():
            return piece                 # "3-D", "2D": already as written
    return piece


def _title_word(word: str, *, force: bool) -> str:
    # A word that already carries a capital after its first letter is a brand
    # or an acronym (ORION, WhatsApp, NNUE, iPhone): it is left exactly as is.
    letters = [c for c in word if c.isalpha()]
    if len(letters) > 1 and any(c.isupper() for c in letters[1:]):
        return word
    core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", word)
    if core.lower() in _ACRONYMS and core == core.lower():
        return word.replace(core, _ACRONYMS[core.lower()], 1)
    if not force and core.lower() in _MINOR and core == core.lower():
        return word
    if "-" in core:
        # "deep-dive" -> "Deep-Dive"; "Wi-Fi" never gets here (internal capital).
        head, *tail = word.split("-")
        return "-".join([_cap(head)] + [_cap(part) for part in tail])
    return _cap(word)


def title_case(text: str) -> str:
    """*text* as a name: major words capitalised, minor words left alone.

    Idempotent, and it never touches a word that already has an internal
    capital, so ``title_case("Check ORION health")`` is ``"Check ORION Health"``.
    """
    text = str(text or "")
    words = list(_WORD.finditer(text))
    if not words:
        return text
    out: list[str] = []
    last = 0
    final = len(words) - 1
    after_colon = True
    for position, match in enumerate(words):
        out.append(text[last:match.start()])
        word = match.group(0)
        force = position == 0 or position == final or after_colon
        out.append(_title_word(word, force=force))
        after_colon = word.endswith(":") or word in ("—", "–")
        last = match.end()
    out.append(text[last:])
    return "".join(out)


def plural(count: int | float, singular: str, plural_form: str | None = None) -> str:
    """'1 card', '0 cards', '2 cards': the number is always shown, and the noun
    agrees with it. (The deck used to say "1 cards" and "1 distraction(s)".)"""
    word = singular if count == 1 else (plural_form or singular + "s")
    shown = f"{count:g}" if isinstance(count, float) else str(count)
    return f"{shown} {word}"


#: What a deck page is CALLED on screen, where that differs from its key. The
#: keys stay stable (voice navigation, saved layouts, tests and the swarm all
#: use them); only the words a person reads change. One table, because the
#: sidebar, the header, the Command Palette, the Overview's quick links and the
#: "opening…" placeholder all name the same page and must say the same thing.
PAGE_DISPLAY_NAMES = {
    "BRAIN": "Brain",
    "WORKBENCH": "Vision Lab",
    "RESEARCH": "Research Console",
    "COMMAND CENTRE": "Command Centre",
    "OPS": "Operations",
    "COGNITION": "Toolkit",
}


def page_display_name(name: str) -> str:
    """The on-screen name of the deck page whose key is *name*."""
    key = str(name)
    return PAGE_DISPLAY_NAMES.get(key.upper()) or title_case(key.lower())


__all__ = ["PAGE_DISPLAY_NAMES", "page_display_name", "plural", "title_case"]
