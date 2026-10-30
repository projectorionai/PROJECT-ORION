"""
ORION reading and pressing his own Command Deck (Mark XXXVII.4).

interface_control could only OPEN a deck page. Asked "show me where that is
on the Command Deck" (owner, 26 Oct 2026) he had no way to know what any
page holds, and nothing on a page could be pressed for him. This module is
the read side and the press side, over the deck's real widgets - the words a
person reads there, not pixels:

  read_page   the headings, labels, buttons and list rows a page shows;
  find        which page holds some words (its name, its section, or - for
              a page that has been opened - its contents);
  locate      the one button on a page that a label names.

Pressing is for ordinary controls. A button that deletes, forgets, approves,
sends, pays, signs out or shuts something down is never pressed here: those
stay the owner's own click, or go through the tool that asks him first.

Runs on the GUI thread only (core_window answers a bus request with it).
"""

from __future__ import annotations

import re
from typing import Any

from PyQt6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QTabBar,
    QToolButton,
    QWidget,
)

#: What one reading returns at most, so a busy page cannot flood a turn.
MAX_LINES = 60
MAX_CHARS = 1800

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

#: A control ORION must not press for anyone: irreversible, or the owner's
#: own decision. Matched on its label, accessible name and tooltip.
RISKY_RE = re.compile(
    r"(?i)\b(?:delete|remove|forget|erase|wipe|purge|clear\s+all|reset|uninstall|"
    r"quit|exit|shut\s*down|restart|reboot|power|format|approve|allow|deny|decline|"
    r"accept|confirm|send|submit|post|publish|pay|buy|purchase|order|sign\s*(?:out|off)|"
    r"log\s*out|revoke|kill|terminate|end\s+task|overwrite|replace\s+all|discard)\b")


def _clean(text: Any, limit: int = 160) -> str:
    raw = str(text or "")
    raw = _TAG_RE.sub(" ", raw).replace("&nbsp;", " ").replace("&amp;", "&")
    return _SPACE_RE.sub(" ", raw).strip()[:limit]


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", _clean(text).lower()).strip()


def _words(text: Any) -> list[str]:
    return [w for w in _norm(text).split() if len(w) > 1]


def _button_text(button: QAbstractButton) -> str:
    """A button's text without its keyboard mnemonic ("&Save" -> "Save")."""
    return re.sub(r"&(.)", r"\1", str(button.text() or ""))


def _button_label(button: QAbstractButton) -> str:
    return (_clean(_button_text(button)) or _clean(button.accessibleName())
            or _clean(button.toolTip(), 60))


def visible_text(page: QWidget, limit: int = MAX_LINES) -> list[str]:
    """The text a person reads on *page*, in the order it was built: labels,
    headings, buttons (marked), the chosen item of each list box and the
    first rows of lists. Hidden widgets are not read; password fields never."""
    lines: list[str] = []
    seen: set[str] = set()

    def add(text: str, prefix: str = "") -> None:
        text = _clean(text)
        if not text or text in seen:
            return
        seen.add(text)
        lines.append(f"{prefix}{text}")

    for widget in page.findChildren(QWidget):
        if len(lines) >= limit:
            break
        try:
            if not widget.isVisibleTo(page):
                continue
            if isinstance(widget, QLabel):
                add(widget.text())
            elif isinstance(widget, QAbstractButton):
                add(_button_label(widget), "[button] ")
            elif isinstance(widget, QGroupBox):
                add(widget.title())
            elif isinstance(widget, QComboBox):
                add(widget.currentText(), "[choice] ")
            elif isinstance(widget, QLineEdit):
                if widget.echoMode() == QLineEdit.EchoMode.Normal:
                    add(widget.text() or widget.placeholderText(), "[field] ")
            elif isinstance(widget, QListWidget):
                for row in range(min(8, widget.count())):
                    item = widget.item(row)
                    if item is not None and not item.isHidden():
                        add(item.text(), "- ")
            elif isinstance(widget, QTabBar):
                for index in range(widget.count()):
                    add(widget.tabText(index), "[tab] ")
        except RuntimeError:
            continue                     # deleted while we walked the tree
    return lines[:limit]


def _page_names(deck: Any) -> list[str]:
    try:
        return list(deck.page_names())
    except Exception:
        return []


def _display(deck: Any, name: str) -> str:
    try:
        return str(deck.display_name(name))
    except Exception:
        return name.title()


def read_page(deck: Any, name: str) -> tuple[bool, str]:
    """(ok, what the page *name* shows) - built if it had not been yet."""
    widget = deck.page_widget(name)
    if widget is None:
        return False, f"The {_display(deck, name)} page could not be opened."
    lines = visible_text(widget)
    title = _display(deck, name)
    if not lines:
        return True, (f"The {title} page is open. It shows a picture or a live view "
                      "rather than text I can read out.")
    body, used = [], 0
    for line in lines:
        if used + len(line) > MAX_CHARS:
            body.append("…")
            break
        body.append(line)
        used += len(line) + 2
    return True, f"The {title} page shows:\n" + "\n".join(body)


def find(deck: Any, query: str) -> tuple[bool, str, str | None]:
    """(ok, what was found, the best page or None) for *query*.

    A page counts when its name, label, section or zone holds every word of
    the query; a page that has been opened also counts for its contents.
    Pages never opened are searched by name only, and the answer says so."""
    words = _words(query)
    if not words:
        return False, "Tell me what to look for on the Command Deck.", None
    hits: list[tuple[int, str, str]] = []
    unbuilt = 0
    for name in _page_names(deck):
        try:
            terms = _norm(deck.page_terms(name))
        except Exception:
            terms = _norm(f"{name} {_display(deck, name)}")
        score = 0
        if all(word in terms for word in words):
            score = 3
        snippet = ""
        widget = deck.live_page(name) if hasattr(deck, "live_page") else None
        if widget is None:
            unbuilt += 1
        else:
            for line in visible_text(widget, limit=200):
                low = _norm(line)
                if all(word in low for word in words):
                    snippet = line
                    score = max(score, 2)
                    break
                if not snippet and sum(word in low for word in words) * 2 > len(words):
                    snippet = line
                    score = max(score, 1)
        if score:
            hits.append((score, name, snippet))
    if not hits:
        pages = "page has" if unbuilt == 1 else "pages have"
        extra = (f" {unbuilt} {pages} not been opened yet, so only their names "
                 "were searched." if unbuilt else "")
        return False, f"Nothing on the Command Deck matches {query!r}.{extra}", None
    hits.sort(key=lambda hit: -hit[0])
    best = hits[0][1]
    parts = []
    for score, name, snippet in hits[:4]:
        parts.append(f"{_display(deck, name)}" + (f" (\"{snippet}\")" if snippet else ""))
    if len(hits) == 1 or hits[0][0] > hits[1][0]:
        message = f"It's on the {parts[0]} page - I've opened it."
        if len(hits) > 1:
            message += " Also mentioned on: " + "; ".join(parts[1:]) + "."
    else:
        message = ("It could be on: " + "; ".join(parts)
                   + f". I've opened {_display(deck, best)}.")
    return True, message, best


def locate(deck: Any, label: str, page: str | None) -> tuple[bool, str, QAbstractButton | None, str | None]:
    """(ok, message, the button, its page) for the button *label* names.

    Looked for on *page*, else on the page on screen, else on every page that
    has been opened. Refused when the button is risky (RISKY_RE), disabled,
    opens a menu, or more than one different button matches."""
    want = _norm(label)
    if not want:
        return False, "Tell me which button to press.", None, None
    order: list[str] = []
    if page:
        order.append(page)
    else:
        current = deck.current_page_name() if hasattr(deck, "current_page_name") else None
        if current:
            order.append(current)
        order += [n for n in _page_names(deck) if n not in order]
    for name in order:
        widget = deck.page_widget(name) if name == page else (
            deck.live_page(name) if hasattr(deck, "live_page") else None)
        if widget is None:
            continue
        buttons = [b for b in widget.findChildren(QAbstractButton) if b.isVisibleTo(widget)]
        exact = [b for b in buttons
                 if want in (_norm(_button_text(b)), _norm(b.accessibleName()))]
        found = exact or [b for b in buttons
                          if want in _norm(_button_text(b)) or want in _norm(b.accessibleName())]
        if not found:
            continue
        labels = sorted({_button_label(b) for b in found})
        if len(labels) > 1:
            return (False, f"More than one button on the {_display(deck, name)} page matches "
                    f"{label!r}: " + ", ".join(labels[:6]) + ". Which one?", None, name)
        button = found[0]
        shown = _button_label(button)
        described = " ".join((_button_text(button), button.accessibleName(), button.toolTip()))
        if RISKY_RE.search(_clean(described, 400)):
            return (False, f"I won't press {shown!r} for you - it can't be undone or it is "
                    "your decision. Press it yourself, or ask me to do it through the "
                    "tool that checks with you first.", None, name)
        if not button.isEnabled():
            return False, f"{shown!r} on the {_display(deck, name)} page is greyed out right now.", None, name
        if isinstance(button, QToolButton) and button.menu() is not None:
            return (False, f"{shown!r} opens a menu; tell me which item you want instead.",
                    None, name)
        return True, f"Pressed {shown!r} on the {_display(deck, name)} page.", button, name
    where = f"the {_display(deck, page)} page" if page else "any open Command Deck page"
    return False, f"There is no {label!r} button on {where}.", None, None


__all__ = ["MAX_CHARS", "MAX_LINES", "RISKY_RE", "find", "locate", "read_page", "visible_text"]
