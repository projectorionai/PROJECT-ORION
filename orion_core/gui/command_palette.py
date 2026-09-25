"""
Command palette (Mark X.12 §1.2; unified across tools and navigation in the
Mark XX design-spec §8).

A Ctrl+K fuzzy-searchable list of every tool ORION exposes AND every Command
Deck page/zone — one search surface for "how do I reach X" instead of two.
Before the design-spec pass, Ctrl+K only indexed the ~100 dispatcher tools;
a user who half-remembered "the chess thing" or a page name had no search
path once it left the tab bar (worse still once the Toolkit/OPS
redistribution, §3/§4, meant those surfaces don't keep a permanent tab).

Choosing a TOOL entry *prefills the message box* with that tool's name and
focuses it — it never fires a tool directly, so a side-effectful action
still goes through the normal send/confirm path. Choosing a PAGE entry
navigates the Command Deck straight there instead, since opening a page has
no side effect worth confirming.

The command-model and ranking logic are pure and unit-tested; the QDialog is a
thin shell over them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

# Palette entry kinds — what happens when you choose one (see CommandPalette
# .command_chosen below).
#
#   KIND_PAGE    navigates the Command Deck straight there.
#   KIND_ACTION  RUNS a real command through the command router, right now.
#   KIND_TOOL    a dispatcher tool: runs it, after confirmation when the tool
#                can change something.
#
# KIND_ACTION exists because the palette was, in the user's words, "largely
# decorative": every entry only ever prefilled a text box for the user to
# then send themselves. An entry labelled "System Scan" that types the words
# "system scan" into a box is a label pretending to be a control. Choosing an
# entry now performs the thing it names — see orion_core.command_router.
KIND_TOOL = "tool"
KIND_PAGE = "page"
KIND_ACTION = "action"


@dataclass(frozen=True)
class PaletteCommand:
    name: str            # dispatcher tool name, or deck page label
    title: str           # human-facing title (name with spaces)
    subtitle: str        # one-line description
    kind: str = KIND_TOOL


def _first_sentence(text: str, limit: int = 100) -> str:
    text = " ".join(str(text or "").split())
    for sep in (". ", " — ", "; ", " ("):
        idx = text.find(sep)
        if 0 < idx <= limit:
            return text[:idx]
    return text[:limit].rstrip()


def build_commands(tool_declarations: Iterable[dict]) -> list[PaletteCommand]:
    """Turn TOOL_DECLARATIONS into palette entries, sorted by tool name."""
    commands: list[PaletteCommand] = []
    for tool in tool_declarations:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        commands.append(PaletteCommand(
            name=name,
            title=name.replace("_", " "),
            subtitle=_first_sentence(tool.get("description") or ""),
            kind=KIND_TOOL,
        ))
    return sorted(commands, key=lambda c: c.name)


def build_page_commands(page_names: Iterable[str]) -> list[PaletteCommand]:
    """Turn the Command Deck's page labels into palette entries.

    Sourced from UnifiedDashboard.page_names() at call time, not a
    hardcoded copy — the palette can never list a page that no longer
    exists, or miss one that was added.
    """
    commands: list[PaletteCommand] = []
    for name in page_names:
        label = str(name or "").strip()
        if not label:
            continue
        commands.append(PaletteCommand(
            name=label,
            title=label.title(),
            subtitle="Command Deck page",
            kind=KIND_PAGE,
        ))
    return sorted(commands, key=lambda c: c.name)


def build_action_commands(
        entries: Iterable[tuple[str, str, str, bool]]) -> list[PaletteCommand]:
    """Turn CommandRouter.palette_entries() into palette entries.

    These are the entries that genuinely DO something when chosen: each one
    resolves to a handler bound to a real subsystem.  They sort first in the
    results because "run the thing" is almost always what the user meant when
    they typed its name.
    """
    commands: list[PaletteCommand] = []
    for command_id, title, subtitle, destructive in entries:
        commands.append(PaletteCommand(
            name=str(command_id),
            title=str(title),
            subtitle=(str(subtitle) + (" (asks first)" if destructive else "")),
            kind=KIND_ACTION,
        ))
    return sorted(commands, key=lambda c: c.name)


def _match_score(query: str, target: str) -> Optional[float]:
    """Score how well *query* matches *target*, or None if it doesn't.

    A contiguous substring scores far higher than a scattered in-order
    subsequence; matches at a word boundary and shorter targets score higher
    still. Deterministic, so ranking is stable and testable.
    """
    q = query.lower()
    t = target.lower()
    if not q:
        return 0.0
    if q in t:
        idx = t.find(q)
        boundary = 10.0 if idx == 0 or t[idx - 1] in " _-" else 0.0
        return 100.0 + boundary - idx * 0.1 - len(t) * 0.01
    # fallback: every query char appears in order somewhere in the target
    ti = 0
    for qc in q:
        ti = t.find(qc, ti)
        if ti == -1:
            return None
        ti += 1
    return 10.0 - len(t) * 0.01


def rank_commands(query: str, commands: Sequence[PaletteCommand],
                  limit: int = 60) -> list[PaletteCommand]:
    """Return the commands matching *query*, best first. Empty query = all."""
    query = str(query or "").strip()
    if not query:
        return list(commands)[:limit]
    scored: list[tuple[float, PaletteCommand]] = []
    for cmd in commands:
        best: Optional[float] = None
        for field in (cmd.name, cmd.title, cmd.subtitle):
            score = _match_score(query, field)
            if score is not None:
                best = score if best is None else max(best, score)
        if best is not None:
            scored.append((best, cmd))
    scored.sort(key=lambda pair: (-pair[0], pair[1].name))
    return [cmd for _, cmd in scored[:limit]]


class CommandPalette(QDialog):
    """Fuzzy launcher over both the dispatcher tool set and the Command Deck's
    pages. Emits (kind, name) — the caller decides what a tool vs. a page
    choice does (prefill vs. navigate); this dialog only ranks and picks.
    """

    command_chosen = pyqtSignal(str, str)   # (kind, name)

    # A small glyph per kind so a mixed results list stays scannable — a
    # page entry should read as "this navigates" at a glance, not require
    # reading the subtitle to tell it apart from a tool entry.
    _KIND_GLYPH = {KIND_PAGE: "⧉", KIND_TOOL: "▸", KIND_ACTION: "⏵"}

    def __init__(self, commands: Sequence[PaletteCommand], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Command palette")
        self.setModal(True)
        self.resize(560, 420)
        self._commands = list(commands)

        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "Search ORION's commands and pages…  (Esc to close)")
        self.results = QListWidget()
        layout.addWidget(self.search)
        layout.addWidget(self.results)

        self.search.textChanged.connect(self._refilter)
        self.search.returnPressed.connect(self._choose_current)
        self.results.itemActivated.connect(lambda _item: self._choose_current())
        self._refilter("")
        self.search.setFocus()

    def _refilter(self, text: str) -> None:
        self.results.clear()
        for cmd in rank_commands(text, self._commands):
            glyph = self._KIND_GLYPH.get(cmd.kind, "▸")
            item = QListWidgetItem(f"{glyph} {cmd.title}\n    {cmd.subtitle}")
            item.setData(Qt.ItemDataRole.UserRole, (cmd.kind, cmd.name))
            self.results.addItem(item)
        if self.results.count():
            self.results.setCurrentRow(0)

    def _choose_current(self) -> None:
        item = self.results.currentItem() or (
            self.results.item(0) if self.results.count() else None)
        if item is None:
            return
        kind, name = item.data(Qt.ItemDataRole.UserRole)
        self.command_chosen.emit(str(kind), str(name))
        self.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override name)
        # Arrow keys from the search field navigate the results list.
        key = event.key()
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.results.count():
            row = self.results.currentRow()
            if key == Qt.Key.Key_Down:
                row = min(row + 1, self.results.count() - 1)
            else:
                row = max(row - 1, 0)
            self.results.setCurrentRow(row)
            return
        super().keyPressEvent(event)
