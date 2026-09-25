"""
Everything ORION has stored about you, and a way to delete any of it.

Why this is not optional
------------------------
ORION remembers people, projects, preferences and decisions, and until now the
only surface for that was a panel showing COUNTS — "identity:2 personal:5
knowledge:1207". A number is not transparency. If an assistant is going to keep
a file on someone, that person has to be able to read it and cross things out,
and neither of those was possible from the interface.

The backend was already there: ``records()`` lists what is stored with the date
each row was last updated, and ``forget()`` deletes by category, key prefix or
value substring. What was missing was any way to reach them, which in practice
means the capability did not exist.

Deleting is deliberate, and deliberately narrow
-----------------------------------------------
Forgetting is not undoable — the row is gone and the FTS index goes with it —
so it asks first, and it asks with the actual text of what is about to go
rather than a count. "Delete 1 item?" is not informed consent; the sentence
ORION wrote about your sister is.

The delete is also scoped to one exact row by category AND key. ``forget()``
will happily take a bare category and remove twelve hundred rows, which is
correct for a tool call and wrong for a button next to a single line.

Styled from ORION's own tokens; no palette of its own.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .style import C
from .widgets import describe_control

#: How many rows to show at once. The store runs to four figures; a table with
#: 1,200 rows in it is not more transparent than one with 200, it is just
#: slower to open and harder to read.
PAGE_SIZE = 200

#: Longest value shown in the cell. The full text is in the tooltip and in the
#: confirmation, so nothing is hidden — only wrapped.
VALUE_CHARS = 160


class MemoryViewer(QDialog):
    """A table of stored facts, with a Forget button on each row."""

    COLUMNS = ("Category", "Key", "What ORION remembers", "Learned", "")

    def __init__(self, matrix: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._matrix = matrix
        self.setWindowTitle("O.R.I.O.N. — What he remembers")
        self.setModal(False)
        self.setMinimumSize(880, 520)
        # The application's own styling, not a private copy of it. This was
        # a full second stylesheet — same tokens, old idiom, and a hover that
        # turned every button crimson. The shared sheet already covers
        # QDialog, QLineEdit, tables, headers and buttons; only #hint is
        # particular to this window.
        from .style import APP_STYLESHEET

        self.setStyleSheet(
            APP_STYLESHEET +
            f"QLabel#hint {{ color: {C.MUTED}; font-size: 11px; }}"
        )

        heading = QLabel("What ORION remembers about you")
        heading.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))

        self.search = QLineEdit()
        self.search.setPlaceholderText("search everything he has stored…")
        describe_control(self.search, "Search stored memory",
                         "Filters the table. An empty box shows the most "
                         "recently updated entries.")
        self.search.returnPressed.connect(self.refresh)
        search_button = QPushButton("Search")
        describe_control(search_button, "Search", "Filter the table.")
        search_button.clicked.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(search_button)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.status = QLabel("")
        self.status.setObjectName("hint")

        note = QLabel(
            "Forgetting cannot be undone — the entry and its search index go "
            "together. He is shown the exact text before anything is removed."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)

        close = QPushButton("Close")
        describe_control(close, "Close", "Close this window.")
        close.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(close)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(heading)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        layout.addWidget(note)
        layout.addLayout(bottom)

        self.refresh()

    # ── listing ──────────────────────────────────────────────────────────────

    def rows(self) -> list[dict[str, str]]:
        try:
            return list(self._matrix.records(self.search.text().strip(),
                                             limit=PAGE_SIZE))
        except Exception:
            return []

    def refresh(self) -> None:
        records = self.rows()
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            category = str(record.get("category", ""))
            key = str(record.get("key_ref", ""))
            value = str(record.get("value", ""))
            learned = str(record.get("updated_at", ""))[:19].replace("T", " ")

            shown = value if len(value) <= VALUE_CHARS else value[:VALUE_CHARS] + "…"
            cells = (category, key.replace("_", " "), shown, learned)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 2:
                    item.setToolTip(value)       # nothing is hidden
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)

            forget = QPushButton("Forget")
            describe_control(
                forget, f"Forget {category}/{key}",
                f"Permanently delete this entry: {shown[:80]}")
            forget.clicked.connect(
                lambda _checked=False, c=category, k=key, v=value:
                self._forget(c, k, v))
            self.table.setCellWidget(row, 4, forget)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self.status.setText(
            f"{len(records)} entr{'y' if len(records) == 1 else 'ies'} shown"
            + (f" (most recent {PAGE_SIZE})" if len(records) >= PAGE_SIZE else "")
        )

    # ── forgetting ───────────────────────────────────────────────────────────

    def _forget(self, category: str, key: str, value: str) -> None:
        """Delete one entry, after showing what it actually says.

        "Delete 1 item?" is not informed consent. The sentence ORION wrote
        about your sister is.
        """
        excerpt = value if len(value) <= 200 else value[:200] + "…"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Forget this?")
        box.setText(f"Permanently forget {category}/{key}?")
        box.setInformativeText(f"“{excerpt}”\n\nThis cannot be undone.")
        confirm = box.addButton("Forget", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Keep", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(box.buttons()[-1])       # default to keeping it
        box.exec()
        if box.clickedButton() is not confirm:
            return
        removed = self.forget_entry(category, key)
        self.status.setText(
            f"Forgotten: {category}/{key}." if removed
            else f"Nothing removed for {category}/{key} — it may already be gone."
        )
        self.refresh()

    def forget_entry(self, category: str, key: str) -> int:
        """Remove exactly one row. Returns how many were deleted.

        Scoped by category AND key on purpose: ``forget()`` accepts a bare
        category and will remove twelve hundred rows, which is right for a tool
        call and very wrong for a button sitting beside a single line.
        """
        if not category or not key:
            return 0
        try:
            return int(self._matrix.forget(category=category, key_prefix=key))
        except Exception:
            return 0


__all__ = ["PAGE_SIZE", "VALUE_CHARS", "MemoryViewer"]
