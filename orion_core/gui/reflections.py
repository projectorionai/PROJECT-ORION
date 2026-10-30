"""A useful home for completed application notes in the Cognition page."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from ..bus import OrionBus
from ..reflections import ReflectionHistory


class ReflectionsPanel(QFrame):
    """Completed notes only: no hidden-page typing or token repaint queue."""

    def __init__(self, bus: OrionBus) -> None:
        super().__init__()
        self.bus = bus
        self.history = ReflectionHistory()
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        heading = QHBoxLayout()
        title = QLabel("Reflections")
        title.setObjectName("panelHeading")
        title.setToolTip("Generated summaries of observed activity, not private model reasoning.")
        heading.addWidget(title, 1)
        self.tasks_button = QPushButton("Open Tasks")
        self.tasks_button.clicked.connect(
            lambda: self.bus.gui_command.emit({"action": "page", "target": "TASKS"}))
        self.clear_button = QPushButton("Clear View")
        self.clear_button.setToolTip("Clear this view; retain the recorded journal.")
        self.clear_button.clicked.connect(self.clear_view)
        heading.addWidget(self.tasks_button)
        heading.addWidget(self.clear_button)
        layout.addLayout(heading)
        self.empty_label = QLabel("No activity notes yet.")
        self.empty_label.setObjectName("mutedLabel")
        layout.addWidget(self.empty_label)
        self.notes = QListWidget()
        self.notes.setWordWrap(True)
        self.notes.setMinimumHeight(120)
        self.notes.setMaximumHeight(280)
        self.notes.setAccessibleName("Recorded reflections and activity notes")
        self.notes.currentRowChanged.connect(self._show_details)
        layout.addWidget(self.notes)
        self.details = QLabel("")
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setObjectName("mutedLabel")
        layout.addWidget(self.details)
        self.bus.thought.connect(self._on_thought_full)
        # Lazy deck construction must not lose the last completed notes. The
        # source publishes at most 50 entries and the model enforces its cap.
        read_latest = getattr(bus, "last_value", None)
        latest = read_latest("dashboard:reflections") if callable(read_latest) else None
        if isinstance(latest, dict) and isinstance(latest.get("entries"), list):
            for payload in latest["entries"][-50:]:
                self.history.add(payload)
        self._refresh()

    def _on_thought_full(self, payload: Any) -> None:
        if self.history.add(payload):
            self._refresh()

    def _refresh(self) -> None:
        self.notes.blockSignals(True)
        self.notes.clear()
        for entry in self.history.entries:
            row = QListWidgetItem(f"{entry.heading}\n{entry.text}")
            row.setToolTip(entry.context or entry.heading)
            self.notes.addItem(row)
        self.notes.blockSignals(False)
        has_notes = bool(self.history.entries)
        self.empty_label.setVisible(not has_notes)
        self.notes.setVisible(has_notes)
        self.clear_button.setEnabled(has_notes)
        if has_notes:
            self.notes.setCurrentRow(0)
            self._show_details(0)
        else:
            self.details.clear()
            self.details.hide()

    def _show_details(self, row: int) -> None:
        entries = list(self.history.entries)
        if not 0 <= row < len(entries):
            self.details.hide()
            return
        entry = entries[row]
        lines = [entry.context] if entry.context else []
        if entry.observed_at:
            lines.append(f"Recorded: {entry.observed_at}")
        if entry.evidence:
            lines.append("Recorded evidence:")
            lines.extend(f"• {item}" for item in entry.evidence)
        self.details.setText("\n".join(lines))
        self.details.setVisible(bool(lines))

    def clear_view(self) -> None:
        self.history.clear()
        self._refresh()
