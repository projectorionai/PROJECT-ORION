"""The existing goals engine, reachable from the Command Deck."""
from __future__ import annotations

import asyncio
from typing import Any

from PyQt6 import sip
from PyQt6.QtWidgets import (
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)


class GoalsDeckView(QWidget):
    def __init__(self, bus: Any, dispatcher: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus, self.dispatcher = bus, dispatcher
        self._busy = False
        self._task = None
        layout = QVBoxLayout(self)
        title = QLabel("Goals")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlainText("Add a goal or refresh to see your saved goals and their pace.")
        layout.addWidget(self.output, 1)
        form = QFormLayout()
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Goal")
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("Optional numeric target")
        self.unit_input = QLineEdit()
        self.unit_input.setPlaceholderText("e.g. km, hours, books…")
        self.deadline_input = QLineEdit()
        self.deadline_input.setPlaceholderText("Optional deadline, e.g. 2026-12-31")
        self.progress_input = QLineEdit()
        self.progress_input.setPlaceholderText("What you did, e.g. I ran 5 km")
        for label, control in (("Goal", self.title_input), ("Target", self.target_input),
                               ("Unit", self.unit_input), ("Deadline", self.deadline_input),
                               ("Progress", self.progress_input)):
            form.addRow(label, control)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.buttons = []
        for label, action in (("Refresh", "list"), ("Add Goal", "create"),
                              ("Log Progress", "log_progress"), ("Review Week", "review")):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, action=action: self._launch(action))
            self.buttons.append(button)
            row.addWidget(button)
        layout.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        bus.dashboard_event.connect(self._on_event)
        self.destroyed.connect(self._cancel_owned)

    def _cancel_owned(self, *_args: Any) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def _arguments(self, action: str) -> dict[str, Any]:
        args: dict[str, Any] = {"action": action}
        title = self.title_input.text().strip()
        if action == "create":
            args.update(title=title, unit=self.unit_input.text().strip(),
                        deadline=self.deadline_input.text().strip())
            if self.target_input.text().strip():
                args["target"] = self.target_input.text().strip()
        elif action == "log_progress":
            args.update(goal=title, text=self.progress_input.text().strip())
        return args

    def _launch(self, action: str) -> None:
        if sip.isdeleted(self) or self._busy:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.status.setText("Goal controls are available while ORION is running.")
            return
        from .. import background
        self._task = background.spawn(self.execute(self._arguments(action)), name="orion-goals-page")

    async def execute(self, args: dict[str, Any]) -> Any:
        if sip.isdeleted(self) or self._busy:
            return None
        self._busy = True
        for button in self.buttons:
            button.setEnabled(False)
        try:
            result = await self.dispatcher.goals_tool(args)
            if sip.isdeleted(self):
                return result
            self.status.setText(result.text)
            if args.get("action") in {"list", "review"}:
                self.output.setPlainText(result.text)
            elif result.ok:
                listing = await self.dispatcher.goals_tool({"action": "list"})
                if not sip.isdeleted(self):
                    self.output.setPlainText(listing.text)
            return result
        except Exception as exc:
            if not sip.isdeleted(self):
                self.status.setText(f"Goals could not be updated: {exc}")
            return None
        finally:
            self._busy = False
            if not sip.isdeleted(self):
                for button in self.buttons:
                    button.setEnabled(True)

    def _on_event(self, channel: str, _payload: Any) -> None:
        if channel == "goals" and self.isVisible() and not self._busy:
            self._launch("list")

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._launch("list")
