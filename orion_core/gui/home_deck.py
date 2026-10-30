"""The Home page of the Command Deck: the smart home at a glance (Mark XXXVII.5).

Every light, room, switch and fan Home Assistant or the Hue bridge knows, each
with its own on/off button; the scenes; the routines (Goodnight, Morning,
Movie, Away). Everything goes through the smart_home tool, so the page, ORION's
voice and the reflexes all do exactly the same thing - and the network work
happens off the GUI thread inside that tool.
"""
from __future__ import annotations

import asyncio
from typing import Any

from PyQt6 import sip
from PyQt6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

#: What the device grid shows, in this order (scenes have their own row).
SHOWN_KINDS = ("room", "light", "switch", "fan", "lock", "cover", "climate", "media_player",
               "input_boolean")
ROUTINE_BUTTONS = (("goodnight", "Goodnight"), ("morning", "Morning"),
                   ("movie", "Movie"), ("away", "Away"))


class HomeDeckView(QWidget):
    def __init__(self, bus: Any, dispatcher: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus, self.dispatcher = bus, dispatcher
        self._busy = False
        self._task: Any = None
        layout = QVBoxLayout(self)
        title = QLabel("Home")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        self.status = QLabel("Reading your smart home…")
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Smart home status")
        layout.addWidget(self.status)

        routines = QHBoxLayout()
        self.buttons: list[QPushButton] = []
        for key, label in ROUTINE_BUTTONS:
            button = QPushButton(label)
            button.setAccessibleName(f"Run the {label} routine")
            button.setToolTip(f"Run the {label} routine")
            button.clicked.connect(lambda _c=False, k=key: self._launch({"action": "routine",
                                                                          "target": k}))
            routines.addWidget(button)
            self.buttons.append(button)
        routines.addStretch(1)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setAccessibleName("Refresh the smart home")
        self.refresh_button.clicked.connect(lambda _c=False: self._launch({"action": "status"}))
        routines.addWidget(self.refresh_button)
        self.pair_button = QPushButton("Pair Hue Bridge")
        self.pair_button.setAccessibleName("Pair a Philips Hue bridge")
        self.pair_button.setToolTip("Press the bridge's round button first, then this within 30 seconds")
        self.pair_button.clicked.connect(lambda _c=False: self._launch({"action": "pair"}))
        routines.addWidget(self.pair_button)
        self.buttons += [self.refresh_button, self.pair_button]
        layout.addLayout(routines)

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setColumnStretch(0, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_host)
        layout.addWidget(scroll, 1)
        self.scenes = QHBoxLayout()
        layout.addLayout(self.scenes)
        self._rows: list[QWidget] = []

    # ── running the tool ─────────────────────────────────────────────────────

    def _launch(self, args: dict[str, Any]) -> None:
        if sip.isdeleted(self) or self._busy:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.status.setText("Smart home controls are available while ORION is running.")
            return
        from .. import background
        self._task = background.spawn(self.execute(args), name="orion-home-page")

    async def execute(self, args: dict[str, Any]) -> Any:
        if sip.isdeleted(self) or self._busy:
            return None
        self._busy = True
        self._enable(False)
        try:
            result = await self.dispatcher.smart_home_tool(args)
            if sip.isdeleted(self):
                return result
            if args.get("action") == "status":
                self.show_house(result)
            else:
                self.status.setText(result.text)
                listing = await self.dispatcher.smart_home_tool({"action": "status"})
                if not sip.isdeleted(self):
                    self.show_house(listing, keep_status=True)
            return result
        except Exception as exc:
            if not sip.isdeleted(self):
                self.status.setText(f"The smart home could not be reached: {exc}")
            return None
        finally:
            self._busy = False
            if not sip.isdeleted(self):
                self._enable(True)

    def _enable(self, on: bool) -> None:
        for button in self.buttons + [b for row in self._rows for b in row.findChildren(QPushButton)]:
            button.setEnabled(on)

    # ── drawing the house ────────────────────────────────────────────────────

    def show_house(self, result: Any, *, keep_status: bool = False) -> None:
        rows: list[dict[str, Any]] = []
        for item in getattr(result, "evidence", None) or []:
            if isinstance(item, dict) and isinstance(item.get("smart_home_devices"), list):
                rows = item["smart_home_devices"]
        if not keep_status:
            self.status.setText(str(getattr(result, "text", "") or ""))
        for widget in self._rows:
            widget.setParent(None)
            widget.deleteLater()
        self._rows = []
        while self.scenes.count():
            item = self.scenes.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        shown = sorted((r for r in rows if r.get("kind") in SHOWN_KINDS),
                       key=lambda r: (SHOWN_KINDS.index(r["kind"]), str(r.get("name")).lower()))
        for index, row in enumerate(shown):
            line = QWidget()
            box = QHBoxLayout(line)
            box.setContentsMargins(0, 0, 0, 0)
            name = str(row.get("name") or row.get("id"))
            state = str(row.get("state") or "")
            detail = state.title() + (f" · {row['brightness']}%" if row.get("brightness") else "")
            label = QLabel(f"{name}  —  {detail}" if detail else name)
            label.setAccessibleName(f"{name}, {detail or 'state unknown'}")
            box.addWidget(label, 1)
            if row.get("kind") not in {"lock", "cover", "climate"} and state != "unavailable":
                on = bool(row.get("on"))
                button = QPushButton("Turn Off" if on else "Turn On")
                button.setAccessibleName(f"Turn {name} {'off' if on else 'on'}")
                button.clicked.connect(lambda _c=False, n=name, o=on: self._launch(
                    {"action": "off" if o else "on", "target": n}))
                box.addWidget(button)
            self.grid.addWidget(line, index, 0)
            self._rows.append(line)
        for row in [r for r in rows if r.get("kind") == "scene"][:8]:
            name = str(row.get("name") or "")
            button = QPushButton(name)
            button.setAccessibleName(f"Set the {name} scene")
            button.clicked.connect(lambda _c=False, n=name: self._launch({"action": "scene",
                                                                          "target": n}))
            self.scenes.addWidget(button)
        self.scenes.addStretch(1)

    def showEvent(self, event: Any) -> None:   # noqa: N802
        super().showEvent(event)
        self._launch({"action": "status"})


__all__ = ["HomeDeckView", "ROUTINE_BUTTONS", "SHOWN_KINDS"]
