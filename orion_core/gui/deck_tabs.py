"""One Command Deck page made of what used to be separate pages (Mark XXXVIII).

Focus and Toolkit both had a focus timer; Telemetry drew the load graphs the
Diagnostics page summarises. Each pair is now one page with tabs, and a part
that is expensive to build is built the first time its tab is opened.
"""
from __future__ import annotations

from typing import Callable

from PyQt6.QtWidgets import QTabWidget, QWidget

Part = QWidget | Callable[[], QWidget]


class DeckTabs(QTabWidget):
    def __init__(self, parts: list[tuple[str, Part]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDocumentMode(True)
        self._factories: dict[int, Callable[[], QWidget]] = {}
        for label, part in parts:
            if isinstance(part, QWidget):
                self.addTab(part, label)
            else:
                index = self.addTab(QWidget(), label)
                self._factories[index] = part
        self.currentChanged.connect(self._build)
        self._build(self.currentIndex())

    def _build(self, index: int) -> None:
        factory = self._factories.pop(index, None)
        if factory is None:
            return
        widget = factory()
        label = self.tabText(index)
        placeholder = self.widget(index)
        self.blockSignals(True)
        try:
            self.removeTab(index)
            self.insertTab(index, widget, label)
            self.setCurrentIndex(index)
        finally:
            self.blockSignals(False)
        if placeholder is not None:
            placeholder.deleteLater()

    def show_tab(self, label: str) -> bool:
        """Bring the tab called *label* forward ("tools", "load")."""
        wanted = str(label or "").strip().lower()
        for index in range(self.count()):
            if self.tabText(index).lower() == wanted:
                self.setCurrentIndex(index)
                return True
        return False

    def part(self, label: str) -> QWidget | None:
        """The (built) widget of the tab called *label*."""
        for index in range(self.count()):
            if self.tabText(index).lower() == str(label).lower():
                self._build(index)
                return self.widget(index)
        return None


__all__ = ["DeckTabs"]
