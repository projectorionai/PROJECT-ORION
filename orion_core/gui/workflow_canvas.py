"""
WorkflowChainCanvas — a real visual diagram of a workflow's steps, for the
AUTOMATION zone's "visual workflow builder" ask (Mark XX design-spec).

WorkflowEngine runs a named workflow as a strictly ordered chain of tool
calls (step 1 -> step 2 -> ... -> step N, no branching, no conditions in
the data model) — so a left-to-right node chain is what "visual" honestly
means for THIS backend. A drag-and-drop DAG editor would overclaim a
branching capability the engine doesn't have. The step table in
AutomationDeckView stays the authoritative editing surface (typing a tool
name and a JSON args blob into a table cell is simply the right interface
for that data); this canvas is a live, read-only overview on top of it —
click a node to jump the table selection to it, select a table row to
highlight the matching node.
"""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsTextItem, QGraphicsView

from ..constants import C


#: Workflow steps are structure, not status — a canvas of blue boxes was a
#: second theme living inside the deck. One neutral edge, and the accent kept
#: for the step you have selected.
_NODE_EDGE = C.ACCENT_DIM


class WorkflowChainCanvas(QGraphicsView):
    step_clicked = pyqtSignal(int)

    _NODE_W = 170
    _NODE_H = 68
    _GAP = 44

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMinimumHeight(130)
        self.setMaximumHeight(130)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._rects: list[Any] = []

    def set_steps(self, steps: list[dict]) -> None:
        self._scene.clear()
        self._rects = []
        if not steps:
            placeholder = self._scene.addText("No steps yet — add one in the step editor.")
            placeholder.setDefaultTextColor(QColor(C.MUTED))
            self._scene.setSceneRect(0, 0, 320, self._NODE_H + 20)
            return
        x = 10.0
        y = 12.0
        prev_right: float | None = None
        for index, step in enumerate(steps):
            tool = str(step.get("tool") or "?")
            args_preview = json.dumps(step.get("args") or {})
            if len(args_preview) > 30:
                args_preview = args_preview[:27] + "..."
            rect = self._scene.addRect(
                0, 0, self._NODE_W, self._NODE_H,
                QPen(QColor(_NODE_EDGE), 2), QBrush(QColor(C.PANEL_HI)))
            rect.setPos(x, y)
            rect.setData(0, index)
            label = QGraphicsTextItem(f"{index + 1}. {tool}\n{args_preview}", rect)
            label.setDefaultTextColor(QColor(C.WHITE))
            label.setPos(8, 6)
            label.setTextWidth(self._NODE_W - 16)
            self._rects.append(rect)
            if prev_right is not None:
                self._scene.addLine(
                    prev_right, y + self._NODE_H / 2, x, y + self._NODE_H / 2,
                    QPen(QColor(_NODE_EDGE), 2))
            prev_right = x + self._NODE_W
            x += self._NODE_W + self._GAP
        self._scene.setSceneRect(0, 0, x, self._NODE_H + 24)

    def highlight_step(self, index: int) -> None:
        for i, rect in enumerate(self._rects):
            selected = i == index
            # Selection is a state, which is one of the three things the
            # accent is for. The steps themselves are structure and stay
            # neutral, so the selected one is the only coloured node.
            colour = QColor(C.PRI) if selected else QColor(_NODE_EDGE)
            rect.setPen(QPen(colour, 3 if selected else 2))

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        index = None
        while item is not None and index is None:
            index = item.data(0)
            item = item.parentItem()
        if isinstance(index, int):
            self.highlight_step(index)
            self.step_clicked.emit(index)
        super().mousePressEvent(event)
