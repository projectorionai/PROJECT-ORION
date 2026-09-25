"""
Live, persistent knowledge-graph visualiser.

The Command/Mission decks used to show only the graph's *counts*.  This widget
renders the actual node-link graph ORION has built: the most-connected entities
and the relationships among them, laid out with a light force-directed
relaxation.  Node positions are retained across snapshots, so as ingestion adds
entities and links the graph visibly grows and settles rather than redrawing
from scratch — the picture correlates with what ORION is learning.

Performance: at most ~42 nodes (O(n²) relaxation ≈ 1.7k ops/tick), and the
relaxation timer only does work while the widget is actually visible.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ..constants import C

# Entity-kind → colour.  Unknown kinds fall back to the ORION red.
_KIND_COLOURS = {
    "person": (90, 200, 255),
    "people": (90, 200, 255),
    "org": (255, 184, 66),
    "organisation": (255, 184, 66),
    "organization": (255, 184, 66),
    "company": (255, 184, 66),
    "project": (120, 255, 165),
    "product": (120, 255, 165),
    "place": (185, 145, 255),
    "location": (185, 145, 255),
    "concept": (255, 96, 132),
    "topic": (255, 96, 132),
    "technology": (0, 229, 255),
}
_DEFAULT_COLOUR = (255, 66, 96)

# A small, distinct ring colour per cluster id (Mark XX design-spec §3:
# connected components computed by KnowledgeGraphEngine.graph_snapshot(),
# rendered here as a thin outline around each node — a secondary signal
# layered on top of the existing kind-colour dot, not replacing it).
# Cycles for clusters beyond the palette length rather than erroring.
_CLUSTER_RING_COLOURS = (
    (255, 205, 90), (111, 160, 224), (79, 191, 133), (185, 145, 255),
    (255, 138, 101), (0, 229, 255), (223, 230, 238),
)


class _Node:
    __slots__ = ("id", "name", "kind", "degree", "cluster", "x", "y", "vx", "vy")

    def __init__(self, node_id: str, name: str, kind: str, degree: int,
                 x: float, y: float, cluster: int = -1) -> None:
        self.id      = node_id
        self.name    = name
        self.kind    = kind
        self.degree  = degree
        self.cluster = cluster
        self.x       = x
        self.y       = y
        self.vx      = 0.0
        self.vy      = 0.0


class KnowledgeGraphWidget(QWidget):
    """Force-directed, persistent view of the knowledge graph."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("graphCanvas")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(200)
        self._nodes: dict[str, _Node] = {}
        # (source, target, weight, confidence, contradicts)
        self._edges: list[tuple[str, str, float, float, bool]] = []
        self._totals: dict[str, Any] = {}
        self._rng = random.Random(7)
        self._timer = QTimer(self)
        self._timer.setInterval(45)                 # ~22 fps while visible
        self._timer.timeout.connect(self._step)

    # The relaxation loop runs ONLY while the widget is actually on screen — it
    # costs nothing when the graph dock is hidden, and it means simply
    # constructing the widget (e.g. in tests) never leaves a timer running.
    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event: Any) -> None:
        self._timer.stop()
        super().hideEvent(event)

    # ── data ──────────────────────────────────────────────────────────────────

    def set_snapshot(self, snapshot: Any) -> None:
        """Merge a {nodes, edges, totals} snapshot, preserving the positions of
        entities that persist so the layout evolves smoothly."""
        if not isinstance(snapshot, dict):
            return
        nodes = snapshot.get("nodes") or []
        edges = snapshot.get("edges") or []
        self._totals = snapshot.get("totals") or {}
        cx = (self.width() or 400) / 2.0
        cy = (self.height() or 300) / 2.0

        seen: set[str] = set()
        for nd in nodes:
            nid = str(nd.get("id"))
            seen.add(nid)
            existing = self._nodes.get(nid)
            if existing is not None:
                existing.degree  = int(nd.get("degree") or 0)
                existing.name    = str(nd.get("name") or nid)
                existing.kind    = str(nd.get("kind") or "")
                existing.cluster = int(nd.get("cluster", -1))
            else:
                ang = self._rng.uniform(0, 2 * math.pi)
                rad = self._rng.uniform(20, 100)
                self._nodes[nid] = _Node(
                    nid, str(nd.get("name") or nid), str(nd.get("kind") or ""),
                    int(nd.get("degree") or 0),
                    cx + math.cos(ang) * rad, cy + math.sin(ang) * rad,
                    cluster=int(nd.get("cluster", -1)))
        for gone in [k for k in self._nodes if k not in seen]:
            del self._nodes[gone]

        self._edges = [
            (str(e.get("source")), str(e.get("target")), float(e.get("weight") or 1.0),
             float(e.get("confidence", 1.0)), bool(e.get("contradicts", False)))
            for e in edges
            if str(e.get("source")) in self._nodes and str(e.get("target")) in self._nodes
        ]
        self.update()

    # ── layout ─────────────────────────────────────────────────────────────────

    def _step(self) -> None:
        if not self.isVisible() or not self._nodes:
            return
        nodes = list(self._nodes.values())
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        for a in nodes:
            fx = fy = 0.0
            for b in nodes:
                if a is b:
                    continue
                dx = a.x - b.x
                dy = a.y - b.y
                d2 = dx * dx + dy * dy + 0.01
                f = 1500.0 / d2
                fx += dx * f
                fy += dy * f
            fx += (cx - a.x) * 0.012                 # gentle centring
            fy += (cy - a.y) * 0.012
            a.vx = (a.vx + fx) * 0.80
            a.vy = (a.vy + fy) * 0.80
        for s, t, w, _conf, _contra in self._edges:
            a = self._nodes.get(s)
            b = self._nodes.get(t)
            if a is None or b is None:
                continue
            f = 0.008 * min(3.0, w)
            dx = b.x - a.x
            dy = b.y - a.y
            a.vx += dx * f
            a.vy += dy * f
            b.vx -= dx * f
            b.vy -= dy * f
        margin = 26.0
        w = max(margin * 2 + 1, self.width())
        h = max(margin * 2 + 1, self.height())
        for a in nodes:
            a.x = min(w - margin, max(margin, a.x + max(-6.0, min(6.0, a.vx))))
            a.y = min(h - margin, max(margin, a.y + max(-6.0, min(6.0, a.vy))))
        self.update()

    # ── paint ──────────────────────────────────────────────────────────────────

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(C.BG))

        if not self._nodes:
            painter.setPen(QColor(C.MUTED))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Knowledge graph warming up — nodes appear as ORION learns.")
            return

        for s, t, w, confidence, contradicts in self._edges:
            a = self._nodes[s]
            b = self._nodes[t]
            # Confidence dims the edge (low-confidence links fade rather
            # than reading as equally certain); a contradicting pair (two
            # relationships between the same entities whose evidence
            # disagreed — KnowledgeGraphEngine's negation heuristic) draws
            # dashed and amber instead of a silent overwrite or an
            # indistinguishable extra line.
            alpha = int((38 + min(120, w * 30)) * max(0.15, min(1.0, confidence)))
            if contradicts:
                pen = QPen(QColor(C.AMBER_RGB[0], C.AMBER_RGB[1], C.AMBER_RGB[2], min(220, alpha + 60)), 1.4)
                pen.setStyle(Qt.PenStyle.DashLine)
            else:
                pen = QPen(QColor(255, 40, 70, alpha), 1.0)
            painter.setPen(pen)
            painter.drawLine(QPointF(a.x, a.y), QPointF(b.x, b.y))

        max_deg = max((n.degree for n in self._nodes.values()), default=1) or 1
        painter.setFont(QFont("Segoe UI", 8))
        for n in self._nodes.values():
            r = 4.0 + 8.0 * (n.degree / max_deg)
            col = _KIND_COLOURS.get(n.kind.lower(), _DEFAULT_COLOUR)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(col[0], col[1], col[2], 40))
            painter.drawEllipse(QPointF(n.x, n.y), r * 2.1, r * 2.1)
            if n.cluster >= 0:
                ring = _CLUSTER_RING_COLOURS[n.cluster % len(_CLUSTER_RING_COLOURS)]
                painter.setPen(QPen(QColor(ring[0], ring[1], ring[2], 150), 1.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(n.x, n.y), r * 2.7, r * 2.7)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*col))
            painter.drawEllipse(QPointF(n.x, n.y), r, r)
            if n.degree >= max(1, max_deg * 0.35):
                painter.setPen(QColor(235, 235, 240))
                painter.drawText(QRectF(n.x + r + 3, n.y - 7, 140, 14),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 n.name[:24])

        if self._totals:
            painter.setPen(QColor(255, 120, 140))
            painter.setFont(QFont("Cascadia Mono", 8))
            t = self._totals
            painter.drawText(
                QRectF(8, 6, self.width() - 16, 16), Qt.AlignmentFlag.AlignLeft,
                f"entities {t.get('entities', '?')} · relationships "
                f"{t.get('relationships', '?')} · events {t.get('events', '?')}")
