"""
StudioDeckView — a scannable multi-panel deck for the creative/academic/agency
subsystems, appended as a selectable tab inside the UnifiedDashboard.

Three panels, all fed purely by ``OrionBus`` signals (no widget ever touches a
service worker):

    1. Creative Audio Tracker — raw vs processed stem counts with a subtle
       canvas visualiser, driven by ``bus.audio_studio_activity``.
    2. Research & Concept Graph — newly indexed literature fragments and
       mechanism-first summaries, driven by ``bus.dashboard_event('literature')``
       and the KNOWLEDGE memory tier.
    3. Creator Campaign Pipeline — a kanban of campaigns by stage, driven by
       ``bus.dashboard_event('pipeline_update')``.

Everything respects the core palette (crimson #ff1a3c on deep-void #050508 with
#0f0f14 panels) and updates smoothly off the bus event streams.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import C
from ..utils import now_stamp


def _heading(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("panelHeading")
    return lab


# ──────────────────────────────────────────────────────────────────────────────
# CANVAS VISUALISER
# ──────────────────────────────────────────────────────────────────────────────

class AudioCanvas(QWidget):
    """A subtle animated bar visualiser of raw vs processed stem counts."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(110)
        self._raw = 0
        self._processed = 0
        self._raw_disp = 0.0
        self._proc_disp = 0.0
        self._scan = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_counts(self, raw: int, processed: int) -> None:
        self._raw = max(0, int(raw))
        self._processed = max(0, int(processed))

    def _tick(self) -> None:
        self._raw_disp += (self._raw - self._raw_disp) * 0.15
        self._proc_disp += (self._processed - self._proc_disp) * 0.15
        self._scan = (self._scan + 2.2) % 360.0
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#07070b"))
        peak = max(1.0, self._raw_disp, self._proc_disp)
        pad = 14
        bar_w = (w - pad * 3) / 2
        self._bar(painter, pad, h, bar_w, self._raw_disp / peak, C.MUTED, "RAW", int(round(self._raw_disp)))
        self._bar(painter, pad * 2 + bar_w, h, bar_w, self._proc_disp / peak, C.PRI, "PROCESSED",
                  int(round(self._proc_disp)))
        # Subtle horizontal scan line.
        sy = int((math.sin(math.radians(self._scan)) * 0.5 + 0.5) * (h - 24)) + 12
        painter.setPen(QPen(QColor(255, 26, 60, 40), 1))
        painter.drawLine(0, sy, w, sy)

    def _bar(self, painter: QPainter, x: float, h: int, bw: float, frac: float,
             colour: str, label: str, value: int) -> None:
        frac = max(0.0, min(1.0, frac))
        top = 22
        bh = (h - top - 22) * frac
        y = h - 22 - bh
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(colour))
        painter.drawRoundedRect(int(x), int(y), int(bw), int(bh), 5, 5)
        painter.setPen(QColor(C.WHITE))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.drawText(int(x), 0, int(bw), 18, Qt.AlignmentFlag.AlignCenter, f"{label}: {value}")


# ──────────────────────────────────────────────────────────────────────────────
# PANEL 1 — CREATIVE AUDIO TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class AudioTrackerPanel(QFrame):
    def __init__(self, bus: OrionBus, service: Any | None = None) -> None:
        super().__init__()
        self.bus = bus
        self.service = service
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(_heading("CREATIVE AUDIO TRACKER"))

        self.canvas = AudioCanvas()
        layout.addWidget(self.canvas)

        self.status = QLabel("Awaiting stem activity…")
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.stems = QListWidget()
        self.stems.setMaximumHeight(150)
        layout.addWidget(self.stems, 1)

        self.bus.audio_studio_activity.connect(self._on_activity)
        if service is not None:
            self._render(service.inventory_snapshot())

    def _on_activity(self, phase: str, data: object) -> None:
        if not isinstance(data, dict):
            return
        if phase == "index":
            self._render(data)
        elif phase == "processing":
            self.status.setText(f"Rendering {data.get('file', '')} "
                                f"[{data.get('backend', '')}]…")
        elif phase == "processed":
            self.status.setText(f"✓ {data.get('file', '')} — {data.get('summary', 'done')}")
        elif phase == "package":
            self.status.setText(f"📦 Packaged {data.get('count', 0)} stem(s): {data.get('package', '')}")
        elif phase == "error":
            self.status.setText(f"⚠ {data.get('file', '')}: {data.get('error', 'processing error')}")

    def _render(self, snap: dict[str, Any]) -> None:
        raw_count = int(snap.get("raw_count", 0))
        proc_count = int(snap.get("processed_count", 0))
        self.canvas.set_counts(raw_count, proc_count)
        self.status.setText(f"{raw_count} raw · {proc_count} processed  (updated {now_stamp()})")
        self.stems.clear()
        for asset in snap.get("processed", [])[:20]:
            self.stems.addItem(f"✓ {asset.get('name', '')}  ({asset.get('size_kb', 0):.0f} KB)")
        for asset in snap.get("raw", [])[:20]:
            self.stems.addItem(f"• {asset.get('name', '')}  ({asset.get('size_kb', 0):.0f} KB)")


# ──────────────────────────────────────────────────────────────────────────────
# PANEL 2 — RESEARCH & CONCEPT GRAPH
# ──────────────────────────────────────────────────────────────────────────────

class ConceptGraphPanel(QFrame):
    def __init__(self, bus: OrionBus, memory: Any, service: Any | None = None) -> None:
        super().__init__()
        self.bus = bus
        self.memory = memory
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(_heading("RESEARCH & CONCEPT GRAPH"))

        self.reader = QPlainTextEdit()
        self.reader.setReadOnly(True)
        self.reader.setObjectName("logBox")
        self.reader.setPlaceholderText(
            "Ingested papers and their mechanism-first summaries appear here.\n"
            "Use the literature_vault tool (or say 'ingest this paper …')."
        )
        layout.addWidget(self.reader, 1)

        self.bus.dashboard_event.connect(self._on_event)
        # Seed from any prior ingests + the KNOWLEDGE tier.
        if service is not None:
            for paper in service.snapshot():
                self._render_paper(paper)
        self._load_from_memory()

    def _on_event(self, channel: str, payload: object) -> None:
        if channel == "literature" and isinstance(payload, dict):
            self._render_paper(payload)

    def _render_paper(self, paper: dict[str, Any]) -> None:
        title = paper.get("title", "Untitled")
        block = [f"━━ {title} ━━  ({paper.get('word_count', 0)} words)"]
        for mech in paper.get("mechanisms", [])[:5]:
            block.append(f"  ⚙ {mech}")
        if paper.get("tables"):
            block.append("  ▦ " + "; ".join(paper["tables"][:2]))
        if paper.get("cross_refs"):
            block.append("  ↹ linked: " + ", ".join(paper["cross_refs"][:4]))
        if paper.get("dois"):
            block.append("  🔖 " + ", ".join(paper["dois"][:2]))
        block.append("")
        # Prepend so the newest paper is at the top.
        existing = self.reader.toPlainText()
        self.reader.setPlainText("\n".join(block) + ("\n" + existing if existing else ""))

    def _load_from_memory(self) -> None:
        try:
            rows = self.memory.query("lit_", limit=8)
        except Exception:
            rows = []
        if not rows:
            return
        seed = ["[from the KNOWLEDGE tier]"]
        for row in rows:
            if str(row.get("key_ref", "")).startswith("lit_"):
                seed.append(f"  • {row.get('value', '')[:160]}")
        if len(seed) > 1:
            existing = self.reader.toPlainText()
            self.reader.setPlainText((existing + "\n" if existing else "") + "\n".join(seed))


# ──────────────────────────────────────────────────────────────────────────────
# PANEL 3 — CREATOR CAMPAIGN PIPELINE (KANBAN)
# ──────────────────────────────────────────────────────────────────────────────

class CampaignKanbanPanel(QFrame):
    def __init__(self, bus: OrionBus, service: Any | None = None) -> None:
        super().__init__()
        self.bus = bus
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(_heading("CREATOR CAMPAIGN PIPELINE"))
        self.totals = QLabel("")
        self.totals.setObjectName("mutedLabel")
        header.addStretch(1)
        header.addWidget(self.totals)
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._board_host = QWidget()
        self._board = QHBoxLayout(self._board_host)
        self._board.setSpacing(8)
        self._board.setContentsMargins(2, 2, 2, 2)
        scroll.setWidget(self._board_host)
        layout.addWidget(scroll, 1)

        self.bus.dashboard_event.connect(self._on_event)
        if service is not None:
            self._render(service.get_snapshot())
        else:
            self._render({"stages": [], "board": {}, "totals": {}})

    def _on_event(self, channel: str, payload: object) -> None:
        if channel == "pipeline_update" and isinstance(payload, dict):
            self._render(payload)

    def _render(self, snap: dict[str, Any]) -> None:
        # Clear existing columns.
        while self._board.count():
            item = self._board.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        totals = snap.get("totals", {})
        if totals:
            self.totals.setText(
                f"{totals.get('campaigns', 0)} campaigns · "
                f"£{totals.get('pipeline_value', 0):,.0f} · "
                f"{totals.get('engagement', 0):,.0f} engagement"
            )
        board = snap.get("board", {})
        stages = snap.get("stages", []) or list(board.keys())
        if not stages:
            placeholder = QLabel("No campaigns yet — create one with the campaign_pipeline tool.")
            placeholder.setObjectName("mutedLabel")
            self._board.addWidget(placeholder)
            self._board.addStretch(1)
            return
        for stage in stages:
            self._board.addWidget(self._column(stage, board.get(stage, [])))
        self._board.addStretch(1)

    def _column(self, stage: str, cards: list[dict[str, Any]]) -> QWidget:
        col = QFrame()
        col.setObjectName("panelFrame")
        col.setMinimumWidth(180)
        col.setMaximumWidth(230)
        cl = QVBoxLayout(col)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(6)
        title = QLabel(f"{stage}  ({len(cards)})")
        title.setStyleSheet(f"color:{C.PRI};font-weight:800;font-size:11px;")
        cl.addWidget(title)
        for card in cards[:12]:
            cl.addWidget(self._card(card))
        cl.addStretch(1)
        return col

    def _card(self, card: dict[str, Any]) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(
            f"background:{C.PANEL};border:1px solid {C.BORDER};border-radius:6px;")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(8, 6, 8, 6)
        fl.setSpacing(2)
        name = QLabel(str(card.get("name", "")))
        name.setStyleSheet(f"color:{C.WHITE};font-weight:700;font-size:11px;")
        name.setWordWrap(True)
        fl.addWidget(name)
        meta = []
        if card.get("brand"):
            meta.append(str(card["brand"]))
        if card.get("value"):
            meta.append(f"£{float(card['value']):,.0f}")
        eng = card.get("engagement_total", 0)
        if eng:
            meta.append(f"{float(eng):,.0f} eng")
        if meta:
            sub = QLabel("  ·  ".join(meta))
            sub.setObjectName("mutedLabel")
            sub.setStyleSheet(f"color:{C.MUTED};font-size:10px;")
            fl.addWidget(sub)
        return frame


# ──────────────────────────────────────────────────────────────────────────────
# THE DECK
# ──────────────────────────────────────────────────────────────────────────────

class StudioDeckView(QWidget):
    """The Studio Deck page: audio · research · campaigns, all bus-driven."""

    def __init__(self, bus: OrionBus, memory: Any, audio_studio: Any | None = None,
                 literature: Any | None = None, pipeline: Any | None = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.bus = bus
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        outer.addWidget(_heading("STUDIO DECK — CREATIVE · ACADEMIC · AGENCY"))

        grid = QGridLayout()
        grid.setSpacing(10)
        self.audio_panel = AudioTrackerPanel(bus, audio_studio)
        self.concept_panel = ConceptGraphPanel(bus, memory, literature)
        self.kanban_panel = CampaignKanbanPanel(bus, pipeline)
        # Audio + research side by side on top; kanban spans the full width below.
        grid.addWidget(self.audio_panel, 0, 0)
        grid.addWidget(self.concept_panel, 0, 1)
        grid.addWidget(self.kanban_panel, 1, 0, 1, 2)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        outer.addLayout(grid, 1)
