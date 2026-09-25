"""The Command Deck's Brain page (Mark XXXII): the Iris over the Signal Lanes.

The Iris shows ORION's state, load and which of his systems are working; the
Signal Lanes show each request stepping through hearing, thinking, acting,
speaking and recall over the last twenty seconds, so a request that stalls is
visible where it stopped. Beside them, the plain facts: the Live channel, the
host's load, the latest tool route and recent activity.

Everything is native Qt fed by signals ORION already emits (see
brain_instruments). It replaced a three.js scene that cost a WebEngine
renderer; the face and the orb are untouched.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..bus import OrionBus
from .brain_instruments import BrainModel, IrisGauge, SignalLanes, _Animator


class CommandOverview(QWidget):
    """The Brain page; each signal updates only what it changes."""

    QUICK_LINKS = ("RESEARCH", "MISSION", "COMMAND CENTRE", "DIAGNOSTICS", "LOG")

    def __init__(self, bus: OrionBus,
                 on_open_page: Callable[[str], Any] | None = None) -> None:
        super().__init__()
        self.bus = bus
        self._open_page = on_open_page
        self.deck_pages: dict[str, str] = {}
        self._recent: deque[str] = deque(maxlen=5)
        self.model = BrainModel()

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        head = QHBoxLayout()
        head.setSpacing(12)
        title = QLabel("BRAIN")
        title.setObjectName("workspaceTitle")
        head.addWidget(title)
        self.mode = QLabel("STANDBY")
        self.mode.setObjectName("deckStatePill")
        self.mode.setAccessibleName("ORION's state")
        head.addWidget(self.mode, 0, Qt.AlignmentFlag.AlignVCenter)
        subtitle = QLabel("What ORION is doing, as it happens")
        subtitle.setObjectName("mutedLabel")
        head.addWidget(subtitle, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addStretch(1)
        root.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(12)
        iris_frame = QFrame()
        iris_frame.setObjectName("panelFrame")
        iris_box = QVBoxLayout(iris_frame)
        iris_box.setContentsMargins(8, 8, 8, 8)
        self.iris = IrisGauge(self.model)
        iris_box.addWidget(self.iris)
        body.addWidget(iris_frame, 3)

        facts = QVBoxLayout()
        facts.setSpacing(10)
        self.link = self._card(facts, "LIVE CHANNEL", "Waiting for connection")
        self.host = self._card(facts, "HOST LOAD", "CPU —   ·   RAM —")
        self.route = self._card(facts, "LATEST ROUTE", "No tool activity yet")
        activity = QFrame()
        activity.setObjectName("panelFrame")
        activity_box = QVBoxLayout(activity)
        activity_box.addWidget(self._heading("RECENT ACTIVITY"))
        self.activity = QLabel("Ready for your next request")
        self.activity.setObjectName("mutedLabel")
        self.activity.setWordWrap(True)
        self.activity.setAlignment(Qt.AlignmentFlag.AlignTop)
        activity_box.addWidget(self.activity, 1)
        facts.addWidget(activity, 1)
        facts_holder = QWidget()
        facts_holder.setLayout(facts)
        facts_holder.setFixedWidth(300)
        body.addWidget(facts_holder)
        root.addLayout(body, 1)

        lanes_frame = QFrame()
        lanes_frame.setObjectName("panelFrame")
        lanes_box = QVBoxLayout(lanes_frame)
        lanes_box.setContentsMargins(4, 4, 4, 4)
        self.lanes = SignalLanes(self.model)
        self.lanes.stalled_count = _stalled_requests
        lanes_box.addWidget(self.lanes)
        root.addWidget(lanes_frame)

        links = QHBoxLayout()
        links.setSpacing(8)
        self._quick_links: dict[str, QPushButton] = {}
        for name in self.QUICK_LINKS:
            button = QPushButton(name.title())
            button.setObjectName("deckTab")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAccessibleName(f"Open {name.title()}")
            button.clicked.connect(lambda _checked=False, page=name: self._navigate(page))
            self._quick_links[name] = button
            links.addWidget(button)
        links.addStretch(1)
        root.addLayout(links)

        self._animator = _Animator(self.model, [self.iris, self.lanes])
        # A heard utterance ends when the speaker goes quiet, which no signal
        # announces; while hidden nothing needs drawing, so nothing polls.
        self._settle = QTimer(self)
        self._settle.setInterval(500)
        self._settle.timeout.connect(self.model.settle)

        bus.state.connect(self._on_state)
        bus.speaking.connect(self._on_speaking)
        bus.connection_state.connect(self._on_connection)
        bus.telemetry_sample.connect(self._on_telemetry)
        bus.agent_activity.connect(self._on_activity)
        bus.log.connect(self._on_log)
        bus.dashboard_event.connect(self._on_dashboard_event)

    # ── layout helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("panelHeading")
        return label

    def _card(self, column: QVBoxLayout, title: str, initial: str) -> QLabel:
        frame = QFrame()
        frame.setObjectName("panelFrame")
        box = QVBoxLayout(frame)
        box.addWidget(self._heading(title))
        value = QLabel(initial)
        value.setObjectName("workspaceTitle")
        value.setWordWrap(True)
        box.addWidget(value, 1)
        column.addWidget(frame)
        return value

    @staticmethod
    def _set(label: QLabel, value: str) -> None:
        if label.text() != value:
            label.setText(value)

    # ── navigation ───────────────────────────────────────────────────────────

    def _navigate(self, page: str) -> None:
        if self._open_page is not None:
            self._open_page(page)

    def attach_deck_pages(self, pages: dict[str, str]) -> None:
        self.deck_pages = dict(pages or {})
        for name, button in self._quick_links.items():
            button.setEnabled(name in self.deck_pages)

    # ── signals ──────────────────────────────────────────────────────────────

    def _on_state(self, state: Any) -> None:
        text = str(state or "STANDBY").upper()
        self._set(self.mode, text)
        self.model.set_state(text)

    def _on_speaking(self, active: Any) -> None:
        if active:
            self._set(self.mode, "SPEAKING")
        self.model.set_speaking(bool(active))

    def _on_connection(self, snapshot: Any) -> None:
        if not isinstance(snapshot, dict):
            return
        provider = str(snapshot.get("provider") or "Local")
        state = str(snapshot.get("state") or "disconnected").replace("_", " ")
        self._set(self.link, f"{provider}  ·  {state.title()}")
        self.model.set_link(provider, state)

    def _on_telemetry(self, sample: Any) -> None:
        if not isinstance(sample, dict):
            return
        try:
            cpu = float(sample.get("cpu", 0.0) or 0.0)
            ram = float(sample.get("ram", 0.0) or 0.0)
            gpu_raw = sample.get("gpu")
            gpu = float(gpu_raw) if gpu_raw is not None else None
        except (TypeError, ValueError):
            return
        self._set(self.host, f"CPU {cpu:.0f}%   ·   RAM {ram:.0f}%")
        self.model.set_telemetry(cpu, ram, gpu)

    def _on_activity(self, agent: str, summary: str) -> None:
        self._add_activity(f"{agent}: {summary}")

    def _on_log(self, message: str) -> None:
        text = str(message)
        if text.startswith("TRACE: "):
            # Request traces: a local transcript marks an utterance heard; a
            # tool start opens the Act lane until its result is reported.
            if " TRANSCRIPTION_COMPLETE" in text:
                self.model.heard_utterance()
            elif " TOOL_EXECUTION_START" in text:
                names = text.split(" — ", 1)[1] if " — " in text else ""
                self.model.tool_started(names)
            return
        if text.startswith(("NET:", "TOOL:", "AUDIO:")):
            self._add_activity(text)

    def _on_dashboard_event(self, channel: str, _payload: Any) -> None:
        if channel == "voice_heard":
            self.model.heard()

    def _add_activity(self, message: str) -> None:
        item = " ".join(str(message).split())[:180]
        if not item or (self._recent and self._recent[0] == item):
            return
        self._recent.appendleft(item)
        if self.isVisible():
            self._set(self.activity, "\n\n".join(self._recent))

    # ── visibility and cost ──────────────────────────────────────────────────

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if self._recent:
            self._set(self.activity, "\n\n".join(self._recent))
        self._settle.start()
        self._animator.set_visible(True)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._settle.stop()
        self._animator.set_visible(False)
        super().hideEvent(event)

    def pulse(self, from_id: str = "core", to_id: str = "", tool: str = "",
              args: dict[str, Any] | None = None, flight: Any = None,
              ok: bool = True, elapsed_ms: float | None = None, **_extra: Any) -> None:
        """The dispatcher's report after every tool call."""
        label = str(tool or to_id or "Activity")
        timing = f"  ·  {elapsed_ms / 1000.0:.1f} s" if elapsed_ms else ""
        self._set(self.route, f"{label}  ·  {'Done' if ok else 'Failed'}{timing}")
        self._add_activity(f"{label}: {'completed' if ok else 'failed'}")
        self.model.tool_finished(label, ok, elapsed_ms)

    def set_attended(self, attended: bool) -> None:
        self._animator.set_attended(attended)

    def release(self) -> bool:
        return False


def _stalled_requests() -> int:
    try:
        from ..request_trace import TRACES
        return len(TRACES.stalled())
    except Exception:
        return 0
