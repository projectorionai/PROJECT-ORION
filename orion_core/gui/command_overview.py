"""The Command Deck's Brain page (Mark XXXII): the Iris over the Signal Lanes.

The Iris shows ORION's state, load and which of his systems are working; the
Signal Lanes show each request stepping through hearing, thinking, acting,
speaking and recall over the last twenty seconds, so a request that stalls is
visible where it stopped. Beside them, the plain facts: the Live channel, the
host's load, the latest tool route and recent activity.

Everything is native Qt fed by signals ORION already emits (see
brain_instruments). It replaced a three.js scene that cost a WebEngine
renderer; the face and the orb are untouched.

Mark XXXV W1.2: in a real run the two instruments are drawn by a Brain host
process (gui/page_host.py), so a stall in ORION's thread no longer freezes
them. This page keeps its own model and forwards each event to the host; if
the host cannot run, the instruments are drawn here as before.
"""

from __future__ import annotations

import re

import time
from collections import deque
from typing import Any, Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from ..bus import OrionBus
from ..constants import CONFIG_DIR
from .brain_instruments import (STATE_WORDS, BrainModel, IrisGauge, SignalLanes, _Animator,
                                display_state)
from .command_palette import command_title
from .labels import page_display_name, plural

_STATES = STATE_WORDS


def _instrument_background() -> str:
    """The panel colour the instruments sit on, made solid.

    A window from another process cannot be translucent over this one, so in
    host mode the two instrument panels are painted this opaque colour and so
    is the host's window: the resting panel over the window backdrop, as it
    composites over a dark desktop."""
    from ..constants import C, GLASS

    def channels(hex_colour: str) -> tuple[int, int, int]:
        value = hex_colour.lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    window = [c * 0.88 for c in channels(C.BG)]
    panel = channels(C.PANEL)
    alpha = float(GLASS.PANEL)
    mixed = [round(p * alpha + w * (1.0 - alpha)) for p, w in zip(panel, window)]
    return "#{:02x}{:02x}{:02x}".format(*mixed)


#: Log lines shown in Recent Activity: what he did, including everything in
#: the background (Mark XXXX.4 - specialists, jobs, research and his proactive
#: lines were filtered out, so the page missed most of what he was doing).
ACTIVITY_PREFIXES = ("NET:", "TOOL:", "AUDIO:", "GUARD:", "AGENT:", "SPECIALIST:", "REASON:",
                     "JOB:", "REFLEX:", "RESEARCH:", "BRIEF:", "REPORT:", "EXPORT:", "REMOTE:",
                     "TELEGRAM:", "SENTINEL:", "INSTALL:", "ORION (proactive):")


def _agent_key(match: "re.Match[str]") -> str:
    return "agent:" + match.group("agent").strip().removesuffix(" Agent").lower()


#: (pattern, key, label, outcome): outcome None starts a WORK bar; otherwise it
#: says whether the work ended well.
_WORK_LINES = (
    (re.compile(r"AGENT: routing request to the (?P<agent>.+?)\.?$"), _agent_key,
     lambda m: m.group("agent"), None),
    (re.compile(r"SPECIALIST: (?P<agent>.+?) started\b"), _agent_key,
     lambda m: m.group("agent"), None),
    (re.compile(r"SPECIALIST: (?P<agent>.+?) (?P<end>DONE|PARTIAL|FAILED|GAVE UP|TIMED OUT)\b"),
     _agent_key, None, lambda m: m.group("end") in ("DONE", "PARTIAL")),
    (re.compile(r"JOB: (?P<id>job-\w+) (?:started in the background|queued) — (?P<label>.+)"),
     lambda m: "job:" + m.group("id"), lambda m: m.group("label"), None),
    (re.compile(r"JOB: (?P<id>job-\w+) (?P<verdict>\w+) after "),
     lambda m: "job:" + m.group("id"), None, lambda m: m.group("verdict") == "finished"),
    (re.compile(r"INSTALL: (?P<id>\S+) started — (?P<name>.+)"),
     lambda m: "install:" + m.group("id"), lambda m: "Installing " + m.group("name"), None),
    (re.compile(r"INSTALL: (?P<id>\S+) (?P<how>finished|failed) — "),
     lambda m: "install:" + m.group("id"), None, lambda m: m.group("how") == "finished"),
    (re.compile(r"TOOL: (?P<tool>\S+) still running after"),
     lambda m: "tool:" + m.group("tool"), lambda m: m.group("tool") + " (background)", None),
    (re.compile(r"TOOL: (?P<tool>\S+) finished (?:in the background - (?P<how>\w+)|after it was cancelled)"),
     lambda m: "tool:" + m.group("tool"), None, lambda m: m.group("how") == "done"),
    # A research run paces itself over its minutes and was never on the lane:
    # "it says he's researching but it's not showing on the brain" (owner,
    # 30 Oct 2026). One run at a time (research.py), so one key.
    (re.compile(r"RESEARCH: started '(?P<topic>.+?)' → "), lambda m: "research",
     lambda m: "Researching " + m.group("topic"), None),
    (re.compile(r"RESEARCH: (?P<state>completed|failed) — "), lambda m: "research", None,
     lambda m: m.group("state") == "completed"),
    (re.compile(r"RESEARCH: (?:cancelled '|fault - )"), lambda m: "research", None, lambda m: False),
)


class _MirroredModel(BrainModel):
    """ORION's copy of the Brain model; each event also goes to the host."""

    def __init__(self, forward: Callable[..., None]) -> None:
        super().__init__()
        self._forward = forward

    def heard(self) -> None:
        super().heard()
        self._forward("heard")

    def heard_utterance(self) -> None:
        super().heard_utterance()
        self._forward("heard_utterance")

    def set_state(self, state: str) -> None:
        super().set_state(state)
        self._forward("set_state", state)

    def set_speaking(self, active: bool) -> None:
        super().set_speaking(active)
        self._forward("set_speaking", bool(active))

    def tool_started(self, names: str) -> None:
        super().tool_started(names)
        self._forward("tool_started", names)

    def tool_finished(self, name: str, ok: bool = True,
                      elapsed_ms: float | None = None) -> None:
        super().tool_finished(name, ok, elapsed_ms)
        self._forward("tool_finished", name, bool(ok), elapsed_ms)

    def set_telemetry(self, cpu: float | None, ram: float | None,
                      gpu: float | None = None) -> None:
        super().set_telemetry(cpu, ram, gpu)
        self._forward("set_telemetry", cpu, ram, gpu)

    def work_started(self, key: str, label: str = "") -> None:
        super().work_started(key, label)
        self._forward("work_started", key, label)

    def work_finished(self, key: str, ok: bool = True) -> None:
        super().work_finished(key, ok)
        self._forward("work_finished", key, bool(ok))

    def set_link(self, provider: str, state: str) -> None:
        super().set_link(provider, state)
        self._forward("set_link", provider, state)


class _RemoteAnimator:
    """The local animator's interface, driving the host's instead."""

    def __init__(self, page: "CommandOverview") -> None:
        self.page = page
        self.visible = False
        self.attended = True

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        if self.page._remote is not None:
            self.page._remote.set_visible(self.visible)

    def set_attended(self, attended: bool) -> None:
        self.attended = bool(attended)
        if self.page._remote is not None:
            self.page._remote.set_attended(self.attended)

    def kick(self) -> None:
        pass


#: The BRAIN page's activity graph: the request lanes or the ECG trace.
#: Owner, 28 Oct 2026: "there must be a graph setting (kind of looking like
#: a ECG machine)". Kept in its own file so a deck update never resets it.
BRAIN_VIEW_PATH = CONFIG_DIR / "brain_view.json"
GRAPH_STYLES = ("Lanes", "ECG", "Voice")


def brain_graph(path: Any = None) -> str:
    import json
    from pathlib import Path
    try:
        data = json.loads(Path(path or BRAIN_VIEW_PATH).read_text(encoding="utf-8"))
        style = str(data.get("graph") or "")
    except (OSError, ValueError, AttributeError):
        return GRAPH_STYLES[0]
    return style if style in GRAPH_STYLES else GRAPH_STYLES[0]


def save_brain_graph(style: str, path: Any = None) -> None:
    import json
    from pathlib import Path

    from ..atomic_io import atomic_write_text
    target = Path(path or BRAIN_VIEW_PATH)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, json.dumps({"graph": style}), encoding="utf-8")
    except OSError:
        pass


class CommandOverview(QWidget):
    """The Brain page; each signal updates only what it changes."""

    QUICK_LINKS = ("RESEARCH", "MISSION", "COMMAND CENTRE", "DIAGNOSTICS", "LOG")
    #: The sidebar's icon for each quick link, so a page looks the same twice.
    _LINK_ICONS = {"RESEARCH": "search", "MISSION": "flag", "COMMAND CENTRE": "gauge",
                   "DIAGNOSTICS": "activity", "LOG": "terminal"}

    def __init__(self, bus: OrionBus,
                 on_open_page: Callable[[str], Any] | None = None, *,
                 remote_instruments: bool = False) -> None:
        super().__init__()
        self.bus = bus
        self._open_page = on_open_page
        self.deck_pages: dict[str, str] = {}
        self._recent: deque[str] = deque(maxlen=5)
        # The composition root asks for the Brain host in a real run; a page
        # built anywhere else (the tests) draws its instruments itself.
        self._remote_wanted = bool(remote_instruments)
        self._remote: Any = None
        self.model = (_MirroredModel(self._forward) if self._remote_wanted
                      else BrainModel())

        from ..constants import SPACE

        # The hierarchy, top to bottom: the live instruments (the thing to
        # look at); one quiet column of plain facts;
        # the request timeline; a row of places to go next. The page name is
        # the deck header's; it is not repeated here.
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE.MD, SPACE.SM, SPACE.MD, SPACE.MD)
        root.setSpacing(SPACE.SM + 4)

        body = QHBoxLayout()
        body.setSpacing(SPACE.SM + 4)
        iris_frame = QFrame()
        iris_frame.setObjectName("panelFrame")
        iris_box = QVBoxLayout(iris_frame)
        iris_box.setContentsMargins(8, 8, 8, 8)
        if self._remote_wanted:
            self.iris = self._slot("ORION state gauge", min_size=(260, 260))
        else:
            self.iris = IrisGauge(self.model)
            self.iris.state_label = display_state
        iris_box.addWidget(self.iris)
        body.addWidget(iris_frame, 3)
        self._instrument_frames = [iris_frame]

        # The plain facts, as ONE panel of short rows rather than five cards:
        # five equal boxes, each with a 16 px headline, read as five things
        # competing for attention, and three of them repeated the gauge. The
        # state pill heads the panel because it is the one live thing in it.
        facts = QVBoxLayout()
        facts.setSpacing(SPACE.SM + 4)
        now = QFrame()
        now.setObjectName("panelFrame")
        now.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._now_box = QVBoxLayout(now)
        self._now_box.setContentsMargins(SPACE.MD - 2, SPACE.SM + 4, SPACE.MD - 2, SPACE.SM + 4)
        self._now_box.setSpacing(SPACE.SM + 2)
        now_head = QHBoxLayout()
        now_head.addWidget(self._heading("Now"))
        now_head.addStretch(1)
        self.mode = QLabel("Ready")
        self.mode.setObjectName("deckStatePill")
        self.mode.setAccessibleName("ORION's state")
        now_head.addWidget(self.mode, 0, Qt.AlignmentFlag.AlignVCenter)
        self._now_box.addLayout(now_head)
        self.link = self._fact("Live Voice", "Waiting for connection")
        self.model_route = self._fact("Answers", "No answer requested yet")
        self.route = self._fact("Latest Action", "No actions requested yet")
        self.host = self._fact("Computer Use", "Processor —   ·   Memory —")
        self.hearing = self._fact("What I Notice", "No sound alerts")
        self.context = QLabel("Camera off\nSound watch off\nCloud frames off")
        self.context.setObjectName("mutedLabel")
        self.context.setAccessibleName("Current local sensing and cloud sharing")
        self.context.setWordWrap(True)
        self.context.ensurePolished()
        self.context.setMinimumHeight(3 * self.context.fontMetrics().lineSpacing() + 4)
        self._now_box.addWidget(self.context)
        self._context_sources: dict[str, Any] = {}
        self._cloud_summary = "Cloud frames off"
        facts.addWidget(now)
        activity = QFrame()
        activity.setObjectName("panelFrame")
        activity_box = QVBoxLayout(activity)
        activity_box.setContentsMargins(SPACE.MD - 2, SPACE.SM + 4, SPACE.MD - 2, SPACE.SM + 4)
        activity_box.addWidget(self._heading("Recent Activity"))
        self.activity = QLabel("Ready")
        self.activity.setObjectName("mutedLabel")
        self.activity.setWordWrap(True)
        self.activity.setAlignment(Qt.AlignmentFlag.AlignTop)
        activity_box.addWidget(self.activity, 1)
        facts.addWidget(activity, 1)
        facts_holder = QWidget()
        facts_holder.setLayout(facts)
        # Scrolls rather than squeezes. With a fixed width and the window's
        # height, Qt shrank the cards below their content on a shorter
        # window: headings were cut in half ("LIVE CHANNEL") and the sensing
        # lines ran over the card border.
        facts_scroll = QScrollArea()
        facts_scroll.setObjectName("zoneScroll")
        facts_scroll.setWidgetResizable(True)
        facts_scroll.setFrameShape(QFrame.Shape.NoFrame)
        facts_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        facts_scroll.setWidget(facts_holder)
        facts_scroll.setFixedWidth(348)
        body.addWidget(facts_scroll)
        root.addLayout(body, 1)

        lanes_frame = QFrame()
        lanes_frame.setObjectName("panelFrame")
        lanes_box = QVBoxLayout(lanes_frame)
        lanes_box.setContentsMargins(SPACE.XS, SPACE.XS, SPACE.XS, SPACE.XS)
        lanes_box.setSpacing(SPACE.XS)
        if self._remote_wanted:
            self.lanes = self._slot("Request timeline: hear, think, act, speak and recall",
                                    height=190)
        else:
            self.lanes = SignalLanes(self.model)
            self.lanes.stalled_count = _stalled_requests
        # The same activity as an ECG trace, drawn here (well under a
        # millisecond a frame, and only while it is the one shown).
        from PyQt6.QtWidgets import QButtonGroup, QStackedWidget
        from .brain_instruments import EcgMonitor
        self.ecg = EcgMonitor()
        self.ecg.setMinimumHeight(190)
        self.graph_stack = QStackedWidget()
        self.graph_stack.addWidget(self.lanes)
        self.graph_stack.addWidget(self.ecg)
        # The owner's voice and his as a live ridgeline (28 Oct 2026).
        from .hud_instruments import VoiceRidgeline
        self.voice = VoiceRidgeline(bus)
        self.voice.setMinimumHeight(190)
        self.graph_stack.addWidget(self.voice)
        graph_head = QHBoxLayout()
        graph_head.setContentsMargins(SPACE.SM, SPACE.XS, SPACE.XS, 0)
        graph_head.addWidget(self._heading("Activity"))
        graph_head.addStretch(1)
        self.graph_toggle = QFrame()
        self.graph_toggle.setObjectName("segmented")
        toggle_row = QHBoxLayout(self.graph_toggle)
        toggle_row.setContentsMargins(3, 3, 3, 3)
        toggle_row.setSpacing(2)
        self._graph_group = QButtonGroup(self.graph_toggle)
        self._graph_group.setExclusive(True)
        self._graph_buttons: dict[str, QPushButton] = {}
        for style in GRAPH_STYLES:
            button = QPushButton(style)
            button.setObjectName("formItem")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAccessibleName(f"Show activity as {style}")
            self._graph_group.addButton(button)
            toggle_row.addWidget(button)
            self._graph_buttons[style] = button
        graph_head.addWidget(self.graph_toggle, 0, Qt.AlignmentFlag.AlignVCenter)
        lanes_box.addLayout(graph_head)
        lanes_box.addWidget(self.graph_stack)
        self.show_graph(brain_graph(), remember=False)
        self._graph_group.buttonToggled.connect(
            lambda button, on: on and self.show_graph(button.text()))
        root.addWidget(lanes_frame)
        self._instrument_frames.append(lanes_frame)

        # Where to go next: labelled as such, with the same line icons as the
        # sidebar, rather than five loose words under the timeline.
        from .line_icons import apply as apply_icon

        links = QHBoxLayout()
        links.setSpacing(SPACE.SM)
        jump = QLabel("Jump To")
        jump.setObjectName("summaryKey")
        links.addWidget(jump, 0, Qt.AlignmentFlag.AlignVCenter)
        self._quick_links: dict[str, QPushButton] = {}
        for name in self.QUICK_LINKS:
            label = page_display_name(name)
            button = QPushButton(label)
            button.setObjectName("ghostButton")
            apply_icon(button, self._LINK_ICONS.get(name, "document"))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAccessibleName(f"Open {label}")
            button.clicked.connect(lambda _checked=False, page=name: self._navigate(page))
            self._quick_links[name] = button
            links.addWidget(button)
        links.addStretch(1)
        root.addLayout(links)

        self._animator: Any = (_RemoteAnimator(self) if self._remote_wanted
                               else _Animator(self.model, [self.iris, self.lanes]))
        # A heard utterance ends when the speaker goes quiet, which no signal
        # announces; while hidden nothing needs drawing, so nothing polls.
        self._settle = QTimer(self)
        self._settle.setInterval(500)
        self._settle.timeout.connect(self.model.settle)
        self._settle.timeout.connect(self._refresh_context)
        if self._remote_wanted:
            self._settle.timeout.connect(self._send_stalled)

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

    def _fact(self, title: str, initial: str) -> QLabel:
        """One row of the Now panel: a quiet name over its current value."""
        name = QLabel(title)
        name.setObjectName("summaryKey")
        value = QLabel(initial)
        value.setObjectName("summaryValue")
        value.setWordWrap(True)
        value.setAccessibleName(title)
        row = QVBoxLayout()
        row.setSpacing(1)
        row.addWidget(name)
        row.addWidget(value)
        self._now_box.addLayout(row)
        return value

    @staticmethod
    def _set(label: QLabel, value: str) -> None:
        if label.text() != value:
            label.setText(value)

    @staticmethod
    def _slot(name: str, *, min_size: tuple[int, int] | None = None,
              height: int | None = None) -> QWidget:
        """Where a host-drawn instrument goes (or the local one, on fallback)."""
        slot = QWidget()
        slot.setAccessibleName(name)
        box = QVBoxLayout(slot)
        box.setContentsMargins(0, 0, 0, 0)
        if min_size is not None:
            slot.setMinimumSize(*min_size)
            slot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if height is not None:
            slot.setFixedHeight(height)
            slot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return slot

    # ── the Brain host (Mark XXXV W1.2) ──────────────────────────────────────

    def _forward(self, method: str, *args: Any) -> None:
        remote = self._remote
        if remote is not None and not remote.failed:
            remote.forward(method, *args)

    def _start_remote(self) -> None:
        from . import page_host
        if not page_host.enabled("brain"):
            self._fall_back("switched off")
            return
        background = _instrument_background()
        try:
            remote = page_host.RemoteBrain(self, background=background)
        except Exception as exc:
            self._fall_back(str(exc))
            return
        remote.hide()                   # only its two parts are shown, in the slots
        remote.on_failed = lambda: self._fall_back("the Brain process could not run")
        remote.snapshot_provider = self.model.snapshot
        self._remote = remote
        for frame in self._instrument_frames:
            frame.setStyleSheet(f"QFrame#panelFrame {{ background: {background}; }}")
        self.iris.layout().addWidget(remote.part("iris"))
        self.lanes.layout().addWidget(remote.part("lanes"))
        remote.forward("load_snapshot", self.model.snapshot())
        remote.set_attended(self._animator.attended)
        remote.set_visible(self.isVisible())

    def _fall_back(self, reason: str) -> None:
        """Draw the instruments here, as before the host existed."""
        if not self._remote_wanted:
            return
        self._remote_wanted = False
        remote, self._remote = self._remote, None
        if remote is not None:
            for name, slot in (("iris", self.iris), ("lanes", self.lanes)):
                try:
                    part = remote.part(name)
                    slot.layout().removeWidget(part)
                    part.hide()
                except Exception:
                    pass
            remote.shutdown()
        for frame in self._instrument_frames:
            frame.setStyleSheet("")
        iris = IrisGauge(self.model)
        iris.state_label = display_state
        lanes = SignalLanes(self.model)
        lanes.stalled_count = _stalled_requests
        self.iris.layout().addWidget(iris)
        self.lanes.layout().addWidget(lanes)
        attended = self._animator.attended
        self._animator = _Animator(self.model, [iris, lanes])
        self._animator.set_attended(attended)
        self._animator.set_visible(self.isVisible())
        try:
            from .. import session_log
            session_log.write(f"BRAIN: instruments drawn in ORION's process - {reason}")
        except Exception:
            pass

    def _send_stalled(self) -> None:
        if self._remote is not None:
            self._remote.set_stalled(_stalled_requests())

    def shutdown(self) -> None:
        if self._remote is not None:
            self._remote.shutdown()

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
        self._set(self.mode, _STATES.get(text, text.replace("_", " ").capitalize()))
        self.model.set_state(text)

    def _on_speaking(self, active: Any) -> None:
        if active:
            self._set(self.mode, "Speaking")
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
        load = f"Processor {cpu:.0f}%   ·   Memory {ram:.0f}%"
        if gpu is not None:
            load += f"   ·   Graphics {gpu:.0f}%"
        self._set(self.host, load)
        self.model.set_telemetry(cpu, ram, gpu)

    def _on_activity(self, agent: str, summary: str) -> None:
        self._add_activity(f"{agent}: {summary}")

    def show_graph(self, style: str, *, remember: bool = True) -> None:
        """Lanes or ECG for the activity graph; *remember* keeps the choice."""
        style = style if style in GRAPH_STYLES else GRAPH_STYLES[0]
        self.graph_stack.setCurrentWidget({"ECG": self.ecg, "Voice": self.voice}.get(style, self.lanes))
        button = self._graph_buttons[style]
        if not button.isChecked():
            self._graph_group.blockSignals(True)
            button.setChecked(True)
            self._graph_group.blockSignals(False)
        if remember:
            save_brain_graph(style)

    def _on_log(self, message: str) -> None:
        text = str(message)
        self.ecg.feed(text)
        if text.startswith("TRACE: "):
            # Request traces: a local transcript marks an utterance heard; a
            # tool start opens the Act lane until its result is reported.
            if " TRANSCRIPTION_COMPLETE" in text:
                self.model.heard_utterance()
            elif " TOOL_EXECUTION_START" in text:
                names = text.split(" — ", 1)[1] if " — " in text else ""
                self.model.tool_started(names)
            return
        self._note_work(text)
        if text.startswith(ACTIVITY_PREFIXES):
            self._add_activity(text)

    def _note_work(self, text: str) -> None:
        """Background work from the log onto the WORK lane (Mark XXXX.4)."""
        for pattern, key_of, label_of, outcome in _WORK_LINES:
            match = pattern.match(text)
            if match is None:
                continue
            key = key_of(match)
            if outcome is None:
                self.model.work_started(key, label_of(match))
            else:
                self.model.work_finished(key, outcome(match))
            return

    def _on_dashboard_event(self, channel: str, payload: Any) -> None:
        if channel == "research" and isinstance(payload, dict) and payload.get("run") \
                and payload.get("engine") != "timed":
            # Dossiers, reading runs and papers: one bar each while they run
            # (the timed run has its own, from its log lines).
            key = "research:" + str(payload["run"])
            if payload.get("status") == "running":
                self.model.work_started(key, "Researching " + str(payload.get("topic") or "")[:80])
            else:
                self.model.work_finished(key, payload.get("status") == "done")
            return
        if channel == "perception" and isinstance(payload, dict) and "cloud_sharing" in payload:
            remaining = payload.get("cloud_calls_remaining", 0)
            if type(remaining) is not int or remaining < 0:
                remaining = 0
            self._cloud_summary = (f"Cloud frames on · {remaining} calls left"
                                   if payload["cloud_sharing"] is True else "Cloud frames off")
            self._refresh_context()
            return
        if channel == "context" and isinstance(payload, dict):
            sources = payload.get("sources")
            if isinstance(sources, dict):
                self._context_sources = sources
                self._refresh_context()
            return
        if channel == "voice_heard":
            self.model.heard()
        elif channel == "provider_route" and isinstance(payload, dict):
            where = "on this computer" if payload.get("local") else "online"
            text = f"{payload.get('provider', '?')}  ·  {where}"
            try:
                text += f"  ·  {float(payload.get('latency_s')):.1f} s"
            except (TypeError, ValueError):
                pass
            if payload.get("fallbacks"):
                text += f"  ·  {plural(int(payload['fallbacks']), 'alternative')} tried"
            self._set(self.model_route, text)
        elif channel == "provider_breaker" and isinstance(payload, dict):
            name = str(payload.get("provider") or "?")
            if self.model_route.text().startswith(name + "  "):
                try:
                    wait = f" {float(payload.get('cooldown_s')):.0f} s"
                except (TypeError, ValueError):
                    wait = ""
                self._set(self.model_route, f"{name}  ·  retrying in{wait}")
        elif channel == "sound_alert" and isinstance(payload, dict):
            category = str(payload.get("category") or "sound").capitalize()
            stamp = time.strftime("%H:%M:%S")
            self._set(self.hearing, f"{category}  ·  {stamp}")
            self._add_activity(f"SOUND: {category.lower()} heard at {stamp}")

    def _refresh_context(self) -> None:
        lines = []
        now = time.time()
        for source, label in (("camera.local", "Camera"), ("audio.local", "Sound watch")):
            state = self._context_sources.get(source, {"state": "off"})
            if not isinstance(state, dict):
                continue
            status = state.get("state", "off")
            if status not in {"active", "starting", "off", "degraded"}:
                status = "degraded"
            fresh_until = state.get("fresh_until")
            if status == "active" and (not isinstance(fresh_until, (int, float))
                                        or fresh_until <= now):
                status = "degraded"
            last = state.get("last_event_at")
            recent = (f" · event {max(0, now - last):.0f}s ago"
                      if isinstance(last, (int, float)) else "")
            lines.append(f"{label} {status}{recent}")
        self._set(self.context, "\n".join([*lines, self._cloud_summary]))

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
        self._refresh_context()
        self._settle.start()
        if self._remote_wanted and self._remote is None:
            self._start_remote()        # first opening: the host costs nothing before it
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
        title = command_title(label)
        timing = f"  ·  {elapsed_ms / 1000.0:.1f} s" if elapsed_ms else ""
        self._set(self.route, f"{title}  ·  {'Done' if ok else 'Could not finish'}{timing}")
        self._add_activity(f"{title}: {'completed' if ok else 'could not finish'}")
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
