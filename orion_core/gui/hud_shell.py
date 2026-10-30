"""
The shell around ORION's face.

The Core Window was a header and a face in a splitter. Everything else — the
log, the telemetry, the content, a place to type — lived on the Command Deck,
a second window. That is two windows to open, two to arrange, and a glance at
the assistant tells you nothing about what it is doing.

So this is the chrome: a telemetry rail down the left, the activity log and a
place to type down the right, found material along the bottom, status along
the bottom edge. One window that answers "what is it doing" without a click.

What this module does NOT do
----------------------------
It does not build, wrap, restyle or touch the face or the orb. ``HudShell``
takes a centre widget and puts it in the middle; what that widget is, and how
it paints, is entirely the caller's business. The face has been rebuilt enough
times to have earned being left alone, and every widget here is deliberately
incapable of interfering with it.

Everything is fed by signals that already existed — ``telemetry_sample``,
``log``, ``content_results``, ``state`` — so the HUD is a new view of ORION,
not a new source of truth about him.

Cheap by construction
---------------------
The Qt thread is the asyncio event loop, so paint time here is stolen directly
from audio and the live socket. Every widget in this file is text and simple
rectangles: no gradients animated per frame, no per-frame layout, no timers
except the one-second clock. Meters only repaint when their value actually
changes, because a bar redrawn at the same width is pure cost.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..constants import C, ELEV, GLASS, SPACE
from .widgets import describe_control

#: The rail's width. Wide enough for "Graphics  100%" with the meter's own
#: padding round it (132 px put the name and the value edge to edge), narrow
#: enough that the face keeps the room it needs.
RAIL_WIDTH = 152

#: The right column's width. This one holds readable log lines, so it is
#: measured from text rather than chosen: ~46 monospace characters, plus the
#: panels' padding.
COLUMN_WIDTH = 320

#: Padding inside every HUD panel (left, top, right, bottom): one value, so
#: the meters, the feed, the drop zone and the command bar share an inner
#: edge instead of four (6, 8, 8 and 6 px before).
PANEL_PADDING = (SPACE.SM + SPACE.XS, SPACE.SM, SPACE.SM + SPACE.XS, SPACE.SM)

#: Most log lines kept. The activity log is a view of the last few minutes;
#: the journals on disk are the record.
LOG_LINES = 400


#: Fixed-pitch families in preference order. A fixed pitch is what makes a
#: column of meters line up; a proportional fallback makes every row a
#: different width. Handed to Qt as a list so its own font matching walks
#: them — asking `QFontDatabase.families()` which of them exists enumerates
#: and sorts every font installed on the machine, which measured 111 ms of a
#: 127 ms window build, on the one code path whose whole job is to open fast.
MONO_FAMILIES = ("JetBrains Mono", "Cascadia Mono", "Consolas",
                 "DejaVu Sans Mono", "Courier New")

#: Built fonts, keyed by (size, bold). The HUD asks for about thirty fonts
#: while it builds and wants roughly six distinct ones; a QFont is immutable
#: as far as this module uses it, so they are shared.
_FONTS: dict[tuple[int, bool], QFont] = {}


def mono(size: int = 9, *, bold: bool = False) -> QFont:
    """ORION's HUD typeface at *size*."""
    key = (int(size), bool(bold))
    font = _FONTS.get(key)
    if font is None:
        font = QFont()
        font.setFamilies(list(MONO_FAMILIES))
        # If none of them is installed, this is what still gets a monospaced
        # face rather than a proportional one.
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(int(size))
        font.setBold(bool(bold))
        _FONTS[key] = font
    return font


#: The friendly face for words (Mark XXXV): headings, labels and the feed are
#: read, not scanned, so they are set in a humanist sans rather than the
#: terminal monospace the HUD used for everything.
UI_FAMILIES = ("Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", "Arial")
_UI_FONTS: dict[tuple[int, bool], QFont] = {}


def ui(size: int = 9, *, bold: bool = False) -> QFont:
    """ORION's reading typeface at *size*."""
    key = (int(size), bool(bold))
    font = _UI_FONTS.get(key)
    if font is None:
        font = QFont()
        font.setFamilies(list(UI_FAMILIES))
        font.setPointSize(int(size))
        if bold:
            font.setWeight(QFont.Weight.DemiBold)
        _UI_FONTS[key] = font
    return font


def _section(title: str) -> QLabel:
    """The caption above each panel - a name in Title Case, as you would say it
    ("Your PC", not "▸ SYS MONITOR")."""
    label = QLabel(title)
    label.setObjectName("hudSection")
    label.setFont(ui(9, bold=True))
    return label


# ── the left rail ─────────────────────────────────────────────────────────────

def _styled_background(widget: "QWidget") -> None:
    """Let a plain QWidget actually paint the background the sheet gives it.

    Qt only honours a stylesheet ``background`` on a QWidget subclass if
    WA_StyledBackground is set; QFrame paints one either way. That is why the
    meters and the drop zone had surfaces while the activity log and the
    command bar sat directly on the void — the rule was written, applied and
    silently ignored, with nothing raised to say so.
    """
    try:
        widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    except Exception:
        pass


class Meter(QFrame):
    """One telemetry reading: a name, a value, and a bar.

    The bar is a plain child widget resized to the percentage rather than a
    QProgressBar, which brings a whole style pipeline for something that is
    one filled rectangle.
    """

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudMeter")
        self._percent = -1.0                      # forces the first paint

        self.name_label = QLabel(name)
        self.name_label.setObjectName("hudMeterName")
        self.name_label.setFont(ui(8))
        self.value_label = QLabel("N/A")
        self.value_label.setObjectName("hudMeterValue")
        self.value_label.setFont(mono(8, bold=True))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(SPACE.SM)
        top.addWidget(self.name_label)
        top.addStretch(1)
        top.addWidget(self.value_label)

        self.track = QWidget()
        self.track.setObjectName("hudTrack")
        self.track.setFixedHeight(3)
        self.fill = QWidget(self.track)
        self.fill.setObjectName("hudFill")
        self.fill.setGeometry(0, 0, 0, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PANEL_PADDING)
        layout.setSpacing(SPACE.SM)
        layout.addLayout(top)
        layout.addWidget(self.track)

        describe_control(self, f"{name} usage", f"Current {name} load.")

    def set_value(self, percent: float | None, text: str = "") -> None:
        """Update the reading. ``None`` means the machine cannot report it."""
        if percent is None:
            self.value_label.setText(text or "N/A")
            if self._percent != -1.0:
                self._percent = -1.0
                self.fill.setFixedWidth(0)
            return
        value = max(0.0, min(100.0, float(percent)))
        self.value_label.setText(text or f"{value:.0f}%")
        # A bar redrawn at the same width is pure cost on the loop that also
        # carries audio, so only move it when it has actually moved.
        if abs(value - self._percent) < 0.5:
            return
        self._percent = value
        self._resize_fill()

    def _resize_fill(self) -> None:
        if self._percent < 0:
            return
        width = max(0, int(self.track.width() * self._percent / 100.0))
        self.fill.setFixedWidth(width)
        # The bar earns its colour: nominal, busy, saturated. Nominal used to
        # be the accent, which made a CPU reading nobody needs to act on the
        # loudest thing on the rail — four crimson bars sitting at 20% teach
        # the eye to ignore crimson, and then a real fault arrives in a colour
        # that has stopped meaning anything.
        colour = (C.BAD if self._percent >= 90 else
                  C.WARN if self._percent >= 70 else C.GOOD)
        self.fill.setStyleSheet(f"#hudFill {{ background: {colour}; }}")

    def resizeEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        super().resizeEvent(event)
        self._resize_fill()


class SysMonitorRail(QWidget):
    """The left rail: what the machine is doing, at a glance.

    Fed by ``bus.telemetry_sample``, which already existed and already ran at
    0.75 s. Readings the host cannot produce show N/A rather than zero — a
    machine with no discrete GPU reporting "0%" reads as an idle GPU, which is
    a different and wrong claim.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudRail")
        self.setFixedWidth(RAIL_WIDTH)

        self.meters: dict[str, Meter] = {}
        layout = QVBoxLayout(self)
        # No margin of its own: the meters line up with the header's edge
        # and the window's margin is the gutter, rather than a second one.
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE.SM)
        layout.addWidget(_section("Your PC"))

        for key, name in (("cpu", "Processor"), ("ram", "Memory"), ("net", "Network"),
                          ("gpu", "Graphics"), ("tmp", "Heat")):
            meter = Meter(name)
            self.meters[key] = meter
            layout.addWidget(meter)

        self.facts = QLabel("ORION has just started")
        self.facts.setObjectName("hudFacts")
        self.facts.setFont(ui(8))
        self.facts.setWordWrap(True)
        facts_frame = QFrame()
        facts_frame.setObjectName("hudMeter")
        facts_layout = QVBoxLayout(facts_frame)
        facts_layout.setContentsMargins(*PANEL_PADDING)
        facts_layout.addWidget(self.facts)
        layout.addWidget(facts_frame)
        layout.addStretch(1)

    def apply_sample(self, sample: Any) -> None:
        """Render one telemetry sample. Never raises on a malformed one."""
        if not isinstance(sample, dict):
            return
        self.meters["cpu"].set_value(sample.get("cpu"))
        self.meters["ram"].set_value(sample.get("ram"))

        net_bps = sample.get("net_bps")
        if net_bps is None:
            self.meters["net"].set_value(None)
        else:
            self.meters["net"].set_value(sample.get("net_percent") or 0.0,
                                         _rate(float(net_bps)))
        # GPU and temperature are supplied by the shared telemetry loop. The
        # probe runs off the qasync/audio thread; this widget only renders the
        # cached result. TMP shows a percentage of a 100°C ceiling and the
        # physical reading together, so the bar and the number agree without
        # hiding the useful unit.
        self.meters["gpu"].set_value(sample.get("gpu"))
        temperature = sample.get("temperature")
        if temperature is None:
            self.meters["tmp"].set_value(None)
        else:
            temp = float(temperature)
            thermal_percent = sample.get("temperature_percent", temp)
            thermal_percent = max(0.0, min(100.0, float(thermal_percent)))
            self.meters["tmp"].set_value(
                thermal_percent, f"{thermal_percent:.0f}% {temp:.0f}°C")

    def set_active(self, duration: str = "") -> None:
        """How long ORION has been running: the one fact under the meters
        (owner, 26 Oct 2026 - the PC's uptime, process count and OS were noise)."""
        duration = str(duration or "").strip()
        self.facts.setText(f"ORION has been active for {duration}" if duration
                           else "ORION has just started")


def _rate(bytes_per_second: float) -> str:
    """A transfer rate a person can read at a glance."""
    if bytes_per_second >= 1_000_000:
        return f"{bytes_per_second / 1_000_000:.1f}MB/s"
    if bytes_per_second >= 1000:
        return f"{bytes_per_second / 1000:.0f}KB/s"
    return f"{bytes_per_second:.0f}B/s"


# ── the right column ──────────────────────────────────────────────────────────

class ActivityLog(QWidget):
    """What ORION has been doing, in his own words.

    Read-only and capped. It is a view of the last few minutes; the journals
    on disk are the record, and an unbounded document in a widget that
    repaints on the event loop is how a long session starts to stutter.

    Built after the first paint
    ---------------------------
    The first ``QPlainTextEdit`` in a process costs about 200 ms — Qt bringing
    up its whole text and font machinery — and every one after it costs a
    tenth of a millisecond. Measured, on this machine: 203.1 ms, then 0.1 ms.

    Paying that during construction means the window cannot appear until it is
    done, which is 200 ms of nothing on the one code path whose job is to open
    quickly. So the widget is created on a zero-delay timer instead: the shell
    paints first, the log arrives on the next turn of the event loop, and any
    lines that arrive in between are buffered rather than lost.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudLogPanel")
        _styled_background(self)
        self.view: QPlainTextEdit | None = None
        self._pending: list[str] = []
        # Every line as written, so "Details" can show the engineering log and
        # switching back re-tells it in the human version (gui/humane.py).
        from collections import deque
        self._raw: deque[str] = deque(maxlen=LOG_LINES)
        self._details = False

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*PANEL_PADDING)
        self._layout.setSpacing(SPACE.SM)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(_section("Terminal"))
        header.addStretch(1)
        self.details_btn = QPushButton("Details")
        self.details_btn.setObjectName("hudDetails")
        self.details_btn.setCheckable(True)
        self.details_btn.setFont(ui(8))
        self.details_btn.toggled.connect(self.set_details)
        describe_control(self.details_btn, "Show every detail",
                         "Show ORION's full technical log instead of the summary.")
        header.addWidget(self.details_btn)
        self._layout.addLayout(header)
        self._layout.addStretch(1)

        QTimer.singleShot(0, self._build_view)

    # ── the human version of a line ──────────────────────────────────────────

    @staticmethod
    def _render(said: Any) -> str:
        """One line of the feed as HTML: you in the human tone, ORION in light,
        events quietly, trouble a step brighter - never in crimson."""
        import html
        text = html.escape(said.text)
        label = html.escape(said.label)
        if said.kind == "you":
            return (f'<span style="color:{C.HUMAN}; font-weight:600;">You</span>'
                    f'&nbsp;&nbsp;<span style="color:{C.WHITE};">{text}</span>')
        if said.kind == "orion":
            return (f'<span style="color:{C.ACCENT}; font-weight:600;">ORION</span>'
                    f'&nbsp;&nbsp;<span style="color:{C.SILVER};">{text}</span>')
        lead = f"{label} &middot; " if label else ""
        if said.kind == "trouble":
            return f'<span style="color:{C.WARN};">&#9888;&nbsp;{lead}{text}</span>'
        return f'<span style="color:{C.MUTED};">{lead}{text}</span>'

    def _show(self, line: str) -> None:
        """Put one raw line into the view, in the current mode."""
        if self.view is None:
            return
        if self._details:
            self.view.appendPlainText(line)
            return
        from .humane import humanise
        said = humanise(line)
        if said is not None:
            self.view.appendHtml(self._render(said))

    def set_details(self, on: bool) -> None:
        """Switch between the human summary and the full technical log."""
        self._details = bool(on)
        if self.details_btn.isChecked() != self._details:
            self.details_btn.setChecked(self._details)
        if self.view is None:
            return
        self.view.clear()
        for line in self._raw:
            self._show(line)
        self._scroll_to_end()

    def _build_view(self) -> None:
        """Create the real text widget and flush anything that arrived first."""
        if self.view is not None:
            return
        view = QPlainTextEdit()
        view.setObjectName("hudLog")
        view.setReadOnly(True)
        view.setFont(ui(9))
        view.setMaximumBlockCount(LOG_LINES)
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        view.setFrameShape(QFrame.Shape.NoFrame)
        describe_control(view, "What's been happening",
                         "Your conversation with ORION and what he has done, most recent last.")
        # Replace the stretch that was holding the space open.
        item = self._layout.takeAt(self._layout.count() - 1)
        del item
        self._layout.addWidget(view, 1)
        self.view = view
        if self._pending:
            for line in self._pending[-LOG_LINES:]:
                self._show(line)
            self._pending.clear()
            self._scroll_to_end()

    def append(self, line: Any) -> None:
        text = str(line or "").rstrip()
        if not text:
            return
        self._raw.append(text)
        if self.view is None:
            # Buffered, not dropped. Boot is the noisiest moment in ORION's
            # life and it happens before this widget exists.
            self._pending.append(text)
            del self._pending[:-LOG_LINES]
            return
        bar = self.view.verticalScrollBar()
        following = bar is None or bar.value() >= bar.maximum() - 2
        self._show(text)
        if following:            # a reader scrolled back up keeps their place
            self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        if self.view is None:
            return
        bar = self.view.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def clear(self) -> None:
        self._pending.clear()
        self._raw.clear()
        if self.view is not None:
            self.view.clear()

    @property
    def line_count(self) -> int:
        if self.view is None:
            return len(self._pending)
        return self.view.blockCount() if self.view.toPlainText() else 0


class DropZone(QFrame):
    """Somewhere to put a file.

    Accepts a drop and hands the paths to a callback; it neither reads nor
    interprets them, because what a dropped file means is the dispatcher's
    decision, not a widget's.
    """

    dropped = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudDrop")
        self.setAcceptDrops(True)
        self.setMinimumHeight(72)

        self.prompt = QLabel("Drop a file")
        self.prompt.setObjectName("hudDropPrompt")
        self.prompt.setFont(ui(9))
        self.prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PANEL_PADDING)
        layout.setSpacing(2)
        layout.addWidget(self.prompt)
        describe_control(self, "File drop area",
                         "Drop a file for ORION to look at.")

    def dragEnterEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        # A local file only, as the Command Deck already checks: a link
        # dragged from a browser was accepted here and then dropped silently.
        data = event.mimeData()
        if data.hasUrls() and any(url.isLocalFile() for url in data.urls()):
            event.acceptProposedAction()
            self.setProperty("active", "true")
            self._restyle()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        self.setProperty("active", "false")
        self._restyle()

    def dropEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        paths = [url.toLocalFile() for url in event.mimeData().urls()
                 if url.toLocalFile()]
        self.setProperty("active", "false")
        self._restyle()
        if paths:
            self.prompt.setText(
                paths[0].rsplit("\\", 1)[-1].rsplit("/", 1)[-1])
            self.dropped.emit(paths)
            event.acceptProposedAction()

    def _restyle(self) -> None:
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)


#: The microphone button, said plainly.
MIC_ON_TEXT = "●  Mic on - I'm listening"
MIC_OFF_TEXT = "○  Mic off"


class CommandBar(QWidget):
    """Type at ORION, stop him, or mute the microphone.

    The three controls are together because they are the three things a person
    reaches for mid-conversation, and two of them were previously a keyboard
    shortcut with no visible affordance at all.
    """

    submitted = pyqtSignal(str)
    interrupted = pyqtSignal()
    mic_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudCommandBar")
        _styled_background(self)
        self._mic_on = True

        self.input = QLineEdit()
        self.input.setObjectName("hudInput")
        self.input.setFont(ui(10))
        self.input.setPlaceholderText("Message ORION…")
        self.input.returnPressed.connect(self._submit)
        describe_control(self.input, "Talk to ORION",
                         "Type a question or something for ORION to do, then press Enter.")

        self.send_btn = QPushButton("▶")
        self.send_btn.setObjectName("hudSend")
        self.send_btn.setFixedWidth(30)
        self.send_btn.clicked.connect(self._submit)
        describe_control(self.send_btn, "Send", "Send what you have typed.")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACE.SM)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)

        self.interrupt_btn = QPushButton("■  Stop Talking   ·   Esc")
        self.interrupt_btn.setObjectName("hudInterrupt")
        self.interrupt_btn.setFont(ui(9, bold=True))
        self.interrupt_btn.clicked.connect(self.interrupted.emit)
        describe_control(self.interrupt_btn, "Interrupt",
                         "Stop ORION mid-sentence and return to listening.")

        self.mic_btn = QPushButton(MIC_ON_TEXT)
        self.mic_btn.setObjectName("hudMic")
        self.mic_btn.setFont(ui(9, bold=True))
        self.mic_btn.clicked.connect(self._toggle_mic)
        describe_control(self.mic_btn, "Microphone",
                         "Turn ORION's microphone on or off.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PANEL_PADDING)
        layout.setSpacing(SPACE.SM)
        # No caption: the box says what it is (owner, 30 Oct 2026: "get rid of
        # this 'Talk to ORION' heading"). Its accessible name stays.
        layout.addLayout(row)
        layout.addWidget(self.interrupt_btn)
        layout.addWidget(self.mic_btn)

    def _submit(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.submitted.emit(text)

    def _toggle_mic(self) -> None:
        self.set_mic(not self._mic_on)
        self.mic_toggled.emit(self._mic_on)

    def set_mic(self, active: bool) -> None:
        """Reflect the microphone's real state without re-emitting.

        Separate from the toggle so the window can follow ``bus.mic_enabled``
        — the mic is turned off by voice and by the gate as well as by this
        button, and a button showing the opposite of the truth is worse than
        no button.
        """
        self._mic_on = bool(active)
        self.mic_btn.setText(MIC_ON_TEXT if self._mic_on else MIC_OFF_TEXT)
        self.mic_btn.setProperty("muted", "false" if self._mic_on else "true")
        style = self.mic_btn.style()
        if style is not None:
            style.unpolish(self.mic_btn)
            style.polish(self.mic_btn)

    @property
    def mic_on(self) -> bool:
        return self._mic_on


# ── status ────────────────────────────────────────────────────────────────────

class StatusChips(QWidget):
    """The stack under the meters, shown only for what needs attention.

    A chip in its "good" state (mind ready, security clear, no routine, not
    recording) is hidden, and with none left the stack is hidden too: four
    tiles saying all is well were clutter (owner, 26 Oct 2026). Recording, a
    security finding, a running routine or a mind that is not ready appear
    the moment they happen."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudChips")
        # As wide as the rail above it, always: a chip given a longer label
        # widened the whole left column and so RESIZED the face beside it - a
        # resize clears its WebGL canvas, which is a black flash (measured:
        # 942 -> 894 -> 876 px, and it never shrank back).
        self.setFixedWidth(RAIL_WIDTH)
        self.chips: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)        # the rail's edges, not a second gutter
        layout.setSpacing(SPACE.SM)
        for key, text in (("core", "Mind\nready"),
                          ("security", "Security\nall clear"),
                          ("protocol", "Routine\nnone running"),
                          ("recording", "Recording\noff")):
            chip = QLabel(text)
            chip.setObjectName("hudChip")
            chip.setFont(ui(8))
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setWordWrap(True)
            chip.setProperty("tone", "good")
            chip.setVisible(False)
            self.chips[key] = chip
            layout.addWidget(chip)
        self.setVisible(False)

    def set_chip(self, key: str, text: str, tone: str = "good") -> None:
        chip = self.chips.get(key)
        if chip is None:
            return
        chip.setText(text)
        chip.setProperty("tone", tone)
        style = chip.style()
        if style is not None:
            style.unpolish(chip)
            style.polish(chip)
        chip.setVisible(tone != "good")
        self.setVisible(any(c.property("tone") != "good" for c in self.chips.values()))


# ── the controls list ─────────────────────────────────────────────────────────

class ControlsPanel(QWidget):
    """Every switch, named, in one column.

    These used to be a ``•••`` menu, two keyboard shortcuts and several
    settings reachable only by asking ORION out loud. A control you cannot see
    is a control you do not have, and the toggles say their current state in
    the label — "AUTO-START: OFF" rather than a checkbox whose meaning depends
    on which way round you read it.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudControls")
        self._buttons: dict[str, QPushButton] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(SPACE.SM, SPACE.SM, SPACE.SM, SPACE.SM)
        self._layout.setSpacing(SPACE.XS)
        self._layout.addWidget(_section("Controls"))
        self._layout.addStretch(1)

    def add(self, key: str, label: str, callback: Callable[[], Any],
            *, icon: str = "▪", hint: str = "",
            primary: bool = False) -> QPushButton:
        """Add one control. Returns the button so a caller can restyle it."""
        from .line_icons import NAMES, apply
        button = QPushButton(label)
        apply(button, icon if icon in NAMES else "sliders")
        button.setObjectName("hudControlPrimary" if primary else "hudControl")
        button.setFont(mono(8, bold=primary))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda: callback())
        describe_control(button, label, hint or f"{label}.")
        self._buttons[key] = button
        self._layout.insertWidget(self._layout.count() - 1, button)
        return button

    def set_state(self, key: str, label: str, *, icon: str = "") -> None:
        """Rewrite a toggle's label so it says what it currently is."""
        button = self._buttons.get(key)
        if button is None:
            return
        button.setText(label)
        if icon:
            from .line_icons import apply
            apply(button, icon)
        describe_control(button, label, f"{label}.")

    def button(self, key: str) -> QPushButton | None:
        return self._buttons.get(key)

    @property
    def keys(self) -> list[str]:
        return list(self._buttons)


# ── composition ───────────────────────────────────────────────────────────────

class HudShell(QWidget):
    """Rail, centre, column — and a strip along the bottom.

    The centre widget is supplied by the caller and is never wrapped,
    restyled or reparented beyond being placed. ORION's face and orb are that
    widget, and they are not this module's business.
    """

    def __init__(self, centre: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudShell")
        # Deliberately NOT _styled_background(self). This widget WRAPS the
        # centre — the face, which is a native/GL surface — so making the shell
        # paint an opaque rectangle across its whole area puts a Qt-painted
        # fill over a child the compositor draws itself. The result is an
        # intermittent black frame: the same black-flicker class of bug as
        # Mark XXIII, arriving by a different route.
        #
        # Nothing is lost by leaving it. The shell's background IS the window's
        # background, the same C.BG, so the parent showing through is exactly
        # what the rule would have painted.

        self.rail = SysMonitorRail()
        self.chips = StatusChips()
        self.log = ActivityLog()
        self.drop = DropZone()
        self.command = CommandBar()
        self.controls = ControlsPanel()

        # The chips sit directly under the meters they qualify rather than at
        # the foot of the column: the foot is where ORION's toggle pill docks
        # (HolographicToggle.dock), and a warning there would be covered.
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(SPACE.SM)
        left.addWidget(self.rail)
        left.addWidget(self.chips)
        left.addStretch(1)

        self.centre = centre
        centre.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        self.centre_stack = QVBoxLayout()
        self.centre_stack.setContentsMargins(0, 0, 0, 0)
        self.centre_stack.setSpacing(SPACE.SM)
        self.centre_stack.addWidget(centre, 1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(SPACE.SM)
        right.addWidget(self.log, 1)
        right.addWidget(self.drop)
        right.addWidget(self.command)

        self.right_column = QWidget()
        self.right_column.setObjectName("hudColumn")
        self.right_column.setFixedWidth(COLUMN_WIDTH)
        self.right_column.setLayout(right)

        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(SPACE.MD)
        body.addLayout(left)
        body.addLayout(self.centre_stack, 1)
        body.addWidget(self.right_column)

    def attach_recorder(self, recorder: Any, stop: Callable[[], Any]) -> None:
        """Read the recorder's in-memory state; no capture or filesystem work here."""
        self._recorder = recorder
        self._recording_stop = self.controls.add(
            "stop_recording", "Stop Recording", stop,
            icon="stop", hint="Stop saving conversation audio and finalise the local recording.")
        self._recording_timer = QTimer(self)
        self._recording_timer.setInterval(500)
        self._recording_timer.timeout.connect(self._refresh_recording)
        self._recording_timer.start()
        self._refresh_recording()

    def _refresh_recording(self) -> None:
        state = self._recorder.snapshot()
        active = bool(state.get("active"))
        if active:
            seconds = int(state.get("audio_seconds", 0))
            detail = "waiting for mic" if state.get("waiting_for_audio") else f"{seconds // 60}:{seconds % 60:02d} saved"
            tone = "warn" if state.get("dropped_blocks") or state.get("errors") else "good"
            self.chips.set_chip("recording", f"Recording\n{detail}", tone)
        else:
            detail = str(state.get("state", "off"))
            self.chips.set_chip("recording", f"Recording\n{detail}", "good" if detail in {"off", "complete"} else "warn")
        self._recording_stop.setEnabled(active)

    def add_below_centre(self, widget: QWidget, stretch: int = 0) -> None:
        """Put something under the face — the content panel goes here."""
        self.centre_stack.addWidget(widget, stretch)


def _hover_shade(primary: str) -> str:
    """The lighter member of *primary*'s family, for hover states.

    Hard-coding ``C.PRI_HI`` here was a quiet bug: the HUD takes its accent as
    an argument so it can be themed, but its hover states stayed crimson
    whatever was passed, so a themed HUD flashed red under the pointer.
    """
    if not primary or primary == C.PRI:
        return C.PRI_HI
    try:
        from .theming import derive_family

        return derive_family(primary).get(C.PRI_HI, C.PRI_HI)
    except Exception:
        return primary


def hud_stylesheet(primary: str = "", panel: str = "", background: str = "",
                   border: str = "") -> str:
    """The HUD's own styling, in ORION's colours.

    Takes its colours as arguments rather than reading ``C`` directly so the
    live theming pass can recolour it the same way it recolours everything
    else: one string, substituted, applied.

    Written in the same visual language as ``APP_STYLESHEET``: translucent
    surfaces rather than opaque fills, one lit hairline along the top rather
    than a box drawn all the way round, radii from the shared scale, and the
    accent colour reserved for state, focus and danger. The HUD is the surface
    the user looks at most, so an older idiom surviving here would make the
    rest of the redesign read as the inconsistency.
    """
    from .style import rgba

    pri = primary or C.PRI
    # Measured by rendering, not chosen: a 0.72 veil of C.PANEL over the void
    # lands eleven levels above it, which is a smudge rather than a surface.
    # On a near-black field the panel has to be genuinely lighter, because
    # there is no shadow to separate it — a shadow is a darkening, and there
    # is nothing left to darken.
    surface = panel or C.PANEL_HI
    void = background or C.BG
    hi = _hover_shade(primary)

    # `border` is honoured for callers that pass one, but the default edge is
    # a hairline of light rather than a grey outline: a box round every
    # element is what made the old HUD read as a wireframe diagram.
    edge = f"1px solid {border}" if border else f"1px solid {rgba(C.WHITE, 0.055)}"
    lip = f"1px solid {rgba(C.WHITE, 0.10)}"
    glass = rgba(surface, GLASS.RAISED)
    raised = rgba(C.PANEL_HI, GLASS.RAISED)
    well = rgba(C.INK, 0.66)
    radius = ELEV.RADIUS[0]

    return f"""
    /* Left unpainted on purpose — see HudShell.__init__. The shell wraps
       the native face, so it must not fill over it. */
    #hudShell {{ background: transparent; }}
    #hudRail, #hudColumn, #hudChips, #hudControls {{ background: transparent; }}

    /* Section labels are furniture — you read past them to the reading
       underneath, so they get weight and spacing rather than colour. */
    QLabel#hudSection {{
        color: {C.MUTED}; padding: 2px 2px; font-weight: 600;
    }}
    /* "Details" swaps the human summary for the full technical log. */
    QPushButton#hudDetails {{
        background: transparent; color: {C.FAINT}; border: 0;
        border-radius: {radius}px; padding: 2px 8px;
    }}
    QPushButton#hudDetails:hover {{ color: {C.SILVER}; }}
    QPushButton#hudDetails:checked {{
        color: {C.WHITE}; background: {rgba(C.WHITE, 0.08)};
    }}

    #hudMeter, #hudLogPanel, #hudCommandBar, #hudDrop {{
        background: {glass}; border: {edge}; border-top: {lip};
        border-radius: {radius}px;
    }}
    QLabel#hudMeterName {{ color: {C.MUTED}; }}
    QLabel#hudMeterValue {{ color: {C.WHITE}; font-weight: 600; }}
    QLabel#hudFacts {{ color: {C.MUTED}; }}
    #hudTrack {{ background: {rgba(C.WHITE, 0.07)}; border-radius: 2px; }}
    #hudFill {{ background: {C.SILVER}; border-radius: 2px; }}

    QPlainTextEdit#hudLog {{
        background: transparent; color: {C.SILVER}; border: 0;
        selection-background-color: {pri};
    }}
    QScrollBar:vertical {{ background: transparent; width: 8px; }}
    QScrollBar::handle:vertical {{
        background: {rgba(C.WHITE, 0.12)}; border-radius: 4px; min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {rgba(C.WHITE, 0.24)}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

    #hudDrop {{ border: 1px dashed {rgba(C.WHITE, 0.12)}; }}
    #hudDrop[active="true"] {{
        border: 1px dashed {pri}; background: {rgba(pri, 0.10)};
    }}
    QLabel#hudDropPrompt {{ color: {C.MUTED}; }}

    QLineEdit#hudInput {{
        background: {well}; color: {C.WHITE};
        border: {edge}; border-top: {lip};
        border-radius: {radius}px; padding: 6px 9px;
    }}
    QLineEdit#hudInput:focus {{ border: 1px solid {pri}; }}

    QPushButton#hudSend {{
        background: {pri}; color: {void}; border: 0;
        border-radius: {radius}px; padding: 6px 0; font-weight: 600;
    }}
    QPushButton#hudSend:hover {{ background: {hi}; }}

    /* Interrupt stops ORION mid-sentence, so it carries the accent — but in
       the TEXT only. It is on screen permanently, and a permanent red bar
       next to a red send button made the quietest corner of the interface
       the loudest. It fills with the accent when you are actually on it. */
    QPushButton#hudInterrupt {{
        background: {glass}; color: {pri};
        border: {edge}; border-top: {lip};
        border-radius: {radius}px; padding: 7px 0;
    }}
    QPushButton#hudInterrupt:hover {{ background: {pri}; color: {void}; }}

    QPushButton#hudMic {{
        background: {raised}; color: {C.SILVER};
        border: {edge}; border-top: {lip};
        border-radius: {radius}px; padding: 7px 0;
    }}
    QPushButton#hudMic[muted="true"] {{
        color: {C.BAD}; border-color: {rgba(C.BAD, 0.5)};
    }}
    QPushButton#hudMic:hover {{ background: {rgba(C.PANEL_HI, GLASS.FLOAT)}; color: {C.WHITE}; }}

    QLabel#hudChip {{
        background: {glass}; border: {edge}; border-top: {lip};
        border-radius: {radius}px; color: {C.MUTED}; padding: 6px 2px;
    }}
    QLabel#hudChip[tone="good"] {{ color: {C.GOOD}; }}
    QLabel#hudChip[tone="warn"] {{
        color: {C.WARN}; border-color: {rgba(C.WHITE, 0.18)};
    }}
    QLabel#hudChip[tone="bad"]  {{
        color: {C.BAD}; border-color: {rgba(C.BAD, 0.5)};
    }}

    QPushButton#hudControl, QPushButton#hudControlPrimary {{
        background: {glass}; color: {C.SILVER};
        border: {edge}; border-top: {lip}; border-radius: {radius}px;
        padding: 7px 10px; text-align: left;
    }}
    QPushButton#hudControl:hover, QPushButton#hudControlPrimary:hover {{
        background: {raised}; color: {C.WHITE};
        border-color: {rgba(C.WHITE, 0.18)};
    }}
    /* A marker down one edge rather than an outline all the way round: it
       says "this is the one" without becoming another red box. */
    QPushButton#hudControlPrimary {{
        color: {C.WHITE}; border-left: 2px solid {pri};
    }}
    """


__all__ = [
    "COLUMN_WIDTH", "LOG_LINES", "PANEL_PADDING", "RAIL_WIDTH",
    "ActivityLog", "CommandBar", "ControlsPanel", "DropZone", "HudShell",
    "Meter", "StatusChips", "SysMonitorRail",
    "hud_stylesheet", "mono",
]
