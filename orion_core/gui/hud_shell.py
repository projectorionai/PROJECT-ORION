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

from ..constants import C, ELEV, GLASS
from .widgets import describe_control

#: The rail's width. Wide enough for "MEM  100%" without eliding, narrow
#: enough that the face keeps the room it needs.
RAIL_WIDTH = 132

#: The right column's width. This one holds readable log lines, so it is
#: measured from text rather than chosen: ~46 monospace characters.
COLUMN_WIDTH = 310

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


def _section(title: str) -> QLabel:
    """The little ▸ SECTION NAME caption above each panel."""
    label = QLabel(f"▸ {title.upper()}")
    label.setObjectName("hudSection")
    label.setFont(mono(8, bold=True))
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

        self.name_label = QLabel(name.upper())
        self.name_label.setObjectName("hudMeterName")
        self.name_label.setFont(mono(8))
        self.value_label = QLabel("N/A")
        self.value_label.setObjectName("hudMeterValue")
        self.value_label.setFont(mono(8, bold=True))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
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
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(5)
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
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(_section("sys monitor"))

        for key, name in (("cpu", "CPU"), ("ram", "MEM"), ("net", "NET"),
                          ("gpu", "GPU"), ("tmp", "TMP")):
            meter = Meter(name)
            self.meters[key] = meter
            layout.addWidget(meter)

        self.facts = QLabel("UP   --:--\nPROC  ---\nOS    ---")
        self.facts.setObjectName("hudFacts")
        self.facts.setFont(mono(8))
        facts_frame = QFrame()
        facts_frame.setObjectName("hudMeter")
        facts_layout = QVBoxLayout(facts_frame)
        facts_layout.setContentsMargins(8, 6, 8, 6)
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

    def set_facts(self, uptime: str = "", processes: Any = "",
                  system: str = "") -> None:
        self.facts.setText(
            f"UP   {uptime or '--:--'}\n"
            f"PROC  {processes if processes != '' else '---'}\n"
            f"OS    {system or '---'}")


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

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)
        self._layout.addWidget(_section("activity log"))
        self._layout.addStretch(1)

        QTimer.singleShot(0, self._build_view)

    def _build_view(self) -> None:
        """Create the real text widget and flush anything that arrived first."""
        if self.view is not None:
            return
        view = QPlainTextEdit()
        view.setObjectName("hudLog")
        view.setReadOnly(True)
        view.setFont(mono(8))
        view.setMaximumBlockCount(LOG_LINES)
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        view.setFrameShape(QFrame.Shape.NoFrame)
        describe_control(view, "Activity log",
                         "What ORION has been doing, most recent last.")
        # Replace the stretch that was holding the space open.
        item = self._layout.takeAt(self._layout.count() - 1)
        del item
        self._layout.addWidget(view, 1)
        self.view = view
        if self._pending:
            view.setPlainText("\n".join(self._pending[-LOG_LINES:]))
            self._pending.clear()
            self._scroll_to_end()

    def append(self, line: Any) -> None:
        text = str(line or "").rstrip()
        if not text:
            return
        if self.view is None:
            # Buffered, not dropped. Boot is the noisiest moment in ORION's
            # life and it happens before this widget exists.
            self._pending.append(text)
            del self._pending[:-LOG_LINES]
            return
        self.view.appendPlainText(text)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        if self.view is None:
            return
        bar = self.view.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def clear(self) -> None:
        self._pending.clear()
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

        self.prompt = QLabel("Drop a file here, or click to browse")
        self.prompt.setObjectName("hudDropPrompt")
        self.prompt.setFont(mono(8))
        self.prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt.setWordWrap(True)
        self.kinds = QLabel("images · video · audio · pdf · docs · code · data")
        self.kinds.setObjectName("hudDropKinds")
        self.kinds.setFont(mono(7))
        self.kinds.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        layout.addWidget(self.prompt)
        layout.addWidget(self.kinds)
        describe_control(self, "File drop area",
                         "Drop a file for ORION to look at.")

    def dragEnterEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("active", "true")
            self._restyle()

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
        self.input.setFont(mono(9))
        self.input.setPlaceholderText("Type a command or question…")
        self.input.returnPressed.connect(self._submit)
        describe_control(self.input, "Command input",
                         "Type a command or question and press Enter.")

        self.send_btn = QPushButton("▶")
        self.send_btn.setObjectName("hudSend")
        self.send_btn.setFixedWidth(30)
        self.send_btn.clicked.connect(self._submit)
        describe_control(self.send_btn, "Send", "Send what you have typed.")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)

        self.interrupt_btn = QPushButton("■  INTERRUPT   [ESC]")
        self.interrupt_btn.setObjectName("hudInterrupt")
        self.interrupt_btn.setFont(mono(8, bold=True))
        self.interrupt_btn.clicked.connect(self.interrupted.emit)
        describe_control(self.interrupt_btn, "Interrupt",
                         "Stop ORION mid-sentence and return to listening.")

        self.mic_btn = QPushButton("●  MICROPHONE ACTIVE")
        self.mic_btn.setObjectName("hudMic")
        self.mic_btn.setFont(mono(8, bold=True))
        self.mic_btn.clicked.connect(self._toggle_mic)
        describe_control(self.mic_btn, "Microphone",
                         "Turn ORION's microphone on or off.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        layout.addWidget(_section("command input"))
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
        self.mic_btn.setText("●  MICROPHONE ACTIVE" if self._mic_on
                             else "○  MICROPHONE MUTED")
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
    """The bottom-left stack: three things that are either true or not."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudChips")
        self.chips: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        for key, text in (("core", "AI CORE\nACTIVE"),
                          ("security", "SEC\nCLEARED"),
                          ("protocol", "PROTOCOL\n—")):
            chip = QLabel(text)
            chip.setObjectName("hudChip")
            chip.setFont(mono(7, bold=True))
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setProperty("tone", "good")
            self.chips[key] = chip
            layout.addWidget(chip)

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
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(3)
        self._layout.addWidget(_section("controls"))
        self._layout.addStretch(1)

    def add(self, key: str, label: str, callback: Callable[[], Any],
            *, icon: str = "▪", hint: str = "",
            primary: bool = False) -> QPushButton:
        """Add one control. Returns the button so a caller can restyle it."""
        button = QPushButton(f"{icon}  {label}")
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
        current = button.text()
        prefix = current.split("  ", 1)[0] if "  " in current else "▪"
        button.setText(f"{icon or prefix}  {label}")
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

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)
        left.addWidget(self.rail, 1)
        left.addWidget(self.chips)

        self.centre = centre
        centre.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        self.centre_stack = QVBoxLayout()
        self.centre_stack.setContentsMargins(0, 0, 0, 0)
        self.centre_stack.setSpacing(6)
        self.centre_stack.addWidget(centre, 1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)
        right.addWidget(self.log, 1)
        right.addWidget(self.drop)
        right.addWidget(self.command)

        self.right_column = QWidget()
        self.right_column.setObjectName("hudColumn")
        self.right_column.setFixedWidth(COLUMN_WIDTH)
        self.right_column.setLayout(right)

        body = QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        body.addLayout(left)
        body.addLayout(self.centre_stack, 1)
        body.addWidget(self.right_column)

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
        color: {C.FAINT}; letter-spacing: 1.4px; padding: 2px 2px;
        font-weight: 600;
    }}

    #hudMeter, #hudLogPanel, #hudCommandBar, #hudDrop {{
        background: {glass}; border: {edge}; border-top: {lip};
        border-radius: {radius}px;
    }}
    QLabel#hudMeterName {{ color: {C.FAINT}; letter-spacing: 0.6px; }}
    QLabel#hudMeterValue {{ color: {C.WHITE}; font-weight: 600; }}
    QLabel#hudFacts {{ color: {C.MUTED}; }}
    #hudTrack {{ background: {rgba(C.WHITE, 0.07)}; border-radius: 2px; }}
    #hudFill {{ background: {C.SILVER}; border-radius: 2px; }}

    QPlainTextEdit#hudLog {{
        background: transparent; color: {C.MUTED}; border: 0;
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
    QLabel#hudDropKinds {{ color: {C.FAINT}; }}

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
    "COLUMN_WIDTH", "LOG_LINES", "RAIL_WIDTH",
    "ActivityLog", "CommandBar", "ControlsPanel", "DropZone", "HudShell",
    "Meter", "StatusChips", "SysMonitorRail",
    "hud_stylesheet", "mono",
]
