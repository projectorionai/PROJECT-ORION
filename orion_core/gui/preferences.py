"""
Preferences — the settings ORION already had, made reachable.

Several capabilities were built without any way to reach them, which in
practice means they were not built. The measured audio-device layer is the
clearest case: ``audio_devices`` can list the short, deduplicated set of real
endpoints and MEASURE whether each one actually carries sound, and none of that
was exposed anywhere, so "ORION can't hear me" still had no answer in the
interface.

What is here
------------
  * **Audio** — microphone and speakers by NAME, from the deduplicated list,
    with a test that plays and records rather than merely opening a stream.
  * **Identity** — what to call ORION, and what he calls you.
  * **Appearance** — the accent colour the whole shell and his face follow.

Everything slow happens off the Qt thread
-----------------------------------------
Under qasync the Qt thread IS the asyncio event loop, so blocking it stops the
audio callback and the Live socket, not merely the repaint. Enumerating devices
takes hundreds of milliseconds on a machine with many endpoints, and the
transport probe deliberately takes about a second because it is timing real
audio. Both run on worker threads and report back through signals.

Styled from ORION's own tokens — crimson on near-black with a silver accent —
rather than carrying a palette of its own, so it stays his rather than looking
like a settings dialog that wandered in.
"""

from __future__ import annotations

import threading
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .style import C
from .widgets import describe_control

#: What the picker calls "whatever the OS is using". Stored as an empty string
#: so an untouched install and a deliberate default are the same thing.
SYSTEM_DEFAULT = "System default"


def _stylesheet() -> str:
    """The application's own styling, plus the two labels only this dialog has.

    This used to be a complete second stylesheet: the same palette tokens in
    the old idiom — opaque fills, a box drawn round everything, and a hover
    state that turned every button crimson. It meant Preferences visibly
    aged behind the rest of the interface every time the shared sheet moved,
    and nothing pointed that out because it was valid QSS either way.
    """
    from .style import APP_STYLESHEET

    return APP_STYLESHEET + (
        f"QLabel#hint {{ color: {C.MUTED}; font-size: 11px; }}"
        f"QLabel#verdict {{ color: {C.MUTED}; font-size: 11px; }}"
    )

class _DeviceRow(QWidget):
    """One direction of audio: a named picker and a test that proves it works."""

    #: (ok, detail) from the worker thread.
    tested = pyqtSignal(bool, str)
    #: (entries, current) once enumeration finishes off-thread.
    listed = pyqtSignal(object, str)

    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = "input" if kind.startswith("in") else "output"
        noun = "microphone" if self.kind == "input" else "speakers"

        self.combo = QComboBox()
        self.combo.setMinimumWidth(320)
        describe_control(self.combo, f"ORION's {noun}",
                         f"Which {noun} ORION uses. Stored by name, so it "
                         f"survives devices being plugged in and out.")
        self.combo.addItem(SYSTEM_DEFAULT, "")

        self.test_button = QPushButton("Test")
        describe_control(self.test_button, f"Test the {noun}",
                         "Plays or records for a moment and reports whether "
                         "sound actually moved — not merely whether the device "
                         "opened.")
        self.test_button.clicked.connect(self._run_test)

        self.verdict = QLabel("")
        self.verdict.setObjectName("verdict")
        self.verdict.setWordWrap(True)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.combo, 1)
        row.addWidget(self.test_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(row)
        layout.addWidget(self.verdict)

        self.tested.connect(self._show_verdict)
        self.listed.connect(self._populate)

    # ── enumeration ──────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Fill the picker off-thread.

        query_devices() has been measured at hundreds of milliseconds on a
        machine with many endpoints, and this runs on the event loop.
        """
        self.combo.setEnabled(False)
        self.verdict.setText("Looking for devices…")

        def work() -> None:
            entries: list[tuple[int, str]] = []
            current = ""
            try:
                from .. import audio_devices

                entries = audio_devices.usable_devices(self.kind)
                current = audio_devices.device_name(self.kind) or ""
            except Exception:
                entries = []
            try:
                self.listed.emit(entries, current)
            except RuntimeError:
                # The dialog was closed while the scan ran: the row is gone
                # and there is nothing left to fill.
                pass

        threading.Thread(target=work, name=f"orion-devices-{self.kind}",
                         daemon=True).start()

    def _populate(self, entries: Any, current: str) -> None:
        self.combo.clear()
        self.combo.addItem(SYSTEM_DEFAULT, "")
        try:
            from ..audio_devices import display_name
        except Exception:
            def display_name(value: str) -> str:
                return value
        for _index, name in (entries or []):
            # Shown cleaned, stored RAW. Windows hands back unresolved driver
            # resource strings for Bluetooth endpoints — a headset arrives as
            # "Headset (@System32\\drivers\\bthhfenum.sys,#2;...;(HyperX Cloud
            # III S Wireless))" — which nobody can pick from. The raw string is
            # still what resolves to a device, so it stays as the item data.
            self.combo.addItem(display_name(name), name)
        if current:
            found = self.combo.findData(current)
            self.combo.setCurrentIndex(found if found >= 0 else 0)
        self.combo.setEnabled(True)
        count = len(entries or [])
        self.verdict.setText(
            f"{count} device{'' if count == 1 else 's'} offered."
            if count else "No devices found — the system default will be used.")

    # ── the measured test ────────────────────────────────────────────────────

    def selected(self) -> str:
        return str(self.combo.currentData() or "")

    def _run_test(self) -> None:
        """Prove it works, rather than assuming.

        Opening a stream is not proof of anything: on Windows, PortAudio's
        DirectSound output opens happily, accepts every write instantly and
        plays nothing at all. Only timing real audio tells them apart.
        """
        self.test_button.setEnabled(False)
        self.verdict.setText("Testing…")
        wanted = self.selected()
        kind = self.kind

        def work() -> None:
            ok, detail = False, "no audio stack"
            try:
                from .. import audio_devices

                # An empty selection means "System default", which sounddevice
                # spells as device=None — not as index 0, which is a real and
                # usually wrong device.
                index = None
                if wanted:
                    for candidate, name in audio_devices.usable_devices(kind):
                        if name == wanted:
                            index = candidate
                            break
                ok, detail = audio_devices.transport_works(kind, index,
                                                           use_cache=False)
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            try:
                self.tested.emit(bool(ok), str(detail))
            except RuntimeError:
                pass        # closed mid-probe, as above

        threading.Thread(target=work, name=f"orion-probe-{self.kind}",
                         daemon=True).start()

    def _show_verdict(self, ok: bool, detail: str) -> None:
        self.test_button.setEnabled(True)
        colour = C.GOOD if ok else C.BAD
        word = "Working" if ok else "No sound"
        self.verdict.setText(f"<span style='color:{colour}'>{word}</span> — {detail}")


class PreferencesDialog(QDialog):
    """ORION's settings. Returns the chosen values; saving is the caller's job,
    so this stays testable without a running assistant."""

    def __init__(self, parent: QWidget | None = None, *,
                 assistant_name: str = "Orion", user_name: str = "",
                 accent: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("O.R.I.O.N. — Preferences")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setStyleSheet(_stylesheet())

        self._accent = QColor(accent) if accent else QColor(C.PRI)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        heading = QLabel("Preferences")
        heading.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        layout.addWidget(heading)

        # ── audio ────────────────────────────────────────────────────────────
        audio_box = QGroupBox("Audio")
        audio_form = QFormLayout(audio_box)
        audio_form.setSpacing(8)
        self.microphone = _DeviceRow("input")
        self.speakers = _DeviceRow("output")
        audio_form.addRow(QLabel("Microphone"), self.microphone)
        audio_form.addRow(QLabel("Speakers"), self.speakers)
        note = QLabel(
            "Devices are stored by name, not by index — an index moves the "
            "moment anything is plugged in. Test measures whether sound "
            "actually moves, because a device can open perfectly and still "
            "play nothing."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        audio_form.addRow(note)
        layout.addWidget(audio_box)

        # ── identity ─────────────────────────────────────────────────────────
        identity_box = QGroupBox("Identity")
        identity_form = QFormLayout(identity_box)
        identity_form.setSpacing(8)
        self.assistant_edit = QLineEdit(assistant_name)
        describe_control(self.assistant_edit, "Assistant name",
                         "What ORION calls himself, spoken and written.")
        self.user_edit = QLineEdit(user_name)
        self.user_edit.setPlaceholderText("what he should call you")
        describe_control(self.user_edit, "Your name",
                         "What ORION calls you.")
        identity_form.addRow(QLabel("He is called"), self.assistant_edit)
        identity_form.addRow(QLabel("You are called"), self.user_edit)
        layout.addWidget(identity_box)

        # ── appearance ───────────────────────────────────────────────────────
        appearance_box = QGroupBox("Appearance")
        appearance_row = QHBoxLayout(appearance_box)
        self.swatch = QLabel()
        self.swatch.setFixedSize(46, 24)
        self._paint_swatch()
        self.colour_button = QPushButton("Accent colour…")
        describe_control(self.colour_button, "Choose the accent colour",
                         "Recolours the shell and ORION's face together.")
        self.colour_button.clicked.connect(self._choose_colour)
        reset = QPushButton("Reset")
        describe_control(reset, "Reset the accent colour",
                         "Return to ORION's own crimson.")
        reset.clicked.connect(self._reset_colour)
        appearance_row.addWidget(self.swatch)
        appearance_row.addWidget(self.colour_button)
        appearance_row.addWidget(reset)
        appearance_row.addStretch(1)
        layout.addWidget(appearance_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.microphone.refresh()
        self.speakers.refresh()

    # ── appearance helpers ───────────────────────────────────────────────────

    def _paint_swatch(self) -> None:
        self.swatch.setStyleSheet(
            f"background: {self._accent.name()}; border: 1px solid {C.BORDER}; "
            f"border-radius: 5px;")
        self.swatch.setToolTip(self._accent.name())
        self.swatch.setAccessibleName(f"Accent colour {self._accent.name()}")

    def _choose_colour(self) -> None:
        chosen = QColorDialog.getColor(self._accent, self, "ORION's accent colour")
        if chosen.isValid():
            self._accent = chosen
            self._paint_swatch()

    def _reset_colour(self) -> None:
        self._accent = QColor(C.PRI)
        self._paint_swatch()

    # ── results ──────────────────────────────────────────────────────────────

    def values(self) -> dict[str, str]:
        """Everything the dialog collected. The caller decides what to persist."""
        return {
            "input_device": self.microphone.selected(),
            "output_device": self.speakers.selected(),
            "assistant_name": self.assistant_edit.text().strip(),
            "user_name": self.user_edit.text().strip(),
            "accent": self._accent.name(),
        }


__all__ = ["SYSTEM_DEFAULT", "PreferencesDialog"]
