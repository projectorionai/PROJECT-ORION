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

import os
import threading
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QColorDialog,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .style import C
from .widgets import describe_control

#: What the picker calls "whatever the OS is using". Stored as an empty string
#: so an untouched install and a deliberate default are the same thing.
SYSTEM_DEFAULT = "System Default"


def _matching_device(entries: Any, spec: str) -> tuple[int, str] | None:
    """Match saved specs, preferring the exact endpoint over a name fragment."""
    if not spec:
        return None
    offered = list(entries or [])
    if spec.isdigit():
        return next((entry for entry in offered if str(entry[0]) == spec), None)
    low = spec.casefold()
    return (next((entry for entry in offered if entry[1].casefold() == low), None)
            or next((entry for entry in offered if low in entry[1].casefold()), None))


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
        self._listed = False
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
        self.test_button.setEnabled(False)
        self._listed = False
        self.verdict.setText("Looking for devices…")

        def work() -> None:
            entries: list[tuple[int, str]] = []
            current = ""
            try:
                from .. import audio_devices

                current = audio_devices.configured_spec(self.kind)
                entries = audio_devices.usable_devices(self.kind)
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
            if found < 0:
                # Saved name fragments and legacy indices must round-trip too.
                # Keep the original spec in item data so an untouched dialog
                # cannot silently rewrite the audio preference.
                match = _matching_device(entries, current)
                if match is not None:
                    found = self.combo.findData(match[1])
                    self.combo.setItemData(found, current)
                else:
                    self.combo.addItem(f"{display_name(current)} (saved choice)", current)
                    found = self.combo.count() - 1
            self.combo.setCurrentIndex(found)
        self._listed = True
        self.combo.setEnabled(True)
        self.test_button.setEnabled(True)
        count = len(entries or [])
        self.verdict.setText(
            f"{count} device{'' if count == 1 else 's'} offered."
            if count else "No devices found — the system default will be used.")

    # ── the measured test ────────────────────────────────────────────────────

    def selected(self) -> str:
        if not self._listed:
            # OK may close the dialog before its off-thread device scan returns.
            # Read only the tiny preferences file; never enumerate hardware here.
            from ..audio_devices import configured_spec
            return configured_spec(self.kind)
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

        # Only an explicit system default may probe device=None.
        def work() -> None:
            ok, detail = False, "no audio stack"
            try:
                from .. import audio_devices

                index = None
                if wanted:
                    matched = _matching_device(audio_devices.usable_devices(kind), wanted)
                    if matched is None:
                        raise ValueError("Selected device is unavailable.")
                    index = matched[0]
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


class _VoiceLearning(QGroupBox):
    """Read prompts on screen while the existing conversation capture learns."""

    updated = pyqtSignal(str)

    def __init__(self, name: Any, parent: QWidget | None = None) -> None:
        super().__init__("Your Voice", parent)
        self._name = name
        self._session: Any = None
        self._closing = False
        self._working = False
        self.status = QLabel("About twenty short lines")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.consent = QCheckBox("Save My Voiceprint (Numbers Only, Never Audio)")
        self.learn = QPushButton("Learn My Voice")
        self.cancel = QPushButton("Cancel Learning")
        self.forget = QPushButton("Forget My Voice")
        describe_control(self.learn, "Learn my voice", "Start the guided voice learning session.")
        describe_control(self.cancel, "Cancel voice learning", "Stop this session without saving samples.")
        describe_control(self.forget, "Forget my voice", "Delete your saved voiceprint and turn voice gates off.")
        self.learn.setEnabled(False)
        self.cancel.setEnabled(False)
        self.consent.toggled.connect(self.learn.setEnabled)
        self.learn.clicked.connect(lambda: self._work("learn"))
        self.cancel.clicked.connect(lambda: self._work("cancel"))
        self.forget.clicked.connect(lambda: self._work("forget"))
        self.updated.connect(self.status.setText)
        row = QHBoxLayout()
        row.addWidget(self.learn)
        row.addWidget(self.cancel)
        row.addWidget(self.forget)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.consent)
        layout.addLayout(row)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _work(self, action: str) -> None:
        if self._working or (self._closing and action != "cancel"):
            return
        self._working = True
        self.learn.setEnabled(False)
        name = str(self._name() or "").strip() or "Owner"
        consent = self.consent.isChecked()

        def work() -> None:
            from .. import voice_enrolment
            from ..voice_speaker_id import SpeakerIdentificationService

            try:
                if action == "learn":
                    service = SpeakerIdentificationService()
                    result = service.start_learning(name, consent=consent, origin="screen",
                                                    on_event=self._emit)
                    if result.ok:
                        self._session = voice_enrolment.current()
                        if self._closing and self._session is not None:
                            self._session.cancel()
                    message = result.text
                elif action == "forget":
                    _ok, message = voice_enrolment.forget_my_voice()
                else:
                    message = (self._session.cancel() if self._session is not None
                               else "No voice learning is running.")
            except Exception as exc:
                message = f"Voice learning could not continue: {exc}"
            finally:
                self._working = False
            self._emit(message)

        threading.Thread(target=work, name="orion-voice-preferences", daemon=True).start()

    def _emit(self, message: str) -> None:
        try:
            self.updated.emit(message)
        except RuntimeError:
            pass                         # the dialog closed while work finished

    def _refresh(self) -> None:
        session = self._session
        listening = session is not None and session.listening
        self.cancel.setEnabled(listening)
        self.learn.setEnabled(self.consent.isChecked() and not listening and not self._working
                              and not self._closing)
        if session is not None:
            self.status.setText(session.describe())

    def cancel_owned(self) -> None:
        self._closing = True
        if self._session is not None and self._session.listening:
            self._work("cancel")


class _ConversationRecording(QGroupBox):
    """Per-session consent and controls; never turns recording on by opening settings."""
    updated = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Conversation Recordings", parent)
        self._working = False
        self._snapshot_key = None
        layout = QVBoxLayout(self)
        self.consent = QCheckBox("Everyone Being Recorded Has Agreed")
        layout.addWidget(self.consent)
        self.start = QPushButton("Start Recording")
        self.stop = QPushButton("Stop Recording")
        describe_control(self.start, "Start conversation recording",
                         "Save local audio only after everyone being recorded has agreed.")
        describe_control(self.stop, "Stop conversation recording",
                         "End capture and finalise the local audio and transcript.")
        row = QHBoxLayout()
        row.addWidget(self.start)
        row.addWidget(self.stop)
        layout.addLayout(row)
        self.status = QLabel("Recording is off.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.consent.toggled.connect(self._refresh)
        self.start.clicked.connect(lambda: self._work("start"))
        self.stop.clicked.connect(lambda: self._work("stop"))
        self.updated.connect(self._finished)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    def _refresh(self, *_args: Any) -> None:
        from ..conversation_recorder import recorder
        service = recorder()
        state = service.snapshot()
        key = (state["active"], state["state"], int(state.get("audio_seconds", 0)))
        if key != self._snapshot_key and not self._working:
            self.status.setText(service.status().text)
        self._snapshot_key = key
        self.start.setEnabled(self.consent.isChecked() and not state["active"] and not self._working)
        self.stop.setEnabled(state["active"] and not self._working)

    def _finished(self, text: str) -> None:
        self._working = False
        self.status.setText(text)
        self._refresh()

    def _work(self, action: str) -> None:
        if self._working or (action == "start" and not self.consent.isChecked()):
            return
        consent = self.consent.isChecked()
        self._working = True
        if action == "start":
            self.consent.setChecked(False)  # fresh agreement required for every session
        self._refresh()

        def work() -> None:
            try:
                from ..conversation_recorder import recorder
                service = recorder()
                result = service.start(consent=consent) if action == "start" else service.stop()
                text = result.text
            except Exception as exc:
                text = f"Recording could not be {action}ed: {exc}"
            try:
                self.updated.emit(text)
            except RuntimeError:
                pass  # settings may close while the bounded writer drain finishes
        threading.Thread(target=work, name="orion-recording-control", daemon=True).start()


class _MessagingConnections(QWidget):
    """Explicit account setup; task delivery never opens a setup window."""

    completed = pyqtSignal(str, object)

    def __init__(self, provider: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._provider = provider
        self._rows: dict[str, tuple[QLabel, QPushButton, QPushButton]] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for platform, title in (("whatsapp", "WhatsApp"), ("discord", "Discord")):
            row = QHBoxLayout()
            label = QLabel(title)
            label.setMinimumWidth(85)
            status = QLabel("Not checked")
            status.setWordWrap(True)
            connect = QPushButton("Connect…")
            check = QPushButton("Check Connection")
            describe_control(connect, f"Connect {title} for background messaging",
                             "Opens ORION's separate sign-in window only when you press this button.")
            describe_control(check, f"Check the {title} connection",
                             "Checks the connection without opening a window or sending a message.")
            connect.clicked.connect(lambda _checked=False, p=platform: self._operate(p, "connect"))
            check.clicked.connect(lambda _checked=False, p=platform: self._operate(p, "check"))
            row.addWidget(label)
            row.addWidget(status, 1)
            row.addWidget(connect)
            row.addWidget(check)
            layout.addLayout(row)
            self._rows[platform] = status, connect, check
        self.completed.connect(self._present)

    def _operate(self, platform: str, action: str) -> None:
        status, connect, check = self._rows[platform]
        if not connect.isEnabled():
            return
        try:
            gateway = self._provider() if callable(self._provider) else self._provider
        except Exception:
            status.setText("Messaging services are unavailable. Try again shortly.")
            return
        if gateway is None:
            status.setText("Messaging services are still starting. Try again shortly.")
            return
        connect.setEnabled(False)
        check.setEnabled(False)
        status.setText("Opening sign-in…" if action == "connect" else "Checking…")

        def work() -> None:
            from ..data import ToolResult
            try:
                from ..execution_policy import background_scope, foreground_scope
                if action == "connect":
                    with foreground_scope("preferences.messaging.connect"):
                        result = gateway.browser_connect(platform)
                else:
                    with background_scope("preferences.messaging.check"):
                        result = gateway.browser_check_connection(platform)
            except Exception as exc:
                result = ToolResult(f"Connection check failed ({type(exc).__name__}).", ok=False)
            try:
                self.completed.emit(platform, result)
            except RuntimeError:
                pass  # Closing Preferences never replays or cancels delivery.

        try:
            threading.Thread(target=work, name=f"orion-messaging-{action}-{platform}", daemon=True).start()
        except RuntimeError:
            status.setText("Unable to start the connection check. Try again shortly.")
            connect.setEnabled(True)
            check.setEnabled(True)

    def _present(self, platform: str, result: Any) -> None:
        from PyQt6 import sip
        if sip.isdeleted(self):
            return
        status, connect, check = self._rows[platform]
        status.setText(str(getattr(result, "text", "Connection status unavailable.")))
        connect.setEnabled(True)
        check.setEnabled(True)


#: (settings key, label, what it does). The labels are names, so Title Case.
WINDOWS_TOGGLES: tuple[tuple[str, str, str], ...] = (
    ("hotkey_enabled", "Global Hotkey",
     "A key chord that brings ORION up from any window."),
    ("dock_enabled", "Docked Screen Face",
     "Allows ORION's face to be parked on the edge of the screen, always on top, never "
     "taking focus. Ask for it with Show Dock in the tray menu, or say \"show your dock\"."),
    ("dock_on_startup", "Show Dock at Startup",
     "Put the docked face on the screen as soon as ORION starts. Off: it only appears "
     "when you ask for it."),
    ("explorer_menu", 'Explorer "Ask ORION" Menu',
     "Adds Ask ORION to the right-click menu of files and folders in File Explorer."),
    ("protocol", "ORION Links",
     "Lets orion:// links and notification buttons open ORION."),
    ("jump_list", "Taskbar Jump List",
     "Adds ORION's tasks to the right-click menu of his taskbar button."),
    ("toasts", "Notifications with Buttons",
     "Windows notifications ORION can answer with a click. Off: plain tray balloons."),
    ("greet_on_unlock", "Greet Me When I Unlock",
     "A short hello when you unlock the PC."),
    ("quiet_when_busy", "Stay Quiet When I'm Busy",
     "No speech or pop-ups during Do Not Disturb, presentations and full-screen games."),
    ("pause_hearing_when_locked", "Pause Listening When Locked",
     "ORION stops hearing the room while the PC is locked."),
    ("start_in_tray_at_login", "Start in Tray at Sign-In",
     "At sign-in ORION starts in the tray (just the icon) and opens his windows when you ask."),
)


class _WindowsIntegration(QGroupBox):
    """How far ORION reaches into Windows: one switch per piece.

    Each box is read from, and saved to, ``windows_settings`` the moment it is
    toggled - no OK button to forget. Registrations (the hotkey, the dock, the
    Explorer entry, orion:// links, the jump list) are made at start-up, so
    they take effect the next time ORION starts; the hint says so.
    """

    #: (key, value) after a switch was saved.
    changed = pyqtSignal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Windows Integration", parent)
        from .. import windows_settings

        self._settings = windows_settings
        self.checks: dict[str, QCheckBox] = {}
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        for key, label, hint in WINDOWS_TOGGLES:
            box = QCheckBox(label)
            box.setChecked(bool(windows_settings.get(key)))
            describe_control(box, label, hint)
            box.toggled.connect(lambda checked, k=key, b=box: self._save(k, bool(checked), b))
            self.checks[key] = box
            layout.addWidget(box)
            if key == "hotkey_enabled":
                self.hotkey_status = QLabel("")
                self.hotkey_status.setObjectName("hint")
                self.hotkey_status.setWordWrap(True)
                self.hotkey_status.setContentsMargins(24, 0, 0, 0)
                self.hotkey_status.setAccessibleName("Active hotkey")
                layout.addWidget(self.hotkey_status)
        self.hint = QLabel("Takes effect after a restart")
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Windows integration status")
        layout.addWidget(self.status)
        self.refresh_hotkey()

    def refresh_hotkey(self) -> None:
        """Show the chord that is live (or the one that will be tried)."""
        text = ""
        try:
            from .. import windows_presence

            text = windows_presence.active_hotkey_text()
            if not text:
                chord = windows_presence.parse_chord(str(self._settings.get("hotkey")))
                text = f"Chord: {chord.label} (registered when ORION starts)."
        except Exception:
            text = ""
        self.hotkey_status.setText(text)

    def _save(self, key: str, value: bool, box: QCheckBox) -> None:
        if self._settings.set_value(key, value):
            self.status.setText("")
            self.changed.emit(key, value)
            self._log(f"[ORION] WINDOWS: {key} = {'on' if value else 'off'} "
                      f"(applies from the next start).")
            return
        # Never leave a box showing a choice that was not kept.
        box.blockSignals(True)
        box.setChecked(not value)
        box.blockSignals(False)
        self.status.setText("That setting could not be saved.")

    def _log(self, text: str) -> None:
        import logging

        logging.getLogger("orion.windows").info(text)
        try:
            bus = getattr(self.window().parent(), "bus", None)
            if bus is not None:
                bus.log.emit(text)
        except Exception:
            pass


class _BrainSelector(QGroupBox):
    """Which brain answers: Cloud, Hybrid or Local Only (brain_mode.py).

    Saved the moment it changes, like the Windows switches, and reported
    through ``changed`` so the running assistant can apply it at once (close
    Live for the local brain, start Ollama for hybrid/local).
    """

    #: The mode that was just saved.
    changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Brain", parent)
        from .. import brain_mode

        self._modes = brain_mode
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        self.combo = QComboBox()
        for mode in brain_mode.MODES:
            self.combo.addItem(brain_mode.LABELS[mode], mode)
            self.combo.setItemData(self.combo.count() - 1, brain_mode.EXPLANATIONS[mode],
                                   Qt.ItemDataRole.ToolTipRole)
        describe_control(self.combo, "Brain",
                         "Which brain answers: the cloud models, the local model first "
                         "(hybrid), or the local model only, with no AI service at all.")
        index = self.combo.findData(brain_mode.stored_mode())
        self.combo.setCurrentIndex(max(0, index))
        layout.addWidget(self.combo)
        self.explanation = QLabel("")
        self.explanation.setObjectName("hint")
        self.explanation.setWordWrap(True)
        self.explanation.setAccessibleName("What this brain does")
        layout.addWidget(self.explanation)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Brain setting status")
        layout.addWidget(self.status)
        self._explain()
        self.combo.currentIndexChanged.connect(self._save)

    def selected(self) -> str:
        return str(self.combo.currentData() or self._modes.DEFAULT_MODE)

    def _explain(self) -> None:
        mode = self.selected()
        text = self._modes.EXPLANATIONS.get(mode, "")
        if self._modes.env_override_active():
            text += " (ORION_BRAIN_MODE is set, and overrides this choice while it is.)"
        self.explanation.setText(text)

    def _save(self, _index: int) -> None:
        mode = self.selected()
        self._explain()
        if self._modes.set_mode(mode) is None:
            self.status.setText("That setting could not be saved.")
            return
        self.status.setText("")
        self.changed.emit(mode)


class _VoiceSelector(QGroupBox):
    """How ORION sounds and talks (voice_settings.py): a style (Professional -
    the cute style was retired in Mark XXXIX) -
    or the online Gemini voice, the offline Kokoro voice and its speed on their
    own. Saved the moment anything changes, like the Brain and the Windows
    switches. The offline voice is read at every line, so it changes at once;
    the online voice from the next conversation."""

    #: (settings key, new value) after something was saved.
    changed = pyqtSignal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Voice", parent)
        from .. import voice_settings

        self._vs = voice_settings
        self._syncing = True
        form = QFormLayout(self)
        form.setSpacing(8)
        self.style = QComboBox()
        for key, style in voice_settings.STYLES.items():
            self.style.addItem(style.label, key)
        self.style.addItem("Custom", "custom")
        describe_control(self.style, "Voice style",
                         "Professional sets the online and the offline voice together, "
                         "and how ORION talks.")
        self.online = QComboBox()
        describe_control(self.online, "Online voice",
                         "The Gemini voice used in conversation. Changes from the next conversation.")
        self.offline = QComboBox()
        describe_control(self.offline, "Offline voice",
                         "The voice ORION uses with no internet, and whenever the online voice cannot speak.")
        self.speed = QDoubleSpinBox()
        low, high = voice_settings.KOKORO_SPEED_LIMITS
        self.speed.setRange(low, high)
        self.speed.setSingleStep(0.05)
        self.speed.setDecimals(2)
        describe_control(self.speed, "Offline speed",
                         "How fast the offline voice talks. 1 is normal; a little faster sounds brighter.")
        self.pause = QComboBox()
        for key, label in (("brisk", "Brisk"), ("natural", "Natural"), ("patient", "Patient")):
            self.pause.addItem(label, key)
        describe_control(self.pause, "Reply pause",
                         "How long you can pause mid-sentence before ORION replies.")
        self.expressive = QCheckBox("Expressive Voice")
        describe_control(self.expressive, "Expressive voice",
                         "ORION's tone follows yours: warmer, quieter or brisker.")
        form.addRow(QLabel("Style"), self.style)
        form.addRow(QLabel("Online Voice"), self.online)
        form.addRow(QLabel("Offline Voice"), self.offline)
        form.addRow(QLabel("Offline Speed"), self.speed)
        form.addRow(QLabel("Reply Pause"), self.pause)
        form.addRow(self.expressive)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Voice setting status")
        form.addRow(self.status)
        self.refresh()
        self.style.activated.connect(self._style_chosen)
        self.online.activated.connect(lambda _i: self._save("cloud_voice", self.online.currentData()))
        self.offline.activated.connect(lambda _i: self._save("kokoro_voice", self.offline.currentData()))
        self.speed.valueChanged.connect(lambda value: self._save("kokoro_speed", value))
        self.pause.activated.connect(lambda _i: self._save("reply_pause", self.pause.currentData()))
        self.expressive.toggled.connect(lambda on: self._save("expressive", bool(on)))

    def refresh(self) -> None:
        """Show what is in effect (an environment override included)."""
        vs = self._vs
        self._syncing = True
        try:
            for combo, candidates, key in ((self.online, vs.CLOUD_CANDIDATES, "cloud_voice"),
                                           (self.offline, vs.KOKORO_CANDIDATES, "kokoro_voice")):
                current = vs.get(key)
                combo.clear()
                for value, label in candidates:
                    combo.addItem(label, value)
                if combo.findData(current) < 0:         # chosen by phrase or file: still shown
                    combo.addItem(str(current), current)
                combo.setCurrentIndex(max(0, combo.findData(current)))
            self.speed.setValue(float(vs.get("kokoro_speed")))
            self.pause.setCurrentIndex(max(0, self.pause.findData(vs.get("reply_pause"))))
            self.expressive.setChecked(bool(vs.get("expressive")))
            self.style.setCurrentIndex(max(0, self.style.findData(vs.chosen_style())))
            overridden = [name for name in vs.ENV_NAMES.values() if os.getenv(name)]
            self.status.setText(("Set by the environment for this run: " + ", ".join(overridden)
                                 + ".") if overridden else "")
        finally:
            self._syncing = False

    def _style_chosen(self, _index: int) -> None:
        key = self.style.currentData()
        if key == "custom":
            return
        message, ok = self._vs.apply_style(key)
        self.refresh()
        if not ok:
            self.status.setText("That voice could not be saved.")
        if ok:
            for setting in ("cloud_voice", "kokoro_voice", "kokoro_speed"):
                self.changed.emit(setting, self._vs.get(setting))

    def _save(self, key: str, value: Any) -> None:
        if self._syncing:
            return
        saved = self._vs.set_value(key, value)
        self.refresh()                       # what is really in effect, not what was asked for
        if saved:
            self.changed.emit(key, self._vs.get(key))
        else:
            self.status.setText("That voice could not be saved.")


class PreferencesDialog(QDialog):
    """ORION's settings. Returns the chosen values; saving is the caller's job,
    so this stays testable without a running assistant."""

    #: (settings key, value): a Windows Integration switch was saved. These are
    #: stored the moment they are toggled, not when OK is pressed.
    windows_setting_changed = pyqtSignal(str, bool)
    #: The brain mode (cloud / hybrid / local) was saved; apply it now.
    brain_mode_changed = pyqtSignal(str)
    #: (voice setting key, value): saved; the live channel picks it up.
    voice_setting_changed = pyqtSignal(str, object)

    def __init__(self, parent: QWidget | None = None, *,
                 assistant_name: str = "Orion", user_name: str = "",
                 accent: str = "", messaging_provider: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("O.R.I.O.N. — Preferences")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setStyleSheet(_stylesheet())

        self._accent = QColor(accent) if accent else QColor(C.PRI)

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(0)
        contents = QWidget()
        layout = QVBoxLayout(contents)
        layout.setSpacing(10)
        scroll.setWidget(contents)
        outer.addWidget(scroll, 1)
        self.resize(620, 720)

        heading = QLabel("Preferences")
        heading.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        layout.addWidget(heading)

        # ── brain (saved as it changes) ──────────────────────────────────────
        self.brain = _BrainSelector(self)
        self.brain.changed.connect(self.brain_mode_changed)
        layout.addWidget(self.brain)

        # ── voice (saved as it changes) ──────────────────────────────────────
        self.voice = _VoiceSelector(self)
        self.voice.changed.connect(self.voice_setting_changed)
        layout.addWidget(self.voice)

        # ── audio ────────────────────────────────────────────────────────────
        audio_box = QGroupBox("Audio")
        audio_form = QFormLayout(audio_box)
        audio_form.setSpacing(8)
        self.microphone = _DeviceRow("input")
        self.speakers = _DeviceRow("output")
        audio_form.addRow(QLabel("Microphone"), self.microphone)
        audio_form.addRow(QLabel("Speakers"), self.speakers)
        layout.addWidget(audio_box)

        messaging_box = QGroupBox("Background Messaging")
        messaging_layout = QVBoxLayout(messaging_box)
        self.messaging_connections = _MessagingConnections(messaging_provider, self)
        messaging_layout.addWidget(self.messaging_connections)
        layout.addWidget(messaging_box)

        # ── identity ─────────────────────────────────────────────────────────
        identity_box = QGroupBox("Identity")
        identity_form = QFormLayout(identity_box)
        identity_form.setSpacing(8)
        self.assistant_edit = QLineEdit(assistant_name)
        # Shown, not offered: his name is fixed in code (the wake words, the
        # spoken-name rule, the transcription hints all depend on it), and an
        # edit here used to be accepted and then silently dropped.
        self.assistant_edit.setReadOnly(True)
        describe_control(self.assistant_edit, "Assistant name",
                         "ORION's own name. It is part of who he is, so it is fixed.")
        self.user_edit = QLineEdit(user_name)
        self.user_edit.setPlaceholderText("What he should call you")
        describe_control(self.user_edit, "Your name",
                         "What ORION calls you.")
        identity_form.addRow(QLabel("He is called"), self.assistant_edit)
        identity_form.addRow(QLabel("You are called"), self.user_edit)
        layout.addWidget(identity_box)
        self.voice_learning = _VoiceLearning(self.user_edit.text, self)
        layout.addWidget(self.voice_learning)
        self.finished.connect(self.voice_learning.cancel_owned)
        self.conversation_recording = _ConversationRecording(self)
        layout.addWidget(self.conversation_recording)

        # ── privacy (saved as it is toggled) ─────────────────────────────────
        privacy_box = QGroupBox("Privacy")
        privacy_form = QFormLayout(privacy_box)
        self.private_log = QCheckBox("Private Log")
        describe_control(self.private_log, "Private log",
                         "His log file keeps what he does, never what either of you says.")
        try:
            from .. import session_log as _session_log
            self.private_log.setChecked(_session_log.load_private())
        except Exception:
            pass
        self.private_log.toggled.connect(self._set_private_log)
        privacy_form.addRow(self.private_log)
        layout.addWidget(privacy_box)

        # ── appearance ───────────────────────────────────────────────────────
        appearance_box = QGroupBox("Appearance")
        appearance_row = QHBoxLayout(appearance_box)
        self.swatch = QLabel()
        self.swatch.setFixedSize(46, 24)
        self._paint_swatch()
        self.colour_button = QPushButton("Accent Colour…")
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

        # ── motion and performance ───────────────────────────────────────────
        motion_box = QGroupBox("Motion and Performance")
        motion_form = QFormLayout(motion_box)
        motion_form.setSpacing(8)
        self.motion = QComboBox()
        describe_control(self.motion, "Motion and performance",
                         "How much the face drifts when idle, how sharply it is "
                         "rendered, and whether deck pages slide.")
        current_motion = "full"
        system_reduced = False
        try:
            from .. import appearance
            for key, profile in appearance.MOTION_PROFILES.items():
                self.motion.addItem(profile.label, key)
            current_motion = appearance.motion_profile().name
            system_reduced = appearance.system_prefers_reduced_motion()
        except Exception:
            self.motion.addItem("Full", "full")
        index = self.motion.findData(current_motion)
        self.motion.setCurrentIndex(max(0, index))
        motion_form.addRow(QLabel("Motion"), self.motion)
        # Mark XXXIX: the Screen Face's designs (appearance.FACE_DESIGNS).
        self.face_design = QComboBox()
        describe_control(self.face_design, "Face design",
                         "How ORION's Screen Face draws its eyes and mouth.")
        try:
            from .. import appearance as _appearance
            for key in _appearance.FACE_DESIGNS:
                self.face_design.addItem(_appearance.DESIGN_LABELS[key], key)
            current_design = _appearance.face_design()
        except Exception:
            self.face_design.addItem("Sentinel", "sentinel")
            current_design = "sentinel"
        self.face_design.setCurrentIndex(max(0, self.face_design.findData(current_design)))
        motion_form.addRow(QLabel("Face Design"), self.face_design)
        if system_reduced:
            motion_note = QLabel("Windows animations are off on this PC")
            motion_note.setObjectName("hint")
            motion_form.addRow(motion_note)
        layout.addWidget(motion_box)

        # ── Windows integration (saved as each box is toggled) ───────────────
        self.windows_integration = _WindowsIntegration(self)
        self.windows_integration.changed.connect(self.windows_setting_changed)
        layout.addWidget(self.windows_integration)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.microphone.refresh()
        self.speakers.refresh()

    def _set_private_log(self, on: bool) -> None:
        try:
            from .. import session_log
            session_log.set_private(on)
        except Exception:
            pass

    # ── appearance helpers ───────────────────────────────────────────────────

    def _paint_swatch(self) -> None:
        self.swatch.setStyleSheet(
            f"background: {self._accent.name()}; border: 1px solid {C.BORDER}; "
            f"border-radius: 5px;")
        self.swatch.setToolTip(self._accent.name())
        self.swatch.setAccessibleName(f"Accent colour {self._accent.name()}")

    def _choose_colour(self) -> None:
        chosen = QColorDialog.getColor(self._accent, self, "ORION's Accent Colour")
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
            "motion": str(self.motion.currentData() or "full"),
            "face_design": str(self.face_design.currentData() or ""),
            "brain_mode": self.brain.selected(),
        }


__all__ = ["SYSTEM_DEFAULT", "WINDOWS_TOGGLES", "PreferencesDialog"]
