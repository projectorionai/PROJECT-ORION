"""
PluginDeckView — the visible face of ORION's plugin ecosystem.

Plugins were previously invisible: they lived as files under
``config/custom_tools/`` and could only be inspected by reading the log at
startup.  This page makes the whole surface legible and operable —

    • every plugin, its capability tier, and its live status;
    • a one-click enable / disable that PERSISTS (writes the registry state,
      so a disabled plugin stays disabled across restarts);
    • the health picture ``plugin doctor`` reports — missing dependencies,
      contract violations, load errors — shown rather than buried;
    • the actions that previously required hand-editing files: create a new
      plugin, backfill missing manifests, re-run the doctor.

Strictly a CONSUMER of :class:`~orion_core.plugin_registry.PluginRegistry`:
it samples on a timer only while visible (the same discipline the Diagnostics
Centre uses), and every action is delegated to the registry rather than
reimplemented here, so the tool path and the GUI path can never disagree.
"""

from __future__ import annotations

import asyncio
from html import escape
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import C
from .. import background
from .labels import plural

# The same OK/DEGRADED/DOWN idea every other surface uses. This file kept a
# private green and yellow and so escaped the status-colour consolidation that
# removed them everywhere else — see tests/test_status_colour_consolidation.py,
# which asserts those two hexes are gone.
_STATUS_COLOURS = {
    "loaded":    C.GOOD,
    "ready":     C.GOOD,
    "disabled":  C.MUTED,
    "held":      C.WARN,
    "needs-deps": C.WARN,
    "invalid":   C.BAD,
    "missing":   C.BAD,
    "error":     C.BAD,
    # Deliberate states, not faults (25 Oct 2026: both showed red as errors).
    "superseded": C.MUTED,
    "quarantined": C.WARN,
}

#: What the row says for a deliberate state, in place of the raw reason.
_STATUS_WORDS = {"superseded": "Superseded", "quarantined": "Quarantined"}


class _PluginRow(QFrame):
    """One plugin: identity, tier, status, and its enable/disable control."""

    def __init__(self, record: dict[str, Any], on_toggle: Any,
                 on_restore: Any = None, on_remove: Any = None) -> None:
        super().__init__()
        self.setObjectName("panelFrame")
        self._name = str(record.get("name") or "")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        status = str(record.get("status") or "?")
        colour = _STATUS_COLOURS.get(status, C.MUTED)
        superseded = str(record.get("superseded_by") or "")
        quarantined = str(record.get("quarantined") or "")
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{colour}; font-size:15px;")
        row.addWidget(dot)

        text = QLabel()
        text.setTextFormat(Qt.TextFormat.RichText)
        text.setWordWrap(True)
        description = escape(str(record.get("description") or "")[:120])
        # Mark XXVI: the row also carries the version and the CAPABILITY
        # DISCLOSURE — what this plugin can touch, and anything it touches
        # without having declared it. That is the fact a user needs in front of
        # them when deciding whether to keep a dropped-in plugin enabled.
        version = escape(str(record.get("version") or ""))
        stamp = f"v{version} · " if version and version != "0.0.0" else ""
        detail = (f"<span style='color:{C.MUTED}'>{stamp}tier "
                  f"{escape(str(record.get('tier')))} · "
                  f"{escape(_STATUS_WORDS.get(status, status))}</span>")
        if superseded:
            detail += (f"<br><span style='color:{C.MUTED}'>Superseded by built-in "
                       f"{escape(superseded)} - held back on purpose, not an error</span>")
        if quarantined:
            detail += (f"<br><span style='color:{C.WARN}'>Quarantined: "
                       f"{escape(quarantined[:110])}</span>")
        permissions = list(record.get("permissions") or [])
        undeclared = list(record.get("undeclared") or [])
        if permissions:
            detail += (f"<br><span style='color:{C.MUTED}'>needs: "
                       f"{escape(', '.join(str(value) for value in permissions))}</span>")
        if undeclared:
            detail += (f"<br><span style='color:#f1c40f'>uses undeclared: "
                       f"{escape(', '.join(str(value) for value in undeclared))}</span>")
        problems = list(record.get("problems") or [])
        missing = list(record.get("missing_deps") or [])
        if missing:
            detail += (f"<br><span style='color:#f1c40f'>needs: "
                       f"{escape(', '.join(str(value) for value in missing))}</span>")
        if problems:
            detail += f"<br><span style='color:{C.PRI}'>{escape(str(problems[0])[:90])}</span>"
        if record.get("last_error"):
            detail += (f"<br><span style='color:{C.PRI}'>"
                       f"{escape(str(record['last_error'])[:90])}</span>")
        text.setText(
            f"<b>{escape(self._name)}</b>"
            + (f" <span style='color:{C.MUTED}'>— {description}</span>" if description else "")
            + f"<br>{detail}"
        )
        row.addWidget(text, 1)

        enabled = bool(record.get("enabled"))
        held = status == "held" or bool(superseded)
        self._held = held
        self.actions: list[QPushButton] = []
        if quarantined and status != "held":
            # Restore or retire it; enable/disable means nothing for a module
            # that is not there.
            for label, slot, tip in (
                    ("Restore", on_restore, "Put the quarantined module back; it loads on the next reload"),
                    ("Remove", on_remove, "Archive this plugin's files out of the active set")):
                button = QPushButton(label)
                button.setObjectName("ghostButton")
                button.setFixedWidth(96)
                button.setAccessibleName(f"{label} plugin {self._name}")
                button.setToolTip(tip)
                button.setEnabled(slot is not None)
                if slot is not None:
                    button.clicked.connect(lambda _=False, s=slot: s(self._name))
                row.addWidget(button)
                self.actions.append(button)
            self.toggle = self.actions[0]
            return
        self.toggle = QPushButton("Superseded" if superseded else
                                  "Held" if held else ("Disable" if enabled else "Enable"))
        self.toggle.setEnabled(not held)
        if not enabled or superseded:
            self.toggle.setObjectName("ghostButton")
        self.toggle.setFixedWidth(96)
        self.toggle.setAccessibleName(f"{self.toggle.text().title()} plugin {self._name}")
        self.toggle.clicked.connect(lambda: on_toggle(self._name, not enabled))
        row.addWidget(self.toggle)
        self.actions.append(self.toggle)


class PluginDeckView(QWidget):
    """The PLUGINS page of the Command Deck."""

    SAMPLE_MS = 4000

    def __init__(self, bus: OrionBus, registry: Any = None) -> None:
        super().__init__()
        self.bus = bus
        self.registry = registry
        self._signature: str = ""
        self._busy = False
        self._state_unavailable = False
        self._action_task = None
        self._refresh_task = None
        self._refresh_again = False
        self._revision = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Plugins")
        title.setObjectName("titleLabel")
        root.addWidget(title)

        self.summary = QLabel("—")
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        self.summary.setObjectName("mutedLabel")
        root.addWidget(self.summary)

        controls = QHBoxLayout()
        self._mutation_buttons: list[QPushButton] = []
        self._controls: list[QPushButton] = []
        for label, slot in (("＋ New Plugin", self._create),
                            ("⟳ Reload", self._refresh),
                            ("Doctor", self._doctor),
                            ("Manifests", self._backfill)):
            button = QPushButton(label)
            button.setStyleSheet(f"QPushButton:disabled {{ color: {C.MUTED}; border-color: {C.MUTED}; }}")
            if label != "＋ New Plugin":
                button.setObjectName("ghostButton")
            button.clicked.connect(slot)
            controls.addWidget(button)
            self._controls.append(button)
            if label in ("＋ New Plugin", "Manifests"):
                self._mutation_buttons.append(button)
        controls.addStretch(1)
        root.addLayout(controls)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._holder = QWidget()
        self._list = QVBoxLayout(self._holder)
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(8)
        self._list.addStretch(1)
        scroll.setWidget(self._holder)
        root.addWidget(scroll, 1)

        self.report = QLabel("")
        self.report.setTextFormat(Qt.TextFormat.PlainText)
        self.report.setObjectName("mutedLabel")
        self.report.setWordWrap(True)
        root.addWidget(self.report)

        self._timer = QTimer(self)
        self._timer.setInterval(self.SAMPLE_MS)
        self._timer.timeout.connect(self._sample)
        self._timer.start()
        self._refresh()

    # ── attachment ───────────────────────────────────────────────────────────

    def attach_registry(self, registry: Any) -> None:
        """Wire the registry in after construction (it is built later in boot)."""
        self.registry = registry
        self._revision += 1
        self._signature = ""
        self._refresh()

    # ── sampling ─────────────────────────────────────────────────────────────

    def _sample(self) -> None:
        if not self.isVisible() or self.registry is None:
            return
        self._refresh(force=False)

    def _refresh(self, *, force: bool = True) -> None:
        if self.registry is None:
            self.summary.setText("The plugin registry is not attached yet.")
            return
        if self._busy:
            self._refresh_again = True
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Standalone Qt consumers without qasync keep their synchronous API.
            try:
                self._apply_snapshot(self.registry.snapshot(), force=force)
            except Exception as exc:
                self.summary.setText(f"Plugin registry unavailable — {exc}")
            return
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_again = self._refresh_again or force
            return

        async def sample() -> None:
            from PyQt6 import sip
            refresh_force = force
            while not sip.isdeleted(self):
                registry = self.registry
                revision = self._revision
                self._refresh_again = False
                try:
                    snapshot = await asyncio.to_thread(registry.snapshot)
                    if sip.isdeleted(self):
                        return
                    if self.registry is registry and revision == self._revision and not self._busy:
                        self._apply_snapshot(snapshot, force=refresh_force)
                except Exception as exc:
                    if not sip.isdeleted(self) and self.registry is registry:
                        self._state_unavailable = True
                        self.summary.setText(f"Plugin registry unavailable — {exc}")
                        self._set_busy(self._busy)
                if sip.isdeleted(self) or self._busy or not self._refresh_again or self.registry is None:
                    return
                refresh_force = True

        self._refresh_task = background.spawn(sample(), name="orion-plugin-panel-refresh")

    def _apply_snapshot(self, snapshot: dict[str, Any], *, force: bool) -> None:
        signature = repr(snapshot)
        self._state_unavailable = bool(snapshot.get("state_error"))
        if force or signature != self._signature:
            self._signature = signature
            self._render(snapshot)
        self._set_busy(self._busy)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for button in self._controls:
            unavailable_mutation = self._state_unavailable and button in self._mutation_buttons
            button.setEnabled(not busy and not unavailable_mutation)
        for row in self._holder.findChildren(_PluginRow):
            for button in row.actions:
                button.setEnabled(not busy and not self._state_unavailable and not row._held)

    def _begin_action(self, registry: Any, operation: Any) -> None:
        if self._busy:
            return
        self._revision += 1
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                self.report.setText(operation().text)
            except Exception as exc:
                self.report.setText(f"Plugin action failed: {exc}")
            self._refresh()
            return
        self._set_busy(True)
        self.report.setText("Working…")

        async def perform() -> None:
            from PyQt6 import sip
            cancelled = False
            try:
                result = await asyncio.to_thread(operation)
                if not sip.isdeleted(self) and self.registry is registry:
                    self.report.setText(result.text)
            except asyncio.CancelledError:
                cancelled = True
                raise
            except Exception as exc:
                if not sip.isdeleted(self) and self.registry is registry:
                    self.report.setText(f"Plugin action failed: {exc}")
            finally:
                if not sip.isdeleted(self):
                    self._set_busy(False)
                    if not cancelled:
                        self._refresh()

        self._action_task = background.spawn(perform(), name="orion-plugin-panel-action")
        if self._action_task is None:
            self._set_busy(False)
            self.report.setText("The action could not start while ORION is shutting down.")

    def _render(self, snapshot: dict[str, Any]) -> None:
        while self._list.count() > 1:                 # keep the trailing stretch
            item = self._list.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        records = list(snapshot.get("plugins") or [])
        state_error = str(snapshot.get("state_error") or "")
        for button in self._mutation_buttons:
            button.setEnabled(not state_error)
        set_aside = ""
        if snapshot.get("quarantined"):
            set_aside += f" · {snapshot['quarantined']} quarantined"
        if snapshot.get("superseded"):
            set_aside += f" · {snapshot['superseded']} superseded"
        self.summary.setText("Preferences unavailable · plugins held" if state_error else (
            f"{snapshot.get('total', 0)} installed · "
            f"{snapshot.get('enabled', 0)} enabled · "
            f"{snapshot.get('healthy', 0)} healthy" + set_aside
        ))
        archived = [item for item in (snapshot.get("archived") or []) if item.get("name")]
        if archived:
            names = ", ".join(sorted({str(item["name"]) for item in archived}))
            self.summary.setToolTip(
                "Archived as superseded by built-in tools (copies kept in "
                f"custom_tools/_superseded): {names}")
        offset = 0
        if state_error:
            explanation = QLabel("Plugin settings could not be read - plugins are paused.")
            explanation.setTextFormat(Qt.TextFormat.PlainText)
            explanation.setObjectName("mutedLabel")
            explanation.setWordWrap(True)
            explanation.setToolTip(escape(state_error))
            self._list.insertWidget(0, explanation)
            offset = 1
            if not records:
                return
        if not records:
            empty = QLabel("No plugins yet")
            empty.setObjectName("mutedLabel")
            empty.setWordWrap(True)
            self._list.insertWidget(0, empty)
            return
        if archived:
            note = QLabel("Superseded by built-in tools and archived (copies kept in "
                          "custom_tools/_superseded): " + ", ".join(
                              f"{item['name']} → {item.get('superseded_by') or '?'}"
                              for item in archived[:6]))
            note.setTextFormat(Qt.TextFormat.PlainText)
            note.setObjectName("mutedLabel")
            note.setWordWrap(True)
            self._list.insertWidget(offset, note)
            offset += 1
        for index, record in enumerate(records):
            shown = {**record, "status": "held", "enabled": False} if state_error else record
            self._list.insertWidget(index + offset, _PluginRow(
                shown, self._toggle, on_restore=self._restore, on_remove=self._remove))

    # ── actions (all delegated to the registry) ──────────────────────────────

    def _toggle(self, name: str, enable: bool) -> None:
        if self.registry is None:
            return
        registry = self.registry
        self._begin_action(registry, lambda: registry.set_enabled(name, enable))

    def _restore(self, name: str) -> None:
        """Put a quarantined plugin's module back (registry.restore_quarantined)."""
        if self.registry is None:
            return
        registry = self.registry
        self._begin_action(registry, lambda: registry.restore_quarantined(name))

    def _remove(self, name: str) -> None:
        """Archive a quarantined plugin out of the active set, after asking."""
        if self.registry is None or self._busy:
            return
        from PyQt6.QtWidgets import QMessageBox
        answer = QMessageBox.question(
            self, "Remove Plugin",
            f"Remove the quarantined plugin '{name}'? Its files are moved to "
            "custom_tools/_superseded, not deleted.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        registry = self.registry
        self._begin_action(registry, lambda: registry.remove(name))

    def _create(self) -> None:
        if self.registry is None or self._busy:
            return
        name, ok = QInputDialog.getText(self, "New plugin", "Plugin name:")
        if not ok or not str(name).strip():
            return
        description, ok = QInputDialog.getText(
            self, "New Plugin", "What should it do?")
        if not ok:
            return
        registry = self.registry
        self._begin_action(registry, lambda: registry.create(str(name), str(description)))

    def _doctor(self) -> None:
        if self.registry is None:
            return
        registry = self.registry
        self._begin_action(registry, registry.doctor)

    def _backfill(self) -> None:
        if self.registry is None:
            return
        registry = self.registry
        def backfill():
            from ..data import ToolResult
            written = registry.backfill_manifests()
            remaining = registry.manifestless()
            if written:
                message = f"Wrote {plural(len(written), 'manifest')}: {', '.join(written)}"
            else:
                message = "No manifests needed writing."
            if remaining:
                message += (f"  Still without one (they fail the contract — "
                            f"run Doctor): {', '.join(remaining)}")
            return ToolResult(message)
        self._begin_action(registry, backfill)

    def showEvent(self, event: Any) -> None:      # noqa: N802  (Qt naming)
        # Rows are rebuilt only when the registry changed: a forced redraw
        # on every show cost 110-230 ms on the GUI thread (28 Oct 2026).
        self._refresh(force=False)
        super().showEvent(event)
