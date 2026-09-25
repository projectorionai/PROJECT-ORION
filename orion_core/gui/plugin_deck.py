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

# The same OK/DEGRADED/DOWN idea every other surface uses. This file kept a
# private green and yellow and so escaped the status-colour consolidation that
# removed them everywhere else — see tests/test_status_colour_consolidation.py,
# which asserts those two hexes are gone.
_STATUS_COLOURS = {
    "loaded":    C.GOOD,
    "ready":     C.GOOD,
    "disabled":  C.MUTED,
    "needs-deps": C.WARN,
    "invalid":   C.BAD,
    "missing":   C.BAD,
    "error":     C.BAD,
}


class _PluginRow(QFrame):
    """One plugin: identity, tier, status, and its enable/disable control."""

    def __init__(self, record: dict[str, Any], on_toggle: Any) -> None:
        super().__init__()
        self.setObjectName("panelFrame")
        self._name = str(record.get("name") or "")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        status = str(record.get("status") or "?")
        colour = _STATUS_COLOURS.get(status, C.MUTED)
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{colour}; font-size:15px;")
        row.addWidget(dot)

        text = QLabel()
        text.setTextFormat(Qt.TextFormat.RichText)
        text.setWordWrap(True)
        description = str(record.get("description") or "")[:120]
        # Mark XXVI: the row also carries the version and the CAPABILITY
        # DISCLOSURE — what this plugin can touch, and anything it touches
        # without having declared it. That is the fact a user needs in front of
        # them when deciding whether to keep a dropped-in plugin enabled.
        version = str(record.get("version") or "")
        stamp = f"v{version} · " if version and version != "0.0.0" else ""
        detail = (f"<span style='color:{C.MUTED}'>{stamp}tier "
                  f"{record.get('tier')} · {status}</span>")
        permissions = list(record.get("permissions") or [])
        undeclared = list(record.get("undeclared") or [])
        if permissions:
            detail += (f"<br><span style='color:{C.MUTED}'>needs: "
                       f"{', '.join(permissions)}</span>")
        if undeclared:
            detail += (f"<br><span style='color:#f1c40f'>⚠ uses undeclared: "
                       f"{', '.join(undeclared)}</span>")
        problems = list(record.get("problems") or [])
        missing = list(record.get("missing_deps") or [])
        if missing:
            detail += (f"<br><span style='color:#f1c40f'>needs: "
                       f"{', '.join(missing)}</span>")
        if problems:
            detail += f"<br><span style='color:{C.PRI}'>{problems[0][:90]}</span>"
        if record.get("last_error"):
            detail += (f"<br><span style='color:{C.PRI}'>"
                       f"{str(record['last_error'])[:90]}</span>")
        text.setText(
            f"<b>{self._name}</b>"
            + (f" <span style='color:{C.MUTED}'>— {description}</span>" if description else "")
            + f"<br>{detail}"
        )
        row.addWidget(text, 1)

        enabled = bool(record.get("enabled"))
        self.toggle = QPushButton("DISABLE" if enabled else "ENABLE")
        if not enabled:
            self.toggle.setObjectName("ghostButton")
        self.toggle.setFixedWidth(96)
        self.toggle.clicked.connect(lambda: on_toggle(self._name, not enabled))
        row.addWidget(self.toggle)


class PluginDeckView(QWidget):
    """The PLUGINS page of the Command Deck."""

    SAMPLE_MS = 4000

    def __init__(self, bus: OrionBus, registry: Any = None) -> None:
        super().__init__()
        self.bus = bus
        self.registry = registry
        self._signature: str = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("PLUGINS")
        title.setObjectName("titleLabel")
        root.addWidget(title)

        self.summary = QLabel("—")
        self.summary.setObjectName("mutedLabel")
        root.addWidget(self.summary)

        controls = QHBoxLayout()
        for label, slot in (("＋ NEW PLUGIN", self._create),
                            ("⟳ RELOAD", self._refresh),
                            ("⚕ DOCTOR", self._doctor),
                            ("MANIFESTS", self._backfill)):
            button = QPushButton(label)
            if label != "＋ NEW PLUGIN":
                button.setObjectName("ghostButton")
            button.clicked.connect(slot)
            controls.addWidget(button)
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
        self._signature = ""
        self._refresh()

    # ── sampling ─────────────────────────────────────────────────────────────

    def _sample(self) -> None:
        if not self.isVisible() or self.registry is None:
            return
        try:
            snapshot = self.registry.snapshot()
        except Exception:
            return
        # Rebuild only when something actually changed — a timer that rebuilds
        # a widget tree every few seconds would fight the user's scroll.
        signature = repr(snapshot)
        if signature != self._signature:
            self._signature = signature
            self._render(snapshot)

    def _refresh(self) -> None:
        if self.registry is None:
            self.summary.setText("The plugin registry is not attached yet.")
            return
        try:
            snapshot = self.registry.snapshot()
        except Exception as exc:
            self.summary.setText(f"Plugin registry unavailable — {exc}")
            return
        self._signature = repr(snapshot)
        self._render(snapshot)

    def _render(self, snapshot: dict[str, Any]) -> None:
        while self._list.count() > 1:                 # keep the trailing stretch
            item = self._list.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        records = list(snapshot.get("plugins") or [])
        self.summary.setText(
            f"{snapshot.get('total', 0)} installed · "
            f"{snapshot.get('enabled', 0)} enabled · "
            f"{snapshot.get('healthy', 0)} healthy"
        )
        if not records:
            empty = QLabel("No plugins yet. ＋ NEW PLUGIN scaffolds a working "
                           "one you can edit straight away.")
            empty.setObjectName("mutedLabel")
            empty.setWordWrap(True)
            self._list.insertWidget(0, empty)
            return
        for index, record in enumerate(records):
            self._list.insertWidget(index, _PluginRow(record, self._toggle))

    # ── actions (all delegated to the registry) ──────────────────────────────

    def _toggle(self, name: str, enable: bool) -> None:
        if self.registry is None:
            return
        try:
            result = self.registry.set_enabled(name, enable)
            self.report.setText(result.text)
        except Exception as exc:
            self.report.setText(f"Could not change '{name}': {exc}")
        self._refresh()

    def _create(self) -> None:
        if self.registry is None:
            return
        name, ok = QInputDialog.getText(self, "New plugin", "Plugin name:")
        if not ok or not str(name).strip():
            return
        description, ok = QInputDialog.getText(
            self, "New plugin", "What should it do?")
        if not ok:
            return
        try:
            result = self.registry.create(str(name), str(description))
            self.report.setText(result.text)
        except Exception as exc:
            self.report.setText(f"Could not create the plugin: {exc}")
        self._refresh()

    def _doctor(self) -> None:
        if self.registry is None:
            return
        try:
            self.report.setText(self.registry.doctor().text)
        except Exception as exc:
            self.report.setText(f"Doctor failed: {exc}")

    def _backfill(self) -> None:
        if self.registry is None:
            return
        try:
            written = self.registry.backfill_manifests()
            remaining = self.registry.manifestless()
            if written:
                message = f"Wrote {len(written)} manifest(s): {', '.join(written)}"
            else:
                message = "No manifests needed writing."
            if remaining:
                message += (f"  Still without one (they fail the contract — "
                            f"run DOCTOR): {', '.join(remaining)}")
            self.report.setText(message)
        except Exception as exc:
            self.report.setText(f"Backfill failed: {exc}")
        self._refresh()

    def showEvent(self, event: Any) -> None:      # noqa: N802  (Qt naming)
        self._refresh()
        super().showEvent(event)
