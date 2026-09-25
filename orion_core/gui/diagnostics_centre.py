"""
OrionDiagnosticsCentre (Mark X.5) — live system observability page.

One glance answers "what is ORION doing right now?":

    • agent/component health (telemetry HealthRegistry, green/amber/red)
    • host memory + CPU usage (psutil, sampled off the hot path)
    • speech state, playback/TTS queue depth and hold state
    • active tool executions and the recent tool ledger
    • event-bus traffic rates (log lines + dashboard events per minute)

Strictly a CONSUMER: it samples telemetry snapshots and dispatcher state on a
2-second QTimer and connects to bus signals for traffic counting.  No service
imports this module (widgets never leak into service layers), and nothing
here blocks — samples are dictionary reads guarded by the registries' own
locks.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import C, HEALTH_COLOURS
from .status_widgets import ProviderStatusStrip


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("panelHeading")
    return label


class _HealthPanel(QFrame):
    """Component health rows rendered from the HealthRegistry snapshot.

    Colours come from HEALTH_COLOURS (constants.py, Mark XX design-spec §9)
    — this panel used to define its own OK/DEGRADED/DOWN triad
    (#2ecc71/#f1c40f/C.PRI), a third independent value set for the same
    concept command_centre.py also redefined on its own.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("AGENT HEALTH"))
        self.body = QLabel("awaiting telemetry…")
        self.body.setObjectName("mutedLabel")
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setWordWrap(True)
        layout.addWidget(self.body)
        layout.addStretch(1)

    def render(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            self.body.setText("no components registered")
            return
        parts = []
        for row in rows[:24]:
            colour = HEALTH_COLOURS.get(str(row.get("status")), C.MUTED)
            detail = str(row.get("detail") or "")[:38]
            parts.append(
                f"<span style='color:{colour}'>●</span> "
                f"{row.get('name')} <span style='color:{C.MUTED}'>{detail}</span>"
            )
        self.body.setText("<br>".join(parts))


class _ResourcePanel(QFrame):
    """Host CPU/RAM plus ORION speech-pipeline gauges."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("panelFrame")
        layout = QGridLayout(self)
        layout.addWidget(_heading("RESOURCES & SPEECH"), 0, 0, 1, 2)
        self.cpu_bar = QProgressBar()
        self.ram_bar = QProgressBar()
        self.gpu_bar = QProgressBar()
        self.vram_bar = QProgressBar()
        for bar in (self.cpu_bar, self.ram_bar, self.gpu_bar, self.vram_bar):
            bar.setRange(0, 100)
            bar.setTextVisible(True)
        layout.addWidget(QLabel("CPU"), 1, 0)
        layout.addWidget(self.cpu_bar, 1, 1)
        layout.addWidget(QLabel("RAM"), 2, 0)
        layout.addWidget(self.ram_bar, 2, 1)
        self.gpu_label = QLabel("GPU")
        layout.addWidget(self.gpu_label, 3, 0)
        layout.addWidget(self.gpu_bar, 3, 1)
        layout.addWidget(QLabel("VRAM"), 4, 0)
        layout.addWidget(self.vram_bar, 4, 1)
        self.speech = QLabel("speech: —")
        self.speech.setObjectName("mutedLabel")
        self.speech.setWordWrap(True)
        layout.addWidget(self.speech, 5, 0, 1, 2)
        self.traffic = QLabel("bus: —")
        self.traffic.setObjectName("mutedLabel")
        layout.addWidget(self.traffic, 6, 0, 1, 2)
        layout.setRowStretch(7, 1)

    def render(self, cpu: float, ram: float, speech: dict[str, Any],
               traffic_per_min: tuple[float, float],
               gpu: dict[str, Any] | None = None) -> None:
        self.cpu_bar.setValue(int(cpu))
        self.ram_bar.setValue(int(ram))
        if gpu and gpu.get("available"):
            self.gpu_bar.setValue(int(gpu.get("util", 0)))
            self.vram_bar.setValue(int(gpu.get("mem_percent", 0)))
            self.gpu_label.setText(
                f"GPU {str(gpu.get('name', '')).replace('NVIDIA ', '')[:16]}"
                + (f" · {gpu.get('temp_c')}°C" if gpu.get("temp_c") else ""))
        else:
            self.gpu_bar.setValue(0); self.vram_bar.setValue(0)
            self.gpu_label.setText("GPU n/a")
        held = " — HELD (resume preserves position)" if speech.get("held") else ""
        self.speech.setText(
            f"speech: {speech.get('state', '?')}{held}   "
            f"playback queue {speech.get('playback_queue_depth', 0)}   "
            f"tts queue {speech.get('tts_queue_depth', 0)}   "
            f"voice: {str(speech.get('local_voice', ''))[:28]}"
        )
        logs_pm, events_pm = traffic_per_min
        self.traffic.setText(
            f"bus traffic: {logs_pm:.0f} log line(s)/min, "
            f"{events_pm:.0f} dashboard event(s)/min"
        )


class _ToolPanel(QFrame):
    """Active tool count + the rolling execution ledger."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(_heading("TOOL EXECUTION"))
        self.active = QLabel("idle")
        self.active.setObjectName("mutedLabel")
        header.addWidget(self.active)
        header.addStretch(1)
        layout.addLayout(header)
        self.ledger = QPlainTextEdit()
        self.ledger.setObjectName("logBox")
        self.ledger.setReadOnly(True)
        self.ledger.setMaximumBlockCount(80)
        layout.addWidget(self.ledger, 1)

    def render(self, active_tools: int, recent: list[dict[str, Any]]) -> None:
        self.active.setText(
            f"{active_tools} running" if active_tools else "idle"
        )
        lines = [
            f"[{r.get('at', '')}] {'✓' if r.get('ok') else '✗'} "
            f"{r.get('tool')} ({r.get('ms')} ms)"
            for r in list(recent)[-30:]
        ]
        self.ledger.setPlainText("\n".join(reversed(lines)) or "no tools executed yet")


class _CrashPanel(QFrame):
    """Recent crashes with a plain-English cause — so "why did ORION crash?"
    has a visible answer in the UI without opening a file by hand."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(_heading("RECENT CRASHES"))
        self.count = QLabel("none")
        self.count.setObjectName("mutedLabel")
        header.addWidget(self.count)
        header.addStretch(1)
        layout.addLayout(header)
        self.body = QLabel("No crashes recorded. ✓")
        self.body.setObjectName("mutedLabel")
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setWordWrap(True)
        layout.addWidget(self.body)
        hint = QLabel("Full reports live in  config/crash_reports/")
        hint.setObjectName("mutedLabel")
        layout.addWidget(hint)
        layout.addStretch(1)

    def render(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            self.count.setText("none")
            self.body.setText(f"<span style='color:{C.GOOD}'>●</span> "
                              "No crashes recorded — all clear.")
            return
        self.count.setText(f"{len(rows)} recent")
        parts = []
        for row in rows[:6]:
            stamp = str(row.get("file", ""))[:15].replace("_", " ")
            why = str(row.get("why") or row.get("summary") or "recorded")[:74]
            parts.append(
                f"<span style='color:{C.PRI}'>▪</span> "
                f"<span style='color:{C.MUTED}'>{stamp}</span>&nbsp; {why}"
            )
        self.body.setText("<br>".join(parts))


class _HealthChecksPanel(QFrame):
    """DiagnosticsEngine's 11 structured checks, plus what the Forge is
    holding back (quarantine, contract failures) and the lesson-corpus
    stats — previously only reachable as buried conversational text, never
    rendered here (Mark X.14)."""

    # A different status vocabulary from HEALTH_COLOURS' OK/DEGRADED/DOWN
    # (this panel reports PASS/WARN/FAIL check outcomes, not component
    # health), but the same three underlying colours — C.GOOD/WARN/BAD —
    # not a fourth independent good/warn/bad triple like the #2ecc71/
    # #f1c40f pair this used to define on its own.
    _COLOURS = {"PASS": C.GOOD, "WARN": C.WARN, "FAIL": C.BAD}

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(_heading("SELF-DIAGNOSTIC"))
        self.verdict = QLabel("no run yet")
        self.verdict.setObjectName("mutedLabel")
        header.addWidget(self.verdict)
        header.addStretch(1)
        layout.addLayout(header)
        self.checks_label = QLabel("awaiting first self-diagnostic…")
        self.checks_label.setObjectName("mutedLabel")
        self.checks_label.setTextFormat(Qt.TextFormat.RichText)
        self.checks_label.setWordWrap(True)
        layout.addWidget(self.checks_label)
        self.forge_label = QLabel("")
        self.forge_label.setObjectName("mutedLabel")
        self.forge_label.setTextFormat(Qt.TextFormat.RichText)
        self.forge_label.setWordWrap(True)
        layout.addWidget(self.forge_label)
        layout.addStretch(1)

    def render_checks(self, at: str, fails: int, warns: int, checks: list[dict[str, Any]]) -> None:
        verdict = "all healthy" if not fails and not warns else f"{fails} fault(s), {warns} warning(s)"
        self.verdict.setText(f"{verdict} — last run {str(at)[11:19] or at}")
        parts = []
        for c in checks:
            colour = self._COLOURS.get(str(c.get("status")), C.MUTED)
            detail = str(c.get("detail") or "")[:60]
            parts.append(
                f"<span style='color:{colour}'>●</span> "
                f"{c.get('name')} <span style='color:{C.MUTED}'>{detail}</span>"
            )
        self.checks_label.setText("<br>".join(parts) or "no checks reported")

    def render_forge(self, snap: dict[str, Any] | None) -> None:
        if not snap:
            self.forge_label.setText("<i>forge not attached this session</i>")
            return
        lessons = snap.get("lessons") or {}
        quarantine = snap.get("quarantine") or []
        contract_failures = snap.get("contract_failures") or []
        parts = [
            f"<b>FORGE</b>  {snap.get('live', 0)} tool(s) live · "
            f"{snap.get('session_ok', 0)}/{snap.get('session_total', 0)} session(s) ok this run · "
            f"lessons: {lessons.get('lessons', 0)} ({lessons.get('unresolved', 0)} unresolved)"
        ]
        if quarantine:
            colour = self._COLOURS["WARN"]
            names = ", ".join(str(row.get("tool", "?")) for row in quarantine[:5])
            parts.append(f"<span style='color:{colour}'>▲</span> {len(quarantine)} quarantined: {names}")
        if contract_failures:
            colour = self._COLOURS["FAIL"]
            names = ", ".join(str(row.get("name", "?")) for row in contract_failures[:5])
            parts.append(f"<span style='color:{colour}'>●</span> {len(contract_failures)} contract failure(s): {names}")
        if not quarantine and not contract_failures:
            parts.append(f"<span style='color:{self._COLOURS['PASS']}'>●</span> nothing held back")
        self.forge_label.setText("<br>".join(parts))


class DiagnosticsCentreView(QWidget):
    """The OrionDiagnosticsCentre page for the unified Command Deck."""

    SAMPLE_MS = 2000
    _TRAFFIC_WINDOW_S = 60.0

    def __init__(
        self,
        bus: OrionBus,
        telemetry: Any,
        dispatcher: Any | None = None,
        worker: Any | None = None,
    ) -> None:
        super().__init__()
        self.bus = bus
        self.telemetry = telemetry
        self.dispatcher = dispatcher
        self.worker = worker
        self._log_stamps: Deque[float] = deque(maxlen=2000)
        self._event_stamps: Deque[float] = deque(maxlen=2000)
        self._last_diagnostic: dict[str, Any] | None = None
        # Traffic taps — count only; rendering happens on the timer.
        bus.log.connect(lambda _m: self._log_stamps.append(time.monotonic()))
        bus.dashboard_event.connect(self._on_dashboard_event)
        bus.dashboard_event.connect(
            lambda _c, _p: self._event_stamps.append(time.monotonic())
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        title = QLabel("ORION DIAGNOSTICS CENTRE")
        title.setObjectName("titleLabel")
        root.addWidget(title)

        # Provider / connection / degraded-mode / token status strip (Sections
        # 2, 4, 5, 6, 7).  Reacts to bus signals; token counts refreshed on the
        # sample timer below.
        self.status_strip = ProviderStatusStrip(bus)
        root.addWidget(self.status_strip)

        row = QHBoxLayout()
        self.health_panel = _HealthPanel()
        self.resource_panel = _ResourcePanel()
        row.addWidget(self.health_panel, 1)
        row.addWidget(self.resource_panel, 1)
        root.addLayout(row)

        self.tool_panel = _ToolPanel()
        root.addWidget(self.tool_panel, 1)

        self.health_checks_panel = _HealthChecksPanel()
        root.addWidget(self.health_checks_panel)

        self.crash_panel = _CrashPanel()
        root.addWidget(self.crash_panel)

        self._timer = QTimer(self)
        self._timer.setInterval(self.SAMPLE_MS)
        self._timer.timeout.connect(self._sample)
        self._timer.start()

    # ── sampling (GUI thread; dictionary reads only) ──────────────────────────

    def _sample(self) -> None:
        if not self.isVisible():
            return  # deck page not open — skip the work entirely
        try:
            self.health_panel.render(self.telemetry.health.snapshot())
        except Exception:
            pass
        # Refresh the token-usage readout from the router's ledger, if attached.
        try:
            ledger = getattr(getattr(self.worker, "router", None), "token_ledger", None)
            self.status_strip.refresh_tokens(ledger)
        except Exception:
            pass
        cpu = ram = 0.0
        try:
            import psutil
            cpu = float(psutil.cpu_percent(interval=None))
            ram = float(psutil.virtual_memory().percent)
        except Exception:
            pass
        speech: dict[str, Any] = {}
        if self.worker is not None:
            try:
                speech = self.worker.speech.telemetry_snapshot()
            except Exception:
                speech = {}
        gpu = None
        try:
            from .. import gpu_stats
            gpu = gpu_stats.sample()
        except Exception:
            gpu = None
        try:
            self.resource_panel.render(cpu, ram, speech, self._traffic_rates(), gpu)
        except Exception:
            pass
        if self.dispatcher is not None:
            try:
                self.tool_panel.render(
                    int(getattr(self.dispatcher, "active_tools", 0)),
                    list(getattr(self.dispatcher, "recent_tools", [])),
                )
            except Exception:
                pass
        try:
            from .. import crash_reporter
            self.crash_panel.render(crash_reporter.recent(6))
        except Exception:
            pass
        if self._last_diagnostic is not None:
            try:
                d = self._last_diagnostic
                self.health_checks_panel.render_checks(
                    d.get("at", ""), int(d.get("fails", 0)), int(d.get("warns", 0)),
                    list(d.get("checks", [])),
                )
            except Exception:
                pass
        try:
            forge = getattr(self.dispatcher, "forge", None)
            snap = forge.snapshot() if forge is not None else None
            self.health_checks_panel.render_forge(snap)
        except Exception:
            pass

    def _on_dashboard_event(self, category: str, payload: Any) -> None:
        # Cache only — rendering happens on the sample timer like everything
        # else on this page, so a burst of events can't hammer the GUI thread.
        if category == "diagnostics" and isinstance(payload, dict):
            self._last_diagnostic = payload

    def _traffic_rates(self) -> tuple[float, float]:
        cutoff = time.monotonic() - self._TRAFFIC_WINDOW_S
        logs = sum(1 for t in self._log_stamps if t >= cutoff)
        events = sum(1 for t in self._event_stamps if t >= cutoff)
        return (float(logs), float(events))
