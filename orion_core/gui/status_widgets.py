"""
Reusable command-centre status components (Section 7).

Accessible, self-contained widgets that subscribe to the resilience bus signals
(connection_state, provider_degraded, provider_diagnosed) and the token ledger,
rendering the futuristic status surface without any subsystem importing the GUI.

Every widget:
  • uses semantic accessible names (screen-reader friendly);
  • sets a ``state`` property that the stylesheet colours (good/warn/bad/neutral);
  • honours a reduced-motion preference (no blinking/animation when set);
  • renders explicit empty / offline / reconnecting / degraded / error states.

Widgets never poll and never touch a background thread; they react to signals
delivered on the GUI thread.
"""

from __future__ import annotations

import os
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..bus import OrionBus


def reduced_motion() -> bool:
    """True when animations should be suppressed (env flag or OS hint)."""
    return os.getenv("ORION_REDUCED_MOTION", "").strip().lower() in {"1", "true", "yes", "on"}


def _restyle(widget: QWidget, state: str) -> None:
    widget.setProperty("state", state)
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


# Connection-state → (chip state colour, short label).
_CONN_STATE_STYLE = {
    "connected": ("good", "CONNECTED"),
    "local_fallback": ("warn", "LOCAL MODEL"),
    "connecting": ("neutral", "CONNECTING…"),
    "reconnecting": ("warn", "RECONNECTING…"),
    "cooling_down": ("warn", "COOLING DOWN"),
    "rotating_credential": ("warn", "ROTATING KEY"),
    "switching_provider": ("warn", "SWITCHING PROVIDER"),
    "degraded_offline": ("bad", "NO PROVIDER"),
    "shutting_down": ("neutral", "SHUTTING DOWN"),
    "disconnected": ("neutral", "OFFLINE"),
}


class ConnectionStatusChip(QLabel):
    """A single accessible chip reflecting the live connection state."""

    def __init__(self, bus: OrionBus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusChip")
        self.setAccessibleName("Connection status")
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._set("disconnected", "OFFLINE", "The live channel is offline.")
        bus.connection_state.connect(self.on_state)

    def _set(self, state: str, label: str, tooltip: str) -> None:
        self.setText(f"◉ {label}")
        self.setToolTip(tooltip)
        self.setAccessibleDescription(tooltip)
        colour = _CONN_STATE_STYLE.get(state, ("neutral", state.upper()))[0]
        _restyle(self, colour)

    def on_state(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        state = str(payload.get("state") or "disconnected")
        colour, label = _CONN_STATE_STYLE.get(state, ("neutral", state.upper()))
        message = str(payload.get("message") or label)
        self.setText(f"◉ {label}")
        self.setToolTip(message)
        self.setAccessibleDescription(message)
        _restyle(self, colour)


class DegradedBanner(QLabel):
    """A prominent, dismissible-in-spirit banner shown only while ALL text
    providers are unavailable, with the configured recovery alternatives."""

    def __init__(self, bus: OrionBus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("degradedBanner")
        self.setWordWrap(True)
        self.setAccessibleName("Degraded mode warning")
        self.hide()
        bus.provider_degraded.connect(self.on_degraded)

    def on_degraded(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if not payload.get("degraded"):
            self.hide()
            return
        alts = payload.get("alternatives") or []
        text = ("⚠ No AI text provider is currently available. Deterministic tools "
                "still work. ")
        if alts:
            text += "Options: " + "; ".join(str(a) for a in alts[:3]) + "."
        else:
            text += "Configure another API/provider or a local LLM."
        self.setText(text)
        self.setAccessibleDescription(text)
        self.show()


class TokenUsageMini(QWidget):
    """A compact, accessible token-usage readout.  Shows explicit empty and
    'Not reported' states; refreshed on demand from a ledger snapshot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Token usage summary")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        self._labels: dict[str, QLabel] = {}
        for key, caption in (("requests", "Requests"), ("total_tokens", "Tokens"),
                             ("estimated_cost_usd", "Est. cost")):
            block = QVBoxLayout()
            cap = QLabel(caption)
            cap.setObjectName("tokenMetric")
            val = QLabel("—")
            val.setObjectName("tokenMetricValue")
            val.setAccessibleName(f"{caption} value")
            block.addWidget(cap)
            block.addWidget(val)
            row.addLayout(block)
            self._labels[key] = val
        self.update_summary(None)

    @staticmethod
    def _fmt(value: Any, prefix: str = "") -> str:
        if value is None:
            return "Not reported"
        if isinstance(value, (int, float)):
            return f"{prefix}{value:,.0f}" if prefix != "$" else f"${value:,.4f}"
        return str(value)

    def update_summary(self, summary: dict[str, Any] | None) -> None:
        if not summary:
            for lbl in self._labels.values():
                lbl.setText("—")
            return
        self._labels["requests"].setText(self._fmt(summary.get("requests")))
        self._labels["total_tokens"].setText(self._fmt(summary.get("total_tokens")))
        self._labels["estimated_cost_usd"].setText(self._fmt(summary.get("estimated_cost_usd"), "$"))


class ProviderStatusStrip(QFrame):
    """A composed status strip: connection chip + token mini + degraded banner.
    Drop-in for the command centre or diagnostics view."""

    def __init__(self, bus: OrionBus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusStrip")
        self.setAccessibleName("Provider and connection status")
        self.bus = bus
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(12)
        self.conn_chip = ConnectionStatusChip(bus)
        self.diag_chip = QLabel("◉ NOMINAL")
        self.diag_chip.setObjectName("statusChip")
        self.diag_chip.setAccessibleName("Last provider diagnosis")
        _restyle(self.diag_chip, "neutral")
        self.token_mini = TokenUsageMini()
        top.addWidget(self.conn_chip, 0)
        top.addWidget(self.diag_chip, 0)
        top.addStretch(1)
        top.addWidget(self.token_mini, 0)
        outer.addLayout(top)
        self.degraded = DegradedBanner(bus)
        outer.addWidget(self.degraded)
        bus.provider_diagnosed.connect(self.on_diagnosis)

    def on_diagnosis(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        category = str(payload.get("category") or "unknown")
        provider = str(payload.get("provider") or "")
        remediation = str(payload.get("remediation") or "")
        colour = "good" if category == "ok" else "bad"
        # The label itself must carry enough context to be useful at a glance —
        # a bare category name ("QUOTA EXHAUSTED") with the provider and the
        # actual remediation hidden behind a hover-only tooltip left the label
        # looking identical on every occurrence, with no visible way to tell
        # which provider or what to do about it.
        label = f"◉ {category.replace('_', ' ').upper()}"
        if provider:
            label += f" ({provider})"
        self.diag_chip.setText(label)
        tip = f"{provider}: {remediation}" if provider else remediation
        self.diag_chip.setToolTip(tip)
        self.diag_chip.setAccessibleDescription(tip)
        _restyle(self.diag_chip, colour)

    def refresh_tokens(self, ledger: Any) -> None:
        """Pull a fresh usage summary from the token ledger, if attached."""
        if ledger is None:
            self.token_mini.update_summary(None)
            return
        try:
            self.token_mini.update_summary(ledger.summary())
        except Exception:
            self.token_mini.update_summary(None)
