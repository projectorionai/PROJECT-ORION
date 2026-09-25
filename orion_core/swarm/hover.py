"""
hover.py — diagnostics that unfold from a node under the pointer (Mark XXIII).

Hovering a subsystem should open a small holographic readout beside it: what
it is doing, how hard it is working, whether it is healthy. That is a
different job from the inspector panel, and the difference drives every
decision here.

The inspector answers "tell me everything about this" — it is opened
deliberately, read carefully, and can afford fifteen rows. A hover readout
answers "what is this?" while the pointer is still moving. It must be
glanceable, so it carries the FEW numbers that matter for that kind of node,
and it must not flicker, so it is gated on dwell rather than on the pointer
merely passing over something.

Three behaviours are load-bearing:

  * DWELL. Sweeping across a dense field crosses dozens of nodes. Opening a
    panel for each would strobe the whole interface. Nothing opens until the
    pointer has rested.

  * WARM SWAP. Once a panel is open, moving to a neighbouring node swaps
    almost immediately — having already asked for detail, the user should not
    re-serve the delay at every node. This is the standard tooltip warm-up
    behaviour, and without it comparing two nodes feels broken.

  * HONEST ATTRIBUTION. Host CPU/GPU is shown on ORION himself and on cluster
    anchors, never on an individual subsystem. ORION cannot measure per-module
    CPU, so printing the machine's 38% next to `module:vision` would invent an
    attribution that does not exist. Same rule as inspect.py's em dash: an
    unmeasured thing is never rendered as a measurement.

Pure Python — no Qt, no GL, injectable clock. The renderer decides how the
panel LOOKS; this decides what it says and how far open it is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .inspect import ABSENT, _count, _ms, _seconds
from .model import Activity, Health, NodeKind, SwarmNode

# How long the pointer must rest before anything opens.
HOVER_DELAY_S = 0.18

# Unfold and fold durations. Folding is faster than unfolding on purpose: a
# panel that lingers after the pointer has left reads as a stuck interface,
# whereas one that opens briskly reads as eager.
UNFOLD_S = 0.22
FOLD_S = 0.13

# After a panel closes, this long remains "warm" — the next hover skips the
# dwell delay and opens fast, so comparing adjacent nodes is fluid.
WARM_S = 0.9

# Fraction of the unfold each successive row lags behind the one above it.
# This is what makes the panel unfold rather than fade: rows arrive in
# sequence, like a readout printing.
ROW_STAGGER = 0.055

# Total stagger is capped so a long panel's last row is not still arriving
# after the animation has notionally finished.
_MAX_TOTAL_STAGGER = 0.55


def _ease(t: float) -> float:
    """Smoothstep. Linear openings look mechanical; this settles."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


def _percent(value: float | None) -> str:
    return ABSENT if value is None else f"{value:.0f}%"


@dataclass(frozen=True, slots=True)
class HoverRow:
    """One line of the readout, with its own arrival progress."""

    label: str
    value: str
    severity: str = "normal"
    reveal: float = 1.0        # 0..1, how far this row has arrived


@dataclass(frozen=True, slots=True)
class HoverPanel:
    """A readout unfolding from one node."""

    node_id: str
    title: str
    subtitle: str
    rows: tuple[HoverRow, ...]
    unfold: float              # 0..1 overall
    alarm: bool                # something on this node needs attention

    @property
    def is_open(self) -> bool:
        return self.unfold > 0.0

    @property
    def visible_rows(self) -> tuple[HoverRow, ...]:
        """Only the rows that have actually started arriving.

        The renderer sizes the panel from this, so the frame grows with the
        content instead of an empty box appearing and then filling."""
        return tuple(r for r in self.rows if r.reveal > 0.0)

    def as_text(self) -> str:
        lines = [f"{self.title} · {self.subtitle}"]
        lines += [f"{r.label}: {r.value}" for r in self.rows]
        return "\n".join(lines)


# ── what each kind of node is worth saying ───────────────────────────────────

def _host_rows(load: Any) -> list[HoverRow]:
    """Machine-level load. Only ever attached to nodes that legitimately own
    it — see the module docstring on attribution."""
    if load is None:
        return []
    cpu = getattr(load, "cpu_percent", None)
    gpu = getattr(load, "gpu_percent", None)
    mem = getattr(load, "mem_percent", None)
    own = getattr(load, "self_cpu_percent", None)
    rows = [
        HoverRow("CPU", _percent(cpu), "alarm" if (cpu or 0) >= 90 else "normal"),
        HoverRow("GPU", _percent(gpu)),
        HoverRow("Memory", _percent(mem), "alarm" if (mem or 0) >= 92 else "normal"),
    ]
    if own is not None:
        rows.append(HoverRow("ORION process", _percent(own)))
    return rows


def _latency_row(node: SwarmNode) -> HoverRow:
    """p95, not the mean.

    An average hides the tail, and the tail is what the user actually feels as
    "ORION went quiet for a second"."""
    return HoverRow("Latency p95", _ms(node.telemetry.latency_p95_ms))


def _failure_row(node: SwarmNode) -> HoverRow:
    telemetry = node.telemetry
    if telemetry.calls <= 0:
        # Never called is not the same as called and never failed.
        return HoverRow("Failures", ABSENT)
    rate = telemetry.error_rate
    return HoverRow("Failures", f"{telemetry.failures:,} ({rate * 100:.0f}%)",
                    "alarm" if rate >= 0.25 else "normal")


def diagnostic_rows(node: SwarmNode, load: Any = None) -> list[HoverRow]:
    """The few numbers worth showing for THIS kind of node.

    Deliberately not the inspector's full list — a hover readout the size of
    the inspector is just the inspector, arriving uninvited."""
    telemetry = node.telemetry
    rows: list[HoverRow] = []

    alarm = node.health in (Health.DEGRADED, Health.DOWN)
    if node.health is not Health.OK or alarm:
        rows.append(HoverRow("Health", node.health.value,
                             "alarm" if alarm else "normal"))

    action = telemetry.current_action.strip()
    if action:
        rows.append(HoverRow("Doing", action))

    if node.kind is NodeKind.CORE or node.is_cluster:
        # ORION himself and the twelve cluster anchors are the only nodes that
        # can honestly claim the machine's load.
        rows.append(HoverRow("Tool calls", _count(telemetry.calls)))
        rows.append(_failure_row(node))
        rows.extend(_host_rows(load))
        return rows

    if node.kind is NodeKind.AGENT:
        rows.append(HoverRow("Invocations", _count(telemetry.calls)))
        rows.append(HoverRow("Last active", _seconds(telemetry.heartbeat_age_s)))
        rows.append(_latency_row(node))
        rows.append(_failure_row(node))
        return rows

    if node.kind in (NodeKind.MCP_SERVER, NodeKind.MCP_TOOL):
        if node.kind is NodeKind.MCP_SERVER:
            rows.append(HoverRow("Tools", _count(len(node.capabilities))))
        rows.append(HoverRow("Requests", _count(telemetry.calls)))
        rows.append(_latency_row(node))
        rows.append(_failure_row(node))
        return rows

    if node.kind is NodeKind.MEMORY_TIER:
        # A memory tier has rows and a recency band, and no latency at all —
        # printing "Latency —" four times would be noise, not information.
        from .memory_field import band_for
        band = band_for(node.label)
        rows.append(HoverRow("Stored", _count(telemetry.queue_depth)))
        rows.append(HoverRow("Recall",
                             "immediate" if band <= 0.7
                             else ("cold storage" if band >= 1.7 else "recallable")))
        return rows

    if node.kind is NodeKind.WORKFLOW:
        rows.append(HoverRow("Steps", _count(len(node.capabilities))))
        rows.append(HoverRow("Runs", _count(telemetry.calls)))
        return rows

    if node.kind is NodeKind.PAGE:
        rows.append(HoverRow("Zone", node.cluster.title()))
        rows.append(HoverRow("Open", "double-click"))
        return rows

    # Subsystems and anything new.
    rows.append(HoverRow("Requests", _count(telemetry.calls)))
    rows.append(_failure_row(node))
    rows.append(_latency_row(node))
    if telemetry.queue_depth:
        rows.append(HoverRow("Queued", _count(telemetry.queue_depth)))
    return rows


def _subtitle(node: SwarmNode) -> str:
    from .inspect import _ACTIVITY_LABELS, _KIND_LABELS
    kind = _KIND_LABELS.get(node.kind, node.kind.value)
    activity = _ACTIVITY_LABELS.get(node.activity, node.activity.value)
    return f"{kind} · {activity}"


def _stagger_rows(rows: list[HoverRow], unfold: float) -> tuple[HoverRow, ...]:
    """Give each row its own arrival, lagging the one above.

    A single alpha on the whole panel is a fade; per-row arrival is an unfold,
    and the difference is what makes it read as an instrument rather than a
    tooltip."""
    if not rows:
        return ()
    stagger = ROW_STAGGER
    if len(rows) > 1:
        stagger = min(stagger, _MAX_TOTAL_STAGGER / (len(rows) - 1))
    span = max(1e-6, 1.0 - stagger * (len(rows) - 1))
    out = []
    for index, row in enumerate(rows):
        progress = _ease((unfold - stagger * index) / span)
        out.append(HoverRow(row.label, row.value, row.severity, progress))
    return tuple(out)


class HoverController:
    """Tracks what the pointer is over and how far its readout has opened.

    A state machine rather than a timer: `panel()` derives everything from
    elapsed time whenever it is asked, so an unhovered field costs nothing and
    the answer is correct without having been ticked at any particular rate.
    """

    __slots__ = ("_target", "_since", "_open", "_open_at", "_closed_at", "_clock",
                 "_delay", "_unfold_s", "_fold_s")

    def __init__(self, clock: Callable[[], float] | None = None,
                 delay_s: float = HOVER_DELAY_S,
                 unfold_s: float = UNFOLD_S,
                 fold_s: float = FOLD_S) -> None:
        self._clock = clock if clock is not None else time.monotonic
        self._target: str | None = None      # what the pointer is over
        self._since = 0.0                    # when it arrived there
        self._open: str | None = None        # what is actually unfolding
        self._open_at = 0.0
        self._closed_at = -1e9
        self._delay = max(0.0, float(delay_s))
        self._unfold_s = max(1e-3, float(unfold_s))
        self._fold_s = max(1e-3, float(fold_s))

    # ── input ────────────────────────────────────────────────────────────────

    def point_at(self, node_id: str | None, now: float | None = None) -> bool:
        """The pointer is over *node_id* (None = over nothing).

        Returns True when this changed the target, so a caller can repaint
        only when something actually moved."""
        now = self._clock() if now is None else float(now)
        node_id = node_id or None
        if node_id == self._target:
            return False
        # Leaving the open node starts its fold; the open panel is kept so it
        # can collapse rather than vanish.
        if self._open is not None and node_id != self._open:
            self._closed_at = now
            self._open = None
        self._target = node_id
        self._since = now
        return True

    def clear(self, now: float | None = None) -> bool:
        return self.point_at(None, now)

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def target(self) -> str | None:
        return self._target

    def is_warm(self, now: float | None = None) -> bool:
        """True while a recently-closed panel still shortcuts the delay."""
        now = self._clock() if now is None else float(now)
        return (now - self._closed_at) <= WARM_S

    def _promote(self, now: float) -> None:
        """Open the pending target once it has been dwelt on long enough."""
        if self._target is None or self._open == self._target:
            return
        delay = 0.0 if self.is_warm(now) else self._delay
        # The unfold is dated from when the dwell was SATISFIED, not from when
        # something happened to ask. Dating it from `now` would make the panel
        # open late whenever frames are sparse — the animation would run at
        # the polling rate rather than in real time, which is exactly the
        # coupling this model exists to avoid.
        ready_at = self._since + delay
        if now >= ready_at:
            self._open = self._target
            self._open_at = ready_at

    def unfold(self, now: float | None = None) -> float:
        """How far open the readout is, 0..1."""
        now = self._clock() if now is None else float(now)
        self._promote(now)
        if self._open is not None:
            return _ease((now - self._open_at) / self._unfold_s)
        # Folding away.
        elapsed = now - self._closed_at
        if elapsed >= self._fold_s:
            return 0.0
        return _ease(1.0 - elapsed / self._fold_s)

    def open_node(self, now: float | None = None) -> str | None:
        """Which node's readout is currently being drawn — which is not always
        the hovered one, since a panel keeps drawing while it folds away."""
        now = self._clock() if now is None else float(now)
        self._promote(now)
        if self._open is not None:
            return self._open
        return None

    # ── the rendering answer ─────────────────────────────────────────────────

    def panel(self, node: SwarmNode | None, load: Any = None,
              now: float | None = None) -> HoverPanel | None:
        """The readout to draw, or None when nothing should be shown.

        *node* is the node identified by `open_node()`; the caller resolves it,
        because this module deliberately holds no reference to the graph."""
        now = self._clock() if now is None else float(now)
        progress = self.unfold(now)
        if progress <= 0.0 or node is None:
            return None
        rows = diagnostic_rows(node, load)
        alarm = (node.health in (Health.DEGRADED, Health.DOWN)
                 or node.activity is Activity.ERROR
                 or any(r.severity == "alarm" for r in rows))
        return HoverPanel(
            node_id=node.id,
            title=node.label,
            subtitle=_subtitle(node),
            rows=_stagger_rows(rows, progress),
            unfold=progress,
            alarm=alarm,
        )

    def reset(self) -> None:
        self._target = None
        self._open = None
        self._closed_at = -1e9
