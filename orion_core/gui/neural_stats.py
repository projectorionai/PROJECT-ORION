"""
NEURAL - ORION's stats as graphs (Mark XXXX.4).

The owner is a neural engineer and asked to SEE how ORION is doing, drawn the
way the BRAIN page draws where his thoughts go: as a graph. This page is that
view, built only from what ORION already measures:

    signal flow   every turn as it travels: input -> hearing -> thinking ->
                  tools -> reply. Edge widths are real counts, node rings
                  are success rates, and a pulse runs along an edge the
                  moment a live turn reaches that stage.
    response      the last turns as stacked bars - hearing, thinking, tools
                  and voice start - with the p50 and p95 lines (request_trace).
    benchmark     every self-benchmark run, per category (self_benchmark).
    providers     requests and tokens in 24 hours, latency, last failure
                  (the router's token ledger and latency average).
    tools         calls, failures and median time per tool (telemetry).

Everything drawn happened; nothing is estimated. ``collect`` is a pure
reducer over those sources, so the numbers are testable without Qt. The page
refreshes only while it is visible, the ledger is read at most every 15 s,
and the pulse animation runs only while a pulse is in flight - the deck often
shares the GPU with the face on a second monitor.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..constants import C

_UI_FAMILY = "Segoe UI Variable Display"
_MONO_FAMILY = "Cascadia Mono"

#: Which latency group each trace stage's time belongs to, up to the reply.
STAGE_GROUPS: dict[str, str] = {
    "CAPTURE_START": "hear", "CAPTURE_COMPLETE": "hear", "TRANSCRIPTION_START": "hear",
    "TRANSCRIPTION_COMPLETE": "hear", "COMMAND_RECEIVED": "hear", "AGENT_SELECTED": "hear",
    "API_REQUEST_START": "think", "API_RESPONSE_RECEIVED": "think",
    "TOOL_EXECUTION_START": "tools", "TOOL_EXECUTION_COMPLETE": "tools",
    "TTS_START": "voice", "PLAYBACK_START": "voice", "REPLY_DELIVERED": "voice",
}
GROUPS = ("hear", "think", "tools", "voice")
GROUP_LABELS = {"hear": "Hear", "think": "Think", "tools": "Tools", "voice": "Voice Start"}
#: The stage at which the owner has his answer.
RESPONSE_STAGES = frozenset({"PLAYBACK_START", "REPLY_DELIVERED"})
ORIGINS = {"voice": "Voice", "text": "Typed", "typed": "Typed", "chat": "Typed",
           "remote": "Phone", "phone": "Phone", "telegram": "Phone"}
MAX_TURNS = 32
#: Tiles about ORION's own process, read live (not from the reducer).
PROCESS_TILES = ("ORION RAM", "ORION CPU")
MAX_TOOLS = 7
MAX_PROVIDERS = 6
LEDGER_EVERY_S = 15.0
BENCH_EVERY_S = 60.0


# ── the numbers (pure) ───────────────────────────────────────────────────────

@dataclass
class Turn:
    request_id: str
    origin: str
    segments: dict[str, float]
    response_ms: float | None
    speak_ms: float
    complete: bool
    failed: bool
    stages: tuple[str, ...]


@dataclass
class ToolStat:
    name: str
    calls: int
    failures: int
    p50_ms: float | None


@dataclass
class ProviderStat:
    name: str
    requests: int
    tokens: int
    latency_s: float | None
    failing: bool


@dataclass
class FlowNode:
    key: str
    label: str
    layer: int
    value: float
    detail: str = ""
    health: float | None = None          # 0..1 success, None when not measured


@dataclass
class FlowEdge:
    src: str
    dst: str
    weight: float


@dataclass
class NeuralSnapshot:
    turns: list[Turn] = field(default_factory=list)
    tiles: list[tuple[str, str]] = field(default_factory=list)
    tools: list[ToolStat] = field(default_factory=list)
    providers: list[ProviderStat] = field(default_factory=list)
    bench: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[FlowNode] = field(default_factory=list)
    edges: list[FlowEdge] = field(default_factory=list)
    p50_ms: float | None = None
    p95_ms: float | None = None


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def origin_label(origin: str) -> str:
    return ORIGINS.get(str(origin or "").lower().split(":")[0], "Other")


def turn_from_trace(trace: Any) -> Turn:
    """One request trace as latency groups, up to the first word of the reply."""
    segments = {group: 0.0 for group in GROUPS}
    response_ms: float | None = None
    speak_ms = 0.0
    failed = bool(getattr(trace, "failed_stage", ""))
    elapsed = 0.0
    stages: list[str] = []
    for stage, ms in trace.stage_timings():
        stage = str(stage)
        stages.append(stage)
        if stage.startswith("FAILED::"):
            failed = True
            continue
        elapsed += max(0.0, float(ms))
        if response_ms is None:
            group = STAGE_GROUPS.get(stage)
            if group:
                segments[group] += max(0.0, float(ms))
            if stage in RESPONSE_STAGES:
                response_ms = elapsed
        elif stage == "PLAYBACK_COMPLETE":
            speak_ms += max(0.0, float(ms))
    return Turn(str(getattr(trace, "request_id", "")), origin_label(getattr(trace, "origin", "")),
                segments, response_ms, speak_ms, bool(getattr(trace, "complete", False)),
                failed, tuple(stages))


def tool_stats(telemetry: Any, top: int = MAX_TOOLS) -> list[ToolStat]:
    if telemetry is None:
        return []
    try:
        usage = telemetry.tool_usage(top=top)
        timers = telemetry.metrics.snapshot().get("timers", {})
    except Exception:
        return []
    out = []
    for row in usage:
        name = str(row.get("tool") or "")
        timer = timers.get(f"tool.{name}.ms") or {}
        out.append(ToolStat(name, int(row.get("calls") or 0), int(row.get("failures") or 0),
                            timer.get("p50_ms")))
    return out


def provider_stats(router: Any, ledger_rows: list[dict[str, Any]] | None,
                   top: int = MAX_PROVIDERS) -> list[ProviderStat]:
    latency = dict(getattr(router, "_latency_ema", {}) or {}) if router is not None else {}
    failures = dict(getattr(router, "_failures", {}) or {}) if router is not None else {}
    rows: dict[str, ProviderStat] = {}
    for row in ledger_rows or []:
        name = str(row.get("key") or "unknown")
        rows[name] = ProviderStat(name, int(row.get("requests") or 0),
                                  int(row.get("total_tokens") or 0),
                                  latency.get(name), name in failures)
    for name, seconds in latency.items():
        rows.setdefault(name, ProviderStat(name, 0, 0, seconds, name in failures))
    ranked = sorted(rows.values(), key=lambda p: (-p.requests, -(p.tokens or 0), p.name))
    return ranked[:top]


def _format_ms(ms: float | None) -> str:
    if ms is None:
        return "–"
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def _flow(turns: list[Turn], tools: list[ToolStat]) -> tuple[list[FlowNode], list[FlowEdge]]:
    nodes: list[FlowNode] = []
    edges: list[FlowEdge] = []
    origins: dict[str, int] = {}
    for turn in turns:
        origins[turn.origin] = origins.get(turn.origin, 0) + 1
    for label in ("Voice", "Typed", "Phone", "Other"):
        if origins.get(label) or (label == "Voice" and not origins):
            nodes.append(FlowNode(f"in:{label}", label, 0, origins.get(label, 0)))
            edges.append(FlowEdge(f"in:{label}", "hear", origins.get(label, 0)))
    heard = [t for t in turns if t.segments["hear"] or "TRANSCRIPTION_COMPLETE" in t.stages]
    hear_ms = percentile([t.segments["hear"] for t in heard], 0.5)
    nodes.append(FlowNode("hear", "Hear", 1, len(turns), _format_ms(hear_ms)))
    thought = [t for t in turns if "API_RESPONSE_RECEIVED" in t.stages or t.response_ms is not None]
    think_ms = percentile([t.segments["think"] for t in thought], 0.5)
    done = sum(1 for t in turns if t.complete)
    nodes.append(FlowNode("think", "Think", 2, len(thought), _format_ms(think_ms),
                          done / len(turns) if turns else None))
    edges.append(FlowEdge("hear", "think", len(thought)))
    for tool in tools:
        key = f"tool:{tool.name}"
        health = 1.0 - tool.failures / tool.calls if tool.calls else None
        nodes.append(FlowNode(key, tool.name.replace("_", " "), 3, tool.calls,
                              _format_ms(tool.p50_ms), health))
        edges.append(FlowEdge("think", key, tool.calls))
    spoken = sum(1 for t in turns if "PLAYBACK_START" in t.stages)
    written = sum(1 for t in turns if "REPLY_DELIVERED" in t.stages and "PLAYBACK_START" not in t.stages)
    nodes.append(FlowNode("out:spoken", "Spoken", 4, spoken))
    nodes.append(FlowNode("out:written", "Written", 4, written))
    edges.append(FlowEdge("think", "out:spoken", spoken))
    edges.append(FlowEdge("think", "out:written", written))
    return nodes, edges


def collect(traces: Iterable[Any], *, telemetry: Any = None, router: Any = None,
            ledger_rows: list[dict[str, Any]] | None = None,
            bench: list[dict[str, Any]] | None = None) -> NeuralSnapshot:
    """Everything the page draws, from ORION's own measurements."""
    all_traces = list(traces)
    turns = [turn_from_trace(t) for t in all_traces][-MAX_TURNS:]
    tools = tool_stats(telemetry)
    providers = provider_stats(router, ledger_rows)
    responses = [t.response_ms for t in turns if t.response_ms is not None]
    p50, p95 = percentile(responses, 0.5), percentile(responses, 0.95)
    complete = sum(1 for t in all_traces if getattr(t, "complete", False))
    failed = sum(1 for t in all_traces if getattr(t, "failed_stage", ""))
    calls = failures = 0
    if telemetry is not None:
        try:
            counters = telemetry.metrics.snapshot().get("counters", {})
            calls = int(counters.get("tool.calls", 0))
            failures = int(counters.get("tool.failures", 0))
        except Exception:
            pass
    latest = (bench or [])[-1] if bench else None
    tiles = [
        ("Turns", f"{len(all_traces)}"),
        ("Completed", f"{100 * complete / len(all_traces):.0f}%" if all_traces else "–"),
        ("Failed", f"{failed}"),
        ("Response p50", _format_ms(p50)),
        ("Response p95", _format_ms(p95)),
        ("Tool Calls", f"{calls}"),
        ("Tool Failures", f"{100 * failures / calls:.0f}%" if calls else "–"),
        ("Benchmark", f"{float(latest.get('overall') or 0):.0f}%" if latest else "–"),
    ]
    nodes, edges = _flow(turns, tools)
    return NeuralSnapshot(turns, tiles, tools, providers, list(bench or []), nodes, edges, p50, p95)


class ResearchActivity:
    """Research running now, kept only from real start and finish events.

    "Co-ordinate with the brain and the neural network of ORION, only if he's
    ACTUALLY researching" (owner, 30 Oct 2026). Fed by the ("research", ...)
    dashboard events every engine sends through research_runs.ActivitySignal
    - its start, its current question and exactly one finish (done, draft,
    failed or cancelled) - so nothing here can say research is running when
    it is not, and a finish always clears it. Pure, so testable without Qt."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, str]] = {}

    def apply(self, payload: Any) -> bool:
        """Take one event; True when what is shown changed."""
        if not isinstance(payload, dict):
            return False
        key = str(payload.get("run") or payload.get("topic") or "").strip()
        if not key:
            return False
        if str(payload.get("status") or "").lower() != "running":
            return self.runs.pop(key, None) is not None
        row = self.runs.setdefault(key, {"topic": "", "question": "", "engine": ""})
        before = dict(row)
        for name in row:
            value = " ".join(str(payload.get(name) or "").split())
            if value:
                row[name] = value[:160]
        return row != before or not any(before.values())

    def active(self) -> bool:
        return bool(self.runs)

    def line(self) -> str:
        """"Researching <topic> · <question>", or "" when nothing runs."""
        if not self.runs:
            return ""
        row = list(self.runs.values())[-1]
        text = f"Researching {row['topic'] or 'a topic'}"
        if row["question"] and row["question"] != row["topic"]:
            text += f"  ·  {row['question']}"
        if len(self.runs) > 1:
            text += f"  ·  +{len(self.runs) - 1} more"
        return text

    def caption(self) -> str:
        """What the radar prints: the current question, else the topic."""
        if not self.runs:
            return ""
        row = list(self.runs.values())[-1]
        return row["question"] or row["topic"] or "Researching"


# ── drawing ──────────────────────────────────────────────────────────────────

def _colour(hex_colour: str, alpha: float = 1.0) -> QColor:
    colour = QColor(hex_colour)
    colour.setAlphaF(max(0.0, min(1.0, alpha)))
    return colour


def _font(size: float, *, mono: bool = False, bold: bool = False) -> QFont:
    font = QFont(_MONO_FAMILY if mono else _UI_FAMILY)
    font.setPointSizeF(size)
    if bold:
        font.setWeight(QFont.Weight.DemiBold)
    return font


GROUP_COLOURS = {"hear": C.ACCENT_DIM, "think": C.PRI, "tools": C.PRI_DIM, "voice": C.SILVER}
BENCH_STYLES = {
    "reasoning": (C.PRI, Qt.PenStyle.SolidLine),
    "coding": (C.CORE, Qt.PenStyle.DashLine),
    "instructions": (C.SILVER, Qt.PenStyle.SolidLine),
    "agentic": (C.MUTED, Qt.PenStyle.DashDotLine),
}


class _Panel(QWidget):
    """A titled, painted instrument. Subclasses draw inside ``body``."""

    title = ""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.snapshot = NeuralSnapshot()

    def set_snapshot(self, snapshot: NeuralSnapshot) -> None:
        self.snapshot = snapshot
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        if self.width() < 40 or self.height() < 40:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.setPen(QPen(_colour(C.BORDER), 1))
            painter.setBrush(_colour(C.PANEL))
            painter.drawRoundedRect(rect, 10, 10)
            painter.setPen(_colour(C.MUTED))
            painter.setFont(_font(8.5, bold=True))
            painter.drawText(rect.adjusted(14, 8, -14, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                             self.title.upper())
            body = rect.adjusted(14, 30, -14, -12)
            if body.width() > 20 and body.height() > 20:
                self.draw(painter, body)
        finally:
            painter.end()

    def draw(self, painter: QPainter, body: QRectF) -> None:
        raise NotImplementedError

    @staticmethod
    def empty(painter: QPainter, body: QRectF, text: str) -> None:
        painter.setPen(_colour(C.FAINT))
        painter.setFont(_font(9))
        painter.drawText(body, Qt.AlignmentFlag.AlignCenter, text)


class SignalFlowGraph(_Panel):
    """input -> hear -> think -> tools -> reply, drawn as a neural network.

    Every node is wired faintly to every node of the next layer (the network),
    the paths his turns really took are drawn over it weighted by their counts,
    and signal flows along those paths while the page is on screen.

    Cost (Mark XXXX.4, owner: "make sure it doesn't affect ORION's performance"):
    the network, the paths and the labels are painted once into a pixmap
    whenever the numbers change (every 1.5 s at most); each animation frame
    only draws that pixmap and a few dozen dots, at 15 frames a second, and
    only while the page is visible."""

    title = "Signal Flow"
    PULSE_S = 0.9
    #: Ambient frame interval while visible (ms).
    FRAME_MS = 66

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)
        self._pulses: list[tuple[str, str, float]] = []
        self._heat: dict[str, float] = {}
        self._positions: dict[str, QPointF] = {}
        self._static: Any = None
        self._static_key: Any = None
        self._radii: dict[str, float] = {}

    def set_snapshot(self, snapshot: NeuralSnapshot) -> None:
        self._static = None                 # the numbers changed: repaint the network once
        super().set_snapshot(snapshot)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        self._static = None
        super().resizeEvent(event)

    def pulse(self, src: str, dst: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._pulses.append((src, dst, now))
        self._heat[dst] = now
        self._pulses = self._pulses[-24:]
        self.update()

    def animating(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        self._pulses = [p for p in self._pulses if now - p[2] < self.PULSE_S]
        return bool(self._pulses) or any(now - t < 1.5 for t in self._heat.values())

    def _layout(self, body: QRectF) -> dict[str, QPointF]:
        columns = (0.06, 0.26, 0.47, 0.72, 0.94)
        layers: dict[int, list[FlowNode]] = {}
        for node in self.snapshot.nodes:
            layers.setdefault(node.layer, []).append(node)
        positions = {}
        for layer, nodes in layers.items():
            count = len(nodes)
            for index, node in enumerate(nodes):
                y = body.top() + body.height() * (index + 1) / (count + 1)
                positions[node.key] = QPointF(body.left() + body.width() * columns[layer], y)
        return positions

    def draw(self, painter: QPainter, body: QRectF) -> None:
        snap = self.snapshot
        if not snap.nodes:
            self.empty(painter, body, "No turns yet")
            return
        now = time.monotonic()
        key = (round(body.width()), round(body.height()))
        if self._static is None or self._static_key != key:
            self._positions = self._layout(body)
            self._static = self._render_static(body)
            self._static_key = key
        pos = self._positions
        painter.drawPixmap(body.topLeft(), self._static)
        top_weight = max([e.weight for e in snap.edges] + [1.0])
        painter.setPen(Qt.PenStyle.NoPen)
        # Signal flowing along the paths his turns really took.
        for index, edge in enumerate(snap.edges):
            a, b = pos.get(edge.src), pos.get(edge.dst)
            if a is None or b is None or not edge.weight:
                continue
            share = edge.weight / top_weight
            curve = self._curve(a, b)
            dots = 1 + int(round(3 * share))
            speed = 0.18 + 0.32 * share
            for k in range(dots):
                t = (now * speed + k / dots + index * 0.137) % 1.0
                if not 0.05 < t < 0.95:
                    continue
                point = curve.pointAtPercent(t)
                painter.setBrush(_colour(C.PRI_HI, 0.16))
                painter.drawEllipse(point, 4.5, 4.5)
                painter.setBrush(_colour(C.CORE, 0.85))
                painter.drawEllipse(point, 1.8, 1.8)
        # Live pulses: a turn reaching a stage right now.
        for src, dst, started in self._pulses:
            a, b = pos.get(src), pos.get(dst)
            if a is None or b is None:
                continue
            t = (now - started) / self.PULSE_S
            if not 0 <= t <= 1:
                continue
            point = self._curve(a, b).pointAtPercent(t)
            for radius, alpha in ((9, 0.18), (5, 0.45), (2.6, 1.0)):
                painter.setBrush(_colour(C.CORE, alpha))
                painter.drawEllipse(point, radius, radius)
        # A node that just fired glows around its ring.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for node_key, fired in self._heat.items():
            heat = max(0.0, 1.0 - (now - fired) / 1.5)
            centre, radius = pos.get(node_key), self._radii.get(node_key)
            if heat and centre is not None and radius:
                painter.setPen(QPen(_colour(C.PRI, 0.45 * heat), 5))
                painter.drawEllipse(centre, radius + 7, radius + 7)

    def _render_static(self, body: QRectF) -> Any:
        """The network, the weighted paths, the nodes and their labels."""
        from PyQt6.QtGui import QPixmap
        ratio = max(1.0, self.devicePixelRatioF())
        pixmap = QPixmap(max(1, int(body.width() * ratio)), max(1, int(body.height() * ratio)))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.translate(-body.left(), -body.top())
            self._paint_network(painter)
        finally:
            painter.end()
        return pixmap

    def _paint_network(self, painter: QPainter) -> None:
        snap, pos = self.snapshot, self._positions
        top_weight = max([e.weight for e in snap.edges] + [1.0])
        top_value = max([n.value for n in snap.nodes] + [1.0])
        # The network: every node to every node of the next layer, faintly.
        layers: dict[int, list[FlowNode]] = {}
        for node in snap.nodes:
            layers.setdefault(node.layer, []).append(node)
        painter.setPen(QPen(_colour(C.ACCENT_DEEP, 0.55), 0.7))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for layer in sorted(layers):
            for a in layers[layer]:
                for b in layers.get(layer + 1, ()):
                    painter.drawPath(self._curve(pos[a.key], pos[b.key]))
        # The paths his turns really took, weighted by how often.
        for edge in snap.edges:
            a, b = pos.get(edge.src), pos.get(edge.dst)
            if a is None or b is None:
                continue
            width = 0.8 + 6.0 * math.sqrt(edge.weight / top_weight) if edge.weight else 0.6
            alpha = 0.18 + 0.5 * (edge.weight / top_weight)
            painter.setPen(QPen(_colour(C.PRI if edge.weight else C.ACCENT_DEEP, alpha), width,
                                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(self._curve(a, b))
        self._radii = {}
        for node in snap.nodes:
            centre = pos[node.key]
            radius = 7.0 + 13.0 * math.sqrt(node.value / top_value) if node.value else 6.0
            self._radii[node.key] = radius
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_colour(C.PRI, 0.10 if node.value else 0.0))
            painter.drawEllipse(centre, radius + 5, radius + 5)
            painter.setPen(QPen(_colour(C.ACCENT_DIM), 1.2))
            painter.setBrush(_colour(C.INK))
            painter.drawEllipse(centre, radius, radius)
            if node.health is not None:
                ring = QRectF(centre.x() - radius - 3, centre.y() - radius - 3,
                              2 * radius + 6, 2 * radius + 6)
                painter.setPen(QPen(_colour(C.BAD if node.health < 0.8 else C.PRI), 2.4,
                                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawArc(ring, 90 * 16, -int(360 * 16 * max(0.0, min(1.0, node.health))))
            painter.setPen(_colour(C.WHITE))
            painter.setFont(_font(8.5, mono=True, bold=True))
            painter.drawText(QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius),
                             Qt.AlignmentFlag.AlignCenter, f"{node.value:.0f}")
            if node.layer == 3:
                # The tool column is stacked tight: its labels go to the side.
                painter.setPen(_colour(C.SILVER))
                painter.setFont(_font(8.5))
                side = QRectF(centre.x() + radius + 7, centre.y() - 8, 150, 16)
                painter.drawText(side, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 node.label)
                if node.detail:
                    painter.setPen(_colour(C.FAINT))
                    painter.setFont(_font(7.5, mono=True))
                    painter.drawText(side.translated(0, 12), Qt.AlignmentFlag.AlignLeft
                                     | Qt.AlignmentFlag.AlignVCenter, node.detail)
                continue
            painter.setPen(_colour(C.SILVER))
            painter.setFont(_font(8.5))
            label = QRectF(centre.x() - 70, centre.y() + radius + 3, 140, 14)
            painter.drawText(label, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, node.label)
            if node.detail:
                painter.setPen(_colour(C.FAINT))
                painter.setFont(_font(7.5, mono=True))
                painter.drawText(label.translated(0, 13), Qt.AlignmentFlag.AlignHCenter
                                 | Qt.AlignmentFlag.AlignTop, node.detail)

    @staticmethod
    def _curve(a: QPointF, b: QPointF) -> QPainterPath:
        path = QPainterPath(a)
        dx = (b.x() - a.x()) * 0.5
        path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
        return path


class ResponseChart(_Panel):
    """Stacked time to the reply, per turn, with p50 and p95."""

    title = "Response Time Per Turn"

    def draw(self, painter: QPainter, body: QRectF) -> None:
        turns = [t for t in self.snapshot.turns if t.response_ms is not None]
        if not turns:
            self.empty(painter, body, "No replies yet")
            return
        legend_h = 16
        plot = body.adjusted(34, 0, 0, -legend_h)
        top = max(max(t.response_ms or 0 for t in turns), self.snapshot.p95_ms or 0, 1.0) * 1.1
        _axis(painter, plot, top)
        step = plot.width() / max(len(turns), 1)
        bar = max(2.0, step * 0.68)
        for index, turn in enumerate(turns):
            x = plot.left() + index * step + (step - bar) / 2
            y = plot.bottom()
            for group in GROUPS:
                height = plot.height() * turn.segments[group] / top
                if height <= 0:
                    continue
                painter.fillRect(QRectF(x, y - height, bar, height), _colour(GROUP_COLOURS[group],
                                                                            0.55 if turn.failed else 0.9))
                y -= height
        for value, label, style in ((self.snapshot.p50_ms, "p50", Qt.PenStyle.DashLine),
                                    (self.snapshot.p95_ms, "p95", Qt.PenStyle.DotLine)):
            if value is None:
                continue
            y = plot.bottom() - plot.height() * value / top
            painter.setPen(QPen(_colour(C.WHITE, 0.75), 1, style))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setFont(_font(7.5, mono=True))
            painter.drawText(QRectF(plot.right() - 90, y - 13, 90, 12), Qt.AlignmentFlag.AlignRight,
                             f"{label} {_format_ms(value)}")
        _legend(painter, QRectF(body.left() + 34, body.bottom() - 12, body.width() - 34, 12),
                [(GROUP_LABELS[g], GROUP_COLOURS[g], Qt.PenStyle.SolidLine) for g in GROUPS])


class BenchmarkChart(_Panel):
    """Every self-benchmark run, per category and overall."""

    title = "Self-Benchmark"

    def draw(self, painter: QPainter, body: QRectF) -> None:
        runs = self.snapshot.bench
        if not runs:
            self.empty(painter, body, "No benchmark runs yet")
            return
        legend_h = 28                       # dates under the axis, then the legend
        plot = body.adjusted(34, 0, -6, -legend_h)
        _axis(painter, plot, 100.0, unit="%")

        def x_at(index: int) -> float:
            return plot.left() + (plot.width() * index / (len(runs) - 1) if len(runs) > 1 else plot.width() / 2)

        def y_at(percent: float) -> float:
            return plot.bottom() - plot.height() * percent / 100.0

        series: list[tuple[str, str, Qt.PenStyle, float]] = []
        for name, (colour, style) in BENCH_STYLES.items():
            series.append((name, colour, style, 1.4))
        for name, colour, style, width in series:
            path = QPainterPath()
            started = False
            for index, run in enumerate(runs):
                cat = (run.get("categories") or {}).get(name)
                if not cat or not cat.get("total"):
                    continue
                point = QPointF(x_at(index), y_at(100.0 * cat["passed"] / cat["total"]))
                if started:
                    path.lineTo(point)
                else:
                    path.moveTo(point)
                    started = True
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(_colour(colour))
                painter.drawEllipse(point, 2.4, 2.4)
            painter.setPen(QPen(_colour(colour), width, style))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
        overall = QPainterPath()
        for index, run in enumerate(runs):
            point = QPointF(x_at(index), y_at(float(run.get("overall") or 0)))
            overall.moveTo(point) if index == 0 else overall.lineTo(point)
        painter.setPen(QPen(_colour(C.WHITE), 2.2))
        painter.drawPath(overall)
        painter.setPen(_colour(C.FAINT))
        painter.setFont(_font(7, mono=True))
        for index in sorted({0, len(runs) - 1}):
            stamp = float(runs[index].get("at") or 0)
            if stamp > 1e9:
                text = time.strftime("%d %b %H:%M", time.localtime(stamp))
                align = Qt.AlignmentFlag.AlignLeft if index == 0 else Qt.AlignmentFlag.AlignRight
                box = QRectF(x_at(index) - (0 if index == 0 else 90), plot.bottom() + 1, 90, 11)
                painter.drawText(box, align | Qt.AlignmentFlag.AlignTop, text)
        last = runs[-1]
        painter.setFont(_font(9, mono=True, bold=True))
        painter.drawText(QRectF(plot.right() - 80, plot.top() - 2, 80, 14), Qt.AlignmentFlag.AlignRight,
                         f"{float(last.get('overall') or 0):.0f}%")
        _legend(painter, QRectF(body.left() + 34, body.bottom() - 12, body.width() - 34, 12),
                [("Overall", C.WHITE, Qt.PenStyle.SolidLine)]
                + [(n.title(), c, s) for n, (c, s) in BENCH_STYLES.items()])


class ProviderBars(_Panel):
    """Requests in 24 hours per provider, with latency and failure."""

    title = "Providers · 24 Hours"

    def draw(self, painter: QPainter, body: QRectF) -> None:
        rows = self.snapshot.providers
        if not rows:
            self.empty(painter, body, "No provider traffic yet")
            return
        top = max([r.requests for r in rows] + [1])
        _bars(painter, body, [(r.name, r.requests / top, 0.0,
                               f"{r.requests} req · {_tokens(r.tokens)} · "
                               f"{'–' if r.latency_s is None else f'{r.latency_s:.1f} s'}",
                               r.failing) for r in rows])


class ToolBars(_Panel):
    """Calls, failures and median time per tool this session."""

    title = "Tools · This Session"

    def draw(self, painter: QPainter, body: QRectF) -> None:
        rows = self.snapshot.tools
        if not rows:
            self.empty(painter, body, "No tool calls yet")
            return
        top = max([r.calls for r in rows] + [1])
        _bars(painter, body, [(r.name.replace("_", " "), r.calls / top,
                               r.failures / top, f"{r.calls} · {_format_ms(r.p50_ms)}",
                               bool(r.failures) and r.failures / max(r.calls, 1) >= 0.2)
                              for r in rows])


def _tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M tok"
    if count >= 1000:
        return f"{count / 1000:.0f}k tok"
    return f"{count} tok"


def _axis(painter: QPainter, plot: QRectF, top: float, unit: str = "ms") -> None:
    painter.setFont(_font(7.5, mono=True))
    for fraction in (0.0, 0.5, 1.0):
        y = plot.bottom() - plot.height() * fraction
        painter.setPen(QPen(_colour(C.BORDER), 1))
        painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        value = top * fraction
        text = (f"{value:.0f}%" if unit == "%" else _format_ms(value))
        painter.setPen(_colour(C.FAINT))
        painter.drawText(QRectF(plot.left() - 34, y - 7, 30, 14),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)


def _legend(painter: QPainter, rect: QRectF, items: list[tuple[str, str, Qt.PenStyle]]) -> None:
    painter.setFont(_font(7.5))
    metrics = QFontMetricsF(painter.font())
    x = rect.left()
    for label, colour, style in items:
        painter.setPen(QPen(_colour(colour), 2.2, style))
        painter.drawLine(QPointF(x, rect.center().y()), QPointF(x + 12, rect.center().y()))
        painter.setPen(_colour(C.MUTED))
        painter.drawText(QPointF(x + 16, rect.center().y() + 3.5), label)
        x += 16 + metrics.horizontalAdvance(label) + 14
        if x > rect.right():
            break


def _bars(painter: QPainter, body: QRectF,
          rows: list[tuple[str, float, float, str, bool]]) -> None:
    row_h = min(26.0, body.height() / max(len(rows), 1))
    label_w = min(130.0, body.width() * 0.32)
    value_w = min(170.0, body.width() * 0.34)
    track_w = max(10.0, body.width() - label_w - value_w - 12)
    for index, (label, share, failed_share, value, alarm) in enumerate(rows):
        y = body.top() + index * row_h
        painter.setFont(_font(8.5))
        painter.setPen(_colour(C.SILVER))
        painter.drawText(QRectF(body.left(), y, label_w - 6, row_h),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         QFontMetricsF(painter.font()).elidedText(label, Qt.TextElideMode.ElideRight,
                                                                  label_w - 6))
        track = QRectF(body.left() + label_w, y + row_h * 0.3, track_w, row_h * 0.4)
        painter.fillRect(track, _colour(C.INK))
        painter.fillRect(QRectF(track.left(), track.top(), track.width() * max(0.0, min(1.0, share)),
                                track.height()), _colour(C.PRI, 0.85))
        if failed_share > 0:
            painter.fillRect(QRectF(track.left(), track.top(), track.width() * min(1.0, failed_share),
                                    track.height()), _colour(C.BAD))
        painter.setFont(_font(8, mono=True))
        painter.setPen(_colour(C.BAD if alarm else C.MUTED))
        painter.drawText(QRectF(track.right() + 8, y, value_w, row_h),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, value)


class _Tile(QWidget):
    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(0)
        self.value = QLabel("–")
        self.value.setObjectName("summaryValue")
        self.value.setFont(_font(15, mono=True, bold=True))
        self.label = QLabel(label)
        self.label.setObjectName("mutedLabel")
        layout.addWidget(self.value)
        layout.addWidget(self.label)
        self.setAccessibleName(label)


# ── the page ─────────────────────────────────────────────────────────────────

class NeuralStatsView(QWidget):
    """The NEURAL deck page."""

    REFRESH_MS = 1500
    PULSE_MS = 33

    def __init__(self, bus: Any = None, *, telemetry: Any = None, router: Any = None,
                 traces: Any = None, bench: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.telemetry = telemetry
        self.router = router
        if traces is None:
            from ..request_trace import TRACES as traces
        self.traces = traces
        if bench is None:
            from .. import self_benchmark
            bench = self_benchmark.history
        self._bench_source = bench
        self._bench: list[dict[str, Any]] = []
        self._bench_at = -1e9
        self._ledger_rows: list[dict[str, Any]] = []
        self._ledger_at = -1e9
        self._seen_events: dict[str, int] = {}
        self._seen_tools: dict[str, int] = {}
        self._primed = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        title = QLabel("Neural")
        title.setObjectName("titleLabel")
        root.addWidget(title)
        tiles = QHBoxLayout()
        tiles.setSpacing(6)
        self.tiles: dict[str, _Tile] = {}
        for label in [label for label, _value in collect([]).tiles] + list(PROCESS_TILES):
            tile = _Tile(label)
            self.tiles[label] = tile
            tiles.addWidget(tile, 1)
        root.addLayout(tiles)
        self._process: Any = None
        # Brain activity as an ECG trace (Mark XXXX.4), fed by the live log.
        from .brain_instruments import EcgMonitor
        self.ecg = EcgMonitor()
        ecg_frame = QWidget()
        ecg_frame.setObjectName("panelFrame")
        ecg_layout = QVBoxLayout(ecg_frame)
        ecg_layout.setContentsMargins(14, 8, 14, 10)
        ecg_layout.setSpacing(4)
        ecg_title = QLabel("Brain Activity")
        ecg_title.setObjectName("panelHeading")
        ecg_layout.addWidget(ecg_title)
        # A real research run and its current question - shown only while
        # one runs (ResearchActivity), hidden the moment it finishes.
        self.research = ResearchActivity()
        self.research_line = QLabel("")
        self.research_line.setObjectName("mutedLabel")
        self.research_line.setAccessibleName("Research in progress")
        self.research_line.setVisible(False)
        ecg_layout.addWidget(self.research_line)
        from .hud_instruments import ToolRadar
        self.radar = ToolRadar(bus)
        self.radar.setMinimumSize(220, 190)
        activity_row = QHBoxLayout()
        activity_row.setSpacing(14)
        activity_row.addWidget(self.ecg, 3)
        activity_row.addWidget(self.radar, 1)
        ecg_layout.addLayout(activity_row, 1)
        ecg_frame.setMaximumHeight(250)
        root.addWidget(ecg_frame)
        if bus is not None and hasattr(bus, "log"):
            bus.log.connect(self._on_log)
        if bus is not None and hasattr(bus, "dashboard_event"):
            bus.dashboard_event.connect(self._on_dashboard_event)
        # A page built after a run started (deck pages are built on first
        # open) starts from the runs in progress, not from nothing.
        try:
            from .. import research_runs
            for row in research_runs.live_activity():
                self.research.apply(row)
        except Exception:
            pass
        self._show_research()
        self.flow = SignalFlowGraph()
        self.response = ResponseChart()
        self.benchmark = BenchmarkChart()
        self.providers = ProviderBars()
        self.tools = ToolBars()
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.addWidget(self.flow, 0, 0, 1, 2)
        grid.addWidget(self.response, 1, 0)
        grid.addWidget(self.benchmark, 1, 1)
        grid.addWidget(self.providers, 2, 0)
        grid.addWidget(self.tools, 2, 1)
        grid.setRowStretch(0, 5)
        grid.setRowStretch(1, 4)
        grid.setRowStretch(2, 4)
        root.addLayout(grid, 1)
        self.panels = (self.flow, self.response, self.benchmark, self.providers, self.tools)

        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(SignalFlowGraph.FRAME_MS)
        self._pulse_timer.timeout.connect(self._animate)

    # ── the ECG: what he does, as it happens ─────────────────────────────────

    def _on_log(self, message: str) -> None:
        self.ecg.feed(message)

    # ── research: only while a real run is in progress ───────────────────────

    def _on_dashboard_event(self, channel: str, payload: Any) -> None:
        if channel == "research" and self.research.apply(payload):
            self._show_research()

    def _show_research(self) -> None:
        text = self.research.line()
        self.research_line.setText(text)
        self.research_line.setToolTip(text)
        self.research_line.setVisible(bool(text))
        self.radar.set_research(self.research.caption())

    # ── data ─────────────────────────────────────────────────────────────────

    def _ledger(self, now: float) -> list[dict[str, Any]]:
        if now - self._ledger_at >= LEDGER_EVERY_S:
            self._ledger_at = now
            ledger = getattr(self.router, "token_ledger", None)
            try:
                self._ledger_rows = (ledger.by_dimension("provider", {"since": time.time() - 86400})
                                     if ledger is not None else [])
            except Exception:
                self._ledger_rows = []
        return self._ledger_rows

    def _benchmarks(self, now: float) -> list[dict[str, Any]]:
        if now - self._bench_at >= BENCH_EVERY_S:
            self._bench_at = now
            try:
                self._bench = list(self._bench_source() or [])
            except Exception:
                self._bench = []
        return self._bench

    def snapshot(self) -> NeuralSnapshot:
        now = time.monotonic()
        try:
            traces = self.traces.all()
        except Exception:
            traces = []
        return collect(traces, telemetry=self.telemetry, router=self.router,
                       ledger_rows=self._ledger(now), bench=self._benchmarks(now))

    def process_tiles(self) -> list[tuple[str, str]]:
        """ORION's own memory and CPU (this process and its page hosts), apart
        from the rest of the PC - the owner watches his RAM (Mark XXXX.4)."""
        try:
            import psutil
            if self._process is None:
                self._process = psutil.Process()
                self._process.cpu_percent(None)
            procs = [self._process, *self._process.children(recursive=True)]
            rss = 0
            for proc in procs:
                try:
                    rss += proc.memory_info().rss
                except psutil.Error:
                    continue
            cpu = self._process.cpu_percent(None) / max(1, psutil.cpu_count() or 1)
            return [("ORION RAM", f"{rss / 2**30:.2f} GB" if rss >= 2**30 else f"{rss / 2**20:.0f} MB"),
                    ("ORION CPU", f"{cpu:.0f}%")]
        except Exception:
            return [(label, "–") for label in PROCESS_TILES]

    def refresh(self) -> NeuralSnapshot:
        snap = self.snapshot()
        for label, value in snap.tiles + self.process_tiles():
            tile = self.tiles.get(label)
            if tile is not None:
                tile.value.setText(value)
        for panel in self.panels:
            panel.set_snapshot(snap)
        self.radar.set_counts({t.name: t.calls for t in snap.tools if t.name})
        self._detect_pulses(snap)
        return snap

    def _detect_pulses(self, snap: NeuralSnapshot) -> None:
        """A pulse for every stage a turn reached, and every tool call, since
        the last refresh. The first refresh only records: history is not
        replayed as if it were happening now."""
        if not self._primed:
            self._primed = True
            self._seen_events = {t.request_id: len(t.stages) for t in snap.turns}
            self._seen_tools = {t.name: t.calls for t in snap.tools}
            return
        for turn in snap.turns:
            seen = self._seen_events.get(turn.request_id, 0)
            for stage in turn.stages[seen:]:
                edge = {
                    "TRANSCRIPTION_COMPLETE": (f"in:{turn.origin}", "hear"),
                    "API_RESPONSE_RECEIVED": ("hear", "think"),
                    "PLAYBACK_START": ("think", "out:spoken"),
                    "REPLY_DELIVERED": ("think", "out:written"),
                }.get(stage)
                if edge:
                    self.flow.pulse(*edge)
            self._seen_events[turn.request_id] = len(turn.stages)
        for tool in snap.tools:
            if tool.calls > self._seen_tools.get(tool.name, 0):
                self.flow.pulse("think", f"tool:{tool.name}")
            self._seen_tools[tool.name] = tool.calls
        if len(self._seen_events) > 4 * MAX_TURNS:
            keep = {t.request_id for t in snap.turns}
            self._seen_events = {k: v for k, v in self._seen_events.items() if k in keep}
        if self.isVisible() and not self._pulse_timer.isActive():
            self._pulse_timer.start()

    def _animate(self) -> None:
        """One ambient frame of the signal flow (15 fps, visible only)."""
        window = self.window()
        if not self.isVisible() or (window is not None and window.isMinimized()):
            self._pulse_timer.stop()
            return
        self.flow.update()

    # ── visibility ───────────────────────────────────────────────────────────

    def showEvent(self, event: Any) -> None:  # noqa: N802
        self._bench_at = self._ledger_at = -1e9
        self.refresh()
        self._timer.start()
        self._pulse_timer.start()
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.stop()
        self._pulse_timer.stop()
        super().hideEvent(event)


__all__ = ["NeuralSnapshot", "NeuralStatsView", "ResearchActivity", "collect", "percentile",
           "turn_from_trace"]
