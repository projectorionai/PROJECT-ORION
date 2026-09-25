"""
RequestTrace — per-request instrumentation for "why did ORION ignore me?".

The problem this solves
-----------------------
"ORION sometimes ignores me" was undiagnosable because a voice request passes
through eight subsystems on three threads and an event loop, and NONE of them
shared an identifier.  When nothing came back there was no way to tell whether

    the microphone never captured,
    the recogniser returned an empty string,
    the request never reached the model,
    the model never replied,
    a tool call hung,
    the reply was synthesised but never played,
    or playback finished but listening was never restored.

Every one of those looks identical from the outside: silence.

What this does
--------------
Each user request gets an id — ``ORION_REQ_000123`` — and every subsystem
stamps its stage against it.  A request that stops is then a request whose last
stage is not ``LISTENING_RESTORED``, and the stage that IS last names the
subsystem that dropped it.

    trace = TRACES.begin(origin="voice")
    trace.stage(Stage.CAPTURE_START)
    ...
    trace.stage(Stage.TRANSCRIPTION_COMPLETE, detail=text)
    ...
    trace.failed(Stage.API_REQUEST_START, "websocket closed", recovery="reconnect")

Nothing here can raise into the caller and nothing blocks: instrumentation that
can break the thing it measures is worse than none.  The store is bounded, so a
long session cannot grow it without limit.

``TRACES.stalled()`` is the query the health model and the diagnostics page
ask; ``TRACES.explain_last_silence()`` is the one a user gets when they say
"why did you ignore me?".
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Iterable


class Stage(str, Enum):
    """The pipeline, in order.  The ORDER is meaningful: the furthest stage a
    request reached is what identifies the subsystem that dropped it."""

    CAPTURE_START = "CAPTURE_START"
    CAPTURE_COMPLETE = "CAPTURE_COMPLETE"
    TRANSCRIPTION_START = "TRANSCRIPTION_START"
    TRANSCRIPTION_COMPLETE = "TRANSCRIPTION_COMPLETE"
    COMMAND_RECEIVED = "COMMAND_RECEIVED"
    AGENT_SELECTED = "AGENT_SELECTED"
    API_REQUEST_START = "API_REQUEST_START"
    API_RESPONSE_RECEIVED = "API_RESPONSE_RECEIVED"
    TOOL_EXECUTION_START = "TOOL_EXECUTION_START"
    TOOL_EXECUTION_COMPLETE = "TOOL_EXECUTION_COMPLETE"
    TTS_START = "TTS_START"
    PLAYBACK_START = "PLAYBACK_START"
    PLAYBACK_COMPLETE = "PLAYBACK_COMPLETE"
    LISTENING_RESTORED = "LISTENING_RESTORED"


#: Stage order for "how far did it get" comparisons.
STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)
_STAGE_INDEX = {stage: i for i, stage in enumerate(STAGE_ORDER)}

#: What the user actually experiences when a request dies at each stage, and
#: which subsystem owns it.  This is the whole point: turning "silence" into a
#: specific, actionable sentence.
_BLAME: dict[Stage, tuple[str, str]] = {
    Stage.CAPTURE_START: (
        "microphone",
        "audio was captured but never handed to the recogniser — the gate "
        "thread or the capture queue stalled"),
    Stage.CAPTURE_COMPLETE: (
        "speech recognition",
        "audio reached the recogniser but no transcription came back"),
    Stage.TRANSCRIPTION_START: (
        "speech recognition",
        "the recogniser started and never finished — it hung or timed out"),
    Stage.TRANSCRIPTION_COMPLETE: (
        "command routing",
        "the words were recognised but the command was never routed"),
    Stage.COMMAND_RECEIVED: (
        "agent routing",
        "the command was received but no agent or handler was selected"),
    Stage.AGENT_SELECTED: (
        "model provider",
        "the request was never sent to the model — the channel was down or "
        "the send blocked"),
    Stage.API_REQUEST_START: (
        "model provider",
        "the request was sent and the model never replied — a hung websocket, "
        "a timeout, or a rate limit"),
    Stage.API_RESPONSE_RECEIVED: (
        "tool execution",
        "the model replied and a tool call never returned"),
    Stage.TOOL_EXECUTION_START: (
        "tool execution",
        "a tool started and never completed — it hung"),
    Stage.TOOL_EXECUTION_COMPLETE: (
        "speech synthesis",
        "the answer was produced but never synthesised into speech"),
    Stage.TTS_START: (
        "speech synthesis",
        "synthesis started and produced no audio"),
    Stage.PLAYBACK_START: (
        "audio playback",
        "playback started and never completed — the output device stalled"),
    Stage.PLAYBACK_COMPLETE: (
        "microphone gate",
        "ORION finished speaking but listening was never restored — this is "
        "the state where he appears alive and ignores everything you say"),
}


@dataclass
class TraceEvent:
    stage: str
    at: float                 # monotonic
    wall: float               # unix time, for the log
    detail: str = ""
    ok: bool = True


@dataclass
class RequestTrace:
    """One request's journey through ORION."""

    request_id: str
    origin: str = "voice"           # voice / text / remote / protocol / gui
    started: float = field(default_factory=time.monotonic)
    started_wall: float = field(default_factory=time.time)
    events: list[TraceEvent] = field(default_factory=list)
    error: str = ""
    failed_stage: str = ""
    recovery: str = ""
    _bus: Any = field(default=None, repr=False)

    # ── recording ─────────────────────────────────────────────────────────────

    def stage(self, stage: Stage | str, detail: str = "") -> "RequestTrace":
        name = stage.value if isinstance(stage, Stage) else str(stage)
        self.events.append(TraceEvent(name, time.monotonic(), time.time(),
                                      str(detail)[:200]))
        self._emit(f"{self.request_id} {name}"
                   + (f" — {str(detail)[:120]}" if detail else ""))
        return self

    def failed(self, stage: Stage | str, error: str,
               recovery: str = "") -> "RequestTrace":
        name = stage.value if isinstance(stage, Stage) else str(stage)
        self.failed_stage = name
        self.error = str(error)[:300]
        self.recovery = str(recovery)[:200]
        self.events.append(TraceEvent(f"FAILED::{name}", time.monotonic(),
                                      time.time(), self.error, ok=False))
        self._emit(f"{self.request_id} FAILED_STAGE={name} ERROR={self.error}"
                   + (f" RECOVERY_ACTION={self.recovery}" if self.recovery else ""))
        return self

    def timeout(self, stage: Stage | str, seconds: float,
                recovery: str = "") -> "RequestTrace":
        return self.failed(stage, f"TIMEOUT after {seconds:.1f}s", recovery)

    def _emit(self, message: str) -> None:
        bus = self._bus
        if bus is None:
            return
        try:
            bus.log.emit(f"TRACE: {message}")
        except Exception:
            pass

    # ── reading ───────────────────────────────────────────────────────────────

    @property
    def last_stage(self) -> str:
        for event in reversed(self.events):
            if event.ok:
                return event.stage
        return ""

    @property
    def complete(self) -> bool:
        return any(e.stage == Stage.LISTENING_RESTORED.value for e in self.events)

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.started

    def elapsed_ms(self) -> float:
        if not self.events:
            return 0.0
        return (self.events[-1].at - self.started) * 1000.0

    def stage_timings(self) -> list[tuple[str, float]]:
        """(stage, ms since the previous stage) — where the latency went."""
        out: list[tuple[str, float]] = []
        previous = self.started
        for event in self.events:
            out.append((event.stage, (event.at - previous) * 1000.0))
            previous = event.at
        return out

    def diagnosis(self) -> str:
        """One sentence naming the subsystem that dropped this request."""
        if self.complete:
            return f"{self.request_id} completed in {self.elapsed_ms():.0f} ms."
        if self.failed_stage:
            head = (f"{self.request_id} failed at {self.failed_stage}: "
                    f"{self.error}")
            return head + (f" Recovery: {self.recovery}." if self.recovery else "")
        last = self.last_stage
        if not last:
            return (f"{self.request_id} never started — nothing was captured "
                    "at all (microphone disabled, or the capture stream is dead).")
        try:
            owner, what = _BLAME[Stage(last)]
        except (KeyError, ValueError):
            return f"{self.request_id} stopped after {last}."
        return (f"{self.request_id} stopped after {last} — {what}. "
                f"Responsible subsystem: {owner}.")

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "origin": self.origin,
            "complete": self.complete,
            "last_stage": self.last_stage,
            "failed_stage": self.failed_stage,
            "error": self.error,
            "recovery": self.recovery,
            "elapsed_ms": round(self.elapsed_ms(), 1),
            "age_s": round(self.age_s, 1),
            "stages": [e.stage for e in self.events],
            "diagnosis": self.diagnosis(),
        }


class TraceStore:
    """Bounded, thread-safe store of recent request traces."""

    #: A request still incomplete after this long is presumed dropped.
    STALL_AFTER_S = 45.0

    def __init__(self, capacity: int = 200) -> None:
        self._lock = RLock()
        self._traces: deque[RequestTrace] = deque(maxlen=capacity)
        self._counter = 0
        self._bus: Any = None

    def attach_bus(self, bus: Any) -> None:
        """Stage lines go to the ORION log once a bus is available."""
        self._bus = bus

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"ORION_REQ_{self._counter:06d}"

    def begin(self, origin: str = "voice", detail: str = "") -> RequestTrace:
        trace = RequestTrace(self.next_id(), origin=origin)
        trace._bus = self._bus
        with self._lock:
            self._traces.append(trace)
        if detail:
            trace.stage(Stage.COMMAND_RECEIVED, detail)
        return trace

    # ── queries ───────────────────────────────────────────────────────────────

    def all(self) -> list[RequestTrace]:
        with self._lock:
            return list(self._traces)

    def latest(self) -> RequestTrace | None:
        with self._lock:
            return self._traces[-1] if self._traces else None

    def stalled(self, older_than: float | None = None) -> list[RequestTrace]:
        """Requests that never reached LISTENING_RESTORED and have gone quiet."""
        limit = self.STALL_AFTER_S if older_than is None else older_than
        return [t for t in self.all()
                if not t.complete and t.age_s > limit]

    def explain_last_silence(self) -> str:
        """The answer to "why did you ignore me?" — in one sentence."""
        stalled = self.stalled()
        if stalled:
            return stalled[-1].diagnosis()
        latest = self.latest()
        if latest is None:
            return "No requests have been recorded this session."
        if latest.complete:
            return (f"The last request ({latest.request_id}) completed normally "
                    f"in {latest.elapsed_ms():.0f} ms — nothing was dropped.")
        return latest.diagnosis()

    def summary(self) -> dict[str, Any]:
        traces = self.all()
        complete = [t for t in traces if t.complete]
        stalled = self.stalled()
        failed = [t for t in traces if t.failed_stage]
        timings = [t.elapsed_ms() for t in complete]
        return {
            "total": len(traces),
            "complete": len(complete),
            "failed": len(failed),
            "stalled": len(stalled),
            "mean_ms": round(sum(timings) / len(timings), 1) if timings else 0.0,
            "worst_ms": round(max(timings), 1) if timings else 0.0,
            "diagnosis": self.explain_last_silence(),
        }

    def report(self, limit: int = 10) -> str:
        """Human-readable recent history for the diagnostics page."""
        lines = ["REQUEST TRACE — most recent first", ""]
        for trace in reversed(self.all()[-limit:]):
            mark = "OK " if trace.complete else ("ERR" if trace.failed_stage else "...")
            lines.append(f"[{mark}] {trace.request_id} ({trace.origin}) "
                         f"{trace.elapsed_ms():.0f} ms")
            lines.append(f"      {trace.diagnosis()}")
        summary = self.summary()
        lines += ["", f"{summary['complete']}/{summary['total']} completed, "
                      f"{summary['failed']} failed, {summary['stalled']} stalled."]
        return "\n".join(lines)

    def bottlenecks(self, limit: int = 5) -> list[tuple[str, float]]:
        """Mean ms spent in each stage across completed requests — where the
        latency actually is, for the provider/pipeline benchmark."""
        totals: dict[str, list[float]] = {}
        for trace in self.all():
            for stage, ms in trace.stage_timings():
                totals.setdefault(stage, []).append(ms)
        means = [(stage, sum(v) / len(v)) for stage, v in totals.items() if v]
        means.sort(key=lambda pair: -pair[1])
        return means[:limit]


#: The shared store.  One counter, so ids are unique across the process.
TRACES = TraceStore()


__all__ = ["RequestTrace", "Stage", "TraceStore", "TRACES", "STAGE_ORDER"]
