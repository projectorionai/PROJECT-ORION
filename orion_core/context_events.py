"""Versioned, in-memory context observations from existing ORION signal loops.

This module owns no devices and performs no persistence or network I/O. A
producer supplies an observation from a sensor it already owns; a later bus
adapter can publish accepted events through ``OrionBus``. The broker keeps only
bounded metadata and never accepts raw frames, audio, or arbitrary dict payloads.

Times are UTC Unix seconds. ``observed_at`` is when the source saw the signal;
``expires_at`` is the latest instant at which it may inform current context.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock
from typing import Callable, Literal, Mapping


EVENT_SCHEMA_VERSION = 1
MAX_EVENT_BYTES = 1024
MAX_TTL_SECONDS = 600.0
MAX_BROKER_EVENTS = 1024
MAX_SEEN_IDS = 4096
_OPAQUE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")


def _text(value: object, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"{name} must be a non-empty string of at most {limit} characters")
    if value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must have no outer whitespace or control characters")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _exact_fields(value: Mapping[str, object], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} fields must be {sorted(fields)}")


class Sensitivity(StrEnum):
    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"


@dataclass(frozen=True, slots=True)
class AudioObservation:
    """A classified sound label, never an audio sample or transcript."""

    label: str
    kind: Literal["audio"] = field(default="audio", init=False)

    def __post_init__(self) -> None:
        _text(self.label, "audio label", 96)


@dataclass(frozen=True, slots=True)
class VisualObservation:
    """A visual classification with an optional coarse region, never a frame."""

    label: str
    region: str | None = None
    kind: Literal["visual"] = field(default="visual", init=False)

    def __post_init__(self) -> None:
        _text(self.label, "visual label", 96)
        if self.region is not None:
            _text(self.region, "visual region", 64)


@dataclass(frozen=True, slots=True)
class SystemObservation:
    """A numeric host metric or state score, never a process dump."""

    metric: str
    value: float
    unit: str | None = None
    kind: Literal["system"] = field(default="system", init=False)

    def __post_init__(self) -> None:
        _text(self.metric, "system metric", 64)
        object.__setattr__(self, "value", _number(self.value, "system value"))
        if self.unit is not None:
            _text(self.unit, "system unit", 24)


ContextPayload = AudioObservation | VisualObservation | SystemObservation


def _payload_to_dict(payload: ContextPayload) -> dict[str, object]:
    if isinstance(payload, AudioObservation):
        return {"kind": "audio", "label": payload.label}
    if isinstance(payload, VisualObservation):
        return {"kind": "visual", "label": payload.label, "region": payload.region}
    if isinstance(payload, SystemObservation):
        return {"kind": "system", "metric": payload.metric,
                "value": payload.value, "unit": payload.unit}
    raise TypeError("context payload must be a typed observation")


def _payload_from_dict(value: object) -> ContextPayload:
    if not isinstance(value, Mapping):
        raise ValueError("payload must be an object")
    kind = value.get("kind")
    if kind == "audio":
        _exact_fields(value, {"kind", "label"}, "audio payload")
        return AudioObservation(label=value["label"])
    if kind == "visual":
        _exact_fields(value, {"kind", "label", "region"}, "visual payload")
        return VisualObservation(label=value["label"], region=value["region"])
    if kind == "system":
        _exact_fields(value, {"kind", "metric", "value", "unit"}, "system payload")
        return SystemObservation(metric=value["metric"], value=value["value"],
                                 unit=value["unit"])
    raise ValueError("unsupported context payload kind")


@dataclass(frozen=True, slots=True)
class ContextEvent:
    """One bounded observation, with source, lifetime, provenance, and privacy."""

    event_id: str
    source: str
    observed_at: float
    expires_at: float
    confidence: float
    sensitivity: Sensitivity
    payload: ContextPayload
    evidence_ref: str | None = None
    schema_version: int = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported context event schema version")
        _text(self.event_id, "event_id", 128)
        _text(self.source, "source", 64)
        if self.evidence_ref is not None:
            _text(self.evidence_ref, "evidence_ref", 128)
            if _OPAQUE_REF.fullmatch(self.evidence_ref) is None:
                raise ValueError("evidence_ref must be an opaque identifier, not a path or URL")
        if not isinstance(self.sensitivity, Sensitivity):
            raise ValueError("sensitivity must be a Sensitivity value")
        if not isinstance(self.payload, (AudioObservation, VisualObservation, SystemObservation)):
            raise ValueError("payload must be a typed observation")
        observed = _number(self.observed_at, "observed_at")
        expires = _number(self.expires_at, "expires_at")
        confidence = _number(self.confidence, "confidence")
        if not observed < expires or expires - observed > MAX_TTL_SECONDS:
            raise ValueError("event lifetime must be positive and at most 600 seconds")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "confidence", confidence)
        encoded = json.dumps(self.to_dict(), ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_EVENT_BYTES:
            raise ValueError("context event exceeds wire size limit")

    def to_dict(self) -> dict[str, object]:
        """Serialize a complete v1 envelope for a future bus or test fixture."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "source": self.source,
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "confidence": self.confidence,
            "evidence_ref": self.evidence_ref,
            "sensitivity": self.sensitivity.value,
            "payload": _payload_to_dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ContextEvent:
        """Read only the exact v1 wire shape; unknown revisions fail closed."""
        if not isinstance(value, Mapping):
            raise ValueError("context event must be an object")
        _exact_fields(value, {"schema_version", "event_id", "source", "observed_at",
                              "expires_at", "confidence", "evidence_ref",
                              "sensitivity", "payload"}, "context event")
        try:
            sensitivity = Sensitivity(value["sensitivity"])
        except (ValueError, TypeError) as exc:
            raise ValueError("unsupported context sensitivity") from exc
        return cls(
            schema_version=value["schema_version"],
            event_id=value["event_id"],
            source=value["source"],
            observed_at=value["observed_at"],
            expires_at=value["expires_at"],
            confidence=value["confidence"],
            evidence_ref=value["evidence_ref"],
            sensitivity=sensitivity,
            payload=_payload_from_dict(value["payload"]),
        )


class PublishResult(StrEnum):
    ACCEPTED = "accepted"
    COALESCED = "coalesced"
    DUPLICATE = "duplicate"
    STALE = "stale"
    EXPIRED = "expired"
    FUTURE = "future"


def _coalescing_key(event: ContextEvent) -> tuple[str, ...]:
    payload = event.payload
    if isinstance(payload, AudioObservation):
        return (event.source, "audio", payload.label)
    if isinstance(payload, VisualObservation):
        return (event.source, "visual", payload.label, payload.region or "")
    return (event.source, "system", payload.metric, payload.unit or "")


@dataclass(frozen=True, slots=True)
class ContextState:
    """A compact, current in-memory view, newest observation first."""

    generated_at: float
    events: tuple[ContextEvent, ...]


class ContextBroker:
    """Thread safe, bounded current context from existing source loops.

    A later observation with the same source and signal replaces the prior one.
    Replayed IDs are suppressed while remembered, even after coalescing. The
    ID cache is also bounded, so very old IDs can be forgotten under sustained
    load; current active event IDs remain protected. No disk state is retained.
    """

    def __init__(self, *, max_events: int = 128, max_seen_ids: int | None = None,
                 future_skew_seconds: float = 5.0,
                 clock: Callable[[], float] = time.time) -> None:
        if type(max_events) is not int or not 1 <= max_events <= MAX_BROKER_EVENTS:
            raise ValueError(f"max_events must be between 1 and {MAX_BROKER_EVENTS}")
        if max_seen_ids is None:
            max_seen_ids = max(32, max_events * 4)
        if (type(max_seen_ids) is not int
                or not max_events <= max_seen_ids <= MAX_SEEN_IDS):
            raise ValueError(f"max_seen_ids must be between max_events and {MAX_SEEN_IDS}")
        if _number(future_skew_seconds, "future_skew_seconds") < 0:
            raise ValueError("future_skew_seconds must be non-negative")
        self.max_events = max_events
        self.max_seen_ids = max_seen_ids
        self.future_skew_seconds = float(future_skew_seconds)
        self._clock = clock
        self._lock = Lock()
        self._active: dict[tuple[str, ...], ContextEvent] = {}
        self._seen_ids: OrderedDict[str, float] = OrderedDict()
        self.evicted_count = 0

    def _prune_locked(self, now: float) -> None:
        for key, event in tuple(self._active.items()):
            if event.expires_at <= now:
                del self._active[key]
        for event_id, expires_at in tuple(self._seen_ids.items()):
            if expires_at <= now:
                del self._seen_ids[event_id]

    def _remember_locked(self, event: ContextEvent) -> None:
        self._seen_ids[event.event_id] = event.expires_at
        while len(self._seen_ids) > self.max_seen_ids:
            self._seen_ids.popitem(last=False)

    def publish(self, event: ContextEvent) -> PublishResult:
        """Store one fresh observation or explain why it was discarded."""
        if not isinstance(event, ContextEvent):
            raise TypeError("publish requires a ContextEvent")
        now = _number(self._clock(), "clock")
        with self._lock:
            self._prune_locked(now)
            if event.expires_at <= now:
                return PublishResult.EXPIRED
            if event.observed_at > now + self.future_skew_seconds:
                return PublishResult.FUTURE
            if event.event_id in self._seen_ids or any(
                    active.event_id == event.event_id for active in self._active.values()):
                return PublishResult.DUPLICATE

            key = _coalescing_key(event)
            previous = self._active.get(key)
            if previous is not None:
                self._remember_locked(event)
                if event.observed_at <= previous.observed_at:
                    return PublishResult.STALE
                self._active[key] = event
                return PublishResult.COALESCED

            if len(self._active) >= self.max_events:
                oldest = min(self._active, key=lambda existing: (
                    self._active[existing].observed_at,
                    self._active[existing].event_id))
                del self._active[oldest]
                self.evicted_count += 1
            self._active[key] = event
            self._remember_locked(event)
            return PublishResult.ACCEPTED

    def snapshot(self) -> tuple[ContextEvent, ...]:
        """All fresh current observations, ordered oldest to newest."""
        now = _number(self._clock(), "clock")
        with self._lock:
            self._prune_locked(now)
            return tuple(sorted(self._active.values(),
                                key=lambda event: (event.observed_at, event.event_id)))

    def summary(self, *, max_items: int = 16) -> ContextState:
        """Bounded fresh state suitable for a future awareness bridge."""
        if type(max_items) is not int or max_items < 0:
            raise ValueError("max_items must be a non-negative integer")
        now = _number(self._clock(), "clock")
        with self._lock:
            self._prune_locked(now)
            latest = sorted(self._active.values(),
                            key=lambda event: (event.observed_at, event.event_id),
                            reverse=True)
            return ContextState(generated_at=now,
                                events=tuple(latest[:min(max_items, self.max_events)]))

    def clear_source(self, source: str) -> int:
        """Drop one source's active context when its sensing permission stops."""
        _text(source, "source", 64)
        with self._lock:
            keys = [key for key, event in self._active.items()
                    if event.source == source]
            for key in keys:
                del self._active[key]
            return len(keys)

    def clear(self) -> None:
        """Release all volatile state at shutdown."""
        with self._lock:
            self._active.clear()
            self._seen_ids.clear()

