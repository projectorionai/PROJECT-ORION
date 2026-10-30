"""Synthetic audio, visual, and host signals for the first context contract."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from orion_core.context_events import (
    AudioObservation,
    ContextBroker,
    ContextEvent,
    PublishResult,
    Sensitivity,
    SystemObservation,
    VisualObservation,
)


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def event(event_id: str, payload=None, *, source: str = "audio.local",
          observed_at: float = 1_000.0, ttl: float = 30.0,
          sensitivity: Sensitivity = Sensitivity.PERSONAL) -> ContextEvent:
    return ContextEvent(
        event_id=event_id,
        source=source,
        observed_at=observed_at,
        expires_at=observed_at + ttl,
        confidence=0.83,
        evidence_ref=f"sample:{event_id}",
        sensitivity=sensitivity,
        payload=payload if payload is not None else AudioObservation("doorbell"),
    )


def test_synthetic_sources_keep_provenance_and_chronological_order():
    clock = Clock()
    broker = ContextBroker(clock=clock)
    audio = event("audio-1", AudioObservation("doorbell"),
                  observed_at=999.0)
    visual = event("visual-1", VisualObservation("person", "entry"),
                   source="camera.local", observed_at=998.0,
                   sensitivity=Sensitivity.SENSITIVE)
    system = event("system-1", SystemObservation("memory_used", 92.0, "%"),
                   source="system.monitor", observed_at=997.0,
                   sensitivity=Sensitivity.PUBLIC)

    for sample in (audio, system, visual):
        assert broker.publish(sample) == PublishResult.ACCEPTED

    assert [sample.event_id for sample in broker.snapshot()] == [
        "system-1", "visual-1", "audio-1"]
    state = broker.summary()
    assert [sample.event_id for sample in state.events] == [
        "audio-1", "visual-1", "system-1"]
    assert state.generated_at == clock.now
    assert state.events[1].source == "camera.local"
    assert state.events[1].evidence_ref == "sample:visual-1"
    assert state.events[1].sensitivity == Sensitivity.SENSITIVE


@pytest.mark.parametrize("payload", [
    AudioObservation("doorbell"),
    VisualObservation("person", "entry"),
    SystemObservation("battery", 14.5, "%"),
])
def test_envelope_round_trips_as_versioned_json(payload):
    original = event("roundtrip", payload)
    encoded = json.loads(json.dumps(original.to_dict()))
    assert ContextEvent.from_dict(encoded) == original
    assert encoded["schema_version"] == 1
    with pytest.raises(FrozenInstanceError):
        original.event_id = "changed"


def test_rejects_unknown_revision_fields_raw_media_and_unbounded_content():
    valid = event("base")
    for alteration in (
        {"schema_version": 2},
        {"schema_version": True},
        {"unexpected": "field"},
        {"payload": {"kind": "audio", "label": "doorbell", "pcm": b"raw"}},
        {"payload": {"kind": "video", "label": "frame"}},
        {"sensitivity": "unknown"},
    ):
        wire = valid.to_dict() | alteration
        with pytest.raises(ValueError):
            ContextEvent.from_dict(wire)

    with pytest.raises(ValueError):
        AudioObservation("x" * 97)
    with pytest.raises(ValueError):
        VisualObservation("person", b"raw frame")
    with pytest.raises(ValueError):
        event("bad", payload=b"raw audio")
    with pytest.raises(ValueError):
        event("bad", ttl=601)
    with pytest.raises(ValueError):
        ContextEvent(
            event_id="path", source="camera.local", observed_at=1_000.0,
            expires_at=1_001.0, confidence=0.5,
            sensitivity=Sensitivity.SENSITIVE,
            payload=VisualObservation("person"),
            evidence_ref="C:/raw/frame.jpg")
    with pytest.raises(ValueError):
        ContextEvent.from_dict(valid.to_dict() | {"confidence": float("nan")})
    with pytest.raises(ValueError, match="wire size limit"):
        ContextEvent(
            event_id="🧠" * 128, source="💬" * 64,
            observed_at=1_000.0, expires_at=1_001.0, confidence=0.5,
            sensitivity=Sensitivity.PUBLIC,
            payload=AudioObservation("🔔" * 96))


def test_repeated_id_is_deduplicated_even_after_its_event_was_coalesced():
    broker = ContextBroker(clock=Clock())
    first = event("a", observed_at=998.0)
    newer = event("b", observed_at=999.0)
    assert broker.publish(first) == PublishResult.ACCEPTED
    assert broker.publish(newer) == PublishResult.COALESCED
    assert broker.publish(first) == PublishResult.DUPLICATE
    assert broker.publish(newer) == PublishResult.DUPLICATE
    assert [sample.event_id for sample in broker.snapshot()] == ["b"]


def test_coalescing_is_per_source_and_signal_and_late_replay_cannot_rewind():
    broker = ContextBroker(clock=Clock())
    assert broker.publish(event("new", observed_at=999.0)) == PublishResult.ACCEPTED
    assert broker.publish(event("late", observed_at=998.0)) == PublishResult.STALE
    assert broker.publish(event("late")) == PublishResult.DUPLICATE
    assert broker.publish(event("other-source", source="audio.other")) == PublishResult.ACCEPTED
    assert broker.publish(event("other-sound", AudioObservation("alarm"))) == PublishResult.ACCEPTED
    assert broker.publish(event("newer", observed_at=1_000.0)) == PublishResult.COALESCED
    assert {sample.event_id for sample in broker.snapshot()} == {
        "newer", "other-source", "other-sound"}


def test_visual_regions_are_distinct_coalescing_keys():
    broker = ContextBroker(clock=Clock())
    entry = event("entry", VisualObservation("person", "entry"),
                  source="camera.local", observed_at=998.0)
    desk = event("desk", VisualObservation("person", "desk"),
                 source="camera.local", observed_at=999.0)
    entry_new = event("entry-new", VisualObservation("person", "entry"),
                      source="camera.local", observed_at=1_000.0)
    assert broker.publish(entry) == PublishResult.ACCEPTED
    assert broker.publish(desk) == PublishResult.ACCEPTED
    assert broker.publish(entry_new) == PublishResult.COALESCED
    assert {sample.event_id for sample in broker.snapshot()} == {"desk", "entry-new"}


def test_expiry_and_future_timestamp_do_not_create_stale_current_state():
    clock = Clock()
    broker = ContextBroker(clock=clock)
    assert broker.publish(event("expired", observed_at=900.0)) == PublishResult.EXPIRED
    assert broker.publish(event("future", observed_at=1_006.0)) == PublishResult.FUTURE
    assert broker.snapshot() == ()
    assert broker.publish(event("fresh", ttl=2.0)) == PublishResult.ACCEPTED
    clock.now = 1_002.0
    assert broker.snapshot() == ()
    assert broker.summary().events == ()


def test_capacity_eviction_uses_observation_time_and_all_caches_are_bounded():
    broker = ContextBroker(max_events=2, max_seen_ids=2, clock=Clock())
    assert broker.publish(event("new", source="a", observed_at=999.0)) == PublishResult.ACCEPTED
    assert broker.publish(event("old", source="b", observed_at=997.0)) == PublishResult.ACCEPTED
    assert broker.publish(event("middle", source="c", observed_at=998.0)) == PublishResult.ACCEPTED
    assert [sample.event_id for sample in broker.snapshot()] == ["middle", "new"]
    assert broker.evicted_count == 1
    assert len(broker._seen_ids) <= 2
    assert len(broker.summary(max_items=1).events) == 1
    with pytest.raises(ValueError):
        broker.summary(max_items=-1)


def test_parallel_producers_cannot_exceed_capacity():
    broker = ContextBroker(max_events=12, clock=Clock())
    samples = [event(f"sample-{i}", SystemObservation(f"metric-{i}", i),
                     source="system.monitor") for i in range(80)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(broker.publish, samples))
    assert all(result == PublishResult.ACCEPTED for result in results)
    assert len(broker.snapshot()) == 12
    assert broker.evicted_count == 68
    assert len(broker._seen_ids) <= broker.max_seen_ids


def test_source_stop_and_shutdown_clear_volatile_context():
    broker = ContextBroker(clock=Clock())
    broker.publish(event("camera", VisualObservation("person"),
                         source="camera.local"))
    broker.publish(event("audio", source="audio.local"))
    assert broker.clear_source("camera.local") == 1
    assert [sample.event_id for sample in broker.snapshot()] == ["audio"]
    assert broker.clear_source("camera.local") == 0
    broker.clear()
    assert broker.snapshot() == ()
    assert broker._seen_ids == {}


def test_capacity_settings_have_absolute_limits():
    with pytest.raises(ValueError):
        ContextBroker(max_events=1025)
    with pytest.raises(ValueError):
        ContextBroker(max_events=4, max_seen_ids=4097)

