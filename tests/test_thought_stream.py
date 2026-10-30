"""
Tests for the free-thinking stream (ThoughtStream + the router's thought path).

* The 'thoughts' profile is reserved: it never appears in normal text routing,
  and generate_thought prefers it with the compact introspection instruction.
* Reflections are broadcast on bus.thought, kept out of the terminal, journalled to
  JSONL, and tagged task='thought' in the token ledger.
* Reflections are frugal: an unchanged context does not re-think; decision
  commentary is rate-limited by the gap timer.
* A thought can never act: the stream holds no dispatcher and only ever emits.
"""

from __future__ import annotations

import asyncio

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.providers import (
    AIProviderProfile,
    OrionProviderSettings,
    ProviderRouter,
)
from orion_core.thought_stream import ThoughtStream


class _Signal:
    def __init__(self, sink, name):
        self._sink, self._name = sink, name
        self._handlers = []

    def emit(self, *payload):
        self._sink.append((self._name, payload[0] if len(payload) == 1 else payload))
        for handler in self._handlers:
            handler(*payload)

    def connect(self, handler):
        self._handlers.append(handler)

    def disconnect(self, handler):
        self._handlers.remove(handler)


class _StubBus:
    def __init__(self):
        self.events = []
        self.log = _Signal(self.events, "log")
        self.thought = _Signal(self.events, "thought")

    def __getattr__(self, name):
        sig = _Signal(self.events, name)
        object.__setattr__(self, name, sig)
        return sig


class _StubMemory:
    def recent_turns(self, limit=6):
        return [{"role": "user", "content": "how is the build going?"}]

    def prompt_context(self, limit=16):
        return ""


def _profiles():
    thoughts = AIProviderProfile(
        name="thoughts", kind="openai_compatible", model="mini",
        api_key="tk", base_url="https://thoughts.example/v1", enabled=True,
    )
    chat = AIProviderProfile(
        name="chat", kind="openai_compatible", model="big",
        api_key="ck", base_url="https://chat.example/v1", enabled=True,
    )
    return {"thoughts": thoughts, "chat": chat}


def _router():
    providers = _profiles()
    settings = OrionProviderSettings(
        active_provider="chat",
        provider_order=["chat", "thoughts"],
        providers=providers,
    )
    bus = _StubBus()
    return ProviderRouter(settings, bus, memory=_StubMemory()), bus


def test_thoughts_profile_never_routes_user_turns():
    router, _bus = _router()
    names = [p.name for p in router.text_profiles()]
    assert "thoughts" not in names
    assert "chat" in names


# ── what thinking is allowed to cost ────────────────────────────────────────
#
# Thinking is LOCAL-ONLY by default now. The router had always preferred a
# local model, but preferring is not requiring: with none running it fell
# through to a paid provider, and a reflection every five minutes forever is a
# bill nobody asked for.
#
# The three tests below describe the paid fallback, so they opt into it
# explicitly. A test that silently relied on the old default would have gone
# on passing while the thing it guarded had been turned off.


@pytest.fixture
def paid_thoughts_allowed(monkeypatch):
    """Let the inner monologue reach a paid provider, as it used to."""
    monkeypatch.setenv("ORION_PAID_THOUGHTS", "1")
    return True


def test_generate_thought_prefers_dedicated_profile_and_tags_usage(paid_thoughts_allowed):
    router, _bus = _router()
    calls = []

    async def fake_chat(profile, prompt, system_extra="", *, instruction=None, task=""):
        calls.append((profile.name, instruction, task))
        return "I noticed the build is green; I intend to keep watch."

    router._openai_compatible_chat = fake_chat
    profile, text = asyncio.run(router.generate_thought("reflect"))
    assert profile.name == "thoughts"
    name, instruction, task = calls[0]
    assert task == "thought"
    assert "application reflection" in instruction
    assert "private reasoning" not in instruction
    assert text.startswith("I noticed")


def test_generate_thought_falls_back_when_slot_unconfigured(paid_thoughts_allowed):
    router, _bus = _router()
    router.settings.providers["thoughts"].enabled = False

    async def fake_chat(profile, prompt, system_extra="", *, instruction=None, task=""):
        return "fallback thought"

    router._openai_compatible_chat = fake_chat
    profile, _text = asyncio.run(router.generate_thought("reflect"))
    assert profile.name == "chat"


# ── local-first thought routing ─────────────────────────────────────────────
# Thinking should be free and instant once a local model is warm, so it leads
# even ahead of the dedicated cloud "thoughts" profile — the dedicated cloud
# profile stays the fallback when no local model is running.

def _router_with_local():
    providers = _profiles()
    providers["local_ollama"] = AIProviderProfile(
        name="local_ollama", kind="openai_compatible", model="qwen2.5",
        api_key="", base_url="http://127.0.0.1:11434/v1", enabled=True,
    )
    settings = OrionProviderSettings(
        active_provider="chat",
        provider_order=["chat", "thoughts", "local_ollama"],
        providers=providers,
    )
    bus = _StubBus()
    return ProviderRouter(settings, bus, memory=_StubMemory()), bus


def test_generate_thought_prefers_local_over_the_dedicated_cloud_profile():
    router, _bus = _router_with_local()

    async def fake_chat(profile, prompt, system_extra="", *, instruction=None, task=""):
        return f"thinking via {profile.name}"

    router._openai_compatible_chat = fake_chat
    profile, text = asyncio.run(router.generate_thought("reflect"))
    assert profile.name == "local_ollama"
    assert "local_ollama" in text


def test_generate_thought_falls_back_to_dedicated_when_local_unavailable(paid_thoughts_allowed):
    router, _bus = _router_with_local()
    router.settings.providers["local_ollama"].enabled = False

    async def fake_chat(profile, prompt, system_extra="", *, instruction=None, task=""):
        return f"thinking via {profile.name}"

    router._openai_compatible_chat = fake_chat
    profile, _text = asyncio.run(router.generate_thought("reflect"))
    assert profile.name == "thoughts"


def test_generate_thought_stream_also_prefers_local():
    router, _bus = _router_with_local()

    async def fake_stream(profile, prompt, system_extra="", *, instruction=None,
                          task="", on_delta=None):
        if on_delta:
            on_delta(f"streamed via {profile.name}")
        return f"streamed via {profile.name}"

    router._openai_compatible_chat_stream = fake_stream
    profile, text = asyncio.run(router.generate_thought_stream("reflect"))
    assert profile.name == "local_ollama"
    assert "local_ollama" in text


def _stream(tmp_path, router, bus):
    # These tests mock the non-streaming generate_thought; disable the live SSE
    # streaming path so the stream deterministically exercises that mock instead
    # of attempting a real network call to the fake provider URL.
    router.generate_thought_stream = None
    return ThoughtStream(
        bus, _StubMemory(), router,
        journal_path=tmp_path / "journal.jsonl",
    )


def test_reflection_broadcasts_and_journals_without_terminal_spam(tmp_path):
    router, bus = _router()

    async def fake_thought(prompt):
        return router.settings.providers["thoughts"], "I am watching the deploy."

    router.generate_thought = fake_thought
    stream = _stream(tmp_path, router, bus)
    asyncio.run(stream._reflection_thought())
    thoughts = [p for (n, p) in bus.events if n == "thought"]
    assert thoughts and thoughts[0]["text"] == "I am watching the deploy."
    assert thoughts[0]["kind"] == "reflection"
    logs = [p for (n, p) in bus.events if n == "log"]
    assert not any("I am watching the deploy." in str(entry) for entry in logs)
    snapshots = [p for (n, p) in bus.events if n == "dashboard_event"
                 and p[0] == "reflections"]
    assert snapshots[-1][1]["entries"][-1]["text"] == "I am watching the deploy."
    lines = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "reflection"
    assert stream.recent[-1]["text"] == "I am watching the deploy."


def test_unchanged_context_does_not_rethink(tmp_path):
    router, bus = _router()
    count = 0

    async def fake_thought(prompt):
        nonlocal count
        count += 1
        return router.settings.providers["thoughts"], "same thought"

    router.generate_thought = fake_thought
    stream = _stream(tmp_path, router, bus)
    asyncio.run(stream._reflection_thought())
    asyncio.run(stream._reflection_thought())   # identical context → skipped
    assert count == 1


def test_decision_commentary_consumes_queue_and_sets_gap(tmp_path):
    router, bus = _router()

    async def fake_thought(prompt):
        assert "dispatch: vision_analyse" in prompt
        return router.settings.providers["thoughts"], "I checked the screen first."

    router.generate_thought = fake_thought
    stream = _stream(tmp_path, router, bus)
    stream._note_decision("dispatch: vision_analyse")
    assert stream._decision_gap_open()
    asyncio.run(stream._decision_thought())
    assert not stream._pending_decisions
    assert not stream._decision_gap_open()      # rate-limited afterwards
    thoughts = [p for (n, p) in bus.events if n == "thought"]
    assert thoughts[0]["kind"] == "decision"


def test_stream_live_deltas_then_end_and_dedup(tmp_path):
    """When the provider streams, thoughts arrive token-by-token (start → deltas
    → end) and the full thought carries the SAME id so the GUI de-duplicates it
    instead of typing it a second time."""
    router, bus = _router()

    async def fake_stream(prompt, on_delta):
        for tok in ("I ", "see ", "the ", "deploy."):
            on_delta(tok)
        return router.settings.providers["thoughts"], "I see the deploy."

    router.generate_thought_stream = fake_stream
    stream = ThoughtStream(bus, _StubMemory(), router, journal_path=tmp_path / "j.jsonl")
    asyncio.run(stream._reflection_thought())

    deltas = [p for (n, p) in bus.events if n == "thought_delta"]
    phases = [d["phase"] for d in deltas]
    assert "start" in phases and "end" in phases and "abort" not in phases
    streamed = "".join(d["text"] for d in deltas if d["phase"] == "delta")
    assert streamed == "I see the deploy."
    thoughts = [p for (n, p) in bus.events if n == "thought"]
    assert thoughts[0]["text"] == "I see the deploy."
    end = next(d for d in deltas if d["phase"] == "end")
    assert end["id"] == thoughts[0]["id"]     # GUI keys de-dup off this id


def test_partial_stream_failure_aborts_and_delivers_full(tmp_path):
    """A stream that dies mid-thought must be aborted (partial dropped) and the
    COMPLETE thought delivered via the non-streaming path — never truncated to
    'just what he noted so far'."""
    router, bus = _router()

    async def flaky_stream(prompt, on_delta):
        on_delta("I no")            # a partial fragment, then the stream dies
        raise RuntimeError("stream cut")

    async def full_thought(prompt):
        return (router.settings.providers["thoughts"],
                "I noticed the deploy stalled and I am watching it closely.")

    router.generate_thought_stream = flaky_stream
    router.generate_thought = full_thought
    stream = ThoughtStream(bus, _StubMemory(), router, journal_path=tmp_path / "j.jsonl")
    asyncio.run(stream._reflection_thought())

    phases = [p["phase"] for (n, p) in bus.events if n == "thought_delta"]
    assert "abort" in phases and "end" not in phases
    thoughts = [p for (n, p) in bus.events if n == "thought"]
    assert thoughts[0]["text"] == \
        "I noticed the deploy stalled and I am watching it closely."


def test_provider_failure_is_contained(tmp_path):
    router, bus = _router()

    async def broken(prompt):
        raise RuntimeError("no provider")

    router.generate_thought = broken
    stream = _stream(tmp_path, router, bus)
    asyncio.run(stream._reflection_thought())   # must not raise
    assert not [p for (n, p) in bus.events if n == "thought"]
    assert any("THOUGHT: skipped" in str(p) for (n, p) in bus.events if n == "log")


def test_failure_cooldown_keeps_observed_activity_useful_without_new_model_calls(tmp_path, monkeypatch):
    import orion_core.thought_stream as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    router, bus = _router()
    calls = []

    async def unavailable(prompt):
        calls.append(prompt)
        raise RuntimeError("No local reflection provider is available")

    router.generate_thought = unavailable
    stream = _stream(tmp_path, router, bus)
    stream._note_decision("Desktop", "Calculator launch requested")
    asyncio.run(stream._decision_thought())
    assert len(calls) == 1
    assert stream._retry_at == 1060.0
    assert stream.recent[-1]["origin"] == "observation"
    assert stream.recent[-1]["evidence"] == ["Desktop Calculator launch requested"]
    assert "requested" in stream.recent[-1]["text"]
    assert "succeeded" not in stream.recent[-1]["text"]
    # New observed activity is useful while generation is cooling down.
    clock[0] = 1015.0
    stream._note_decision("Browser", "Page navigation requested")
    asyncio.run(stream._decision_thought())
    assert len(calls) == 1
    assert len(stream.recent) == 2
    assert stream.recent[-1]["evidence"] == ["Browser Page navigation requested"]
    assert len([p for name, p in bus.events if name == "log"]) == 1


def test_identical_provider_errors_are_coalesced_and_backoff_is_bounded(tmp_path, monkeypatch):
    import orion_core.thought_stream as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    router, bus = _router()

    async def unavailable(prompt):
        raise RuntimeError("Provider unavailable")

    router.generate_thought = unavailable
    stream = _stream(tmp_path, router, bus)
    asyncio.run(stream._reflection_thought())
    clock[0] = stream._retry_at
    asyncio.run(stream._reflection_thought())
    assert stream._retry_at == 1180.0
    assert len([p for name, p in bus.events if name == "log"]) == 1
    for _ in range(10):
        clock[0] = stream._retry_at
        asyncio.run(stream._reflection_thought())
    assert stream._retry_at - clock[0] == stream.MAX_FAILURE_RETRY_S
    assert len([p for name, p in bus.events if name == "log"]) == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_local_only_unavailability_never_spams_the_log(tmp_path, monkeypatch, streaming):
    import orion_core.thought_stream as module

    monkeypatch.delenv("ORION_PAID_THOUGHTS", raising=False)
    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    router, bus = _router()
    # Exercise the real policy rather than matching a truncated error string.
    monkeypatch.setattr(router, "_thought_profiles", lambda prompt: [router.settings.providers["chat"]])
    stream = ThoughtStream(bus, _StubMemory(), router, journal_path=tmp_path / "j.jsonl")
    if not streaming:
        router.generate_thought_stream = None
    else:
        async def duplicate_call(prompt):
            pytest.fail("an expected idle state must not retry non-streamed")
        router.generate_thought = duplicate_call
    for _ in range(12):
        stream._note_decision("Desktop", "Calculator launch requested")
        asyncio.run(stream._decision_thought())
        clock[0] = stream._retry_at
    assert not [p for name, p in bus.events if name == "log"]
    assert stream.recent[-1]["origin"] == "observation"
    assert "requested" in stream.recent[-1]["text"]


def test_a_real_failure_can_be_reported_again_after_recovery(tmp_path, monkeypatch):
    import orion_core.thought_stream as module
    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    router, bus = _router()
    failed = [True]

    async def generate(prompt):
        if failed[0]:
            raise RuntimeError("Provider unavailable")
        return router.settings.providers["thoughts"], "Review the latest task outcome."

    router.generate_thought = generate
    stream = _stream(tmp_path, router, bus)
    asyncio.run(stream._reflection_thought())
    failed[0] = False
    clock[0] = stream._retry_at
    asyncio.run(stream._reflection_thought())
    failed[0] = True
    stream._last_context_digest = ""
    asyncio.run(stream._reflection_thought())
    assert len([p for name, p in bus.events if name == "log"]) == 2


def test_success_resets_failure_backoff(tmp_path, monkeypatch):
    import orion_core.thought_stream as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    router, bus = _router()
    calls = []

    async def recovering(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("Temporarily unavailable")
        return router.settings.providers["thoughts"], "Review the latest task outcome."

    router.generate_thought = recovering
    stream = _stream(tmp_path, router, bus)
    asyncio.run(stream._reflection_thought())
    clock[0] = stream._retry_at
    asyncio.run(stream._reflection_thought())
    assert stream._failure_count == 0
    assert stream._retry_at == 0.0
    assert stream.recent[-1]["origin"] == "generated_reflection"


def test_empty_model_response_also_cools_down_instead_of_repeating_expensive_calls(tmp_path):
    router, bus = _router()
    calls = []

    async def empty(prompt):
        calls.append(prompt)
        return router.settings.providers["thoughts"], " "

    router.generate_thought = empty
    stream = _stream(tmp_path, router, bus)
    asyncio.run(stream._reflection_thought())
    asyncio.run(stream._reflection_thought())
    assert len(calls) == 1
    assert stream._failure_count == 1
    assert "provider returned no reflection text" in [
        payload for name, payload in bus.events if name == "log"][0]


def test_late_page_snapshot_is_bounded_and_excludes_raw_conversation(tmp_path):
    router, bus = _router()
    stream = _stream(tmp_path, router, bus)
    for number in range(75):
        stream._emit("A useful observation.", "reflection", "", number,
                     context={"focus": "Writing", "recent_conversation": "private turn"})
    latest = [payload for name, payload in bus.events
              if name == "dashboard_event" and payload[0] == "reflections"][-1][1]
    assert len(latest["entries"]) == 50
    assert latest["entries"][-1]["context"] == {"focus": "Writing"}
    assert "private turn" not in json.dumps(latest)
    assert not [payload for name, payload in bus.events if name == "log"]


def test_external_forge_and_maintenance_notes_join_late_page_history_once(tmp_path):
    router, bus = _router()
    stream = _stream(tmp_path, router, bus)
    note = {"id": "forge:round-12", "at": "09:00:00", "origin": "observation",
            "text": "Candidate checks passed; activation is awaiting approval.",
            "context": {"active_project": "ORION", "secret_field": "excluded"},
            "evidence": ["Benchmark: 12 cases passed"]}
    bus.thought.emit(note)
    bus.thought.emit(dict(note))
    assert len(stream.recent) == 1
    assert stream.recent[0]["context"] == {"active_project": "ORION"}
    assert stream.recent[0]["origin"] == "observation"
    snapshots = [payload[1] for name, payload in bus.events
                 if name == "dashboard_event" and payload[0] == "reflections"]
    assert snapshots[-1]["entries"][0]["text"] == note["text"]
    lines = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert note["id"] == "forge:round-12"  # Signal consumers retain their original event.
    assert not [payload for name, payload in bus.events if name == "log"]


def test_own_note_is_not_journalled_twice_by_the_external_observer(tmp_path):
    router, bus = _router()
    stream = _stream(tmp_path, router, bus)
    stream._emit("A useful reflection.", "reflection", "thoughts", 1)
    assert len(stream.recent) == 1
    assert len((tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_shutdown_disconnects_the_external_note_observer_and_ignores_late_delivery(tmp_path):
    router, bus = _router()
    stream = _stream(tmp_path, router, bus)
    assert stream._remember_external_note in bus.thought._handlers
    stream.stop()
    stream.stop()  # Closing twice is harmless.
    assert stream._remember_external_note not in bus.thought._handlers
    bus.thought.emit({"text": "Late note", "origin": "observation"})
    stream._remember_external_note({"text": "Queued callback after disconnect"})
    stream._note_decision("Late activity")
    assert not stream.recent
    assert not stream._pending_decisions
    assert not (tmp_path / "journal.jsonl").exists()


def test_late_model_completion_cannot_publish_after_shutdown(tmp_path):
    router, bus = _router()
    stream = _stream(tmp_path, router, bus)

    async def late(prompt):
        stream.stop()
        return router.settings.providers["thoughts"], "Too late."

    router.generate_thought = late
    assert asyncio.run(stream._think("summarise", "reflection")) is False
    assert not stream.recent
    assert not (tmp_path / "journal.jsonl").exists()


def test_late_model_failure_cannot_publish_a_fallback_after_shutdown(tmp_path):
    router, bus = _router()
    stream = _stream(tmp_path, router, bus)

    async def late_failure(prompt):
        stream.stop()
        raise RuntimeError("Closed while generating")

    router.generate_thought = late_failure
    assert asyncio.run(stream._think("summarise", "decision",
                                    evidence=["Calculator launch requested"])) is False
    assert not stream.recent
    assert not [payload for name, payload in bus.events if name == "log"]


def test_stream_holds_no_dispatcher():
    router, bus = _router()
    stream = ThoughtStream(bus, _StubMemory(), router)
    assert not hasattr(stream, "dispatcher")


def test_thinking_does_not_reach_a_paid_provider_by_default(monkeypatch):
    """The change that stops the bill.

    With no local model running and no explicit opt-in, a thought is simply
    not had. Silence costs nothing and a missing reflection is invisible; a
    recurring charge is neither.
    """
    monkeypatch.delenv("ORION_PAID_THOUGHTS", raising=False)
    router, _bus = _router()
    router.local_text_profiles = lambda: []

    reached = []

    async def fake_chat(profile, prompt, system_extra="", *, instruction=None,
                        task=""):
        reached.append(profile.name)
        return "should never happen"

    router._openai_compatible_chat = fake_chat
    with pytest.raises(Exception):
        asyncio.run(router.generate_thought("reflect"))
    assert reached == [], f"a paid provider was used to think: {reached}"
