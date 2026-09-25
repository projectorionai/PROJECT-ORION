"""
Command router + health model.

Two rules from the brief drive every test here:

  §20  "No fake controls. No dead buttons. No placeholder functionality
        presented as complete."
  §23  "The UI should reflect actual state rather than decorative indicators."

So: every registered command must run a real handler and report a real result,
and every health probe must be able to say UNKNOWN rather than green-by-default.
"""

from __future__ import annotations

import asyncio

import pytest

from orion_core.command_router import Command, CommandRouter
from orion_core.data import ToolResult
from orion_core.health_model import (
    DEGRADED, OFFLINE, ONLINE, RECOVERING, UNKNOWN, HealthModel,
)


class _Sig:
    def __init__(self):
        self.messages = []

    def emit(self, *a):
        self.messages.append(a if len(a) != 1 else a[0])

    def connect(self, *a):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Sig()
        object.__setattr__(self, name, sig)
        return sig


# ── command router ────────────────────────────────────────────────────────────

def test_every_registered_command_has_a_real_handler():
    router = CommandRouter(_Bus())
    for command in router.commands():
        assert callable(command.handler), command.id
        assert command.subsystem, f"{command.id} names no owning subsystem"
        assert command.title and command.subtitle, command.id


def test_a_command_actually_runs_and_returns_a_result():
    router = CommandRouter(_Bus())
    ran = []

    async def handler():
        ran.append(True)
        return ToolResult("did the thing")

    router.register(Command("test.thing", "Test Thing", "Does it.",
                            "TestSubsystem", handler))
    result = asyncio.run(router.run("test.thing"))
    assert ran == [True], "choosing a command must execute it, not prefill text"
    assert result.ok
    assert result.text == "did the thing"


def test_the_result_reaches_the_ui_as_an_event_not_a_widget_call():
    bus = _Bus()
    router = CommandRouter(bus)

    async def handler():
        return ToolResult("done")

    router.register(Command("test.event", "T", "s", "Sub", handler))
    asyncio.run(router.run("test.event"))

    channels = [m[0] for m in bus.dashboard_event.messages]
    assert "command_started" in channels
    assert "command_result" in channels


def test_a_failing_command_reports_instead_of_raising():
    router = CommandRouter(_Bus())

    async def handler():
        raise RuntimeError("subsystem exploded")

    router.register(Command("test.boom", "Boom", "s", "Sub", handler))
    result = asyncio.run(router.run("test.boom"))
    assert result.ok is False
    assert "subsystem exploded" in result.text


def test_an_unknown_command_is_reported_not_silently_ignored():
    router = CommandRouter(_Bus())
    result = asyncio.run(router.run("nope.nothing"))
    assert result.ok is False
    assert "No command with id" in result.text


def test_a_handler_returning_none_still_produces_a_result():
    router = CommandRouter(_Bus())

    async def handler():
        return None

    router.register(Command("test.none", "Nothing", "s", "Sub", handler))
    assert asyncio.run(router.run("test.none")).ok


def test_diagnostic_commands_are_always_available():
    """These need no subsystem, so they must never be missing."""
    router = CommandRouter(_Bus())
    ids = router.ids()
    assert "diagnostics.last_silence" in ids
    assert "diagnostics.traces" in ids
    assert "audio.devices" in ids


def test_the_why_did_you_ignore_me_command_answers():
    router = CommandRouter(_Bus())
    result = asyncio.run(router.run("diagnostics.last_silence"))
    assert result.text


def test_the_audio_device_command_checks_real_devices():
    router = CommandRouter(_Bus())
    result = asyncio.run(router.run("audio.devices"))
    assert "listening on" in result.text
    assert "Output check:" in result.text


def test_commands_are_only_registered_when_the_subsystem_exists():
    bare = CommandRouter(_Bus())
    assert not any(c.id.startswith("protocol.") for c in bare.commands()), (
        "a protocol command with no ProtocolManager would fail when clicked")


def test_protocol_commands_appear_when_protocols_are_supplied():
    class Protocols:
        def all_protocols(self):
            return {"morning": {"description": "Morning start-up."},
                    "emergency": {"description": "Urgent events."}}

        async def run(self, name):
            return ToolResult(f"ran {name}")

    router = CommandRouter(_Bus(), protocols=Protocols())
    ids = router.ids()
    assert "protocol.morning" in ids
    assert "protocol.emergency" in ids
    assert asyncio.run(router.run("protocol.emergency")).text == "ran emergency"


def test_palette_entries_expose_id_title_subtitle_and_confirmation():
    router = CommandRouter(_Bus())
    entries = router.palette_entries()
    assert entries
    for command_id, title, subtitle, destructive in entries:
        assert command_id and title and subtitle
        assert isinstance(destructive, bool)


def test_palette_action_entries_carry_the_command_id_as_their_name():
    from orion_core.gui.command_palette import KIND_ACTION, build_action_commands

    router = CommandRouter(_Bus())
    commands = build_action_commands(router.palette_entries())
    assert commands
    assert all(c.kind == KIND_ACTION for c in commands)
    assert all("." in c.name for c in commands), (
        "actions are dispatched by command id, so the id must be the name")


# ── health model ──────────────────────────────────────────────────────────────

def test_a_probe_that_raises_is_unknown_never_green():
    health = HealthModel(_Bus())

    def broken():
        raise RuntimeError("cannot read state")

    health.register("Thing", broken)
    snapshot = health.snapshot()
    assert snapshot["Thing"]["status"] == UNKNOWN
    assert "cannot read state" in snapshot["Thing"]["detail"]


def test_an_unregistered_subsystem_is_absent_not_healthy():
    health = HealthModel(_Bus())
    assert "Globe" not in health.snapshot()


def test_a_probe_returning_junk_becomes_unknown():
    health = HealthModel(_Bus())
    health.register("Odd", lambda: {"status": "SPLENDID"})
    assert health.snapshot()["Odd"]["status"] == UNKNOWN


def test_overall_takes_the_worst_critical_subsystem():
    health = HealthModel(_Bus())
    health.register("Fine", lambda: {"status": ONLINE}, critical=True)
    health.register("Broken", lambda: {"status": OFFLINE}, critical=True)
    assert health.overall() == OFFLINE


def test_a_non_critical_failure_does_not_condemn_the_whole_system():
    health = HealthModel(_Bus())
    health.register("Core", lambda: {"status": ONLINE}, critical=True)
    health.register("Cosmetic", lambda: {"status": OFFLINE})
    assert health.overall() == ONLINE


def test_problems_lists_only_what_needs_attention():
    health = HealthModel(_Bus())
    health.register("A", lambda: {"status": ONLINE})
    health.register("B", lambda: {"status": DEGRADED, "detail": "slow"})
    health.register("C", lambda: {"status": RECOVERING})
    names = {n for n, _s, _d in health.problems()}
    assert names == {"B", "C"}


def test_render_shows_every_subsystem_and_its_state():
    health = HealthModel(_Bus())
    health.register("Speaker", lambda: {"status": OFFLINE, "detail": "no stream"})
    text = health.render()
    assert "Speaker" in text
    assert "OFFLINE" in text
    assert "no stream" in text
    assert "need attention" in text


def test_publish_pushes_a_health_event():
    bus = _Bus()
    health = HealthModel(bus)
    health.register("A", lambda: {"status": ONLINE})
    health.publish()
    assert any(m[0] == "health" for m in bus.dashboard_event.messages)


def test_defaults_register_only_supplied_subsystems():
    health = HealthModel(_Bus())
    health.register_defaults()
    registered = set(health.registered())
    assert "OrionBus" in registered
    assert "Network" in registered
    assert "RequestPipeline" in registered
    # Nothing was supplied, so none of these may be claimed:
    assert "Microphone" not in registered
    assert "Globe" not in registered
    assert "LLM" not in registered


def test_a_disabled_microphone_reads_degraded_not_online():
    from types import SimpleNamespace

    health = HealthModel(_Bus())
    worker = SimpleNamespace(
        mic=SimpleNamespace(_stream=object()),
        microphone_enabled=False,
        recogniser=SimpleNamespace(available=True),
        speech=None,
        connected=True,
    )
    health.register_defaults(worker=worker)
    assert health.snapshot()["Microphone"]["status"] == DEGRADED


def test_a_missing_capture_stream_reads_offline():
    from types import SimpleNamespace

    health = HealthModel(_Bus())
    worker = SimpleNamespace(
        mic=SimpleNamespace(_stream=None),
        microphone_enabled=True,
        recogniser=SimpleNamespace(available=True),
        speech=None,
        connected=True,
    )
    health.register_defaults(worker=worker)
    assert health.snapshot()["Microphone"]["status"] == OFFLINE


def test_swarm_only_navigation_is_reported_as_degraded():
    """It is the state that made the globe unreachable, so it must be visible."""
    from types import SimpleNamespace

    health = HealthModel(_Bus())
    deck = SimpleNamespace(
        page_names=lambda: ["GLOBE", "LOG"],
        swarm_navigation=True,
        _pages=[],
    )
    health.register_defaults(deck=deck)
    entry = health.snapshot()["CommandDeck"]
    assert entry["status"] == DEGRADED
    assert "tabs hidden" in entry["detail"]


def test_an_unopened_globe_is_unknown_rather_than_broken_or_fine():
    from types import SimpleNamespace

    health = HealthModel(_Bus())
    globe = SimpleNamespace(_built=False, view=None, _rebuilds=0)
    deck = SimpleNamespace(
        page_names=lambda: ["GLOBE"],
        swarm_navigation=False,
        _pages=[("GLOBE", globe)],
    )
    health.register_defaults(deck=deck)
    entry = health.snapshot()["Globe"]
    assert entry["status"] == UNKNOWN
    assert "lazy" in entry["detail"]
