"""
Clean exits, and ORION knowing what actually broke.

Both come from one report: every shutdown ended in a wall of tracebacks that
described the bus being deleted rather than any real fault, and self-repair
announced that it had captured something without saying what.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from orion_core.selfrepair import Incident, SelfRepairAgent  # noqa: E402


class _Log:
    def __init__(self): self.lines = []
    def emit(self, text): self.lines.append(text)


class _DeadLog:
    def emit(self, text):
        raise RuntimeError("wrapped C/C++ object of type OrionBus has been deleted")


class _Bus:
    def __init__(self, log=None):
        self.log = log or _Log()
        self.dashboard_event = _Log()
        self.connection_state = type("_S", (), {"connect": lambda *a: None})()


def _agent(bus=None):
    agent = SelfRepairAgent.__new__(SelfRepairAgent)
    agent.bus = bus or _Bus()
    agent.telemetry = None
    agent._loop = None
    agent._tasks = set()
    agent._shutting_down = False
    return agent


def _incident(**kw):
    fields = dict(id="inc-1", at="now", error_type="AttributeError",
                  message="'NoneType' object has no attribute 'render'",
                  file="C:/orion/orion_core/gui/face.py", line=214,
                  module="orion_core.gui.face")
    fields.update(kw)
    return Incident(**fields)


# ── the shutdown wall of tracebacks ──────────────────────────────────────────

def test_a_loop_error_after_the_bus_dies_does_not_raise():
    """The handler outlived the bus, so reporting a pending task raised FROM
    INSIDE the exception handler — which asyncio then reported again, once per
    task. A dozen tracebacks on every clean exit, none about a real fault."""
    agent = _agent(_Bus(log=_DeadLog()))
    agent._loop_exception_handler(None, {"message": "Task was destroyed"})


def test_a_loop_error_is_still_reported_while_the_bus_is_alive():
    agent = _agent()
    agent._loop_exception_handler(None, {"message": "Task was destroyed"})
    assert any("Task was destroyed" in line for line in agent.bus.log.lines)


def test_shutdown_cancels_repairs_still_in_flight():
    """Otherwise asyncio destroys them pending and prints a traceback each.

    Shutdown is now two steps, and the split is the whole point.
    ``shutdown()`` cancels but deliberately KEEPS its references; ``drain()``
    then awaits them. Clearing the set inside shutdown() — which is what this
    test used to assert — dropped the last reference while every task was still
    in the 'cancelling' state, so the loop never got a turn to deliver a single
    CancelledError. The user's console showed the result: over a hundred
    "Task was destroyed but it is pending!" lines on one exit.
    """
    async def _run():
        agent = _agent()

        async def _slow():
            await asyncio.sleep(30)

        task = asyncio.get_running_loop().create_task(_slow())
        agent._tasks.add(task)

        agent.shutdown()
        assert task.cancelling() or task.cancelled() or task.done()
        assert agent._tasks, (
            "references dropped mid-cancellation — the task will be destroyed "
            "while still pending")

        settled = await agent.drain(timeout=2.0)
        assert settled == 1
        assert task.done(), "the cancellation was never actually delivered"
        assert agent._tasks == set()

    asyncio.run(_run())


def test_shutdown_stops_new_repairs_being_scheduled():
    """A coroutine created after shutdown begins is never awaited — which is
    exactly the "coroutine ... was never awaited" warning."""
    agent = _agent()
    agent.shutdown()
    agent._schedule_auto_repair(_incident())
    assert agent._tasks == set()


def test_shutdown_is_safe_with_no_loop_and_twice_over():
    agent = _agent()
    agent.shutdown()
    agent.shutdown()


# ── he says what went wrong ──────────────────────────────────────────────────

def test_the_diagnosis_names_the_fault_in_plain_terms():
    text = _agent().diagnose(_incident())
    assert "does not exist" in text


def test_the_diagnosis_says_where():
    text = _agent().diagnose(_incident())
    assert "face.py" in text and "214" in text


def test_the_diagnosis_says_whether_he_can_fix_it():
    agent = _agent()
    missing = agent.diagnose(_incident(error_type="ModuleNotFoundError",
                                       message="No module named 'chess'"))
    assert "install it himself" in missing
    denied = agent.diagnose(_incident(error_type="PermissionError"))
    assert "cannot repair" in denied


def test_a_transient_fault_is_named_as_external():
    text = _agent().diagnose(_incident(error_type="ConnectionError",
                                       transient=True))
    assert "self-recovering" in text


def test_an_informational_incident_is_not_presented_as_repairable():
    text = _agent().diagnose(_incident(actionable=False))
    assert "nothing to repair" in text


def test_an_unknown_error_type_still_gets_a_useful_line():
    text = _agent().diagnose(_incident(error_type="WeirdCustomError"))
    assert "unrecognised" in text
    assert "face.py" in text


def test_a_fault_with_no_traceback_still_diagnoses():
    text = _agent().diagnose(_incident(file="", line=0, module=""))
    assert "without a traceback" in text


def test_the_diagnosis_is_one_line():
    """It goes to the log beside the summary; a paragraph would bury it."""
    for error in ("AttributeError", "KeyError", "RuntimeError", "Whatever"):
        assert "\n" not in _agent().diagnose(_incident(error_type=error))


# ── standby is silence until he is asked for ─────────────────────────────────

def _worker():
    from orion_core.live_worker import GenAILiveWorker
    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    worker.quiet_mode = False
    worker.standby_mode = False
    worker.bus = type("_B", (), {
        "log": _Log(), "state": _Log(),
    })()
    worker.spoken = []
    worker._say = lambda text: worker.spoken.append(text)
    return worker


def test_going_to_standby_makes_him_silent_not_merely_unprompted():
    """quiet_mode alone still answered. The ask was: say nothing at all."""
    worker = _worker()
    assert worker._handle_quiet_command("orion go to standby") is True
    assert worker.standby_mode is True
    assert worker._held_in_standby("what is the weather") is True


def test_standby_collapses_him_to_the_orb():
    """face3d's _ORB_STATES turns STANDBY into the orb form, so the visual
    matches the behaviour rather than a materialised ORION sitting mute."""
    worker = _worker()
    worker._handle_quiet_command("orion, standby")
    assert "STANDBY" in worker.bus.state.lines


def test_the_wake_phrase_the_user_actually_uses_releases_him(monkeypatch):
    from orion_core import audio_devices
    ready = audio_devices.DeviceCheck(
        "output", "test", True, True, 1, "test output", "output verified")
    ready_in = audio_devices.DeviceCheck(
        "input", "test", True, True, 2, "test input", "input verified")
    monkeypatch.setattr(audio_devices, "verify",
                        lambda kind, spec="", **_k: ready if kind == "output" else ready_in)
    monkeypatch.setattr(audio_devices, "preferred_index", lambda kind: 1 if kind == "output" else 2)
    worker = _worker()
    worker._handle_quiet_command("go into standby")
    assert worker._handle_quiet_command("orion i need you") is True
    assert worker.standby_mode is False
    assert worker.quiet_mode is False
    assert "LISTENING" in worker.bus.state.lines


def test_the_wake_phrase_is_never_swallowed_by_standby():
    """Standby is full silence, so nothing else will release it — if the wake
    phrase were held back too, he could not be brought out at all."""
    worker = _worker()
    worker._handle_quiet_command("standby")
    assert worker._held_in_standby("orion i need you") is False


def test_a_power_command_still_reaches_him_in_standby():
    """Otherwise standby would be a state ORION cannot be shut down from."""
    worker = _worker()
    worker._handle_quiet_command("standby")
    assert worker._held_in_standby("orion shut down") is False


def test_nothing_is_held_back_when_he_is_not_in_standby():
    assert _worker()._held_in_standby("what is the weather") is False


def test_he_acknowledges_once_and_then_stops():
    worker = _worker()
    worker._handle_quiet_command("orion go to standby")
    from orion_core.voice_variety import BANK
    # One short acknowledgement, phrased afresh each time (voice_variety).
    assert len(worker.spoken) == 1 and worker.spoken[0] in BANK["standing_by"]


def test_explicit_standby_disables_microphone_capture():
    worker = _worker()

    class _Mic:
        def __init__(self): self.enabled = True; self.drained = False
        def set_enabled(self, enabled): self.enabled = enabled
        def _drain(self): self.drained = True

    mic = _Mic()
    worker.microphone_enabled = True
    worker.mic = mic
    worker.tool_busy = True
    worker._clear_turn = lambda: None

    worker._handle_quiet_command("go into standby")

    assert worker.standby_mode is True
    assert worker.microphone_enabled is False
    assert mic.enabled is False
    assert mic.drained is True


# ── observability ORION can use himself (Mark XXXV W1.5) ─────────────────────
#
# "Why did you stop talking at 23:41?" must be answered from his own records,
# with the exact lines quoted. The session log now stamps every line with the
# full date (the time alone was ambiguous across midnight and the rotated
# files) and redacts secrets and e-mail addresses before they reach the disk.
# Every fixture below is synthetic: no owner file is ever read by a test.

import json  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from datetime import date, datetime  # noqa: E402

_STAMPED = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}  \S")
_FAKE_KEY = "AIza" + "SyFAKE" + "k3y" * 10
_FAKE_EMAIL = "owner.person@example.com"
_NOW = datetime(2026, 10, 10, 9, 30)
# The shape latency.reentry_line writes (Mark XXXV): it names the task that died.
_LOOP_LINE = ("2026-10-09 23:41:07  LOOP: Cannot enter into task 'orion-live-receive' "
              "(LiveWorker._receive_loop) - 'tool:camera' (Dispatcher.dispatch) pumped the "
              "Windows message queue inside its own step at open_camera() at "
              "camera_devices.py:72; 'orion-live-receive' (LiveWorker._receive_loop) may never resume.")
_NET_1011 = ("2026-10-09 23:41:09  NET: WebSocket error 1011 (server error) \u2014 1011 None. "
             "Internal error encountered.")
_BACK = "2026-10-09 23:51:14  NET: gemini live channel synchronised."
_OLD_PERF = "21:15:30  PERF: event loop blocked 2400 ms by open() at audio.py:3047"
_FORCED = "2026-10-10 00:02:38  FORCED by the shutdown watchdog after 25.0 s."


def _journal(**row):
    return json.dumps(row, default=str)


_JOURNAL_1011 = _journal(at="2026-10-09T23:41:09", event="error", provider="gemini",
                         code=1011, type="APIError", established=True, seconds=664.0,
                         summary="WebSocket error 1011 (server error) \u2014 1011 None. "
                                 "Internal error encountered.")


def _incident_fixture(root):
    """A config dir modelled on the real shapes: an old time-only log, a
    stamped evening log, a rotation just before midnight, the Live journal,
    a forced shutdown and one crash report."""
    from orion_core.self_diagnosis import Sources
    diag, crashes = root / "diagnostics", root / "crash_reports"
    diag.mkdir(parents=True)
    crashes.mkdir(parents=True)
    (diag / "session.log.2").write_text("\n".join([          # before the upgrade
        "20:58:00  ===== session 2026-10-09 20:58:00 (pid 4100) =====",
        "20:58:03  NET: gemini live channel synchronised.",
        _OLD_PERF,
        "21:15:33  FACE: WebGL context lost x1; renderer stalled x2 (longest 950 ms)",
        "22:29:50  SYS: shutdown requested by user.",
    ]) + "\n", encoding="utf-8")
    (diag / "session.log.1").write_text("\n".join([
        "2026-10-09 22:30:00  ===== session 2026-10-09 22:30:00 (pid 4242) =====",
        "2026-10-09 22:30:05  NET: gemini live channel synchronised.",
        "2026-10-09 23:20:11  NET: Error code 1006 \u2014 no close frame received or sent",
        "2026-10-09 23:38:40  YOU: what's the weather tomorrow",
        "2026-10-09 23:38:43  ORION: Tomorrow looks dry, around fourteen degrees.",
        "2026-10-09 23:40:58  VOICE: resumed.",
        "2026-10-09 23:41:02  TOOL: camera_tools requested.",
        _LOOP_LINE,
        _NET_1011,
        "2026-10-09 23:42:30  SYS: Command Deck \u2192 Library.",
        "2026-10-09 23:44:12  PERF: event loop blocked 412 ms in read() at camera.py:140",
        # Written before redaction existed: the reader must redact it too.
        f"2026-10-09 23:45:00  NET: provider check failed for key={_FAKE_KEY} ({_FAKE_EMAIL})",
        "2026-10-09 23:51:12  NET: live channel closed; re-establishing.",
        _BACK,
        "2026-10-09 23:51:20  VOICE: resumed.",
        "2026-10-09 23:59:58  NET: server requested reconnect (go_away); rotating channel.",
    ]) + "\n", encoding="utf-8")
    (diag / "session.log").write_text("\n".join([          # rotated just before midnight
        "2026-10-10 00:00:01  NET: gemini live channel synchronised.",
        "2026-10-10 00:02:11  SYS: shutdown requested by user.",
        "2026-10-10 00:02:13  SYS: farewell speech failed - timeout",
    ]) + "\n", encoding="utf-8")
    (diag / "live_sessions.jsonl").write_text("\n".join([
        _journal(at="2026-10-09T22:30:05", event="connected", provider="gemini",
                 model="models/live", resumed=False, stage=0),
        _journal(at="2026-10-09T23:20:11", event="error", provider="gemini", code=1006,
                 type="ConnectionClosedError", established=True, seconds=3006.0,
                 summary="Error code 1006 \u2014 no close frame received or sent"),
        _JOURNAL_1011,
        _journal(at="2026-10-09T23:51:14", event="connected", provider="gemini",
                 model="models/live", resumed=True, stage=0),
        _journal(at="2026-10-10T00:02:12", event="closed", provider="gemini",
                 seconds=658.0, reason="shutdown"),
    ]) + "\n", encoding="utf-8")
    (diag / "last_shutdown.txt").write_text(
        _FORCED + "\nStill inside: worker > microphone\n"
        "3 traced step(s), 24.1 s between them. Slowest first:\n"
        "   22.00 s  worker > microphone\n", encoding="utf-8")
    (crashes / "faulthandler.log").write_text(
        "\n===== faulthandler armed 20261009_223000_000 =====\n", encoding="utf-8")
    (crashes / "20261010_000240_100_native-exit.txt").write_text(
        "O.R.I.O.N. CRASH REPORT\n" + "=" * 60 + "\n"
        "time     : 2026-10-10 00:02:40 (recorded 2026-10-10 09:00:01)\n"
        "kind     : native-exit\n"
        "summary  : Windows fatal exception: access violation in orion_core/audio.py:3162 close.\n",
        encoding="utf-8")
    return Sources.under(root)


# -- the session log itself --

def test_every_session_log_line_carries_the_full_date():
    from orion_core import session_log
    at = datetime(2026, 10, 9, 23, 41, 7).timestamp()
    line = session_log.format_line(at, "NET: live channel closed; re-establishing.")
    assert line == "2026-10-09 23:41:07  NET: live channel closed; re-establishing."
    assert _STAMPED.match(line)


def test_a_multi_line_message_keeps_one_stamp():
    """Its other lines are indented, so no reader mistakes them for entries."""
    from orion_core import session_log
    line = session_log.format_line(0.0, "DEBUG: trace\nTraceback (most recent call last):\n  x")
    first, *rest = line.split("\n")
    assert _STAMPED.match(first)
    assert rest and all(part.startswith("    ") for part in rest)


@pytest.mark.parametrize("secret", [
    _FAKE_KEY,
    "AQ." + "Ab8RN6Lz" + "x9Y8z7W6v5U4t3S2r1Q0p",
    "sk-ant-" + "api03-" + "a1B2c3D4e5F6g7H8i9J0",
    "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1",
    "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
    "github_pat_" + "11ABCDEFG0123456789_abcdefghij",
    "xoxb-" + "123456789012-abcdefABCDEF",
    "eyJhbGciOiJIUzI1NiJ9" + ".eyJzdWIiOiIxMjM0NTY3ODkwIn0" + ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9",
])
def test_a_secret_shape_never_reaches_the_session_log(secret):
    from orion_core.session_log import redact
    out = redact(f"NET: provider said {secret} was refused")
    assert secret not in out and "<redacted>" in out
    assert out.startswith("NET: provider said ") and out.endswith(" was refused")


@pytest.mark.parametrize("line, secret", [
    ("NET: GET https://example.com/v1/models?key=abc123def456&alt=sse", "abc123def456"),
    ("NET: Authorization: Bearer abc.def-123456789", "abc.def-123456789"),
    ("NET: retrying with Bearer 0a1b2c3d4e5f6a7b8c", "0a1b2c3d4e5f6a7b8c"),
    ('TOOL: config {"password": "hunter22", "user": "x"}', "hunter22"),
    ("SYS: token=0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef"),
    ("SYS: client_secret: s3cr3tValue99", "s3cr3tValue99"),
    ('SYS: {"api_key": "plainvalue123"}', "plainvalue123"),
    (f"SYS: signed in as {_FAKE_EMAIL}", _FAKE_EMAIL),
])
def test_a_secret_in_a_credential_context_is_redacted(line, secret):
    from orion_core.session_log import redact
    out = redact(line)
    assert secret not in out
    assert "<redacted>" in out or "<email>" in out


@pytest.mark.parametrize("line, exposed", [
    ("NET: Authorization: Basic YTo=", ("YTo=",)),
    ("NET: Authorization: Negotiate c3ludGhldGljLW9ubHk=", ("c3ludGhldGljLW9ubHk=",)),
    ("NET: authorization=NTLM c3ludGhldGljLXBvb2Y=", ("c3ludGhldGljLXBvb2Y=",)),
    ("NET: Authorization: CustomProof opaque-proof-value", ("opaque-proof-value",)),
    ('SYS: {"Authorization": "Negotiate synthetic proof tail"}', ("synthetic", "proof", "tail")),
    ("SYS: {'Authorization': 'NTLM synthetic proof tail'}", ("synthetic", "proof", "tail")),
    ('NET: Authorization: Digest username="synthetic-owner", nonce="synthetic-nonce", response="synthetic-proof"',
     ("synthetic-owner", "synthetic-nonce", "synthetic-proof")),
    ('SYS: {"password": "ab"}', ("ab",)),
    ('SYS: {"client_secret": "first second third"}', ("first", "second", "third")),
    ("SYS: password='first second third'", ("first", "second", "third")),
    ('SYS: {"client_secret": "first&second"}', ("first", "second")),
])
def test_explicit_credentials_are_redacted_completely_even_when_short_or_quoted(line, exposed):
    """W1.5: explicit credential fields must not leave a quoted value's tail on disk."""
    from orion_core.session_log import redact
    out = redact(line)
    assert "<redacted>" in out
    assert all(part not in out for part in exposed)


@pytest.mark.parametrize("surface", ["format_line", "diagnostic_quote"])
def test_authorization_value_is_redacted_in_written_and_historical_evidence(surface):
    from orion_core import session_log, self_diagnosis
    secret = "c3ludGhldGljLWV2aWRlbmNl"
    line = f"NET: Authorization: Negotiate {secret}"
    at = datetime(2026, 10, 12, 12, 0)
    if surface == "format_line":
        out = session_log.format_line(at.timestamp(), line)
    else:
        out = self_diagnosis._quote(self_diagnosis.Event(at, "session.log", line))
    assert secret not in out
    assert "Authorization:" in out and "<redacted>" in out


@pytest.mark.parametrize("line", [
    r"FILE: saved C:\Users\owner\ORION\config\diagnostics\session.log",
    "FORGE: loaded chess_nnue_3fa9c2d1e8b7a6f5c4d3e2f1a0b9c8d7e6f5a4b3.bin",
    "PERF: event loop blocked 412 ms in open() at camera.py:88",
    "===== session 2026-10-09 23:41:07 (pid 4242) =====",
    "TRACE: ORION_REQ_000123 reached LISTENING_RESTORED in 812 ms",
    "TOOL: token_usage ok; tokens: 1200 of max_tokens=8192",
    "FACE: loaded avatar@2x.png texture",
    "SYS: request 123e4567-e89b-12d3-a456-426614174000 done at 23:41:07",
    "NET: Error code 1008 \u2014 1008 None. Request is missing required authentication credential.",
    "AUDIO: microphone switched to [3] Headset (task-manager)",
    "SYS: bearer instruments online; commit 3fa9c2d; python 3.13",
    "NET: authorization policy checks passed; no credentials were supplied",
    "SYS: max_authorization_tokens=4; not an HTTP header",
])
def test_redaction_leaves_ordinary_log_text_alone(line):
    from orion_core.session_log import redact
    assert redact(line) == line


def test_redaction_runs_on_the_writer_thread_not_the_callers(tmp_path, monkeypatch):
    """write() runs on the GUI/asyncio thread and must stay cheap."""
    from orion_core import session_log
    threads: list[str] = []
    real = session_log.redact

    def spy(text):
        threads.append(threading.current_thread().name)
        return real(text)

    monkeypatch.setattr(session_log, "redact", spy)
    path = session_log.enable(tmp_path / "session.log")
    try:
        session_log.write(f"NET: provider refused {_FAKE_KEY} for {_FAKE_EMAIL}")
        session_log.flush()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and "refused" not in (
                path.read_text(encoding="utf-8") if path.exists() else ""):
            time.sleep(0.05)
        text = path.read_text(encoding="utf-8")
    finally:
        session_log._enabled = False
    assert threads and set(threads) == {"orion-session-log"}
    assert _FAKE_KEY not in text and _FAKE_EMAIL not in text
    assert "<redacted>" in text and "<email>" in text
    assert all(_STAMPED.match(line) for line in text.splitlines())


def test_rotation_still_keeps_three_old_files(tmp_path):
    from orion_core import session_log
    for name, text in (("session.log", "current"), ("session.log.1", "one"),
                       ("session.log.2", "two"), ("session.log.3", "three")):
        (tmp_path / name).write_text(text)
    session_log._rotate(tmp_path / "session.log")
    assert not (tmp_path / "session.log").exists()
    assert [(tmp_path / f"session.log.{n}").read_text() for n in (1, 2, 3)] == [
        "current", "one", "two"]
    assert (session_log.KEEP, session_log.MAX_BYTES) == (3, 2_000_000)


# -- when the question is about --

@pytest.mark.parametrize("said, now, start, end, focus", [
    ("why did you stop talking at 23:41?", _NOW,
     datetime(2026, 10, 9, 23, 35), datetime(2026, 10, 9, 23, 56), datetime(2026, 10, 9, 23, 41)),
    ("23:41", datetime(2026, 10, 10, 23, 50),
     datetime(2026, 10, 10, 23, 35), datetime(2026, 10, 10, 23, 56), datetime(2026, 10, 10, 23, 41)),
    ("at 23:41", datetime(2026, 10, 10, 23, 41, 30),        # this very minute counts
     datetime(2026, 10, 10, 23, 35), datetime(2026, 10, 10, 23, 56), datetime(2026, 10, 10, 23, 41)),
    ("what happened last night around 11:40", _NOW,
     datetime(2026, 10, 9, 23, 25), datetime(2026, 10, 9, 23, 55), datetime(2026, 10, 9, 23, 40)),
    ("11:40 pm", _NOW,
     datetime(2026, 10, 9, 23, 34), datetime(2026, 10, 9, 23, 55), datetime(2026, 10, 9, 23, 40)),
    ("11:40", _NOW,                         # 12-hour: the most recent past reading
     datetime(2026, 10, 9, 23, 34), datetime(2026, 10, 9, 23, 55), datetime(2026, 10, 9, 23, 40)),
    ("at 9:05 this morning", _NOW,
     datetime(2026, 10, 10, 8, 59), datetime(2026, 10, 10, 9, 20), datetime(2026, 10, 10, 9, 5)),
    ("on 8 oct at 14:05", _NOW,
     datetime(2026, 10, 8, 13, 59), datetime(2026, 10, 8, 14, 20), datetime(2026, 10, 8, 14, 5)),
    ("what went wrong just now", _NOW,
     datetime(2026, 10, 10, 9, 20), _NOW, None),
    ("why did you freeze earlier", _NOW,
     datetime(2026, 10, 10, 5, 30), _NOW, None),
    ("what happened last night", _NOW,
     datetime(2026, 10, 9, 18, 0), datetime(2026, 10, 10, 6, 0), None),
    ("yesterday", _NOW,
     datetime(2026, 10, 9), datetime(2026, 10, 10), None),
    ("", _NOW, datetime(2026, 10, 10, 9, 0), _NOW, None),
])
def test_the_time_asked_about_resolves_to_a_window(said, now, start, end, focus):
    from orion_core.self_diagnosis import resolve_window
    window = resolve_window(said, now)
    assert (window.start, window.end, window.focus) == (start, end, focus)


def test_minutes_ago_centres_on_that_moment():
    from orion_core.self_diagnosis import resolve_window
    window = resolve_window("why did you go quiet 10 minutes ago", _NOW)
    assert window.focus == datetime(2026, 10, 10, 9, 20)
    assert window.start < window.focus < window.end <= _NOW


# -- old, time-only logs --

def test_time_only_lines_take_their_date_from_the_header_across_midnight():
    from orion_core.self_diagnosis import parse_session_log
    text = "\n".join([
        "23:50:00  ===== session 2026-10-08 23:50:00 (pid 7) =====",
        "23:58:10  NET: gemini live channel synchronised.",
        "00:03:00  VOICE: resumed.",
        "    a continuation line",
        "00:04:00  PERF: event loop blocked 900 ms by read() at mic.py:1",
    ])
    events, carry = parse_session_log(text, "session.log.1")
    assert [e.at for e in events] == [
        datetime(2026, 10, 8, 23, 50), datetime(2026, 10, 8, 23, 58, 10),
        datetime(2026, 10, 9, 0, 3), datetime(2026, 10, 9, 0, 3), datetime(2026, 10, 9, 0, 4)]
    assert all(e.inferred for e in events)
    assert carry == (date(2026, 10, 9), 240)
    # the next, newer file continues from where this one stopped
    more, _ = parse_session_log("00:10:00  VOICE: resumed.", "session.log", carry=carry)
    assert more[0].at == datetime(2026, 10, 9, 0, 10)


def test_an_undated_prefix_is_dated_back_from_a_header_or_the_file_time():
    from orion_core.self_diagnosis import parse_session_log
    events, _ = parse_session_log(
        "23:59:00  VOICE: resumed.\n00:01:00  ===== session 2026-10-09 00:01:00 (pid 8) =====", "x")
    assert events[0].at == datetime(2026, 10, 8, 23, 59)
    modified = datetime(2026, 10, 9, 0, 30).timestamp()
    events, _ = parse_session_log("23:59:00  VOICE: a\n00:20:00  VOICE: b", "x", mtime=modified)
    assert [e.at for e in events] == [datetime(2026, 10, 8, 23, 59), datetime(2026, 10, 9, 0, 20)]


@pytest.mark.parametrize("line, detail", [
    ("PERF: event loop blocked 412 ms in open() at camera.py:88",
     "412 ms|open() at camera.py:88"),
    ("PERF: event loop blocked 412 ms by open_stream() at audio.py:42",
     "412 ms|open_stream() at audio.py:42"),
    ("PERF: event loop blocked 180 ms by open_stream() at audio.py:42  "
     "(+3 shorter block(s) not listed)",
     "180 ms|open_stream() at audio.py:42"),
    ("PERF: event loop STILL blocked after 3.0 s by open_stream() at audio.py:42",
     "over 3.0 s (still blocked when this line was written)|open_stream() at audio.py:42"),
])
def test_any_perf_line_is_read_as_a_loop_block_with_its_culprit(line, detail):
    """The watchdog's wording may change (lead, Mark XXXV): any PERF line with
    a duration is a loop block, and its culprit survives a trailing count."""
    from orion_core.self_diagnosis import classify
    kind, weight, found = classify(line)
    assert kind == "loop_block" and weight > 0
    assert found == detail


def test_a_retried_task_is_explained_as_held_up_not_dead():
    """From Mark XXXX.4 the refused step is retried (latency.retry_refused_step):
    the new line must not be read back as a task that died."""
    from datetime import datetime

    from orion_core.self_diagnosis import Event, _sentence, classify
    line = ("LOOP: Cannot enter into task 'orion-send-realtime' (LiveWorker._send_realtime) "
            "- 'orion-companion-forget' (CompanionPanel.forget) pumped the Windows message "
            "queue inside its own step at exec() at ops_deck.py:900; 'orion-send-realtime' "
            "(LiveWorker._send_realtime) is stepped again once that step ends.")
    kind, weight, detail = classify(line)
    assert kind == "reentry" and detail.startswith("'orion-send-realtime'")
    said = _sentence(Event(at=datetime(2026, 10, 28, 9, 43), source="session.log", text=line,
                           kind=kind, weight=weight, detail=detail))
    assert "held up" in said and "died" not in said


@pytest.mark.parametrize("line, kind", [
    # the shapes ORION's own code writes (live_worker, audio, face, latency)
    ("NET: server requested reconnect (go_away); rotating channel.", "go_away"),
    ("NET: live channel closed; re-establishing.", "channel_closed"),
    ("NET: temporary connection issue; retrying shortly.", "channel_closed"),
    ("NET: gemini live channel synchronised.", "reconnect"),
    ("NET: Error code 1008 — 1008 None. Request is missing required authentication "
     "credential.", "live_error"),
    ("VOICE: Live audio stopped arriving; the voice channel was released.", "voice_released"),
    ("NET: no live audio provider available; text fallback is armed.", "no_provider"),
    ("AUDIO: microphone switched to [3] Headset.", "audio_device"),
    ("TOOL: camera_tools failed - timeout", "tool_error"),
    ("FACE: WebGL context lost x1; renderer stalled x2 (longest 950 ms)", "face"),
    ("AUDIO: leaving standby.", "reconnect"),
    ("ORION: voice muted — my replies will appear as text.", "muted"),
    ("REPAIR: ignoring a scheduling fault in the repair machinery itself (RuntimeError: "
     "Cannot enter into task <Task pending name='x'>) - not a defect in ORION's code.", "reentry"),
    ("SYS: shutdown requested by user.", "shutdown"),
    ("SYS: Command Deck → Library.", ""),
    ("YOU: what's the weather tomorrow", ""),
    ("VOICE: Live never answered 'a riot shut down for now please' (heard here 6s ago; Live "
     "was sent 7.9 s of the microphone, loudest -18 dBFS) - sending it as a text turn.",
     "live_missed"),
    ("NET: refreshing the live channel - it stopped hearing the microphone.", "live_refresh"),
])
def test_real_line_shapes_are_classified(line, kind):
    from orion_core.self_diagnosis import classify
    assert classify(line)[0] == kind


def test_a_missed_request_is_explained_with_what_reached_the_cloud():
    from orion_core.self_diagnosis import Event, _sentence, classify
    line = ("VOICE: Live never answered 'how much have you progressed' (heard here 6s ago; the "
            "microphone was not reaching Live (a tool was running)) - sending it as a text turn.")
    kind, weight, detail = classify(line)
    said = _sentence(Event(at=datetime(2026, 10, 28, 16, 56), source="session.log", text=line,
                           kind=kind, weight=weight, detail=detail))
    assert said.startswith("At 16:56:00 my cloud voice missed 'how much have you progressed'")
    assert "a tool was running" in said


def test_a_long_freeze_outranks_short_blocks_and_repeats_are_not_all_quoted(tmp_path):
    from orion_core import self_diagnosis
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    lines = [f"2026-10-10 09:1{n % 10}:0{n % 6}  PERF: event loop blocked {120 + n} ms by "
             f"tick() at face3d.py:2615" for n in range(12)]
    lines += [
        "2026-10-10 09:14:00  PERF: event loop STILL blocked after 3.0 s by open_stream() at audio.py:42",
        "2026-10-10 09:14:02  PERF: event loop blocked 5210 ms by open_stream() at audio.py:42",
    ]
    (diag / "session.log").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    answer = self_diagnosis.explain("why did you freeze at 9:14",
                                    sources=self_diagnosis.Sources.under(tmp_path), now=_NOW)
    head, _, quoted = answer.partition("Exact log lines")
    assert "blocked for 5210 ms by open_stream() at audio.py:42" in head
    assert "worst 5210 ms" in head
    assert "STILL blocked after 3.0 s" in quoted
    assert quoted.count("tick() at face3d.py:2615") <= 3 and "near-identical" in quoted


# -- asked about a recent incident, ORION quotes his own log correctly --

def test_asked_why_he_stopped_talking_he_names_the_cause_and_quotes_the_lines(tmp_path):
    from orion_core import self_diagnosis
    answer = self_diagnosis.explain("why did you stop talking at 23:41?",
                                    sources=_incident_fixture(tmp_path), now=_NOW)
    head, marker, quoted = answer.partition("Exact log lines")
    assert marker, answer
    # the cause, in plain English, before any quote: the LOOP line names the
    # task that died, which is the strongest evidence there is for silence
    assert "Cannot enter into task" in head and "orion-live-receive" in head
    assert "receives my Live voice channel" in head and "died" in head
    assert "open_camera() at camera_devices.py:72" in head
    assert "nothing said to me was heard or answered" in head
    assert "1011" in head                                       # the server error alongside
    assert "23:40:58 to 23:51:20" in head and "10 minutes" in head   # the silence
    assert "23:51:14" in head                                    # and the recovery
    assert "it was re-established at 23:51:14" in head           # a close, with its reconnect
    # the exact lines, verbatim, with their timestamps
    for line in (_LOOP_LINE, _NET_1011, _BACK, _JOURNAL_1011):
        assert line in quoted, line
    # nothing unrelated: an earlier error, routine chatter, the later shutdown
    for unrelated in ("23:20:11", "Command Deck", "FORCED", "00:02:"):
        assert unrelated not in quoted, unrelated
    # and never the key or the address, although the old file holds both
    assert _FAKE_KEY not in answer and "AIza" not in answer and _FAKE_EMAIL not in answer
    assert len(answer) <= self_diagnosis.MAX_CHARS
    assert sum(1 for line in quoted.splitlines() if line.startswith("- ")) <= self_diagnosis.MAX_LINES


def test_asked_about_the_exit_he_names_the_forced_shutdown(tmp_path):
    from orion_core import self_diagnosis
    answer = self_diagnosis.explain("why did you not close properly", when="00:02",
                                    sources=_incident_fixture(tmp_path), now=_NOW)
    head = answer.partition("Exact log lines")[0]
    assert "FORCED" in head and "worker > microphone" in head
    assert _FORCED in answer and "Cannot enter into task" not in answer


def test_an_old_format_line_is_quoted_with_its_inferred_date(tmp_path):
    from orion_core import self_diagnosis
    answer = self_diagnosis.explain("why did you freeze at 21:15?",
                                    sources=_incident_fixture(tmp_path), now=_NOW)
    head = answer.partition("Exact log lines")[0]
    assert "2400 ms" in head and "open() at audio.py:3047" in head
    assert f"session.log.2, date inferred 2026-10-09 | {_OLD_PERF}" in answer


def test_when_nothing_explains_it_he_says_so_and_shows_the_nearest_evidence(tmp_path):
    from orion_core import self_diagnosis
    answer = self_diagnosis.explain("what happened at 22:00?",
                                    sources=_incident_fixture(tmp_path), now=_NOW)
    assert answer.startswith("I found nothing") and "will not guess" in answer
    assert "Most likely" not in answer and "most likely" not in answer
    assert "Nearest notable evidence" in answer
    assert "Searched:" in answer and "session.log.2" in answer


def _exit_fault_fixture(root):
    """A shutdown at 20:28:50, a channel closing with it, and a native fault on
    the way out - the faulthandler dump written as Windows writes it (CRLF)."""
    import os
    diag, crashes = root / "diagnostics", root / "crash_reports"
    diag.mkdir(parents=True)
    crashes.mkdir(parents=True)
    (diag / "session.log").write_bytes("\r\n".join([
        "2026-10-10 20:21:25  ===== session 2026-10-10 20:21:25 (pid 9001) =====",
        "2026-10-10 20:27:40  ORION: Of course.",
        "2026-10-10 20:28:50  SYS: shutdown requested by user.",
        "2026-10-10 20:28:54  NET: live channel closed; re-establishing.",
    ]).encode("utf-8") + b"\r\n")
    fault = crashes / "faulthandler.log"
    fault.write_bytes("\r\n".join([
        "", "===== faulthandler armed 20261010_202125_233 =====",
        "Windows fatal exception: access violation", "",
        "Thread 0x00002b98 (most recent call first):",
        '  File "orion_core\\app.py", line 360 in _force_exit',
        '  File "threading.py", line 1012 in run', "",
        "Current thread 0x000037ac (most recent call first):",
        "  <no Python frame>", ""]).encode("utf-8"))
    stamp = datetime(2026, 10, 10, 20, 28, 56).timestamp()
    os.utime(fault, (stamp, stamp))
    from orion_core.self_diagnosis import Sources
    return Sources.under(root)


def test_a_windows_written_faulthandler_dump_is_read(tmp_path):
    """Bytes read as they are keep each CRLF's '\\r', and faulthandler's
    '...=====$' session marker then never matched: a real access violation
    read back as '0 fatal dump(s)'."""
    from orion_core import self_diagnosis
    answer = self_diagnosis.explain("why did you crash at 20:28",
                                    sources=_exit_fault_fixture(tmp_path),
                                    now=datetime(2026, 10, 10, 21, 0))
    head = answer.partition("Exact log lines")[0]
    assert "faulted natively while exiting" in head
    assert ("- faulthandler.log (session armed 2026-10-10 20:21:25) | "
            "Windows fatal exception: access violation") in answer
    assert "1 fatal dump(s)" in answer


def test_an_older_dump_is_pinned_to_its_sessions_last_record(tmp_path):
    """Only the newest dump has a time. An older one lies between its
    session's start and the next start - hours or days - so it is placed
    after that session's last record; a handled first-chance COM exception is
    not a crash at all; and the session's own connect before it died is not
    reported as a recovery."""
    from orion_core import self_diagnosis
    diag, crashes = tmp_path / "diagnostics", tmp_path / "crash_reports"
    diag.mkdir()
    crashes.mkdir()
    (crashes / "faulthandler.log").write_text("\n".join([
        "", "===== faulthandler armed 20261009_100000_000 =====",
        "Windows fatal exception: code 0x8001010d", "",
        "Current thread 0x000011aa (most recent call first):", "  <no Python frame>", "",
        "===== faulthandler armed 20261009_120000_000 =====",
        "Windows fatal exception: access violation", "",
        "Current thread 0x000037ac (most recent call first):",
        '  File "C:\\ORION\\orion_core\\audio.py", line 3047 in _open',
        '  File "C:\\ORION\\orion_core\\audio.py", line 3120 in _start', "",
        "===== faulthandler armed 20261009_180000_000 =====", ""]), encoding="utf-8")
    (diag / "live_sessions.jsonl").write_text("\n".join([
        _journal(at="2026-10-09T12:00:05", event="connected", provider="gemini",
                 model="models/live", resumed=False, stage=0),
        _journal(at="2026-10-09T12:09:05", event="closed", provider="gemini",
                 seconds=540.0, reason="go_away"),
        _journal(at="2026-10-09T12:09:06", event="connected", provider="gemini",
                 model="models/live", resumed=True, stage=0),
        _journal(at="2026-10-09T18:00:04", event="connected", provider="gemini",
                 model="models/live", resumed=False, stage=0),
    ]) + "\n", encoding="utf-8")
    answer = self_diagnosis.explain("why did you crash at 12:15",
                                    sources=self_diagnosis.Sources.under(tmp_path),
                                    now=datetime(2026, 10, 9, 20, 0))
    head = answer.partition("Exact log lines")[0]
    assert head.splitlines()[1].startswith("A fatal native error ended my session that "
                                           "started 12:00 on Fri 9 Oct")
    assert "some time from 12:09 to 12:19 on Fri 9 Oct" in head
    assert "orion_core/audio.py:3047 _open" in head
    assert "Recovery" not in head
    assert "0x8001010d" not in answer and "faulthandler.log (1 fatal dump(s))" in answer


def test_what_a_shutdown_causes_is_not_offered_as_the_cause(tmp_path):
    """The channel closing at shutdown is not 'had to be re-established', and
    a fault on the way out does not outrank the shutdown that preceded it."""
    from orion_core import self_diagnosis
    answer = self_diagnosis.explain("why did you stop talking at 20:28",
                                    sources=_exit_fault_fixture(tmp_path),
                                    now=datetime(2026, 10, 10, 21, 0))
    head = answer.partition("Exact log lines")[0]
    assert head.splitlines()[1].startswith("At 20:28:50 I was shutting down")
    assert "re-established" not in head
    assert "2026-10-10 20:28:54  NET: live channel closed; re-establishing." in answer


def test_an_empty_or_missing_log_dir_is_said_plainly(tmp_path):
    from orion_core import self_diagnosis
    for root in (tmp_path / "missing", tmp_path / "empty"):
        if root.name == "empty":
            (root / "diagnostics").mkdir(parents=True)
        answer = self_diagnosis.explain("what went wrong just now",
                                        sources=self_diagnosis.Sources.under(root), now=_NOW)
        assert answer.startswith("I have no records") and "will not guess" in answer
        assert "session.log (absent)" in answer


def test_a_huge_log_is_read_in_bounded_time(tmp_path):
    from orion_core import self_diagnosis
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    routine = "2026-10-10 08:00:00  SYS: Command Deck \u2192 Library.\n"
    with open(diag / "session.log", "w", encoding="utf-8") as handle:
        handle.write(routine * 500_000)                       # ~26 MB
        handle.write("2026-10-10 09:25:05  LOOP: Cannot enter into task 'orion-live-receive' "
                     "- 'Task-9' pumped the Windows message queue inside its own step at "
                     "open() at camera.py:88; it may never resume.\n")
    assert (diag / "session.log").stat().st_size > 8 * self_diagnosis.LOG_TAIL_BYTES
    started = time.perf_counter()
    answer = self_diagnosis.explain("what went wrong just now",
                                    sources=self_diagnosis.Sources.under(tmp_path), now=_NOW)
    elapsed = time.perf_counter() - started
    assert "Cannot enter into task" in answer.partition("Exact log lines")[0]
    assert elapsed < 5.0, f"read took {elapsed:.1f}s"


# -- the tool --

class _AnyBus:
    """Every attribute is a signal that records nothing."""

    def __getattr__(self, name):
        signal = type("_Signal", (), {"emit": lambda *a, **k: None,
                                      "connect": lambda *a, **k: None})()
        object.__setattr__(self, name, signal)
        return signal


def _dispatcher(diagnostics=None):
    from orion_core.dispatcher import OrionDispatcher
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _AnyBus()
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)
    d.diagnostics = diagnostics
    return d


def test_the_diagnostics_tool_answers_an_incident_end_to_end(tmp_path, monkeypatch):
    """Through dispatch() itself, with no diagnostics engine at all: reading
    the logs needs none."""
    from orion_core import self_diagnosis
    sources = _incident_fixture(tmp_path)
    monkeypatch.setattr(self_diagnosis, "default_sources", lambda: sources)
    monkeypatch.setattr(self_diagnosis, "_now", lambda: _NOW)
    result = asyncio.run(_dispatcher().dispatch("diagnostics", {
        "action": "incident", "when": "23:41",
        "question": "why did you stop talking at 23:41?"}))
    assert result.ok, result.text
    assert _LOOP_LINE in result.text and _JOURNAL_1011 in result.text
    assert _FAKE_KEY not in result.text and _FAKE_EMAIL not in result.text


def test_a_bare_why_is_still_the_capability_matrix(tmp_path, monkeypatch):
    from orion_core import self_diagnosis
    from orion_core.data import ToolResult

    class _Engine:
        async def capability_report(self, deep=True):
            return ToolResult("CAPABILITY MATRIX")

    sources = _incident_fixture(tmp_path)
    monkeypatch.setattr(self_diagnosis, "default_sources", lambda: sources)
    monkeypatch.setattr(self_diagnosis, "_now", lambda: _NOW)
    d = _dispatcher(_Engine())
    assert asyncio.run(d.diagnostics_tool({"action": "why"})).text == "CAPABILITY MATRIX"
    timed = asyncio.run(d.diagnostics_tool({"action": "why", "when": "23:41"}))
    assert _LOOP_LINE in timed.text
    for alias in ("what happened", "logs", "events"):
        assert _LOOP_LINE in asyncio.run(d.diagnostics_tool(
            {"action": alias, "question": "why did you stop talking at 23:41"})).text


def test_the_schema_teaches_the_model_to_ask_for_an_incident():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "diagnostics")
    first = tool["description"].split(". ")[0]
    assert "stopped talking" in first          # what find_tool shows of it
    assert "action='incident'" in tool["description"]
    assert {"when", "question"} <= set(tool["parameters"]["properties"])


# ── W1.3: the richer shutdown report is still read correctly ───────────────

def test_a_forced_report_with_loop_detail_still_names_the_step():
    """The watchdog's report now carries where the step was awaiting and
    whether the event loop was blocked; the diagnosis must still find the
    step it interrupted."""
    from orion_core.self_diagnosis import parse_shutdown_report
    from orion_core.shutdown_trace import ShutdownTrace

    now = [100.0]
    trace = ShutdownTrace(clock=lambda: now[0])
    trace.begin()
    with trace.phase("farewell (speaking the goodbye)"):
        now[0] += 3.0
    with trace.phase("remote gateway"):
        now[0] += 22.0
        trace.last_beat = now[0] - 20.0
        text = "2026-10-09 11:30:12  " + trace.report(forced=True)
    events = parse_shutdown_report(text, "last_shutdown.txt")
    assert events[0].kind == "forced_shutdown"
    assert events[0].detail == "remote gateway"
    assert "event loop: BLOCKED - it last ran 20.0 s ago" in text


def test_a_clean_report_with_a_step_given_up_is_still_a_clean_shutdown():
    from orion_core.self_diagnosis import parse_shutdown_report
    from orion_core.shutdown_trace import ShutdownTrace

    now = [100.0]
    trace = ShutdownTrace(clock=lambda: now[0])
    trace.begin()
    with trace.phase("MCP host"):
        now[0] += 3.0
        trace.note_gave_up(trace.current, 3.0, "mcp_host.py:1030 in stop()")
    text = "2026-10-11 13:40:00  " + trace.report()
    events = parse_shutdown_report(text, "last_shutdown.txt")
    assert events[0].kind == "shutdown"                 # not mistaken for FORCED
    assert "MCP host: gave up after 3.0 s; cancelled; waiting at mcp_host.py:1030" in text

# ── folded in from test_crash_last_real_fault.py (Mark XXXX.4: a real crash after handled COM dumps in the same session is) ────

from orion_core.crash_reporter import last_real_fault

COM = r"""Windows fatal exception: code 0x8001010e

Current thread 0x00001a2c (most recent call first):
  File "C:\orion\orion_core\media_session.py", line 50 in _now_playing
"""
AV = r"""Windows fatal exception: access violation

Current thread 0x00002b3c (most recent call first):
  File "C:\orion\orion_core\app.py", line 3495 in _shutdown
"""


def test_com_dumps_alone_are_not_a_crash():
    assert last_real_fault(COM * 3) is None


def test_an_access_violation_after_com_dumps_is_found():
    found = last_real_fault(COM * 40 + AV)
    assert found is not None and found["error"] == "access violation"


def test_an_access_violation_before_com_dumps_is_still_found():
    found = last_real_fault(AV + COM * 2)
    assert found is not None and found["error"] == "access violation"
