"""Mark XXXX.3: a private log keeps what ORION did, never what was said."""

from __future__ import annotations

import asyncio
import json

import pytest

from orion_core import session_log


@pytest.fixture
def private_off(tmp_path, monkeypatch):
    monkeypatch.setattr(session_log, "_privacy_path", lambda: tmp_path / "privacy.json")
    monkeypatch.setattr(session_log, "_private", False)
    return tmp_path


@pytest.mark.parametrize("line, kept", [
    ("YOU: Can you sing a song for me?", "YOU: [private]"),
    ("ORION: Okie-dokie, shutting down now.", "ORION: [private]"),
    ("ORION (proactive): Security note: something", "ORION (proactive): [private]"),
    ("TRACE: ORION_REQ_000002 TRANSCRIPTION_COMPLETE — can you sing", "TRACE: ORION_REQ_000002 TRANSCRIPTION_COMPLETE"),
    ("TRACE: ORION_REQ_000003 API_REQUEST_START —  Ryan, shut down", "TRACE: ORION_REQ_000003 API_REQUEST_START"),
    ("EMOTION: alert @ 0.25 — User expressed urgent (now)", "EMOTION: alert @ 0.25 [private]"),
    ("SPECIALIST: Reliability Agent DONE — The crash was caused by", "SPECIALIST: Reliability Agent DONE [private]"),
    ("COG: restored. Research sessions: camping_spots_devon", "COG: restored. Research sessions: [private]"),
    ("ORION[local]: here is the answer", "ORION[local]: [private]"),
    ("ORION[gemini_text]: sure thing", "ORION[gemini_text]: [private]"),
    ("PHONE: turn the lights off", "PHONE: [private]"),
    ("PHONE (voice): what's on today", "PHONE (voice): [private]"),
    ("VOICE: not answering 'pass the salt' - not addressed to ORION",
     "VOICE: not answering [private] - not addressed to ORION"),
])
def test_words_are_removed(line, kept):
    assert session_log.private_line(line) == kept


@pytest.mark.parametrize("line", [
    "TOOL: calculate requested.",
    "TRACE: ORION_REQ_000003 TOOL_EXECUTION_START — shutdown_orion",
    "NET: live channel closed; re-establishing.",
    "EMOTION: sad",
    "PERF: startup 8.6 s",
])
def test_what_he_did_is_kept(line):
    assert session_log.private_line(line) == line


def test_the_file_line_is_private_only_when_switched_on(private_off):
    assert "sing" in session_log.format_line(0, "YOU: sing for me")
    session_log.set_private(True)
    try:
        assert session_log.format_line(0, "YOU: sing for me").endswith("YOU: [private]")
        saved = json.loads((private_off / "privacy.json").read_text(encoding="utf-8"))
        assert saved == {"private_log": True}
        session_log._private = False
        assert session_log.load_private() is True
    finally:
        session_log.set_private(False)
    assert session_log.load_private() is False


def test_the_tool_switches_it(private_off):
    from orion_core.dispatch_files import FilesDispatchMixin as FileDispatchMixin

    holder = type("D", (FileDispatchMixin,), {})()
    holder.security = object()
    on = asyncio.run(holder.security_watch_tool({"action": "private_log", "enabled": True}))
    assert on.ok and session_log.private()
    status = asyncio.run(holder.security_watch_tool({"action": "private_log"}))
    assert "on" in status.text
    asyncio.run(holder.security_watch_tool({"action": "private_log_off"}))
    assert not session_log.private()
