"""
Shoulder (CAP-03) — read the active screen on request, describe it, act on it.

    "analyse images ... identify what course of action to take dependent on the
     prompt."
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.screen_read import ScreenReader, ScreenReading  # noqa: E402


PNG = b"\x89PNG\r\n\x1a\n fake image bytes"


def _reader(**kw):
    kw.setdefault("capture", lambda: PNG)
    return ScreenReader(**kw)


async def test_reads_and_describes_the_screen():
    async def describe(img, prompt):
        assert img == PNG
        return "A VS Code window with a Python traceback.", "model"

    r = await _reader(describe=describe).read("what's the error?")
    assert r.ok
    assert "traceback" in r.description.lower()
    assert r.via == "model"


async def test_prompt_drives_an_action():
    async def describe(img, prompt):
        return "A login form asking for an API key.", "model"

    async def generate(ask):
        assert "login form" in ask          # the description is handed to the model
        assert "API key" in ask
        return "Paste the key from your password manager into the field."

    r = await _reader(describe=describe, generate=generate).read("what do I do here?")
    assert r.ok
    assert "password manager" in r.action
    assert "password manager" in r.describe()


async def test_no_describer_degrades_gracefully():
    r = await _reader(describe=None).read("anything")
    assert not r.ok
    assert "no way to see" in r.note.lower()


async def test_capture_failure_is_reported_not_raised():
    def boom():
        raise RuntimeError("no display")
    r = await ScreenReader(capture=boom).read("x")
    assert not r.ok
    assert "capture" in r.note.lower()


async def test_empty_capture_is_handled():
    r = await ScreenReader(capture=lambda: b"").read("x")
    assert not r.ok


async def test_a_read_is_logged_once():
    logged = []

    async def describe(img, prompt):
        return "the desktop", "ocr"

    await _reader(describe=describe, log=logged.append).read("hi")
    assert any("screen" in m.lower() for m in logged)


async def test_no_passive_capture_without_a_call():
    # Constructing the reader must NOT capture anything — privacy model.
    calls = []
    ScreenReader(capture=lambda: calls.append(1) or PNG)
    assert calls == []      # nothing captured until read() is awaited


# ── the tool wiring ──────────────────────────────────────────────────────────

def test_tool_and_schema_present():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"screen_read"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "screen_read")
    assert "action" in tool["parameters"]["properties"]
    assert "prompt" in tool["parameters"]["properties"]
