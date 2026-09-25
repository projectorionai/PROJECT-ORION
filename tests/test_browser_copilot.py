"""
Tests for the BrowserCopilot (#11) — ORION's real, visible-browser co-pilot.

These are hermetic: they never launch Chrome. They cover the pure JavaScript
builders (escaping, parameterisation), bus narration ("showing his workings"),
browser discovery, security guarding, and graceful degradation when no browser
is installed or no page is open.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.browser_copilot import (
    BrowserCopilot,
    js_click_text,
    js_fill,
    js_highlight,
    js_links,
    js_read_text,
    js_scroll_by,
    js_scroll_to_text,
    js_submit,
)
from orion_core.data import ToolResult


class _RecordingSignal:
    def __init__(self) -> None:
        self.emitted: list = []

    def emit(self, *args) -> None:
        self.emitted.append(args[0] if len(args) == 1 else args)

    def connect(self, *a, **k) -> None:
        pass


class _FakeBus:
    def __init__(self) -> None:
        self.log = _RecordingSignal()
        self.browser_step = _RecordingSignal()


def _copilot() -> tuple[BrowserCopilot, _FakeBus]:
    bus = _FakeBus()
    return BrowserCopilot(bus), bus


# ── pure JS builders ──────────────────────────────────────────────────────────

def test_scroll_and_read_builders_are_parameterised():
    assert "500" in js_scroll_by(500)
    assert "window.scrollBy" in js_scroll_by(500)
    assert "1234" in js_read_text(1234)
    # ints are coerced, so a non-int-looking float still renders a clean int
    assert "800" in js_scroll_by(800.0)


def test_text_builders_json_escape_their_input():
    # A hostile phrase with quotes/brackets must be JSON-escaped, never injected
    # raw into the script body.
    payload = 'both"; alert(1); //'
    js = js_scroll_to_text(payload)
    assert '"; alert(1)' not in js.replace(chr(92) + '"', "")  # no un-escaped break-out
    assert "alert(1)" in js  # the literal survives, but only inside a JSON string
    assert js_highlight(payload) == js_scroll_to_text(payload)


def test_click_and_fill_builders_emit_expected_hooks():
    click = js_click_text("Sign in")
    assert ".click()" in click
    assert "__orionHighlight" in click  # shows its workings on-screen

    fill = js_fill("email", "user@example.com")
    assert "dispatchEvent" in fill
    assert "'input'" in fill and "'change'" in fill
    assert "user@example.com" in fill

    assert "Enter" in js_submit()
    assert "href" in js_links()


# ── narration / "showing his workings" ────────────────────────────────────────

def test_narrate_emits_log_and_structured_step():
    cop, bus = _copilot()
    cop._current_url = "https://example.com"
    cop._narrate("Navigating", "https://example.com")
    assert any("Navigating" in str(line) for line in bus.log.emitted)
    step = bus.browser_step.emitted[-1]
    assert step["step"] == "Navigating"
    assert step["detail"] == "https://example.com"
    assert step["url"] == "https://example.com"


# ── discovery ─────────────────────────────────────────────────────────────────

def test_find_browser_honours_env_override(tmp_path):
    fake_exe = tmp_path / "chrome.exe"
    fake_exe.write_text("stub")
    cop, _ = _copilot()
    old = os.environ.get("ORION_BROWSER_PATH")
    os.environ["ORION_BROWSER_PATH"] = str(fake_exe)
    try:
        assert cop.find_browser() == str(fake_exe)
    finally:
        if old is None:
            os.environ.pop("ORION_BROWSER_PATH", None)
        else:
            os.environ["ORION_BROWSER_PATH"] = old


# ── input validation & security ───────────────────────────────────────────────

def test_open_requires_a_url():
    cop, _ = _copilot()
    result = asyncio.run(cop.open("   "))
    assert isinstance(result, ToolResult) and not result.ok


def test_security_guard_blocks_destructive_text():
    cop, _ = _copilot()
    # Guarding runs before any browser work, so no page is needed.
    result = asyncio.run(cop.fill("query", "rm -rf /important"))
    assert not result.ok
    assert "blocked" in result.text.lower()


# ── graceful degradation ──────────────────────────────────────────────────────

def test_actions_without_an_open_page_degrade_cleanly():
    cop, _ = _copilot()
    for coro in (
        cop.scroll(amount=200),
        cop.read(),
        cop.links(),
        cop.click("anything"),
        cop.highlight("anything"),
        cop.fill("q", "hello"),
        cop.submit(),
        cop.screenshot(),
    ):
        result = asyncio.run(coro)
        assert isinstance(result, ToolResult) and not result.ok


def test_missing_browser_returns_actionable_message(monkeypatch):
    cop, _ = _copilot()
    monkeypatch.setattr(cop, "find_browser", lambda: None)
    failure = asyncio.run(cop.ensure())
    assert isinstance(failure, ToolResult) and not failure.ok
    assert "chrome" in failure.text.lower()
    # open() surfaces the same actionable failure rather than raising.
    opened = asyncio.run(cop.open("https://example.com"))
    assert not opened.ok
