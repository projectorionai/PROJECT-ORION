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
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

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


# ── latest-launch browser reliability regressions ────────────────────────────

class _WebControl:
    def __init__(self):
        self.calls = []
        self.failure = ""

    def _call(self, method, *args, **kwargs):
        self.calls.append((method, args, threading.get_ident()))
        return ToolResult(f"{method} refused" if self.failure == method else method,
                          ok=self.failure != method)

    def focus_window(self, title):
        return self._call("focus_window", title)

    def send_hotkeys(self, keys):
        return self._call("send_hotkeys", keys)

    def type_text(self, text):
        return self._call("type_text", text)

    def click(self, *args, **kwargs):
        return self._call("click", *args, **kwargs)

    def scroll(self, amount):
        return self._call("scroll", amount)

    def _foreground_title(self):
        return "Save As"


class _WebVision:
    def __init__(self):
        self.scans = []
        self.elements = []
        self.ocr_result = ToolResult("OCR unavailable", ok=False)
        self.fields = {}

    def _visible_window_titles(self):
        return ["Speedtest - Microsoft Edge"]

    def _detect_elements_sync(self, kinds, limit):
        self.scans.append((kinds, limit))
        return self.elements

    async def find_element(self, label, kinds="all"):
        assert kinds == "input"
        return self.fields.get(label)

    async def ocr(self, path):
        return self.ocr_result


class _WebVerifier:
    def __init__(self):
        self.calls = []
        self.result = ToolResult("clicked")

    async def _resolved_click(self, label, resolved, attempts):
        self.calls.append((label, resolved, attempts))
        return self.result


@pytest.fixture
def web_controller(monkeypatch):
    from orion_core.web import WebController
    async def instant(_seconds):
        pass
    monkeypatch.setattr(asyncio, "sleep", instant)
    return WebController(_FakeBus(), _WebControl(), _WebVision(), _WebVerifier())


def _web_element(name, role="Button"):
    return {"name": name, "role": role, "center": (100, 150),
            "rect": (80, 130, 40, 40), "enabled": True}


@pytest.mark.parametrize("action", ["accept_cookies", "close_popup", "read_page", "close_tab",
                                    "summarise_current", "download_current", "submit"])
def test_unfocused_browser_never_reads_or_sends_input(web_controller, action):
    web_controller.control.failure = "focus_window"
    result = asyncio.run(getattr(web_controller, action)())
    assert not result.ok
    assert all(call[0] == "focus_window" for call in web_controller.control.calls)
    assert web_controller.vision.scans == []


def test_cookie_labels_use_one_scan_and_accept_link_controls(web_controller):
    web_controller.vision.elements = [_web_element("Accept all", "Hyperlink")]
    assert asyncio.run(web_controller.accept_cookies()).ok
    assert len(web_controller.vision.scans) == 1
    assert web_controller.verifier.calls[0][1]["element"]["role"] == "Hyperlink"


def test_missing_cookie_prompt_does_not_repeat_browser_tree_scans(web_controller):
    assert not asyncio.run(web_controller.accept_cookies()).ok
    assert len(web_controller.vision.scans) == 1
    assert web_controller.verifier.calls == []


def test_uncertain_cookie_click_preserves_receipt_and_stops(web_controller):
    web_controller.vision.elements = [_web_element("Accept all"), _web_element("OK")]
    receipt = ToolResult("Click issued but unverified; inspect before retrying.", ok=False)
    web_controller.verifier.result = receipt
    assert asyncio.run(web_controller.accept_cookies()) is receipt
    assert len(web_controller.verifier.calls) == 1


def test_duplicate_cookie_controls_do_not_guess(web_controller):
    web_controller.vision.elements = [_web_element("Accept all"), _web_element("Accept all")]
    assert not asyncio.run(web_controller.accept_cookies()).ok
    assert web_controller.verifier.calls == []


def test_native_browser_launch_failure_is_reported(web_controller, monkeypatch):
    from orion_core import web
    monkeypatch.setattr(web, "open_url", lambda _url: False)
    assert not asyncio.run(web_controller.open("example.com")).ok


def test_navigation_preserves_keyboard_failure_and_never_types(web_controller):
    web_controller.control.failure = "send_hotkeys"
    result = asyncio.run(web_controller.navigate("example.com"))
    assert not result.ok and "refused" in result.text
    assert not any(call[0] == "type_text" for call in web_controller.control.calls)


def test_navigation_inputs_run_off_live_event_loop(web_controller):
    event_loop_thread = threading.get_ident()
    assert asyncio.run(web_controller.navigate("example.com")).ok
    inputs = [call for call in web_controller.control.calls if call[0] in {"type_text", "send_hotkeys"}]
    assert inputs and all(call[2] != event_loop_thread for call in inputs)


def test_partial_form_cannot_be_submitted(web_controller):
    web_controller.vision.fields = {"email": _web_element("email", "Edit")}
    result = asyncio.run(web_controller.fill_form({"email": "a@example.com", "name": "Ada"}, submit=True))
    assert not result.ok and "not submitted" in result.text
    assert not any(call[0] == "send_hotkeys" and call[1] == ("enter",)
                   for call in web_controller.control.calls)


def test_form_typing_failure_cannot_be_reported_as_success(web_controller):
    web_controller.vision.fields = {"email": _web_element("email", "Edit")}
    web_controller.control.failure = "type_text"
    result = asyncio.run(web_controller.fill_form({"email": "a@example.com"}, submit=True))
    assert not result.ok and "refused" in result.text
    assert not any(call[0] == "send_hotkeys" and call[1] == ("enter",)
                   for call in web_controller.control.calls)


def test_page_read_preserves_ocr_failure(web_controller):
    assert asyncio.run(web_controller.read_page()) is web_controller.vision.ocr_result


def test_narrated_scroll_uses_wheel_notches(web_controller):
    assert asyncio.run(web_controller._smooth_scroll_down()).ok
    assert [call[1] for call in web_controller.control.calls] == [(-1,)] * 5


def test_file_dialog_refuses_to_type_into_a_normal_page(web_controller, monkeypatch):
    monkeypatch.setattr(web_controller.control, "_foreground_title", lambda: "Example - Microsoft Edge")
    assert not asyncio.run(web_controller.handle_file_dialog("C:/Downloads/page.html")).ok
    assert web_controller.control.calls == []


def test_about_blank_is_not_converted_to_a_broken_https_url(web_controller):
    assert web_controller._normalise("about:blank") == "about:blank"
    assert web_controller._normalise("example.com") == "https://example.com"


@pytest.mark.parametrize("arguments", [
    {"action": "switch_tab", "index": "not a number"},
    {"action": "browse", "steps": "many"},
    {"action": "browse", "dwell": float("nan")},
    {"action": "fill_form", "fields": {}, "submit": "false"},
    {"action": "scroll", "amount": None},
])
def test_bad_web_arguments_return_failures_instead_of_escaping(web_controller, arguments):
    from orion_core.dispatch_web import WebDispatchMixin
    dispatcher = WebDispatchMixin()
    dispatcher.web = web_controller
    result = asyncio.run(dispatcher.web_control(arguments))
    assert not result.ok
    assert web_controller.control.calls == []


def test_confirm_dialog_alias_completes_the_advertised_download_workflow(web_controller):
    from orion_core.dispatch_web import WebDispatchMixin
    dispatcher = WebDispatchMixin()
    dispatcher.web = web_controller
    assert asyncio.run(dispatcher.web_control({"action": "confirm_dialog", "path": "C:/Downloads/page.html"})).ok
    assert web_controller.control.calls[-1][1] == ("enter",)


def test_closed_cdp_socket_is_reported_without_busy_polling():
    from aiohttp import WSMsgType
    class ClosedSocket:
        closed = False
        calls = 0
        async def send_str(self, _text):
            pass
        async def receive(self):
            self.calls += 1
            return SimpleNamespace(type=WSMsgType.CLOSED, data=None)
        async def close(self):
            self.closed = True
    cop, _ = _copilot()
    socket = cop._ws = ClosedSocket()
    with pytest.raises(RuntimeError, match="disconnected"):
        asyncio.run(cop._cdp("Runtime.evaluate"))
    assert socket.calls == 1 and cop._ws is None


def test_closed_cdp_connection_reattaches_without_launching_another_browser(monkeypatch):
    cop, _ = _copilot()
    cop._ws = SimpleNamespace(closed=True, close=_async_noop)
    cop._proc = SimpleNamespace(poll=lambda: None)
    async def connected():
        cop._ws = SimpleNamespace(closed=False)
        return True
    monkeypatch.setattr(cop, "_connect_debugger", connected)
    monkeypatch.setattr(cop, "find_browser", lambda: pytest.fail("must reuse the existing browser"))
    assert asyncio.run(cop.ensure()) is None


@pytest.mark.parametrize("was_background", [False, True])
def test_browser_mode_change_after_socket_teardown_retires_old_process(monkeypatch, tmp_path, was_background):
    from orion_core import browser_copilot
    cop, _ = _copilot()
    events = []
    cop.profile_dir = tmp_path / "browser-profile"
    cop._headless = was_background
    cop._ws = None  # _cdp has already torn down the disconnected transport.
    cop._proc = SimpleNamespace(poll=lambda: None, terminate=lambda: events.append("terminate"))
    replacement = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(cop, "find_browser", lambda: "browser.exe")
    def launch(args, **kwargs):
        assert events == ["terminate"]
        assert ("--headless=new" in args) == (not was_background)
        events.append("launch")
        return replacement
    async def connected():
        return True
    monkeypatch.setattr(browser_copilot.subprocess, "Popen", launch)
    monkeypatch.setattr(cop, "_connect_debugger", connected)
    assert asyncio.run(cop.ensure(background=not was_background)) is None
    assert events == ["terminate", "launch"]
    assert cop._proc is replacement


async def _async_noop():
    pass


def test_browser_script_errors_are_not_silent_success(monkeypatch):
    cop, _ = _copilot()
    async def failed_script(*args):
        return {"exceptionDetails": {"text": "Uncaught ReferenceError"}}
    monkeypatch.setattr(cop, "_cdp", failed_script)
    with pytest.raises(RuntimeError, match="ReferenceError"):
        asyncio.run(cop._evaluate("missingFunction()"))


def test_browser_form_preserves_submission_failure(monkeypatch):
    cop, _ = _copilot()
    receipt = ToolResult("Submission connection lost", ok=False)
    async def fill(*args):
        return ToolResult("filled")
    async def submit():
        return receipt
    monkeypatch.setattr(cop, "fill", fill)
    monkeypatch.setattr(cop, "submit", submit)
    assert asyncio.run(cop.form({"email": "a@example.com"}, submit=True)) is receipt
