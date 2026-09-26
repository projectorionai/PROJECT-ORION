"""
The phone page, driven in a real browser against a live gateway.

String checks cannot catch ordering bugs in the page's own JavaScript. This
runs it in headless Chromium: a QR link pairs without also popping the manual
code prompt, an approval raised while the phone was away appears once on
reconnect and resolves on the desktop, and only a refused device (not a busy
or rate-limited desktop) is asked to pair again.

Skipped when Playwright or a Chromium build is not available.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sync_api = pytest.importorskip("playwright.sync_api")

from orion_core.remote import RemoteGateway  # noqa: E402


class _Signal:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Memory:
    def log_episode(self, *a):
        pass

    def prompt_context(self, limit=18):
        return ""


def _serve(gateway, ready, holder):
    async def main():
        from aiohttp import web
        # As the gateway's own stop(): an open event stream must not hold
        # teardown for aiohttp's default grace.
        runner = web.AppRunner(gateway._build_app(),
                               shutdown_timeout=gateway._SHUTDOWN_GRACE_S)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        holder["port"] = site._server.sockets[0].getsockname()[1]
        holder["stop"] = asyncio.Event()
        ready.set()
        await holder["stop"].wait()
        await runner.cleanup()
    holder["loop"] = asyncio.new_event_loop()
    holder["loop"].run_until_complete(main())


def _launch(pw):
    """Chromium from Playwright, the installed Chrome, or a known path."""
    candidates = [{}, {"channel": "chrome"}]
    for path in (os.getenv("ORION_TEST_CHROMIUM", ""), "/opt/pw-browsers/chromium"):
        if path and Path(path).is_file():
            candidates.append({"executable_path": path})
    for options in candidates:
        try:
            return pw.chromium.launch(args=["--no-sandbox"], **options)
        except Exception:
            continue
    pytest.skip("no Chromium build available for Playwright")


def test_the_phone_page_pairs_catches_up_and_only_repairs_when_refused(tmp_path):
    gateway = RemoteGateway(object(), _Memory(), _Bus(), config_dir=tmp_path)
    ready, holder = threading.Event(), {}
    server = threading.Thread(target=_serve, args=(gateway, ready, holder), daemon=True)
    server.start()
    assert ready.wait(10), "the gateway did not start"
    base = f"http://127.0.0.1:{holder['port']}"
    try:
        with sync_api.sync_playwright() as pw:
            browser = _launch(pw)
            # Routes do not see requests a service worker answers, and the
            # page's CSP (rightly) forbids the eval Playwright polls with.
            context = browser.new_context(service_workers="block", bypass_csp=True)
            page = context.new_page()
            dialogs, errors = [], []
            page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
            page.on("pageerror", lambda e: errors.append(str(e)))

            # A QR link pairs by itself, cleans the URL, and asks nothing.
            page.goto(f"{base}/?pair={gateway.begin_pairing()}", wait_until="domcontentloaded")
            page.wait_for_function(
                "document.body.innerText.includes('Paired with ORION.')", timeout=15000)
            device = page.evaluate("localStorage.getItem('orion_device')")
            assert device
            assert "pair=" not in page.evaluate("location.href")
            assert dialogs == [], "the manual pairing prompt popped up over a QR link"

            # An approval raised while away appears once after reconnecting.
            gateway.confirmations.create("send_email", {"to": "bob"}, device, "Email Bob")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function(
                "document.body.innerText.includes('Approve: Email Bob?')", timeout=15000)
            assert page.locator("text=Approve: Email Bob?").count() == 1
            page.locator("button:has-text('Deny')").first.click()
            page.wait_for_function(
                "!document.body.innerText.includes('Approve: Email Bob?')", timeout=15000)
            deadline = 50
            while gateway.confirmations.pending_for(device) and deadline:
                page.wait_for_timeout(100)
                deadline -= 1
            assert not gateway.confirmations.pending_for(device)

            # Busy or rate-limited: no prompt. Refused: pair again.
            reset = "(async()=>{access='';accessExp=0;return await ensureAccess();})()"
            page.route("**/v1/auth/token", lambda r: r.fulfill(status=429, body="{}"))
            assert page.evaluate(reset) is False
            assert dialogs == []
            page.unroute("**/v1/auth/token")
            page.route("**/v1/auth/token", lambda r: r.fulfill(status=401, body="{}"))
            page.evaluate(reset)
            assert dialogs, "a refused device was not asked to pair again"

            assert errors == [], errors
            browser.close()
    finally:
        loop = holder.get("loop")
        gateway._closing = True
        if loop is not None and "stop" in holder:
            loop.call_soon_threadsafe(holder["stop"].set)
        server.join(timeout=10)
