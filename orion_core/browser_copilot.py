"""
BrowserCopilot — ORION drives a REAL, visible browser and shows his workings.

Rather than a headless automation black box, this opens an actual Chrome/Edge
window the user can watch, and narrates every step (navigate, scroll, highlight,
read, click, research) over the bus so the user sees exactly what ORION is
doing.  It speaks the **Chrome DevTools Protocol** over a WebSocket, so it needs
no Selenium/Playwright — only aiohttp (already a dependency) and a
Chromium-based browser (Chrome or Edge), which every Windows box has.

Design:
  • foreground  → a visible browser window the user watches him work in;
  • background  → the same, headless, so the user keeps working uninterrupted;
  • a dedicated user-data-dir keeps ORION's session out of the user's own
    profile (Chrome refuses the debug port on an already-open default profile);
  • every action narrates itself to the activity log, and returns a structured
    ToolResult. It also emits a bus.browser_step event carrying the same step
    in machine-readable form — that signal currently has NO subscriber, so the
    narration the user actually sees is the log line. The event is kept because
    it costs nothing and is the seam a GUI or phone mirror would attach to, but
    describing it as something that "lets the phone mirror his reasoning" would
    be claiming a feature that does not exist.

All browser I/O is async and time-boxed; if no browser is found or the debug
port never comes up, the service degrades to an actionable message.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation

_DEBUG_PORT = 9222
_CONNECT_TIMEOUT_S = 15.0
_CDP_TIMEOUT_S = 20.0


# ── JavaScript the co-pilot injects (kept as pure builders so they unit-test) ──

def js_scroll_by(amount: int) -> str:
    return f"window.scrollBy({{top: {int(amount)}, behavior: 'smooth'}}); 'scrolled';"


def js_scroll_to_text(text: str) -> str:
    """Find the first element whose text contains *text*, scroll it into view,
    and flash a highlight box around it.  Returns the matched text or ''."""
    safe = json.dumps(str(text))
    return _HIGHLIGHT_HELPER + f"""
    (function(){{
        var needle = {safe}.toLowerCase();
        var walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
        var node;
        while ((node = walk.nextNode())) {{
            if (node.nodeValue && node.nodeValue.toLowerCase().indexOf(needle) !== -1
                && node.parentElement) {{
                var el = node.parentElement;
                el.scrollIntoView({{behavior:'smooth', block:'center'}});
                __orionHighlight(el, {safe});
                return (node.nodeValue || '').trim().slice(0, 240);
            }}
        }}
        return '';
    }})();
    """


def js_highlight(text: str) -> str:
    return js_scroll_to_text(text)


def js_read_text(max_chars: int) -> str:
    n = int(max_chars)
    return (
        "(function(){var t=(document.body&&document.body.innerText)||'';"
        f"return t.replace(/\\n{{3,}}/g,'\\n\\n').slice(0,{n});}})();"
    )


def js_links() -> str:
    return """
    (function(){
        var out=[], seen={};
        var as=document.querySelectorAll('a[href]');
        for (var i=0;i<as.length && out.length<40;i++){
            var a=as[i], t=(a.innerText||'').trim().replace(/\\s+/g,' ');
            var h=a.href;
            if (t && h && h.indexOf('javascript:')!==0 && !seen[h]){
                seen[h]=1; out.push({text:t.slice(0,90), href:h});
            }
        }
        return JSON.stringify(out);
    })();
    """


def js_click_text(text: str) -> str:
    """Click the first link/button whose visible text contains *text*."""
    safe = json.dumps(str(text))
    return _HIGHLIGHT_HELPER + f"""
    (function(){{
        var needle = {safe}.toLowerCase();
        var els = document.querySelectorAll('a,button,[role=button],input[type=submit],[onclick]');
        for (var i=0;i<els.length;i++){{
            var el=els[i], t=(el.innerText||el.value||'').toLowerCase();
            if (t && t.indexOf(needle)!==-1){{
                el.scrollIntoView({{behavior:'smooth', block:'center'}});
                __orionHighlight(el, {safe});
                el.click();
                return true;
            }}
        }}
        return false;
    }})();
    """


def js_fill(hint: str, value: str) -> str:
    """Find the input/textarea whose placeholder, aria-label, name, id or
    associated <label> contains *hint* (or the first field if hint is blank),
    highlight it, and set its value firing the events frameworks listen for."""
    safe_hint = json.dumps(str(hint))
    safe_val = json.dumps(str(value))
    return _HIGHLIGHT_HELPER + f"""
    (function(){{
        var hint = {safe_hint}.toLowerCase();
        var val = {safe_val};
        var fields = document.querySelectorAll(
            'input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=checkbox]):not([type=radio]),'
            + 'textarea,[contenteditable=true],[contenteditable=""]');
        function labelText(el){{
            var t = (el.getAttribute('placeholder')||'') + ' ' + (el.getAttribute('aria-label')||'')
                  + ' ' + (el.name||'') + ' ' + (el.id||'') + ' ' + (el.getAttribute('title')||'');
            if (el.id){{ var lab = document.querySelector('label[for="'+el.id+'"]'); if(lab) t += ' ' + (lab.innerText||''); }}
            return t.toLowerCase();
        }}
        var target = null;
        if (hint){{
            for (var i=0;i<fields.length;i++){{ if (labelText(fields[i]).indexOf(hint)!==-1){{ target=fields[i]; break; }} }}
        }}
        if (!target && fields.length) target = fields[0];
        if (!target) return false;
        target.scrollIntoView({{behavior:'smooth', block:'center'}});
        __orionHighlight(target, hint || 'field');
        target.focus();
        if (target.isContentEditable){{ target.textContent = val; }}
        else {{
            var proto = target.tagName==='TEXTAREA' ? window.HTMLTextAreaElement.prototype
                                                     : window.HTMLInputElement.prototype;
            var setter = Object.getOwnPropertyDescriptor(proto,'value');
            if (setter && setter.set) setter.set.call(target, val); else target.value = val;
        }}
        target.dispatchEvent(new Event('input', {{bubbles:true}}));
        target.dispatchEvent(new Event('change', {{bubbles:true}}));
        return true;
    }})();
    """


def js_submit() -> str:
    """Press Enter on the focused field and, failing that, submit its form."""
    return """
    (function(){
        var el = document.activeElement;
        if (!el) return false;
        ['keydown','keypress','keyup'].forEach(function(type){
            try{ el.dispatchEvent(new KeyboardEvent(type,{key:'Enter',code:'Enter',
                keyCode:13,which:13,bubbles:true,cancelable:true})); }catch(e){}
        });
        if (el.form){ try{ el.form.requestSubmit ? el.form.requestSubmit() : el.form.submit(); }catch(e){} }
        return true;
    })();
    """


# A small helper injected alongside highlight/click JS: draws a labelled outline
# box over an element so the user SEES what ORION is looking at, then fades it.
_HIGHLIGHT_HELPER = """
    window.__orionHighlight = window.__orionHighlight || function(el, label){
        try{
            var r = el.getBoundingClientRect();
            var box = document.createElement('div');
            box.style.cssText = 'position:fixed;z-index:2147483647;pointer-events:none;'
                + 'border:2px solid #ff1a3c;border-radius:6px;box-shadow:0 0 0 2px rgba(223,230,238,.6),0 0 18px rgba(255,26,60,.6);'
                + 'transition:opacity .5s ease;'
                + 'left:'+(r.left-3)+'px;top:'+(r.top-3)+'px;width:'+(r.width+6)+'px;height:'+(r.height+6)+'px;';
            var tag = document.createElement('div');
            tag.textContent = 'ORION';
            tag.style.cssText='position:absolute;top:-20px;left:-2px;background:#ff1a3c;color:#fff;'
                + 'font:700 10px/1 Segoe UI, sans-serif;letter-spacing:2px;padding:3px 6px;border-radius:4px;';
            box.appendChild(tag); document.body.appendChild(box);
            setTimeout(function(){box.style.opacity='0';}, 2600);
            setTimeout(function(){box.remove();}, 3200);
        }catch(e){}
    };
"""


class BrowserCopilot:
    """Drives a live Chromium browser over the DevTools Protocol."""

    def __init__(self, bus: OrionBus, *, debug_port: int = _DEBUG_PORT,
                 profile_dir: str | Path | None = None) -> None:
        self.bus = bus
        self.debug_port = int(debug_port)
        self.profile_dir = Path(profile_dir) if profile_dir else (CONFIG_DIR / "browser_profile")
        self._proc: Any = None
        self._session: Any = None
        self._ws: Any = None
        self._headless = False
        self._next_id = 0
        self._current_url = ""
        self._lock = asyncio.Lock()

    # ── discovery ─────────────────────────────────────────────────────────────

    def find_browser(self) -> str | None:
        """Locate a Chromium-based browser executable to drive over CDP.

        The user's own default comes first, then Edge, then anything else that
        speaks CDP. Chrome used to be hard-coded ahead of both, which meant a
        machine with Chrome merely installed had ORION driving a browser the
        user does not use — different profile, different logins, different
        extensions — while every other part of ORION opened links in Edge.
        Asking browser.py keeps the co-pilot and plain link-opening on the
        same browser, which is the whole point of "the default browser must be
        used each time".
        """
        env = os.getenv("ORION_BROWSER_PATH") or os.getenv("CHROME_BIN")
        if env and os.path.exists(env):
            return env
        from .browser import browser_path
        preferred = browser_path()
        # Only if it is Chromium-based: CDP is what this driver speaks, so a
        # Firefox default must fall through rather than be launched and hang.
        if preferred and Path(preferred).stem.lower() in {
                "msedge", "chrome", "chromium", "brave", "vivaldi", "opera"}:
            return preferred
        if sys.platform == "win32":
            pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
            pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
            local = os.environ.get("LOCALAPPDATA", "")
            candidates = [
                Path(pf86) / "Microsoft/Edge/Application/msedge.exe",
                Path(pf) / "Microsoft/Edge/Application/msedge.exe",
                Path(local) / "Microsoft/Edge/Application/msedge.exe",
                Path(pf) / "Google/Chrome/Application/chrome.exe",
                Path(pf86) / "Google/Chrome/Application/chrome.exe",
                Path(local) / "Google/Chrome/Application/chrome.exe",
            ]
        elif sys.platform == "darwin":
            candidates = [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            ]
        else:
            candidates = []
            for name in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser"):
                found = shutil.which(name)
                if found:
                    candidates.append(Path(found))
        for path in candidates:
            if path and Path(path).exists():
                return str(path)
        for name in ("msedge", "microsoft-edge", "chrome", "google-chrome", "chromium"):
            found = shutil.which(name)
            if found:
                return found
        return None

    # ── narration ─────────────────────────────────────────────────────────────

    def _narrate(self, step: str, detail: str = "") -> None:
        """Show ORION's workings: log line + a structured browser_step event."""
        line = f"BROWSER: {step}" + (f" — {detail}" if detail else "")
        try:
            self.bus.log.emit(line)
        except Exception:
            pass
        try:
            self.bus.browser_step.emit({"step": step, "detail": detail, "url": self._current_url})
        except Exception:
            pass

    # ── CDP transport ─────────────────────────────────────────────────────────

    async def _cdp(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one CDP command and return its result, skipping event frames."""
        if self._ws is None:
            raise RuntimeError("no browser session")
        self._next_id += 1
        msg_id = self._next_id
        await self._ws.send_str(json.dumps({"id": msg_id, "method": method,
                                            "params": params or {}}))
        deadline = time.monotonic() + _CDP_TIMEOUT_S
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(self._ws.receive(), timeout=_CDP_TIMEOUT_S)
            data = getattr(raw, "data", None)
            if not isinstance(data, str):
                continue
            try:
                frame = json.loads(data)
            except Exception:
                continue
            if frame.get("id") == msg_id:          # our response (skip events)
                if "error" in frame:
                    raise RuntimeError(str(frame["error"]))
                return frame.get("result", {})
        raise TimeoutError(f"CDP {method} timed out")

    async def _evaluate(self, expression: str) -> Any:
        result = await self._cdp("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": True,
        })
        return (result.get("result") or {}).get("value")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def ensure(self, *, background: bool = False) -> ToolResult | None:
        """Launch (or reuse) the browser and connect the debugger.  Returns None
        on success, or an actionable ToolResult on failure."""
        if self._ws is not None and self._headless == background:
            return None
        if self._ws is not None:
            await self.close()                     # mode change → fresh session
        exe = self.find_browser()
        if exe is None:
            return ToolResult(
                "I couldn't find Chrome or Edge to drive. Install Google Chrome "
                "(or set ORION_BROWSER_PATH) and I'll take it from there.", ok=False)
        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        args = [exe, f"--remote-debugging-port={self.debug_port}",
                f"--user-data-dir={self.profile_dir}", "--no-first-run",
                "--no-default-browser-check", "--disable-features=Translate",
                "about:blank"]
        if background:
            args.insert(1, "--headless=new")
        self._headless = background
        self._narrate("Opening a browser window" if not background
                      else "Opening a background browser", os.path.basename(exe))
        try:
            self._proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            return ToolResult(f"Couldn't launch the browser: {exc}", ok=False)
        connected = await self._connect_debugger()
        if not connected:
            return ToolResult(
                "The browser launched but its debugger never came up — another "
                "instance may be holding the debug port. Close other Chrome "
                "windows started by ORION and try again.", ok=False)
        return None

    async def _connect_debugger(self) -> bool:
        from aiohttp import ClientSession, ClientTimeout
        self._session = ClientSession(timeout=ClientTimeout(total=_CDP_TIMEOUT_S))
        base = f"http://127.0.0.1:{self.debug_port}"
        ws_url = None
        deadline = time.monotonic() + _CONNECT_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                async with self._session.get(f"{base}/json") as resp:
                    targets = await resp.json()
                page = next((t for t in targets if t.get("type") == "page"), None)
                if page is None:
                    async with self._session.get(f"{base}/json/new?about:blank") as resp:
                        page = await resp.json()
                ws_url = page.get("webSocketDebuggerUrl")
                if ws_url:
                    break
            except Exception:
                await asyncio.sleep(0.4)
        if not ws_url:
            await self._teardown_session()
            return False
        try:
            self._ws = await self._session.ws_connect(ws_url, max_msg_size=0)
            await self._cdp("Page.enable")
            await self._cdp("Runtime.enable")
            return True
        except Exception:
            await self._teardown_session()
            return False

    async def _teardown_session(self) -> None:
        for closer in (self._ws, self._session):
            try:
                if closer is not None:
                    await closer.close()
            except Exception:
                pass
        self._ws = None
        self._session = None

    async def close(self) -> ToolResult:
        await self._teardown_session()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        self._narrate("Closed the browser")
        return ToolResult("Closed the browser.")

    # ── actions ───────────────────────────────────────────────────────────────

    def _safe(self, text: str, label: str) -> str:
        return SecuritySanitiser.guard_text(str(text or ""), label)

    async def open(self, url: str, *, background: bool = False) -> ToolResult:
        try:
            url = self._safe(url, "browser.url").strip()
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        if not url:
            return ToolResult("Give me a URL or a search to open.", ok=False)
        if not url.startswith(("http://", "https://")):
            # Treat a bare phrase as a web search.
            from urllib.parse import quote_plus
            url = "https://www.google.com/search?q=" + quote_plus(url)
        async with self._lock:
            failure = await self.ensure(background=background)
            if failure is not None:
                return failure
            self._current_url = url
            self._narrate("Navigating", url)
            try:
                await self._cdp("Page.navigate", {"url": url})
                await self._await_ready()
            except Exception as exc:
                return ToolResult(f"Navigation failed: {exc}", ok=False)
            title = await self._evaluate("document.title") or ""
            self._narrate("Loaded", str(title)[:120])
            return ToolResult(f"Opened {url} — “{str(title)[:100]}”.")

    async def _await_ready(self, timeout: float = 12.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = await self._evaluate("document.readyState")
            except Exception:
                state = None
            if state in ("interactive", "complete"):
                return
            await asyncio.sleep(0.25)

    async def launch(self, *, background: bool = False) -> ToolResult:
        """Bring the browser up (no navigation) so it's ready to drive."""
        async with self._lock:
            failure = await self.ensure(background=background)
            if failure is not None:
                return failure
        return ToolResult("The browser is up and ready."
                          if not background else "A background browser is ready.")

    async def fill(self, hint: str, value: str, *, submit: bool = False) -> ToolResult:
        try:
            hint = self._safe(hint, "browser.field")
            value = self._safe(value, "browser.value")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        async with self._lock:
            if self._ws is None:
                return ToolResult("No browser is open. Ask me to open a page first.", ok=False)
            self._narrate("Typing", f"“{value[:60]}” into {hint or 'the field'}")
            ok = await self._evaluate(js_fill(hint, value))
            if not ok:
                return ToolResult(
                    f"I couldn't find a field matching “{hint}” to type into.", ok=False)
            if submit:
                self._narrate("Submitting")
                await self._evaluate(js_submit())
                await self._await_ready(timeout=8.0)
                self._current_url = await self._evaluate("location.href") or self._current_url
                return ToolResult(f"Typed “{value[:60]}” and submitted.")
            return ToolResult(f"Typed “{value[:60]}” into {hint or 'the field'}.")

    async def form(self, values: dict[str, str], *, submit: bool = False) -> ToolResult:
        if not values:
            return ToolResult("No form values were supplied.", ok=False)
        filled: list[str] = []
        for hint, value in values.items():
            res = await self.fill(str(hint), str(value))
            if res.ok:
                filled.append(str(hint))
        if not filled:
            return ToolResult("None of the fields matched anything on this page.", ok=False)
        if submit:
            await self.submit()
        return ToolResult(f"Filled {len(filled)} field(s): {', '.join(filled)}"
                          + (" and submitted." if submit else "."))

    async def submit(self) -> ToolResult:
        async with self._lock:
            if self._ws is None:
                return ToolResult("No browser is open.", ok=False)
            self._narrate("Submitting")
            await self._evaluate(js_submit())
            await self._await_ready(timeout=8.0)
            self._current_url = await self._evaluate("location.href") or self._current_url
            return ToolResult("Submitted.")

    async def scroll(self, *, amount: int = 800, to_text: str = "") -> ToolResult:
        async with self._lock:
            if self._ws is None:
                return ToolResult("No browser is open. Ask me to open a page first.", ok=False)
            if to_text:
                try:
                    to_text = self._safe(to_text, "browser.text")
                except SecurityViolation as exc:
                    return ToolResult(str(exc), ok=False)
                self._narrate("Scrolling to", to_text)
                found = await self._evaluate(js_scroll_to_text(to_text))
                if found:
                    return ToolResult(f"Scrolled to and highlighted: “{str(found)[:120]}”.")
                return ToolResult(f"I couldn't find “{to_text}” on this page.", ok=False)
            self._narrate("Scrolling", f"{amount}px")
            await self._evaluate(js_scroll_by(amount))
            return ToolResult("Scrolled.")

    async def highlight(self, text: str) -> ToolResult:
        try:
            text = self._safe(text, "browser.text")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        async with self._lock:
            if self._ws is None:
                return ToolResult("No browser is open yet.", ok=False)
            self._narrate("Highlighting", text)
            found = await self._evaluate(js_highlight(text))
            if found:
                return ToolResult(f"Highlighted “{str(found)[:120]}”.")
            return ToolResult(f"I couldn't find “{text}” to highlight.", ok=False)

    async def read(self, max_chars: int = 4000) -> ToolResult:
        async with self._lock:
            if self._ws is None:
                return ToolResult("No page is open to read.", ok=False)
            self._narrate("Reading the page")
            text = await self._evaluate(js_read_text(max_chars)) or ""
            return ToolResult(str(text) or "The page had no readable text.")

    async def links(self) -> ToolResult:
        async with self._lock:
            if self._ws is None:
                return ToolResult("No page is open.", ok=False)
            self._narrate("Collecting links")
            raw = await self._evaluate(js_links()) or "[]"
            try:
                items = json.loads(raw)
            except Exception:
                items = []
            if not items:
                return ToolResult("No links found on this page.")
            lines = [f"- {i['text']} → {i['href']}" for i in items[:30]]
            return ToolResult("Links on this page:\n" + "\n".join(lines))

    async def click(self, text: str) -> ToolResult:
        try:
            text = self._safe(text, "browser.text")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        async with self._lock:
            if self._ws is None:
                return ToolResult("No browser is open.", ok=False)
            self._narrate("Clicking", text)
            ok = await self._evaluate(js_click_text(text))
            if ok:
                await self._await_ready(timeout=6.0)
                self._current_url = await self._evaluate("location.href") or self._current_url
                return ToolResult(f"Clicked “{text}”.")
            return ToolResult(f"I couldn't find anything to click matching “{text}”.", ok=False)

    async def screenshot(self) -> ToolResult:
        async with self._lock:
            if self._ws is None:
                return ToolResult("No page to capture.", ok=False)
            self._narrate("Capturing a screenshot")
            try:
                import base64
                result = await self._cdp("Page.captureScreenshot", {"format": "jpeg", "quality": 70})
                data = base64.b64decode(result.get("data", ""))
            except Exception as exc:
                return ToolResult(f"Screenshot failed: {exc}", ok=False)
            return ToolResult("Captured the current page.",
                              media={"data": data, "mime_type": "image/jpeg"})
