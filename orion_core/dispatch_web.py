"""
Dispatch domain — Web, browser, external integrations and geo.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
import time
import webbrowser  # noqa: F401  (kept: see .browser.open_url)
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line
from .browser import open_url
from .execution_policy import require_foreground



#: What people call the countries ORION is most often in, by the name the
#: location service stores ("United Kingdom").
_COUNTRY_NAMES: dict[str, frozenset[str]] = {
    "united kingdom": frozenset({"uk", "the uk", "united kingdom", "the united kingdom",
                                 "britain", "great britain", "gb", "england", "the country",
                                 "this country", "here in the uk", "over here"}),
}


def _own_country(place: str, temporal: Any) -> bool:
    """Whether *place* names the country the user is in (so it means their
    own town), judged from the stored location - never assumed."""
    try:
        where = str(temporal.locality() if temporal is not None else "").lower()
    except Exception:
        return False
    said = re.sub(r"[^a-z ]", "", str(place or "").lower()).strip()
    for country, names in _COUNTRY_NAMES.items():
        if country in where and said in names:
            return True
    return False

class WebDispatchMixin:
    """Web, browser, external integrations and geo."""

    def web_search(self, args: dict[str, Any]) -> ToolResult | Awaitable[ToolResult]:
        query = SecuritySanitiser.guard_text(str(args.get("query") or ""), "web_search.query")
        if not query:
            return ToolResult("No search query supplied.", ok=False)
        from .execution_policy import is_background
        if is_background():
            return self._background_web_search(query)
        raw = args.get("open", args.get("show"))
        show = raw is True or str(raw or "").strip().lower() in {"true", "1", "yes", "on"}
        return self._foreground_web_search(query, show)

    async def _foreground_web_search(self, query: str, show: bool) -> ToolResult:
        """Real results for the model to answer from; the browser only on request.

        The foreground search used to open Bing and return "Secure web search
        opened" — the model was handed nothing, so "latest AI models",
        "events at my university" and "best AI PCs" (the owner's second most
        common kind of request) were answered from training memory, stale
        and stated as current. The results now come back as text from the
        same bounded backends the background path uses; the browser opens
        only when the user asked to see the search, or as a visible fallback
        when no backend answered (and then the result says it read nothing).
        """
        found = await self._background_web_search(query, timeout=15.0)
        receipt = dict(found.receipt or {})
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        if found.ok:
            if show:
                receipt["browser_opened"] = bool(open_url(url))
                note = ("\n(Also opened in the browser, as asked.)" if receipt["browser_opened"]
                        else "\n(The browser could not be opened to show it.)")
                return replace(found, text=found.text + note, receipt=receipt)
            return found
        receipt["browser_opened"] = bool(open_url(url))
        why = first_line(found.text, 160)
        if receipt["browser_opened"]:
            return ToolResult(
                f"I could not read any search results ({why}). I opened the search for "
                f"'{query}' in the browser so the user can see it, but its contents are NOT "
                "known to me: do not state any result as fact; say it is on screen, or "
                "read it with web_control read_page.", ok=False, receipt=receipt)
        return ToolResult(f"I could not read any search results ({why}), and the browser "
                          "could not be opened either.", ok=False, receipt=receipt)

    async def _background_web_search(self, query: str, timeout: float = 20.0) -> ToolResult:
        """Fetch bounded results without opening or modifying user browser tabs."""
        from . import web_search_backends
        try:
            async with asyncio.timeout(timeout):
                hits = await web_search_backends.search(query, limit=6)
            results = [{"title": str(hit.title)[:300], "url": str(hit.url)[:2000],
                        "snippet": str(hit.snippet)[:2000]} for hit in hits[:6]
                       if urlparse(str(hit.url)).scheme.lower() in {"http", "https"}]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolResult(f"Background search failed: {first_line(exc, 160)}", ok=False,
                              receipt={"status": "failed", "query": query, "results": [], "browser_opened": False})
        if not results:
            return ToolResult("Search returned no usable results.", ok=False,
                              receipt={"status": "failed", "query": query, "results": [], "browser_opened": False})
        text = "Search results:\n" + "\n".join(
            f"{index}. {hit['title']}\n{hit['url']}\n{hit['snippet']}" for index, hit in enumerate(results, 1))
        return ToolResult(text, receipt={"status": "completed", "query": query,
                                        "results": results, "browser_opened": False}, evidence=results)

    async def fetch_url(self, args: dict[str, Any]) -> ToolResult:
        """Read a URL through the fetch MCP instead of opening it in a browser."""
        url = str(args.get("url") or args.get("link") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ToolResult("fetch_url needs an http(s) URL. For a local file, use process_file.",
                              ok=False)
        try:
            start = max(0, int(args.get("start_index") or 0))
            length = max(1000, min(20_000, int(args.get("max_length") or 8000)))
        except (TypeError, ValueError):
            return ToolResult("start_index and max_length must be whole numbers.", ok=False)
        from .execution_policy import is_background
        if is_background():
            from .background_http import fetch_text
            return await fetch_text(url, start_index=start, max_length=length)
        host = getattr(self, "mcp_host", None)
        if host is None:
            return ToolResult("The fetch MCP host is unavailable.", ok=False)
        result = await host.call("fetch", "fetch", {
            "url": url, "start_index": start, "max_length": length})
        return ToolResult(result, ok=not mcp_call_failed(result))

    def open_news(self, args: dict[str, Any]) -> ToolResult:
        from .execution_policy import require_foreground
        deferred = require_foreground("browser_window")
        if deferred is not None:
            return deferred
        if not self.news_articles:
            return ToolResult("No briefing stories are cached yet.", ok=False)
        query = SecuritySanitiser.guard_text(
            str(args.get("query") or args.get("topic") or args.get("title") or ""),
            "open_news.query",
        ).strip().lower()
        chosen: list[dict[str, str]] = []
        index = args.get("index")
        if index is not None:
            try:
                position = int(index) - 1
                if 0 <= position < len(self.news_articles):
                    chosen = [self.news_articles[position]]
            except (TypeError, ValueError):
                pass
        if not chosen and query:
            chosen = [
                article for article in self.news_articles
                if query in article["title"].lower() or query in article["topic"].lower()
            ]
        if not chosen:
            return ToolResult(
                "No cached briefing story matches that request. Cached stories:\n"
                + "\n".join(
                    f"{i}. ({a['topic']}) {a['title']}"
                    for i, a in enumerate(self.news_articles, 1)
                ),
                ok=False,
            )
        opened: list[str] = []
        for article in chosen[:3]:
            if article.get("url"):
                open_url(article["url"])
                opened.append(article["title"])
        if not opened:
            return ToolResult("The matched stories carry no usable link.", ok=False)
        return ToolResult("Opened in browser: " + "; ".join(opened))

    def browser_control(self, args: dict[str, Any]) -> ToolResult | Awaitable[ToolResult]:
        action = str(args.get("action") or "go_to").lower().strip()
        if action not in {"search", "web_search"}:
            from .execution_policy import require_foreground
            deferred = require_foreground("browser_window")
            if deferred is not None:
                return deferred
        if action in {"go_to", "open", "new_tab"}:
            url = str(args.get("url") or args.get("target") or "").strip()
            if not url:
                return ToolResult("No URL supplied.", ok=False)
            url = self._normalise_url(SecuritySanitiser.guard_text(url, "browser_control.url"))
            open_url(url)
            return ToolResult(f"Browser opened: {url}.")
        if action in {"search", "web_search"}:
            # "Search in the browser" is a request to SEE it: open it as well
            # as reading the results back.
            return self.web_search({"query": args.get("query") or args.get("text") or "",
                                    "open": True})
        return ToolResult(f"Browser action '{action}' is not supported by the native dispatcher.", ok=False)

    async def web_control(self, args: dict[str, Any]) -> ToolResult:
        blocked = require_foreground("shared_browser_control")
        if blocked is not None:
            return blocked
        try:
            if "submit" in args and not isinstance(args["submit"], bool):
                return ToolResult("submit must be true or false.", ok=False)
            return await self._web_control_action(args)
        except (ValueError, TypeError, OverflowError, SecurityViolation) as exc:
            return ToolResult(f"Invalid web_control arguments: {first_line(str(exc))}", ok=False)

    async def _web_control_action(self, args: dict[str, Any]) -> ToolResult:
        if self.web is None:
            return ToolResult("Web controller is not available.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        w = self.web
        if action in {"open", "go_to"}:
            return await w.open(str(args.get("url") or args.get("target") or ""))
        if action in {"navigate", "goto"}:
            return await w.navigate(str(args.get("url") or args.get("target") or ""))
        if action == "new_tab":
            return await w.new_tab(str(args.get("url") or ""))
        if action == "close_tab":
            return await w.close_tab()
        if action == "switch_tab":
            return await w.switch_tab(int(args.get("index", 1)))
        if action in {"accept_cookies", "accept"}:
            return await w.accept_cookies()
        if action in {"reject_cookies", "reject"}:
            return await w.reject_cookies()
        if action in {"close_popup", "dismiss"}:
            return await w.close_popup()
        if action in {"click", "click_text", "click_element"}:
            return await w.click(str(args.get("text") or args.get("target") or args.get("query") or ""))
        if action == "scroll":
            return await w.scroll(int(args.get("amount", -5)))
        if action == "submit":
            return await w.submit()
        if action in {"fill", "type", "type_text"}:
            label = str(args.get("field") or args.get("label") or args.get("target") or "").strip()
            if not label:
                return ToolResult("Typing requires a field label; use fill_form or supply field.", ok=False)
            return await w.fill_form({label: str(args.get("value") or args.get("text") or "")},
                                     submit=args.get("submit", False))
        if action in {"fill_form", "form"}:
            fields = args.get("fields")
            if not isinstance(fields, dict):
                return ToolResult("fill_form requires a 'fields' object of label→value.", ok=False)
            return await w.fill_form(fields, submit=bool(args.get("submit")))
        if action in {"search", "web_search", "search_web", "google"}:
            return await self._web_control_search(w, args)
        if action in {"read_page", "read"}:
            return await w.read_page()
        if action in {"browse", "narrated_scroll", "read_down", "explore"}:
            dwell = float(args.get("dwell", 1.6))
            if not math.isfinite(dwell):
                return ToolResult("dwell must be a finite number of seconds.", ok=False)
            return await w.narrated_scroll(
                url=str(args.get("url") or ""),
                steps=int(args.get("steps", 6)),
                dwell=dwell,
            )
        if action in {"summarise", "summarize", "evaluate", "review"}:
            return await w.summarise_current(url=str(args.get("url") or ""))
        if action in {"download"}:
            return await w.download_current()
        if action in {"file_dialog", "confirm_dialog", "upload", "save_as"}:
            return await w.handle_file_dialog(str(args.get("path") or ""))
        return ToolResult(
            f"Unsupported web_control action: {action}. Use open, navigate, search, new_tab, "
            "close_tab, switch_tab, accept_cookies, reject_cookies, close_popup, "
            "click, scroll, fill_form, type_text, submit, read_page, browse (narrated slow-scroll), summarise, "
            "download, or file_dialog.",
            ok=False,
        )

    async def _web_control_search(self, w: Any, args: dict[str, Any]) -> ToolResult:
        """Search in the visible browser AND hand the model the results.

        'search' was rejected as unsupported, so the model navigated to a
        search URL it composed itself and then had nothing to read back."""
        query = SecuritySanitiser.guard_text(
            str(args.get("query") or args.get("text") or args.get("value") or "").strip(),
            "web_control.query")
        if not query:
            return ToolResult("search needs a query.", ok=False)
        shown = await w.open(f"https://www.bing.com/search?q={quote_plus(query)}")
        found = await self._background_web_search(query, timeout=15.0)
        receipt = dict(found.receipt or {})
        receipt["browser_opened"] = bool(getattr(shown, "ok", False))
        lead = (f"Searched for '{query}' in the browser." if receipt["browser_opened"] else
                f"The browser did not open the search ({first_line(getattr(shown, 'text', ''), 120)}).")
        if found.ok:
            return ToolResult(f"{lead}\n{found.text}", receipt=receipt, evidence=found.evidence)
        return ToolResult(
            f"{lead} I could not read the results directly ({first_line(found.text, 120)}); "
            "use read_page to read what the browser shows before stating any result.",
            ok=receipt["browser_opened"], receipt=receipt)

    async def web_automation_tool(self, args: dict[str, Any]) -> ToolResult:
        """Browser co-pilot (#11): ORION drives a REAL, visible browser window
        and narrates every step over the bus, so the user watches him work."""
        action = str(args.get("action") or "").lower().strip()
        if action != "close":
            blocked = require_foreground("generic_browser_automation")
            if blocked is not None:
                return blocked
        copilot = getattr(self, "web_copilot", None)
        if copilot is None:
            return ToolResult(
                "The browser co-pilot is unavailable — no Chromium browser was "
                "wired up.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        background = bool(args.get("headless") or args.get("background"))
        # Accept both the legacy 'selector'/'target' aliases and plain 'text';
        # the co-pilot matches by visible text, not brittle CSS selectors.
        target = str(args.get("text") or args.get("target") or args.get("selector") or "")

        if action == "launch":
            return await copilot.launch(background=background)
        if action in {"go_to", "goto", "open", "navigate"}:
            return await copilot.open(str(args.get("url") or target), background=background)
        if action in {"click", "click_selector", "smart_click", "click_text"}:
            return await copilot.click(target)
        if action in {"type", "type_text", "fill"}:
            return await copilot.fill(
                str(args.get("field") or args.get("selector") or args.get("hint") or ""),
                str(args.get("text") or args.get("value") or ""),
                submit=bool(args.get("submit")))
        if action == "form":
            return await copilot.form(dict(args.get("values") or {}),
                                      submit=bool(args.get("submit")))
        if action == "submit":
            return await copilot.submit()
        if action in {"scroll", "read_down"}:
            return await copilot.scroll(
                amount=int(args.get("amount", 800)),
                to_text=str(args.get("to_text") or args.get("text") or ""))
        if action == "highlight":
            return await copilot.highlight(target)
        if action in {"read", "read_page"}:
            return await copilot.read(int(args.get("max_chars", 4000)))
        if action == "links":
            return await copilot.links()
        if action in {"screenshot", "capture"}:
            return await copilot.screenshot()
        if action == "close":
            return await copilot.close()
        return ToolResult(
            f"Unsupported web_automation action: {action}. Use launch, go_to, "
            "click, type, form, submit, scroll, highlight, read, links, "
            "screenshot, or close.", ok=False)

    async def flight_search(self, args: dict[str, Any]) -> ToolResult:
        """Find flights by reading the results page a person would read.

        There is no free pricing API worth wiring to, so this drives the
        browser co-pilot to the search and reads what rendered. The parsing
        of places, dates and money lives in ``flights`` and is pure; only the
        browser and the model are touched here.
        """
        blocked = require_foreground("generic_browser_flight_search")
        if blocked is not None:
            return blocked
        from .flights import FlightFinder, FlightRequestError, spoken_summary, \
            written_report

        ai = getattr(self, "ai", None)
        router = getattr(ai, "router", None) or getattr(self, "router", None)
        finder = FlightFinder(
            bus=getattr(self, "bus", None),
            router=router,
            copilot=getattr(self, "web_copilot", None),
        )

        try:
            search = await finder.search(args)
        except FlightRequestError as exc:
            # An ambiguous request is a question, not a failure — ORION asks
            # rather than searching the wrong day or the wrong Birmingham.
            return ToolResult(str(exc), ok=False)
        except Exception as exc:  # pragma: no cover - browser/network faults
            return ToolResult(
                f"The flight search didn't complete: {first_line(str(exc))}",
                ok=False)

        if getattr(self, "web_copilot", None) is None:
            return ToolResult(
                f"I can't drive a browser to search, but here is the search "
                f"for {search.route} on {search.depart}: {search.url}",
                ok=False)

        self.last_flight_search = search
        return ToolResult(f"{spoken_summary(search)}\n\n{written_report(search)}")

    async def outlook_mail(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "read_inbox").lower().strip()
        limit = int(args.get("limit") or 10)
        # Tool calls are direct user requests, so Outlook may be started here
        # (launch=True); passive surveys elsewhere remain attach-only.
        if action in {"read_inbox", "read", "inbox", "check"}:
            return await self.outlook.read_inbox(
                limit=limit, unread_only=bool(args.get("unread_only")), launch=True
            )
        if action in {"priority", "priority_emails", "important"}:
            return await self.outlook.priority_emails(limit=limit, launch=True)
        if action in {"read_email", "open_email", "full_body"}:
            entry_id = str(args.get("entry_id") or "").strip()
            if not entry_id:
                return ToolResult("An entry_id from a previous inbox read is required.", ok=False)
            return await self.outlook.read_email_body(entry_id)
        if action in {"draft", "create_draft", "compose"}:
            return await self.outlook.create_draft(
                to=str(args.get("to") or ""),
                subject=str(args.get("subject") or ""),
                body=str(args.get("body") or ""),
                cc=str(args.get("cc") or ""),
            )
        if action in {"send_draft", "send"}:
            return await self.outlook.send_draft(
                draft_ref=str(args.get("draft_ref") or args.get("draft") or ""),
                confirm=bool(args.get("confirm")),
            )
        if action in {"discard_draft", "discard"}:
            return await self.outlook.discard_draft(
                str(args.get("draft_ref") or args.get("draft") or "")
            )
        if action in {"pending_drafts", "drafts"}:
            drafts = self.outlook.pending_drafts()
            if not drafts:
                return ToolResult("No drafts are awaiting approval.")
            return ToolResult(
                "Drafts awaiting approval:\n" + "\n".join(
                    f"- {d['ref']}: to {d['to']}, subject '{d['subject']}'" for d in drafts
                )
            )
        return ToolResult(
            f"Unsupported outlook action: {action}. Use read_inbox, priority, "
            "read_email, draft, send_draft, discard_draft, or pending_drafts.",
            ok=False,
        )

    async def open_owner_tasks(self, limit: int = 12, *, source: str = "") -> list[dict[str, str]]:
        """Notion's open tasks minus the ones the user already said are done
        or removed, remembered as "the tasks just read out" so a following
        "I've done all of those" can name them. Raises if Notion is unreadable."""
        cognition = getattr(self, "cognition", None)
        dismissed = getattr(cognition, "is_dismissed", None)
        limit = max(1, min(100, int(limit)))
        # Apply local completion knowledge before limiting the visible list.
        # A full first page of dismissed rows must not hide later open work.
        rows = await self.notion.open_tasks(limit=None)
        rows = [r for r in rows if not (dismissed is not None and dismissed(r["title"]))][:limit]
        if cognition is not None and source:
            await asyncio.to_thread(cognition.note_mentioned_tasks,
                                    [r["title"] for r in rows], source)
        return rows

    @staticmethod
    def _task_title_list(value: Any) -> list[str]:
        """Titles from a string, a list, or lists of either (model arguments
        arrive in every shape); ';' or newlines separate several in one string."""
        names: list[str] = []

        def _walk(item: Any) -> None:
            if isinstance(item, (list, tuple)):
                for inner in item:
                    _walk(inner)
                return
            if item is None:
                return
            for part in re.split(r"\s*[;\n]\s*", str(item)):
                clean = part.strip(" -•\t'\"")
                if clean and clean.casefold() not in {n.casefold() for n in names}:
                    names.append(clean)

        _walk(value)
        return names

    async def complete_owner_tasks(self, titles: Any = None, *, all_listed: bool = False) -> ToolResult:
        """Mark tasks the user says are finished as done everywhere ORION keeps
        them: the Notion board (when one is configured) and ORION's own task
        list, plus a permanent "completed by user" record so no briefing,
        proactive check or restored conversation raises them again.

        17 Oct 2026: told "I've done all of those tasks", ORION had only
        clear_overdue (his own list, and none were overdue), said they were
        cleared, and the same six Notion tasks were read out next morning."""
        cognition = getattr(self, "cognition", None)
        mentioned = cognition.last_mentioned_tasks() if cognition is not None else []
        names = self._task_title_list(titles)
        if all_listed:
            names += [t for t in mentioned if t.casefold() not in {n.casefold() for n in names}]
        # A fragment ("the work shift one") resolves to the exact title just
        # read out when exactly one of them contains it.
        local_titles = ([str(t.get("title") or "") for t in cognition.list_tasks()]
                        if cognition is not None else [])
        known = list(dict.fromkeys(mentioned + local_titles))
        resolved: list[str] = []
        unresolved: list[str] = []
        for name in names:
            folded = name.casefold()
            exact = [t for t in known if t.casefold() == folded]
            partial = list({t.casefold(): t for t in known if folded in t.casefold()}.values())
            if not exact and len(partial) > 1:
                unresolved.append(f"{name} (multiple tasks match; use a full title)")
                continue
            title = exact[0] if exact else partial[0] if len(partial) == 1 else name
            if title.casefold() not in {t.casefold() for t in resolved}:
                resolved.append(title)
        if len(resolved) > 30:
            return ToolResult("Complete at most 30 named tasks at a time; no tasks were changed.", ok=False)
        if not resolved:
            if unresolved:
                return ToolResult("No tasks were changed: " + "; ".join(unresolved), ok=False)
            return ToolResult(
                "I don't have a recent list of tasks I read out to you, so I can't tell "
                "which ones you mean. Name them and I'll mark them done.", ok=False)
        notion = getattr(self, "notion", None)
        notion_ready = bool(notion is not None and getattr(notion, "available", False)
                            and getattr(notion, "tasks_db", ""))
        on_board: list[str] = []
        remembered: list[str] = []
        problems: list[str] = []
        memory_problems: list[str] = []
        uncertain_on_board: list[dict[str, Any]] = []
        for title in resolved:
            board_title = ""
            if notion_ready:
                try:
                    result = await notion.complete_task(title)
                    receipt = result.receipt or {}
                    if result.ok and receipt.get("completed") is True and receipt.get("task"):
                        board_title = str(receipt["task"])
                        on_board.append(board_title)
                    else:
                        problems.append(f"{title} ({first_line(result.text, 120)})")
                        if receipt.get("verification_pending"):
                            uncertain_on_board.append(dict(receipt))
                except Exception as exc:
                    problems.append(f"{title} ({first_line(exc, 120)})")
            # A verified remote title is authoritative. On board failure only
            # exact known titles can be remembered; never dismiss a fragment
            # or complete the first vaguely matching local task.
            local_title = board_title or (title if not notion_ready or any(
                t.casefold() == title.casefold() for t in known) else "")
            if cognition is not None and local_title:
                try:
                    await asyncio.to_thread(cognition.dismiss_task, local_title, "completed by user")
                    remembered.append(local_title)
                except Exception as exc:
                    memory_problems.append(f"{local_title} ({first_line(exc, 120)})")
        if cognition is None and not on_board:
            return ToolResult("I couldn't mark those tasks done: neither the Notion board "
                              "nor my own task list is available.", ok=False)
        lines = []
        if on_board:
            lines.append(f"Marked done on the Notion board: {'; '.join(on_board)}.")
        local_only = [t for t in remembered if t not in on_board]
        if local_only:
            lines.append(f"Recorded as done locally: {'; '.join(local_only)}.")
        if problems:
            lines.append("The board could not be updated for: " + "; ".join(problems) + ".")
        if memory_problems:
            lines.append("Completion could not be saved locally for: " + "; ".join(memory_problems) + ".")
        if unresolved:
            lines.append("No changes for: " + "; ".join(unresolved) + ".")
        return ToolResult(" ".join(lines), ok=not (problems or memory_problems or unresolved),
                          receipt={"completed_on_board": on_board, "recorded_done": remembered,
                                   "board_problems": problems, "memory_problems": memory_problems,
                                   "unresolved": unresolved, "unverified_on_board": uncertain_on_board})

    async def owner_task_view(self) -> Any:
        """ORION's list + the Notion board, one overdue rule (owner_tasks) -
        the same view the briefing reads out."""
        from . import owner_tasks
        return await owner_tasks.gather(getattr(self, "notion", None),
                                        getattr(self, "cognition", None))

    async def list_owner_tasks(self, limit: int = 30) -> ToolResult:
        from . import owner_tasks
        view = await self.owner_task_view()
        rows = view.rows[:max(1, min(100, int(limit)))]
        cognition = getattr(self, "cognition", None)
        if cognition is not None:
            await asyncio.to_thread(cognition.note_mentioned_tasks,
                                    [r["title"] for r in rows], "awareness")
        text = owner_tasks.render(rows)
        if view.board_configured and not view.board_read:
            text += (f"\nThe Notion board could not be read ({view.board_error}), "
                     "so tasks kept there are not shown.")
        return ToolResult(text, ok=not (view.board_configured and not view.board_read and not rows),
                          receipt={"counts": view.counts(), "board_read": view.board_read})

    async def clear_overdue_owner_tasks(self) -> ToolResult:
        """Clear every overdue task the owner has been told about - ORION's
        own list AND the board the briefing reads.

        25 Oct 2026: the briefing listed five overdue Notion rows; "clear the
        several due tasks" reached clear_overdue, which only knew ORION's own
        (empty) list and answered "Nothing is overdue". Now both stores are
        read by the same rule. A board task is cleared the way remove_task
        clears one - ORION stops raising it (a permanent tombstone) and the
        board itself is left as it is; "mark them done" ticks them off there."""
        from . import owner_tasks
        cognition = getattr(self, "cognition", None)
        if cognition is None:
            return ToolResult("Cognitive state is not available.", ok=False)
        view = await self.owner_task_view()
        local_removed = await asyncio.to_thread(cognition.clear_overdue)
        board_cleared: list[str] = []
        problems: list[str] = []
        for row in view.overdue:
            if row.get("source") != "notion":
                continue
            title = str(row.get("title") or "").strip()
            try:
                if await asyncio.to_thread(cognition.dismiss_task, title, "cleared overdue"):
                    board_cleared.append(title)
            except Exception as exc:
                problems.append(f"{title} ({first_line(exc, 100)})")
        unread = view.board_configured and not view.board_read
        removed = list(local_removed) + board_cleared
        if not removed and not problems:
            if unread:
                return ToolResult(
                    "Nothing on my own list is overdue, but the Notion board could not be read "
                    f"({view.board_error}), so I could not clear the tasks kept there. "
                    "Do not tell the user nothing was overdue.", ok=False,
                    receipt={"cleared": [], "board_read": False})
            return ToolResult("Nothing is overdue - on my own list or on the task board.",
                              receipt={"cleared": [], "board_read": view.board_read})
        lines = []
        if local_removed:
            lines.append(f"Removed {len(local_removed)} overdue task(s) from my list: "
                         + "; ".join(local_removed) + ".")
        if board_cleared:
            lines.append(f"Cleared {len(board_cleared)} overdue Notion task(s): "
                         + "; ".join(board_cleared) + ". I will not raise them again; they "
                         "stay on the board itself (ask me to mark them done to tick them off there).")
        if problems:
            lines.append("Could not clear: " + "; ".join(problems) + ".")
        if unread:
            lines.append(f"The Notion board could not be read ({view.board_error}).")
        return ToolResult(" ".join(lines), ok=not problems,
                          receipt={"cleared": removed, "cleared_local": list(local_removed),
                                   "cleared_board": board_cleared, "problems": problems,
                                   "board_read": view.board_read})

    async def notion_workspace(self, args: dict[str, Any]) -> ToolResult:
        from .action_aliases import normalise_action
        action = normalise_action("notion_workspace", args.get("action") or "list_tasks")
        if action in {"list_tasks", "tasks", "todo"}:
            if (args.get("include_done") or not getattr(self.notion, "available", False)
                    or not getattr(self.notion, "tasks_db", "")):
                return await self.notion.list_tasks(
                    limit=int(args.get("limit") or 12),
                    include_done=bool(args.get("include_done")))
            try:
                rows = await self.open_owner_tasks(
                    limit=int(args.get("limit") or 12), source="list_tasks")
            except Exception as exc:
                return ToolResult(f"Notion task query failed: {first_line(exc)}", ok=False)
            self.bus.dashboard_event.emit("tasks", rows)
            return ToolResult(self.notion.format_tasks(rows))
        if action in {"create_task", "add_task", "new_task"}:
            return await self.notion.create_task(
                title=str(args.get("title") or args.get("task") or ""),
                due=str(args.get("due") or args.get("date") or ""),
                notes=str(args.get("notes") or ""),
            )
        if action in {"complete_task", "finish_task", "done", "complete_tasks",
                      "finish_tasks", "mark_done", "completed"}:
            return await self.complete_owner_tasks(
                [args.get("titles"), args.get("query"), args.get("title")],
                all_listed=bool(args.get("all_listed")))
        if action in {"update_task", "edit_task"}:
            query = str(args.get("query") or args.get("task") or args.get("title") or "")
            result = await self.notion.update_task(
                query, title=str(args.get("title") or "") if args.get("query") or args.get("task") else "",
                due=str(args["due"] or "") if "due" in args else None,
                status=str(args.get("status") or ""))
            cognition = getattr(self, "cognition", None)
            if not result.ok or cognition is None:
                return result
            receipt = dict(result.receipt or {})
            original, title = str(receipt.get("original_task") or ""), str(receipt.get("task") or "")
            if not original or not title or "completed" not in receipt:
                return ToolResult("The edit returned no verified task identity; refresh the board.", ok=False)
            try:
                if receipt["completed"] or receipt.get("terminal") or (
                        not args.get("status") and cognition.is_dismissed(original)):
                    for name in {original, title}:
                        await asyncio.to_thread(cognition.dismiss_task, name, "completed by user")
                else:
                    if original != title:
                        await asyncio.to_thread(cognition.dismiss_task, original, "renamed to " + title)
                    await asyncio.to_thread(cognition.add_task, title, due=str(receipt.get("date") or ""), restore=True)
            except Exception as exc:
                receipt["memory_problem"] = first_line(exc, 160)
                return ToolResult(result.text + " The board was updated, but local memory could not be saved: "
                                  + receipt["memory_problem"], ok=False, receipt=receipt)
            return result
        if action in {"archive_task", "remove_task", "delete_task"}:
            title = str(args.get("title") or "").strip()
            result = await self.notion.archive_task_exact(title)
            if result.ok and title and getattr(self, "cognition", None) is not None:
                self.cognition.dismiss_task(title)
            return result
        if action in {"upcoming_events", "calendar", "schedule", "agenda"}:
            return await self.notion.upcoming_events(
                days=int(args.get("days") or 7), limit=int(args.get("limit") or 10)
            )
        if action in {"create_event", "schedule_event", "book"}:
            return await self.notion.create_event(
                title=str(args.get("title") or ""),
                start=str(args.get("start") or args.get("date") or ""),
                end=str(args.get("end") or ""),
            )
        if action in {"projects", "project_overview"}:
            return await self.notion.project_overview(limit=int(args.get("limit") or 10))
        return ToolResult(
            f"Unsupported notion action: {action}. Use list_tasks, create_task, "
            "complete_task, update_task, archive_task, upcoming_events, create_event, or projects.",
            ok=False,
        )

    async def telephony_tool(self, args: dict[str, Any]) -> ToolResult:
        """Ring a phone, or text one, for real.

        Three guards, because a phone call cannot be un-sent, is often
        recorded at the far end, and may be heard by someone who is not the
        user:

        * **Default-deny numbers.** Only what is in
          ``config/telephony_contacts.json``. ORION cannot dial a number
          somebody said out loud, or one a web page contained.
        * **Screening.** The spoken message goes through the spillage guard
          first, and screening fails CLOSED — if it cannot check the text,
          nothing is said.
        * **Confirmation.** Calls and texts cost money and arrive instantly,
          so ``confirm=true`` is required. The model is expected to say who
          it is about to ring and what it will say, and only then ask.
        """
        gateway = getattr(self, "telephony", None)
        if gateway is None:
            return ToolResult("Telephony is not available.", ok=False)
        action = str(args.get("action") or "call").lower().strip()

        if action in {"status", "check"}:
            names = ", ".join(gateway.contacts.names()) or "nobody"
            # Say which ROUTE is live and, when none is, which credential is
            # actually missing. Naming a file and a restart was wrong about
            # both whenever the real problem was an empty Account SID.
            if gateway.available():
                route = ("a telephony server" if gateway._mcp_connected()
                         else "Twilio directly")
                state = f"ready — I can dial through {route}."
            else:
                state = gateway.why_unavailable()
            return ToolResult(
                f"Telephony: {state} Numbers ORION may dial: {names}.")
        if action in {"contacts", "who", "numbers"}:
            names = gateway.contacts.names()
            return ToolResult(
                "ORION may ring: " + ", ".join(names) if names else
                "No numbers are allowed yet. Copy "
                "config/telephony_contacts.example.json to "
                "telephony_contacts.json and fill it in.")

        to = str(args.get("to") or args.get("contact") or "").strip()
        message = str(args.get("message") or args.get("text") or "").strip()
        if not to:
            return ToolResult("Who should I ring? Use action='contacts' to "
                              "see the numbers I'm allowed to dial.", ok=False)
        if not message:
            return ToolResult("What should I say?", ok=False)
        texting = action in {"text", "sms", "message"}
        verb = "text" if texting else "ring"
        if not bool(args.get("confirm")):
            return ToolResult(
                f"This will {verb} {to} for real and it costs money. Tell "
                f"them what you are about to say, then call again with "
                f"confirm=true.", ok=False)

        # confirm=true got us this far, and on its own it is a convention
        # rather than a gate: nothing stops the model sending it on the first
        # call. Spending money goes behind the same human-issued, single-use,
        # expiring token that terminating a process does, with the number
        # bound to it so the one approved on screen is the one dialled.
        guard = getattr(self, "system_guard", None)
        if guard is not None:
            from .system_guard import ActionIntent, DecisionKind

            intent = (ActionIntent.SEND_MESSAGE if texting
                      else ActionIntent.PLACE_CALL)
            blocked = require_foreground("telephony_confirmation_dialog")
            if blocked is not None:
                return blocked
            decision = guard.request_confirmation(
                intent, origin="telephony",
                payload={"to": to, "message": message, "action": action})
            if decision.kind is DecisionKind.CONFIRM:
                try:
                    self.bus.confirm_action.emit({
                        "token": decision.token,
                        "intent": intent.value,
                        # "message", not "detail": the dialog reads this
                        # key, and anything else leaves it asking someone
                        # to approve a phone call without saying which.
                        "message": (f"{verb.title()} {to} and say: "
                                    f"{message[:160]}"),
                    })
                except Exception:
                    pass
                return ToolResult(
                    f"About to {verb} {to} and say: {message[:120]}. "
                    f"Approve the on-screen prompt to go ahead.", ok=False)

        if texting:
            result = await gateway.text(to, message)
        else:
            result = await gateway.call(to, message,
                                        reason=str(args.get("reason") or ""))
        return ToolResult(result.detail, ok=result.ok)

    def _mcp_spend_gate(self, server: str, tool: str, arguments: dict[str, Any], *,
                        confirmed: bool) -> ToolResult | None:
        """None when the MCP call may run now; otherwise the reply to give.

        Real world actions. The Twilio entry carried the warning from
        the day it was added and nothing enforced it; then the first-class
        ``mcp__twilio__*`` tools arrived and skipped even the check the
        generic ``mcp call`` had — a second, unbolted door to the same bill.
        Both doors now come through here. ``confirm=true`` is only the model
        agreeing with itself, so a spending server always needs the
        human-issued, single-use token the telephony tool uses; with no guard
        available it is refused rather than waved through.
        """
        try:
            reason = self.mcp_host.requires_confirmation(server, tool)
        except Exception:
            reason = ""
        if not reason:
            return None
        if not confirmed:
            return ToolResult(
                f"{reason}. Tell the user exactly what this will do and any "
                f"cost; if they agree, call again with confirm=true and approve "
                f"the on-screen prompt.", ok=False)
        guard = getattr(self, "system_guard", None)
        if guard is None:
            return ToolResult(f"{reason}. Human confirmation is unavailable "
                              "in this session.", ok=False)
        from .system_guard import ActionIntent, DecisionKind

        blocked = require_foreground("mcp_confirmation_dialog")
        if blocked is not None:
            return blocked
        decision = guard.request_confirmation(
            ActionIntent.SEND_MESSAGE, origin=f"mcp:{server}",
            payload={"server": server, "tool": tool, "arguments": dict(arguments)})
        if decision.kind is DecisionKind.CONFIRM:
            try:
                self.bus.confirm_action.emit({
                    "token": decision.token,
                    "intent": "mcp_spend",
                    "message": f"Run {server}.{tool}? {reason}.",
                })
            except Exception:
                pass
            return ToolResult(f"{reason}. Approve the on-screen prompt to go ahead.",
                              ok=False)
        if decision.kind is not DecisionKind.EXECUTE:
            return ToolResult(f"{reason}. Not permitted: {decision.message}", ok=False)
        return None

    async def mcp_tool(self, args: dict[str, Any]) -> ToolResult:
        """Model Context Protocol gateway: discover and call tools from the
        connected MCP servers (Gmail, Google Calendar, filesystem, …)."""
        if self.mcp_host is None:
            return ToolResult("The MCP host is not available.", ok=False)
        action = str(args.get("action") or "list").lower().strip()
        if action in {"add", "declare", "enable", "on", "reconnect", "restart"}:
            blocked = require_foreground("mcp_configuration_dialog")
            if blocked is not None:
                return blocked
            # Registering a command then enabling it was a shell-execution
            # bypass: the model could supply python -c or powershell itself.
            # Bind the command/spec to a token only the local UI can release.
            guard = getattr(self, "system_guard", None)
            if guard is None:
                return ToolResult("MCP executable configuration requires local human confirmation.", ok=False)
            from .system_guard import ActionIntent, DecisionKind
            from .mcp_host import load_config, _spec_fingerprint
            server = str(args.get("server") or args.get("name") or "").strip()
            payload: dict[str, Any] = {"mcp_configuration": dict(args)}
            command = str(args.get("command") or "")
            command_args = args.get("arguments") or args.get("args") or []
            if action not in {"add", "declare"}:
                spec = (await asyncio.to_thread(load_config)).get("servers", {}).get(server)
                if not isinstance(spec, dict):
                    return ToolResult(f"No configured MCP server named '{server}'.", ok=False)
                payload["spec_fingerprint"] = _spec_fingerprint(spec)
                command, command_args = str(spec.get("command") or ""), spec.get("args") or []
            decision = guard.request_confirmation(ActionIntent.MCP_CONFIGURE, origin="mcp:configuration",
                                                  payload=payload)
            if decision.kind is DecisionKind.CONFIRM:
                self.bus.confirm_action.emit({"token": decision.token, "intent": "mcp_configure",
                    "message": (f"{action.title()} MCP server '{server}'? This permits external code to run on your computer.\n"
                                f"Executable: {command!r}\nArguments: {command_args!r}")})
            return ToolResult("Approve the local MCP configuration prompt to proceed.", ok=False)
        if action in {"list", "catalogue", "catalog", "discover"}:
            return ToolResult("Connected MCP tools:\n" + self.mcp_host.describe())
        if action in {"servers", "status"}:
            return ToolResult(self.mcp_host.list_servers())
        if action in {"disable", "off"}:
            server = str(args.get("server") or args.get("name") or "").strip()
            want_on = action in {"enable", "on"}
            return ToolResult(await self.mcp_host.set_enabled(server, want_on))
        if action in {"call", "invoke", "run"}:
            server = str(args.get("server") or "").strip()
            tool   = str(args.get("tool") or args.get("name") or "").strip()
            if not server or not tool:
                return ToolResult(
                    "mcp 'call' needs a 'server' and a 'tool'. Use action='list' "
                    "to see what's available.", ok=False)
            arguments = args.get("arguments") or args.get("args") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            blocked = self._mcp_spend_gate(server, tool, arguments,
                                           confirmed=bool(args.get("confirm")))
            if blocked is not None:
                return blocked
            result = await self.mcp_host.call(server, tool, arguments)
            return ToolResult(result, ok=not mcp_call_failed(result))
        return ToolResult(
            f"Unsupported mcp action '{action}'. Use list, servers, call, "
            "enable, disable, reconnect, or add.", ok=False)

    async def _request_channel_send(self, kind: str, contact: str, message: str, *,
                                    title: str = "", priority: str = "") -> ToolResult:
        """Arm the owner's approval for one message on one of ORION's own
        channels. Nothing is sent here: the approved token runs exactly this
        recipient and these words (DesktopDispatchMixin._run_confirmed_channel)."""
        from . import message_channels
        receipt = {"kind": "messaging", "platform": kind, "backend": "builtin_api",
                   "attempted": False, "sent": False}
        if not message.strip():
            return ToolResult("There is nothing to send.", ok=False,
                              receipt={**receipt, "status": "empty"})
        if not await asyncio.to_thread(message_channels.configured, kind):
            return ToolResult(message_channels.SETUP[kind], ok=False,
                              receipt={**receipt, "status": "not_configured"})
        suppressed = getattr(self.messaging, "delivery_suppressed", None)
        if callable(suppressed) and suppressed():
            self.bus.log.emit(f"MESSAGING: {kind} delivery simulated (dry-run).")
            return ToolResult(f"{kind.title()} message simulated (dry-run).",
                              receipt={**receipt, "status": "simulated"})
        guard = getattr(self, "system_guard", None)
        if guard is None:
            return ToolResult("Human confirmation is unavailable in this session, so nothing "
                              "was sent.", ok=False, receipt={**receipt, "status": "unavailable"})
        # Asked from anywhere (a spoken turn, a job, a schedule), the approval
        # is the same non-modal, token-bound prompt a plugin's message gets:
        # it never takes the owner's focus, and nothing is sent without it.
        from .execution_policy import is_background
        from .system_guard import ActionIntent, DecisionKind
        payload = {"channel": kind, "to": contact, "message": message,
                   "title": title, "priority": priority}
        decision = guard.request_confirmation(ActionIntent.SEND_MESSAGE,
                                              origin=f"channel:{kind}", payload=payload)
        prompt = message_channels.describe_send(kind, contact, message)
        if decision.kind is not DecisionKind.CONFIRM:
            return ToolResult(f"Not sent: {decision.message}", ok=False,
                              receipt={**receipt, "status": "rejected"})
        try:
            self.bus.confirm_action.emit({"token": decision.token, "intent": "channel_message",
                                          "background": is_background(), "channel": kind,
                                          "to": contact, "body": message,
                                          "message": prompt + "?"})
        except Exception:
            guard.cancel(decision.token)
            return ToolResult("The approval could not be shown, so nothing was sent.", ok=False,
                              receipt={**receipt, "status": "unavailable"})
        return ToolResult(f"{prompt} - waiting for the user's approval on screen; nothing has "
                          "been sent yet.", receipt={**receipt, "status": "awaiting_confirmation"})

    async def messaging_tool(self, args: dict[str, Any]) -> ToolResult:
        """Real-time messaging gateway (WhatsApp, Telegram) and email.

        Email is here rather than in a mail-specific tool because it is the
        same decision every time: deliver it properly if ORION can, otherwise
        put it in front of the user with everything filled in — and say which
        of the two happened.
        """
        if self.messaging is None:
            return ToolResult("Messaging gateway is unavailable.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        platform = str(args.get("platform") or "").lower().strip()
        if action == "close":
            return await asyncio.to_thread(self.messaging.close)
        if action in {"status", "connection_status", "check_connection", "connect"}:
            method = (self.messaging.browser_connect if action == "connect" else
                      self.messaging.browser_check_connection if action == "check_connection" else self.messaging.browser_status)
            return await asyncio.to_thread(method, platform or "whatsapp")
        from . import message_channels
        if action in {"channels", "channel_status", "setup"}:
            return ToolResult(await asyncio.to_thread(message_channels.status))
        kind = message_channels.channel(platform) if action == "send" else ""
        # While the Telegram or Discord plugin is still installed it keeps
        # its own path (its registry state, secrets and approval); the
        # built-in channel takes over once it is retired.
        legacy = {"telegram": "telegram_message", "discord": "discord_message"}.get(kind)
        registry = getattr(self, "plugins", None)
        if legacy and registry is not None:
            try:
                if registry.get(legacy) is not None:
                    kind = ""
            except Exception:
                kind = ""
        if kind and (kind in {"push", "ifttt"}
                     or await asyncio.to_thread(message_channels.configured, kind)):
            # ORION's own channels (message_channels): the service's API, sent
            # only after the owner approves the exact words on screen.
            return await self._request_channel_send(
                kind, str(args.get("contact") or args.get("to") or ""),
                str(args.get("message") or args.get("text") or ""),
                title=str(args.get("title") or ""), priority=str(args.get("priority") or ""))
        owner_loop = asyncio.get_running_loop()
        invoker = getattr(self, "invoke_plugin", None)
        if callable(invoker) and action == "send" and platform in {"discord", "telegram"}:
            from .messaging import BOT_TIMEOUT_S
            def registered_plugin(name, arguments):
                # Request the central human-approval workflow. This flag can
                # queue a prompt; only its exact host-issued capability grants
                # permission to execute. Model arguments cannot grant it.
                request = {**arguments, "confirm": True}
                pending = asyncio.run_coroutine_threadsafe(
                    invoker(name, request, timeout_s=BOT_TIMEOUT_S), owner_loop)
                return pending.result(timeout=BOT_TIMEOUT_S + 5.0)
            readiness = {}
            check_secrets = getattr(self, "plugin_secret_available", None)
            if callable(check_secrets):
                try:
                    for name in (("discord_message",) if platform == "discord" else ("telegram_message",)):
                        readiness[name] = await check_secrets(name)
                except Exception:
                    return ToolResult("Messaging account readiness could not be read. No delivery was started.", ok=False,
                                      receipt={"kind": "messaging", "status": "unavailable", "attempted": False, "sent": False})
            self.messaging.bind_plugin_sender(registered_plugin, readiness if callable(check_secrets) else None)
        if action in {"email", "mail"} or (action == "send" and platform in {"email", "gmail", "mail"}):
            return await asyncio.to_thread(self.messaging.send_email,
                str(args.get("to") or args.get("contact") or ""),
                str(args.get("subject") or ""),
                str(args.get("body") or args.get("message") or ""))
        if action == "send":
            bind = getattr(self.messaging, "bind_desktop", None)
            if callable(bind):
                bind(getattr(self, "control", None))
            bind_browser = getattr(self.messaging, "bind_browser", None)
            if callable(bind_browser) and platform in {"whatsapp", "discord"}:
                bind_browser()
            contact = str(args.get("contact") or args.get("to") or "")
            message = str(args.get("message") or args.get("text") or "")
            return await asyncio.to_thread(self.messaging.send_text, platform, contact, message)
        return ToolResult(
            f"Unsupported messaging action: {action}. Use 'send' (platform "
            "whatsapp/telegram/discord), 'email', 'status', 'check_connection', 'connect' or 'close'.", ok=False)

    async def social_media_tool(self, args: dict[str, Any]) -> ToolResult:
        """Real-account social automation (#TikTok upload, Instagram DM read/
        reply) via a REAL, visible, narrated browser logged into the user's
        OWN accounts — kept separate from web_automation's general narrated
        browsing (orion_core/social_automation.py)."""
        social = self.social
        if social is None:
            return ToolResult("Social automation is not configured.", ok=False)
        if not social.available:
            return ToolResult(
                "Social automation is unavailable — install it with "
                "'pip install playwright' then run 'playwright install chrome' once.",
                ok=False,
            )
        action = str(args.get("action") or "").lower().strip()
        if action == "tiktok_upload":
            return await social.tiktok_upload(
                video_path=str(args.get("video_path") or ""),
                caption=str(args.get("caption") or ""),
                hashtags=str(args.get("hashtags") or ""),
            )
        if action in {"instagram_check_dms", "check_dms"}:
            return await social.instagram_check_dms(int(args.get("limit") or 10))
        if action in {"instagram_draft_reply", "draft_reply"}:
            return await social.instagram_draft_reply(
                contact=str(args.get("contact") or ""),
                message=str(args.get("message") or ""),
            )
        if action in {"instagram_reply", "reply"}:
            return await social.instagram_reply(
                reply_ref=str(args.get("reply_ref") or ""),
                confirm=bool(args.get("confirm")),
            )
        return ToolResult(
            f"Unsupported social_media action: {action}. Use tiktok_upload, "
            "instagram_check_dms, instagram_draft_reply, or instagram_reply.",
            ok=False,
        )

    async def geo_tool(self, args: dict[str, Any]) -> ToolResult:
        """Phase 8: worldwide geospatial intelligence — locate any place, list
        the settlements near it, or describe its administrative hierarchy."""
        action = str(args.get("action") or "locate").lower().strip()
        query = str(args.get("query") or args.get("place") or "")
        if action in {"weather", "forecast", "conditions", "temperature", "rain"}:
            return await self._weather(query, str(args.get("when") or args.get("question") or ""))
        if action in {"places", "find_places", "nearest", "near_me", "poi", "find_nearby"}:
            return await self._places(str(args.get("what") or query),
                                      str(args.get("near") or args.get("from") or ""))
        if action in {"distance", "how_far", "howfar", "directions", "navigate",
                      "route", "show_route", "route_on_globe"}:
            destination = str(args.get("to") or args.get("destination") or query).strip()
            origin = str(args.get("from") or args.get("origin") or "").strip()
            if not destination:
                return ToolResult("Where to?", ok=False)
            if action in {"route", "show_route", "route_on_globe"}:
                return await self._route_on_globe(origin, destination)
            if action in {"directions", "navigate"}:
                from .journeys import directions_url
                mode = str(args.get("mode") or "driving")
                if not open_url(directions_url(origin, destination, mode)):
                    return ToolResult("I couldn't open the browser for directions.", ok=False)
                return ToolResult(f"Opened {mode} directions to {destination}"
                                  + (f" from {origin}" if origin else "") + " in Google Maps.")
            return await self._distance(origin, destination)
        if self.geo is None:
            return ToolResult("The geospatial engine is not available.", ok=False)
        if not query:
            return ToolResult("A place is required.", ok=False)
        if action in {"locate", "find", "fly"}:
            return await self.geo.locate(query)
        if action in {"nearby", "towns_near", "near", "within"}:
            radius = float(args.get("radius_km") or args.get("radius") or 50.0)
            return await self.geo.towns_near(query, radius)
        if action in {"describe", "admin", "administrative", "detail"}:
            return await self.geo.describe(query)
        return ToolResult(
            f"Unsupported geo action: {action}. Use locate, nearby, describe or weather.",
            ok=False,
        )

    async def _journey_ends(self, origin: str, destination: str) -> Any:
        """((lat, lon, name), (lat, lon, name)) for the start and the end -
        the start is where the user is when no origin is given - or a
        ToolResult saying why not."""
        if self.geo is None:
            return ToolResult("I can't look places up right now.", ok=False)
        found = await self.geo.geocode(destination)
        if not found:
            return ToolResult(f"I couldn't find '{destination}'.", ok=False)
        to = found[0]
        # The geocoder names the region ('Greater London'); say it as asked.
        to_name = destination.title() if destination.lower() in to.name.lower() else to.name
        if origin and not re.fullmatch(r"(?i)(here|home|me|my (house|place|location))", origin):
            start_found = await self.geo.geocode(origin)
            if not start_found:
                return ToolResult(f"I couldn't find '{origin}'.", ok=False)
            first = start_found[0]
            from_name = origin.title() if origin.lower() in first.name.lower() else first.name
            start = (first.lat, first.lon, from_name)
        else:
            temporal = getattr(getattr(self, "live_worker", None), "temporal", None)
            point = temporal.coordinates() if temporal is not None else None
            if point is None:
                return ToolResult(f"I don't know where you are yet — tell me where to start "
                                  f"from and I'll work it out for {to_name}.", ok=False)
            here = str(getattr(temporal, "locality", lambda: "")() or "").split(",")[0].strip()
            start = (point[0], point[1], here or "you")
        return start, (to.lat, to.lon, to_name)

    async def _distance(self, origin: str, destination: str) -> ToolResult:
        """How far *destination* is from *origin* (or from where the user is):
        straight line always, road distance and driving time when the router
        answers."""
        from . import journeys
        ends = await self._journey_ends(origin, destination)
        if isinstance(ends, ToolResult):
            return ends
        (lat1, lon1, from_name), (lat2, lon2, to_name) = ends
        crow = journeys.crow_km(lat1, lon1, lat2, lon2)
        drivable = crow < 3000
        by_road = await journeys.road(lat1, lon1, lat2, lon2) if drivable else None
        found = by_road if isinstance(by_road, tuple) else None
        return ToolResult(journeys.describe(from_name, to_name, crow,
                                            found[0] if found else None,
                                            found[1] if found else None,
                                            road_tried=drivable, no_road=by_road is False,
                                            ferry=bool(found and found[2])))

    async def _route_on_globe(self, origin: str, destination: str) -> ToolResult:
        """Draw the driving route on the 3-D globe (the page is put on screen)
        and say how far and how long."""
        from . import journeys
        ends = await self._journey_ends(origin, destination)
        if isinstance(ends, ToolResult):
            return ends
        (lat1, lon1, from_name), (lat2, lon2, to_name) = ends
        shape = await journeys.route_shape(lat1, lon1, lat2, lon2)
        if shape is None:
            return ToolResult(f"I couldn't find a road route from {from_name} to {to_name}.",
                              ok=False)
        km, seconds, coords, ferry = shape
        summary = (f"{from_name} to {to_name}: {journeys._miles_km(km)} by road"
                   f"{' and ferry' if ferry else ''}, about {journeys.spoken_duration(seconds)} "
                   f"without traffic{', including the crossing' if ferry else ''}.")
        bus = getattr(self, "bus", None)
        if bus is not None:
            bus.dashboard_event.emit("globe_route", {
                "from": from_name, "to": to_name, "km": round(km, 1),
                "seconds": round(seconds), "coords": coords, "summary": summary})
        return ToolResult("Showing the route on the globe — " + summary)

    async def _places(self, what: str, near: str) -> ToolResult:
        """Shops, chemists, cash machines, takeaways... around the user (or a
        named place), nearest first, with open-now where it is certain."""
        from . import places
        what = what.strip()
        if not what:
            return ToolResult("What are you looking for?", ok=False)
        if near and not re.fullmatch(r"(?i)(here|me|home|my (house|place|location))", near):
            if self.geo is None:
                return ToolResult("I can't look places up right now.", ok=False)
            found = await self.geo.geocode(near)
            if not found:
                return ToolResult(f"I couldn't find '{near}'.", ok=False)
            lat, lon = found[0].lat, found[0].lon
            label = near.title() if near.lower() in found[0].name.lower() else found[0].name
        else:
            temporal = getattr(getattr(self, "live_worker", None), "temporal", None)
            point = temporal.coordinates() if temporal is not None else None
            if point is None:
                return ToolResult("I don't know where you are yet — tell me the town and I'll "
                                  "look there.", ok=False)
            lat, lon = point
            label = ""
        try:
            results = await places.search(what, lat, lon)
        except Exception as exc:
            return ToolResult(f"The map service didn't answer: {first_line(exc, 90)}", ok=False)
        return ToolResult(places.describe(results, what, label), ok=bool(results))

    async def _weather(self, place: str, when: str) -> ToolResult:
        """The forecast for a named place, or - with none - where the user is
        (the live temporal service's location, never a guess)."""
        from . import weather
        place = place.strip()
        temporal = getattr(getattr(self, "live_worker", None), "temporal", None)
        if place and _own_country(place, temporal):
            # Common sense: "the weather in the UK" from someone in Colchester
            # means Colchester's - a country has no one forecast, and its
            # middle (geocode) is a field nobody lives in (owner, 26 Oct 2026).
            place = ""
        if place and not re.fullmatch(r"(?i)(here|home|outside|near me|my area|local(ly)?)", place):
            if self.geo is None:
                return ToolResult("I can't look places up right now.", ok=False)
            found = await self.geo.geocode(place)
            if not found:
                return ToolResult(f"I couldn't find '{place}' to check its weather.", ok=False)
            lat, lon, name = found[0].lat, found[0].lon, found[0].name
        else:
            temporal = getattr(getattr(self, "live_worker", None), "temporal", None)
            point = temporal.coordinates() if temporal is not None else None
            if point is None:
                return ToolResult("I don't know where you are yet — tell me the town and "
                                  "I'll check its weather.", ok=False)
            lat, lon = point
            name = str(getattr(temporal, "locality", lambda: "")() or "").split(",")[0].strip()
        try:
            forecast = await weather.fetch(lat, lon, name)
        except Exception as exc:
            return ToolResult(f"The weather service didn't answer: {first_line(exc, 90)}", ok=False)
        return ToolResult(weather.answer(forecast, when))

    async def aviation_tool(self, args: dict[str, Any]) -> ToolResult:
        """Live aircraft: what's overhead, what's near a place, one flight's
        latest state, or the lot drawn on the Globe.

        Providers, back-off, caching and wording live in ``aviation``; this
        only hands it the location sources. The user's locality comes from the
        LIVE temporal service (the live worker's), never the state file: a
        bare dispatcher then has no location and asks, rather than a test
        sweep quietly reaching the network with the real home coordinates.
        """
        from .aviation import run_tool
        text, ok = await run_tool(
            args,
            geo=getattr(self, "geo", None),
            temporal=getattr(getattr(self, "live_worker", None), "temporal", None),
            bus=getattr(self, "bus", None),
        )
        return ToolResult(text, ok=ok)


_MCP_FAILURE_PREFIXES = ("No connected MCP server named", "MCP tool '")
_MCP_FAILURE_MARKERS = ("' timed out.", "' failed: ", "has no tool '")


def mcp_call_failed(text: str) -> bool:
    """Whether an MCPHost.call() reply is one of its own failure messages.

    It returns plain strings for success and failure alike; reporting every
    one as ok=True told the model a timed-out call had worked. Its replies
    now carry their outcome (``MCPReply.kind``), which is authoritative; the
    wording checks below remain for plain strings (stubs, older callers)."""
    kind = getattr(text, "kind", None)
    if isinstance(kind, str):
        return bool(kind)
    text = str(text or "")
    head = text[:200]
    return (head.startswith(_MCP_FAILURE_PREFIXES)
            and any(marker in head for marker in _MCP_FAILURE_MARKERS + ("Available:",))) \
        or head.startswith("Server '") and "has no tool '" in head \
        or head.startswith("MCP server '") and "could not restart:" in head \
        or head.startswith("(tool error)")
