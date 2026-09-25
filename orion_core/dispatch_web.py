"""
Dispatch domain — Web, browser, external integrations and geo.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import webbrowser  # noqa: F401  (kept: see .browser.open_url)
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line
from .browser import open_url


class WebDispatchMixin:
    """Web, browser, external integrations and geo."""

    def web_search(self, args: dict[str, Any]) -> ToolResult:
        query = SecuritySanitiser.guard_text(str(args.get("query") or ""), "web_search.query")
        if not query:
            return ToolResult("No search query supplied.", ok=False)
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        open_url(url)
        return ToolResult(f"Secure web search opened for: {query}.")

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
        host = getattr(self, "mcp_host", None)
        if host is None:
            return ToolResult("The fetch MCP host is unavailable.", ok=False)
        result = await host.call("fetch", "fetch", {
            "url": url, "start_index": start, "max_length": length})
        return ToolResult(result, ok=not mcp_call_failed(result))

    def open_news(self, args: dict[str, Any]) -> ToolResult:
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

    def browser_control(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "go_to").lower().strip()
        if action in {"go_to", "open", "new_tab"}:
            url = str(args.get("url") or args.get("target") or "").strip()
            if not url:
                return ToolResult("No URL supplied.", ok=False)
            url = self._normalise_url(SecuritySanitiser.guard_text(url, "browser_control.url"))
            open_url(url)
            return ToolResult(f"Browser opened: {url}.")
        if action in {"search", "web_search"}:
            return self.web_search({"query": args.get("query") or args.get("text") or ""})
        return ToolResult(f"Browser action '{action}' is not supported by the native dispatcher.", ok=False)

    async def web_control(self, args: dict[str, Any]) -> ToolResult:
        if self.web is None:
            return ToolResult("Web controller is not available.", ok=False)
        action = str(args.get("action") or "").lower().strip()
        w = self.web
        if action in {"open", "go_to"}:
            return await w.open(str(args.get("url") or ""))
        if action == "navigate":
            return await w.navigate(str(args.get("url") or ""))
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
        if action in {"fill_form", "form"}:
            fields = args.get("fields")
            if not isinstance(fields, dict):
                return ToolResult("fill_form requires a 'fields' object of label→value.", ok=False)
            return await w.fill_form(fields, submit=bool(args.get("submit")))
        if action in {"read_page", "read"}:
            return await w.read_page()
        if action in {"browse", "narrated_scroll", "read_down", "explore"}:
            return await w.narrated_scroll(
                url=str(args.get("url") or ""),
                steps=int(args.get("steps", 6)),
                dwell=float(args.get("dwell", 1.6)),
            )
        if action in {"summarise", "summarize", "evaluate", "review"}:
            return await w.summarise_current(url=str(args.get("url") or ""))
        if action in {"download"}:
            return await w.download_current()
        if action in {"file_dialog", "upload", "save_as"}:
            return await w.handle_file_dialog(str(args.get("path") or ""))
        return ToolResult(
            f"Unsupported web_control action: {action}. Use open, navigate, new_tab, "
            "close_tab, switch_tab, accept_cookies, reject_cookies, close_popup, "
            "fill_form, read_page, browse (narrated slow-scroll), summarise, "
            "download, or file_dialog.",
            ok=False,
        )

    async def web_automation_tool(self, args: dict[str, Any]) -> ToolResult:
        """Browser co-pilot (#11): ORION drives a REAL, visible browser window
        and narrates every step over the bus, so the user watches him work."""
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

    async def notion_workspace(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list_tasks").lower().strip()
        if action in {"list_tasks", "tasks", "todo"}:
            return await self.notion.list_tasks(
                limit=int(args.get("limit") or 12),
                include_done=bool(args.get("include_done")),
            )
        if action in {"create_task", "add_task", "new_task"}:
            return await self.notion.create_task(
                title=str(args.get("title") or args.get("task") or ""),
                due=str(args.get("due") or args.get("date") or ""),
                notes=str(args.get("notes") or ""),
            )
        if action in {"complete_task", "finish_task", "done"}:
            return await self.notion.complete_task(
                str(args.get("query") or args.get("title") or "")
            )
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
            "complete_task, upcoming_events, create_event, or projects.",
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

        Real money and real world. The Twilio entry carried the warning from
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
                f"{reason}. Tell the user what this will do and what it will "
                f"cost; if they agree, call again with confirm=true and approve "
                f"the on-screen prompt.", ok=False)
        guard = getattr(self, "system_guard", None)
        if guard is None:
            # A bare dispatcher (no guard wired) keeps the older, weaker
            # flag-only gate rather than becoming unable to act at all. The
            # running app always has a guard (see OrionDispatcher.__init__).
            return None
        from .system_guard import ActionIntent, DecisionKind

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
        if action in {"list", "catalogue", "catalog", "discover"}:
            return ToolResult("Connected MCP tools:\n" + self.mcp_host.describe())
        if action in {"servers", "status"}:
            return ToolResult(self.mcp_host.list_servers())
        if action in {"enable", "disable", "on", "off"}:
            server = str(args.get("server") or args.get("name") or "").strip()
            want_on = action in {"enable", "on"}
            return ToolResult(await self.mcp_host.set_enabled(server, want_on))
        if action in {"reconnect", "restart"}:
            server = str(args.get("server") or args.get("name") or "").strip()
            if not server:
                return ToolResult("Which MCP server should I reconnect?", ok=False)
            ok = await self.mcp_host.reconnect(server)
            return ToolResult(
                f"MCP server '{server}' reconnected." if ok
                else f"MCP server '{server}' did not come back up.", ok=ok)
        if action in {"add", "declare"}:
            raw_args = args.get("arguments") or args.get("args") or []
            return ToolResult(self.mcp_host.add_server(
                str(args.get("server") or args.get("name") or ""),
                str(args.get("command") or ""),
                [str(a) for a in raw_args] if isinstance(raw_args, list) else [],
                args.get("env") if isinstance(args.get("env"), dict) else {},
                str(args.get("description") or ""),
            ))
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

    def messaging_tool(self, args: dict[str, Any]) -> ToolResult:
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
        if action in {"email", "mail"} or (action == "send" and platform in {"email", "gmail", "mail"}):
            return self.messaging.send_email(
                str(args.get("to") or args.get("contact") or ""),
                str(args.get("subject") or ""),
                str(args.get("body") or args.get("message") or ""))
        if action == "send":
            contact = str(args.get("contact") or "")
            message = str(args.get("message") or "")
            return self.messaging.send_text(platform, contact, message)
        return ToolResult(
            f"Unsupported messaging action: {action}. Use 'send' (platform "
            "whatsapp/telegram/discord) or 'email'.", ok=False)

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
        if self.geo is None:
            return ToolResult("The geospatial engine is not available.", ok=False)
        action = str(args.get("action") or "locate").lower().strip()
        query = str(args.get("query") or args.get("place") or "")
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
            f"Unsupported geo action: {action}. Use locate, nearby, or describe.",
            ok=False,
        )

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
