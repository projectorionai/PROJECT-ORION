"""
Outlook integration — read, draft, summarise and (with explicit approval)
send email through the locally installed Outlook client via COM automation.

Design principles:

    • SAFETY FIRST — ORION can *create* drafts freely, but an email is only
      ever transmitted through ``send_draft(..., confirm=True)``.  Drafts are
      real Outlook drafts, so the user can always review or send them from
      Outlook itself.  Nothing is auto-sent.

    • THREADING — every COM call runs inside ``asyncio.to_thread`` with its
      own COM apartment (CoInitialize/CoUninitialize), because Outlook COM
      objects must never be touched from the qasync GUI loop.

    • GRACEFUL DEGRADATION — if pywin32 or Outlook is absent, every method
      returns a clear, spoken-friendly explanation instead of raising.

    • LAZY CONNECTION (Mark X.5) — Outlook is NEVER launched as a side effect
      of startup.  Every COM call first tries ``GetActiveObject`` to join an
      Outlook instance that is already running; only operations explicitly
      marked ``launch=True`` (direct user requests through the outlook_mail
      tool) may fall back to ``Dispatch``, which starts Outlook.  Passive
      callers — the dashboard email panel, the proactive survey and the
      morning briefing — are attach-only, so a machine without Outlook open
      stays that way.  Connection-state transitions are logged on the bus.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable

from .bus import OrionBus
from .data import ToolResult
from .security import SecuritySanitiser
from .utils import first_line

# Outlook OlDefaultFolders / OlItemType / OlImportance constants
OL_FOLDER_INBOX = 6
OL_MAIL_ITEM = 0
OL_IMPORTANCE_HIGH = 2


class OutlookService:
    """COM bridge to the local Outlook client with an approval-gated send path."""

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._availability: bool | None = None
        # Pending drafts created this session: short ref → Outlook EntryID.
        # (EntryIDs are long and unspeakable; refs like 'draft-1' are voice-friendly.)
        self._drafts: dict[str, dict[str, str]] = {}
        self._draft_counter = 0
        # Lazy-connection state machine: DISCONNECTED → ATTACHED | LAUNCHED.
        self._connection_state = "DISCONNECTED"

    def _note_state(self, state: str, detail: str = "") -> None:
        """Log connection-state transitions once per change (thread-safe emit)."""
        if state == self._connection_state:
            return
        self._connection_state = state
        self.bus.log.emit(f"MAIL: connection {state}" + (f" — {detail}" if detail else ""))

    @property
    def connection_state(self) -> str:
        return self._connection_state

    # ── availability ──────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        if self._availability is None:
            try:
                import win32com.client  # noqa: F401
                import pythoncom        # noqa: F401
                self._availability = True
            except Exception:
                self._availability = False
                self.bus.log.emit(
                    "MAIL: pywin32 not installed (pip install pywin32); Outlook offline."
                )
        return self._availability

    def _unavailable(self) -> ToolResult:
        return ToolResult(
            "Outlook integration is unavailable on this host. Install pywin32 "
            "(pip install pywin32) and ensure the Outlook desktop client is configured.",
            ok=False,
        )

    # ── COM plumbing ──────────────────────────────────────────────────────────

    async def _run_com(
        self, fn: Callable[[Any, Any], ToolResult], launch: bool = False
    ) -> ToolResult:
        """
        Execute *fn(outlook_app, mapi_namespace)* inside a COM-initialised thread.

        Attach-first policy: joins a running Outlook via ``GetActiveObject``.
        Only when ``launch=True`` (an explicit user request) does it fall back
        to ``Dispatch``, which starts the Outlook process.  Win32 bindings are
        imported here, inside the worker — never at module or startup time.
        """

        def _worker() -> ToolResult:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            try:
                try:
                    outlook = win32com.client.GetActiveObject("Outlook.Application")
                    self._note_state("ATTACHED", "joined the running Outlook instance")
                except Exception:
                    if not launch:
                        self._note_state("DISCONNECTED", "Outlook not running; attach-only call declined to start it")
                        return ToolResult(
                            "Outlook is not currently running, sir. Open Outlook — or ask me "
                            "for your mail directly and I shall start it for the task.",
                            ok=False,
                        )
                    outlook = win32com.client.Dispatch("Outlook.Application")
                    self._note_state("LAUNCHED", "started Outlook on explicit request")
                namespace = outlook.GetNamespace("MAPI")
                return fn(outlook, namespace)
            finally:
                pythoncom.CoUninitialize()

        try:
            return await asyncio.to_thread(_worker)
        except Exception as exc:
            self._note_state("DISCONNECTED", first_line(exc))
            return ToolResult(f"Outlook operation failed: {first_line(exc)}", ok=False)

    # ── reading ───────────────────────────────────────────────────────────────

    async def read_inbox(
        self, limit: int = 10, unread_only: bool = False, launch: bool = False
    ) -> ToolResult:
        """Latest inbox messages as a compact, speakable digest."""
        if not self.available:
            return self._unavailable()
        limit = max(1, min(25, int(limit or 10)))

        def _read(outlook: Any, namespace: Any) -> ToolResult:
            inbox = namespace.GetDefaultFolder(OL_FOLDER_INBOX)
            items = inbox.Items
            items.Sort("[ReceivedTime]", True)
            emails = self._collect_items(items, limit, unread_only=unread_only)
            if not emails:
                return ToolResult(
                    "The inbox holds no " + ("unread " if unread_only else "") + "messages."
                )
            self.bus.dashboard_event.emit("emails", emails)
            lines = [f"Inbox digest — {len(emails)} message(s):"]
            for mail in emails:
                flag = "● UNREAD" if mail["unread"] else "read"
                pri  = " ⚑ HIGH" if mail["high_importance"] else ""
                lines.append(
                    f"{mail['index']}. [{flag}{pri}] {mail['sender']} — {mail['subject']}"
                    f" ({mail['received']})\n    {mail['preview']}"
                )
            return ToolResult("\n".join(lines))

        return await self._run_com(_read, launch=launch)

    async def priority_emails(self, limit: int = 5, launch: bool = False) -> ToolResult:
        """Unread or high-importance messages — the morning-briefing feed."""
        if not self.available:
            return self._unavailable()
        limit = max(1, min(15, int(limit or 5)))

        def _read(outlook: Any, namespace: Any) -> ToolResult:
            inbox = namespace.GetDefaultFolder(OL_FOLDER_INBOX)
            items = inbox.Items
            items.Sort("[ReceivedTime]", True)
            collected = self._collect_items(items, limit * 4, unread_only=False)
            priority = [
                m for m in collected if m["unread"] or m["high_importance"]
            ][:limit]
            if not priority:
                return ToolResult("No priority email requires attention.")
            self.bus.dashboard_event.emit("emails", priority)
            lines = [f"Priority email — {len(priority)} message(s) need attention:"]
            for mail in priority:
                pri = " (high importance)" if mail["high_importance"] else ""
                lines.append(
                    f"- {mail['sender']}: {mail['subject']}{pri} — {mail['preview'][:140]}"
                )
            return ToolResult("\n".join(lines))

        return await self._run_com(_read, launch=launch)

    def _collect_items(self, items: Any, limit: int, unread_only: bool) -> list[dict[str, Any]]:
        """Walk a sorted Outlook Items collection into plain dictionaries."""
        emails: list[dict[str, Any]] = []
        index = 0
        item = items.GetFirst()
        while item is not None and len(emails) < limit and index < 200:
            index += 1
            try:
                if getattr(item, "Class", 0) != 43:  # olMail
                    item = items.GetNext()
                    continue
                unread = bool(getattr(item, "UnRead", False))
                if unread_only and not unread:
                    item = items.GetNext()
                    continue
                body = re.sub(r"\s+", " ", str(getattr(item, "Body", "") or ""))[:220]
                emails.append({
                    "index": len(emails) + 1,
                    "entry_id": str(getattr(item, "EntryID", "") or ""),
                    "sender": str(getattr(item, "SenderName", "") or "unknown sender"),
                    "subject": str(getattr(item, "Subject", "") or "(no subject)")[:140],
                    "received": str(getattr(item, "ReceivedTime", "") or "")[:19],
                    "unread": unread,
                    "high_importance": int(getattr(item, "Importance", 1) or 1) == OL_IMPORTANCE_HIGH,
                    "preview": body,
                })
            except Exception:
                pass
            item = items.GetNext()
        return emails

    async def read_email_body(self, entry_id: str) -> ToolResult:
        """Full body of a specific message (for summarisation upstream)."""
        if not self.available:
            return self._unavailable()

        def _read(outlook: Any, namespace: Any) -> ToolResult:
            item = namespace.GetItemFromID(entry_id)
            body = str(getattr(item, "Body", "") or "")
            return ToolResult(
                f"From: {getattr(item, 'SenderName', '')}\n"
                f"Subject: {getattr(item, 'Subject', '')}\n"
                f"Received: {str(getattr(item, 'ReceivedTime', ''))[:19]}\n\n"
                f"{body[:8000]}"
            )

        # Reading a specific message is always a direct user request.
        return await self._run_com(_read, launch=True)

    # ── drafting and approval-gated sending ──────────────────────────────────

    async def create_draft(self, to: str, subject: str, body: str, cc: str = "") -> ToolResult:
        """
        Compose a draft.  The draft is saved into Outlook's Drafts folder and
        registered under a speakable reference (e.g. 'draft-1').  It is NOT
        sent — sending requires an explicit, separate approval call.
        """
        if not self.available:
            return self._unavailable()
        to = SecuritySanitiser.guard_text(str(to or "").strip(), "outlook.to")
        subject = SecuritySanitiser.guard_text(str(subject or "").strip(), "outlook.subject")
        body = SecuritySanitiser.guard_text(str(body or "").strip(), "outlook.body")
        if not to or not body:
            return ToolResult("A recipient and a body are required to draft an email.", ok=False)

        def _draft(outlook: Any, namespace: Any) -> ToolResult:
            mail = outlook.CreateItem(OL_MAIL_ITEM)
            mail.To = to
            if cc.strip():
                mail.CC = cc.strip()
            mail.Subject = subject or "(no subject)"
            mail.Body = body
            mail.Save()
            self._draft_counter += 1
            ref = f"draft-{self._draft_counter}"
            self._drafts[ref] = {
                "entry_id": str(mail.EntryID),
                "to": to,
                "subject": subject,
                "preview": body[:200],
            }
            self.bus.dashboard_event.emit("drafts", self.pending_drafts())
            self.bus.banner.emit(f"EMAIL DRAFT READY: {ref} → {to}", 2)
            return ToolResult(
                f"Draft saved as {ref} (also visible in Outlook's Drafts folder).\n"
                f"To: {to}\nSubject: {subject}\n"
                "Awaiting approval — say the word and I shall send it "
                f"(send_draft with reference '{ref}' and confirm=true)."
            )

        # Drafting is always a direct user request — Outlook may be started.
        return await self._run_com(_draft, launch=True)

    async def send_draft(self, draft_ref: str, confirm: bool = False) -> ToolResult:
        """
        Transmit a previously created draft.  Refuses without confirm=True —
        this is the approval gate demanded by the Mark VIII specification.
        """
        if not self.available:
            return self._unavailable()
        draft_ref = str(draft_ref or "").strip().lower()
        record = self._drafts.get(draft_ref)
        if record is None:
            known = ", ".join(sorted(self._drafts)) or "none"
            return ToolResult(
                f"No pending draft matches '{draft_ref}'. Pending drafts: {known}.", ok=False
            )
        if not confirm:
            return ToolResult(
                f"Approval required before transmission. {draft_ref} is addressed to "
                f"{record['to']} with subject '{record['subject']}'. "
                "Repeat the request with explicit confirmation to send.",
                ok=False,
            )

        def _send(outlook: Any, namespace: Any) -> ToolResult:
            mail = namespace.GetItemFromID(record["entry_id"])
            mail.Send()
            self._drafts.pop(draft_ref, None)
            self.bus.dashboard_event.emit("drafts", self.pending_drafts())
            self.bus.banner.emit(f"EMAIL SENT → {record['to']}", 3)
            return ToolResult(
                f"Sent. {draft_ref} is on its way to {record['to']} "
                f"(subject: '{record['subject']}')."
            )

        # Approved transmission is a direct user request — Outlook may be started.
        return await self._run_com(_send, launch=True)

    def pending_drafts(self) -> list[dict[str, str]]:
        """Session drafts awaiting approval (consumed by the dashboard panel)."""
        return [
            {"ref": ref, "to": rec["to"], "subject": rec["subject"], "preview": rec["preview"]}
            for ref, rec in self._drafts.items()
        ]

    async def discard_draft(self, draft_ref: str) -> ToolResult:
        """Forget a pending draft reference (the Outlook draft itself remains)."""
        if self._drafts.pop(str(draft_ref or "").strip().lower(), None) is None:
            return ToolResult(f"No pending draft named '{draft_ref}'.", ok=False)
        self.bus.dashboard_event.emit("drafts", self.pending_drafts())
        return ToolResult(
            f"Reference {draft_ref} withdrawn. The draft remains in Outlook's "
            "Drafts folder should you wish to revisit it."
        )
