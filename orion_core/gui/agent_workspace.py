"""
AgentWorkspaceView — one shared template, six domain fills (Mark XX design-
spec §6): each specialist agent gets a real professional workspace instead
of sharing one generic combo-box-and-textbox form.

Four fixed regions, populated per-agent:
    BRIEF     — the request, optional context, domain quick-start prompts,
                and file/folder attachment (see below)
    FINDINGS  — the answer, plus the toolbelt's evidence trail (finding.tools,
                surfaced via ToolResult.evidence — previously gathered and
                discarded at the dispatch() boundary, now shown here)
    CONTEXT   — this session's past consults with this specialist, so a
                follow-up question isn't starting from nothing
    TOOLS     — which read-only instruments this specialist can actually
                reach for right now (AgentManager.available_instruments()),
                so "what does this agent know how to look up" is visible
                instead of implicit

Every workspace talks to the SAME AgentManager.dispatch() the dropdown-panel
in dashboard.py and voice/text agent_dispatch calls already use — nothing
here is a parallel or weaker implementation, only a dedicated presentation
of the same specialist.

Attachments (any specialist, not creative-only): BRIEF's "Add File(s)" / "Add
Folder" buttons run the selection through the SAME IngestionEngine that backs
the LIBRARY deck — no bespoke parsing here — and fold each file's already-
computed directive/summary into the plain-text ``context`` string every
dispatch() call already accepts. That reuses IngestionEngine's offline
extraction (code/prose/PDF/DOCX/XLSX/images) instead of duplicating it, keeps
attachments within the existing 2000-char context budget (a distilled
one-liner per file, not a raw dump), and means a file attached once is
recognised instantly on a later visit via IngestionEngine's own fingerprint
cache. Model-driven tool calls (AgentToolbelt/investigate()) were considered
and rejected for this: the specialist may or may not choose to call a tool,
where attaching a file must deterministically put it in front of the
specialist, and the toolbelt factory is wired once, globally, in app.py, with
no per-call override in AgentManager.dispatch() to hang a per-workspace
attachment tool from.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from .. import background

_MAX_HISTORY = 40


def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panelFrame")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("panelHeading")
    layout.addWidget(heading)
    return frame, layout


class AgentWorkspaceView(QWidget):
    """One specialist's professional workspace: Brief / Findings / Context / Tools."""

    def __init__(
        self,
        bus: OrionBus,
        agent_manager: Any,
        agent_name: str,
        page_title: str,
        quick_prompts: Sequence[str] = (),
        ingestion: Any | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.bus = bus
        self.agent_manager = agent_manager
        self.agent_name = agent_name
        self.page_title = page_title
        self.quick_prompts = list(quick_prompts)
        # Shared IngestionEngine — the SAME instance the LIBRARY deck uses, so
        # a file attached here is already known (or becomes known) there too.
        # None when the caller hasn't wired one in, which degrades the attach
        # buttons to a clear status message rather than a crash.
        self.ingestion = ingestion
        self._history: list[dict[str, Any]] = []
        self._attachments: list[dict[str, Any]] = []
        self._build_ui()
        self._refresh_header()
        self._refresh_tools()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(self.page_title.upper())
        title.setObjectName("titleLabel")
        header.addWidget(title)
        header.addStretch(1)
        self.activity_label = QLabel("")
        self.activity_label.setObjectName("mutedLabel")
        header.addWidget(self.activity_label)
        outer.addLayout(header)

        info = self._agent_info()
        subtitle = QLabel(info.get("focus", ""))
        subtitle.setObjectName("mutedLabel")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self._build_brief_panel(), 0, 0)
        grid.addWidget(self._build_tools_panel(), 0, 1)
        grid.addWidget(self._build_findings_panel(), 1, 0, 1, 2)
        grid.addWidget(self._build_context_panel(), 2, 0, 1, 2)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 2)
        outer.addLayout(grid, 1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedLabel")
        outer.addWidget(self.status_label)

    def _build_brief_panel(self) -> QFrame:
        frame, layout = _panel("BRIEF")
        self.request_input = QPlainTextEdit()
        self.request_input.setPlaceholderText("What do you need from this specialist?")
        self.request_input.setMaximumHeight(90)
        layout.addWidget(self.request_input)

        self.context_input = QLineEdit()
        self.context_input.setPlaceholderText("Optional extra context")
        layout.addWidget(self.context_input)

        attach_row = QHBoxLayout()
        attach_row.setSpacing(4)
        add_files_btn = QPushButton("Add File(s)")
        add_files_btn.setToolTip(
            "Attach files so this specialist can see what's in them")
        add_files_btn.clicked.connect(self._pick_files)
        attach_row.addWidget(add_files_btn)
        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.setToolTip(
            "Attach a folder so this specialist can see what's in it")
        add_folder_btn.clicked.connect(self._pick_folder)
        attach_row.addWidget(add_folder_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_attachments)
        attach_row.addWidget(clear_btn)
        attach_row.addStretch(1)
        layout.addLayout(attach_row)

        self.attachments_label = QLabel("No files attached.")
        self.attachments_label.setObjectName("mutedLabel")
        self.attachments_label.setWordWrap(True)
        layout.addWidget(self.attachments_label)

        if self.quick_prompts:
            prompts_label = QLabel("Quick start")
            prompts_label.setObjectName("mutedLabel")
            layout.addWidget(prompts_label)
            prompt_row = QHBoxLayout()
            prompt_row.setSpacing(4)
            for prompt in self.quick_prompts:
                btn = QPushButton(prompt)
                btn.clicked.connect(lambda _c=False, p=prompt: self._use_quick_prompt(p))
                prompt_row.addWidget(btn)
            layout.addLayout(prompt_row)

        consult_btn = QPushButton("Consult")
        consult_btn.clicked.connect(self._consult)
        layout.addWidget(consult_btn)
        layout.addStretch(1)
        return frame

    def _build_tools_panel(self) -> QFrame:
        frame, layout = _panel("INSTRUMENTS")
        self.tools_label = QLabel("")
        self.tools_label.setObjectName("mutedLabel")
        self.tools_label.setWordWrap(True)
        layout.addWidget(self.tools_label)
        layout.addStretch(1)
        return frame

    def _build_findings_panel(self) -> QFrame:
        frame, layout = _panel("FINDINGS")
        self.findings_output = QPlainTextEdit()
        self.findings_output.setReadOnly(True)
        self.findings_output.setObjectName("logBox")
        self.findings_output.setPlaceholderText("The specialist's answer appears here.")
        layout.addWidget(self.findings_output, 1)
        return frame

    def _build_context_panel(self) -> QFrame:
        frame, layout = _panel("CONTEXT — THIS SESSION")
        self.history_list = QListWidget()
        self.history_list.itemActivated.connect(self._on_history_selection)
        layout.addWidget(self.history_list, 1)
        return frame

    # ── header / tools refresh ───────────────────────────────────────────────

    def _agent_info(self) -> dict[str, Any]:
        try:
            for row in self.agent_manager.describe():
                if row.get("name") == self.agent_name:
                    return row
        except Exception:
            pass
        return {}

    def _refresh_header(self) -> None:
        info = self._agent_info()
        calls = info.get("calls", 0)
        last_active = info.get("last_active") or "never"
        self.activity_label.setText(f"{calls} call(s) · last active {last_active}")

    def _refresh_tools(self) -> None:
        try:
            names = self.agent_manager.available_instruments()
        except Exception:
            names = []
        if names:
            self.tools_label.setText("Can consult:\n" + "\n".join(f"• {n}" for n in sorted(names)))
        else:
            self.tools_label.setText("No read-only instruments attached yet — "
                                     "answers come from persona + reasoning alone.")

    # ── actions ───────────────────────────────────────────────────────────────

    def _use_quick_prompt(self, prompt: str) -> None:
        self.request_input.setPlainText(prompt)

    def _run(self, factory) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        background.spawn(factory())

    def _consult(self) -> None:
        request = self.request_input.toPlainText().strip()
        if not request:
            self.status_label.setText("Type a request first.")
            return
        self._run(self._consult_async)

    async def _consult_async(self) -> None:
        request = self.request_input.toPlainText().strip()
        context = self._build_context()
        self.status_label.setText(f"Consulting the {self.page_title}…")
        result = await self.agent_manager.dispatch(
            request, agent_name=self.agent_name, context=context)
        self._render_result(request, result)
        self._refresh_header()

    def _build_context(self) -> str:
        """Free-text context plus a digest of anything attached below.

        Attached files/folders are not dumped in full — ``handle``/``dispatch``
        truncate ``context`` to 2000 chars regardless of what a workspace
        sends — so each attachment contributes the one-line directive/summary
        IngestionEngine already distilled for it (the same digest the LIBRARY
        deck's table shows), not raw file contents.
        """
        parts: list[str] = []
        manual = self.context_input.text().strip()
        if manual:
            parts.append(manual)
        if self._attachments:
            lines = [
                f"- [{a['category']}] {Path(a['path']).name} — {a['digest']}"
                for a in self._attachments
            ]
            parts.append("Attached for reference:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _render_result(self, request: str, result: Any) -> None:
        self.findings_output.setPlainText(result.text)
        if getattr(result, "evidence", None):
            lines = [f"\n\nSources consulted ({len(result.evidence)}):"]
            for item in result.evidence:
                marker = "✓" if item.get("ok") else "✗"
                lines.append(f"  {marker} {item.get('summary', '')}")
            self.findings_output.appendPlainText("\n".join(lines))
        self.status_label.setText("" if result.ok else "The specialist reported a problem.")
        entry = {
            "at": datetime.now().strftime("%H:%M:%S"),
            "request": request,
            "answer": result.text,
        }
        self._history.append(entry)
        del self._history[:-_MAX_HISTORY]
        item = QListWidgetItem(f"{entry['at']}  {request[:70]}")
        item.setData(Qt.ItemDataRole.UserRole, len(self._history) - 1)
        self.history_list.addItem(item)
        self.history_list.scrollToBottom()

    def _on_history_selection(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None or not (0 <= index < len(self._history)):
            return
        entry = self._history[index]
        self.findings_output.setPlainText(entry["answer"])
        self.request_input.setPlainText(entry["request"])

    # ── attachments (file/folder import) ────────────────────────────────────

    def _pick_files(self) -> None:
        if self.ingestion is None:
            self.status_label.setText("File import is not available on this workspace.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Attach file(s) for the {self.page_title}")
        if paths:
            self._run(lambda: self._attach_async(paths, is_folder=False))

    def _pick_folder(self) -> None:
        if self.ingestion is None:
            self.status_label.setText("File import is not available on this workspace.")
            return
        path = QFileDialog.getExistingDirectory(
            self, f"Attach a folder for the {self.page_title}")
        if path:
            self._run(lambda: self._attach_async([path], is_folder=True))

    async def _attach_async(self, paths: Sequence[str], is_folder: bool) -> None:
        """Read file(s)/a folder through the shared IngestionEngine off the UI
        thread — ``asyncio.to_thread``, the same idiom every other blocking
        call in this app uses to stay off the Qt event loop — then fold a
        compact digest of each result into this panel's attachments. Reuses
        IngestionEngine's own fingerprint cache, so a file attached before
        (here or on the LIBRARY deck) attaches again instantly.
        """
        label = Path(paths[0]).name if len(paths) == 1 else f"{len(paths)} item(s)"
        self.status_label.setText(f"Reading {label} for the {self.page_title}…")
        try:
            if is_folder:
                batch = await asyncio.to_thread(self.ingestion.ingest_folder, paths[0])
                results = list(batch.results)
            else:
                results = await asyncio.to_thread(
                    lambda: [self.ingestion.ingest_file(p) for p in paths])
        except Exception as exc:
            self.status_label.setText(f"Could not read that: {exc}")
            return
        added = 0
        for result in results:
            if not result.ok:
                continue
            digest = result.directive or result.summary or result.note or "(no summary available)"
            self._attachments.append({
                "path": result.path,
                "category": result.category or "file",
                "digest": digest,
            })
            added += 1
        self._refresh_attachments()
        skipped = len(results) - added
        if added:
            detail = f" ({skipped} unreadable)" if skipped else ""
            self.status_label.setText(f"Attached {added} file(s){detail}.")
        else:
            self.status_label.setText("No readable files found in that selection.")

    def _refresh_attachments(self) -> None:
        if not self._attachments:
            self.attachments_label.setText("No files attached.")
            self.attachments_label.setToolTip("")
            return
        names = [Path(a["path"]).name for a in self._attachments]
        shown = ", ".join(names[:6]) + ("…" if len(names) > 6 else "")
        self.attachments_label.setText(f"{len(names)} attached: {shown}")
        self.attachments_label.setToolTip(
            "\n".join(f"{Path(a['path']).name} — {a['digest']}" for a in self._attachments))

    def _clear_attachments(self) -> None:
        self._attachments = []
        self._refresh_attachments()
        self.status_label.setText("Attachments cleared.")
