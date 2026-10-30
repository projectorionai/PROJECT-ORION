"""
Where we are: the continuity brief every Live session starts with.

9 Oct 2026, 11:04: the Live channel rotated (go_away) while a "Federal Law"
research call was running. The new channel resumed from the server's last
saved point, which came before the request, so it held neither the request,
the tool call nor its result. The user's words were never logged either (their
turn never completed on the old channel), and the late result was spoken in
the local voice, where the model never heard it. Asked a minute later about
"the one you just researched into federal law", ORION said he did not recall a
recent research session, and later called it "a brief cognitive
discontinuity". Memory was intact; nothing had told the new session.

A restart is the same failure at full size: every Live session after a
restart began with no recent conversation at all, and every update is a
restart. In the owner's journal (26 Sep - 12 Oct) 28 of 75 Live sessions began
that way; 41 of the 47 in-process reconnects resumed with a handle, and those
lose only what came after the server's last saved point - which is exactly
what a long tool call straddles.

Two texts are built here:

* :func:`build_brief` for a FRESH session (first launch or restart, a
  resumption the server refused, a model change): the last turns with their
  times, jobs and tool calls still running or recently finished, what is
  waiting on the user, and whether ORION has just restarted - after an update
  or not. It goes into the session's system instruction.
* :func:`build_update` for a RESUMED session: only what the restored context
  can lack, i.e. what happened after the server's last saved point. It is sent
  as a context-only client turn (no reply is requested).

Both are bounded (:data:`MAX_TOKENS`), redacted (``session_log.redact``: keys,
tokens, e-mail addresses) and deterministic for the same inputs and clock. They
read a handful of indexed SQLite rows; the caller runs them off the event loop.

:class:`ActivityLedger` is the in-process record of Live tool calls that makes
"still running" and "finished while you were away" possible: the job store
holds only work started through the job tool, and a research call made inside
a voice turn is not a job.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .session_log import redact

#: Hard ceiling for either text. Estimated at four characters a token, the
#: usual ratio for English prose (no tokenizer is available offline).
MAX_TOKENS = 1500
CHARS_PER_TOKEN = 4
MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN

#: How many recent turns a fresh session is given, and how long each may be.
#: ORION's own replies are clipped harder: the gist is what carries over.
TURNS = 8
USER_CHARS = 220
ORION_CHARS = 170
#: How many turns after the saved point a resumed session is given.
UPDATE_TURNS = 6
#: Tool calls, jobs and waiting items listed at most (per kind).
TOOL_LINES = 6
JOB_LINES = 5
PENDING_LINES = 4
OUTCOME_CHARS = 200
#: A finished tool call older than this is history, not "where we are".
RECENT_TOOL_S = 30 * 60.0
#: A spoken file reference belongs to the latest displayed search, not an
#: indefinite memory of whichever file happened to be touched last.
FILE_REFERENCE_S = 30 * 60.0
FILE_REFERENCE_LIMIT = 40
#: Jobs that finished within this window (or within an hour of the last
#: conversation, for a restart the next morning) are listed.
JOB_WINDOW_S = 12 * 3600.0
#: Code newer than the last conversation by more than this is an update.
CODE_SLACK_S = 60.0

HEADER = ("[WHERE WE ARE at {when} - continuity brief from your own records. Background "
          "for you; never read it out.]")
UPDATE_HEADER = ("[CONTEXT UPDATE - the live channel reconnected at {when} and resumed your "
                 "session from its last saved point ({saved}). This happened after that "
                 "point, so the restored conversation does not contain it:]")
FOOTER = ("If asked what you were doing, where you were or what you found, answer from this "
          "- for detail use job(action='list' or 'result') or recall_conversation - rather "
          "than saying you do not recall.")
UPDATE_FOOTER = "Background only: do not reply to this note or read it out."
RUNNING_TITLE = ("Still running (do not start these again; their results will be reported "
                 "when they finish):")
INTERRUPTED_NOTE = " [interrupted by the reconnect - not answered yet]"

#: Gateway lookups say nothing about where the conversation is.
_UNTRACKED = frozenset({"find_tool"})
#: Argument keys worth naming, most telling first.
_ARG_KEYS = ("action", "topic", "query", "need", "path", "folder", "file", "target",
             "name", "app", "url", "title", "question", "text")
_OUTPUT_PATH = re.compile(
    r"(?:^|\b)(?:Filed|Location):\s*[\"`]?((?:[A-Za-z]:[\\/]|/)[^\r\n\"<>|*?]+)", re.I)


def _recorded_output_paths(result: str) -> tuple[str, ...]:
    """Read exact labelled output locations from legacy tool/job text.

    Research tools emit 'Filed:' or 'Location:'. Spaces belong to that
    location; splitting on whitespace used to retain only C:\\Users\\...\\OneDrive.
    Unlabelled prose is deliberately not interpreted as a path.
    """
    paths = []
    for match in _OUTPUT_PATH.finditer(str(result)):
        path = match.group(1).strip().rstrip("`.")
        if 0 < len(path) <= 1024 and redact(path) == path and path not in paths:
            paths.append(path)
        if len(paths) >= 3:
            break
    return tuple(paths)


def wall_time() -> float:
    """The wall clock (epoch seconds) - one place, so callers that are tested
    with a patched ``time`` module never need ``time.time`` themselves."""
    return time.time()


class FileReferences:
    """Bounded, in-process references to the most recently displayed files.

    Only explicit anaphora is interpreted ("that file", "the second one").
    Ordinary names and bare numbers remain search terms. Callers use this
    for reads and opening, never to infer a write or delete target.
    """

    def __init__(self, clock_fn: Any = time.monotonic) -> None:
        self._clock = clock_fn
        self._lock = threading.Lock()
        self._paths: tuple[str, ...] = ()
        self._selected: str | None = None
        self._at = 0.0

    def remember(self, paths: Iterable[str], *, selected: str | None = None) -> None:
        unique: dict[str, str] = {}
        for value in paths:
            path = str(value)
            if path and len(path) <= 4096:
                unique.setdefault(os.path.normcase(os.path.normpath(path)), path)
            if len(unique) >= FILE_REFERENCE_LIMIT:
                break
        items = tuple(unique.values())
        with self._lock:
            self._paths = items
            self._selected = selected if selected in items else (items[0] if len(items) == 1 else None)
            self._at = self._clock()

    def resolve(self, value: str) -> tuple[bool, str | None]:
        """(is a reference, resolved path). An unresolved reference stays
        unresolved; it must never become a new search or an implicit path."""
        text = " ".join(str(value).lower().strip().rstrip(".!?").split())
        single = text in {"it", "this", "that", "this file", "that file", "the file",
                          "this folder", "that folder", "the folder", "this one", "that one"}
        ordinal = re.fullmatch(
            r"(?:the )?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|\d+(?:st|nd|rd|th))"
            r"(?: (?:one|file|folder|result|match))?", text)
        numbered = re.fullmatch(r"(?:#|(?:result|match|file)\s+#?)(\d+)", text)
        if not single and ordinal is None and numbered is None:
            return False, None
        with self._lock:
            if not self._paths or self._clock() - self._at > FILE_REFERENCE_S:
                return True, None
            if single:
                return True, self._selected
            key = ordinal.group(1) if ordinal else numbered.group(1)
            words = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh",
                     "eighth", "ninth", "tenth")
            index = (-1 if key == "last" else words.index(key) if key in words
                     else int(re.match(r"\d+", key).group()) - 1)
            if (index < 0 and key != "last") or index >= len(self._paths):
                return True, None
            return True, self._paths[index]

    def select(self, path: str) -> None:
        """Record a successful explicit read/open without discarding the
        numbered list it came from."""
        path = str(path)
        if not path or len(path) > 4096:
            self.remember([])
            return
        with self._lock:
            if path not in self._paths or self._clock() - self._at > FILE_REFERENCE_S:
                self._paths = (path,)
            self._selected = path
            self._at = self._clock()


def estimate_tokens(text: str) -> int:
    """Approximate token count at :data:`CHARS_PER_TOKEN` characters each."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN if text else 0


def one_line(value: Any, limit: int) -> str:
    """*value* redacted, on one line, and at most *limit* characters.

    Redacted BEFORE clipping, so a key cut in half by the clip is still
    recognised by its shape."""
    clean = " ".join(redact(str(value or "")).split())
    if len(clean) <= limit:
        return clean
    return clean[: max(1, limit - 1)].rstrip() + "…"


def _epoch(stamp: Any) -> float | None:
    """'2026-10-09T10:05:31Z' (utils.utc_stamp) as epoch seconds."""
    try:
        if isinstance(stamp, (float, int)) and not isinstance(stamp, bool):
            import math
            return float(stamp) if math.isfinite(stamp) else None
        return datetime.strptime(str(stamp)[:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def clock(at: Any, now: float) -> str:
    """Local wall-clock time, with the day when it is not today."""
    try:
        moment = datetime.fromtimestamp(float(at))
    except (TypeError, ValueError, OverflowError, OSError):
        return "earlier"
    days = (datetime.fromtimestamp(now).date() - moment.date()).days
    hhmm = moment.strftime("%H:%M")
    if days == 0:
        return hhmm
    if days == 1:
        return f"yesterday {hhmm}"
    if 1 < days < 7:
        return f"{moment.strftime('%a')} {hhmm}"
    return f"{moment.day} {moment.strftime('%b')} {hhmm}"


def _ago(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return "moments"
    minutes = round(seconds / 60.0)
    if minutes < 90:
        return f"{minutes} minutes"
    hours = round(seconds / 3600.0)
    if hours < 36:
        return f"{hours} hours"
    return f"{round(seconds / 86400.0)} days"


# ── the tool ledger ───────────────────────────────────────────────────────────

def describe_call(name: Any, args: Any) -> tuple[str, str]:
    """(tool name, a short redacted argument summary) for one call. A call
    through the gateway's ``use_tool`` is described as the tool it ran."""
    tool = str(name or "tool")
    params: dict[str, Any] = dict(args) if isinstance(args, dict) else {}
    if tool == "use_tool":
        inner = params.get("arguments", params.get("args", {}))
        if isinstance(inner, str):
            try:
                inner = json.loads(inner) if inner.strip() else {}
            except ValueError:
                inner = {}
        tool = str(params.get("name") or params.get("tool") or tool)
        params = inner if isinstance(inner, dict) else {}
    parts: list[str] = []
    for key in _ARG_KEYS:
        value = params.get(key)
        if value is None or isinstance(value, (dict, list, tuple, set)) or not str(value).strip():
            continue
        text = one_line(value, 80)
        parts.append(text if key == "action" else f"{key} '{text}'")
        if len(parts) >= 3:
            break
    return tool, ", ".join(parts)


@dataclass
class ToolActivity:
    """One Live tool call, from request to outcome."""

    name: str
    detail: str = ""
    started: float = 0.0
    #: Which Live session asked (the worker's session serial number).
    session: int = 0
    #: The user request this work belongs to; reconnects preserve its identity.
    request_id: int = 0
    finished: float | None = None
    ok: bool | None = None
    outcome: str = ""
    #: Exact model-visible paths are kept separately from the clipped prose.
    #: Otherwise a 200-character outcome can end halfway through a filename.
    file_paths: tuple[str, ...] = ()
    #: The model was answered "still running" and never got the result.
    detached: bool = False

    @property
    def running(self) -> bool:
        return self.finished is None

    def label(self) -> str:
        return f"{self.name} ({self.detail})" if self.detail else self.name


class ActivityLedger:
    """The last few Live tool calls, in memory. Thread-safe: calls are noted
    on the event loop and read by the brief builder on a worker thread."""

    LIMIT = 24

    def __init__(self, limit: int = LIMIT, clock_fn: Any = wall_time) -> None:
        self._items: deque[ToolActivity] = deque(maxlen=max(1, int(limit)))
        self._lock = threading.Lock()
        self._clock = clock_fn

    def started(self, name: Any, args: Any = None, *, session: int = 0,
                request_id: int = 0) -> ToolActivity:
        tool, detail = describe_call(name, args)
        activity = ToolActivity(name=tool, detail=detail, started=self._clock(),
                                session=int(session or 0), request_id=int(request_id or 0))
        if tool not in _UNTRACKED:
            with self._lock:
                self._items.append(activity)
        return activity

    def finished(self, activity: ToolActivity, payload: Any = None, *,
                 cancelled: bool = False) -> None:
        with self._lock:
            activity.finished = self._clock()
            if cancelled:
                activity.ok, activity.outcome = False, "cancelled"
                return
            data = payload if isinstance(payload, dict) else {}
            activity.ok = bool(data.get("ok", True))
            activity.outcome = one_line(data.get("result", ""), OUTCOME_CHARS)
            receipt = data.get("receipt")
            paths = receipt.get("file_paths", ()) if isinstance(receipt, dict) else ()
            if isinstance(paths, (list, tuple)):
                # Omit an overlong or redacted path instead of manufacturing a
                # truncated location which cannot actually be opened.
                activity.file_paths = tuple(path for path in paths[:3]
                                            if isinstance(path, str) and 0 < len(path) <= 1024
                                            and redact(path) == path and "\n" not in path
                                            and "\r" not in path)
            if not activity.file_paths:
                activity.file_paths = _recorded_output_paths(str(data.get("result") or ""))

    def snapshot(self) -> list[ToolActivity]:
        """Copies, oldest first."""
        with self._lock:
            return [replace(item) for item in self._items]


def late_result_note(activity: ToolActivity, now: float | None = None) -> str:
    """The context note for a tool result the model never received: it was
    answered "still running", or asked for on a channel that has since gone."""
    now = wall_time() if now is None else now
    status = "finished" if activity.ok else "did not succeed"
    outcome = f": {activity.outcome}" if activity.outcome else ""
    paths = _file_path_note(activity)
    return (f"[CONTEXT UPDATE - background result at {clock(activity.finished or now, now)}: "
            f"{activity.label()} {status}{outcome}{paths}. It was still running when you last "
            "answered, so you have not seen this result; use it if the user asks. "
            f"{UPDATE_FOOTER}]")


# ── what changed since the last conversation ─────────────────────────────────

_CODE_TIME: dict[str, float | None] = {}


def _newest_source(top: Path) -> float:
    """Newest mtime of the .py files under *top*. os.scandir carries the stat
    with each entry on Windows: MEASURED 1.6 ms for orion_core's 462 files,
    against 48 ms for Path.rglob."""
    newest = 0.0
    stack = [str(top)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name != "__pycache__" and not entry.name.startswith((".", "_quarantine")):
                            stack.append(entry.path)
                    elif entry.name.endswith(".py"):
                        try:
                            newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
                        except OSError:
                            continue
        except OSError:
            continue
    return newest


def code_time() -> float | None:
    """When the code ORION is running last changed: the build (frozen) or the
    newest source file (a source run). Measured once per process."""
    if "value" in _CODE_TIME:
        return _CODE_TIME["value"]
    value: float | None = None
    try:
        if getattr(sys, "frozen", False):
            from .build_freshness import BUILD_INFO_NAME, read_build_info
            from .constants import RESOURCE_DIR
            stamps = []
            try:
                stamps.append(float((read_build_info() or {}).get("built_at") or 0.0))
            except (TypeError, ValueError):
                pass
            for path in (Path(sys.executable), Path(RESOURCE_DIR) / BUILD_INFO_NAME):
                try:
                    stamps.append(path.stat().st_mtime)
                except OSError:
                    continue
            value = max(stamps, default=0.0) or None
        else:
            from .constants import PACKAGE_DIR
            value = _newest_source(Path(PACKAGE_DIR)) or None
    except Exception:
        value = None
    _CODE_TIME["value"] = value
    return value


# ── reading the stores ────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str
    content: str
    at: float | None


def _turns(memory: Any, limit: int) -> list[Turn]:
    """The newest *limit* episodes, oldest first, from the durable log."""
    recall = getattr(memory, "recall_episodes", None)
    if recall is None:
        return []
    try:
        rows = recall("", limit=limit) or []
    except Exception:
        return []
    turns = [Turn(str(row.get("role") or ""), str(row.get("content") or ""),
                  _epoch(row.get("created_at"))) for row in rows if isinstance(row, dict)]
    return [t for t in reversed(turns) if t.content.strip()]


def _speaker(role: str) -> str:
    low = role.lower()
    if low.startswith("user"):
        who = "User"
    elif low.startswith(("orion", "assistant")):
        who = "ORION"
    else:
        who = role[:16] or "Note"
    return f"{who} (phone)" if "remote" in low else who


def _turn_line(turn: Turn, now: float, note: str = "") -> str:
    limit = USER_CHARS if _speaker(turn.role).startswith("User") else ORION_CHARS
    return f"- {clock(turn.at, now)} {_speaker(turn.role)}: {one_line(turn.content, limit)}{note}"


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", "", str(text).lower()).split()


def _same_words(a: str, b: str) -> bool:
    return bool(_words(a)) and _words(a) == _words(b)


def _store(store: Any) -> Any:
    if store is not None:
        return store
    try:
        from .job_store import default_store
        return default_store()
    except Exception:
        return None


def _job_label(job: dict[str, Any]) -> str:
    label = str(job.get("label") or job.get("name") or "job")
    args = job.get("args") if isinstance(job.get("args"), dict) else {}
    _tool, detail = describe_call(job.get("name"), args)
    if detail and detail.lower() not in label.lower():
        label = f"{label} ({detail})"
    return one_line(label, 110)


def _job_progress(job: dict[str, Any]) -> str:
    bits = [str(job.get("state") or "").replace("_", " ")]
    fraction = job.get("fraction")
    if isinstance(fraction, (int, float)):
        bits.append(f"{int(round(float(fraction) * 100))}%")
    if job.get("total"):
        bits.append(f"step {job.get('step') or 0} of {job.get('total')}")
    note = one_line(job.get("note") or "", 80)
    if note:
        bits.append(note)
    return ", ".join(b for b in bits if b)


def _job_result(job: dict[str, Any]) -> str:
    result = str(job.get("result") or "")
    first = one_line(result.strip().splitlines()[0] if result.strip() else "", 140)
    paths = _recorded_output_paths(result)
    if paths and paths[0] not in first:
        first = f"{first} (output {paths[0]})".strip()
    return first or "no result recorded"


def _jobs(store: Any, since: float) -> tuple[list[dict], list[dict]]:
    """(jobs still live, jobs that ended - finished or interrupted - since
    *since*), oldest first. Read-only queries on the job store."""
    if store is None:
        return [], []
    try:
        from .job_store import LIVE, State
        live = store.list(states=[s.value for s in LIVE], limit=JOB_LINES)
        ended = [job for job in store.list(
            states=[State.COMPLETED.value, State.FAILED.value, State.CANCELLED.value,
                    State.INTERRUPTED.value], limit=JOB_LINES * 3)
            if float(job.get("updated_at") or 0.0) >= since][:JOB_LINES]
        return list(reversed(live)), list(reversed(ended))
    except Exception:
        return [], []


def _approvals(store: Any, now: float) -> list[str]:
    """Approvals a job is waiting for, as lines."""
    if store is None:
        return []
    try:
        rows = store.pending_approvals()
    except Exception:
        return []
    lines = []
    for row in rows[:PENDING_LINES]:
        try:
            job = store.get(row.get("job_id")) or {}
        except Exception:
            job = {}
        reason = one_line(row.get("reason") or "", 90)
        lines.append(f"approval for step {int(row.get('step_idx') or 0) + 1} of "
                     f"{_job_label(job) if job else 'a job'} ({row.get('tool')})"
                     + (f": {reason}" if reason else "")
                     + f", open until {clock(row.get('expires_at'), now)}")
    return lines


def _tool_line(activity: ToolActivity, now: float) -> str:
    if activity.running:
        return f"- {activity.label()} - asked at {clock(activity.started, now)}, still running"
    when = clock(activity.finished, now)
    paths = _file_path_note(activity)
    if activity.ok:
        return f"- {when} {activity.label()}: {activity.outcome or 'done'}{paths}"
    return f"- {when} {activity.label()}: FAILED - {activity.outcome or 'no reason given'}{paths}"


def _file_path_note(activity: ToolActivity) -> str:
    paths = [path for path in activity.file_paths if path not in activity.outcome]
    return " [exact paths: " + "; ".join(paths) + "]" if paths else ""


def _job_line(job: dict[str, Any], now: float) -> str:
    state = str(job.get("state") or "")
    if state in {"completed", "failed", "cancelled"}:
        return (f"- {clock(job.get('finished_at') or job.get('updated_at'), now)} job "
                f"{_job_label(job)} {state}: {_job_result(job)}")
    if state == "interrupted":
        return f"- job {_job_label(job)}: {one_line(job.get('note') or 'interrupted', 90)}"
    return (f"- job {_job_label(job)}: {_job_progress(job)}, started "
            f"{clock(job.get('started_at') or job.get('created_at'), now)}")


# ── assembling the texts ─────────────────────────────────────────────────────

def _research_lines(store: Any, *, since: float, now: float) -> list[str]:
    """Durable observations survive a restart; reading never replays work."""
    try:
        if store is None:
            from . import research_runs
            if research_runs._DEFAULT is None and not research_runs.DEFAULT_PATH.exists():
                return []
            store = research_runs.default_store()
        if getattr(store, "storage_error", ""):
            return ["- Research records unavailable; use research(action='status') for the error."]
        rows = store.list(limit=128)
        active = [row for row in rows if row.get("research_state") in
                  {"running", "accepted", "cancel_requested"}]
        ended = [row for row in rows if row not in active
                 and (_epoch(row.get("updated_at")) or 0) > since]
        active_slots = 2 if ended else 3
        selected = active[:active_slots]
        selected += ended[:3 - len(selected)]
        lines = []
        for row in sorted(selected, key=lambda item: _epoch(item.get("updated_at")) or 0):
            updated = _epoch(row.get("updated_at"))
            state = one_line(row.get("research_state"), 32)
            line = (f"- {clock(updated, now)} research run {one_line(row.get('run_id'), 64)} "
                    f"[{state}] {one_line(row.get('topic'), 160)}: "
                    f"{one_line(row.get('result'), 160)}")
            path = row.get("selected_path")
            if isinstance(path, str) and path and len(path) <= 1024 and redact(path) == path:
                line += f"; recorded artifact: {path}"
            if state in {"running", "accepted", "cancel_requested"}:
                line += "; do not start again; query this run's status"
            elif state == "interrupted":
                line += "; not resumed; do not claim this work completed"
            lines.append(line)
        if len(active) > active_slots:
            lines.append(f"- {len(active) - active_slots} additional active research runs; "
                         "use research(action='runs') to see their IDs; do not restart them.")
        if lines:
            return [*lines, "- Use research(action='status', run_id='...') for current "
                    "state and artifact checks, or action='open' for that run. A saved partial "
                    "artifact does not prove completed research."]
        return []
    except Exception:
        return ["- Research records unavailable; check research status before making claims."]

@dataclass
class Brief:
    """A continuity text and what went into it."""

    text: str = ""
    kind: str = "none"            # "brief" | "update" | "none"
    tokens: int = 0
    build_ms: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)


def _render(head: Sequence[str], blocks: Sequence[list], tail: Sequence[str]) -> str:
    out = list(head)
    for title, lines in blocks:
        if lines:
            out.append(title)
            out.extend(lines)
    out.extend(tail)
    return "\n".join(out)


def _fit(head: Sequence[str], blocks: Sequence[tuple[str, Sequence[str]]],
         tail: Sequence[str], trim: Sequence[tuple[int, int]]) -> str:
    """Render within MAX_CHARS. *trim* names the blocks to shed lines from, in
    order, oldest line first, each down to a minimum; a hard clip at a line
    boundary is the last resort."""
    parts = [[title, list(lines)] for title, lines in blocks]
    text = _render(head, parts, tail)
    for index, keep in trim:
        lines = parts[index][1]
        while len(text) > MAX_CHARS and len(lines) > keep:
            del lines[0]
            text = _render(head, parts, tail)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit("\n", 1)[0]
    return text


def build_brief(memory: Any = None, *, reason: str = "launch", now: float | None = None,
                interrupted: str = "", ledger: ActivityLedger | None = None, store: Any = None,
                pending: Iterable[str] = (), process_started: float | None = None,
                code_stamp: float | None = None, model_changed: bool = False,
                research_store: Any = None) -> Brief:
    """The full brief for a FRESH Live session. *reason* is ``"launch"`` for
    the first session of this run and ``"reconnect"`` for a later one."""
    began = time.perf_counter()
    now = wall_time() if now is None else float(now)
    turns = _turns(memory, TURNS + 4)
    started = now if process_started is None else float(process_started)
    head = [HEADER.format(when=clock(now, now))]

    earlier = [t.at for t in turns if t.at is not None and t.at < started - 1.0]
    previous_end = max(earlier) if earlier else None
    if reason == "launch":
        if previous_end is None:
            head.append("- You have just started; there is no earlier conversation on record.")
        else:
            updated = code_stamp is not None and code_stamp > previous_end + CODE_SLACK_S
            line = (f"- You started at {clock(started, now)}"
                    + (" after an update to your software" if updated else "")
                    + f"; the previous conversation ended {clock(previous_end, now)} "
                    f"({_ago(started - previous_end)} earlier).")
            if updated:
                line += (" If it comes up, say simply that you restarted and what the user "
                         "was last doing; describe the update only if asked.")
            head.append(line)
    else:
        head.append(f"- The live channel reconnected at {clock(now, now)} as a fresh session"
                    + (" on a different voice model" if model_changed else "")
                    + " (the previous one could not be resumed), so the earlier "
                    "conversation is only what is listed here.")

    shown = turns[-TURNS:]
    turn_lines = [_turn_line(t, now) for t in shown]
    said = str(interrupted or "").strip()
    if said:
        users = [i for i, t in enumerate(shown) if _speaker(t.role).startswith("User")]
        if users and _same_words(shown[users[-1]].content, said):
            turn_lines[users[-1]] += INTERRUPTED_NOTE
        else:
            turn_lines.append(_turn_line(Turn("user", said, now), now, INTERRUPTED_NOTE))

    activities = ledger.snapshot() if ledger is not None else []
    running = [a for a in activities if a.running][-TOOL_LINES:]
    finished = [a for a in activities if not a.running
                and now - float(a.finished or 0.0) <= RECENT_TOOL_S][-TOOL_LINES:]
    store = _store(store)
    window_start = now - JOB_WINDOW_S
    if previous_end is not None:
        window_start = min(window_start, previous_end - 3600.0)
    live_jobs, ended_jobs = _jobs(store, window_start)
    stopped = [j for j in ended_jobs if j.get("state") == "interrupted"]
    done = [j for j in ended_jobs if j.get("state") != "interrupted"]
    in_flight = [_tool_line(a, now) for a in running] + [_job_line(j, now) for j in live_jobs]
    finished_lines = [_tool_line(a, now) for a in finished] + [_job_line(j, now) for j in done]
    waiting = [one_line(p, 160) for p in pending if str(p or "").strip()]
    waiting = (waiting + _approvals(store, now))[:PENDING_LINES]

    research = _research_lines(research_store, since=window_start, now=now)
    blocks = [("Recorded research (execution evidence; check status before reopening):", research),
              ("Last conversation (oldest first):", turn_lines),
              (RUNNING_TITLE, in_flight),
              ("Stopped when you last shut down, not resumed:",
               [_job_line(j, now) for j in stopped]),
              ("Recently finished:", finished_lines),
              ("Waiting on the user:", [f"- {w}" for w in waiting])]
    text = _fit(head, blocks, [FOOTER], trim=((1, 2), (4, 1), (3, 1), (2, 2), (5, 1), (0, 2)))
    return Brief(text=text, kind="brief", tokens=estimate_tokens(text),
                 build_ms=(time.perf_counter() - began) * 1000.0,
                 counts={"turns": len(turn_lines), "running": len(in_flight),
                         "stopped": len(stopped), "finished": len(finished_lines),
                         "waiting": len(waiting), "research": max(0, len(research) - 1)})


def build_update(memory: Any = None, *, since: float | None, now: float | None = None,
                 interrupted: str = "", ledger: ActivityLedger | None = None, store: Any = None,
                 pending: Iterable[str] = (), research_store: Any = None) -> Brief:
    """What a RESUMED session needs: what happened after its saved point
    (*since*, epoch seconds - when the server last sent a resumable handle).
    Empty (kind ``"none"``) when nothing did."""
    began = time.perf_counter()
    now = wall_time() if now is None else float(now)
    since = now - 300.0 if since is None else float(since)

    # A saved handle does not prove that later turns were restored, even when
    # they completed immediately afterwards. Keep the saved boundary strict.
    later = [t for t in _turns(memory, UPDATE_TURNS + 6)
             if t.at is not None and t.at > since][-UPDATE_TURNS:]
    turn_lines = [_turn_line(t, now) for t in later]
    said = str(interrupted or "").strip()
    if said and not any(_same_words(t.content, said) for t in later):
        turn_lines.append(_turn_line(Turn("user", said, now), now, INTERRUPTED_NOTE))

    activities = ledger.snapshot() if ledger is not None else []
    running = [a for a in activities if a.running][-TOOL_LINES:]
    finished = [a for a in activities if not a.running
                and float(a.finished or 0.0) > since][-TOOL_LINES:]
    store = _store(store)
    live_jobs, ended_jobs = _jobs(store, since)
    in_flight = [_tool_line(a, now) for a in running] + [
        _job_line(j, now) for j in live_jobs if float(j.get("updated_at") or 0.0) > since]
    finished_lines = [_tool_line(a, now) for a in finished] + [
        _job_line(j, now) for j in ended_jobs]
    waiting = [one_line(p, 160) for p in pending if str(p or "").strip()][:PENDING_LINES]

    research = _research_lines(research_store, since=since, now=now)
    blocks = [("Recorded research changed since then:", research),
              ("Said since then:", turn_lines),
              (RUNNING_TITLE, in_flight),
              ("Finished since then (you have not seen these results):", finished_lines),
              ("Waiting on the user:", [f"- {w}" for w in waiting])]
    if not any(lines for _title, lines in blocks):
        return Brief(kind="none", build_ms=(time.perf_counter() - began) * 1000.0)
    head = [UPDATE_HEADER.format(when=clock(now, now), saved=clock(since, now))]
    text = _fit(head, blocks, [UPDATE_FOOTER], trim=((1, 2), (3, 1), (2, 2), (4, 1), (0, 2)))
    return Brief(text=text, kind="update", tokens=estimate_tokens(text),
                 build_ms=(time.perf_counter() - began) * 1000.0,
                 counts={"turns": len(turn_lines), "running": len(in_flight),
                         "finished": len(finished_lines), "waiting": len(waiting),
                         "research": max(0, len(research) - 1)})


__all__ = ["ActivityLedger", "Brief", "FileReferences", "MAX_TOKENS", "ToolActivity", "build_brief",
           "build_update", "clock", "code_time", "describe_call", "estimate_tokens",
           "late_result_note", "one_line", "wall_time"]
