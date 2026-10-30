"""
The supervisor that runs scheduled plugins while nobody is watching.

A plugin with a ``schedule`` in its manifest is a standing instruction: check
the news at six, sweep the inbox on the half hour, compile the dossier
overnight. This is what actually carries those out.

Unattended work has to fail quietly and visibly at the same time
-----------------------------------------------------------------
Nobody is at the keyboard when these run, so two things matter more than they
would for a tool the user invoked:

  * **Nothing a plugin does may stop ORION.** A scheduled plugin that raises,
    hangs, or returns nonsense is contained: the exception is caught, the run
    is recorded as failed, and the next one is still scheduled.
  * **Repeated failure mutes.** A plugin that fails ``MUTE_AFTER`` times in a
    row stops being scheduled and says so once. Otherwise a broken plugin on
    a fifteen-minute interval writes ninety-six identical errors a day into
    the log and buries everything else.
  * **Cancellation is incomplete work.** It stays in the run history without
    changing the plugin's failure streak. Cancellation of an await does not
    establish whether an external side effect completed.

A missed window is caught up once
--------------------------------
If the machine was asleep at 06:00, the overdue job runs once when it wakes.
The next occurrence is calculated from the current clock after it finishes;
the supervisor does not replay every missed occurrence in a burst.

An overdue interval job also runs once, then starts its interval again from
completion. An explicit skip-missed policy would need a separate setting.

One at a time
-------------
Runs are serialised. Several plugins coming due in the same minute execute one
after another rather than together — they share ORION's dispatcher, its
provider quota and this machine, and a scheduler that can start six things at
once is a scheduler that will.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Iterable, Literal

#: Consecutive failures before a plugin stops being scheduled.
MUTE_AFTER = 3

#: How long a single scheduled run may take before it is abandoned. Generous,
#: because a research plugin legitimately takes minutes — but finite, because
#: a hung run would otherwise hold the queue for ever.
RUN_TIMEOUT_S = 600.0

#: How often the supervisor wakes to see what is due. Cron resolution is one
#: minute, so anything finer is wasted wakeups; anything coarser risks
#: stepping over a minute entirely.
TICK_S = 20.0

#: Runs kept in memory for the status report. The journals on disk are the
#: record; this is "what happened recently".
HISTORY = 50

RunStatus = Literal["running", "succeeded", "failed", "timed_out", "cancelled"]


@dataclass
class _SecretScope:
    plugin: str
    values: dict[str, str]
    active: bool = True
    # Reporting-only snapshots survive credential revocation in copied tasks.
    # They never participate in scoped_plugin_secret lookup.
    redactions: tuple[str, ...] = ()


_SECRET_SCOPE: ContextVar[_SecretScope | None] = ContextVar("orion_plugin_secret_scope", default=None)


@contextmanager
def plugin_secret_scope(plugin: str, values: dict[str, str]):
    """Supply only declared keys, with revocation shared by copied contexts."""
    scope = _SecretScope(plugin, dict(values), redactions=tuple(dict.fromkeys(
        (*plugin_redaction_values(), *(value for value in values.values() if value)))))
    token = _SECRET_SCOPE.set(scope)
    try:
        yield scope
    finally:
        scope.active = False
        scope.values.clear()
        _SECRET_SCOPE.reset(token)


def plugin_redaction_values() -> tuple[str, ...]:
    """Inherited reporting filters, separate from active credential access."""
    scope = _SECRET_SCOPE.get()
    return scope.redactions if scope is not None else ()


def scoped_plugin_secret(plugin: str, name: str) -> str | None:
    """None means no scheduled scope; an empty string denies scoped fallback.

    Context follows asyncio tasks and asyncio.to_thread. Revocation also reaches
    child tasks that retained their copied context. This is trusted-code hygiene,
    not an isolation boundary against arbitrary Python or deliberate logging.
    """
    scope = _SECRET_SCOPE.get()
    if scope is None:
        return None
    if not scope.active or scope.plugin != plugin:
        return ""
    return scope.values.get(name, "")


def redact_plugin_secrets(value: object, values: Iterable[str] | None = None) -> str:
    if values is None:
        values = plugin_redaction_values()
    text = str(value)
    from urllib.parse import quote
    for secret in sorted({str(item) for item in values if item}, key=len, reverse=True):
        text = text.replace(secret, "[redacted]").replace(quote(secret, safe=""), "[redacted]")
    return text


def redact_plugin_result(result: Any) -> Any:
    """Copy reporting fields before registry/history/telemetry can retain them."""
    from .data import ToolResult
    if not isinstance(result, ToolResult) or not plugin_redaction_values():
        return result

    def clean(value: Any, depth: int = 0) -> Any:
        if depth > 20:
            return "[nested content omitted]"
        if isinstance(value, str):
            return redact_plugin_secrets(value)
        if isinstance(value, dict):
            return {clean(key, depth + 1): clean(item, depth + 1) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item, depth + 1) for item in value]
        if isinstance(value, tuple):
            return tuple(clean(item, depth + 1) for item in value)
        return value

    return replace(result, text=clean(result.text), receipt=clean(result.receipt),
                   evidence=clean(result.evidence), media=clean(result.media), chain=clean(result.chain),
                   execution=clean(result.execution))


@dataclass
class Run:
    """One execution of a scheduled plugin."""

    plugin: str
    started_at: float
    finished_at: float = 0.0
    ok: bool = False
    detail: str = ""
    status: RunStatus = "running"

    @property
    def seconds(self) -> float:
        return max(0.0, (self.finished_at or time.time()) - self.started_at)

    def as_dict(self) -> dict[str, Any]:
        return {"plugin": self.plugin, "ok": self.ok, "status": self.status,
                "seconds": round(self.seconds, 2), "detail": self.detail[:200],
                "at": datetime.fromtimestamp(self.started_at).isoformat(
                    timespec="seconds")}


@dataclass
class Entry:
    """A plugin the supervisor is watching, and its state."""

    name: str
    schedule: Any
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Environment variables this plugin declared. Only these are fetched
    #: from the vault, and only for the duration of its own run.
    secrets: tuple[str, ...] = ()
    next_due: datetime | None = None
    failures: int = 0
    muted: bool = False
    runs: int = 0
    last: Run | None = None


class PluginScheduler:
    """Runs scheduled plugins, one at a time, and keeps out of the way."""

    def __init__(self, dispatcher: Any = None, bus: Any = None,
                 vault: Any = None, *, clock: Callable[[], datetime] | None = None,
                 registry: Any = None
                 ) -> None:
        self.dispatcher = dispatcher
        self.bus = bus
        self.vault = vault
        self.registry = registry
        from .time_service import TIME
        self._clock = clock or TIME.now
        self.entries: dict[str, Entry] = {}
        self.history: list[Run] = []
        self._stop = asyncio.Event()
        self._running = False

    # ── registration ─────────────────────────────────────────────────────────

    def add(self, manifest: Any) -> Entry | None:
        """Watch a plugin, if its manifest asked to be scheduled.

        Returns the entry, or None for an ordinary on-request plugin. A
        manifest whose schedule will not parse is skipped with a line in the
        log rather than taking the load down — the manifest loader has already
        validated it, so reaching here means something stranger.
        """
        name = str(getattr(manifest, "name", "") or "").strip()
        if not name:
            return None
        try:
            schedule = manifest.parsed_schedule
        except Exception as exc:
            self._log(f"'{name}' has a schedule I can't read — {exc}")
            return None
        if schedule is None:
            return None

        if self.vault is not None:
            missing = self.vault.missing(name, getattr(manifest, "secrets", ()))
            if missing:
                # Said now, not at three in the morning when it fires and
                # fails on a key nobody knew was absent.
                self._log(f"'{name}' is scheduled but its vault is missing "
                          f"{', '.join(missing)} — it will run without them.")

        entry = Entry(name=name, schedule=schedule,
                      secrets=tuple(getattr(manifest, "secrets", ()) or ()))
        if entry.secrets:
            self._log(f"'{name}' uses task-scoped vault secrets. Legacy plugins must use "
                      "jv_plugins.runtime.secret(plugin, name) or scoped_plugin_secret; "
                      "scheduled vault values are no longer injected into os.getenv.")
        entry.next_due = schedule.next_due(self._clock())
        self.entries[name] = entry
        from .plugin_schedule import describe

        self._log(f"'{name}' scheduled — {describe(schedule)}")
        return entry

    def remove(self, name: str) -> bool:
        return self.entries.pop(str(name), None) is not None

    # ── the loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Wake periodically, run whatever is due, and never die."""
        # No await between checking and claiming ownership: duplicate starts
        # on this event loop must not dispatch the same due action twice.
        if self._running:
            return
        self._running = True
        try:
            while not self._stop.is_set():
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # The supervisor outliving its plugins is the whole point.
                    self._log(f"scheduler recovered from {type(exc).__name__}: {exc}")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=TICK_S)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._running

    async def _tick(self) -> None:
        from .plugin_schedule import _instant
        now = self._clock()
        due = [entry for entry in self.entries.values()
               if self._enabled(entry.name) and not entry.muted and entry.next_due is not None
               and _instant(entry.next_due) <= _instant(now)]
        # Oldest deadline first, so a schedule that has been waiting longest
        # is not starved by one that comes due every minute.
        for entry in sorted(due, key=lambda e: _instant(e.next_due or now)):
            if self._stop.is_set():
                return
            # A preceding awaited run may have removed, replaced, muted or
            # rescheduled this entry. The due-list snapshot is not authority
            # to execute a plugin that is no longer scheduled.
            current = self._clock()
            if (self.entries.get(entry.name) is not entry or not self._enabled(entry.name) or entry.muted
                    or entry.next_due is None or _instant(entry.next_due) > _instant(current)):
                continue
            await self._execute(entry)
            entry.next_due = entry.schedule.next_due(self._clock())

    def _enabled(self, name: str) -> bool:
        if self.registry is None:
            return True
        try:
            return bool(self.registry.is_enabled(name))
        except Exception:
            return False

    @contextmanager
    def _with_secrets(self, entry: Entry):
        """Supply declared secrets without mutating the process environment."""
        secrets: dict[str, str] = {}
        if self.vault is not None and entry.secrets:
            try:
                found = self.vault.secrets_for(entry.name, entry.secrets)
                secrets = {key: value for key, value in found.items()
                           if key in entry.secrets and isinstance(value, str)}
            except Exception as exc:
                raise RuntimeError("Scheduled plugin secrets could not be loaded; restore or repair the vault.") from exc
        with plugin_secret_scope(entry.name, secrets):
            yield secrets

    async def _execute(self, entry: Entry) -> Run:
        """Contain plugin failures and preserve caller cancellation."""
        run = Run(plugin=entry.name, started_at=time.time())
        entry.runs += 1
        redactions = plugin_redaction_values()
        try:
            if self.dispatcher is None:
                raise RuntimeError("no dispatcher is attached")
            with self._with_secrets(entry) as secrets:
                redactions = plugin_redaction_values()
                from .execution_policy import background_scope
                with background_scope("plugin_schedule", run_id=f"{entry.name}:{entry.runs}"):
                    result = await asyncio.wait_for(
                        self.dispatcher.dispatch(entry.name, dict(entry.arguments)),
                        timeout=RUN_TIMEOUT_S)
            ok = getattr(result, "ok", None)
            text = getattr(result, "text", None)
            if type(ok) is not bool or not isinstance(text, str):
                run.status = "failed"
                run.detail = "Dispatcher returned an invalid outcome: expected boolean ok and text result."
            else:
                run.ok = ok
                run.status = "succeeded" if ok else "failed"
                run.detail = redact_plugin_secrets(text, redactions)[:400]
        except asyncio.TimeoutError:
            run.ok = False
            run.status = "timed_out"
            run.detail = (f"timed out after {RUN_TIMEOUT_S:g}s; "
                          "completion and side effects are unverified")
        except asyncio.CancelledError:
            run.ok = False
            run.status = "cancelled"
            run.detail = "cancelled before a result; completion and side effects are unverified"
            run.finished_at = time.time()
            self._record(entry, run)
            raise
        except Exception as exc:
            run.ok = False
            run.status = "failed"
            run.detail = redact_plugin_secrets(f"{type(exc).__name__}: {exc}", redactions)

        run.finished_at = time.time()
        self._record(entry, run)
        return run

    def _record(self, entry: Entry, run: Run) -> None:
        entry.last = run
        self.history.append(run)
        del self.history[:-HISTORY]

        if run.status == "cancelled":
            self._log(f"'{entry.name}' — {run.detail}")
            return

        if run.ok:
            if entry.failures:
                self._log(f"'{entry.name}' is working again.")
            entry.failures = 0
            return

        entry.failures += 1
        if entry.failures >= MUTE_AFTER:
            entry.muted = True
            self._log(f"'{entry.name}' failed {entry.failures} times in a row "
                      f"and will not be scheduled again this session — "
                      f"{run.detail}")
        else:
            self._log(f"'{entry.name}' failed ({entry.failures}/{MUTE_AFTER}) "
                      f"— {run.detail}")

    def unmute(self, name: str) -> bool:
        """Put a muted plugin back on the schedule, after it has been fixed."""
        entry = self.entries.get(str(name))
        if entry is None:
            return False
        entry.muted = False
        entry.failures = 0
        entry.next_due = entry.schedule.next_due(self._clock())
        return True

    # ── reporting ────────────────────────────────────────────────────────────

    def status(self) -> list[dict[str, Any]]:
        """What is scheduled, when it next runs, and how it last went."""
        from .plugin_schedule import describe

        rows = []
        for entry in sorted(self.entries.values(), key=lambda e: e.name):
            rows.append({
                "plugin": entry.name,
                "schedule": describe(entry.schedule),
                "next": (entry.next_due.strftime("%a %d %b %H:%M")
                         if entry.next_due and not entry.muted else "—"),
                "runs": entry.runs,
                "muted": entry.muted,
                "last": entry.last.as_dict() if entry.last else None,
            })
        return rows

    def _log(self, line: str) -> None:
        if self.bus is None:
            return
        try:
            self.bus.log.emit(f"[Schedule] {line}")
        except Exception:
            pass


__all__ = ["HISTORY", "MUTE_AFTER", "RUN_TIMEOUT_S", "TICK_S",
           "Entry", "PluginScheduler", "Run", "RunStatus", "scoped_plugin_secret",
           "redact_plugin_secrets", "redact_plugin_result", "plugin_secret_scope"]
