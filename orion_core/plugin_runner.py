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

A missed window is a missed run
-------------------------------
If the machine was asleep at 06:00 the 06:00 job does not fire at 09:14 when
it wakes. Cron schedules are wall-clock (see ``plugin_schedule``) and a
briefing three hours late is worse than one that did not arrive — the honest
catch-up path is ``catch_up``, which reports the gap rather than pretending.

Interval schedules are different by nature: "every thirty minutes" has no
particular minute to miss, so after a sleep the next run is simply thirty
minutes from waking.

One at a time
-------------
Runs are serialised. Several plugins coming due in the same minute execute one
after another rather than together — they share ORION's dispatcher, its
provider quota and this machine, and a scheduler that can start six things at
once is a scheduler that will.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable

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


@dataclass
class Run:
    """One execution of a scheduled plugin."""

    plugin: str
    started_at: float
    finished_at: float = 0.0
    ok: bool = False
    detail: str = ""

    @property
    def seconds(self) -> float:
        return max(0.0, (self.finished_at or time.time()) - self.started_at)

    def as_dict(self) -> dict[str, Any]:
        return {"plugin": self.plugin, "ok": self.ok,
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
                 vault: Any = None, *, clock: Callable[[], datetime] | None = None
                 ) -> None:
        self.dispatcher = dispatcher
        self.bus = bus
        self.vault = vault
        self._clock = clock or datetime.now
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
        now = self._clock()
        due = [entry for entry in self.entries.values()
               if not entry.muted and entry.next_due is not None
               and entry.next_due <= now]
        # Oldest deadline first, so a schedule that has been waiting longest
        # is not starved by one that comes due every minute.
        for entry in sorted(due, key=lambda e: e.next_due or now):
            if self._stop.is_set():
                return
            await self._execute(entry)
            entry.next_due = entry.schedule.next_due(self._clock())

    @contextmanager
    def _with_secrets(self, entry: Entry):
        """Put this plugin's declared secrets in the environment, briefly.

        Scoped to one run and restored afterwards, so a plugin cannot read
        another's key by looking at os.environ a moment later. This is only
        safe because runs are serialised — two plugins running at once would
        see each other's variables, which is the reason the supervisor
        executes one at a time rather than merely preferring to.

        A variable that already existed is restored to its old value rather
        than deleted; the process environment is not ours to tidy.
        """
        if self.vault is None or not entry.secrets:
            yield {}
            return
        try:
            secrets = self.vault.secrets_for(entry.name, entry.secrets)
        except Exception as exc:
            self._log(f"'{entry.name}' could not read its secrets - {exc}")
            yield {}
            return

        previous = {name: os.environ.get(name) for name in secrets}
        os.environ.update(secrets)
        try:
            yield secrets
        finally:
            for name, was in previous.items():
                if was is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = was

    async def _execute(self, entry: Entry) -> Run:
        """Run one plugin. Never raises."""
        run = Run(plugin=entry.name, started_at=time.time())
        entry.runs += 1
        try:
            if self.dispatcher is None:
                raise RuntimeError("no dispatcher is attached")
            with self._with_secrets(entry):
                result = await asyncio.wait_for(
                    self.dispatcher.dispatch(entry.name, dict(entry.arguments)),
                    timeout=RUN_TIMEOUT_S)
            run.ok = bool(getattr(result, "ok", True))
            run.detail = str(getattr(result, "text", result) or "")[:400]
        except asyncio.TimeoutError:
            run.ok = False
            run.detail = f"gave up after {RUN_TIMEOUT_S:.0f}s"
        except asyncio.CancelledError:
            run.ok = False
            run.detail = "cancelled at shutdown"
            run.finished_at = time.time()
            self._record(entry, run)
            raise
        except Exception as exc:
            run.ok = False
            run.detail = f"{type(exc).__name__}: {exc}"

        run.finished_at = time.time()
        self._record(entry, run)
        return run

    def _record(self, entry: Entry, run: Run) -> None:
        entry.last = run
        self.history.append(run)
        del self.history[:-HISTORY]

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
           "Entry", "PluginScheduler", "Run"]
