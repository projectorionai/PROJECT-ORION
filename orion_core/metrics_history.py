"""
Rolling metrics history (Phase — July 2026 improvement pass, Priority 3.1).

Telemetry, ``resource_monitor`` and ``system_guard`` are real-time only: the
Command Centre shows the current second and nothing else. This module adds a
lightweight, local-first SQLite store so *trends* over days and weeks become
visible — audio latency creeping up, ORION's own memory growth, provider
cooldown frequency, and which tools actually get used.

Two tables:

    samples(ts, metric, value)   — downsampled time-series for numeric trends.
    tool_usage(tool, calls, ...) — cumulative per-tool invocation counts that
                                   survive restarts, for the usage audit
                                   (Priority 3.4).

Design rules, consistent with the rest of the local store layer
(``geo_cache.db``, ``news_cache.db``, ``token_usage.db``):

* Sampling is throttled to at most one row-set per ``interval_s`` (default
  60 s), so a long-running node accretes ~1 point/minute, not per frame.
* Old rows auto-prune beyond ``retention_days`` on each sample, so the file
  stays small without a maintenance job.
* Never raises into a caller — history must never break a live turn; every
  public method swallows storage faults and degrades to a no-op.
"""

from __future__ import annotations

from .db import apply_pragmas

import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Optional


class MetricsHistory:
    """Throttled, self-pruning on-disk history of key telemetry series."""

    def __init__(
        self,
        db_path: Path,
        *,
        interval_s: float = 60.0,
        retention_days: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.interval_s = max(1.0, float(interval_s))
        self.retention_s = max(3600.0, float(retention_days) * 86400.0)
        self._clock = clock or time.time
        self._lock = RLock()
        self._last_sample_at = 0.0
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self.conn)
        self.conn.row_factory = sqlite3.Row
        self._initialise()

    def _initialise(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS samples (
                    ts     REAL NOT NULL,
                    metric TEXT NOT NULL,
                    value  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_samples_metric_ts
                    ON samples(metric, ts);
                CREATE TABLE IF NOT EXISTS tool_usage (
                    tool     TEXT PRIMARY KEY,
                    calls    INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    last_at  REAL NOT NULL DEFAULT 0
                );
                """
            )
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass

    # ── writing ────────────────────────────────────────────────────────────────

    def record(self, series: dict[str, float], ts: float | None = None) -> bool:
        """Write one time-series row-set immediately (bypasses the throttle)."""
        if not series:
            return False
        stamp = float(ts if ts is not None else self._clock())
        rows = [(stamp, str(k), float(v)) for k, v in series.items()
                if isinstance(v, (int, float))]
        if not rows:
            return False
        try:
            with self._lock:
                self.conn.executemany(
                    "INSERT INTO samples(ts, metric, value) VALUES (?, ?, ?)", rows)
                self.conn.commit()
            return True
        except Exception:
            return False

    def sample(self, series: dict[str, float], ts: float | None = None) -> bool:
        """Throttled record: writes at most once per ``interval_s``.  Returns
        True when a row-set was actually written."""
        now = float(ts if ts is not None else self._clock())
        with self._lock:
            if now - self._last_sample_at < self.interval_s:
                return False
            self._last_sample_at = now
        written = self.record(series, ts=now)
        if written:
            self.prune(now=now)
        return written

    def record_tool_usage(self, usage: dict[str, dict[str, Any]],
                          ts: float | None = None) -> None:
        """Upsert cumulative per-tool counts.  ``usage`` maps tool → {calls,
        failures}; values are absolute cumulative totals (last-writer-wins)."""
        if not usage:
            return
        stamp = float(ts if ts is not None else self._clock())
        try:
            with self._lock:
                for tool, stats in usage.items():
                    self.conn.execute(
                        """
                        INSERT INTO tool_usage(tool, calls, failures, last_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(tool) DO UPDATE SET
                            calls=excluded.calls,
                            failures=excluded.failures,
                            last_at=excluded.last_at
                        """,
                        (str(tool), int(stats.get("calls", 0)),
                         int(stats.get("failures", 0)), stamp),
                    )
                self.conn.commit()
        except Exception:
            pass

    def prune(self, now: float | None = None) -> int:
        """Delete samples older than the retention window; returns rows removed."""
        cutoff = float(now if now is not None else self._clock()) - self.retention_s
        try:
            with self._lock:
                cur = self.conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
                self.conn.commit()
                return cur.rowcount or 0
        except Exception:
            return 0

    # ── reading ────────────────────────────────────────────────────────────────

    def metrics(self) -> list[str]:
        try:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT DISTINCT metric FROM samples ORDER BY metric").fetchall()
            return [r["metric"] for r in rows]
        except Exception:
            return []

    def series(self, metric: str, limit: int = 500,
               since_ts: float | None = None) -> list[tuple[float, float]]:
        """Chronological (ts, value) points for one metric (most recent ``limit``)."""
        try:
            with self._lock:
                if since_ts is not None:
                    rows = self.conn.execute(
                        "SELECT ts, value FROM samples WHERE metric = ? AND ts >= ? "
                        "ORDER BY ts DESC LIMIT ?",
                        (metric, float(since_ts), int(limit)),
                    ).fetchall()
                else:
                    rows = self.conn.execute(
                        "SELECT ts, value FROM samples WHERE metric = ? "
                        "ORDER BY ts DESC LIMIT ?",
                        (metric, int(limit)),
                    ).fetchall()
            return [(r["ts"], r["value"]) for r in reversed(rows)]
        except Exception:
            return []

    def trend(self, metric: str, window_s: float | None = None) -> Optional[dict[str, float]]:
        """Summary of a metric over the last ``window_s`` seconds (default all):
        first, last, min, max, mean, delta and sample count."""
        since = (self._clock() - window_s) if window_s else None
        points = self.series(metric, limit=100000, since_ts=since)
        if not points:
            return None
        values = [v for _ts, v in points]
        first, last = values[0], values[-1]
        return {
            "metric": metric,
            "count": float(len(values)),
            "first": round(first, 3),
            "last": round(last, 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "mean": round(sum(values) / len(values), 3),
            "delta": round(last - first, 3),
        }

    def tool_usage(self, top: int | None = None) -> list[dict[str, Any]]:
        """Cumulative per-tool usage, busiest first — the feature-audit view."""
        try:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT tool, calls, failures, last_at FROM tool_usage "
                    "ORDER BY calls DESC").fetchall()
            out = [dict(r) for r in rows]
            return out[:top] if top else out
        except Exception:
            return []

    def summary(self) -> dict[str, Any]:
        """Compact overview for the Command Centre / a status tool."""
        try:
            with self._lock:
                total = self.conn.execute(
                    "SELECT COUNT(*) AS n FROM samples").fetchone()
                span = self.conn.execute(
                    "SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM samples").fetchone()
            lo, hi = (span["lo"], span["hi"]) if span else (None, None)
            hours = round((hi - lo) / 3600.0, 1) if lo and hi else 0.0
            return {
                "samples": int(total["n"]) if total else 0,
                "span_hours": hours,
                "metrics": len(self.metrics()),
                "tools_tracked": len(self.tool_usage()),
            }
        except Exception:
            return {"samples": 0, "span_hours": 0.0, "metrics": 0, "tools_tracked": 0}
