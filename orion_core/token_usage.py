"""
Token-usage ledger (Section 6).

Records authoritative per-request token usage for every configured model/API,
keyed by a unique request ID so retries, streaming completion, reconnects and
provider failover can never double-count.  Estimated pre-request counts are
labelled and reconciled against the provider's reported usage afterwards.

Persistence uses a dedicated SQLite database (the project's existing pattern —
one DB file per subsystem under config/), and every write is thread-safe.  Key
identifiers are stored as masked aliases only — never the key itself.  Model
context-window limits come from the versioned model_registry and are NEVER
guessed; unknown values surface as ``None`` → "Not reported by provider".
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from . import model_registry
from .db import apply_pragmas

try:
    from .constants import CONFIG_DIR
except Exception:  # pragma: no cover
    CONFIG_DIR = Path(".")

TOKEN_DB_PATH = CONFIG_DIR / "token_usage.db"

# Simple, versioned heuristic tokenizer for PRE-request estimates only.  Real
# usage always comes from the provider and reconciles over this.
ESTIMATOR_NAME = "heuristic-chars-v1"
ESTIMATOR_VERSION = "1.0"
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Rough pre-request token estimate (chars/4).  Deliberately labelled as an
    estimate everywhere it is used; reconciled away once real usage arrives."""
    if not text:
        return 0
    return max(1, round(len(str(text)) / _CHARS_PER_TOKEN))


def mask_key(api_key: str | None, alias: str | None = None) -> str:
    """Return a stable, non-reversible alias for a key — NEVER the key itself."""
    if alias:
        return str(alias)
    key = str(api_key or "").strip()
    if not key or key.lower() in {"local", "none", "no-key"}:
        return "local"
    digest = hashlib.sha256(key.encode()).hexdigest()[:6]
    return f"…{key[-4:]}#{digest}" if len(key) >= 4 else f"key#{digest}"


@dataclass
class UsageRecord:
    request_id: str
    provider: str
    model: str
    key_alias: str = "local"
    session_id: str = ""
    task: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    estimated: bool = False
    tokenizer: str = ""
    tokenizer_version: str = ""
    streaming: bool = False
    failed: bool = False
    at: float = 0.0

    def resolved_total(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        parts = [t for t in (self.input_tokens, self.output_tokens) if t is not None]
        return sum(parts) if parts else None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    request_id          TEXT PRIMARY KEY,
    at                  REAL,
    provider            TEXT,
    model               TEXT,
    key_alias           TEXT,
    session_id          TEXT,
    task                TEXT,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cached_input_tokens INTEGER,
    reasoning_tokens    INTEGER,
    total_tokens        INTEGER,
    context_window      INTEGER,
    estimated           INTEGER,
    tokenizer           TEXT,
    tokenizer_version   TEXT,
    cost_usd            REAL,
    cost_source         TEXT,
    streaming           INTEGER,
    failed              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_usage_at ON usage_events(at);
CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_events(provider);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_events(session_id);
"""


class TokenUsageLedger:
    """Thread-safe, dedup-by-request-id usage store with reconciliation."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._lock = RLock()
        path = str(db_path) if db_path is not None else str(TOKEN_DB_PATH)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self.conn)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    # ── recording ─────────────────────────────────────────────────────────────

    def record(self, record: UsageRecord) -> bool:
        """Insert a usage event.  Idempotent by ``request_id`` — a duplicate
        (retry, streaming re-emit, reconnect, failover echo) is ignored, so
        counts never inflate.  Returns True if a NEW row was written."""
        at = record.at or time.time()
        window = model_registry.context_window(record.model)
        cost, cost_source = model_registry.estimated_cost_usd(
            record.model, record.input_tokens, record.output_tokens)
        with self._lock:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO usage_events (
                    request_id, at, provider, model, key_alias, session_id, task,
                    input_tokens, output_tokens, cached_input_tokens, reasoning_tokens,
                    total_tokens, context_window, estimated, tokenizer, tokenizer_version,
                    cost_usd, cost_source, streaming, failed
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.request_id, at, record.provider, record.model, record.key_alias,
                    record.session_id, record.task,
                    record.input_tokens, record.output_tokens, record.cached_input_tokens,
                    record.reasoning_tokens, record.resolved_total(), window,
                    1 if record.estimated else 0, record.tokenizer, record.tokenizer_version,
                    cost, cost_source, 1 if record.streaming else 0, 1 if record.failed else 0,
                ),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def reconcile(self, request_id: str, usage: Any) -> bool:
        """Replace an estimated record's counts with the provider's AUTHORITATIVE
        usage once the response completes.  ``usage`` is a NormalizedUsage (or any
        object exposing the same attributes).  Returns True if a row was updated."""
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cached = getattr(usage, "cached_input_tokens", None)
        reasoning = getattr(usage, "reasoning_tokens", None)
        total = getattr(usage, "total_tokens", None)
        with self._lock:
            row = self.conn.execute(
                "SELECT model FROM usage_events WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                return False
            if total is None:
                parts = [t for t in (input_tokens, output_tokens) if t is not None]
                total = sum(parts) if parts else None
            cost, cost_source = model_registry.estimated_cost_usd(
                row["model"], input_tokens, output_tokens)
            self.conn.execute(
                """UPDATE usage_events SET
                    input_tokens=?, output_tokens=?, cached_input_tokens=?,
                    reasoning_tokens=?, total_tokens=?, cost_usd=?, cost_source=?,
                    estimated=0
                   WHERE request_id=?""",
                (input_tokens, output_tokens, cached, reasoning, total,
                 cost, cost_source, request_id),
            )
            self.conn.commit()
            return True

    def mark_failed(self, request_id: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE usage_events SET failed=1 WHERE request_id=?", (request_id,))
            self.conn.commit()

    # ── aggregation ────────────────────────────────────────────────────────────

    def _where(self, filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        f = filters or {}
        for col in ("provider", "model", "session_id", "task"):
            if f.get(col):
                clauses.append(f"{col}=?")
                params.append(f[col])
        if f.get("since") is not None:
            clauses.append("at>=?")
            params.append(float(f["since"]))
        if f.get("until") is not None:
            clauses.append("at<=?")
            params.append(float(f["until"]))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def summary(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        where, params = self._where(filters)
        with self._lock:
            row = self.conn.execute(
                f"""SELECT
                        COUNT(*) AS requests,
                        SUM(CASE WHEN failed=1 THEN 1 ELSE 0 END) AS failed,
                        SUM(input_tokens) AS input_tokens,
                        SUM(output_tokens) AS output_tokens,
                        SUM(cached_input_tokens) AS cached_input_tokens,
                        SUM(reasoning_tokens) AS reasoning_tokens,
                        SUM(total_tokens) AS total_tokens,
                        SUM(cost_usd) AS cost_usd
                    FROM usage_events{where}""",
                params,
            ).fetchone()
        return {
            "requests": row["requests"] or 0,
            "failed_requests": row["failed"] or 0,
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "cached_input_tokens": row["cached_input_tokens"],
            "reasoning_tokens": row["reasoning_tokens"],
            "total_tokens": row["total_tokens"],
            "estimated_cost_usd": round(row["cost_usd"], 6) if row["cost_usd"] is not None else None,
        }

    def by_dimension(self, dimension: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if dimension not in {"provider", "model", "session_id", "task", "key_alias"}:
            raise ValueError(f"unsupported dimension: {dimension}")
        where, params = self._where(filters)
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT {dimension} AS key,
                        COUNT(*) AS requests,
                        SUM(total_tokens) AS total_tokens,
                        SUM(cost_usd) AS cost_usd
                    FROM usage_events{where}
                    GROUP BY {dimension}
                    ORDER BY total_tokens DESC""",
                params,
            ).fetchall()
        return [{"key": r["key"], "requests": r["requests"],
                 "total_tokens": r["total_tokens"],
                 "estimated_cost_usd": round(r["cost_usd"], 6) if r["cost_usd"] is not None else None}
                for r in rows]

    _BUCKETS = {
        "hour": "%Y-%m-%d %H:00",
        "day": "%Y-%m-%d",
        "week": "%Y-W%W",
        "month": "%Y-%m",
    }

    def timeseries(self, bucket: str = "day", filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        fmt = self._BUCKETS.get(bucket)
        if fmt is None:
            raise ValueError(f"unsupported bucket: {bucket}")
        where, params = self._where(filters)
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT strftime('{fmt}', at, 'unixepoch') AS bucket,
                        COUNT(*) AS requests,
                        SUM(total_tokens) AS total_tokens,
                        SUM(cost_usd) AS cost_usd
                    FROM usage_events{where}
                    GROUP BY bucket ORDER BY bucket""",
                params,
            ).fetchall()
        return [{"bucket": r["bucket"], "requests": r["requests"],
                 "total_tokens": r["total_tokens"],
                 "estimated_cost_usd": round(r["cost_usd"], 6) if r["cost_usd"] is not None else None}
                for r in rows]

    def timeseries_by(self, dimension: str, bucket: str = "hour",
                      filters: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
        """Bucketed usage split by a dimension in ONE query — the command-deck
        graph's data source ({provider: [{bucket, total_tokens, …}, …]})."""
        if dimension not in {"provider", "model", "session_id", "task", "key_alias"}:
            raise ValueError(f"unsupported dimension: {dimension}")
        fmt = self._BUCKETS.get(bucket)
        if fmt is None:
            raise ValueError(f"unsupported bucket: {bucket}")
        where, params = self._where(filters)
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT {dimension} AS key,
                        strftime('{fmt}', at, 'unixepoch') AS bucket,
                        COUNT(*) AS requests,
                        SUM(total_tokens) AS total_tokens,
                        SUM(cost_usd) AS cost_usd
                    FROM usage_events{where}
                    GROUP BY {dimension}, bucket ORDER BY bucket""",
                params,
            ).fetchall()
        series: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            series.setdefault(str(r["key"] or "unknown"), []).append({
                "bucket": r["bucket"], "requests": r["requests"],
                "total_tokens": r["total_tokens"],
                "estimated_cost_usd": round(r["cost_usd"], 6) if r["cost_usd"] is not None else None,
            })
        return series

    def context_report(self, model: str, tokens_in_context: int | None) -> dict[str, Any]:
        """Per-request context-window picture for a model, clearly distinguishing
        the context limit from quota/rate-limit/billing (which are NOT this)."""
        window = model_registry.context_window(model)
        return {
            "model": model,
            "context_window": window,                       # None → Not reported
            "tokens_in_context": tokens_in_context,
            "remaining_context": model_registry.remaining_context(model, tokens_in_context),
            "context_utilisation_pct": model_registry.context_utilisation(model, tokens_in_context),
            "registry_version": model_registry.REGISTRY_VERSION,
            "note": ("context_window is a per-request input limit — not account "
                     "quota, rate limit, or billing credit."),
        }

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass
