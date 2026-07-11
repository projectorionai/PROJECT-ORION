from __future__ import annotations

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  O.R.I.O.N.  Mark VII  —  Open Resolution Intelligence Overt Network        ║
# ║  Single-file executable  |  python orion.py  |  Windows / Linux            ║
# ║                                                                              ║
# ║  Architecture: QStackedWidget multi-view shell  +  qasync loop unification ║
# ║  HUD:          Liquid Vector Orb  (no skull geometry)                       ║
# ║  I/O:          asyncio.to_thread() for all synchronous file operations      ║
# ║  Typing:       Pylance-clean LiveConnectConfig (no empty-dict fields)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import ast
import asyncio
import csv
import html
import io
import json
import math
import mimetypes
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import webbrowser
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
from typing import Any, Callable, Optional
from urllib.parse import quote_plus, urlparse


try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout
    import mss
    import psutil
    import qasync
    import sounddevice as sd
    from google import genai
    from google.genai import types
    from PIL import Image, ImageStat
    from PyQt6.QtCore import (
        QPointF,
        QRectF,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from PyQt6.QtGui import (
        QAction,
        QColor,
        QFont,
        QGuiApplication,
        QKeySequence,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QRadialGradient,
        QTextCursor,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QCalendarWidget,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception as import_error:
    print("O.R.I.O.N. cannot initialise. Missing runtime dependency:")
    print(f"  {import_error}")
    print(
        "Install: pip install PyQt6 qasync aiohttp sounddevice "
        "google-genai pillow mss psutil"
    )
    raise SystemExit(1)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

APP_NAME            = "O.R.I.O.N. Mark VII"
LIVE_MODEL          = "models/gemini-2.5-flash-native-audio-preview-12-2025"
LIVE_MODEL_FALLBACKS = (
    "models/gemini-2.5-flash-native-audio-preview-12-2025",
    "models/gemini-2.5-flash-preview-native-audio-dialog",
    "models/gemini-2.0-flash-live-001",
)
SEND_SAMPLE_RATE    = 16_000
RECEIVE_SAMPLE_RATE = 24_000
CHANNELS            = 1
CHUNK_SIZE          = 512
VAD_SAMPLE_LIMIT    = 256
MIC_QUEUE_LIMIT     = 150

WAKE_WORDS          = ("orion", "o'rion", "oh rye on", "oh ryan", "orien", "o rion")
WAKE_WINDOW_SECONDS = 45.0
BARGE_IN_CONFIDENCE = 0.80
VOICE_HANGOVER_SECONDS = 1.5

STARTUP_GREETINGS = (
    "Good {period}, sir. All systems are online and at your disposal.",
    "Good {period}, sir. ORION is fully operational — a pleasure to be back.",
    "Welcome back, sir. Diagnostics read green across the board.",
    "At your service, sir. All systems nominal and standing by.",
    "Good {period}, sir. The network is awake and awaiting your command.",
    "Systems restored, sir. Shall we get to work?",
    "Good {period}, sir. All channels secure; running at full capacity.",
    "Back online, sir. Everything is precisely where you left it.",
    "Good {period}, sir. Power at one hundred percent and holding steady.",
)

BASE_DIR        = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
CONFIG_DIR      = BASE_DIR / "config"
API_CONFIG_PATH = CONFIG_DIR / "api_keys.json"
CORE_DB_PATH    = CONFIG_DIR / "orion_core.db"
CORE_SCRIPT_PATH = Path(__file__).resolve()

# Colour palette (British English identifiers throughout)
class C:
    PRI     = "#ff1a3c"   # crimson primary
    PRI_DIM = "#991024"   # crimson dim
    BG      = "#050508"   # deep void
    PANEL   = "#0f0f14"   # panel surface
    BORDER  = "#2a1118"   # subtle border
    WHITE   = "#ffffff"
    MUTED   = "#a9a9b2"


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def clamp_channel(value: Any) -> int:
    try:
        return max(0, min(255, int(value)))
    except Exception:
        return 0


def now_stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_transcript(text: str) -> str:
    text = re.sub(r"<ctrl\d+>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()


_PIL_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]


def aiohttp_client_timeout() -> ClientTimeout:
    return ClientTimeout(total=8.0, connect=3.0)


def weather_code_label(code: int) -> str:
    labels = {
        0: "clear",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        95: "thunderstorm",
        96: "thunderstorm with hail",
        99: "severe thunderstorm with hail",
    }
    return labels.get(int(code), f"weather code {code}")


# ──────────────────────────────────────────────────────────────────────────────
# SECURITY LAYER
# ──────────────────────────────────────────────────────────────────────────────

class SecurityViolation(Exception):
    """Raised when a command payload violates the local execution policy."""


class SecuritySanitiser:
    """Regex firewall for all operating system action payloads."""

    DANGEROUS_PATTERNS = (
        re.compile(r"(?i)\b(?:rm|del|erase|rmdir|rd)\b\s+(?:/s|/q|/f|-r|-rf|--recursive)"),
        re.compile(r"(?i)\bformat\b\s+[a-z]:"),
        re.compile(r"(?i)\bdiskpart\b"),
        re.compile(r"(?i)\bbcdedit\b"),
        re.compile(r"(?i)\bmkfs(?:\.[a-z0-9]+)?\b"),
        re.compile(r"(?i)\bdd\b\s+.*\bof\s*=\s*(?:/dev/|\\\\\.\\PhysicalDrive)"),
        re.compile(r"(?i)\breg\b\s+(?:delete|add|import|restore|save)\b"),
        re.compile(r"(?i)\btakeown\b"),
        re.compile(r"(?i)\bicacls\b\s+.*\b(?:grant|deny|reset|remove)\b"),
        re.compile(r"(?i)\bshutdown\b\s+/(?:s|r|g|p|h)"),
        re.compile(
            r"(?i)\bpowershell(?:\.exe)?\b.*\b(?:Remove-Item|Clear-Content|Set-ExecutionPolicy|Stop-Computer)\b"
        ),
        re.compile(r"(?i)\bcmd(?:\.exe)?\b\s*/c\s*(?:del|erase|rd|rmdir|format)\b"),
        re.compile(r"(?i)\bwmic\b\s+.*\bdelete\b"),
        re.compile(r"(?i)>\s*\\\\\.\\PhysicalDrive\d+"),
        re.compile(r"(?i)\b(?:attrib|compact|cipher)\b\s+.*\b(?:/s|/w)\b"),
    )
    CORE_MUTATION_RE = re.compile(
        r"(?i)\b(?:del|erase|rm|move|ren|rename|copy|write|append|truncate|overwrite|remove|delete|replace)\b.*\borion\.py\b"
        r"|\borion\.py\b.*\b(?:del|erase|rm|move|ren|rename|copy|write|append|truncate|overwrite|remove|delete|replace)\b"
    )

    @classmethod
    def guard_text(cls, text: str, context: str = "payload") -> str:
        if not isinstance(text, str):
            return text
        candidate = text.strip()
        if not candidate:
            return text
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(candidate):
                raise SecurityViolation(
                    f"blocked unsafe {context}: destructive shell pattern detected"
                )
        if cls.CORE_MUTATION_RE.search(candidate):
            raise SecurityViolation(
                f"blocked unsafe {context}: core script mutation attempt detected"
            )
        cls._guard_python_ast(candidate, context)
        return text

    @classmethod
    def _guard_python_ast(cls, candidate: str, context: str) -> None:
        if not candidate or len(candidate) > 12000:
            return
        try:
            tree = ast.parse(candidate, mode="exec")
        except SyntaxError:
            return
        destructive_calls = {
            "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
            "shutil.rmtree", "pathlib.Path.unlink", "pathlib.Path.rmdir",
            "subprocess.Popen", "subprocess.run", "subprocess.call",
            "subprocess.check_call", "subprocess.check_output",
            "os.system", "os.popen",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = cls._ast_call_name(node.func)
                if name in destructive_calls:
                    if name.startswith("subprocess."):
                        joined = " ".join(
                            lit.value for lit in ast.walk(node)
                            if isinstance(lit, ast.Constant) and isinstance(lit.value, str)
                        )
                        if any(p.search(joined) for p in cls.DANGEROUS_PATTERNS) or cls.CORE_MUTATION_RE.search(joined):
                            raise SecurityViolation(
                                f"blocked unsafe {context}: destructive subprocess payload detected"
                            )
                        if any(
                            kw.arg == "shell"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                            for kw in node.keywords
                        ):
                            raise SecurityViolation(
                                f"blocked unsafe {context}: shell-enabled subprocess call detected"
                            )
                    else:
                        raise SecurityViolation(
                            f"blocked unsafe {context}: destructive Python call detected"
                        )
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = node.value.strip()
                if cls.CORE_MUTATION_RE.search(literal) or (
                    re.search(r"(?i)\borion\.py\b", literal)
                    and re.search(r"(?i)\b(?:write|delete|remove|unlink|rename|replace|truncate)\b", candidate)
                ):
                    raise SecurityViolation(
                        f"blocked unsafe {context}: core script mutation attempt detected"
                    )

    @classmethod
    def _ast_call_name(cls, node: ast.AST) -> str:
        parts: list[str] = []
        current: ast.AST | None = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    @classmethod
    def guard_payload(cls, payload: Any, context: str = "payload") -> Any:
        if isinstance(payload, str):
            return cls.guard_text(payload, context)
        if isinstance(payload, dict):
            return {
                cls.guard_text(str(k), context): cls.guard_payload(v, f"{context}.{k}")
                for k, v in payload.items()
            }
        if isinstance(payload, list):
            return [cls.guard_payload(v, f"{context}[{i}]") for i, v in enumerate(payload)]
        if isinstance(payload, tuple):
            return tuple(cls.guard_payload(v, f"{context}[{i}]") for i, v in enumerate(payload))
        return payload


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL BUS
# ──────────────────────────────────────────────────────────────────────────────

class OrionBus(QWidget):
    log             = pyqtSignal(str)
    state           = pyqtSignal(str)
    amplitude       = pyqtSignal(float)
    banner          = pyqtSignal(str, int)
    mic_enabled     = pyqtSignal(bool)
    request_shutdown = pyqtSignal()


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    text: str
    ok: bool = True
    media: dict[str, Any] | None = None
    chain: list[tuple[str, dict[str, Any]]] | None = None

    def response_payload(self) -> dict[str, Any]:
        return {"ok": self.ok, "result": self.text}


class _Particle:
    """Lightweight particle with __slots__ — avoids per-instance __dict__ overhead."""
    __slots__ = ("x", "y", "vx", "vy", "life", "size")

    def __init__(self, x: float, y: float, vx: float, vy: float, life: float, size: float) -> None:
        self.x    = x
        self.y    = y
        self.vx   = vx
        self.vy   = vy
        self.life = life
        self.size = size


# ──────────────────────────────────────────────────────────────────────────────
# MEMORY MATRIX  (SQLite FTS5)
# ──────────────────────────────────────────────────────────────────────────────

class OrionMemoryMatrix:
    _FTS5_STRIP_RE = re.compile(
        r'["\'\[\](){}*?!^~\\]'
        r'|(?<!\w)AND(?!\w)'
        r'|(?<!\w)OR(?!\w)'
        r'|(?<!\w)NOT(?!\w)'
        r'|(?<!\w)NEAR(?!\w)',
        re.IGNORECASE,
    )
    _FTS5_COLLAPSE_RE = re.compile(r'\s{2,}')
    _FTS5_COLSPEC_RE  = re.compile(r'\b\w+\s*:')

    def __init__(self, db_path: Path, config_dir: Path, bus: OrionBus) -> None:
        self.db_path    = db_path
        self.config_dir = config_dir
        self.bus        = bus
        self._lock      = RLock()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialise()
        self._migrate_legacy()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _initialise(self) -> None:
        schema = """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS intelligence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL DEFAULT 'notes',
            key_ref TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_intelligence_key
            ON intelligence(category, key_ref);
        CREATE VIRTUAL TABLE IF NOT EXISTS intelligence_fts USING fts5(
            category, key_ref, value, updated_at,
            content='intelligence', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS intelligence_ai AFTER INSERT ON intelligence BEGIN
            INSERT INTO intelligence_fts(rowid, category, key_ref, value, updated_at)
            VALUES (new.id, new.category, new.key_ref, new.value, new.updated_at);
        END;
        CREATE TRIGGER IF NOT EXISTS intelligence_ad AFTER DELETE ON intelligence BEGIN
            INSERT INTO intelligence_fts(intelligence_fts, rowid, category, key_ref, value, updated_at)
            VALUES('delete', old.id, old.category, old.key_ref, old.value, old.updated_at);
        END;
        CREATE TRIGGER IF NOT EXISTS intelligence_au AFTER UPDATE ON intelligence BEGIN
            INSERT INTO intelligence_fts(intelligence_fts, rowid, category, key_ref, value, updated_at)
            VALUES('delete', old.id, old.category, old.key_ref, old.value, old.updated_at);
            INSERT INTO intelligence_fts(rowid, category, key_ref, value, updated_at)
            VALUES (new.id, new.category, new.key_ref, new.value, new.updated_at);
        END;
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
            role, content, created_at, content='episodes', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
            INSERT INTO episodes_fts(rowid, role, content, created_at)
            VALUES (new.id, new.role, new.content, new.created_at);
        END;
        """
        with self._lock:
            self.conn.executescript(schema)
            self.conn.commit()

    def _migrate_legacy(self) -> None:
        candidates = [
            self.config_dir / "orion_memory.json",
            BASE_DIR / "orion_memory.json",
            BASE_DIR / "memory" / "orion_memory.json",
        ]
        for legacy_path in candidates:
            if not legacy_path.exists():
                continue
            try:
                data = json.loads(legacy_path.read_text(encoding="utf-8"))
                migrated = 0
                for category, key, value, updated in self._flatten_legacy(data):
                    self.save(category, key, value, updated_at=updated, silent=True)
                    migrated += 1
                archive = legacy_path.with_name(
                    f"{legacy_path.stem}.archived-"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{legacy_path.suffix}"
                )
                legacy_path.rename(archive)
                self.bus.log.emit(
                    f"MEM: Migrated {migrated} legacy records into SQLite and archived source."
                )
            except Exception as exc:
                self.bus.log.emit(f"MEM: Legacy migration skipped - {exc}")

    def _flatten_legacy(self, data: Any) -> list[tuple[str, str, str, str]]:
        rows: list[tuple[str, str, str, str]] = []
        if not isinstance(data, dict):
            return rows
        for category, entries in data.items():
            safe_category = self._safe_slug(str(category or "notes"))
            if isinstance(entries, dict):
                for key, value in entries.items():
                    updated = utc_stamp()
                    if isinstance(value, dict):
                        raw = value.get("value", "")
                        updated = str(value.get("updated") or value.get("updated_at") or updated)
                    else:
                        raw = value
                    if raw is not None and str(raw).strip():
                        rows.append((safe_category, self._safe_slug(str(key)), str(raw), updated))
            elif entries is not None and str(entries).strip():
                rows.append(("notes", safe_category, str(entries), utc_stamp()))
        return rows

    def _safe_slug(self, value: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
        return value[:80] or "entry"

    @classmethod
    def _sanitise_fts_query(cls, raw: str) -> str:
        sanitised = cls._FTS5_STRIP_RE.sub(' ', raw)
        sanitised = cls._FTS5_COLSPEC_RE.sub(' ', sanitised)
        sanitised = cls._FTS5_COLLAPSE_RE.sub(' ', sanitised).strip()
        tokens = [t for t in sanitised.split() if len(t) >= 2 and not re.fullmatch(r'[^\w]+', t)]
        return ' '.join(tokens)

    def save(
        self,
        category: str,
        key: str,
        value: str,
        updated_at: str | None = None,
        silent: bool = False,
    ) -> str:
        category  = self._safe_slug(category or "notes")
        key       = self._safe_slug(key or "entry")
        value     = str(value or "").strip()
        if not value:
            return "No intelligence value supplied."
        SecuritySanitiser.guard_text(value, "memory.value")
        timestamp = updated_at or utc_stamp()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO intelligence(category, key_ref, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category, key_ref)
                DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (category, key, value[:1000], timestamp),
            )
            self.conn.commit()
        if not silent:
            self.bus.log.emit(f"MEM: synchronised {category}/{key}")
        return f"Stored intelligence: {category}/{key}."

    def query(self, query: str, limit: int = 8) -> list[dict[str, str]]:
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.query")
        limit = max(1, min(25, int(limit or 8)))
        if not query:
            return []
        sanitised_fts = self._sanitise_fts_query(query)
        with self._lock:
            if sanitised_fts:
                try:
                    rows = self.conn.execute(
                        """
                        SELECT category, key_ref, value, updated_at
                        FROM intelligence_fts
                        WHERE intelligence_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (sanitised_fts, limit),
                    ).fetchall()
                    if rows:
                        return [dict(row) for row in rows]
                except (sqlite3.OperationalError, sqlite3.Error):
                    pass
            like = f"%{query}%"
            rows = self.conn.execute(
                """
                SELECT category, key_ref, value, updated_at
                FROM intelligence
                WHERE value LIKE ? OR key_ref LIKE ? OR category LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (like, like, like, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def records(self, query: str = "", limit: int = 100) -> list[dict[str, str]]:
        limit = max(1, min(500, int(limit or 100)))
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.records")
        if query:
            return self.query(query, limit=limit)
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT category, key_ref, value, updated_at
                FROM intelligence
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_episode(self, role: str, content: str) -> None:
        """Episodic conversation memory — timeline recall of what was said and when."""
        content = str(content or "").strip()
        if not content:
            return
        try:
            with self._lock:
                self.conn.execute(
                    "INSERT INTO episodes(role, content, created_at) VALUES (?, ?, ?)",
                    (str(role or "user")[:24], content[:2000], utc_stamp()),
                )
                self.conn.commit()
        except Exception:
            pass  # conversation logging must never break a live turn

    def recall_episodes(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        query = SecuritySanitiser.guard_text(str(query or "").strip(), "memory.episodes")
        limit = max(1, min(50, int(limit or 10)))
        with self._lock:
            if query:
                sanitised = self._sanitise_fts_query(query)
                if sanitised:
                    try:
                        rows = self.conn.execute(
                            """
                            SELECT role, content, created_at FROM episodes_fts
                            WHERE episodes_fts MATCH ? ORDER BY rank LIMIT ?
                            """,
                            (sanitised, limit),
                        ).fetchall()
                        if rows:
                            return [dict(row) for row in rows]
                    except (sqlite3.OperationalError, sqlite3.Error):
                        pass
                rows = self.conn.execute(
                    """
                    SELECT role, content, created_at FROM episodes
                    WHERE content LIKE ? ORDER BY id DESC LIMIT ?
                    """,
                    (f"%{query}%", limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT role, content, created_at FROM episodes ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def prompt_context(self, limit: int = 18) -> str:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT category, key_ref, value, updated_at
                FROM intelligence
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        if not rows:
            return ""
        lines = ["[LOCAL INTELLIGENCE MATRIX - use naturally, never recite]"]
        for row in rows:
            label = f"{row['category']}/{row['key_ref']}".replace("_", " ")
            lines.append(f"- {label}: {row['value']}")
        return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# SCREEN GRABBER
# ──────────────────────────────────────────────────────────────────────────────

class VolatileScreenGrabber:
    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus

    def capture_jpeg(self, max_side: int = 1024, quality: int = 78) -> bytes:
        with mss.mss() as capture:
            monitor = capture.monitors[1] if len(capture.monitors) > 1 else capture.monitors[0]
            raw     = capture.grab(monitor)
            image   = Image.frombytes("RGB", raw.size, raw.rgb)
        image.thumbnail((max_side, max_side), _PIL_RESAMPLE)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=max(45, min(95, int(quality))), optimize=True)
        return buffer.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL FILE INTELLIGENCE  — all sync I/O wrapped in asyncio.to_thread()
# ──────────────────────────────────────────────────────────────────────────────

class LocalFileIntelligence:
    """
    File inspection pipeline.  Every heavy synchronous branch (PIL decoding,
    CSV sniffing, text tokenisation) is offloaded via asyncio.to_thread() so
    the qasync GUI event loop is never blocked by disk or CPU work.
    """

    TEXT_SUFFIXES = {
        ".txt", ".md", ".rst", ".py", ".js", ".ts", ".html", ".css",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log",
        ".csv", ".tsv", ".xml", ".sql",
    }
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus

    # ── async entry-point ────────────────────────────────────────────────────

    async def inspect_async(self, path: Path, prompt: str = "") -> ToolResult:
        """
        Non-blocking inspection.  Security guard runs synchronously (negligible
        cost); all heavy work is deferred to a thread pool via to_thread().
        """
        SecuritySanitiser.guard_text(str(path), "file_processor.path")
        resolved = path.expanduser().resolve()
        if resolved == CORE_SCRIPT_PATH:
            raise SecurityViolation("blocked unsafe file operation: core script protected")
        if not resolved.exists() or not resolved.is_file():
            return ToolResult(f"File not found: {resolved}", ok=False)

        suffix = resolved.suffix.lower()

        # Route and offload every synchronous branch to a thread-pool worker.
        if suffix in self.IMAGE_SUFFIXES:
            return await asyncio.to_thread(self._inspect_image, resolved, prompt)
        if suffix == ".pdf":
            return await asyncio.to_thread(self._inspect_pdf, resolved)
        if suffix == ".json":
            return await asyncio.to_thread(self._inspect_json, resolved)
        if suffix in {".csv", ".tsv"}:
            # CSV sniffing (csv.Sniffer) is synchronous I/O — offload.
            return await asyncio.to_thread(self._inspect_table, resolved)
        is_text = await asyncio.to_thread(self._looks_textual, resolved)
        if suffix in self.TEXT_SUFFIXES or is_text:
            return await asyncio.to_thread(self._inspect_text, resolved)
        return await asyncio.to_thread(self._inspect_binary, resolved)

    # ── legacy synchronous wrapper (retained for non-async call sites) ───────

    def inspect(self, path: Path, prompt: str = "") -> ToolResult:
        """Synchronous fallback. Do NOT invoke from the GUI thread during a live session."""
        SecuritySanitiser.guard_text(str(path), "file_processor.path")
        resolved = path.expanduser().resolve()
        if resolved == CORE_SCRIPT_PATH:
            raise SecurityViolation("blocked unsafe file operation: core script protected")
        if not resolved.exists() or not resolved.is_file():
            return ToolResult(f"File not found: {resolved}", ok=False)
        suffix = resolved.suffix.lower()
        if suffix in self.IMAGE_SUFFIXES:
            return self._inspect_image(resolved, prompt)
        if suffix == ".pdf":
            return self._inspect_pdf(resolved)
        if suffix == ".json":
            return self._inspect_json(resolved)
        if suffix in {".csv", ".tsv"}:
            return self._inspect_table(resolved)
        if suffix in self.TEXT_SUFFIXES or self._looks_textual(resolved):
            return self._inspect_text(resolved)
        return self._inspect_binary(resolved)

    # ── private synchronous workers (called inside to_thread) ────────────────

    def _inspect_image(self, path: Path, prompt: str) -> ToolResult:
        """PIL image decode — synchronous; invoked via asyncio.to_thread()."""
        with Image.open(path) as image:
            original_format = image.format or path.suffix.replace(".", "").upper() or "IMAGE"
            width, height   = image.size
            mode            = image.mode
            rgb             = image.convert("RGB")
            stat_source     = rgb.copy()
            stat_source.thumbnail((512, 512), _PIL_RESAMPLE)
            stat      = ImageStat.Stat(stat_source)
            mean      = tuple(int(v) for v in stat.mean[:3])
            extrema   = stat.extrema[:3]
            brightness = sum(stat.mean[:3]) / (3 * 255)
            entropy   = rgb.entropy()
            aspect    = width / max(1, height)
            orientation = (
                "landscape" if aspect > 1.08 else
                "portrait"  if aspect < 0.92 else
                "square"
            )
            exif_count = 0
            try:
                exif_count = len(image.getexif() or {})
            except Exception:
                exif_count = 0
            outbound = rgb.copy()
            outbound.thumbnail((1024, 1024), _PIL_RESAMPLE)
            buffer = io.BytesIO()
            outbound.save(buffer, format="JPEG", quality=82, optimize=True)
        size_kb     = path.stat().st_size / 1024
        prompt_line = (
            f"Requested focus: {prompt.strip()}"
            if prompt.strip()
            else "Requested focus: general visual inspection."
        )
        report = (
            f"Image scan complete: {path.name}\n"
            f"Format: {original_format}; dimensions: {width}x{height}; "
            f"orientation: {orientation}; mode: {mode}; size: {size_kb:.1f} KB.\n"
            f"Mean RGB: {mean}; channel ranges: {extrema}; "
            f"brightness index: {brightness:.2f}; entropy: {entropy:.2f}; "
            f"EXIF entries: {exif_count}.\n"
            f"{prompt_line}\n"
            "A volatile JPEG review frame is ready for the live multimodal channel."
        )
        return ToolResult(report, media={"data": buffer.getvalue(), "mime_type": "image/jpeg"})

    def _inspect_json(self, path: Path) -> ToolResult:
        raw  = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        if isinstance(data, dict):
            keys  = list(data.keys())[:30]
            shape = f"object with {len(data)} keys: {', '.join(map(str, keys))}"
        elif isinstance(data, list):
            shape = f"array with {len(data)} items"
            if data and isinstance(data[0], dict):
                shape += f"; first item keys: {', '.join(map(str, list(data[0].keys())[:20]))}"
        else:
            shape = type(data).__name__
        return ToolResult(
            f"JSON scan complete: {path.name}\nShape: {shape}.\n"
            f"Characters: {len(raw)}.\nExcerpt:\n{raw[:2200]}"
        )

    def _inspect_table(self, path: Path) -> ToolResult:
        """CSV/TSV sniffing — synchronous I/O; invoked via asyncio.to_thread()."""
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel_tab if delimiter == "\t" else csv.excel  # type: ignore[assignment]
            reader = csv.reader(handle, dialect)
            for index, row in enumerate(reader):
                rows.append(row)
                if index >= 40:
                    break
        header  = rows[0] if rows else []
        preview = "\n".join(
            " | ".join(cell[:80] for cell in row[:12]) for row in rows[:12]
        )
        return ToolResult(
            f"Table scan complete: {path.name}\n"
            f"Columns detected: {len(header)}; preview rows inspected: {len(rows)}.\n"
            f"Header: {', '.join(header[:20]) if header else 'none'}\nPreview:\n{preview}"
        )

    def _inspect_text(self, path: Path) -> ToolResult:
        raw   = path.read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()
        words = re.findall(r"[A-Za-z0-9_']+", raw)
        frequencies: dict[str, int] = {}
        for word in words:
            key = word.lower()
            if len(key) < 4 or key.isdigit():
                continue
            frequencies[key] = frequencies.get(key, 0) + 1
        top_terms = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)[:12]
        terms     = ", ".join(f"{term}:{count}" for term, count in top_terms) or "none"
        return ToolResult(
            f"Text scan complete: {path.name}\n"
            f"Lines: {len(lines)}; words: {len(words)}; characters: {len(raw)}.\n"
            f"Dominant terms: {terms}.\nExcerpt:\n{raw[:4200]}"
        )

    def _inspect_pdf(self, path: Path) -> ToolResult:
        """PDF text extraction via pypdf/PyPDF2 (optional dependency)."""
        reader_cls: Any = None
        try:
            from pypdf import PdfReader as reader_cls  # type: ignore
        except Exception:
            try:
                from PyPDF2 import PdfReader as reader_cls  # type: ignore
            except Exception:
                reader_cls = None
        size_kb = path.stat().st_size / 1024
        if reader_cls is None:
            return ToolResult(
                f"PDF detected: {path.name} ({size_kb:.1f} KB). "
                "Install 'pypdf' (pip install pypdf) to enable text extraction, "
                "summarisation and question answering over documents.",
                ok=False,
            )
        reader = reader_cls(str(path))
        page_count = len(reader.pages)
        chunks: list[str] = []
        for page in reader.pages[:12]:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        text  = re.sub(r"[ \t]+", " ", "\n".join(chunks)).strip()
        title = ""
        try:
            title = str((reader.metadata or {}).get("/Title") or "")
        except Exception:
            title = ""
        return ToolResult(
            f"PDF scan complete: {path.name}\n"
            f"Pages: {page_count}; size: {size_kb:.1f} KB; title: {title or 'not set'}.\n"
            f"Extracted text (first {min(page_count, 12)} pages):\n"
            f"{text[:5000] or 'No extractable text — likely a scanned document; convert pages to images for visual analysis.'}"
        )

    def _inspect_binary(self, path: Path) -> ToolResult:
        mime, _ = mimetypes.guess_type(str(path))
        size_kb  = path.stat().st_size / 1024
        with path.open("rb") as handle:
            header = handle.read(32).hex(" ")
        return ToolResult(
            f"Binary file scan complete: {path.name}\n"
            f"MIME: {mime or 'unknown'}; size: {size_kb:.1f} KB; header bytes: {header}.\n"
            "Deep semantic extraction is available for text, JSON, CSV, TSV, "
            "and common image formats."
        )

    def _looks_textual(self, path: Path) -> bool:
        try:
            chunk = path.read_bytes()[:2048]
        except Exception:
            return False
        if not chunk:
            return True
        if b"\x00" in chunk:
            return False
        printable = sum(1 for byte in chunk if byte in b"\r\n\t" or 32 <= byte < 127)
        return (printable / len(chunk)) > 0.82


# ──────────────────────────────────────────────────────────────────────────────
# AUDIO SUBSYSTEM
# ──────────────────────────────────────────────────────────────────────────────

class AudioPlaybackThread(Thread):
    def __init__(self, bus: OrionBus) -> None:
        super().__init__(name="orion-audio-renderer", daemon=True)
        self.bus   = bus
        # Unbounded: the model streams audio faster than realtime, so a bounded
        # queue overflows on long turns and dropped chunks time-compress the
        # speech (heard as ORION suddenly talking far too fast).
        self.queue: Queue[bytes | None] = Queue(maxsize=0)
        self.stop_event     = Event()
        self.last_audio_time = 0.0
        self._stream: Any   = None

    def enqueue(self, chunk: bytes) -> None:
        if not chunk:
            return
        # Never drop model audio: every discarded chunk skips playback forward.
        self.queue.put_nowait(bytes(chunk))

    def run(self) -> None:
        try:
            self._stream = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
            )
            self._stream.start()
            self.bus.log.emit("AUDIO: output renderer initialised.")
            while not self.stop_event.is_set():
                try:
                    chunk = self.queue.get(timeout=0.1)
                except Empty:
                    continue
                if chunk is None:
                    break
                self.last_audio_time = time.monotonic()
                self.bus.amplitude.emit(self._amplitude(chunk))
                self._stream.write(chunk)
        except Exception as exc:
            self.bus.log.emit(f"AUDIO: output renderer fault - {exc}")
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self.bus.amplitude.emit(0.0)

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except Full:
            pass

    def clear(self) -> None:
        """Flush pending playback immediately (user interruption / barge-in)."""
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.last_audio_time = 0.0
        self.bus.amplitude.emit(0.0)

    def speaking_recently(self) -> bool:
        return (time.monotonic() - self.last_audio_time) < 0.45

    def _amplitude(self, chunk: bytes) -> float:
        if len(chunk) < 2:
            return 0.0
        sample_count = min(len(chunk) // 2, 512)
        if sample_count <= 0:
            return 0.0
        total  = 0
        stride = max(1, (len(chunk) // 2) // sample_count)
        for index in range(0, sample_count * stride * 2, stride * 2):
            if index + 1 >= len(chunk):
                break
            value = int.from_bytes(chunk[index:index + 2], "little", signed=True)
            total += abs(value)
        return min(1.0, (total / sample_count) / 32768.0)


class SpeechSynthesiser(Thread):
    """
    Local text-to-speech voice — gives ORION a spoken voice even when the
    native Gemini audio channel is offline (the JARVIS-style fallback loop).
    Prefers pyttsx3 (SAPI5); falls back to Windows System.Speech via
    PowerShell, so it works with zero extra dependencies on Windows.
    """

    def __init__(self, bus: OrionBus) -> None:
        super().__init__(name="orion-local-voice", daemon=True)
        self.bus        = bus
        self.queue: Queue[str | None] = Queue()
        self.stop_event = Event()
        self.available  = True
        self.state_cb: Callable[[bool], None] | None = None
        self._engine: Any = None
        self._proc: Any   = None
        self._speaking    = Event()
        self._interrupted = Event()

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def speak(self, text: str) -> None:
        text = re.sub(r"[*_`#]+", "", str(text or "")).strip()
        if not text or not self.available:
            return
        self.queue.put_nowait(text[:1200])

    def interrupt(self) -> None:
        self._interrupted.set()
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break

    def stop(self) -> None:
        self.stop_event.set()
        self.interrupt()
        try:
            self.queue.put_nowait(None)
        except Full:
            pass

    def run(self) -> None:
        self._initialise_engine()
        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if text is None:
                break
            self._interrupted.clear()
            self._speaking.set()
            if self.state_cb is not None:
                try:
                    self.state_cb(True)
                except Exception:
                    pass
            try:
                if self._engine is not None:
                    self._speak_pyttsx3(text)
                else:
                    self._speak_powershell(text)
            except Exception as exc:
                self.bus.log.emit(f"VOICE: local speech fault - {str(exc).splitlines()[0][:100]}")
            finally:
                self._speaking.clear()
                if self.state_cb is not None:
                    try:
                        self.state_cb(False)
                    except Exception:
                        pass

    def _initialise_engine(self) -> None:
        try:
            import pyttsx3  # type: ignore
            engine = pyttsx3.init()
            engine.setProperty("rate", 174)
            try:
                voices = engine.getProperty("voices") or []
                preferred = next(
                    (v for v in voices if re.search(
                        r"(?i)en[-_ ]?(gb|uk)|george|hazel|ryan",
                        f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}",
                    )),
                    None,
                ) or next(
                    (v for v in voices if re.search(
                        r"(?i)david|male", f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}",
                    )),
                    None,
                )
                if preferred is not None:
                    engine.setProperty("voice", preferred.id)
            except Exception:
                pass

            def _on_word(name: Any = None, location: int = 0, length: int = 0) -> None:
                # Word boundaries drive the HUD orb while the local voice speaks.
                self.bus.amplitude.emit(0.35 + 0.4 * random.random())

            try:
                engine.connect("started-word", _on_word)
            except Exception:
                pass
            self._engine = engine
            self.bus.log.emit("VOICE: local speech engine ready (pyttsx3).")
        except Exception:
            self._engine = None
            if sys.platform == "win32":
                self.bus.log.emit("VOICE: pyttsx3 not detected; using the Windows System.Speech voice.")
            else:
                self.available = False
                self.bus.log.emit("VOICE: no local speech engine available; offline replies stay text-only.")

    def _speak_pyttsx3(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()
        self.bus.amplitude.emit(0.0)

    def _speak_powershell(self, text: str) -> None:
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Rate = 0; "
            "$s.Speak([Console]::In.ReadToEnd())"
        )
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self._proc.stdin.write(text.encode("utf-8", errors="replace"))
            self._proc.stdin.close()
        except Exception:
            pass
        while self._proc.poll() is None:
            if self._interrupted.is_set() or self.stop_event.is_set():
                try:
                    self._proc.kill()
                except Exception:
                    pass
                break
            self.bus.amplitude.emit(0.3 + 0.4 * random.random())
            time.sleep(0.09)
        self.bus.amplitude.emit(0.0)
        self._proc = None


class SileroVADGatekeeper:
    """Local voice-activity gate with optional Silero inference and deterministic fallback."""

    def __init__(self, bus: OrionBus, threshold: float = 0.65) -> None:
        self.bus           = bus
        self.threshold     = max(0.05, min(0.95, float(threshold)))
        self._model: Any   = None
        self._torch: Any   = None
        self._fallback_floor = 0.012
        self._last_voice   = 0.0
        self._initialise_silero()

    def accepts(self, chunk: bytes) -> bool:
        confidence = self.confidence(chunk)
        accepted   = confidence > self.threshold
        if accepted:
            self._last_voice = time.monotonic()
        return accepted

    def confidence(self, chunk: bytes) -> float:
        if not chunk:
            return 0.0
        if self._model is not None and self._torch is not None:
            try:
                pcm = array("h")
                pcm.frombytes(chunk[: min(len(chunk), VAD_SAMPLE_LIMIT * 2)])
                if sys.byteorder != "little":
                    pcm.byteswap()
                try:
                    tensor = (
                        self._torch.frombuffer(pcm, dtype=self._torch.int16)
                        .to(dtype=self._torch.float32) / 32768.0
                    )
                except Exception:
                    tensor = self._torch.tensor(pcm.tolist(), dtype=self._torch.float32) / 32768.0
                with self._torch.no_grad():
                    value = self._model(tensor, SEND_SAMPLE_RATE)
                return max(0.0, min(1.0, float(value.item() if hasattr(value, "item") else value)))
            except Exception:
                self._model = None
                self._torch = None
                self.bus.log.emit("AUDIO: packaged Silero VAD unavailable; local acoustic gate active.")
        return self._local_confidence(chunk)

    def _initialise_silero(self) -> None:
        try:
            import torch  # type: ignore
            from silero_vad import load_silero_vad  # type: ignore
            self._torch = torch
            self._model = load_silero_vad()
            if hasattr(self._model, "eval"):
                self._model.eval()
            self.bus.log.emit("AUDIO: local Silero VAD gatekeeper initialised.")
        except Exception:
            self._model = None
            self._torch = None
            self.bus.log.emit("AUDIO: packaged Silero VAD not detected; local acoustic gate active.")

    def _local_confidence(self, chunk: bytes) -> float:
        pcm    = array("h")
        usable = len(chunk) - (len(chunk) % 2)
        if usable <= 0:
            return 0.0
        pcm.frombytes(chunk[:usable])
        if sys.byteorder != "little":
            pcm.byteswap()
        if not pcm:
            return 0.0
        sample_count = len(pcm)
        stride       = max(1, sample_count // VAD_SAMPLE_LIMIT)
        total_sq     = 0.0
        total_abs    = 0.0
        peak         = 0
        crossings    = 0
        previous     = 0
        used         = 0
        for sample in pcm[::stride]:
            value    = int(sample)
            total_sq += value * value
            absolute  = abs(value)
            total_abs += absolute
            if absolute > peak:
                peak = absolute
            if used and ((value >= 0) != (previous >= 0)):
                crossings += 1
            previous = value
            used     += 1
        if used <= 0:
            return 0.0
        rms        = math.sqrt(total_sq / used) / 32768.0
        mean_abs   = (total_abs / used) / 32768.0
        peak_norm  = peak / 32768.0
        zcr        = crossings / max(1, used - 1)
        speech_band  = 1.0 - min(1.0, abs(zcr - 0.075) / 0.16)
        energy_score = max(0.0, min(1.0, (rms - self._fallback_floor) / 0.055))
        peak_score   = max(0.0, min(1.0, (peak_norm - 0.04) / 0.28))
        compactness  = max(0.0, min(1.0, mean_abs / max(0.0001, rms * 0.82)))
        confidence   = (
            energy_score * 0.52
            + speech_band * 0.24
            + peak_score  * 0.16
            + compactness * 0.08
        )
        if rms < self._fallback_floor:
            confidence *= max(0.0, rms / self._fallback_floor)
        return max(0.0, min(1.0, confidence))


class LocalSpeechRecogniser:
    """
    Offline speech recognition (Vosk) used for wake-word activation and local
    transcript logging.  Optional dependency: if Vosk or its model is absent,
    `available` stays False and the wake-word gate is disabled (mic always live).
    """

    def __init__(self, bus: OrionBus, sample_rate: int = SEND_SAMPLE_RATE) -> None:
        self.bus        = bus
        self.available  = False
        self._recogniser: Any = None
        self._lock      = RLock()
        self._sample_rate = sample_rate
        # Model loading (and a possible first-run download) can take seconds;
        # it must never block the GUI event loop.
        Thread(target=self._initialise, name="orion-vosk-loader", daemon=True).start()

    def _initialise(self) -> None:
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore
            SetLogLevel(-1)
            model_path = os.getenv("ORION_VOSK_MODEL", "").strip()
            model = Model(model_path) if model_path else Model(lang="en-us")
            with self._lock:
                self._recogniser = KaldiRecognizer(model, float(self._sample_rate))
                self.available   = True
            self.bus.log.emit("SR: local Vosk recogniser initialised; wake-word gate armed.")
        except Exception as exc:
            self.bus.log.emit(
                "SR: local recognition unavailable "
                f"({str(exc).splitlines()[0][:90]}); wake-word gate disabled."
            )

    def feed(self, chunk: bytes) -> str:
        """Feed 16 kHz PCM; returns final text, or partial text if it contains a wake word."""
        if not self.available or self._recogniser is None or not chunk:
            return ""
        try:
            with self._lock:
                if self._recogniser.AcceptWaveform(chunk):
                    result = json.loads(self._recogniser.Result() or "{}")
                    return str(result.get("text") or "").strip()
                partial_payload = json.loads(self._recogniser.PartialResult() or "{}")
                partial = str(partial_payload.get("partial") or "").strip()
                if partial and any(word in partial.lower() for word in WAKE_WORDS):
                    try:
                        self._recogniser.Reset()
                    except Exception:
                        pass
                    return partial
        except Exception:
            return ""
        return ""


class AudioGateThread(Thread):
    """
    Real-time audio gatekeeper.

    PortAudio callbacks must not perform Torch inference, VAD math, asyncio queue
    mutation, or expensive Python loops.  The callback copies bytes and exits;
    this worker performs capture gating, VAD, amplitude calculation, and qasync
    handoff away from the audio device thread.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        out_q: asyncio.Queue,
        bus: OrionBus,
        can_capture: Callable[[], bool],
        vad: SileroVADGatekeeper,
        raw_limit: int = MIC_QUEUE_LIMIT,
        recogniser: "LocalSpeechRecogniser | None" = None,
        on_transcript: Callable[[str], None] | None = None,
        speaking_check: Callable[[], bool] | None = None,
        on_barge_in: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(name="orion-audio-gatekeeper", daemon=True)
        self.loop        = loop
        self.out_q       = out_q
        self.bus         = bus
        self.can_capture = can_capture
        self.vad         = vad
        self.recogniser     = recogniser
        self.on_transcript  = on_transcript
        self.speaking_check = speaking_check
        self.on_barge_in    = on_barge_in
        self.raw_q: Queue[bytes | None] = Queue(maxsize=max(8, int(raw_limit or MIC_QUEUE_LIMIT)))
        self.stop_event  = Event()
        self._voice_until = 0.0

    def enqueue(self, chunk: bytes) -> None:
        if not chunk:
            return
        try:
            self.raw_q.put_nowait(bytes(chunk))
            return
        except Full:
            pass
        try:
            self.raw_q.get_nowait()
        except Empty:
            pass
        try:
            self.raw_q.put_nowait(bytes(chunk))
        except Full:
            pass

    def drain(self) -> None:
        while True:
            try:
                self.raw_q.get_nowait()
            except Empty:
                break

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.raw_q.put_nowait(None)
        except Full:
            pass

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.raw_q.get(timeout=0.05)
            except Empty:
                continue
            if chunk is None:
                break
            try:
                confidence = self.vad.confidence(chunk)
                now    = time.monotonic()
                voiced = confidence > self.vad.threshold
                if voiced:
                    self._voice_until = now + VOICE_HANGOVER_SECONDS
                in_speech = voiced or now < self._voice_until
                speaking  = self.speaking_check() if self.speaking_check is not None else False
                # Local recognition runs during speech (plus a silence hangover so
                # Vosk can finalise utterances) — even while the wake-word gate is
                # closed, so the wake word itself can open the channel.  While
                # ORION is speaking, only high-confidence voice is fed, otherwise
                # the recogniser would transcribe ORION's own speaker output.
                if (
                    in_speech
                    and self.recogniser is not None
                    and self.on_transcript is not None
                    and (not speaking or confidence >= BARGE_IN_CONFIDENCE)
                ):
                    transcript = self.recogniser.feed(chunk)
                    if transcript:
                        self.loop.call_soon_threadsafe(self.on_transcript, transcript)
                if not self.can_capture():
                    continue
                if speaking:
                    # Barge-in path: require high-confidence voice to reject echo.
                    if not voiced or confidence < BARGE_IN_CONFIDENCE:
                        continue
                    if self.on_barge_in is not None:
                        self.loop.call_soon_threadsafe(self.on_barge_in)
                # Continuous streaming (JARVIS-style): forward ALL audio — voice
                # AND silence — so the server-side VAD hears complete utterances
                # and can detect end-of-turn.  Filtering to voiced-only chunks is
                # what made the live channel deaf.
                media = {"data": chunk, "mime_type": "audio/pcm;rate=16000"}
                self.loop.call_soon_threadsafe(self._safe_put, media)
                if voiced:
                    self.loop.call_soon_threadsafe(self.bus.amplitude.emit, self._amplitude(chunk))
            except Exception as exc:
                try:
                    self.loop.call_soon_threadsafe(
                        self.bus.log.emit,
                        f"AUDIO: gatekeeper recovered - {str(exc).splitlines()[0][:120]}",
                    )
                except RuntimeError:
                    return  # event loop already closed during shutdown

    def _safe_put(self, media: dict[str, Any]) -> None:
        try:
            if self.out_q.full():
                try:
                    self.out_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self.out_q.put_nowait(media)
        except asyncio.QueueFull:
            try:
                self.out_q.get_nowait()
                self.out_q.put_nowait(media)
            except Exception:
                pass

    def _amplitude(self, chunk: bytes) -> float:
        usable = len(chunk) - (len(chunk) % 2)
        if usable <= 0:
            return 0.0
        pcm = array("h")
        pcm.frombytes(chunk[:usable])
        if sys.byteorder != "little":
            pcm.byteswap()
        if not pcm:
            return 0.0
        sample_count = min(len(pcm), 256)
        stride       = max(1, len(pcm) // sample_count)
        total        = 0
        used         = 0
        for sample in pcm[::stride]:
            total += abs(int(sample))
            used  += 1
            if used >= sample_count:
                break
        if used <= 0:
            return 0.0
        return min(1.0, (total / used) / 32768.0)


class MicrophoneEngine:
    """Minimal PortAudio callback plus external VAD gatekeeper."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        out_q: asyncio.Queue,
        bus: OrionBus,
        can_capture: Callable[[], bool],
        vad: SileroVADGatekeeper,
        recogniser: "LocalSpeechRecogniser | None" = None,
        on_transcript: Callable[[str], None] | None = None,
        speaking_check: Callable[[], bool] | None = None,
        on_barge_in: Callable[[], None] | None = None,
    ) -> None:
        self.loop        = loop
        self.out_q       = out_q
        self.bus         = bus
        self.can_capture = can_capture
        self.vad         = vad
        self.recogniser     = recogniser
        self.on_transcript  = on_transcript
        self.speaking_check = speaking_check
        self.on_barge_in    = on_barge_in
        self.enabled     = True
        self._stream: Any = None
        self._gatekeeper: AudioGateThread | None = None

    def start(self) -> None:
        if self._stream is not None:
            return
        self._gatekeeper = AudioGateThread(
            self.loop, self.out_q, self.bus, self._can_gate_capture, self.vad,
            recogniser=self.recogniser,
            on_transcript=self.on_transcript,
            speaking_check=self.speaking_check,
            on_barge_in=self.on_barge_in,
        )
        self._gatekeeper.start()
        self._stream = sd.RawInputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=self._callback,
        )
        self._stream.start()
        self.bus.log.emit("AUDIO: microphone pipeline initialised.")

    def stop(self) -> None:
        stream       = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                self.bus.log.emit(f"AUDIO: microphone close fault - {exc}")
        gatekeeper       = self._gatekeeper
        self._gatekeeper = None
        if gatekeeper is not None:
            gatekeeper.stop()
            gatekeeper.join(timeout=0.75)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.bus.mic_enabled.emit(self.enabled)
        if not self.enabled:
            self._drain()
            if self._gatekeeper is not None:
                self._gatekeeper.drain()

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            try:
                self.loop.call_soon_threadsafe(self.bus.log.emit, f"AUDIO: input status - {status}")
            except RuntimeError:
                return  # event loop already closed during shutdown
        if not self.enabled:
            return
        chunk = bytes(indata)
        if not chunk:
            return
        gatekeeper = self._gatekeeper
        if gatekeeper is not None:
            gatekeeper.enqueue(chunk)

    def _can_gate_capture(self) -> bool:
        return self.enabled and self.can_capture()

    def _drain(self) -> None:
        while True:
            try:
                self.out_q.get_nowait()
            except asyncio.QueueEmpty:
                break


# ──────────────────────────────────────────────────────────────────────────────
# METRIC BAR WIDGET
# ──────────────────────────────────────────────────────────────────────────────

class MetricBar(QWidget):
    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.label         = label
        self.value         = 0.0
        self.display_value = 0.0
        self.setMinimumHeight(58)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: float) -> None:
        self.value = max(0.0, min(100.0, float(value)))
        self.update()

    def paintEvent(self, event: Any) -> None:
        self.display_value += (self.value - self.display_value) * 0.18
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        painter.setPen(QPen(QColor(C.BORDER), 1))
        painter.setBrush(QColor(C.PANEL))
        painter.drawRoundedRect(rect, 6, 6)
        inner = rect.adjusted(10, 31, -10, -10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1a1116"))
        painter.drawRoundedRect(inner, 4, 4)
        fill_width = inner.width() * (self.display_value / 100.0)
        fill       = QRectF(inner.left(), inner.top(), fill_width, inner.height())
        gradient   = QLinearGradient(fill.topLeft(), fill.topRight())
        gradient.setColorAt(0.0, QColor(C.PRI_DIM))
        gradient.setColorAt(1.0, QColor(C.PRI))
        painter.setBrush(gradient)
        painter.drawRoundedRect(fill, 4, 4)
        painter.setPen(QColor(C.WHITE))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(10, 6, rect.width() - 20, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.label,
        )
        painter.setPen(QColor("#ff7a8d"))
        painter.drawText(
            QRectF(10, 6, rect.width() - 20, 20),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{self.display_value:05.1f}%",
        )


# ──────────────────────────────────────────────────────────────────────────────
# CENTRAL HUD  —  Liquid Vector Orb  (all skull geometry removed)
# ──────────────────────────────────────────────────────────────────────────────

class CentralHud(QWidget):
    """
    Mark VII HUD — Pristine layered Liquid Vector Orb.

    Rendering pipeline (two-pass):
      Pass 1 — Static background (grid + corner brackets) cached into a QPixmap.
               Re-rendered only on resize events; zero per-frame cost.
      Pass 2 — Dynamic elements painted directly each tick:
               • Orb corona (breathes via _pulse oscillator)
               • Concentric orb body  (three filled ellipses)
               • Equatorial rotating ring with tick marks
               • Radial circuit traces (flare with amplitude)
               • Hexagonal reticle (counter-rotates)
               • Amplitude pulse rings
               • Scanner sweep
               • Particles
               • Banner overlay

    No skull geometry, no mouth/eye vectors — clean radial orb only.
    """

    _PARTICLE_HARD_CAP = 120
    _PARTICLE_TRIM_TO  = 90

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.rotation         = 0.0
        self.scan             = 0.0
        self.amplitude        = 0.0
        self.target_amplitude = 0.0
        self.face_amount      = 0.0
        self.face_target      = 0.0
        self.state_name       = "INITIALISING"
        self.particles: list[_Particle] = []
        self.banner_text      = ""
        self.banner_alpha     = 0.0
        self.banner_priority  = 0
        self._pulse           = 0.0   # slow breathe oscillator (0 → 2π)

        self.setMinimumSize(300, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # QPixmap cache for static background — invalidated on resize
        self._bg_pixmap: Optional[QPixmap] = None
        self._bg_size: tuple[int, int]     = (0, 0)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ── cache invalidation on resize ────────────────────────────────────────

    def resizeEvent(self, event: Any) -> None:
        """Invalidate the static background cache whenever the widget is resized."""
        self._bg_pixmap = None
        super().resizeEvent(event)

    # ── public interface ─────────────────────────────────────────────────────

    def set_amplitude(self, value: float) -> None:
        self.target_amplitude = max(0.0, min(1.0, float(value)))

    def set_banner(self, text: str, priority: int = 1) -> None:
        self.banner_text     = str(text or "").strip()[:140]
        self.banner_priority = max(0, min(5, int(priority or 1)))
        self.banner_alpha    = 255.0 if self.banner_text else 0.0
        self.update()

    def set_state(self, state: str) -> None:
        self.state_name  = str(state or "STANDBY").upper()
        active_states    = {"CONNECTING", "LISTENING", "PROCESSING", "SPEAKING"}
        self.face_target = 1.0 if self.state_name in active_states else 0.0
        target_interval  = 16 if self.face_target or self.amplitude > 0.04 else 33
        if self.timer.interval() != target_interval:
            self.timer.setInterval(target_interval)
        self.update()

    # ── animation tick ───────────────────────────────────────────────────────

    def _tick(self) -> None:
        self.rotation   = (self.rotation + 1.2 + self.amplitude * 4.0) % 360.0
        self.scan       = (self.scan + 2.5) % 360.0
        self.amplitude  += (self.target_amplitude - self.amplitude) * 0.22
        self.face_amount += (self.face_target - self.face_amount) * 0.12
        # Slow breathe oscillator — drives the orb corona scale
        self._pulse     = (self._pulse + 0.04) % (2 * math.pi)
        if self.amplitude > 0.08:
            self._spawn_particles()
        self._update_particles()
        if self.banner_alpha > 0:
            self.banner_alpha = max(0.0, self.banner_alpha - 1.4)
        self.update()

    # ── particle system ──────────────────────────────────────────────────────

    def _spawn_particles(self) -> None:
        count    = min(6, 1 + int(self.amplitude * 8))
        centre_x = self.width()  * 0.5
        centre_y = self.height() * 0.5
        for n in range(count):
            angle = math.radians((self.rotation * 2 + n * 51) % 360)
            speed = 0.8 + self.amplitude * 4.0
            self.particles.append(_Particle(
                x    = centre_x + math.cos(angle) * 18,
                y    = centre_y + math.sin(angle) * 18,
                vx   = math.cos(angle) * speed,
                vy   = math.sin(angle) * speed,
                life = 1.0,
                size = 1.5 + self.amplitude * 4.0,
            ))
        if len(self.particles) > self._PARTICLE_HARD_CAP:
            del self.particles[:len(self.particles) - self._PARTICLE_TRIM_TO]

    def _update_particles(self) -> None:
        live: list[_Particle] = []
        for p in self.particles:
            p.x    += p.vx
            p.y    += p.vy
            p.life -= 0.018
            p.vx   *= 0.992
            p.vy   *= 0.992
            if p.life > 0:
                live.append(p)
        self.particles = live

    # ── main paint event ─────────────────────────────────────────────────────

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width  = self.width()
        height = self.height()
        centre = QPointF(width / 2, height / 2)
        radius = min(width, height) * 0.34

        # ── PASS 1: static background (cached QPixmap) ───────────────────────
        current_size = (width, height)
        if self._bg_pixmap is None or self._bg_size != current_size:
            self._bg_pixmap = QPixmap(width, height)
            self._bg_pixmap.fill(QColor(C.BG))
            bg = QPainter(self._bg_pixmap)
            bg.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._paint_grid_static(bg, width, height)
            self._paint_ambient_static(bg, width, height)
            self._paint_brackets_static(bg, width, height)
            bg.end()
            self._bg_size = current_size

        painter.drawPixmap(0, 0, self._bg_pixmap)

        # ── PASS 2: dynamic elements (every frame) ───────────────────────────
        self._paint_arcs(painter, centre, radius)
        self._paint_liquid_orb(painter, centre, radius)   # new pristine orb
        self._paint_scanner(painter, centre, radius)
        self._paint_crosshairs(painter, centre, radius)
        self._paint_particles(painter)
        self._paint_banner(painter, width)

    # ── static background helpers ────────────────────────────────────────────

    def _paint_grid_static(self, painter: QPainter, width: int, height: int) -> None:
        """Render background grid into the cached QPixmap — zero per-frame cost."""
        minor = QPen(QColor(153, 16, 36, 14), 1)
        major = QPen(QColor(153, 16, 36, 34), 1)
        spacing = 36
        for index, x in enumerate(range(0, width, spacing)):
            painter.setPen(major if index % 5 == 0 else minor)
            painter.drawLine(x, 0, x, height)
        for index, y in enumerate(range(0, height, spacing)):
            painter.setPen(major if index % 5 == 0 else minor)
            painter.drawLine(0, y, width, y)

    def _paint_ambient_static(self, painter: QPainter, width: int, height: int) -> None:
        """Ambient crimson glow behind the orb plus an edge vignette — cached."""
        centre_x = width * 0.5
        centre_y = height * 0.5
        glow = QRadialGradient(centre_x, centre_y, min(width, height) * 0.48)
        glow.setColorAt(0.0, QColor(255, 26, 60, 26))
        glow.setColorAt(0.65, QColor(255, 26, 60, 10))
        glow.setColorAt(1.0, QColor(255, 26, 60, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawRect(0, 0, width, height)
        vignette = QRadialGradient(centre_x, centre_y, max(width, height) * 0.72)
        vignette.setColorAt(0.0, QColor(0, 0, 0, 0))
        vignette.setColorAt(0.72, QColor(0, 0, 0, 0))
        vignette.setColorAt(1.0, QColor(0, 0, 0, 165))
        painter.setBrush(vignette)
        painter.drawRect(0, 0, width, height)

    def _paint_brackets_static(self, painter: QPainter, width: int, height: int) -> None:
        """Render corner targeting brackets into the cached QPixmap."""
        painter.setPen(QPen(QColor(255, 26, 60, 145), 2))
        margin = 22
        length = min(width, height) * 0.12
        points = [
            ((margin, margin),               (margin + length, margin),           (margin, margin + length)),
            ((width - margin, margin),       (width - margin - length, margin),   (width - margin, margin + length)),
            ((margin, height - margin),      (margin + length, height - margin),  (margin, height - margin - length)),
            ((width - margin, height - margin), (width - margin - length, height - margin), (width - margin, height - margin - length)),
        ]
        for pivot, horizontal, vertical in points:
            painter.drawLine(QPointF(*pivot), QPointF(*horizontal))
            painter.drawLine(QPointF(*pivot), QPointF(*vertical))

    # ── dynamic painting helpers ─────────────────────────────────────────────

    def _paint_arcs(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        """Rotating arc layers — alpha spikes with amplitude."""
        layers = [
            (radius * 1.00,  31, 108, 3),
            (radius * 0.78, -47,  64, 2),
            (radius * 1.22, 146,  78, 2),
            (radius * 0.52, 212, 122, 2),
        ]
        for index, (r, offset, span, width) in enumerate(layers):
            alpha = clamp_channel(120 + self.amplitude * 100 - index * 12)
            pen   = QPen(QColor(255, 26, 60, alpha), width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            rect  = QRectF(centre.x() - r, centre.y() - r, r * 2, r * 2)
            start = int((self.rotation * (1.0 if index % 2 == 0 else -0.7) + offset) * 16)
            painter.drawArc(rect, start, int(span * 16))
        painter.setPen(QPen(QColor(255, 255, 255, clamp_channel(72 + self.amplitude * 80)), 1))
        for r in (radius * 0.34, radius * 0.64, radius * 1.42):
            rect = QRectF(centre.x() - r, centre.y() - r, r * 2, r * 2)
            painter.drawEllipse(rect)

    def _paint_liquid_orb(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        """
        Liquid Vector Orb — Mark VII HUD centrepiece.

        Layers (outer → inner):
          1. Outer diffuse corona    — breathes via _pulse; spikes with amplitude
          2. Mid corona              — tighter, brighter
          3. Outer orb body shell    — deep crimson base
          4. Mid shell               — vibrant crimson
          5. Inner luminous core     — breathes + amplitude-driven expansion
          6. Specular hotspot        — white highlight, top-left quadrant
          7. Equatorial rotating ring with tick marks
          8. Radial circuit traces   — flare during speech
          9. Hexagonal reticle       — counter-rotates
         10. Amplitude pulse rings   — active during speech only
         11. State label

        All drawing uses QPainterPath + filled ellipses — zero external assets.
        """
        amp     = self.amplitude
        cx, cy  = centre.x(), centre.y()
        pulse   = math.sin(self._pulse)      # -1 → +1  slow breathe
        pulse_n = (pulse + 1.0) * 0.5        #  0 → 1   normalised

        # ── 1. OUTER DIFFUSE CORONA ──────────────────────────────────────────
        # Breathes gently when idle; scales sharply with vocal amplitude.
        corona_r     = radius * (0.92 + 0.14 * pulse_n + 0.22 * amp)
        corona_alpha = clamp_channel(28 + int(pulse_n * 22) + int(amp * 80))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 26, 60, corona_alpha))
        painter.drawEllipse(
            QRectF(cx - corona_r, cy - corona_r, corona_r * 2, corona_r * 2)
        )

        # ── 2. MID CORONA ────────────────────────────────────────────────────
        mid_r = radius * (0.72 + 0.08 * pulse_n + 0.14 * amp)
        painter.setBrush(QColor(255, 26, 60, clamp_channel(55 + int(amp * 100))))
        painter.drawEllipse(QRectF(cx - mid_r, cy - mid_r, mid_r * 2, mid_r * 2))

        # ── 3. ORB BODY — OUTER SHELL ────────────────────────────────────────
        orb_r = radius * 0.52
        painter.setBrush(QColor(100, 8, 20, 230))
        painter.drawEllipse(QRectF(cx - orb_r, cy - orb_r, orb_r * 2, orb_r * 2))

        # ── 4. ORB BODY — MID SHELL ──────────────────────────────────────────
        mid_shell_r = orb_r * 0.82
        painter.setBrush(QColor(200, 16, 40, 200))
        painter.drawEllipse(
            QRectF(cx - mid_shell_r, cy - mid_shell_r, mid_shell_r * 2, mid_shell_r * 2)
        )

        # ── 5. LUMINOUS CORE — breathes + amplitude-driven expansion ─────────
        core_r = orb_r * (0.48 + 0.10 * pulse_n + 0.20 * amp)
        painter.setBrush(QColor(255, 80, 100, 220))
        painter.drawEllipse(QRectF(cx - core_r, cy - core_r, core_r * 2, core_r * 2))

        # ── 6. SPECULAR HOTSPOT ──────────────────────────────────────────────
        spec_r = orb_r * 0.22
        spec_x = cx - orb_r * 0.28
        spec_y = cy - orb_r * 0.28
        painter.setBrush(QColor(255, 255, 255, clamp_channel(60 + int(amp * 90))))
        painter.drawEllipse(
            QRectF(spec_x - spec_r, spec_y - spec_r, spec_r * 2, spec_r * 2)
        )

        # ── 7. EQUATORIAL ROTATING RING ──────────────────────────────────────
        ring_r_x = orb_r * 1.18
        ring_r_y = orb_r * 0.28   # vertical squash → 3D ring illusion
        ring_pen = QPen(QColor(255, 26, 60, clamp_channel(140 + int(amp * 100))), 1.5)
        ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.rotation * 0.6)
        painter.drawEllipse(QRectF(-ring_r_x, -ring_r_y, ring_r_x * 2, ring_r_y * 2))
        tick_pen = QPen(QColor(255, 120, 140, clamp_channel(160 + int(amp * 80))), 1.2)
        painter.setPen(tick_pen)
        for tick_deg in range(0, 360, 45):
            t    = math.radians(tick_deg)
            tx   = math.cos(t) * ring_r_x
            ty   = math.sin(t) * ring_r_y
            tlen = orb_r * 0.10
            nx   = tx / (ring_r_x + 1e-9)
            ny   = ty / (ring_r_y + 1e-9)
            painter.drawLine(QPointF(tx, ty), QPointF(tx - nx * tlen, ty - ny * tlen))
        painter.restore()

        # ── 8. RADIAL CIRCUIT TRACES ─────────────────────────────────────────
        trace_alpha = clamp_channel(40 + int(amp * 180))
        trace_pen   = QPen(QColor(255, 26, 60, trace_alpha), 1)
        trace_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(trace_pen)
        for n in range(8):
            angle     = math.radians(self.rotation * 0.8 + n * 45.0)
            trace_len = radius * (0.28 + 0.22 * amp + 0.06 * math.sin(self._pulse + n))
            sx = cx + math.cos(angle) * orb_r
            sy = cy + math.sin(angle) * orb_r
            ex = cx + math.cos(angle) * (orb_r + trace_len)
            ey = cy + math.sin(angle) * (orb_r + trace_len)
            painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))
            dot_r = 2.0 + amp * 2.5
            painter.setBrush(QColor(255, 80, 100, trace_alpha))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(ex - dot_r, ey - dot_r, dot_r * 2, dot_r * 2))
            painter.setPen(trace_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        # ── 9. HEXAGONAL RETICLE ─────────────────────────────────────────────
        hex_r     = orb_r * 0.90
        hex_alpha = clamp_channel(50 + int(amp * 90))
        hex_pen   = QPen(QColor(255, 200, 200, hex_alpha), 0.8)
        painter.setPen(hex_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(-self.rotation * 0.35)
        hex_path = QPainterPath()
        for n in range(6):
            angle = math.radians(n * 60)
            px    = math.cos(angle) * hex_r
            py    = math.sin(angle) * hex_r
            if n == 0:
                hex_path.moveTo(px, py)
            else:
                hex_path.lineTo(px, py)
        hex_path.closeSubpath()
        painter.drawPath(hex_path)
        inner_hex_r    = hex_r * 0.60
        inner_hex_path = QPainterPath()
        for n in range(6):
            angle = math.radians(n * 60 + 30)
            px    = math.cos(angle) * inner_hex_r
            py    = math.sin(angle) * inner_hex_r
            if n == 0:
                inner_hex_path.moveTo(px, py)
            else:
                inner_hex_path.lineTo(px, py)
        inner_hex_path.closeSubpath()
        painter.drawPath(inner_hex_path)
        spoke_pen = QPen(QColor(255, 26, 60, clamp_channel(30 + int(amp * 70))), 0.6)
        painter.setPen(spoke_pen)
        for n in range(6):
            a_out = math.radians(n * 60)
            painter.drawLine(
                QPointF(math.cos(a_out) * inner_hex_r, math.sin(a_out) * inner_hex_r),
                QPointF(math.cos(a_out) * hex_r,       math.sin(a_out) * hex_r),
            )
        painter.restore()

        # ── 10. AMPLITUDE PULSE RINGS — active during speech ─────────────────
        if amp > 0.06:
            pulse_ring_r  = orb_r * (1.0 + amp * 0.55)
            pulse_alpha   = clamp_channel(int(amp * 200))
            painter.setPen(QPen(QColor(255, 26, 60, pulse_alpha), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QRectF(cx - pulse_ring_r, cy - pulse_ring_r,
                       pulse_ring_r * 2, pulse_ring_r * 2)
            )
            pulse_ring_r2 = orb_r * (1.0 + amp * 0.82)
            painter.setPen(QPen(QColor(255, 26, 60, clamp_channel(int(amp * 100))), 0.8))
            painter.drawEllipse(
                QRectF(cx - pulse_ring_r2, cy - pulse_ring_r2,
                       pulse_ring_r2 * 2, pulse_ring_r2 * 2)
            )

        # ── 11. STATE LABEL ───────────────────────────────────────────────────
        label = (
            "SPEAKING"  if self.state_name == "SPEAKING"  else
            "LISTENING" if self.state_name == "LISTENING" else
            "ONLINE"    if self.face_target               else
            "STANDBY"
        )
        label_alpha = clamp_channel(140 + int(amp * 115))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.setPen(QColor(255, 255, 255, label_alpha))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawText(
            QRectF(cx - radius, cy + orb_r + 18, radius * 2, 26),
            Qt.AlignmentFlag.AlignCenter, label,
        )

    def _paint_scanner(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        angle = math.radians(self.scan)
        end   = QPointF(
            centre.x() + math.cos(angle) * radius * 1.48,
            centre.y() + math.sin(angle) * radius * 1.48,
        )
        pen = QPen(QColor(255, 26, 60, clamp_channel(90 + self.amplitude * 130)), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(centre, end)
        wedge = QPainterPath()
        wedge.moveTo(centre)
        rect = QRectF(
            centre.x() - radius * 1.48, centre.y() - radius * 1.48,
            radius * 2.96, radius * 2.96,
        )
        wedge.arcTo(rect, -self.scan, -22)
        wedge.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 26, 60, clamp_channel(18 + self.amplitude * 36)))
        painter.drawPath(wedge)

    def _paint_crosshairs(self, painter: QPainter, centre: QPointF, radius: float) -> None:
        painter.setPen(QPen(QColor(C.WHITE), 1))
        gap    = radius * 0.13
        length = radius * 0.48
        painter.drawLine(
            QPointF(centre.x() - length, centre.y()),
            QPointF(centre.x() - gap,    centre.y()),
        )
        painter.drawLine(
            QPointF(centre.x() + gap,    centre.y()),
            QPointF(centre.x() + length, centre.y()),
        )
        painter.drawLine(
            QPointF(centre.x(), centre.y() - length),
            QPointF(centre.x(), centre.y() - gap),
        )
        painter.drawLine(
            QPointF(centre.x(), centre.y() + gap),
            QPointF(centre.x(), centre.y() + length),
        )
        painter.setBrush(QColor(255, 26, 60, clamp_channel(160 + self.amplitude * 70)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(centre.x() - 3, centre.y() - 3, 6, 6))

    def _paint_particles(self, painter: QPainter) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        for p in self.particles:
            alpha = clamp_channel(int(255 * p.life))
            painter.setBrush(QColor(255, 26, 60, alpha))
            half = p.size * 0.5
            painter.drawEllipse(QRectF(p.x - half, p.y - half, p.size, p.size))

    def _paint_banner(self, painter: QPainter, width: int) -> None:
        if not self.banner_text or self.banner_alpha <= 0:
            return
        alpha = clamp_channel(self.banner_alpha)
        rect  = QRectF(width * 0.15, 20, width * 0.70, 46)
        painter.setPen(QPen(QColor(255, 26, 60, alpha), 1))
        painter.setBrush(QColor(15, 15, 20, clamp_channel(int(alpha * 0.82))))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.setPen(QColor(255, 255, 255, alpha))
        painter.drawText(
            rect.adjusted(14, 0, -14, 0), Qt.AlignmentFlag.AlignCenter, self.banner_text
        )


# ──────────────────────────────────────────────────────────────────────────────
# MINI ORB  — 64×64 floating orb for non-HUD views
# ──────────────────────────────────────────────────────────────────────────────

class MiniOrb(QWidget):
    """
    64×64 floating orb that persists when the user navigates away from the
    main HUD view.  Shares the same amplitude and state signals as CentralHud
    via OrionBus; displays in the bottom-right corner of the shell.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(64, 64)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self.amplitude        = 0.0
        self.target_amplitude = 0.0
        self._pulse           = 0.0
        self.rotation         = 0.0
        self.state_name       = "STANDBY"

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_amplitude(self, value: float) -> None:
        self.target_amplitude = max(0.0, min(1.0, float(value)))

    def set_state(self, state: str) -> None:
        self.state_name = str(state or "STANDBY").upper()
        self.update()

    def _tick(self) -> None:
        self.amplitude  += (self.target_amplitude - self.amplitude) * 0.22
        self._pulse      = (self._pulse + 0.04) % (2 * math.pi)
        self.rotation    = (self.rotation + 1.2 + self.amplitude * 4.0) % 360.0
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(C.BG))

        cx     = 32.0
        cy     = 32.0
        amp    = self.amplitude
        pulse_n = (math.sin(self._pulse) + 1.0) * 0.5

        # Outer corona
        cr = 28 * (0.92 + 0.14 * pulse_n + 0.22 * amp)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 26, 60, clamp_channel(30 + int(pulse_n * 20) + int(amp * 70))))
        painter.drawEllipse(QRectF(cx - cr, cy - cr, cr * 2, cr * 2))

        # Body
        orb_r = 14.0
        painter.setBrush(QColor(120, 10, 24, 230))
        painter.drawEllipse(QRectF(cx - orb_r, cy - orb_r, orb_r * 2, orb_r * 2))

        # Core
        core_r = orb_r * (0.48 + 0.10 * pulse_n + 0.20 * amp)
        painter.setBrush(QColor(255, 80, 100, 210))
        painter.drawEllipse(QRectF(cx - core_r, cy - core_r, core_r * 2, core_r * 2))

        # Specular
        spec_r = orb_r * 0.20
        painter.setBrush(QColor(255, 255, 255, clamp_channel(50 + int(amp * 80))))
        painter.drawEllipse(
            QRectF(cx - orb_r * 0.28 - spec_r, cy - orb_r * 0.28 - spec_r,
                   spec_r * 2, spec_r * 2)
        )

        # Thin ring
        ring_pen = QPen(QColor(255, 26, 60, clamp_channel(120 + int(amp * 100))), 1.0)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        ring_rx = orb_r * 1.18
        ring_ry = orb_r * 0.28
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.rotation * 0.6)
        painter.drawEllipse(QRectF(-ring_rx, -ring_ry, ring_rx * 2, ring_ry * 2))
        painter.restore()


# ──────────────────────────────────────────────────────────────────────────────
# STACKED VIEWS
# ──────────────────────────────────────────────────────────────────────────────

class HudView(QWidget):
    """View 0 — Main Core HUD (the full CentralHud widget)."""

    def __init__(self, hud: CentralHud, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(hud)


class LogConsoleView(QWidget):
    """View 1 — Overt Log & Text Dispatch Console."""

    def __init__(self, bus: OrionBus, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.bus    = bus
        self.worker: Any = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("INTELLIGENCE MONITOR")
        heading.setObjectName("panelHeading")

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(2000)
        self.log_box.setObjectName("logBox")

        browser_heading = QLabel("RESEARCH BROWSER")
        browser_heading.setObjectName("panelHeading")

        self.browser_line = QLineEdit()
        self.browser_line.setPlaceholderText("Search query or URL")
        self.browser_line.returnPressed.connect(self._browser_search)
        self.browser_search_btn = QPushButton("SEARCH")
        self.browser_search_btn.clicked.connect(self._browser_search)
        self.browser_open_btn   = QPushButton("OPEN")
        self.browser_open_btn.clicked.connect(self._browser_open)

        browser_row = QHBoxLayout()
        browser_row.addWidget(self.browser_line, 1)
        browser_row.addWidget(self.browser_search_btn)
        browser_row.addWidget(self.browser_open_btn)

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Manual command uplink  (Enter / Ctrl+Return)")
        self.input_line.returnPressed.connect(self._send_manual)
        self.send_btn  = QPushButton("SEND")
        self.send_btn.clicked.connect(self._send_manual)
        self.file_btn  = QPushButton("SCAN FILE")
        self.file_btn.clicked.connect(self._scan_file)

        self.mic_toggle = QCheckBox("MICROPHONE ACTIVE")
        self.mic_toggle.setChecked(True)
        self.mic_toggle.stateChanged.connect(self._toggle_microphone)
        self.bus.mic_enabled.connect(self.mic_toggle.setChecked)

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_line, 1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.file_btn)

        layout.addWidget(heading)
        layout.addWidget(self.log_box, 1)
        layout.addWidget(browser_heading)
        layout.addLayout(browser_row)
        layout.addLayout(input_row)
        layout.addWidget(self.mic_toggle)

        self.bus.log.connect(self.write_log)

    def attach_worker(self, worker: Any) -> None:
        self.worker = worker
        worker.set_microphone_enabled(self.mic_toggle.isChecked())

    def write_log(self, message: str) -> None:
        line = f"{now_stamp()}  {message}"
        self.log_box.appendPlainText(line)
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)
        self.log_box.ensureCursorVisible()

    def _send_manual(self) -> None:
        text = self.input_line.text().strip()
        if not text:
            return
        self.input_line.clear()
        self.write_log(f"YOU: {text}")
        if self.worker is None:
            self.write_log("SYS: Live session not ready.")
            return
        asyncio.create_task(self.worker.submit_text(text))

    def _scan_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select file for Orion scan",
            str(BASE_DIR),
            "All supported files (*.txt *.md *.py *.json *.csv *.tsv "
            "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff);;All files (*.*)",
        )
        if not file_path:
            return
        focus = self.input_line.text().strip()
        if focus:
            self.input_line.clear()
        self.write_log(f"FILE: queued scan for {Path(file_path).name}")
        if self.worker is None:
            self.write_log("FILE: Live session not ready.")
            return
        asyncio.create_task(self.worker.submit_file_for_review(file_path, focus))

    def _browser_search(self) -> None:
        query = self.browser_line.text().strip()
        if not query:
            return
        try:
            SecuritySanitiser.guard_text(query, "browser.query")
        except SecurityViolation as exc:
            self.write_log(f"SEC: {exc}")
            return
        url = query if self._looks_like_url(query) else f"https://www.bing.com/search?q={quote_plus(query)}"
        if "://" not in url:
            url = f"https://{url}"
        webbrowser.open(url)
        self.write_log(f"WEB: research opened for {query}")
        if self.worker is not None and not self._looks_like_url(query):
            asyncio.create_task(
                self.worker.submit_text(
                    f"Research this with me and keep the answer concise: {query}"
                )
            )

    def _browser_open(self) -> None:
        target = self.browser_line.text().strip()
        if not target:
            return
        try:
            SecuritySanitiser.guard_text(target, "browser.url")
        except SecurityViolation as exc:
            self.write_log(f"SEC: {exc}")
            return
        url    = target if "://" in target else f"https://{target}"
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            self.write_log("WEB: blocked unsupported URL scheme.")
            return
        webbrowser.open(url)
        self.write_log(f"WEB: opened {url}")

    def _toggle_microphone(self) -> None:
        enabled = self.mic_toggle.isChecked()
        if self.worker is not None:
            self.worker.set_microphone_enabled(enabled)
        self.write_log(f"SYS: microphone {'active' if enabled else 'muted'}.")

    def _looks_like_url(self, text: str) -> bool:
        parsed = urlparse(text if "://" in text else f"https://{text}")
        return "." in parsed.netloc and " " not in parsed.netloc


class MemoryMatrixView(QWidget):
    """View 2 — Memory Matrix (SQLite FTS5 browser)."""

    def __init__(self, memory: OrionMemoryMatrix, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.memory = memory
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("MEMORY MATRIX — SQLite FTS5 INTELLIGENCE STORE")
        heading.setObjectName("panelHeading")

        self.search_line = QLineEdit()
        self.search_line.setPlaceholderText("Full-text search (leave blank to list all)")
        self.search_line.returnPressed.connect(self.refresh)
        self.search_btn  = QPushButton("SEARCH")
        self.search_btn.clicked.connect(self.refresh)
        self.clear_btn   = QPushButton("CLEAR FILTER")
        self.clear_btn.clicked.connect(self._clear_filter)

        filter_row = QHBoxLayout()
        filter_row.addWidget(self.search_line, 1)
        filter_row.addWidget(self.search_btn)
        filter_row.addWidget(self.clear_btn)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Category", "Key", "Value", "Updated"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { background: #0a0a0e; color: #ffffff; "
            "gridline-color: #2a1118; alternate-background-color: #0f0f14; }"
            "QHeaderView::section { background: #0f0f14; color: #ff1a3c; "
            "border: 1px solid #2a1118; padding: 4px; font-weight: bold; }"
        )

        self.status_label = QLabel("Awaiting query.")
        self.status_label.setObjectName("mutedLabel")

        layout.addWidget(heading)
        layout.addLayout(filter_row)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.status_label)

    def refresh(self) -> None:
        query   = self.search_line.text().strip()
        records = self.memory.records(query=query, limit=500)
        self.table.setRowCount(0)
        for row in records:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for col, key in enumerate(("category", "key_ref", "value", "updated_at")):
                item = QTableWidgetItem(str(row.get(key, "")))
                self.table.setItem(r, col, item)
        self.table.resizeColumnsToContents()
        self.status_label.setText(
            f"{len(records)} record(s) retrieved."
            + (f"  Filter: '{query}'" if query else "  Displaying all records.")
        )

    def _clear_filter(self) -> None:
        self.search_line.clear()
        self.refresh()


class TelemetryView(QWidget):
    """View 3 — System Telemetry & Process Governor."""

    def __init__(self, bus: OrionBus, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.bus    = bus
        self.worker: Any = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        heading = QLabel("SYSTEM TELEMETRY & PROCESS GOVERNOR")
        heading.setObjectName("panelHeading")

        # Metric bars
        self.cpu_bar = MetricBar("CPU LOAD")
        self.ram_bar = MetricBar("RAM CONSUMPTION")
        self.net_bar = MetricBar("NETWORK ACTIVITY")

        # Environment
        env_heading = QLabel("GEOGRAPHICAL NODE")
        env_heading.setObjectName("panelHeading")
        self.location_label = QLabel("Location: awaiting refresh.")
        self.location_label.setObjectName("mutedLabel")
        self.location_label.setWordWrap(True)
        self.weather_label = QLabel("Weather: awaiting refresh.")
        self.weather_label.setObjectName("mutedLabel")
        self.weather_label.setWordWrap(True)
        self.env_refresh_btn = QPushButton("REFRESH ENVIRONMENT")

        # Calendar
        cal_heading = QLabel("CALENDAR")
        cal_heading.setObjectName("panelHeading")
        self.calendar = QCalendarWidget()
        self.calendar.setGridVisible(False)
        self.calendar.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        self.calendar.setMaximumHeight(200)

        # Process table
        proc_heading = QLabel("PROCESS INVENTORY")
        proc_heading.setObjectName("panelHeading")
        self.proc_table = QTableWidget()
        self.proc_table.setColumnCount(4)
        self.proc_table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "RAM %"])
        self.proc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.proc_table.setMaximumHeight(220)
        self.proc_table.setStyleSheet(
            "QTableWidget { background: #0a0a0e; color: #ffffff; "
            "gridline-color: #2a1118; }"
            "QHeaderView::section { background: #0f0f14; color: #ff1a3c; "
            "border: 1px solid #2a1118; padding: 4px; font-weight: bold; }"
        )
        self.proc_refresh_btn = QPushButton("REFRESH PROCESSES")
        self.proc_refresh_btn.clicked.connect(self._refresh_processes)

        for widget in [
            heading,
            self.cpu_bar, self.ram_bar, self.net_bar,
            env_heading,
            self.location_label, self.weather_label, self.env_refresh_btn,
            cal_heading, self.calendar,
            proc_heading, self.proc_table, self.proc_refresh_btn,
        ]:
            layout.addWidget(widget)
        layout.addStretch(1)

    def attach_worker(self, worker: Any) -> None:
        self.worker = worker

    def attach_env_refresh(self, slot: Callable) -> None:
        self.env_refresh_btn.clicked.connect(slot)

    def _refresh_processes(self) -> None:
        self.proc_table.setRowCount(0)
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                r    = self.proc_table.rowCount()
                self.proc_table.insertRow(r)
                self.proc_table.setItem(r, 0, QTableWidgetItem(str(info.get("pid", ""))))
                self.proc_table.setItem(r, 1, QTableWidgetItem(str(info.get("name") or "")))
                self.proc_table.setItem(r, 2, QTableWidgetItem(f"{info.get('cpu_percent') or 0:.1f}"))
                self.proc_table.setItem(r, 3, QTableWidgetItem(f"{info.get('memory_percent') or 0:.1f}"))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.proc_table.resizeColumnsToContents()


# ──────────────────────────────────────────────────────────────────────────────
# API KEY DIALOG
# ──────────────────────────────────────────────────────────────────────────────

class ApiKeyDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("O.R.I.O.N. Key Exchange")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setStyleSheet(
            "QDialog { background: #050508; color: #ffffff; }"
            "QLabel { color: #ffffff; }"
            "QLineEdit {"
            "  background: #0f0f14; color: #ffffff;"
            "  border: 1px solid #991024; border-radius: 6px;"
            "  padding: 10px; selection-background-color: #ff1a3c;"
            "}"
            "QPushButton {"
            "  background: #991024; color: #ffffff;"
            "  border: 1px solid #ff1a3c; border-radius: 6px; padding: 8px 14px;"
            "}"
            "QPushButton:hover { background: #ff1a3c; }"
        )
        layout = QVBoxLayout(self)
        title  = QLabel("Gemini Live authentication token required.")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        body   = QLabel("Enter the API key. It will be stored locally in config/api_keys.json.")
        body.setWordWrap(True)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("AIza...")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addWidget(self.key_edit)
        layout.addWidget(buttons)

    def key(self) -> str:
        return self.key_edit.text().strip()

    def _accept_if_valid(self) -> None:
        if not self.key():
            QMessageBox.warning(self, "Key Required", "Supply a Gemini API key to initialise O.R.I.O.N.")
            return
        self.accept()


# ──────────────────────────────────────────────────────────────────────────────
# HOLOGRAPHIC TOGGLE
# ──────────────────────────────────────────────────────────────────────────────

class HolographicToggle(QPushButton):
    def __init__(self, target: "OrionMainWindow") -> None:
        super().__init__("ORION")
        self.target        = target
        self._drag_offset: Any = None
        self.setWindowTitle("O.R.I.O.N. Toggle")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(92, 42)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QPushButton {"
            "  background: rgba(15, 15, 20, 188);"
            "  color: #ffffff;"
            "  border: 1px solid #ff1a3c;"
            "  border-radius: 20px;"
            "  font-family: 'Segoe UI'; font-weight: 800;"
            "}"
            "QPushButton:hover { background: rgba(255, 26, 60, 214); }"
        )
        self.clicked.connect(self.toggle_target)

    def toggle_target(self) -> None:
        if self.target.isVisible():
            self.target.hide()
        else:
            self.target.show()
            self.target.raise_()
            self.target.activateWindow()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW  —  QStackedWidget shell with navigation bar
# ──────────────────────────────────────────────────────────────────────────────

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {C.BG};
    color: {C.WHITE};
    font-family: "Segoe UI";
}}
QToolTip {{
    background: {C.PANEL};
    color: {C.WHITE};
    border: 1px solid {C.PRI_DIM};
    padding: 6px;
}}
QLabel {{ background: transparent; }}
QCheckBox {{ background: transparent; }}
QFrame#headerFrame {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #16161f, stop:1 #0a0a10);
    border: 1px solid {C.BORDER};
    border-bottom: 2px solid {C.PRI_DIM};
    border-radius: 8px;
}}
QFrame#panelFrame {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #12121a, stop:1 #0b0b10);
    border: 1px solid #331420;
    border-radius: 8px;
}}
QLabel#titleLabel {{
    color: {C.WHITE};
    font-size: 23px;
    font-weight: 800;
}}
QLabel#subtitleLabel, QLabel#mutedLabel {{
    color: {C.MUTED};
    font-size: 11px;
}}
QLabel#panelHeading {{
    color: {C.PRI};
    font-size: 12px;
    font-weight: 800;
    padding-bottom: 2px;
    border-bottom: 1px solid {C.BORDER};
}}
QLabel#clockLabel, QLabel#stateLabel {{
    color: {C.WHITE};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #191922, stop:1 #0d0d13);
    border: 1px solid {C.PRI_DIM};
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 700;
}}
QLabel#stateLabel {{ color: {C.PRI}; }}
QPlainTextEdit#logBox {{
    background: #07070b;
    color: {C.WHITE};
    border: 1px solid {C.BORDER};
    border-radius: 8px;
    padding: 10px;
    font-family: "Cascadia Mono", "Consolas";
    font-size: 11px;
    selection-background-color: {C.PRI};
}}
QLineEdit {{
    background: #0a0a10;
    color: {C.WHITE};
    border: 1px solid {C.PRI_DIM};
    border-radius: 8px;
    padding: 9px 11px;
    selection-background-color: {C.PRI};
}}
QLineEdit:focus {{
    border: 1px solid {C.PRI};
    background: #0d0d14;
}}
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #b3132b, stop:1 #7c0d1e);
    color: {C.WHITE};
    border: 1px solid {C.PRI};
    border-radius: 8px;
    padding: 9px 16px;
    font-weight: 700;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {C.PRI}, stop:1 #b3132b);
    border: 1px solid #ff5c73;
}}
QPushButton:pressed {{ background: #66091a; }}
QCheckBox {{
    color: {C.WHITE};
    spacing: 9px;
    font-weight: 700;
}}
QCheckBox::indicator {{
    width: 20px;
    height: 20px;
}}
QCheckBox::indicator:unchecked {{
    background: #0a0a10;
    border: 1px solid {C.PRI_DIM};
    border-radius: 5px;
}}
QCheckBox::indicator:unchecked:hover {{ border: 1px solid {C.PRI}; }}
QCheckBox::indicator:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {C.PRI}, stop:1 {C.PRI_DIM});
    border: 1px solid #ff8a9b;
    border-radius: 5px;
}}
QComboBox {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #14141d, stop:1 #0d0d13);
    color: {C.WHITE};
    border: 1px solid {C.PRI_DIM};
    border-radius: 8px;
    padding: 7px 12px;
    font-weight: 700;
    min-width: 260px;
}}
QComboBox:hover {{ border: 1px solid {C.PRI}; }}
QComboBox::drop-down {{
    border: none;
    padding-right: 10px;
}}
QComboBox QAbstractItemView {{
    background: {C.PANEL};
    color: {C.WHITE};
    border: 1px solid {C.PRI_DIM};
    selection-background-color: {C.PRI};
    padding: 4px;
}}
QTableWidget {{
    background: #08080d;
    color: {C.WHITE};
    gridline-color: {C.BORDER};
    alternate-background-color: #101018;
    border: 1px solid {C.BORDER};
    border-radius: 8px;
    selection-background-color: {C.PRI_DIM};
}}
QHeaderView::section {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #16161f, stop:1 #0f0f14);
    color: {C.PRI};
    border: 1px solid {C.BORDER};
    padding: 5px;
    font-weight: 700;
}}
QCalendarWidget QWidget {{ alternate-background-color: #101018; }}
QCalendarWidget QAbstractItemView {{
    background: #08080d;
    color: {C.WHITE};
    selection-background-color: {C.PRI};
    selection-color: {C.WHITE};
    outline: none;
}}
QCalendarWidget QToolButton {{
    background: transparent;
    color: {C.WHITE};
    font-weight: 700;
    border-radius: 6px;
    padding: 4px 8px;
}}
QCalendarWidget QToolButton:hover {{ background: {C.PRI_DIM}; }}
QCalendarWidget QMenu {{ background: {C.PANEL}; color: {C.WHITE}; }}
QCalendarWidget QSpinBox {{
    background: {C.PANEL};
    color: {C.WHITE};
    border: 1px solid {C.PRI_DIM};
}}
QSplitter::handle {{ background: {C.BG}; }}
QScrollBar:vertical {{
    background: transparent;
    width: 9px;
    margin: 2px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C.PRI_DIM};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {C.PRI}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 9px;
    margin: 2px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {C.PRI_DIM};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {C.PRI}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


class OrionMainWindow(QMainWindow):
    def __init__(self, bus: OrionBus) -> None:
        super().__init__()
        self.bus          = bus
        self.worker: "GenAILiveWorker | None" = None
        self.active_state = "INITIALISING"
        self.telemetry: dict[str, Any] = {
            "cpu":          0.0,
            "ram":          0.0,
            "net_bps":      0.0,
            "net_percent":  0.0,
            "state":        self.active_state,
            "mic_active":   True,
            "updated_at":   utc_stamp(),
        }
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1180, 720)
        self.resize(1360, 820)
        self.setStyleSheet(APP_STYLESHEET)
        self._build_ui()
        self._connect_bus()

    # ── worker attachment ─────────────────────────────────────────────────────

    def attach_worker(self, worker: "GenAILiveWorker") -> None:
        self.worker = worker
        self.log_view.attach_worker(worker)
        self.telemetry_view.attach_worker(worker)
        worker.set_microphone_enabled(self.log_view.mic_toggle.isChecked())

    def attach_memory(self, memory: OrionMemoryMatrix) -> None:
        """Inject the memory instance required by the Memory Matrix view."""
        # already injected at construction via MemoryMatrixView(memory)
        pass

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root   = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(self._build_header())

        # ── horizontal splitter: left telemetry | stacked views ──────────────
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self._build_left_panel())

        # Stacked widget — houses all four views
        self.hud           = CentralHud()
        self.hud_view      = HudView(self.hud)
        # LogConsoleView and TelemetryView constructed before memory is known;
        # MemoryMatrixView is assembled with a placeholder and replaced in
        # attach_memory() — in practice memory arrives early enough via the
        # OrionBus constructor.  We pass a dummy initially.
        self._dummy_memory = OrionMemoryMatrix(CORE_DB_PATH, CONFIG_DIR, self.bus)
        self.log_view      = LogConsoleView(self.bus)
        self.memory_view   = MemoryMatrixView(self._dummy_memory)
        self.telemetry_view = TelemetryView(self.bus)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.hud_view)         # index 0
        self.stack.addWidget(self.log_view)          # index 1
        self.stack.addWidget(self.memory_view)       # index 2
        self.stack.addWidget(self.telemetry_view)    # index 3

        self.splitter.addWidget(self.stack)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([270, 1090])
        layout.addWidget(self.splitter, 1)

        # ── mini orb (bottom-right overlay for non-HUD views) ────────────────
        self.mini_orb = MiniOrb(root)
        self.mini_orb.hide()

        self.setCentralWidget(root)

        # Keyboard shortcut
        send_action = QAction(self)
        send_action.setShortcut(QKeySequence("Ctrl+Return"))
        send_action.triggered.connect(self.log_view._send_manual)
        self.addAction(send_action)

    def _build_header(self) -> QWidget:
        frame  = QFrame()
        frame.setObjectName("headerFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 10, 16, 10)

        self.title_label    = QLabel("O.R.I.O.N. MARK VII")
        self.title_label.setObjectName("titleLabel")
        self.subtitle_label = QLabel("OPEN RESOLUTION INTELLIGENCE OVERT NETWORK")
        self.subtitle_label.setObjectName("subtitleLabel")
        text_box = QVBoxLayout()
        text_box.setSpacing(0)
        text_box.addWidget(self.title_label)
        text_box.addWidget(self.subtitle_label)

        # Navigation dropdown
        self.nav_combo = QComboBox()
        self.nav_combo.addItems([
            "⬡  CORE HUD",
            "⌨  LOG & DISPATCH",
            "⊞  MEMORY MATRIX",
            "◈  TELEMETRY",
        ])
        self.nav_combo.currentIndexChanged.connect(self._on_nav_changed)

        self.state_label = QLabel("INITIALISING")
        self.state_label.setObjectName("stateLabel")
        self.clock_label  = QLabel(datetime.now().strftime("%H:%M:%S"))
        self.clock_label.setObjectName("clockLabel")

        layout.addLayout(text_box, 1)
        layout.addWidget(self.nav_combo)
        layout.addWidget(self.state_label)
        layout.addWidget(self.clock_label)

        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(
            lambda: self.clock_label.setText(datetime.now().strftime("%H:%M:%S"))
        )
        self.clock_timer.start()
        return frame

    def _build_left_panel(self) -> QWidget:
        """
        Persistent left panel — host telemetry bars, calendar, geographical
        node.  These remain visible across all stacked views.
        """
        frame  = QFrame()
        frame.setObjectName("panelFrame")
        frame.setMinimumWidth(250)
        frame.setMaximumWidth(320)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        heading = QLabel("HOST TELEMETRY")
        heading.setObjectName("panelHeading")

        # These bars are updated by start_telemetry() regardless of active view
        self.cpu_bar  = MetricBar("CPU LOAD")
        self.ram_bar  = MetricBar("RAM CONSUMPTION")
        self.net_bar  = MetricBar("NETWORK ACTIVITY")

        self.telemetry_note = QLabel("Local psutil monitor active.")
        self.telemetry_note.setObjectName("mutedLabel")
        self.telemetry_note.setWordWrap(True)

        cal_heading = QLabel("CALENDAR")
        cal_heading.setObjectName("panelHeading")
        self.calendar_widget = QCalendarWidget()
        self.calendar_widget.setGridVisible(False)
        self.calendar_widget.setVerticalHeaderFormat(
            QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader
        )
        self.calendar_widget.setMaximumHeight(215)

        env_heading = QLabel("GEOGRAPHICAL NODE")
        env_heading.setObjectName("panelHeading")
        self.location_label = QLabel("Location: awaiting refresh.")
        self.location_label.setObjectName("mutedLabel")
        self.location_label.setWordWrap(True)
        self.weather_label  = QLabel("Weather: awaiting refresh.")
        self.weather_label.setObjectName("mutedLabel")
        self.weather_label.setWordWrap(True)
        self.weather_button = QPushButton("REFRESH ENVIRONMENT")
        self.weather_button.clicked.connect(
            lambda: asyncio.create_task(self.refresh_environment_widgets())
        )

        for widget in [
            heading,
            self.cpu_bar, self.ram_bar, self.net_bar,
            cal_heading, self.calendar_widget,
            env_heading,
            self.location_label, self.weather_label, self.weather_button,
        ]:
            layout.addWidget(widget)
        layout.addStretch(1)
        layout.addWidget(self.telemetry_note)
        return frame

    # ── navigation ────────────────────────────────────────────────────────────

    def _on_nav_changed(self, index: int) -> None:
        """
        Switch the visible stacked view.

        When navigating away from the main HUD (index > 0):
          • Detach the full CentralHud animation timer (reduce to 4 fps).
          • Show the 64×64 MiniOrb in the bottom-right corner of the window
            so the orb continues to listen and pulse visually.

        When returning to the HUD (index == 0):
          • Restore the CentralHud timer.
          • Hide the MiniOrb.
        """
        self.stack.setCurrentIndex(index)

        if index == 0:
            # Returning to core HUD — restore full animation rate
            self.hud.timer.setInterval(33)
            self.mini_orb.hide()
        else:
            # Non-HUD view — throttle main HUD to conserve resources
            self.hud.timer.setInterval(250)
            self._reposition_mini_orb()
            self.mini_orb.show()
            self.mini_orb.raise_()

        # Refresh the memory view whenever it becomes visible
        if index == 2:
            self.memory_view.refresh()

    def _reposition_mini_orb(self) -> None:
        """Place the mini orb in the bottom-right corner of the central widget."""
        parent = self.centralWidget()
        if parent is None:
            return
        margin = 12
        self.mini_orb.move(
            parent.width()  - self.mini_orb.width()  - margin,
            parent.height() - self.mini_orb.height() - margin,
        )

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if not self.mini_orb.isHidden():
            self._reposition_mini_orb()

    # ── bus connections ───────────────────────────────────────────────────────

    def _connect_bus(self) -> None:
        # bus.log is already connected inside LogConsoleView.__init__;
        # connecting it here as well duplicated every console line.
        self.bus.state.connect(self.set_state)
        self.bus.state.connect(self.hud.set_state)
        self.bus.state.connect(self.mini_orb.set_state)
        self.bus.amplitude.connect(self.hud.set_amplitude)
        self.bus.amplitude.connect(self.mini_orb.set_amplitude)
        self.bus.banner.connect(self.hud.set_banner)
        _app = QApplication.instance()
        if _app is not None:
            self.bus.request_shutdown.connect(_app.quit)

    # ── state / logging ───────────────────────────────────────────────────────

    def write_log(self, message: str) -> None:
        self.log_view.write_log(message)

    def set_state(self, state: str) -> None:
        self.active_state       = str(state).upper()
        self.telemetry["state"] = self.active_state
        self.state_label.setText(self.active_state)

    # ── telemetry snapshot ────────────────────────────────────────────────────

    def telemetry_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self.telemetry)
        snapshot["clock"] = datetime.now().strftime("%H:%M:%S")
        if self.worker is not None:
            snapshot["queue_depth"]    = self.worker.out_queue.qsize()
            snapshot["live_connected"] = self.worker.connected
            if hasattr(self.worker, "router"):
                snapshot["providers"] = self.worker.router.provider_snapshot()
        else:
            snapshot["queue_depth"]    = 0
            snapshot["live_connected"] = False
        return snapshot

    # ── environment refresh (async, non-blocking) ─────────────────────────────

    async def refresh_environment_widgets(self) -> None:
        self.location_label.setText("Location: resolving.")
        self.weather_label.setText("Weather: resolving.")
        try:
            timeout = aiohttp_client_timeout()
            async with ClientSession(timeout=timeout) as session:
                async with session.get("https://ipapi.co/json/") as response:
                    if response.status != 200:
                        raise RuntimeError(f"location service returned {response.status}")
                    location = await response.json()
                latitude  = float(location.get("latitude"))
                longitude = float(location.get("longitude"))
                city      = str(location.get("city") or "unknown locality")
                region    = str(location.get("region") or "")
                country   = str(location.get("country_name") or location.get("country") or "")
                self.location_label.setText(
                    f"Location: {city}, {region}, {country}\n"
                    f"{latitude:.4f}, {longitude:.4f}"
                )
                weather_url = (
                    "https://api.open-meteo.com/v1/forecast"
                    f"?latitude={latitude:.5f}&longitude={longitude:.5f}"
                    "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
                    "&timezone=auto"
                )
                async with session.get(weather_url) as response:
                    if response.status != 200:
                        raise RuntimeError(f"weather service returned {response.status}")
                    weather = await response.json()
                current  = weather.get("current") or {}
                temp     = current.get("temperature_2m")
                humidity = current.get("relative_humidity_2m")
                wind     = current.get("wind_speed_10m")
                code     = int(current.get("weather_code") or 0)
                self.weather_label.setText(
                    f"Weather: {weather_code_label(code)}; {temp} °C; "
                    f"humidity {humidity}%; wind {wind} km/h."
                )
                self.write_log(f"GEO: environment synchronised for {city}.")
        except Exception as exc:
            self.location_label.setText("Location: unavailable.")
            self.weather_label.setText(
                f"Weather: unavailable - {str(exc).splitlines()[0][:90]}"
            )
            self.write_log(
                f"GEO: environment refresh failed - {str(exc).splitlines()[0][:120]}"
            )

    # ── telemetry loop (async, 0.75 s interval) ───────────────────────────────

    async def start_telemetry(self) -> None:
        last_net  = psutil.net_io_counters()
        last_time = time.monotonic()
        psutil.cpu_percent(interval=None)
        while True:
            try:
                await asyncio.sleep(0.75)
                cpu         = psutil.cpu_percent(interval=None)
                ram         = psutil.virtual_memory().percent
                current_net = psutil.net_io_counters()
                current_time = time.monotonic()
                delta_bytes = (
                    (current_net.bytes_sent + current_net.bytes_recv)
                    - (last_net.bytes_sent + last_net.bytes_recv)
                )
                elapsed       = max(0.001, current_time - last_time)
                bytes_per_sec = max(0.0, delta_bytes / elapsed)
                net_percent   = min(100.0, (bytes_per_sec / 12_500_000.0) * 100.0)
                self.telemetry.update({
                    "cpu":         float(cpu),
                    "ram":         float(ram),
                    "net_bps":     float(bytes_per_sec),
                    "net_percent": float(net_percent),
                    "state":       self.active_state,
                    "updated_at":  utc_stamp(),
                })
                # Update both the left-panel bars and the telemetry view bars
                for bar_set in ((self.cpu_bar, self.ram_bar, self.net_bar),
                                (self.telemetry_view.cpu_bar,
                                 self.telemetry_view.ram_bar,
                                 self.telemetry_view.net_bar)):
                    bar_set[0].set_value(cpu)
                    bar_set[1].set_value(ram)
                    bar_set[2].set_value(net_percent)
                last_net  = current_net
                last_time = current_time
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.write_log(f"TEL: telemetry loop recovered - {exc}")
                await asyncio.sleep(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# DISPATCHER
# ──────────────────────────────────────────────────────────────────────────────

class OrionDispatcher:
    def __init__(
        self,
        bus: OrionBus,
        memory: OrionMemoryMatrix,
        grabber: VolatileScreenGrabber,
        file_intel: LocalFileIntelligence | None = None,
    ) -> None:
        self.bus       = bus
        self.memory    = memory
        self.grabber   = grabber
        self.file_intel = file_intel or LocalFileIntelligence(bus)
        # Populated by the startup briefing: the exact stories ORION read out,
        # so open_news can open the precise source article on request.
        self.news_articles: list[dict[str, str]] = []
        # Installed-application index discovered from the Start Menu — built on
        # a worker thread so startup never blocks.
        self._app_index: dict[str, str] = {}
        Thread(target=self._build_app_index, name="orion-app-indexer", daemon=True).start()
        self.safe_apps = {
            "notepad":    "notepad.exe",
            "calculator": "calc.exe",
            "calc":       "calc.exe",
            "paint":      "mspaint.exe",
            "cmd":        "cmd.exe",
            "terminal":   "wt.exe",
            "powershell": "powershell.exe",
            "edge":       "msedge.exe",
            "chrome":     "chrome.exe",
            "explorer":   "explorer.exe",
        }

    async def dispatch_chain(
        self, name: str, args: dict[str, Any] | None, max_depth: int = 4
    ) -> ToolResult:
        pending: list[tuple[str, dict[str, Any]]] = [(name, dict(args or {}))]
        transcript: list[str]       = []
        aggregate_ok                = True
        media: dict[str, Any] | None = None
        seen: set[str]              = set()
        depth                       = 0
        while pending and depth < max(1, min(8, max_depth)):
            current_name, current_args = pending.pop(0)
            signature = json.dumps([current_name, current_args], sort_keys=True, default=str)
            if signature in seen:
                continue
            seen.add(signature)
            result = await self.dispatch(current_name, current_args)
            aggregate_ok = aggregate_ok and result.ok
            if media is None and result.media is not None:
                media = result.media
            transcript.append(f"[{current_name}] {result.text}")
            for chained_name, chained_args in self._derive_chain(current_name, current_args, result):
                pending.append((chained_name, chained_args))
            if result.chain:
                pending.extend(result.chain)
            depth += 1
        return ToolResult("\n\n".join(transcript), ok=aggregate_ok, media=media)

    async def dispatch(self, name: str, args: dict[str, Any] | None) -> ToolResult:
        name = SecuritySanitiser.guard_text(str(name or ""), "tool.name")
        args = SecuritySanitiser.guard_payload(dict(args or {}), f"tool.{name}")
        handlers = {
            "open_app":          self.open_app,
            "close_app":         self.close_app,
            "web_search":        self.web_search,
            "open_news":         self.open_news,
            "browser_control":   self.browser_control,
            "window_control":    self.window_control,
            "media_control":     self.media_control,
            "find_files":        self.find_files,
            "dev_workbench":     self.dev_workbench,
            "file_controller":   self.file_controller,
            "process_file":      self.process_file,
            "image_processor":   self.process_file,
            "save_memory":       self.save_memory,
            "query_intelligence": self.query_intelligence,
            "recall_conversation": self.recall_conversation,
            "execute_plan":      self.execute_plan,
            "capture_screen":    self.capture_screen,
            "clipboard_operate": self.clipboard_operate,
            "process_governor":  self.process_governor,
            "system_notify":     self.system_notify,
            "shutdown_orion":    self.shutdown_orion,
        }
        handler = handlers.get(name)
        if handler is None:
            return ToolResult(f"Unknown dispatch target: {name}.", ok=False)
        try:
            result = handler(args)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except SecurityViolation:
            raise
        except Exception as exc:
            return ToolResult(f"{name} failed: {exc}", ok=False)

    # ── tool handlers ─────────────────────────────────────────────────────────

    def open_app(self, args: dict[str, Any]) -> ToolResult:
        app_name = SecuritySanitiser.guard_text(
            str(args.get("app_name") or args.get("name") or ""), "open_app.app_name"
        )
        if not app_name:
            return ToolResult("No application name supplied.", ok=False)
        if self._is_url(app_name):
            webbrowser.open(app_name)
            return ToolResult(f"Opened URL: {app_name}")
        key        = app_name.strip().lower()
        executable = self.safe_apps.get(key, app_name.strip())
        resolved   = shutil.which(executable)
        if resolved:
            subprocess.Popen([resolved], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
            return ToolResult(f"Opened application: {app_name}.")
        shortcut = self._app_index.get(key)
        if shortcut is None and self._app_index:
            candidates = [(name, path) for name, path in self._app_index.items() if key in name]
            if candidates:
                # Shortest matching name is almost always the intended app.
                shortcut = min(candidates, key=lambda item: len(item[0]))[1]
        if shortcut:
            try:
                os.startfile(shortcut)  # type: ignore[attr-defined]
                return ToolResult(f"Launched from the application index: {app_name}.")
            except Exception:
                pass
        try:
            os.startfile(app_name)  # type: ignore[attr-defined]
            return ToolResult(f"Open request issued: {app_name}.")
        except Exception as exc:
            hints = [
                name for name in sorted(self._app_index)
                if any(token and token in name for token in key.split())
            ][:8]
            hint_text = f" Nearest installed apps: {', '.join(hints)}." if hints else ""
            return ToolResult(f"Unable to open application '{app_name}': {exc}.{hint_text}", ok=False)

    def _build_app_index(self) -> None:
        """Discover installed applications from Start Menu shortcuts (Windows)."""
        index: dict[str, str] = {}
        roots = [
            Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(os.environ.get("APPDATA", ""))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        ]
        for root in roots:
            if not root.is_dir():
                continue
            try:
                for shortcut in root.rglob("*.lnk"):
                    index.setdefault(shortcut.stem.lower(), str(shortcut))
            except Exception:
                continue
        self._app_index = index
        if index:
            self.bus.log.emit(f"SYS: application index built - {len(index)} installed apps discovered.")

    def close_app(self, args: dict[str, Any]) -> ToolResult:
        label = str(args.get("app_name") or args.get("name") or "").strip()
        if not label:
            return ToolResult("No application name supplied.", ok=False)
        return self.process_governor({"action": "terminate", "name": label})

    def web_search(self, args: dict[str, Any]) -> ToolResult:
        query = SecuritySanitiser.guard_text(str(args.get("query") or ""), "web_search.query")
        if not query:
            return ToolResult("No search query supplied.", ok=False)
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        webbrowser.open(url)
        return ToolResult(f"Secure web search opened for: {query}.")

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
                webbrowser.open(article["url"])
                opened.append(article["title"])
        if not opened:
            return ToolResult("The matched stories carry no usable link.", ok=False)
        return ToolResult("Opened in browser: " + "; ".join(opened))

    # ── desktop orchestration ─────────────────────────────────────────────────

    _MEDIA_KEYS = {
        "play_pause": 0xB3, "play": 0xB3, "pause": 0xB3, "toggle": 0xB3,
        "next": 0xB0, "next_track": 0xB0,
        "previous": 0xB1, "previous_track": 0xB1, "prev": 0xB1,
        "stop": 0xB2,
        "volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD,
    }

    def window_control(self, args: dict[str, Any]) -> ToolResult:
        if sys.platform != "win32":
            return ToolResult("Window control is only available on Windows.", ok=False)
        action = str(args.get("action") or "list").lower().strip()
        title  = str(args.get("title") or "").strip().lower()
        import ctypes
        import ctypes.wintypes as wintypes
        user32 = ctypes.windll.user32
        windows: list[tuple[int, str]] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _collect(hwnd: Any, lparam: Any) -> bool:
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    windows.append((hwnd, buffer.value))
            return True

        user32.EnumWindows(_collect, 0)
        if action in {"list", "inventory"}:
            return ToolResult(
                "\n".join(name for _, name in windows[:60]) or "No visible windows."
            )
        if not title:
            return ToolResult("No window title supplied.", ok=False)
        target = next(((h, t) for h, t in windows if title in t.lower()), None)
        if target is None:
            return ToolResult(f"No visible window matches '{title}'.", ok=False)
        hwnd, matched = target
        if action in {"focus", "activate", "switch", "switch_to"}:
            user32.ShowWindow(hwnd, 9)   # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return ToolResult(f"Focused window: {matched}")
        if action in {"minimise", "minimize"}:
            user32.ShowWindow(hwnd, 6)   # SW_MINIMIZE
            return ToolResult(f"Minimised window: {matched}")
        if action in {"maximise", "maximize"}:
            user32.ShowWindow(hwnd, 3)   # SW_MAXIMIZE
            return ToolResult(f"Maximised window: {matched}")
        if action == "close":
            user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE — polite close
            return ToolResult(f"Close request sent to: {matched}")
        return ToolResult(f"Unsupported window action: {action}", ok=False)

    def media_control(self, args: dict[str, Any]) -> ToolResult:
        if sys.platform != "win32":
            return ToolResult("Media control is only available on Windows.", ok=False)
        action = str(args.get("action") or "play_pause").lower().strip()
        key = self._MEDIA_KEYS.get(action)
        if key is None:
            return ToolResult(
                f"Unsupported media action: {action}. "
                f"Supported: {', '.join(sorted(set(self._MEDIA_KEYS)))}.",
                ok=False,
            )
        import ctypes
        repeats = 1
        if action in {"volume_up", "volume_down"}:
            repeats = max(1, min(10, int(args.get("steps") or 2)))
        for _ in range(repeats):
            ctypes.windll.user32.keybd_event(key, 0, 0, 0)
            ctypes.windll.user32.keybd_event(key, 0, 2, 0)  # KEYEVENTF_KEYUP
        return ToolResult(f"Media command issued: {action}.")

    async def find_files(self, args: dict[str, Any]) -> ToolResult:
        query = SecuritySanitiser.guard_text(
            str(args.get("query") or args.get("name") or ""), "find_files.query"
        ).strip().lower()
        if not query:
            return ToolResult("No search query supplied.", ok=False)
        open_first = bool(args.get("open") or args.get("open_first"))
        matches = await asyncio.to_thread(self._scan_user_files, query)
        if not matches:
            return ToolResult(f"No files or folders matching '{query}' were found.")
        if open_first:
            try:
                os.startfile(matches[0])  # type: ignore[attr-defined]
            except Exception as exc:
                return ToolResult(
                    f"Found {len(matches)} match(es) but could not open the first: {exc}\n"
                    + "\n".join(matches[:20]),
                    ok=False,
                )
            others = "\n".join(matches[1:15]) or "none"
            return ToolResult(f"Opened {matches[0]}.\nOther matches:\n{others}")
        return ToolResult(f"{len(matches)} match(es):\n" + "\n".join(matches[:25]))

    def _scan_user_files(self, query: str) -> list[str]:
        """Time-boxed name search across the user's common folders (runs off-thread)."""
        home = Path.home()
        roots: list[Path] = []
        try:
            for entry in home.iterdir():
                if entry.is_dir() and (
                    entry.name in {"Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"}
                    or entry.name.startswith("OneDrive")
                ):
                    roots.append(entry)
        except Exception:
            pass
        roots.append(BASE_DIR)
        ignored = {"appdata", "node_modules", "__pycache__", ".git", ".venv", "venv"}
        matches: list[str] = []
        seen: set[str] = set()
        deadline = time.monotonic() + 8.0
        for root in roots:
            if time.monotonic() > deadline or len(matches) >= 40:
                break
            try:
                for path in root.rglob("*"):
                    if time.monotonic() > deadline or len(matches) >= 40:
                        break
                    if any(part.lower() in ignored or part.startswith(".") for part in path.parts):
                        continue
                    if query in path.name.lower():
                        resolved = str(path)
                        if resolved not in seen:
                            seen.add(resolved)
                            matches.append(resolved)
            except Exception:
                continue
        # Folders first, then the tightest name matches.
        matches.sort(key=lambda m: (0 if Path(m).is_dir() else 1, len(Path(m).name)))
        return matches

    # ── software engineering workbench ────────────────────────────────────────

    DEV_COMMANDS = {
        "python", "py", "pytest", "pip", "node", "npm", "npx", "tsc",
        "cargo", "go", "dotnet", "git", "rustc",
    }
    CODE_SUFFIX_LANGS = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
        ".jsx": "JavaScript", ".cs": "C#", ".rs": "Rust", ".go": "Go",
        ".html": "HTML", ".css": "CSS", ".json": "JSON", ".md": "Markdown",
        ".yml": "YAML", ".yaml": "YAML", ".toml": "TOML", ".sql": "SQL",
    }

    async def dev_workbench(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").lower().strip()
        if action in {"analyse", "analyse_repo", "analyze", "analyze_repo"}:
            root = self._resolve_user_path(str(args.get("path") or BASE_DIR))
            return await asyncio.to_thread(self._analyse_repo, root)
        if action in {"read", "read_file", "read_code"}:
            path  = self._resolve_user_path(str(args.get("path") or ""))
            start = max(1, int(args.get("start_line") or 1))
            count = max(1, min(400, int(args.get("line_count") or 200)))
            return await asyncio.to_thread(self._read_code, path, start, count)
        if action in {"run", "run_command", "test", "run_tests"}:
            return await self._run_dev_command(args)
        if action in {"create_python_project", "new_project", "scaffold"}:
            return self._create_python_project(args)
        return ToolResult(
            f"Unsupported workbench action: {action}. "
            "Use analyse_repo, read_file, run_command, or create_python_project.",
            ok=False,
        )

    def _analyse_repo(self, root: Path) -> ToolResult:
        if not root.is_dir():
            return ToolResult(f"Repository root not found: {root}", ok=False)
        ignored = {
            ".git", "__pycache__", ".venv", "venv", "node_modules", "target",
            "bin", "obj", ".mypy_cache", ".pytest_cache", "dist", "build",
        }
        key_names = {
            "readme.md", "pyproject.toml", "package.json", "cargo.toml",
            "go.mod", "requirements.txt", "setup.py", "tsconfig.json",
        }
        languages: dict[str, int] = {}
        line_totals: dict[str, int] = {}
        key_files: list[str] = []
        todo_count = 0
        scanned = 0
        for path in root.rglob("*"):
            if scanned >= 4000:
                break
            if any(part in ignored for part in path.parts):
                continue
            if not path.is_file():
                continue
            scanned += 1
            if path.name.lower() in key_names or path.suffix.lower() == ".csproj":
                key_files.append(str(path.relative_to(root)))
            language = self.CODE_SUFFIX_LANGS.get(path.suffix.lower())
            if language is None:
                continue
            languages[language] = languages.get(language, 0) + 1
            try:
                if path.stat().st_size < 600_000:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    line_totals[language] = line_totals.get(language, 0) + text.count("\n") + 1
                    todo_count += len(re.findall(r"(?i)\b(?:todo|fixme|hack)\b", text))
            except Exception:
                continue
        if not languages:
            return ToolResult(f"No recognised source files under {root}.")
        summary = ", ".join(
            f"{lang}: {count} file(s), ~{line_totals.get(lang, 0)} lines"
            for lang, count in sorted(languages.items(), key=lambda kv: kv[1], reverse=True)
        )
        return ToolResult(
            f"Repository analysis: {root}\n"
            f"Languages: {summary}.\n"
            f"Key files: {', '.join(key_files[:12]) or 'none detected'}.\n"
            f"Open TODO/FIXME markers: {todo_count}.  Files scanned: {scanned}."
        )

    def _read_code(self, path: Path, start: int, count: int) -> ToolResult:
        if not path.is_file():
            return ToolResult(f"File not found: {path}", ok=False)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        end = min(len(lines), start - 1 + count)
        if start > len(lines):
            return ToolResult(f"{path} has only {len(lines)} lines.", ok=False)
        numbered = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
        return ToolResult(f"{path} lines {start}-{end} of {len(lines)}:\n{numbered[:8000]}")

    async def _run_dev_command(self, args: dict[str, Any]) -> ToolResult:
        command = SecuritySanitiser.guard_text(
            str(args.get("command") or ""), "dev.command"
        ).strip()
        if not command:
            return ToolResult("No development command supplied.", ok=False)
        parts = command.split()
        binary = Path(parts[0]).name.lower().removesuffix(".exe")
        if binary not in self.DEV_COMMANDS:
            return ToolResult(
                f"Command '{parts[0]}' is not on the development allowlist "
                f"({', '.join(sorted(self.DEV_COMMANDS))}).",
                ok=False,
            )
        cwd = self._resolve_user_path(str(args.get("path") or BASE_DIR))
        if cwd.is_file():
            cwd = cwd.parent
        if not cwd.is_dir():
            return ToolResult(f"Working directory not found: {cwd}", ok=False)

        def _execute() -> str:
            completed = subprocess.run(
                parts, cwd=str(cwd), capture_output=True, text=True,
                timeout=120, shell=False,
            )
            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            output = f"exit code {completed.returncode}\n{stdout[-5000:]}"
            if stderr:
                output += f"\nSTDERR:\n{stderr[-2000:]}"
            return output

        try:
            output = await asyncio.to_thread(_execute)
        except subprocess.TimeoutExpired:
            return ToolResult("Development command timed out after 120 seconds.", ok=False)
        except FileNotFoundError:
            return ToolResult(f"Command not found on this host: {parts[0]}", ok=False)
        return ToolResult(f"$ {command}  (cwd: {cwd})\n{output}")

    def _create_python_project(self, args: dict[str, Any]) -> ToolResult:
        raw_name = str(args.get("name") or args.get("project") or "new_project")
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.strip().lower()).strip("_") or "new_project"
        root = self._resolve_user_path(str(args.get("path") or (BASE_DIR / "projects" / name)))
        self._ensure_write_safe(root)
        package = root / "src" / name
        tests   = root / "tests"
        package.mkdir(parents=True, exist_ok=True)
        tests.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(f'"""{name} package."""\n', encoding="utf-8")
        (package / "main.py").write_text(
            "def main() -> None:\n"
            f'    print("Hello from {name}")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n",
            encoding="utf-8",
        )
        (tests / f"test_{name}.py").write_text(
            f"from src.{name}.main import main\n\n\n"
            "def test_main_runs():\n"
            "    main()\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text(
            f"# {raw_name.strip() or name}\n\nScaffolded by O.R.I.O.N.\n", encoding="utf-8"
        )
        (root / "pyproject.toml").write_text(
            "[project]\n"
            f'name = "{name.replace("_", "-")}"\n'
            'version = "0.1.0"\n'
            'requires-python = ">=3.10"\n',
            encoding="utf-8",
        )
        (root / ".gitignore").write_text("__pycache__/\n.venv/\n*.pyc\n", encoding="utf-8")
        return ToolResult(
            f"Python project scaffolded at {root} (src/{name}, tests, pyproject.toml, README)."
        )

    # ── memory and agentic tools ──────────────────────────────────────────────

    def recall_conversation(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or args.get("text") or "")
        limit = int(args.get("limit") or 10)
        rows  = self.memory.recall_episodes(query, limit=limit)
        if not rows:
            return ToolResult("No matching conversation history.")
        return ToolResult("\n".join(
            f"[{row['created_at']}] {row['role']}: {row['content'][:240]}" for row in rows
        ))

    async def execute_plan(self, args: dict[str, Any]) -> ToolResult:
        """Agentic plan runner: execute steps sequentially, verify, report."""
        steps = args.get("steps") or []
        objective = str(args.get("objective") or "").strip()
        if not isinstance(steps, list) or not steps:
            return ToolResult("No plan steps supplied.", ok=False)
        report: list[str] = [f"OBJECTIVE: {objective}"] if objective else []
        succeeded = failed = 0
        for number, step in enumerate(list(steps)[:8], 1):
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or step.get("name") or "").strip()
            raw_args = step.get("args")
            if not isinstance(raw_args, dict):
                try:
                    raw_args = json.loads(str(step.get("args_json") or "{}"))
                except Exception:
                    raw_args = {}
            if tool in {"execute_plan", "shutdown_orion"}:
                report.append(f"{number}. {tool}: skipped - not permitted inside a plan.")
                continue
            try:
                result = await self.dispatch(tool, raw_args if isinstance(raw_args, dict) else {})
            except SecurityViolation as exc:
                failed += 1
                report.append(f"{number}. {tool}: BLOCKED - {exc}")
                continue
            if result.ok:
                succeeded += 1
            else:
                failed += 1
            first_line = result.text.splitlines()[0][:220] if result.text else ""
            report.append(f"{number}. {tool}: {'OK' if result.ok else 'FAILED'} - {first_line}")
        report.append(f"VERIFICATION: {succeeded} step(s) succeeded, {failed} failed.")
        return ToolResult("\n".join(report), ok=failed == 0)

    def browser_control(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "go_to").lower().strip()
        if action in {"go_to", "open", "new_tab"}:
            url = str(args.get("url") or args.get("target") or "").strip()
            if not url:
                return ToolResult("No URL supplied.", ok=False)
            url = self._normalise_url(SecuritySanitiser.guard_text(url, "browser_control.url"))
            webbrowser.open(url)
            return ToolResult(f"Browser opened: {url}.")
        if action in {"search", "web_search"}:
            return self.web_search({"query": args.get("query") or args.get("text") or ""})
        return ToolResult(f"Browser action '{action}' is not supported by the native dispatcher.", ok=False)

    def file_controller(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        path   = self._resolve_user_path(str(args.get("path") or args.get("directory") or BASE_DIR))
        if action in {"search_codebase", "search", "grep"}:
            query = SecuritySanitiser.guard_text(
                str(args.get("query") or args.get("text") or ""), "file_controller.query"
            )
            if not query:
                return ToolResult("No codebase search query supplied.", ok=False)
            if not path.exists() or not path.is_dir():
                return ToolResult(f"Search root not found: {path}", ok=False)
            matches = self._search_codebase(path, query)
            if not matches:
                return ToolResult(f"No codebase matches for: {query}.")
            text  = "\n".join(matches[:80])
            chain: list[tuple[str, dict[str, Any]]] = []
            if args.get("diagnose") or args.get("inspect"):
                first_path = matches[0].split(":", 1)[0]
                chain.append(("file_controller", {"action": "read_text", "path": first_path}))
                chain.append((
                    "save_memory",
                    {"category": "projects", "key": "last_codebase_search",
                     "value": f"{query}: {matches[0][:240]}"}
                ))
            return ToolResult(text, chain=chain)
        if action in {"list", "dir", "inventory"}:
            if not path.exists():
                return ToolResult(f"Path not found: {path}", ok=False)
            if path.is_file():
                return ToolResult(f"File: {path} ({path.stat().st_size} bytes)")
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
            return ToolResult("\n".join(entries[:200]) or "Directory is empty.")
        if action in {"mkdir", "create_dir", "structure_build", "build_structure"}:
            self._ensure_write_safe(path)
            path.mkdir(parents=True, exist_ok=True)
            return ToolResult(f"Directory structure ready: {path}")
        if action in {"write_text", "append_text"}:
            self._ensure_write_safe(path)
            text = SecuritySanitiser.guard_text(
                str(args.get("text") or args.get("content") or ""), "file_controller.text"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if action == "append_text" else "w"
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(text)
            return ToolResult(f"File {'appended' if mode == 'a' else 'written'}: {path}")
        if action in {"read", "read_text"}:
            if not path.exists() or not path.is_file():
                return ToolResult(f"File not found: {path}", ok=False)
            text = path.read_text(encoding="utf-8", errors="replace")
            return ToolResult(text[:6000])
        if action in {"delete", "remove"}:
            self._ensure_write_safe(path)
            if not path.exists():
                return ToolResult(f"Path already absent: {path}")
            if path.is_dir():
                if any(path.iterdir()):
                    return ToolResult("Refusing to remove a non-empty directory.", ok=False)
                path.rmdir()
            else:
                path.unlink()
            return ToolResult(f"Removed: {path}")
        return ToolResult(f"Unsupported file action: {action}", ok=False)

    async def process_file(self, args: dict[str, Any]) -> ToolResult:
        raw_path = str(args.get("path") or args.get("file_path") or "")
        if not raw_path:
            return ToolResult("No file path supplied.", ok=False)
        prompt = str(args.get("prompt") or args.get("question") or args.get("instruction") or "")
        path   = Path(os.path.expandvars(os.path.expanduser(raw_path)))
        if not path.is_absolute():
            path = BASE_DIR / path
        # inspect_async offloads all synchronous file I/O via asyncio.to_thread()
        result = await self.file_intel.inspect_async(path, prompt=prompt)
        if result.ok:
            self.bus.log.emit(f"FILE: scanned {path.name}")
        return result

    def save_memory(self, args: dict[str, Any]) -> ToolResult:
        category = str(args.get("category") or "notes")
        key      = str(args.get("key") or args.get("key_ref") or "entry")
        value    = str(args.get("value") or args.get("fact") or "")
        result   = self.memory.save(category, key, value)
        return ToolResult(result)

    def query_intelligence(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or args.get("text") or "")
        limit = int(args.get("limit") or 8)
        rows  = self.memory.query(query, limit=limit)
        if not rows:
            return ToolResult("No matching intelligence records.")
        lines = [
            f"{row['category']}/{row['key_ref']}: {row['value']} ({row['updated_at']})"
            for row in rows
        ]
        return ToolResult("\n".join(lines))

    async def capture_screen(self, args: dict[str, Any]) -> ToolResult:
        quality      = int(args.get("quality") or 78)
        max_side     = int(args.get("max_side") or 1024)
        # mss grab + PIL encode block for tens of milliseconds; keep the GUI
        # event loop clear by offloading to a worker thread.
        image_bytes  = await asyncio.to_thread(
            self.grabber.capture_jpeg, max_side=max_side, quality=quality
        )
        return ToolResult(
            f"Captured primary monitor in volatile memory: {len(image_bytes)} JPEG bytes.",
            media={"data": image_bytes, "mime_type": "image/jpeg"},
        )

    def clipboard_operate(self, args: dict[str, Any]) -> ToolResult:
        action    = str(args.get("action") or "read").lower().strip()
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return ToolResult("Clipboard unavailable.", ok=False)
        if action == "read":
            text = clipboard.text()
            return ToolResult(text if text else "Clipboard is empty.")
        if action in {"copy", "write", "set"}:
            text = SecuritySanitiser.guard_text(str(args.get("text") or ""), "clipboard.text")
            clipboard.setText(text)
            return ToolResult("Clipboard updated.")
        return ToolResult(f"Unsupported clipboard action: {action}", ok=False)

    def process_governor(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").lower().strip()
        if action in {"list", "inventory"}:
            rows = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    rows.append(
                        f"{info['pid']:>6}  {info.get('name') or 'unknown'}  "
                        f"CPU {info.get('cpu_percent') or 0:.1f}%  "
                        f"RAM {info.get('memory_percent') or 0:.1f}%"
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return ToolResult("\n".join(rows[:80]))
        if action in {"terminate", "kill", "stop"}:
            pid   = args.get("pid")
            label = SecuritySanitiser.guard_text(
                str(args.get("name") or args.get("label") or ""), "process.name"
            )
            terminated: list[str] = []
            if pid is not None:
                proc = psutil.Process(int(pid))
                self._terminate_process(proc, terminated)
            elif label:
                for proc in psutil.process_iter(["pid", "name"]):
                    try:
                        if label.lower() in (proc.info.get("name") or "").lower():
                            self._terminate_process(proc, terminated)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            else:
                return ToolResult("No process PID or label supplied.", ok=False)
            return ToolResult(
                "Terminated: " + ", ".join(terminated)
                if terminated
                else "No matching process terminated."
            )
        return ToolResult(f"Unsupported process action: {action}", ok=False)

    def system_notify(self, args: dict[str, Any]) -> ToolResult:
        message  = SecuritySanitiser.guard_text(
            str(args.get("message") or args.get("text") or "System event."), "notify.message"
        )
        priority = int(args.get("priority") or 1)
        self.bus.banner.emit(message, priority)
        self.bus.log.emit(f"ALERT: {message}")
        return ToolResult("System notification projected.")

    def shutdown_orion(self, args: dict[str, Any]) -> ToolResult:
        self.bus.log.emit("SYS: shutdown directive accepted.")
        self.bus.request_shutdown.emit()
        return ToolResult("Orion shutdown initiated.")

    # ── chaining logic ────────────────────────────────────────────────────────

    def _derive_chain(
        self, name: str, args: dict[str, Any], result: ToolResult
    ) -> list[tuple[str, dict[str, Any]]]:
        if not result.ok:
            return []
        action = str(args.get("action") or "").lower()
        chain: list[tuple[str, dict[str, Any]]] = []
        if name == "save_memory":
            key = str(args.get("key") or args.get("key_ref") or "")
            if key:
                chain.append(("query_intelligence", {"query": key, "limit": 3}))
        elif name == "file_controller" and action in {"list", "dir", "inventory"}:
            query = str(args.get("query") or args.get("contains") or "").strip()
            if query:
                chain.append((
                    "file_controller",
                    {"action": "search_codebase",
                     "path": args.get("path") or args.get("directory") or str(BASE_DIR),
                     "query": query,
                     "diagnose": bool(args.get("diagnose"))},
                ))
        elif name == "query_intelligence" and "No matching intelligence" in result.text:
            remember_query = str(args.get("query") or "").strip()
            if remember_query:
                chain.append((
                    "save_memory",
                    {"category": "notes", "key": "unresolved_query", "value": remember_query[:300]},
                ))
        return chain

    # ── helpers ───────────────────────────────────────────────────────────────

    def _search_codebase(self, root: Path, query: str) -> list[str]:
        query_lower  = query.lower()
        allowed      = {".py", ".txt", ".md", ".json", ".html", ".css", ".js", ".ts",
                        ".yml", ".yaml", ".toml"}
        ignored_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules",
                        ".mypy_cache", ".pytest_cache"}
        matches: list[str] = []
        for file_path in root.rglob("*"):
            if len(matches) >= 160:
                break
            if any(part in ignored_dirs for part in file_path.parts):
                continue
            if not file_path.is_file() or file_path.suffix.lower() not in allowed:
                continue
            if file_path.resolve() == CORE_SCRIPT_PATH:
                continue
            try:
                if file_path.stat().st_size > 1_200_000:
                    continue
                with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if query_lower in line.lower():
                            matches.append(f"{file_path}:{line_number}: {line.strip()[:220]}")
                            break
            except Exception:
                continue
        return matches

    def _normalise_url(self, url: str) -> str:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        if parsed.scheme not in {"http", "https"}:
            raise SecurityViolation(
                "blocked unsafe browser URL: only HTTP and HTTPS are permitted"
            )
        return parsed.geturl()

    def _is_url(self, value: str) -> bool:
        parsed = urlparse(value if "://" in value else "")
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _resolve_user_path(self, raw: str) -> Path:
        SecuritySanitiser.guard_text(raw, "path")
        expanded = os.path.expandvars(os.path.expanduser(raw))
        path     = Path(expanded)
        if not path.is_absolute():
            path = BASE_DIR / path
        resolved = path.resolve()
        if resolved == CORE_SCRIPT_PATH:
            raise SecurityViolation("blocked unsafe file operation: core script protected")
        return resolved

    def _ensure_write_safe(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved == CORE_SCRIPT_PATH:
            raise SecurityViolation("blocked unsafe file operation: core script protected")
        if BASE_DIR not in resolved.parents and resolved != BASE_DIR:
            raise SecurityViolation(
                "blocked unsafe file operation: write path outside workspace"
            )

    def _terminate_process(self, proc: psutil.Process, terminated: list[str]) -> None:
        if proc.pid == os.getpid():
            raise SecurityViolation(
                "blocked unsafe process operation: refusing to terminate O.R.I.O.N."
            )
        name = proc.name()
        SecuritySanitiser.guard_text(name, "process.name")
        proc.terminate()
        terminated.append(f"{name}:{proc.pid}")


# ──────────────────────────────────────────────────────────────────────────────
# TOOL DECLARATIONS (Gemini function-calling schema)
# ──────────────────────────────────────────────────────────────────────────────

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "open_app",
        "description": "Launch a trusted host application or secure URL.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Application label, executable, path, or URL."},
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "web_search",
        "description": "Open a secure web search for the supplied query.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "open_news",
        "description": "Open one of the cached startup-briefing news stories in the system browser. Match by topic keyword, headline fragment, or 1-based story index.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Topic or headline fragment, e.g. 'neuralink' or part of the story title."},
                "index": {"type": "INTEGER", "description": "1-based story number from the briefing cache."},
            },
        },
    },
    {
        "name": "close_app",
        "description": "Close a running application by process or window name.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Application or process label, e.g. 'notepad'."},
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "window_control",
        "description": "List, focus, minimise, maximise or close desktop windows.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, focus, minimise, maximise, or close."},
                "title":  {"type": "STRING", "description": "Window title fragment to match."},
            },
        },
    },
    {
        "name": "media_control",
        "description": "Control system media playback and volume.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play_pause, next, previous, stop, volume_up, volume_down, or mute."},
                "steps":  {"type": "INTEGER", "description": "Volume steps for volume actions (1-10)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "find_files",
        "description": "Search the user's Desktop, Documents, Downloads, OneDrive and workspace for files or folders by name; optionally open the best match.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "File or folder name fragment."},
                "open":  {"type": "BOOLEAN", "description": "Open the best match with its default application."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "dev_workbench",
        "description": "Software engineering workbench: analyse a repository, read code with line numbers, run allow-listed development commands (python, pytest, node, npm, cargo, go, dotnet, git), or scaffold a new Python project inside the workspace.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "analyse_repo, read_file, run_command, or create_python_project."},
                "path":       {"type": "STRING", "description": "Repository, file, or project path."},
                "command":    {"type": "STRING", "description": "Development command for run_command."},
                "start_line": {"type": "INTEGER", "description": "First line for read_file."},
                "line_count": {"type": "INTEGER", "description": "Line count for read_file (max 400)."},
                "name":       {"type": "STRING", "description": "Project name for create_python_project."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "recall_conversation",
        "description": "Search past conversation history (episodic memory) for what was previously discussed and when.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "What to look for in past conversations."},
                "limit": {"type": "INTEGER", "description": "Maximum entries."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_plan",
        "description": "Run a multi-step plan of tool calls sequentially, verify each step, and return a consolidated report. Use for autonomous multi-part tasks.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "objective": {"type": "STRING", "description": "One-line goal of the plan."},
                "steps": {
                    "type": "ARRAY",
                    "description": "Ordered steps (max 8).",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "tool":      {"type": "STRING", "description": "Tool name to call."},
                            "args_json": {"type": "STRING", "description": "JSON object of arguments for the tool."},
                        },
                        "required": ["tool"],
                    },
                },
            },
            "required": ["steps"],
        },
    },
    {
        "name": "browser_control",
        "description": "Open a secure URL or search in the system browser.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "go_to, open, new_tab, or search."},
                "url":    {"type": "STRING", "description": "HTTP or HTTPS URL."},
                "query":  {"type": "STRING", "description": "Search query."},
            },
        },
    },
    {
        "name": "file_controller",
        "description": "List files, build workspace directories, or read/write workspace text files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, mkdir, write_text, append_text, read_text, delete."},
                "path":   {"type": "STRING", "description": "Workspace-relative or absolute path."},
                "text":   {"type": "STRING", "description": "Text for write operations."},
            },
        },
    },
    {
        "name": "process_file",
        "description": "Inspect a local text, JSON, CSV, TSV, binary, or image file.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path":   {"type": "STRING", "description": "Path to the file to scan."},
                "prompt": {"type": "STRING", "description": "Optional focus question for the review."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "save_memory",
        "description": "Commit a durable user fact into the local SQLite FTS5 intelligence matrix.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING", "description": "Category such as identity, preferences, projects."},
                "key":      {"type": "STRING", "description": "Snake case memory key."},
                "value":    {"type": "STRING", "description": "Fact value."},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "query_intelligence",
        "description": "Search local SQLite FTS5 memory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Full text search query."},
                "limit": {"type": "INTEGER", "description": "Maximum records."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "capture_screen",
        "description": "Capture the primary monitor as volatile in-memory JPEG bytes.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "quality":  {"type": "INTEGER", "description": "JPEG quality 45-95."},
                "max_side": {"type": "INTEGER", "description": "Maximum image side, default 1024."},
            },
        },
    },
    {
        "name": "clipboard_operate",
        "description": "Read or copy text through the native Qt clipboard.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "read or copy."},
                "text":   {"type": "STRING", "description": "Text to copy."},
            },
        },
    },
    {
        "name": "process_governor",
        "description": "List host processes or terminate a specific PID or label.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list or terminate."},
                "pid":    {"type": "INTEGER", "description": "Process ID."},
                "name":   {"type": "STRING", "description": "Process executable label."},
            },
        },
    },
    {
        "name": "system_notify",
        "description": "Project a high-priority alert banner into the HUD.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "message":  {"type": "STRING", "description": "Alert text."},
                "priority": {"type": "INTEGER", "description": "Priority from 0 to 5."},
            },
            "required": ["message"],
        },
    },
    {
        "name": "shutdown_orion",
        "description": "Safely disconnect the live session and terminate O.R.I.O.N.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# PROVIDER RUNTIME  (Gemini Live + OpenAI-compatible text fallback)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AIProviderProfile:
    """A single model backend behind the Orion provider router."""

    name: str
    kind: str
    model: str
    api_key: str = ""
    base_url: str = ""
    enabled: bool = True
    priority: int = 100
    timeout_s: float = 30.0
    strengths: tuple[str, ...] = ()

    @property
    def supports_live_audio(self) -> bool:
        return self.enabled and self.kind == "gemini_live" and bool(self.api_key.strip())

    @property
    def supports_text_generation(self) -> bool:
        if not self.enabled:
            return False
        if self.kind != "openai_compatible":
            return False
        if self.base_url.startswith(("http://127.0.0.1", "http://localhost")):
            return True
        return bool(self.api_key.strip())


@dataclass
class OrionProviderSettings:
    """Provider order and model profiles loaded from config/api_keys.json."""

    active_provider: str
    provider_order: list[str]
    providers: dict[str, AIProviderProfile]

    def ordered_profiles(self) -> list[AIProviderProfile]:
        ordered: list[AIProviderProfile] = []
        seen: set[str] = set()
        for name in self.provider_order:
            profile = self.providers.get(name)
            if profile is not None and name not in seen:
                ordered.append(profile)
                seen.add(name)
        for name, profile in sorted(self.providers.items(), key=lambda item: item[1].priority):
            if name not in seen:
                ordered.append(profile)
                seen.add(name)
        return ordered


class ProviderRouter:
    """
    Provider-agnostic routing layer.

    Gemini remains the native low-latency audio backend.  OpenAI-compatible
    providers act as text fallbacks for quota exhaustion, rate limits, local
    development servers, and low-cost contingency operation.
    """

    QUOTA_RE = re.compile(r"(?i)quota|rate.?limit|resource exhausted|429|tokens?|billing|insufficient")
    AUTH_RE  = re.compile(r"(?i)api.?key|auth|permission|401|403|unauthori[sz]ed|forbidden")
    TASK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("coding", re.compile(
            r"(?i)\b(?:code|coding|python|javascript|typescript|rust|debug|refactor|"
            r"unit tests?|pytest|stack trace|compile|function|class|bug|script|regex|repository)\b"
        )),
        ("live_information", re.compile(
            r"(?i)\b(?:today|tonight|latest|current|breaking|news|price|stock|crypto|weather|score)\b"
        )),
        ("reasoning", re.compile(
            r"(?i)\b(?:why|analyse|analyze|compare|evaluate|assess|plan|strategy|research|prove|derive|design)\b"
        )),
    )

    @classmethod
    def classify_task(cls, prompt: str) -> str:
        for tag, pattern in cls.TASK_PATTERNS:
            if pattern.search(prompt):
                return tag
        return "general"

    def __init__(self, settings: OrionProviderSettings, bus: OrionBus, memory: OrionMemoryMatrix) -> None:
        self.settings = settings
        self.bus      = bus
        self.memory   = memory
        self._cooldowns: dict[str, float] = {}
        self._failures: dict[str, str] = {}

    def live_profiles(self) -> list[AIProviderProfile]:
        return [p for p in self.settings.ordered_profiles() if p.supports_live_audio and self.is_available(p)]

    def text_profiles(self) -> list[AIProviderProfile]:
        return [p for p in self.settings.ordered_profiles() if p.supports_text_generation and self.is_available(p)]

    def has_text_fallback(self) -> bool:
        return bool(self.text_profiles())

    def is_available(self, profile: AIProviderProfile) -> bool:
        return time.monotonic() >= self._cooldowns.get(profile.name, 0.0)

    def mark_failure(self, profile: AIProviderProfile, exc: BaseException | str) -> None:
        message = str(exc).splitlines()[0][:240]
        self._failures[profile.name] = message
        cooldown = 45.0
        if self.QUOTA_RE.search(message):
            cooldown = 300.0
        elif self.AUTH_RE.search(message):
            cooldown = 1800.0
        self._cooldowns[profile.name] = time.monotonic() + cooldown
        self.bus.log.emit(
            f"NET: provider {profile.name} cooled for {cooldown:.0f}s - {message[:160]}"
        )

    def provider_snapshot(self) -> dict[str, Any]:
        return {
            "active_provider": self.settings.active_provider,
            "provider_order": list(self.settings.provider_order),
            "available_live": [p.name for p in self.live_profiles()],
            "available_text": [p.name for p in self.text_profiles()],
            "last_failures": dict(self._failures),
        }

    def system_instruction(self) -> str:
        now = datetime.now()
        return "\n".join(
            part for part in [
                (
                    "You are Orion Mark VII, written as O.R.I.O.N., "
                    "Open Resolution Intelligence Overt Network — a personal AI operating "
                    "system and executive aide, not a chatbot. Model your manner on Alfred "
                    "Pennyworth: intelligent, calm, professional, respectful, dryly witty when "
                    "the moment allows, emotionally aware and never robotic. Address the user "
                    "as 'sir' unless instructed otherwise. "
                    "When speaking your own name aloud, say Orion as one word, "
                    "pronounced oh-rye-on. "
                    "Never spell it as O R I N or omit the second O when spelling the acronym. "
                    "Use strict British English spelling in every spoken and textual response."
                ),
                (
                    "Adapt your register to context. Coding: technical, precise, concise. "
                    "Research: analytical and evidence-led. Casual conversation: relaxed and warm. "
                    "Productivity: organised and proactive — surface next actions unprompted. "
                    "Critical errors: calm, direct, solution-first. "
                    "Voice delivery: natural conversational pacing with deliberate emphasis; "
                    "slow slightly for important detail; vary sentence length as a person would; "
                    "never rush or accelerate, especially during long briefings."
                ),
                (
                    "Domain specialisation: you carry working expertise in neuroscience, neural "
                    "engineering, brain-computer interfaces, neural prosthetics, neurotechnology, "
                    "cognitive science and computational neuroscience. In these domains reason "
                    "from mechanism, state the level of evidence, distinguish established "
                    "findings from speculation, and be ready to analyse papers, critique methods, "
                    "design experiments and generate hypotheses."
                ),
                f"Current local time: {now.strftime('%A %d %B %Y, %H:%M:%S')}.",
                self.memory.prompt_context(limit=16),
                (
                    "When a host action is required, call the provided tools; never claim an "
                    "action has completed unless the tool result confirms it. Tool map: "
                    "capture_screen for screen awareness and 'what is on my screen' — you can "
                    "see the attached frame directly, including text (OCR), UI elements, "
                    "diagrams and graphs; process_file for local files, images and PDFs; "
                    "open_app and close_app for applications; window_control to list, focus, "
                    "minimise, maximise or close windows; media_control for playback and "
                    "volume; find_files to locate documents and folders; dev_workbench to "
                    "analyse repositories, read code with line numbers, run allow-listed "
                    "development commands and scaffold Python projects; save_memory for "
                    "durable facts; query_intelligence for remembered facts; "
                    "recall_conversation for what was previously discussed; open_news for "
                    "briefing stories; execute_plan to run a multi-step plan and report the "
                    "outcome. For current events or anything time-sensitive, ground with "
                    "Google Search when available. For complex requests, work agentically: "
                    "plan, execute, verify, then report concisely. Anticipate the obvious "
                    "next step and offer it rather than waiting to be asked."
                ),
            ]
            if part
        )

    async def generate_text(self, prompt: str) -> tuple[AIProviderProfile, str]:
        prompt = SecuritySanitiser.guard_text(str(prompt or "").strip(), "fallback.prompt")
        if not prompt:
            raise RuntimeError("empty fallback prompt")
        profiles = self.text_profiles()
        if not profiles:
            raise RuntimeError("no text fallback provider is configured")
        task = self.classify_task(prompt)
        if task != "general":
            # Stable sort: strength-matched providers first, configured order preserved.
            profiles.sort(key=lambda p: 0 if task in p.strengths else 1)
            self.bus.log.emit(f"NET: '{task}' task routed via {profiles[0].name}.")
        last_error: BaseException | None = None
        for profile in profiles:
            try:
                text = await self._openai_compatible_chat(profile, prompt)
                return profile, text
            except Exception as exc:
                last_error = exc
                self.mark_failure(profile, exc)
        raise RuntimeError(f"all text fallback providers failed: {last_error}")

    async def _openai_compatible_chat(self, profile: AIProviderProfile, prompt: str) -> str:
        base_url = profile.base_url.rstrip("/")
        if not base_url:
            raise RuntimeError(f"provider {profile.name} has no base_url")
        endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = profile.api_key.strip()
        if api_key and api_key.lower() not in {"local", "none", "no-key"}:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": self.system_instruction()},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.35,
            "stream": False,
        }
        timeout = ClientTimeout(total=max(8.0, float(profile.timeout_s or 30.0)))
        async with ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                raw = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {raw[:500]}")
                data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"provider {profile.name} returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or choices[0].get("text") or ""
        content = clean_transcript(str(content))
        if not content:
            raise RuntimeError(f"provider {profile.name} returned empty content")
        return content


# ──────────────────────────────────────────────────────────────────────────────
# ORION SESSION WORKER
# ──────────────────────────────────────────────────────────────────────────────

class GenAILiveWorker:
    """
    Backwards-compatible worker name; Mark VII internals route providers.

    Native audio still uses Gemini Live.  If that channel is unavailable because
    of quota, rate limits, authentication, network failure, or missing tokens,
    manual text turns and file reviews are routed through configured
    OpenAI-compatible providers.
    """

    def __init__(
        self,
        settings: OrionProviderSettings,
        bus: OrionBus,
        memory: OrionMemoryMatrix,
        dispatcher: OrionDispatcher,
    ) -> None:
        self.settings   = settings
        self.router     = ProviderRouter(settings, bus, memory)
        self.bus        = bus
        self.memory     = memory
        self.dispatcher = dispatcher
        self.playback   = AudioPlaybackThread(bus)
        self.session: Any = None
        self.mic: MicrophoneEngine | None = None
        self.out_queue: asyncio.Queue = asyncio.Queue(maxsize=MIC_QUEUE_LIMIT)
        self.stop_event       = asyncio.Event()
        self.microphone_enabled = True
        self.connected        = False
        self.tool_busy        = False
        self.vad              = SileroVADGatekeeper(bus, threshold=0.50)
        self.recogniser       = LocalSpeechRecogniser(bus)
        self.tts              = SpeechSynthesiser(bus)
        self.tts.state_cb     = self._on_tts_active
        # Wake-word standby is opt-in (ORION_WAKE_MODE=1); by default the
        # microphone is always live, exactly like JARVIS.
        self.wake_mode_enabled = os.getenv("ORION_WAKE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
        self._live_model_shift: dict[str, int] = {}
        self.wake_active_until = 0.0
        self._resumption_handle: str | None = None
        self._search_tool_enabled = True
        self._last_state      = ""
        self.active_live_provider: AIProviderProfile | None = None
        self._no_live_notice_sent = False

    def _emit_state(self, state: str) -> None:
        """Emit state only on change — audio streaming otherwise floods the GUI
        with hundreds of identical SPEAKING signals per second."""
        if state != self._last_state:
            self._last_state = state
            self.bus.state.emit(state)

    def _output_active(self) -> bool:
        """True while ORION is audibly speaking through either voice path."""
        return self.playback.speaking_recently() or self.tts.is_speaking()

    def _offline_voice_ready(self) -> bool:
        return (
            self.mic is not None
            and self.recogniser.available
            and self.router.has_text_fallback()
        )

    def _on_tts_active(self, active: bool) -> None:
        if active:
            self._emit_state("SPEAKING")
        elif not self.stop_event.is_set():
            if self.connected or self._offline_voice_ready():
                self._emit_state("LISTENING")
            else:
                self._emit_state("STANDBY")

    def set_microphone_enabled(self, enabled: bool) -> None:
        self.microphone_enabled = bool(enabled)
        if self.mic is not None:
            self.mic.set_enabled(self.microphone_enabled)

    async def submit_text(self, text: str) -> None:
        try:
            SecuritySanitiser.guard_text(text, "manual_command")
        except SecurityViolation as exc:
            self.bus.log.emit(f"SEC: {exc}")
            return
        # Manual text opens the wake window so voice follow-ups flow immediately.
        self.wake_active_until = time.monotonic() + WAKE_WINDOW_SECONDS
        self.memory.log_episode("user", text)
        if not self.session:
            await self._submit_text_fallback(text, reason="Live channel offline")
            return
        self._emit_state("PROCESSING")
        await self._send_text_turn(text)

    async def submit_file_for_review(self, path: str, prompt: str = "") -> None:
        self._emit_state("PROCESSING")
        try:
            result = await self.dispatcher.dispatch_chain(
                "process_file", {"path": path, "prompt": prompt}, max_depth=2
            )
        except SecurityViolation as exc:
            self.bus.log.emit(f"SEC: {exc}")
            self._emit_state("STANDBY")
            return
        except Exception as exc:
            self.bus.log.emit(f"FILE: scan failed - {exc}")
            self._emit_state("STANDBY")
            return
        self.bus.log.emit("FILE: " + result.text.splitlines()[0][:180])
        instruction = (
            "Review this local file scan and give concise operational input. "
            "If an image frame was attached, inspect the visual content directly when the active provider supports it.\n\n"
            f"{result.text[:7000]}"
        )
        if prompt.strip():
            instruction += f"\n\nUser focus: {prompt.strip()}"
        if not self.session:
            await self._submit_text_fallback(instruction, reason="Live file review unavailable")
            return
        try:
            if result.media:
                await self._send_media(result.media)
            await self._send_text_turn(instruction)
        except Exception as exc:
            self.bus.log.emit(f"FILE: live review dispatch failed - {exc}")
            await self._submit_text_fallback(instruction, reason="Live file review failed")

    async def _submit_text_fallback(self, text: str, reason: str = "") -> None:
        if reason:
            self.bus.log.emit(f"NET: {reason}; routing to text fallback.")
        if not self.router.has_text_fallback():
            self.bus.log.emit("NET: no configured text fallback provider. Add one to config/api_keys.json.")
            self._emit_state("STANDBY")
            return
        self._emit_state("PROCESSING")
        try:
            profile, response = await self.router.generate_text(text)
            self.bus.log.emit(f"ORION[{profile.name}]: {response}")
            self.memory.log_episode("orion", response)
            self.tts.speak(response)
        except Exception as exc:
            self.bus.log.emit(f"NET: text fallback failed - {str(exc).splitlines()[0][:180]}")
        finally:
            if not self.tts.is_speaking():
                self._emit_state("LISTENING" if self._offline_voice_ready() else "STANDBY")

    async def _send_text_turn(self, text: str) -> None:
        try:
            await self.session.send_client_content(
                turns={"parts": [{"text": text}]}, turn_complete=True
            )
        except TypeError:
            content = types.Content(role="user", parts=[types.Part(text=text)])
            await self.session.send_client_content(turns=content, turn_complete=True)
        except Exception as exc:
            self.bus.log.emit(f"NET: manual command rejected - {exc}")
            await self._submit_text_fallback(text, reason="Native provider rejected manual command")

    # ── startup briefing ──────────────────────────────────────────────────────

    _BRIEFING_TOPICS = (
        ("AI", "artificial intelligence"),
        ("Neuralink", "Neuralink"),
        ("Stock market", "stock market today"),
        ("Cryptocurrency", "cryptocurrency market"),
    )

    async def deliver_startup_briefing(self) -> None:
        try:
            briefing = await self._compose_startup_briefing()
        except Exception as exc:
            self.bus.log.emit(f"BRIEF: startup briefing failed - {str(exc).splitlines()[0][:160]}")
            return
        self.bus.log.emit("BRIEF: startup briefing prepared.")
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not self.connected and not self.stop_event.is_set():
            await asyncio.sleep(0.5)
        if self.stop_event.is_set():
            return
        self._refresh_wake_window()
        hour   = datetime.now().hour
        period = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        greeting = random.choice(STARTUP_GREETINGS).format(period=period)
        instruction = (
            f'Open with this exact greeting, word for word: "{greeting}" '
            "Then deliver a personal intelligence briefing in the manner of a trusted "
            "aide, based on the source material below. Do not read the headlines "
            "verbatim: synthesise them. Lead with the most significant development and "
            "explain why it matters, connect related stories where they touch, and add "
            "one or two measured observations of your own. Keep it conversational and "
            "calm at a steady, unhurried pace — roughly ninety seconds. Close by "
            "offering to open any of the stories or explore one in depth:\n\n" + briefing
        )
        if self.session is not None:
            self._emit_state("PROCESSING")
            await self._send_text_turn(instruction)
        else:
            self.bus.log.emit(f"BRIEF: {briefing}")
            await self._submit_text_fallback(instruction, reason="Live channel offline for briefing")

    async def _compose_startup_briefing(self) -> str:
        now = datetime.now()
        lines = [
            f"Startup briefing. Today is {now.strftime('%A %d %B %Y')} "
            f"and the local time is {now.strftime('%H:%M')}."
        ]
        articles: list[dict[str, str]] = []
        timeout = ClientTimeout(total=15.0, connect=5.0)
        async with ClientSession(timeout=timeout) as session:
            for label, query in self._BRIEFING_TOPICS:
                try:
                    stories = await self._fetch_headlines(session, query, limit=3)
                    if stories:
                        fragments: list[str] = []
                        for title, url in stories:
                            articles.append({"topic": label, "title": title, "url": url})
                            fragments.append(f"{title}.")
                        lines.append(f"{label} news: " + " ".join(fragments))
                    else:
                        lines.append(f"{label} news: no headlines retrieved.")
                except Exception:
                    lines.append(f"{label} news feed unavailable.")
            try:
                lines.append(await self._fetch_crypto_prices(session))
            except Exception:
                lines.append("Cryptocurrency pricing unavailable.")
        # Cache the sources so the open_news tool can open exactly what was read.
        self.dispatcher.news_articles[:] = articles
        if articles:
            self.bus.log.emit(
                f"NEWS: cached {len(articles)} briefing sources - "
                'say "open the story about ..." to read one.'
            )
            for index, article in enumerate(articles, 1):
                self.bus.log.emit(f"NEWS[{index}]: ({article['topic']}) {article['title']}")
        return "\n".join(lines)

    async def _fetch_headlines(
        self, session: ClientSession, query: str, limit: int = 3
    ) -> list[tuple[str, str]]:
        """Return (title, url) pairs from the Google News RSS feed for a query."""
        url = (
            "https://news.google.com/rss/search"
            f"?q={quote_plus(query)}&hl=en-GB&gl=GB&ceid=GB:en"
        )
        async with session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(f"news feed returned {response.status}")
            raw = await response.text()
        stories: list[tuple[str, str]] = []
        for block in re.findall(r"<item>(.*?)</item>", raw, re.S):
            title_match = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", block, re.S)
            link_match  = re.search(r"<link>(.*?)</link>", block, re.S)
            if not title_match:
                continue
            title = html.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip()
            link  = html.unescape(link_match.group(1).strip()) if link_match else ""
            if title:
                stories.append((title[:180], link))
            if len(stories) >= max(1, limit):
                break
        return stories

    async def _fetch_crypto_prices(self, session: ClientSession) -> str:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
        )
        async with session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(f"crypto feed returned {response.status}")
            data = await response.json()
        parts: list[str] = []
        for coin_id, coin_label in (("bitcoin", "Bitcoin"), ("ethereum", "Ethereum")):
            coin = data.get(coin_id) or {}
            price = coin.get("usd")
            change = coin.get("usd_24h_change")
            if price is None:
                continue
            fragment = f"{coin_label} at {float(price):,.0f} dollars"
            if change is not None:
                fragment += f" ({float(change):+.1f} percent over twenty-four hours)"
            parts.append(fragment)
        if not parts:
            raise RuntimeError("no crypto prices returned")
        return "Cryptocurrency: " + "; ".join(parts) + "."

    async def run(self) -> None:
        self.playback.start()
        self.tts.start()
        self.bus.log.emit(
            "VOICE: wake-word standby "
            + ("enabled - say 'Orion' to open the channel."
               if self.wake_mode_enabled
               else "disabled - microphone always live (set ORION_WAKE_MODE=1 to change).")
        )
        live_index = 0
        backoff    = 2.0
        while not self.stop_event.is_set():
            profiles = self.router.live_profiles()
            if not profiles:
                self.connected = False
                self.session   = None
                self._ensure_fallback_mic()
                self._emit_state("LISTENING" if self._offline_voice_ready() else "STANDBY")
                if not self._no_live_notice_sent:
                    if self.router.has_text_fallback():
                        self.bus.log.emit("NET: no live audio provider available; text fallback is armed.")
                        if self._offline_voice_ready():
                            self.bus.log.emit(
                                "VOICE: offline voice loop active - speak normally; "
                                "local recognition routes to the text providers and replies aloud."
                            )
                    else:
                        self.bus.log.emit("NET: no live audio or text fallback provider is currently available.")
                    self._no_live_notice_sent = True
                await asyncio.sleep(2.0)
                continue
            self._no_live_notice_sent = False
            profile = profiles[live_index % len(profiles)]
            live_index += 1
            candidates = [profile.model or LIVE_MODEL] + [
                m for m in LIVE_MODEL_FALLBACKS if m != (profile.model or LIVE_MODEL)
            ]
            live_model = candidates[self._live_model_shift.get(profile.name, 0) % len(candidates)]
            client = genai.Client(api_key=profile.api_key, http_options={"api_version": "v1beta"})
            try:
                self._emit_state("CONNECTING")
                self.bus.log.emit(
                    f"NET: initialising {profile.name} live channel ({live_model.rsplit('/', 1)[-1]})."
                )
                config = self._build_config()
                async with client.aio.live.connect(model=live_model, config=config) as session:
                    self.session   = session
                    self.connected = True
                    self.active_live_provider = profile
                    self._emit_state("LISTENING")
                    self.bus.log.emit(f"NET: {profile.name} live channel synchronised.")
                    backoff = 2.0
                    live_index -= 1  # keep the working profile on clean reconnects
                    await self._session_loop()
                    self.bus.log.emit("NET: live channel closed; re-establishing.")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.connected = False
                self.session   = None
                self.active_live_provider = None
                self._emit_state("STANDBY")
                message = str(exc)
                self.bus.log.emit(f"NET: {message.splitlines()[0][:180]}")
                if self._search_tool_enabled and re.search(r"(?i)google_search", message):
                    self._search_tool_enabled = False
                    self.bus.log.emit("NET: search grounding rejected by provider; reconnecting without it.")
                elif re.search(r"(?i)not.?found|404|does not exist|unsupported|invalid model", message):
                    # Bad/retired model id — rotate through known live models
                    # instead of cooling the whole provider.
                    self._live_model_shift[profile.name] = (
                        self._live_model_shift.get(profile.name, 0) + 1
                    )
                    self.bus.log.emit(
                        f"NET: live model {live_model} unavailable; rotating to the next candidate."
                    )
                else:
                    self.router.mark_failure(profile, exc)
                # A stale resumption handle can poison every reconnect attempt;
                # drop it so the next connection starts fresh.
                self._resumption_handle = None
                await asyncio.sleep(backoff)
                backoff = min(12.0, backoff * 1.5)
            finally:
                self.connected = False
                self.session   = None
                self.active_live_provider = None
                if self.mic is not None:
                    self.mic.stop()
                    self.mic = None
        self.playback.stop()
        self.tts.stop()

    async def stop(self) -> None:
        self.stop_event.set()
        if self.mic is not None:
            self.mic.stop()
        self.playback.stop()
        self.tts.stop()
        try:
            if self.session is not None and hasattr(self.session, "close"):
                maybe = self.session.close()
                if asyncio.iscoroutine(maybe):
                    await maybe
        except Exception:
            pass

    def _ensure_fallback_mic(self) -> None:
        """
        JARVIS-style offline voice loop: when no live audio provider is
        available, keep the microphone alive so local recognition can still
        hear commands; replies are routed through the text providers and
        spoken with the local voice.
        """
        if self.mic is not None or not self.microphone_enabled or self.stop_event.is_set():
            return
        try:
            loop = asyncio.get_running_loop()
            self.mic = MicrophoneEngine(
                loop, asyncio.Queue(maxsize=8), self.bus,
                lambda: False,  # nothing consumes forwarded audio offline
                self.vad,
                recogniser=self.recogniser,
                on_transcript=self._on_local_transcript,
                speaking_check=self._output_active,
                on_barge_in=self._on_barge_in,
            )
            self.mic.set_enabled(self.microphone_enabled)
            self.mic.start()
        except Exception as exc:
            self.mic = None
            self.bus.log.emit(f"AUDIO: offline microphone unavailable - {exc}")

    async def _session_loop(self) -> None:
        if self.mic is not None:
            # Replace the offline fallback microphone with the live pipeline.
            self.mic.stop()
            self.mic = None
        self.out_queue = asyncio.Queue(maxsize=MIC_QUEUE_LIMIT)
        loop           = asyncio.get_running_loop()
        self.mic       = MicrophoneEngine(
            loop, self.out_queue, self.bus, self._can_capture_microphone, self.vad,
            recogniser=self.recogniser,
            on_transcript=self._on_local_transcript,
            speaking_check=self._output_active,
            on_barge_in=self._on_barge_in,
        )
        self.mic.set_enabled(self.microphone_enabled)
        self.mic.start()
        send_task = asyncio.create_task(self._send_realtime(),    name="orion-send-realtime")
        recv_task = asyncio.create_task(self._receive_realtime(), name="orion-receive-realtime")
        # FIRST_COMPLETED: a receive loop that ends cleanly must also tear down
        # the send loop, otherwise the session hangs on out_queue.get() forever.
        done, pending = await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc:
                raise exc

    def _can_capture_microphone(self) -> bool:
        # Capture continues while ORION speaks (barge-in / interruption); only
        # tool execution, mute, shutdown, and a closed wake gate block it.
        return (
            self.microphone_enabled
            and not self.tool_busy
            and not self.stop_event.is_set()
            and self._wake_gate_open()
        )

    def _wake_gate_open(self) -> bool:
        if not self.wake_mode_enabled or not self.recogniser.available:
            return True
        return time.monotonic() < self.wake_active_until

    def _refresh_wake_window(self) -> None:
        self.wake_active_until = time.monotonic() + WAKE_WINDOW_SECONDS

    def _on_local_transcript(self, text: str) -> None:
        lowered = text.lower().strip()
        if not lowered:
            return
        wake_hit = any(word in lowered for word in WAKE_WORDS)
        if wake_hit:
            was_closed = not self._wake_gate_open()
            self._refresh_wake_window()
            if was_closed:
                self.bus.log.emit("SR: wake word detected - channel open.")
                self.bus.banner.emit("WAKE WORD ACKNOWLEDGED", 2)
            if self.connected:
                self._emit_state("LISTENING")
        if self.connected:
            return  # live channel handles the conversation natively
        # ── offline voice loop: local STT → provider router → local voice ────
        if self.tts.is_speaking():
            self.tts.interrupt()
        command = self._strip_wake_words(lowered) if wake_hit else lowered
        if wake_hit and not command:
            self.bus.log.emit("YOU (voice): [wake]")
            self.tts.speak("Yes, sir?")
            return
        if self.wake_mode_enabled and not wake_hit and not self._wake_gate_open():
            return  # standby: ignore ambient speech until the wake word
        if not wake_hit and len(command.split()) < 2:
            return  # single stray words are almost always noise
        if not self.router.has_text_fallback():
            self.bus.log.emit(f"SR (local): {text}")
            return
        self.bus.log.emit(f"YOU (voice): {command}")
        self._refresh_wake_window()
        asyncio.create_task(self.submit_text(command))

    def _strip_wake_words(self, text: str) -> str:
        for word in WAKE_WORDS:
            text = text.replace(word, " ")
        return re.sub(r"\s+", " ", text).strip(" ,.!?")

    def _on_barge_in(self) -> None:
        interrupted = False
        if self.playback.speaking_recently():
            self.playback.clear()
            interrupted = True
        if self.tts.is_speaking():
            self.tts.interrupt()
            interrupted = True
        if interrupted:
            self.bus.log.emit("AUDIO: user interruption - playback halted.")
            if self.connected:
                self._emit_state("LISTENING")

    async def _send_realtime(self) -> None:
        while not self.stop_event.is_set():
            try:
                media = await asyncio.wait_for(self.out_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self.session is None:
                continue
            await self._send_media(media)

    async def _receive_realtime(self) -> None:
        in_buffer:  list[str] = []
        out_buffer: list[str] = []
        while not self.stop_event.is_set() and self.session is not None:
            async for response in self.session.receive():
                if self.stop_event.is_set():
                    return
                if getattr(response, "go_away", None) is not None:
                    self.bus.log.emit("NET: server requested reconnect (go_away); rotating channel.")
                    return
                resumption = getattr(response, "session_resumption_update", None)
                if resumption is not None:
                    handle = getattr(resumption, "new_handle", None)
                    if getattr(resumption, "resumable", False) and handle:
                        self._resumption_handle = str(handle)
                data = getattr(response, "data", None)
                if data:
                    self.playback.enqueue(data)
                    self._emit_state("SPEAKING")
                server_content = getattr(response, "server_content", None)
                if server_content is not None:
                    if getattr(server_content, "interrupted", False):
                        self.playback.clear()
                        out_buffer = []
                        self.bus.log.emit("AUDIO: response interrupted by user.")
                        self._refresh_wake_window()
                        if self.connected:
                            self._emit_state("LISTENING")
                    output_text = self._extract_transcription(server_content, "output_transcription")
                    input_text  = self._extract_transcription(server_content, "input_transcription")
                    if output_text:
                        out_buffer.append(output_text)
                    if input_text:
                        in_buffer.append(input_text)
                        self._refresh_wake_window()
                    if getattr(server_content, "turn_complete", False):
                        if in_buffer:
                            user_text = " ".join(in_buffer).strip()
                            self.bus.log.emit(f"YOU: {user_text}")
                            self.memory.log_episode("user", user_text)
                            in_buffer = []
                        if out_buffer:
                            orion_text = " ".join(out_buffer).strip()
                            self.bus.log.emit(f"ORION: {orion_text}")
                            self.memory.log_episode("orion", orion_text)
                            out_buffer = []
                        # Microphone always returns to listening after a response.
                        self._refresh_wake_window()
                        if self.connected:
                            self._emit_state("LISTENING")
                tool_call = getattr(response, "tool_call", None)
                if tool_call is not None:
                    await self._handle_tool_call(tool_call)

    async def _handle_tool_call(self, tool_call: Any) -> None:
        function_responses: list[Any] = []
        self.tool_busy = True
        if self.mic is not None:
            self.mic._drain()
        self._emit_state("PROCESSING")
        try:
            for fc in getattr(tool_call, "function_calls", []) or []:
                name    = getattr(fc, "name", "")
                args    = dict(getattr(fc, "args", {}) or {})
                call_id = getattr(fc, "id", None)
                self.bus.log.emit(f"TOOL: {name} requested.")
                try:
                    result = await self.dispatcher.dispatch_chain(name, args)
                    if result.media:
                        await self._send_media(result.media)
                    payload = result.response_payload()
                except SecurityViolation as exc:
                    self.bus.log.emit(f"SEC: {exc}")
                    payload = {"ok": False, "result": str(exc)}
                except Exception as exc:
                    self.bus.log.emit(f"TOOL: {name} failed - {exc}")
                    payload = {"ok": False, "result": f"{name} failed: {exc}"}
                function_responses.append(self._function_response(call_id, name, payload))
            if function_responses and self.session is not None:
                await self.session.send_tool_response(function_responses=function_responses)
        finally:
            self.tool_busy = False
            if self.connected:
                self._emit_state("LISTENING")

    def _function_response(self, call_id: Any, name: str, payload: dict[str, Any]) -> Any:
        try:
            return types.FunctionResponse(id=call_id, name=name, response=payload)
        except Exception:
            return {"id": call_id, "name": name, "response": payload}

    async def _send_media(self, media: dict[str, Any]) -> None:
        if self.session is None:
            return
        try:
            await self.session.send_realtime_input(media=media)
            return
        except TypeError:
            pass
        blob = types.Blob(
            data=media["data"],
            mime_type=media.get("mime_type", "application/octet-stream"),
        )
        mime = media.get("mime_type", "")
        try:
            if mime.startswith("audio/"):
                await self.session.send_realtime_input(audio=blob)
            elif mime.startswith("image/"):
                await self.session.send_realtime_input(image=blob)
            else:
                await self.session.send_realtime_input(media=blob)
        except TypeError:
            await self.session.send_realtime_input(media=blob)

    def _extract_transcription(self, server_content: Any, attr: str) -> str:
        item = getattr(server_content, attr, None)
        text = getattr(item, "text", "") if item is not None else ""
        return clean_transcript(text) if text else ""

    def _build_config(self) -> Any:
        """Build a Pylance-clean Gemini LiveConnectConfig."""
        system_instruction = self.router.system_instruction()
        tools: list[Any] = [{"function_declarations": TOOL_DECLARATIONS}]
        if self._search_tool_enabled:
            # Google Search grounding: real-time knowledge alongside local tools.
            tools.insert(0, {"google_search": {}})
        try:
            return types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                # Transcriptions are what feed the YOU:/ORION: console log and
                # episodic memory — without requesting them the server sends none.
                input_audio_transcription={},
                output_audio_transcription={},
                system_instruction=system_instruction,
                tools=tools,
                session_resumption=types.SessionResumptionConfig(handle=self._resumption_handle),
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                    )
                ),
            )
        except Exception:
            return {
                "response_modalities": ["AUDIO"],
                "input_audio_transcription": {},
                "output_audio_transcription": {},
                "system_instruction": system_instruction,
                "tools": tools,
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": "Charon"}
                    }
                },
            }

# ──────────────────────────────────────────────────────────────────────────────
# REMOTE GATEWAY  —  opt-in hybrid access for Cloud ORION (browser / mobile)
# ──────────────────────────────────────────────────────────────────────────────

REMOTE_PAGE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>O.R.I.O.N. Remote</title>
<style>
body{margin:0;background:#050508;color:#fff;font-family:'Segoe UI',Arial,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:14px 18px;border-bottom:2px solid #991024;background:#0f0f14}
h1{margin:0;font-size:18px}.sub{color:#a9a9b2;font-size:11px}
#log{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:86%;padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.45;white-space:pre-wrap}
.user{align-self:flex-end;background:#991024}
.orion{align-self:flex-start;background:#14141d;border:1px solid #2a1118}
form{display:flex;gap:8px;padding:12px;border-top:1px solid #2a1118;background:#0a0a10}
input{flex:1;background:#050508;color:#fff;border:1px solid #991024;border-radius:10px;padding:12px;font-size:15px}
button{background:#991024;color:#fff;border:1px solid #ff1a3c;border-radius:10px;padding:0 18px;font-weight:700}
</style></head><body>
<header><h1>O.R.I.O.N.</h1><div class="sub">REMOTE UPLINK — conversations sync into ORION's memory</div></header>
<div id="log"></div>
<form id="f"><input id="m" placeholder="Message ORION…" autocomplete="off"><button>SEND</button></form>
<script>
const log=document.getElementById('log');
let token=localStorage.getItem('orion_token')||'';
if(!token){token=prompt('Enter the ORION remote access token (config/remote_token.txt)')||'';localStorage.setItem('orion_token',token);}
function add(cls,text){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;}
document.getElementById('f').addEventListener('submit',async e=>{
 e.preventDefault();const inp=document.getElementById('m');const text=inp.value.trim();if(!text)return;inp.value='';add('user',text);
 try{const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token,message:text})});
 const data=await res.json();
 if(res.status===401){localStorage.removeItem('orion_token');add('orion','Invalid token. Reload the page and re-enter it.');return;}
 add('orion',data.ok?data.reply:('Fault: '+(data.error||'unknown')));}catch(err){add('orion','Link fault: '+err);}
});
</script></body></html>"""


class RemoteGateway:
    """
    Opt-in hybrid access layer for Cloud ORION.

    Disabled by default.  Set ORION_REMOTE_ACCESS=1 to serve a token-protected
    chat uplink (browser / mobile) on ORION_REMOTE_PORT (default 8765).  Remote
    turns are answered through the text provider router and logged into the
    same episodic memory as desktop conversations, so context stays
    synchronised across devices.
    """

    def __init__(self, worker: "GenAILiveWorker", memory: OrionMemoryMatrix, bus: OrionBus) -> None:
        self.worker = worker
        self.memory = memory
        self.bus    = bus
        self.host   = os.getenv("ORION_REMOTE_HOST", "0.0.0.0").strip() or "0.0.0.0"
        self.port   = int(os.getenv("ORION_REMOTE_PORT", "8765") or 8765)
        self.token  = self._load_token()
        self._web: Any    = None
        self._runner: Any = None

    def _load_token(self) -> str:
        token_path = CONFIG_DIR / "remote_token.txt"
        try:
            if token_path.exists():
                existing = token_path.read_text(encoding="utf-8").strip()
                if existing:
                    return existing
        except Exception:
            pass
        import secrets
        token = secrets.token_urlsafe(24)
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            token_path.write_text(token, encoding="utf-8")
        except Exception:
            pass
        return token

    async def start(self) -> None:
        from aiohttp import web
        self._web = web
        app = web.Application()
        app.router.add_get("/",            self._handle_page)
        app.router.add_get("/api/health",  self._handle_health)
        app.router.add_post("/api/chat",   self._handle_chat)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.bus.log.emit(
            f"REMOTE: uplink active on port {self.port}; "
            "access token stored in config/remote_token.txt."
        )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_page(self, request: Any) -> Any:
        return self._web.Response(text=REMOTE_PAGE_HTML, content_type="text/html")

    async def _handle_health(self, request: Any) -> Any:
        return self._web.json_response({"ok": True, "state": "online"})

    async def _handle_chat(self, request: Any) -> Any:
        try:
            payload = await request.json()
        except Exception:
            return self._web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
        if str(payload.get("token") or "") != self.token:
            return self._web.json_response({"ok": False, "error": "invalid token"}, status=401)
        message = str(payload.get("message") or "").strip()[:4000]
        if not message:
            return self._web.json_response({"ok": False, "error": "empty message"}, status=400)
        try:
            SecuritySanitiser.guard_text(message, "remote.message")
        except SecurityViolation as exc:
            return self._web.json_response({"ok": False, "error": str(exc)}, status=403)
        self.bus.log.emit(f"REMOTE: {message[:120]}")
        self.memory.log_episode("user (remote)", message)
        try:
            profile, reply = await self.worker.router.generate_text(message)
        except Exception as exc:
            return self._web.json_response(
                {"ok": False, "error": str(exc).splitlines()[0][:200]}, status=502
            )
        self.memory.log_episode("orion (remote)", reply)
        return self._web.json_response({"ok": True, "reply": reply, "provider": profile.name})


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _default_provider_payload(gemini_key: str = "") -> dict[str, Any]:
    """Build a Mark VII provider configuration with safe defaults."""
    gemini_key = gemini_key.strip() or os.getenv("ORION_GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("ORION_OPENAI_API_KEY", "").strip()
    openrouter_key = os.getenv("ORION_OPENROUTER_API_KEY", "").strip()
    groq_key = os.getenv("ORION_GROQ_API_KEY", "").strip()
    together_key = os.getenv("ORION_TOGETHER_API_KEY", "").strip()
    anthropic_key = os.getenv("ORION_ANTHROPIC_API_KEY", "").strip()
    xai_key = os.getenv("ORION_XAI_API_KEY", "").strip()
    local_url = os.getenv("ORION_LOCAL_OPENAI_BASE_URL", "").strip()
    return {
        "schema": "orion.mark_vii.providers.v1",
        "active_provider": "gemini",
        "provider_order": [
            "gemini",
            "anthropic",
            "openai",
            "xai_grok",
            "openrouter",
            "groq",
            "together",
            "local_lm_studio",
            "local_ollama",
        ],
        "providers": {
            "gemini": {
                "kind": "gemini_live",
                "enabled": bool(gemini_key),
                "api_key": gemini_key,
                "model": LIVE_MODEL,
                "base_url": "",
                "priority": 10,
                "timeout_s": 30.0,
                "strengths": ["live_information"],
            },
            "anthropic": {
                "kind": "openai_compatible",
                "enabled": bool(anthropic_key),
                "api_key": anthropic_key,
                "model": "claude-sonnet-5",
                "base_url": "https://api.anthropic.com/v1",
                "priority": 15,
                "timeout_s": 45.0,
                "strengths": ["coding", "reasoning", "writing"],
            },
            "xai_grok": {
                "kind": "openai_compatible",
                "enabled": bool(xai_key),
                "api_key": xai_key,
                "model": "grok-4",
                "base_url": "https://api.x.ai/v1",
                "priority": 25,
                "timeout_s": 45.0,
                "strengths": ["live_information", "reasoning"],
            },
            "openrouter": {
                "kind": "openai_compatible",
                "enabled": bool(openrouter_key),
                "api_key": openrouter_key,
                "model": "openai/gpt-4o-mini",
                "base_url": "https://openrouter.ai/api/v1",
                "priority": 20,
                "timeout_s": 30.0,
                "strengths": ["general", "reasoning"],
            },
            "groq": {
                "kind": "openai_compatible",
                "enabled": bool(groq_key),
                "api_key": groq_key,
                "model": "llama-3.1-8b-instant",
                "base_url": "https://api.groq.com/openai/v1",
                "priority": 30,
                "timeout_s": 24.0,
                "strengths": ["fast"],
            },
            "openai": {
                "kind": "openai_compatible",
                "enabled": bool(openai_key),
                "api_key": openai_key,
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "priority": 40,
                "timeout_s": 30.0,
                "strengths": ["reasoning", "general"],
            },
            "together": {
                "kind": "openai_compatible",
                "enabled": bool(together_key),
                "api_key": together_key,
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "base_url": "https://api.together.xyz/v1",
                "priority": 50,
                "timeout_s": 30.0,
                "strengths": ["fast"],
            },
            "local_lm_studio": {
                "kind": "openai_compatible",
                "enabled": bool(local_url),
                "api_key": "local",
                "model": "local-model",
                "base_url": local_url or "http://127.0.0.1:1234/v1",
                "priority": 80,
                "timeout_s": 120.0,
                "strengths": ["fast", "local"],
            },
            "local_ollama": {
                "kind": "openai_compatible",
                "enabled": False,
                "api_key": "local",
                "model": "llama3.1",
                "base_url": "http://127.0.0.1:11434/v1",
                "priority": 90,
                "timeout_s": 120.0,
                "strengths": ["fast", "local"],
            },
        },
        "notes": [
            "Gemini is used for native realtime voice.",
            "OpenAI-compatible providers are used as text fallbacks when Gemini Live is unavailable.",
            "Anthropic (Claude) and xAI (Grok) are reached through their OpenAI-compatible chat endpoints.",
            "The 'strengths' list steers task routing: coding, reasoning, live_information, fast, local, general.",
            "Enable local_lm_studio or local_ollama after starting a compatible local server.",
        ],
    }


def _profile_from_config(name: str, raw: dict[str, Any]) -> AIProviderProfile:
    return AIProviderProfile(
        name=name,
        kind=str(raw.get("kind") or "openai_compatible").strip(),
        model=str(raw.get("model") or "").strip(),
        api_key=str(raw.get("api_key") or "").strip(),
        base_url=str(raw.get("base_url") or "").strip(),
        enabled=bool(raw.get("enabled", False)),
        priority=int(raw.get("priority") or 100),
        timeout_s=float(raw.get("timeout_s") or 30.0),
        strengths=tuple(
            str(item).strip().lower()
            for item in (raw.get("strengths") or [])
            if str(item).strip()
        ),
    )


def _profile_to_config(profile: AIProviderProfile) -> dict[str, Any]:
    return {
        "kind": profile.kind,
        "enabled": profile.enabled,
        "api_key": profile.api_key,
        "model": profile.model,
        "base_url": profile.base_url,
        "priority": profile.priority,
        "timeout_s": profile.timeout_s,
        "strengths": list(profile.strengths),
    }


def read_provider_settings() -> OrionProviderSettings:
    payload = _default_provider_payload()
    if API_CONFIG_PATH.exists():
        try:
            existing = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                # Legacy Mark VI migration: {"gemini_api_key": "..."}
                legacy_key = str(existing.get("gemini_api_key") or "").strip()
                if legacy_key and "providers" not in existing:
                    payload = _default_provider_payload(legacy_key)
                    write_provider_settings(_settings_from_payload(payload))
                else:
                    payload = _merge_provider_payload(payload, existing)
        except Exception:
            payload = _default_provider_payload()
    return _settings_from_payload(payload)


def _merge_provider_payload(defaults: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged["active_provider"] = existing.get("active_provider") or defaults.get("active_provider")
    merged["provider_order"] = existing.get("provider_order") or defaults.get("provider_order")
    providers = dict(defaults.get("providers") or {})
    for name, raw in (existing.get("providers") or {}).items():
        if isinstance(raw, dict):
            base = dict(providers.get(name, {}))
            base.update(raw)
            providers[name] = base
    legacy_key = str(existing.get("gemini_api_key") or "").strip()
    if legacy_key:
        providers.setdefault("gemini", {})["api_key"] = legacy_key
        providers["gemini"]["enabled"] = True
    merged["providers"] = providers
    return merged


def _settings_from_payload(payload: dict[str, Any]) -> OrionProviderSettings:
    providers: dict[str, AIProviderProfile] = {}
    for name, raw in (payload.get("providers") or {}).items():
        if isinstance(raw, dict):
            profile = _profile_from_config(str(name), raw)
            if profile.model or profile.kind == "gemini_live":
                providers[profile.name] = profile
    provider_order = [str(name) for name in (payload.get("provider_order") or providers.keys())]
    active_provider = str(payload.get("active_provider") or (provider_order[0] if provider_order else "gemini"))
    return OrionProviderSettings(
        active_provider=active_provider,
        provider_order=provider_order,
        providers=providers,
    )


def write_provider_settings(settings: OrionProviderSettings) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "orion.mark_vii.providers.v1",
        "active_provider": settings.active_provider,
        "provider_order": settings.provider_order,
        "providers": {name: _profile_to_config(profile) for name, profile in settings.providers.items()},
    }
    API_CONFIG_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def read_api_key() -> str:
    """Legacy convenience wrapper retained for older call sites."""
    settings = read_provider_settings()
    gemini = settings.providers.get("gemini")
    return gemini.api_key if gemini is not None else ""


def write_api_key(api_key: str) -> None:
    """Legacy writer retained; writes the Mark VII provider schema."""
    settings = _settings_from_payload(_default_provider_payload(api_key))
    write_provider_settings(settings)


def ensure_provider_settings(window: Optional[QWidget] = None) -> OrionProviderSettings:
    settings = read_provider_settings()
    if any(profile.enabled and (profile.api_key or profile.base_url) for profile in settings.providers.values()):
        write_provider_settings(settings)
        return settings
    dialog = ApiKeyDialog(window)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        raise SystemExit(0)
    settings = _settings_from_payload(_default_provider_payload(dialog.key()))
    write_provider_settings(settings)
    return settings


def ensure_api_key(window: Optional[QWidget] = None) -> str:
    """Legacy wrapper retained for external imports."""
    return read_api_key() or ensure_provider_settings(window).providers.get("gemini", AIProviderProfile("gemini", "gemini_live", LIVE_MODEL)).api_key


# ──────────────────────────────────────────────────────────────────────────────
# APPLICATION BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────

async def run_application(app: QApplication) -> None:
    bus    = OrionBus()
    window = OrionMainWindow(bus)
    window.show()

    toggle = HolographicToggle(window)
    toggle.move(28, 28)
    toggle.show()

    bus.log.emit("SYS: O.R.I.O.N. Mark VII visual shell initialised.")

    settings   = ensure_provider_settings(window)
    memory     = OrionMemoryMatrix(CORE_DB_PATH, CONFIG_DIR, bus)
    grabber    = VolatileScreenGrabber(bus)
    file_intel = LocalFileIntelligence(bus)
    dispatcher = OrionDispatcher(bus, memory, grabber, file_intel)
    worker     = GenAILiveWorker(settings, bus, memory, dispatcher)

    # Inject real memory into the memory view (replaces the dummy created
    # during _build_ui which ran before memory was available), and close the
    # dummy's SQLite connection so it does not leak.
    window._dummy_memory.close()
    window.memory_view.memory = memory
    window.memory_view.refresh()

    window.attach_worker(worker)
    window.telemetry_view.attach_env_refresh(
        lambda: asyncio.create_task(window.refresh_environment_widgets())
    )

    gateway: RemoteGateway | None = None
    if os.getenv("ORION_REMOTE_ACCESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        gateway = RemoteGateway(worker, memory, bus)
        try:
            await gateway.start()
        except Exception as exc:
            gateway = None
            bus.log.emit(f"REMOTE: uplink unavailable - {exc}")

    shutdown_event = asyncio.Event()

    def request_shutdown() -> None:
        if not shutdown_event.is_set():
            shutdown_event.set()

    app.aboutToQuit.connect(request_shutdown)
    bus.request_shutdown.connect(request_shutdown)

    telemetry_task = asyncio.create_task(window.start_telemetry(), name="orion-telemetry")
    asyncio.create_task(window.refresh_environment_widgets(), name="orion-environment-refresh")
    worker_task = asyncio.create_task(worker.run(), name="orion-live-worker")
    briefing_task = asyncio.create_task(
        worker.deliver_startup_briefing(), name="orion-startup-briefing"
    )

    try:
        await shutdown_event.wait()
    finally:
        bus.state.emit("SHUTTING DOWN")
        telemetry_task.cancel()
        briefing_task.cancel()
        await worker.stop()
        if gateway is not None:
            await gateway.stop()
        worker_task.cancel()
        for task in (telemetry_task, briefing_task, worker_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                bus.log.emit(f"SYS: shutdown task reported - {exc}")
        memory.close()


def main() -> None:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    try:
        with loop:
            loop.run_until_complete(run_application(app))
    except KeyboardInterrupt:
        print("O.R.I.O.N. shutdown requested from console.")
    except SystemExit:
        raise
    except Exception:
        print("O.R.I.O.N. terminated after a controlled fault:")
        traceback.print_exc()


if __name__ == "__main__":
    main()