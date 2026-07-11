"""
Creative Audio Workspace (additive module).

``AudioStudioService`` gives ORION a headless stem-processing pipeline: it
watches directories for raw audio assets (WAV/MP3), gain-stages vocal takes,
converts formats and organises dry stems into export packages.

Two processing backends, chosen automatically:

    • ffmpeg  — when the ``ffmpeg`` binary is on PATH it is used for loudness
                normalisation (EBU R128 ``loudnorm``) and any format
                conversion.  This is the richer path.
    • raw     — otherwise, 16-bit PCM WAV files are peak-normalised in pure
                Python via ``wave`` + ``array`` (no third-party deps).  MP3s
                cannot be decoded without ffmpeg and degrade with a clear
                ``bus.log`` warning.

Every long operation runs off the event loop through ``asyncio.to_thread``.
Real-time telemetry is streamed to the GUI through ``bus.audio_studio_activity``
so the Studio Deck can visualise rendering without ever touching a worker.

Downward-dependency rule respected: this module imports only constants, bus,
data, security and utils — nothing above it.
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import subprocess
import time
import wave
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import BASE_DIR
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line, utc_stamp

STUDIO_ROOT = BASE_DIR / "audio_studio"
RAW_DIR = STUDIO_ROOT / "raw"
PROCESSED_DIR = STUDIO_ROOT / "processed"
PACKAGES_DIR = STUDIO_ROOT / "packages"

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}


@dataclass
class AssetRecord:
    path: str
    name: str
    suffix: str
    size_kb: float
    processed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "name": self.name, "suffix": self.suffix,
                "size_kb": round(self.size_kb, 1), "processed": self.processed}


@dataclass
class StudioInventory:
    raw: list[AssetRecord] = field(default_factory=list)
    processed: list[AssetRecord] = field(default_factory=list)
    at: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "raw_count": len(self.raw),
            "processed_count": len(self.processed),
            "raw": [a.to_dict() for a in self.raw[:60]],
            "processed": [a.to_dict() for a in self.processed[:60]],
            "at": self.at or utc_stamp(),
        }


class AudioStudioService:
    """Headless stem pipeline with ffmpeg or pure-Python fallback."""

    SCAN_INTERVAL = 8.0
    TARGET_PEAK_DBFS = -3.0
    LOUDNORM_I = -16.0        # integrated loudness target (LUFS)

    def __init__(self, bus: OrionBus, telemetry: Any | None = None,
                 watch_dirs: Optional[list[Path]] = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.watch_dirs: list[Path] = [RAW_DIR] + list(watch_dirs or [])
        self._ffmpeg: Optional[str] = shutil.which("ffmpeg")
        self._inventory = StudioInventory()
        self._known: set[str] = set()
        self._stop = asyncio.Event()
        for directory in (RAW_DIR, PROCESSED_DIR, PACKAGES_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        if self._ffmpeg is None:
            self.bus.log.emit(
                "STUDIO: ffmpeg not found on PATH — using pure-Python WAV peak "
                "normalisation; MP3/FLAC processing needs ffmpeg (add it to PATH)."
            )
        else:
            self.bus.log.emit("STUDIO: ffmpeg detected — loudnorm + format conversion enabled.")

    # ── telemetry helper ──────────────────────────────────────────────────────

    def _activity(self, phase: str, data: dict[str, Any]) -> None:
        try:
            self.bus.audio_studio_activity.emit(phase, data)
        except RuntimeError:
            pass
        if self.telemetry is not None:
            self.telemetry.metrics.incr(f"studio.{phase}")

    # ── indexing ──────────────────────────────────────────────────────────────

    async def index_assets(self) -> ToolResult:
        inventory = await asyncio.to_thread(self._scan)
        self._inventory = inventory
        snap = inventory.snapshot()
        self._activity("index", snap)
        return ToolResult(
            f"Audio studio indexed {snap['raw_count']} raw asset(s) and "
            f"{snap['processed_count']} processed stem(s). "
            f"Backend: {'ffmpeg' if self._ffmpeg else 'pure-Python WAV'}."
        )

    def _scan(self) -> StudioInventory:
        inventory = StudioInventory(at=utc_stamp())
        for directory in self.watch_dirs:
            inventory.raw.extend(self._scan_dir(directory, processed=False))
        inventory.processed.extend(self._scan_dir(PROCESSED_DIR, processed=True))
        return inventory

    def _scan_dir(self, directory: Path, processed: bool) -> list[AssetRecord]:
        records: list[AssetRecord] = []
        if not directory.is_dir():
            return records
        try:
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                    try:
                        size_kb = path.stat().st_size / 1024
                    except OSError:
                        size_kb = 0.0
                    records.append(AssetRecord(
                        path=str(path), name=path.name, suffix=path.suffix.lower(),
                        size_kb=size_kb, processed=processed,
                    ))
        except OSError:
            pass
        return records

    def inventory_snapshot(self) -> dict[str, Any]:
        return self._inventory.snapshot()

    # ── vocal processing ──────────────────────────────────────────────────────

    async def process_vocal_take(self, path: str, target_dbfs: float | None = None,
                                 convert_to: str = "wav") -> ToolResult:
        """Gain-stage (normalise) a vocal take and write a processed stem."""
        try:
            path = SecuritySanitiser.guard_text(str(path or "").strip(), "studio.path")
        except SecurityViolation as exc:
            return ToolResult(str(exc), ok=False)
        source = Path(os.path.expandvars(os.path.expanduser(path)))
        if not source.is_absolute():
            source = RAW_DIR / source
        if not source.is_file():
            return ToolResult(f"Audio file not found: {source}", ok=False)
        if source.suffix.lower() not in AUDIO_SUFFIXES:
            return ToolResult(f"Unsupported audio format: {source.suffix}", ok=False)
        target_dbfs = self.TARGET_PEAK_DBFS if target_dbfs is None else float(target_dbfs)
        convert_to = (convert_to or "wav").lower().lstrip(".")
        out_name = f"{source.stem}_processed.{convert_to}"
        out_path = PROCESSED_DIR / out_name
        self._activity("processing", {"file": source.name, "progress": 0.0,
                                      "backend": "ffmpeg" if self._ffmpeg else "raw"})
        try:
            if self._ffmpeg is not None:
                report = await asyncio.to_thread(self._process_ffmpeg, source, out_path, target_dbfs)
            else:
                if source.suffix.lower() != ".wav" or convert_to != "wav":
                    self._activity("error", {"file": source.name})
                    return ToolResult(
                        f"Without ffmpeg I can only normalise 16-bit WAV to WAV, sir. "
                        f"'{source.name}' needs ffmpeg on PATH for {source.suffix}→{convert_to}.",
                        ok=False,
                    )
                report = await asyncio.to_thread(self._process_raw_wav, source, out_path, target_dbfs)
        except Exception as exc:
            self._activity("error", {"file": source.name, "error": first_line(exc, 80)})
            return ToolResult(f"Processing failed: {first_line(exc)}", ok=False)
        self._activity("processed", {"file": out_path.name, "progress": 1.0, **report})
        # Refresh the inventory so the GUI reflects the new stem.
        self._inventory = self._scan()
        self._activity("index", self._inventory.snapshot())
        return ToolResult(
            f"Processed '{source.name}' → '{out_path.name}': {report.get('summary', 'done')}."
        )

    def _process_ffmpeg(self, source: Path, out_path: Path, target_dbfs: float) -> dict[str, Any]:
        # loudnorm gives a broadcast-consistent level; -ar 44100 standardises rate.
        cmd = [
            self._ffmpeg or "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source),
            "-af", f"loudnorm=I={self.LOUDNORM_I}:TP={target_dbfs}:LRA=11",
            "-ar", "44100", str(out_path),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or "ffmpeg failed").strip()[:200])
        return {"summary": f"loudnorm to {self.LOUDNORM_I} LUFS, peak {target_dbfs} dBTP",
                "backend": "ffmpeg"}

    def _process_raw_wav(self, source: Path, out_path: Path, target_dbfs: float) -> dict[str, Any]:
        """Peak-normalise a 16-bit PCM WAV in pure Python."""
        with wave.open(str(source), "rb") as w:
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            n_frames = w.getnframes()
            frames = w.readframes(n_frames)
        if sampwidth != 2:
            raise RuntimeError(f"raw normaliser handles 16-bit PCM only (this is {sampwidth * 8}-bit)")
        samples = array("h")
        samples.frombytes(frames)
        if len(samples) == 0:
            raise RuntimeError("empty audio file")
        peak = max(1, max(abs(int(s)) for s in samples))
        target_amp = 32767.0 * (10 ** (target_dbfs / 20.0))
        gain = target_amp / peak
        # Clamp to avoid overflow on the (unlikely) case gain > 1 pushes past int16.
        normalised = array("h", (
            max(-32768, min(32767, int(round(s * gain)))) for s in samples
        ))
        with wave.open(str(out_path), "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(sampwidth)
            out.setframerate(framerate)
            out.writeframes(normalised.tobytes())
        applied_db = 20 * math.log10(gain) if gain > 0 else 0.0
        duration = n_frames / float(framerate or 1)
        return {"summary": f"peak-normalised to {target_dbfs} dBFS ({applied_db:+.1f} dB), "
                           f"{channels}ch {framerate} Hz {duration:.1f}s",
                "backend": "raw", "gain_db": round(applied_db, 2)}

    # ── stem packaging ────────────────────────────────────────────────────────

    async def export_stem_package(self, name: str = "") -> ToolResult:
        """Collect processed stems into a dated package folder with a manifest."""
        return await asyncio.to_thread(self._export_package, name)

    def _export_package(self, name: str) -> ToolResult:
        import json
        processed = self._scan_dir(PROCESSED_DIR, processed=True)
        if not processed:
            return ToolResult("There are no processed stems to package yet, sir.", ok=False)
        slug = SecuritySanitiser.guard_text(str(name or "session"), "studio.pkg")
        slug = "".join(c if c.isalnum() else "_" for c in slug.lower()).strip("_")[:40] or "session"
        folder = PACKAGES_DIR / f"{datetime.now():%Y-%m-%d}_{slug}"
        folder.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, Any]] = []
        for record in processed:
            dest = folder / record.name
            try:
                shutil.copy2(record.path, dest)
                manifest.append(record.to_dict())
            except OSError:
                continue
        (folder / "manifest.json").write_text(
            json.dumps({"package": slug, "created": utc_stamp(), "stems": manifest}, indent=2),
            encoding="utf-8",
        )
        self._activity("package", {"package": slug, "count": len(manifest),
                                   "folder": str(folder)})
        return ToolResult(
            f"Exported {len(manifest)} stem(s) to package '{folder.name}', sir "
            f"(manifest.json written)."
        )

    # ── background directory monitor ──────────────────────────────────────────

    async def run(self) -> None:
        if self.telemetry is not None:
            self.telemetry.health.register("audio_studio")
        try:
            while not self._stop.is_set():
                inventory = await asyncio.to_thread(self._scan)
                current = {a.path for a in inventory.raw} | {a.path for a in inventory.processed}
                if current != self._known:
                    self._known = current
                    self._inventory = inventory
                    self._activity("index", inventory.snapshot())
                if self.telemetry is not None:
                    self.telemetry.health.beat("audio_studio", "OK",
                                               f"{len(inventory.raw)} raw / {len(inventory.processed)} done")
                await asyncio.sleep(self.SCAN_INTERVAL)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()
