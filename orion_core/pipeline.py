"""
Creator Agency & Campaign Pipeline (additive module).

``AgencyPipelineService`` is a deterministic, fully-offline tracker for creator
workflows: brand partnerships, content drops and engagement metrics.  It models
each campaign as it moves through a fixed set of stages (a kanban), records the
content scheduled/published under it, and logs performance over time.

State persists to ``config/pipeline/pipeline.json`` so it survives restarts
with no database or network dependency.  Every mutation — and a periodic
heartbeat — emits ``bus.dashboard_event("pipeline_update", snapshot)`` so the
Studio Deck kanban stays in sync purely through the bus.

The dispatcher exposes ``campaign_pipeline`` with actions update_stage,
log_performance and get_pipeline_snapshot (plus create/schedule helpers).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .utils import utc_stamp

PIPELINE_DIR = CONFIG_DIR / "pipeline"
PIPELINE_FILE = PIPELINE_DIR / "pipeline.json"


class Stage(str, Enum):
    """Ordered negotiation → delivery workflow (the kanban columns)."""

    LEAD = "Lead"
    NEGOTIATION = "Negotiation"
    CONTRACTED = "Contracted"
    IN_PRODUCTION = "In Production"
    SCHEDULED = "Scheduled"
    PUBLISHED = "Published"
    PAID = "Paid / Complete"

    @classmethod
    def order(cls) -> list["Stage"]:
        return [cls.LEAD, cls.NEGOTIATION, cls.CONTRACTED, cls.IN_PRODUCTION,
                cls.SCHEDULED, cls.PUBLISHED, cls.PAID]

    @classmethod
    def coerce(cls, value: str) -> Optional["Stage"]:
        v = str(value or "").strip().lower().replace("_", " ")
        for stage in cls:
            if v == stage.value.lower() or v == stage.name.lower() or v in stage.value.lower():
                return stage
        return None


@dataclass
class ContentDrop:
    title: str
    platform: str
    scheduled: str = ""      # ISO date
    status: str = "planned"  # planned | scheduled | published


@dataclass
class PerformanceLog:
    metric: str
    value: float
    at: str = ""


@dataclass
class Campaign:
    id: str
    name: str
    brand: str = ""
    stage: str = Stage.LEAD.value
    value: float = 0.0
    deadline: str = ""
    notes: str = ""
    created: str = ""
    updated: str = ""
    content: list[ContentDrop] = field(default_factory=list)
    performance: list[PerformanceLog] = field(default_factory=list)

    def engagement_total(self) -> float:
        return sum(p.value for p in self.performance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "brand": self.brand, "stage": self.stage,
            "value": self.value, "deadline": self.deadline, "notes": self.notes,
            "content": [asdict(c) for c in self.content],
            "performance": [asdict(p) for p in self.performance],
            "engagement_total": self.engagement_total(),
            "updated": self.updated,
        }


class AgencyPipelineService:
    """Offline campaign/contract/content tracker with bus-driven snapshots."""

    HEARTBEAT_INTERVAL = 12.0

    def __init__(self, bus: OrionBus, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self._campaigns: dict[str, Campaign] = {}
        self._stop = asyncio.Event()
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── ids ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _slug(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")[:40] or "campaign"

    def _resolve(self, ref: str) -> Optional[Campaign]:
        ref_l = str(ref or "").strip().lower()
        if ref_l in self._campaigns:
            return self._campaigns[ref_l]
        for camp in self._campaigns.values():
            if ref_l == camp.id or ref_l in camp.name.lower() or ref_l in camp.brand.lower():
                return camp
        return None

    # ── mutations ─────────────────────────────────────────────────────────────

    def create_campaign(self, name: str, brand: str = "", value: float = 0.0,
                        deadline: str = "", notes: str = "") -> ToolResult:
        name = str(name or "").strip()
        if not name:
            return ToolResult("A campaign needs a name, sir.", ok=False)
        cid = self._slug(name)
        if cid in self._campaigns:
            return ToolResult(f"Campaign '{name}' already exists.", ok=False)
        self._campaigns[cid] = Campaign(
            id=cid, name=name, brand=brand, value=float(value or 0.0),
            deadline=deadline, notes=notes, created=utc_stamp(), updated=utc_stamp(),
        )
        self._commit()
        return ToolResult(f"Campaign '{name}' created at the Lead stage, sir.")

    def update_stage(self, ref: str, stage: str) -> ToolResult:
        camp = self._resolve(ref)
        if camp is None:
            return self._not_found(ref)
        target = Stage.coerce(stage)
        if target is None:
            return ToolResult(
                f"Unknown stage '{stage}'. Valid stages: "
                + ", ".join(s.value for s in Stage.order()) + ".", ok=False,
            )
        old = camp.stage
        camp.stage = target.value
        camp.updated = utc_stamp()
        self._commit()
        return ToolResult(f"'{camp.name}' moved from {old} → {target.value}, sir.")

    def schedule_content(self, ref: str, title: str, platform: str = "",
                        scheduled: str = "") -> ToolResult:
        camp = self._resolve(ref)
        if camp is None:
            return self._not_found(ref)
        if not title.strip():
            return ToolResult("A content title is required, sir.", ok=False)
        status = "scheduled" if scheduled.strip() else "planned"
        camp.content.append(ContentDrop(title=title.strip(), platform=platform.strip(),
                                        scheduled=scheduled.strip(), status=status))
        camp.updated = utc_stamp()
        self._commit()
        return ToolResult(f"Content '{title}' added to '{camp.name}' ({status}), sir.")

    def log_performance(self, ref: str, metric: str, value: float) -> ToolResult:
        camp = self._resolve(ref)
        if camp is None:
            return self._not_found(ref)
        camp.performance.append(PerformanceLog(metric=str(metric or "engagement"),
                                               value=float(value or 0.0), at=utc_stamp()))
        camp.updated = utc_stamp()
        self._commit()
        return ToolResult(
            f"Logged {metric}={value:,.0f} for '{camp.name}' "
            f"(total engagement {camp.engagement_total():,.0f}), sir."
        )

    def delete_campaign(self, ref: str, confirm: bool = False) -> ToolResult:
        """Permanently remove a campaign — gated on the user's explicit consent."""
        camp = self._resolve(ref)
        if camp is None:
            return self._not_found(ref)
        if not confirm:
            return ToolResult(
                f"Deleting the campaign '{camp.name}'"
                + (f" ({camp.brand})" if camp.brand else "")
                + " is permanent, sir. Confirm and I shall remove it.",
                ok=False,
            )
        key = next((k for k, c in self._campaigns.items() if c is camp), None)
        name = camp.name
        if key is not None:
            self._campaigns.pop(key, None)
        self._commit()
        self.bus.log.emit(f"PIPELINE: campaign '{name}' deleted on request.")
        return ToolResult(f"Deleted the campaign '{name}', sir.")

    def _not_found(self, ref: str) -> ToolResult:
        known = ", ".join(c.name for c in self._campaigns.values()) or "none yet"
        return ToolResult(f"No campaign matches '{ref}', sir. Known campaigns: {known}.", ok=False)

    # ── snapshot ──────────────────────────────────────────────────────────────

    def get_snapshot(self) -> dict[str, Any]:
        board: dict[str, list[dict[str, Any]]] = {s.value: [] for s in Stage.order()}
        for camp in self._campaigns.values():
            board.setdefault(camp.stage, []).append(camp.to_dict())
        return {
            "stages": [s.value for s in Stage.order()],
            "board": board,
            "totals": {
                "campaigns": len(self._campaigns),
                "pipeline_value": round(sum(c.value for c in self._campaigns.values()), 2),
                "engagement": round(sum(c.engagement_total() for c in self._campaigns.values()), 2),
            },
            "at": utc_stamp(),
        }

    def snapshot_text(self) -> ToolResult:
        snap = self.get_snapshot()
        lines = [f"Campaign pipeline — {snap['totals']['campaigns']} campaign(s), "
                 f"£{snap['totals']['pipeline_value']:,.0f} in play, "
                 f"{snap['totals']['engagement']:,.0f} total engagement:"]
        for stage in snap["stages"]:
            cards = snap["board"].get(stage, [])
            if cards:
                names = ", ".join(f"{c['name']}"
                                  + (f" ({c['brand']})" if c['brand'] else "") for c in cards)
                lines.append(f"  {stage}: {names}")
        return ToolResult("\n".join(lines) if len(lines) > 1
                          else "The pipeline is empty, sir. Create a campaign to begin.")

    # ── persistence ───────────────────────────────────────────────────────────

    def _commit(self) -> None:
        self._save()
        snap = self.get_snapshot()
        self.bus.dashboard_event.emit("pipeline_update", snap)
        if self.telemetry is not None:
            self.telemetry.metrics.gauge("pipeline.campaigns", float(len(self._campaigns)))

    def _save(self) -> None:
        try:
            payload = {"schema": "orion.pipeline.v1", "saved": utc_stamp(),
                       "campaigns": [c.to_dict() for c in self._campaigns.values()]}
            PIPELINE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            self.bus.log.emit(f"PIPELINE: save failed - {exc}")

    def _load(self) -> None:
        if not PIPELINE_FILE.exists():
            return
        try:
            data = json.loads(PIPELINE_FILE.read_text(encoding="utf-8"))
            for raw in data.get("campaigns", []):
                camp = Campaign(
                    id=str(raw.get("id") or self._slug(raw.get("name", ""))),
                    name=str(raw.get("name", "")), brand=str(raw.get("brand", "")),
                    stage=str(raw.get("stage", Stage.LEAD.value)),
                    value=float(raw.get("value", 0.0)), deadline=str(raw.get("deadline", "")),
                    notes=str(raw.get("notes", "")), updated=str(raw.get("updated", "")),
                )
                for c in raw.get("content", []):
                    camp.content.append(ContentDrop(**{k: c.get(k, "") for k in
                                                       ("title", "platform", "scheduled", "status")}))
                for p in raw.get("performance", []):
                    camp.performance.append(PerformanceLog(metric=str(p.get("metric", "")),
                                                           value=float(p.get("value", 0.0)),
                                                           at=str(p.get("at", ""))))
                self._campaigns[camp.id] = camp
        except Exception as exc:
            self.bus.log.emit(f"PIPELINE: load recovered - {exc}")

    # ── periodic heartbeat ────────────────────────────────────────────────────

    async def run(self) -> None:
        if self.telemetry is not None:
            self.telemetry.health.register("pipeline")
        # Emit an initial snapshot so the GUI has state on connect.
        self.bus.dashboard_event.emit("pipeline_update", self.get_snapshot())
        try:
            while not self._stop.is_set():
                self.bus.dashboard_event.emit("pipeline_update", self.get_snapshot())
                if self.telemetry is not None:
                    self.telemetry.health.beat("pipeline", "OK", f"{len(self._campaigns)} campaigns")
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        self._stop.set()
