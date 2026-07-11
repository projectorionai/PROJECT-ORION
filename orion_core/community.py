"""
Community layer (Phase 10) + E-commerce Intelligence Hub aggregation (Phase 8).

The ORION Network foundation: shareable, privacy-controlled bundles of
knowledge and market intelligence that users can export, hand to others and
import — the local, offline-first substrate a future sync service would build
on.  Nothing leaves the machine unless the user explicitly exports it, and
private items are never included.

    CommunityHub    — export/import knowledge packs and product research with
                      a signed-ish manifest (schema, author, privacy) and
                      optional anonymisation.

    EcommerceHub    — aggregates the live entrepreneurial picture (scored
                      product opportunities, virality reads, research log,
                      knowledge packs, active brand) into one snapshot for the
                      dashboard and the ``commerce_hub`` tool.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .constants import CONFIG_DIR
from .data import ToolResult
from .memory import MemoryAgent
from .utils import first_line, utc_stamp

COMMUNITY_DIR = CONFIG_DIR / "community"
MANIFEST_SCHEMA = "orion.community.bundle.v1"

# Rough PII scrubbers for anonymised exports.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


class CommunityHub:
    """Privacy-controlled export/import of knowledge and market intelligence."""

    def __init__(self, bus: OrionBus, memory: MemoryAgent, knowledge: Any | None = None,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.knowledge = knowledge
        self.telemetry = telemetry
        COMMUNITY_DIR.mkdir(parents=True, exist_ok=True)

    # ── manifest ──────────────────────────────────────────────────────────────

    def _manifest(self, kind: str, privacy: str) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "kind": kind,
            "privacy": privacy,              # "public" | "shared" | "private"
            "exported_at": utc_stamp(),
            "author": "",                    # user fills in if desired
            "orion_version": "MarkIX",
        }

    # ── knowledge pack sharing ────────────────────────────────────────────────

    def export_pack(self, pack_id: str, privacy: str = "shared") -> ToolResult:
        if self.knowledge is None:
            return ToolResult("Knowledge system unavailable.", ok=False)
        pack = self.knowledge._packs.get(pack_id)
        if pack is None:
            return ToolResult(f"No installed pack '{pack_id}'.", ok=False)
        bundle = {"manifest": self._manifest("knowledge_pack", privacy),
                  "pack": pack.to_dict()}
        path = COMMUNITY_DIR / f"pack_{pack_id}_{datetime.now():%Y%m%d}.orionpack.json"
        try:
            path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            return ToolResult(f"Export failed: {first_line(exc)}", ok=False)
        return ToolResult(f"Exported '{pack.title}' to {path} ({privacy}). "
                          "Share this file; recipients import it with community import.")

    def import_pack(self, path: str) -> ToolResult:
        if self.knowledge is None:
            return ToolResult("Knowledge system unavailable.", ok=False)
        p = Path(path).expanduser()
        if not p.is_file():
            return ToolResult(f"File not found: {p}", ok=False)
        try:
            bundle = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            return ToolResult(f"Unreadable bundle: {first_line(exc)}", ok=False)
        if bundle.get("manifest", {}).get("schema") != MANIFEST_SCHEMA:
            return ToolResult("Not a valid ORION community bundle.", ok=False)
        pack = bundle.get("pack")
        if not isinstance(pack, dict):
            return ToolResult("Bundle contains no knowledge pack.", ok=False)
        result = self.knowledge.install(pack)
        return ToolResult(f"Imported from community bundle. {result.text}")

    # ── product research sharing (privacy-first) ──────────────────────────────

    def export_research(self, privacy: str = "shared", anonymise: bool = True) -> ToolResult:
        rows = [r for r in self.memory.records(limit=500)
                if r.get("category") == "product_research"]
        if not rows:
            return ToolResult("No product research to export yet.", ok=False)
        items = []
        for r in rows:
            value = str(r.get("value", ""))
            key = str(r.get("key_ref", ""))
            if "private" in value.lower() or "private" in key.lower():
                continue  # never export items the user marked private
            if anonymise:
                value = _EMAIL_RE.sub("[email]", value)
                value = _PHONE_RE.sub("[phone]", value)
            items.append({"product": key, "finding": value, "at": r.get("updated_at", "")})
        bundle = {"manifest": self._manifest("product_research", privacy), "items": items}
        path = COMMUNITY_DIR / f"research_{datetime.now():%Y%m%d}.orionresearch.json"
        try:
            path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            return ToolResult(f"Export failed: {first_line(exc)}", ok=False)
        return ToolResult(f"Exported {len(items)} research item(s) to {path} "
                          f"({privacy}{', anonymised' if anonymise else ''}). "
                          "Private items were excluded.")

    def import_research(self, path: str) -> ToolResult:
        p = Path(path).expanduser()
        if not p.is_file():
            return ToolResult(f"File not found: {p}", ok=False)
        try:
            bundle = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            return ToolResult(f"Unreadable bundle: {first_line(exc)}", ok=False)
        if bundle.get("manifest", {}).get("schema") != MANIFEST_SCHEMA:
            return ToolResult("Not a valid ORION community bundle.", ok=False)
        items = bundle.get("items") or []
        count = 0
        for item in items:
            product = str(item.get("product", "")).strip()
            finding = str(item.get("finding", "")).strip()
            if product and finding:
                self.memory.matrix.save("product_research", f"community_{product}",
                                        f"[community] {finding}", silent=True)
                count += 1
        return ToolResult(f"Imported {count} community research item(s) into product research.")

    def list_bundles(self) -> list[str]:
        return [p.name for p in COMMUNITY_DIR.glob("*.json")]


class EcommerceHub:
    """Aggregates the live entrepreneurial intelligence picture (Phase 8)."""

    def __init__(self, bus: OrionBus, memory: MemoryAgent, dropship: Any,
                 knowledge: Any | None = None) -> None:
        self.bus = bus
        self.memory = memory
        self.dropship = dropship
        self.knowledge = knowledge

    def snapshot(self) -> dict[str, Any]:
        research = self.dropship.research_log(limit=30)
        opportunities = []
        for r in research:
            value = r.get("value", "")
            m = re.search(r"score\s+([\d.]+)", value)
            opportunities.append({
                "product": r.get("key_ref", ""),
                "score": float(m.group(1)) if m else None,
                "note": value[:120],
            })
        opportunities.sort(key=lambda o: (o["score"] is None, -(o["score"] or 0)))
        return {
            "at": utc_stamp(),
            "brand": self.memory.active_project or "Hausables",
            "opportunities": opportunities[:15],
            "opportunity_count": len(opportunities),
            "knowledge_packs": (self.knowledge.list_packs() if self.knowledge else []),
        }

    def report(self) -> ToolResult:
        snap = self.snapshot()
        lines = [f"E-COMMERCE INTELLIGENCE HUB — {snap['at'][:10]}  (brand: {snap['brand']})",
                 f"Scored product opportunities: {snap['opportunity_count']}"]
        for o in snap["opportunities"][:10]:
            score = f"{o['score']:.0f}/100" if o["score"] is not None else "unscored"
            lines.append(f"- {o['product']}: {score}")
        if snap["knowledge_packs"]:
            lines.append("Knowledge packs: " + ", ".join(p["title"] for p in snap["knowledge_packs"]))
        self.bus.dashboard_event.emit("commerce_hub", snap)
        return ToolResult("\n".join(lines))
