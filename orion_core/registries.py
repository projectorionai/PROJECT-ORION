"""
System registries (Phase 3; widened in the Mark XX architectural-audit
pass, Track G) — the map of what ORION can do.

Three queryable registries, populated once at app wiring, so any layer
(ExecutiveCore, Forge, diagnostics, future implementers) can check what
already exists BEFORE building anything new:

    CapabilityRegistry — every dispatchable tool (from the handler table)
                         plus forged tools as they activate
    ModuleRegistry     — every constructed service/engine, its role,
                         dependencies, version and (when a Telemetry ref is
                         supplied) its LIVE health
    FeatureRegistry    — completed feature passes and their scope, so work
                         is never rebuilt

Audit finding this widens: ModuleRegistry previously stored only
role/type/available, was hand-populated from one 16-item tuple in app.py,
and had no health, dependency or version fields at all. Health is
deliberately NOT reinvented here as a second, parallel concept — Telemetry
already runs a real, self-registered HealthRegistry (~30+ components beat
their own status there). ModuleRegistry.describe()/all_described() simply
cross-reference that live data by name rather than tracking a second,
static copy that could drift from the truth. The module list in app.py
also grew substantially (roughly doubled) to cover the subsystems the audit
named as missing — memory, the provider router, the dispatcher, the bus,
workflow_engine, vision, desktop control, Outlook/Notion, the agent
manager, and more — though "every subsystem" would mean touching dozens of
constructors across the tree to self-register at construction time; this
pass widens the curated list rather than converting every module to
call registries.modules.register(self) on itself, which is a larger,
separately-scoped change.

Pure data, no behaviour, no persistence — the source of truth remains the
dispatcher and app wiring; these are the queryable index over it.
"""

from __future__ import annotations

from typing import Any

from .data import ToolResult


class CapabilityRegistry:
    """Every dispatchable tool name → short description."""

    def __init__(self) -> None:
        self._capabilities: dict[str, str] = {}

    def register(self, name: str, description: str = "") -> None:
        name = str(name or "").strip()
        if name:
            self._capabilities[name] = str(description or "")[:200]

    def register_from_dispatcher(self, dispatcher: Any,
                                 declarations: list[dict[str, Any]] | None = None) -> int:
        described = {str(d.get("name")): str(d.get("description") or "")[:200]
                     for d in (declarations or [])}
        for name in dispatcher.handler_table():
            self.register(name, described.get(name, ""))
        return len(self._capabilities)

    def has(self, name: str) -> bool:
        return str(name or "").strip() in self._capabilities

    def all(self) -> dict[str, str]:
        return dict(self._capabilities)

    def search(self, query: str) -> list[str]:
        query = str(query or "").lower().strip()
        if not query:
            return sorted(self._capabilities)
        return sorted(n for n, d in self._capabilities.items()
                      if query in n.lower() or query in d.lower())


class ModuleRegistry:
    """Every constructed service → its role, dependencies, version and
    availability. Health is cross-referenced live from Telemetry.health
    rather than stored here — see module docstring."""

    def __init__(self) -> None:
        self._modules: dict[str, dict[str, Any]] = {}

    def register(self, name: str, service: Any, role: str = "",
                 dependencies: list[str] | None = None, version: str = "") -> None:
        name = str(name or "").strip()
        if not name:
            return
        self._modules[name] = {
            "role": str(role or "")[:200],
            "type": type(service).__name__ if service is not None else "None",
            "available": service is not None,
            "dependencies": [str(d).strip() for d in (dependencies or []) if str(d).strip()],
            "version": str(version or ""),
        }

    def get(self, name: str) -> dict[str, Any] | None:
        return self._modules.get(str(name or "").strip())

    def all(self) -> dict[str, dict[str, Any]]:
        return dict(self._modules)

    def available(self) -> list[str]:
        return sorted(n for n, m in self._modules.items() if m["available"])

    @staticmethod
    def _health_lookup(telemetry: Any | None) -> dict[str, str]:
        if telemetry is None or getattr(telemetry, "health", None) is None:
            return {}
        try:
            return {row["name"]: row["status"] for row in telemetry.health.snapshot()}
        except Exception:
            return {}

    def health_for(self, name: str, telemetry: Any | None = None) -> str:
        """The module's live status from Telemetry.health, or 'UNKNOWN' when
        no Telemetry is supplied or the module never registered a heartbeat
        there (not every module beats — event-driven ones may simply never
        have called telemetry.health.register(name))."""
        return self._health_lookup(telemetry).get(str(name or "").strip(), "UNKNOWN")

    def describe(self, name: str, telemetry: Any | None = None) -> dict[str, Any] | None:
        """The full record for one module, including live health."""
        record = self.get(name)
        if record is None:
            return None
        full = dict(record)
        full["health"] = self.health_for(name, telemetry)
        return full

    def all_described(self, telemetry: Any | None = None) -> dict[str, dict[str, Any]]:
        health = self._health_lookup(telemetry)
        return {
            name: {**record, "health": health.get(name, "UNKNOWN")}
            for name, record in self._modules.items()
        }


class FeatureRegistry:
    """Completed feature passes — checked before rebuilding anything."""

    def __init__(self) -> None:
        self._features: dict[str, str] = {}

    def register(self, name: str, scope: str) -> None:
        name = str(name or "").strip()
        if name:
            self._features[name] = str(scope or "")[:300]

    def has(self, name: str) -> bool:
        return str(name or "").strip() in self._features

    def all(self) -> dict[str, str]:
        return dict(self._features)


class SystemRegistries:
    """The three registries plus the dispatcher-facing report."""

    def __init__(self) -> None:
        self.capabilities = CapabilityRegistry()
        self.modules = ModuleRegistry()
        self.features = FeatureRegistry()

    def report(self, query: str = "", telemetry: Any | None = None) -> ToolResult:
        if query:
            matches = self.capabilities.search(query)
            if not matches:
                return ToolResult(f"No capability matching '{query}'.")
            lines = [f"Capabilities matching '{query}':"]
            caps = self.capabilities.all()
            lines.extend(f"- {m}" + (f" — {caps[m][:90]}" if caps[m] else "")
                         for m in matches[:15])
            return ToolResult("\n".join(lines))
        lines = [
            f"System registries: {len(self.capabilities.all())} capabilities, "
            f"{len(self.modules.available())}/{len(self.modules.all())} modules "
            f"available, {len(self.features.all())} feature passes recorded.",
        ]
        unavailable = [n for n, m in self.modules.all().items()
                       if not m["available"]]
        if unavailable:
            lines.append("Unavailable modules: " + ", ".join(sorted(unavailable)[:10]))
        if telemetry is not None:
            described = self.modules.all_described(telemetry)
            unhealthy = sorted(
                n for n, m in described.items() if m["health"] in {"DEGRADED", "DOWN"})
            if unhealthy:
                lines.append("Unhealthy modules: " + ", ".join(unhealthy[:10]))
        return ToolResult("\n".join(lines))


__all__ = ["CapabilityRegistry", "FeatureRegistry", "ModuleRegistry",
           "SystemRegistries"]
