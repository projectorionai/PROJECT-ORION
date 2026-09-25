"""
HealthModel — one honest answer to "is ORION actually working?".

The brief asks for a unified health model over LLM, STT, TTS, microphone,
speaker, audio engine, OrionBus, MCP, memory, database, globe, Command Deck,
Command Palette, protocols, agent manager and network, with states
ONLINE / DEGRADED / OFFLINE / RECOVERING / UNKNOWN, and — importantly — that
"the UI should reflect actual state rather than decorative indicators".

That last clause is the design constraint that shapes this module.  Every probe
here reports what it can genuinely determine, and reports UNKNOWN when it
cannot.  UNKNOWN is a first-class answer, not a failure: an indicator that
shows green because nothing checked it is exactly the decorative indicator the
brief forbids.

Probes are:
  • cheap        — they read state that already exists; nothing here performs
                   network I/O or opens a device, so refreshing the panel can
                   never itself cause a stutter;
  • isolated     — a probe that raises is reported as UNKNOWN with the reason,
                   and never takes the snapshot down;
  • registered   — a subsystem that was never wired in is absent, not green.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

ONLINE = "ONLINE"
DEGRADED = "DEGRADED"
OFFLINE = "OFFLINE"
RECOVERING = "RECOVERING"
UNKNOWN = "UNKNOWN"

#: Worst-first, for rolling up an overall verdict.
_SEVERITY = {OFFLINE: 4, DEGRADED: 3, RECOVERING: 2, UNKNOWN: 1, ONLINE: 0}


@dataclass
class Probe:
    """One subsystem's health check."""

    name: str
    check: Callable[[], dict[str, Any]]
    critical: bool = False   # does its failure mean ORION is broken?


@dataclass
class Health:
    status: str = UNKNOWN
    detail: str = ""
    at: float = field(default_factory=time.time)

    def describe(self) -> dict[str, Any]:
        return {"status": self.status, "detail": self.detail, "at": self.at}


def _ok(detail: str = "") -> dict[str, Any]:
    return {"status": ONLINE, "detail": detail}


def _degraded(detail: str) -> dict[str, Any]:
    return {"status": DEGRADED, "detail": detail}


def _offline(detail: str) -> dict[str, Any]:
    return {"status": OFFLINE, "detail": detail}


def _unknown(detail: str) -> dict[str, Any]:
    return {"status": UNKNOWN, "detail": detail}


def _starting(detail: str) -> dict[str, Any]:
    """A subsystem that has not come up YET.

    Distinct from UNKNOWN, and the distinction matters on screen. UNKNOWN means
    "nobody could tell me"; a microphone engine that simply has not been started
    two seconds into boot is not unknown — we know precisely what it is. It was
    being reported as UNKNOWN, and because UNKNOWN outranks ONLINE in the
    roll-up, one not-yet-started subsystem dragged the NAV panel's whole
    "System Health" line to UNKNOWN even when everything else was fine. That is
    the "System health: Unknown" the user kept seeing.
    """
    return {"status": RECOVERING, "detail": detail}


class HealthModel:
    """Registry of subsystem probes plus a rendered snapshot."""

    def __init__(self, bus: Any = None) -> None:
        self.bus = bus
        self._probes: dict[str, Probe] = {}

    # ── registration ──────────────────────────────────────────────────────────

    def register(self, name: str, check: Callable[[], dict[str, Any]],
                 critical: bool = False) -> None:
        self._probes[name] = Probe(name, check, critical)

    def registered(self) -> list[str]:
        return sorted(self._probes)

    # ── reading ───────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Run every probe.  A probe that raises becomes UNKNOWN, never green."""
        out: dict[str, dict[str, Any]] = {}
        for name, probe in sorted(self._probes.items()):
            try:
                result = probe.check() or {}
                status = str(result.get("status") or UNKNOWN).upper()
                if status not in _SEVERITY:
                    status = UNKNOWN
                out[name] = {
                    "status": status,
                    "detail": str(result.get("detail") or "")[:200],
                    "critical": probe.critical,
                }
            except Exception as exc:
                out[name] = {
                    "status": UNKNOWN,
                    "detail": f"probe failed: {type(exc).__name__}: {exc}"[:200],
                    "critical": probe.critical,
                }
        return out

    def overall(self, snapshot: dict[str, dict[str, Any]] | None = None) -> str:
        snapshot = snapshot if snapshot is not None else self.snapshot()
        if not snapshot:
            return UNKNOWN
        critical = [v for v in snapshot.values() if v.get("critical")]
        pool = critical or list(snapshot.values())
        worst = max(pool, key=lambda v: _SEVERITY.get(v["status"], 1))
        return worst["status"]

    def problems(self, snapshot: dict[str, dict[str, Any]] | None = None
                 ) -> list[tuple[str, str, str]]:
        snapshot = snapshot if snapshot is not None else self.snapshot()
        return [(name, v["status"], v["detail"])
                for name, v in snapshot.items()
                if v["status"] in (OFFLINE, DEGRADED, RECOVERING)]

    def render(self, snapshot: dict[str, dict[str, Any]] | None = None) -> str:
        snapshot = snapshot if snapshot is not None else self.snapshot()
        lines = [f"ORION SYSTEM HEALTH — {self.overall(snapshot)}", ""]
        width = max((len(n) for n in snapshot), default=10)
        for name, value in snapshot.items():
            mark = {ONLINE: "OK", DEGRADED: "!!", OFFLINE: "XX",
                    RECOVERING: "~~", UNKNOWN: "??"}.get(value["status"], "??")
            line = f"  [{mark}] {name.ljust(width)}  {value['status']}"
            if value["detail"]:
                line += f" — {value['detail']}"
            lines.append(line)
        problems = self.problems(snapshot)
        lines += ["", (f"{len(problems)} subsystem(s) need attention."
                       if problems else "All registered subsystems nominal.")]
        return "\n".join(lines)

    def publish(self) -> dict[str, dict[str, Any]]:
        """Snapshot and push it to the UI as an event."""
        snapshot = self.snapshot()
        try:
            self.bus.dashboard_event.emit("health", {
                "overall": self.overall(snapshot), "subsystems": snapshot})
        except Exception:
            pass
        return snapshot

    # ── the standard ORION probe set ──────────────────────────────────────────

    def register_defaults(self, *, worker: Any = None, dispatcher: Any = None,
                          router: Any = None, memory: Any = None,
                          recovery: Any = None, protocols: Any = None,
                          deck: Any = None, command_router: Any = None,
                          mcp: Any = None, security: Any = None) -> None:
        """Wire the probes for whichever subsystems were actually supplied.

        A subsystem that is not passed in is simply not registered — the panel
        then shows nothing for it, rather than an indicator that is green
        because nobody ever checked.
        """
        self.register("OrionBus", lambda: (
            _ok("signal hub live") if self.bus is not None
            else _offline("no bus")), critical=True)

        if router is not None:
            def llm() -> dict[str, Any]:
                try:
                    if not router.text_available():
                        return _offline("no text provider is usable")
                    profiles = router.text_profiles()
                    names = ", ".join(getattr(p, "name", "?") for p in profiles[:3])
                    if getattr(router, "degraded", False):
                        return _degraded(f"degraded; available: {names}")
                    return _ok(f"{len(profiles)} provider(s): {names}")
                except Exception as exc:
                    return _unknown(f"provider state unreadable: {exc}")
            self.register("LLM", llm, critical=True)

        if worker is not None:
            self.register("LiveChannel", lambda: (
                _ok("connected") if getattr(worker, "connected", False)
                else _degraded("offline — local fallback in use")))

            def stt() -> dict[str, Any]:
                rec = getattr(worker, "recogniser", None)
                if rec is None:
                    return _unknown("no recogniser attached")
                return (_ok("local recogniser ready")
                        if getattr(rec, "available", False)
                        else _degraded("local recognition unavailable"))
            self.register("STT", stt)

            def mic() -> dict[str, Any]:
                engine = getattr(worker, "mic", None)
                if engine is None:
                    return _starting("capture engine still starting")
                if not getattr(worker, "microphone_enabled", False):
                    return _degraded("microphone disabled")
                if getattr(engine, "_stream", None) is None:
                    return _offline("no capture stream open")
                return _ok("capturing")
            self.register("Microphone", mic, critical=True)

            def tts() -> dict[str, Any]:
                speech = getattr(worker, "speech", None)
                if speech is None:
                    return _starting("voice stack still starting")
                engine = getattr(speech, "tts", None)
                backend = getattr(engine, "_active_backend", "") or "local"
                if engine is not None and not getattr(engine, "available", True):
                    return _offline("no voice engine available")
                return _ok(f"backend: {backend}")
            self.register("TTS", tts)

            def speaker() -> dict[str, Any]:
                speech = getattr(worker, "speech", None)
                playback = getattr(speech, "playback", None) if speech else None
                if playback is None:
                    return _starting("renderer still starting")
                if getattr(playback, "_stream", None) is None:
                    return _offline("no output stream open")
                if playback.held():
                    return _degraded("playback held")
                return _ok("output stream open")
            self.register("Speaker", speaker, critical=True)

        if recovery is not None:
            def audio_engine() -> dict[str, Any]:
                state = recovery.health()
                return {"status": state.get("status", UNKNOWN),
                        "detail": f"out {state.get('output')} / in {state.get('input')}"
                                  + (f" — {state['illegal']}" if state.get("illegal") else "")}
            self.register("AudioEngine", audio_engine, critical=True)

        if memory is not None:
            def memory_probe() -> dict[str, Any]:
                """Ask memory something only a working store can answer.

                This used to look for a ``conn``/``_conn`` attribute, which
                MemoryAgent has never exposed — so it ALWAYS returned UNKNOWN.
                Because Memory is a critical probe, that single wrong attribute
                name also dragged the whole NAV panel's System Health to
                UNKNOWN, which is exactly what the user was seeing on screen.
                Exercising the real public API is both correct and a better
                test: it proves the store answers, not merely that a handle
                exists.
                """
                try:
                    snapshot = memory.tiers_snapshot()
                except Exception as exc:
                    return _offline(f"memory store error: {exc}")
                if isinstance(snapshot, dict) and snapshot:
                    total = sum(v for v in snapshot.values() if isinstance(v, (int, float)))
                    tiers = len(snapshot)
                    return _ok(f"{int(total)} records across {tiers} tiers")
                return _ok("store responding")
            self.register("Memory", memory_probe, critical=True)

        # Security posture — the SECURITY ring read "NO DATA" purely because
        # nothing ever registered a probe for it, not because the sentinel was
        # down. An unregistered subsystem is invisible by design (see the module
        # docstring), so the fix is to register it, not to fake a value.
        if security is not None:
            def security_probe() -> dict[str, Any]:
                try:
                    if hasattr(security, "posture"):
                        posture = security.posture()
                        if isinstance(posture, dict):
                            alerts = int(posture.get("alerts", 0) or 0)
                            if alerts:
                                return _degraded(f"{alerts} open alert(s)")
                            return _ok(str(posture.get("summary") or "nominal")[:120])
                    monitoring = bool(getattr(security, "enabled", True))
                    alerts = list(getattr(security, "alerts", []) or [])
                    if not monitoring:
                        return _degraded("monitoring is switched off")
                    if alerts:
                        return _degraded(f"{len(alerts)} alert(s) on record")
                    return _ok("monitoring, no alerts")
                except Exception as exc:
                    return _unknown(f"posture unreadable: {exc}")
            self.register("Security", security_probe)

        if dispatcher is not None:
            self.register("AgentManager", lambda: (
                _ok(f"{len(getattr(dispatcher, 'declarations', []) or [])} tools")
                if dispatcher is not None else _unknown("")))

        if protocols is not None:
            def protocol_probe() -> dict[str, Any]:
                available = protocols.all_protocols()
                unbound = [n for n, s in available.items()
                           if s.get("native")
                           and s["native"] not in getattr(protocols, "_native", {})]
                if unbound:
                    return _degraded(f"{len(available)} protocols; "
                                     f"not wired: {', '.join(unbound)}")
                return _ok(f"{len(available)} protocols ready")
            self.register("Protocols", protocol_probe)

        if command_router is not None:
            self.register("CommandPalette", lambda: _ok(
                f"{len(command_router.commands())} executable commands"))

        if deck is not None:
            def deck_probe() -> dict[str, Any]:
                pages = deck.page_names()
                sidebar = getattr(deck, "_sidebar", None)
                if sidebar is not None and not sidebar.isHidden():
                    return _ok(f"{len(pages)} pages, sidebar navigation active")
                if getattr(deck, "swarm_navigation", False):
                    return _degraded(f"{len(pages)} pages; tabs hidden")
                return _ok(f"{len(pages)} pages, tab navigation active")
            self.register("CommandDeck", deck_probe)

            def globe_probe() -> dict[str, Any]:
                for name, widget in getattr(deck, "_pages", []):
                    if name != "GLOBE":
                        continue
                    if not getattr(widget, "_built", False):
                        return _unknown("not opened yet (renderer is lazy)")
                    if getattr(widget, "view", None) is None:
                        return _offline("renderer absent after build")
                    rebuilds = getattr(widget, "_rebuilds", 0)
                    if rebuilds:
                        return _degraded(f"recovered from {rebuilds} render crash(es)")
                    return _ok("renderer live")
                return _unknown("no globe page registered")
            self.register("Globe", globe_probe)

        if mcp is not None:
            def mcp_probe() -> dict[str, Any]:
                servers = getattr(mcp, "servers", None) or {}
                if not servers:
                    return _unknown("no MCP servers configured")
                return _ok(f"{len(servers)} server(s)")
            self.register("MCP", mcp_probe)

        def network() -> dict[str, Any]:
            if router is None:
                return _unknown("no connectivity source")
            try:
                return (_ok("online") if router.is_online()
                        else _degraded("offline — local capabilities only"))
            except Exception as exc:
                return _unknown(str(exc))
        self.register("Network", network)

        def traces() -> dict[str, Any]:
            from .request_trace import TRACES
            summary = TRACES.summary()
            if summary["stalled"]:
                return _degraded(f"{summary['stalled']} stalled request(s): "
                                 f"{summary['diagnosis']}")
            if summary["total"] == 0:
                return _unknown("no requests yet this session")
            return _ok(f"{summary['complete']}/{summary['total']} completed, "
                       f"mean {summary['mean_ms']:.0f} ms")
        self.register("RequestPipeline", traces)


__all__ = ["DEGRADED", "HealthModel", "OFFLINE", "ONLINE", "RECOVERING",
           "UNKNOWN", "Health", "Probe"]
