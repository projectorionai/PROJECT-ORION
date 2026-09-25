"""
Dispatch domain — Agents, briefings, globe, telemetry, protocols, planning and executive tools.

Split out of the monolithic ``dispatcher.py`` (July 2026 improvement
pass, Priority 2.1). These handlers are mixed into ``OrionDispatcher``;
they run with the same ``self`` and are routed by its ``handler_table``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtWidgets import QApplication

from .constants import BASE_DIR, is_protected_path
from .data import ToolResult
from .security import SecuritySanitiser, SecurityViolation
from .utils import first_line
from . import background


def _busiest_provider(runway: Any) -> str:
    """The provider that has burned the most tokens today — the one worth
    governing. Empty when nothing has been used yet."""
    try:
        providers = runway._known_providers()
    except Exception:
        providers = []
    best, best_used = "", 0
    for name in providers:
        try:
            used = runway.used_today(name)
        except Exception:
            used = 0
        if used > best_used:
            best, best_used = name, used
    return best


class ProductivityDispatchMixin:
    """Agents, briefings, globe, telemetry, protocols, planning and executive tools."""

    def focus_tool(self, args: dict[str, Any]) -> ToolResult:
        """Deep-focus work blocks (Mark XXV): start a timed, intention-named block
        with a break to follow, log distractions, and measure the habit."""
        engine = getattr(self, "focus", None)
        if engine is None:
            from .focus import FocusEngine
            engine = FocusEngine()
            self.focus = engine
        from .focus import human_minutes
        action = str(args.get("action") or "status").lower().strip()

        if action in {"start", "begin", "go", "work"}:
            label = str(args.get("label") or args.get("task") or args.get("intention") or "")
            if not label:
                return ToolResult("What are we focusing on? Give the block an intention.",
                                  ok=False)
            minutes = args.get("minutes") or args.get("duration")
            s = engine.start(
                label=label, preset=str(args.get("preset") or args.get("cadence") or ""),
                minutes=int(minutes) if str(minutes or "").strip().isdigit() else None,
                break_minutes=(int(args["break_minutes"])
                               if str(args.get("break_minutes") or "").strip().isdigit() else None),
                kind=str(args.get("kind") or "deep"),
                energy=(int(args["energy"]) if str(args.get("energy") or "").strip().isdigit() else None))
            return ToolResult(
                f'Focus on — "{s.label}" for {human_minutes(s.planned_minutes)}, '
                f"then a {human_minutes(s.break_minutes)} break. I'll hold the line; "
                "tell me if something breaks it.")

        if action in {"status", "check", "how_long", "remaining"}:
            s = engine.active()
            if s is None:
                t = engine.today()
                return ToolResult(
                    f"No focus block running. Today: {t['completed']} completed, "
                    f"{t['focus_minutes']} minutes focused.")
            if s.is_break_due():
                over = human_minutes(s.elapsed_minutes() - s.planned_minutes)
                return ToolResult(
                    f'"{s.label}" — the {human_minutes(s.planned_minutes)} block is up '
                    f"(running {over} over). Good moment to stop and take your "
                    f"{human_minutes(s.break_minutes)} break.")
            return ToolResult(
                f'"{s.label}" — {human_minutes(s.remaining_minutes())} left'
                + (f", {s.interruptions} interruption(s) so far." if s.interruptions else "."))

        if action in {"interrupt", "distraction", "distracted", "broke"}:
            s = engine.interrupt()
            if s is None:
                return ToolResult("No focus block running to interrupt.")
            return ToolResult(f"Logged the distraction ({s.interruptions} this block). "
                              "Back to it when you can.")

        if action in {"done", "complete", "finish", "stop"}:
            energy = int(args["energy"]) if str(args.get("energy") or "").strip().isdigit() else None
            s = engine.complete(energy=energy, notes=str(args.get("notes") or ""))
            if s is None:
                return ToolResult("No focus block was running.")
            streak = engine.streak()
            return ToolResult(
                f'Done — {human_minutes(s.elapsed_minutes())} on "{s.label}". '
                f"Take your {human_minutes(s.break_minutes)} break. "
                f"That's a {streak}-day focus streak.")

        if action in {"cancel", "abandon", "abort"}:
            s = engine.cancel()
            return ToolResult("Focus block cancelled." if s else "Nothing to cancel.")

        if action in {"rhythm", "when", "best_time", "chronotype"}:
            from .insights import daily_rhythm, render_rhythm
            return ToolResult(render_rhythm(daily_rhythm(
                engine, days=int(args.get("days") or 90))))
        if action in {"stats", "progress", "summary"}:
            days = int(args["days"]) if str(args.get("days") or "").strip().isdigit() else 7
            st = engine.stats(days=days)
            return ToolResult(
                f"Last {st['days']} days: {st['sessions']} blocks, {st['completed']} completed "
                f"({st['completion_rate']}%), {st['focus_minutes']} minutes focused, "
                f"{st['interruptions']} distractions. {st['streak']}-day streak.")

        return ToolResult(f"Unsupported focus action: {action}.", ok=False)

    def decision_tool(self, args: dict[str, Any]) -> ToolResult:
        """Decision journal + calibration (Mark XXVI): record a judgement with its
        reasoning and confidence BEFORE the outcome is known, review it later, and
        find out whether the confidence was justified."""
        engine = getattr(self, "decisions", None)
        if engine is None:
            from .decisions import DecisionJournal
            engine = DecisionJournal()
            self.decisions = engine
        action = str(args.get("action") or "due").lower().strip()

        def _ident() -> int | None:
            raw = str(args.get("id") or "").strip().lstrip("#")
            return int(raw) if raw.isdigit() else None

        if action in {"record", "log", "add", "new", "decide"}:
            try:
                decision = engine.record(
                    str(args.get("question") or args.get("decision") or ""),
                    str(args.get("choice") or args.get("option") or ""),
                    confidence=int(args.get("confidence") or 70),
                    prediction=str(args.get("prediction") or ""),
                    rationale=str(args.get("rationale") or args.get("why") or ""),
                    review_days=int(args.get("review_days") or 30),
                    tags=[t.strip() for t in str(args.get("tags") or "").split(",") if t.strip()])
            except (ValueError, TypeError) as exc:
                return ToolResult(f"I need both the decision and what you chose. {exc}",
                                  ok=False)
            return ToolResult(
                f"Logged decision #{decision.id}: {decision.choice} "
                f"({decision.confidence}% confident). I'll bring it back on "
                f"{decision.review_at[:10]} to see how it went.")

        if action in {"due", "review", "pending"}:
            due = engine.due()
            if not due:
                open_count = len(engine.open_decisions())
                return ToolResult(
                    f"No decisions due for review. {open_count} still open."
                    if open_count else "No decisions logged yet.")
            lines = []
            for decision in due[:8]:
                lines.append(f"  {decision.summary()}")
                if decision.prediction:
                    lines.append(f"     you predicted: {decision.prediction}")
            return ToolResult(
                f"{len(due)} decision(s) ready to review — what actually happened?\n"
                + "\n".join(lines)
                + "\n\nTell me: decision resolve id=<n> outcome=right|wrong|mixed")

        if action in {"resolve", "outcome", "close"}:
            ident = _ident()
            if ident is None:
                return ToolResult("Which decision? Give its id.", ok=False)
            try:
                decision = engine.resolve(
                    ident, str(args.get("outcome") or ""),
                    actual=str(args.get("actual") or args.get("result") or ""),
                    lesson=str(args.get("lesson") or ""))
            except ValueError as exc:
                return ToolResult(str(exc), ok=False)
            if decision is None:
                return ToolResult(f"I have no decision #{ident}.", ok=False)
            calibration = engine.calibration()
            tail = ""
            if calibration["brier"] is not None:
                tail = (f" Across {calibration['scored']} scored decision(s) your "
                        f"Brier score is {calibration['brier']}.")
            return ToolResult(
                f"Recorded: #{decision.id} came out {decision.outcome.upper()}."
                + (f" Lesson noted." if decision.lesson else "") + tail)

        if action in {"defer", "postpone", "later"}:
            ident = _ident()
            decision = engine.defer(ident, int(args.get("days") or 30)) if ident else None
            if decision is None:
                return ToolResult("Which open decision should I push back?", ok=False)
            return ToolResult(f"Pushed #{decision.id} back to "
                              f"{decision.review_at[:10]}.")

        if action in {"calibration", "score", "how_am_i_doing"}:
            calibration = engine.calibration()
            if not calibration["scored"]:
                return ToolResult(
                    "Nothing scored yet — resolve a few decisions and I can tell "
                    "you whether your confidence is justified.")
            lines = [f"Calibration over {calibration['scored']} resolved decision(s):",
                     f"  Brier score : {calibration['brier']} (0 is perfect, 0.25 is a coin flip)",
                     f"  Hit rate    : {calibration['hit_rate']}%"]
            for band in calibration["bins"]:
                lines.append(f"  said {band['stated']}% ({band['n']}x) → actually "
                             f"{band['actual']}%  [{band['gap']:+d}]")
            lines.append("")
            lines.append(calibration["verdict"])
            return ToolResult("\n".join(lines))

        if action in {"lessons", "learned"}:
            lessons = engine.lessons()
            if not lessons:
                return ToolResult("No lessons recorded yet.")
            return ToolResult("What past decisions taught you:\n"
                              + "\n".join(f"  - {line}" for line in lessons))

        if action in {"list", "open", "all"}:
            open_decisions = engine.open_decisions()
            if not open_decisions:
                return ToolResult("No open decisions.")
            return ToolResult(f"{len(open_decisions)} open decision(s):\n"
                              + "\n".join(f"  {d.summary()}" for d in open_decisions[:10]))

        if action in {"stats", "summary"}:
            st = engine.stats()
            return ToolResult(
                f"{st['total']} decision(s) logged - {st['open']} open, "
                f"{st['resolved']} resolved ({st['right']} right, {st['wrong']} wrong, "
                f"{st['mixed']} mixed), {st['due']} due for review. Average stated "
                f"confidence {st['average_confidence']}%.")

        return ToolResult(f"Unsupported decision action: {action}.", ok=False)

    def finance_tool(self, args: dict[str, Any]) -> ToolResult:
        """Offline personal/business finance (Mark XXVI): ledger + runway."""
        engine = getattr(self, "finance", None)
        if engine is None:
            from .finance import FinanceEngine
            engine = FinanceEngine()
            self.finance = engine
        action = str(args.get("action") or "report").lower().strip()

        if action in {"account", "add_account"}:
            a = engine.add_account(
                name=str(args.get("name") or "Main"), kind=str(args.get("kind") or "current"),
                opening_balance=float(args.get("balance") or args.get("opening_balance") or 0))
            tail = f", opening {a.opening_balance:.2f}" if a.opening_balance else ""
            return ToolResult(f"Added {a.kind} account '{a.name}'{tail}.")

        if action in {"txn", "spend", "expense", "income", "add_txn"}:
            if args.get("amount") is None:
                return ToolResult("How much, and on what?", ok=False)
            direction = ("in" if action == "income"
                         or str(args.get("direction") or "").lower() in {"in", "income"} else "out")
            t = engine.add_txn(
                float(args["amount"]), direction, account=str(args.get("account") or ""),
                category=str(args.get("category") or ""),
                merchant=str(args.get("merchant") or args.get("note") or ""))
            verb = "Income" if direction == "in" else "Spent"
            on = f" on {t.merchant}" if t.merchant else ""
            return ToolResult(f"{verb} {abs(float(args['amount'])):.2f}{on}. "
                              f"Cash now {engine.liquid_cash():.2f}.")

        if action in {"balance", "cash"}:
            acc = str(args.get("account") or "")
            if acc:
                return ToolResult(f"{acc}: {engine.balance(acc):.2f}")
            return ToolResult(f"Liquid cash: {engine.liquid_cash():.2f} across "
                              f"{len(engine.store.accounts())} account(s).")

        if action == "runway":
            r = engine.runway_months()
            burn = engine.monthly_net_burn()
            if r is None:
                return ToolResult(f"Not burning cash — income covers the outgoings. "
                                  "Runway is effectively unlimited at the current rate.")
            return ToolResult(f"Runway: {r} months — {engine.liquid_cash():.2f} cash at a "
                              f"burn of {burn:.2f}/month.")

        if action in {"subscription", "add_subscription", "sub"}:
            if not str(args.get("name") or "") or args.get("amount") is None:
                return ToolResult("A subscription needs a name and an amount.", ok=False)
            s = engine.add_subscription(str(args["name"]), float(args["amount"]),
                                        cadence_days=int(args.get("cadence_days") or 30))
            return ToolResult(f"Tracking '{s.name}' at {s.amount:.2f} every {s.cadence_days} days.")

        if action in {"upcoming", "renewals"}:
            ups = engine.upcoming_subscriptions(days=int(args.get("days") or 14))
            if not ups:
                return ToolResult("No subscriptions renewing in that window.")
            return ToolResult("Renewing soon:\n" + "\n".join(
                f"  {s.name}: {s.amount:.2f}" for s in ups))

        if action in {"import", "import_csv"}:
            text = str(args.get("text") or args.get("csv") or "")
            path = str(args.get("path") or "")
            if not text and path:
                try:
                    text = Path(path).read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    return ToolResult(f"Couldn't read {path}: {first_line(exc, 120)}", ok=False)
            n = engine.import_csv(text, account=str(args.get("account") or ""))
            return ToolResult(f"Imported {n} transactions. Cash now {engine.liquid_cash():.2f}.")

        r = engine.report()
        runway = f"{r['runway_months']} months" if r["runway_months"] is not None else "unlimited (not burning)"
        return ToolResult(f"Cash {r['cash']:.2f} · burn {r['monthly_burn']:.2f}/mo · runway "
                          f"{runway} · {r['subscriptions']} subs "
                          f"({r['monthly_subscriptions']:.2f}/mo).")

    def wellbeing_tool(self, args: dict[str, Any]) -> ToolResult:
        """Offline wellbeing check-ins, trends, and correlation with focus (XXVI)."""
        engine = getattr(self, "wellbeing", None)
        if engine is None:
            from .wellbeing import WellbeingEngine
            engine = WellbeingEngine()
            self.wellbeing = engine
        action = str(args.get("action") or "checkin").lower().strip()

        if action in {"checkin", "log"}:
            factors = args.get("factors")
            if isinstance(factors, str):
                factors = [f.strip() for f in factors.split(",") if f.strip()]
            c = engine.checkin(
                energy=args.get("energy"), mood=args.get("mood"), stress=args.get("stress"),
                sleep_hours=args.get("sleep_hours") or args.get("sleep"),
                note=str(args.get("note") or ""), factors=factors)
            bits = []
            if c.energy is not None:
                bits.append(f"energy {c.energy}/5")
            if c.mood_valence is not None:
                bits.append(f"mood {c.mood_valence:+d}")
            if c.sleep_hours is not None:
                bits.append(f"slept {c.sleep_hours:g}h")
            return ToolResult("Logged" + (": " + ", ".join(bits) if bits else " a check-in") + ".")

        if action == "today":
            t = engine.today()
            if not t["logged"]:
                return ToolResult("No check-in yet today — how's your energy, 1 to 5?")
            return ToolResult(f"Today: {t['checkins']} check-in(s), average energy "
                              f"{t['avg_energy']}/5.")

        if action == "trend":
            tr = engine.trend(days=int(args.get("days") or 14))
            if not tr["checkins"]:
                return ToolResult("Not enough check-ins yet to show a trend.")
            return ToolResult(
                f"Last {tr['days']}d ({tr['checkins']} check-ins): energy {tr['avg_energy']}/5 "
                f"({tr['energy_direction']}), mood {tr['avg_mood']}, sleep {tr['avg_sleep']}h, "
                f"stress {tr['avg_stress']}/5.")

        if action in {"patterns", "insights", "life", "cross"}:
            from .insights import collect_series, find_patterns, render
            series = collect_series(
                focus=getattr(self, "focus", None),
                wellbeing=engine,
                study=getattr(self, "study", None),
                finance=getattr(self, "finance", None),
                days=int(args.get("days") or 60))
            return ToolResult(render(find_patterns(series), series))
        if action in {"correlate", "correlation", "insight"}:
            cors = engine.correlate(focus=getattr(self, "focus", None),
                                    days=int(args.get("days") or 30))
            if not cors:
                return ToolResult("Not enough overlapping data yet to correlate wellbeing "
                                  "with your focus — keep logging both.")
            return ToolResult("Correlations (−1 to +1):\n" + "\n".join(
                f"  {c['pair']}: r={c['coefficient']} (n={c['n']})" for c in cors))

        return ToolResult(f"Unsupported wellbeing action: {action}.", ok=False)

    async def agent_dispatch(self, args: dict[str, Any]) -> ToolResult:
        return await self.agent_manager.dispatch(
            request=str(args.get("request") or args.get("query") or ""),
            agent_name=str(args.get("agent") or "auto"),
            context=str(args.get("context") or ""),
        )

    async def reason_tool(self, args: dict[str, Any]) -> ToolResult:
        """Deliberate on a hard question: panel, critique, verification (Track C).

        Distinct from ``agent_dispatch``, which is one specialist answering in
        one pass. This is the expensive path — several specialists working
        independently with read-only instruments, a red team attacking their
        drafts, a chair synthesising what survived, and the result checked
        against the evidence store — and the tier decides how much of that
        machinery actually convenes.
        """
        if self.reasoning is None:
            return ToolResult(
                "The reasoning engine is not available; answer in the general "
                "ORION capacity or use agent_dispatch for a single specialist.",
                ok=False,
            )
        question = str(args.get("question") or args.get("request")
                       or args.get("query") or "")
        action = str(args.get("action") or "").strip().lower()
        if action in {"last", "snapshot", "trace"}:
            snapshot = self.reasoning.snapshot()
            if not snapshot.get("ran"):
                return ToolResult("I have not reasoned through anything yet this session.")
            return ToolResult(
                f"Last deliberation — '{snapshot['question']}'\n"
                f"Tier: {snapshot['tier']} ({snapshot['rationale']})\n"
                f"Panel: {', '.join(snapshot['panel']) or 'none'}\n"
                f"Lookups: {snapshot['lookups']} · {snapshot['elapsed_s']}s\n"
                f"Verification: {snapshot['verification']}\n"
                + (f"\nRed team:\n{snapshot['critique']}" if snapshot.get("critique") else "")
            )
        outcome = await self.reasoning.reason(
            question,
            tier=str(args.get("tier") or args.get("effort") or ""),
            context=str(args.get("context") or ""),
            agent=str(args.get("agent") or ""),
        )
        return outcome.to_tool_result()

    async def strategy_tool(self, args: dict[str, Any]) -> ToolResult:
        """Search a decision space exhaustively rather than guessing at it (Track D).

        Distinct from ``reason``, which thinks hard about a question: this
        enumerates and scores thousands of concrete option combinations, keeps
        only what nothing else beats outright, and spends the model on the
        survivors. Use it when the decision genuinely has independent axes.
        """
        if self.strategy is None:
            return ToolResult(
                "The strategy engine is not available; use 'reason' to think "
                "the decision through instead.", ok=False)
        action = str(args.get("action") or "").strip().lower()
        if action in {"last", "snapshot"}:
            snapshot = self.strategy.snapshot()
            if not snapshot.get("ran"):
                return ToolResult("I have not searched a decision space yet this session.")
            return ToolResult(
                f"Last strategy search — '{snapshot['objective']}'\n"
                f"Dimensions: {', '.join(snapshot['dimensions'])}\n"
                f"Objectives: {', '.join(snapshot['objectives'])}\n"
                f"{snapshot['considered']:,} considered → {snapshot['survivors']:,} "
                f"feasible → {snapshot['frontier']} on the frontier "
                f"({snapshot['elapsed_s']}s)\n\n{snapshot['shortlist']}")
        report = await self.strategy.strategise(
            str(args.get("objective") or args.get("decision")
                or args.get("question") or ""),
            max_candidates=self._bounded_int(args.get("max_candidates"), 20000, 1000, 100000),
            top=self._bounded_int(args.get("top"), 5, 1, 12),
            judge=str(args.get("judge") or "").strip().lower() not in {"false", "no", "0"},
            context=str(args.get("context") or ""),
        )
        return report.to_tool_result()

    @staticmethod
    def _bounded_int(value: Any, fallback: int, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(value)))
        except (TypeError, ValueError):
            return fallback

    async def morning_briefing(self, args: dict[str, Any]) -> ToolResult:
        """The daily briefing — at whatever time of day it is asked for.

        The tool's NAME is historical. "Here is your morning briefing" at
        6:25 pm came from that name and from a button that literally sent
        "Give me my morning briefing." The overnight-catch-up treatment is
        only honoured in the morning; any other time it is the briefing for
        the part of the day it actually is, and the reply says so.
        """
        from .briefing import MorningBriefingService

        now_period = MorningBriefingService.greeting_period()
        period = str(args.get("period") or "general").strip().lower()
        if period == "morning" and now_period != "morning":
            period = "general"
        label = f"{now_period} briefing"
        if self.on_briefing_request is not None:
            background.spawn(self.on_briefing_request(period))
            return ToolResult(
                f"Your {label} is underway — composing the intelligence picture "
                f"now. (It is the {now_period}: call it your {label}, never a "
                f"morning briefing unless it is the morning.)")
        briefing = await self.briefing.compose_source_material(period=period)
        return ToolResult(briefing)

    def globe_tool(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").strip().lower()
        place = str(args.get("place") or args.get("location") or args.get("query") or "").strip()
        # Zoom / reset the camera without changing location.
        zoom_words = {"zoom_in": "in", "zoom in": "in", "in": "in", "closer": "in",
                      "zoom_out": "out", "zoom out": "out", "out": "out", "back": "out",
                      "reset": "reset", "reset_view": "reset", "orbit": "reset"}
        if action in zoom_words:
            direction = zoom_words[action]
            self.bus.globe_zoom.emit(direction)
            if direction == "reset":
                return ToolResult("Pulling the globe back out to the whole Earth.")
            return ToolResult(f"Zooming {direction} on the globe.")
        if not place:
            return ToolResult("Where should I take you on the globe? A city, "
                              "region, country or postcode all work.", ok=False)
        # Drive the GUI globe over the bus (the GlobeView geocodes + fetches news).
        # A postcode / ZIP resolves through the OSM geocoder just like a place name.
        self.bus.globe_request.emit(place)
        return ToolResult(f"Taking you to {place} on the globe — fetching the "
                          "regional news and footage now.")

    def max_zoom_in_globe_tool(self, args: dict[str, Any]) -> ToolResult:
        """Deterministic tool: drive the on-screen globe to its maximum zoom.

        This is a resolved, non-LLM action — it hands the request to the GUI
        globe over the bus, where a bounded, cancellable controller repeatedly
        issues the existing zoom-in step until the active backend (desktop
        Cesium or a browser-controlled globe) reaches its closest zoom.  Because
        it is registered here it executes without consulting any text provider;
        optional bounds (max_attempts, max_seconds, delay_between_s, tolerance,
        unchanged_threshold) may be supplied to tune the safeguards."""
        options: dict[str, Any] = {}
        for key in ("max_attempts", "max_seconds", "delay_between_s",
                    "unchanged_threshold", "tolerance", "blind_attempts"):
            if key in args and args[key] is not None:
                options[key] = args[key]
        self.bus.globe_max_zoom.emit(options)
        return ToolResult(
            "Maximising zoom on the globe now — flying the camera to the "
            "closest level the view supports.")

    def resource_status_tool(self, args: dict[str, Any]) -> ToolResult:
        """Report current system resource pressure without interrupting work.

        Read-only and non-destructive: it samples host/own-process load and
        returns a graduated assessment. It never terminates or throttles any
        process (there is deliberately no capability to do so).

        With ``action='trends'`` it instead reports the rolling history —
        metric trends over time and the tool-usage audit (Priority 3.1/3.4)."""
        if str(args.get("action") or "").lower().strip() in {"trends", "history", "usage", "audit"}:
            return self._resource_trends(args)
        from .resource_monitor import ResourceMonitor, SystemResourceReader
        reader = getattr(self, "_resource_reader", None)
        monitor = getattr(self, "_resource_monitor", None)
        if reader is None:
            reader = SystemResourceReader()
            self._resource_reader = reader
        if monitor is None:
            monitor = ResourceMonitor()
            self._resource_monitor = monitor
        if not reader.available():
            return ToolResult("Resource monitoring is unavailable (psutil not installed).", ok=False)
        reading = reader.read()
        if reading is None:
            return ToolResult("Could not read resource metrics on this host.", ok=False)
        procs = reader.top_processes(limit=5)
        assessment = monitor.observe(reading, processes=procs)
        parts = [
            f"CPU {reading.cpu_percent:.0f}%, physical memory (RAM) {reading.mem_percent:.0f}%",
        ]
        if reading.swap_percent is not None:
            parts.append(f"swap {reading.swap_percent:.0f}%")
        if reading.self_rss_mb is not None:
            parts.append(f"ORION working set {reading.self_rss_mb:.0f} MB")
        renderers = _webengine_children_mb()
        if renderers:
            # The face, brain and globe run in separate Chromium processes —
            # half of ORION's real footprint was missing from this report.
            parts.append(f"ORION's web renderers {renderers:.0f} MB")
        detail = "; ".join(parts)
        text = f"{assessment.message} [{detail}]"
        from . import resource_governor
        governor = resource_governor.GOVERNOR
        if governor is not None:
            status = governor.status()
            text += f"\nResource governor: {status['state']}"
            if status["shedding"]:
                text += " — easing off " + ", ".join(status["shedding"])
            text += "."
        return ToolResult(text, ok=True)

    def _resource_trends(self, args: dict[str, Any]) -> ToolResult:
        """Rolling-history view: metric trends over time + the tool-usage audit."""
        history = getattr(self.telemetry, "history", None) if self.telemetry else None
        if history is None:
            return ToolResult(
                "The rolling metrics history isn't enabled on this node.", ok=False)
        summary = history.summary()
        lines = [
            f"Rolling history: {summary['samples']} samples over "
            f"{summary['span_hours']}h across {summary['metrics']} metrics."
        ]
        usage = history.tool_usage(top=10)
        if usage:
            lines.append("\nMost-used tools (cumulative):")
            for row in usage:
                fails = f", {row['failures']} failed" if row.get("failures") else ""
                lines.append(f"  {row['tool']}: {row['calls']} calls{fails}")
        else:
            lines.append("\nNo tool usage recorded yet.")
        for metric in ("g.audio.playback_latency_ms", "t.tool.dispatch.p95",
                       "c.tool.calls", "c.remote.turns"):
            trend = history.trend(metric)
            if trend:
                lines.append(
                    f"\n{metric}: now {trend['last']}, "
                    f"range {trend['min']}–{trend['max']}, Δ{trend['delta']:+g} "
                    f"over {int(trend['count'])} points")
        return ToolResult("\n".join(lines), ok=True)

    def token_usage_tool(self, args: dict[str, Any]) -> ToolResult:
        """Report accurate token usage for configured models/APIs (Section 6).

        Read-only. Uses authoritative usage recorded by the provider router and
        versioned model metadata for context-window limits; unknown values are
        shown as 'Not reported by provider', never fabricated."""
        # "How much have I got left?" is the question people actually ask about
        # usage, and the ledger alone cannot answer it — it knows what was
        # spent, not what the limit is. See usage_runway.
        action = str(args.get("action") or "").lower().strip()
        if action in {"remaining", "runway", "left", "budget", "how_long"}:
            ledger = getattr(self, "token_ledger", None)
            if ledger is None:
                return ToolResult("No usage has been recorded yet.", ok=False)
            from .usage_runway import UsageRunway
            runway = UsageRunway(ledger)
            provider = str(args.get("provider") or "").strip()
            if provider:
                return ToolResult(runway.runway(provider).describe())
            return ToolResult(runway.report())
        if action in {"set_budget", "budget_set"}:
            ledger = getattr(self, "token_ledger", None)
            provider = str(args.get("provider") or "").strip()
            limit = args.get("daily_tokens") or args.get("limit")
            if ledger is None or not provider or not limit:
                return ToolResult(
                    "Tell me the provider and its daily token allowance.", ok=False)
            from .usage_runway import UsageRunway
            budget = UsageRunway(ledger).set_budget(provider, int(limit))
            return ToolResult(
                f"Noted — {budget.provider} has {budget.daily_tokens:,} tokens a day.")
        ledger = getattr(self, "token_ledger", None)
        if ledger is None:
            router = getattr(self, "ai", None)
            ledger = getattr(getattr(router, "router", None), "token_ledger", None)
        if ledger is None:
            return ToolResult("Token-usage telemetry is not attached.", ok=False)
        filters: dict[str, Any] = {}
        for key in ("provider", "model", "session_id", "task"):
            if args.get(key):
                filters[key] = args[key]
        summary = ledger.summary(filters or None)
        def fmt(v: Any) -> str:
            return "Not reported by provider" if v is None else str(v)
        lines = [
            "Token usage:",
            f"- Requests: {summary['requests']} ({summary['failed_requests']} failed)",
            f"- Input tokens: {fmt(summary['input_tokens'])}",
            f"- Output tokens: {fmt(summary['output_tokens'])}",
            f"- Cached input tokens: {fmt(summary['cached_input_tokens'])}",
            f"- Reasoning tokens: {fmt(summary['reasoning_tokens'])}",
            f"- Total tokens: {fmt(summary['total_tokens'])}",
            f"- Estimated cost (USD): {fmt(summary['estimated_cost_usd'])}",
        ]
        try:
            by_model = ledger.by_dimension("model", filters or None)
            if by_model:
                lines.append("By model:")
                for row in by_model[:8]:
                    lines.append(f"  • {row['key']}: {fmt(row['total_tokens'])} tokens, {row['requests']} req")
        except Exception:
            pass
        return ToolResult("\n".join(lines))

    async def protocol_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.protocols is None:
            return ToolResult("Protocols are not available.", ok=False)
        action = str(args.get("action") or "run").lower().strip()
        name = str(args.get("name") or args.get("protocol") or "")
        if action in {"run", "engage", "execute"}:
            return await self.protocols.run(name or str(args.get("query") or ""))
        if action in {"list", "protocols"}:
            return ToolResult(self.protocols.list_text())
        if action in {"create", "save", "define"}:
            steps = args.get("steps")
            if not isinstance(steps, list):
                return ToolResult("create requires a 'steps' list of {tool, args}.", ok=False)
            return self.protocols.create(name, steps, str(args.get("description") or ""))
        if action in {"delete", "remove"}:
            return self.protocols.delete(name)
        return ToolResult("Unsupported protocol action. Use run, list, create, or delete.", ok=False)

    def reminder_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.reminders is None:
            return ToolResult("Reminders are not available.", ok=False)
        action = str(args.get("action") or "add").lower().strip()
        if action in {"add", "set", "create", "remind"}:
            minutes = args.get("minutes")
            return self.reminders.add(
                text=str(args.get("text") or args.get("task") or ""),
                minutes=float(minutes) if minutes is not None else None,
                at=str(args.get("at") or ""),
                phrase=str(args.get("phrase") or args.get("query") or ""),
            )
        if action in {"list", "show"}:
            return self.reminders.list_text()
        if action in {"cancel", "clear", "delete"}:
            rid = args.get("id")
            return self.reminders.cancel(int(rid) if rid is not None else None)
        return ToolResult("Unsupported reminder action. Use add, list, or cancel.", ok=False)

    def sentinel_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.sentinel is None:
            return ToolResult("The system sentinel is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "report", "check"}:
            return self.sentinel.status()
        if action in {"enable", "on"}:
            self.sentinel.set_enabled(True)
            return ToolResult("Ambient monitoring enabled.")
        if action in {"disable", "off"}:
            self.sentinel.set_enabled(False)
            return ToolResult("Ambient monitoring disabled.")
        return ToolResult("Unsupported sentinel action. Use status, enable, or disable.", ok=False)

    async def audio_studio_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.audio_studio is None:
            return ToolResult("The audio studio is not available.", ok=False)
        action = str(args.get("action") or "index_assets").lower().strip()
        if action in {"index_assets", "index", "scan"}:
            return await self.audio_studio.index_assets()
        if action in {"process_vocal_take", "process", "normalise", "normalize"}:
            return await self.audio_studio.process_vocal_take(
                path=str(args.get("path") or args.get("file") or ""),
                target_dbfs=float(args["target_dbfs"]) if args.get("target_dbfs") is not None else None,
                convert_to=str(args.get("convert_to") or "wav"),
            )
        if action in {"export_stem_package", "export", "package"}:
            return await self.audio_studio.export_stem_package(str(args.get("name") or ""))
        return ToolResult(
            "Unsupported audio_studio action. Use index_assets, process_vocal_take, "
            "or export_stem_package.", ok=False,
        )

    def campaign_pipeline_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.pipeline is None:
            return ToolResult("The campaign pipeline is not available.", ok=False)
        action = str(args.get("action") or "get_pipeline_snapshot").lower().strip()
        ref = str(args.get("campaign") or args.get("name") or args.get("ref") or "")
        if action in {"get_pipeline_snapshot", "snapshot", "status", "board"}:
            return self.pipeline.snapshot_text()
        if action in {"create", "create_campaign", "new"}:
            return self.pipeline.create_campaign(
                name=str(args.get("name") or ""), brand=str(args.get("brand") or ""),
                value=float(args.get("value") or 0.0), deadline=str(args.get("deadline") or ""),
                notes=str(args.get("notes") or ""),
            )
        if action in {"update_stage", "stage", "move"}:
            return self.pipeline.update_stage(ref, str(args.get("stage") or ""))
        if action in {"log_performance", "log", "performance"}:
            return self.pipeline.log_performance(
                ref, str(args.get("metric") or "engagement"), float(args.get("value") or 0.0)
            )
        if action in {"schedule_content", "content", "schedule"}:
            return self.pipeline.schedule_content(
                ref, str(args.get("title") or ""), str(args.get("platform") or ""),
                str(args.get("scheduled") or ""),
            )
        if action in {"delete", "remove", "delete_campaign"}:
            return self.pipeline.delete_campaign(ref, bool(args.get("confirm")))
        return ToolResult(
            "Unsupported campaign_pipeline action. Use get_pipeline_snapshot, create, "
            "update_stage, log_performance, schedule_content, or delete.", ok=False,
        )

    async def autoplan_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.plan_executor is None:
            return ToolResult("The plan executor is not available.", ok=False)
        steps = args.get("steps")
        if not isinstance(steps, list):
            return ToolResult("autoplan requires a 'steps' list of {tool, args, on_fail}.", ok=False)
        return await self.plan_executor.execute(
            steps, objective=str(args.get("objective") or ""),
            max_retries=int(args.get("max_retries", 2)),
        )

    async def workspace_control(self, args: dict[str, Any]) -> ToolResult:
        if self.workspace is None:
            return ToolResult("Workspace manager is not available.", ok=False)
        action = str(args.get("action") or "snapshot").lower().strip()
        if action in {"snapshot", "capture"}:
            snap = await self.workspace.snapshot_workspace()
            context_note = f" — {snap.active_context}" if snap.active_context else ""
            return ToolResult(
                f"Workspace snapshot: {snap.summary()} "
                f"(active: {snap.active_window}{context_note}).")
        if action in {"save", "save_state"}:
            return await self.workspace.save_workspace_state(str(args.get("name") or ""))
        if action in {"restore", "restore_state", "resume"}:
            return await self.workspace.restore_workspace_state(str(args.get("name") or ""))
        if action in {"track", "changes", "track_changes"}:
            return await self.workspace.track_changes()
        if action in {"resume_context", "context"}:
            return ToolResult(self.memory.resume_context(str(args.get("project") or "")) or
                              "No prior context to resume.")
        if action in {"set_project", "project"}:
            name = self.memory.set_active_project(str(args.get("project") or args.get("name") or ""))
            return ToolResult(f"Active project set to '{name or 'none'}'.")
        return ToolResult(
            f"Unsupported workspace action: {action}. Use snapshot, save, restore, "
            "track_changes, resume_context, or set_project.",
            ok=False,
        )

    async def execute_plan(self, args: dict[str, Any]) -> ToolResult:
        """Agentic plan runner: execute steps, verify each one, report honestly.

        This was a second, weaker plan loop sitting beside ``autoplan`` — no
        verification, no retries, a hard cap of eight steps and no dependency
        support. Both now run through the one PlanExecutor, so ``execute_plan``
        inherits visual verification, bounded retries, dependency waves and
        concurrent execution of independent steps. The legacy inline loop
        remains only as the fallback for a runtime with no executor wired.
        """
        steps = args.get("steps") or []
        objective = str(args.get("objective") or "").strip()
        if not isinstance(steps, list) or not steps:
            return ToolResult("No plan steps supplied.", ok=False)
        if self.plan_executor is not None:
            return await self.plan_executor.execute(
                steps, objective=objective,
                max_retries=int(args.get("max_retries", 2)),
            )
        return await self._execute_plan_inline(steps, objective)

    async def _execute_plan_inline(self, steps: list[Any], objective: str) -> ToolResult:
        """Sequential fallback used only when no PlanExecutor is available."""
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
            if tool in {"execute_plan", "shutdown_orion", "restart_orion"}:
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
            line = result.text.splitlines()[0][:220] if result.text else ""
            report.append(f"{number}. {tool}: {'OK' if result.ok else 'FAILED'} - {line}")
        report.append(f"VERIFICATION: {succeeded} step(s) succeeded, {failed} failed.")
        return ToolResult("\n".join(report), ok=failed == 0)

    async def job_tool(self, args: dict[str, Any]) -> ToolResult:
        """Run read-only work detached from the conversation, and collect it later."""
        if self.jobs is None:
            return ToolResult("The background job manager is not available.", ok=False)
        action = str(args.get("action") or "list").lower().strip()

        if action in {"run", "start", "background", "detach"}:
            tool = str(args.get("tool") or args.get("tool_name") or "").strip()
            if not tool:
                return ToolResult(
                    "job(action='run') needs a 'tool' to background, e.g. "
                    "job(action='run', tool='research', args={'topic': '...'}).",
                    ok=False)
            tool_args = args.get("args") or args.get("tool_args") or {}
            if not isinstance(tool_args, dict):
                try:
                    tool_args = json.loads(str(tool_args))
                except Exception:
                    tool_args = {}
            outcome = self.jobs.start(tool, tool_args, label=str(args.get("label") or ""))
            if isinstance(outcome, ToolResult):
                return outcome
            return ToolResult(
                f"Started {outcome.id} in the background ({outcome.label}). "
                "Carry on — I'll tell you when it lands, or ask "
                f"job(action='result', job_id='{outcome.id}').")

        if action in {"list", "status", "jobs"}:
            job_id = str(args.get("job_id") or "").strip()
            if not job_id:
                return self.jobs.report()
            job = self.jobs.get(job_id)
            return (ToolResult(job.summary()) if job is not None
                    else ToolResult(f"No job matches '{job_id}'.", ok=False))

        if action in {"result", "collect", "output"}:
            return self.jobs.collect(str(args.get("job_id") or ""))

        if action in {"cancel", "stop", "abort"}:
            return self.jobs.cancel(str(args.get("job_id") or ""))

        return ToolResult(
            f"Unsupported job action: {action}. Use run, list, result, or cancel.",
            ok=False)

    async def proactive_check(self, args: dict[str, Any]) -> ToolResult:
        if self.proactive is None:
            return ToolResult("Proactive intelligence is not available.", ok=False)
        return await self.proactive.check_now()

    def ai_mode(self, args: dict[str, Any]) -> ToolResult:
        """Report the active intelligence mode, or recommend a model to use.

        'recommend' answers "what's the cheapest good model", "what can I run
        for free", "best model for coding" — a curated free-and-cheap shortlist
        (see model_advisor), which is what the user actually wants when they ask
        about models.
        """
        action = str(args.get("action") or "status").lower().strip()
        if action in {"recommend", "models", "cheapest", "which_model",
                      "suggest", "advise"}:
            from . import model_advisor
            task = str(args.get("task") or args.get("for") or args.get("query")
                       or ("cheapest" if action == "cheapest" else "")).strip()
            return ToolResult(model_advisor.advise(task))
        if action in {"governor", "runway", "budget", "should_i_switch",
                      "cost", "spend"}:
            return self._cost_governor_report(args)
        if self.ai is None:
            return ToolResult("Mode information is unavailable.", ok=False)
        return self.ai.report()

    def _cost_governor_report(self, args: dict[str, Any]) -> ToolResult:
        """CAP-02: given live usage and machine load, say whether to switch to a
        cheaper/free model and whether the CPU should hand off to the GPU."""
        from .cost_governor import CostGovernor
        from .usage_runway import UsageRunway
        # Same ledger discovery as token_usage_tool: prefer an attached ledger,
        # else the router's.
        ledger = getattr(self, "token_ledger", None)
        if ledger is None:
            router = getattr(self, "router", None)
            ledger = getattr(router, "token_ledger", None) or \
                getattr(getattr(router, "router", None), "token_ledger", None)
        if ledger is None:
            return ToolResult(
                "I can't see the token ledger yet, so I have no usage to govern. "
                "Ask me to 'recommend a cheap model' for the shortlist instead.",
                ok=False)
        provider = str(args.get("provider") or "").strip()
        if not provider:
            # Whichever provider has burned the most today is the one to govern.
            runway = UsageRunway(ledger)
            provider = _busiest_provider(runway) or "gemini"
        else:
            runway = UsageRunway(ledger)
        local = str(args.get("local") or args.get("host") or "").lower() in {
            "1", "true", "local", "yes", "on"}
        decision = CostGovernor(runway).assess(provider, local=local)
        return ToolResult(decision.message)

    def emotion_tool(self, args: dict[str, Any]) -> ToolResult:
        """Mark X.7: inspect or pin ORION's emotional rendering."""
        if self.emotion is None:
            return ToolResult("The emotion engine is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "current", "describe"}:
            desc = self.emotion.describe()
            return ToolResult(
                f"Emotional state: {desc['current']} (baseline {desc['baseline']}"
                + (f", pinned {desc['manual']}" if desc.get("manual") else "")
                + f"). Last sentiment: {desc['last_sentiment']['sentiment']} "
                f"@ {desc['last_sentiment']['confidence']:.2f}."
            )
        if action in {"set", "pin", "express"}:
            return ToolResult(self.emotion.set_emotion(str(args.get("name") or "")))
        if action in {"auto", "clear", "release"}:
            return ToolResult(self.emotion.set_emotion("auto"))
        return ToolResult(
            f"Unsupported emotion action: {action}. Use status, set (with name), or auto.",
            ok=False,
        )

    async def momentum_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.momentum is None:
            return ToolResult("The momentum engine is not available.", ok=False)
        action = str(args.get("action") or "focus").lower().strip()
        if action in {"standup", "stand_up", "status"}:
            return self.momentum.standup()
        if action in {"plan", "milestones", "roadmap"}:
            return await self.momentum.plan(
                str(args.get("project") or ""), str(args.get("goal") or args.get("text") or ""))
        return self.momentum.focus()

    async def executive_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.executive is None:
            return ToolResult("Executive assistant mode is not available.", ok=False)
        action = str(args.get("action") or "status").lower().strip()
        if action in {"status", "overview"}:
            return await self.executive.status()
        if action in {"prioritise", "prioritize", "priorities"}:
            return await self.executive.prioritise()
        if action in {"schedule", "book"}:
            return await self.executive.schedule(
                str(args.get("title") or args.get("text") or ""),
                str(args.get("when") or args.get("due") or ""),
                str(args.get("notes") or ""))
        if action in {"meeting_summary", "summarise_meeting", "minutes"}:
            return await self.executive.summarise_meeting(
                str(args.get("transcript") or args.get("text") or ""),
                str(args.get("title") or ""))
        if action in {"plan", "workflow", "plan_workflow"}:
            return await self.executive.plan_workflow(
                str(args.get("objective") or args.get("text") or ""))
        if action in {"progress", "monitor"}:
            return await self.executive.progress()
        if action in {"track", "track_project"}:
            return await self.executive.track_project(
                str(args.get("project") or args.get("name") or ""),
                str(args.get("notes") or ""))
        # ── Phase 3: ExecutiveCore actions ────────────────────────────────────
        if self.executive_core is not None:
            if action in {"challenge", "stress_test", "devil"}:
                return await self.executive_core.challenge(
                    str(args.get("decision") or args.get("idea")
                        or args.get("text") or ""),
                    str(args.get("context") or ""))
            if action in {"focus", "daily_focus"}:
                return await self.executive_core.focus()
            if action in {"recommend", "recommendations"}:
                return await self.executive_core.recommend()
            if action in {"goals", "review_goals"}:
                return await self.executive_core.review_goals()
            if action in {"blindspots", "blind_spots"}:
                return await self.executive_core.blind_spots()
        return ToolResult(
            f"Unsupported executive action: {action}. Use status, prioritise, "
            "schedule, meeting_summary, plan, progress, track, challenge, "
            "focus, recommend, goals, or blindspots.",
            ok=False,
        )

    # ── Phase 3: Personal AI Operating System tools ──────────────────────────

    def skill_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.skills is None:
            return ToolResult("The skills system is not available.", ok=False)
        action = str(args.get("action") or "list").lower().strip()
        name = str(args.get("name") or args.get("skill") or "")
        if action in {"list", "installed"}:
            return self.skills.list_skills()
        if action in {"describe", "info", "show"}:
            return self.skills.describe(name)
        if action in {"install", "add"}:
            return self.skills.install(str(args.get("source")
                                           or args.get("path") or ""))
        if action in {"remove", "uninstall"}:
            return self.skills.remove(name)
        if action in {"enable", "activate"}:
            return self.skills.set_enabled(name, True)
        if action in {"disable", "deactivate"}:
            return self.skills.set_enabled(name, False)
        if action in {"reload", "refresh"}:
            count = self.skills.reload()
            return ToolResult(f"Skill registry reloaded — {count} skill(s).")
        return ToolResult("Unsupported skill action. Use list, describe, "
                          "install, remove, enable, disable, or reload.", ok=False)

    async def plugin_tool(self, args: dict[str, Any]) -> ToolResult:
        """Manage ORION's CODE plugins (``skill`` handles data-only packages).

        This is the surface the plugin runtime never had: without it a plugin
        could only be managed by editing files on disk and restarting.
        """
        if self.plugins is None:
            return ToolResult("The plugin registry is not available.", ok=False)
        action = str(args.get("action") or "list").lower().strip()
        name = str(args.get("name") or args.get("plugin") or "").strip()

        if action in {"list", "installed", "plugins"}:
            return self.plugins.list_plugins()
        if action in {"describe", "info", "show", "about"}:
            return self.plugins.describe(name)
        if action in {"enable", "activate", "on"}:
            return self.plugins.set_enabled(name, True)
        if action in {"disable", "deactivate", "off"}:
            # Disabling must take effect NOW, not at the next restart: a plugin
            # with event hooks would otherwise keep reacting while the user
            # believes it is off. Its tool stays registered until the next
            # reload, but it stops receiving events immediately.
            bridge = getattr(self, "plugin_events", None)
            if bridge is not None:
                try:
                    bridge.forget(name)
                except Exception:
                    pass
            return self.plugins.set_enabled(name, False)
        if action in {"create", "new", "scaffold", "make"}:
            raw_events = args.get("events")
            if isinstance(raw_events, str):
                raw_events = [e.strip() for e in raw_events.split(",") if e.strip()]
            return self.plugins.create(
                name,
                str(args.get("description") or args.get("purpose") or ""),
                str(args.get("tier") or "confirm"),
                version=str(args.get("version") or "1.0.0"),
                kind=str(args.get("kind") or "tool"),
                events=raw_events,
            )
        if action in {"install", "add"}:
            return self.plugins.install(
                str(args.get("source") or args.get("path") or ""))
        if action in {"remove", "uninstall", "delete"}:
            bridge = getattr(self, "plugin_events", None)
            if bridge is not None:
                try:
                    bridge.forget(name)      # a removed plugin must stop reacting
                except Exception:
                    pass
            return self.plugins.remove(name)
        if action in {"doctor", "health", "check", "diagnose"}:
            return self.plugins.doctor(events=getattr(self, "plugin_events", None))
        if action in {"backfill", "manifests"}:
            written = self.plugins.backfill_manifests()
            remaining = self.plugins.manifestless()
            if written:
                text = (f"Wrote {len(written)} manifest(s) at the safe "
                        f"'confirm' tier: " + ", ".join(written))
            else:
                text = "No manifests needed writing, sir."
            if remaining:
                text += ("\nStill without one — they fail the plugin contract, "
                         "so run 'plugin doctor': " + ", ".join(remaining))
            return ToolResult(text)
        if action in {"export", "bundle", "share", "package"}:
            return self.plugins.export_bundle(
                name, str(args.get("source") or args.get("path") or ""))
        if action in {"unmute", "revive", "unsilence"}:
            bridge = getattr(self, "plugin_events", None)
            if bridge is None:
                return ToolResult("Plugin event hooks are not active this session.",
                                  ok=False)
            if not name:
                revived = [n for n, st in bridge.stats.items()
                           if st.muted and bridge.unmute(n)]
                if not revived:
                    return ToolResult("No plugin is muted, sir.")
                return ToolResult("Listening again to: " + ", ".join(sorted(revived))
                                  + ". If one keeps faulting or blocking, it will be "
                                  "muted again.")
            if bridge.unmute(name):
                return ToolResult(f"'{name}' is listening again.")
            return ToolResult(f"'{name}' was not muted.")
        if action in {"hooks", "events", "subscriptions"}:
            bridge = getattr(self, "plugin_events", None)
            if bridge is None:
                return ToolResult("Plugin event hooks are not active this session.",
                                  ok=False)
            return ToolResult(bridge.report())
        if action in {"audit", "permissions", "capabilities", "trust"}:
            return self.plugins.audit(name)
        if action in {"update", "check_update", "upgrade"}:
            return self.plugins.check_update(
                name, str(args.get("version") or args.get("source") or ""))
        if action in {"deps", "dependencies", "install_deps"}:
            return await self.plugins.install_dependencies(name)
        if action in {"reload", "refresh"}:
            return await self._reload_plugins()
        return ToolResult(
            "Unsupported plugin action. Use list, describe, enable, disable, "
            "create, install, remove, reload, deps, backfill, audit, update, export, hooks, unmute, or doctor.",
            ok=False)

    async def _reload_plugins(self) -> ToolResult:
        """Hot-reload every enabled plugin — no restart.

        Reuses the Forge's ReflectiveModuleLoader so a reloaded plugin stays
        visible in forge.health() rather than living in a second, parallel
        registry.
        """
        forge = getattr(self, "forge", None)
        loader = getattr(forge, "loader", None)
        if loader is None:
            return ToolResult(
                "The module loader is not available, so I cannot hot-reload "
                "plugins in this session.", ok=False)
        from .constants import CONFIG_DIR
        from .plugin_manifest import load_plugins
        try:
            plan = await load_plugins(
                loader, self, CONFIG_DIR / "custom_tools", self.bus,
                registry=self.plugins,
                # Without this a hot-reloaded plugin keeps working as a tool but
                # silently stops REACTING — its event hooks would be lost.
                events=getattr(self, "plugin_events", None))
        except Exception as exc:
            return ToolResult(f"Plugin reload failed: {exc}", ok=False)
        loaded = len(plan.loadable)
        skipped = len(plan.skipped)
        text = f"Reloaded plugins — {loaded} live"
        text += f", {skipped} held back." if skipped else "."
        if plan.skipped:
            text += "\n" + "\n".join(
                f"  ✗ {n}: {r}" for n, r in list(plan.skipped.items())[:8])
        return ToolResult(text)

    async def workflow_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.automation is None:
            return ToolResult("The automation engine is not available.", ok=False)
        return await self.automation.handle(args)

    async def creator_intel_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.creator_intel is None:
            return ToolResult("The creator intelligence suite is not "
                              "available.", ok=False)
        return await self.creator_intel.handle(args)

    async def briefing_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.briefing_engine is None:
            return ToolResult("The dynamic briefing engine is not "
                              "available.", ok=False)
        action = str(args.get("action") or "brief").lower().strip()
        specialist = {"research", "project", "mission", "missions",
                      "security", "opportunity", "opportunities"}
        if action in specialist:
            return await self.briefing_engine.brief(action)
        if action in {"brief", "now", "morning", "midday", "evening"}:
            period = action if action in {"morning", "midday", "evening"} \
                else str(args.get("period") or "")
            return await self.briefing_engine.brief(period)
        if action in {"check", "triggers", "alerts"}:
            return await self.briefing_engine.check_triggers()
        return ToolResult("Unsupported briefing action. Use brief, morning, "
                          "midday, evening, research, mission, security, "
                          "opportunity, or check.", ok=False)

    async def mission_tool(self, args: dict[str, Any]) -> ToolResult:
        if self.missions is None:
            return ToolResult("The mission engine is not available.", ok=False)
        return await self.missions.handle(args)

    async def avatar_tool(self, args: dict[str, Any]) -> ToolResult:
        """Avatar system control: state, face tracking, status."""
        action = str(args.get("action") or "status").lower().strip()
        if action in {"track", "face_tracking", "watch", "eyes_on"}:
            if self.face_tracker is None:
                return ToolResult("Face tracking is not available on this "
                                  "system.", ok=False)
            return await asyncio.to_thread(self.face_tracker.start)
        if action in {"untrack", "stop_tracking", "eyes_off"}:
            if self.face_tracker is None:
                return ToolResult("Face tracking is not available.", ok=False)
            return await asyncio.to_thread(self.face_tracker.stop)
        # "show me what you're looking at" — the live camera view. A GUI-thread
        # widget, so it is toggled directly rather than through to_thread.
        if action in {"show_camera", "camera_view", "show_view", "live_camera"}:
            window = getattr(self, "live_camera_window", None)
            if window is None:
                return ToolResult("The live camera view is not available.", ok=False)
            window.start()
            return ToolResult("Live camera view open — that's what I'm seeing.")
        if action in {"hide_camera", "close_camera", "hide_view"}:
            window = getattr(self, "live_camera_window", None)
            if window is None:
                return ToolResult("The live camera view is not available.", ok=False)
            window.stop()
            return ToolResult("Live camera view closed.")
        if self.avatar is None:
            return ToolResult("The avatar system is not available.", ok=False)
        if action in {"state", "set_state"}:
            state = str(args.get("state") or "")
            self.avatar.on_state(state)
            return ToolResult(f"Avatar state requested: {state or 'idle'}.")
        if action in {"notify", "notification"}:
            self.avatar.engine.notify(warning=False)
            return ToolResult("Avatar notification pulse fired.")
        if action in {"warn", "warning"}:
            self.avatar.engine.notify(warning=True)
            return ToolResult("Avatar warning state raised.")
        status = self.avatar.status()
        tracker = ""
        if self.face_tracker is not None:
            tracker_result = self.face_tracker.status()
            tracker = " " + getattr(tracker_result, "text", "")
        return ToolResult(
            f"Avatar: state {status['state']}"
            f"{'' if status['settled'] else ' (transitioning)'}, "
            f"face tracking {'on' if status['tracking'] else 'off'}."
            f"{tracker}")

    def capabilities_tool(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "")
        live = "" if query else self._live_self_description()
        if self.registries is None:
            return ToolResult("The system registries are not available."
                              + (f"\n\n{live}" if live else ""), ok=False)
        report = self.registries.report(query, telemetry=self.telemetry)
        if live:
            report = ToolResult(live + "\n\n" + report.text, ok=report.ok)
        return report

    def _live_self_description(self) -> str:
        """What ORION can do RIGHT NOW, read from live state — not from a
        boot-time snapshot or an old prompt. The registries are built once at
        startup, so MCP servers that connected later, forged tools and a voice
        channel that has since dropped were all invisible to "what can you
        do?". Each line is a fact checked now; a line that cannot be checked
        is left out rather than guessed."""
        lines = ["RIGHT NOW:"]
        worker = getattr(self, "live_worker", None)
        if worker is not None:
            if getattr(worker, "session", None) is not None:
                model = str(getattr(worker, "active_live_model", "") or "").rsplit("/", 1)[-1]
                lines.append(f"- Voice: Gemini Live connected{f' ({model})' if model else ''}.")
            else:
                lines.append("- Voice: Gemini Live is not connected; I'm on local voice and "
                             "the text models (I can still act through my tools).")
        router = getattr(getattr(self, "ai", None), "router", None)
        if router is not None:
            try:
                text = [p.name for p in router.text_profiles()]
                lines.append(f"- Thinking: {len(text)} text model provider(s) available"
                             + (f" ({', '.join(text[:5])})." if text else
                                " — none reachable at the moment."))
            except Exception:
                pass
            try:
                from .provider_capabilities import Capability, supports
                seeing = [p.name for p in router.text_profiles()
                          if supports(p, Capability.VISION) or "vision" in p.strengths]
                keyed = [p.name for p in router.settings.ordered_profiles()
                         if p.supports_live_audio and str(router.active_key(p) or "").strip()]
                lines.append("- Seeing: " + ("available (" + ", ".join((seeing + keyed)[:4]) + ")."
                                             if seeing or keyed else
                                             "the camera works, but no vision-capable model is configured."))
            except Exception:
                pass
        declarations = getattr(self, "TOOL_DECLARATIONS", None) or []
        mcp_routes = getattr(self, "_mcp_routes", {}) or {}
        forged = getattr(self, "_tool_handlers", {}) or {}
        by_server: dict[str, int] = {}
        for server, _tool in mcp_routes.values():
            by_server[server] = by_server.get(server, 0) + 1
        lines.append(f"- Tools: {len(declarations)} callable"
                     + (f", including {sum(by_server.values())} from MCP servers ("
                        + ", ".join(f"{k} {v}" for k, v in sorted(by_server.items())) + ")"
                        if by_server else "")
                     + (f" and {len(forged)} I forged myself" if forged else "") + ".")
        host = getattr(self, "mcp_host", None)
        if host is not None:
            try:
                dead = [name for name, conn in getattr(host, "servers", {}).items()
                        if hasattr(conn, "is_alive") and not conn.is_alive()]
                if dead:
                    lines.append("- MCP servers down: " + ", ".join(dead) + ".")
            except Exception:
                pass
        try:
            from . import resource_governor
            governor = resource_governor.GOVERNOR
            if governor is not None:
                status = governor.status()
                lines.append(f"- Resources: memory {status['memory_used_percent'] or '?'}% "
                             f"used, governor {status['state']}"
                             + (f" (easing off {', '.join(status['shedding'])})"
                                if status["shedding"] else "") + ".")
        except Exception:
            pass
        return "\n".join(lines) if len(lines) > 1 else ""

    def workflow_patterns_tool(self, args: dict[str, Any]) -> ToolResult:
        """Tool sequences the user keeps repeating — candidates worth saving
        as a named workflow.

        Reports only. Turning one into a real workflow goes through the
        ordinary `workflow` tool with the user's agreement; and because the
        detector records argument KEYS rather than values (so a rolling log
        of activity never holds real queries or paths), the actual step
        arguments have to be supplied at that point rather than replayed.
        """
        detector = getattr(self, "pattern_detector", None)
        if detector is None:
            return ToolResult("Pattern detection is not available.", ok=False)
        try:
            sequences = detector.detect_repeated_sequences(
                min_occurrences=max(2, int(args.get("min_occurrences") or 3)))
        except Exception as exc:
            return ToolResult(f"Could not analyse tool patterns: {exc}", ok=False)
        if not sequences:
            return ToolResult(
                "No repeated tool sequences noticed yet — I need to see the same "
                "run of tools a few times before it's worth naming.")
        lines = ["Repeated tool sequences I've noticed:"]
        for seq in sequences[:8]:
            lines.append(
                f"  {seq.describe()} — {seq.occurrences} times "
                f"(suggested name: {seq.suggested_name()})")
        lines.append(
            "Say the word and I'll save any of these as a workflow — I'll need "
            "the actual arguments for each step, which I deliberately don't retain.")
        return ToolResult("\n".join(lines))

    async def chess_tool(self, args: dict[str, Any]) -> ToolResult:
        """Play chess against ORION's own engine (or Stockfish, on request)."""
        if self.chess is None:
            return ToolResult("Chess is not available.", ok=False)
        action = str(args.get("action") or "board_state").lower().strip()
        if action in {"set_opponent", "opponent", "use_own_engine", "switch_engine"}:
            who = str(args.get("opponent") or args.get("who")
                      or args.get("engine") or "orion")
            return self.chess.set_opponent(who)
        if action in {"new_game", "new", "start"}:
            elo = args.get("elo")
            colour = str(args.get("player_colour") or args.get("colour")
                        or args.get("color") or "white")
            # An opponent named on new_game applies from here on, so "play me,
            # you play, not the computer" is one call rather than two.
            opponent = args.get("opponent") or args.get("versus")
            if opponent:
                self.chess.set_opponent(str(opponent))
            result = await self.chess.new_game(
                elo=int(elo) if elo is not None else None, player_colour=colour)
            # Starting a game is a complete interaction, not merely a backend
            # state change.  The board owns the experience, so route there
            # deterministically instead of relying on the model to issue a
            # second, often-generic Command Deck navigation call.
            if result.ok:
                try:
                    self.bus.gui_command.emit({"action": "chess", "target": ""})
                except Exception:
                    pass
            return result
        if action in {"move", "make_move", "play"}:
            move = str(args.get("move") or args.get("uci") or args.get("san") or "")
            if not move:
                return ToolResult("No move supplied.", ok=False)
            return await self.chess.make_move(move)
        if action in {"set_elo", "elo", "difficulty", "strength"}:
            elo = args.get("elo")
            if elo is None:
                return ToolResult("No ELO value supplied.", ok=False)
            return await self.chess.set_elo(int(elo))
        if action in {"resign", "give_up"}:
            return await self.chess.resign()
        if action in {"rating", "elo_rating", "my_rating", "your_rating", "stats"}:
            return self.chess.rating()
        if action in {"practice", "practise", "train", "self_play"}:
            return self.chess.start_practice(int(args.get("games") or 10))
        if action in {"stop_practice", "stop_practise", "stop_training"}:
            return self.chess.stop_practice()
        if action in {"analysis", "analyst", "analysis_engine", "set_analyst"}:
            return self.chess.set_analyst(str(args.get("engine") or args.get("opponent")
                                              or "orion"))
        if action in {"brain", "describe_brain", "learning"}:
            return ToolResult(self.chess.brain.describe())
        # Moving through the game by voice — "go back three moves", "show me
        # move 12", "back to the live position", "flip the board". The board
        # on screen follows (the service tells it over the bus).
        try:
            count = max(1, int(args.get("count") or 1))
        except (TypeError, ValueError):
            count = 1
        if action in {"back", "step_back", "previous", "prev", "undo_view"}:
            return self.chess.step(-count)
        if action in {"forward", "step_forward", "next"}:
            return self.chess.step(count)
        if action in {"goto", "goto_move", "show_move", "jump"}:
            move_number = args.get("move_number")
            if move_number is None:
                return ToolResult("Which move number should I show?", ok=False)
            return self.chess.goto_move(int(move_number),
                                        args.get("colour") or args.get("color"))
        if action in {"first", "beginning", "start_position", "goto_start"}:
            return self.chess.goto_ply(0)
        if action in {"live", "latest", "current", "return_to_live", "resume"}:
            return self.chess.resume_live()
        if action in {"flip", "flip_board", "rotate"}:
            return self.chess.flip()
        if action in {"history", "moves", "move_list"}:
            return self.chess.history_text()
        if action in {"analyse_move", "analyze_move", "explain_move"}:
            move_number = args.get("move_number")
            kwargs: dict[str, Any] = {
                "move_number": int(move_number) if move_number is not None else None}
            colour = args.get("colour") or args.get("color")
            if colour:
                kwargs["colour"] = str(colour)
            return await self.chess.analyse_move(**kwargs)
        if action in {"analyse_game", "analyze_game", "analyse", "analyze", "summary"}:
            return await self.chess.analyse_game()
        if action in {"board_state", "status", "board"}:
            state = self.chess.board_state()
            if not state.get("available"):
                return ToolResult("No chess game is active.", ok=False)
            history = ", ".join(state.get("history_san") or []) or "(no moves yet)"
            versus = (f"me — my own engine, rated {self.chess.brain.rating_text()}"
                      if getattr(self.chess, "opponent", "orion") == "orion"
                      else f"Stockfish at about {state['elo']} ELO")
            return ToolResult(
                f"{'Your' if state['active'] else 'Last'} game against {versus} — "
                f"{state['turn']} to move. "
                f"Moves: {history}. {state.get('status', '')}"
            )
        return ToolResult(
            f"Unsupported chess action: {action}. Use new_game, move, set_elo, "
            "set_opponent, resign, analyse_move, analyse_game, board_state, rating, "
            "practice, stop_practice, analysis, brain, back, forward, goto_move, "
            "first, live, flip or history.",
            ok=False,
        )


def _webengine_children_mb() -> float:
    """Resident memory of this process's QtWebEngine children, in MB (0 if none)."""
    try:
        import psutil
        total = 0.0
        for child in psutil.Process().children(recursive=True):
            try:
                if "QtWebEngine" in child.name():
                    total += child.memory_info().rss / 2**20
            except Exception:
                continue
        return total
    except Exception:
        return 0.0
