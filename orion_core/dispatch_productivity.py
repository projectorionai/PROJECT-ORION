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


def _storage_text() -> str:
    """Free space on each fixed drive: 'C: 120 GB free of 476 GB (75% used)'.
    Removable and optical drives are left out (an empty card reader is not
    'full'); '' when nothing can be read."""
    try:
        import psutil
        lines = []
        for part in psutil.disk_partitions(all=False):
            opts = str(part.opts or "").lower()
            if "cdrom" in opts or "removable" in opts or not part.fstype:
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                continue
            gb = 1024 ** 3
            lines.append(f"{part.mountpoint.rstrip(chr(92))} {usage.free / gb:,.0f} GB free of "
                         f"{usage.total / gb:,.0f} GB ({usage.percent:.0f}% used)")
        return ("Storage: " + "; ".join(lines) + ".") if lines else ""
    except Exception:
        return ""


def _battery_text() -> str:
    try:
        import psutil
        battery = psutil.sensors_battery()
    except Exception:
        battery = None
    if battery is None:
        return "This PC has no battery — it runs from the mains."
    plugged = "charging" if battery.power_plugged else "on battery"
    left = ""
    secs = battery.secsleft
    if not battery.power_plugged and isinstance(secs, (int, float)) and secs > 0:
        hours, rest = divmod(int(secs) // 60, 60)
        left = f", about {hours} h {rest} min left" if hours else f", about {rest} min left"
    return f"Battery at {battery.percent:.0f}%, {plugged}{left}."


def _timer_words(seconds: float) -> str:
    """'ten-minute', '30-second', '1 hour 30 minute' - the timer's own name."""
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours and not minutes and not secs:
        return f"{hours}-hour"
    if not hours and minutes and not secs:
        return f"{minutes}-minute"
    if not hours and not minutes:
        return f"{secs}-second"
    parts = ([f"{hours} hour"] if hours else []) + ([f"{minutes} minute"] if minutes else []) \
        + ([f"{secs} second"] if secs else [])
    return " ".join(parts)


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
                    confidence=int(args["confidence"] if args.get("confidence") is not None else 70),
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
        try:
            return ProductivityDispatchMixin._finance_tool(self, args)
        except (ValueError, TypeError, OverflowError) as exc:
            return ToolResult(f"Finance request couldn't be completed: {first_line(exc, 160)}", ok=False)

    def _finance_tool(self, args: dict[str, Any]) -> ToolResult:
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
            return ToolResult(f"{verb} {t.amount:.2f}{on}. "
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
                                        cadence_days=int(args["cadence_days"] if args.get("cadence_days") is not None else 30))
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

        if action not in {"report", "status", "summary", "overview"}:
            return ToolResult(f"Unsupported finance action: {action}.", ok=False)
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
                sleep_hours=(args["sleep_hours"] if args.get("sleep_hours") is not None else args.get("sleep")),
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

    async def goals_tool(self, args: dict[str, Any]) -> ToolResult:
        """Personal goals (goals.py): outcomes, measures, pace, plans. Every
        database touch runs on a worker thread — this is reached from voice."""
        from . import goals as goals_module
        engine = getattr(self, "goal_tracker", None)
        if engine is None:
            # A bare dispatcher (tests, a partial boot): the same store, with
            # whatever reminder service and model router already exist.
            momentum = getattr(self, "momentum", None)
            engine = await asyncio.to_thread(
                goals_module.GoalsEngine, reminders=getattr(self, "reminders", None),
                router=getattr(momentum, "router", None),
                companion=getattr(self, "companion", None),
                cognition=getattr(self, "cognition", None))
            self.goal_tracker = engine
        args = dict(args or {})
        if goals_module.resolve_action(args.get("action")) == "plan":
            return await goals_module.plan_goal(engine, args)
        return await asyncio.to_thread(goals_module.tool_action, engine, args)

    async def conversation_recording_tool(self, args: dict[str, Any]) -> ToolResult:
        from . import conversation_recorder
        return await conversation_recorder.handle(
            action=args.get("action", "status"), consent=args.get("consent", False),
            session_id=args.get("session_id", ""), query=args.get("query", ""),
            record_only=args.get("record_only", True))

    async def agent_dispatch(self, args: dict[str, Any]) -> ToolResult:
        # background=true detaches a read-only specialist run as a job that is
        # announced when it lands. A call already running inside a job (the
        # detached run itself, a plan, a workflow) stays in that job.
        from . import jobs as _jobs
        background = (str(args.get("background") or "").strip().lower()
                      in {"1", "true", "yes", "on"}
                      and _jobs.current_control() is None)
        # The model names the request several ways; "No request supplied"
        # failed two owner requests on 26-27 Oct (capability-gap ledger).
        request = next((str(args[key]) for key in ("request", "query", "task", "prompt",
                                                   "question", "message", "instruction",
                                                   "goal", "topic", "text")
                        if str(args.get(key) or "").strip()), "")
        if not request.strip():
            return ToolResult("No request supplied: call agent_dispatch again with the owner's "
                              "words in 'request'.", ok=False)
        return await self.agent_manager.dispatch(
            request=request,
            agent_name=str(args.get("agent") or "auto"),
            context=str(args.get("context") or ""),
            background=background,
            jobs=getattr(self, "jobs", None),
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

        section = str(args.get("section") or "").strip().lower().replace(" ", "_")
        if section in {"markets", "market", "stocks", "stock_market", "shares", "indices",
                       "crypto", "cryptocurrency", "bitcoin"}:
            return await self._markets_now(section)
        now_period = MorningBriefingService.greeting_period()
        period = str(args.get("period") or "general").strip().lower()
        if period == "morning" and now_period != "morning":
            period = "general"
        label = f"{now_period} briefing"
        if self.on_briefing_request is not None:
            # SINGLE FLIGHT. Each call used to spawn its own delivery, and 3 of
            # 10 sessions heard the briefing twice: the model called this again
            # while the first was being composed (often in reply to the
            # delivery's own prompt), or narrated one itself from "underway".
            # A request while one is composing or speaking joins it.
            running = getattr(self, "_briefing_delivery", None)
            if running is not None and not running.done():
                return ToolResult(
                    f"Your {label} is already on its way - it is being composed and "
                    "delivered right now. Do NOT start, summarise or narrate a briefing "
                    "yourself and do not call this tool again; at most say it is coming.",
                    receipt={"status": "already_delivering", "period": period})
            task = background.spawn(self.on_briefing_request(period), name="briefing delivery")
            try:
                self._briefing_delivery = task
            except AttributeError:
                pass
            return ToolResult(
                f"Your {label} is being composed now and the briefing system will deliver "
                "it to the user as its own message in a moment. Do NOT compose, summarise "
                "or narrate a briefing yourself, and do not call this tool again; just "
                "acknowledge briefly (for example 'coming right up'). (It is the "
                f"{now_period}: call it your {label}, never a morning briefing unless it "
                "is the morning.)", receipt={"status": "delivering", "period": period})
        briefing = await self.briefing.compose_source_material(period=period)
        return ToolResult(briefing)

    def briefing_in_flight(self) -> bool:
        """True while a requested briefing is being composed or delivered, so
        another way in (a typed "yes" to the startup offer) can join it instead
        of starting a second one."""
        running = getattr(self, "_briefing_delivery", None)
        return running is not None and not running.done()

    async def _markets_now(self, section: str) -> ToolResult:
        """'How's the stock market doing?' is one question, not a briefing:
        the four indices in the right tense (markets.py) and, unless only
        shares were asked about, the main cryptocurrencies - in a second."""
        from aiohttp import ClientSession, ClientTimeout
        from . import markets
        crypto_only = section in {"crypto", "cryptocurrency", "bitcoin"}
        shares_only = section in {"stocks", "stock_market", "shares", "indices"}
        parts: list[str] = []
        async with ClientSession(timeout=ClientTimeout(total=8.0, connect=4.0)) as session:
            if not crypto_only:
                quotes = await markets.fetch(session)
                parts.append(markets.summary(quotes) if quotes
                             else "The share prices aren't available right now.")
            crypto = getattr(getattr(self, "briefing", None), "_crypto_section", None)
            if not shares_only and crypto is not None:
                parts.append(await crypto(session))
        text = " ".join(p for p in parts if p).strip()
        ok = bool(text) and "unavailable" not in text.lower()
        return ToolResult(text or "Market prices aren't available right now.", ok=ok or len(parts) > 1)

    #: What the user may call each globe overlay or layer.
    _GLOBE_SHOWS = {
        "radar": "radar", "rain_radar": "radar", "weather_radar": "radar", "rain": "radar",
        "day_night": "daynight", "daynight": "daynight", "night": "daynight",
        "day_and_night": "daynight", "daylight": "daynight", "night_and_day": "daynight",
        "where_it_is_night": "daynight", "sun": "daynight",
        "terminator": "daynight", "city_lights": "daynight", "sunlight": "daynight",
        "measure": "measure", "distance": "measure", "ruler": "measure",
        "events": "events", "natural_events": "events", "disasters": "events",
        "wildfires": "events", "fires": "events", "storms": "events", "hurricanes": "events",
        "typhoons": "events", "volcanoes": "events", "floods": "events",
        "earthquakes": "earthquakes", "quakes": "earthquakes",
        "satellites": "satellites", "iss": "satellites", "space_station": "satellites",
        "tsunami": "tsunami", "tsunamis": "tsunami",
        "weather_warnings": "weather", "warnings": "weather",
    }
    _GLOBE_SPOKEN = {"radar": "the rain radar", "daynight": "day and night with city lights",
                     "measure": "distance measuring", "events": "natural events",
                     "earthquakes": "earthquakes", "satellites": "space stations",
                     "tsunami": "tsunami bulletins", "weather": "UK weather warnings"}

    def globe_tool(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").strip().lower()
        place = str(args.get("place") or args.get("location") or args.get("query") or "").strip()
        # Overlays and world layers: "show the rain radar", "hide the storms".
        show = str(args.get("show") or args.get("layer") or args.get("overlay") or "").strip().lower()
        hide = str(args.get("hide") or "").strip().lower()
        if action in {"show", "hide", "on", "off"} and not (show or hide) and place:
            show, place = place.lower(), ""
        target = (hide or show).replace(" ", "_").replace("-", "_")
        for article in ("the_", "a_", "an_", "live_"):
            target = target.removeprefix(article)
        if target:
            name = self._GLOBE_SHOWS.get(target) or self._GLOBE_SHOWS.get(target.removesuffix("s"))
            if name is None:
                return ToolResult(
                    "The globe can show: the rain radar, day and night, natural events "
                    "(fires, storms, volcanoes), earthquakes, space stations, tsunami "
                    "bulletins, UK weather warnings, and measure distances.", ok=False)
            on = not hide and action not in {"hide", "off"}
            self.bus.dashboard_event.emit("globe_overlay", {"name": name, "on": on})
            spoken = self._GLOBE_SPOKEN[name]
            if name == "measure" and on:
                return ToolResult("Measuring on the globe: click two or more points and I'll "
                                  "show the distance along the Earth's surface.")
            return ToolResult(f"{'Showing' if on else 'Hiding'} {spoken} on the globe.")
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

    async def resource_status_async(self, args: dict[str, Any]) -> ToolResult:
        """resource_status off the UI loop (it samples process CPU)."""
        return await asyncio.to_thread(self.resource_status_tool, args)

    def resource_status_tool(self, args: dict[str, Any]) -> ToolResult:
        """Report current system resource pressure without interrupting work.

        Read-only and non-destructive: it samples host/own-process load and
        returns a graduated assessment. It never terminates or throttles any
        process (there is deliberately no capability to do so).

        With ``action='trends'`` it instead reports the rolling history —
        metric trends over time and the tool-usage audit (Priority 3.1/3.4)."""
        action = str(args.get("action") or "").lower().strip()
        if action in {"trends", "history", "usage", "audit"}:
            return self._resource_trends(args)
        if action in {"storage", "disk", "disks", "space", "drives"}:
            return ToolResult(_storage_text() or "I couldn't read the drives.")
        if action in {"battery", "power"}:
            return ToolResult(_battery_text())
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
        storage = _storage_text()
        if storage:
            text += f"\n{storage}"
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
        if action in {"timer", "set_timer", "countdown", "start_timer"}:
            # A kitchen timer is a reminder with a length and no task:
            # "set a timer for ten minutes" has nothing to remember but the time.
            try:
                seconds = (float(args.get("seconds") or 0) + 60 * float(args.get("minutes") or 0)
                           + 3600 * float(args.get("hours") or 0))
            except (TypeError, ValueError):
                seconds = 0.0
            if seconds <= 0:
                return ToolResult("How long should the timer be?", ok=False)
            if seconds > 24 * 3600:
                return ToolResult("A timer can run for up to a day — for longer, set a reminder.",
                                  ok=False)
            label = str(args.get("text") or args.get("label") or "").strip()
            length = _timer_words(seconds)
            result = self.reminders.add(text=label or f"your {length} timer is up",
                                        minutes=seconds / 60.0)
            due = re.search(r"\(at (\d{1,2}:\d{2})\)", result.text or "")
            if result.ok and due:
                named = f" for the {label}" if label else ""
                return ToolResult(f"{length[0].upper()}{length[1:]} timer{named} started — "
                                  f"it goes off at {due.group(1)}.")
            return result
        if action in {"add", "set", "create", "remind"}:
            minutes = args.get("minutes")
            return self.reminders.add(
                text=str(args.get("text") or args.get("task") or ""),
                minutes=minutes,
                at=str(args.get("at") or ""),
                phrase=str(args.get("phrase") or args.get("query") or ""),
                channel=args.get("channel") or "",
                recipient=str(args.get("recipient") or ""),
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

    async def sing_tool(self, args: dict[str, Any]) -> ToolResult:
        """ORION sings (singing.handle): a short song in his own voice, tuned
        to a melody over a soft backing, played through his voice output.
        The rendering runs off the event loop inside singing.handle."""
        from . import singing
        return await singing.handle(self, args)

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
            if not result.ok and str(step.get("on_fail") or "").lower() not in {
                    "continue", "skip", "ignore", "retry"}:
                # The same default as PlanExecutor: a failed step that changes
                # something stops the plan; what follows assumed it worked.
                from .job_store import READ_ONLY, side_effect_class
                if side_effect_class(tool, raw_args if isinstance(raw_args, dict) else {}) \
                        != READ_ONLY:
                    report.append(f"STOPPED: step {number} ({tool}) failed, so the steps "
                                  "after it were not run.")
                    break
        report.append(f"VERIFICATION: {succeeded} step(s) succeeded, {failed} failed.")
        return ToolResult("\n".join(report), ok=failed == 0)

    def _running_research(self) -> str:
        """'Researching X - N of M minutes elapsed.' while a research run is
        going, else ''."""
        research = getattr(self, "research", None)
        task = (getattr(research, "_active", None) or {}).get("task")
        if task is None or task.done():
            return ""
        try:
            return research.status().text
        except Exception:
            return ""

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
                report = self.jobs.report()
                # Research runs on its own task, not as a job: "what are you
                # working on?" left out the research he said was running
                # (owner, 30 Oct 2026: "how do I know if he's actually still
                # doing something?").
                researching = self._running_research()
                if researching:
                    return ToolResult(f"{researching}\n{report.text}", ok=report.ok,
                                      receipt=report.receipt)
                return report
            job = self.jobs.get(job_id)
            return (ToolResult(job.summary()) if job is not None
                    else ToolResult(f"No job matches '{job_id}'.", ok=False))

        if action in {"result", "collect", "output"}:
            return self.jobs.collect(str(args.get("job_id") or ""))

        if action in {"cancel", "stop", "abort"}:
            return self.jobs.cancel(str(args.get("job_id") or ""))

        # Pause, resume and priority change WHEN work runs, never WHETHER a
        # side-effecting step is allowed: resuming an interrupted job still
        # stops at the job store's bound approval for any step that changes
        # something. Approving is deliberately not an action here — a tool
        # call can come from the model or from text it read, and an approval
        # must come from the user (the Tasks page or the phone's card).
        if action in {"pause", "hold"} and hasattr(self.jobs, "pause"):
            return self.jobs.pause(str(args.get("job_id") or ""))
        if action in {"resume", "continue", "unpause"} and hasattr(self.jobs, "resume"):
            return await self.jobs.resume(str(args.get("job_id") or ""))
        if (action in {"priority", "prioritise", "prioritize", "sooner"}
                and hasattr(self.jobs, "reprioritise")):
            return self.jobs.reprioritise(str(args.get("job_id") or ""),
                                          str(args.get("priority") or "user"))

        return ToolResult(
            f"Unsupported job action: {action}. Use run, list, result, cancel, "
            "pause, resume or priority.", ok=False)

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
        if action in {"set_brain", "switch_brain", "set_brain_mode", "use_brain",
                      "change_brain", "switch", "set_mode"}:
            return self._set_brain_mode(args)
        if action in {"brain", "brain_mode", "which_brain", "brain_status"}:
            return self._brain_mode_report()
        if self.ai is None:
            return ToolResult("Mode information is unavailable.", ok=False)
        return self.ai.report()

    def _brain_mode_report(self) -> ToolResult:
        """Which brain answers (brain_mode.py), and the local model it uses."""
        lines_of = getattr(self.ai, "brain_lines", None)
        if callable(lines_of):
            try:
                return ToolResult("\n".join(lines_of()))
            except Exception:
                pass
        from . import brain_mode
        return ToolResult(f"Brain mode: {brain_mode.describe()}")

    def _set_brain_mode(self, args: dict[str, Any]) -> ToolResult:
        """Switch between the cloud, hybrid and local brains — saved, and
        applied now when the live worker is attached (apply_brain_mode)."""
        from . import brain_mode
        wanted = brain_mode.normalise(args.get("mode") or args.get("brain")
                                      or args.get("target") or args.get("task"))
        if wanted is None:
            return ToolResult("Which brain should I use: cloud, hybrid or local?", ok=False)
        apply = getattr(self, "apply_brain_mode", None)
        if callable(apply):
            ok, message = apply(wanted)
            return ToolResult(message, ok=bool(ok))
        if brain_mode.set_mode(wanted) is None:
            return ToolResult("I couldn't save the brain setting.", ok=False)
        return ToolResult(f"Brain set to {brain_mode.LABELS[wanted]} — "
                          f"{brain_mode.EXPLANATIONS[wanted]} It takes full effect when "
                          "I next start.")

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
        action = str(args.get("action") or "status").lower().strip()
        if action in {"insights", "insight", "trend", "mood_trend", "history", "week"}:
            # Mark XXXIX: the emotional_insight_generator plugin, rebuilt -
            # how the user has been over days, from their own words.
            return self._mood_insights(args)
        if self.emotion is None:
            return ToolResult("The emotion engine is not available.", ok=False)
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
        if action in {"show", "display", "flash", "tour", "cycle", "show_all"}:
            # Held on his face for a moment, or one after another - his own
            # window, so only for the owner's turn (require_own_window).
            from .execution_policy import require_own_window
            blocked = require_own_window("gui_expression")
            if blocked is not None:
                return blocked
            if action in {"tour", "cycle", "show_all"}:
                message, ok = self.emotion.tour(args.get("names") or None)
            else:
                message, ok = self.emotion.show(str(args.get("name") or ""), args.get("seconds"))
            return ToolResult(message, ok=ok)
        return ToolResult(
            f"Unsupported emotion action: {action}. Use status, set (with name), show "
            "(with name), tour, insights, or auto.",
            ok=False,
        )

    async def _mood_insights(self, args: dict[str, Any]) -> ToolResult:
        """How the user has been over the last few days (mood_insights.py),
        read from memory off the event loop."""
        from . import mood_insights
        try:
            days = int(float(args.get("days") or 7))
        except (TypeError, ValueError):
            days = 7
        memory = getattr(self, "memory", None)
        try:
            text = await asyncio.to_thread(mood_insights.insights, memory, days)
        except Exception as exc:
            return ToolResult(f"I couldn't look back over your week - {first_line(exc)}.", ok=False)
        return ToolResult(text)

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
            return await asyncio.to_thread(self.plugins.list_plugins)
        if action in {"describe", "info", "show", "about"}:
            return await asyncio.to_thread(self.plugins.describe, name)
        if action in {"enable", "activate", "on"}:
            return await asyncio.to_thread(self.plugins.set_enabled, name, True)
        if action in {"disable", "deactivate", "off"}:
            result = await asyncio.to_thread(self.plugins.set_enabled, name, False)
            if not result.ok:
                return result
            bridge = getattr(self, "plugin_events", None)
            # Bound services use the saved preference gate and can resume
            # after re-enable. Legacy services must be detached explicitly.
            if bridge is not None and getattr(bridge, "registry", None) is not self.plugins:
                try:
                    bridge.forget(name)
                except Exception:
                    pass
            scheduler = getattr(self, "plugin_scheduler", None)
            if scheduler is not None and getattr(scheduler, "registry", None) is not self.plugins:
                scheduler.remove(name)
            return result
        if action in {"create", "new", "scaffold", "make"}:
            raw_events = args.get("events")
            if isinstance(raw_events, str):
                raw_events = [e.strip() for e in raw_events.split(",") if e.strip()]
            return await asyncio.to_thread(self.plugins.create,
                name,
                str(args.get("description") or args.get("purpose") or ""),
                str(args.get("tier") or "confirm"),
                version=str(args.get("version") or "1.0.0"),
                kind=str(args.get("kind") or "tool"),
                events=raw_events,
            )
        if action in {"install", "add"}:
            return await asyncio.to_thread(self.plugins.install,
                str(args.get("source") or args.get("path") or ""))
        if action in {"restore", "unquarantine", "restore_quarantined"} and callable(
                getattr(self.plugins, "restore_quarantined", None)):
            return await asyncio.to_thread(self.plugins.restore_quarantined, name)
        if action in {"remove", "uninstall", "delete"}:
            result = await asyncio.to_thread(self.plugins.remove, name)
            if not result.ok:
                return result
            removed_name = next((item["name"] for item in (result.evidence or [])
                                 if item.get("kind") == "plugin_removed" and isinstance(item.get("name"), str)), name)
            bridge = getattr(self, "plugin_events", None)
            if bridge is not None:
                try:
                    bridge.forget(removed_name)      # retire the actual registered callable identity
                except Exception:
                    pass
            from .plugin_registry import detach_runtime
            detach_runtime(self, removed_name)
            return result
        if action in {"doctor", "health", "check", "diagnose"}:
            return await asyncio.to_thread(self.plugins.doctor, events=getattr(self, "plugin_events", None))
        if action in {"backfill", "manifests"}:
            written, remaining = await asyncio.to_thread(
                lambda: (self.plugins.backfill_manifests(), self.plugins.manifestless()))
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
            return await asyncio.to_thread(self.plugins.export_bundle,
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
            return await asyncio.to_thread(lambda: self.plugins.audit(name))
        if action in {"update", "check_update", "upgrade"}:
            return await asyncio.to_thread(lambda: self.plugins.check_update(
                name, str(args.get("version") or args.get("source") or "")))
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
                events=getattr(self, "plugin_events", None),
                scheduler=getattr(self, "plugin_scheduler", None))
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
            ready = getattr(worker, "_transport_ready", None)
            connected = (bool(ready()) if callable(ready) else
                         bool(getattr(worker, "connected", False) and getattr(worker, "session", None)))
            if connected:
                model = str(getattr(worker, "active_live_model", "") or "").rsplit("/", 1)[-1]
                lines.append(f"- Voice: Gemini Live connected{f' ({model})' if model else ''}.")
            else:
                lines.append("- Voice: Gemini Live is not connected; I'm on local voice and "
                             "the text models (I can still act through my tools).")
        router = getattr(getattr(self, "ai", None), "router", None)
        if router is not None:
            try:
                text = [p.name for p in router.text_profiles()]
                lines.append(f"- Thinking: {len(text)} text model provider(s) eligible for routing"
                             + (f" ({', '.join(text[:5])})." if text else
                                " — none eligible at the moment."))
            except Exception:
                pass
            try:
                from .provider_capabilities import Capability, supports
                seeing = [p.name for p in router.text_profiles()
                          if supports(p, Capability.VISION) or "vision" in p.strengths]
                keyed = [p.name for p in router.live_profiles()]
                lines.append("- Seeing: " + ("model routes eligible (" + ", ".join((seeing + keyed)[:4])
                                             + "); camera permission and capture not checked."
                                             if seeing or keyed else
                                             "no eligible vision model route; camera readiness not checked."))
            except Exception:
                pass
        declarations = getattr(self, "TOOL_DECLARATIONS", None) or []
        mcp_routes = getattr(self, "_mcp_routes", {}) or {}
        forged = getattr(self, "_tool_handlers", {}) or {}
        by_server: dict[str, int] = {}
        for server, _tool in mcp_routes.values():
            by_server[server] = by_server.get(server, 0) + 1
        lines.append(f"- Tools: {len(declarations)} registered"
                     + (f", including {sum(by_server.values())} from MCP servers ("
                        + ", ".join(f"{k} {v}" for k, v in sorted(by_server.items())) + ")"
                        if by_server else "")
                     + (f" and {len(forged)} dynamic handler(s)" if forged else "") + ".")
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
            from .execution_policy import require_own_window
            if result.ok and require_own_window("gui_chess_navigation") is None:
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
        # Lichess: friends play him remotely through his BOT account
        # (chess_lichess.py — outbound only, nothing listens on this PC).
        if action in {"lichess_start", "take_challenges", "go_online", "lichess_online"}:
            return await self.chess.lichess_start()
        if action in {"lichess_stop", "stop_challenges", "go_offline", "lichess_offline"}:
            return await self.chess.lichess_stop()
        if action in {"lichess_status", "lichess"}:
            return self.chess.lichess_status()
        if action in {"challenge", "lichess_challenge", "challenge_friend"}:
            username = str(args.get("username") or args.get("friend") or "")
            if not username:
                return ToolResult("Who should I challenge? I need their Lichess username.",
                                  ok=False)
            try:
                minutes = float(args.get("minutes") or 5)
                increment = int(args.get("increment") if args.get("increment") is not None else 3)
            except (TypeError, ValueError):
                minutes, increment = 5.0, 3
            return await self.chess.lichess_challenge(
                username, minutes, increment,
                str(args.get("rated") or "").strip().lower() in {"true", "1", "yes", "rated"},
                str(args.get("colour") or args.get("color") or "random").lower())
        if action in {"lichess_allow", "allow_friend", "lichess_disallow", "remove_friend"}:
            username = str(args.get("username") or args.get("friend") or "")
            if not username:
                return ToolResult("Which Lichess username?", ok=False)
            return self.chess.lichess_allow(
                username, allowed=action in {"lichess_allow", "allow_friend"})
        if action in {"lichess_anyone", "accept_anyone"}:
            enabled = str(args.get("enabled") if args.get("enabled") is not None
                          else "true").strip().lower() in {"true", "1", "yes", "on"}
            return self.chess.lichess_accept_anyone(enabled)
        if action in {"lichess_resign"}:
            return await self.chess.lichess_resign()
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
