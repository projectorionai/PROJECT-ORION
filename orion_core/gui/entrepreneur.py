"""
EntrepreneurDeck — an upgraded toolkit page for the swipeable Command Deck.

Every widget is a real, working, offline-capable tool for a modern founder:

    • Focus timer      — Pomodoro with a task label, a completed-session count,
                         and a spoken "focus block complete" when a block ends.
    • Margin & markup  — cost, price, platform fee %, shipping and VAT →
                         true profit/margin/markup, plus a suggested price to
                         hit a target margin.
    • Ad performance    — spend, revenue and orders → ROAS, ACOS, CPA and the
                         break-even ROAS/CPA implied by a product margin.
    • Currency         — live ECB conversion with a swap button (offline-safe).
    • World clock      — key commerce timezones with LSE/NYSE market open/closed.
    • Idea capture     — ideas saved to long-term memory, with delete.
    • KPI tracker      — log revenue + orders; running total, order count, AOV.
    • Deadline countdown — named deadlines with live days/hours remaining.
    • Reminders        — quick spoken reminders through the ReminderService.
    • Protocols        — run any JARVIS-style protocol with one click.

The widgets that drive live systems (reminders, protocols) are only shown when
those managers are wired in.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus


def _num(text: Any, default: float = 0.0) -> float:
    try:
        return float(str(text).replace(",", "").replace("£", "").replace("$", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return default


def _setup(widget: QWidget, title: str) -> QVBoxLayout:
    """Give *widget* a panel layout with a heading; return the layout to fill."""
    widget.setObjectName("panelFrame")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(6)
    heading = QLabel(title)
    heading.setObjectName("panelHeading")
    layout.addWidget(heading)
    return layout


# ──────────────────────────────────────────────────────────────────────────────
# FOCUS TIMER
# ──────────────────────────────────────────────────────────────────────────────

class FocusTimer(QFrame):
    def __init__(self, bus: OrionBus) -> None:
        super().__init__()
        self.bus = bus
        self.setObjectName("panelFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)
        layout.addWidget(self._h("FOCUS TIMER"))

        self._work, self._break = 25 * 60, 5 * 60
        self._remaining = self._work
        self._on_break = False
        self._running = False
        self._sessions = 0

        self.task = QLineEdit()
        self.task.setPlaceholderText("What are you focusing on?")
        self.clock = QLabel("25:00")
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock.setStyleSheet("font-size:32px;font-weight:800;color:#ff5c73;")
        self.phase = QLabel("Focus block — ready when you are, sir.")
        self.phase.setObjectName("mutedLabel")
        self.phase.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bar = QProgressBar(); self.bar.setRange(0, self._work); self.bar.setValue(self._work)
        self.bar.setTextVisible(False)

        row = QHBoxLayout()
        self.start_btn = QPushButton("START"); self.start_btn.clicked.connect(self._toggle)
        reset = QPushButton("RESET"); reset.clicked.connect(self._reset)
        row.addWidget(self.start_btn); row.addWidget(reset)

        for w in (self.task, self.clock, self.phase, self.bar):
            layout.addWidget(w)
        layout.addLayout(row)

        self._timer = QTimer(self); self._timer.setInterval(1000); self._timer.timeout.connect(self._tick)

    def _h(self, t: str) -> QLabel:
        lab = QLabel(t); lab.setObjectName("panelHeading"); return lab

    def _toggle(self) -> None:
        self._running = not self._running
        self.start_btn.setText("PAUSE" if self._running else "START")
        self._timer.start() if self._running else self._timer.stop()

    def _reset(self) -> None:
        self._timer.stop(); self._running = False; self._on_break = False
        self._remaining = self._work; self.start_btn.setText("START")
        self.phase.setText("Focus block — ready when you are, sir.")
        self.bar.setRange(0, self._work); self.bar.setValue(self._work); self._render()

    def _tick(self) -> None:
        self._remaining -= 1
        if self._remaining <= 0:
            if not self._on_break:
                self._sessions += 1
                task = self.task.text().strip()
                note = f" on {task}" if task else ""
                self.bus.speak_request.emit(f"Focus block complete{note}, sir. That's {self._sessions} today. Time for a short break.")
            else:
                self.bus.speak_request.emit("Break over, sir. Back to it.")
            self._on_break = not self._on_break
            self._remaining = self._break if self._on_break else self._work
            total = self._break if self._on_break else self._work
            self.bar.setRange(0, total)
            self.phase.setText(f"Break — step away, sir. ({self._sessions} done)"
                               if self._on_break else f"Focus block — go. ({self._sessions} done)")
        self.bar.setValue(self._remaining); self._render()

    def _render(self) -> None:
        m, s = divmod(max(0, self._remaining), 60)
        self.clock.setText(f"{m:02d}:{s:02d}")


# ──────────────────────────────────────────────────────────────────────────────
# MARGIN & MARKUP (with fees / shipping / VAT + target price)
# ──────────────────────────────────────────────────────────────────────────────

class MarginCalculator(QFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = _setup(self, "MARGIN & PRICING")
        self.cost = QLineEdit(); self.cost.setPlaceholderText("Unit cost (e.g. 6.50)")
        self.price = QLineEdit(); self.price.setPlaceholderText("Sale price (e.g. 24.99)")
        self.fee = QLineEdit(); self.fee.setPlaceholderText("Platform fee % (e.g. 8)")
        self.ship = QLineEdit(); self.ship.setPlaceholderText("Shipping cost (e.g. 3.20)")
        self.vat = QLineEdit(); self.vat.setPlaceholderText("VAT % (e.g. 20)")
        self.target = QLineEdit(); self.target.setPlaceholderText("Target margin % (for price idea)")
        for w in (self.cost, self.price, self.fee, self.ship, self.vat, self.target):
            w.textChanged.connect(self._calc); layout.addWidget(w)
        self.out = QLabel("Enter a cost and price, sir."); self.out.setObjectName("mutedLabel")
        self.out.setWordWrap(True); layout.addWidget(self.out)

    def _calc(self) -> None:
        cost, price = _num(self.cost.text()), _num(self.price.text())
        fee, ship, vat = _num(self.fee.text()), _num(self.ship.text()), _num(self.vat.text())
        tgt = _num(self.target.text())
        lines = []
        if price > 0:
            net = price / (1 + vat / 100) if vat else price     # ex-VAT revenue
            fees = net * fee / 100
            profit = net - cost - ship - fees
            margin = profit / net * 100 if net else 0
            markup = profit / (cost + ship) * 100 if (cost + ship) else 0
            lines.append(f"Profit/unit: £{profit:.2f}  Margin: {margin:.1f}%  Markup: {markup:.1f}%")
        if tgt > 0 and (cost + ship) > 0:
            # Price (ex-VAT) so that profit/net = tgt%, accounting for the fee.
            denom = (1 - tgt / 100 - fee / 100)
            if denom > 0:
                net_needed = (cost + ship) / denom
                gross = net_needed * (1 + vat / 100)
                lines.append(f"To hit {tgt:.0f}% margin: sell at £{gross:.2f}")
        self.out.setText("\n".join(lines) or "Enter a cost and price, sir.")


# ──────────────────────────────────────────────────────────────────────────────
# AD PERFORMANCE (ROAS / ACOS / CPA / break-even)
# ──────────────────────────────────────────────────────────────────────────────

class AdCalculator(QFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = _setup(self, "AD PERFORMANCE")
        self.spend = QLineEdit(); self.spend.setPlaceholderText("Ad spend (e.g. 200)")
        self.revenue = QLineEdit(); self.revenue.setPlaceholderText("Revenue from ads (e.g. 900)")
        self.orders = QLineEdit(); self.orders.setPlaceholderText("Orders (e.g. 30)")
        self.margin = QLineEdit(); self.margin.setPlaceholderText("Product margin % (e.g. 55)")
        for w in (self.spend, self.revenue, self.orders, self.margin):
            w.textChanged.connect(self._calc); layout.addWidget(w)
        self.out = QLabel("Enter spend and revenue, sir."); self.out.setObjectName("mutedLabel")
        self.out.setWordWrap(True); layout.addWidget(self.out)

    def _calc(self) -> None:
        spend, rev = _num(self.spend.text()), _num(self.revenue.text())
        orders, margin = _num(self.orders.text()), _num(self.margin.text())
        if spend <= 0:
            self.out.setText("Enter spend and revenue, sir."); return
        roas = rev / spend
        acos = spend / rev * 100 if rev else 0
        lines = [f"ROAS: {roas:.2f}x   ACOS: {acos:.1f}%"]
        if orders > 0:
            cpa = spend / orders
            lines.append(f"CPA: £{cpa:.2f}   AOV: £{rev/orders:.2f}")
        if margin > 0:
            be_roas = 100 / margin
            verdict = "profitable" if roas >= be_roas else "below break-even"
            lines.append(f"Break-even ROAS: {be_roas:.2f}x — currently {verdict}")
            if orders > 0 and rev > 0:
                be_cpa = (rev / orders) * margin / 100
                lines.append(f"Break-even CPA: £{be_cpa:.2f}")
        self.out.setText("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# CURRENCY CONVERTER (with swap)
# ──────────────────────────────────────────────────────────────────────────────

class CurrencyConverter(QFrame):
    CURRENCIES = ("GBP", "USD", "EUR", "CNY", "JPY", "AUD", "CAD", "INR", "AED")

    def __init__(self, bus: OrionBus) -> None:
        super().__init__()
        layout = _setup(self, "CURRENCY CONVERTER")
        self.bus = bus
        self.amount = QLineEdit("100"); self.amount.textChanged.connect(self._convert)
        layout.addWidget(self.amount)
        row = QHBoxLayout()
        self.src = QComboBox(); self.src.addItems(self.CURRENCIES)
        self.dst = QComboBox(); self.dst.addItems(self.CURRENCIES); self.dst.setCurrentText("USD")
        swap = QPushButton("⇄"); swap.setFixedWidth(38); swap.clicked.connect(self._swap)
        self.src.currentTextChanged.connect(self._convert); self.dst.currentTextChanged.connect(self._convert)
        row.addWidget(self.src); row.addWidget(swap); row.addWidget(self.dst)
        layout.addLayout(row)
        self.out = QLabel("Live ECB rate."); self.out.setObjectName("mutedLabel")
        self.out.setWordWrap(True); layout.addWidget(self.out)
        self._convert()

    def _swap(self) -> None:
        s, d = self.src.currentText(), self.dst.currentText()
        self.src.setCurrentText(d); self.dst.setCurrentText(s); self._convert()

    def _convert(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._do())

    async def _do(self) -> None:
        amount = _num(self.amount.text())
        src, dst = self.src.currentText(), self.dst.currentText()
        if src == dst:
            self.out.setText(f"{amount:.2f} {src} = {amount:.2f} {dst}"); return
        try:
            from aiohttp import ClientSession, ClientTimeout
            url = f"https://api.frankfurter.app/latest?amount={amount}&from={src}&to={dst}"
            async with ClientSession(timeout=ClientTimeout(total=8.0)) as s:
                async with s.get(url) as r:
                    data = await r.json()
            value = (data.get("rates") or {}).get(dst)
            if value is None:
                raise ValueError("no rate")
            rate = value / amount if amount else 0
            self.out.setText(f"{amount:.2f} {src} = {value:,.2f} {dst}\n1 {src} = {rate:.4f} {dst}  (live ECB)")
        except Exception:
            self.out.setText("Rate unavailable offline, sir — reconnect for live currency.")


# ──────────────────────────────────────────────────────────────────────────────
# WORLD CLOCK + MARKET STATUS
# ──────────────────────────────────────────────────────────────────────────────

class WorldClock(QFrame):
    ZONES = (("London", 0), ("New York", -5), ("Los Angeles", -8),
             ("Dubai", 4), ("Shenzhen", 8), ("Tokyo", 9))

    def __init__(self) -> None:
        super().__init__()
        layout = _setup(self, "WORLD CLOCK & MARKETS")
        self.labels: list[QLabel] = []
        for _ in self.ZONES:
            lab = QLabel(); lab.setObjectName("mutedLabel"); layout.addWidget(lab); self.labels.append(lab)
        self.markets = QLabel(); self.markets.setObjectName("mutedLabel"); self.markets.setWordWrap(True)
        layout.addWidget(self.markets)
        self._timer = QTimer(self); self._timer.setInterval(1000); self._timer.timeout.connect(self._tick)
        self._timer.start(); self._tick()

    def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        for lab, (name, off) in zip(self.labels, self.ZONES):
            local = now + timedelta(hours=off); hour = local.hour
            state = "🌙" if (hour < 7 or hour >= 21) else "☀"
            lab.setText(f"{state}  {name:<12} {local.strftime('%H:%M %a')}")
        lse = self._market_open(now + timedelta(hours=0), 8, 0, 16, 30)
        nyse = self._market_open(now + timedelta(hours=-5), 9, 30, 16, 0)
        self.markets.setText(f"LSE: {'OPEN' if lse else 'closed'}   ·   NYSE: {'OPEN' if nyse else 'closed'}")

    @staticmethod
    def _market_open(local: datetime, oh: int, om: int, ch: int, cm: int) -> bool:
        if local.weekday() >= 5:
            return False
        mins = local.hour * 60 + local.minute
        return oh * 60 + om <= mins < ch * 60 + cm


# ──────────────────────────────────────────────────────────────────────────────
# IDEA CAPTURE (with delete)
# ──────────────────────────────────────────────────────────────────────────────

class IdeaCapture(QFrame):
    def __init__(self, bus: OrionBus, memory: Any) -> None:
        super().__init__()
        layout = _setup(self, "IDEA CAPTURE")
        self.bus = bus; self.memory = memory
        row = QHBoxLayout()
        self.input = QLineEdit(); self.input.setPlaceholderText("Capture a product or business idea…")
        self.input.returnPressed.connect(self._save)
        save = QPushButton("SAVE"); save.clicked.connect(self._save)
        row.addWidget(self.input, 1); row.addWidget(save)
        layout.addLayout(row)
        self.list = QListWidget(); self.list.setMaximumHeight(120); layout.addWidget(self.list)
        delete = QPushButton("DELETE SELECTED"); delete.clicked.connect(self._delete); layout.addWidget(delete)
        self._load()

    def _save(self) -> None:
        idea = self.input.text().strip()
        if not idea:
            return
        self.input.clear()
        key = f"idea_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            self.memory.remember("knowledge", key, f"Business idea: {idea}")
        except Exception:
            pass
        self.list.insertItem(0, f"💡 {idea}")
        self.bus.log.emit(f"IDEA: captured — {idea[:60]}")

    def _delete(self) -> None:
        item = self.list.currentItem()
        if item:
            self.list.takeItem(self.list.row(item))

    def _load(self) -> None:
        try:
            for r in self.memory.query("Business idea", limit=12):
                val = str(r.get("value", "")).replace("Business idea: ", "")
                if val:
                    self.list.addItem(f"💡 {val}")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# KPI TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class KPITracker(QFrame):
    def __init__(self, bus: OrionBus, memory: Any) -> None:
        super().__init__()
        layout = _setup(self, "KPI TRACKER")
        self.bus = bus; self.memory = memory
        row = QHBoxLayout()
        self.rev = QLineEdit(); self.rev.setPlaceholderText("Revenue today (£)")
        self.ord = QLineEdit(); self.ord.setPlaceholderText("Orders")
        log = QPushButton("LOG"); log.clicked.connect(self._log)
        row.addWidget(self.rev); row.addWidget(self.ord); row.addWidget(log)
        layout.addLayout(row)
        self.out = QLabel(); self.out.setObjectName("mutedLabel"); self.out.setWordWrap(True)
        layout.addWidget(self.out)
        self._refresh()

    def _log(self) -> None:
        rev, orders = _num(self.rev.text()), _num(self.ord.text())
        if rev <= 0 and orders <= 0:
            return
        self.rev.clear(); self.ord.clear()
        key = f"kpi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            self.memory.remember("project", key, json.dumps({"rev": rev, "orders": orders}), project="kpi")
        except Exception:
            pass
        self._refresh()

    def _refresh(self) -> None:
        total_rev = total_ord = 0.0; n = 0
        try:
            for r in self.memory.recall("project", project="kpi", limit=200):
                try:
                    d = json.loads(r.get("value") or "{}")
                    total_rev += float(d.get("rev", 0)); total_ord += float(d.get("orders", 0)); n += 1
                except Exception:
                    continue
        except Exception:
            pass
        aov = total_rev / total_ord if total_ord else 0
        self.out.setText(f"Logged entries: {n}\nTotal revenue: £{total_rev:,.2f}\n"
                         f"Total orders: {int(total_ord)}   AOV: £{aov:,.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# DEADLINE COUNTDOWN
# ──────────────────────────────────────────────────────────────────────────────

class DeadlineCountdown(QFrame):
    def __init__(self, bus: OrionBus, memory: Any) -> None:
        super().__init__()
        layout = _setup(self, "DEADLINE COUNTDOWN")
        self.bus = bus; self.memory = memory
        row = QHBoxLayout()
        self.name = QLineEdit(); self.name.setPlaceholderText("Deadline name")
        self.date = QLineEdit(); self.date.setPlaceholderText("YYYY-MM-DD")
        add = QPushButton("ADD"); add.clicked.connect(self._add)
        row.addWidget(self.name, 1); row.addWidget(self.date); row.addWidget(add)
        layout.addLayout(row)
        self.list = QListWidget(); self.list.setMaximumHeight(120); layout.addWidget(self.list)
        self._deadlines: list[tuple[str, str]] = []
        self._load()
        self._timer = QTimer(self); self._timer.setInterval(30000); self._timer.timeout.connect(self._render)
        self._timer.start()

    def _add(self) -> None:
        name, date = self.name.text().strip(), self.date.text().strip()
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            self.bus.log.emit("DEADLINE: use YYYY-MM-DD."); return
        if not name:
            return
        self._deadlines.append((name, date)); self.name.clear(); self.date.clear()
        try:
            self.memory.remember("project", f"deadline_{name[:20]}", date, project="deadlines")
        except Exception:
            pass
        self._render()

    def _load(self) -> None:
        try:
            for r in self.memory.recall("project", project="deadlines", limit=50):
                key = str(r.get("key_ref", "")).replace("deadline_", "")
                self._deadlines.append((key, str(r.get("value", ""))))
        except Exception:
            pass
        self._render()

    def _render(self) -> None:
        self.list.clear()
        now = datetime.now()
        for name, date in sorted(self._deadlines, key=lambda d: d[1]):
            try:
                target = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                continue
            delta = target - now
            if delta.total_seconds() < 0:
                self.list.addItem(f"⛔ {name} — passed")
            else:
                days = delta.days; hours = delta.seconds // 3600
                self.list.addItem(f"⏳ {name} — {days}d {hours}h ({date})")


# ──────────────────────────────────────────────────────────────────────────────
# REMINDERS
# ──────────────────────────────────────────────────────────────────────────────

class RemindersWidget(QFrame):
    def __init__(self, bus: OrionBus, reminders: Any) -> None:
        super().__init__()
        layout = _setup(self, "REMINDERS")
        self.bus = bus; self.reminders = reminders
        row = QHBoxLayout()
        self.text = QLineEdit(); self.text.setPlaceholderText("Remind me to…")
        self.mins = QLineEdit(); self.mins.setPlaceholderText("mins"); self.mins.setFixedWidth(60)
        add = QPushButton("SET"); add.clicked.connect(self._add)
        self.text.returnPressed.connect(self._add)
        row.addWidget(self.text, 1); row.addWidget(self.mins); row.addWidget(add)
        layout.addLayout(row)
        self.list = QListWidget(); self.list.setMaximumHeight(120); layout.addWidget(self.list)
        clear = QPushButton("CLEAR ALL"); clear.clicked.connect(self._clear); layout.addWidget(clear)
        self.bus.dashboard_event.connect(self._on_event)
        self._timer = QTimer(self); self._timer.setInterval(5000); self._timer.timeout.connect(self._refresh)
        self._timer.start(); self._refresh()

    def _add(self) -> None:
        text = self.text.text().strip()
        if not text:
            return
        mins = _num(self.mins.text(), 0)
        result = self.reminders.add(text=text, minutes=mins if mins > 0 else None,
                                    phrase="" if mins > 0 else text)
        self.bus.log.emit(f"REMINDER: {result.text}")
        self.text.clear(); self.mins.clear(); self._refresh()

    def _clear(self) -> None:
        self.reminders.cancel(); self._refresh()

    def _on_event(self, channel: str, _payload: Any) -> None:
        if channel == "reminder_fired":
            self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        for r in self.reminders.active():
            import time as _t
            remaining = max(0, int(r.due_at - _t.monotonic()))
            self.list.addItem(f"⏰ {r.text} — at {r.wall_due} ({remaining // 60}m {remaining % 60}s)")


# ──────────────────────────────────────────────────────────────────────────────
# PROTOCOLS
# ──────────────────────────────────────────────────────────────────────────────

class ProtocolsWidget(QFrame):
    def __init__(self, bus: OrionBus, protocols: Any) -> None:
        super().__init__()
        layout = _setup(self, "PROTOCOLS")
        self.bus = bus; self.protocols = protocols
        self.combo = QComboBox()
        for name in protocols.all_protocols():
            self.combo.addItem(name.replace("_", " ").title(), name)
        run = QPushButton("ENGAGE PROTOCOL"); run.clicked.connect(self._run)
        layout.addWidget(self.combo); layout.addWidget(run)
        self.out = QLabel("Select and engage a protocol, sir."); self.out.setObjectName("mutedLabel")
        self.out.setWordWrap(True); layout.addWidget(self.out)

    def _run(self) -> None:
        name = str(self.combo.currentData() or "")
        if not name:
            return
        self.out.setText(f"Engaging {name.replace('_', ' ')} protocol…")

        async def _go() -> None:
            result = await self.protocols.run(name)
            self.out.setText(result.text.splitlines()[0][:160])

        try:
            asyncio.get_running_loop().create_task(_go())
        except RuntimeError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# DECK
# ──────────────────────────────────────────────────────────────────────────────

class EntrepreneurDeck(QWidget):
    """The entrepreneur toolkit page (scrollable grid of tools)."""

    def __init__(self, bus: OrionBus, memory: Any, reminders: Any | None = None,
                 protocols: Any | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus = bus
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        heading = QLabel("ENTREPRENEUR TOOLKIT")
        heading.setObjectName("panelHeading")
        outer.addWidget(heading)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        grid = QGridLayout(holder); grid.setSpacing(10)

        widgets: list[QWidget] = [
            FocusTimer(bus), MarginCalculator(), AdCalculator(),
            CurrencyConverter(bus), WorldClock(), IdeaCapture(bus, memory),
            KPITracker(bus, memory), DeadlineCountdown(bus, memory),
        ]
        if reminders is not None:
            widgets.append(RemindersWidget(bus, reminders))
        if protocols is not None:
            widgets.append(ProtocolsWidget(bus, protocols))

        for i, w in enumerate(widgets):
            grid.addWidget(w, i // 3, i % 3)

        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)
