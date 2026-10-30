"""
The stock market, said the way a person would say it.

Two things were wrong with what ORION said about markets:

  * The greeting named one company: "Nvidia is up 2.3 percent" was the whole
    market report. An index is the market; a ticker is a position nobody
    asked ORION to hold.
  * Both the greeting and the briefing stated every move as if it were live.
    At 8 a.m. in the UK the S&P 500's latest quote is YESTERDAY's close (New
    York opens at 14:30 UK time), so "the S and P is up 0.8 percent" was a
    day-old move presented as today's.

Here each quote knows whether its exchange is open now, closed for the day,
or not open yet (so the number is from the last session), and the sentence
says so: "the FTSE 100 is up 0.2 percent, and on Friday the S and P 500
closed up 0.7 percent". UK first, because that is where the user is.

Quotes come from Yahoo's public chart endpoint (no key). Everything is
best-effort: a missing quote is left out, never guessed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus

#: (symbol, how it is said aloud). The ampersand is spelled out so a speech
#: engine does not read "S ampersand P".
INDICES: tuple[tuple[str, str], ...] = (
    ("^FTSE", "the FTSE 100"),
    ("^GSPC", "the S and P 500"),
    ("^IXIC", "the Nasdaq"),
    ("^DJI", "the Dow"),
)
#: What the Markets chart shows, by group: (symbol, name on screen). These
#: are only drawn, never spoken - the spoken report stays on the indices.
#: config/markets.json replaces any group: {"Stocks": [["TSLA", "Tesla"]]}.
WATCH: dict[str, tuple[tuple[str, str], ...]] = {
    "Indices": (("^FTSE", "FTSE 100"), ("^FTMC", "FTSE 250"),
                ("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq")),
    "ETFs": (("VUSA.L", "Vanguard S&P 500"), ("ISF.L", "iShares FTSE 100"),
             ("VWRL.L", "Vanguard All-World"), ("EQQQ.L", "Invesco Nasdaq 100")),
    "Stocks": (("NVDA", "Nvidia"), ("AAPL", "Apple"), ("MSFT", "Microsoft"),
               ("AZN.L", "AstraZeneca")),
}
WATCH_FILE = "markets.json"
_HEADERS = {"User-Agent": "Mozilla/5.0 (ORION markets)"}
FLAT_PCT = 0.1          # below this a move is "flat", not "up 0.0 percent"
BIG_PCT = 2.0           # a move worth leading with


@dataclass
class Quote:
    symbol: str
    label: str
    price: float
    change: float | None          # percent against the previous close
    state: str                    # "open", "closed" (today's session over), "previous"
    session: datetime | None      # when the quoted session traded, exchange-local
    currency: str = ""            # Yahoo's code: "GBp" is pence, "GBP" pounds
    previous: float | None = None # the previous close
    times: tuple[int, ...] = ()   # the session's prices (intraday fetch only),
    points: tuple[float, ...] = ()  # as unix seconds and price
    period: tuple[int, int] | None = None  # the regular session, unix seconds

    def move(self) -> str:
        if self.change is None:
            return f"at {self.price:,.0f}"
        if abs(self.change) < FLAT_PCT:
            return "flat"
        return f"{'up' if self.change > 0 else 'down'} {abs(self.change):.1f} percent"


def _zone(name: str | None, offset_s: Any) -> timezone | Any:
    try:
        from zoneinfo import ZoneInfo
        if name:
            return ZoneInfo(name)
    except Exception:
        pass
    try:
        return timezone(timedelta(seconds=int(offset_s or 0)))
    except Exception:
        return timezone.utc


def parse(symbol: str, label: str, meta: dict[str, Any], now: datetime | None = None) -> Quote | None:
    """A Quote from Yahoo chart metadata, or None when it holds no price."""
    price = meta.get("regularMarketPrice")
    if price is None:
        return None
    now = now or datetime.now(timezone.utc)
    previous = meta.get("chartPreviousClose") or meta.get("previousClose")
    change = None
    if previous:
        change = (float(price) - float(previous)) / float(previous) * 100.0
    zone = _zone(meta.get("exchangeTimezoneName"), meta.get("gmtoffset"))
    traded = meta.get("regularMarketTime")
    session = datetime.fromtimestamp(int(traded), tz=timezone.utc).astimezone(zone) if traded else None
    regular = ((meta.get("currentTradingPeriod") or {}).get("regular") or {})
    start, end = regular.get("start"), regular.get("end")
    stamp = now.timestamp()
    if start and end and float(start) <= stamp < float(end):
        state = "open"
    elif session is not None and session.date() == now.astimezone(zone).date():
        state = "closed"
    else:
        state = "previous"
    return Quote(symbol, label, float(price), change, state, session,
                 currency=str(meta.get("currency") or ""),
                 previous=float(previous) if previous else None,
                 period=(int(start), int(end)) if start and end else None)


def series(result: dict[str, Any]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """The session's (times, prices) from one Yahoo chart result, gaps dropped."""
    stamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
    pairs = [(int(t), float(c)) for t, c in zip(stamps, quote.get("close") or [])
             if t is not None and c is not None]
    return tuple(t for t, _ in pairs), tuple(c for _, c in pairs)


def price_text(quote: Quote, value: float | None = None) -> str:
    """A price as it is written: index points bare, London's pence in pounds."""
    amount = quote.price if value is None else value
    if quote.symbol.startswith("^"):
        return f"{amount:,.2f}"
    if quote.currency == "GBp":
        return f"£{amount / 100:,.2f}"
    sign = {"GBP": "£", "USD": "$", "EUR": "€"}.get(quote.currency)
    if sign:
        return f"{sign}{amount:,.2f}"
    return f"{amount:,.2f} {quote.currency}".strip()


def watchlist(config_dir: Any = None) -> dict[str, tuple[tuple[str, str], ...]]:
    """WATCH with the owner's own groups from config/markets.json laid over it."""
    groups = dict(WATCH)
    try:
        if config_dir is None:
            from .constants import CONFIG_DIR as config_dir
        data = json.loads((Path(config_dir) / WATCH_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return groups
    if isinstance(data, dict):
        for name, rows in data.items():
            try:
                chosen = tuple((str(symbol), str(label)) for symbol, label in rows)
            except (TypeError, ValueError):
                continue
            if chosen:
                groups[str(name)] = chosen
    return groups


async def fetch(session: Any, indices: Iterable[tuple[str, str]] = INDICES,
                now: datetime | None = None, *, intraday: bool = False) -> list[Quote]:
    """Quotes for *indices*, concurrently, in the order given; failures dropped.
    *intraday* also brings the session's five-minute prices (for the chart)."""
    async def one(symbol: str, label: str) -> Quote | None:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(symbol)}"
               + ("?range=1d&interval=5m" if intraday else "?range=1d&interval=1d"))
        try:
            async with session.get(url, headers=_HEADERS) as response:
                if response.status != 200:
                    return None
                data = await response.json(content_type=None)
            result = ((data.get("chart") or {}).get("result") or [{}])[0] or {}
            quote = parse(symbol, label, result.get("meta") or {}, now)
            if quote is not None and intraday:
                quote.times, quote.points = series(result)
            return quote
        except Exception:
            return None

    pairs = list(indices)
    results = await asyncio.gather(*(one(s, l) for s, l in pairs), return_exceptions=True)
    return [r for r in results if isinstance(r, Quote)]


def _when(quote: Quote, now: datetime) -> str:
    """'yesterday', 'on Friday' or 'on 2 October' for an earlier session."""
    if quote.session is None:
        return "at the last close"
    local_today = now.astimezone(quote.session.tzinfo).date() if quote.session.tzinfo else now.date()
    days = (local_today - quote.session.date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"on {quote.session.strftime('%A')}"
    return f"on {quote.session.day} {quote.session.strftime('%B')}"


def phrase(quote: Quote, now: datetime | None = None) -> str:
    """One index, with the right tense: 'is up', 'closed up', 'closed up ... on Friday'."""
    now = now or datetime.now(timezone.utc)
    move = quote.move()
    if quote.state == "open":
        return f"{quote.label} is {move}" if move != "flat" else f"{quote.label} is flat"
    if quote.state == "closed":
        return f"{quote.label} closed {move}"
    return f"{quote.label} closed {move} {_when(quote, now)}"


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def summary(quotes: list[Quote], now: datetime | None = None, *, brief: bool = False) -> str:
    """A spoken market sentence. *brief* (the greeting): the home index and
    the S&P, plus the Nasdaq only when it tells a different story; nothing
    at all when no quote arrived."""
    now = now or datetime.now(timezone.utc)
    if not quotes:
        return ""
    chosen = list(quotes)
    if brief:
        by_symbol = {q.symbol: q for q in quotes}
        chosen = [q for q in (by_symbol.get("^FTSE"), by_symbol.get("^GSPC")) if q]
        nasdaq, spx = by_symbol.get("^IXIC"), by_symbol.get("^GSPC")
        if nasdaq is not None and nasdaq.change is not None and (
                spx is None or spx.change is None
                or abs(nasdaq.change - spx.change) >= 0.5 or abs(nasdaq.change) >= BIG_PCT):
            chosen.append(nasdaq)
        if not chosen:
            chosen = quotes[:2]
    # Live markets first, then today's closes, then earlier sessions - the
    # order in which the numbers are current. Closes from the same session
    # share one clause: "on Friday the S and P 500 closed up 0.7 percent and
    # the Nasdaq up 1.2 percent", not "on Friday" three times over.
    rank = {"open": 0, "closed": 1, "previous": 2}
    chosen.sort(key=lambda q: rank.get(q.state, 3))
    big = [q for q in chosen if q.change is not None and abs(q.change) >= BIG_PCT]
    clauses: list[str] = [phrase(q, now) for q in chosen if q.state == "open"]
    groups: dict[str, list[Quote]] = {}
    for q in chosen:
        if q.state != "open":
            key = "today" if q.state == "closed" else _when(q, now)
            groups.setdefault(key, []).append(q)
    for when, members in groups.items():
        first, rest = members[0], members[1:]
        text = f"{first.label} closed {first.move()}"
        text += "".join(f", {q.label} {q.move()}" for q in rest[:-1])
        if rest:
            text += f" and {rest[-1].label} {rest[-1].move()}"
        clauses.append(text if when == "today" else f"{when} {text}")
    sentence = _join(clauses)
    if big:
        mover = max(big, key=lambda q: abs(q.change or 0))
        lead = "A rough day" if (mover.change or 0) < 0 else "A strong day"
        if mover.state == "previous":
            lead = "A big move last session"
        return f"{lead} in the markets: {sentence}."
    return f"In the markets, {sentence}."


__all__ = ["INDICES", "WATCH", "Quote", "parse", "fetch", "phrase", "price_text", "series",
           "summary", "watchlist"]
