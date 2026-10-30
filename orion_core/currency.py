"""
Currency — "how much is 50 dollars in pounds?", from today's reference rates.

The calculator converted stone to kilos but answered "I could not read that as
a calculation" to any amount of money. Rates come from Frankfurter (European
Central Bank reference rates, keyless, ~30 currencies, published each working
day), cached for an hour. The answer says whose rate and which day it is, and
that a bank or card will charge a little more - the reference rate is the
mid-market price, not what anyone pays.

parse() is pure (tested without the network); convert() makes the one call.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

RATES_URL = "https://api.frankfurter.dev/v1/latest?base={base}&symbols={target}"
CACHE_S = 3600.0
_CACHE: dict[tuple[str, str], tuple[float, float, str]] = {}

#: Spoken and written names -> ISO code. Bare "dollar" is the US dollar,
#: bare "pound" sterling; the qualified forms come first so they win.
_NAMES: tuple[tuple[str, str], ...] = (
    (r"australian dollars?|aussie dollars?|aud|a\$", "AUD"),
    (r"canadian dollars?|cad|c\$", "CAD"),
    (r"new zealand dollars?|kiwi dollars?|nzd", "NZD"),
    (r"hong kong dollars?|hkd", "HKD"),
    (r"singapore dollars?|sgd", "SGD"),
    (r"us dollars?|american dollars?|dollars?|bucks|usd|\$", "USD"),
    (r"pounds? sterling|british pounds?|pounds?|quid|sterling|gbp|£", "GBP"),
    (r"euros?|eur|€", "EUR"),
    (r"japanese yen|yen|jpy|¥", "JPY"),
    (r"chinese yuan|yuan|renminbi|rmb|cny", "CNY"),
    (r"indian rupees?|rupees?|inr|₹", "INR"),
    (r"swiss francs?|francs?|chf", "CHF"),
    (r"swedish kronor|swedish krona|kronor|sek", "SEK"),
    (r"norwegian kroner|norwegian krone|nok", "NOK"),
    (r"danish kroner|danish krone|dkk", "DKK"),
    (r"polish zloty|zloty|zlotys|pln", "PLN"),
    (r"czech korun[ay]|koruna|czk", "CZK"),
    (r"hungarian forints?|forints?|huf", "HUF"),
    (r"turkish lira|lira|try", "TRY"),
    (r"south african rand|rand|zar", "ZAR"),
    (r"brazilian reais|brazilian real|reais|brl", "BRL"),
    (r"mexican pesos?|pesos?|mxn", "MXN"),
    (r"south korean won|korean won|won|krw", "KRW"),
    (r"thai baht|baht|thb", "THB"),
    (r"israeli shekels?|shekels?|ils", "ILS"),
    (r"icelandic kr[oó]n(?:a|ur)|isk", "ISK"),
    (r"romanian lei|lei|ron", "RON"),
    (r"malaysian ringgit|ringgit|myr", "MYR"),
    (r"philippine pesos?|php", "PHP"),
    (r"indonesian rupiah|rupiah|idr", "IDR"),
    (r"bulgarian lev|lev|bgn", "BGN"),
)
_NAME_ALT = "|".join(f"(?:{pattern})" for pattern, _code in _NAMES)
SPOKEN = {"USD": "US dollars", "GBP": "pounds", "EUR": "euros", "JPY": "yen", "AUD": "Australian dollars",
          "CAD": "Canadian dollars", "NZD": "New Zealand dollars", "HKD": "Hong Kong dollars",
          "SGD": "Singapore dollars", "CNY": "yuan", "INR": "rupees", "CHF": "Swiss francs",
          "SEK": "Swedish kronor", "NOK": "Norwegian kroner", "DKK": "Danish kroner",
          "PLN": "zloty", "CZK": "Czech koruna", "HUF": "forints", "TRY": "Turkish lira",
          "ZAR": "rand", "BRL": "Brazilian reais", "MXN": "Mexican pesos", "KRW": "won",
          "THB": "baht", "ILS": "shekels", "ISK": "Icelandic krónur", "RON": "lei",
          "MYR": "ringgit", "PHP": "Philippine pesos", "IDR": "rupiah", "BGN": "lev"}

_AMOUNT = r"(?P<amount{n}>\d[\d,]*(?:\.\d+)?)\s*(?P<scale{n}>k|thousand|million|m|grand)?"
_PATTERN = re.compile(
    rf"(?:(?P<pre>{_NAME_ALT})\s*{_AMOUNT.format(n=1)}|{_AMOUNT.format(n=2)}\s*(?P<post>{_NAME_ALT}))"
    rf"\s+(?:in|to|into|as|in terms of)\s+(?P<target>{_NAME_ALT})(?![\w$£€¥₹])", re.I)


def code_of(name: str) -> str | None:
    text = str(name or "").strip().lower()
    for pattern, code in _NAMES:
        if re.fullmatch(pattern, text, re.I):
            return code
    return None


@dataclass(frozen=True)
class Request:
    amount: float
    base: str
    target: str


def parse(text: str) -> Request | None:
    """The amount and the two currencies in a question, or None."""
    text = str(text or "").replace("’", "'")
    # "two grand" is pounds in British speech, with or without the word.
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s*grand\b", r"£\1k", text, flags=re.I)
    m = _PATTERN.search(text)
    if not m:
        return None
    base = code_of(m.group("pre") or m.group("post") or "")
    target = code_of(m.group("target"))
    amount_text = m.group("amount1") or m.group("amount2") or ""
    if not (base and target and amount_text):
        return None
    amount = float(amount_text.replace(",", ""))
    scale = (m.group("scale1") or m.group("scale2") or "").lower()
    amount *= {"k": 1e3, "thousand": 1e3, "grand": 1e3, "m": 1e6, "million": 1e6}.get(scale, 1.0)
    if base == target or amount <= 0:
        return None
    return Request(amount, base, target)


def _money(value: float, code: str) -> str:
    whole = code in {"JPY", "KRW", "HUF", "ISK", "IDR"} or value >= 1000 or float(value).is_integer()
    figure = f"{value:,.0f}" if whole else f"{value:,.2f}"
    return f"{figure} {SPOKEN.get(code, code)}"


def describe(request: Request, rate: float, day: str) -> str:
    converted = request.amount * rate
    try:
        from datetime import date
        published = date.fromisoformat(day)
        when = ("today's" if published == date.today()
                else f"{published.strftime('%A')}'s")
    except ValueError:
        when = "the latest"
    return (f"{_money(request.amount, request.base)} is about {_money(round(converted, 2), request.target)} "
            f"at {when} European Central Bank reference rate "
            f"(1 {request.base} = {rate:.4g} {request.target}). A bank or card will usually "
            f"give you a little less.")


async def convert(text: str, session: Any = None) -> str | None:
    """The spoken conversion, None if *text* isn't one; raises on a fetch failure."""
    request = parse(text)
    if request is None:
        return None
    key = (request.base, request.target)
    hit = _CACHE.get(key)
    if hit is not None and time.monotonic() - hit[0] < CACHE_S:
        return describe(request, hit[1], hit[2])
    url = RATES_URL.format(base=request.base, target=request.target)

    async def get(client: Any) -> dict[str, Any]:
        async with client.get(url) as response:
            if response.status != 200:
                raise RuntimeError(f"the rates service answered HTTP {response.status}")
            return await response.json(content_type=None)

    if session is None:
        from aiohttp import ClientSession, ClientTimeout
        async with ClientSession(timeout=ClientTimeout(total=8.0, connect=4.0)) as own:
            data = await get(own)
    else:
        data = await get(session)
    rate = float((data.get("rates") or {})[request.target])
    day = str(data.get("date") or "today")
    _CACHE[key] = (time.monotonic(), rate, day)
    return describe(request, rate, day)


__all__ = ["Request", "SPOKEN", "code_of", "convert", "describe", "parse"]
