"""
TemporalPresence (Mark X.7, Phase 3) — dynamic temporal awareness.

Replaces the canned greeting rotation with a greeting that actually knows
where it stands in time:

    "Good morning, sir. It is Tuesday the 8th of July, 7:03 in the morning —
     a mild summer day, 19 degrees and partly cloudy. You have two calendar
     entries today, and it has been eleven hours since we last spoke."

Composed from whichever of these are healthy — each is optional, each fails
alone, and the whole composition is capped by a hard time budget so startup
can never hang on a feed:

    • clock, weekday, date, day period            (always available)
    • season                                      (month-derived, hemisphere-aware)
    • weather                                     (Open-Meteo; coordinates from a
                                                   one-time IP lookup cached to
                                                   config/temporal.json, so
                                                   subsequent runs work offline-ish)
    • calendar load                               (Notion, when configured)
    • time since the previous conversation        (episodic memory)
    • one market note outside weekends            (Yahoo quote, best-effort)

Entirely decoupled: the worker calls ``compose_greeting()`` and receives a
string; no widget, no bus dependency beyond logging.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from aiohttp import ClientSession, ClientTimeout

from .bus import OrionBus
from .constants import CONFIG_DIR
from .utils import first_line, weather_code_label

STATE_PATH = CONFIG_DIR / "temporal.json"

_SEASONS_NORTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

_ORDINALS = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}


def _ordinal(day: int) -> str:
    return f"{day}{_ORDINALS.get(day, 'th')}"


class TemporalPresence:
    """Composes time-, weather- and context-aware greetings."""

    TIME_BUDGET_S = 6.0        # the whole enrichment must fit inside this

    def __init__(
        self,
        bus: OrionBus,
        memory: Any | None = None,
        notion: Any | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self.bus = bus
        self.memory = memory
        self.notion = notion
        self.telemetry = telemetry
        self._state = self._load_state()
        if self.telemetry is not None:
            self.telemetry.health.register("temporal")

    # ── persistence (cached coordinates + locality) ───────────────────────────

    def _load_state(self) -> dict[str, Any]:
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    async def compose_greeting(self) -> str:
        """The full temporal greeting; degrades feed by feed, never raises."""
        now = datetime.now()
        period = ("morning" if now.hour < 12 else
                  "afternoon" if now.hour < 17 else "evening")
        opening = (
            f"Good {period}, sir. It is {now.strftime('%A')} the "
            f"{_ordinal(now.day)} of {now.strftime('%B')}, "
            f"{now.strftime('%H:%M')}"
        )
        try:
            extras = await asyncio.wait_for(
                self._enrich(now), timeout=self.TIME_BUDGET_S)
        except (asyncio.TimeoutError, Exception) as exc:   # noqa: BLE001
            self.bus.log.emit(f"TEMPORAL: enrichment trimmed - {first_line(exc, 70)}")
            extras = []
        sentence = opening
        if extras and extras[0].startswith("—"):
            sentence += f" {extras.pop(0)}"
        sentence += "."
        tail = " ".join(extras).strip()
        greeting = f"{sentence} {tail}".strip()
        if self.telemetry is not None:
            self.telemetry.health.beat("temporal", "OK", f"{len(extras) + 1} element(s)")
        return greeting

    # ── enrichment feeds (each optional, all concurrent) ─────────────────────

    async def _enrich(self, now: datetime) -> list[str]:
        results = await asyncio.gather(
            self._weather_fragment(now),
            self._calendar_fragment(),
            self._last_seen_fragment(),
            self._market_fragment(now),
            return_exceptions=True,
        )
        fragments: list[str] = []
        for r in results:
            if isinstance(r, str) and r.strip():
                fragments.append(r.strip())
        return fragments

    def locality(self) -> str:
        """The cached 'city, region, country' string, or '' if unknown yet."""
        parts = [self._state.get(k) for k in ("city", "region", "country")]
        return ", ".join(p for p in parts if p)

    def coordinates(self) -> tuple[float, float] | None:
        lat, lon = self._state.get("latitude"), self._state.get("longitude")
        return (float(lat), float(lon)) if lat is not None and lon is not None else None

    async def prime_locality(self, router: Any | None = None) -> str:
        """
        Resolve the PC's location once at startup (IP geolocation, cached) and
        push it to the ProviderRouter so ORION already knows where he is —
        weather, news and 'near me' default here without asking.
        """
        if not self.locality():
            timeout = ClientTimeout(total=5.0, connect=2.5)
            try:
                async with ClientSession(timeout=timeout) as session:
                    async with session.get("https://ipapi.co/json/") as response:
                        if response.status == 200:
                            loc = await response.json()
                            self._state.update({
                                "latitude": float(loc.get("latitude")),
                                "longitude": float(loc.get("longitude")),
                                "city": str(loc.get("city") or ""),
                                "region": str(loc.get("region") or ""),
                                "country": str(loc.get("country_name") or loc.get("country") or ""),
                            })
                            await asyncio.to_thread(self._save_state)
            except Exception as exc:
                self.bus.log.emit(f"TEMPORAL: location lookup failed - {first_line(exc, 70)}")
        locality = self.locality()
        if locality and router is not None and hasattr(router, "set_locality"):
            router.set_locality(locality)
            self.bus.log.emit(f"TEMPORAL: current location resolved — {locality}.")
        return locality

    async def _weather_fragment(self, now: datetime) -> str:
        season = _SEASONS_NORTH.get(now.month, "")
        lat = self._state.get("latitude")
        lon = self._state.get("longitude")
        timeout = ClientTimeout(total=4.0, connect=2.0)
        try:
            async with ClientSession(timeout=timeout) as session:
                if lat is None or lon is None:
                    async with session.get("https://ipapi.co/json/") as response:
                        if response.status != 200:
                            raise RuntimeError(f"location {response.status}")
                        loc = await response.json()
                    lat = float(loc.get("latitude"))
                    lon = float(loc.get("longitude"))
                    self._state.update({
                        "latitude": lat, "longitude": lon,
                        "city": str(loc.get("city") or ""),
                        "region": str(loc.get("region") or ""),
                        "country": str(loc.get("country_name") or loc.get("country") or ""),
                    })
                    await asyncio.to_thread(self._save_state)
                url = (
                    "https://api.open-meteo.com/v1/forecast"
                    f"?latitude={lat:.4f}&longitude={lon:.4f}"
                    "&current=temperature_2m,weather_code&timezone=auto"
                )
                async with session.get(url) as response:
                    if response.status != 200:
                        raise RuntimeError(f"weather {response.status}")
                    data = await response.json()
            current = data.get("current") or {}
            temp = current.get("temperature_2m")
            label = weather_code_label(int(current.get("weather_code") or 0))
            if temp is None:
                raise RuntimeError("no temperature")
            return (f"— a {label} {season} {'morning' if now.hour < 12 else 'day'}"
                    f" at {float(temp):.0f} degrees")
        except Exception:
            # Offline / feed down: the season still gives temporal texture.
            return f"— a {season} {'morning' if now.hour < 12 else 'day'}" if season else ""

    async def _calendar_fragment(self) -> str:
        if self.notion is None or not getattr(self.notion, "available", False):
            return ""
        try:
            result = await self.notion.upcoming_events(days=1, limit=6)
            if not result.ok:
                return ""
            count = max(0, len([l for l in result.text.splitlines() if l.strip()]) - 1)
            if count <= 0:
                return "Your calendar is clear today."
            plural = "entries" if count != 1 else "entry"
            return f"You have {count} calendar {plural} today."
        except Exception:
            return ""

    async def _last_seen_fragment(self) -> str:
        if self.memory is None:
            return ""
        try:
            episodes = await asyncio.to_thread(self.memory.recall_episodes, "", 1)
        except Exception:
            return ""
        if not episodes:
            return ""
        raw = str(episodes[0].get("created_at") or "")
        try:
            last = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return ""
        hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
        if hours < 0.5:
            return ""
        if hours < 1.5:
            return "We spoke less than two hours ago."
        if hours < 24.0:
            return f"It has been {int(round(hours))} hours since we last spoke."
        days = int(hours // 24)
        return f"It has been {days} day{'s' if days != 1 else ''} since we last spoke."

    async def _market_fragment(self, now: datetime) -> str:
        if now.weekday() >= 5:      # weekend: markets closed, say nothing
            return ""
        timeout = ClientTimeout(total=4.0, connect=2.0)
        headers = {"User-Agent": "Mozilla/5.0 (ORION temporal)"}
        try:
            async with ClientSession(timeout=timeout) as session:
                url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
                       f"{quote_plus('NVDA')}?range=1d&interval=1d")
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        raise RuntimeError(f"quote {response.status}")
                    data = await response.json(content_type=None)
            meta = (((data.get("chart") or {}).get("result") or [{}])[0].get("meta") or {})
            price = meta.get("regularMarketPrice")
            previous = meta.get("chartPreviousClose") or meta.get("previousClose")
            if price is None or not previous:
                return ""
            change = (float(price) - float(previous)) / float(previous) * 100.0
            if abs(change) < 1.0:
                return ""       # only worth a mention when it moved
            direction = "up" if change > 0 else "down"
            return f"Nvidia is {direction} {abs(change):.1f} percent."
        except Exception:
            return ""
