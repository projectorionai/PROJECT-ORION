"""
Aviation — live aircraft around a place, from public ADS-B feeds.

"What's that plane?" and "what's flying over us?" are questions about the last
few seconds of the sky, so the answer has to come from a live feed and say how
old it is. Two free, keyless sources are wired, tried in this order:

  adsb.lol   ``/v2/lat/{lat}/lon/{lon}/dist/{nm}`` — community receivers,
             readsb JSON in feet, knots and ft/min. Measured 2026-09-25:
             ~130 ms a call cold, ~30 ms on a reused connection; a burst of a
             few calls inside a second earns an HTML 429 with NO Retry-After,
             which had cleared 10 s later.
  OpenSky    anonymous ``/api/states/all`` over a bounding box — SI units, but
             400 credits a day for anonymous use (``x-rate-limit-remaining``
             read 399 after one call) and a 10 s time resolution, so it is the
             fallback and is spaced 30 s apart to make the day's budget last.

Every provider sits behind its own circuit breaker. A 429 opens it for the
server's Retry-After (or a per-provider default when none is sent); any other
fault opens it for an exponentially growing spell; a success closes it. A
tripped provider is skipped rather than waited on, so one bad feed never
stalls a turn — the next one is asked instead.

Privacy: a provider is told a grid cell, not the house. Queries go out with the
centre rounded to 0.1° (about 11 km × 7 km in the UK) and the radius padded by
the worst-case rounding error; the radius filter, distances and bearings are
then worked out here from the precise point.

Honesty: nothing is invented. A field the source didn't send stays empty — no
callsign means "unidentified", adsb.lol sends no registration country so none
is claimed — and every answer carries the age of the positions it describes.
When no feed will answer, the last picture is returned flagged stale with its
age, never passed off as live.

Pure asyncio + httpx, no Qt. The Globe (gui/globe.py) and the ``aviation`` tool
share one service through :func:`get_service`, so they share its cache and
never poll a provider twice for the same cell.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable

ADSB_LOL_AREA = "https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{nm}"
ADSB_LOL_HEX = "https://api.adsb.lol/v2/hex/{hex}"
ADSB_LOL_CALLSIGN = "https://api.adsb.lol/v2/callsign/{callsign}"
OPENSKY_STATES = "https://opensky-network.org/api/states/all"
USER_AGENT = "ORION-AI-OS (personal assistant; live aircraft layer)"

EARTH_RADIUS_KM = 6371.0088
FT_TO_M = 0.3048
KT_TO_MS = 0.514444
FPM_TO_MS = 0.00508

CACHE_TTL_S = 10.0          # one region is fetched at most this often
STALE_LIMIT_S = 120.0       # the oldest picture ever handed back (flagged stale)
MAX_AIRCRAFT = 500          # the most a report — and so the Globe — carries
MAX_PARSED = 2000           # rows read from one response; bounds memory
MAX_RADIUS_KM = 250.0
MAX_CACHED = 16             # snapshots kept (distinct cells / radii)
GRID_DEG = 0.1              # provider queries use a centre snapped to this grid
# Snapping to a 0.1° grid moves the centre at most 0.05° in each axis: 5.56 km
# of latitude plus at most 5.56 km of longitude, 7.9 km diagonally. Padding the
# radius by that much means the coarse query still covers the precise circle.
PRIVACY_PAD_KM = 8.0

# readsb emitter categories C1–C5 are surface vehicles and fixed obstacles, and
# adsb.lol also lists airport transmitters typed "TWR"/"GND" (three of the five
# "aircraft" at Heathrow in the first probe were towers). None of them fly.
_NOT_AIRCRAFT_CATEGORIES = {"C1", "C2", "C3", "C4", "C5"}
_NOT_AIRCRAFT_TYPES = {"TWR", "GND"}

_COMPASS = ("north", "north-east", "east", "south-east",
            "south", "south-west", "west", "north-west")


# ── the normalised record ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Aircraft:
    """One aircraft as a source reported it, in SI units.

    ``last_contact`` is the epoch second (LOCAL clock) of the position: each
    parser converts the source's own "seconds since seen" onto this machine's
    clock, so dead-reckoning and "N s old" are not skewed by server clocks.
    """

    icao24: str
    callsign: str                   # "" when the transponder sent none
    lat: float
    lon: float
    altitude_m: float | None        # barometric; None when unknown or on the ground
    velocity_ms: float | None       # ground speed
    heading_deg: float | None       # track over the ground, clockwise from north
    vertical_rate: float | None     # m/s, positive climbing
    on_ground: bool
    origin_country: str             # registration country AS THE SOURCE STATES IT
    last_contact: float
    source: str
    registration: str = ""          # only when the source sends it
    type_code: str = ""             # ICAO type designator, only when sent

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class Contact:
    """An aircraft placed relative to the point that was asked about."""

    aircraft: Aircraft
    distance_km: float
    bearing_deg: float


@dataclass
class AirspaceReport:
    lat: float
    lon: float
    radius_km: float
    contacts: list[Contact] = field(default_factory=list)
    total: int = 0                  # inside the radius, before the result cap
    source: str = ""                # provider label; "" when nobody answered
    fetched_at: float = 0.0         # epoch s, local clock
    stale: bool = False             # could not refresh; this is an older picture
    error: str = ""                 # why fresh data could not be had
    retry_in_s: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.source) and not self.stale

    def age_s(self, now: float | None = None) -> float | None:
        if not self.fetched_at:
            return None
        return max(0.0, (time.time() if now is None else now) - self.fetched_at)


@dataclass
class TrackResult:
    ident: str
    aircraft: Aircraft | None = None
    source: str = ""
    fetched_at: float = 0.0
    error: str = ""
    retry_in_s: float = 0.0


@dataclass(frozen=True)
class Centre:
    lat: float
    lon: float
    label: str


@dataclass
class LayerState:
    """What the Globe's aircraft layer should be showing.

    Owned by the service, not the view: ``show`` may be asked for before the
    Globe page has ever been built (it is created lazily), and the view picks
    this up when it appears. ``focus_seq`` changes on every ``show`` so the view
    knows to clear the old region and fly to the new one exactly once.
    """

    on: bool = False
    lat: float = 0.0
    lon: float = 0.0
    radius_km: float = 60.0
    label: str = ""
    focus_seq: int = 0


# ── geometry ─────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, 0–360."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def compass_word(bearing: float) -> str:
    return _COMPASS[int(((bearing % 360.0) + 22.5) // 45) % 8]


def coarse(value: float) -> float:
    """The coarse grid value a provider is told instead of the precise one."""
    return round(round(value / GRID_DEG) * GRID_DEG, 4)


# ── parsing ──────────────────────────────────────────────────────────────────

def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_adsb_lol(payload: Any, received_at: float) -> list[Aircraft]:
    """readsb/ADSBExchange-v2 JSON (``{"ac": [...], "now": ms}``) → Aircraft."""
    rows = payload.get("ac") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("no 'ac' list in the response")
    out: list[Aircraft] = []
    for row in rows[:MAX_PARSED]:
        if not isinstance(row, dict):
            continue
        lat, lon = _num(row.get("lat")), _num(row.get("lon"))
        icao = _text(row.get("hex")).lower()
        if lat is None or lon is None or not icao:
            continue                              # can't be placed: not drawn
        if (_text(row.get("category")) in _NOT_AIRCRAFT_CATEGORIES
                or _text(row.get("t")).upper() in _NOT_AIRCRAFT_TYPES):
            continue
        alt_raw = row.get("alt_baro")
        on_ground = alt_raw == "ground"
        alt_ft = None if on_ground else _num(alt_raw)
        if alt_ft is None and not on_ground:
            alt_ft = _num(row.get("alt_geom"))
        speed_kt = _num(row.get("gs"))
        heading = _num(row.get("track"))
        if heading is None:
            heading = _num(row.get("true_heading"))   # never mag_heading: off by variation
        rate_fpm = _num(row.get("baro_rate"))
        if rate_fpm is None:
            rate_fpm = _num(row.get("geom_rate"))
        seen = _num(row.get("seen_pos"))
        if seen is None:
            seen = _num(row.get("seen")) or 0.0
        out.append(Aircraft(
            icao24=icao,
            callsign=_text(row.get("flight")),
            lat=lat, lon=lon,
            altitude_m=None if alt_ft is None else alt_ft * FT_TO_M,
            velocity_ms=None if speed_kt is None else speed_kt * KT_TO_MS,
            heading_deg=None if heading is None else heading % 360.0,
            vertical_rate=None if rate_fpm is None else rate_fpm * FPM_TO_MS,
            on_ground=on_ground,
            origin_country="",                     # adsb.lol does not send one
            last_contact=received_at - max(0.0, seen),
            source="adsb.lol",
            registration=_text(row.get("r")),
            type_code=_text(row.get("t")),
        ))
    return out


def parse_opensky(payload: Any, received_at: float) -> list[Aircraft]:
    """OpenSky ``/states/all`` (state vectors as positional arrays) → Aircraft."""
    if not isinstance(payload, dict):
        raise ValueError("not a JSON object")
    states = payload.get("states") or []           # null when the box is empty
    if not isinstance(states, list):
        raise ValueError("'states' is not a list")
    server_now = _num(payload.get("time"))
    out: list[Aircraft] = []
    for s in states[:MAX_PARSED]:
        if not isinstance(s, (list, tuple)) or len(s) < 12:
            continue
        lat, lon = _num(s[6]), _num(s[5])
        icao = _text(s[0]).lower()
        if lat is None or lon is None or not icao:
            continue
        stamp = _num(s[3]) or _num(s[4])
        if server_now is not None and stamp is not None:
            contact = received_at - max(0.0, server_now - stamp)
        else:
            contact = stamp or received_at
        heading = _num(s[10])
        on_ground = bool(s[8])
        altitude = None if on_ground else _num(s[7])
        if altitude is None and not on_ground and len(s) > 13:
            altitude = _num(s[13])               # geometric, when baro is missing
        out.append(Aircraft(
            icao24=icao,
            callsign=_text(s[1]),
            lat=lat, lon=lon,
            altitude_m=altitude,
            velocity_ms=_num(s[9]),
            heading_deg=None if heading is None else heading % 360.0,
            vertical_rate=_num(s[11]),
            on_ground=on_ground,
            origin_country=_text(s[2]),
            last_contact=contact,
            source="OpenSky",
        ))
    return out


# ── providers ────────────────────────────────────────────────────────────────

class ProviderError(Exception):
    def __init__(self, provider: str, reason: str, *, retry_after: float | None = None,
                 rate_limited: bool = False) -> None:
        super().__init__(f"{provider} {reason}")
        self.provider = provider
        self.reason = reason
        self.retry_after = retry_after
        self.rate_limited = rate_limited


def retry_after_seconds(headers: Any, now: float | None = None) -> float | None:
    """Seconds to wait from a 429's headers, or None if it didn't say.

    Reads the standard ``Retry-After`` (seconds or an HTTP date) and OpenSky's
    own ``X-Rate-Limit-Retry-After-Seconds``; clamped to 1 s – 6 h so a bad
    header can neither hammer the server nor switch the feed off for days.
    """
    for key in ("x-rate-limit-retry-after-seconds", "retry-after"):
        raw = headers.get(key) if headers is not None else None
        if not raw:
            continue
        raw = str(raw).strip()
        try:
            seconds = float(raw)
        except ValueError:
            try:
                when = parsedate_to_datetime(raw).timestamp()
            except (TypeError, ValueError, IndexError):
                continue
            seconds = when - (time.time() if now is None else now)
        return min(6 * 3600.0, max(1.0, seconds))
    return None


@dataclass
class _Breaker:
    open_until: float = 0.0         # monotonic; provider skipped until then
    last_request: float = -1e9      # monotonic; for the politeness spacing
    failures: int = 0
    reason: str = ""


class FlightProvider:
    """A live-position source. Subclasses implement ``area`` and ``lookup``;
    this base does the HTTP, error translation and politeness spacing."""

    name = ""
    label = ""                  # the name ORION says
    min_interval_s = 0.0        # never two requests closer together than this
    default_backoff_s = 30.0    # a 429 with no Retry-After opens the breaker this long

    def __init__(self) -> None:
        self.breaker = _Breaker()
        self.clock: Callable[[], float] = time.monotonic
        self.sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
        self.wall: Callable[[], float] = time.time
        self.credits_remaining: int | None = None

    def cooling_for(self, now: float) -> float:
        return max(0.0, self.breaker.open_until - now)

    def spacing_for(self, now: float) -> float:
        return max(0.0, self.breaker.last_request + self.min_interval_s - now)

    async def _get_json(self, client: Any, url: str, params: dict[str, Any] | None = None) -> Any:
        import httpx
        wait = self.spacing_for(self.clock())
        if wait > 0:
            await self.sleep(wait)
        self.breaker.last_request = self.clock()
        try:
            response = await client.get(url, params=params)
        except httpx.TimeoutException:
            raise ProviderError(self.label, "didn't answer in time") from None
        except httpx.HTTPError as exc:
            raise ProviderError(self.label, f"couldn't be reached ({type(exc).__name__})") from None
        remaining = response.headers.get("x-rate-limit-remaining")
        if remaining is not None:
            try:
                self.credits_remaining = int(remaining)
            except ValueError:
                pass
        if response.status_code == 429:
            raise ProviderError(self.label, "is rate-limiting", rate_limited=True,
                                retry_after=retry_after_seconds(response.headers, self.wall()))
        if response.status_code == 404:
            return None                           # a lookup for something not there
        if response.status_code >= 400:
            raise ProviderError(self.label, f"answered HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise ProviderError(self.label, "sent a reply that isn't JSON") from None

    async def area(self, client: Any, lat: float, lon: float, radius_km: float) -> list[Aircraft]:
        raise NotImplementedError

    async def lookup(self, client: Any, icao24: str = "", callsign: str = "") -> list[Aircraft]:
        return []


class AdsbLolProvider(FlightProvider):
    name = "adsb.lol"
    label = "adsb.lol"
    min_interval_s = 2.0
    default_backoff_s = 15.0    # its 429 carries no Retry-After; measured clear in <10 s

    async def area(self, client: Any, lat: float, lon: float, radius_km: float) -> list[Aircraft]:
        nm = max(1, min(250, math.ceil(radius_km / 1.852)))
        url = ADSB_LOL_AREA.format(lat=f"{lat:.2f}", lon=f"{lon:.2f}", nm=nm)
        payload = await self._get_json(client, url)
        try:
            return parse_adsb_lol(payload, self.wall())
        except ValueError as exc:
            raise ProviderError(self.label, f"sent an unexpected reply ({exc})") from None

    async def lookup(self, client: Any, icao24: str = "", callsign: str = "") -> list[Aircraft]:
        found: list[Aircraft] = []
        if icao24:
            payload = await self._get_json(client, ADSB_LOL_HEX.format(hex=icao24))
            if payload is not None:
                found = parse_adsb_lol(payload, self.wall())
        if not found and callsign:
            payload = await self._get_json(client, ADSB_LOL_CALLSIGN.format(callsign=callsign))
            if payload is not None:
                found = parse_adsb_lol(payload, self.wall())
        return found


class OpenSkyProvider(FlightProvider):
    name = "opensky"
    label = "OpenSky"
    # Anonymous use is 400 credits a day and positions only move every 10 s;
    # 30 s apart makes the budget last hours of an open Globe, not one.
    min_interval_s = 30.0
    default_backoff_s = 60.0

    async def area(self, client: Any, lat: float, lon: float, radius_km: float) -> list[Aircraft]:
        dlat = radius_km / 111.32
        dlon = radius_km / (111.32 * max(0.05, math.cos(math.radians(lat))))
        params = {"lamin": f"{lat - dlat:.2f}", "lomin": f"{lon - dlon:.2f}",
                  "lamax": f"{lat + dlat:.2f}", "lomax": f"{lon + dlon:.2f}"}
        payload = await self._get_json(client, OPENSKY_STATES, params)
        try:
            return parse_opensky(payload or {}, self.wall())
        except ValueError as exc:
            raise ProviderError(self.label, f"sent an unexpected reply ({exc})") from None

    async def lookup(self, client: Any, icao24: str = "", callsign: str = "") -> list[Aircraft]:
        # A callsign search would need EVERY state in the world (the most
        # expensive request there is, ~1 MB); only the address is looked up.
        if not icao24:
            return []
        payload = await self._get_json(client, OPENSKY_STATES, {"icao24": icao24})
        return parse_opensky(payload or {}, self.wall())


# ── the service ──────────────────────────────────────────────────────────────

@dataclass
class _Snapshot:
    cell: tuple[float, float]
    radius_km: float                # the padded radius the provider was asked for
    aircraft: tuple[Aircraft, ...]
    source: str
    fetched_mono: float
    fetched_wall: float
    ttl_s: float


class AviationService:
    """Area queries, flight lookup and the Globe layer's state, over a
    fallback chain of providers with a shared, bounded TTL cache."""

    def __init__(self, providers: list[FlightProvider] | None = None, *,
                 transport: Any = None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
                 ttl_s: float = CACHE_TTL_S) -> None:
        self.providers = providers if providers is not None else [AdsbLolProvider(), OpenSkyProvider()]
        for p in self.providers:
            p.clock, p.wall, p.sleep = clock, wall, sleep
        self.clock, self.wall, self.ttl_s = clock, wall, ttl_s
        self._transport = transport
        self._cache: OrderedDict[tuple[float, float, float], _Snapshot] = OrderedDict()
        self._client: Any = None
        self._lock: asyncio.Lock | None = None
        self._loop: Any = None
        self.layer = LayerState()
        self.last_fetch_ms: float | None = None   # duration of the last good provider call

    # ── plumbing ──────────────────────────────────────────────────────────────

    def _loop_lock(self) -> asyncio.Lock:
        # One client and one lock per event loop: both belong to the loop they
        # were first used on, and tests (and a restart) bring new loops.
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._lock is None:
            self._loop, self._lock, self._client = loop, asyncio.Lock(), None
        return self._lock

    def _make_client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            timeout=httpx.Timeout(8.0, connect=4.0),
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            transport=self._transport, follow_redirects=False)

    async def _client_now(self) -> Any:
        """The loop's client; call with the loop lock held. Measured: importing
        httpx costs ~390 ms and building a client ~730 ms (its SSL context
        loads the CA bundle), both blocking. On the GUI loop that is a second
        of frozen Globe on the first aircraft request, so it is built on a
        worker thread (construction binds nothing to a loop)."""
        if self._client is None or self._client.is_closed:
            self._client = await asyncio.to_thread(self._make_client)
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except RuntimeError:
                pass                      # its loop is gone; nothing to release

    def _fresh(self, cell: tuple[float, float], need_km: float, now: float,
               max_age: float | None = None) -> _Snapshot | None:
        best: _Snapshot | None = None
        for snap in self._cache.values():
            if snap.cell != cell or snap.radius_km < need_km:
                continue
            limit = snap.ttl_s if max_age is None else max_age
            if now - snap.fetched_mono > limit:
                continue
            if best is None or snap.fetched_mono > best.fetched_mono:
                best = snap
        return best

    def _store(self, snap: _Snapshot) -> None:
        key = (*snap.cell, snap.radius_km)
        self._cache[key] = snap
        self._cache.move_to_end(key)
        while len(self._cache) > MAX_CACHED:
            self._cache.popitem(last=False)

    def _explain(self, failures: list[tuple[FlightProvider, str]], now: float) -> tuple[str, float]:
        """One sentence naming each provider's real reason, and when to retry."""
        if not failures:
            return "No flight data source is configured.", 0.0
        reasons = [f"{p.label} {why}" for p, why in failures]
        waits = [w for w in (max(p.cooling_for(now), p.spacing_for(now)) for p, _ in failures) if w > 0]
        sentence = "; ".join(reasons)      # provider names keep their own case
        if waits:
            retry = min(waits)
            return f"{sentence}. I've backed off and will retry in {math.ceil(retry)} s.", retry
        return f"{sentence}.", 0.0

    def _trip(self, provider: FlightProvider, exc: ProviderError, now: float) -> None:
        b = provider.breaker
        b.failures += 1
        if exc.rate_limited:
            wait = exc.retry_after if exc.retry_after is not None else provider.default_backoff_s
        else:
            wait = min(300.0, 5.0 * 2 ** (b.failures - 1))
        b.open_until = now + wait
        b.reason = exc.reason

    async def _call(self, provider: FlightProvider, failures: list, fn: Callable[[], Awaitable[Any]]) -> Any:
        """Run one provider call through its breaker. Returns None on failure
        (recording why); a provider still cooling off is skipped, not awaited,
        and one whose only obstacle is politeness spacing longer than a couple
        of seconds is skipped too, so a turn never waits on it."""
        now = self.clock()
        cooling = provider.cooling_for(now)
        if cooling > 0:
            what = "is rate-limiting" if "rate" in provider.breaker.reason else provider.breaker.reason
            failures.append((provider, f"{what} (backing off)"))
            return None
        if provider.spacing_for(now) > 2.5:
            failures.append((provider, "was asked moments ago (spacing requests out)"))
            return None
        started = time.perf_counter()
        try:
            result = await fn()
        except ProviderError as exc:
            self._trip(provider, exc, self.clock())
            failures.append((provider, exc.reason))
            return None
        except Exception as exc:          # a parser fault is a provider fault
            err = ProviderError(provider.label, f"failed ({type(exc).__name__})")
            self._trip(provider, err, self.clock())
            failures.append((provider, err.reason))
            return None
        self.last_fetch_ms = (time.perf_counter() - started) * 1000.0
        provider.breaker.failures = 0
        provider.breaker.open_until = 0.0
        provider.breaker.reason = ""
        return result

    # ── area queries ──────────────────────────────────────────────────────────

    async def nearby(self, lat: float, lon: float, radius_km: float = 50.0, *,
                     limit: int = MAX_AIRCRAFT) -> AirspaceReport:
        """Aircraft within *radius_km* of the precise point, nearest first."""
        radius_km = max(1.0, min(MAX_RADIUS_KM, float(radius_km)))
        cell = (coarse(lat), coarse(lon))
        need = math.ceil((radius_km + PRIVACY_PAD_KM) / 10.0) * 10.0
        snapshot, error, retry = await self._area(cell, need)
        report = AirspaceReport(lat=lat, lon=lon, radius_km=radius_km,
                                error=error, retry_in_s=retry)
        if snapshot is None:
            return report
        contacts = []
        for ac in snapshot.aircraft:
            d = haversine_km(lat, lon, ac.lat, ac.lon)
            if d <= radius_km:
                contacts.append(Contact(ac, d, bearing_deg(lat, lon, ac.lat, ac.lon)))
        contacts.sort(key=lambda c: c.distance_km)
        report.total = len(contacts)
        report.contacts = contacts[:max(0, int(limit))]
        report.source = snapshot.source
        report.fetched_at = snapshot.fetched_wall
        report.stale = bool(error)
        return report

    async def overhead(self, lat: float, lon: float, radius_km: float = 25.0) -> AirspaceReport:
        return await self.nearby(lat, lon, radius_km)

    async def _area(self, cell: tuple[float, float], need_km: float) -> tuple[_Snapshot | None, str, float]:
        # Serialised: the Globe's poll and a spoken question arriving together
        # make ONE request — the second finds the first one's answer cached.
        async with self._loop_lock():
            now = self.clock()
            snap_ = self._fresh(cell, need_km, now)
            if snap_ is not None:
                return snap_, "", 0.0
            failures: list[tuple[FlightProvider, str]] = []
            client = await self._client_now()
            for provider in self.providers:
                aircraft = await self._call(
                    provider, failures,
                    lambda p=provider: p.area(client, cell[0], cell[1], need_km))
                if aircraft is None:
                    continue
                snap_ = _Snapshot(cell=cell, radius_km=need_km, aircraft=tuple(aircraft),
                                  source=provider.label, fetched_mono=self.clock(),
                                  fetched_wall=self.wall(),
                                  ttl_s=max(self.ttl_s, provider.min_interval_s))
                self._store(snap_)
                return snap_, "", 0.0
            error, retry = self._explain(failures, self.clock())
            return self._fresh(cell, need_km, self.clock(), max_age=STALE_LIMIT_S), error, retry

    # ── one flight ────────────────────────────────────────────────────────────

    async def track(self, ident: str) -> TrackResult:
        """The latest state of one aircraft, by callsign or ICAO24 address."""
        raw = re.sub(r"\s+", "", str(ident or "")).upper()
        result = TrackResult(ident=raw)
        if not raw or not re.fullmatch(r"~?[A-Z0-9-]{2,10}", raw):
            result.error = "That doesn't look like a callsign or an ICAO24 code."
            return result
        icao = raw.lower() if re.fullmatch(r"~?[0-9A-F]{6}", raw) else ""
        callsign = raw if re.fullmatch(r"[A-Z0-9]{2,8}", raw) else ""
        # Anything already on screen answers without a request.
        now = self.clock()
        for snap_ in reversed(self._cache.values()):
            if now - snap_.fetched_mono > snap_.ttl_s:
                continue
            for ac in snap_.aircraft:
                if (icao and ac.icao24 == icao) or (callsign and ac.callsign.upper() == callsign):
                    result.aircraft, result.source, result.fetched_at = ac, snap_.source, snap_.fetched_wall
                    return result
        async with self._loop_lock():
            failures: list[tuple[FlightProvider, str]] = []
            answered = False
            client = await self._client_now()
            for provider in self.providers:
                found = await self._call(
                    provider, failures,
                    lambda p=provider: p.lookup(client, icao24=icao, callsign=callsign))
                if found is None:
                    continue
                answered = True
                match = [ac for ac in found
                         if (icao and ac.icao24 == icao)
                         or (callsign and ac.callsign.upper() == callsign)]
                if match:
                    freshest = max(match, key=lambda ac: ac.last_contact)
                    result.aircraft, result.source = freshest, provider.label
                    result.fetched_at = self.wall()
                    return result
            if not answered:
                result.error, result.retry_in_s = self._explain(failures, self.clock())
        return result

    # ── the Globe layer ───────────────────────────────────────────────────────

    def show_layer(self, lat: float, lon: float, radius_km: float, label: str = "") -> LayerState:
        self.layer = LayerState(on=True, lat=lat, lon=lon,
                                radius_km=max(1.0, min(MAX_RADIUS_KM, float(radius_km))),
                                label=label, focus_seq=self.layer.focus_seq + 1)
        return self.layer

    def hide_layer(self) -> None:
        self.layer.on = False

    def status(self) -> dict[str, Any]:
        now = self.clock()
        return {p.label: {"cooling_s": round(p.cooling_for(now), 1),
                          "failures": p.breaker.failures,
                          "reason": p.breaker.reason,
                          "credits_remaining": p.credits_remaining}
                for p in self.providers}


_SERVICE: AviationService | None = None


def get_service() -> AviationService:
    """The process-wide service the tool and the Globe share."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AviationService()
    return _SERVICE


# ── the Globe payload ────────────────────────────────────────────────────────

def globe_batch(report: AirspaceReport, *, reset: bool = False) -> dict[str, Any]:
    """A compact, JSON-ready batch for the page's ``orionAircraft``."""
    def r(v: float | None, nd: int) -> float | None:
        return None if v is None else round(v, nd)
    return {
        "reset": bool(reset),
        "source": report.source,
        "fetched": round(report.fetched_at, 1),
        "stale": report.stale,
        "centre": {"lat": round(report.lat, 5), "lon": round(report.lon, 5),
                   "radius_km": report.radius_km},
        "aircraft": [{
            "id": c.aircraft.icao24, "cs": c.aircraft.callsign,
            "lat": round(c.aircraft.lat, 5), "lon": round(c.aircraft.lon, 5),
            "alt": r(c.aircraft.altitude_m, 0), "v": r(c.aircraft.velocity_ms, 1),
            "hdg": r(c.aircraft.heading_deg, 1), "vr": r(c.aircraft.vertical_rate, 2),
            "gnd": c.aircraft.on_ground, "t": round(c.aircraft.last_contact, 1),
            "reg": c.aircraft.registration, "typ": c.aircraft.type_code,
            "ctry": c.aircraft.origin_country, "d": round(c.distance_km, 2),
        } for c in report.contacts[:MAX_AIRCRAFT]],
    }


# ── speech ───────────────────────────────────────────────────────────────────

def _article(word: str) -> str:
    # Letters whose NAME starts with a vowel sound: an A320, an F900, an H25B.
    return "an" if word[:1].upper() in "AEFHILMNORSX" else "a"


def _identity(ac: Aircraft) -> str:
    extras = [ac.registration] if ac.registration and ac.registration != ac.callsign else []
    if ac.type_code:
        extras.append(f"{_article(ac.type_code)} {ac.type_code}")
    if ac.origin_country:
        extras.append(f"registered in {ac.origin_country}")
    name = ac.callsign or f"an unidentified aircraft ({ac.icao24})"
    return f"{name} ({', '.join(extras)})" if extras else name


def _metres(value: float) -> str:
    return f"{int(round(value / 100.0) * 100):,} m"


def _distance(km: float) -> str:
    return "under 1 km" if km < 1.0 else f"{km:.0f} km"


def describe_contact(c: Contact, *, detail: bool = False) -> str:
    """'RYR4TK (EI-DWF, a B38M) at 11,200 m, heading 140°, 6 km to the
    north-east' — only the parts the source actually supplied."""
    ac = c.aircraft
    bits = [_identity(ac)]
    if ac.on_ground:
        bits.append("on the ground")
    else:
        if ac.altitude_m is not None:
            bits.append(f"at {_metres(ac.altitude_m)}")
        if ac.heading_deg is not None:
            bits.append(f"heading {ac.heading_deg:.0f}°")
        if detail and ac.velocity_ms is not None:
            bits.append(f"{ac.velocity_ms * 3.6:,.0f} km/h")
    bits.append(f"{_distance(c.distance_km)} to the {compass_word(c.bearing_deg)}")
    return bits[0] + " " + ", ".join(bits[1:])


def summarise(report: AirspaceReport, *, where: str = "", airborne_only: bool = False,
              now: float | None = None) -> str:
    """A spoken answer. Leads with the reason and the age when the data could
    not be refreshed; never presents an old picture as the current one."""
    if not report.source:
        return report.error or "No flight data source answered."
    place = f" of {where}" if where else ""
    radius = f"{report.radius_km:g} km"
    contacts = report.contacts
    ground = [c for c in contacts if c.aircraft.on_ground]
    if airborne_only:
        contacts = [c for c in contacts if not c.aircraft.on_ground]
    age = report.age_s(now) or 0.0
    parts: list[str] = []
    if report.stale:
        parts.append(f"I couldn't get a fresh picture: {report.error} "
                     f"This is the last one I have, {age:.0f} s old.")
    counted = len(contacts) if airborne_only else report.total
    if not contacts:
        flying = "flying" if airborne_only else "showing"
        parts.append(f"Nothing is {flying} within {radius}{place} right now.")
    else:
        noun = "aircraft"
        verb = "is" if counted == 1 else "are"
        state = " in the air" if airborne_only else ""
        line = f"There {verb} {counted} {noun}{state} within {radius}{place}"
        if airborne_only and ground:
            line += f", plus {len(ground)} on the ground"
        elif not airborne_only and ground:
            line += f" ({len(ground)} of them on the ground)"
        parts.append(line + ".")
        parts.append(f"The closest is {describe_contact(contacts[0], detail=True)}.")
        rest = contacts[1:5]
        if rest:
            parts.append("Then " + "; ".join(describe_contact(c) for c in rest) + ".")
    if airborne_only and not contacts and ground:
        parts.append(f"{len(ground)} {'is' if len(ground) == 1 else 'are'} on the ground.")
    if not report.stale:
        parts.append(f"Positions from {report.source}, {age:.0f} s old.")
    return " ".join(parts)


def describe_track(result: TrackResult, centre: Centre | None = None,
                   now: float | None = None) -> tuple[str, bool]:
    if result.error:
        return result.error, False
    ac = result.aircraft
    if ac is None:
        return (f"I can't see {result.ident} on the public ADS-B feeds right now. "
                "It may be on the ground, out of receiver range, or not "
                "transmitting under that identity."), True
    bits = [_identity(ac), "is"]
    if ac.on_ground:
        bits.append("on the ground")
    else:
        motion = []
        if ac.altitude_m is not None:
            motion.append(f"at {_metres(ac.altitude_m)}")
        if ac.heading_deg is not None:
            motion.append(f"heading {ac.heading_deg:.0f}°")
        if ac.velocity_ms is not None:
            motion.append(f"at {ac.velocity_ms * 3.6:,.0f} km/h")
        if ac.vertical_rate is not None and abs(ac.vertical_rate) >= 1.0:
            motion.append(("climbing" if ac.vertical_rate > 0 else "descending")
                          + f" {abs(ac.vertical_rate):.0f} m/s")
        bits.append(", ".join(motion) if motion else "airborne")
    ns = "N" if ac.lat >= 0 else "S"
    ew = "E" if ac.lon >= 0 else "W"
    text = f"{' '.join(bits)}, over {abs(ac.lat):.2f}°{ns} {abs(ac.lon):.2f}°{ew}"
    if centre is not None:
        d = haversine_km(centre.lat, centre.lon, ac.lat, ac.lon)
        b = bearing_deg(centre.lat, centre.lon, ac.lat, ac.lon)
        text += f" — {_distance(d)} {compass_word(b)} of {centre.label or 'you'}"
    age = max(0.0, (time.time() if now is None else now) - ac.last_contact)
    return f"{text}. Position from {result.source}, {age:.0f} s old.", True


# ── the tool ─────────────────────────────────────────────────────────────────

NO_LOCATION = ("I don't know where you are yet, so I can't say what's overhead. "
               "Tell me your town or postcode and I'll look at the sky there.")
_HERE = {"", "here", "me", "us", "home", "overhead", "above", "above us",
         "above me", "my location", "near me", "near us", "where i am"}
_ACTIONS = {
    "overhead": "overhead", "above": "overhead", "over_us": "overhead", "sky": "overhead",
    "what_is_overhead": "overhead",
    "nearby": "nearby", "near": "nearby", "around": "nearby", "list": "nearby",
    "track": "track", "find": "track", "follow": "track", "where": "track", "lookup": "track",
    "show": "show", "map": "show", "globe": "show", "draw": "show", "display": "show",
    "hide": "hide", "off": "hide", "clear": "hide", "stop": "hide",
}
_DEFAULT_RADIUS = {"overhead": 25.0, "nearby": 50.0, "show": 60.0}


async def resolve_centre(args: dict[str, Any], *, geo: Any = None,
                         temporal: Any = None) -> Centre | str:
    """Where to look: explicit coordinates, then a named place, then the
    user's own locality. Returns a sentence (asking) when none is known —
    never a guessed location."""
    lat, lon = _num(args.get("lat")), _num(args.get("lon"))
    if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
        return Centre(lat, lon, f"{lat:.2f}, {lon:.2f}")
    place = _text(args.get("place")) or _text(args.get("location"))
    if place.lower() not in _HERE:
        if geo is None:
            return f"I can't look up where '{place}' is right now — the map service isn't available."
        try:
            results = await geo.geocode(place, limit=1)
        except Exception:
            results = []
        if not results:
            return f"I couldn't find '{place}' on the map. Try a town, city or postcode."
        p = results[0]
        return Centre(float(p.lat), float(p.lon), str(getattr(p, "name", "") or place))
    coords = None
    try:
        coords = temporal.coordinates() if temporal is not None else None
    except Exception:
        coords = None
    if coords:
        try:
            label = str(temporal.locality() or "").split(",")[0].strip()
        except Exception:
            label = ""
        return Centre(float(coords[0]), float(coords[1]), label)
    return NO_LOCATION


def _radius(args: dict[str, Any], default: float) -> float:
    value = _num(args.get("radius_km"))
    if value is None:
        value = _num(args.get("radius"))
    if value is None or value <= 0:
        value = default
    return max(1.0, min(MAX_RADIUS_KM, value))


def _emit(bus: Any, signal: str, *payload: Any) -> None:
    try:
        getattr(bus, signal).emit(*payload)
    except Exception:
        pass                              # no GUI (headless / tests): nothing to drive


async def run_tool(args: Any, *, service: AviationService | None = None, geo: Any = None,
                   temporal: Any = None, bus: Any = None) -> tuple[str, bool]:
    """The ``aviation`` tool: (spoken text, ok)."""
    args = args if isinstance(args, dict) else {}
    service = service or get_service()
    raw_action = str(args.get("action") or "overhead").strip().lower().replace(" ", "_")
    action = _ACTIONS.get(raw_action, raw_action)

    if action == "hide":
        service.hide_layer()
        _emit(bus, "dashboard_event", "aviation", {"action": "hide"})
        return "The aircraft layer is off; the Globe has stopped polling for flights.", True

    if action == "track":
        ident = (_text(args.get("callsign")) or _text(args.get("icao24"))
                 or _text(args.get("ident")) or _text(args.get("query")))
        if not ident:
            return ("Which aircraft? Give me a callsign such as BAW123, or a "
                    "six-character ICAO24 code such as 4070ea."), False
        where = {k: args[k] for k in ("lat", "lon", "place", "location") if k in args}
        centre = await resolve_centre(where, geo=geo, temporal=temporal)
        return describe_track(await service.track(ident),
                              centre if isinstance(centre, Centre) else None)

    if action not in _DEFAULT_RADIUS:
        return (f"Unsupported aviation action: {raw_action}. Use overhead, nearby, "
                "track, show, or hide."), False

    centre = await resolve_centre(args, geo=geo, temporal=temporal)
    if isinstance(centre, str):
        return centre, False
    radius = _radius(args, _DEFAULT_RADIUS[action])

    if action == "show":
        service.show_layer(centre.lat, centre.lon, radius, centre.label)
        # Open the Globe first: its page takes a moment to load, and it reads
        # the layer from the service (the fetch below lands in the shared cache).
        _emit(bus, "gui_command", {"action": "globe", "target": ""})
        _emit(bus, "dashboard_event", "aviation", {"action": "show"})
        report = await service.nearby(centre.lat, centre.lon, radius)
        place = f" of {centre.label}" if centre.label else ""
        if not report.source:
            return (f"{report.error} The Globe is open on {centre.label or 'your area'} "
                    "and will draw the aircraft as soon as a feed answers."), False
        parts = []
        if report.stale:
            parts.append(f"I couldn't get a fresh picture: {report.error}")
        if report.contacts:
            parts.append(f"Showing {len(report.contacts)} aircraft within {radius:g} km"
                         f"{place} on the Globe; click one for its details. They keep "
                         "moving and refresh while the Globe is open.")
            parts.append(f"The closest is {describe_contact(report.contacts[0], detail=True)}.")
        else:
            parts.append(f"The Globe is on {centre.label or 'your area'}, but nothing is "
                         f"showing within {radius:g} km right now; I'll keep watching "
                         "while it's open.")
        parts.append(f"Positions from {report.source}, {report.age_s() or 0:.0f} s old.")
        return " ".join(parts), report.ok

    report = await service.nearby(centre.lat, centre.lon, radius)
    return summarise(report, where=centre.label, airborne_only=(action == "overhead")), report.ok


__all__ = [
    "Aircraft", "AirspaceReport", "AviationService", "AdsbLolProvider", "Centre",
    "Contact", "FlightProvider", "LayerState", "NO_LOCATION", "OpenSkyProvider",
    "ProviderError", "TrackResult", "bearing_deg", "compass_word", "describe_contact",
    "describe_track", "get_service", "globe_batch", "haversine_km", "parse_adsb_lol",
    "parse_opensky", "resolve_centre", "retry_after_seconds", "run_tool", "summarise",
]
