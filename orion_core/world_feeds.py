"""Official world feeds shared by the Globe and ORION's read-only MCP server.

Imports do not start a connection or a timer. Downloads occur only on demand,
with persistent, transactional request deadlines shared between processes.
Satellite positions are SGP4 predictions, never observations or imagery.
"""

from __future__ import annotations

import asyncio
import html
import json
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .constants import CONFIG_DIR
from .db import apply_pragmas
from .lazy_import import LazyModule

aiohttp = LazyModule("aiohttp")

MAX_BYTES = 2_097_152
MAX_ITEMS = 2000
MAX_SATELLITES = 64
LAYER_NAMES = ("satellites", "earthquakes", "tsunami", "weather", "events")
#: Positions kept for a moving event's track (a storm's path).
MAX_TRACK = 60


@dataclass(frozen=True)
class Feed:
    key: str
    layer: str
    url: str
    source: str
    interval: int
    coverage: str


FEEDS = (
    Feed("stations", "satellites",
         "https://celestrak.org/NORAD/elements/gp.php?GROUP=STATIONS&FORMAT=JSON",
         "CelesTrak", 7200,
         "CelesTrak space stations group; predicted positions, not all satellites"),
    Feed("earthquakes", "earthquakes",
         "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
         "USGS", 120, "USGS earthquake catalogue, past day; detection coverage varies"),
    Feed("ntwc", "tsunami", "https://www.tsunami.gov/events/xml/PAAQAtom.xml",
         "NOAA National Tsunami Warning Center", 120,
         "NOAA NTWC bulletins; information statements can report no tsunami threat"),
    Feed("ptwc", "tsunami", "https://www.tsunami.gov/events/xml/PHEBAtom.xml",
         "NOAA Pacific Tsunami Warning Center", 120,
         "NOAA PTWC bulletins; not a globally comprehensive active-warning service"),
    Feed("weather", "weather",
         "https://www.metoffice.gov.uk/public/data/PWSCache/WarningsRSS/Region/UK",
         "Met Office", 120, "United Kingdom severe weather warning summaries"),
    Feed("eonet", "events",
         "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&days=30&limit=300",
         "NASA EONET", 1800,
         "NASA Earth Observatory Natural Event Tracker: open wildfires, severe storms, "
         "volcanoes, floods and ice from the past 30 days; curated, not exhaustive"),
)
_FEED_BY_KEY = {feed.key: feed for feed in FEEDS}


def _iso(timestamp: float | None) -> str | None:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _date(value: Any) -> float | None:
    if not isinstance(value, str) or len(value) > 100:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (ValueError, TypeError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _number(value: Any, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a coordinate or measurement")  # noqa: TRY004 — malformed feed data
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid coordinate or measurement") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError("Coordinate or measurement is outside its bounds")
    return result


def _text(value: Any, limit: int = 1000) -> str:
    text = str(value or "")[:limit * 2]
    text = re.sub(r"<[^>]*>", " ", text)
    return " ".join(html.unescape(text).split())[:limit]


def _link(value: Any, feed: Feed) -> str:
    """Only official HTTPS content links survive; links are never fetched."""
    try:
        candidate = str(value or "")
        parts = urlsplit(candidate)
        expected = urlsplit(feed.url).hostname
        permitted = {expected}
        if feed.layer == "weather":
            permitted.update({"weather.metoffice.gov.uk", "www.metoffice.gov.uk"})
        if parts.scheme == "https" and parts.hostname in permitted and \
                parts.port in (None, 443) and not parts.username and not parts.password \
                and len(candidate) <= 1500:
            return candidate
    except ValueError:
        pass
    return feed.url


def _json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8-sig"),
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError(f"Non-finite JSON number: {value}")))
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid JSON feed") from exc


def _earthquakes(raw: bytes, feed: Feed) -> list[dict[str, Any]]:
    body = _json(raw)
    if not isinstance(body, dict) or body.get("type") != "FeatureCollection" or \
            not isinstance(body.get("features"), list):
        raise ValueError("Expected a USGS GeoJSON FeatureCollection")
    features = body["features"]
    if len(features) > MAX_ITEMS:
        raise ValueError("Earthquake feed contains too many records")
    items = []
    for feature in features:
        try:
            props, geometry = feature["properties"], feature["geometry"]
            if geometry.get("type") != "Point" or props.get("type") != "earthquake":
                continue
            coords = geometry["coordinates"]
            lon = _number(coords[0], -180, 180)
            lat = _number(coords[1], -90, 90)
            depth = _number(coords[2], -20, 1000)
            magnitude = None if props.get("mag") is None else _number(props["mag"], -10, 15)
            event_time = _number(props["time"], 0, 32_503_680_000_000) / 1000
            identifier = _text(feature["id"], 100)
            if not identifier:
                continue
            items.append({"id": identifier, "title": _text(props.get("title") or props.get("place"), 300),
                          "latitude": lat, "longitude": lon, "altitude_km": 0,
                          "magnitude": magnitude, "depth_km": depth,
                          "event_at": _iso(event_time), "source": feed.source,
                          "source_url": _link(props.get("url"), feed),
                          "details": _text(props.get("place"), 500),
                          "review_status": _text(props.get("status"), 30)})
        except (KeyError, TypeError, IndexError, ValueError, AttributeError):
            continue
    if features and not items:
        raise ValueError("USGS feed contained no valid earthquake records")
    return items


def _eonet(raw: bytes, feed: Feed) -> list[dict[str, Any]]:
    """NASA EONET open events: each at its latest reported position, a moving
    one (a storm) with its recent track."""
    body = _json(raw)
    if not isinstance(body, dict) or not isinstance(body.get("events"), list):
        raise ValueError("Expected an EONET events document")
    events = body["events"]
    if len(events) > MAX_ITEMS:
        raise ValueError("EONET feed contains too many records")
    items = []
    for event in events:
        try:
            identifier = _text(event["id"], 100)
            points = []
            for geometry in event.get("geometry") or []:
                if not isinstance(geometry, dict) or geometry.get("type") != "Point":
                    continue
                coords = geometry["coordinates"]
                when = _date(geometry.get("date"))
                points.append((when or 0.0, _number(coords[1], -90, 90),
                               _number(coords[0], -180, 180), geometry))
            if not identifier or not points:
                continue
            points.sort(key=lambda point: point[0])
            when, lat, lon, latest = points[-1]
            categories = [c for c in event.get("categories") or [] if isinstance(c, dict)]
            category = _text(categories[0].get("title"), 60) if categories else "Natural event"
            magnitude = None
            unit = ""
            if latest.get("magnitudeValue") is not None:
                magnitude = _number(latest["magnitudeValue"], -1e6, 1e9)
                unit = _text(latest.get("magnitudeUnit"), 20)
            sources = [_text(src.get("id"), 40) for src in event.get("sources") or []
                       if isinstance(src, dict) and src.get("id")]
            details = [category]
            if magnitude is not None:
                details.append(f"{magnitude:g} {unit}".strip())
            if len(points) > 1:
                details.append(f"{len(points)} reported positions")
            if sources:
                details.append("Reported by " + ", ".join(sources[:4]))
            item = {"id": identifier, "title": _text(event.get("title"), 300) or category,
                    "latitude": lat, "longitude": lon, "altitude_km": 0,
                    "category": category, "magnitude": magnitude, "magnitude_unit": unit,
                    "event_at": _iso(when) if when else None, "source": feed.source,
                    "source_url": _link(event.get("link"), feed),
                    "details": " · ".join(details)}
            if len(points) > 1:
                item["track"] = [[round(p[1], 4), round(p[2], 4)] for p in points[-MAX_TRACK:]]
            items.append(item)
        except (KeyError, TypeError, IndexError, ValueError, AttributeError):
            continue
    if events and not items:
        raise ValueError("EONET feed contained no valid events")
    return items


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml(raw: bytes, feed: Feed) -> list[dict[str, Any]]:
    # UTF-8 only prevents a UTF-16 encoding from hiding declarations from this
    # check. ElementTree does not fetch external entities, and declarations
    # are refused altogether before parsing to prevent entity expansion.
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("Expected a UTF-8 XML feed") from exc
    upper = decoded.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise ValueError("XML declarations and entities are not permitted")
    try:
        root = ET.fromstring(decoded)
    except (ET.ParseError, RecursionError) as exc:
        raise ValueError("Invalid XML feed") from exc
    if _local(root.tag) not in {"rss", "feed"}:
        raise ValueError("Expected an RSS or Atom feed")
    entries = []
    for index, element in enumerate(root.iter()):
        if index > 16_000:
            raise ValueError("XML feed contains too many elements")
        if _local(element.tag) in {"entry", "item"}:
            entries.append(element)
            if len(entries) > 200:
                raise ValueError("Warning feed contains too many records")
    items = []
    for entry in entries:
        values = {_local(child.tag): "".join(child.itertext()) for child in entry}
        title = _text(values.get("title"), 400)
        if not title:
            continue
        links = [child.get("href") or child.text for child in entry
                 if _local(child.tag) == "link" and child.get("rel", "alternate") == "alternate"]
        link = _link(next(iter(links), None), feed)
        published = values.get("updated") or values.get("published") or values.get("pubDate")
        item: dict[str, Any] = {
            "id": _text(values.get("id") or values.get("guid") or f"{feed.key}:{title}", 400),
            "title": title, "source": feed.source, "source_url": link,
            "event_at": _iso(_date(published)),
            "details": _text(values.get("summary") or values.get("description") or values.get("content"), 1500),
            "kind": "bulletin" if feed.layer == "tsunami" else "warning_summary",
        }
        # GeoRSS point is an event location, not a threatened coastline.
        point = values.get("point")
        if point:
            try:
                parts = point.split()
                if len(parts) == 2:
                    item.update(latitude=_number(parts[0], -90, 90),
                                longitude=_number(parts[1], -180, 180), altitude_km=0,
                                location_kind="feed_event_location")
            except ValueError:
                pass
        items.append(item)
    if entries and not items:
        raise ValueError("Warning feed contained no valid entries")
    return items


_OMM_NUMERIC = {
    "NORAD_CAT_ID": (1, 999_999_999), "MEAN_MOTION": (0.01, 30),
    "ECCENTRICITY": (0, 0.9999999), "INCLINATION": (0, 180),
    "RA_OF_ASC_NODE": (0, 360), "ARG_OF_PERICENTER": (0, 360),
    "MEAN_ANOMALY": (0, 360), "BSTAR": (-10, 10),
    "MEAN_MOTION_DOT": (-100, 100), "MEAN_MOTION_DDOT": (-100, 100),
    "EPHEMERIS_TYPE": (0, 9), "ELEMENT_SET_NO": (0, 999999), "REV_AT_EPOCH": (0, 99999999),
}


def _stations(raw: bytes, _feed: Feed) -> list[dict[str, Any]]:
    body = _json(raw)
    if not isinstance(body, list) or len(body) > MAX_SATELLITES:
        raise ValueError("Expected a bounded CelesTrak stations OMM list")
    items = []
    for row in body:
        try:
            if not isinstance(row, dict):
                continue
            normal = {key: str(_number(row[key], *bounds)) for key, bounds in _OMM_NUMERIC.items()}
            for key in ("NORAD_CAT_ID", "EPHEMERIS_TYPE", "ELEMENT_SET_NO", "REV_AT_EPOCH"):
                normal[key] = str(int(float(normal[key])))
            epoch = _date(row.get("EPOCH"))
            if epoch is None:
                continue
            normal.update(EPOCH=datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f"),
                          CLASSIFICATION_TYPE="U", OBJECT_NAME=_text(row.get("OBJECT_NAME"), 150),
                          OBJECT_ID=_text(row.get("OBJECT_ID"), 30))
            items.append(normal)
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            continue
    if body and not items:
        raise ValueError("CelesTrak feed contained no valid orbital elements")
    return items


def parse_feed(feed: Feed, raw: bytes) -> list[dict[str, Any]]:
    if len(raw) > MAX_BYTES:
        raise ValueError("Feed exceeds the response size limit")
    if feed.layer == "satellites":
        return _stations(raw, feed)
    if feed.layer == "earthquakes":
        return _earthquakes(raw, feed)
    if feed.layer == "events":
        return _eonet(raw, feed)
    return _xml(raw, feed)


def _validate_cache(feed: Feed, rows: Any) -> list[dict[str, Any]]:
    """Cached files get the same coordinate and link checks as fresh data."""
    if not isinstance(rows, list) or len(rows) > MAX_ITEMS or \
            any(not isinstance(row, dict) for row in rows):
        raise ValueError("Invalid cached feed")
    if feed.layer == "satellites":
        return _stations(json.dumps(rows, allow_nan=False).encode("utf-8"), feed)
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or \
                not isinstance(row.get("title"), str) or len(row["title"]) > 400:
            raise ValueError("Invalid cached record")
        if row.get("source_url") != _link(row.get("source_url"), feed):
            raise ValueError("Invalid cached source link")
        has_latitude, has_longitude = "latitude" in row, "longitude" in row
        if has_latitude != has_longitude or (
                feed.layer in ("earthquakes", "events") and not has_latitude):
            raise ValueError("Incomplete cached coordinates")
        track = row.get("track")
        if track is not None:
            if not isinstance(track, list) or len(track) > MAX_TRACK:
                raise ValueError("Invalid cached track")
            for point in track:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Invalid cached track point")
                _number(point[0], -90, 90)
                _number(point[1], -180, 180)
        if has_latitude:
            _number(row["latitude"], -90, 90)
            _number(row["longitude"], -180, 180)
            _number(row.get("altitude_km", 0), 0, 100000)
        if feed.layer == "earthquakes":
            _number(row.get("depth_km"), -20, 1000)
            if row.get("magnitude") is not None:
                _number(row["magnitude"], -10, 15)
        if row.get("event_at") is not None and _date(row["event_at"]) is None:
            raise ValueError("Invalid cached event date")
    return rows


def _geodetic(position: tuple[float, float, float], jd: float) -> tuple[float, float, float]:
    """TEME km to WGS84 geodetic position using Vallado's GMST rotation.

    UT1 is approximated by UTC; no polar-motion/Earth-orientation correction
    is claimed. Appropriate for this visualisation, not collision avoidance.
    """
    century = (jd - 2451545.0) / 36525.0
    seconds = (67310.54841 + (876600 * 3600 + 8640184.812866) * century
               + 0.093104 * century ** 2 - 6.2e-6 * century ** 3)
    theta = (seconds * math.pi / 43200) % (2 * math.pi)
    x, y, z = position
    ecef_x = x * math.cos(theta) + y * math.sin(theta)
    ecef_y = -x * math.sin(theta) + y * math.cos(theta)
    radius = math.hypot(ecef_x, ecef_y)
    longitude = math.atan2(ecef_y, ecef_x)
    flattening = 1 / 298.257223563
    eccentricity_squared = flattening * (2 - flattening)
    latitude = math.atan2(z, radius * (1 - eccentricity_squared))
    for _ in range(8):
        normal = 6378.137 / math.sqrt(1 - eccentricity_squared * math.sin(latitude) ** 2)
        latitude = math.atan2(z + eccentricity_squared * normal * math.sin(latitude), radius)
    normal = 6378.137 / math.sqrt(1 - eccentricity_squared * math.sin(latitude) ** 2)
    altitude = radius / max(1e-12, abs(math.cos(latitude))) - normal
    if radius < 1e-9:
        altitude = abs(z) - 6356.752314245
    return math.degrees(latitude), math.degrees(longitude), altitude


def predict_satellites(rows: list[dict[str, Any]], now: float) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from sgp4 import omm
        from sgp4.api import Satrec, jday
    except ImportError:
        return [], "Satellite prediction needs the sgp4 integration dependency"
    instant = datetime.fromtimestamp(now, timezone.utc)
    jd, fraction = jday(instant.year, instant.month, instant.day, instant.hour,
                        instant.minute, instant.second + instant.microsecond / 1_000_000)
    items = []
    skipped = 0
    for row in rows[:MAX_SATELLITES]:
        try:
            epoch = _date(row.get("EPOCH"))
            if epoch is None or not -86400 <= now - epoch <= 7 * 86400:
                skipped += 1
                continue
            satellite = Satrec()
            omm.initialize(satellite, row)
            code, position, _velocity = satellite.sgp4(jd, fraction)
            if code:
                skipped += 1
                continue
            lat, lon, altitude = _geodetic(position, jd + fraction)
            _number(lat, -90, 90)
            _number(lon, -180, 180)
            _number(altitude, 0, 100_000)
            items.append({"id": row["NORAD_CAT_ID"], "title": row["OBJECT_NAME"] or row["NORAD_CAT_ID"],
                          "latitude": lat, "longitude": lon, "altitude_km": altitude,
                          "epoch": _iso(epoch), "event_at": _iso(epoch), "predicted_at": _iso(now),
                          "source": "CelesTrak", "source_url": FEEDS[0].url,
                          "details": "SGP4 predicted position from orbital elements; not an observation",
                          "position_kind": "sgp4_prediction", "earth_rotation": "GMST; UTC approximates UT1"})
        except (KeyError, ValueError, TypeError, OverflowError, ZeroDivisionError):
            skipped += 1
    return items, f"{skipped} orbital records were too old or could not be propagated" if skipped else None


class FeedFailure(Exception):
    def __init__(self, message: str, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after(value: str | None, now: float) -> float:
    try:
        delay = float(value or "")
    except ValueError:
        when = _date(value)
        delay = (when - now) if when is not None else 0
    # A provider may ask for a longer wait than our normal retry ceiling.
    return max(0, delay) if math.isfinite(delay) else 0


class WorldFeedService:
    """An on-demand async service; call ``close`` when its owner shuts down."""

    def __init__(self, cache_path: Path | None = None, *, clock: Callable[[], float] = time.time):
        self.cache_path = Path(cache_path) if cache_path else CONFIG_DIR / "world_feeds.db"
        self._clock = clock
        self._session: aiohttp.ClientSession | None = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._closed = False

    @contextmanager
    def _database(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.cache_path, timeout=1.0)
        connection.row_factory = sqlite3.Row
        try:
            apply_pragmas(connection)
            connection.execute("CREATE TABLE IF NOT EXISTS feeds (key TEXT PRIMARY KEY, data TEXT, "
                               "fetched REAL DEFAULT 0, next_attempt REAL DEFAULT 0, "
                               "failures INTEGER DEFAULT 0, error TEXT, busy_until REAL DEFAULT 0)")
            connection.commit()
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _read(self, feed: Feed) -> dict[str, Any]:
        with self._database() as database:
            row = database.execute("SELECT * FROM feeds WHERE key = ?", (feed.key,)).fetchone()
        if row is None:
            return {"data": None, "fetched": 0, "next_attempt": 0, "error": None, "busy_until": 0}
        state = dict(row)
        try:
            encoded = state["data"]
            if encoded is not None and len(encoded) > MAX_BYTES * 2:
                raise ValueError("Oversized cached feed")
            state["data"] = _json(encoded.encode("utf-8")) if encoded is not None else None
            if state["data"] is not None:
                state["data"] = _validate_cache(feed, state["data"])
            for key in ("fetched", "next_attempt", "busy_until"):
                state[key] = _number(state[key], 0, 32_503_680_000)
        except (ValueError, TypeError, RecursionError):
            state.update(data=None, fetched=0, next_attempt=0, busy_until=0,
                         error="Saved feed cache is invalid")
        return state

    def _claim(self, feed: Feed, now: float) -> bool:
        # Reserve the provider deadline BEFORE the request. A process crash or
        # cancellation cannot make the other consumer download immediately.
        with self._database() as database:
            database.execute("BEGIN IMMEDIATE")
            database.execute("INSERT OR IGNORE INTO feeds (key) VALUES (?)", (feed.key,))
            row = database.execute("SELECT next_attempt FROM feeds WHERE key=?", (feed.key,)).fetchone()
            if row[0] > now:
                return False
            database.execute("UPDATE feeds SET next_attempt=?, busy_until=? WHERE key=?",
                             (now + feed.interval, now + 15, feed.key))
        return True

    def _store(self, feed: Feed, now: float, data: list | None, error: str | None,
               retry_after: float = 0) -> None:
        with self._database() as database:
            if error is None:
                database.execute("UPDATE feeds SET data=?, fetched=?, failures=0, error=NULL, "
                                 "busy_until=0, next_attempt=? WHERE key=?",
                                 (json.dumps(data, allow_nan=False), now, now + feed.interval, feed.key))
            else:
                row = database.execute("SELECT failures FROM feeds WHERE key=?", (feed.key,)).fetchone()
                failures = min(20, (row[0] if row else 0) + 1)
                delay = max(feed.interval, min(86400, 120 * 2 ** (failures - 1)), retry_after)
                database.execute("UPDATE feeds SET failures=?, error=?, busy_until=0, next_attempt=? WHERE key=?",
                                 (failures, _text(error, 250), now + delay, feed.key))

    async def _download(self, feed: Feed) -> bytes:
        if self._closed:
            raise asyncio.CancelledError
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10, connect=4, sock_read=5),
                connector=aiohttp.TCPConnector(limit=5), trust_env=False,
                headers={"User-Agent": "ORION-Mark-XXXIII/1.0 (personal world feed reader)",
                         "Accept": "application/json, application/atom+xml, application/rss+xml, text/xml"})
        async with self._session.get(feed.url, allow_redirects=False) as response:
            if response.status != 200:
                delay = _retry_after(response.headers.get("Retry-After"), self._clock())
                if response.status in {301, 302, 303, 307, 308, 403, 404, 410}:
                    delay = max(delay, 6 * 3600)
                raise FeedFailure(f"{feed.source} returned HTTP {response.status}", delay)
            if response.content_length is not None and response.content_length > MAX_BYTES:
                raise FeedFailure("Official feed exceeds the response size limit")
            buffer = bytearray()
            async for chunk in response.content.iter_chunked(32_768):
                buffer.extend(chunk)
                if len(buffer) > MAX_BYTES:
                    raise FeedFailure("Official feed exceeds the response size limit")
            return bytes(buffer)

    async def _refresh(self, feed: Feed) -> None:
        try:
            claimed = await asyncio.to_thread(self._claim, feed, self._clock())
            if not claimed:
                return
            raw = await self._download(feed)
            data = await asyncio.to_thread(parse_feed, feed, raw)
            await asyncio.to_thread(self._store, feed, self._clock(), data, None)
        except asyncio.CancelledError:
            # Reservation remains valid after cancellation. No immediate retry.
            raise
        except FeedFailure as exc:
            try:
                await asyncio.to_thread(self._store, feed, self._clock(), None, str(exc), exc.retry_after)
            except (OSError, sqlite3.Error):
                pass
        except (aiohttp.ClientError, TimeoutError, ValueError, OSError, sqlite3.Error) as exc:
            try:
                await asyncio.to_thread(self._store, feed, self._clock(), None,
                                        f"{type(exc).__name__}: {str(exc)[:150]}")
            except (OSError, sqlite3.Error):
                pass

    async def snapshot(self, layers: tuple[str, ...] = LAYER_NAMES, *, refresh: bool = True) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("World feed service has been closed")
        if not isinstance(layers, (tuple, list)) or not layers or len(layers) > len(LAYER_NAMES) or \
                len(set(layers)) != len(layers) or any(name not in LAYER_NAMES for name in layers):
            raise ValueError("Choose distinct supported world layers")
        if not isinstance(refresh, bool):
            raise TypeError("refresh must be a Boolean")
        feeds = [feed for feed in FEEDS if feed.layer in layers]
        if refresh:
            pending = []
            for feed in feeds:
                task = self._tasks.get(feed.key)
                if task is None or task.done():
                    task = asyncio.create_task(self._refresh(feed))
                    self._tasks[feed.key] = task
                pending.append(asyncio.shield(task))
            await asyncio.gather(*pending)
        states = await asyncio.gather(*(asyncio.to_thread(self._read, feed) for feed in feeds),
                                      return_exceptions=True)
        now = self._clock()
        result: dict[str, Any] = {"generated_at": _iso(now), "layers": {}}
        for name in layers:
            selected = [(feed, state) for feed, state in zip(feeds, states) if feed.layer == name]
            items: list[dict[str, Any]] = []
            errors = []
            fetched = []
            next_attempts = []
            stale = False
            unavailable = False
            loading = False
            for feed, state in selected:
                if isinstance(state, BaseException):
                    errors.append("World feed cache could not be read")
                    unavailable = True
                    continue
                data = state["data"]
                if data is None:
                    unavailable = True
                else:
                    if name == "satellites":
                        predicted, issue = await asyncio.to_thread(predict_satellites, data, now)
                        items.extend(predicted)
                        if issue:
                            errors.append(issue)
                            stale = True
                            unavailable = not predicted
                    else:
                        items.extend(data)
                    fetched.append(state["fetched"])
                    stale |= now - state["fetched"] > feed.interval
                if state.get("error"):
                    errors.append(f"{feed.source}: {state['error']}")
                    stale |= data is not None
                loading |= state.get("busy_until", 0) > now
                next_attempts.append(state["next_attempt"])
            status = "unavailable" if unavailable and not items else "stale" if stale or unavailable else \
                "healthy" if items else "empty"
            if loading and not items and not errors:
                status = "loading"
            result["layers"][name] = {
                "items": items, "fetched_at": _iso(min(fetched)) if fetched else None,
                "source": "; ".join(feed.source for feed, _state in selected),
                "source_url": selected[0][0].url,
                "source_urls": [feed.url for feed, _state in selected],
                "stale": bool(stale or unavailable), "status": status,
                "error": "; ".join(errors) or None,
                "coverage": "; ".join(feed.coverage for feed, _state in selected),
                "next_refresh_at": _iso(min(next_attempts)) if next_attempts else None,
            }
        return result

    async def close(self) -> None:
        self._closed = True
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._tasks.clear()


_service: WorldFeedService | None = None


def get_world_feeds() -> WorldFeedService:
    global _service
    if _service is None or _service._closed:
        _service = WorldFeedService()
    return _service


async def shutdown_world_feeds() -> None:
    global _service
    if _service is not None:
        await _service.close()
        _service = None
