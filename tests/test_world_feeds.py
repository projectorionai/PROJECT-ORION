"""Provider fixtures, persistent rate limits and feed lifecycle regressions."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import ClassVar

import pytest

from orion_core import world_feeds as world


def _feed(key):
    return next(feed for feed in world.FEEDS if feed.key == key)


def _quake(*, longitude=-2, latitude=52, magnitude=3.4):
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature", "id": "fixture-one",
        "geometry": {"type": "Point", "coordinates": [longitude, latitude, 8.2]},
        "properties": {"type": "earthquake", "time": 1790729000000,
                       "mag": magnitude, "title": "M 3.4 — fixture", "place": "Test location",
                       "url": "https://earthquake.usgs.gov/earthquakes/eventpage/fixture-one",
                       "status": "reviewed", "tsunami": 1},
    }]}


def _encoded(body):
    return json.dumps(body).encode()


def _omm():
    return {"OBJECT_NAME": "ISS fixture", "OBJECT_ID": "1998-067A", "EPOCH": "2026-09-30T12:00:00.000000",
            "MEAN_MOTION": 15.5, "ECCENTRICITY": 0.0004, "INCLINATION": 51.64,
            "RA_OF_ASC_NODE": 25, "ARG_OF_PERICENTER": 45, "MEAN_ANOMALY": 80,
            "EPHEMERIS_TYPE": 0, "CLASSIFICATION_TYPE": "U", "NORAD_CAT_ID": 25544,
            "ELEMENT_SET_NO": 999, "REV_AT_EPOCH": 50000, "BSTAR": 0.00015,
            "MEAN_MOTION_DOT": 0.0001, "MEAN_MOTION_DDOT": 0}


def test_official_destinations_and_celestrak_policy():
    assert all(feed.url.startswith("https://") for feed in world.FEEDS)
    assert _feed("stations").interval >= 7200
    assert "GROUP=STATIONS&FORMAT=JSON" in _feed("stations").url
    assert len({feed.key for feed in world.FEEDS}) == len(world.FEEDS)


def test_earthquake_record_is_attributed_without_converting_tsunami_flag_to_warning():
    item = world.parse_feed(_feed("earthquakes"), _encoded(_quake()))[0]
    assert item["longitude"] == -2 and item["latitude"] == 52
    assert item["depth_km"] == 8.2 and item["magnitude"] == 3.4
    assert item["event_at"].endswith("Z") and item["source"] == "USGS"
    assert "tsunami" not in item


@pytest.mark.parametrize("longitude,latitude", [(181, 52), (-2, -91), ("bad", 50), (True, 50), (float("nan"), 50)])
def test_invalid_coordinates_cannot_become_globe_entities(longitude, latitude):
    with pytest.raises(ValueError):
        world.parse_feed(_feed("earthquakes"), _encoded(_quake(longitude=longitude, latitude=latitude)))


@pytest.mark.parametrize("body", [[], {}, {"type": "FeatureCollection", "features": "wrong"},
                                 {"type": "FeatureCollection", "features": [None]}])
def test_invalid_json_shape_is_a_failure_not_a_healthy_empty_feed(body):
    with pytest.raises(ValueError):
        world.parse_feed(_feed("earthquakes"), _encoded(body))


def test_actual_empty_json_feed_is_supported():
    assert world.parse_feed(_feed("earthquakes"), _encoded({"type": "FeatureCollection", "features": []})) == []


def test_atom_bulletin_without_coordinates_stays_a_bulletin():
    raw = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>noaa:1</id>
    <title>Tsunami Information Statement</title><updated>2026-09-30T12:00:00Z</updated>
    <link href="https://www.tsunami.gov/events/example.html"/>
    <summary>No tsunami threat to the listed area.</summary></entry></feed>'''
    item = world.parse_feed(_feed("ntwc"), raw)[0]
    assert item["kind"] == "bulletin" and "latitude" not in item
    assert item["details"].startswith("No tsunami threat")
    assert item["source_url"].endswith("example.html")


def test_weather_rss_keeps_direct_link_and_plain_text():
    raw = b'''<rss><channel><item><title>Yellow warning of rain</title>
    <description>&lt;b&gt;Heavy rain&lt;/b&gt; possible</description>
    <link>https://weather.metoffice.gov.uk/warnings-and-advice/uk-warnings</link>
    <pubDate>Wed, 30 Sep 2026 12:00:00 GMT</pubDate></item></channel></rss>'''
    item = world.parse_feed(_feed("weather"), raw)[0]
    assert item["details"] == "Heavy rain possible"
    assert item["source"] == "Met Office"
    assert item["source_url"].startswith("https://weather.metoffice.gov.uk/")
    assert "latitude" not in item


def test_official_georss_point_is_not_claimed_as_a_warning_polygon():
    raw = b'''<feed xmlns:g="http://www.georss.org/georss"><entry><title>Bulletin</title>
    <g:point>38.1 142.8</g:point></entry></feed>'''
    item = world.parse_feed(_feed("ptwc"), raw)[0]
    assert item["latitude"] == 38.1 and item["longitude"] == 142.8
    assert item["location_kind"] == "feed_event_location"


@pytest.mark.parametrize("raw", [b'<!DOCTYPE rss [<!ENTITY x "secret">]><rss/>',
                                 b'<rss><channel><item></rss>', b'<html/>', b'<?xml version="1.0"?><rss/>'.decode().encode("utf-16")])
def test_xml_faults_are_refused(raw):
    with pytest.raises(ValueError):
        world.parse_feed(_feed("weather"), raw)


def test_response_size_and_item_count_are_bounded():
    with pytest.raises(ValueError, match="size"):
        world.parse_feed(_feed("weather"), b"x" * (world.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="many"):
        world.parse_feed(_feed("earthquakes"), _encoded({"type": "FeatureCollection", "features": [None] * 2001}))
    with pytest.raises(ValueError, match="bounded"):
        world.parse_feed(_feed("stations"), _encoded([_omm()] * 65))


@pytest.mark.parametrize("url", ["https://evil.example/x", "file:///etc/passwd", "javascript:alert(1)",
                                 "https://www.metoffice.gov.uk:8000/x", "https://user@www.metoffice.gov.uk/x"])
def test_links_are_official_https_only(url):
    assert world._link(url, _feed("weather")) == _feed("weather").url


def test_sgp4_positions_move_from_valid_orbital_elements():
    pytest.importorskip("sgp4")
    rows = world.parse_feed(_feed("stations"), _encoded([_omm()]))
    epoch = world._date(_omm()["EPOCH"])
    first, issue = world.predict_satellites(rows, epoch)
    second, second_issue = world.predict_satellites(rows, epoch + 60)
    assert issue is None and second_issue is None
    assert len(first) == len(second) == 1
    assert 350 < first[0]["altitude_km"] < 450
    assert abs(first[0]["latitude"]) < 52
    assert abs(first[0]["longitude"] - second[0]["longitude"]) > 0.1
    assert first[0]["position_kind"] == "sgp4_prediction"
    assert first[0]["predicted_at"] != second[0]["predicted_at"]
    assert first[0]["epoch"] == second[0]["epoch"]


def test_too_old_orbital_elements_are_not_presented_as_current():
    pytest.importorskip("sgp4")
    rows = world.parse_feed(_feed("stations"), _encoded([_omm()]))
    items, issue = world.predict_satellites(rows, world._date(_omm()["EPOCH"]) + 8 * 86400)
    assert not items and "too old" in issue


def test_geodetic_conversion_has_correct_equatorial_altitude_and_rotation():
    lat, lon, altitude = world._geodetic((6778.137, 0, 0), 2451545.0)
    assert lat == pytest.approx(0)
    assert altitude == pytest.approx(400)
    # J2000 GMST = 280.460618 degrees, so ECI x maps to 79.539382 E.
    assert lon == pytest.approx(79.539382, abs=1e-5)


async def test_concurrent_clients_share_the_persistent_deadline(tmp_path, monkeypatch):
    now = 1790769600.0
    calls = []

    async def download(self, feed):
        calls.append(feed.key)
        await asyncio.sleep(0.02)
        return _encoded(_quake())

    monkeypatch.setattr(world.WorldFeedService, "_download", download)
    one = world.WorldFeedService(tmp_path / "world.db", clock=lambda: now)
    two = world.WorldFeedService(tmp_path / "world.db", clock=lambda: now)
    try:
        await asyncio.gather(one.snapshot(("earthquakes",)), two.snapshot(("earthquakes",)))
        assert calls == ["earthquakes"]
        later = await two.snapshot(("earthquakes",))
        assert later["layers"]["earthquakes"]["status"] == "healthy"
        assert len(later["layers"]["earthquakes"]["items"]) == 1
    finally:
        await one.close()
        await two.close()


async def test_satellite_requests_respect_two_hours_even_after_restart(tmp_path, monkeypatch):
    now = [1790769600.0]
    calls = []

    async def download(self, feed):
        calls.append(feed.key)
        return _encoded([_omm()])

    monkeypatch.setattr(world.WorldFeedService, "_download", download)
    service = world.WorldFeedService(tmp_path / "world.db", clock=lambda: now[0])
    await service.snapshot(("satellites",))
    await service.close()
    service = world.WorldFeedService(tmp_path / "world.db", clock=lambda: now[0])
    try:
        now[0] += 7199
        await service.snapshot(("satellites",))
        assert calls == ["stations"]
        now[0] += 2
        await service.snapshot(("satellites",))
        assert calls == ["stations", "stations"]
    finally:
        await service.close()


async def test_failure_keeps_last_good_data_and_backs_off(tmp_path, monkeypatch):
    now = [1790769600.0]
    calls = []

    async def download(self, feed):
        calls.append(feed.key)
        if len(calls) > 1:
            raise world.FeedFailure("HTTP 429", retry_after=3600)
        return _encoded(_quake())

    monkeypatch.setattr(world.WorldFeedService, "_download", download)
    service = world.WorldFeedService(tmp_path / "world.db", clock=lambda: now[0])
    try:
        first = await service.snapshot(("earthquakes",))
        now[0] += 121
        failed = await service.snapshot(("earthquakes",))
        data = failed["layers"]["earthquakes"]
        assert data["items"] == first["layers"]["earthquakes"]["items"]
        assert data["status"] == "stale" and data["stale"]
        assert "429" in data["error"]
        now[0] += 300
        await service.snapshot(("earthquakes",))
        assert len(calls) == 2
    finally:
        await service.close()


async def test_unavailable_is_distinct_from_successful_empty(tmp_path, monkeypatch):
    async def download(self, feed):
        if feed.layer == "weather":
            raise world.FeedFailure("provider unavailable")
        return b'<feed/>'

    monkeypatch.setattr(world.WorldFeedService, "_download", download)
    service = world.WorldFeedService(tmp_path / "world.db")
    try:
        data = (await service.snapshot(("weather", "tsunami")))["layers"]
        assert data["weather"]["status"] == "unavailable"
        assert data["weather"]["error"] and not data["weather"]["items"]
        assert data["tsunami"]["status"] == "empty"
        assert data["tsunami"]["error"] is None and not data["tsunami"]["stale"]
    finally:
        await service.close()


async def test_cancellation_closes_download_and_preserves_deadline(tmp_path, monkeypatch):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def download(self, feed):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(world.WorldFeedService, "_download", download)
    service = world.WorldFeedService(tmp_path / "world.db")
    snapshot = asyncio.create_task(service.snapshot(("satellites",)))
    await entered.wait()
    await service.close()
    assert cancelled.is_set()
    with pytest.raises(asyncio.CancelledError):
        await snapshot
    assert service._read(_feed("stations"))["next_attempt"] > service._clock() + 7100
    with pytest.raises(RuntimeError, match="closed"):
        await service.snapshot()


def test_feed_database_uses_the_shared_write_policy(tmp_path):
    service = world.WorldFeedService(tmp_path / "world.db")
    with service._database() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_database_handles_are_closed_on_success_and_failure(tmp_path, monkeypatch):
    original = world.sqlite3.connect
    handles = []

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        handles.append(connection)
        return connection

    monkeypatch.setattr(world.sqlite3, "connect", connect)
    service = world.WorldFeedService(tmp_path / "world.db")
    service._read(_feed("earthquakes"))
    with pytest.raises(RuntimeError), service._database():
        raise RuntimeError("fixture")
    for handle in handles:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            handle.execute("SELECT 1")


@pytest.mark.parametrize("changed", [{"latitude": 100}, {"source_url": "https://evil.example/x"},
                                     {"longitude": None}, {"event_at": "wrong"}])
async def test_corrupt_cached_records_are_refused(tmp_path, changed):
    service = world.WorldFeedService(tmp_path / "world.db")
    feed = _feed("earthquakes")
    now = service._clock()
    assert service._claim(feed, now)
    rows = world.parse_feed(feed, _encoded(_quake()))
    rows[0].update(changed)
    service._store(feed, now, rows, None)
    try:
        result = (await service.snapshot(("earthquakes",), refresh=False))["layers"]["earthquakes"]
        assert result["status"] == "unavailable" and not result["items"]
        assert "invalid" in result["error"]
    finally:
        await service.close()


async def test_corrupt_cache_is_not_a_healthy_empty(tmp_path):
    service = world.WorldFeedService(tmp_path / "world.db")
    feed = _feed("earthquakes")
    assert service._claim(feed, service._clock())
    with service._database() as database:
        database.execute("UPDATE feeds SET data='[NaN]', fetched=100 WHERE key=?", (feed.key,))
    try:
        data = (await service.snapshot(("earthquakes",), refresh=False))["layers"]["earthquakes"]
        assert data["status"] == "unavailable"
        assert "invalid" in data["error"]
    finally:
        await service.close()


@pytest.mark.parametrize("field,value", [("fetched", None), ("next_attempt", "broken"),
                                         ("busy_until", None), ("next_attempt", -1)])
async def test_corrupt_cached_timestamps_report_unavailable_without_crashing(tmp_path, field, value):
    service = world.WorldFeedService(tmp_path / "world.db")
    feed = _feed("earthquakes")
    now = service._clock()
    assert service._claim(feed, now)
    service._store(feed, now, world.parse_feed(feed, _encoded(_quake())), None)
    with service._database() as database:
        # Field names are fixed by this fixture, not supplied by callers.
        database.execute(f"UPDATE feeds SET {field}=? WHERE key=?", (value, feed.key))
    try:
        data = (await service.snapshot(("earthquakes",), refresh=False))["layers"]["earthquakes"]
        assert data["status"] == "unavailable" and "invalid" in data["error"]
        assert data["next_refresh_at"] is None
    finally:
        await service.close()


@pytest.mark.parametrize("value,expected", [("120", 120), ("-3", 0), ("NaN", 0),
                                            ("864000", 864000),
                                            ("Wed, 30 Sep 2026 12:00:00 GMT", 120),
                                            ("not a date", 0)])
def test_retry_after_values(value, expected):
    now = world._date("2026-09-30T11:58:00Z")
    assert world._retry_after(value, now) == expected


async def test_http_redirect_is_refused_and_blocked_provider_gets_long_backoff(tmp_path):
    requested = []

    class Response:
        status = 302
        headers: ClassVar = {"Location": "http://127.0.0.1/private"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Session:
        def get(self, url, **kwargs):
            requested.append((url, kwargs))
            return Response()

        async def close(self):
            pass

    service = world.WorldFeedService(tmp_path / "world.db")
    service._session = Session()
    try:
        data = (await service.snapshot(("weather",)))["layers"]["weather"]
        assert data["status"] == "unavailable" and "HTTP 302" in data["error"]
        assert requested == [(_feed("weather").url, {"allow_redirects": False})]
        assert service._read(_feed("weather"))["next_attempt"] > service._clock() + 21500
    finally:
        await service.close()


async def test_status_does_not_start_network(tmp_path, monkeypatch):
    async def forbidden(*_args):
        raise AssertionError("Status must not download")

    monkeypatch.setattr(world.WorldFeedService, "_download", forbidden)
    service = world.WorldFeedService(tmp_path / "world.db")
    try:
        data = await service.snapshot(refresh=False)
        assert set(data["layers"]) == set(world.LAYER_NAMES)
        assert all(layer["status"] == "unavailable" for layer in data["layers"].values())
        assert service._session is None
    finally:
        await service.close()
