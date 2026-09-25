"""
Tests for pinpoint UK-postcode geocoding (the 'M1 1AE flies to the sea' fix).

A postcode-shaped query is routed to postcodes.io for its EXACT centroid before
Nominatim's coarse free-text search or the Open-Meteo fallback (which cannot
resolve postcodes at all) can return a wrong point.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.geo import (
    GeoIntelligenceEngine,
    looks_like_uk_postcode,
    normalise_uk_postcode,
)


class _Signal:
    def emit(self, *a):
        pass

    def connect(self, *a, **k):
        pass


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


def test_postcode_shape_detection():
    assert looks_like_uk_postcode("M1 1AE")
    assert looks_like_uk_postcode("m11ae")          # no space, lowercase
    assert looks_like_uk_postcode("SW1A 1AA")
    assert looks_like_uk_postcode("EC1A1BB")
    assert not looks_like_uk_postcode("Manchester")
    assert not looks_like_uk_postcode("90210")        # US ZIP, not UK
    assert not looks_like_uk_postcode("")


def test_postcode_normalisation():
    assert normalise_uk_postcode("m11ae") == "M1 1AE"
    assert normalise_uk_postcode("SW1A   1AA") == "SW1A 1AA"
    assert normalise_uk_postcode("ec1a1bb") == "EC1A 1BB"


def _engine(tmp_path, monkeypatch, fake_get_json):
    monkeypatch.setattr("orion_core.geo.GEO_CACHE_PATH", tmp_path / "geo_cache.db")
    engine = GeoIntelligenceEngine(_StubBus())
    engine._get_json = fake_get_json      # type: ignore[assignment]
    return engine


def test_postcode_uses_postcodes_io_exact_centroid(tmp_path, monkeypatch):
    calls: list[str] = []

    async def fake_get_json(url, timeout=12.0):
        calls.append(url)
        # postcodes.io exact-lookup shape for M1 1AE (Manchester, Devon).
        assert "api.postcodes.io/postcodes/" in url
        return {
            "status": 200,
            "result": {
                "postcode": "M1 1AE",
                "latitude": 50.5470,
                "longitude": -3.4977,
                "admin_district": "Teignbridge",
                "admin_ward": "Manchester",
                "region": "South West",
                "country": "England",
            },
        }

    engine = _engine(tmp_path, monkeypatch, fake_get_json)
    places = asyncio.run(engine.geocode("M1 1AE"))
    assert len(places) == 1
    p = places[0]
    assert p.kind == "postcode"
    assert abs(p.lat - 50.5470) < 1e-6 and abs(p.lon - -3.4977) < 1e-6
    assert p.country == "England"
    # Nominatim was never consulted — postcodes.io answered exactly.
    assert all("nominatim" not in u for u in calls)


def test_unknown_postcode_falls_through_to_nominatim(tmp_path, monkeypatch):
    urls: list[str] = []

    async def fake_get_json(url, timeout=12.0):
        urls.append(url)
        if "api.postcodes.io" in url:
            return {"status": 404, "error": "Postcode not found"}
        # Nominatim, now constrained to GB for a postcode-shaped query.
        assert "countrycodes=gb" in url
        return [{
            "lat": "51.5", "lon": "-0.12", "name": "Somewhere",
            "address": {"country": "United Kingdom", "postcode": "ZZ99 9ZZ"},
            "addresstype": "postcode",
        }]

    engine = _engine(tmp_path, monkeypatch, fake_get_json)
    places = asyncio.run(engine.geocode("ZZ99 9ZZ"))
    assert len(places) == 1                     # recovered via Nominatim
    assert any("api.postcodes.io" in u for u in urls)
    assert any("nominatim" in u for u in urls)
