"""Reject guessed origins and distinguish address precision from accuracy."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from orion_core.geo import GeoIntelligenceEngine, Place
from orion_core.globe_locations import GlobeLocation, valid_coordinates


@pytest.mark.parametrize("lat,lon", [(None, 0), (0, None), (True, 0), (0, False),
                                    ("bad", "0"), (float("nan"), 0), (0, float("inf")),
                                    (90.1, 0), (0, -180.1)])
def test_source_positions_cannot_create_plausible_invalid_markers(lat, lon):
    assert not valid_coordinates(lat, lon)


def test_position_precision_is_preserved_and_provenance_round_trips():
    place = Place("Target", 51.507400123, -0.127800987, kind="building",
                  display="Target, London, England", source="OpenStreetMap Nominatim")
    stored = GeoIntelligenceEngine._place_from_dict(place.to_dict())
    target = GlobeLocation.from_place(stored, "Target")
    assert target.lat == place.lat and target.lon == place.lon
    assert target.label == place.display
    assert target.detail == "Building reference point · OpenStreetMap Nominatim"
    assert target.camera_range == 3000


def test_postcode_reference_is_not_claimed_as_an_exact_house():
    target = GlobeLocation(51.5, -0.1, "SW1A 1AA", "postcodes.io", "postcode")
    assert target.detail == "Postcode representative point · postcodes.io"


def test_legacy_postcode_cache_does_not_invent_provider_provenance():
    place = GeoIntelligenceEngine._place_from_dict({"name": "SW1A 1AA", "lat": 51.5,
                                                  "lon": -0.1, "kind": "postcode"})
    target = GlobeLocation.from_place(place, "SW1A 1AA")
    assert "Saved geocoder result" in target.detail
    assert "postcodes.io" not in target.detail


@pytest.mark.parametrize("record", [{}, {"name": "Missing"}, {"lat": 0},
                                      {"lat": 0, "lon": True}, {"lat": "NaN", "lon": 0},
                                      {"lat": 0, "lon": 0, "admin": "bad"}, None, []])
def test_malformed_cache_never_defaults_to_the_gulf_of_guinea(record):
    assert GeoIntelligenceEngine._place_from_dict(record) is None


@pytest.mark.parametrize("record", [{"lat": "0"}, {"lat": True, "lon": "0"},
                                      {"lat": "nan", "lon": "0"}, {"lat": 91, "lon": 0},
                                      {"lat": 0, "lon": 0, "address": "bad"},
                                      {"lat": 0, "lon": 0, "extratags": [1]}])
def test_bad_provider_record_is_refused_without_crashing(record):
    engine = GeoIntelligenceEngine.__new__(GeoIntelligenceEngine)
    assert engine._place_from_nominatim(record) is None


def test_partial_invalid_cache_is_filtered_and_bad_cache_is_refetched():
    async def scenario():
        engine = GeoIntelligenceEngine.__new__(GeoIntelligenceEngine)
        engine.telemetry = None
        stored = []
        calls = []
        engine._cache_put = lambda *args: stored.append(args)
        async def fetch(url):
            calls.append(url)
            return [{"name": "Valid", "lat": "51.5", "lon": "-0.1"}]
        engine._get_json = fetch
        engine._cache_get = lambda _key: [{"name": "Bad"}]
        found = await engine.geocode("Valid")
        assert len(calls) == 1 and len(found) == 1
        assert found[0].source == "OpenStreetMap Nominatim"
        engine._cache_get = lambda _key: [{"name": "Bad"}, found[0].to_dict()]
        assert (await engine.geocode("Valid"))[0].lat == 51.5
        assert len(calls) == 1
    asyncio.run(scenario())


def test_reverse_cache_does_not_collapse_adjacent_positions():
    async def scenario():
        engine = GeoIntelligenceEngine.__new__(GeoIntelligenceEngine)
        keys = []
        engine._cache_get = lambda key: keys.append(key)
        engine._cache_put = lambda *_args: None
        async def fetch(_url):
            return {"name": "Reference", "lat": "51.5", "lon": "-0.1"}
        engine._get_json = fetch
        await engine.reverse(51.500001, -0.100001)
        await engine.reverse(51.500009, -0.100009)
        assert keys == ["reverse::51.500001::-0.100001", "reverse::51.500009::-0.100009"]
        assert await engine.reverse(True, 0) is None
        assert len(keys) == 2
    asyncio.run(scenario())


@pytest.mark.parametrize("result", [None, [], "bad", {"latitude": True, "longitude": 0},
                                      {"latitude": "inf", "longitude": 0},
                                      {"latitude": 0, "longitude": 181}])
def test_postcode_provider_refuses_malformed_or_invalid_positions(result):
    async def scenario():
        engine = GeoIntelligenceEngine.__new__(GeoIntelligenceEngine)
        async def fetch(_url):
            return {"status": 200, "result": result}
        engine._get_json = fetch
        assert await engine._geocode_uk_postcode("SW1A 1AA") is None
    asyncio.run(scenario())


def test_label_prefers_full_address_for_same_named_towns():
    place = SimpleNamespace(lat=51.5, lon=-0.1, name="Same name", country="UK",
                            display="Same name, County, UK", kind="town", source="Test")
    assert GlobeLocation.from_place(place, "Same name").label == place.display


def test_voice_coordinates_bypass_geocoder_and_ambiguous_names_do_not_claim_a_match():
    async def scenario():
        engine = GeoIntelligenceEngine.__new__(GeoIntelligenceEngine)
        requests = []
        engine.bus = SimpleNamespace(globe_request=SimpleNamespace(emit=requests.append))
        async def forbidden(*_args, **_kwargs):
            raise AssertionError("Voice coordinates reached the geocoder")
        engine.geocode = forbidden
        assert (await engine.locate("51.507400123,-0.127800987")).ok
        assert not (await engine.locate("91,0")).ok
        assert requests == ["51.507400123,-0.127800987"]
        async def multiple(query, limit):
            assert limit == 6
            return [Place("Same name", 51, -1, display="Same name, First county"),
                    Place("Same name", 52, -2, display="Same name, Second county")]
        engine.geocode = multiple
        result = await engine.locate("Same name")
        assert "Choose the full address" in result.text
        assert "First county" in result.text and "Second county" in result.text
        assert not result.text.startswith("Located")
        assert requests[-1] == "Same name"
    asyncio.run(scenario())
