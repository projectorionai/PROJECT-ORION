"""Exact navigation must never turn an invalid pair into a place search."""

from __future__ import annotations

import pytest

from orion_core.globe_coordinates import MAX_COORDINATE_TEXT, parse_coordinates


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("51.507400123, -0.127800987", (51.507400123, -0.127800987)),
        ("  ( 51.5074 ; −0.1278 )  ", (51.5074, -0.1278)),
        ("51.5074 N, 0.1278 W", (51.5074, -0.1278)),
        ("51.5074°n 0.1278°w", (51.5074, -0.1278)),
        ("-33.8688 S; 151.2093 E", (-33.8688, 151.2093)),
        ("lat:51.5074, lon:-0.1278", (51.5074, -0.1278)),
        ("latitude = 51.5074 longitude = 0.1278 W", (51.5074, -0.1278)),
        ("lng=-0.1278; LAT=51.5074", (51.5074, -0.1278)),
        ("(longitude: -0.1278, latitude: 51.5074)", (51.5074, -0.1278)),
        ("geo:51.507400123,-0.127800987", (51.507400123, -0.127800987)),
        ("GEO:0,0", (0.0, 0.0)),
        ("+90,+180", (90.0, 180.0)),
        ("-90,-180", (-90.0, -180.0)),
        ("90S 180W", (-90.0, -180.0)),
        (".5,-.125", (0.5, -0.125)),
        ("5.15074e1, -1.278e-1", (51.5074, -0.1278)),
    ],
)
def test_exact_pairs_preserve_decimal_precision(query, expected):
    assert parse_coordinates(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "", "   ", "Birmingham", "Manchester, UK", "10 Downing Street, London",
        "Area 51, Nevada", "Paris, 16th arrondissement", "SW1A 1AA", "90210",
        "Northern Ireland", "Latitude Festival", "London (West)",
    ],
)
def test_ordinary_places_are_left_for_geocoding(query):
    assert parse_coordinates(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "90.000000001,0", "-90.000000001,0", "0,180.000000001", "0,-180.000000001",
        "90.000000000000000001,0", "0,-180.000000000000000001",
        "51.5,", ", -0.1", "51.5, -0.1, 1", "51.5;;-0.1", "51.5 -0.1", "51.5",
        ",", "(,)", ";;",
        "(51.5,-0.1", "51.5,-0.1)", "((51.5,-0.1))", "51.5abc,-0.1", "51.5,London",
        "nan,0", "0,inf", "-infinity,0", "1e1000,0", "geo:nan,0", "geo:51.5,-0.1,2",
        "geo:51.5,-0.1;crs=wgs84", "geo:51.5,-0.1?z=3", "geo:51.5,-0.1#point",
        "geo:", "lat:51.5", "latitude:51.5, lat:0.1", "lon:51.5, lng:0.1",
        "lat:51.5, longitude:invalid", "51.5E,0.1N", "51.5 N 0.1 N",
        "-51.5N,0.1W", "+51.5S,0.1W", "51.5N,-0.1E", "51.5N,+0.1W",
        "-0N,0E", "0N,+0W",
        "N51,W0", "51x,0y", "51°30N,0°7W", "51°", "lat 51.5 lon -0.1",
    ],
)
def test_coordinate_looking_errors_cannot_fall_back_to_place_search(query):
    with pytest.raises(ValueError):
        parse_coordinates(query)


@pytest.mark.parametrize("value", [True, False, None, 51.5, 90, (51.5, -0.1), [51.5, -0.1]])
def test_non_text_input_is_refused(value):
    with pytest.raises(ValueError):
        parse_coordinates(value)


def test_queries_are_bounded_before_parsing_or_place_search():
    with pytest.raises(ValueError, match="characters"):
        parse_coordinates("a" * (MAX_COORDINATE_TEXT + 1))
    assert parse_coordinates("a" * MAX_COORDINATE_TEXT) is None


def test_longitude_is_never_wrapped_to_another_location():
    with pytest.raises(ValueError, match="Longitude"):
        parse_coordinates("0,540")


def test_signed_zero_hemisphere_is_preserved():
    import math

    latitude, longitude = parse_coordinates("0S,0W")
    assert math.copysign(1.0, latitude) == -1.0
    assert math.copysign(1.0, longitude) == -1.0
