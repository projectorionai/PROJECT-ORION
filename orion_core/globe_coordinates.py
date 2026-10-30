"""Parse exact WGS84 locations without sending malformed coordinates to search."""

from __future__ import annotations

import math
import re
from decimal import Decimal

MAX_COORDINATE_TEXT = 256

_NUMBER = r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|nan|inf(?:inity)?)"
_VALUE = rf"(?P<number>{_NUMBER})\s*°?\s*(?P<direction>[NSEW]?)"
_VALUE_PATTERN = re.compile(rf"^\s*{_VALUE}\s*$", re.IGNORECASE)
_LABEL = r"(?:latitude|lat|longitude|long|lon|lng)"
_LABELLED = re.compile(
    rf"^\s*(?P<first_label>{_LABEL})\s*[:=]\s*(?P<first>.*?)"
    rf"\s*(?:[,;]\s*|\s+)(?P<second_label>{_LABEL})\s*[:=]\s*(?P<second>.*?)\s*$",
    re.IGNORECASE,
)
_SUFFIX_PAIR = re.compile(
    rf"^\s*(?P<first>{_NUMBER}\s*°?\s*[NS])\s+"
    rf"(?P<second>{_NUMBER}\s*°?\s*[EW])\s*$",
    re.IGNORECASE,
)
_LABEL_START = re.compile(
    rf"^\s*{_LABEL}\b\s*(?:[:=]|\s+(?:[+-]?[0-9.]|nan\b|inf\b))", re.IGNORECASE
)
_NUMERIC_START = re.compile(r"^[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)")
_NUMBER_START = re.compile(r"^[+-]?(?:[0-9]|\.[0-9])")
_CARDINAL_START = re.compile(r"^[NSEW]\s*[+-]?(?:[0-9]|\.[0-9])", re.IGNORECASE)
_DEGREE_START = re.compile(r"^[+-]?[0-9]+\s*°")
_COORDINATE_CHARACTERS = re.compile(r"^[0-9a-zA-Z+\-.,;:=()°\s]*$")


def _looks_like_coordinates(text: str) -> bool:
    """Distinguish coordinate input from names such as '10 Downing Street'."""
    unwrapped = text.strip("() ")
    if unwrapped.lower().startswith("geo:") or _LABEL_START.match(unwrapped):
        return True
    pieces = [piece.strip() for piece in re.split(r"[,;]", unwrapped)]
    if len(pieces) > 1:
        return not any(pieces) or any(
            bool(_VALUE_PATTERN.fullmatch(piece))
            or bool(_NUMERIC_START.match(piece))
            or bool(_CARDINAL_START.match(piece))
            or bool(_DEGREE_START.match(piece))
            for piece in pieces
        ) or all(_NUMBER_START.match(piece) and not re.search(r"\s", piece) for piece in pieces)
    if _NUMERIC_START.match(unwrapped) or _CARDINAL_START.match(unwrapped) or _DEGREE_START.match(unwrapped):
        return True
    # Clear cardinal input remains a coordinate request even if the separators
    # or hemisphere ordering are wrong. Ordinary place names still use search.
    if _COORDINATE_CHARACTERS.fullmatch(unwrapped):
        return bool(
            re.search(r"[0-9]\s*°?\s*[NSEW]\b", unwrapped, re.IGNORECASE)
            or re.fullmatch(rf"\s*{_NUMBER}\s+{_NUMBER}\s*", unwrapped, re.IGNORECASE)
        )
    return False


def _value(text: str, *, latitude: bool) -> float:
    match = _VALUE_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("Use decimal latitude and longitude, with optional N/S and E/W suffixes.")
    number = match.group("number")
    direction = match.group("direction").upper()
    value = float(number)
    if not math.isfinite(value):
        raise ValueError("Coordinates must be finite numbers.")
    positive, negative = ("N", "S") if latitude else ("E", "W")
    if direction:
        if direction not in (positive, negative):
            raise ValueError("Latitude uses N/S; longitude uses E/W.")
        if (number.startswith("-") and direction == positive) or (
            number.startswith("+") and direction == negative
        ):
            raise ValueError("The coordinate sign contradicts its hemisphere suffix.")
        value = abs(value) * (-1.0 if direction == negative else 1.0)
    bound = 90.0 if latitude else 180.0
    # Check the original decimal before float conversion can round a value just
    # outside the pole or antimeridian back onto the boundary.
    if Decimal(number).copy_abs() > bound:
        name = "Latitude" if latitude else "Longitude"
        raise ValueError(f"{name} must be between {-bound:g} and {bound:g} degrees.")
    return value


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Return exact latitude/longitude, or ``None`` for an ordinary place name.

    Decimal pairs use a comma or semicolon. Parentheses, hemisphere suffixes,
    labelled latitude/longitude (in either order), and ``geo:lat,lon`` are
    accepted. Geo URI altitude and parameters are refused rather than silently
    discarded. Malformed coordinate input raises ``ValueError`` so callers can
    show an error without falling back to a nearby city or postcode centroid.
    No rounding, longitude wrapping, geocoding or network access takes place.
    """
    if not isinstance(text, str):
        # Callers handle all rejected location queries through one error path.
        raise ValueError("A coordinate or place query must be text.")  # noqa: TRY004
    if len(text) > MAX_COORDINATE_TEXT:
        raise ValueError(f"Keep location queries within {MAX_COORDINATE_TEXT} characters.")
    query = text.strip().replace("−", "-")
    if not query:
        return None
    if not _looks_like_coordinates(query):
        return None
    if query.startswith("(") or query.endswith(")"):
        if not (query.startswith("(") and query.endswith(")")):
            raise ValueError("Coordinate parentheses must be balanced.")
        query = query[1:-1].strip()
    if query.lower().startswith("geo:"):
        query = query[4:].strip()
        # A third ordinate or URI parameter cannot be mistaken for a pair.
        if query.count(",") != 1 or ";" in query or "?" in query or "#" in query:
            raise ValueError("Use geo:latitude,longitude without altitude or URI parameters.")
    labelled = _LABELLED.fullmatch(query)
    if labelled is not None:
        labels = (labelled.group("first_label").lower(), labelled.group("second_label").lower())
        first_is_lat = labels[0] in ("lat", "latitude")
        second_is_lat = labels[1] in ("lat", "latitude")
        if first_is_lat == second_is_lat:
            raise ValueError("Provide one latitude label and one longitude label.")
        first, second = labelled.group("first"), labelled.group("second")
        lat_text, lon_text = (first, second) if first_is_lat else (second, first)
    else:
        pieces = re.split(r"[,;]", query)
        if len(pieces) == 2:
            lat_text, lon_text = pieces
        else:
            suffix_pair = _SUFFIX_PAIR.fullmatch(query)
            if suffix_pair is None:
                raise ValueError("Provide latitude, longitude separated by a comma or semicolon.")
            lat_text, lon_text = suffix_pair.group("first"), suffix_pair.group("second")
    return _value(lat_text, latitude=True), _value(lon_text, latitude=False)
