"""Location provenance for the Globe; coordinate precision is not accuracy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def valid_coordinates(lat: Any, lon: Any) -> bool:
    """Reject absent, boolean, non-finite and out-of-range source positions."""
    if isinstance(lat, bool) or isinstance(lon, bool):
        return False
    try:
        return math.isfinite(float(lat)) and math.isfinite(float(lon)) and (
            -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180
        )
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class GlobeLocation:
    lat: float
    lon: float
    label: str
    source: str
    kind: str

    @property
    def detail(self) -> str:
        if self.kind == "coordinate":
            return "Supplied WGS84 coordinate"
        if self.source == "postcodes.io":
            return "Postcode representative point · postcodes.io"
        if self.source == "Open-Meteo":
            return "Settlement centre · Open-Meteo"
        kind = self.kind.replace("_", " ") or "mapped place"
        return f"{kind.capitalize()} reference point · {self.source}"

    @property
    def camera_range(self) -> float:
        # A closer view is useful for an explicitly supplied position, road or
        # postcode. The chosen view height never asserts imagery accuracy.
        return 3000.0 if self.kind in {
            "coordinate", "postcode", "house", "building", "road", "residential",
        } else 16000.0

    @classmethod
    def from_place(cls, place: Any, query: str) -> GlobeLocation | None:
        if not valid_coordinates(getattr(place, "lat", None), getattr(place, "lon", None)):
            return None
        label = getattr(place, "display", "") or ", ".join(
            str(x) for x in (getattr(place, "name", ""), getattr(place, "country", "")) if x
        ) or query
        # Old cache entries have no provenance. Do not label them as an
        # authoritative postcode lookup merely because their kind is postcode.
        source = getattr(place, "source", "") or "Saved geocoder result"
        return cls(float(place.lat), float(place.lon), str(label)[:512],
                   str(source)[:80], str(getattr(place, "kind", ""))[:80])
