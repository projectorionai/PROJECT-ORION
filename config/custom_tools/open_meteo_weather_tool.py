"""Weather, without an account.

Open-Meteo is free for non-commercial use and needs no key at all, which makes
it the right default: a weather tool that first requires signing up for an API
key is a weather tool nobody turns on. Everything here is a plain HTTPS GET
against a documented public endpoint.

Places are resolved through Open-Meteo's own geocoding, except UK postcodes,
which it handles poorly — those go to postcodes.io, which ORION already uses
elsewhere for the same reason.
"""

from __future__ import annotations

from orion_core.data import ToolResult

import json
import urllib.error
import urllib.parse
import urllib.request

FORECAST = "https://api.open-meteo.com/v1/forecast"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
POSTCODES = "https://api.postcodes.io/postcodes/"
TIMEOUT = 15

#: WMO weather codes, in the words a person would use. The API returns a
#: number; "code 61" is not an answer to "what's the weather".
CONDITIONS = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}


def _get(url: str, params: dict) -> dict:
    query = urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "ORION/1.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _looks_like_uk_postcode(text: str) -> bool:
    stripped = text.replace(" ", "").upper()
    return (5 <= len(stripped) <= 7
            and stripped[0].isalpha() and stripped[-1].isalpha()
            and any(ch.isdigit() for ch in stripped))


def _locate(place: str) -> tuple[float, float, str] | None:
    """Turn a place name into coordinates, or None if it cannot be found."""
    place = place.strip()
    if not place:
        return None
    if _looks_like_uk_postcode(place):
        try:
            payload = _get(POSTCODES + urllib.parse.quote(place), {})
            result = payload.get("result") or {}
            if result.get("latitude") is not None:
                name = result.get("admin_district") or result.get("parish") or place
                return (float(result["latitude"]), float(result["longitude"]),
                        f"{place.upper()} ({name})")
        except Exception:
            pass        # fall through to the general geocoder
    try:
        payload = _get(GEOCODE, {"name": place, "count": 1, "language": "en",
                                 "format": "json"})
    except Exception:
        return None
    results = payload.get("results") or []
    if not results:
        return None
    first = results[0]
    label = ", ".join(part for part in
                      (first.get("name"), first.get("admin1"),
                       first.get("country")) if part)
    return float(first["latitude"]), float(first["longitude"]), label


def _describe(code: int) -> str:
    return CONDITIONS.get(int(code), f"weather code {code}")


def _run_impl(**kwargs) -> ToolResult:
    """Execute the plugin. Returns a short line for ORION to speak."""
    place = str(kwargs.get("place") or kwargs.get("location") or "").strip()
    days = max(0, min(7, int(kwargs.get("days") or 0)))
    if not place:
        return ToolResult("Which place? Give me a town, a city or a UK postcode — "
                "no account or API key is needed.", ok=False)

    located = _locate(place)
    if located is None:
        return ToolResult(f"I could not find anywhere called {place!r}.", ok=False)
    latitude, longitude, label = located

    params = {
        "latitude": latitude, "longitude": longitude,
        "current": ["temperature_2m", "apparent_temperature", "weather_code",
                    "wind_speed_10m", "relative_humidity_2m"],
        "timezone": "auto",
    }
    if days:
        params["daily"] = ["weather_code", "temperature_2m_max",
                           "temperature_2m_min", "precipitation_probability_max"]
        params["forecast_days"] = days
    try:
        data = _get(FORECAST, params)
    except urllib.error.URLError as exc:
        return ToolResult(f"The weather service could not be reached ({exc.reason}).", ok=False)
    except Exception as exc:
        return ToolResult(f"The weather lookup failed ({exc}).", ok=False)

    current = data.get("current") or {}
    if not current or any(current.get(key) is None for key in (
            "temperature_2m", "apparent_temperature", "weather_code",
            "wind_speed_10m", "relative_humidity_2m")):
        return ToolResult("The weather service returned incomplete current conditions.", ok=False)
    units = data.get("current_units") or {}
    degrees = units.get("temperature_2m", "°C")
    lines = [
        f"{label}: {_describe(current.get('weather_code', 0))}, "
        f"{current.get('temperature_2m')}{degrees} "
        f"(feels like {current.get('apparent_temperature')}{degrees}), "
        f"wind {current.get('wind_speed_10m')} "
        f"{units.get('wind_speed_10m', 'km/h')}, "
        f"humidity {current.get('relative_humidity_2m')}%."
    ]

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    for index, date in enumerate(dates):
        lines.append(
            f"  {date}: {_describe(daily['weather_code'][index])}, "
            f"{daily['temperature_2m_min'][index]}–"
            f"{daily['temperature_2m_max'][index]}{degrees}, "
            f"{daily['precipitation_probability_max'][index]}% chance of rain.")
    return ToolResult("\n".join(lines), ok=True)


def get_tool_schema():
    """Loader contract; inspection does not contact the external service."""
    return {'name': 'open_meteo_weather',
     'description': 'Current weather and a short forecast for any town, city or UK postcode. Free '
                    'and needs no account or API key.',
     'parameters': {'type': 'object',
                    'properties': {'place': {'type': 'string',
                                             'description': 'Town, city or UK postcode, e.g. '
                                                            "'Birmingham' or 'B1 1AA'."},
                                   'days': {'type': 'integer',
                                            'description': 'How many days of forecast to add, '
                                                           '0-7. Omit for current conditions '
                                                           'only.'}},
                    'required': ['place']}}


def run(**kwargs) -> ToolResult:
    """Report failures explicitly, including invalid input and malformed replies."""
    try:
        return _run_impl(**kwargs)
    except Exception as exc:
        return ToolResult(f"Plugin action failed ({type(exc).__name__}); completion was not verified.", ok=False)
