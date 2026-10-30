"""
Weather — the forecast a person actually asks for, with the obvious advice.

Routing already sent "what's the weather in Leeds" and "do I need a coat
today" to the geo tool, but geo could only locate, list nearby towns and
describe places: there was no weather to give. The greeting fetched a
one-line reading for its own use and nothing else could ask.

This answers the questions people ask about weather, from Open-Meteo (free,
keyless, worldwide, hourly):

    now        what it is like outside right now, and what it feels like
    today      the high and low, when rain is likely, the wind
    tonight    how cold it gets, rain overnight
    tomorrow   / a weekday / the weekend / the week ahead
    advice     the common-sense part: a coat or not, an umbrella, sunscreen,
               a very windy day, ice — said plainly, only when it applies

UK conventions: degrees Celsius, wind in miles an hour, times in words.

Pure parsing and wording (parse, answer) are separate from the one network
call (fetch), so everything a person hears can be tested without the network.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .utils import spoken_time, weather_code_label

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_S = 600.0                 # one place's forecast is reused for ten minutes
_CACHE: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}

WAKING = range(7, 23)           # the hours whose weather people plan around
RAIN_LIKELY = 50                # percent: "take an umbrella"
RAIN_POSSIBLE = 30              # percent: "a shower is possible"
COAT_BELOW = 11.0               # feels-like °C: wear a coat
JACKET_BELOW = 16.0             # feels-like °C: a light jacket
WINDY_GUST = 40.0               # mph
SUNSCREEN_UV = 6.0

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_ADVICE_WORDS = re.compile(
    r"\b(coat|jacket|umbrella|brolly|raincoat|waterproof|wear|layers?|jumper|sunscreen|"
    r"sun ?cream|shorts|washing|wash|dry|cycle|bike|walk)\b", re.I)


def is_wet(code: int) -> bool:
    """Open-Meteo codes for drizzle, rain, snow, showers and storms."""
    return 51 <= code <= 67 or 71 <= code <= 77 or 80 <= code <= 86 or code >= 95


def request_url(lat: float, lon: float, days: int = 7) -> str:
    return (f"{FORECAST_URL}?latitude={lat:.4f}&longitude={lon:.4f}"
            "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,"
            "wind_gusts_10m,relative_humidity_2m,is_day,precipitation"
            "&hourly=temperature_2m,apparent_temperature,precipitation_probability,"
            "weather_code,wind_gusts_10m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
            "apparent_temperature_min,precipitation_probability_max,precipitation_sum,"
            "wind_gusts_10m_max,uv_index_max,sunrise,sunset"
            f"&forecast_days={max(1, min(16, days))}&timezone=auto&wind_speed_unit=mph")


# ── the parsed forecast ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Hour:
    at: datetime
    temp: float | None
    feels: float | None
    rain_chance: float | None
    code: int
    gust: float | None


@dataclass(frozen=True)
class Day:
    on: date
    code: int
    high: float | None
    low: float | None
    feels_low: float | None
    rain_chance: float | None
    rain_mm: float | None
    gust: float | None
    uv: float | None
    sunrise: datetime | None
    sunset: datetime | None


@dataclass
class Forecast:
    place: str
    now: datetime                      # local time at the place
    temp: float | None
    feels: float | None
    code: int
    wind: float | None
    gust: float | None
    humidity: float | None
    hours: list[Hour] = field(default_factory=list)
    days: list[Day] = field(default_factory=list)
    offset_s: int | None = None        # the place's UTC offset, for its real clock

    def clock(self) -> datetime:
        """The real local time at the place now. `now` is the start of the
        forecast's 15-minute reading, up to a quarter of an hour behind."""
        if self.offset_s is None:
            return self.now
        from datetime import timedelta, timezone
        return (datetime.now(timezone.utc) + timedelta(seconds=self.offset_s)).replace(tzinfo=None)

    def day(self, on: date) -> Day | None:
        return next((d for d in self.days if d.on == on), None)

    def hours_on(self, on: date, hours: range = WAKING, after: datetime | None = None) -> list[Hour]:
        return [h for h in self.hours if h.at.date() == on and h.at.hour in hours
                and (after is None or h.at > after)]


def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _when(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _column(block: dict[str, Any], key: str, index: int) -> Any:
    values = block.get(key) or []
    return values[index] if index < len(values) else None


def parse(data: dict[str, Any], place: str = "") -> Forecast:
    """An Open-Meteo forecast response as a Forecast. Raises ValueError when
    the response has no current reading."""
    current = data.get("current") or {}
    now = _when(current.get("time"))
    if now is None:
        raise ValueError("the forecast has no current reading")
    hourly, daily = data.get("hourly") or {}, data.get("daily") or {}
    hours = []
    for i, stamp in enumerate(hourly.get("time") or []):
        at = _when(stamp)
        if at is None:
            continue
        hours.append(Hour(at, _num(_column(hourly, "temperature_2m", i)),
                          _num(_column(hourly, "apparent_temperature", i)),
                          _num(_column(hourly, "precipitation_probability", i)),
                          int(_num(_column(hourly, "weather_code", i)) or 0),
                          _num(_column(hourly, "wind_gusts_10m", i))))
    days = []
    for i, stamp in enumerate(daily.get("time") or []):
        try:
            on = date.fromisoformat(str(stamp))
        except ValueError:
            continue
        days.append(Day(on, int(_num(_column(daily, "weather_code", i)) or 0),
                        _num(_column(daily, "temperature_2m_max", i)),
                        _num(_column(daily, "temperature_2m_min", i)),
                        _num(_column(daily, "apparent_temperature_min", i)),
                        _num(_column(daily, "precipitation_probability_max", i)),
                        _num(_column(daily, "precipitation_sum", i)),
                        _num(_column(daily, "wind_gusts_10m_max", i)),
                        _num(_column(daily, "uv_index_max", i)),
                        _when(_column(daily, "sunrise", i)),
                        _when(_column(daily, "sunset", i))))
    return Forecast(place=place, now=now, temp=_num(current.get("temperature_2m")),
                    feels=_num(current.get("apparent_temperature")),
                    code=int(_num(current.get("weather_code")) or 0),
                    wind=_num(current.get("wind_speed_10m")),
                    gust=_num(current.get("wind_gusts_10m")),
                    humidity=_num(current.get("relative_humidity_2m")),
                    hours=hours, days=days,
                    offset_s=int(data["utc_offset_seconds"]) if "utc_offset_seconds" in data else None)


async def fetch(lat: float, lon: float, place: str = "", *, session: Any = None) -> Forecast:
    """The forecast for a point (cached ten minutes per ~5 km cell)."""
    key = (round(float(lat) / 0.05) * 0.05, round(float(lon) / 0.05) * 0.05)
    hit = _CACHE.get(key)
    if hit is not None and time.monotonic() - hit[0] < CACHE_S:
        return parse(hit[1], place)
    url = request_url(lat, lon)
    if session is None:
        from aiohttp import ClientSession, ClientTimeout
        async with ClientSession(timeout=ClientTimeout(total=8.0, connect=4.0)) as own:
            data = await _get(own, url)
    else:
        data = await _get(session, url)
    _CACHE[key] = (time.monotonic(), data)
    if len(_CACHE) > 64:
        _CACHE.pop(next(iter(_CACHE)))
    return parse(data, place)


async def _get(session: Any, url: str) -> dict[str, Any]:
    async with session.get(url) as response:
        if response.status != 200:
            raise RuntimeError(f"the weather service answered HTTP {response.status}")
        return await response.json(content_type=None)


# ── which day was asked about ────────────────────────────────────────────────

def which(question: str, forecast: Forecast) -> tuple[str, list[date]]:
    """(the spoken name of the period, the dates it covers) for a question:
    today / tonight / tomorrow / a weekday / the weekend / the week."""
    text = str(question or "").lower()
    today = forecast.now.date()
    known = [d.on for d in forecast.days]
    if re.search(r"\b(this|the) week\b|\bweek ahead\b|\bnext few days\b|\bnext 7 days\b", text):
        return "the week ahead", known[:7]
    if "weekend" in text:
        # The weekend in progress or the next one: Saturday and Sunday, or
        # just Sunday when asked on a Sunday - never Sunday plus next Saturday.
        first = next((d for d in known if d.weekday() >= 5 and d >= today), None)
        ahead = [d for d in known if first is not None and 0 <= (d - first).days <= 1
                 and d.weekday() >= 5]
        return "this weekend", ahead
    if "tomorrow" in text:
        return "tomorrow", [d for d in known if (d - today).days == 1]
    for index, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", text):
            match = [d for d in known if d.weekday() == index and d > today][:1]
            if today.weekday() == index and "next" not in text:
                match = [today]
            return name.title(), match
    if re.search(r"\btonight\b|\bthis evening\b|\bovernight\b", text):
        return "tonight", [today]
    return "today", [today]


# ── advice: the common-sense part ────────────────────────────────────────────

def _spoken_hour(hour: int) -> str:
    return spoken_time(hour, 0).replace(" o'clock", "")


def advice(forecast: Forecast, on: date, *, tonight: bool = False) -> list[str]:
    """Plain advice for one day, only what applies: coat, umbrella, wind,
    sunscreen, ice. For today it looks at the hours still to come."""
    today = on == forecast.now.date()
    span = range(17, 24) if tonight else WAKING
    hours = forecast.hours_on(on, span, after=forecast.now if today else None)
    if today and not hours and not tonight:
        hours = forecast.hours_on(on, range(forecast.now.hour, 24), after=forecast.now)
    day = forecast.day(on)
    out: list[str] = []

    chances = [(h.rain_chance or 0.0, h) for h in hours]
    likely = next((h for c, h in chances if c >= RAIN_LIKELY), None)
    peak = max((c for c, _h in chances), default=day.rain_chance if day and day.rain_chance else 0.0)
    raining_now = today and is_wet(forecast.code) and not tonight
    if raining_now:
        out.append("Take an umbrella — it's raining now")
    elif likely is not None:
        when = "" if likely.at.hour <= (forecast.now.hour + 1 if today else 7) else \
            f" from around {_spoken_hour(likely.at.hour)}"
        out.append(f"Take an umbrella — rain is likely{when} ({peak:.0f} percent)")
    elif peak >= RAIN_POSSIBLE:
        out.append(f"A shower is possible ({peak:.0f} percent), so a light waterproof wouldn't hurt")

    feels = [h.feels for h in hours if h.feels is not None]
    if today and forecast.feels is not None and not tonight:
        feels.append(forecast.feels)
    coldest = min(feels) if feels else (day.feels_low if day else None)
    if coldest is not None:
        if coldest < COAT_BELOW:
            out.append(f"Wear a coat — it'll feel like {coldest:.0f} degrees")
        elif coldest < JACKET_BELOW:
            out.append(f"A light jacket should do — it'll feel like {coldest:.0f} degrees at the coolest")
        elif not out:
            out.append("No coat needed")

    gusts = [h.gust for h in hours if h.gust is not None]
    strongest = max(gusts) if gusts else (day.gust if day else None)
    if strongest is not None and strongest >= WINDY_GUST:
        out.append(f"It'll be very windy, with gusts up to {strongest:.0f} miles an hour")
    if day is not None and day.uv is not None and day.uv >= SUNSCREEN_UV and not tonight:
        out.append(f"Wear sunscreen — the UV index reaches {day.uv:.0f}")
    low = day.low if day is not None else None
    if low is not None and low <= 0.5:
        out.append(f"Watch for ice — it drops to {_degrees(low)}")
    return out


_RAIN_WORDS = re.compile(r"\b(rain(ing|y)?|umbrella|brolly|wet|showers?|drizzle|dry|pour(ing)?)\b", re.I)


def rain_verdict(forecast: Forecast, on: date, *, tonight: bool = False,
                 umbrella: bool = False) -> str:
    """A straight answer to "will it rain?" / "do I need an umbrella?" for
    the hours that matter — including the no, which advice() leaves unsaid."""
    today = on == forecast.now.date()
    hours = forecast.hours_on(on, range(17, 24) if tonight else WAKING,
                              after=forecast.now if today else None)
    if today and not tonight and is_wet(forecast.code):
        return "Yes — it's raining now" + (", so take an umbrella" if umbrella else "")
    peak = max((h.rain_chance or 0.0 for h in hours), default=0.0)
    likely = next((h for h in hours if (h.rain_chance or 0.0) >= RAIN_LIKELY), None)
    if likely is not None:
        return (f"Yes — rain is likely from around {_spoken_hour(likely.at.hour)} "
                f"({peak:.0f} percent)" + (", so take an umbrella" if umbrella else ""))
    if peak >= RAIN_POSSIBLE:
        return (f"Possibly — there's a {peak:.0f} percent chance of a shower"
                + (", so a small umbrella wouldn't hurt" if umbrella else ""))
    span = "tonight" if tonight else "during the day" if not today else "for the rest of the day"
    if any(is_wet(h.code) for h in hours):
        return (f"Probably not — at most a {peak:.0f} percent chance of a light shower {span}"
                + (", so you can leave the umbrella" if umbrella else ""))
    return (f"No — it should stay dry {span}"
            + (", so no umbrella needed" if umbrella else "")
            + (f" (at most a {peak:.0f} percent chance)" if peak >= 10 else ""))


def _degrees(value: float | None) -> str:
    if value is None:
        return "an unknown temperature"
    rounded = round(value)
    return f"minus {abs(rounded)} degrees" if rounded < 0 else f"{rounded} degrees"


# ── the spoken answer ────────────────────────────────────────────────────────

def _rain_window(forecast: Forecast, on: date) -> str:
    hours = forecast.hours_on(on, WAKING, after=forecast.now if on == forecast.now.date() else None)
    likely = [h for h in hours if (h.rain_chance or 0) >= RAIN_LIKELY]
    if not likely:
        return ""
    start, end = likely[0].at.hour, likely[-1].at.hour
    if end - start <= 1:
        return f"rain likely around {_spoken_hour(start)}"
    return f"rain likely from around {_spoken_hour(start)} to {_spoken_hour(end + 1)}"


def _day_line(forecast: Forecast, day: Day, name: str, rain_clause: bool = True) -> str:
    label = weather_code_label(day.code)
    waking = forecast.hours_on(day.on, WAKING)
    if is_wet(day.code) and waking and not any(
            (h.rain_chance or 0.0) >= RAIN_POSSIBLE for h in waking):
        # A day's code is its worst hour: rain at 3 a.m. - or one model run's
        # drizzle the probabilities give 20 percent - made a dry day "slight
        # rain", contradicting "no umbrella needed".
        label = (f"mostly dry, with a small chance of {label}"
                 if any(is_wet(h.code) for h in waking)
                 else f"{label} overnight, dry during the day")
    bits = [label]
    if day.low is not None and day.high is not None:
        bits.append(f"{_degrees(day.low)} to {_degrees(day.high)}".replace(" degrees to", " to"))
    rain = _rain_window(forecast, day.on) if rain_clause else ""
    if rain:
        bits.append(rain)
    elif rain_clause and day.rain_chance is not None and day.rain_chance >= RAIN_POSSIBLE:
        bits.append(f"a {day.rain_chance:.0f} percent chance of rain")
    if day.gust is not None and day.gust >= WINDY_GUST:
        bits.append(f"gusts up to {day.gust:.0f} miles an hour")
    return f"{name}: " + ", ".join(bits)


_SUN_WORDS = re.compile(r"\b(sun ?rise|sun ?set|get(?:s|ting)? (?:dark|light)|dark tonight|"
                        r"daylight|dawn|dusk|sun (?:come up|go down|goes down|comes up))\b", re.I)


def sun_times(forecast: Forecast, question: str = "") -> str:
    """Sunrise and sunset, as asked: 'when does it get dark' is sunset today
    (or tomorrow's once it has gone), 'sunrise' the next one."""
    where = f" in {forecast.place}" if forecast.place else ""
    name, dates = which(question, forecast)
    on = dates[0] if dates else forecast.now.date()
    day = forecast.day(on)
    if day is None or day.sunrise is None or day.sunset is None:
        return f"I don't have the sunrise and sunset{where} for {name}."
    text = str(question or "").lower()
    wants_rise = bool(re.search(r"sun ?rise|get(?:s|ting)? light|dawn|come up|comes up", text))
    wants_set = bool(re.search(r"sun ?set|dark|dusk|go down|goes down", text)) or not wants_rise
    now = forecast.clock()
    today = on == now.date()
    parts = []
    if wants_set:
        when = day.sunset
        if today and when <= now:
            nxt = forecast.day(on.fromordinal(on.toordinal() + 1))
            parts.append(f"the sun set at {when.strftime('%H:%M')}"
                         + (f"; tomorrow it sets at {nxt.sunset.strftime('%H:%M')}"
                            if nxt and nxt.sunset else ""))
        else:
            left = ""
            if today:
                mins = int((when - now).total_seconds() // 60)
                left = f", in {mins // 60} hour{'s' if mins // 60 != 1 else ''} {mins % 60} minutes" \
                    if mins >= 60 else f", in {mins} minutes"
            parts.append(f"sunset is at {when.strftime('%H:%M')}{left}")
    if wants_rise:
        rise_day = day
        if today and day.sunrise <= now:
            rise_day = forecast.day(on.fromordinal(on.toordinal() + 1)) or day
        label = "tomorrow" if rise_day.on != now.date() and today else ""
        parts.append(f"sunrise{' ' + label if label else ''} is at {rise_day.sunrise.strftime('%H:%M')}")
    span = (day.sunset - day.sunrise).total_seconds() / 3600 if day.sunset and day.sunrise else None
    if name == "today":
        lead = f"In {forecast.place}, " if forecast.place else ""
    else:
        lead = f"{name.capitalize()}{where}: "
    text_out = lead + "; ".join(parts)
    text_out = text_out[0].upper() + text_out[1:]
    if span is not None and wants_set and wants_rise:
        text_out += f" — {span:.1f} hours of daylight"
    return text_out + "."


#: "When will it stop raining / clear up / get better / be dry again?"
_IMPROVE_WORDS = re.compile(
    r"\b(?:stop(?:s|ping)?\s+(?:raining|being\s+(?:bad|wet|rubbish|awful|grim|miserable)|"
    r"(?:the\s+)?rain)|clear(?:s|ing)?\s+up|dr(?:y|ies)\s+(?:up|out)|"
    r"(?:get|gets|getting|be)\s+(?:better|nicer|drier|dry|sunny|warmer)|improv(?:e|es|ing)|"
    r"(?:next|first)\s+(?:dry|sunny|nice|good)\s+(?:day|spell)|(?:dry|sunny|nice)\s+day|"
    r"brighten(?:s|ing)?\s+up|when\s+(?:will|does|is)\s+(?:the\s+)?(?:rain|sun))\b", re.I)


def _dry_hour(forecast: Forecast, hour: Any) -> bool:
    return not is_wet(hour.code) and (hour.rain_chance or 0.0) < RAIN_POSSIBLE


def improvement(forecast: Forecast) -> str:
    """When the bad weather lets up, in specifics (owner, 26 Oct 2026: asked
    when it would stop being bad, he heard "the drizzle may clear within a
    couple of hours ... Sunday currently appearing the most likely day").
    The hour the rain stops today, the first dry day with its temperatures,
    and the best day of the week - all from the forecast, nothing guessed."""
    where = f" in {forecast.place}" if forecast.place else ""
    today = forecast.now.date()
    parts: list[str] = []
    rest_of_today = forecast.hours_on(today, range(forecast.now.hour, 24), after=forecast.now)
    wet_now = is_wet(forecast.code) or any(
        not _dry_hour(forecast, h) for h in rest_of_today[:2])
    if wet_now:
        clear = next((h for i, h in enumerate(rest_of_today)
                      if all(_dry_hour(forecast, g) for g in rest_of_today[i:i + 3])), None)
        if clear is not None:
            parts.append(f"The rain{where} should stop around {_spoken_hour(clear.at.hour)} today, "
                         "and stay dry for a few hours after")
        else:
            parts.append(f"It stays wet{where} for the rest of today")
    else:
        parts.append(f"It's dry{where} right now")

    def dry_day(day: Day) -> bool:
        waking = forecast.hours_on(day.on, WAKING)
        chance = max((h.rain_chance or 0.0 for h in waking),
                     default=day.rain_chance if day.rain_chance is not None else 0.0)
        return chance < RAIN_POSSIBLE

    def label(on: date) -> str:
        delta = (on - today).days
        return "today" if delta == 0 else "tomorrow" if delta == 1 else on.strftime("%A")

    ahead = [d for d in forecast.days if d.on > today][:7]
    first_dry = next((d for d in ahead if dry_day(d)), None)
    if first_dry is not None:
        temps = (f", {_degrees(first_dry.low)} to {_degrees(first_dry.high)}".replace(" degrees to", " to")
                 if first_dry.low is not None and first_dry.high is not None else "")
        parts.append(f"The first properly dry day is {label(first_dry.on)} "
                     f"({weather_code_label(first_dry.code)}{temps})")
    elif ahead:
        least = min(ahead, key=lambda d: d.rain_chance if d.rain_chance is not None else 100.0)
        chance = f"{least.rain_chance:.0f} percent" if least.rain_chance is not None else "the lowest"
        parts.append(f"No fully dry day in the next {len(ahead)} days; the best bet is "
                     f"{label(least.on)}, at {chance} chance of rain")
    warm = [d for d in ahead if d.high is not None]
    if warm:
        best = max(warm, key=lambda d: (dry_day(d), d.high))
        if first_dry is None or best.on != first_dry.on:
            parts.append(f"the nicest-looking day is {label(best.on)}, up to {_degrees(best.high)}")
    wet_days = [label(d.on) for d in ahead[:4] if not dry_day(d)]
    if wet_days and first_dry is not None and wet_days[0] != label(first_dry.on):
        parts.append("before that, expect rain " + (
            wet_days[0] if len(wet_days) == 1 else ", ".join(wet_days[:-1]) + " and " + wet_days[-1]))
    text = ". ".join(p[0].upper() + p[1:] for p in parts)
    return text + "."


def answer(forecast: Forecast, question: str = "") -> str:
    """The spoken answer to a weather question, advice first when the
    question was really about what to wear or carry."""
    if _SUN_WORDS.search(question or ""):
        return sun_times(forecast, question)
    if _IMPROVE_WORDS.search(question or ""):
        return improvement(forecast)
    name, dates = which(question, forecast)
    where = f" in {forecast.place}" if forecast.place else ""
    wants_advice = bool(_ADVICE_WORDS.search(question or ""))
    if not dates:
        return (f"The forecast{where} doesn't reach {name} yet — it covers the next "
                f"{len(forecast.days)} days.")

    if name in ("this weekend", "the week ahead"):
        lines = []
        for on in dates:
            day = forecast.day(on)
            if day is not None:
                label = "Today" if on == forecast.now.date() else \
                    "Tomorrow" if (on - forecast.now.date()).days == 1 else on.strftime("%A")
                lines.append(_day_line(forecast, day, label))
        head = f"{name[0].upper()}{name[1:]}{where} —"
        tips = advice(forecast, dates[0]) if wants_advice and dates else []
        return head + " " + "; ".join(lines) + "." + (f" {'. '.join(tips)}." if tips else "")

    on = dates[0]
    day = forecast.day(on)
    tonight = name == "tonight"
    tips = advice(forecast, on, tonight=tonight)
    parts: list[str] = []
    asked_rain = bool(_RAIN_WORDS.search(question or ""))
    if asked_rain:
        # The question was "will it rain / do I need an umbrella": answer it
        # first, yes or no, and drop the rain tip it would repeat.
        umbrella = bool(re.search(r"\b(umbrella|brolly)\b", question or "", re.I))
        tips = [t for t in tips if not t.startswith(("Take an umbrella", "A shower"))]
        tips.insert(0, rain_verdict(forecast, on, tonight=tonight, umbrella=umbrella))
        wants_advice = True
    if name == "today" and forecast.temp is not None:
        now_line = (f"Right now{where} it's {_degrees(forecast.temp)} and "
                    f"{weather_code_label(forecast.code)}")
        if forecast.feels is not None and abs(forecast.feels - forecast.temp) >= 2:
            now_line += f", feeling like {_degrees(forecast.feels)}"
        parts.append(now_line + ".")
        if day is not None:
            rest = []
            if day.high is not None and forecast.now.hour < 15 and day.high - forecast.temp >= 1.5:
                rest.append(f"rising to {_degrees(day.high)}")
            rain = _rain_window(forecast, on)
            if rain:
                rest.append(rain)
            elif not asked_rain and (not tips or not tips[0].startswith(("Take", "A shower"))):
                rest.append("dry for the rest of the day")
            if rest:
                parts.append("Later: " + ", ".join(rest) + ".")
    elif tonight and day is not None:
        evening = forecast.hours_on(on, range(18, 24), after=forecast.now)
        lows = [h.temp for h in evening if h.temp is not None]
        coldest = min(lows) if lows else day.low
        wet = any((h.rain_chance or 0) >= RAIN_LIKELY for h in evening)
        parts.append(f"Tonight{where}: down to {_degrees(coldest)}"
                     + (", with rain likely." if wet else ", staying dry."))
    elif day is not None:
        # A yes/no rain verdict already leads: the line need not repeat it.
        parts.append(_day_line(forecast, day, f"{name.capitalize() if name == 'tomorrow' else name}{where}",
                               rain_clause=not asked_rain) + ".")
    if any("rain likely" in part for part in parts):
        # The forecast line already says when; the tip needn't say it twice.
        tips = ["Take an umbrella" if t.startswith("Take an umbrella — rain is likely") else t
                for t in tips]
    if tips:
        advice_text = ". ".join(tips) + "."
        parts = [advice_text] + parts if wants_advice else parts + [advice_text]
    return " ".join(parts) or f"I couldn't read the forecast{where}."


def glance(forecast: Forecast) -> tuple[str, str]:
    """(headline, detail) WRITTEN for a panel rather than spoken: digits and
    degree signs, the conditions now, then today's advice and tomorrow."""
    where = f"{forecast.place} · " if forecast.place else ""
    head = f"{where}{_short(forecast.temp)} {weather_code_label(forecast.code)}"
    if forecast.feels is not None and forecast.temp is not None and abs(forecast.feels - forecast.temp) >= 2:
        head += f", feels {_short(forecast.feels)}"
    today = forecast.day(forecast.now.date())
    if today is not None and today.high is not None and today.low is not None:
        head += f"  ·  {_short(today.low)} to {_short(today.high)}"
    lines = []
    tips = advice(forecast, forecast.now.date())
    rain = _rain_window(forecast, forecast.now.date())
    if rain:
        lines.append(rain[0].upper() + rain[1:])
    lines += [t.split(" — ")[0] for t in tips if not (rain and t.startswith("Take an umbrella"))][:2]
    tomorrow = next((d for d in forecast.days if (d.on - forecast.now.date()).days == 1), None)
    if tomorrow is not None and tomorrow.low is not None and tomorrow.high is not None:
        lines.append(f"Tomorrow {_short(tomorrow.low)} to {_short(tomorrow.high)}, "
                     f"{weather_code_label(tomorrow.code)}"
                     + (f", {tomorrow.rain_chance:.0f}% rain" if (tomorrow.rain_chance or 0) >= RAIN_POSSIBLE
                        else ""))
    return head, "  ·  ".join(lines)


def _short(value: float | None) -> str:
    return "—" if value is None else f"{round(value)}°"


__all__ = ["Day", "Forecast", "Hour", "advice", "answer", "fetch", "glance", "is_wet", "parse",
           "rain_verdict", "request_url", "which"]
