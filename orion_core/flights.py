"""
Finding flights, by reading the page a person would read.

ORION opens a flight-search page in its existing browser, reads the visible
text, and summarises offers. Page layout, sign-in requirements and prices can
change; a search result is not a guaranteed fare or a completed booking.

Three parts, deliberately separated
-----------------------------------
``resolve_airport``, ``parse_date`` and ``search_url`` are pure: strings in,
strings out, no network. They are where nearly all the mistakes live, so they
are where nearly all the tests are. ``read_offers`` turns page text into rows.
``FlightFinder`` is the only part that touches a browser or a model.

Four things this refuses to do
------------------------------
**It will not guess a date.** An unparseable date returns ``None`` and ORION
asks, rather than quietly searching today and presenting the answer with the
same confidence as a correct one. A wrong date that looks right is worse than
an admitted failure — the user books it.

**It will not carry a stale search.** Google Flights' ``tfs`` parameter is a
base64 blob encoding a complete itinerary, and it *overrides* the readable
query beside it. Pasting a fixed one into a URL builder means every search
silently inherits whichever route was hard-coded. The query here carries the
route in readable text and nothing else.

**It will not send a city where a code belongs.** "Birmingham" is an airport in
Alabama as well as the United Kingdom, and a search that sends
the bare word gets whichever the ranker prefers. Cities resolve to IATA codes,
and a city with several airports resolves to its *metropolitan* code — LON, not
LHR — so the search covers Gatwick and Stansted too, which is what a person
asking about London actually meant.

**It will not compare prices by stripping the punctuation.** "£1,234" and
"£234" become 1234 and 234 that way, which is right by luck; "£1,234.56" and
"£999" become 123456 and 999, which is not. Money is parsed as money.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from urllib.parse import quote_plus

#: Longest page text handed to a model. Google Flights' DOM text runs long and
#: the offers are near the top; the rest is footer, legal and cookie prose.
MAX_PAGE_CHARS = 14_000

#: Most offers reported. Beyond about five, a spoken list stops being listenable
#: and the user should be looking at the page instead.
MAX_OFFERS = 5

#: Seconds allowed for the results to render before the text is read. Google
#: Flights streams prices in after the shell paints, so reading immediately
#: reliably returns a page with no prices on it.
RENDER_WAIT_S = 6.0


# ── airports ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Airport:
    """A resolved place. ``metro`` means the code covers several airports."""

    code: str
    name: str
    country: str = ""
    metro: bool = False

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"


#: Metropolitan codes: one search across every airport serving the city. This
#: is what someone means by "a flight to London" — not Heathrow specifically.
_METRO: dict[str, tuple[str, str, str]] = {
    "london":        ("LON", "London (all airports)", "United Kingdom"),
    "new york":      ("NYC", "New York (all airports)", "United States"),
    "paris":         ("PAR", "Paris (all airports)", "France"),
    "tokyo":         ("TYO", "Tokyo (all airports)", "Japan"),
    "milan":         ("MIL", "Milan (all airports)", "Italy"),
    "rome":          ("ROM", "Rome (all airports)", "Italy"),
    "moscow":        ("MOW", "Moscow (all airports)", "Russia"),
    "chicago":       ("CHI", "Chicago (all airports)", "United States"),
    "washington":    ("WAS", "Washington DC (all airports)", "United States"),
    "buenos aires":  ("BUE", "Buenos Aires (all airports)", "Argentina"),
    "sao paulo":     ("SAO", "São Paulo (all airports)", "Brazil"),
    "osaka":         ("OSA", "Osaka (all airports)", "Japan"),
    "seoul":         ("SEL", "Seoul (all airports)", "South Korea"),
    "stockholm":     ("STO", "Stockholm (all airports)", "Sweden"),
    "toronto":       ("YTO", "Toronto (all airports)", "Canada"),
    "montreal":      ("YMQ", "Montréal (all airports)", "Canada"),
    "beijing":       ("BJS", "Beijing (all airports)", "China"),
    "shanghai":      ("SHA", "Shanghai (all airports)", "China"),
    "jakarta":       ("JKT", "Jakarta (all airports)", "Indonesia"),
    "berlin":        ("BER", "Berlin Brandenburg", "Germany"),
}

#: Single-airport cities, and the named airports of the metropolitan ones for
#: when someone asks for Heathrow rather than London.
_AIRPORTS: dict[str, tuple[str, str, str]] = {
    # United Kingdom and Ireland
    "heathrow":         ("LHR", "London Heathrow", "United Kingdom"),
    "gatwick":          ("LGW", "London Gatwick", "United Kingdom"),
    "stansted":         ("STN", "London Stansted", "United Kingdom"),
    "luton":            ("LTN", "London Luton", "United Kingdom"),
    "london city":      ("LCY", "London City", "United Kingdom"),
    "birmingham":       ("BHX", "Birmingham", "United Kingdom"),
    "manchester":       ("MAN", "Manchester", "United Kingdom"),
    "edinburgh":        ("EDI", "Edinburgh", "United Kingdom"),
    "glasgow":          ("GLA", "Glasgow", "United Kingdom"),
    "bristol":          ("BRS", "Bristol", "United Kingdom"),
    "newcastle":        ("NCL", "Newcastle", "United Kingdom"),
    "liverpool":        ("LPL", "Liverpool John Lennon", "United Kingdom"),
    "leeds":            ("LBA", "Leeds Bradford", "United Kingdom"),
    "belfast":          ("BFS", "Belfast International", "United Kingdom"),
    "cardiff":          ("CWL", "Cardiff", "United Kingdom"),
    "aberdeen":         ("ABZ", "Aberdeen", "United Kingdom"),
    "southampton":      ("SOU", "Southampton", "United Kingdom"),
    "east midlands":    ("EMA", "East Midlands", "United Kingdom"),
    "nottingham":       ("EMA", "East Midlands", "United Kingdom"),
    "dublin":           ("DUB", "Dublin", "Ireland"),
    "cork":             ("ORK", "Cork", "Ireland"),
    # Europe
    "cdg":              ("CDG", "Paris Charles de Gaulle", "France"),
    "charles de gaulle": ("CDG", "Paris Charles de Gaulle", "France"),
    "orly":             ("ORY", "Paris Orly", "France"),
    "beauvais":         ("BVA", "Paris Beauvais", "France"),
    "amsterdam":        ("AMS", "Amsterdam Schiphol", "Netherlands"),
    "frankfurt":        ("FRA", "Frankfurt", "Germany"),
    "munich":           ("MUC", "Munich", "Germany"),
    "hamburg":          ("HAM", "Hamburg", "Germany"),
    "dusseldorf":       ("DUS", "Düsseldorf", "Germany"),
    "cologne":          ("CGN", "Cologne Bonn", "Germany"),
    "madrid":           ("MAD", "Madrid Barajas", "Spain"),
    "barcelona":        ("BCN", "Barcelona El Prat", "Spain"),
    "malaga":           ("AGP", "Málaga", "Spain"),
    "alicante":         ("ALC", "Alicante", "Spain"),
    "palma":            ("PMI", "Palma de Mallorca", "Spain"),
    "ibiza":            ("IBZ", "Ibiza", "Spain"),
    "seville":          ("SVQ", "Seville", "Spain"),
    "valencia":         ("VLC", "Valencia", "Spain"),
    "tenerife":         ("TFS", "Tenerife South", "Spain"),
    "lanzarote":        ("ACE", "Lanzarote", "Spain"),
    "gran canaria":     ("LPA", "Gran Canaria", "Spain"),
    "lisbon":           ("LIS", "Lisbon", "Portugal"),
    "porto":            ("OPO", "Porto", "Portugal"),
    "faro":             ("FAO", "Faro", "Portugal"),
    "madeira":          ("FNC", "Madeira", "Portugal"),
    "fiumicino":        ("FCO", "Rome Fiumicino", "Italy"),
    "ciampino":         ("CIA", "Rome Ciampino", "Italy"),
    "malpensa":         ("MXP", "Milan Malpensa", "Italy"),
    "linate":           ("LIN", "Milan Linate", "Italy"),
    "bergamo":          ("BGY", "Milan Bergamo", "Italy"),
    "venice":           ("VCE", "Venice Marco Polo", "Italy"),
    "naples":           ("NAP", "Naples", "Italy"),
    "florence":         ("FLR", "Florence", "Italy"),
    "pisa":             ("PSA", "Pisa", "Italy"),
    "bologna":          ("BLQ", "Bologna", "Italy"),
    "turin":            ("TRN", "Turin", "Italy"),
    "catania":          ("CTA", "Catania", "Italy"),
    "zurich":           ("ZRH", "Zurich", "Switzerland"),
    "geneva":           ("GVA", "Geneva", "Switzerland"),
    "vienna":           ("VIE", "Vienna", "Austria"),
    "brussels":         ("BRU", "Brussels", "Belgium"),
    "copenhagen":       ("CPH", "Copenhagen", "Denmark"),
    "oslo":             ("OSL", "Oslo Gardermoen", "Norway"),
    "helsinki":         ("HEL", "Helsinki Vantaa", "Finland"),
    "athens":           ("ATH", "Athens", "Greece"),
    "thessaloniki":     ("SKG", "Thessaloniki", "Greece"),
    "santorini":        ("JTR", "Santorini", "Greece"),
    "crete":            ("HER", "Heraklion, Crete", "Greece"),
    "rhodes":           ("RHO", "Rhodes", "Greece"),
    "istanbul":         ("IST", "Istanbul", "Turkey"),
    "antalya":          ("AYT", "Antalya", "Turkey"),
    "prague":           ("PRG", "Prague", "Czechia"),
    "warsaw":           ("WAW", "Warsaw Chopin", "Poland"),
    "krakow":           ("KRK", "Kraków", "Poland"),
    "budapest":         ("BUD", "Budapest", "Hungary"),
    "bucharest":        ("OTP", "Bucharest Otopeni", "Romania"),
    "sofia":            ("SOF", "Sofia", "Bulgaria"),
    "zagreb":           ("ZAG", "Zagreb", "Croatia"),
    "split":            ("SPU", "Split", "Croatia"),
    "dubrovnik":        ("DBV", "Dubrovnik", "Croatia"),
    "nice":             ("NCE", "Nice Côte d'Azur", "France"),
    "lyon":             ("LYS", "Lyon", "France"),
    "marseille":        ("MRS", "Marseille", "France"),
    "toulouse":         ("TLS", "Toulouse", "France"),
    "bordeaux":         ("BOD", "Bordeaux", "France"),
    "reykjavik":        ("KEF", "Reykjavík Keflavík", "Iceland"),
    "malta":            ("MLA", "Malta", "Malta"),
    "cyprus":           ("LCA", "Larnaca", "Cyprus"),
    "larnaca":          ("LCA", "Larnaca", "Cyprus"),
    # North America
    "los angeles":      ("LAX", "Los Angeles", "United States"),
    "san francisco":    ("SFO", "San Francisco", "United States"),
    "miami":            ("MIA", "Miami", "United States"),
    "boston":           ("BOS", "Boston Logan", "United States"),
    "seattle":          ("SEA", "Seattle Tacoma", "United States"),
    "atlanta":          ("ATL", "Atlanta", "United States"),
    "dallas":           ("DFW", "Dallas Fort Worth", "United States"),
    "denver":           ("DEN", "Denver", "United States"),
    "las vegas":        ("LAS", "Las Vegas", "United States"),
    "orlando":          ("MCO", "Orlando", "United States"),
    "houston":          ("IAH", "Houston Intercontinental", "United States"),
    "philadelphia":     ("PHL", "Philadelphia", "United States"),
    "phoenix":          ("PHX", "Phoenix Sky Harbor", "United States"),
    "san diego":        ("SAN", "San Diego", "United States"),
    "austin":           ("AUS", "Austin", "United States"),
    "nashville":        ("BNA", "Nashville", "United States"),
    "detroit":          ("DTW", "Detroit", "United States"),
    "minneapolis":      ("MSP", "Minneapolis St Paul", "United States"),
    "honolulu":         ("HNL", "Honolulu", "United States"),
    "jfk":              ("JFK", "New York JFK", "United States"),
    "newark":           ("EWR", "Newark", "United States"),
    "laguardia":        ("LGA", "New York LaGuardia", "United States"),
    "o'hare":           ("ORD", "Chicago O'Hare", "United States"),
    "midway":           ("MDW", "Chicago Midway", "United States"),
    "dulles":           ("IAD", "Washington Dulles", "United States"),
    "reagan":           ("DCA", "Washington Reagan", "United States"),
    "pearson":          ("YYZ", "Toronto Pearson", "Canada"),
    "vancouver":        ("YVR", "Vancouver", "Canada"),
    "calgary":          ("YYC", "Calgary", "Canada"),
    "mexico city":      ("MEX", "Mexico City", "Mexico"),
    "cancun":           ("CUN", "Cancún", "Mexico"),
    # Middle East and Africa
    "dubai":            ("DXB", "Dubai", "United Arab Emirates"),
    "abu dhabi":        ("AUH", "Abu Dhabi", "United Arab Emirates"),
    "doha":             ("DOH", "Doha Hamad", "Qatar"),
    "tel aviv":         ("TLV", "Tel Aviv Ben Gurion", "Israel"),
    "riyadh":           ("RUH", "Riyadh", "Saudi Arabia"),
    "jeddah":           ("JED", "Jeddah", "Saudi Arabia"),
    "cairo":            ("CAI", "Cairo", "Egypt"),
    "marrakesh":        ("RAK", "Marrakesh", "Morocco"),
    "casablanca":       ("CMN", "Casablanca", "Morocco"),
    "johannesburg":     ("JNB", "Johannesburg", "South Africa"),
    "cape town":        ("CPT", "Cape Town", "South Africa"),
    "nairobi":          ("NBO", "Nairobi", "Kenya"),
    "lagos":            ("LOS", "Lagos", "Nigeria"),
    "accra":            ("ACC", "Accra", "Ghana"),
    "addis ababa":      ("ADD", "Addis Ababa", "Ethiopia"),
    "mauritius":        ("MRU", "Mauritius", "Mauritius"),
    # Asia and Oceania
    "haneda":           ("HND", "Tokyo Haneda", "Japan"),
    "narita":           ("NRT", "Tokyo Narita", "Japan"),
    "incheon":          ("ICN", "Seoul Incheon", "South Korea"),
    "hong kong":        ("HKG", "Hong Kong", "Hong Kong"),
    "singapore":        ("SIN", "Singapore Changi", "Singapore"),
    "bangkok":          ("BKK", "Bangkok Suvarnabhumi", "Thailand"),
    "phuket":           ("HKT", "Phuket", "Thailand"),
    "kuala lumpur":     ("KUL", "Kuala Lumpur", "Malaysia"),
    "manila":           ("MNL", "Manila", "Philippines"),
    "hanoi":            ("HAN", "Hanoi", "Vietnam"),
    "ho chi minh city": ("SGN", "Ho Chi Minh City", "Vietnam"),
    "bali":             ("DPS", "Denpasar, Bali", "Indonesia"),
    "delhi":            ("DEL", "Delhi", "India"),
    "mumbai":           ("BOM", "Mumbai", "India"),
    "bangalore":        ("BLR", "Bengaluru", "India"),
    "chennai":          ("MAA", "Chennai", "India"),
    "hyderabad":        ("HYD", "Hyderabad", "India"),
    "kolkata":          ("CCU", "Kolkata", "India"),
    "goa":              ("GOI", "Goa", "India"),
    "colombo":          ("CMB", "Colombo", "Sri Lanka"),
    "kathmandu":        ("KTM", "Kathmandu", "Nepal"),
    "karachi":          ("KHI", "Karachi", "Pakistan"),
    "lahore":           ("LHE", "Lahore", "Pakistan"),
    "islamabad":        ("ISB", "Islamabad", "Pakistan"),
    "dhaka":            ("DAC", "Dhaka", "Bangladesh"),
    "guangzhou":        ("CAN", "Guangzhou", "China"),
    "shenzhen":         ("SZX", "Shenzhen", "China"),
    "chengdu":          ("CTU", "Chengdu", "China"),
    "taipei":           ("TPE", "Taipei Taoyuan", "Taiwan"),
    "sydney":           ("SYD", "Sydney", "Australia"),
    "melbourne":        ("MEL", "Melbourne", "Australia"),
    "brisbane":         ("BNE", "Brisbane", "Australia"),
    "perth":            ("PER", "Perth", "Australia"),
    "adelaide":         ("ADL", "Adelaide", "Australia"),
    "auckland":         ("AKL", "Auckland", "New Zealand"),
    "wellington":       ("WLG", "Wellington", "New Zealand"),
    "christchurch":     ("CHC", "Christchurch", "New Zealand"),
    # South America
    "rio de janeiro":   ("GIG", "Rio de Janeiro", "Brazil"),
    "lima":             ("LIM", "Lima", "Peru"),
    "santiago":         ("SCL", "Santiago", "Chile"),
    "bogota":           ("BOG", "Bogotá", "Colombia"),
    "medellin":         ("MDE", "Medellín", "Colombia"),
    "quito":            ("UIO", "Quito", "Ecuador"),
    "montevideo":       ("MVD", "Montevideo", "Uruguay"),
    "havana":           ("HAV", "Havana", "Cuba"),
    "barbados":         ("BGI", "Barbados", "Barbados"),
    "jamaica":          ("KIN", "Kingston", "Jamaica"),
}

#: What people actually say, mapped to what the table is keyed on.
_ALIASES: dict[str, str] = {
    "nyc": "new york", "new york city": "new york", "manhattan": "new york",
    "la": "los angeles", "l.a.": "los angeles", "sf": "san francisco",
    "vegas": "las vegas", "dc": "washington", "washington dc": "washington",
    "brum": "birmingham", "the smoke": "london",
    "amsterdam schiphol": "amsterdam", "schiphol": "amsterdam",
    "barca": "barcelona", "munchen": "munich", "koln": "cologne",
    "firenze": "florence", "roma": "rome", "milano": "milan",
    "napoli": "naples", "venezia": "venice", "torino": "turin",
    "lisboa": "lisbon", "sevilla": "seville", "mallorca": "palma",
    "majorca": "palma", "menorca": "palma", "canaries": "tenerife",
    "wien": "vienna", "praha": "prague", "warszawa": "warsaw",
    "krakau": "krakow", "cracow": "krakow", "constantinople": "istanbul",
    "saigon": "ho chi minh city", "bombay": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "bengaluru": "bangalore", "peking": "beijing",
    "uae": "dubai", "qatar": "doha", "sao paolo": "sao paulo",
    "new delhi": "delhi", "changi": "singapore",
    "são paulo": "sao paulo", "zürich": "zurich",
    "düsseldorf": "dusseldorf", "málaga": "malaga",
    "montréal": "montreal", "cancún": "cancun",
    "bogotá": "bogota", "medellín": "medellin",
    "kraków": "krakow", "marrakech": "marrakesh",
    "ibiza town": "ibiza", "tenerife south": "tenerife",
    "gran canaria las palmas": "gran canaria",
}

#: Words that surround a place name in speech and are never part of one.
_PLACE_NOISE = re.compile(
    r"^\s*(?:the\s+|an?\s+|to\s+|from\s+|in\s+|at\s+|for\s+)+|"
    r"\s*(?:airport|international|intl\.?|city\s+centre|city\s+center)\s*$",
    re.IGNORECASE,
)

#: Every IATA code the table knows, for accepting one typed directly.
_KNOWN_CODES: dict[str, tuple[str, str, str]] = {
    entry[0]: entry for entry in (*_METRO.values(), *_AIRPORTS.values())
}


def _normalise_place(raw: str) -> str:
    text = str(raw or "").strip().lower()
    text = text.replace(".", " ").replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Applied repeatedly: "to the airport" needs both ends stripped, and each
    # pass can expose another prefix underneath.
    for _ in range(3):
        stripped = _PLACE_NOISE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    return text


def resolve_airport(raw: str) -> Airport | None:
    """A place name or IATA code as an :class:`Airport`, or ``None``.

    A city with several airports resolves to its metropolitan code, so
    "London" searches Heathrow, Gatwick, Stansted, Luton and City together —
    which is what the word means when a person says it. Asking for "Heathrow"
    still gets LHR alone.
    """
    text = _normalise_place(raw)
    if not text:
        return None

    # A bare code, typed or spelled out. Checked against the table rather than
    # accepted on shape alone: "the" and "for" are three letters too.
    upper = text.upper()
    if len(upper) == 3 and upper in _KNOWN_CODES:
        code, name, country = _KNOWN_CODES[upper]
        return Airport(code, name, country, metro=code in
                       {entry[0] for entry in _METRO.values()})

    text = _ALIASES.get(text, text)

    if text in _METRO:
        code, name, country = _METRO[text]
        return Airport(code, name, country, metro=True)
    if text in _AIRPORTS:
        code, name, country = _AIRPORTS[text]
        return Airport(code, name, country)

    # "london heathrow", "paris cdg", "tokyo narita" — a city qualified by its
    # airport. The longest match wins so "london city" beats "london".
    best: tuple[str, tuple[str, str, str]] | None = None
    for key, entry in _AIRPORTS.items():
        if key in text and (best is None or len(key) > len(best[0])):
            best = (key, entry)
    if best is not None:
        code, name, country = best[1]
        return Airport(code, name, country)
    for key, entry in _METRO.items():
        if key in text:
            code, name, country = entry
            return Airport(code, name, country, metro=True)
    return None


# ── dates ─────────────────────────────────────────────────────────────────────

_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12,
    "dec": 12,
}

_WEEKDAYS: dict[str, int] = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}


def parse_date(raw: str, *, today: date | None = None) -> str | None:
    """A date expression as ``YYYY-MM-DD``, or ``None`` if it is not one.

    ``None`` is the important return. Falling back to today's date would hand
    back a plausible-looking answer for a question that was never understood,
    and a flight search for the wrong day reads exactly like one for the right
    day. ORION asks instead.
    """
    text = str(raw or "").strip().lower()
    if not text:
        return None
    now = today or date.today()

    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        try:
            parts = [int(p) for p in text.split("-")]
            return date(*parts).isoformat()
        except ValueError:
            return None

    # Day first: this is a British assistant, and 03/04 is the third of April.
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d.%m.%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass

    fixed = {
        "today": 0, "tonight": 0, "tomorrow": 1,
        "the day after tomorrow": 2, "day after tomorrow": 2,
        "overmorrow": 2,
    }
    if text in fixed:
        return (now + timedelta(days=fixed[text])).isoformat()

    match = re.fullmatch(r"in\s+(a|an|\d{1,3})\s+(day|week|month)s?", text)
    if match:
        count = 1 if match.group(1) in {"a", "an"} else int(match.group(1))
        unit = match.group(2)
        days = count * {"day": 1, "week": 7, "month": 30}[unit]
        return (now + timedelta(days=days)).isoformat()

    if text in {"next week", "a week today", "this time next week"}:
        return (now + timedelta(days=7)).isoformat()
    if text in {"next month", "a month today"}:
        return (now + timedelta(days=30)).isoformat()

    # "friday", "next friday", "this friday" — always forward, never today.
    match = re.fullmatch(r"(?:(next|this|coming)\s+)?([a-z]+)", text)
    if match and match.group(2) in _WEEKDAYS:
        target = _WEEKDAYS[match.group(2)]
        ahead = (target - now.weekday()) % 7
        if ahead == 0:
            ahead = 7            # "Friday" said on a Friday means the next one
        if match.group(1) == "next" and ahead < 7:
            ahead += 7
        return (now + timedelta(days=ahead)).isoformat()

    return _parse_spoken_date(text, now)


def _parse_spoken_date(text: str, now: date) -> str | None:
    """"15 March", "March 15th", "15th of March 2027"."""
    cleaned = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text)
    cleaned = cleaned.replace(" of ", " ").replace(",", " ")

    month: int | None = None
    for name, number in _MONTHS.items():
        if re.search(rf"\b{name}\b", cleaned):
            # A longer name wins: "mar" also matches inside "march".
            if month is None or len(name) > len(
                    next(n for n, v in _MONTHS.items() if v == month)):
                month = number
    if month is None:
        return None

    year: int | None = None
    year_match = re.search(r"\b(20\d{2})\b", cleaned)
    if year_match:
        year = int(year_match.group(1))
        cleaned = cleaned.replace(year_match.group(1), " ")

    day_match = re.search(r"\b(\d{1,2})\b", cleaned)
    if not day_match:
        return None
    day = int(day_match.group(1))

    if year is None:
        # No year given: the next time that date comes round. Someone asking in
        # December about "10 January" means five weeks away, not eleven months
        # ago.
        year = now.year
        try:
            if date(year, month, day) < now:
                year += 1
        except ValueError:
            return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None          # 31 February


# ── the search URL ────────────────────────────────────────────────────────────

_CABINS = {
    "economy": "economy", "coach": "economy", "standard": "economy",
    "premium": "premium economy", "premium economy": "premium economy",
    "business": "business", "business class": "business",
    "first": "first", "first class": "first",
}


def search_url(
    origin: Airport,
    destination: Airport,
    depart: str,
    return_date: str | None = None,
    passengers: int = 1,
    cabin: str = "economy",
    currency: str = "GBP",
) -> str:
    """A Google Flights URL carrying this search and nothing else.

    Google Flights also accepts a ``tfs`` parameter: a base64 blob encoding a
    complete itinerary, which takes precedence over the readable query beside
    it. A builder that pastes a fixed one in returns a URL for whichever route
    was baked into the constant, whatever arguments it was called with. There
    is no ``tfs`` here on purpose — the readable query is the whole search.
    """
    seats = max(1, min(9, int(passengers or 1)))
    cabin_phrase = _CABINS.get(str(cabin or "").strip().lower(), "economy")

    parts = ["Flights"]
    if cabin_phrase != "economy":
        parts = [f"{cabin_phrase.title()} class flights"]
    parts.append(f"from {origin.code} to {destination.code}")
    parts.append(f"on {depart}")
    if return_date:
        parts.append(f"returning {return_date}")
    if seats > 1:
        parts.append(f"for {seats} passengers")

    return (
        "https://www.google.com/travel/flights"
        f"?q={quote_plus(' '.join(parts))}"
        f"&curr={quote_plus(str(currency or 'GBP').upper())}"
        "&hl=en&gl=GB"
    )


# ── offers ────────────────────────────────────────────────────────────────────

@dataclass
class Offer:
    """One flight option as read off the page."""

    airline: str = ""
    departure: str = ""
    arrival: str = ""
    duration: str = ""
    stops: int | None = None
    price: str = ""
    currency: str = ""

    @property
    def amount(self) -> float | None:
        """The price as a number, or ``None`` if there is not one.

        Stripping every non-digit turns "£1,234.56" into 123456, which sorts
        above "£999" and makes the cheapest flight the dearest. The decimal
        part is separated before the thousands separators are removed.
        """
        return money(self.price)

    @property
    def stops_phrase(self) -> str:
        if self.stops is None:
            return ""
        if self.stops == 0:
            return "non-stop"
        return f"{self.stops} stop{'s' if self.stops > 1 else ''}"


_MONEY = re.compile(
    r"(?P<symbol>[£$€¥₹]|\b(?:GBP|USD|EUR|JPY|INR|AUD|CAD|CHF)\b)?\s*"
    r"(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
)

_SYMBOL_TO_CODE = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY", "₹": "INR"}


def money(text: str) -> float | None:
    """The numeric value of a price string, honouring its separators."""
    match = _MONEY.search(str(text or ""))
    if not match:
        return None
    number = match.group("number")
    # A single group of exactly three after a comma is a thousands separator;
    # the decimal point is the only thing that introduces a fraction.
    try:
        return float(number.replace(",", ""))
    except ValueError:
        return None


def currency_of(text: str) -> str:
    match = _MONEY.search(str(text or ""))
    symbol = (match.group("symbol") if match else None) or ""
    return _SYMBOL_TO_CODE.get(symbol, symbol.upper())


#: A price has a currency on it. `_MONEY` deliberately does not require one —
#: a model returns the amount and the currency in separate fields — but when
#: reading a page, an optional symbol means the first bare number wins, and on
#: a flight page the first bare number is the hour of the departure time. That
#: is how a fare of "07" came out of "07:35".
_AMOUNT = r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
_PRICE = re.compile(
    rf"[£$€¥₹]\s*(?:{_AMOUNT})|\b(?:GBP|USD|EUR|JPY|INR|AUD|CAD|CHF)\s*(?:{_AMOUNT})"
)

_TIME = re.compile(r"\b(\d{1,2}:\d{2}\s*(?:AM|PM)?|\d{1,2}\s*(?:AM|PM))\b",
                   re.IGNORECASE)
_DURATION = re.compile(r"\b(\d{1,2}\s*hr?s?(?:\s*\d{1,2}\s*min)?|"
                       r"\d{1,2}h\s*\d{1,2}m?)\b", re.IGNORECASE)
_STOPS = re.compile(r"\b(nonstop|non-stop|direct|(\d)\s*stops?)\b",
                    re.IGNORECASE)


def read_offers(page_text: str, limit: int = MAX_OFFERS) -> list[Offer]:
    """Best-effort structured offers from raw page text.

    Deliberately conservative: a row needs two times and a price carrying a
    currency before it is believed. Half-read rows presented as flights are
    worse than an honest "I couldn't read the page" — the user acts on these.

    This is the fallback. The model path in :class:`FlightFinder` reads the
    same text far better; this exists so a flight search still returns
    something when no model is reachable.
    """
    rows: list[Offer] = []
    lines = [line.strip() for line in str(page_text or "").splitlines()
             if line.strip()]

    index = 0
    while index < len(lines) and len(rows) < limit:
        # Google Flights renders one offer as a run of short lines, so a whole
        # offer is read from a small window rather than from a single line.
        window_end = min(len(lines), index + _WINDOW)
        window = " | ".join(lines[index:window_end])

        times = _TIME.findall(window)
        price_match = _PRICE.search(window)
        if len(times) < 2 or price_match is None:
            index += 1
            continue

        duration = _DURATION.search(window)
        stop_match = _STOPS.search(window)
        stops: int | None = None
        if stop_match:
            stops = int(stop_match.group(2)) if stop_match.group(2) else 0

        price = price_match.group(0).strip()
        rows.append(Offer(
            airline=_airline_near(lines, index),
            departure=times[0].strip(),
            arrival=times[1].strip(),
            duration=(duration.group(0).strip() if duration else ""),
            stops=stops,
            price=price,
            currency=currency_of(price),
        ))
        # Step past the offer just read. Advancing one line at a time makes
        # overlapping windows re-read the same flight from a different offset,
        # which is how a two-flight page produced five options.
        step = 1
        for offset in range(index, window_end):
            if _PRICE.search(lines[offset]):
                step = offset - index + 1
                break
        index += step

    return rows


#: Lines that one offer is rendered across. Wide enough to reach from the
#: carrier's name to its fare, narrow enough not to reach the next flight's.
_WINDOW = 7


_AIRLINE_HINT = re.compile(
    r"\b(British Airways|Ryanair|easyJet|Jet2|Wizz Air|Vueling|Aer Lingus|"
    r"Lufthansa|Air France|KLM|Iberia|TAP|Swiss|Austrian|Brussels Airlines|"
    r"SAS|Finnair|Norwegian|Turkish Airlines|Emirates|Qatar Airways|Etihad|"
    r"Virgin Atlantic|American|Delta|United|JetBlue|Southwest|Alaska|"
    r"Air Canada|WestJet|Qantas|Singapore Airlines|Cathay|ANA|JAL|"
    r"Air India|IndiGo|Etihad|Ethiopian|Kenya Airways|LATAM|Avianca|"
    r"Air New Zealand|Eurowings|Transavia|Pegasus|Aegean|ITA Airways|"
    r"Scandinavian|LOT|Croatia Airlines|TAROM|Air Serbia|Royal Air Maroc)\b",
    re.IGNORECASE,
)


def _airline_near(lines: list[str], index: int) -> str:
    """The carrier named closest to an offer row, if one is."""
    for offset in range(-2, 7):
        position = index + offset
        if 0 <= position < len(lines):
            match = _AIRLINE_HINT.search(lines[position])
            if match:
                return match.group(0)
    return ""


def cheapest(offers: Iterable[Offer]) -> Offer | None:
    priced = [offer for offer in offers if offer.amount is not None]
    if not priced:
        return None
    return min(priced, key=lambda offer: offer.amount or 0.0)


# ── what ORION says and writes ────────────────────────────────────────────────

@dataclass
class FlightSearch:
    """Everything one search produced."""

    origin: Airport
    destination: Airport
    depart: str
    return_date: str | None = None
    passengers: int = 1
    cabin: str = "economy"
    url: str = ""
    offers: list[Offer] = field(default_factory=list)
    note: str = ""

    @property
    def route(self) -> str:
        return f"{self.origin} → {self.destination}"


def _say_date(iso: str) -> str:
    """"2027-03-15" as "Monday the 15th of March" — a date said, not read."""
    try:
        when = date.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso
    day = when.day
    suffix = ("th" if 11 <= day <= 13 else
              {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th"))
    return when.strftime(f"%A the {day}{suffix} of %B")


def spoken_summary(search: FlightSearch) -> str:
    """A few sentences ORION can say without them becoming a recitation."""
    route = f"{search.origin.name} to {search.destination.name}"
    when = _say_date(search.depart)

    if not search.offers:
        detail = f" {search.note}" if search.note else ""
        return (f"I couldn't read any prices for {route} on {when}."
                f"{detail} I've left the search open so you can look yourself.")

    best = cheapest(search.offers)
    lead = (f"I found {len(search.offers)} option"
            f"{'s' if len(search.offers) != 1 else ''} for {route} on {when}.")

    if best is not None:
        parts = [f"The cheapest is {best.price}"]
        if best.airline:
            parts.append(f"with {best.airline}")
        if best.departure:
            parts.append(f"leaving at {best.departure}")
        if best.stops_phrase:
            parts.append(best.stops_phrase)
        lead += " " + ", ".join(parts) + "."

    # Two more, not four. A spoken list stops being listenable after about
    # three items, and the written report has all of them.
    others = [offer for offer in search.offers if offer is not best][:2]
    for offer in others:
        bits = [bit for bit in (offer.airline, offer.price,
                                offer.departure and f"at {offer.departure}",
                                offer.stops_phrase) if bit]
        if bits:
            lead += " There's also " + ", ".join(bits) + "."

    return lead + " The full list is on screen."


def written_report(search: FlightSearch) -> str:
    """The version with everything in it, for the panel and the log."""
    lines = [
        f"Flights — {search.route}",
        f"Departing  {_say_date(search.depart)}  ({search.depart})",
    ]
    if search.return_date:
        lines.append(
            f"Returning  {_say_date(search.return_date)}  ({search.return_date})")
    if search.passengers > 1:
        lines.append(f"Passengers {search.passengers}")
    if search.cabin and search.cabin != "economy":
        lines.append(f"Cabin      {search.cabin.title()}")
    lines.append(f"Searched   {datetime.now().strftime('%d %b %Y, %H:%M')}")
    lines.append("")

    if not search.offers:
        lines.append(search.note or "No prices could be read from the page.")
    else:
        best = cheapest(search.offers)
        for number, offer in enumerate(search.offers, 1):
            head = f"{number}. {offer.airline or 'Airline not named'}"
            if offer is best:
                head += "   ← cheapest"
            lines.append(head)
            detail = [bit for bit in (
                f"{offer.departure} → {offer.arrival}"
                if offer.departure and offer.arrival else "",
                offer.duration, offer.stops_phrase, offer.price) if bit]
            lines.append("   " + "  ·  ".join(detail))

    lines.append("")
    lines.append(search.url)
    return "\n".join(lines)


def as_content_rows(search: FlightSearch) -> list[dict[str, str]]:
    """The offers as rows the content panel already knows how to render."""
    rows: list[dict[str, str]] = []
    best = cheapest(search.offers)
    for offer in search.offers:
        bits = [bit for bit in (
            offer.price,
            f"{offer.departure}–{offer.arrival}"
            if offer.departure and offer.arrival else "",
            offer.duration, offer.stops_phrase) if bit]
        title = f"{offer.airline or 'Flight'} · " + "  ".join(bits)
        rows.append({
            "title": title + ("   (cheapest)" if offer is best else ""),
            "url": search.url,
            "topic": f"{search.origin.code}→{search.destination.code}"
                     f" {search.depart}",
        })
    return rows



# ── the part that touches the world ───────────────────────────────────────────

class FlightRequestError(ValueError):
    """The request could not be understood well enough to search.

    Carries the sentence ORION should say. Raised rather than returned so an
    ambiguous request can never be mistaken for a completed search — the one
    failure mode that matters here is answering confidently about the wrong
    day or the wrong Birmingham.
    """


class FlightFinder:
    """Opens the search, reads the page, turns it into offers.

    The browser co-pilot and the model router are both injected and both
    optional. Without a browser there is nothing to read, and the search falls
    back to handing over the URL — which is still a useful answer. Without a
    model the deterministic reader runs instead, which finds fewer offers but
    invents none.
    """

    def __init__(self, bus: Any = None, router: Any = None,
                 copilot: Any = None) -> None:
        self.bus = bus
        self.router = router
        self.copilot = copilot

    # ── the request ──────────────────────────────────────────────────────────

    def understand(self, args: dict[str, Any]) -> FlightSearch:
        """Turn loose arguments into a search, or say what is missing.

        Every failure here is a question ORION can ask, not a stack trace: the
        model calling this tool passes whatever the user said, and "Brum" and
        "a week on Thursday" are both perfectly ordinary ways to say it.
        """
        origin_raw = str(args.get("origin") or args.get("from") or "").strip()
        dest_raw = str(args.get("destination") or args.get("to") or "").strip()
        if not origin_raw:
            raise FlightRequestError("Where are you flying from?")
        if not dest_raw:
            raise FlightRequestError("Where would you like to fly to?")

        origin = resolve_airport(origin_raw)
        if origin is None:
            raise FlightRequestError(
                f"I don't know an airport for {origin_raw!r}. "
                "Could you give me the city or its three-letter code?")
        destination = resolve_airport(dest_raw)
        if destination is None:
            raise FlightRequestError(
                f"I don't know an airport for {dest_raw!r}. "
                "Could you give me the city or its three-letter code?")
        if origin.code == destination.code:
            raise FlightRequestError(
                f"That's {origin.name} in both directions — "
                "where did you want to fly to?")

        depart_raw = str(args.get("depart") or args.get("date")
                         or args.get("when") or "").strip()
        if not depart_raw:
            raise FlightRequestError("Which day did you want to fly?")
        depart = parse_date(depart_raw)
        if depart is None:
            raise FlightRequestError(
                f"I couldn't work out a date from {depart_raw!r}. "
                "Could you say it as a day and month?")

        return_raw = str(args.get("return_date") or args.get("returning")
                         or args.get("back") or "").strip()
        return_date: str | None = None
        if return_raw:
            return_date = parse_date(return_raw)
            if return_date is None:
                raise FlightRequestError(
                    f"I understood the outbound date, but not {return_raw!r} "
                    "for the return. Could you say it as a day and month?")
            if return_date < depart:
                raise FlightRequestError(
                    "The return date is before the outbound one — "
                    "did you mean them the other way round?")

        try:
            passengers = max(1, min(9, int(args.get("passengers") or 1)))
        except (TypeError, ValueError):
            passengers = 1

        cabin = _CABINS.get(str(args.get("cabin") or "").strip().lower(),
                            "economy")

        search = FlightSearch(
            origin=origin, destination=destination, depart=depart,
            return_date=return_date, passengers=passengers, cabin=cabin,
        )
        search.url = search_url(
            origin, destination, depart, return_date, passengers, cabin,
            str(args.get("currency") or "GBP"),
        )
        return search

    # ── the search ───────────────────────────────────────────────────────────

    async def search(self, args: dict[str, Any]) -> FlightSearch:
        search = self.understand(args)
        self._say(f"Searching {search.route} on {search.depart}.")

        page = await self._read_page(search)
        if not page:
            search.note = ("I couldn't read the results page — the browser "
                           "didn't come back with any text.")
            return search

        offers = await self._extract(page, search)
        if not offers:
            offers = read_offers(page)
            if offers:
                search.note = "Read from the page directly."
        if not offers:
            search.note = ("The page loaded but no prices were on it yet — "
                           "Google Flights sometimes needs a moment longer.")

        search.offers = offers[:MAX_OFFERS]
        self._publish(search)
        return search

    async def _read_page(self, search: FlightSearch) -> str:
        """Open the search and return the rendered text."""
        if self.copilot is None:
            return ""
        import asyncio
        try:
            opened = await self.copilot.open(search.url)
            if opened is not None and not getattr(opened, "ok", True):
                return ""
            # Prices stream in after the shell paints. Reading immediately
            # reliably returns a page with no prices on it.
            await asyncio.sleep(RENDER_WAIT_S)
            result = await self.copilot.read(MAX_PAGE_CHARS)
        except Exception:
            return ""
        text = getattr(result, "text", "") or ""
        return text if getattr(result, "ok", True) else ""

    async def _extract(self, page: str, search: FlightSearch) -> list[Offer]:
        """Ask the model to read the page, then check what it says.

        A model reading a cluttered page will occasionally produce a tidy,
        plausible offer that is not on it. Every price is therefore checked
        against the page text before the offer is kept — the user acts on
        these numbers, and an invented fare is the one error that costs money.
        """
        if self.router is None or not page.strip():
            return []
        prompt = (
            f"Below is the text of a Google Flights results page for "
            f"{search.origin.code} to {search.destination.code} on "
            f"{search.depart}.\n\n"
            f"Return a JSON array of up to {MAX_OFFERS} flight options, "
            f"cheapest first, using exactly these keys:\n"
            '[{"airline": "", "departure": "HH:MM", "arrival": "HH:MM", '
            '"duration": "", "stops": 0, "price": "", "currency": ""}]\n\n'
            "Copy every price exactly as it is written on the page, including "
            "its symbol. Do not estimate, convert or round anything. If the "
            "page has no flight prices on it, return []. Return only the JSON "
            "array.\n\n"
            f"{page[:MAX_PAGE_CHARS]}"
        )
        try:
            _profile, raw = await self.router.generate_text(
                prompt,
                system_extra="You extract structured data from page text. "
                             "You return only JSON, never prose.",
            )
        except Exception:
            return []

        rows = _json_array(raw)
        offers: list[Offer] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            price = str(row.get("price") or "").strip()
            if price and not _price_on_page(price, page):
                continue          # not on the page; do not report it
            stops: int | None
            try:
                stops = int(row["stops"]) if row.get("stops") is not None else None
            except (TypeError, ValueError):
                stops = None
            offers.append(Offer(
                airline=str(row.get("airline") or "").strip(),
                departure=str(row.get("departure") or "").strip(),
                arrival=str(row.get("arrival") or "").strip(),
                duration=str(row.get("duration") or "").strip(),
                stops=stops,
                price=price,
                currency=str(row.get("currency") or "").strip()
                         or currency_of(price),
            ))
        return offers

    # ── telling the rest of ORION ────────────────────────────────────────────

    def _say(self, line: str) -> None:
        bus = self.bus
        if bus is None:
            return
        try:
            bus.log.emit(f"[FLIGHTS] {line}")
        except Exception:
            pass

    def _publish(self, search: FlightSearch) -> None:
        """Put the offers on the content panel, where they can be clicked."""
        bus = self.bus
        if bus is None or not search.offers:
            return
        try:
            bus.content_results.emit(as_content_rows(search))
        except Exception:
            pass


def _json_array(raw: str) -> list[Any]:
    """The JSON array in a model reply, however it was fenced."""
    import json

    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


#: A number standing on its own: not part of a longer number, and not half of
#: a clock time. The lookarounds are the whole point — "7" sits inside "07:35"
#: and inside "3 hr 35 min", so a plain substring test finds any small number
#: on any flight page.
_STANDALONE_NUMBER = re.compile(
    r"(?<![\d.,:])(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r"(?![\d.,]|\s*:)"
)


def _page_amounts(page: str) -> set[float]:
    """Every number written on the page as a number in its own right."""
    amounts: set[float] = set()
    for match in _STANDALONE_NUMBER.finditer(str(page or "")):
        try:
            amounts.add(float(match.group(1).replace(",", "")))
        except ValueError:
            continue
    return amounts


def _price_on_page(price: str, page: str) -> bool:
    """Whether a quoted price genuinely appears in the page text.

    Compared as a number rather than as text, so the page and the model may
    punctuate thousands differently and still agree — and so a fare the model
    invented cannot be waved through by its digits happening to occur inside a
    departure time.
    """
    amount = money(price)
    if amount is None:
        return False
    return amount in _page_amounts(page)


__all__ = [
    "MAX_OFFERS", "MAX_PAGE_CHARS", "RENDER_WAIT_S",
    "Airport", "Offer", "FlightSearch", "FlightFinder", "FlightRequestError",
    "as_content_rows", "cheapest", "currency_of", "money",
    "parse_date", "read_offers", "resolve_airport", "search_url",
    "spoken_summary", "written_report",
]
