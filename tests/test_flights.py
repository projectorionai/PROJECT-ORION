"""
Finding flights without inventing them.

Nearly every way this can go wrong is a parsing mistake, and the expensive
ones all look like success: the wrong Birmingham, the wrong day, a fare that
was never on the page. So the pure layer — places, dates, URLs, money — carries
most of these tests, and the browser layer is exercised with fakes.

The behaviour these protect most carefully is the refusal to guess. An
unparseable date must come back as a question, because a flight search for the
wrong day reads exactly like one for the right day and the user books it.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.flights import (  # noqa: E402
    MAX_OFFERS,
    Airport,
    FlightFinder,
    FlightRequestError,
    FlightSearch,
    Offer,
    as_content_rows,
    cheapest,
    currency_of,
    money,
    parse_date,
    read_offers,
    resolve_airport,
    search_url,
    spoken_summary,
    written_report,
)

#: A Saturday, so weekday arithmetic has a known starting point.
SATURDAY = date(2026, 9, 19)


# ── places ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("said, code", [
    ("birmingham", "BHX"),
    ("Birmingham", "BHX"),
    ("BHX", "BHX"),
    ("manchester", "MAN"),
    ("to New York", "NYC"),
    ("the airport at manchester", "MAN"),
    ("nyc", "NYC"),
    ("vegas", "LAS"),
    ("Dubai International", "DXB"),
])
def test_places_people_actually_say(said, code):
    airport = resolve_airport(said)
    assert airport is not None and airport.code == code


def test_a_city_with_several_airports_searches_all_of_them():
    """"A flight to London" does not mean Heathrow specifically. Resolving to
    LHR silently drops Gatwick, Stansted, Luton and City from the search."""
    london = resolve_airport("london")
    assert london.code == "LON"
    assert london.metro is True


def test_but_a_named_airport_is_still_exact():
    assert resolve_airport("heathrow").code == "LHR"
    assert resolve_airport("gatwick").code == "LGW"
    assert resolve_airport("london city").code == "LCY"


def test_a_named_airport_beats_the_city_it_is_in():
    """"Paris CDG" is a request for Charles de Gaulle, not for all of Paris."""
    assert resolve_airport("paris cdg").code == "CDG"
    assert resolve_airport("paris").code == "PAR"


def test_accents_and_local_spellings_resolve():
    assert resolve_airport("Zürich").code == "ZRH"
    assert resolve_airport("roma").code == "ROM"
    assert resolve_airport("bombay").code == "BOM"


def test_an_unknown_place_is_admitted_not_guessed():
    assert resolve_airport("atlantis") is None
    assert resolve_airport("") is None
    assert resolve_airport(None) is None


def test_a_three_letter_word_is_not_an_airport_code():
    """"the" and "for" are three letters. Shape alone cannot mean a code."""
    assert resolve_airport("zzz") is None


# ── dates ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("said, expected", [
    ("2027-03-15", "2027-03-15"),
    ("15/03/2027", "2027-03-15"),
    ("15.03.2027", "2027-03-15"),
    ("today", "2026-09-19"),
    ("tomorrow", "2026-09-20"),
    ("the day after tomorrow", "2026-09-21"),
    ("in 3 days", "2026-09-22"),
    ("in a week", "2026-09-26"),
    ("next week", "2026-09-26"),
    ("15 March", "2027-03-15"),
    ("March 15th", "2027-03-15"),
    ("15th of March 2027", "2027-03-15"),
])
def test_dates_people_actually_say(said, expected):
    assert parse_date(said, today=SATURDAY) == expected


def test_a_british_assistant_reads_the_day_first():
    """03/04 is the third of April here, not the fourth of March."""
    assert parse_date("03/04/2027", today=SATURDAY) == "2027-04-03"


def test_a_weekday_always_means_the_next_one():
    """Said on a Saturday, "Saturday" means the one coming, not today —
    nobody searches for a flight they would have to be on already."""
    assert parse_date("saturday", today=SATURDAY) == "2026-09-26"
    assert parse_date("friday", today=SATURDAY) == "2026-09-25"


def test_next_friday_is_further_out_than_friday():
    plain = parse_date("friday", today=SATURDAY)
    next_one = parse_date("next friday", today=SATURDAY)
    assert next_one > plain


def test_a_date_with_no_year_rolls_forward():
    """Asked in September about "10 January", they mean four months away."""
    assert parse_date("10 January", today=SATURDAY) == "2027-01-10"
    assert parse_date("25 December", today=SATURDAY) == "2026-12-25"


def test_the_longest_month_name_wins():
    """"march" contains "mar"; reading the abbreviation gives the same month
    here but the comparison has to be right for it to keep being right."""
    assert parse_date("3 March 2027", today=SATURDAY) == "2027-03-03"
    assert parse_date("3 May 2027", today=SATURDAY) == "2027-05-03"


@pytest.mark.parametrize("nonsense", [
    "", "   ", "blorp", "31 February", "sometime soonish", "the usual", None,
])
def test_a_date_it_cannot_read_is_refused_not_guessed(nonsense):
    """The load-bearing test.

    Falling back to today returns a plausible-looking answer to a question
    that was never understood, and a search for the wrong day is
    indistinguishable from one for the right day. The user books it.
    """
    assert parse_date(nonsense, today=SATURDAY) is None


# ── the URL ───────────────────────────────────────────────────────────────────

def test_the_url_carries_this_search():
    url = search_url(resolve_airport("birmingham"), resolve_airport("new york"),
                     "2027-03-15")
    assert "BHX" in url and "NYC" in url and "2027-03-15" in url


def test_the_url_carries_no_stale_itinerary():
    """Google Flights' `tfs` parameter is a base64 blob encoding a complete
    itinerary, and it overrides the readable query beside it. A builder that
    pastes a fixed one in returns whichever route was baked into the constant,
    whatever it was called with."""
    url = search_url(resolve_airport("birmingham"), resolve_airport("malaga"),
                     "2027-03-15")
    assert "tfs=" not in url


def test_the_url_reflects_its_arguments():
    one = search_url(resolve_airport("london"), resolve_airport("rome"),
                     "2027-03-15")
    two = search_url(resolve_airport("london"), resolve_airport("athens"),
                     "2027-06-02")
    assert one != two


def test_a_return_and_a_cabin_reach_the_query():
    url = search_url(resolve_airport("london"), resolve_airport("tokyo"),
                     "2027-03-15", "2027-03-29", 2, "business")
    assert "2027-03-29" in url and "Business" in url and "2" in url


def test_passenger_count_is_bounded():
    """Nine is the maximum a single booking takes; zero is not a trip."""
    for count, expected in ((0, "1"), (1, "1"), (40, "9")):
        url = search_url(resolve_airport("london"), resolve_airport("rome"),
                         "2027-03-15", passengers=count)
        if expected == "1":
            assert "passengers" not in url
        else:
            assert f"{expected}+passengers" in url


# ── money ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, value", [
    ("£149", 149.0),
    ("£1,234", 1234.0),
    ("£1,234.56", 1234.56),
    ("$899", 899.0),
    ("€89.50", 89.5),
    ("from £212 return", 212.0),
    ("free", None),
    ("", None),
])
def test_money_is_parsed_as_money(text, value):
    assert money(text) == value


def test_the_cheapest_is_the_cheapest():
    """Stripping every non-digit turns "£1,234.56" into 123456, which sorts
    above "£999" and makes the dearest flight look like the cheapest."""
    offers = [Offer(airline="A", price="£1,234.56"),
              Offer(airline="B", price="£999"),
              Offer(airline="C", price="£1,100")]
    assert cheapest(offers).airline == "B"


def test_an_unpriced_list_has_no_cheapest():
    assert cheapest([Offer(airline="A"), Offer(airline="B")]) is None
    assert cheapest([]) is None


def test_currency_is_read_from_the_symbol():
    assert currency_of("£149") == "GBP"
    assert currency_of("$149") == "USD"
    assert currency_of("€149") == "EUR"
    assert currency_of("149") == ""


# ── reading the page ──────────────────────────────────────────────────────────

PAGE = """
Departing flights
British Airways
07:35
10:55
3 hr 20 min
Nonstop
£212
Ryanair
06:10
09:45
3 hr 35 min
Nonstop
£89
Lufthansa
12:05
18:40
6 hr 35 min
1 stop
£154
Prices include taxes and fees
"""


def test_offers_are_read_off_the_page():
    offers = read_offers(PAGE)
    assert offers, "nothing was read from a page that plainly has flights on it"
    assert any(offer.price == "£212" for offer in offers)


def test_a_row_needs_two_times_and_a_price_to_be_believed():
    """Half-read rows presented as flights are worse than an honest failure —
    the user acts on these."""
    assert read_offers("Cookies and privacy. Sign in. Help centre.") == []
    assert read_offers("") == []


def test_the_reader_is_bounded():
    many = "\n".join(PAGE for _ in range(20))
    assert len(read_offers(many, limit=MAX_OFFERS)) <= MAX_OFFERS


def test_a_fare_needs_a_currency_on_it():
    """Found end to end, not by unit test.

    The money pattern makes the currency symbol optional, because a model
    returns the amount and the currency in separate fields. Reading a page
    with that same optional symbol means the first bare number wins — and on a
    flight page the first bare number is the hour of the departure time. ORION
    reported the cheapest fare as "3", read out of "3 hr 20 min".
    """
    offers = read_offers(PAGE)
    for offer in offers:
        assert any(symbol in offer.price for symbol in "£$€¥₹"), (
            f"{offer.price!r} is a bare number, not a fare")


def test_one_flight_is_one_offer():
    """Also found end to end.

    The reader looks at a window of lines because one offer is rendered across
    several. Advancing a line at a time makes overlapping windows read the
    same flight from a different offset: a two-flight page produced five
    options, three of them the same flights at wrong prices.
    """
    offers = read_offers(PAGE)
    assert len(offers) == 3, f"3 flights on the page, {len(offers)} read"
    assert len({offer.departure for offer in offers}) == len(offers)


def test_the_fares_read_are_the_fares_on_the_page():
    prices = {offer.price for offer in read_offers(PAGE)}
    assert prices == {"£212", "£89", "£154"}


def test_stops_are_read_as_numbers():
    offers = read_offers(PAGE)
    phrases = {offer.stops_phrase for offer in offers}
    assert "non-stop" in phrases


# ── what ORION says ───────────────────────────────────────────────────────────

def _search(**kwargs) -> FlightSearch:
    base = dict(
        origin=resolve_airport("birmingham"),
        destination=resolve_airport("malaga"),
        depart="2027-03-15",
        url="https://example.invalid/search",
        offers=[Offer("Ryanair", "06:10", "09:45", "3 hr 35 min", 0, "£89", "GBP"),
                Offer("Jet2", "11:20", "15:00", "3 hr 40 min", 0, "£154", "GBP")],
    )
    base.update(kwargs)
    return FlightSearch(**base)


def test_the_spoken_version_leads_with_the_cheapest():
    said = spoken_summary(_search())
    assert "£89" in said and "Ryanair" in said


def test_the_spoken_version_says_the_date_rather_than_reading_it():
    said = spoken_summary(_search())
    assert "2027-03-15" not in said
    assert "March" in said and "15th" in said


def test_the_spoken_version_stays_listenable():
    """Beyond about three items a spoken list stops being a list and becomes a
    recitation. The written report has all of them."""
    many = _search(offers=[Offer(f"Airline {n}", "07:00", "10:00",
                                 price=f"£{100 + n}") for n in range(5)])
    assert spoken_summary(many).count("There's also") <= 2


def test_an_empty_result_is_admitted_plainly():
    said = spoken_summary(_search(offers=[]))
    assert "couldn't" in said.lower()
    assert "£" not in said, "no price should be implied when none was read"


def test_the_written_report_has_everything():
    report = written_report(_search())
    assert "Ryanair" in report and "Jet2" in report
    assert "cheapest" in report
    assert "https://example.invalid/search" in report


def test_the_report_names_the_route_both_ends():
    report = written_report(_search())
    assert "BHX" in report and "AGP" in report


def test_offers_become_content_panel_rows():
    rows = as_content_rows(_search())
    assert len(rows) == 2
    assert all(row.get("title") for row in rows), (
        "the content panel drops rows with no title")
    assert all(row.get("url") for row in rows)


# ── understanding a request ───────────────────────────────────────────────────

def _finder() -> FlightFinder:
    return FlightFinder()


def test_a_plain_request_is_understood():
    search = _finder().understand({
        "origin": "Birmingham", "destination": "Malaga", "depart": "2027-03-15"})
    assert search.origin.code == "BHX"
    assert search.destination.code == "AGP"
    assert search.depart == "2027-03-15"
    assert "BHX" in search.url


@pytest.mark.parametrize("args, expected_word", [
    ({"destination": "Malaga", "depart": "2027-03-15"}, "from"),
    ({"origin": "Birmingham", "depart": "2027-03-15"}, "to"),
    ({"origin": "Birmingham", "destination": "Malaga"}, "day"),
    ({"origin": "Atlantis", "destination": "Malaga", "depart": "2027-03-15"},
     "don't know"),
    ({"origin": "Birmingham", "destination": "Malaga", "depart": "the usual"},
     "couldn't work out"),
])
def test_a_request_it_cannot_understand_becomes_a_question(args, expected_word):
    """Every failure here is something ORION can ask, not a stack trace."""
    with pytest.raises(FlightRequestError) as caught:
        _finder().understand(args)
    assert expected_word in str(caught.value).lower() or \
        expected_word in str(caught.value)


def test_flying_somewhere_from_itself_is_queried():
    with pytest.raises(FlightRequestError):
        _finder().understand({"origin": "London", "destination": "LON",
                              "depart": "2027-03-15"})


def test_a_return_before_the_outbound_is_queried():
    with pytest.raises(FlightRequestError) as caught:
        _finder().understand({"origin": "Birmingham", "destination": "Malaga",
                              "depart": "2027-03-15",
                              "return_date": "2027-03-01"})
    assert "other way round" in str(caught.value)


def test_an_unreadable_return_date_is_not_silently_dropped():
    """Losing the return leg turns a round trip into a one-way search and the
    price it reports is then half the real one."""
    with pytest.raises(FlightRequestError):
        _finder().understand({"origin": "Birmingham", "destination": "Malaga",
                              "depart": "2027-03-15", "return_date": "whenever"})


# ── the browser and model layer ───────────────────────────────────────────────

class _Result:
    def __init__(self, text: str = "", ok: bool = True) -> None:
        self.text, self.ok = text, ok


class _Copilot:
    """A browser that returns a fixed page."""

    def __init__(self, page: str = PAGE, ok: bool = True) -> None:
        self.page, self.ok, self.opened = page, ok, []

    async def open(self, url: str, **_kwargs) -> _Result:
        self.opened.append(url)
        return _Result("opened", self.ok)

    async def read(self, max_chars: int = 4000) -> _Result:
        return _Result(self.page[:max_chars], self.ok)


class _Router:
    """A model that returns whatever it was given."""

    def __init__(self, reply: str) -> None:
        self.reply, self.prompts = reply, []

    async def generate_text(self, prompt: str, **_kwargs):
        self.prompts.append(prompt)
        return None, self.reply


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """The real search waits for prices to stream in; the fakes are instant."""
    monkeypatch.setattr("orion_core.flights.RENDER_WAIT_S", 0.0)


def test_a_search_opens_the_page_it_built():
    copilot = _Copilot()
    finder = FlightFinder(copilot=copilot)
    search = _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                                 "depart": "2027-03-15"}))
    assert copilot.opened == [search.url]
    assert search.offers, "the deterministic reader found nothing on a real page"


def test_the_model_reads_the_page_when_there_is_one():
    copilot = _Copilot()
    router = _Router('[{"airline": "Ryanair", "departure": "06:10", '
                     '"arrival": "09:45", "duration": "3 hr 35 min", '
                     '"stops": 0, "price": "£89", "currency": "GBP"}]')
    finder = FlightFinder(router=router, copilot=copilot)
    search = _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                                 "depart": "2027-03-15"}))
    assert [offer.airline for offer in search.offers] == ["Ryanair"]
    assert router.prompts, "the model was never asked"


def test_a_price_that_is_not_on_the_page_is_not_reported():
    """The one error that costs money.

    A model reading a cluttered page will occasionally produce a tidy,
    plausible offer that is not on it. Every price is checked against the page
    text before the offer is kept.
    """
    copilot = _Copilot()
    router = _Router('[{"airline": "Imaginary Air", "departure": "05:00", '
                     '"arrival": "08:00", "stops": 0, "price": "£7", '
                     '"currency": "GBP"}]')
    finder = FlightFinder(router=router, copilot=copilot)
    search = _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                                 "depart": "2027-03-15"}))
    assert not any(offer.airline == "Imaginary Air" for offer in search.offers)


def test_a_fenced_json_reply_is_still_read():
    copilot = _Copilot()
    router = _Router('```json\n[{"airline": "Jet2", "price": "£154"}]\n```')
    finder = FlightFinder(router=router, copilot=copilot)
    search = _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                                 "depart": "2027-03-15"}))
    assert search.offers[0].airline == "Jet2"


def test_a_model_talking_prose_falls_back_to_reading_the_page():
    copilot = _Copilot()
    router = _Router("I'm afraid I can't help with that.")
    finder = FlightFinder(router=router, copilot=copilot)
    search = _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                                 "depart": "2027-03-15"}))
    assert search.offers, "the deterministic reader should have caught this"


def test_a_model_that_raises_does_not_lose_the_search():
    class _Broken:
        async def generate_text(self, *_args, **_kwargs):
            raise RuntimeError("no provider is reachable")

    finder = FlightFinder(router=_Broken(), copilot=_Copilot())
    search = _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                                 "depart": "2027-03-15"}))
    assert search.offers, "an offline model should not cost us the page"


def test_no_browser_still_yields_a_usable_search():
    """The URL is a real answer even when nothing can be read."""
    finder = FlightFinder()
    search = _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                                 "depart": "2027-03-15"}))
    assert search.url and not search.offers
    assert search.note


def test_a_browser_that_fails_is_reported_not_invented():
    finder = FlightFinder(copilot=_Copilot(ok=False))
    search = _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                                 "depart": "2027-03-15"}))
    assert not search.offers
    assert "couldn't read" in search.note.lower()


def test_a_page_with_no_prices_says_so():
    finder = FlightFinder(copilot=_Copilot(page="No results found. Try again."))
    search = _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                                 "depart": "2027-03-15"}))
    assert not search.offers and search.note


def test_offers_reach_the_content_panel():
    published: list = []

    class _Signal:
        def emit(self, rows):
            published.append(rows)

    class _Bus:
        content_results = _Signal()
        log = _Signal()

    finder = FlightFinder(bus=_Bus(), copilot=_Copilot())
    _run(finder.search({"origin": "Birmingham", "destination": "Malaga",
                        "depart": "2027-03-15"}))
    assert published, "the offers never reached the panel"
    assert all(row.get("title") for row in published[-1])


# ── wiring ────────────────────────────────────────────────────────────────────

def test_the_tool_is_declared_and_routed():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    declared = {tool["name"] for tool in TOOL_DECLARATIONS}
    assert "flight_search" in declared

    source = (ROOT / "orion_core" / "dispatcher.py").read_text(encoding="utf-8")
    assert '"flight_search":' in source, "declared but unreachable"


def test_it_runs_one_at_a_time():
    """It drives the single shared browser; two at once fight over one tab."""
    from orion_core.concurrency import ToolClass, classify

    assert classify("flight_search", {}) is not ToolClass.PARALLEL


def test_asking_about_flights_finds_the_tool():
    """ORION has shipped a tool the resolver could not route to before — the
    words that matter have to be in the schema and vocabulary, not just in the
    implementation."""
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.tool_resolver import lexical_scores

    asks = [
        "find me a flight from birmingham to new york next friday",
        "how much is a plane ticket to dubai",
        "cheapest flights to malaga in march",
        "book me a return to JFK",
    ]
    for ask in asks:
        scores = lexical_scores(ask, TOOL_DECLARATIONS)
        best = max(scores.items(), key=lambda pair: pair[1])[0]
        assert best == "flight_search", f"{ask!r} routed to {best}"


def test_an_unrelated_question_does_not_route_here():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.tool_resolver import lexical_scores

    scores = lexical_scores("what's the weather in paris", TOOL_DECLARATIONS)
    best = max(scores.items(), key=lambda pair: pair[1])[0]
    assert best != "flight_search"


# ── live aircraft (aviation.py) ──────────────────────────────────────────────
#
# A different question from booking: what is in the sky NOW. The failures that
# matter are an old picture passed off as live, an identity field invented, the
# user's exact location sent to a third party, and a rate-limited feed hammered.
# Every provider here is an httpx.MockTransport; no test reaches the network.

import httpx  # noqa: E402

from orion_core import aviation as av  # noqa: E402

# Shapes copied from real adsb.lol replies (probed 2026-09-25): an airborne
# jet, a taxiing one, a tower transmitter and a ground service vehicle.
_ADSB_ROWS = [
    {"hex": "4070ea", "type": "adsb_icao", "flight": "EXS4TC  ", "r": "G-JZHW", "t": "B738",
     "alt_baro": 36000, "gs": 501.5, "track": 20.42, "baro_rate": -640,
     "lat": 50.60, "lon": -3.45, "seen_pos": 1.5, "category": "A3"},
    {"hex": "408121", "type": "adsb_icao", "flight": "MAINT   ", "r": "G-VPIE", "t": "A339",
     "alt_baro": "ground", "gs": 0.0, "true_heading": 180.0, "lat": 50.55, "lon": -3.50,
     "seen_pos": 4.7, "category": "A5"},
    {"hex": "42584c", "type": "adsb_icao_nt", "r": "TWR", "t": "TWR", "alt_baro": "ground",
     "lat": 50.56, "lon": -3.49, "seen_pos": 0.2},
    {"hex": "4258aa", "type": "adsb_icao_nt", "alt_baro": "ground", "category": "C2",
     "lat": 50.56, "lon": -3.48, "seen_pos": 0.2},
    {"hex": "43c001", "type": "mlat"},                                    # no position
]
_HOME = (50.5461, -3.4987)          # a made-up point, not anyone's address


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    async def sleep(self, seconds):
        self.t += seconds


def _service(handler, clock=None):
    clock = clock or _Clock()
    return av.AviationService(transport=httpx.MockTransport(handler), clock=clock,
                              sleep=clock.sleep), clock


def _adsb_ok(request):
    return httpx.Response(200, json={"ac": _ADSB_ROWS, "now": 0})


def test_aviation_adsb_lol_rows_are_normalised_to_si_without_invention():
    rows = av.parse_adsb_lol({"ac": _ADSB_ROWS, "now": 0}, received_at=5000.0)
    by_id = {a.icao24: a for a in rows}
    assert set(by_id) == {"4070ea", "408121"}           # tower, vehicle, no-fix dropped
    jet = by_id["4070ea"]
    assert jet.callsign == "EXS4TC" and jet.registration == "G-JZHW" and jet.type_code == "B738"
    assert jet.altitude_m == pytest.approx(36000 * 0.3048)
    assert jet.velocity_ms == pytest.approx(501.5 * 0.514444)
    assert jet.vertical_rate == pytest.approx(-640 * 0.00508)
    assert jet.last_contact == pytest.approx(5000.0 - 1.5)
    assert jet.origin_country == ""                       # adsb.lol never says; never guessed
    taxi = by_id["408121"]
    assert taxi.on_ground and taxi.altitude_m is None and taxi.heading_deg == 180.0


def test_aviation_opensky_states_parse_and_empty_box_is_not_an_error():
    assert av.parse_opensky({"time": 100, "states": None}, 200.0) == []
    state = ["3c6444", "DLH9LF  ", "Germany", 95, 99, 8.5, 50.1, 10972.8, False,
             232.0, 98.5, -1.3, None, 11277.6, "1000", False, 0]
    (ac,) = av.parse_opensky({"time": 100, "states": [state]}, received_at=200.0)
    assert (ac.callsign, ac.origin_country, ac.altitude_m) == ("DLH9LF", "Germany", 10972.8)
    assert ac.last_contact == pytest.approx(195.0)        # 5 s old, on OUR clock


def test_aviation_provider_is_told_a_grid_cell_not_the_precise_point():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return _adsb_ok(request)

    svc, _ = _service(handler)
    report = asyncio.run(svc.overhead(*_HOME, radius_km=25))
    # 25 km + 8 km rounding pad -> 40 km bucket -> 22 nm, around 50.5, -3.5.
    assert seen == ["https://api.adsb.lol/v2/lat/50.50/lon/-3.50/dist/22"]
    # ...while distances are still measured from the precise point, nearest first.
    assert report.ok and [c.aircraft.icao24 for c in report.contacts] == ["408121", "4070ea"]
    assert report.contacts[0].distance_km == pytest.approx(
        av.haversine_km(*_HOME, 50.55, -3.50), rel=1e-6)


def test_aviation_cache_makes_one_request_and_old_data_is_flagged_not_live():
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return _adsb_ok(request) if len(calls) == 1 else httpx.Response(503)

    svc, clock = _service(handler)
    first = asyncio.run(svc.nearby(*_HOME, 25))
    second = asyncio.run(svc.nearby(*_HOME, 10))          # smaller circle: same snapshot
    assert len(calls) == 1 and second.fetched_at == first.fetched_at
    clock.t += 45                                         # past every TTL: must refetch
    third = asyncio.run(svc.nearby(*_HOME, 25))
    assert third.stale and not third.ok and third.contacts
    assert "answered HTTP 503" in third.error
    text = av.summarise(third, where="Testville")
    assert text.startswith("I couldn't get a fresh picture") and "s old" in text


def test_aviation_rate_limit_backs_off_and_falls_back_with_the_real_reason():
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        if request.url.host == "api.adsb.lol":        # real shape: HTML, no Retry-After
            return httpx.Response(429, text="<html>429 Too Many Requests</html>")
        return httpx.Response(429, headers={"X-Rate-Limit-Retry-After-Seconds": "60"})

    svc, clock = _service(handler)
    report = asyncio.run(svc.overhead(*_HOME))
    assert hosts == ["api.adsb.lol", "opensky-network.org"]
    assert not report.ok and report.retry_in_s == pytest.approx(15.0)
    assert "adsb.lol is rate-limiting" in report.error
    assert "OpenSky is rate-limiting" in report.error
    assert "I've backed off and will retry in 15 s" in report.error
    clock.t += 5                                          # both still cooling: no request
    asyncio.run(svc.overhead(*_HOME))
    assert len(hosts) == 2
    assert svc.status()["OpenSky"]["cooling_s"] == pytest.approx(55.0)


def test_aviation_retry_after_header_forms():
    assert av.retry_after_seconds({"retry-after": "120"}) == 120.0
    assert av.retry_after_seconds({"x-rate-limit-retry-after-seconds": "999999"}) == 6 * 3600.0
    assert av.retry_after_seconds({}) is None


def test_aviation_with_no_known_location_asks_and_never_fetches():
    def handler(request):                                 # pragma: no cover - must not run
        raise AssertionError("fetched without a location")

    svc, _ = _service(handler)
    text, ok = asyncio.run(av.run_tool({"action": "overhead"}, service=svc))
    assert not ok and text == av.NO_LOCATION


class _Temporal:
    def coordinates(self):
        return _HOME

    def locality(self):
        return "Testville, Devon, UK"


def test_aviation_summary_speaks_only_supplied_fields():
    svc, _ = _service(_adsb_ok)
    text, ok = asyncio.run(av.run_tool({}, service=svc, temporal=_Temporal()))
    assert ok
    assert text.startswith("There is 1 aircraft in the air within 25 km of Testville, "
                           "plus 1 on the ground.")
    # 36,000 ft -> 10,973 m -> "11,000 m"; 501.5 kt -> 928.8 km/h.
    assert "EXS4TC (G-JZHW, a B738) at 11,000 m, heading 20°, 929 km/h" in text
    assert "registered in" not in text                    # the source didn't say
    assert "Positions from adsb.lol" in text


def test_aviation_show_opens_the_globe_and_turns_the_layer_on():
    svc, _ = _service(_adsb_ok)
    emitted = []

    class Sig:
        def __init__(self, name):
            self.name = name

        def emit(self, *payload):
            emitted.append((self.name, payload))

    class Bus:
        gui_command = Sig("gui_command")
        dashboard_event = Sig("dashboard_event")

    text, ok = asyncio.run(av.run_tool({"action": "show", "lat": 50.5, "lon": -3.5,
                                        "radius_km": 40}, service=svc, bus=Bus()))
    assert ok and svc.layer.on and svc.layer.radius_km == 40 and svc.layer.focus_seq == 1
    assert ("gui_command", ({"action": "globe", "target": ""},)) in emitted
    assert ("dashboard_event", ("aviation", {"action": "show"})) in emitted
    batch = av.globe_batch(asyncio.run(svc.nearby(50.5, -3.5, 40)), reset=True)
    assert batch["reset"] and len(batch["aircraft"]) == 2
    asyncio.run(av.run_tool({"action": "hide"}, service=svc, bus=Bus()))
    assert not svc.layer.on


def test_aviation_questions_route_to_aviation_not_flight_search():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.tool_resolver import lexical_scores

    for ask in ("what planes are flying over us right now",
                "what's that aircraft overhead",
                "show me live air traffic on the map"):
        scores = lexical_scores(ask, TOOL_DECLARATIONS)
        best = max(scores.items(), key=lambda pair: pair[1])[0]
        assert best == "aviation", f"{ask!r} routed to {best}"
