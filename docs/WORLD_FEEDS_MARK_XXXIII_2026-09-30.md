# World feeds in Mark XXXIII

The desktop Globe and `orion_world` MCP server use one implementation and one
local SQLite cache. The MCP server is enabled by the configuration migration;
an existing explicit disabled choice is retained. There are no credentials or
account registrations for these feeds.

## Sources and coverage

| Layer | Official source | What ORION returns |
| --- | --- | --- |
| Satellites | [CelesTrak GP data formats](https://celestrak.org/NORAD/documentation/gp-data-formats.php) | The space stations group in OMM JSON, with SGP4 predicted latitude, longitude and altitude, orbital epoch and prediction time. This is a bounded useful group, rather than a download of every satellite. |
| Earthquakes | [USGS GeoJSON summary feeds](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php) | Past-day earthquakes, source links, event time, magnitude, depth and catalogue review status. Regional detection coverage differs. The USGS `tsunami` flag is not converted into a tsunami warning. |
| Tsunami bulletins | [NOAA product retrieval](https://www.tsunami.gov/?page=productRetrieval) | NTWC and PTWC Atom bulletins, including information statements which can explicitly report no threat. These are not a globally comprehensive list of active warnings. A GeoRSS point, when supplied, is the bulletin's event location, not a threatened coastline. |
| UK weather warnings | [Met Office RSS guidance](https://weather.metoffice.gov.uk/guides/rss) | UK severe weather warning summaries, attribution and direct links to the full official warning. The source's content links remain available. A summary without geometry is displayed in the list rather than assigned an invented coordinate. |

### Fixed endpoints

```text
https://celestrak.org/NORAD/elements/gp.php?GROUP=STATIONS&FORMAT=JSON
https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson
https://www.tsunami.gov/events/xml/PAAQAtom.xml
https://www.tsunami.gov/events/xml/PHEBAtom.xml
https://www.metoffice.gov.uk/public/data/PWSCache/WarningsRSS/Region/UK
```

The public RSS endpoint is used for UK weather warnings. ORION does not
silently register for the separate Weather DataHub product. The Met Office
[RSS terms](https://weather.metoffice.gov.uk/guides/rss) require attribution
and a functional direct link to the corresponding content page.

## Satellite calculation

Install the optional integration dependency with the project's integrations
requirements, or install `sgp4>=2.25,<3` into the same Python runtime as ORION.
The reviewed runtime currently has `sgp4` 2.27. Missing SGP4 reports an
unavailable prediction; it does not manufacture orbital positions.

The [upstream SGP4 OMM implementation](https://github.com/brandon-rhodes/python-sgp4/blob/master/sgp4/omm.py)
initialises the model from OMM fields. ORION propagates in TEME, rotates using
Vallado's GMST expression, then converts to WGS84 geodetic coordinates.
UTC approximates UT1; polar motion is not applied. This is a visualisation,
not a precision collision avoidance or flight operation calculation.
Orbital elements older than seven days, failed propagations and below-ground
positions are omitted with a visible diagnostic. Every successful position
is explicitly labelled `sgp4_prediction`.

## Request policy and failure handling

- CelesTrak orbital downloads have a persistent minimum interval of two hours,
  including across process restarts and the GUI/MCP boundary. This follows
  [CelesTrak's usage policy](https://celestrak.org/usage-policy.php).
- Other sources have a two-minute minimum interval. All downloads are on
  demand; importing the service or discovering MCP tools starts no download.
- A SQLite transaction reserves the next request time before making a
  connection. Concurrent consumers cannot independently request the same
  dataset. A cancelled request retains its reservation.
- Failed requests use exponential backoff. HTTP 429 honours `Retry-After`;
  redirects, access refusals and discontinued endpoints wait at least six
  hours. ORION never follows a redirect to a caller-selected destination.
- Each request has a ten-second total timeout, a four-second connection
  timeout, a five-second read timeout and a two-MiB response ceiling.
- XML declarations and entities are refused; JSON non-finite numbers,
  malformed shapes, excessive record counts and invalid coordinates fail
  validation. Official content links are HTTPS and restricted to the source
  domains. The MCP interface accepts no URL.
- Last successful data survives a failure and is labelled stale. A failed or
  uninitialised source is never presented as an empty healthy feed. A genuine
  successful empty feed has `status: empty`.
- `close()` cancels pending downloads, awaits their cancellation, closes the
  HTTP session and closes every database connection. Application shutdown
  calls `shutdown_world_feeds()` after stopping the MCP host.

## Service interface

```python
from orion_core.world_feeds import get_world_feeds, shutdown_world_feeds

snapshot = await get_world_feeds().snapshot(
    ("satellites", "earthquakes", "tsunami", "weather"), refresh=True
)
# refresh=False reads cached data and calculates satellite positions without
# starting a download. There is no force-refresh bypass of provider deadlines.
await shutdown_world_feeds()
```

The result contains `generated_at` and `layers`. Each layer contains `items`,
`fetched_at`, `source`, `source_url`, `source_urls`, `coverage`, `status`,
`stale`, `error` and `next_refresh_at`. Times are ISO UTC. Layer status is one
of `healthy`, `empty`, `stale`, `unavailable` or `loading`.

The five read-only MCP tools are `world_satellites`, `world_earthquakes`,
`world_tsunami`, `world_weather` and `world_feed_status`. Feed tools accept
only an optional integer `limit` from 1 to 200, default 50. A feed result
includes `record_count` and `truncated`. The status tool reads the cache
without downloading. An unavailable source has an MCP error flag as well as
the structured source diagnostic; partial or stale data remains visible.

## Verification

Fixtures exercise source formats, finite coordinate validation, hostile
links, entity declarations, item/byte limits, SGP4 movement and epoch age,
known J2000 Earth rotation, genuine empty feeds, unavailable/stale data,
concurrent cache consumers, process restart deadlines, provider backoff,
request cancellation and database handle closure. MCP tests exercise
handshakes, discovery, read-only annotations, bounded structured results and
configuration migration without re-enabling a previously disabled server.

Live source reachability is recorded separately in the project audit. An
HTTP refusal or provider outage remains a source failure even when all
fixture tests pass.
