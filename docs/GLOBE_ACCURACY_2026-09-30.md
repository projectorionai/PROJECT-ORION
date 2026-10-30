# Globe location accuracy — Mark XXXIII

This phase follows local source commit `903e034`. It repairs the existing
desktop Globe and adds a safe Lichess setup helper. It does not reinstate the
cancelled Command Deck redesign, deploy the prepared cloud service or alter
the phone during its deferred acceptance period.

## How to navigate

Open **GLOBE** and enter a place, full address, UK postcode or decimal WGS84
latitude and longitude. Examples of supported coordinate formats:

```text
51.507400123, -0.127800987
51.5074 N, 0.1278 W
longitude: -0.1278, latitude: 51.5074
geo:51.5074,-0.1278
```

Coordinate input bypasses geocoding and news requests. Full precision is
retained internally; the visible readout uses six decimal places. Invalid
coordinate input produces an error without being sent to a place search.
Latitude must be within ±90° and longitude within ±180°. Non-finite values,
contradictory hemisphere signs, altitude and unsupported URI parameters are
refused. Decimal boundary checks run before float conversion.

Place searches request up to six matches. Where several locations match, the
native **Matching locations** dropdown shows full addresses and coordinates.
Choose a result before the camera moves. Voice location requests also report
ambiguity rather than announcing that the first result is the intended town.
A new query or Reset clears these choices; a replaced renderer rejects old
choices and responses.

## Bugs corrected

| Finding | Correction |
| --- | --- |
| Camera flew towards latitude minus 0.12° while its marker used the original point | Centre the actual WGS84 point using Cesium's local target frame and a vertical view; cancel earlier camera flights |
| Readout rounded to three decimal places | Display six decimals while preserving the supplied float internally |
| Search silently accepted the first same-named place | Require an explicit choice among bounded, fully labelled results |
| Google map card geocoded the raw name independently | Embed and open the same resolved latitude/longitude as the Globe |
| Missing cached coordinates defaulted to `(0, 0)` | Refuse malformed cache/provider records; refetch an entirely invalid cache |
| Reverse-address cache rounded to four decimals | Match the six-decimal request precision so nearby points do not share a coarser cache key |
| Older completed requests remained queued while the page loaded | Remove their navigation JavaScript when a newer request or Reset takes ownership |
| Coordinate requests waited for unrelated place/news services | Resolve immediately and publish the selection before optional news |
| A requested address could fall back to a town gazetteer | Refuse that coarse fallback for recognised postcode/street/premises queries |
| Focused world reports could leave another place's marker and map visible | Clear the previous selection, centre the supplied source point and correlate its map; unlocated bulletins acquire no marker |

Source provenance now survives cache round trips. Legacy records without
provenance are labelled as saved geocoder results rather than being assigned
an invented provider.

## What accuracy means

The renderer can centre a supplied position precisely. That does not establish
the accuracy of a geocoder, satellite prediction, earthquake solution, GPS
sensor or imagery tile. The source label distinguishes supplied coordinates,
postcode representative points, mapped object reference points and settlement
centres. A UK postcode point is not an individual premises or a moving car.
No numerical source-error radius is invented when a provider supplies none.

The existing CelesTrak, USGS, NOAA and Met Office integrations remain active
with their documented coverage, caps, freshness and read-only World MCP tools.
Satellite positions are bounded space-station predictions for a snapshot,
not an observed global catalogue or continuous tracking service. Bulletins
without official coordinates remain unlocated. Existing aviation and zoom
controls remain connected. See [World feeds](WORLD_FEEDS_MARK_XXXIII_2026-09-30.md).

The desktop Globe requires online engine assets and imagery. It is separate
from the phone's BRouter/offline route work. No phone installation or driving
test was performed in this phase.

## Verification

The focused regression gate covers parsing, source/cache validation, voice
lookup, cancellation, native Qt result selection, camera controls and execution
of the actual embedded JavaScript in isolated Chromium. Browser fixtures block
all network requests; the native chooser runs in its own process to preserve
the main suite's Qt lifecycle. Detailed final counts are recorded in the
chronological audit and the ignored local XML evidence. The combined final
Globe/BOT setup gate passed **199 tests**, with zero failures or skips, in
49.78 seconds. The account helper's 25 tests use mocked requests only.

A separate check used the real pinned **Cesium 1.115** engine and the same CDN
document origin as the Qt page. Imagery and account requests were blocked.
It measured London, Sydney, both poles and both sides of the date line:

- Marker distance from the supplied Cartesian target: **0 metres** in all six cases.
- Largest projected camera-centre error: **0.003247 pixels** at a 1280 × 900 viewport.
- Camera range: **3,000 metres**, subject only to floating-point error.
- Page JavaScript errors: **0**.

These measurements verify renderer maths against test coordinates; they do
not certify real-world imagery or source accuracy. Earlier harness attempts
were corrected for the actual CDN allowlist and document origin before this
successful check. Evidence: `reports/mark_xxxiii_2026_09_30/globe_navigation_real_engine.json`.

## Lichess setup

See [the numbered account guide](CHESS_LICHESS_BOT.md). The new helper verifies
an account by default. `-Upgrade` requires the exact authenticated username
after a permanent-conversion warning, then checks the same account's BOT title.
It does not save credentials or convert an account during automated tests.
Account and live rated-game acceptance still require the user's setup.

## Useful further improvements

These are proposals, not installed capabilities:

1. Saved Globe locations and a measurement tool for distance/bearing.
2. Optional selected-satellite following with an explicit prediction clock.
3. Official warning boundaries or source uncertainty overlays where supplied.
4. A repeatable measured chess strength suite alongside actual Lichess ratings.
5. Resume installed-device voice timings and offline driving acceptance when
   the user is home.

## Primary references

- [Cesium target-centred camera flights](https://cesium.com/learn/cesiumjs/ref-doc/Camera.html#flyToBoundingSphere)
- [Lichess official BOT conversion specification](https://github.com/lichess-org/api/blob/master/doc/specs/tags/bot/api-bot-account-upgrade.yaml)
