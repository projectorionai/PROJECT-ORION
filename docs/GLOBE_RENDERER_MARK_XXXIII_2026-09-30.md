# Globe renderer — Mark XXXIII

## Scope and chronology

This stage follows the earlier Mark XXXIII fixes and source commit `e7a5f50`.
It extends the existing CesiumJS page in `orion_core/gui/globe.py`. The current
repository has no separate `Globe.js` implementation. The user set aside the
wider Command Deck redesign on 30 September; it is not awaiting a selection.
The subsequent [coordinate accuracy phase](GLOBE_ACCURACY_2026-09-30.md) repairs
target centring, ambiguous search, source provenance and map correlation.

## Available controls

Open **GLOBE**, then select **Satellites**, **Earthquakes**, **Tsunami bulletins**
or **UK weather warnings**. Each selected layer requests an official report
snapshot through the shared world feed service. **Refresh** updates the
snapshot; provider cache intervals and retry limits still apply. Satellite
positions are recalculated from cached orbital elements without downloading
those elements on each UI refresh.

Click a plotted point or a report card for its source, coverage, event time,
retrieval time and details. Lists use eight cards per page, with **Previous**
and **Next**, so reports without coordinates remain accessible. The map admits
64 satellite predictions, 250 earthquake records, 200 tsunami bulletins and
200 weather warning records. If a source supplies more records, its count
shows the displayed subset against the source total. Duplicate identifiers
within a layer are represented once; identifiers are namespaced by layer.

Existing place search, postcode resolution, voice zoom, bounded maximum zoom,
map cards, news, footage and aviation remain connected. Reset cancels an
in-flight place/news request before resetting the camera. Source links open
through the existing system-browser interceptor. They do not depend on a
popup handler which the embedded Qt page does not implement.

## Rendering and safety

- Static reports use one `PointPrimitiveCollection` per layer. Positions are
  changed in a batch, followed by an explicit render request.
- No continuous idle orbit is started. Cesium's existing rendering and tile
  budgets remain in place. Hiding the tab disables its render loop and stops
  the GUI's feed timers and owned feed requests.
- Rebuilding after a graphics context loss removes event handlers, timers and
  pick handlers before creating a replacement viewer. Closing the widget or
  app calls the idempotent `GlobeView.shutdown()` method.
- Page generations and selection snapshots reject responses which arrive
  after a reload, hide, close or layer change. A newer place request cancels
  and supersedes the old one.
- Public strings are inserted through `textContent`, including news and
  event details. Links require HTTPS and reject embedded credentials and
  script/data URLs. The page-to-Python bridge accepts only named read-only
  layers, refresh and camera reset; it accepts no network URL or arbitrary
  command.
- Bulletins without official coordinates remain in the list. Coordinates
  embedded in prose are not inferred as either an epicentre or an affected
  coastline. Provider errors are distinct from an empty healthy feed and
  from a cached report whose next update is due.
- Official imagery attribution remains visible with Cesium's credit display.
  The Esri service metadata checked on 30 September 2026 names Vantor,
  Earthstar Geographics and the GIS User Community for imagery; labels name
  Esri, HERE, Garmin, OpenStreetMap contributors and the GIS user community.
  OpenStreetMap fallback retains its own provider attribution.

The style keeps the existing page layout, with grey, crimson, black and white
controls and keyboard-operable buttons. The cyan control rail and redundant
gesture instructions have been removed.

## Verification

The final focused gate passed **30 tests**, including execution of the actual
embedded JavaScript in isolated Chromium, plus the existing camera/aviation
tests. It checks repeated batches, point caps, duplicate identifiers, unsafe
links, markup injection, absent renderer, stale reports, satellite prediction
timestamps, hidden rendering, teardown, Reset ownership and navigation races.
It also individually opens all 25 unlocated bulletin records in a paginated
fixture, with at most eight cards present for that layer at once.

A separate run loaded the real Cesium **1.115** engine and public Esri imagery
with a cached snapshot from the live official feed diagnostic. It produced:

| Observation | Result |
| --- | --- |
| Reports admitted | 246 |
| Plotted points | 244 |
| Imagery tiles loaded | Yes |
| Page JavaScript errors | 0 |
| Recorded console errors | 0 |
| Rendered frames in three settled idle seconds | 1 |
| Rendered frames in three hidden seconds | 0 |
| Horizontal overflow at 1280, 736 and 320 px | None |
| Explicit renderer teardown | Complete |

Evidence is saved locally under `reports/mark_xxxiii_2026_09_30/`:
`globe-renderer-live.json`, `globe-world-live.png`, `globe-bulletin-live.png`
and `world-renderer-snapshot.json`. These generated reports remain excluded
from source publication. Cached source records are labelled as saved if their
normal refresh interval has elapsed. Satellite coordinates are predictions
for the snapshot time, rather than an observation or continuous live track.

The headless browser check verifies rendering and lifecycle behaviour. It
does not establish a crash-free result on every graphics driver or replace
hands-on checking of the Qt desktop page. Public CDN/imagery availability and
provider coverage remain external dependencies. The desktop Globe is not the
phone's offline navigation map.

## Primary references

- [Cesium explicit render behaviour](https://cesium.com/learn/cesiumjs/ref-doc/Scene.html#requestRenderMode)
- [Cesium Viewer credit container and render-loop ownership](https://cesium.com/learn/cesiumjs/ref-doc/Viewer.html)
- [Cesium point collection batching and destruction](https://cesium.com/learn/cesiumjs/ref-doc/PointPrimitiveCollection.html)
- [Esri World Imagery metadata](https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer?f=pjson)
- [Esri label service metadata](https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer?f=pjson)

See `WORLD_FEEDS_MARK_XXXIII_2026-09-30.md` for provider coverage, source
licensing and the read-only MCP tools used by ORION.
