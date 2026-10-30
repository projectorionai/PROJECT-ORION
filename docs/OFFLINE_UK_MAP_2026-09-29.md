# ORION phone offline UK map — 29 September 2026

## Installed state

- Android package: `com.orion.companion`, version 3.6 (`versionCode 28`).
- Final installed APK SHA-256:
  `33723DAF1C206659911B7481909A70EB6C29609606BD170BEFB4786FA6AFC7C6`.
- The signed APK was installed over the existing phone app through wireless
  Android debugging. The existing phone data remained in place.
- Both Mapsforge packs were transferred to the app's external `maps/`
  directory. The on-phone SHA-256 hashes matched the downloaded originals:

| Pack | Bytes | SHA-256 |
| --- | ---: | --- |
| Great Britain | 1,647,208,232 | `23C80F8464A30DB90BD9E0A8AB94079B52367312A8F6C211903E908A89C10338` |
| Ireland and Northern Ireland | 338,508,734 | `66C068BA52A4CE0401A43A8E12D930E455C860EAA1950DEA36AE859897297158` |

The app opened both maps and rendered road-level detail offline. It displayed
a live phone position with an accuracy estimate, and Follow mode centred the
map on that position. Movement, speed and heading were not road-tested.
The main screen's Listen, Offline map and Phone tools controls were opened on
the phone; Listen started the microphone foreground service.

## Scope

The installed maps are OSM-derived **vector road maps**, not satellite
photographs. The screen supports pan, zoom, foreground live position and
country-wide map display. A supplied coordinate changes the view centre;
it never replaces the measured phone position. The in-app map does not yet
calculate car routes, search addresses offline or speak turn directions.
The route control opens OsmAnd for that work, using OsmAnd's own downloaded
data. A separate route graph and routing engine are needed before those
features can move into ORION itself.

Detailed, country-wide offline satellite imagery needs a licensed source and
far more storage than the road maps. No Google or OSM online tiles are cached
for offline use. A future imagery layer should accept an explicitly licensed
local raster pack, verify its coverage and checksum, and keep road labels
visible over the image.

## Next capabilities worth adding

1. Offline address and point-of-interest search with a local search index.
2. An in-app car routing engine and downloadable routing graph, including
   rerouting, spoken turns and route progress.
3. Optional licensed satellite packs by region, with versioned downloads and
   clear storage controls.
4. GPX trip recording and export, saved places and home/work shortcuts.
5. Road-level speed-limit data with source date and confidence; never infer
   legal speed from GPS alone.

Map data © OpenStreetMap contributors, ODbL. Sources:
[Mapsforge map downloads](https://download.mapsforge.org/maps/v5/europe/),
[OpenStreetMap copyright](https://www.openstreetmap.org/copyright).
