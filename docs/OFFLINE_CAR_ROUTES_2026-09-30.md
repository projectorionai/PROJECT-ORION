# Offline car routes in ORION

## What is integrated

The native Android map now has **Places** and **Routes** controls. It keeps the existing Mapsforge UK road maps and moving GPS marker, and adds:

- Phone-private saved destinations with names and coordinates. Save the current GPS position, add coordinates, show a place, calculate a route or remove the place.
- Local car route calculation through BRouter's official Android service, using the `car-fast` profile. The map displays the returned road geometry and BRouter's actual GPX manoeuvre descriptions and offsets.
- Explicit planning between two supplied coordinate points without GPS. The planned start remains route input; it never replaces the measured phone position. Use **Routes → Plan between coordinates** whilst preparing a journey.
- Along-route distance, the next supplied instruction and its distance, an off-route indication and a destination-nearby indication. These use GPS and saved route geometry; no traffic estimate or invented arrival time is shown.
- Import of a connected GPX track, saved locally for reopening. Import works without BRouter. Waypoint-only files and disconnected segments are rejected because joining them would invent missing road geometry.
- Coordinate navigation requests from the phone assistant enter the same native route flow. They wait for a fresh GPS fix before routing. Named address searches still use OsmAnd because an offline address index has not been bundled.

## Set up before driving

1. Install BRouter from its official distribution or F-Droid, and open it once to finish setup.
2. Use BRouter's downloader to obtain every road region crossed by the journey. Its `.rd5` routing graph files are separate from ORION's Mapsforge `.map` files. Installing a map does not install a routing graph.
3. Ensure BRouter's `car-fast` profile is installed. ORION does not substitute a cycling profile if a car profile is missing.
4. In ORION, open **Maps → Routes → BRouter setup** to reopen BRouter. Then choose **Car route to coordinates**, or select a saved place and choose **Calculate car route**.
5. Turn on precise Location permission and GPS. Routing requires a fix less than 30 seconds old with reported accuracy within 100 metres.
6. After obtaining the route, switch off mobile data to check the saved map, route and moving position whilst safely stationary. GPS itself does not require mobile data, although obtaining a first fix can take longer without assisted data.

BRouter reads its selected base directory from its private `config15.dat` and uses `<base>/brouter/segments4/`, `<base>/brouter/profiles2/car-fast.brf` and `<base>/brouter/modes/`. Let its own setup choose a directory compatible with Android scoped storage. Do not assume that copying files to `/sdcard/brouter` will configure a current installation.

The service interface and storage arrangement are documented in the [official BRouter AIDL contract](https://github.com/abrensch/brouter/blob/master/brouter-routing-app/src/main/aidl/btools/routingapp/IBRouterService.aidl), [service implementation](https://github.com/abrensch/brouter/blob/master/brouter-routing-app/src/main/java/btools/routingapp/BRouterService.java) and [Android instructions](https://github.com/abrensch/brouter/blob/master/docs/users/android_quickstart.md).

## Scope and limits

ORION calculates and displays offline car routes **inside the app when BRouter and the required graph regions are installed**. It does not bundle a UK-wide car routing graph or download one automatically. The maps are vector roads and places, not satellite photography.

For this owner's wireless setup on 30 September, BRouter 1.7.9 was installed from the official F-Droid distribution. Its own setup created `Android/media/btools.routingapp/brouter`, including `profiles2/car-fast.brf`. Seven official routing cells covering the UK were downloaded on the PC and installed there: `W10_N45`, `W10_N50`, `W5_N50`, `E0_N50`, `W10_N55`, `W5_N55` and `W5_N60`. The 294,157,904-byte dataset was checked against the HTTPS response lengths and SHA-256 hashes of the PC and phone copies. These are transfer-integrity observations, not an independent guarantee of every road restriction. The application does not silently perform this one-off owner setup on customer phones.

Version 3.16 adds **Routes → Start guidance**. A user-started location foreground service follows GPS whilst the screen is locked, speaks supplied manoeuvres using a downloaded UK phone voice, and offers cautious automatic recalculation for the current BRouter car route. An explicit assistant coordinate-navigation request starts the same guidance after its route is calculated whilst the app is visible. Closing the map leaves an explicitly started journey running; its notification has a Stop action. The service does not restart after process termination or device reboot. Download an offline UK voice in Android's Text-to-speech settings; if none is installed, the map shows that speech is unavailable and keeps visual guidance. Directions never use a cloud key.

Rerouting requires three distinct accurate GPS fixes outside the route corridor over at least eight seconds. Attempts are at least sixty seconds apart, retain the original destination and cannot replace a newer saved journey. Imported and reopened GPX routes do not acquire BRouter provenance: their supplied turns can be read, but automatic car recalculation is disabled. After reopening a route, calculate it with BRouter again to enable recalculation. GPS loss, stale/poor fixes or revoked precise Location permission suspend or stop guidance. Native offline address search remains absent. **Routes → Spoken guidance · OsmAnd** remains available for its own address search and guidance after maps are downloaded. Road data and restrictions can be out of date; follow road signs and prepare the journey whilst parked.

Imported tracks may be walking or cycling tracks; importing GPX does not certify that it is suitable for a car. A plain track without validated instruction offsets is drawn without invented manoeuvres. The instruction reader follows BRouter's [GPX format implementation](https://github.com/abrensch/brouter/blob/master/brouter-core/src/main/java/btools/router/FormatGpx.java).

The phone must calculate at least one route using installed BRouter and real graph data before this integration can be described as verified on that device. No data coverage or locked-screen guidance is inferred from an engine installation.

## Reliability and privacy

- Routing runs on a dedicated worker, with a 30-second BRouter calculation budget and a 45-second UI deadline. Cancellation unbinds the service and suppresses late results. Android cannot forcibly cancel an outstanding synchronous Binder transaction; another call is refused whilst that transaction remains active.
- Returned GPX is compressed to reduce Binder payload size. The reply and expanded data have limits; import is capped at 4 MB, 50,000 track points, 4,096 instruction points and 32 XML levels.
- DTDs and entity declarations are rejected before parsing; external entity resolution is refused. Coordinates, instruction offsets, geometry agreement and UTF-8 are checked. Malformed files and BRouter error messages leave the existing map and route intact.
- Route matching prefers the nearby route section, reacquires the whole track if needed, and shows off-route status rather than treating a distant GPS fix as a valid road position. Poor or stale fixes suspend manoeuvre presentation. No ETA is inferred from straight-line distance.
- Places and the last route remain in app-private storage on the phone. They are not sent to an AI provider, MCP service or PC. Saved places are capped at 128. Writes use temporary files and replacement; a corrupt place store is not silently overwritten.
- Version 3.15 shares a per-file revision and lock across map screens. A superseded calculation or delayed clear cannot overwrite or erase a newer requested journey. Route restoration is bounded and cancelled/malformed writes preserve the last usable file.
- Version 3.16 reads route and revision atomically, binds guidance to that exact journey and reserves a reroute only if it remains current. GPS work and main-thread results use bounded latest slots. Announcements use actual route distances, retain deduplication across GPS jitter/backtracking, and do not replay passed turns. Old announcements expire or are discarded after movement; other ORION speech takes precedence. Navigation speech and its three-second acoustic tail are excluded from command capture even when echo cancellation is available.

## Verification

`OfflineRouteTest.kt` exercises genuine-style BRouter GPX, manoeuvre offsets, along-track progress, off-route fixes, duplicate points, antimeridian geometry, route reacquisition, disconnected or malformed tracks, untrusted XML, coordinate limits, compressed replies, decompression bounds and invalid UTF-8. `OfflinePlacesTest.kt` checks durable save/remove, capacity limits and preservation of corrupt stores. Combined Android test/build/lint results are recorded in the integration audit; real driving and voice-guidance behaviour require device checks.

`OfflineGuidanceTest.kt` adds twenty-seven cases for actual turn distances, near-turn offsets, deduplication, crossing continuity, conservative arrival, stale/invalid fixes, fresh watchdog repeats, sustained deviations and reroute cooldown. Route-store and audio-lease tests cover competing journeys and stale speech callbacks. These are source/unit checks; Android foreground lifecycle, real BRouter calculation, locked-screen audio and a physical journey remain device acceptance gates. See the [3.16 record](PHONE_NAVIGATION_VOICE_3_16_2026-09-30.md).

The service declaration/start policy follows Android's [foreground service types](https://developer.android.com/develop/background-work/services/fgs/service-types) and [background-start restrictions](https://developer.android.com/develop/background-work/services/fgs/restrictions-bg-start). It starts from the visible app with precise location already granted. Its local speech selects an installed, non-network `en-GB` voice through the [Android Voice contract](https://developer.android.com/reference/android/speech/tts/Voice), uses navigation audio attributes and releases transient audio focus after each announcement.

## Intent contract

Open `OfflineMapActivity` with `coordinate_query` in `latitude,longitude` form. Add the Boolean extra `route_to_coordinate=true` to request a native car route after a suitable GPS fix. Omitting the extra retains the existing coordinate-view behaviour. This activity remains unexported; external applications cannot issue arbitrary route or file operations through it.
