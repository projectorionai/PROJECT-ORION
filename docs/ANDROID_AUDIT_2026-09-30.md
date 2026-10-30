# ORION phone audit — 30 September 2026

## Scope and verified changes

This pass inspected the Android manifest, console and options navigation, screenshot path, phone AI request construction, MCP discovery path, voice performance records, offline map renderer, map pack manager and OsmAnd hand-off. It also inspected the current phone installation through wireless ADB. It is a bounded audit; it does not prove that every phone feature works in every Android state, network or third-party app.

| Area | Finding | Action and evidence |
| --- | --- | --- |
| Screenshots | `PhoneConsoleActivity` set `FLAG_SECURE`, blocking ordinary Android screenshots. | Removed the flag. Added **Options → Phone → Save screenshot**, which captures only the ORION window and lets the user choose a PNG destination. Temporary capture files are deleted after saving or cancellation. |
| AI request size | Up to 128,000 history characters and 20 entries could accompany every request, even for a short follow-up. | Capped the sent conversation at 32,000 characters and 16 entries, retaining the current request and the end of a long preceding answer. This reduces the maximum input size; a real latency improvement is not claimed until measured on the phone. |
| Offline position map | The Mapsforge view shows roads, live GPS position, heading and follow mode, but contains no route graph, route calculation or spoken manoeuvres. | The app now labels it **position only** and explains the OsmAnd hand-off. Both ORION map packs are present on the connected phone: Great Britain (about 1.5 GB) and Ireland/Northern Ireland (about 323 MB). The app validates each pack before using it. |
| Offline driving directions | `navigate` opens OsmAnd. ORION cannot assert that OsmAnd has enough routing data for a requested journey. | The connected phone has an OsmAnd South West England regional map and world basemap. Full-UK offline turn-by-turn navigation is therefore **not ready** on this phone. Download the regions for the whole planned route in OsmAnd while on Wi-Fi, then test a route with mobile data off before driving. |
| MCP latency | A configured MCP server can add up to six seconds to a first AI reply when tool discovery runs or its cache expires. | Recorded as an open performance issue. The current cache lasts five minutes and failed discovery has a 30-second cooldown. A later change should pre-warm discovery outside the reply path and measure cold versus warm replies. |
| Charon latency | A prior physical-phone sample recorded 3.263 seconds from voice start to first audio in Live mode; direct TTS previously started in about 1.352 seconds for a short typed reply. | No voice engine was changed in this pass. Performance diagnostics now include sent conversation characters so future measurements can separate prompt size from text and speech delays. |
| Native actions | Call, SMS, e-mail, navigation and app-opening receipts describe what Android accepted; several hand-offs still need the user to finish an action in another app. | Kept action receipts precise. No third-party delivery or route completion is claimed without observation. |

## Verification

- Android release build completed on the local toolchain.
- All 63 Android JVM unit tests passed.
- Android lint completed with 0 errors and 113 warnings. Most warnings concern Kotlin style and text internationalisation; the notable remaining warnings cover fixed orientation on legacy screens, JavaScript in the locally intercepted Three.js avatar WebView, and an application-context singleton reported as a possible static field leak. These warnings require separate review before a customer release.
- Version 3.10 (`versionCode` 32) was installed over the existing app via wireless ADB; the installed package reports the expected version.
- The staged APK is 2,268,564 bytes with SHA-256 `8877b2aa356bdfe566a08e82407feef2684bd1079df9a190a4f752c4977336bc`.
- The PNG capture flow has not been exercised end to end on the physical phone during this pass because another application was in the foreground. The compiled capture code and normal Android screenshot flag are verified, but the user should confirm the document picker and resulting PNG.

## Highest-value next additions

1. **Route readiness check:** before car travel, show which regional routing maps are installed, whether the planned journey is covered, free storage, GPS permission and last GPS fix. This can prevent an apparently ready map from failing mid-journey.
2. **Driver view:** large route controls, minimal text, spoken route state and interruption handling. It must depend on a verified navigation engine and avoid encouraging screen interaction while driving.
3. **MCP pre-warming:** discover configured tools in a background worker on connection or app start; show connection health separately from AI generation and measure cold and warm request times.
4. **Fast local intents:** answer safe device-state questions and open known installed apps without a cloud reasoning round trip when the wording is unambiguous; keep sensitive sends and account actions under native review.
5. **On-device resilience:** queue only safe drafts and reminders during a network outage, with explicit status and no false claim that a message, e-mail or PC action completed.
6. **Customer readiness:** separate user-owned API credentials, clear quota and model capability checks, an exportable diagnostic bundle with secrets removed, and upgrade tests for Android Keystore and revoked permissions.

These are proposals, not capabilities claimed present in 3.10.
