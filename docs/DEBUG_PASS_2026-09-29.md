# ORION debugging pass — 29 September 2026

## Faults corrected

- The phone terminal loaded and decrypted saved conversation history on the UI thread. It now reads history on its worker, discards stale loads, and renders only the latest result. Streamed answers update the display in short batches and follow the latest text only when the reader was already at the bottom.
- Fatal background-listening failures could leave the requested-listening flag set, causing repeated restarts. Fatal paths now clear it. Quota exhaustion waits for an available voice model instead of retrying every few seconds. Direct TTS honours its selected mode rather than silently switching to Live voice.
- MCP replies are ignored after cancellation or a configuration change. An active TLS call is cancelled off the UI thread. Reused remote command IDs are rejected within a connection, and the phone app picker handles a package-manager failure.
- The offline map discards stale or less accurate location fixes, clears its vehicle marker outside installed coverage, validates imported packs, and stops a download when Wi-Fi is lost. Rotation preserves the in-progress transfer. The Packs dialog now presents its selectable list.
- Saved empty hosted-provider settings could block a key added through the environment later. The merge now activates a later key for hosted profiles whilst preserving explicit saved credentials and disabled choices. Phone AI credential provisioning refuses cleartext requests from non-loopback LAN clients and rechecks pairing after approval I/O.

## Verification

| Check | Result |
| --- | --- |
| Android signed release build | ORION 3.7, version code 29; success |
| Android unit tests | 56 passed, 0 failed |
| Android lint | 0 errors, 108 warnings |
| Wireless handset install | Success; `PhoneConsoleActivity` resumed |
| Android fatal exceptions dated 29 September | 0 in the handset log after launch |
| Python full repository suite | 8,126 passed, 15 skipped, 7 subtests passed; 314 warnings; 9m 46s |
| Ruff fatal and undefined-name checks | Passed |
| Source inventory | 463 source/deployment files, 0 Python syntax errors, 0 direct known blocking calls inside `async def` bodies, 67 IPv4 literals |

The installed APK is `releases/android/orion-companion.apk`, SHA-256
`d7f74d00191ca2cffa8b27b89980f18cc887ec55de53167b4f396ddcffbc410a`.
It uses the existing Android signing identity so installation preserves phone
data and downloaded maps. The broad Python suite's warnings are principally
third-party deprecations; pytest could not update its cache under this
workspace's existing permissions. The Android warnings are advisory and
include old Gradle API usage and accessibility checks.

## Limits of this pass

- Automated tests and a launch check do not prove microphone pickup, Bluetooth playback, live provider quotas, hand-off between mobile networks, or every phone action on a physical device. Those need live interaction with the handset and relevant accounts.
- The offline packs provide a two-dimensional OpenStreetMap-derived road map and live location. They do not provide satellite imagery or in-app turn-by-turn routing. Routing currently opens OsmAnd when it is installed.
- The proposed cleaner phone layouts are review previews only. The installed 3.7 app contains functional debugging changes, not the unapproved visual redesign.
