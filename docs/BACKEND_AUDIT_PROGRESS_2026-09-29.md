# ORION backend audit progress — 29 September 2026

The backend contains 399 Python files under `orion_core` and 454 test modules.
Pytest currently collects 8,080 cases under `tests/`; the full run also
includes tests in other repository directories.
The running remote gateway is `aiohttp.web`, rather than FastAPI/Uvicorn; the
network and shutdown checks in this audit target that actual implementation.
The source inventory checked 458 Python, Android and deployment files: no
Python syntax errors and no direct known blocking calls inside `async def`
bodies. It found 66 IPv4 literals, mostly loopback addresses. The external
probe addresses are in `orion_core/connectivity.py`; the `100.64.0.0/10`
entries are Tailscale policy ranges, not server bind addresses. Details are in
`BACKEND_SOURCE_INVENTORY_2026-09-28.json` (regenerated on this date).

Ruff's fatal/undefined-name checks across `orion_core` now pass. Its first
pass found type-only `aiohttp` references in `antigravity_research.py`; these
are now declared under `TYPE_CHECKING` without forcing the optional HTTP
library to load at import time.

## Confirmed backend faults corrected

`ConnectivityMonitor.__init__` called a three-host synchronous network probe
even when constructed inside the desktop or headless asyncio event loop. On a
broken connection it could block startup for several seconds. In a running
loop it now starts with an explicit *checking connectivity* state; the existing
background task probes in a worker thread. Outside a running loop, the
constructor retains its synchronous behaviour. Two regression tests cover
both cases.

`AntigravityResearch._request` previously allowed HTTP redirects while sending
the Gemini key in `x-goog-api-key`. A redirect could have sent that header to
another host. It now refuses redirects and treats any 3xx response as a
failure. A regression test checks the request option.

The opt-in proactivity engine launched a persistent task that desktop shutdown
neither cancelled nor awaited. The shutdown path now owns that task alongside
the other background workers. The existing diagnostics and housekeeping tasks
are also awaited after cancellation, avoiding pending tasks when the loop
closes.

The shared hosted-provider transport, model discovery and both Charon speech
requests could follow HTTP redirects while carrying API keys. They now refuse
redirects. A 3xx chat response fails explicitly and closes its connection;
model discovery returns no models, and speech reports provider unavailability.
Focused tests cover these boundaries without using a live credential.

`AIProviderProfile.is_local` used a string prefix to decide whether a model
server needed cloud credentials. An address such as
`http://127.0.0.1@outside.example` could be misclassified as local. It now
parses the URL and accepts only exact loopback hostnames over HTTP, without
embedded user credentials. Tests cover normal loopback and deceptive URLs.

The browser reading tool did an unused accessibility lookup before its actual
page scan. That extra desktop query has been removed, reducing latency for
phone-initiated page reading without changing the returned text.

The credential boundary pass found that Gemini embeddings, grounded search,
Brave search, HIBP breach checks and the Lichess bot API also accepted
redirects on requests carrying credentials. Those requests now refuse
redirects. HIBP account names are URL encoded; failures no longer log raw
request exception text. Official Google citation wrappers still resolve only
through recognised Google hosts.

Remote pairing and revocation no longer report success when the signing key
or device registry fails to persist. A corrupt existing key is preserved and
reported explicitly instead of silently being replaced. Revoked devices are
also denied cached screen data and pending device-action replies at once.
LAN hints now include only RFC 1918 addresses; Tailscale `100.64.0.0/10`
addresses are not presented as local Wi-Fi addresses.

The remote capability gate exposed a `use_tool` wrapper that could pass an
approved call through to a desktop-only `FORBID` tool. It now classifies and
executes the underlying tool, including saved approvals and multi-step plans.
MCP idle parking also waits for in-progress offloaded argument validation,
so a server cannot be stopped between validation and invocation.
MCP child processes no longer inherit short credential variable names such
as `DB_PASS` and `GITHUB_PAT`, or database connection URLs, unless a server's
configuration explicitly supplies them.

The fallback speech synthesiser silently cut every reply at 4,000 characters.
It now queues the complete text in bounded parts, and an interruption clears
all remaining parts. The Android app previously split speech at 1,800
characters although the host rejects requests over 1,200; version 3.0 uses
the host's limit. No live handset audio check was possible in this pass.

The desktop Study panel could grade a different card from the one revealed
because voice review shared its mutable cursor. Grading now names the visible
card explicitly. Chess evaluation now scores an immutable board copy and
ignores late scores once the displayed position changes.

Desktop text entry could report an unverified write as a successful tool
result when an editor could not be read back. It now returns a failed receipt
and tells the user to inspect the text, so a remote multi-step plan stops
instead of claiming the Notepad task is complete.

The Hostinger and Oracle setup scripts were reviewed for current headless
deployment. The Hostinger unit now advertises its public HTTPS origin in
reverse-proxy mode, and setup preserves existing firewall and environment
settings. The Oracle setup validates a Caddy candidate before replacing the
live file and preserves existing provider settings. These scripts pass Bash
syntax checks; systemd, Caddy and a live VM have not been exercised on this
Windows host. The standalone Twilio service has no conversational audio engine
and is documented as unavailable for live calls.

## Verification completed

| Area | Result |
| --- | --- |
| Remote, provider and phone integration | 35 passed |
| Startup, reliability, gateway and complete replies | 118 passed |
| Chess, cognition, tasks and desktop navigation | 137 passed |
| Voice and hosted provider behaviour | 67 passed |
| Security and MCP routes | 94 passed, 2 skipped |
| Antigravity request boundary and connectivity | 15 passed |
| Desktop lifecycle and opt-in proactivity | 54 passed |
| Hosted provider, Charon stream and redirect boundaries | 34 passed |
| Provider classification, rotation and local model memory | 84 passed, 1 skipped |
| Embeddings, search, breach and chess credential boundaries | 97 passed, 1 deselected |
| Remote gateway, pairing persistence and endpoint selection | 124 passed |
| MCP policy, idle handling and phone gateway tools | 54 passed |
| Long speech, remote voice and audio lifecycle | 22 passed |
| Cognition, chess engine/view, Tasks and Grandmaster | 161 passed |
| Desktop control, phone-to-PC task routing and approval | 107 passed |
| Final remote/provider/speech/shutdown cross-check | 19 passed |
| Full repository `pytest -q --maxfail=1` | 8,109 passed, 17 skipped, 7 subtests passed; 4 warnings; 10m 42s |

Android 3.0 (`versionCode` 22) was built with the installed signing identity:
49 unit tests passed, lint reported no errors and 80 warnings, and the signed
APK was verified with `apksigner`. Its SHA-256 is
`e99f4b7391710be60121cb694423984f95a9c4e0dace69fc7efc11c16b7e7632`.
The temporary installer-only HTTPS route returned this exact file; `/api/health`
on that route returned 404, confirming the gateway was not exposed through it.

After the user confirmed version 3.0 works with the PC off but reported a wait
before speech, Android 3.1 (`versionCode` 23) began queuing complete factual
answer phrases for Charon whilst later text is still generated. Commands that
may control the phone still wait for their action receipt. The Android build,
52 unit tests and lint passed. The signed APK SHA-256 is
`f13f09f0c51dc555d56d5b6b513e78c5ba09259d4f133e49c7547336336b0fb0`;
the installer-only HTTPS GET returned the same hash and `/api/health` remained
404. A real-phone timing check remains open.

Android 3.2 (`versionCode` 24) adds private phone timing diagnostics and stops
silently upscaling a selected Lite text model for long questions. It also
adds a reviewed battery and connection action, a capability panel, and
first-run entry for a customer's own Gemini key. The signed APK, 52 Android
unit tests and lint passed; its SHA-256 is
`851c42c6cb96e113f54f14436231f296b62dcfe15b8599b855fb2a86cfd66b0c`.
The installer-only HTTPS GET matched and `/api/health` remained 404. The
remaining customer release gates and physical phone tests are recorded in
`docs/ANDROID_FUTURE_READINESS_AUDIT_2026-09-29.md`.

The focused groups overlap, so their figures are test executions rather than
a count of unique cases. The full run completed after the code fixes and is
the repository-wide regression result. Its four warnings concern a Starlette
test-client deprecation, two PEFT test fixtures and Python's deprecated
`audioop` import in the Twilio bridge.

## Remaining scope

The syntax inventory and full suite are broad checks, not individual
behavioural proof for all 399 Python modules. Device-dependent calls,
external API credentials, an Oracle cloud instance and a live phone were not
available to this pass. Continue file-by-file review by domain, starting
with remote authentication, provider routing, MCP execution and startup
lifecycle, then the desktop tools and remaining utility modules.
