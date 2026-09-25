# ORION Command Deck and backend audit — 25 September 2026

## Architecture found

ORION's Window 2 is `UnifiedDashboard`, a PyQt6 `QMainWindow` containing native widgets. Its cross component telemetry flows through `OrionBus` Qt signals. The phone gateway and Twilio bridge use aiohttp. There is no FastAPI app or browser DOM behind Window 2, so adding FastAPI dependency injection, Pydantic request models, or a new SSE transport to the desktop would introduce parallel infrastructure without fixing its actual data path. The remote phone already has an authenticated SSE stream with a 20 second keepalive and bounded subscriber queues.

The existing palette lives in the application's Qt style sheet. The redesign adds no colour literals and reuses its `workspaceTitle`, `panelFrame`, `panelHeading`, `mutedLabel`, and `deckTab` selectors. The separate face renderer and globe still have their own graphics requirements; the deprecated Window 2 Brain renderer was removed.

## 2D Command Deck options

| Path | Component and layout logic | State and data flow | Styling approach |
| --- | --- | --- | --- |
| **A. SVG node topology** | A `QGraphicsView`/`QPainter` scene with selectable nodes, explicit keyboard focus, pan and zoom, and compact detail cards. A browser version would use `<svg><g class="nodes">…</g></svg>` and an adjacent `<aside>` for the selected node. | Keep node positions and selection in the view; feed route activity from `OrionBus`; update only affected paths and detail fields. | Use current Qt palette values for pens and brushes. A browser port would map existing CSS variables to path stroke, glow and focus states. Avoid full scene animation when idle. |
| **B. Bento dashboard** | A `QGridLayout` with independent status, health, route, task, log and latency cards. A browser version would use a `<main class="dashboard-grid">` of semantic `<section>` elements. | Each card subscribes to its relevant bus signal, holds a small bounded history, and changes only its own labels or graph. | Reuse the present QSS cards and typography; a browser port would use the existing CSS variables with Grid/Flexbox and short opacity transitions. |
| **C. Telemetry and navigation split pane — implemented** | A `QSplitter` holds a persistent, scrollable zone and page sidebar on the left and the existing `QStackedWidget` on the right. The Overview page contains state, Live channel, host load, latest route, recent activity and direct links. A browser equivalent would be `<aside><nav>…</nav></aside><main><section>…</section></main>`. | The stack stays the single source of page identity. The overview subscribes to Qt signals; a state, connection, sample or tool event updates only its target label. Activity is capped at five entries. Existing pages remain lazy where already configured. | All controls use existing QSS object names and palette. There is no new WebEngine instance or Three.js canvas in Window 2. |

Option C fits the current desktop architecture and gives every page a stable, labelled target. The internal `BRAIN` route key remains for saved settings and voice commands; its visible label is **Overview**.

## Findings and changes

| Area | Finding | Change and verification |
| --- | --- | --- |
| Live speech | A cancelled Live response carrying audio and `turn_complete` in one event skipped completion, leaving output muted for later turns. | The receive loop discards that audio but processes completion, clears the pending turn and resets the mute. A two response regression test confirms the next turn plays. |
| Long research | Browser reading and paper generation could occupy a Live tool turn for minutes, outliving the voice connection and giving no timely spoken acknowledgement. | Research detaches by default, returns an immediate acknowledgement, keeps progress in the Research console, and announces completion over `speak_request`. Explicit `wait` remains available. |
| Provider routing | Unknown 429 reset times imposed a flat five minute pause. | Bounded exponential cooldown now grows from 5 to 300 seconds; a successful response resets the streak. Provider supplied retry delays and daily limits retain their specialised handling. Existing fallback and aiohttp transport timeouts remain in place. |
| Remote approvals | A confirmation token was broadcast to every paired phone; approval resolution did not verify the owning device. | Approval events are sent to the owner's SSE stream and resolution checks device identity. Remote task lookup also checks ownership. |
| Shared request state | Concurrent remote chats could overwrite a shared current device before a gated tool call. | A `ContextVar` scopes device identity to each request. |
| Remote event loop | Endpoint discovery and memory writes could block aiohttp; non object JSON raised server errors. | Endpoint discovery is cached behind an async lock and runs in a thread; chat memory and grounding work are offloaded; request bodies must be JSON objects. Auth rate buckets now evict stale entries and use the socket peer rather than client supplied `X-Forwarded-For`. |
| Twilio media | The media WebSocket lacked signature validation; overlapping calls could overwrite the engine's per call tool gate. | The upgrade verifies the Twilio signature, keeps a heartbeat, permits one active media call and releases its lock on every exit path. |
| Window 2 | The Brain page created a WebEngine and Three.js renderer merely to navigate pages. | The Brain renderer and its renderer specific tests were removed. The Overview and persistent sidebar use native Qt widgets and per signal updates. |

## Data flow

`Live worker / dispatcher / system metrics → OrionBus Qt signals → individual Overview labels and existing deck pages` is the desktop path. It needs no localhost WebSocket. The phone path is `aiohttp → authenticated SSE → bounded per client queue → phone`, with a keepalive frame every 20 seconds and queue removal on disconnect. Approval events are scoped to the paired device.

## What the Gemini result establishes

The local regression reproduces one concrete cause of subsequent silent turns and verifies its fix. Separate tests verify that long research returns control to the Live session while work continues. These are offline tests; a real long running Gemini Live conversation and the user's microphone and speaker hardware have not been exercised here. The [official Live API session guidance](https://ai.google.dev/gemini-api/docs/live-api/session-management) describes finite connection lifetime, GoAway and session resumption, which ORION already handles, but there is no basis to claim a vendor side bug has been fixed. The [tool guidance](https://ai.google.dev/gemini-api/docs/live-api/tools) says the client must return tool responses to the session. Capture connection state, turn IDs, tool duration, audio queue depth and the last `turn_complete` on a real reproduction before closing the issue fully.

## Remaining priorities

1. Run an end to end soak test with long spoken prompts and completed research on the user's actual Gemini account and audio devices. Verify reconnect and session resumption during playback.
2. Add a bounded execution policy for every Live tool that may wait on a browser, model or desktop task, not only research. A tool call still holds the receive loop while it runs.
3. Standardise strict request schemas across the aiohttp gateway and add size limits for every endpoint. The current object check prevents top level type errors but is not a complete schema layer.
4. Introduce a supervised shared HTTP client pool for provider calls when profiling proves connection creation is significant. Keep per provider timeout budgets and isolate malformed responses.
5. Retire old `swarm` and `BRAIN` internal names through a migration after confirming saved routes and spoken aliases, and run a visual QA pass on the user's display scaling and fonts.

## Verification

The final focused voice, provider, remote, telephony and deck regression set passed: **174 tests**. A full run collected roughly 6,800 tests and reached 94% without an assertion failure, then Windows terminated the Python process with native status `0xC0000409`. The final test file group passed separately (**391 tests**). The suite therefore has no clean single process completion in this environment; the crash needs its own native Qt/offscreen investigation. No external provider call, physical audio session or real Twilio call was made.

The working tree contained substantial unrelated changes before this audit; these were preserved.
