# O.R.I.O.N. — Capabilities as of 18 September 2026

**Open Resolution Intelligence Overt Network**  
**Written against:** 25.0.0, “Mark XXV — Presence & Focus”  
**Current source:** 32.0.0, “Mark XXXII — Iris & Signal Lanes”. See the README
and `orion_core/__init__.py` for the current release identity; the inventory
below remains a dated snapshot.
**Scope:** the current source tree, including later additions whose individual
modules use Mark XXVI development labels. This is a dated feature inventory,
not a claim that every integration has been tested on every machine.

## Overview

ORION is a modular personal desktop assistant built with Python, PyQt6 and
asyncio/qasync. Its interface combines a visual assistant, voice/text
interaction, a command dashboard, persistent local memory and tools for
research, productivity and computer control. Windows is the primary desktop
platform; an optional headless service supports remote clients.

The source declares **138 model-callable tools** in
`orion_core/dispatch_schema.py` and wires **26 dashboard pages** in
`orion_core/app.py`. These are registered interfaces, not a count of independently
verified integrations. Plugins can extend the tool surface. Hardware, provider
configuration and optional dependencies determine availability.

## Camera and electronics inspection

The electronics workbench provides a camera-led workflow for PCBs and electronic
assemblies:

1. Start the selected camera and position the board in the live preview.
2. View local board outlines and candidate regions when the geometry scanner
   can extract them. These overlays show geometry, not component identity.
3. Capture a still. ORION checks image quality and can attempt OCR using an
   available local engine.
4. With a configured vision-capable provider, request a summary, observations,
   legible markings, limitations and suggested next steps.
5. Inspect numbered observations on the retained still and ask follow-up
   questions about that same capture.

The workbench shares the application's camera tracker instead of opening a
second connection. Opening the panel alone does not activate the camera.
Selection, unavailable/stopped/stale-frame handling, cancellation and
capture/result correlation are represented in the implementation and tests.
Voice-triggered inspections also feed results into the workbench.

**Practical limits:** photographs do not measure voltage, current, resistance,
continuity, temperature or internal PCB layers. A visible defect can be a
suspicion; it is not proof of an electrical fault. Exact part identification
depends on legible markings and model accuracy. The local contour scanner is
not a trained component detector. OCR and models can be wrong; ORION retains
limitations instead of certifying a board as safe or fault-free.

Key implementation: `electronics_inspection.py`, `electronics_controller.py`,
`board_detect.py`, `gui/electronics_workbench.py` and the camera/vision services.

## Interaction and interface

| Area | Implemented surface | Requirements or limits |
| --- | --- | --- |
| Conversation | Voice/text interaction, recall, tool dispatch and assistant state | Quality depends on the selected model and context |
| Live voice | Gemini Live sessions, speech queues, interruption and device controls | Configured provider, audio hardware and network for cloud voice |
| Offline speech | Local transcription engine selection and speech output | Installed backends and models; first use can require downloads |
| Visual presence | Avatar/orb visualisations, assistant state and compact navigation | Rendering performance depends on hardware |
| Screen understanding | Capture, OCR, image analysis and visual verification | OS access; model interpretation needs a compatible vision provider |
| Optional camera features | Shared preview and gesture support | Feature-specific packages, models and hardware |

The dashboard contains Spellscape, Workbench, Mission, Research, Development,
Coding, Widgets, Toolkit, Marketing, Studio, Design, Fashion, Entertainment,
Library, Ops, Cognition, Automation, Command Centre, Plugins, Diagnostics,
Globe, Log, Memory, Telemetry, Chess and Security. Some pages expose shared
services or specialist agents rather than standalone applications.

The current avatar also includes a software-rendered head and viseme-driven
lip movement. Audio includes an echo guard, measured device-transport probes
and a push-to-talk/voice gate. These reduce specific rendering and audio failure
modes; lip-sync accuracy and device compatibility still depend on the setup.

Recent interface additions include live accent theming with readability checks,
clickable cards for research and briefing results, a browsable memory view with
deletion controls, and opt-in clipboard suggestions. Clipboard suggestions
screen for recognised credentials and send text for assistance only after an
explicit action; pattern matching cannot recognise every possible secret.

## Memory, research and documents

- Persistent local SQLite-backed memory, conversation retrieval and knowledge
  graph services support recall and context reuse.
- Prompt memory balances categories so bulk knowledge does not crowd out
  personal context, and indexes omitted keys for later lookup. Personal profile
  names and starter missions live in ignored local configuration.
- Document ingestion, file intelligence and a literature vault organise
  material for later retrieval. Format support depends on installed readers.
- Research services organise questions, searches, evidence and outputs;
  source quality and model claims still need assessment.
- Report and document export support structured output, including optional
  Word export. Local files and exports remain user data.
- Generic knowledge packs cover programming, AI, business, marketing and
  personal development. They are starting material, not guaranteed current
  or authoritative expertise.

## Productivity and personal organisation

| Capability | Current implementation |
| --- | --- |
| Missions | Durable missions, goals, tasks, progress and linked notes/files/research |
| Study | Study material, flashcards/review and model-assisted generation where configured |
| Focus | Focus sessions and a Cognition dashboard sharing engines with voice tools |
| Finance | Local manual/CSV-based accounts, transactions, subscriptions and cash-runway calculations; no built-in live bank connection |
| Wellbeing | Local self-reported mood, energy, stress and sleep tracking linked to focus/study information |
| Planning | Reminders, workflows, jobs, plan execution, briefings and executive/proactive services |
| Integrations | Outlook and Notion services when applications/accounts and credentials are configured |

Finance and wellbeing are organisational tools. Their presence does not imply
professional financial advice, medical assessment or independent verification
of entered data.

## Desktop, development and automation

- Application/window control, clipboard operations, file discovery/processing,
  desktop input, browser navigation and screen-based verification.
- A visible-browser copilot and web automation, subject to the browser session,
  permissions and supported page behaviour.
- Development workspaces, codebase assistance, debugger/Docker interfaces,
  diagnostics, self-repair proposals and change-awareness services.
  Bounded auto-repair is enabled by default for eligible changes with passing
  isolated tests, protected-file exclusions and session limits; set
  `ORION_AUTOREPAIR=0` to disable it. Applied source repairs keep backups.
- Workflows, jobs, capability registration, data-only skills, code plugins and
  optional MCP integrations.
- A bounded, session-local undo stack supports opted-in file operations. It is
  not a universal undo facility, does not survive process restarts and excludes
  operations whose previous state cannot be retained safely.
- Messaging/social-media, creator/content and commerce tools that need the
  relevant accounts. Availability does not establish authorisation or success.
  The source includes reviewed Discord and Telegram messaging plugins; their
  bot credentials and contacts stay in local configuration. Plugin tools are
  additional to the 138 declarations counted above.

Computer-control and code plugins can perform significant actions with the
running user's permissions. Security checks and confirmation controls exist,
but are not an OS sandbox or a guarantee against all unsafe actions.
The process-governor termination path uses a human-confirmed, expiring token;
PID actions also check process creation time before acting so a reused PID
cannot silently change the approved target.

## Additional services

Flight search builds a route/date query, reads the visible browser page and
summarises the offers it finds, cheapest first. Places may be cities, airport
names or IATA codes, and a city with several airports is searched as a whole
(London means Heathrow, Gatwick, Stansted, Luton and City together). Dates are
read from ordinary phrasing.

Two refusals are deliberate. A date it cannot parse produces a question rather
than a guess, because a search for the wrong day is indistinguishable from one
for the right day. And every fare a model reads off the page is checked against
the page text before it is reported, because a plausible invented price is the
one error here that costs money.

It requires a working browser session and page access; it does not guarantee
fares and cannot book or pay for anything.

The repository includes chess play/analysis through python-chess and an optional
Stockfish engine, media/gaming helpers, audio-studio services, globe/geographic
interfaces, telemetry, resource monitoring, backup interfaces, a security
dashboard and tools intended for authorised security labs. Some features need
separate tools such as Nmap or capture drivers.

External accounts, operating-system support, network services and installed
binaries can limit these services. The source export does not bundle downloaded
engines or the developer's installed application.

## Telephony

ORION has two ways to reach a phone, and they are deliberately different.

Through the paired Android app (`phone_action`) he hands over an intent and
the app opens the dialler pre-filled; you tap. Nothing is dialled
automatically, it costs nothing, and it needs no setup.

Through a connected Twilio server he places the call himself. Reminders carry
a delivery channel — DESKTOP, MOBILE_PUSH, SMS or VOICE_CALL — read from the
phrase as well as from an argument, so "remind me at eight about the news and
call me" rings your phone at eight.

Real calls dial immediately and cost money, so the whole path is built around
refusing. A number must appear in `config/telephony_contacts.json` before it
can be dialled at all. Anything ORION is about to say is screened for card
numbers, tokens and other credentials first, and that check fails closed — if
the screening cannot run, nothing is dialled.

Two-way calls run over a separate service (`orion-telephony`) that bridges
Twilio's 8 kHz audio to the conversational engine. Inbound callers must also
be in the contact book, and a voice on the telephone cannot make ORION *do*
anything: questions are answered freely, but a tool that changes something,
spends money or leaves the machine needs a code delivered to the paired phone
and read back. Caller ID is trivially spoofed and a recording of someone
saying "yes" is easy to obtain, so voice is treated as a convenience and never
as authority.

## Working unattended

A plugin's manifest may declare when it runs — `"schedule": "cron(0 6 * * *)"`
or `"interval_seconds": 1800` — and a supervisor carries it out whether or not
anyone is at the keyboard. A plugin that fails three times running is muted and
says so once, rather than writing the same error ninety-six times a day.

Schedules are wall-clock, so six in the morning stays six in the morning after
the clocks change. A consequence worth knowing: if the machine was asleep at
06:00 the job does not fire in a burst at 09:14. A briefing three hours late is
worse than one that did not arrive; `catch_up` is the honest path for the gap.

Each plugin declares the secrets it needs and receives only those, for the
duration of its own run. On Windows they are sealed with DPAPI, which ties them
to the Windows account; elsewhere with an owner-only key file. The scoping is
the property that always holds — the encryption protects against another user
or a leaked file, not against someone already running as ORION.

`research action=dossier` runs an investigation end to end and files it to
`research/dossiers/` with deduplicated citations, an executive summary and the
three or four things that actually matter, then announces itself. It prefers a
Tavily or Brave MCP server when one is connected and falls back to the built-in
encyclopaedia and news lookup otherwise — and the dossier says which was used,
because "no sources found" and "no search server configured" are different
problems.

## Running on a server

`python orion.py --headless` starts the brain with no window, no audio capture
and no screen grabbing, serving the phone and web clients over HTTPS. On such a
node the face is suspended entirely rather than drawn into a buffer nobody
reads, while the sensory state behind it keeps flowing over the bus.

`deploy/` carries what a Hostinger or any Ubuntu/Debian VPS needs: two systemd
units with restart policies and memory and CPU ceilings, an idempotent setup
script that configures a null audio device and locks the firewall to SSH and
HTTPS, a Caddy configuration, a state-sync script and a verification checklist.

Desktop and node share knowledge by shipping consistent snapshots one direction
at a time — never a shared mount. SQLite's locking does not work over NFS or
SSHFS, and the failure is silent corruption rather than an error.

## Models, offline operation and remote access

Provider routing supports configured cloud providers and compatible local model
servers. Retry, timeout, fallback and capability selection help handle failures.
Vision requires an image-capable route; a text-only local model does not gain
vision from being installed.

Local memory, geometry scanning, many organisational services and supported
speech engines can work without a cloud model. Broad offline conversation
depends on a working local model. Web research, external integrations and
cloud-backed features need network access.

An optional remote uplink, browser/PWA interface and Android companion source
are present. Remote use needs separate authentication and network configuration.
A headless cloud instance does not automatically control a different PC's camera
or desktop. Consult the remote/deployment guides before enabling access.

## Startup and responsiveness

The launcher performs lightweight dependency checks, records timing from the
launcher onward and reports startup phase budgets. Heavy optional imports and
camera work are deferred where implemented. Camera conversion/scanning uses
worker paths to reduce GUI-thread work. Creating the workbench does not start
camera hardware.

Recent repair-journal reads have a fixed byte budget instead of loading the
whole history. Repeated identical fault captures are sampled, while distinct
faults and repair/resolution events remain recorded.

These changes improve the startup path and make slow phases measurable. This
summary makes no universal startup-time or frame-rate guarantee: models,
drivers, storage, network and the enabled feature set matter.

## Publication and validation status

The snapshot includes a publication checker, a public-config allowlist for
standalone builds and a restrictive Docker build context. Personalised installs
retain credentials, memories and records locally. Export public source
separately from installs and Git history; see [Publishing ORION](PUBLISHING.md).

Regression tests cover electronics inspection, camera lifecycle, startup and
publication boundaries. Controlled data and mocked services do not establish
live hardware compatibility, provider availability or an end-to-end test of
every integration. Roadmaps remain historical design context, not promised
capabilities merely because they describe a proposed feature.

---

## Voice features matrix (updated 2026-09-20)

Which parts of ORION's hearing work on which input path. Until this update the
right-hand column was mostly "no", and nothing said so.

| capability | Gemini Live | offline transcription | push-to-talk |
|---|---|---|---|
| gender guess (`speaker_id`) | yes | yes | yes |
| voiceprint identity (`voice_speaker_id`) | **yes** (was no) | yes | yes |
| tone / emotion (`voice_tone`) | **yes** (was stale) | yes | yes |
| only-my-voice gate | **yes** (was no) | yes | yes |
| wake word / barge-in | yes | n/a | n/a |

Push-to-talk gates the same capture path, so it inherits whatever that path
does; it is listed separately only because people ask.

### What changed and why it was broken

`voice_presence` — the voiceprint comparison, the tone reading and the gate —
was fed from exactly one place: `speech_offline.transcribe_pcm`, which is
handed a complete utterance by an upstream VAD.

The Live path has no such moment. It forwards **every** chunk, voice and
silence alike, because the server's own VAD is what detects end-of-turn;
filtering to voiced chunks is what once made the live channel deaf. So there
was no complete utterance anywhere on that path to read, and half of ORION's
hearing worked only in the half of the pipeline most conversations do not take.

`orion_core/live_presence.py` assembles the missing utterance inside the
capture gate, on the same terms the gender tracker is already fed: voiced, not
ORION's own output, past the echo guard, while capture is allowed. When speech
ends, the whole thing is handed to `voice_presence` off the capture thread —
that thread has a 32 ms deadline and a speaker embedding costs 13 ms of it.

### One honest limitation

On the Live path the gate decides **part-way through** an utterance rather than
at the end, because a stranger has to be cut off before the turn completes
rather than after it has been answered. That means the opening second or so of
a stranger's speech does reach the model before ORION stops forwarding.

Nobody can be identified before they have spoken. That is a property of the
problem, not of the implementation.

### Which tool is which

Two tools with confusingly similar names, and they are not alternatives:

* **`speaker_id`** — a pitch-based guess at whether a voice sounded male or
  female. It cannot say which person spoke and cannot be made to.
* **`voice_speaker_id`** — a trained 256-dimension voiceprint compared against
  people who have enrolled. This is the one that answers "who is speaking",
  and it backs the only-my-voice gate.

The dispatcher attribute behind the second was called `speaker_id` until
2026-09-20, which made it the exact opposite of the tool of that name. It is
now `voiceprint_service`. The public tool names are unchanged.

### Turning the gate on

```
"enrol my voice"          → records ~5s and stores a voiceprint
"only listen to my voice" → ORION ignores anyone else
"answer anyone"           → back to normal
```

The gate **fails open**: too little speech, nobody enrolled, a missing encoder
or a borderline score all resolve to listening. Wrongly ignoring you looks like
a broken microphone and is miserable to diagnose; wrongly answering a stranger
costs one reply. Those are not comparable.
