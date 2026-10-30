# Common sense with loose requests, and improvement driven by what was asked

Owner's request (4 Oct 2026): ORION needed "absolutely specific" prompts — "open a
browser" or "open an app" failed — and the forge should improve ORION continuously.

## Measured on this PC before any change

Requests resolved against the real application catalogue (304 entries: registry App
Paths, Start Menu, Store apps, aliases); nothing launched.

| Request | Before | After |
|---|---|---|
| open up a browser | **Tor Browser** | Microsoft Edge (the Windows default browser), alternatives named |
| zoom | **"Uninstall Zoom Workplace"** | Zoom Workplace |
| open the web | CapCut Web | the default browser |
| spreadsheet / slides | Spreadsheet Compare / nothing | Excel / PowerPoint |
| music, text editor, command line, pdf reader, video player, alarm… (17 kinds) | nothing | the user's default or most-used app of that kind |
| firefox (not installed) | document search, then "not found" | "Firefox is not installed. You do have: Google Chrome, Microsoft Edge…" |
| open an app | document search for the words | "Which application? The ones you open most are: …" |
| my downloads / documents | nothing | the real folder (Documents is OneDrive-redirected here) |

## What changed

**app_resolver.py** — 31 kinds of application with their everyday phrasings
(normalised exactly as requests are, or "the internet" never matches "internet"),
Windows default-app associations (`UserChoice` ProgIds, Store apps via their
AppUserModelID), known folders via `User Shell Folders`, a filter for Start-menu
shortcuts that are not the application (uninstallers, manuals, websites), and
`AppUsage` (`CONFIG_DIR/app_usage.json`): launch counts break ties, and opening a
different app of the same kind within two minutes of a guess is learned as a
correction. `open_app(remember_as=...)` records an explicit "when I say X, open Y".
A failed lookup rebuilds a catalogue older than two minutes, so an app installed
while ORION runs is found.

**Closing and switching** — `close_app` matched process names by SUBSTRING ("close
code" also killed `codex.exe`; "close edge" every `msedgewebview2.exe`) and
`TerminateProcess`ed every match, losing unsaved documents. Now: whole names,
`WM_CLOSE` exactly as clicking X does, four seconds' grace, a "save changes?"
prompt left to the user, termination only for windowless background apps; kinds
work ("close the browser" closes the one in front, or asks). Windows are found by
their application's processes when no title says it (Spotify is titled with the
song), and `force_foreground` verifies focus actually moved.

**Media** — `media_session.py` reads Windows' media sessions
(`winrt-Windows.Media.Control`). "Play" used the play/pause TOGGLE key and paused
music already playing; it now plays, says "already playing", and with nothing
open starts the user's music app. `now_playing` names the app and track.

**Safety** — `open_app` passed free text to `ShellExecute`, so "logoff" would sign
the user out with no confirmation on Windows Pro (bypassing the power guard);
guarded programs are refused and the literal fallback is limited to URI schemes.

**File search** — when Everything (installed here) answered "nothing", the search
still fell through to the Windows index and a whole-drive walk: 25.5 s for every
name that does not exist. Everything's complete answer is now final (0.3 s). With
no exact match, a relaxed pass offers the closest names (plurals, abbreviations,
spelling slips, filler words ignored) and never opens an ambiguous one.

## Continuous improvement

**capability_gaps.py** — a ledger of what the user asked for and did not get:
every unsuccessful tool call (classified: a clarifying question, confirmation or
guard is not a gap), including forged and MCP tools, which were never counted in
telemetry before; and every Live reply in which ORION declined ("I can't…")
without trying a tool. Similar requests cluster by shared words, by the capability
ORION named, or by meaning (local encoder, when loaded). The self-improvement
heartbeat leads its evidence with recurring unmet requests; a cluster shown three
times without recurring retires.

**The forge** — autonomous forging now needs strong evidence (a recurring unmet
request, a failing diagnostic or fresh tool failures). The real journal to 30 Sep
held 40 failed forges, nine of them variations on "emotion", and recent successes
such as `viseme_argument_adapter` invented from log noise; log text alone now
yields a note, not a tool. A successful autonomous forge is announced ("Earlier
you asked me: 'convert 30 celsius to fahrenheit'… I've built a tool for it") and
marks that need addressed; if the user keeps asking, the next review is told the
tool did not meet it. `forge(action='review')` runs a review on request;
`forge/diagnostics(action='gaps')` and "what can't you do yet?" show the ledger.
The journal is compacted past 512 KB with every recorded failure preserved.

**Intent network** — once seeded, the network never re-learned its corpus: better
tool descriptions reached an existing install only one replay sample at a time.
The corpus is now fingerprinted; a change triggers a background refresh (10
epochs; 3 left a toy change unlearnt) that keeps the user's own phrasings. App
kinds are taught as training phrases ("open the music app", "close the browser").

## Routing measurement

`tests/data/intent_routing_cases.py` (394 everyday requests, its odd half the
held-out set) plus 22 loose app requests and 4 maths requests; encoder cold;
means over 5 seeds; both conditions on identical sets (5 Oct 2026). Single-seed
comparisons are not usable: any vocabulary edit reshuffles the seeded network
by ±2 points on unrelated tools.

| | full top-1 | full top-3 | held-out top-1 | loose top-1 | loose top-3 | maths top-1 |
|---|---|---|---|---|---|---|
| before | 69.5 ±0.7 | 84.0 | 69.7 | 36.4 | 56.4 | — (no tool) |
| after | **69.7 ±0.4** | **84.1** | 69.2 | **86.4** | **95.5** | **100** |

The app-kind training phrases alone were worth +13.6 points of loose top-1
(68.2 -> 81.8) with the full set unchanged.

## Also in this pass (5 Oct)

* `calculator.py` + `calculate` tool: exact arithmetic (AST, bounded powers)
  and UK-aware unit conversion ("12 stone in kg", "5 ft 10 in cm", UK pints);
  typed conversions answer instantly through a reflex that fires only when the
  calculator recognises the request. The offline brain's maths answered "what's
  12 stone in kilos" with "That's 12." and had unbounded powers; it now uses
  the calculator.
* World time in the same tool (`tzdata`, which Windows lacks): "what time is it
  in Tokyo", "3pm UK time in New York", daylight saving included.
* Subprocess decoding: 16 `text=True` calls decoded child output as cp1252; one
  undefined byte (an emoji in a file or device name) made the reader thread
  fail and the caller silently got `stdout=None`. UTF-8 producers (git,
  Tailscale, Docker) now decode as UTF-8; everything has `errors="replace"`.
* Reminders: "tonight at 8" was set for 08:00, "on Monday at 10" for today,
  and "half an hour", "an hour", "twenty minutes", "at noon", "half past four",
  "tomorrow morning" set no time; all parsed now (25 phrasings verified).
* Contacts: "call my mum" finds "Mum ❤️" and "dave" finds the only Dave — never
  a guess between two (it names them and asks); WhatsApp names resolve through
  the contacts ("wa.me/mum" was a dead link) or open WhatsApp's own picker.

Two words were removed from `open_app`'s vocabulary after a per-case check: a
longer vocabulary diluted "open" itself, and "open spotfy" went to `find_files`.

## Found in real use (5 Oct, from ORION's own records)

The resolver's shadow log (`resolver_shadow.db`: what was said, which tool the
model called) and the tool-usage audit in the installed app's config were the
best evidence of what goes wrong. Four findings came straight from them.

* **Your words reached ORION in pieces.** Gemini Live streams the transcript of
  what you say as sub-word chunks that carry their own spacing. Each chunk was
  stripped and the pieces re-joined with spaces, so everything local — reflexes,
  the capability-gap ledger, the learning intent network, stored memories — saw
  "Ye ah , I 'm so ex cited", "che st", "sto ck fi sh". The model itself hears
  the audio and was unaffected. `utils.join_transcript` keeps the chunks'
  spacing; with no spacing anywhere it falls back to the old join.
* **Every tool was credited to the previous thing you said.** A tool call
  arrives before the turn ends, but the utterance was only recorded at the end
  of the turn — so "Oh my god, bro" followed by "shut down" taught
  "Oh my god, bro → shutdown_orion". Learned reflexes, the intent network, the
  resolver evaluation and the gap ledger all received shifted pairs. Tool calls
  are now attributed to the current turn's words; a turn whose words have not
  arrived yet teaches nothing.
* **ORION learned from his own voice.** His speech, picked up by the
  microphone, was taken as yours ("My objective is to assist you, sir…" →
  morning_briefing). Anything matching a line he spoke in the last two minutes
  is no longer learned or recorded as a request. The source-run intent network
  (all 18 of its learned examples were shifted, garbled or echoes) was set
  aside as `config/brain/intent_v2.set-aside-2026-10-05-shifted-pairs.npz`; it
  rebuilds from its seed corpus. The shadow log's older rows are shifted too.
* **"Scroll down" never visibly scrolled.** On Windows pyautogui passes its
  number straight through as the raw wheel delta, where one notch is 120, so
  the model's `scroll(-5)` was a twenty-fourth of a notch ("you don't actually
  scroll down" in the log). Amounts are now notches; a scroll with no amount
  goes down five; with the pointer over ORION, the desktop or the taskbar, the
  front window is scrolled instead.

Also: "Ryan," / "Orin," at the start of a command (how the transcriber hears
the name) now count as "Orion," for the instant reflexes; and windows can be
moved to another screen by name ("put it on the first screen"), restoring and
re-maximising as needed — the log shows this asked three times.

## Things that never worked on the installed runtime

* **Volume and mute.** pycaw 20251023 replaced `GetSpeakers().Activate()` with
  an `.EndpointVolume` property, so every volume, mute and pinch-gesture call
  failed with "Volume control is unavailable". pycaw was also never declared,
  so the build environment (and the exe) lacked it. Fixed for both APIs and
  declared. "Set the volume to 70" used to clamp to 100%, an explicit 0 became
  50%, and "unmute" set the volume to 100%; levels are now understood as
  0.4, 40, "40%", "forty" or "half", unmute keeps your level, and there are
  relative steps ("turn it down a bit") and a status read.
* **Brightness on a desktop monitor.** Only laptop panels (WMI) were handled;
  external monitors are now driven over DDC/CI, verified on the XV252Q.
* **Wi-Fi/Ethernet "toggles"** only ever disabled, and named adapters that do
  not exist here ("Ethernet" vs "Ethernet 2/3"). They now find the real
  adapters, switch either way, and open the settings page when Windows needs
  administrator rights.
* **Deleting a file** was permanent, and the "undo" copy was kept as decoded
  text — a photo or .docx "restored" that way was corrupt. Deletes now go to
  the Recycle Bin (`recycle_bin.py`, the shell's own API, verified in-bin),
  undo restores from the bin or from the exact bytes, and the cleanup
  assistant uses the bin when send2trash is absent (it always was in the exe).
* **"What's using my CPU?"** listed processes in PID order with 0.0% each:
  psutil's first per-process reading is always zero. Processes are now sampled
  twice, shown as Task Manager shows them, off the UI thread.

## Playing something by name

`media_control` could pause and skip but not play a named song. "Play about
time by Loyle Carner", "put some Queen on" and "play my Going out playlist" now
work in the Spotify app through Spotify's own search, read with UI Automation:
the search link must be URL-encoded (with raw spaces Spotify searched only the
first word), result rows keep stale names so the title is read from each row's
link, the row's Play button is pressed through UIA Invoke (the mouse never
moves), and a track link is only a fallback because Spotify ignores one while
something is playing. Songs are preferred over podcast chapters and live,
karaoke or sped-up versions unless asked for. Without Spotify, or when it has
no match, the top YouTube result opens. Success is confirmed through Windows'
media sessions (every session, so a browser tab counts), and whatever else was
playing is paused. Measured: about 2–3 s from request to music.

## The everyday questions (5 October, evening)

Found by printing every tool's vocabulary next to what the tool can actually
do, and by putting forty everyday spoken requests through the reflexes and the
router. Each gap was a promise the router made that no tool kept.

* **Weather.** "What's the weather in Leeds?" and "do I need a coat?" were
  routed to `geo`, which had no weather at all. `weather.py` (Open-Meteo,
  keyless) answers now, today, tonight, tomorrow, a weekday, the weekend or the
  week, with plain advice: coat or not, an umbrella yes / possibly / no with the
  hour, strong wind, sunscreen, ice. A day whose code says rain only overnight
  is said to be dry by day rather than contradicting the umbrella answer.
  Sunrise and sunset too ("when does it get dark"), timed on the place's real
  clock. The greeting's weather words lacked the freezing and snow-shower codes.
* **How far, the route, directions, places.** `journeys.py`: straight-line and
  road distance with a driving time (OSRM), ferries named, "no road" told apart
  from "unavailable" — the router snaps an unreachable end to the nearest road
  it can reach (New York became a Portuguese headland), so ends moved more than
  25 km are rejected. The route can be drawn on the Globe, by voice or from the
  Globe's new route form. Directions open in Google Maps. `places.py` finds the
  nearest pharmacy, cash machine, chippy or EV charger (OpenStreetMap), and
  says "open now" only when the opening hours are read with certainty.
* **Dates.** `dates.py` answers "how many days until Christmas", "what day is
  4 May", "three weeks from today", "when is Easter", "how long until 5pm",
  exactly — models get day counts and weekdays wrong. UK bank holidays come from
  GOV.UK (`bank_holidays.py`), and the greeting now says "It's the Early May
  bank holiday" or "Tomorrow is the Summer bank holiday".
* **Money and chance.** "50 dollars in pounds" at the ECB reference rate
  (`currency.py`); coin flips, dice and random numbers from the operating
  system's secure source, never invented.
* **The machine.** Bluetooth and screen-off existed only in the vocabulary;
  Wi-Fi off needed administrator rights. Radios now switch through Windows'
  Radio API as the user. Free disk space and the battery are reported.
  Timers are reminders with a length ("ten-minute timer started — it goes off
  at 20:53"). "How's the stock market doing?" answers in a second instead of
  composing the whole briefing.
* **Context.** A request answered instantly on the PC never reached the model,
  so "and tomorrow?" had nothing to refer to; the next typed turn carries it.
* **Camera.** A vision rule can take a photo of the moment it fires ("if
  someone comes in while I'm out, take a photo"); photos have a self-timer by
  voice and in the Vision lab.
