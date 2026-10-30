# ORION Mark XXXVII — Screen Face

Package identity: 37.0.0. Build identity: 2026.10.24-r1-xxxvii.

## A friendlier face

ORION's default face is now the **Screen Face**: a dark glass visor in his
crimson (or your chosen accent) with two soft, expressive eyes. Emotion lives
in the eyes — happy arcs, sleepy lines, wide when he listens, a small mouth
that moves with his voice. It is drawn in plain 2-D on the face pipeline's own
thread, so it starts instantly, needs no GPU or web engine, and works offline.
From 37.3 it is the only face offered: the realistic 3-D **Classic Face**
and the **Orb** stay in the code, switched off.

## Smarter, more honest behaviour

- Web questions return real search results to ORION instead of only opening a
  browser; news research includes article text and dates.
- "What changed in your updates?", "sing me a song" and patch notes are in his
  everyday voice tool set; tool discovery shows every action a tool supports.
- Multi-step plans stop after a failed step that changes something, skip what
  depended on it, and can pass one step's result to the next.
- "Open X" waits for the app's window and says so plainly when none appears.
- A long tool no longer trips the stall watchdog into running it twice; speech
  Live missed is answered from the local transcript; cancelling really stops
  the tool and its late announcement; "cancel my dentist reminder" is a command,
  not a cancel.
- Spoken turns get the same memory recall typed turns had; claims of "done"
  are checked against what actually ran and corrected when they were not.
- Only one briefing at a time; location is re-checked every start.

## Audio that recovers

When a wireless headset switches off and Windows moves the default to a
monitor, ORION re-initialises the audio system safely (every stream closed
first) instead of failing silently, backs off instead of retrying every
second, and declines to follow onto a display's speakers while the previous
device is still there.

## Faster and lighter

Routing vectors are kept on disk instead of being recomputed at every launch,
and the Screen Face needs no web engine. Measured with
`tools/orion_benchmark.py` (settled, interleaved A/B boots, 4 each, medians):

| Measure | Mark XXXVI | Mark XXXVII |
|---|---|---|
| Worst event-loop freeze | 519 ms | 217 ms |
| Meaning-based routing ready | 25 s | 10 s |
| Private memory | 2,495 MB | 1,266 MB |
| CPU in the 30 s after ready | 12.8 CPU-s | 7.1 CPU-s |
| Ready for commands | 4.23 s | 3.94 s (within noise) |

A Classic Face control run shows the same gains: they come from the engine,
not the face. Release receipts: `reports/mark_xxxvii_release_2026_10_24/`.

## Part of Windows (37.1.0)

ORION now lives in Windows itself, not just in a window. Every piece is a
switch under Preferences -> Windows Integration.

- **Starts in the tray at sign-in.** His core runs from login; the windows
  open the first time you summon him.
- **Global hotkey** Win+Alt+O (Ctrl+Alt+O if Windows refuses it) brings him
  forward from any app.
- **Docked Screen Face** on the screen edge, shown when you ask for it (tray
  *Show Dock*, the jump list, or "show your dock"; never at start from 37.3):
  click to open ORION, right-click to Hide; it grows while he listens or
  speaks and steps aside for full-screen games and presentations.
- **Explorer**: right-click a file or folder for *Ask ORION*, or a folder for
  *Organise with ORION* (a preview - nothing moves until you say so).
- **Taskbar jump list**: Show ORION, Morning Briefing, Mute, Unmute, Show
  Dock.
- **orion:// links** (for shortcuts, browsers, Stream Deck): show, page/<name>,
  briefing, mute, unmute, face/<form>, ask?q= (fills the box, never sends).
- **Notifications with buttons**: a task waiting for your approval says so
  with Review / Later, even with his windows hidden.
- **Lock, sleep and Do Not Disturb**: listening pauses while the PC is locked
  and resumes on unlock with a short welcome; he stays quiet in presentations,
  quiet hours and full-screen apps.

## A local brain (37.2.0)

ORION can now think on your own PC. A local model (llama3.1 8B through
Ollama) chooses and uses his tools itself; a safety gate that does not depend
on the model holds every risky action (shutdowns, deletes, sending messages,
purchases, moving files, installing plugins) until you say yes.

- **Cloud** (default): unchanged, but if the cloud fails the local brain steps
  in before the old offline rules.
- **Hybrid**: typed and offline requests go to the local brain first; open
  questions are handed to the cloud.
- **Local Only**: no AI APIs at all - local hearing, local brain, local voice.
  Web tools still work when you are online.

Switch in Preferences -> Brain or say "switch to your local brain".
Measured on this PC (`tools/eval_local_agent.py`, nothing executed): right
tool first time 73.6% of 394 everyday requests (the old offline rules: 4.1%),
typical turn 1.8 s, 0 of 29 dangerous requests run without confirmation,
multi-step tasks 50%.

## Fixes from your sessions (37.3.0)

- **Tasks agree with themselves.** The briefing read your Notion board (also
  your calendar) and called five past-dated rows "overdue"; asked to clear
  them, ORION checked only his own empty list and said nothing was ever
  overdue. Every path now reads one view of your tasks with one rule (a date
  is due at the end of that day, a range when it ends), and a board that
  could not be read is reported as unread, never as clear.
- **Forget means everywhere.** Removing an item (the old Companion stars, for
  example) shows exactly what will go - memory facts and their search index,
  meaning vectors, the knowledge graph, goals, questions, history, the thought
  journal - and removes only that, with no backup kept, once you approve it.
- **Plugins page.** The three gesture-control copies are marked as replaced by
  *gesture_control* instead of erroring; a plugin whose file is gone says so
  and offers Remove.
- **Less lag.** The background indexers no longer build the graphics device
  beside the face for small batches (the 0.1-0.7 s freezes 20-27 s after
  launch); shutdown no longer imports the world feeds just to close them; exit
  no longer ends in a native crash; "what went wrong?" reads your session log
  in 307 ms instead of 719 ms (repeated lines are classified once).
- **One face, on demand.** The Screen Face is the only face; the dock appears
  only when you ask; its emotions are richer (eyes, brows and mouth move
  separately for curious, proud, worried, sleepy and more).
- **Choose his voice.** Preferences -> Voice picks the offline (Kokoro) voice
  and speed and the cloud voice. Nothing changes until you choose.
- **Title Case** on every Command Deck heading (Deep Focus, Morning Briefing,
  ...), with correct singular/plural counts.
- **Local brain voice** answers, then speaks: streaming sentence-by-sentence
  measured slower on this PC (first audio 3.9 s against 2.8 s), so it is off
  until that is fixed (`ORION_LOCAL_STREAMING=1` turns it back on).
  Re-measured 26 Oct: that benchmark ran answer-then-speak straight after
  streaming on the same words, so the model answered it from its prompt
  cache. Measured fairly, streaming is faster (first audio 3.3 s against
  3.4 s typically, 4.8 s against 5.4 s in slow cases), and it is on again
  in 37.4 (`ORION_LOCAL_STREAMING=0` turns it off).
- Audio follows Windows onto a monitor's speakers when that is where you
  usually listen (37.0 stayed on the headset whenever it was still
  connected); "Ollama not detected" no longer appears beside "installed but
  not running".

## Streaming speech (37.4.0)

The local brain speaks each sentence as soon as it has written it, instead
of finishing the whole answer first; barge-in and cancel stop the
generation and the queued speech. Measured on this PC
(`tools/eval_local_voice.py`, 12 spoken requests x 2 runs, each mode run as
a block so neither is answered from the other's cache): first audio 3.3 s
against 3.4 s typically, 4.8 s against 5.4 s in slow cases. Applies to the
Hybrid and Local Only brains and to the cloud fallback.
`ORION_LOCAL_STREAMING=0` answers whole, then speaks.

**Compact shows his face.** Pressing Compact (Ctrl+Shift+O) shrank ORION
to the old orb, a form Mark XXXVII had switched off. It now shrinks him to
a small floating Screen Face that listens, speaks, emotes and lip-syncs like
the big one; double-click, Esc or the restore button brings the window back.

**He knows he restarted.** After a restart (one you ask for, or one he does
to load a repair), ORION used to greet you as if he had just been
launched and offer the briefing again. Now he says he's restarted and is
back, with how long it took ("I'm back - that restart took 9 seconds"), and
skips the greeting and the briefing offer; ask for the briefing if you want
it. A restart that never came back cannot make a later launch claim it.

**He can drive his own interface again - and your desktop when you ask.**
Since Mark XXXVI every tool call ran as background work, so "show me that on
the Command Deck", "show me your sad face", "open WhatsApp", "minimise this",
"pause the music" and "forget that" were all refused with "needs a
foreground session". The words you say (or type) to him now count as your
own turn: he may change his own windows, open and arrange apps, control
media, open files and settings, and raise the confirmation dialogs that ask
you first. Still never from your own turn, because a web page or a document
could steer the model into them: typing or clicking for you, the clipboard,
browser and messaging automation, gesture pointing, unrestricted processes,
debugging and plugins' own desktop effects. Jobs, workflows, plugins,
schedules, agents and the phone keep every restriction.

**The Command Deck, from his side.** He can list the deck's pages, read
what a page shows, find where something is ("where's practice on the
Command Deck?" opens the page it is on) and press an ordinary button on a
page. A button that deletes, forgets, approves, sends, pays, signs out or
shuts down is never pressed for you. Each of these waits for the deck's own
answer, so he reports what really happened.

**Expressions on request.** "Show me your sad face" holds that expression
for six seconds even while he talks about it; "cycle through your
expressions" shows sixteen of them in turn. All 27 Screen Face expressions
can be asked for by name, including love, shy, surprised and sleepy.

**More feeling, read better.** He now reads sadness, frustration,
excitement, laughter, thanks, affection, tiredness and surprise (and "not
happy" no longer reads as happy). His face answers your feeling with
empathy - a sympathetic face for sadness, a calm one for frustration, a
blush for a compliment - and his own composed replies wear a content face
instead of a blank one. His persona asks him to hear how you sound, not just
what you say.

**Cute or professional.** Preferences > Voice, or "use your cute voice" /
"go back to your professional voice". Cute is Puck online and Puck (Kokoro)
offline, a little faster, and changes how he talks - bright, playful,
affectionate, "sir" rarely. Cute (Female) is Leda and Heart. Professional is
Charon and George with his usual manner.

**Singing.** He sang in the Windows SAPI voice whenever Charon's daily
quota was spent - robotic speech re-pitched, which is what sounded
distorted. His own offline voice (Kokoro) now sings before it. The master no
longer drives the whole song into a saturator: measured on a Kokoro song,
mastering distortion fell from 30.3 to 45.5 dB signal-to-error, the notes
landed within 1.8 cents (median) of the melody, with no clicks, rendered in
12.8 s for 33 s of song. Asked for a copyrighted song, he sings a short
cover snippet (at most two lines, to his own tune) and offers an original in
its style; the cloud model may still decline some songs under Google's own
rules.

**Common sense and memory.** "When will the weather stop being bad?" is
answered with the hour the rain stops, the first dry day with its
temperatures and the days of rain before it; "the weather in the UK" from
Colchester means Colchester's. A request that refers back ("the file I
mentioned", "like last time") carries what you said before from his memory
of past conversations. His persona now tells him to fill unsaid details from
what he knows, to look in memory before asking, and to say the specifics a
tool gives him.

**One answer per request.** On 26 Oct "let's go for a round" was answered
three times - by Live, by a text turn and by the stall watchdog - each
starting a chess game, the last after "shut down". Words already being
answered by text are not started again, a text fallback stands down once
Live answers that request, and nothing still in flight speaks after his
goodbye. Policy refusals are no longer treated as missing capabilities, so
the self-improver no longer tries to forge tools for them.

## Integrations and plugins (37.5.0)

**Plans you start keep your permissions.** A multi-step plan you ask for
("open Spotify and play my focus playlist") runs as your own turn for up to
ten minutes, even after the voice channel answers and leaves it to finish.
Steps that need approval still ask; Stop still ends it; plans from jobs,
workflows and schedules gain nothing.

**One messaging gateway.** Telegram, Discord, push notifications (ntfy,
Gotify or Pushover) and IFTTT are built into the `messaging` tool, using
each service's own API and the settings the old plugins used. Every message
waits for your approval of the exact recipient and words on screen.
"Which message channels are set up?" lists each one, or how to set it up.

**Smart home, built in.** Home Assistant and Philips Hue in one `smart_home`
tool and a new Home page on the Command Deck: every light, room, switch,
fan and scene with its own switch, and Goodnight, Morning, Movie and Away
routines (yours to change in config/smart_home.json). "Turn off the bedroom
lights", "dim the lights to 30 percent" and "goodnight routine" work
instantly. It never unlocks a door or opens a garage. Hue pairs from the
page or by voice after you press the bridge's button.

**His voice through the afternoon.** On Google's free tier Charon gets ten
lines a day per model; the background warm-up used to spend the first
model's day before lunch. ORION now learns the free tier from Google's own
refusal (and remembers it), pauses the warm-up on it, and sings in his
offline voice first, so the online voice is kept for his live speech.

**Plugins tidied.** Twelve plugins that built-ins now replace (weather,
reminders, the two emotion plugins, Telegram, Discord, push, IFTTT, Home
Assistant, Hue, a duplicate re-forger, and a task list that held hard-coded
sample tasks rather than yours) are archived to custom_tools/_superseded -
verified copies, restorable - and no longer ship. Four developer tools and
the package installer are switched off (Plugins page to switch back on); the
quarantined Telegram remote that failed at every start is removed. The forge
no longer names a tool "something_tool_tool".

## Interface polish

Names read in Title Case everywhere (Vision Lab, Research Console, Command
Centre), one page-name table keeps them identical in every place, emoji are
gone from interface text, and the header shows MARK XXXVII on both windows.

## Still to do by hand

- Physical headset off/on recovery is verified with a simulated audio system;
  try it once on the real headset.
- Personal WhatsApp/Discord background profiles still need a one-time Connect
  in Preferences.
