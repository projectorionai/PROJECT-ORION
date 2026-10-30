# ORION Mark XXXX — Aegis

Package identity: 40.0.0. Build identity: 2026.10.27-r3-xl.

## Conversation

- Answers no longer stop mid-sentence and wait for "keep going". ORION already
  asked the live model to finish a reply the server cut off, but only when the
  turn held no words from the user — and a spoken question's own transcript
  arrives in the same turn as the answer, so it almost never ran (3 times in
  two weeks). It now holds back only when the user spoke after the answer
  began, and one answer may be carried on up to three times. The rest arrives
  while the first part is still playing.
- "Yes, go on" to a briefing is answered at once: the briefing is composed
  while the offer waits (it took about six seconds after the yes).

## Singing

- Asked to sing, ORION sings through the sing tool every time — his tuned
  voice over piano. His speaking voice cannot hold a tune, and the old
  instruction to sing "in your own voice" had him recite Twinkle Twinkle.
- Typed requests ("sing me a lullaby", "sing happy birthday to Sam") go
  straight to the singer.
- If the live model starts reciting lyrics anyway, its speech stops at the
  first lyric and those lyrics are sung properly instead.

## Security

- Threat guard (`threat_guard.py`), built on Windows Defender's engine: a
  program starting from a temp or Downloads folder is checked for a valid
  signature and, if unsigned, scanned by Defender; new programs, scripts and
  archives in Downloads are scanned once they finish downloading; a new
  program set to start with Windows is checked and scanned; a changed hosts
  file is reported. A threat Defender confirms is always spoken; something
  that could not be cleared is spoken once; a clean result is shown, not
  spoken. What was told is remembered across restarts for a week.
- One message per check: an installer runs as `setup.exe` and `setup.tmp`, and
  the two each got a spoken note ("'Torvis-setup.exe' is running from a temp
  directory", twice). One program is now one finding, and one pass is one
  message.
- Windows Defender's own health — real-time protection off, definitions out of
  date — is checked every fifteen minutes; the check existed but was never run.
- Security audit ("how secure is my computer?"): firewall, Defender, User
  Account Control, Remote Desktop, Remote Assistance, SMBv1, apps reachable
  from the network, unsigned auto-start programs and phone control, scored,
  with the fix for each failure.
- Scan a file or the Downloads folder on request (antivirus `file`,
  `downloads`).

## Phone control

- A master switch: "stop everything from my phone" (or "phone control off")
  and nothing from the phone reaches this computer until it is switched back
  on — at the computer only.
- A record of what the phone made this computer do — the tool, when, and what
  happened, never the words — kept for 90 days ("what did my phone do?").

## Desktop

- Win+Alt+M mutes and unmutes the microphone from any window.

## Globe

- Smoother: up to 60 frames a second while it moves (still nothing drawn at
  rest), a short glide after a zoom or drag, and a lighter frame in motion.
  A resolution change now redraws at once instead of clearing the canvas.
- A live readout of where the cursor points, the camera's altitude and its
  heading.
- Double-click a spot to fly in towards it; arrow keys steer, + and − zoom,
  Home resets.
- ORION's crimson-and-silver HUD: glass panels, a corner frame, silver type.

## Research

- A page is downloaded up to 2 MB and cleaned off the event loop: a 5 MB page
  took 117 ms to clean beside his voice.

## 40.1.0 — finishing your sentences, tones of voice

- He waits for a real pause before answering. The live channel ended the
  owner's turn on the first breath; it now waits 1.2 s of silence with the
  low end-of-speech sensitivity (Reply Pause: Brisk 0.7 s, Natural 1.2 s,
  Patient 1.7 s — Preferences > Voice, or "give me more time to finish",
  "don't cut me off", "reply faster"). The change applies the moment he is
  quiet: the channel is refreshed with the conversation kept. The local
  brain's quick endpoint stretches with it.
- Expressive voice (affective dialog): his tone follows yours. Probed against
  the live model first: every natural request and every one of ORION's own
  notes was answered; a curt "say only: ready" came back silent 2 times in 3,
  so two silent replies in a session set it aside, and a server that refuses
  it is reconnected without it.
- His delivery has a range — warm, bright, hushed late at night, brisk,
  grave, deadpan with a beat before the line, curious — instead of
  "understated intonation".
- Personality: opinions with a reason, one follow-up when you share
  something that matters, callbacks to earlier in the conversation, no "As an
  AI". Asked plainly what he is, he says so.
- An online voice change applies once he finishes speaking, not at the next
  conversation.
- Coming back: the offline return line is never the one he used last; the
  Mark is said as a number ("Mark forty"), never letter by letter.
- "Next face design", "try another look" and "change your face" step through
  the designs.
- Phone control off now stops the phone's conversation too, not only its
  tools.
- A new network port is named by its program; Windows' own services and
  ORION are left alone; a signed program is shown, an unsigned one told once.
- Tests: six test files built a Qt application nobody held, so each crashed
  the run (exit 127) when it ran on its own; a heredoc had written literal
  backspaces into a regular expression.

## 40.2.0 — he benchmarks himself, plans paid campaigns, and clicks when asked

- Self-benchmark ("benchmark yourself"; diagnostics `benchmark`): reasoning,
  coding (model-written code checked against an import and call allow-list,
  then run in its own time-limited process), exact instruction following and
  agentic skill (his own resolver's tool choice, and multi-step plans as tool
  calls), scored by code and compared with the last run
  (`tools/self_benchmark.py` runs the same outside the app). First real run:
  78% overall - reasoning 12/12, coding 5/6, instructions 3/5, agentic 11/16.
  After the fixes it found: agentic 16/16, instructions 4/5, 88% overall.
- What the benchmark found and what changed: "how secure is my computer"
  routed to the cyber course and "open spotify" to media control (tool
  vocabulary, both now right, no everyday case lost); his name rule beat an
  exact instruction (EXACT_DIRECTIVE, and the name rule now yields when the
  user asks for another spelling); code wrapped in prose was not found.
- Accuracy rule: anything beyond simple mental arithmetic goes through the
  calculate tool, and a multi-step answer is checked against the question.
- Meta and TikTok Shop (brand_growth `ad_plan`, `ad_metrics`,
  `ad_copy_check`): break-even and target CPA/ROAS from price and margin, the
  daily budget the learning phase needs, structure (Advantage+ / GMV Max,
  Spark Ads, creators), a creative grid, kill and scale rules; a campaign's
  CTR, CPC, CPM, conversion, CPA, ROAS, frequency and profit with a scale /
  hold / fix / kill verdict; and copy that ad review tends to reject.
- Desktop: pointer, keyboard, window focus and clipboard work in the owner's
  own live turn ("press Start Camera" was refused four times); a refused
  click no longer scrolls and re-reads the screen for 45 s, and a click by
  name gives up after 25 s.
- The UI shows MARK XXXX · 40 (core window and Command Deck) and a HUD layer:
  crimson gradient hairlines, spaced capitals, monospaced readouts, crimson
  progress bars, slim scrollbars.

## 40.3.0 — his keys sealed, a private log, approvals by button

- Keys sealed to the Windows account (`key_seal.py`, DPAPI through ctypes, so
  the built app needs nothing extra): every provider key and integration
  token in `api_keys.json`, the bot tokens and webhooks in `messaging.json`
  and the phone's signing key. The file stays readable JSON, so a new key can
  still be pasted in as plain text; it is sealed the next time ORION reads
  it. A copied config folder opens nowhere else: a value sealed by another
  account reads as empty, and is never written over.
- Private log (Preferences > Privacy, the Security page, or "private log
  on"): the session log keeps what he did - tools, timings, connections,
  errors - and never what was said: your words, his replies, the transcript
  in a trace line, the phrase an emotion was read from, a specialist's
  answer. The window still shows them.
- Telegram approvals with buttons: an action that needs approving comes back
  with Approve and Decline. A tap counts only from the allowed private chat
  that asked, carries the confirmation's single-use token and expires with
  it; phone control off still stops it. Before this, an approval asked for
  over Telegram was filed under a device no phone app or desk ever listed,
  so it could never be given.
- The Security page: the audit's score with every check, failures first in
  crimson with their fix (Run Audit); Defender's real-time protection,
  definitions age, last quick scan and recorded threats; the phone-control
  master switch with the phone's last steps; the private-log switch.
- Live ad accounts (brand_growth `ad_report`): the last N days of each Meta
  and TikTok campaign, read from the accounts themselves and run through the
  scale / hold / fix / kill analysis. Read-only - no budget change, pause or
  spend. Needs `integrations.meta_ads` (access_token, ad_account_id) or
  `integrations.tiktok_ads` (access_token, advertiser_id).
- Start-up report: warm and cold starts are told apart. Warm starts took
  3.4-3.5 s on 27 Oct and are now held to 4 s; the first start after Windows
  came up took 8.6 s, 2.8 s of it before ORION's first line ran (Windows
  reading him from disk), and is held to 9 s and named as such.

## 40.4.0 — his stats as graphs, and a full debug

- NEURAL, a Command Deck page beside BRAIN ("show me your stats"): a signal
  flow graph of every turn - input, hearing, thinking, each tool, the reply -
  with edge widths from real counts, success rings and a pulse that runs
  along an edge the moment a live turn reaches it; the response time of each
  turn split into hearing, thinking, tools and voice start with its p50 and
  p95; every self-benchmark run per category; providers (requests, tokens,
  latency, failure) and tools (calls, failures, median time).
- A full debug, by two read-only review passes over the code and the logs,
  every finding checked before it was fixed:
  - Research read only the first chunk of each page (usually the <head>), so
    ordinary pages were dropped as thin - a Mark XXXX regression, now read to
    the end or 2 MB.
  - Phone control off did not stop the phone's agent-task route, and phone or
    Telegram messages were logged and remembered before being refused.
  - A key sealed by another Windows account was erased by the next save.
  - The benchmark's code check could be stepped round (`__builtins__.eval`,
    `operator.attrgetter`, format strings).
  - A signed program's all-clear silenced a later warning about an unsigned
    one of the same name for a week; a second threat of one family went
    untold.
  - The private log still wrote `ORION[local]:` replies, the phone's own words
    and "not answering '...'".
  - Expressive voice counted interrupted turns as silent, and once set aside
    could not be switched back on without a restart.
  - The phone switch read a locked or damaged file as "on"; its activity log
    was never pruned.
  - The Downloads scan could outlast the 45 s a live turn waits.
  - A mid-session Live 1008 "missing authentication credential" made him say
    the key should be replaced (the key worked 7 s later); it now reconnects
    quietly.
  - One session's handled COM dumps could hide a real crash behind them.
  - The reasoning provider's primary model returned 5xx at the start of 7 of
    9 sessions; the alternate that answers is now tried first for 6 hours.

### From the 27 Oct 23:18 launch

- He said a security audit was running when nothing ran. Two causes, both
  fixed: the Cybersecurity Agent was refused the security checks (classed
  as "not read-only"), and an `agent_dispatch` that matched no specialist
  returned as a success, which the model described as a background task.
  It now says "Nothing was dispatched - do not say a task is running".
- Everything he does reaches the BRAIN page: a WORK lane shows each
  specialist agent, background job, tool still running after its turn
  answered, and install, from start to finish; MCP, plugin and forged tools
  are drawn like built-in ones; a call made through `use_tool` is labelled
  with the tool that ran (its lane used to stay open); Recent Activity now
  includes agents, specialists, jobs, research, reflexes, remote, security
  and his proactive lines. A specialist that answers directly now logs its
  end as well as its start.
- He can install software (`software_install`, winget): search, and install
  only on your click - the exact package, id and version are shown in his
  confirmation dialog, which also says it accepts the package's licence. The
  phone and background jobs cannot install.
- Lighter on CPU and memory (measured by a profiling pass): the offline
  voice -200 MB and -33% CPU, offline speech-to-text -40% CPU, every small
  model's threads sleep between runs, the globe gives back about 0.5 GB two
  minutes after it leaves the screen, and the offline voice and screen-text
  reader are let go when memory is short. Running small models on the GPU
  (DirectML) costs about 245 MB of system memory on this PC rather than
  saving it, so they stay as they are: the GPU saves CPU time, not memory.

## 40.8.0 — he keeps hearing you, away mode and his own troubleshooting guide

- **"He randomly stops hearing me."** Six times since 26 Oct his own
  recogniser heard a whole request and Live produced nothing for it, while
  the channel stayed up. Now:
  - the microphone stream is closed properly whenever it pauses (he speaks,
    a tool runs), as the Live API asks for any pause over a second - it
    never was, which can leave the server holding audio;
  - notes added to his context (remembered facts, late tool results, who is
    speaking) wait until you have finished your sentence instead of
    arriving in the middle of it;
  - every miss says how much of your audio reached Live and how loud it was,
    or why the microphone was not reaching it, and repairs a microphone left
    switched off;
  - a miss with your audio delivered refreshes the live channel (same
    conversation) at the next quiet moment, so the next request is heard -
    on 28 Oct the channel stayed deaf for two requests in a row.
- **Sharper offline hearing on the GPU**: small.en instead of base.en. With
  room noise mixed into a 9.5 s request it made no word errors at 0 dB
  (base.en 6.2%) and 2.1% at -3 dB (9.4%), in 266-272 ms - less than base.en
  took on the GPU before. Dropping timestamps nobody read made every decode
  faster (base.en 229 -> 133 ms).
- **Agents, second round**: code and error shapes, questions about ORION's
  own running and named online services now lean on their specialist. On
  58 new requests written before the change, the right specialist 82.6% ->
  89.1% (keywords alone: 30.4%); small talk sent to a specialist 2 -> 1 of 12.
- **Robots on more boards**: the rover did not compile for the Leonardo and
  Micro (their Serial is USB, not HardwareSerial); it now compiles for the
  Uno, Nano, Mega, Leonardo and Micro with every warning on, and USB robots
  work out of the box (pyserial ships with ORION).
- **Away mode.** "Away mode on" keeps the PC awake for the phone while the
  owner is out, and locks it after ten minutes without input - once, and
  again after he comes back and leaves. "Away mode off" gives sleep back to
  Windows; so does ORION closing.
- **His own troubleshooting guide**, built into the app so it always matches
  it: can't hear, slow, the GPU, robots, firmware, privacy, away mode, the
  agents, updates and training his local brain.
- Both were added without moving everyday requests: the offline tool finder
  ranks the right tool in its top three for 83.2% of the development set,
  with or without them.

## 40.7.0 — agents that understand the question, and erasing your data

40.6.0 was never installed on its own; it ships in this release.

- **Specialists chosen by meaning.** The agents were picked by keywords: on
  55 requests written after the change and never used to tune it, the right
  specialist 6 times (10.9%); now 43 (78.2%), using ORION's own sentence
  encoder, with strong keywords still deciding and the keyword router as the
  fallback. Small talk stays with ORION (1 of 12 went to a specialist, "take
  a screenshot"). About 4 ms a request.
- **Agents keep working offline.** When the cloud fails, a specialist answers
  on the local model (Ollama, on the GPU) instead of giving up.
- **A second look for study work.** A request that asks for care ("double
  check", "carefully") or is coursework, an exam, an essay or a dissertation
  gets a reviewing pass before the answer is given.
- **Erase all my data** (the right to erasure). "Erase my data" says what
  would go; approving the on-screen confirmation arms it, and ORION restarts
  and sends it to the Recycle Bin before any store opens. Personal keeps his
  keys, settings, pairings and models; everything is a fresh install.
- The robot tools and the erase tool have their own words in the tool
  finder, chosen so everyday requests ("stop the music") stay where they were.

## 40.6.0 — a robot body, his hearing on the GPU, and nothing personal in the source

- **Robots.** `robot/` holds ORION Robot Link: a C99 library and C++11
  wrapper for a robot's controller (no heap; compiles clean with every
  warning as an error for Windows, 32- and 64-bit Raspberry Pi and the
  Arduino Uno), a software robot to try it on, and a rover sketch for an Uno
  (4.9 KB of flash, 589 bytes of RAM). The safety is in the library: a 500 ms
  watchdog, a latching emergency stop, and range checks that refuse rather
  than clamp. ORION's tools: `robot_control` (motion only on the owner's own
  turn; stop and emergency stop always) and `robot_firmware` (writes C++,
  compiles it with every warning on, flashes only when asked).
- **Hearing on the GPU.** Offline speech recognition (Whisper) runs on the
  RTX 3060 Ti: 824 ms to 271-317 ms for 9.5 s of speech, the same words. It is
  proved with one short run before it is trusted, and falls back to the CPU
  on any failure. The app now carries the cuDNN it needs; cuBLAS is in
  %USERPROFILE%\ORION\gpu-runtime.
- **Six new instruments** in the Command Deck, all from real data: a voice
  ridgeline (BRAIN › Activity › Voice), a reactor gauge (Diagnostics › Load),
  a memory constellation (Memory), a 24-hour day ring (Cognition), a tool
  radar (NEURAL) and a market heat map (Widgets).
- **Nothing personal in the source.** The owner's name, PC, network address,
  colleges, tasks and business are gone from every tracked file. The
  creator's name is read from this PC's identity.json, and the public
  export refuses anything personal this PC knows about its owner.
- **Mark XXXX**, not Mark XL, everywhere. He still says "Mark forty".

## 40.5.1 — his creator in his own story

- Asked how he began, he credited the AI tool he was built with and never
  named the person who made him. His origin now names the person who
  made him (from this PC's config, never the source) and has developed him through every version, and he is told
  never to credit the AI tools or models he was built with. The two
  origin memories that credited a tool were reworded the same way, and a
  memory of who made and developed him was added (on this PC only).

## 40.5.0 — plain words, graphs that look the part, and a faster deck

40.4.1 was never installed; everything in it ships here.

- He hears again after a dialog. The OPS deck's Forget question ran a
  second Qt loop inside its task; every task the loop tried to wake
  meanwhile was refused ("Cannot enter into task") and never ran again -
  orion-send-realtime among them, so Gemini Live got no more microphone
  audio and he answered what the small local recogniser misheard ("fine how
  are you doing today" for "How should I relax"). The dialog no longer
  nests a loop, and any task refused that way is now offered its step again
  every 50 ms until it gets in, instead of dying with the session open.
- Plain words, wit kept: everyday vocabulary and short sentences, no
  'complexion', 'splendid' or scene-setting, 'sir' at most once a reply -
  still sarcastic and funny.
- The five overdue Notion tasks from before university no longer open every
  briefing: board tasks due before the fresh start (config/fresh_start.json)
  are left out. The board itself is untouched.
- BRAIN page: an Activity graph with a Lanes / ECG setting, remembered.
  NEURAL page: the same ECG (Hear, Think, Act, Speak, Work) above the
  network graph, which is now drawn as a network and animated from a cached
  layer (about 1 ms a frame).
- Markets as a chart on the Widgets page: indices (FTSE 100, FTSE 250,
  S&P 500, Nasdaq), ETFs and stocks, today's line against the previous close,
  hover for time and price, the rest of the group beside it. Rising is white,
  falling is crimson. config/markets.json replaces any group.
- Desktop control: the pointer moves like a hand (curved, eased, an
  occasional overshoot on long moves) and types with a human rhythm; say
  "fast" for the old instant moves. Clicks, scrolls, drags and typing the
  owner asks for are no longer refused as "needs a foreground session" -
  each was checked under its own name, none of them on the allowed list;
  background work stays blocked. OCR finds the right word on screen 99.7%
  of the time (was 93.8%) and tolerates one misread thin letter.
- Change detection works in the installed app: the build records a hash of
  every module, so "what changed in you" compares real code, not the four
  files on disk.
- The UI is spaced out: the ORION pill no longer sits on the brand, the
  header has three zones, panels share one padding, every deck page starts
  on the same line, and the deck's top bar shows no mark. The face window
  reads "O.R.I.O.N" and MARK XXXX. "What's Been Happening" is "Terminal".
- A faster Command Deck. Measured, then fixed:
  - an event filter on the whole application ran Python on every Qt event
    (31% of a heavy page's first open) - now a sweep per window and page;
  - every page re-parsed the full stylesheet when built (23%), which also
    kept a theme change from reaching it - pages now inherit it;
  - Diagnostics read the process list on the GUI thread every five seconds
    (330-360 ms) - now a thread reads it;
  - Plugins redrew every row on each show and re-parsed every plugin each
    scan - now it redraws on change, and audits are cached by file hash;
  - the Workbench listed cameras on the GUI thread at start (400-1360 ms);
  - the Memory table measured all 500 rows to fit its columns;
  - the deck's page history was written, flushed, on the GUI thread on every
    page switch.
- The globe keeps a disk cache (256 MB, no cookies): Cesium and the map
  tiles it has shown load from disk when its process starts again, instead
  of being downloaded on every open.
- He remembers where he began: ORION v0.01 (ORION CORE), switched on 19 June
  2026 after the plan was written on 11 June, through v0.14 and Mark VI to
  Mark XXXX - a line in his identity, and the full story (his first night
  included) in his memory on the owner's PC only.
- Old tasks cleared for university: jobs, goals, missions, reminders,
  research runs, focus and study history, standing questions, workflows and
  the task fields of his state, in both configs (to the Recycle Bin).
- Retired things removed, not kept as refusals: the cute voice style (its
  table, tool actions, schema wording and reflex words); 62 superseded and
  quarantined plugin copies; a disabled list naming plugins that no longer
  exist. Three working plugins that wrote files unattended now ask on the
  phone, so the start-up warning about them is gone.
- Seven tool results that reported a failure as success now say they
  failed, so the model cannot describe them as done.
- The NEURAL page shows ORION's own memory and CPU, apart from the PC's.
- Ollama, when ORION starts it, uses flash attention and an 8-bit
  conversation cache: about half the memory for the same context.
- Unused test files removed from the PC (old versions, reference copies,
  stale worktrees, old reports and build copies); the suite stays.

## Next

1. Encrypt his memory databases. Whole-database encryption needs SQLCipher
   (a package download) or EFS (Windows Pro; this PC runs Home).
2. The local brain's training stage. The dataset builder, LoRA trainer and
   exporter from 25 October are parked unfinished (90 of 95 of their tests
   pass); training itself needs CUDA PyTorch, bitsandbytes and a base model -
   all downloads.
3. Ad-account budget changes behind an approval, once reading has been used
   for a while.
