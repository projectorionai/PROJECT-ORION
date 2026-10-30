# ORION Mark XXXXI (41.0.0) - Ascendant

## Mark XXXXI.1.1 (41.1.1), 30 Oct 2026

Built from the shared checkout after the 30 Oct round; rollback to 41.1.0 is kept in
`.audit_tmp/rollback_mark4111_20261030`.

- **Dropped files are read.** Before, a dropped file was announced as its whole path and
  no model ever saw it (`process_file` looked for a router that does not exist). Now the
  HUD, the Command Deck and Explorer's "Ask ORION" read the file and give its content to
  the conversation before he speaks, and he says one line ("Got report.pdf: a 3-page PDF,
  titled ... Want a summary?", or why it could not be read). Notepad's UTF-16, large and
  scanned PDFs, Word tables, PowerPoint, spreadsheets and code files are all read: 22 of 22
  readable samples (before: several not at all). The reply is said even in quiet mode,
  because the owner asked for it by dropping the file.
- **Research opens and cites its sources.** The timed run and the dossier never opened a
  page: the "sources" were search snippets. Now they open the top results, number the
  sources, cite them in the notes, mark uncited claims `[unsourced]` (which never become
  evidence), drop off-topic pages and label an encyclopaedia-only run. Multi-word
  encyclopaedia lookups work, and short runs no longer skip their last question. Offline
  fixture: sources opened 0/7 -> 4/7, notes citing a source 0/3 -> 3/3, key facts in the
  notes 1/4 -> 4/4, invented citations 1 -> 0; the same number of model calls. A run shows
  on the BRAIN work lane and the NEURAL page only while it is going, and the "research is
  ready" line is short.
- **Calendar, Focus and Fitness pages** in the Command Deck (Operations). Calendar: the
  next thing first, then the week from the Notion calendar, his reminders and goal
  deadlines, and one line to add a reminder. Focus: one big countdown (15, 25 or 45
  minutes, or body doubling, with ORION's face beside it and a check-in halfway), the task
  broken into small steps that are kept across restarts, and a gentle stop (a block
  stopped past halfway still counts). Fitness: workouts, weekly totals and bars, a streak,
  a weekly target, and health goals counted from what is logged.
- **Fixes from his own logs (22):** a quiet room no longer puts him into standby; radar
  frames and globe network errors no longer crash a page; Notion tasks without a title and
  Notion action names; algebra in the calculator (safely, off the main thread); a fallback
  list for the microphone hotkey; the security sentinel's routine processes, Defender and
  Explorer listeners; agent requests by other names; Whisper falling back to the faster
  base model when it reloads on the CPU; one research run at a time per topic; receipts'
  default action; every delete asks first.
- **Review fixes (7 + 3):** a notice said through Live can no longer make a tool call (a
  prompt-injection path); a notice is not repeated after he is interrupted; the GPU
  reaper thread starts under its lock; the home-network link skips VPN and WSL adapters,
  unregisters a site that failed to bind (logged once) and follows a port change in the
  firewall rule; security notes held during the quiet gap are said afterwards instead of
  lost; programs ORION starts are port-checked again.
- **He talks about his life like a friend** when asked casually, and the "Talk to ORION"
  caption is gone from the face.
- **Phone app 3.18: Calendar, Focus and Fitness on the phone**, under the header or by
  asking ("open my calendar", "start a focus block", "log a workout"), working with the PC
  off; workouts and focus blocks are encrypted on the phone. The home-network pin is kept
  for the same address and port. 190 unit tests pass (184 before); the production APK is
  2,072,205 bytes (3.17: 2,038,121; three unused resources removed). APKs:
  `releases/android/orion-companion-3.18-debug-compatible.apk` (updates the installed app,
  also served by the PC) and `orion-companion-3.18-production.apk`. Not yet tried on a
  phone.

## Mark XXXXI.1 (41.1.0), installed 29 Oct 2026 evening

Executable `9c0224b03bd90221...`; rollback to 41.0.0 kept in
`.audit_tmp/rollback_mark4110_20261029`. Staged and installed parity: every group passed.
Full suite: 14,839 passed, 10 skipped, 1 failed - the shutdown test caught the new LAN
link raising on a gateway that never finished starting (`remote.py`); fixed, and its
suites re-run (129 passed).

- **The phone, straight to the PC on home Wi-Fi.** A second HTTPS listener on the PC's
  home-network address (port 8765, ORION's own certificate) answers private-network
  devices only; the phone pins that certificate (confirmed once with the 16-character
  code on the Security page), tries the home network first, keeps Tailscale as a
  fallback, and finds the PC again if its address changes. The firewall rule is for
  private networks only and needs Administrator, so ORION shows the command.
- **His PC face on the phone.** The Three.js orb and face are gone; the app draws the
  PC's Screen Face (same five designs, colours, timing and emotion rules). Differences:
  the mouth follows loudness only, the gaze is random, an emotion fades after 10 s.
  App 3.17: `releases/android/orion-companion-3.17-debug-compatible.apk` (updates the
  owner's installed app; also served by the PC at `/download/orion-companion.apk`) and
  `orion-companion-3.17-production.apk`. 184 Android unit tests passed.
- **Offline.** The app answers what it can on the phone, queues the rest and uses
  Android's own recogniser. What does not work offline, PC and phone:
  [OFFLINE_CAPABILITIES.md](OFFLINE_CAPABILITIES.md).
- **Shutting the PC down from the phone** already existed (`remote_power.py`): paired
  phone only, a confirmation that expires in 90 s, a 60 s countdown the phone can
  cancel, and a normal Windows shutdown (never forced), so ORION closes cleanly.
- **Specialists** (offline, the agent-routing test sets): seen 91.3% -> 96.1%, fresh
  89.7% -> 93.1%, unseen 94.0% both; small talk sent to a specialist 1 -> 0.
- **Multi-part requests**: each part is ranked for its own tool. Planner goals with two
  parts, parts with their tool in the top three: 8 -> 11 of 14 (encoder cold), 9 -> 11
  (warm); every part covered 3 -> 5 of 7. Everyday top-1 unchanged (73.1% cold, 80.5%
  warm); about 1-5 ms more per multi-part request.
- **Learning memory**: his own tools' phrases are never pushed out by new lessons
  (they were from the 2,086th lesson), and 2,000 distinct phrasings are kept instead of
  the last 400 turns. Taught phrasings: 92.4% right after 1,500 later lessons, a
  re-training and a restart. About 1 MB of RAM.

Installed 29 Oct 2026: executable `950ec781a3ac801f...`, rolling back to Mark XXXX.7
(40.7.0, `a2f7244e...`) kept in `.audit_tmp/rollback_mark4100_20261029`. Staged and
installed parity: every group passed. Full test suite before the build: 14,789 passed,
10 skipped, 0 failed. Mark XXXXI includes the never-installed 40.8.0 (away mode, his
troubleshooting guide).

## Security notes: once, in his own voice

- **The spam.** On 29 Oct, six spoken notes in three minutes, each repeating "an
  unknown program has started accepting connections ... on port N". A program he
  could not name was keyed by its port, and Windows hands out a new port every time,
  so every 20-second pass was "new". Reading the program's name and path together
  also lost the name whenever only the path was protected, a program that had
  already exited was "unknown", and Store apps (WhatsApp) read as unsigned.
- **Now:** one finding per program with all its ports; his own processes skipped; a
  program already gone ignored; one he cannot name is shown on the banner and
  never spoken; Store apps count as signed; each thing and each piece of advice is
  said once; a note that is not a confirmed threat is spoken at most once every 30
  minutes (`security_sentinel.SPOKEN_GAP_S`).
- **The voice change.** Notices went to the local voice, whose Charon comes from a
  free-tier text-to-speech quota of ten lines a day per model, then the Windows
  voice. With the Live channel open and idle, he now says a notice himself, in the
  voice he speaks in; it is also in his context, so he can say what it was about.
  Offline or busy, the local voice says it. Nothing is said twice.

## Faster, lighter

Measured on the installed app with the owner's real config, one warm, muted launch
each, the same script both times:

| | 40.7.0 | 41.0.0 |
|---|---|---|
| ORION's own process start to ready | 5.01 s | 4.47 s |
| first paint / commands ready | 2.12 / 4.59 s | 1.58 / 4.09 s |
| peak private memory (his process tree) | 2,431 MB | 1,917 MB |
| working set | 754 MB | 754 MB |
| GPU memory in the first 150 s | 0 MB | 0 MB |

Single runs: the timings carry noise; the memory drop is well outside it. Isolated
source boots (5 interleaved each, median): ready 4.06 s both; private memory
1,272 -> 875 MB; CPU in the first 30 s after ready 7.0 -> 5.1 s.

- Numeric libraries capped at 4 threads (`orion.py`): 12 threads committed 363 MB of
  buffers at `import numpy`, 4 commit 106 MB, with no slowdown measured.
- GPU (DirectML) sessions are released after 5 minutes unused and rebuilt on the next
  use (`accelerator.py`); each held about 1 GB of committed memory for the whole
  session. Nothing in an idle session uses the GPU; OCR, object detection and
  meaning search load it when used, Whisper uses CUDA.
- Shutdown: the goodbye waited behind a queued notice and its network voice (16 s on
  29 Oct). It now cuts what is queued and plays his cached farewell at once
  (simulation: speech done at 3.5 s instead of 15.0 s; not yet measured on the app).

## Learning

- `reward_learning.py`: praise right after a turn rewards the route he took and teaches
  his routing network again; moving on without complaint is a smaller reward; a
  correction is a penalty and blocks the phrase from becoming a reflex; a retry that
  works with a different tool teaches the network away from the first one.
- A read-only request rewarded three times makes him offer an instant shortcut; it is
  installed only on the owner's yes (the existing guard test forbids automatic
  promotion). A rewarded multi-step task is remembered and given to him as a hint the
  next time - never replayed blindly.
- His self-benchmark reports rewards, penalties, offers and routing trend, and scores
  routing with his learned network. Deeper answers re-check their own arithmetic
  (10 of 10 wrong sums caught, 0 of 14 false alarms on the test set).

## His own model, trained

Llama 3.2 3B Instruct (the identical weights from the ungated
`unsloth/Llama-3.2-3B-Instruct`, Llama 3.2 Community Licence) trained as a LoRA on 890
rows of his own turns and the training set: 3 epochs, 91 minutes on the RTX 3060 Ti,
peak 5.08 GiB, validation loss 0.188 -> 0.129. The adapter is merged into the base
(`tools.train_lora merge`). It is NOT yet in Ollama: Ollama 0.40.11 refuses adapters
and, on Windows, imports safetensors only through an MLX runtime it does not ship.
The remaining step is converting the merged model to GGUF with llama.cpp's
converter; `tools.train_export register --merged` then imports it, and `evaluate`
scores it against the current model before the owner decides to activate it.
