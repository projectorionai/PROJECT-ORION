# Voice reliability and chess learning — 19 October 2026

This pass changes source code. The installed executable has not been rebuilt or
replaced. Physical playback quality and a chess rating increase remain separate
acceptance results. Existing private configuration, learned champions and paused
automation schedules have been preserved.

## Evidence and voice repairs

The latest installed session contained twelve completed event-loop stalls, with
a maximum of 513 ms. Two coincided with the intent learner, two with semantic
warming, and one with import warming. A Windows default-output change was
followed by a renderer message naming the previous output. Charon cache warming
also reached quota. These are separate observations; the log alone does not
identify a single cause of every audible stutter.

The implementation now addresses the following reproducible failure paths:

- Local speech and Live PCM reserve the shared output at utterance boundaries.
  Native audio waiting for local speech keeps FIFO order. A stream of new local
  notices yields to waiting Live audio after the current utterance. Hold,
  interruption, mute, shutdown and cancellation retain their respective behaviour.
- Removing local text from its queue and marking it as waiting is atomic with
  interruption's busy check, preventing a cancelled reply from escaping through
  that handoff. Idle synthesis uses a wake event without waiting under its lock.
- An honoured interruption rejects cancelled PCM before it reaches playback.
  Finish-speaking mode continues accepting its audio. A scheduled Live handoff
  processes final bundled audio and completion before reconnecting.
- Duplicate Live sessions preserve the current microphone and queue. A handshake
  completing after shutdown cannot reopen capture.
- Output reopening resolves the current Windows endpoint on the renderer thread
  and records the device actually opened. Strict name matching rejects ambiguous
  endpoints and preserves the previously measured audio API preference. It never
  tears down PortAudio to refresh devices. Resetting the microphone retains the
  system-default `None` selection instead of turning it into device zero.
- All intent-learning paths check the existing conversation-quiet probe before
  every minibatch, including the first. Prediction remains lock-free; learning
  still progresses under the existing bounded waiting policy.
- Playback exposes total and mid-utterance underflows. Optional telemetry failures
  cannot discard submitted audio, reopen a healthy device, or prevent TTS cleanup.
  The echo-reference hook is resolved once per write buffer rather than per slice.

`tools/diagnose_voice.py` adds a read-only diagnostic report from an existing
session log. It selects the latest session, counts scheduling and capture faults,
notes positive evidence of stale output reopening and summarises cache/quota
status. It copies no conversation, credentials, provider body or device name.
Parsing and retained timing samples are bounded; existing report files are never
overwritten.

```powershell
.venv\Scripts\python.exe tools\diagnose_voice.py --log "PATH\TO\session.log"
```

An optional `--output new-report.json` writes a new report. A missing underflow
message in an old log cannot establish uninterrupted playback. The new counters
are available through the speech manager's telemetry snapshot.

## Durable chess pathways

All four collection entry points share `grandmaster/curriculum.py`: background
self-play, bounded practice, offline training and the continuous improvement
worker. Each reproducible shuffled block contains six opening attempts and one
each of pawn, rook, bishop and knight endings, queen conversion and rook
conversion. Endgame starts can be mirrored between colours.

The per-pathway ledger records attempts, completed games, capped games and added
positions in the existing SQLite dataset. Completed experience and its progress
commit together. A capped game advances the lesson without supplying an invented
draw; interruption preserves the pending lesson. Restarting reconstructs the
lesson sequence from durable progress.

Shuffling avoids a strict alternating pattern aligning the endgames with the
dataset's every-tenth-game validation split. The regression corpus checks both
opening and endgame presence in both splits. This checks the scheduling defect;
it does not prove distribution balance for every possible run with censored games.

Quiet-only collection skips targets that the training filter would immediately
reject. A bounded cache reuses quiet-position decisions. Search tables and history
reset between games. Existing data, independent arena evaluation, candidate
promotion gates, ownership leases and foreground resource priority remain intact.

This supplies a persistent learning pathway when training is running. It does
not start a paused worker or recreate a paused heartbeat. See
[continuous training](CHESS_CONTINUOUS_TRAINING.md) for existing start/stop controls
and checkpoints. No measured Elo gain is claimed from these code changes.

## Verification

The final frozen cohort passed **1,337 tests across 51 modules**, with zero
failures/errors and one skip for the optional Berserk chat dependency. AST parsing
and critical Ruff checks passed across 1,114 Python source/test files. All source
hashes stayed unchanged during the final run, and the existing capability baseline
reported no removals or duplicate registrations.

A separate real-engine smoke completed three endgame games and stored 139 quiet,
terminal-labelled positions in a fresh scratch database in 3.63 seconds. Terminal
results were independently checked against the final boards. It trained/promoted
no model and provides no Elo estimate.

The audit retains before-edit files, behavioural regressions against those
preimages, test receipts, source hashes and a sanitised installed-session report
under `reports/voice_chess_audit_2026_10_19/`. No live provider request, microphone
capture, owner training worker, account change or executable installation was
started. The bounded scratch self-play experiment touched only its audit folder.

The continuation plan is recorded in the local handoff notes. Synthetic tests
establish state and ordering properties;
controlled listening and completed independent matches establish acoustic quality
and chess strength respectively.
