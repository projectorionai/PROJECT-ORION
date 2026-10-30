# Continuing chess improvement

The user authorized ongoing training on 27 September 2026. The former **ORION
chess improvement** heartbeat was subsequently paused; it must not be recreated
or resumed automatically. The code supports resumable manual workers and
ORION's opt-in training controls. This document describes their behavior, not
evidence that a worker or schedule is currently running. The first run evaluates
the already trained candidate against the current classical champion.

The worker is `tools/improve_chess.py`. It runs below normal priority, uses one
CPU thread for numerical work, and has a 55-minute run budget. It does not need
CUDA, external chess engines, online accounts, or network downloads.

Each cycle:

1. Evaluate the pending candidate, if any, before training another one.
2. Play 80 completed pairs from distinct randomized openings, reversing colours
   and allowing both engines 0.08 seconds per move. Snapshot the classical
   knowledge and candidate so results remain comparable across restarts.
3. Promote only at the full horizon and only if the existing one-sided
   confidence bound exceeds a 50% score. An incomplete pair is retried after
   interruption. Censored games are not invented draws. A changed incumbent
   invalidates the previous arena's results. Each match records evaluator
   source hashes and runtime versions. Changed or previously unrecorded code
   starts a fresh match; prior completed results are archived. An edit during
   a run checkpoints it so the next worker loads the new implementation.
4. Collect 80 additional completed self-play games, train up to 1,024 steps with
   validation/early stopping, and evaluate any improved candidate. Training and
   validation use the same quiet-position filter: no check, legal capture or
   legal promotion. This aligns the candidate's static evaluation with the
   stored quiescence targets. Otherwise collect more experience. Repeat within
   the run budget and resume next time.

## Learning pathways (19 October 2026)

Every collection entry point now shares the `mixed-v1` offline curriculum:
background self-play, bounded practice, `tools.train_chess` and
`tools.improve_chess`. Each block of twelve attempts contains six opening games
and one each of pawn, rook, bishop and knight endgames, plus queen and rook
conversion. A deterministic new shuffle per block preserves restart stability
and prevents strict alternation from aligning all endgames with the dataset's
every-tenth-game validation split. Endgame starts are mirrored between colours.
This gives the worker
shorter routes to completed endgame experience without inventing outcomes for
unfinished games. These are training starts, not a claim that the engine has
mastered the lessons or gained a measured rating.

`positions.sqlite3` stores each pathway's attempts, completed games, censored
games and added positions in `curriculum_progress`. A completed game and its
progress commit in the same transaction. Capped games advance the rotation;
interrupted games retain the pending lesson. The next worker reads the shared
cursor from this ledger, so restarting does not repeatedly reset to openings.
Historical game rows and neural checkpoints are preserved. Reports expose a
`curriculum` object; totals cover experience collected after this upgrade.

Quiet-only collection avoids computing targets for tactical positions it would
discard during training. A bounded cache reuses quiet-position checks across
batches. Search tables and history reset between games. Workers still use
bounded episodes, low process priority and the existing ownership lease.
These changes do not enlarge the CPU/GPU budgets or bypass foreground priority.

The promotion arena remains independent: randomized opening games, reversed
colours, equal time and the full confidence gate. Endgame curriculum results
never count as promotion evidence. There is no guaranteed Elo increase.

Checkpoint state: `config/grandmaster/improvement.json`. It records the phase,
PID, current pair count, wins/draws/losses, last training result, and up to 200
arena verdicts. `evaluation/` holds the current immutable match snapshot;
`evaluation/history/` retains full completed/invalidated match records.
`training-quiet-v2.nnue` retains the continuous worker's quiet-position learning
progress. Its first pass starts from the material prior; the earlier
`training.nnue` and all archived game rows are preserved. Filtering is a
training view, not a deletion of tactical positions. A lower validation loss
still requires the same independent playing-strength gate before promotion.
The OS lock `improvement.lock`
prevents concurrent worker instances and automatically releases after a crash.

Relative Elo is descriptive **against the opponent in that equal-time match**.
It is not an absolute FIDE/Lichess rating, and is never added to the old quoted
2600 rating. Small samples are marked provisional; all-win/all-loss estimates
are left unbounded instead of inventing finite ratings. Repeated training does
not guarantee uninterrupted strength gains. Rejected candidates leave the
playing champion unchanged.

Any explicitly re-enabled heartbeat should stay quiet during routine progress
or rejection and report a promoted champion, failure or required user action.
Training does not rebuild or restart the installed executable.

## Start or resume manually

From the project root:

```powershell
$root = (Get-Location).Path
Start-Process -FilePath "$root\.neural-debug-env\Scripts\python.exe" `
  -ArgumentList '-m','tools.improve_chess','--seconds','3300' `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput "$root\reports\neural_chess_2026_09_27\continuous.stdout.log" `
  -RedirectStandardError "$root\reports\neural_chess_2026_09_27\continuous.stderr.log"
```

Check the saved PID and state before launching; a duplicate is rejected by the
lock. Startup errors go to the stderr log. A worker error is also recorded in
the state when possible, preserving previous progress.

## Stop

To stop the chess training, pause the heartbeat and create
`config/grandmaster/improvement.stop`. The worker notices this marker and
checkpoints completed pairs. The heartbeat must never remove that marker
without a user request to resume. The marker can also be created manually:

```powershell
New-Item -ItemType File -Path config/grandmaster/improvement.stop -Force
```

## Verification

Focused tests cover existing Grandmaster and neural behavior, quiet training
views and separate checkpoint lineage, resumption, duplicate prevention,
completed-pair accounting, stale opponents and source versions, archived match
results, censored results, full-horizon promotion, relative-rating edge cases
and PID cleanup:

```powershell
.neural-debug-env\Scripts\python.exe -m pytest tests/test_grandmaster.py tests/test_neural_chess_training.py tests/test_chess_improvement.py tests/test_quiet_chess_training.py -q
```

No Elo gain is claimed until real evaluation games complete.

Future work: add progress display to the existing training controls,
build a fixed tactical/endgame regression suite separate from both training and
promotion, and add reviewed PGN imports with provenance, game-level splits and
explicit licensing. Consider adaptive lesson allocation only after measuring
per-pathway completion and a separate benchmark; keep opening coverage and the
independent playing-strength gate. Stronger engines or local tablebases can
provide optional teachers, but their availability, resource use and labels
must be explicit rather than silently relaxing this offline baseline.
