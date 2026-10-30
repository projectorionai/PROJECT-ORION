# Mark XXXIII chess benchmark — 30 September 2026

ORION's own classical engine completed **48 games in 24 colour pairs** against Stockfish 18 on the host PC. It won 28, drew 10 and lost 10. The fitted performance was **1800.8 on Stockfish's UCI_Elo scale**, with a 95% bootstrap interval of **1722.3–1899.0**.

This measurement applies to the following contract. It is not a FIDE or Lichess rating, and it does not establish performance at longer time controls, on another processor, or with a different trained checkpoint.

| Setting | Value |
| --- | --- |
| Engine under test | ORION's shipped classical tables and seeded opening theory |
| Personal learning / promoted NNUE | Neither used |
| ORION move budget | 0.08 seconds |
| Stockfish move budget | 0.08 seconds |
| Opponent | Stockfish 18, one thread, 32 MiB hash |
| Opponent UCI_Elo levels | 1320, 1600, 1900 |
| Openings | First eight positions in the versioned fixed suite |
| Colours | Both colours for each opening and opponent level |
| Rating fit | Logistic maximum likelihood |
| Interval | 2000 percentile bootstrap samples of colour pairs, stratified by opponent |
| Seed | 20260927 |
| Move cap | 600 plies; only completed games counted |
| Runtime | Python 3.12.14; python-chess 1.11.2; Windows 11 |
| Verified current host CPU | AMD Ryzen 5 5600X 6-Core Processor |

The complete local record is `reports/mark_xxxiii_2026_09_30/chess-rating/results.json`. It contains every game, moves, termination reason, source checksums, opponent executable checksum and resource readings. The run finished at 01:36:45 BST. Its lowest recorded available memory was 1373 MiB. Reports and learning data are kept outside the source commit.

The immutable run contract contains the supplied label `AMD Ryzen 5 3600`, copied from the original task description. A subsequent local `Win32_Processor` check identified this PC as a **Ryzen 5 5600X**, consistent with the preceding live telemetry audit. This corrects the host description; it does not alter the played games or pretend that the same result was measured on a Ryzen 5 3600. The original contract is retained with a separate metadata correction in the local evidence directory.

Reproduce a fresh measurement without changing the playing engine or training data:

```powershell
& '.neural-debug-env\Scripts\python.exe' -m tools.measure_chess `
  --output 'reports/chess-rating-new-run' --pairs 24 `
  --move-seconds 0.08 --opponent-seconds 0.08 `
  --levels 1320 1600 1900 --seconds 1800 `
  --machine 'AMD Ryzen 5 5600X 6-Core Processor'
```

Use a new output directory. The harness snapshots its contract and refuses incompatible resumption. More games and a separate longer-time-control run would improve the evidence; a training-loss reduction alone cannot increase the displayed rating.

The older 2604 figure was described with ORION at 1.5 seconds per move and its opponent at 0.3 seconds. It is a different time-budget claim, and the current app no longer presents it as unconditional present strength. The remote game's **Casual handicap** selector remains a nominal difficulty control, not a measured rating.

## Training ownership

The desktop swarm and continuous trainer now share the same OS lease on `grandmaster/improvement.lock`. Starting or queuing a second writer returns the verified current trainer's state instead of starting competing checkpoint writers. The UI periodically reads that state while visible.

The explicit Stop training control asks a verified continuous worker to save and stop using `improvement.stop`. Closing the desktop leaves an independently scheduled trainer running. No process is terminated by an unverified PID, and no live or online game is started as part of a rating audit.
