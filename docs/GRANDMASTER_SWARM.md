# Grandmaster swarm and remote bridge

The installed source uses `GrandmasterBridge` through `ChessService.lichess`.
The existing `ChessBrain` remains the baseline search/evaluation implementation.
The reported 2604 Elo is not re-measured by this change. Neither 2800 Elo nor a
fixed frame rate can be established by unit tests. Promotion requires actual
validation games; no randomly initialised network replaces the playing engine.

| Phase | Implementation | Ownership boundary |
|---|---|---|
| 1 | `identity.py`, `transport.py` | SQLite identity mapping, strict authorisation, outbound HTTPS; aiohttp ndjson and berserk chat |
| 2 | `search.py`, `lifetime.py`, launcher guard | Spawned ProcessPoolExecutor; immutable initial FEN, move history, final FEN and limits; cancellation acknowledgement |
| 3 | `strength.py` | Depth/node caps, completed-root multi-PV Boltzmann sampling, asynchronous think delay; rated games use full strength |
| 4 | `dataset.py`, `selfplay.py` | One daemon process, randomised openings, root Dirichlet exploration, completed-game WDL and bounded quiescence targets |
| 5 | `nnue.py`, `weights.py`, provider/prompt hooks | Worker-only PyTorch, sparse HalfKP inputs, shared GPU lease, measured free-memory gate, disposable CUDA process |
| 6 | `arena.py` | Paired colour reversal, held-out validation, fixed-horizon promotion, immutable files, atomic active manifest |
| 7 | `book.py` | Periodic approved PGN downloads, bounded ZIP inflation, game deduplication, Polyglot compilation, mmap book/Syzygy readers |
| 8 | `bridge.py`, `supervisor.py`, `chess_remote.py`, service integration | Continuous game reader, stale-move cancellation, contextual commentary, remote viewport and administrative controls |

All eight patches are in `reports/grandmaster-patches/`. They describe only this
change relative to the working tree at the start of implementation. They do
not contain pre-existing unrelated edits. The final source is already patched;
do not apply the patches a second time to this checkout.

## Activation in the existing qasync application

Install `requirements/chess-grandmaster.txt` into the interpreter used by ORION.
Install a CUDA-enabled Torch wheel compatible with the host driver. The lean
frozen build excludes Torch; set `ORION_BUILD_CHESS_TRAINING=1` when building a
training-enabled standalone application. Missing Torch defers training with a
status message. With CPU-only Torch, short training passes use one CPU thread;
CUDA remains preferred when available and still requires the VRAM admission
checks below. Playing and CPU mmap inference do not import Torch.

Create a Lichess BOT account and supply its `bot:play` token through the existing
key store or `ORION_LICHESS_BOT_TOKEN`. The integration never upgrades an account,
opens a listener, configures port forwarding, or exposes a token in a URL.
Endpoints and event semantics follow the [Lichess Bot API](https://lichess.org/api).

```python
# Execute from an existing async command handler in ORION's qasync loop.
service = dispatcher.chess
await service.lichess.authorise(
    "jordan_chess", "Jordan, long-term friend", elo=1600, commentary=True
)
ok, status = await service.lichess.start()
if not ok:
    raise RuntimeError(status)

from orion_core.grandmaster.book import DEFAULT_SOURCES
await service.grandmaster.start(pgn_sources=DEFAULT_SOURCES)

# Administrative operations; queuing training never overrides GPU admission.
await service.lichess.set_handicap(1900)
await service.grandmaster.force_training()
# await service.lichess.abort_all()
# await service.grandmaster.stop_training()
```

Friends challenge the bot's Lichess profile. Authorisation is checked at
challenge acceptance, resumed-game admission and move submission. The existing
JSON allow-list migrates once; revocations cannot be resurrected by migration.
`accept_anyone` is disabled in the new bridge. Handles, memory entity labels,
handicaps and commentary preferences persist in `chess_lichess.sqlite3` beside
the existing settings file. For example:

```python
await service.lichess.revoke("jordan_chess")
```

The remote panel renders its own board, preserving local game navigation. A
16 ms precise Qt timer coalesces pending snapshots and stops when idle/hidden.
Server positions remain authoritative when move acknowledgements arrive late;
terminal game events render their final position before completion is reported.
Search, parsing, SQLite work, file export and training are outside the GUI loop.
Actual display frame timing requires measurement on the target desktop.

## Mathematical and resource contracts

HalfKP uses `1 + 64 * 10 * 64 = 40961` rows, including a zero padding row. For
each king perspective it sums at most 30 active non-king piece rows into a
128-dimensional accumulator, adds a bias and clips to `[0, 1]`. Concatenating
the side-to-move and opposing accumulators produces 256 values. Two 32-unit
linear/ClippedReLU layers feed one linear logit. Training computes:

```python
target = 0.8 * sigmoid(q_stm / 300.0) + 0.2 * ((wdl_stm + 1.0) / 2.0)
loss = binary_cross_entropy_with_logits(predicted_logit, target)
inference_centipawns = 300.0 * predicted_logit
```

WDL is `-1/0/+1` from the position's side to move. Search evaluations use the
same perspective; only presentation converts to White's perspective. Incomplete
self-play episodes and interrupted quiescence targets are discarded. Complete
games are split by game ID into training and validation sets, avoiding trajectory
leakage. Root exploration mixes the Boltzmann prior with Dirichlet alpha `0.3`;
noise never contaminates stored value targets.

Training is value learning from self-play outcomes with search distillation,
not an implemented AlphaZero policy network. The handicap's nominal Elo is an
uncalibrated control, not a guaranteed opponent rating.

Fresh networks now start with an exact material-balance path through two
accumulator channels, using ORION's existing piece values. Other channels can
learn positional corrections. A trainable piece-square factor shares lessons
across king contexts. Export folds it into the existing HalfKP table, preserving
the file format, inference size and accumulator cost. Existing checkpoints load
without altering their predictions. This improves the starting point for limited
desktop training; it is not itself evidence of increased playing strength.

Training uses a seeded, distinct validation sample of up to 1,024 positions from
held-out complete games, cosine learning-rate decay, checks every 32 steps, and
retains the best checkpoint. Four consecutive checks without improvement end a
pass. Repeated copies of one position no longer count as a validation corpus.
Decoded games are cached within a bounded 128-game cache. The playing arena
remains the independent promotion gate.

For a bounded offline session using ORION's own search, run this from the source
root with a Python environment containing NumPy, python-chess and Torch:

```powershell
python -m tools.train_chess --games 80 --steps 768 --device auto
```

Collection and training each default to a five-minute budget. Completed games
enter `config/grandmaster/positions.sqlite3`; a successful candidate is retained
in `training.nnue` and progress/metrics in `last_training.json`. `--games 0`
trains the existing dataset. `--fresh` explicitly starts a new candidate, and
`--compare-baseline` also measures a randomly initialized, unfactorized network
under the same data, seed and training limits. This command never changes the
playing champion. The existing training swarm runs the full independent arena.
Approved champions are now adopted between searches in local games as well as
remote games. An already-running/frozen application needs updated source/build
before it can use these changes.

The worker admits CUDA only when measured free VRAM supports at least 768 MiB
of allocator budget plus a 1536 MiB reserve. The allocator ceiling is 1280 MiB
and the device budget never assumes more than 8 GiB. This uses PyTorch's
[per-process allocator limit](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.set_per_process_memory_fraction.html).
That limit is not a machine-wide quota: external applications and driver/context
allocations require the free-memory reserve and target-machine monitoring.

User prompts mark foreground activity. Interactive provider scopes keep that
mark alive; local language model requests acquire the shared GPU lease. Training
checks cancellation every batch, while the supervisor monitors pre-emption every
25 ms. A requested stop has a 500 ms grace period before process termination.
The lease is released only after the process is confirmed dead, including after
repeated cancellation. CUDA cannot promise instantaneous release of a kernel
already executing; [`empty_cache()`](https://docs.pytorch.org/docs/main/notes/cuda.html)
alone cannot release live tensors or the process's CUDA context. Set
`ORION_GPU_LEASES=1`; training refuses a configuration allowing shared GPU leases.

There are two single-worker search pools, one self-play process and at most one
training/arena/book-maintenance child. Numerical libraries use one CPU thread
per worker. Processes run below normal priority. Each search pool admits one
running request and at most one waiter. Python process spawning and joins are
offloaded; no `join()` or synchronous berserk call runs on qasync.

The compressed dataset retains 50,000 games and has a 2 GiB main-database page
ceiling. At roughly 100 labelled positions per retained game this holds about
five million tuples; generation throughput is hardware/workload dependent.

## Promotion, books and persistence

Unpromoted training progress is retained separately in `training.nnue` so a
rejected arena candidate does not reset learning. Playing weights use a custom
ORION float32 `.nnue` format with validated architecture, exact tensor sizes,
finite weights, padding invariants and SHA-256 names. They are not compatible
with Stockfish `.nnue` files.

The arena uses 80 completed pairs with colours reversed on identical randomised
openings and equal time budgets. It discards censored games and makes a single
decision at its fixed horizon. The candidate must improve held-out BCE and its
one-sided Hoeffding lower bound on pair score must exceed `0.5`, at per-candidate
alpha `0.01`. This conservative gate can reject small real gains and does not
establish a FIDE or Lichess Elo. Repeated candidates are not a family-wise 99%
claim. A changed incumbent invalidates a stale arena result.

Promotion writes an immutable model before atomically replacing `active.json`.
Workers validate and load replacements between searches, clear the transposition
table and replace the model reference. Existing searches retain their weights;
neither the process nor a live game restarts. Corrupt/missing candidates leave
the current evaluator intact.

The GUI's Start training swarm control enables the default
[PGN Mentor Carlsen archive](https://www.pgnmentor.com/files.html), refreshed at
most once daily. Programmatic `start(pgn_sources=())` disables downloads. Parsing
requires both recorded ratings to be at least 2500, processes mainlines only,
and deduplicates games across archive versions. Archives and inflated PGNs are
bounded to 32 MiB, and archive member paths are never extracted. Polyglot
castling uses king-to-rook encoding and entries are sorted by unsigned Zobrist key.

Place trusted Syzygy `.rtbw` and `.rtbz` files in
`config/grandmaster/syzygy/` (relative to the active brain's configuration
directory). No tablebase download is automatic. The
[python-chess Syzygy implementation](https://python-chess.readthedocs.io/en/latest/syzygy.html)
maps table files lazily, with at most four open mappings here. Missing tables
fall back to search; root selection considers WDL, DTZ and the fifty-move clock.

Commentary is limited to one pending request and at most one message per 45
seconds per game. It uses the primary provider router with a background budget,
an eight-second generation deadline and no tool execution. Identity labels are
context for tone; the prompt forbids disclosing relationship details or inventing
unverified sacrifices/mates. Sending uses the synchronous
[berserk Bot API](https://lichess-org.github.io/berserk/api.html) in a dedicated
single I/O worker with HTTP timeouts and redirects disabled.

## Verification limits

Offline tests cover real spawn search/cancellation, loop heartbeats, stale move
suppression, quiet-stream POST failure, authorisation reservations, dataset POV,
Torch/mmap inference parity and gradients, a CPU training pass, promotion,
Polyglot castling, archive path handling and cancellation-safe GPU lease ordering.
No live account challenge, Lichess chat, GPU kernel, actual tablebase corpus or
long-running strength tournament was run during implementation. CUDA tests need
the target RTX 3060 Ti and CUDA-enabled Torch; the isolated verification runtime
has CPU-only Torch. Installation of dependencies alone does not rebuild an
already running or frozen ORION executable.
