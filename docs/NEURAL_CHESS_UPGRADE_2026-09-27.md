# Neural learning and chess training — 27 September 2026

Implemented in the source checkout and trained locally on CPU. Existing unrelated
workspace changes were preserved. The installed/frozen executable was not rebuilt
or restarted; it must be updated before it can run these source changes.

## Changes

- Dense and sparse online networks now save Adam moments and clocks alongside
  weights. Sparse feature rows have individual update clocks, preventing rare or
  new words from receiving an oversized first update. Legacy checkpoints remain
  readable, and invalid training weights/optimizer state are rejected.
- Chess NNUE starts with material knowledge and learns positional corrections.
  A shared piece-square factor transfers lessons across king contexts and folds
  into the existing inference table on export. Inference format and size remain
  unchanged; PyTorch is still confined to training.
- Training works with CPU-only Torch, uses deterministic sampling, validates up
  to 1,024 distinct positions, reduces the learning rate, stops on validation
  stagnation and exports the best checkpoint. A bounded decoded-game cache avoids
  repeatedly decompressing the same games.
- Legacy chess practice no longer invents draws for interrupted/capped games.
  Illegal trajectories are rejected, and the smaller chess learner persists
  replay examples across restarts.
- Local play now adopts arena-approved NNUE models between searches, as remote
  play already did. Candidate weights do not automatically become playing weights.

## Training performed

The offline runner played **80 completed games** with ORION's own search, collecting
**6,322 positions** in approximately 216 seconds. No external engine, online
account, downloaded game corpus or GPU was used. Training used 722 distinct
development-validation positions from held-out game IDs. The final model retained
step 192 after stopping at step 320; validation BCE improved from 0.595219 with
material initialization to 0.586109 after learning.

A separate evaluation generated **24 fresh games** with a new seed. It retained
**1,971 distinct positions**, excluding every position already present in training
(ignoring move counters). These games were not used for fitting or architecture
selection. Targets combine ORION's quiescence evaluation and actual game results.

| Network | Fresh-game BCE, lower is better | Mean absolute probability error |
|---|---:|---:|
| Previous randomly initialized network, after training | 0.689275 | 0.326057 |
| Material initialization, before training | 0.398256 | 0.069060 |
| Upgraded network, after training | **0.393689** | **0.063563** |

The final model has **42.9% lower prediction loss** than the previous trained
network on this evaluation. Most of the improvement comes from the material
prior; learning provides a further improvement. This is a short-budget self-play
prediction benchmark, **not an Elo measurement or proof of stronger match play**.
The existing independent playing arena remains required for promotion.

The language-to-action network was also trained from 5,325 schema/vocabulary
examples covering 144 actions. A 37-case held-out routing check showed no top-1
or top-3 improvement from freshly training with the changed optimizer, and one
fewer top-10 hit. The better baseline initialization was therefore retained in
the new checkpoint format for future online learning: top-1 21/37, top-3 31/37,
top-10 35/37. No evaluation phrases were used as training examples.

## Saved state and reproduction

- `config/brain/intent_v2.npz`: trained routing weights for 144 actions, loaded by
  the existing intent learner on startup.
- `config/grandmaster/positions.sqlite3`: completed self-play games and labels.
- `config/grandmaster/training.nnue`: final chess candidate, SHA-256
  `5d9687b8d2cbda6869961fe36ee0d3b112f16db8a546203af8839b1121c3cae6`.
- `reports/neural_chess_2026_09_27/`: source snapshots, training comparisons,
  independent evaluation data/scripts, metrics and test logs.

No `active.json` was created. The current playing engine remains in place until
the candidate passes the full Grandmaster arena. No unattended training process
or recurring job has been left running.

Continue a bounded training session from the source root:

```powershell
.neural-debug-env\Scripts\python.exe -m tools.train_chess --games 80 --steps 768 --device cpu
```

Use the training swarm in an updated ORION build to run its existing full
training-and-arena promotion workflow. For a different Python environment,
install the project's optional chess training dependencies first.

## Verification

**286 tests passed** across neural continuity, chess search/service/dispatch,
Grandmaster training and promotion, dataset policy, intent routing, and existing
audit regressions. Tests cover exact resumed NumPy learning, rare sparse-feature
updates, legacy/corrupt checkpoints, Torch/exported inference parity, material
initialization and constrained updates, replay persistence, invalid trajectories,
unfinished games and local adoption of approved weights. CPU training and fresh
self-play evaluation were actually executed. CUDA and a full playing-strength
arena were not run.
