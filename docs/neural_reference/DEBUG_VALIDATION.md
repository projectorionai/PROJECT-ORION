**O.R.I.O.N. neural reference — debugging notes**

This records the earlier reference-package debugging pass. The subsequent [repository-wide audit](../ORION_BUG_AUDIT_2026-09-26.md) includes production repairs and the latest results. The reference suite now has **46 passing tests**, including deterministic memory-pressure pause/resume coverage.

Scope of the earlier pass below: the new reference package and its architecture guide. That pass did not change production `orion_core` files, project dependencies, the running assistant, trained weights or active adapters.

**Confirmed defects corrected**

- Repeated request cancellation previously released a generation before its thread stopped. A cancellation barrier now retains ownership until it completes.
- Cancellation during a SQLite commit could leave a durable submitted job absent from the scheduler's in-memory state. Public state transitions now complete atomically despite cancellation.
- A second scheduler start could create a second executor. Repeated starts are rejected.
- Replanning lacked an atomic revision check. The comparison now occurs inside the scheduler lock, with persisted revision numbers.
- A failed step ID could be reused after an intermediate replan. Execution receipts now reserve attempted IDs across revisions.
- State changes were visible before their journal commit. Copy-on-write snapshots are published after successful persistence, and receipts merge into the latest controlled state.
- Static path-policy checks were deferred until individual execution, allowing earlier effects in a structurally invalid plan. Every file action is now checked before admission and again before execution.
- Chroma IDs were global despite owner-filtered reads. Identical IDs could overwrite another owner's record. Upserts now use an owner-scoped digest and a versioned collection.
- Non-ASCII authentication headers caused a `TypeError`; byte comparisons now reject them with 401.
- Provider and SQLite failures could escape as unhandled HTTP errors. They now produce sanitised 503 responses; scope violations produce 403.
- Resource-monitor failure could silently kill the monitoring task. It now pauses dispatch and exposes a diagnostic category.
- `psutil.disk_usage` on Windows rejected `WindowsPath`. The executor now passes a string path; the live probe verifies it.
- Decoder-only generation now explicitly omits token-type IDs and fixes the returned generation format.
- Training data validation now runs before model allocation, rejects malformed/duplicate/leaking/oversized preference examples, and atomically reserves a fresh candidate directory.

Additional checks cover non-finite embeddings, exclusive screenshot-file creation, bounded file verification, subprocess cleanup, and user control changes during execution.

**Validation performed**

Latest complete runs: **45 tests passed under unittest and 45 passed under pytest**, with no failures or skips in the prepared environment. The existing repository configuration also collected all 45 successfully. Ruff lint/format checks, syntax compilation and dependency compatibility checks passed.

The complete suite is in `test_reference.py`, `test_regressions.py`, `test_integration.py` and `test_ml.py`. The raw most recent run is saved in `debug-test-results.txt`.

`debug-pytest-results.txt` records the isolated pytest run; `debug-collection-results.txt` records collection with the existing root configuration. The optional integration/ML modules skip cleanly when their dependencies are absent, verified separately in the bundled base Python. A full run using the repository's production autouse fixtures stopped at an inaccessible shared pytest temporary directory before executing tests; the successful reference-only pytest run excludes those unrelated fixtures. The entire production O.R.I.O.N. test suite was not run during this pass.

- Real FastAPI TestClient requests: authentication, malformed input, preview without execution, goal execution, errors, busy admission, pause/replan and telemetry.
- Real SQLite commits, restart recovery and injected disk failures.
- Real persistent Chroma, tested across separate processes with deterministic local embeddings: owner isolation, corrections, persistence, empty results and invalid vectors.
- Live Windows resource monitoring and the fixed `nvidia-smi` command. These are read-only diagnostics.
- Real tiny Qwen3 forward/generation through `NeuralEngine`, with a logits processor forcing known JSON tokens. This verifies integration, not reasoning quality.
- Real one-step TRL DPO training on a fresh PEFT adapter, including precomputed reference probabilities, evaluation and safetensors saving. Assertions verify that at least one adapter parameter changes and every frozen base parameter remains unchanged.
- Ruff lint/format checks, Python syntax compilation and dependency compatibility checking.

**Reproduction**

Use the isolated `.neural-debug-env` in the repository root:

```powershell
.\.neural-debug-env\Scripts\python.exe -m unittest docs.neural_reference.test_reference docs.neural_reference.test_regressions docs.neural_reference.test_integration docs.neural_reference.test_ml -v
.\.neural-debug-env\Scripts\python.exe -m pytest docs/neural_reference --confcutdir=docs/neural_reference -p no:cacheprovider -q
.\.neural-debug-env\Scripts\ruff.exe check docs/neural_reference
.\.neural-debug-env\Scripts\ruff.exe format --check docs/neural_reference
```

The exact installed package snapshot is `debug-environment.txt`: Windows, CPython 3.12.14, CPU-only PyTorch 2.8.0+cpu, Transformers 4.57.6, TRL 0.24.0, PEFT 0.18.1, FastAPI 0.141.1 and Chroma 1.5.9. CPU PyTorch came from the official `https://download.pytorch.org/whl/cpu` index. This snapshot describes the debug environment; it is not a tested CUDA deployment lockfile.

Known test-library notices: Starlette deprecates its httpx-backed TestClient in favour of httpx2; PEFT warns that the deliberately in-memory tiny model has no base config directory. Neither notice indicates a failed assertion or a downloaded model.

The recorded pytest run also reports that the shared `.pytest_cache` is not writable in this sandbox. The reproduction command disables that optional cache; no application behaviour depends on it.

**Remaining verification boundary**

CUDA/NF4 kernels, loading a pretrained 4B checkpoint, GPU VRAM/latency, full-size DPO, useful learned behaviour, real application interaction and screenshots were not exercised. No pretrained model weights were downloaded and no adapter was deployed. The Windows path policy remains an application-level guard, not a race-proof OS sandbox. Production capability approvals, reconciliation, goal supervision and automatic adapter promotion remain the explicit integration work described in the architecture guide.

The corrected memory schema uses `orion_v2_<embedding_version>` collections. If an earlier reference was run, re-ingest authoritative records into the new collection rather than assuming the old IDs were owner-scoped. Existing journal snapshots without a revision field load with revision zero.
