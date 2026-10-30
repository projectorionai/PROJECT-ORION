# O.R.I.O.N. repository bug audit — 26 September 2026

## Scope and result

This audit covers the existing desktop application, the remote gateway, the Android companion build, and the optional neural reference pipeline. The application currently uses Qt/qasync and aiohttp; the FastAPI neural adapter remains an optional integration, not a replacement silently installed into the production application.

The review combined the complete Python test suite, targeted failure reproduction, static checks, compilation, real local HTTP tests, Android compilation/lint, and syntax checks of embedded browser scripts. It did not involve a line-by-line manual review of every one of the 356 core Python modules. Existing uncommitted work was preserved; before editing existing files, their hashes were checked against an audit baseline and their original contents saved.

**Latest complete-suite result after the follow-up repairs: 7,117 passed, 3 skipped, zero failures, with 7 subtests passed in 290.39 seconds.** All 772 Python files in the follow-up verification manifest remained unchanged during that run. The optional neural suite separately passed all 46 tests during the main audit with all its dependencies installed. The additional defects found after that audit are documented in the follow-up section below.

## Defects fixed

| Area | Reproduced defect | Change |
| --- | --- | --- |
| Dense neural learning | Loading a checkpoint reset Adam's moment estimates but retained its old step count, producing incorrect subsequent updates. | Restart the optimiser clock with the moments; preserve training-history statistics. A numerical regression compares the next update with a correctly initialised optimiser. |
| Dense and sparse checkpoints | Incorrect parameter shapes and NaN weights were accepted; truncated NPZ archives escaped the recovery path. | Validate shapes and finite values before returning a network. Treat truncated/corrupt archives as failed loads. Six tests cover both network types. |
| Background work | `spawn()` accepted Futures, but naming, draining and failure reporting assumed every handle was a Task. | Wrap plain Futures in owned Tasks; preserve existing Task identity and propagate cancellation even before the wrapper starts. |
| Remote authentication | Non-ASCII token signatures raised `TypeError` inside `compare_digest`, producing server errors. | Reject malformed signatures before comparison. Verified through the real aiohttp HTTP boundary. |
| Remote confirmations | Non-ASCII approval credentials also raised instead of being rejected. | Reject malformed credentials while leaving the pending action untouched. |
| Approval ownership | Nested action parameters remained shared with the caller and could change after the request was created. | Take a deep copy when registering the proposed action. |
| Approval execution | The internal execution method accepted pending, denied, expired or unregistered confirmations; repeated callers could execute an approved action twice. | Require the registered approved object, recheck expiry, and synchronously consume the authorisation before awaiting execution. Concurrent replay is refused. The HTTP route already checked approvals; this closes the lower-level execution gap as well. |
| Policy changes | A previously approved plugin action could execute after its policy became desktop-only. | Recheck current capability policy at execution. |
| Annotations | Static checks found 22 unresolved references across the application, desktop dispatcher and renderer. | Restore the missing imports; keep the provider settings import under `TYPE_CHECKING` so startup remains lazy. |
| Test collection | The duplicate-suite check descended into hidden dependency caches that pytest itself excludes, then reported third-party tests as duplicate ORION suites. | Match pytest's directory exclusions and prune before descent. Regression cases still detect an actual stray project copy. |
| Signal audit | The signal-wiring check scanned third-party code in hidden environments, potentially counting unrelated code as ORION signal producers. Its setup took 188.75 seconds. | Exclude hidden tooling directories before descent. Setup took 5.20 seconds in the focused run and 3.63 seconds in the final full run. A regression proves real project sources remain included. |
| Swarm/deck coverage | An obsolete import attempted to load `clusters` from `swarm.model`; a broad exception handler silently skipped the test. | Import the real `swarm.clusters` module and allow unexpected failures to fail the check. The zone consistency test now runs and passes. |
| Neural integration tests | Tests depended on the host's actual RAM usage. Above 90%, the scheduler correctly paused work and two API tests timed out. | Control the resource probe in tests and explicitly verify pause, hysteresis and resume. Production resource limits remain unchanged. |
| Windows compatibility | The reference file guard used an API deprecated in Python 3.13 and removed in 3.15. | Prefer `ntpath.isreserved`, retaining the Python 3.12 fallback. |

The new production regression module contains 21 cases. Twelve cases failed before the initial repairs, and another six checkpoint cases failed before the persistence repair. Additional cases exercise HTTP error responses and background-task cancellation. Two collection regression cases, one signal-audit regression and one resource-pressure integration test were also added.

## Validation

| Check | Result |
| --- | --- |
| First complete production-environment run | 7,055 passed, 3 failed, 4 skipped; 7 subtests passed. |
| Complete run before test-quality improvements | 7,081 passed, 4 skipped, zero failures; 7 subtests passed in 484.62 seconds. |
| Main audit complete production-environment run | 7,083 passed, 3 skipped, zero failures; 7 subtests passed in 292.62 seconds. |
| Follow-up complete production-environment run | **7,117 passed, 3 skipped, zero failures; 7 subtests passed in 290.39 seconds.** |
| Focused repaired-path integration run | 122 passed. |
| Final focused approval, checkpoint, background, collection and shutdown checks | 80 passed. |
| Signal audit and restored swarm/deck check | 14 passed in 6.11 seconds. |
| Optional neural environment | 46 passed, including real FastAPI/SQLite/Chroma integration, local tiny-transformer generation, and a real CPU DPO adapter training step; 7 subtests passed. |
| Python syntax compilation | Passed for `orion_core`, launcher, tools, tests and neural reference. |
| Undefined-name/syntax static checks | Passed across the source tree, excluding generated exports, environments and private configuration. |
| Neural reference lint and formatting | Passed. |
| Capability inventory | Passed: no missing baseline capabilities or duplicate entries. |
| Embedded browser code | 11 embedded JavaScript/JSON blocks parsed successfully; scripts were not executed. |
| Android companion | Offline Kotlin compilation, debug lint and unit-test tasks completed successfully with Gradle 9.5.0 and Android Studio's JDK. There are no Android unit-test result files, so this is not evidence of Android behavioural test coverage. |

Two failures in the first full run were caused by editing `app.py` after pytest had imported it: `inspect.getsource()` used old line numbers against the new file. Both shutdown checks passed in a fresh process. The remaining failure was the collection defect described above. The final production run was started after production code edits were finished.

## Remaining limits and maintenance observations

- Android lint reports **66 warnings and no errors**. These include target/dependency age, Gradle deprecations, accessibility/style issues and warnings around the existing LAN/self-signed-certificate design. The reviewed trust-manager path checks an explicit certificate pin, and WebView trust requires the configured origin and a user-approved fingerprint. The static context warning points to a client created with `applicationContext`. These observations do not constitute a complete mobile security audit.
- `pip check` reports two requirements from `resemblyzer`: the legacy `typing` distribution and the `webrtcvad` distribution name. Runtime imports of `typing`, `webrtcvad` and `resemblyzer` all pass: Python supplies `typing`, and `webrtcvad-wheels` 2.0.14 supplies the VAD module. No obsolete backport was installed and no production packages were changed.
- Optional tests report an upstream Starlette/httpx deprecation and PEFT's warning about the intentionally in-memory tiny model's missing base configuration. The DPO test checks that adapter weights change, base weights remain frozen and the adapter is saved.
- Two full-suite skips are the optional API/ML modules, which pass in the separate neural environment. A symlink test requires host privileges. The previously skipped swarm/deck check was repaired and now passes. Skips are reported rather than counted as passes.
- No live paid-provider requests, real outbound messages, destructive desktop actions, phone deployment, microphone/camera end-to-end session or GPU model training were performed. CUDA/NF4 inference on the RTX 3060 Ti and full-size LoRA/DPO training still require a hardware acceptance run. The CPU integration results do not establish those capabilities.
- The existing generated-code verifier is a subprocess verification harness with resource limits; it is not an operating-system security sandbox. This audit did not turn it into one.
- The full suite uses isolated stores and the existing audit hook that rejects writes to the real `config/`. No model weights or active adapters were deployed. No commits, resets, merges or pushes were performed.

## Evidence and reproduction

Evidence is saved under `reports/audit_2026_09_26/`: baseline hashes and file snapshots, a diff against that baseline, pre-fix failures, focused test output, complete-suite JUnit results, optional neural test output, Android build output and embedded-script results. These baseline snapshots distinguish this audit's edits from work already present in the shared checkout.

Run the production suite from the repository root with its existing environment:

```powershell
$auditTemp = Join-Path (Get-Location).Path ('.audit-run-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $auditTemp) { throw 'Test directory must be new' }
$env:QT_QPA_PLATFORM = 'offscreen'
& '.venv\Scripts\python.exe' -m pytest -q -ra -p no:cacheprovider --basetemp $auditTemp
```

Run the optional integration tests with the prepared neural environment:

```powershell
& '.neural-debug-env\Scripts\python.exe' -m pytest docs/neural_reference --confcutdir=docs/neural_reference -q -p no:cacheprovider
```

Run the embedded-script syntax check without executing browser code:

```powershell
& '.neural-debug-env\Scripts\python.exe' reports/audit_2026_09_26/check_embedded_scripts.py
```

The neural design and integration instructions remain in [the architectural guide](ORION_NEURAL_ARCHITECTURE_GUIDE.md).

## Follow-up review: additional edge cases

A further inspection found three additional defect groups despite the preceding green suite. They were present in the audited snapshot, rather than introduced by subsequent changes. The extra regressions are in `tests/test_audit_followup.py`.

1. **Checkpoint metadata recovery.** Dense and sparse loaders validated weight arrays but trusted the JSON header. Non-object statistics raised `AttributeError`; malformed counters, NaN losses, infinite learning rates and invalid metadata could enter the network. A statistic named `note` could even replace the training-statistics method. Headers now validate dimensions, hyperparameters, statistics and metadata before publishing a model. Valid history and metadata survive loading, including a zero learning rate used to freeze updates.
2. **Missing background error reports.** `tasks.spawn()` indexed the first line of an exception's message. An empty message, common for `TimeoutError()`, caused the reporting code itself to fail and then silently discard the original fault. It now uses the exception type when no message exists.
3. **Test configuration-protection gaps.** The audit hook ignored the integer flags supplied by `os.open()` and only inspected a rename's destination. Low-level writes and moving real configuration files out of their directory could therefore bypass it. The hook now checks write flags and both rename endpoints, and normalises byte paths safely. Reproductions used simulated audit events and scratch paths; no real configuration files were changed.

The initial follow-up reproductions produced **27 failures and 5 passes**. After the fixes, **178 affected tests passed**. Two more valid-checkpoint compatibility cases were then added; the final follow-up module has 34 cases. The final full-suite follow-up passed **7,117 tests**, with **3 skips**, **7 subtests passed** and **zero failures** in **290.39 seconds**. Syntax parsing passed for **796 Python files**, including local custom tools; targeted static checks and the capability inventory also passed. Before/after logs and a separate follow-up patch are saved beside the original audit evidence.
