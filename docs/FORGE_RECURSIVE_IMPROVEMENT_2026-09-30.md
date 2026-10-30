# Measured Forge improvement — Mark XXXIII

## What now works

The `forge` tool has three new actions: `benchmarks`, `improve` and
`improvement_status`. Improvement runs one to three generations within a total
30–600 second budget. It snapshots the baseline, checks explicit generated
acceptance tests, runs a separate fixed benchmark, compares correctness and
resource measurements, and feeds the measured result into the next generation.
One measured experiment runs at a time to avoid stacking container memory/CPU
budgets. Callable parameters must remain compatible with the measured baseline;
a faster candidate cannot qualify by changing its input contract.
Failed candidates inform the persistent Forge lesson store. An unchanged
candidate stops the loop without repeated testing or further model calls.

Correctness takes priority. A candidate must pass every registered case,
complete repeated measurements in the same immutable image, and avoid known
per-case latency, CPU or peak-memory regressions. A correctness improvement, or
a speed gain beyond the configured 10%/absolute/noise thresholds, can qualify
for review. An ordinary passing test or a confident model statement cannot.

The best qualifying source and tests are retained as inert review JSON, outside
the dynamic tool directory. The current tool is preserved. The loop does not
activate its candidates, install dependencies, alter permissions or edit core
source. Personal Forge's existing explicit generation/activation workflow and
customer verification-only policy remain separate.

Recent run summaries and full measurements are retained in the private
`config/self_improvement/benchmarks/improvements.json` ledger, capped at forty
runs. Candidate artefacts are retained separately for review; they are not
automatically discarded with the summary history. Cancelled writes are drained
before releasing the store lock. A manual source change invalidates the basis
for adoption and requires a fresh comparison.

## Registered benchmarks

| Identifier | Contract | Independent cases | Repetitions |
| --- | --- | ---: | ---: |
| `numeric_summary_v1` | Finite-number summary, empty input, stable summation and invalid-input rejection | 10 | 5 |
| `latency_report_v1` | ORION's supplied-duration analysis: nearest-rank percentiles, partial coverage, limits and honest timing disclosures | 17 | 5 |

`latency_report_v1` is tied to the original `orion_latency_report` plugin's
calculation contract. Its installed adapter imports ORION and returns
`ToolResult`; that adapter is never admitted into the isolated pure-code trial.
Only the exact reviewed wrapper can select the separately tested standalone
kernel reference. The receipt records both source identities and explicitly
states that wrapper overhead was not measured. An unknown or edited host
wrapper is refused rather than weakening source admission.

Other tools need an independently reviewed benchmark before this measured
workflow can assess them. The catalogue does not establish correctness for
every task or a general intelligence rating. Public benchmark inputs can be
optimised for; they are not a security proof or secret hold-out evaluation.

## Runtime boundary

Every measured snapshot uses required type checking and the restricted Linux
verifier, regardless of the owner's ordinary process-verification setting.
The runner resolves the reviewed local image to an immutable Docker image ID
before generation. It never downloads an image implicitly. The container has
no network, a read-only filesystem and staged source mount, an unprivileged
user, one CPU, 64 PIDs, bounded memory and a timeout. No source checkout,
credentials, GPU or Docker socket is mounted.

The trusted measurement probe is separate from generated tests. The host
checks outputs against fixed expectations and validates the complete receipt,
source digest, case/repetition identities and runner-owned execution evidence.
Container results retain the conformance probe's actual tool schema; a printed
test dictionary or benchmark receipt cannot replace it.
Wall/CPU measurements and Linux process peak RSS are observations of this
contained Python calculation, not live phone or AI-provider timings.

Build and verify the reviewed image from the repository root:

```powershell
docker build --pull=false -f deploy/forge-verifier.Dockerfile -t orion-forge-verifier:mark33 .
$env:ORION_FORGE_CONTAINER_IMAGE = 'orion-forge-verifier:mark33'
.\.neural-debug-env\Scripts\python.exe tools/verify_forge_benchmarks.py --exercise-loop --output reports/forge-acceptance.json
```

The verifier image is already built on this PC. Docker Desktop must be running.
The script makes zero model requests. Its two-generation exercise uses clearly
labelled faulty/corrected fixtures, never replaces an installed tool, and tests
whether independent failures actually guide a later qualifying generation.

Example tool arguments:

```json
{"action":"benchmarks"}
```

```json
{"action":"improve","tool_name":"orion_latency_report","benchmark":"latency_report_v1","generations":3,"budget_seconds":300,"tool_plan":"Preserve the exact successful timing-analysis contract and reduce repeated sorting without changing percentile results."}
```

```json
{"action":"improvement_status","run_id":"the returned run identifier"}
```

The autonomous heartbeat can propose a measured review only for an exact
registered target/benchmark pair when operational evidence supports it. It
shares the session Forge cap, waits for provider availability, uses at most two
generations and attempts a target once per session. It cannot invent a
benchmark or use a timing-analysis optimisation to claim that microphone or
cloud inference latency has improved.

## Thoughts and useful evidence

Cognition now owns **Reflections**: completed concise notes, timestamps,
focus/project context and recorded evidence, with Open tasks and Clear view.
The duplicate Operations panel and its token/typewriter repaint queue are
removed. Routine generated reflections and successful maintenance activity do
not enter the terminal. Genuine failures remain diagnostic, with backoff and
repeated-error suppression.

Maintenance and Forge observations enter the same bounded snapshot, so opening
Cognition later still shows them. Clear view retains the private journal.
These are generated summaries and recorded activity, never a transcript of a
provider's hidden reasoning. When optional reflection generation is unavailable,
grounded context can supply a factual note without spending AI tokens.

## Neural training limits

The repository's small NumPy networks support intent routing and chess
intuition. Its consent-gated feedback system distinguishes a returned tool
success from independently verified postconditions. Those components can learn
from curated, correctly labelled outcomes; training them does not retrain
Gemini's hosted weights or make unrelated tasks more reliable automatically.

This phase improves Forge through independent evaluation, measured feedback and
retained lessons. It does not enable a local language model, invent training
labels, export personal conversations or claim a new chess/AI rating.

## Actual acceptance

Both reference implementations passed in the existing restricted container:
**135 observations** across 27 independent cases and five repetitions. The
complete fixture loop rejected the first faulty candidate and retained the
corrected second generation for review. No model call or live activation was
made. Receipts are retained in the ignored Mark XXXIII report directory.

See the [delivery ledger](../PLANS.md) for the final project gate and remaining
physical-phone, driving, account and Globe acceptance work.
