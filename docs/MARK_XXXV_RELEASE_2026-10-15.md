# ORION Mark XXXV — Reliability

Desktop package 35.0.0, build `2026.10.15-xxxv`. This release follows the 15 October
capability audit. The companion Android package has not been rebuilt in this pass.

## Research and follow-through

Research accepts work only after registering a durable run. Records contain a
run ID, topic, accepted inputs, timestamps, state and verified artifact locations.
The bounded, atomic ledger lives at `CONFIG_DIR/research_runs.json`. On a process
restart, unfinished work becomes interrupted; it is not silently replayed.
Unreadable or unsupported ledgers are preserved and block new research until
repaired rather than being replaced with an empty success-shaped record.

Use the research tool's `runs`/`history`, `status`, `stop` and `open` actions with
an explicit `run_id` to select an investigation. Multiple investigations retain
separate results. Status checks revalidate saved artifacts; a recorded path does
not guarantee the file still exists. Older documents remain discoverable through
the existing file and report searches; they are not moved into the new ledger.

Programme queues advance after a task settles. Provider failures, empty final
summaries and failed sections cannot become completed research merely because
notes exist. Cancelling during synchronous publication records cancel-requested
until the writer settles, then retains any real partial artifact while reporting
the cancelled investigation honestly. Cancellation cannot stop a writer already
executing in another thread.

Text fallback preserves bounded typed observations separately from clipped prose.
It retains exact selected paths and run IDs, and distinguishes a returned handler
from a completed operation. Fresh and reconnected session briefs read durable
research evidence without restarting work. A model still makes reasoning errors;
these records support verification and recovery rather than guaranteeing judgment.

## Responsiveness and capability evidence

Optional metrics database initialization and sampling run on workers, with
shutdown handling for a store that finishes initialization after cancellation.
Native window operations also use a worker. Geometry/minimize/maximize operations
read back actual OS state and report a refused/no-op request as a failure. Existing
ambiguous-title safeguards remain in place. Started OS or network effects may
still finish after cancellation; they are not automatically repeated.

Capability health includes evidence level, check time and readiness freshness.
Dependency discovery or registration does not establish authenticated service or
device operation. Shallow OCR health never constructs inference engines. Optional
EasyOCR/PaddleOCR fallbacks require cached model assets and compatible adapters.
Native OCR confidence is unknown when its API supplies no measured value.

Video analysis distinguishes decoder failure, unavailable recognition, recognition
failure and no detected speech. No frames and no transcript cannot establish
completed analysis; a model is not asked to invent a summary of missing evidence.
Partial usable evidence is reported with its limitations.

## Messaging and plugin compatibility

Failed async bot outcomes and refused SMTP recipients remain failures. A browser
compose fallback is a draft. SMTP acceptance proves server handoff, not final
delivery or reading. Blocking delivery uses the dispatch worker; timeouts or
cancellation cannot undo effects already accepted by a service.

Scheduled plugin vault values are no longer injected into `os.environ`. Shipped
secret-bearing adapters use the scoped helper. Third-party scheduled plugins
using temporary environment variables must migrate explicitly:

```python
from orion_core.jv_plugins.runtime import secret
token = secret("actual_manifest_plugin_name", "DECLARED_SECRET_NAME")
```

Only declared owner/key pairs are exposed to the active schedule. Missing,
wrong-owner or expired scope values never fall back to ambient credentials.
Known scoped strings are redacted from host-managed results/errors/history.
Trusted Python still runs in-process and can access user files; this is not an
OS sandbox or a guarantee against malicious plugins. Thread timeouts cannot kill
arbitrary blocked native code. A stronger process/permission mode remains work.

## Validation and retained limits

The release evidence is recorded in `reports/mark_xxxv_2026_10_15/REPORT.md` and
its machine-readable validation records after the final build. Component reports
record reproduced failures and focused regressions. Final acceptance requires the
full isolated source suite, capability preservation, a clean-source standalone
build, all 18 installed parity checks, frozen regression probes, source/runtime
hash matching, and preservation of personal files with a matching rollback.

Tests use fake providers, synthetic credentials, temporary stores and bounded test
windows. No real mail/messages, calls, purchases, security scans, provider jobs or
physical microphone/camera acceptance are performed for release verification.
No measured cold/warm startup or end-to-end inference speedup is claimed.

The earlier streamlining reduced the measured workspace from 33.784 GiB to about
17.323 GiB (partial scans with inaccessible paths). The active app was 1.278 GiB;
development environments, history, personal data and rollback are separate. This
release retains working local inference/native assets. Its final footprint and
session-created scratch cleanup are recorded with the final evidence. Further
historical/environment consolidation needs an exact retention and recovery plan.

Remaining priorities include source provenance/contradictions, broader typed
operation receipts, actual account/device acceptance, measured startup profiles,
bounded verified Stockfish acquisition, registry recovery and a genuinely isolated
plugin mode. The final report ends with a detailed continuation prompt.
