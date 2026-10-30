# ORION Mark XXXV performance refinement — 35.0.2

Desktop package `35.0.2`, build `2026.10.15-r2-xxxv`, follows the verified
35.0.1 refinement. Final installed acceptance and exact measurements belong to
`reports/performance_2026_10_15/REPORT.md` after its release gates pass.

This pass prioritizes launch responsiveness, task preparation, saved preferences
and research recovery while retaining the existing capability registrations:

- Thread-safe startup resource work has explicit asynchronous ownership and
  cancellation cleanup. Qt objects and native COM-sensitive setup keep their
  owning thread. A usable window, command readiness and provider readiness remain
  distinct; moving I/O off-loop is not itself a measured boot speedup.
- Routing caches retain the full and gateway menus without repeatedly rebuilding
  their indexes. Description/vocabulary changes and encoder/session changes
  invalidate the corresponding data. Retention has explicit entry/byte limits;
  an oversized menu is still scored without dropping its tools.
- Saved identity preferences survive unreadable or damaged files, unchanged
  settings avoid redundant writes, and unsuccessful persistence cannot claim a
  successful name change.
- Research publication records bind intended files to the actual run. Recovery
  verifies recorded artifacts without scanning arbitrary private directories,
  reports interrupted/partial output honestly and never replays model work.
- Provider prompts distinguish the configured research-report directory from
  legacy research folders and use resolved Windows shell folders. Saved receipts
  remain the authority for actual file identity and existence.
- Direct video-frame model fallback has a bounded attempt with explicit client
  cleanup. Deadline expiry can continue to OCR; cancellation cannot admit a late
  request after client construction. In-flight SDK work owns eventual cleanup.
- Diagnostics distinguish loose source coverage from the actual bundled module
  inventory. Invasive import checks run in an owned, bounded temporary child,
  with dependency-only checks and unavailable inventory reported explicitly.

Controlled source-startup measurements use temporary data, offscreen Qt and a
software face, with capture and real inference unavailable. Routing measurements
exclude task execution and model generation. Their reports retain conditions,
distributions and limitations; neither is a promise about a real provider, GPU,
microphone or cold disk. Read the final report for the measured outcomes,
preservation evidence, skipped tests and refreshed remaining-fixes prompt.
