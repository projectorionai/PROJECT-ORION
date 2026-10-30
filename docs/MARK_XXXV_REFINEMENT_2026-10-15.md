# ORION Mark XXXV refinement — 35.0.1

Desktop package `35.0.1`, build `2026.10.15-r1-xxxv`, follows the original
Mark XXXV reliability release. The Android companion is unchanged. Final source,
installed-build validation, reproduced failures, coverage and remaining limits
are recorded in `reports/refinement_2026_10_15/REPORT.md` after acceptance.

This pass preserves the existing tools, pages, local assets and provider fallbacks.
Its changes strengthen their behavior rather than add unrelated features:

- Legacy paper writing preserves prior files, accepts external destinations and
  distinguishes an unverified draft from completed sourced research. Dossiers
  reuse guarded publication rather than deleting an altered reservation on error.
- A process-held research writer lease prevents a second reader from declaring a
  live run interrupted or replacing the first writer's records. Secondary readers
  refresh without writing; a competing writer reports busy. Process death releases
  the OS lease, and the next owner records interruption without replaying work.
- Shallow dependency discovery does not execute unloaded parent packages. Provider
  settings persistence runs off the interface loop. Cancelling a camera/screen
  watch signals its worker to stop before further captures; native work already
  in flight may still settle afterward.
- Unreadable or ambiguous MCP configuration cannot fall back to new enabled
  defaults, overwrite retained policy, or bypass confirmation on a cached server.
  Remote registry checks preserve valid legacy records while refusing ambiguous
  revocation/hash fields and duplicate JSON fields before new key creation.
- A failed tool can have partially sent, saved or changed something. Plans and
  workflows therefore retry read-only failures, but do not automatically repeat
  mutations after either an exception or an unsuccessful result.
- Workflows reject invalid or oversized definitions as a whole and retain complete
  JSON arguments. Corrupt libraries and observed external edits are preserved.
  Overlong input is rejected explicitly rather than silently shortened; unchanged
  built-in libraries need no redundant startup write.
- Finance imports commit accepted rows together, roll them back together on a
  database error and avoid creating accounts for wholly invalid input. Reports
  apply one date cutoff to cash, trailing burn and runway, while retaining future
  postings in the full ledger, and reuse calculated totals.
  Duplicate account names are refused, account kinds normalize case, missing
  accounts do not masquerade as zero balances, and spoken amounts match stored
  rounding. Invalid subscriptions and explicit zero values receive honest handling.

Regression work uses temporary files/databases, fake providers/transports/captures
and bounded disposable processes. It does not establish live account delivery,
physical device quality, research factual correctness or perfect reasoning.
Installed native host checks and frozen offline probes are separate release gates.
No measured full-application cold-start or model-inference speedup is implied by
fewer imports, commits or queries. Research multi-file crash reconciliation,
broader operation receipts, provider/device acceptance and startup profiling remain
explicit follow-up work.
