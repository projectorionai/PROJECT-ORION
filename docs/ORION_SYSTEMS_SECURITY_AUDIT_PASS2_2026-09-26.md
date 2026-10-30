# O.R.I.O.N. systems audit — second pass

26 September 2026. Windows desktop; Ryzen 5 3600 and RTX 3060 Ti.

This pass examined the already modified checkout for additional reproducible defects. It preserved the first audit's Forge verification, deployment receipts, evolution history and MCP policy, and retained the existing user changes. The focus was file ownership under concurrency, recoverability, streamed-response integrity and credential handling on error paths.

The first new regression run produced **12 failures**, before the relevant production edits. Separate tests subsequently reproduced a model-download collision, an inherited-credential leak and two failures through the live HTTP streaming route. These failures are recorded under `reports/architecture_security_pass2_2026_09_26/`; none required a paid provider, real user-file organisation or an external MCP server.

## Newly established defects and fixes

| Finding | Mechanism and observed failure | Applied correction |
|---|---|---|
| High: file organisation could overwrite another file | The destination was selected during planning, then `shutil.move` was called later. A file created in that interval was overwritten, contradicting the organiser's stated contract. | Claim the destination without replacement, retry collisions, and record the actual chosen path. Windows uses no-replace rename; other filesystems use a hard link or exclusive copy. |
| High: undo evidence was lost after an unsuccessful restore | Missing files and move errors were ignored, after which the entire journal was deleted and success was reported. | Keep unresolved entries, report an incomplete undo and allow retry. Successfully restored entries are removed from the remaining journal. |
| Medium: two operations in one second shared an undo log | Second-resolution filenames caused the latter operation to replace the earlier undo record. | Add microseconds and a random suffix, and serialise organiser mutations inside the process. |
| High: MCP protocol errors leaked configured credentials | Successful/tool-error results used the connection redactor, but the JSON-RPC exception handler interpolated the exception directly into its response. | Redact protocol errors before returning them, retaining their actionable schema message and `protocol` failure kind. |
| High: explicitly inherited MCP credentials were absent from redaction | Redaction values came only from the configuration's `env` object. A secret supplied through `inherit_env` could appear in launch diagnostics or server output. | Build the redaction set from the actual filtered child environment before starting the process. |
| Medium: concurrent model downloads shared one staging file | Both workers opened `<model>.part`. A valid download could fail publication with a Windows sharing violation while the other attempt held the same file open. | Give each attempt an exclusively created temporary file; flush, verify and publish its own bytes, and clean up only its own path. Reuse bounded Windows rename retries for transient locks during publication. |
| High: incomplete model streams were accepted as successful answers | End of the HTTP body was treated as successful completion even without a terminal event. The live router also silently ignored mid-stream error events returned with HTTP 200. | Share the corrected decoder with the live router. Require `[DONE]` or an explicit finish reason; propagate stream error events. Preserve compatibility with servers that finish using a reason without a separate `[DONE]` marker. |
| Medium: streaming byte limits counted characters | Both transport decoders added `len(text)` to a byte budget. Multi-byte UTF-8 output could exceed the declared limit before rejection. | Count encoded UTF-8 bytes before buffering or emitting each fragment. Tests cover a four-byte character against a three-byte allowance. |
| High: backup archives copied live SQLite storage files independently | Copying the main file and WAL/SHM sidecars is not a consistent database snapshot. The regression demonstrates that the archived main file alone lacks committed WAL state. | Use SQLite's online backup API through a read-only source connection. Archive the resulting standalone database and omit its transient sidecars. Each database snapshot has a bounded progress deadline. |
| High: backups could replace each other or publish incomplete output | Backup names had one-second precision; writing directly to the final ZIP exposed partial archives. Individual file read failures were silently skipped. | Use distinct names and private staging; publish only after the archive closes successfully. A failed read rejects the backup and preserves previous archives. Reject destinations inside the source configuration tree to prevent self-inclusion. |
| Medium: selecting an unknown backup restored a different one | A failed substring lookup silently fell back to the newest archive. | Prefer an exact match, allow an unambiguous partial match, and reject missing or ambiguous selections. The default latest selection applies only when no name is supplied. |
| Medium: repeating a restore overwrote previously reviewed files | Extraction reused the same destination directory for an archive. | Extract to private staging and publish a fresh review directory for each restore. Reuse the existing bounded Windows rename retry for transient sharing locks. |
| Medium: relocated configuration directories could not be backed up | Archive names were calculated relative to the application directory, raising when configuration lived elsewhere. | Derive paths relative to the configured source root and preserve the canonical `config/` layout inside the archive. |

## Verification of the live path

The router still had its own SSE parser even though `provider_transport.py` contained a newer one. Testing only the helper would therefore have left the live defect intact. The added HTTP tests run the actual `ProviderRouter._openai_compatible_chat_stream` method against a local aiohttp server and reproduce both premature EOF and a JSON error event after partial output. The live method now calls the shared decoder.

This inspection also corrects a statement in the first report: `SessionPool`, `EndpointGate` and the native Ollama transport exist, but the live router's request methods do not use them. The original pooling claim has been corrected. This patch wires the decoder; it does not claim that pooled sessions, provider endpoint admission or native Ollama residency controls are deployed. Connecting those facilities requires a separate compatibility migration and end-to-end tests of cancellation, failover, shutdown and local-server identification.

## Recovery and concurrency reasoning

**File moves:** choosing an unused name and then calling an overwrite-capable operation is a check/use race. Rechecking immediately before that operation still leaves a race. The replacement operation itself must refuse an occupied destination. The organiser now retries when that refusal occurs. Its process-wide lock also prevents an undo from interleaving with an organiser publication in another worker.

**Model downloads:** a checksum over incoming bytes is meaningful only if those bytes belong to the file being published. Sharing a staging pathname breaks that ownership assumption. Temporary files are now exclusive to each invocation; rejecting one download cannot remove another attempt's candidate.

**SQLite backups:** a transaction visible through SQLite may reside partly in WAL. Opening the source through SQLite and using its backup API produces a consistent snapshot for that database. A backup of several databases is still a sequence of snapshots, not one application-wide transaction. Non-database files can also change between reads.

**Undo:** a failed operation must remain represented in durable recovery metadata. Partial undo now reduces the journal to the unresolved entries instead of erasing the possibility of retry. The organiser still records completed moves after performing them; crash-atomic journalling across filesystem moves is not introduced by this patch.

**Stream completion:** receiving bytes is not evidence that the reply finished. The parser distinguishes text fragments from completion/error evidence. Already emitted fragments may have been displayed before a later failure; the decoder reports that failure rather than accepting the partial answer as complete.

## Second-pass review of the original six phases

1. **Concurrency and resources:** isolated model-download staging; serialised organiser mutation; bounded database snapshot progress; encoded streaming limits. Existing retry, provider circuit and GPU admission code was retained.
2. **Forge verification and rollback:** existing mechanisms were exercised by the complete suite. This pass adds failure reproductions before production fixes and preserves the prior rollback boundaries.
3. **SQLite evolution history:** the history implementation was retained. The backup change now snapshots SQLite databases, including evolution/history stores under the configuration tree, using the database engine rather than copying live sidecars.
4. **Agent coordination:** the previous deduplication change was retained. Concurrency tests in this pass target overlapping downloads and file-operation ownership rather than adding another scheduler.
5. **MCP integration and security:** closed two distinct diagnostic credential paths while preserving typed errors and inherited-environment configuration.
6. **Atomic review and verification:** numbered patches pair changes with their regression tests; hashes distinguish this pass from the pre-existing dirty checkout.

## Audit artefact correction

The initial full run in this pass reported **7,200 passed, 2 skipped and 1 failed**. The failure was the repository's duplicate-test-tree guard: the first audit had saved original test files under `reports/.../before/tests`, making audit evidence eligible for pytest discovery.

`reports/` is now explicitly excluded from collection, alongside the existing generated `build/` and `release/` directories. The integrity check mirrors that exclusion. Regression cases confirm that reports are excluded while an ordinary stray test tree is still detected. This corrects an error in the first audit's artefact handling; it does not suppress production test failures.

## Dependency impact and compatibility

No runtime dependency was added. Downloads and file recovery use standard-library facilities. Backups use standard-library SQLite plus ORION's existing bounded rename retry. MCP keeps its existing configuration format; the new redaction set includes credentials explicitly inherited at launch.

Successful calls retain the same public interfaces and result types. Failure behaviour changes deliberately: truncated streams raise, incomplete backups fail, unknown/ambiguous restore names fail, and incomplete undo reports failure with a retained journal. Restoring the same archive now creates another review directory. Existing backup ZIP files and undo JSON records remain readable.

The strict regression standard is evidence, not a mathematical proof of all possible behaviour. Tests run with isolated data and mocked download responses; the streaming integration tests use a local HTTP server. Real cross-volume/network filesystem behaviour, GPU/VRAM stress, live provider accounts and hostile-code isolation are not established by these checks. The earlier container-verification limitation remains unchanged.

## Validation and patch guide

The verification logs, XML results, pre-edit files and patch manifest are in `reports/architecture_security_pass2_2026_09_26/`.

* Initial failing reproductions: `regressions-before.log`, `model-before.log`, `inherited-before.log` and `router-before.log`.
* Focused integration run: **140 passed**, including existing provider, MCP and perception tests (`focused2.log`).
* Mypy: passed on the modified download, backup and organiser modules with `--check-untyped-defs`, `--follow-imports=skip` and `--ignore-missing-imports` (`typecheck.log`). This is scoped checking, not a whole-application type proof.
* Reference examples: **25 passed, 2 skipped, 7 subtests passed** (`reference.log`).
* The first full run after the fixes found a transient model-publication lock and a missing shared SQLite-policy call. Both were corrected, with **28 targeted checks passing** (`final-fixes.log`). SQLite reader/writer connections now use the shared cache and timeout policy; a test also confirms that backing up a source in DELETE journal mode does not change that mode.
* **Final full production run: 7,228 passed, 2 skipped, zero failures**, in 374.59 seconds (`final.log`, `final.xml`). Both skips require Windows symlink privileges. The earlier failed complete run is retained as `complete.log` and `complete.xml`.
* Critical lint and the source-publication audit passed (`lint.log`, `public-source.log`). The syntax inventory covers **808 Python files and 254,035 lines**, with no syntax errors. Capability comparison found no missing declarations or duplicates. Root-level collection succeeded with the generated copies excluded (`root-collection.log`).
* All **seven patches** passed reverse checks against the edited checkout, then forward checks and application against an isolated baseline copy. All **15 resulting file hashes** matched the source. No runtime dependency was added and no installed executable was rebuilt.

The [numbered patch guide](../reports/architecture_security_pass2_2026_09_26/patches/README.md) lists the files and focused tests for each change. Patches are already applied in this checkout and are based on the saved start-of-pass files, not Git HEAD. Do not reapply them here.
