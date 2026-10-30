# Supplied JV plugins — Mark XXXIII

The collection from `%USERPROFILE%\Downloads\JV Plugins` is installed as twelve ORION tools. Each has a module and versioned manifest in `config/custom_tools`. The implementation lives in `orion_core/jv_plugins` and receives ORION's native services explicitly at application startup.

The old JARVIS contract (`PLUGIN`, `run(parameters, player, session_memory)`) cannot be loaded as an ORION tool directly. The adapters below replace it with `get_tool_schema()` and `run(**kwargs)`. Importing or listing them does not start timers, poll Telegram, open a camera, send messages, sign into a launcher or deploy a server.

## Installed capabilities

| Supplied source | Installed tool | What works | Required setup or deliberate limit |
| --- | --- | --- | --- |
| `document_review.py` | `jv_document_review` | Structured location, issue and suggested improvement; optional provider review of supplied document text | A text provider is needed only when generating new findings; references are not falsely described as checked |
| `quiz.py` | `jv_quiz` | Persistent questions, choice-label/number grading, explanations and manual grading of alternative/open answers | Controlled through conversational tool actions; it does not claim to have opened JARVIS's missing quiz widget |
| `pomodoro.py` | `jv_pomodoro` | Repeating work/break phases, long fourth break, completion statistics, spoken prompts and cancellation | Runs while ORION is open; an interrupted phase is not counted as completed |
| `water_reminder.py` | `jv_water_reminder` | Persisted intake log, chosen goal, bounded reminder cadence and cancellation | Intake is supplied by the user; no automatic medical target or intake measurement |
| `excel_writer.py` | `jv_excel_writer` | Verified XLSX files with styled tables, range formulas and bar/line charts | `openpyxl`; provide source figures or leave unknown cells blank. Files are under `config/jv_plugins/exports` |
| `calorie_counter (4).py` | `jv_calorie_counter` | Capture the selected webcam or read a supplied image, then request a nutrition estimate through ORION's configured vision router | Vision provider; webcam capture also needs the native camera service/OpenCV. Portion assumptions and uncertainty are stated |
| `game_updater.py` | `jv_game_updater` | Supplied Steam/Epic discovery and launcher navigation; Steam install/update/progress; installed Epic launch/update request; current-user daily scheduling | Client, account and Windows accessibility support must exist. Epic progress and new Epic installation are not fabricated; licence/claim settings default off; unattended shutdown is refused |
| `chat_takeover.py` | `jv_chat_takeover` | Read a named conversation window, draft a reply, monitor changes for 1–30 minutes, send explicitly supplied final text through native messaging | Window accessibility/OCR and text provider. Background drafts remain drafts; recipient/platform/text must be supplied for an outward action |
| `telegram_remote.py`, `_telegram_ops.py` | `jv_telegram_remote` | Explicit Bot API polling, permitted private chats, outgoing messages, queued tool requests, local approval/rejection and native dispatch | Bot token and explicit chat allowlist. Other chats/groups are ignored; requests expire after ten minutes and approval is consumed before execution |
| `cloud_agent.py`, `cloud.py`, `lazyimport.py` | `jv_cloud_agent` | Pair with the existing ORION HTTPS service, inspect its task feed, delegate a supported tool, retrieve a task and request a supported reminder/briefing | A deployed ORION node. This ports the integration to ORION's protocol; it does not duplicate the old JARVIS server or claim that a VPS already exists |
| Cloud `reminder.py` | `jv_reminder` | Native persistent spoken reminders, list and cancellation | Uses ORION's existing reminder service |
| New capability | `jv_forge_trial` | Propose a named experiment with acceptance checks, explicitly run one bounded Forge attempt, inspect evidence, roll its current generation back | Existing Forge verifier and provider. Trials use unique names, preserve current tools and require review before running |

`source_inventory.json` records the supplied files' names, sizes and SHA-256 digests. Helper modules and the README are supporting files. The reviewed game updater retains its supplied implementation with namespaced imports and a bounded subprocess wrapper. The user confirmed redistribution permission on 30 September 2026. No licence or author metadata was present in the supplied collection; this records the user's confirmation without inventing terms or removing attribution.

## Runtime configuration

Runtime settings, tokens, quiz answers and wellbeing records are local and excluded from Git. Secrets belong in ORION's per-plugin vault or the relevant environment variable; never enter provider or bot keys into conversation text.

Telegram settings in `config/jv_plugins/telegram_remote.json`:

```json
{
  "allowed_chat_ids": ["YOUR_PRIVATE_CHAT_ID"]
}
```

Store `TELEGRAM_BOT_TOKEN` in the vault under plugin `jv_telegram_remote`, or set that environment variable for the ORION process. Start polling explicitly with `jv_telegram_remote(action="start")`. A permitted owner can send:

```text
/tool {"name":"sentinel","arguments":{"action":"status"}}
```

Use `pending` to inspect the exact request, then `approve` or `reject` with its identifier on the desktop. Sensitive native actions retain their native safeguards. The adapter does not expose raw Python, shell execution or a remote approval command.

Cloud setup uses the existing `deploy/hostinger_setup.sh`, `deploy/Dockerfile`, Caddy and headless entry point. A cloud account, host and credentials are still required. Pair through:

```text
jv_cloud_agent(action="connect", url="https://YOUR_ORION_NODE", pairing_code="FRESH-CODE")
```

The refresh credential is sealed through ORION's plugin vault; pairing refuses a plaintext-only vault. The device id and origin are saved without credentials. Protected calls issue fresh access tokens through `/v1/auth/token`. Redirects are refused so bearer tokens cannot move to another origin. An optional `ORION_CLOUD_URL` or `ORION_CLOUD_TOKEN` environment override is supported; remove an override as well as saved settings when disconnecting.

Delegation uses `/v1/agent/tasks` with `tool` and `args`, and retains the cloud node's permitted-tool checks. It reports a denial or missing service rather than silently using the desktop. The legacy bulk-memory synchronisation and recurring cloud reminder protocol are not exposed by the current ORION server. Use the reviewed deployment synchronisation script separately and configure a real server scheduler before adding recurring cloud work. `jobs` shows the node's native task feed; it does not invent a legacy remote queue.

Game updater settings can be stored in `config/jv_plugins/game_updater.json`, including `install_drive`, `steam_account`, `claim_free` and `accept_eula`. Licence acceptance and free-title claims require deliberate opt-in. Scheduled Windows launcher work runs as the current logged-in user; the original SYSTEM/highest-privilege schedule could not reliably interact with the user's launcher. Frozen builds use a durable `--game-updater-scheduled` entry point before the UI singleton, rather than incorrectly passing a Python script to `ORION.exe`. OS process queries and scheduling commands are now bounded. An accepted launcher operation may continue after a caller times out, so a timeout never implies cancellation of a download.

## Forge improvement workflow

1. Record the observed failure, expected behaviour and a measurable acceptance check.
2. Propose a distinct trial capability. Core files, keys, permission policies and existing tools are outside its scope.
3. Review the proposal, then run it once. Forge supplies contract validation, sandbox checks and its diagnosis-driven repair budget.
4. Inspect the result and exercise the acceptance checks on examples excluded from generation.
5. Keep a verified trial for further evaluation, or roll its current generation back. A revised experiment receives a new proposal.

The new adapter has no background recursive loop and does not claim that its tests prove general intelligence. Forge's existing generation rollback rejects a stale receipt if newer code or a newer handler has replaced the trial. Runtime rollback receipts exist within the current process; the deployment ledger retains recovery evidence after a restart.

Reserved `trial_` tools now have an enforced, fail-closed source admission policy through Forge's existing security hook. Both the module and explicit acceptance tests must use pure functions, JSON/value data and enumerated standard-library exports. Network, subprocess, device and synthetic-input imports are rejected, as are file access, dynamic imports, reflection, `eval`/`exec`, private attributes, dynamically selected callables and dynamic `str.format` traversal. Trial requirements and dependency installation are refused. Imported module objects cannot be passed around to expose their internal namespaces. Generation, every LLM or mechanical repair, every verification pass and publication reapply the gate; persisted loading checks the exact source bytes subsequently compiled, including rehabilitation from quarantine. Syntax, source-size and AST-size failures are rejected rather than skipped. Ordinary Forge tools retain their existing, broader policy.

This is a source admission filter, **not a Python or OS sandbox**. It does not prove immunity to Python/interpreter or approved-library defects, resource exhaustion, a compromised Python environment, or hostile objects supplied by another trusted caller. Trials are intended to analyse JSON/value inputs. Manifest permissions remain capability disclosures rather than containment. Existing verification and rollback still apply; use the restricted verifier account/container and review generated code before allowing customer-submitted code. Passing model-written tests is not independent proof that a capability is correct or intelligent.

## Further useful additions

| Addition | Practical improvement | Acceptance check before release |
| --- | --- | --- |
| Provider budget and latency policy | Route by task capability, known quota cooldown and measured first-token time | No unsupported model request; no repeated retry against an exhausted daily quota |
| Failure-to-regression collector | Turn reproducible incidents into redacted fixtures for the Forge proposal ledger | Secrets and personal messages absent from fixtures; bug demonstrably fails before its fix |
| Capability contract catalogue | Show which tools are usable on this phone, this PC or an existing cloud node | A denied or absent host capability cannot be reported as completed |
| Personal routine engine | Combine explicit reminders, focus blocks and calendar availability | Time-zone transitions, duplicate alerts and cancellation tested |
| Driving readiness checker | Check downloaded map coverage, current positioning and an installed offline routing engine before travel | Entire intended route works with data disabled; a satellite basemap is not mislabelled as turn-by-turn routing |
| Evidence-based memory consolidation | Consolidate repeated facts with provenance, freshness and user correction | Contradictory facts remain identifiable; personal history is never trained into customer examples |

These are proposals for additional work, rather than claims of installed features. Adding arbitrary MCP servers is not a substitute for capability tests, customer-owned credentials and clear failure results.

## Verification

The focused regression suite checks side-effect-free discovery, schema/manifest/signature agreement, Gemini-compatible free-form parameters, hidden quiz references and scoring, duplicate timer cancellation, real XLSX/chart generation, formula injection refusal, Telegram chat restrictions and single-use approval, cloud payload compatibility, honest missing configuration, native reminder forwarding and one-run Forge trials. Gaming checks read actual temporary client manifests, observe a mocked OS launch request, reject unknown/malformed AppIDs, retain safe defaults on malformed settings and validate the frozen scheduled command. Existing plugin registry and manifest gates are also rerun: **80 tests passed**. Live Telegram sends, paid/free game acquisition, real desktop conversation sends and VPS deployment are not performed by these tests.

The new native adapters pass the focused Ruff check. The retained supplied launcher integration has 72 pre-existing lint/style findings, predominantly broad Windows fallback exception handling; these are retained separately from functional test results and remain a refactoring concern before customer redistribution.

The subsequent combined Forge/security/plugin gate passed **484 tests**. The final dedicated trial-policy suite passed **79 tests**, including a real verifier run for a pure JSON analysis module with positive and invalid-input assertions. Its rejection cases cover I/O imports and aliases, module namespace leakage, reflection, dynamic formatting/callables, unsafe acceptance tests, oversized or unparseable source, declared/missing dependencies, LLM/legacy/mechanical repairs, pre-publication, persisted loading, source-snapshot changes and quarantine rehabilitation. The new policy and tests pass Ruff; critical Ruff checks also pass on the touched Forge/security/loader integration. These tests make no live generation, provider or secret-bearing network requests.
