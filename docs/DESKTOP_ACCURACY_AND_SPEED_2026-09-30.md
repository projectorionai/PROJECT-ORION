# Mark XXXIII: desktop accuracy, speed and original plugins

## Desktop targeting

Named targeting resolves the foreground accessibility tree once, preferring
exact interactive controls. Equal-ranked matches return candidates instead
of guessing. Ambiguity cannot become an OCR or scroll-search click. An
observed automation ID stays a native selector, never an OCR label.

Immediately before input, including after pointer travel, verification checks
the foreground, runtime identity, name, role, enabled state, visibility,
rectangle and hit-tested control. Covered/moved targets are refused. Input
or capture failures do not repeat the click. Pixel change is evidence of a
response, not proof that an application completed the intended operation.

Physical virtual-desktop pixels retain negative monitor origins. Disconnected
monitors, invalid points and gaps between offset screens fail instead of
being clamped onto another control. Worker-scoped DPI policy is restored.
Screenshots attach `media.coordinates` with `origin`, `native_size` and
`image_size` from the same capture:

```text
desktop_x = origin_x + image_x * native_width / image_width
desktop_y = origin_y + image_y * native_height / image_height
```

Prefer names or observed automation IDs. Canvas-only applications, animated
OCR regions, inaccessible/elevated windows and overlays can still need a
fresh inspection or manual action.

### Installed local MCP

`orion_desktop` exposes `desktop_target_status`, `desktop_monitors`,
`desktop_controls` and `desktop_find_target`. It injects no input, captures
no background screenshots, and leaves input ownership with native
`desktop_control`. Discovery performs no capture. Bounds, roles, foreground
changes and ambiguous matches have explicit failure results.

The private configuration was migrated preserving existing disabled choices.
A real subprocess negotiated `2025-11-25`, listed all four tools, reported
Windows accessibility available and exited cleanly. It issued **zero clicks
and zero screenshots**. Targeting regressions use controlled doubles; actual
click acceptance across the user's applications remains manual. Frozen-build
data now includes the desktop and World MCP scripts. No new complete frozen
desktop build was produced in this phase.

## Prompt and transport speed

The old memory builder fetched every full stored value then discarded almost
all of them. It also skipped the two newest bulk-category facts after user
priority rounds. Indexed, bounded slices now preserve facts and query access
using one SQLite snapshot; user-category priority remains.

The isolated benchmark used **50,000 synthetic facts, 18 categories,
1,200-byte values and seven paired repetitions**. Median `prompt_context`
changed **457.948 ms → 3.458 ms**. Full-value transfer changed from 50,000
rows to 18 rows, plus 40 recall keys. This measures that synthetic memory
operation, not microphone, network inference, phone speech or total latency.

Provider prompt preparation runs off the live event loop within its original
deadline. Successful catalogues are shared for five minutes by endpoint,
transport kind and credential digest; waiters retain independent deadlines.
Completed SSE/NDJSON responses drain bounded trailing data briefly instead
of waiting another half second. Normal EOF still allows connection reuse.

Recognised routine Gemini requests use supported lower-thinking fields only
on the official hostname. `ORION_MODEL_THINKING=1` preserves explicit thinking
policy. Primary references: [OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
and [thinking](https://ai.google.dev/gemini-api/docs/thinking).

Following the user's Google AI Studio-only account confirmation, the private
`openrouter`, `groq` and `thoughts` profiles were disabled. All credentials,
models and integration values were preserved. Gemini Live, text/reasoning and
hosted Gemma remain enabled. This check created no new external request or
account. Availability and quota remain account-specific.

## Original installed plugins

| Plugin | Working function | Limits |
| --- | --- | --- |
| `orion_latency_report` | Calculate stage means, nearest-rank p50/p95, maxima and the largest mean stage from supplied durations. | 256 samples; 16 distinct stages; exclusive durations only. Measures no live service. |
| `orion_batch_planner` | Propose dependency-respecting batches and critical path for supplied tasks. | 64 tasks; 512 edges; concurrency 8. Writes, external/destructive/unknown tasks are serial barriers. Executes nothing. |

Both load through the native manifest/module contract with no extra account
or dependency. Receipts disclose calculation/proposal status. These are
ORION's original implementations, separate from the supplied JV bundle.

## Customer Forge verification

Personal Forge retains its activation policy. Customer verification is opt-in:

```powershell
docker build --pull=false -f deploy/forge-verifier.Dockerfile -t orion-forge-verifier:mark33 .
$env:ORION_FORGE_CUSTOMER_MODE = '1'
$env:ORION_FORGE_SANDBOX = 'container'
$env:ORION_FORGE_CONTAINER_IMAGE = 'orion-forge-verifier:mark33'
```

Customer mode requires container verification and passed type checking. It
refuses generated dependency installation and host import/activation/reload
or rehabilitation. Successful candidates are inert JSON in `forge_candidates`,
outside `custom_tools`, with hashes and `activated: false`. Malformed policy
fails closed. Review remains necessary before separate execution/publication.

Each attempt snapshots execution mode, type-check policy and image. Fast
children cannot bypass the output budget; memory-limit failures cannot pass.
Unconfirmed container cleanup produces failed verification evidence.

Docker Desktop's Linux engine was started locally and the verifier image built.
A real arithmetic candidate passed type checking, contract probing and
positive/negative assertions in **7.516 seconds**, without host dependency
installation or generated-module import. The same-command boundary probe
confirmed UID 65534, denied outbound network and `/work` writes, and measured
512 MiB memory, PID limit 64 and one-CPU cgroup limits. Only disposable files
were mounted; no checkout, credentials, Docker socket or GPU. These are tested
boundaries, not proof against every runtime defect. The owner's global Forge
mode was not changed.

## Acceptance retained

The user confirmed a Lichess account and JV redistribution permission. This
records their confirmation without inventing licence terms or removing
attribution. Lichess token, BOT status, rated games and public rating remain
unverified. Phone fixes/installations/audio/driving acceptance are deferred
until home. Cloud deployment remains prepared only. Private installations
and learned state are preserved.

Focused gate: **219 passed, one optional environment skip**. Oversized
synthetic pytest IDs initially exceeded Windows' environment-string limit;
bounded IDs corrected that harness fault. Critical Ruff and whitespace checks
passed. The final whole-project gate passed **8,622 tests with 21 skips and
zero failures in 578.39 seconds**. The separate browser gate passed 34 tests,
resolving twelve process-permission skips. Nine optional/environment skip
conditions remain; the main audit records them. The World feed service also
now defers HTTP imports and applies shared SQLite pragmas, with two new real
settings/import regressions and 113 affected tests passing.
