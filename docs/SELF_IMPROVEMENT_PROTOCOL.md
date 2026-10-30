# ORION self-improvement protocol

Updated 16 October 2026. This protocol describes implemented checks and the
coordination contract for future work. The enforcement column is authoritative:
a written requirement is not evidence that a runtime gate already exists.

## Purpose and ownership

An improvement must remove a demonstrated obstacle to a user outcome. Prefer
repairing or extending an existing capability to adding another tool with the
same purpose. A log message, model suggestion or successful import alone does
not establish a need or a useful result.

Every work item should identify its purpose, accountable owner, parent job,
claimed files/resources, input identities, intended output, side effects,
dependencies, acceptance criteria, time/tool/provider budget and cancellation
behaviour. Evidence belongs to that work item. The integrator coordinates shared
files; two workers must not independently publish to the same target.

| Responsibility | Purpose | Required delivery |
| --- | --- | --- |
| Coordinator | Select the smallest useful set of workers, resolve dependencies and own integration | Work contract, ownership and final outcome |
| Implementer | Make one bounded change within its claim | Candidate, rationale, changed paths and limitations |
| Independent verifier | Challenge the candidate with evidence the implementer cannot declare for itself | Reproduction, acceptance result and exact artifact identity |
| Domain specialist | Supply expertise needed by the objective | Relevant observations and a concrete contribution to the result |

These are responsibilities, not a requirement to spawn four agents for every
request. ORION already has engineering, learning, research, neuroscience, AI/ML,
security, reliability, integrations and continuity specialists. The owner's
code-only continuation implemented optional declarative work contracts in their
registry, briefings and receipts. Adding a specialist requires a
distinct useful contribution and an explicit completion contract; panel size is
a ceiling, not a target. This implementation pass used four concurrent
roles and an additional independent review role after capacity became available.

## State and evidence

Use this delivery sequence: observed need → proposed change → verified candidate
→ admitted publication → observed runtime outcome. A failure, cancellation or
uncertain external effect retains partial evidence and a named next action.
Do not silently convert partial work into completion.

| Gate | Required evidence | Enforcement on 16 October |
| --- | --- | --- |
| Need | Recurring unmet request or a concrete diagnostic/failure, with enough context to reproduce | Existing capability-gap/Forge logic |
| Reuse | Search existing tools and identify the additional behaviour | Coordination requirement; duplicate-tool retirement belongs to the Forge |
| Tool purpose | Nonempty callable description and valid object parameters | Registry now checks literal schemas using the shared contract; computed/decorated exports still require runtime validation |
| Call compatibility | Advertised arguments can be passed to the handler | Shared schema/handler contract enforces mandatory argument requiredness in registry, loader and sandbox; declared optional inputs need Python defaults |
| Detailed purpose preservation | Paths, quotes, newlines and Unicode survive generation | Registry scaffold now preserves the same purpose in manifest and callable schema |
| Fault reproduction | Dedicated tests actually fail on the original source | Required before automatic source repair |
| Improvement | The same meaningful checks pass on the candidate, without hiding checks through skips or changed collection | Required before automatic source repair |
| Identity | Source provenance, verification input snapshot and exact candidate bytes | Automatic repair captures a shared input snapshot; publication checks source/draft/candidate identities |
| Recovery | Verified original backup, atomic publication and identity-checked undo | Existing repair machinery preserved |
| Tool activation | Revalidate a detached schema/callable generation and retain an activation receipt | Loader refuses stale registrations, schema/handler mismatches and native/MCP name collisions before publication |
| Deployment ownership | One durable claim and a current source preimage | Exclusive receipt creation refuses concurrent claims and stale backups; recovery preserves newer source/module/route/schema generations |
| Specialist work | Purpose, inputs, deliverable, acceptance, owner, cancellation and optional limits | Optional work_contract is validated on disk/direct registration, included in prompts/descriptions/receipts, and can reduce existing step/read ceilings |
| Scheduled outcomes | Explicit boolean outcome and text; cancellation distinguished from failure | Supervisor records succeeded/failed/timed_out/cancelled and does not mute a capability for a cancelled run |
| Runtime outcome | Observe the actual result after loading the change | Separate acceptance; this source pass does not establish installed behaviour |

## Automatic source-repair experiment

`SelfRepairAgent.attempt_auto_repair` retains its protected-file list, small-diff
limit, per-session capacity and isolated verification. The source-change path
now also requires:

1. Reject a draft if its source has changed since generation.
2. Copy the package, matching dedicated tests and root/test conftest once. Hash
   those verification inputs. Each before/after run gets a fresh working copy
   and private configuration directory from that shared snapshot.
3. Exercise the original source. A meaningful assertion failure is required;
   an already passing suite does not demonstrate improvement.
4. Exercise the exact UTF-8 candidate bytes against the same snapshot. Retain
   the same collected test identities, pass all originally exercised checks
   and refuse a candidate that skips formerly passing/failing checks.
5. Treat timeout, missing interpreter/report, collection/setup/teardown errors,
   empty/all-skipped suites and pytest infrastructure errors as no verdict.
6. Remove inherited Python/test path overrides and externally auto-loaded pytest
   plugins. Copy the actual root hooks and check that package discovery points
   to the isolated copy in the child process that runs pytest.
7. Pass captured candidate/source hashes into locked publication. Check both
   original draft provenance and verified source identity; a mutable incident
   must not weaken those preconditions. Publish the captured candidate bytes.
8. Journal the incident ID, source/candidate/input digests, before/after verdicts
   and application/decline result. Preserve backup and undo receipts.

The verifier's public helper still returns `(True | False | None, detail)`.
Human-reviewed repair previews continue to show the candidate test verdict;
the strict before/after experiment applies to unattended source publication.
Dependency resolution retains its existing separate gates.

These are isolation measures for known project tests. They are not OS-level
network/device/process containment, proof of semantic correctness or proof that
the installed executable contains the patch. Generated tools retain their
existing runtime sandbox and admission boundaries.

## Acceptance and learning

A useful new tool needs a normal case, invalid-input case and honest failure
case. Define its output and side effects before treating it as ready. A repair
needs a reproduced failure, candidate acceptance evidence and a recoverable
previous generation. Separate source, installed, account and physical-device
acceptance in both records and the command deck.

Keep learning specific: record what failed, the supported cause, what changed,
which checks exercised it and the observed result. Repeated attempts with no new
evidence should stop consuming generation budget and return a concrete blocker
or alternative. Do not create tools from noisy logs simply to record activity.

Cancellation stops admitting new effects, reconciles owned work and retains
partial output. A cancelled await does not prove a native process stopped or an
external write rolled back. Retry uncertain writes only after reconciliation.

## Enforced capability contracts

Declarative specialist manifests may supply a complete `work_contract` object:

```json
{
  "purpose": "Diagnose a reported plugin fault from observations",
  "inputs": ["User objective", "Relevant incident observations"],
  "deliverable": "Supported cause, bounded repair proposal and remaining unknowns",
  "acceptance": ["Cite observations", "Distinguish reproduction from inference"],
  "owner": "reliability",
  "cancellation": "propagate",
  "limits": {"max_steps": 2, "max_readings": 2}
}
```

Legacy manifests inherit a default contract; their personas and routing remain
intact. Inputs and natural-language acceptance conditions are descriptive, not
automatic validators of semantic correctness. Receipts say `answered`, `briefed`
or `failed`; acceptance remains `unverified`. Step limits bound investigation
turns with the existing final closing pass. Read limits also consume the
parent's existing instrument budget and preserve its read-only authority.
Direct and closing answers retain their actual provider attribution; empty or
tool-protocol-only replies are failures. Optional observer errors cannot erase
an otherwise delivered contribution, and cancellation propagates.

The generated-tool contract now rejects a handler's mandatory parameter when
the schema advertises it as optional. The properties-only signature helper
retains its legacy API; full-schema admission uses the shared complete checker.
Autofix does not invent a semantic default for a declared optional input just
to report a complete repair. Malformed required lists produce validation
problems rather than unhashable-key exceptions.

Activation snapshots its schema and handler, checks them again before
publication, and preserves a newer completed registration if optional indexing
was slow. Cancellation before publication leaves the previous route active.
The outer Forge recovery tracks the exact imported module even when its
contract is rejected, and refuses to undo newer routes, declarations, imports
or source. A receipt is flushed before candidate source is replaced; failed
receipt admission cannot overwrite another publisher's backup. The publication
gate itself is protected from unattended source repair.

These checks coordinate ORION's publishers. They do not claim a filesystem
transaction with unrelated external editors or prove that a generated tool's
behaviour meets its stated purpose.

## Explicit remaining integration work

- Semantic duplicate capability detection and domain-specific tool-effect
  acceptance remain separate work; neither a similar description nor a valid
  schema proves two tools interchangeable or a real-world effect successful.
- Specialist work contracts do not yet acquire file/resource claims or validate
  semantic acceptance automatically. Those remain coordinator responsibilities.
- The owner deferred the UI redesign because the previews need more
  decluttering. They remain exploratory references, not an approved design or
  installed GUI change. The continuation is code-only.
- A source patch remains a source delivery until a separately verified build
  and installed-runtime acceptance establish the behaviour the user runs.
