**O.R.I.O.N. local neural architecture and implementation guide**

Prepared for integration, 26 September 2026. The code under `docs/neural_reference/` is an opt-in reference package. It does not change the running assistant or replace the existing modules. UK English is used throughout; library identifiers retain their original spelling.

**1. Architectural recommendation**

Build a neuro-symbolic control system around a pretrained local transformer. Retain the small online networks for intent classification and routing. The transformer proposes bounded plans; deterministic code decides whether each action is valid, authorised and complete. Long-term memory, workflow state and model weights have separate lifecycles.

Training a foundation model from scratch on this desktop is not a practical route to advanced reasoning. A pretrained model, retrieval, typed tools, external verification and carefully evaluated adapters provide a much stronger starting point. Personal fine-tuning can improve recurring behaviours; it does not establish general intelligence or guarantee reliable autonomous reasoning.

The inspected checkout already has useful foundations:

| Existing module | Integration recommendation |
|---|---|
| `orion_core/neural.py` | Retain its NumPy MLPs and replay memory for inexpensive online learning. |
| `orion_core/intent_brain.py` | Use its learned tool scores to shortlist capabilities before transformer planning. |
| `orion_core/semantic.py` | Reuse its BGE-small ONNX encoder on the CPU. Avoid loading a second embedding model. |
| `orion_core/memory.py`, `fact_memory.py` | Keep facts and explicit preferences authoritative; add vector retrieval as an index. |
| `orion_core/tool_gateway.py`, `dispatcher.py` | Preserve existing permission checks and verification when integrating tool execution. |
| `orion_core/workflow_engine.py` | Extend with dependency graphs and durable state; avoid two independent owners of one job. |
| `orion_core/resource_governor.py` | Admit inference, indexing and training according to resource pressure. |
| `orion_core/verification.py` | Feed independently observed outcomes into task state and learning. |

The checked-out HTTP server uses `aiohttp`, including `remote.py`; it is not currently a FastAPI application. The reference supplies the requested FastAPI adapter while keeping the core independent of either HTTP framework. These observations describe inspected code, not a full security audit.

```mermaid
flowchart TD
    U[User request and authenticated session] --> A[FastAPI or existing aiohttp adapter]
    A --> C[Trusted capability policy and task budget]
    A --> M[Preferences, episodic memory and semantic retrieval]
    M --> N[Local transformer planner]
    C --> N
    N --> V[Schema, logic and DAG validation]
    V --> Q[Durable priority scheduler]
    Q --> B[Windows execution broker]
    B --> O[OS observations and postcondition checks]
    O --> Q
    O --> N
    O --> T[Telemetry and curated feedback]
    T --> M
    T --> L[Separate LoRA or DPO training process]
    L --> E[Held-out evaluation and promotion gate]
    E --> N
```

For production, separate inference, orchestration and desktop execution into processes. The Windows broker should run in the logged-in interactive session, without administrator privileges. Only the broker receives OS capabilities. The trainer runs separately with no desktop-control credentials. The reference combines orchestration and the restricted executor in one process for readability; it is not an OS security sandbox.

**2. Hardware and model budget**

Use a 4-bit **Qwen3-4B-Instruct-2507** as an initial planner candidate. It is a 4B instruction model with a non-thinking output mode; choose it based on O.R.I.O.N.'s held-out tasks, rather than treating model size as proof of reasoning quality. Its advertised long context is not a sensible default on this GPU. See the [official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

Assuming the standard 8 GB RTX 3060 Ti configuration, start with these engineering budgets, then measure actual allocation and latency:

| Component | Initial setting |
|---|---|
| Transformer weights | NF4 4-bit, FP16 computation; raw 4B weights alone are roughly 2 GB before other allocations. |
| Prompt and generation | At most 4,096 input tokens and 1,024 output tokens in the reference. |
| GPU concurrency | One generation at a time; no simultaneous local training. |
| GPU headroom | Aim to reserve 1.5–2 GB for the desktop and other applications; reduce context/model size if measured usage requires it. |
| Retrieval | Existing BGE-small ONNX encoder on CPU; 4 relevant memories initially. |
| CPU workers | Start with 2–4 compute threads and preserve capacity for audio/UI. |
| Training | Microbatch 1, low-rank adapters, gradient accumulation and checkpointing. |
| Host RAM/storage | Measure available RAM; 16 GB can be restrictive and 32 GB provides more headroom. Store databases on a local SSD. |

Weights are only one allocation: KV cache, activations, quantisation metadata, temporary buffers, vocabulary logits and optimiser state also matter. Do not promise a tokens-per-second figure without measuring the actual driver, wheel, model and context combination.

4B inference is the first target. 4B DPO at useful sequence lengths may exceed available VRAM. If the memory probe fails, use a smaller model for both inference and its adapter, reduce genuinely unnecessary context, or train the 4B adapter elsewhere. An adapter trained for a smaller architecture cannot be attached to the 4B planner.

**3. Python libraries and installation**

| Layer | Libraries |
|---|---|
| Inference | `torch`, `transformers`, `accelerate`, `bitsandbytes`, `safetensors` |
| Adapter training | `peft`, `trl`, `datasets` |
| HTTP and validation | `fastapi`, `uvicorn`, `pydantic` |
| Retrieval | `chromadb`; existing `onnxruntime` and `tokenizers`; optionally `sentence-transformers` for another encoder |
| Task logic | Standard-library `asyncio`, `graphlib`, `heapq`, `sqlite3`; optional `z3-solver` for scheduling/resource constraints |
| Windows control | `psutil`, `subprocess`, `pathlib`, `os`, `pyautogui`; application-specific UI Automation through `uiautomation` or `pywinauto` |
| Observability | Existing O.R.I.O.N. telemetry; optional `prometheus-client`, OpenTelemetry and `nvidia-ml-py` |

Use an isolated CPython 3.12 environment as the initial compatibility target. Install a CUDA-enabled PyTorch wheel using the [official Windows selector](https://pytorch.org/get-started/locally/) for the installed driver. Verify `torch.cuda.is_available()` and a small GPU allocation before downloading a model. Installing a CUDA toolkit alone does not turn a CPU-only PyTorch wheel into a GPU build.

Current official documentation includes Windows NVIDIA support for bitsandbytes. The old blanket claim that it requires Linux is no longer appropriate; the exact wheel/driver combination still needs a smoke test. See [bitsandbytes hardware compatibility](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).

`neural_reference/requirements.in` contains compatibility starting points, including TRL 0.24.0 for the training example. It is **not** a resolved CUDA deployment lockfile. The debugging pass now includes an isolated CPU test environment, an exact installed-package snapshot in `neural_reference/debug-environment.txt`, and a validation report in `neural_reference/DEBUG_VALIDATION.md`. Resolve the deployment environment separately and run CUDA inference/training smoke tests before freezing it. Keep serving and training environments separate where practical.

Provision the model once from a reviewed repository revision. Record the repository, commit, file hashes, licence, tokenizer revision and package lock. Runtime loads use `local_files_only=True`, `trust_remote_code=False` and safetensors. Do not allow a model-generated repository name, Python module or adapter path to be loaded dynamically.

**4. Neuro-symbolic planning and sequential reasoning**

The reasoning pipeline is:

1. Authenticate the request and derive permitted tools from trusted session policy.
2. Retrieve explicit preferences and relevant past episodes.
3. Ask the local transformer for a short structured plan, dependencies and necessary observations.
4. Validate every argument and the entire graph before dispatch.
5. Check live preconditions immediately before each action.
6. Execute one bounded action and verify its postcondition.
7. Use observations to continue, revise the pending plan, or request missing information.

This provides the useful operational part of chain-of-thought prompting: task decomposition and sequential planning. The public artefact is a concise action plan and its evidence, not a transcript of private internal reasoning. Verbose reasoning text is neither a permission grant nor proof that a plan is correct. The selected model can produce structured plans without a special hidden-thinking mode.

`contracts.py` defines a discriminated union of permitted actions, forbids extra fields and rejects duplicate identifiers, missing dependencies and cycles. `NeuralEngine` in `neural.py` owns the local model, serialises GPU access and validates the generated JSON. Malformed output fails closed; it cannot become a partial executable plan. A production constrained decoder can reduce malformed JSON, but the validator remains necessary.

Example validated plan:

```json
{
  "summary": "Check resources and create the requested note.",
  "steps": [
    {"id": "inspect", "depends_on": [], "action": {"tool": "resources"}},
    {"id": "note", "depends_on": ["inspect"],
     "action": {"tool": "create_text", "path": "note.txt", "text": "Meeting at 10:00."}}
  ]
}
```

The note's contents are already known. If it must contain measured CPU usage, first run the inspection phase, return its observation to the planner, and only then construct the note action. Dependencies alone do not bind outputs into later arguments. The reference intentionally does not implement string interpolation such as `${step.output}`; add a typed, validated reference mechanism if required.

Symbolic rules express actual invariants: a prerequisite must have succeeded; a destination must remain inside its granted root; only one task may own the desktop; and completed actions cannot change during replanning. For time-window/resource optimisation, add Z3 constraints over start times, durations and exclusive resources. Keep permission checks in ordinary auditable code even if a solver is introduced.

Use bounded recovery: at most one schema-repair attempt and, initially, three planning phases per goal, with a token/time/tool budget. Do not automatically retry irreversible operations. Unsupported requests should produce a structured clarification or escalation in production; the reference reports a validation failure when it cannot obtain an executable plan.

**5. Semantic memory and preference management**

`memory.py` supplies the requested local retrieval mechanism. It stores vectors in persistent Chroma with cosine distance, performs embedding and database calls away from the event loop, and filters results by authenticated owner. Chroma's [persistent client](https://docs.trychroma.com/reference/python) and [collection configuration](https://docs.trychroma.com/docs/collections/configure) support this pattern.

Memory IDs are hashed together with their owner before upsert, so identical IDs from different owners cannot overwrite each other. Vectors are checked for shape, finite values and non-zero norm before normalisation. The corrected implementation uses `orion_v2_<embedding_version>` collections; re-ingest authoritative records into these new collections if an earlier reference version was used. It does not silently mix old, unscoped IDs with the corrected schema.

Use four distinct stores:

| Memory | Contents | Update policy |
|---|---|---|
| Working state | Current goal, graph, observations, deadlines, receipts | Updated after every transition; journalled. |
| Explicit preferences | Language, preferred applications, formatting, permitted working roots | Versioned structured records; user corrections take effect immediately. |
| Episodic memory | Short, verified summaries of completed tasks and useful failures | Curated after execution; retrieval supplies context. |
| Procedural memory | Reviewed tool recipes with schemas, tests and prerequisites | Versioned artefacts; never executable text copied from retrieval. |

For this checkout, pass the already-warmed `orion_core.semantic.ENCODER.encode` to `SemanticMemory`. Keep SQLite/FTS and structured facts as the source of truth; Chroma can be rebuilt as an index. FAISS is a valid alternative, but you must supply durable metadata, identity filtering, deletion and index synchronisation yourself.

Do not conflate retrieval with learning weights. A correction such as “use UK spelling” belongs in the preference store immediately. A change in output style across many independent examples may later justify adapter training.

Retain source IDs, timestamps, ownership, sensitivity, validity intervals and superseded IDs. The reference includes ownership, source, kind, timestamp and an active flag; expiry, supersession transactions, deletion and retention remain integration work. Correct a fact by updating its authoritative record and replacing or invalidating its vector entry. Deletion must also reach training snapshots and backups according to retention policy.

Chunk documents before ingestion, keep retrieval within a token budget, and combine semantic similarity with lexical/identifier matching. Start with four results; calibrate any distance threshold on labelled examples. Cosine distance is not a probability of truth. Re-embed into a new collection whenever encoder revision, pooling, normalisation or dimensionality changes, then switch the index pointer after validation.

Retrieved documents, webpages and tool outputs remain untrusted input even when labelled “memory”. They cannot create permissions or change the system prompt. Do not store passwords, API tokens or raw clipboard/screenshot material by default. The reference journal contains task arguments/results and therefore needs restrictive local access and a retention policy; it is not encrypted or tamper-evident.

**6. Windows execution and safety boundaries**

`LocalPCAgent` in `pc_agent.py` demonstrates resource monitoring, bounded text reads, exclusive text-file creation, fixed Notepad launch, a fixed `nvidia-smi` diagnostic and optional screenshots. Every action is selected from a registry. There is no model-supplied shell string, executable name, working directory, environment map or Python evaluation.

The reference file policy rejects absolute/UNC/device paths, traversal, alternate data streams, reserved Windows names, executable extensions, reparse points and hard-linked existing files. It uses exclusive creation so an existing file is not silently overwritten, then verifies the resulting content hash. A failed/partial write remains an incident to inspect; it is not automatically deleted or overwritten.

These checks defend against malformed model output. They do **not** eliminate a race with a malicious process running under the same Windows account. For a stronger broker, use a restricted token/account, NTFS ACLs, handle-based identity checks, reparse-aware opens, and Windows Job Objects for child-process limits. Check the final object through its handle, and protect relevant directory components from substitution. See Microsoft's [reparse-point guidance](https://learn.microsoft.com/en-us/windows/win32/fileio/reparse-points-and-file-operations). A Python path-prefix check alone is not a sandbox.

Use a server-issued capability record with principal, goal ID, tool names, resource roots, allowed effect, quota, expiry and plan revision. For actions requiring approval, bind the one-use approval to a canonical action digest and the relevant file/window state. A changed plan or destination invalidates that approval. The model must never be able to submit `approved=true` or invent the principal. The reference demonstrates static single-user capability grants; dynamic approval tokens are not implemented.

Suggested autonomy tiers:

| Operation | Policy |
|---|---|
| Resource probes, ordinary reads in a granted working directory | Autonomous. |
| Creating new data files in a dedicated workspace | Autonomous under the configured capability and quota. |
| Replacing/moving files | Stage a diff/manifest, verify preconditions, maintain recovery data; obtain approval unless this exact scope is already granted. |
| Desktop interaction | Named application, bounded UI transaction, foreground/element verification and postcondition check. |
| Deletion, administrator operations, service/registry/security changes, external messages | Separate narrowly scoped capability and explicit authorisation; absent from this reference. |

Prefer application APIs or UI Automation over raw coordinates. Windows services in Session 0 should not be expected to control the user's desktop. The broker belongs in the interactive session. Serialise all GUI actions with a desktop lease and yield when the user takes control. Screenshots or window text can contain instructions from other applications; treat them as observations only.

For an application-specific UI Automation adapter, the reviewed wrapper can look like this:

```python
def replace_editor_text(pid: int, text: str, expected_title: str):
    # pid and expected_title come from a trusted app-session lease, not the LLM.
    from pywinauto import Application
    app = Application(backend="uia").connect(process=pid)
    window = app.window(title=expected_title)
    window.wait("exists visible enabled", timeout=5)
    editor = window.child_window(auto_id="KnownEditorId", control_type="Edit")
    editor.wait("exists enabled", timeout=5)
    editor.set_edit_text(text)
    if editor.window_text() != text:
        raise RuntimeError("Editor postcondition failed")
```

This is a template requiring the actual application's inspected automation ID and supported Value/Text pattern. It is not a universal Notepad implementation. Wrap it in the same policy/lease checks before registering it. Test against the application's current version. For the PyAutoGUI fallback, retain `FAILSAFE=True`, verify coordinates against a fresh screenshot and known window rectangle, and check state again afterwards. Focus can still change between checks; PyAutoGUI does not provide a security boundary. See its [official documentation](https://pyautogui.readthedocs.io/en/latest/).

**7. DAG scheduling, priority and interruption**

`scheduler.py` supplies an asynchronous priority manager. It admits at most 128 active jobs, builds the ready set from satisfied dependencies, and rebuilds a small priority heap at each dispatch. Smaller priority numbers run first; ageing gradually advances older jobs. This avoids obsolete heap entries after reprioritisation. The example serialises desktop execution; production can add bounded parallel read-only workers while retaining exclusive leases for the desktop and GPU.

Each transition is stored through `journal.py` in SQLite WAL mode. A durable claim precedes the OS action; a receipt follows it. A persistence error stops further dispatch. Keep this database outside OneDrive/network-synchronised folders. Completed jobs also need archival/retention because the reference keeps their snapshots in memory.

The state vocabulary is `pending`, `running`, `succeeded`, `failed`, `blocked`, `cancelled` and `uncertain`, with a separate pause flag. A failed dependency blocks its descendants. Unrelated branches can still finish. On restart, in-flight steps become uncertain and recovered jobs remain paused.

“Exactly once” execution cannot generally be guaranteed across SQLite and arbitrary Windows applications. A crash may occur after a file was created but before its receipt was committed. Reconcile its expected hash/window/app state; do not blindly replay it. The reference refuses to resume uncertain jobs. A trusted reconciliation endpoint with evidence validation remains production work.

Pause and cancellation are cooperative. The current atomic action finishes, then the scheduler stops dispatching that job. Killing an `asyncio` task does not undo a file write, and cancelling a thread await does not stop its underlying work. The neural engine retains its admission slot until an outstanding generation finishes. Production should place inference in a supervised process for recovery from a hung native call. See Python's [task and cancellation documentation](https://docs.python.org/3/library/asyncio-task.html).

Replanning requires a paused, quiescent job. Completed steps remain identical; only pending work is replaced. Failed attempts use new IDs, including across multiple revisions. The scheduler compares a persisted revision number inside its transaction lock, so a concurrent control request cannot be overwritten by a stale plan. New snapshots become visible only after the journal commit. Repeated cancellation waits for an in-flight commit or generation to finish before releasing ownership. A revised plan remains paused until the caller resumes it; an authorised goal supervisor may make that decision automatically within the original scope.

The included RAM-pressure watcher pauses dispatch above 90% usage and resumes it below 80%. These are initial tunable thresholds. Integrate GPU utilisation/free memory, audio latency and user activity through O.R.I.O.N.'s existing resource governor. Critical user stop controls must remain responsive even during a resource pause.

For longer autonomous goals, add a durable goal supervisor above this scheduler:

```python
async def advance_goal(goal, state, planner, scheduler, verifier, policy):
    # state and each submitted job ID must be committed in the same goal store.
    if state.user_paused or state.phases >= policy.max_phases:
        return "paused"
    if await verifier.goal_complete(goal, state.receipts):
        return "complete"
    proposal = await planner.plan(
        goal, state.memories, state.observations, policy.allowed_tools,
    )
    policy.validate_remaining_scope(proposal, state.receipts)
    # Check action fingerprints to avoid replaying completed effects.
    job_id = await scheduler.submit(state.owner, proposal, state.priority)
    await state.record_submission(job_id, proposal)
    return "waiting_for_observations"
```

This supervisor sketch names application-specific interfaces that must be supplied; it is not part of the executable reference. Resume it on persisted scheduler events. Goal completion must come from deterministic checks or user acceptance where possible, not the model's self-assessment. Set an overall deadline, maximum phases, maximum tool calls and escalation policy. The executable API currently schedules one planning phase and exposes explicit replanning.

**8. Continuous learning without uncontrolled drift**

Separate three speeds of improvement:

1. **Immediate:** update an explicit preference or corrected fact; no gradient step.
2. **Online and inexpensive:** update the existing intent router or a small strategy scorer from independently verified outcomes, using bounded replay.
3. **Periodic:** train a candidate LoRA adapter on curated demonstrations/preferences, evaluate it, and promote it only if it improves the required behaviours without unacceptable regressions.

For the third level, choose DPO before PPO-style RLHF. DPO uses paired preferred/rejected responses and a frozen reference policy, avoiding a separately trained reward model and rollout-heavy PPO loop. It is still optimisation against a dataset, not autonomous learning from arbitrary conversation text. The [DPO paper](https://arxiv.org/abs/2305.18290) describes the objective:

\[
\mathcal L_{DPO}=-\mathbb E\log\sigma\left[\beta\left(
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{ref}(y^+\mid x)}-
\log\frac{\pi_\theta(y^-\mid x)}{\pi_{ref}(y^-\mid x)}\right)\right].
\]

The positive and negative answers must address the same request and relevant context. A failed network call is not automatically a bad plan. A user's correction can supply a preference label; a deterministic verifier can supply a narrower task-success label. A successful tool invocation alone does not prove the requested goal was satisfied.

Record an immutable feedback event containing: original prompt/context snapshot, proposed plan, observed tool receipts, correction, accepted/rejected response pair where available, feedback source, consent for training, base/adapter/policy versions, and task/session IDs. Keep full sensitive context in the protected store rather than telemetry labels. Do not use a model's own invented success label as training truth.

An eligible conversational preference example has this shape:

```python
example = {
    "prompt": [{"role": "user", "content": "Create a new note; keep my existing note."}],
    "chosen": [{"role": "assistant", "content": "Create a new file after checking the destination."}],
    "rejected": [{"role": "assistant", "content": "Replace the existing note."}],
}
```

For the planner, use complete schema-valid plan outputs rather than these explanatory sentences. Preserve the actual system/tool context, tokenisation and chat template used for the task. Pair-level labels and provenance must be reviewed by the curation process; schema-valid does not mean safe or preferred.

`train_dpo.py` implements a fresh NF4/LoRA candidate using PyTorch, Transformers, PEFT and TRL. Its initial settings are rank 8, `q_proj`/`v_proj` targets, microbatch 1, accumulation 16, learning rate `5e-6`, one epoch and precomputed reference log probabilities. These are starting hyperparameters, not measured optimums. PEFT documents [quantised adapter training](https://huggingface.co/docs/peft/developer_guides/quantization); the API example is tied to [TRL 0.24.0](https://huggingface.co/docs/trl/v0.24.0/dpo_trainer).

The default token budgets are 768 prompt tokens and 1,024 total tokens. Before allocating the GPU model, `training_data.py` rejects empty splits, malformed messages, identical/token-equivalent answers, duplicate pairs, exact tokenised-prompt leakage between training and validation, and oversized examples. It never silently cuts off schemas or preferred answers. Session-level and semantic deduplication still require curation. Real planner prompts may exceed these values. Increase budgets only after profiling VRAM, or reduce the serving schema/tool shortlist consistently. Do not train on truncated prompts that differ materially from inference. Candidate directories are reserved atomically to prevent two runs overwriting the same output.

Each run starts a new adapter from the same immutable base and trains on the curated current-plus-replay dataset. Thus the adapter-disabled base is the intended frozen reference. If continuing an existing adapter instead, explicitly preserve the desired frozen reference adapter/model and validate that TRL uses it; `ref_model=None` is not a universal instruction to copy whatever adapter was previously active.

Suggested anti-forgetting policy: start replay sampling at 50% recent corrections, 30% diverse older tasks and 20% fixed general/schema/safety anchors. These proportions require tuning. Balance by task family so repeated phrasing from one conversation cannot dominate. Keep weights frozen outside LoRA, use a small learning rate, limit training duration, and evaluate retained skills after every candidate. LoRA alone does not prevent forgetting in the model-plus-adapter behaviour.

Split by entire task/session and reserve a chronological hold-out. Deduplicate near-identical examples before the split. Keep a final test set out of hyperparameter selection. A useful first curation target is a few hundred diverse high-quality pairs; it is a planning heuristic, not a sufficient-data guarantee. Where only correct demonstrations exist, begin with supervised fine-tuning rather than manufacturing weak negative examples.

**9. Training admission, promotion and rollback**

The learning supervisor should trigger only when enough eligible feedback has accumulated and the machine is idle with adequate memory. Its execution sequence is:

1. Snapshot and version the eligible dataset; preserve the previous active adapter.
2. Stop admitting GPU inference, drain requests, then unload/terminate the inference process.
3. Start a training child process with fixed arguments and a resource/time budget.
4. Train into a new candidate directory. A failure leaves the active adapter untouched.
5. Run held-out evaluations against the current adapter and frozen base.
6. If the preconfigured promotion gate passes, update the active manifest atomically.
7. Start inference with that adapter, run a smoke test, and roll back on failure.

`torch.cuda.empty_cache()` does not release memory held by live model tensors. On an 8 GB desktop, plan for sequential training/inference. Native Windows can be tried first; a separate WSL2 training environment is a fallback for package compatibility, while Windows desktop automation remains in the host session.

The manifest should record adapter/base hashes, tokenizer revision, dataset snapshot, exact dependencies, policy version and evaluation results. `NeuralEngine` accepts an optional adapter directory at start-up; only the trusted manifest should supply it. Never train the live adapter in place, automatically merge it into the base, or let model output edit its own promotion rules.

A promotion gate should require: no failures on the fixed prohibited-action suite; retained schema and tool-selection performance; task success within an agreed non-inferiority margin; measurable improvement on the targeted correction set; and acceptable p95 latency/VRAM. Use paired evaluations and uncertainty intervals; a few successes do not establish improvement. The test suite is evidence about its cases, not proof of universal safety.

The training command, candidate loading and evaluation design are provided here. Automatic dataset curation, training admission, an independent evaluation runner, manifest promotion and rollback supervision must still be integrated before enabling unattended weight updates.

**10. Telemetry that improves future approaches**

The reference journal records attempts, tool outcome, duration and job snapshots. The FastAPI planner receives aggregate tool statistics alongside observations, allowing it to adjust future proposals. Keep this advisory: poor historical tool performance does not override a user's required tool, and high success does not authorise a new action.

Extend each event with goal/job/step IDs, provider/base/adapter versions, policy version, action fingerprint, error category, retry count, pre/postcondition evidence IDs and user correction. Separate `tool_returned_ok`, `postcondition_verified` and `goal_accepted`. Notepad process creation, for example, is not proof that a document was edited successfully.

Track goal success by task family, dependency failures, invalid plans, denied actions, human interventions, retrieval recall on a labelled set, p50/p95 latency, token use, GPU peak allocation and rollback frequency. Include denominators. Do not put file content, user names or credentials into metric labels.

For a small online strategy selector, use a smoothed success estimate `(successes + 1) / (attempts + 2)` and penalise latency. Restrict selection to already-approved alternative strategies and require enough samples before changing the default. Use small exploration budgets only for harmless actions. This is a lightweight contextual decision layer, not a substitute for DPO. Freeze permission and evaluation policy regardless of learned strategy scores.

**11. FastAPI integration**

The executable package comprises:

| File | Responsibility |
|---|---|
| `contracts.py` | Strict tool schemas and DAG invariants. |
| `neural.py` | Quantised model, bounded generation and optional candidate adapter loading. |
| `memory.py` | CPU embedding callback and persistent vector retrieval. |
| `pc_agent.py` | Capability-limited Windows operations. |
| `journal.py` | Durable snapshots and small telemetry events. |
| `scheduler.py` | Priorities, dependencies, pause/cancel, replanning and conservative recovery. |
| `api.py` | Authenticated `/neural` routes and lifecycle management. |
| `train_dpo.py` | Separate candidate-training command. |
| `test_reference.py` | Dependency-light contract, execution-policy and scheduler checks. |
| `async_utils.py` | Cancellation barriers for commits and workers that cannot safely be abandoned. |
| `training_data.py` | Preference validation before GPU allocation. |
| `test_regressions.py` | Cancellation, revision, durability, file-policy and live system-probe tests. |
| `test_integration.py` | Real FastAPI, SQLite and persistent Chroma checks. |
| `test_ml.py` | Tiny local CPU transformer and DPO/LoRA integration checks. |

Example application factory, after provisioning model files and the existing encoder:

```python
# Save as an application-specific launcher once the reference package is adopted.
import os
from pathlib import Path
from orion_core.semantic import ENCODER
from docs.neural_reference.api import create_app
from docs.neural_reference.journal import Journal
from docs.neural_reference.memory import SemanticMemory
from docs.neural_reference.neural import NeuralEngine
from docs.neural_reference.pc_agent import LocalPCAgent
from docs.neural_reference.scheduler import Scheduler

def build_app():
    data = Path(os.environ["LOCALAPPDATA"]) / "ORION" / "neural"
    if not ENCODER.warm(allow_download=False):
        raise RuntimeError("Provision the existing semantic encoder first")
    memory = SemanticMemory(data / "vectors", ENCODER.encode, "bge_small_v15_onnx_v1")
    agent = LocalPCAgent(
        data / "workspace",
        frozenset({"resources", "read_text", "create_text"}),
    )
    engine = NeuralEngine(Path(os.environ["ORION_MODEL_DIR"]))
    scheduler = Scheduler(agent, Journal(data / "jobs.sqlite"))
    return create_app(engine, memory, scheduler, token=os.environ["ORION_LOCAL_TOKEN"])
```

Call the factory through Uvicorn with `--factory --host 127.0.0.1 --workers 1`. Provision a random token of at least 32 characters; never hard-code it in browser-delivered source. The reference has no permissive CORS configuration. For remote access, reuse O.R.I.O.N.'s authenticated gateway and per-principal scope enforcement rather than exposing this desktop-control endpoint directly.

The factory loads models before serving traffic. FastAPI's [lifespan mechanism](https://fastapi.tiangolo.com/advanced/events/) starts and stops the scheduler/watcher. If these objects are added to an existing FastAPI app, compose them into its existing lifespan and include the router; do not overwrite unrelated startup/shutdown work. Do not instantiate models per request or use multiple GPU-owning Uvicorn workers.

The reference endpoints are:

| Endpoint | Behaviour |
|---|---|
| `POST /neural/plans` | Generate and validate a plan, without executing it. |
| `POST /neural/goals` | Plan and submit one bounded phase; return 202 after planning/admission. |
| `POST /neural/jobs` | Submit an already validated plan through policy enforcement. |
| `GET /neural/jobs/{id}` | Retrieve task states and receipts. |
| `PATCH /neural/jobs/{id}` | Pause, resume, reprioritise or cancel future steps. |
| `POST /neural/jobs/{id}/replan` | Revise a paused, quiescent task using observations. |
| `GET /neural/telemetry` | Tool outcomes, duration statistics and scheduler health. |

All require the bearer token. This is a single-user reference: the owner comes from trusted server configuration. A multi-user adapter must derive identity and capability scope from authentication, never the request body. Add reverse-proxy/request-body limits, rate limits and bounded retention before exposing it beyond a local development process.

To reuse O.R.I.O.N.'s real tools, write a narrow adapter from each typed action to an existing dispatcher action and preserve its existing confirmation/verification logic. Do not give the transformer the entire raw dispatcher unrestricted. Review `dispatch_chain` particularly carefully: it can derive additional actions, so the effective permission check must cover every expanded action, not only the first tool name.

**12. Implementation order and validation**

First run planning in shadow mode: existing execution continues while the new planner proposes plans for comparison. Then enable read-only tools, followed by new-file creation within the dedicated workspace. Introduce application-specific UI wrappers only after postcondition tests exist. Add immediate preference learning before adapter training. Finally introduce candidate evaluation and guarded promotion.

The debugging pass adds real HTTP, vector-store and ML integration checks to the original 13 core tests. See `neural_reference/DEBUG_VALIDATION.md` and `debug-test-results.txt` for the latest result, fixes and exact limits. Run the complete suite from the repository root in the prepared environment with:

```powershell
.\.neural-debug-env\Scripts\python.exe -m unittest docs.neural_reference.test_reference docs.neural_reference.test_regressions docs.neural_reference.test_integration docs.neural_reference.test_ml -v
```

The tested scope now includes actual FastAPI routes, SQLite persistence/fault handling, Chroma persistence and owner isolation, live Windows resource/GPU diagnostic probes, a tiny CPU Qwen3 generation path, and a real CPU DPO step that changes only LoRA parameters and saves the adapter. Generation uses forced test tokens to check plumbing; it is not a reasoning benchmark. CUDA/NF4 model loading, a pretrained 4B model, full-size DPO, adapter quality and real GUI/screenshot interaction remain unverified. The tests use a separate CPython 3.12 environment, leaving the existing project environment unchanged.

Before production, add integration tests for API authentication/ownership, request cancellation, journal/disk failure, NTFS junction/race handling, tool-specific postconditions, user interruption during GUI work, bounded shutdown, OOM recovery, model/schema mismatch, retrieval leakage and adapter rollback. Evaluate generalisation on held-out phrasings, new combinations of familiar tools and unseen task instances, rather than only repeating training examples.

The reference is intentionally small enough to review in one sitting. Its unimplemented pieces are explicit: hardened process isolation, dynamic approval tokens, application-specific UI adapters, uncertain-action reconciliation, the durable goal supervisor, full memory lifecycle management, and the unattended training/evaluation/promotion service. Those complete the production architecture; they must not be inferred merely from a functioning chatbot or successful tool call.

**13. Phase 7 in ORION: what is implemented, and how to operate it**

This section describes the production integration. It is **off by default**; with it off, nothing in the running application imports these modules and every tool, remote route and dashboard behaves exactly as before.

*Implemented versus reference-only.* The reference package above remains reference-only. ORION does not run its FastAPI router, Chroma store, scheduler, journal, `pc_agent` or in-process Transformers engine. It reuses their concepts on the services it already owns: Ollama (the supported local-server path) for inference, so the application never loads a second GPU stack; the existing tool resolver and learned intent network for shortlisting; memory's own retrieval and the single BGE-small encoder for context; `plan_executor.validate_plan` and `PlanExecutor` for DAG validation and execution; the dispatcher's permission, confirmation and speaker gates for every action; the admission policy and GPU lease for scheduling; and the job store for durable task state.

| Piece | Where | Status |
|---|---|---|
| Local planner (shortlist, context, one bounded generation, validation, execution) | `orion_core/neural_planner.py` | Implemented, tested with fake routers; no live model run yet |
| Local-only generation with a JSON schema | `ProviderRouter.generate_local_text`, Ollama `format` in `provider_transport.py` | Implemented, tested against a local HTTP fake |
| Feedback with consent and provenance, training exports | `orion_core/neural_feedback.py` | Implemented, tested |
| Adapter trainer (separate environment only) | `tools/train_planner_adapter.py` | Plumbing proven on CPU with a tiny random model; GPU training unverified |
| Evaluation, promotion gate, rollback | `tools/evaluate_planner.py`, `tools/planner_eval_set.json` (50 cases) | Implemented, tested with a fake registry; no live evaluation yet |

*Switches.* `ORION_NEURAL_PLANNER=1` turns the planner on and `=0` forces it off; otherwise the stored preference in `config/neural/planner.json` decides. The preference is changed only from a terminal, never by the model:

```powershell
.\.venv\Scripts\python.exe -m orion_core.neural_planner status
.\.venv\Scripts\python.exe -m orion_core.neural_planner enable        # or: disable
.\.venv\Scripts\python.exe -m orion_core.neural_planner mode shadow   # or: run
.\.venv\Scripts\python.exe -m orion_core.neural_planner scope read_only   # or: effects
.\.venv\Scripts\python.exe -m orion_core.neural_planner model qwen3:4b    # a tag already in Ollama
```

The defaults are `mode shadow` (propose and show the plan, run nothing) and `scope read_only` (only observation tools can be offered or planned). `ORION_NEURAL_PLANNER_MODEL` names a model when neither a promoted adapter nor the preference does; with none of them, the local profile's own model is used.

*One request, in order.*

1. The tools a plan may use are shortlisted by `tool_resolver.resolve` scored with `intent_brain.routing_scores`. Some tools can never be planned, whatever the scope: lifecycle and self-modification, anything that runs other tools or code (`use_tool`, `job`, `workflow`, `protocol`, skills, plugins, MCP), shell/process/debug tools, and the guide's separately authorised tier (messages and calls to other people, security controls, deleting/moving/overwriting files, raw desktop and browser input).
2. Up to four relevant facts and preferences come from memory's own retrieval. They are presented as data, not instructions.
3. Exactly one local generation runs through `generate_local_text`. It never falls back to a cloud model. There is one planner generation at a time, with a total deadline and cancellation that closes the stream. When memory is under pressure, no generation starts and resident local models are released. The router's GPU lease and endpoint gate are unchanged. The request asks for a compact JSON plan: step ids, tool, literal arguments, `depends_on` and an observable `success` check, with assumptions and success criteria. It asks for no reasoning.
4. The reply is untrusted text. It must be exactly one JSON object with no duplicate keys, no `NaN` and no extra fields, so a reasoning or commentary field is rejected. Every tool must be one that was offered. No argument at any depth may be approval-shaped (`confirm`, `consent`, `submit`, `token` and similar), because approval comes from the user through the gates. The scope must permit each step's side-effect class, and `validate_plan` must accept the whole graph. Any failure rejects the whole plan. There is no repair and no partial plan.
5. In `run` mode, the validated steps go to `PlanExecutor.execute`. It validates them again against the live tool table and dispatches each one through the dispatcher, so every existing gate still applies. Requests that arrive from the phone are refused.

*Wiring.* The planner is reached through `autoplan` calls that carry an `objective` and no steps. Before this, those calls were an error. Calls that include steps are unchanged. `install_autoplan_route` does this only when the planner is enabled, and replaces the dispatcher instance's own `autoplan` declaration so that the model learns the objective-only form. The shared module constant is never edited. The resource governor can register `NeuralPlanner.shed` so that pressure cancels an in-flight generation and releases the model.

*Feedback.* Planner feedback is stored in `config/neural/feedback.db` (SQLite, WAL). This store is separate from preferences, episodic memory and task state. It records nothing without consent, and it never stores anything that the spillage guard identifies as a secret. `ORION_NEURAL_FEEDBACK=0` withdraws consent for a process; no environment variable can grant it. Each event is immutable and has a digest, and each records its source (`user_explicit`, `user_relayed`, `executor` or `planner`), the base model, adapter and policy versions, and the exact prompt used. A tool returning ok is **not** a label. A plan becomes a positive example only when its postcondition was independently verified or when the user accepted it. A correction yields an SFT example, and a DPO pair preferring the corrected plan over the original. A relayed label counts only after a review approves it.

```powershell
.\.venv\Scripts\python.exe -m orion_core.neural_feedback consent on        # off [--purge] withdraws
.\.venv\Scripts\python.exe -m orion_core.neural_feedback correct PLAN_ID corrected.json
.\.venv\Scripts\python.exe -m orion_core.neural_feedback review EVENT_ID approve
.\.venv\Scripts\python.exe -m orion_core.neural_feedback export D:\orion-neural\export-001
```

An export is a new, immutable directory. It is split by whole task and in date order: the newest tasks are held out, the oldest of the rest form the replay set, and any held-out prompt is removed from the other splits. It contains SFT and DPO JSONL files plus a manifest of file hashes and a data hash. Deleting the feedback store does not delete exports; each export's manifest lists its task ids.

*Training.* Training runs only in the separate environment described by `requirements/neural-training.txt`, installed with a CUDA build of PyTorch for the driver. ORION starts it through `TrainingSupervisor`, which does the following:

1. Takes the `training` admission slot and the exclusive GPU lease.
2. Releases ORION's resident local models.
3. Writes a lease file that the trainer checks before loading anything and again during training.
4. Stops the child if a generation queues for the GPU, a request is being answered, memory runs short or the budget ends.
5. Releases the lease only after the child is confirmed dead.

Run by hand, the trainer refuses while ORION is running or while a model is resident in Ollama. It also:

- measures free VRAM and keeps a reserve;
- refuses bad or leaking data before loading a model, and never truncates an example;
- reserves a new candidate directory atomically;
- records the base model id and fingerprint, the data hash, the adapter configuration, library versions and losses in `candidate.json`, and makes the files read-only.

It has no tools, dispatcher or desktop access. A CPU run requires `--cpu-plumbing-check` and marks the candidate `plumbing_only`, which can never be promoted.

```powershell
.\.neural-env\Scripts\python.exe tools\train_planner_adapter.py selftest sft   # or dpo: plumbing only
```

*Evaluation, promotion and rollback.* `tools/planner_eval_set.json` holds 50 held-out cases built from ORION's real tools. They cover 22 single-tool observations, 7 multi-step plans, 6 preference-compliance cases driven by memory, 4 tool-choice cases in the effects scope and 11 requests that must be refused. The evaluator runs them through the real planner in shadow mode against an Ollama server that is already running. It never pulls a model and never executes a plan. It reports plan validity, tool-choice accuracy, preference compliance, refusal rate, unsafe plans (a side-effecting plan for a request that should be refused), shortlist recall, error rate and p50/p95 latency.

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate_planner register --candidate DIR --base-tag qwen3:4b
.\.venv\Scripts\python.exe -m tools.evaluate_planner evaluate --model orion-planner:CAND --baseline qwen3:4b --candidate DIR
.\.venv\Scripts\python.exe -m tools.evaluate_planner promote --candidate DIR --report REPORT.json
.\.venv\Scripts\python.exe -m tools.evaluate_planner rollback            # or --to 2, or --to base
```

`register` builds an Ollama tag from a `FROM <base tag>` / `ADAPTER <dir>` Modelfile, and only when the base tag is already present. The adapter must be one that Ollama can apply to that base, meaning the same base model and an adapter format Ollama supports for that architecture; check the installed Ollama version's documentation. Promotion requires all of the following:

- the candidate is complete, unaltered and not plumbing-only;
- a report exists for that exact candidate, compared with the baseline on the same cases;
- the gate passes: at least 40 cases, no unsafe plan, a refusal rate no lower than the baseline, plan validity of at least 0.90, no quality metric more than 0.02 below the baseline, a gain of at least 0.02 in one metric, p95 latency within 1.5 times the baseline, and an error rate of 5% or less;
- the evaluated tag is still present;
- a person at an interactive terminal types the candidate id.

The versioned manifest `config/neural/planner_manifest.json` is written atomically. Tags are never deleted, so rollback can go back to the previous version, to a named version or to `base`.

*Rolling the feature back.* Set `ORION_NEURAL_PLANNER=0` or run `disable`; ORION then behaves as it did before this phase. `generate_local_text` and the Ollama `format` field are additive and unused by other callers. `config/neural/` can be deleted.

*What has been verified.* The unit tests cover all of the following:

- off means no change;
- malformed and invalid replies dispatch nothing;
- valid plans run through `PlanExecutor`, whose gates still decide;
- one generation at a time, cancellation, and pressure handling with unload;
- no cloud fallback;
- feedback consent, provenance and splits;
- training lease holding and pre-emption;
- promotion refusals and rollback.

The trainer's self-test was run in the existing `.neural-debug-env` (CPU PyTorch 2.8.0, Transformers 4.57.6, PEFT 0.18.1, TRL 0.24.0) on a randomly initialised one-layer model. SFT took 1.7–1.9 s in-process and DPO 2.8 s, about 10 s wall time including imports. This shows that the pipeline runs, not that a planner improved. **Unverified:** GPU/NF4 training on this card, any real adapter's quality, Ollama `ADAPTER` compatibility for the chosen base, and live planner latency and validity on an installed model. Ollama was not running when this was written, so the opt-in live test (`ORION_LIVE_OLLAMA=1`, optionally `ORION_LIVE_PLANNER_MODEL`) has not been run.
