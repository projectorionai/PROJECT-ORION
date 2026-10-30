"""
Train a planner adapter CANDIDATE — in the separate neural-training environment.

    .neural-env\\Scripts\\python.exe tools\\train_planner_adapter.py ^
        --data EXPORT_DIR --base-model MODEL_DIR --base-model-id ID ^
        --output-root CANDIDATES_DIR [--mode sft|dpo] [--time-budget-s 3600]

    .neural-env\\Scripts\\python.exe tools\\train_planner_adapter.py selftest

ORION starts this through ``orion_core.neural_planner.TrainingSupervisor``: it
takes the TRAINING admission slot and the exclusive GPU lease, releases the
local model's memory, and passes ``--lease-file``. The lease is checked before
anything is loaded and again during training; if ORION stops, the lease
expires or the file disappears, training stops. Run by hand, it refuses while
ORION is running (a local generation could start on the same card) or while a
local model is resident in Ollama.

What it is not allowed to be:

* part of ORION — it is never imported by the app and imports nothing from
  it, so the app's environment never loads a second GPU stack;
* an actor — it reads a feedback export and a local base model and writes one
  new candidate directory. No tools, no dispatcher, no desktop, no network
  beyond asking the local Ollama server what is loaded;
* a deployer — a candidate is inert until ``tools/evaluate_planner.py`` has
  evaluated it and a person has promoted it.

A candidate directory is reserved atomically, never reused, and complete only
once ``candidate.json`` exists (written last). It records the data hash, the
base model's identity and fingerprint, the adapter configuration, the exact
library versions and the measured losses; its files are made read-only.

CUDA is required for real training. ``--device cpu`` exists to prove the
plumbing on a tiny model and marks the candidate ``plumbing_only``, which the
promotion gate refuses: a tiny CPU run shows the pipeline works, not that the
planner got better. ``selftest`` does exactly that on a randomly initialised
one-layer model it builds in a temporary directory (no download).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_REFUSED, EXIT_FAILED = 0, 3, 4
CANDIDATE_FORMAT = "orion-planner-candidate-v1"
EXPORT_FORMAT = "orion-planner-feedback-v1"
#: Kept identical to orion_core.single_instance.MUTEX_NAME (a test checks);
#: copied rather than imported so this file never imports ORION.
ORION_MUTEX = "Global\\ORION-single-instance-9f3c"
OLLAMA_PS = "http://127.0.0.1:11434/api/ps"
REQUIRED = {"sft": ("torch", "transformers", "peft"),
            "dpo": ("torch", "transformers", "peft", "trl", "datasets")}
GIB = 1024 ** 3


class Refused(Exception):
    """A clear, expected reason not to train. Nothing was written."""


# ── output ───────────────────────────────────────────────────────────────────

def emit(result: dict[str, Any]) -> int:
    """The last stdout line is the machine-readable result the supervisor reads."""
    print(json.dumps(result, sort_keys=True, default=str), flush=True)
    return {"candidate": EXIT_OK, "selftest": EXIT_OK, "refused": EXIT_REFUSED,
            "deferred": EXIT_REFUSED}.get(result.get("status"), EXIT_FAILED)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ── the environment ──────────────────────────────────────────────────────────

def check_environment(mode: str) -> None:
    missing = [name for name in REQUIRED[mode] if importlib.util.find_spec(name) is None]
    if missing:
        raise Refused(
            f"this is not the neural-training environment (missing {', '.join(missing)}). "
            "Create it with requirements/neural-training.txt and run this file with that "
            "environment's Python; ORION's own environment never installs these.")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, int(pid))   # QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                  # exists, access denied
        try:
            code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259                # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def orion_running() -> bool:
    """Whether ORION's single-instance mutex exists (Windows)."""
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = wintypes.HANDLE
    handle = kernel32.OpenMutexW(0x00100000, False, ORION_MUTEX)  # SYNCHRONIZE
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == 5


def resident_models(url: str = OLLAMA_PS) -> list[str]:
    """Models the local Ollama server holds in memory now; [] if it is not running."""
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:        # noqa: S310
            data = json.loads(response.read(262_144).decode("utf-8"))
    except Exception:
        return []
    rows = data.get("models") if isinstance(data, dict) else None
    return [str(r.get("name") or r.get("model") or "") for r in rows or [] if isinstance(r, dict)]


def read_lease(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Refused("the lease file could not be read; ORION has not admitted this run") \
            from None
    if not isinstance(data, dict):
        raise Refused("the lease file is not a lease")
    return data


def lease_problem(lease: dict[str, Any], now: float | None = None) -> str:
    """'' while the lease is valid, else why it is not."""
    now = time.time() if now is None else now
    if lease.get("resource") != "gpu" or lease.get("class") != "training":
        return "the lease is not a GPU lease for training work"
    if not lease.get("ticket"):
        return "the lease carries no admission ticket"
    if float(lease.get("expires_at") or 0) <= now:
        return "the lease has expired"
    if not _pid_alive(int(lease.get("holder_pid") or 0)):
        return "the ORION process that holds the lease is no longer running"
    return ""


def admit(lease_file: Path | None, *, device: str) -> dict[str, Any]:
    """Refuse unless this run cannot overlap a local generation."""
    lease = read_lease(lease_file)
    if lease is not None:
        problem = lease_problem(lease)
        if problem:
            raise Refused(problem)
        return {"admitted_by": "orion", "ticket": lease["ticket"]}
    if orion_running():
        raise Refused("ORION is running. Start training from ORION, which admits it as "
                      "TRAINING work holding the GPU lease, or close ORION first.")
    loaded = [m for m in resident_models() if m]
    if loaded and device != "cpu":
        raise Refused(f"a local model is loaded ({', '.join(loaded[:3])}); unload it or "
                      "start training from ORION")
    return {"admitted_by": "standalone", "ticket": ""}


class Lock:
    """One trainer per output root, across processes."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".training.lock"
        self.fd: int | None = None

    def __enter__(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = int(self.path.read_text(encoding="ascii").strip() or 0)
            except (OSError, ValueError):
                holder = 0
            if holder and _pid_alive(holder):
                raise Refused(f"another trainer (pid {holder}) is using {self.path.parent}") \
                    from None
            self.path.unlink(missing_ok=True)             # a crashed run's lock
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.fd, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


# ── data ─────────────────────────────────────────────────────────────────────

def load_export(directory: Path, mode: str) -> tuple[dict[str, Any], dict[str, list[dict]]]:
    """The export's manifest and its rows, after checking every file against
    the manifest (the same check as orion_core.neural_feedback.verify_export)."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Refused(f"{directory} is not a feedback export (no readable manifest.json)") \
            from None
    if manifest.get("format") != EXPORT_FORMAT:
        raise Refused("the export's format is not one this trainer understands")
    files = manifest.get("files") or {}
    for name, digest in files.items():
        if Path(name).name != name:
            raise Refused(f"unexpected file name in the export manifest: {name}")
        try:
            text = (directory / name).read_text(encoding="utf-8")
        except OSError:
            raise Refused(f"{name} is missing from the export") from None
        if sha256_text(text) != digest:
            raise Refused(f"{name} was changed after the export was written")
    if sha256_text(_canonical(files)) != manifest.get("data_hash"):
        raise Refused("the export's data hash does not match its files")
    splits: dict[str, list[dict]] = {}
    for split in ("train", "replay", "heldout"):
        name = f"{mode}_{split}.jsonl"
        text = (directory / name).read_text(encoding="utf-8") if name in files else ""
        splits[split] = [json.loads(line) for line in text.splitlines() if line.strip()]
    return manifest, splits


def _messages_ok(messages: Any, *, final_assistant: bool) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if (not isinstance(message, dict) or message.get("role") not in
                {"system", "user", "assistant"} or not isinstance(message.get("content"), str)
                or not message["content"].strip()):
            return False
    return not final_assistant or messages[-1]["role"] == "assistant"


def validate_rows(splits: dict[str, list[dict]], mode: str, *, min_examples: int) -> None:
    """Refuse bad data before any model is loaded."""
    training = splits["train"] + splits["replay"]
    if len(training) < min_examples:
        raise Refused(f"{len(training)} training example(s); at least {min_examples} are "
                      "required (curate more before training)")
    prompts: dict[str, set[str]] = {}
    for split, rows in splits.items():
        seen: set[str] = set()
        prompts[split] = set()
        for row in rows:
            if mode == "sft":
                messages = row.get("messages")
                if not _messages_ok(messages, final_assistant=True):
                    raise Refused(f"a {split} example is not a valid conversation")
                prompt = [m for m in messages[:-1] if m["role"] == "user"]
                completion = messages[-1]["content"]
                key = _canonical(messages)
            else:
                if not (_messages_ok(row.get("prompt"), final_assistant=False)
                        and _messages_ok(row.get("chosen"), final_assistant=True)
                        and _messages_ok(row.get("rejected"), final_assistant=True)
                        and len(row["chosen"]) == 1 and len(row["rejected"]) == 1):
                    raise Refused(f"a {split} preference pair is malformed")
                if row["chosen"] == row["rejected"]:
                    raise Refused(f"a {split} preference pair has identical answers")
                prompt = [m for m in row["prompt"] if m["role"] == "user"]
                completion = row["chosen"][0]["content"]
                key = _canonical([row["prompt"], row["chosen"], row["rejected"]])
            try:
                if not isinstance(json.loads(completion), dict):
                    raise ValueError
            except ValueError:
                raise Refused(f"a {split} target is not a JSON plan") from None
            if key in seen:
                raise Refused(f"duplicate example in {split}")
            seen.add(key)
            prompts[split].add(_canonical(prompt))
    if prompts["heldout"] & (prompts["train"] | prompts["replay"]):
        raise Refused("a held-out prompt also appears in the training data")


# ── the base model ───────────────────────────────────────────────────────────

def fingerprint(base: Path) -> dict[str, Any]:
    """Identity of a local base model: hashes of its configuration and
    tokenizer, and the names and sizes of its weight files (hashing several
    gigabytes on every run is not worth it; --hash-weights does it)."""
    if not (base / "config.json").is_file():
        raise Refused(f"{base} has no config.json; provision the base model locally first")
    small = {}
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
                 "merges.txt", "chat_template.jinja"):
        if (base / name).is_file():
            small[name] = sha256_file(base / name)
    weights = {p.name: p.stat().st_size for p in sorted(base.glob("*.safetensors"))}
    if not weights:
        raise Refused("the base model has no .safetensors weights (other formats are refused)")
    return {"files": small, "weights": weights,
            "digest": sha256_text(_canonical({"files": small, "weights": weights}))}


# ── training ─────────────────────────────────────────────────────────────────

class Stop(Exception):
    """Stop now: the lease went away or the time budget ran out."""


class Guard:
    """Checked between optimisation steps."""

    def __init__(self, lease_file: Path | None, deadline: float) -> None:
        self.lease_file = lease_file
        self.deadline = deadline
        self._last = 0.0

    def check(self) -> None:
        if time.monotonic() > self.deadline:
            raise Stop("the time budget ran out")
        if self.lease_file is None or time.monotonic() - self._last < 2.0:
            return
        self._last = time.monotonic()
        if not self.lease_file.exists():
            raise Stop("ORION withdrew the lease")
        problem = lease_problem(read_lease(self.lease_file) or {})
        if problem:
            raise Stop(problem)


def _ids(value: Any) -> list[int]:
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    return [int(v) for v in list(value)]


def encode_sft(tokenizer: Any, messages: list[dict], max_length: int) -> tuple[list[int], list[int]]:
    prompt = _ids(tokenizer.apply_chat_template(messages[:-1], tokenize=True,
                                                add_generation_prompt=True))
    full = _ids(tokenizer.apply_chat_template(messages, tokenize=True))
    if full[:len(prompt)] != prompt:
        raise Refused("the chat template does not extend the prompt with the answer; "
                      "an SFT mask cannot be placed safely")
    if len(full) > max_length:
        raise Refused(f"an example is {len(full)} tokens; the budget is {max_length}. "
                      "Curate it or raise the budget — examples are never truncated.")
    return full, [-100] * len(prompt) + full[len(prompt):]


def load_model(base: Path, device: str, quantise: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True,
                                              trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False,
                              "use_safetensors": True}
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = torch.float16
        if quantise:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(base), **kwargs)
    model.config.use_cache = False
    return model, tokenizer


def gate_gpu(reserve_gb: float, need_gb: float) -> dict[str, float]:
    """A MEASURED free-memory gate (the chess trainer's pattern): refuse
    unless the card has room for the run plus a reserve for the desktop."""
    import torch
    if not torch.cuda.is_available():
        raise Refused("CUDA is not available in this environment (a CPU-only PyTorch "
                      "wheel?). Real training needs the CUDA build for this driver.")
    free, total = torch.cuda.mem_get_info()
    usable = free - reserve_gb * GIB
    if usable < need_gb * GIB:
        raise Refused(f"measured {free / GIB:.2f} GiB free of {total / GIB:.2f} GiB; this run "
                      f"needs {need_gb:.1f} GiB plus a {reserve_gb:.1f} GiB reserve")
    torch.cuda.set_per_process_memory_fraction(max(0.05, min(1.0, usable / total)))
    return {"free_gib": round(free / GIB, 3), "total_gib": round(total / GIB, 3),
            "budget_gib": round(usable / GIB, 3)}


def train_sft(model: Any, tokenizer: Any, splits: dict[str, list[dict]], args: Any,
              guard: Guard) -> dict[str, Any]:
    import torch
    device = next(model.parameters()).device
    rows = splits["train"] + splits["replay"]
    encoded = [encode_sft(tokenizer, r["messages"], args.max_length) for r in rows]
    heldout = [encode_sft(tokenizer, r["messages"], args.max_length) for r in splits["heldout"]]

    def mean_loss(items: list[tuple[list[int], list[int]]]) -> float | None:
        if not items:
            return None
        model.eval()
        total = 0.0
        with torch.no_grad():
            for ids, labels in items:
                out = model(input_ids=torch.tensor([ids], device=device),
                            labels=torch.tensor([labels], device=device))
                total += float(out.loss)
        model.train()
        return total / len(items)

    with model.disable_adapter():
        heldout_before = mean_loss(heldout)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.learning_rate)
    rng = random.Random(args.seed)
    losses: list[float] = []
    steps = 0
    model.train()
    for _epoch in range(args.epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for position, index in enumerate(order, 1):
            guard.check()
            ids, labels = encoded[index]
            out = model(input_ids=torch.tensor([ids], device=device),
                        labels=torch.tensor([labels], device=device))
            value = float(out.loss.detach())
            if not math.isfinite(value):
                raise RuntimeError("the loss became non-finite; nothing was saved")
            (out.loss / args.accumulation).backward()
            losses.append(value)
            if position % args.accumulation == 0 or position == len(order):
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                steps += 1
                if args.max_steps and steps >= args.max_steps:
                    break
        if args.max_steps and steps >= args.max_steps:
            break
    return {"optimiser_steps": steps, "examples": len(encoded),
            "train_loss_first": losses[0] if losses else None,
            "train_loss_last": losses[-1] if losses else None,
            "heldout_loss_base": heldout_before, "heldout_loss_adapter": mean_loss(heldout)}


def train_dpo(model: Any, tokenizer: Any, splits: dict[str, list[dict]], args: Any,
              guard: Guard, output: Path) -> dict[str, Any]:
    """DPO with the reference recipe (TRL 0.24): a fresh adapter over the
    immutable base, whose adapter-disabled self is the frozen reference."""
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import DPOConfig, DPOTrainer

    class _Guarded(TrainerCallback):
        def on_step_end(self, _args, state, control, **_kw):   # noqa: ANN001
            try:
                guard.check()
            except Stop:
                control.should_training_stop = True
                raise
            return control

    cpu = args.device == "cpu"
    config = DPOConfig(
        output_dir=str(output / "trainer"), beta=0.1, learning_rate=args.learning_rate,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulation, num_train_epochs=args.epochs,
        max_steps=args.max_steps or -1, max_length=args.max_length,
        max_prompt_length=args.max_prompt_length, fp16=not cpu, bf16=False,
        gradient_checkpointing=not cpu, optim="adamw_torch" if cpu else "paged_adamw_8bit",
        use_cpu=cpu, precompute_ref_log_probs=True, precompute_ref_batch_size=1,
        eval_strategy="no", save_strategy="no", logging_steps=10, report_to="none",
        seed=args.seed, disable_tqdm=True)
    def rows(split_rows: list[dict]) -> Any:
        return Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")}
                                  for r in split_rows])

    # The held-out pairs go in at construction so the trainer tokenises them
    # the same way as the training pairs; they are only ever evaluated.
    trainer = DPOTrainer(model=model, ref_model=None, args=config, processing_class=tokenizer,
                         train_dataset=rows(splits["train"] + splits["replay"]),
                         eval_dataset=rows(splits["heldout"]) if splits["heldout"] else None,
                         callbacks=[_Guarded()])
    result = trainer.train()
    metrics = {"training_loss": float(result.training_loss),
               "optimiser_steps": int(result.global_step)}
    if splits["heldout"]:
        metrics["heldout"] = {k: float(v) for k, v in trainer.evaluate().items()
                              if isinstance(v, (int, float))}
    return metrics


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "peft", "trl", "datasets", "bitsandbytes"):
        if importlib.util.find_spec(name) is not None:
            try:
                out[name] = str(__import__(name).__version__)
            except Exception:
                out[name] = "unknown"
    return out


def _freeze(directory: Path) -> dict[str, str]:
    """Hash every file and make it read-only: a candidate is immutable."""
    hashes = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        hashes[path.relative_to(directory).as_posix()] = sha256_file(path)
        path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return hashes


def run(args: Any) -> dict[str, Any]:
    check_environment(args.mode)
    if args.device == "cpu" and not args.cpu_plumbing_check:
        raise Refused("CPU training is refused: it cannot produce a useful planner adapter. "
                      "Pass --cpu-plumbing-check to prove the pipeline on a tiny model; the "
                      "candidate is then marked plumbing_only and can never be promoted.")
    data_dir, base = Path(args.data), Path(args.base_model)
    manifest, splits = load_export(data_dir, args.mode)
    validate_rows(splits, args.mode, min_examples=args.min_examples)
    base_print = fingerprint(base)
    if args.hash_weights:
        base_print["weight_sha256"] = {p.name: sha256_file(p)
                                       for p in sorted(base.glob("*.safetensors"))}
    if not args.base_model_id.strip():
        raise Refused("record which base model this is (--base-model-id)")
    lease_file = Path(args.lease_file) if args.lease_file else None
    admission = (admit(lease_file, device=args.device) if not args.selftest_admission
                 else {"admitted_by": "selftest", "ticket": ""})
    root = Path(args.output_root)
    with Lock(root):
        import torch
        torch.manual_seed(args.seed)
        memory = gate_gpu(args.reserve_gb, args.need_gb) if args.device == "cuda" else {}
        guard = Guard(lease_file, time.monotonic() + float(args.time_budget_s))
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        candidate_id = f"cand-{stamp}-{manifest['data_hash'][:8]}-{uuid.uuid4().hex[:6]}"
        output = root / candidate_id
        output.mkdir(parents=False, exist_ok=False)        # reserved: never shared, never reused
        model, tokenizer = load_model(base, args.device, quantise=args.device == "cuda")
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if args.device == "cuda":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = {"r": args.rank, "lora_alpha": args.alpha, "lora_dropout": args.dropout,
                "target_modules": list(args.target_modules), "bias": "none"}
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora))
        started = time.time()
        try:
            metrics = (train_sft(model, tokenizer, splits, args, guard) if args.mode == "sft"
                       else train_dpo(model, tokenizer, splits, args, guard, output))
        except Stop as stop:
            return {"status": "deferred", "reason": f"training stopped: {stop}",
                    "incomplete": str(output)}
        adapter = output / "adapter"
        model.save_pretrained(str(adapter), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter))
        trainer_dir = output / "trainer"
        if trainer_dir.exists():
            import shutil
            shutil.rmtree(trainer_dir, ignore_errors=True)
        files = _freeze(output)
        record = {
            "format": CANDIDATE_FORMAT, "candidate_id": candidate_id, "created_at": time.time(),
            "mode": args.mode, "base_model_id": args.base_model_id.strip(),
            "base_model_fingerprint": base_print, "data_dir": str(data_dir),
            "data_hash": manifest["data_hash"], "data_counts": manifest.get("counts"),
            "policy_version": manifest.get("policy_version"),
            "adapter_config": lora,
            "adapter_sha256": sha256_text(_canonical(files)), "files": files,
            "training": {"learning_rate": args.learning_rate, "epochs": args.epochs,
                         "max_steps": args.max_steps, "accumulation": args.accumulation,
                         "max_length": args.max_length, "seed": args.seed,
                         "device": args.device,
                         "quantisation": "nf4" if args.device == "cuda" else "none",
                         "seconds": round(time.time() - started, 2)},
            "gpu_memory": memory, "metrics": metrics, "versions": _versions(),
            "admission": admission,
            "plumbing_only": args.device == "cpu",
        }
        target = output / "candidate.json"
        temporary = output / f".candidate-{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)
        target.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        return {"status": "candidate", "candidate_id": candidate_id, "path": str(output),
                "plumbing_only": record["plumbing_only"], "metrics": metrics}


# ── the plumbing self-test ───────────────────────────────────────────────────

def selftest(mode: str) -> dict[str, Any]:
    """Build a random one-layer model and a four-example export in a temporary
    directory, then run the real path on CPU. Proves plumbing, nothing else."""
    check_environment(mode)
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    words = ["[UNK]", "[PAD]", "[EOS]", "[BOS]", "system", "user", "assistant",
             "Request:", "check", "memory", "search", "news", "notes", "status", "{", "}"]
    raw = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(words)},
                                     unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]",
                                        pad_token="[PAD]", eos_token="[EOS]",
                                        bos_token="[BOS]", padding_side="left")
    tokenizer.chat_template = (
        "{{bos_token}} {% for m in messages %}{{m['role']}} {{m['content']}} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, eos_token_id=2,
        pad_token_id=1, bos_token_id=3))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="orion-planner-selftest-") as scratch:
        root = Path(scratch)
        base = root / "base"
        model.save_pretrained(str(base), safe_serialization=True)
        tokenizer.save_pretrained(str(base))
        plans = ['{"summary":"check memory","steps":[{"id":"a","tool":"resource_status",'
                 '"args":{},"depends_on":[],"success":"memory shown"}],'
                 '"success_criteria":["memory shown"]}',
                 '{"summary":"search news","steps":[{"id":"a","tool":"web_search","args":'
                 '{"query":"news"},"depends_on":[],"success":"results"}],'
                 '"success_criteria":["results"]}']
        rows = []
        for index, goal in enumerate(("check memory", "search news", "memory status",
                                      "news notes")):
            prompt = [{"role": "system", "content": "plan"},
                      {"role": "user", "content": f"Request: {goal}"}]
            if mode == "sft":
                rows.append({"messages": prompt + [{"role": "assistant",
                                                    "content": plans[index % 2]}]})
            else:
                rows.append({"prompt": prompt,
                             "chosen": [{"role": "assistant", "content": plans[index % 2]}],
                             "rejected": [{"role": "assistant",
                                           "content": plans[(index + 1) % 2]}]})
        export = root / "export"
        export.mkdir()
        files = {}
        for split, chunk in (("train", rows[:2]), ("replay", rows[2:3]), ("heldout", rows[3:])):
            for kind in ("sft", "dpo"):
                name = f"{kind}_{split}.jsonl"
                text = "".join(json.dumps(r, sort_keys=True) + "\n"
                               for r in (chunk if kind == mode else []))
                (export / name).write_text(text, encoding="utf-8")
                files[name] = sha256_text(text)
        (export / "manifest.json").write_text(json.dumps({
            "format": EXPORT_FORMAT, "files": files, "data_hash": sha256_text(_canonical(files)),
            "counts": {}, "policy_version": "selftest"}), encoding="utf-8")
        args = build_parser().parse_args([
            "--data", str(export), "--base-model", str(base),
            "--base-model-id", "selftest/tiny-random-qwen3", "--output-root", str(root / "out"),
            "--mode", mode, "--device", "cpu", "--cpu-plumbing-check", "--min-examples", "1",
            "--max-steps", "2", "--accumulation", "1", "--max-length", "256",
            "--max-prompt-length", "128", "--time-budget-s", "300", "--learning-rate", "1e-3"])
        args.selftest_admission = True
        result = run(args)
        if result.get("status") != "candidate":
            return result
        record = json.loads((Path(result["path"]) / "candidate.json").read_text("utf-8"))
        writable = [p for p in Path(result["path"]).rglob("*")
                    if p.is_file() and os.access(p, os.W_OK)]
        for path in Path(result["path"]).rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IWRITE)   # let the temp dir be removed
        return {"status": "selftest", "mode": mode, "plumbing_only": record["plumbing_only"],
                "metrics": record["metrics"], "adapter_files": sorted(record["files"]),
                "writable_files_after_freeze": len(writable),
                "seconds": round(time.monotonic() - started, 2),
                "note": "A random one-layer model on CPU: this proves the pipeline runs, "
                        "not that any planner improved."}


# ── command line ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="a feedback export directory")
    parser.add_argument("--base-model", required=True, help="local base model directory")
    parser.add_argument("--base-model-id", required=True,
                        help="the base model's identity, e.g. its repository and revision")
    parser.add_argument("--output-root", required=True, help="where candidates are created")
    parser.add_argument("--mode", choices=("sft", "dpo"), default="sft")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-plumbing-check", action="store_true")
    parser.add_argument("--lease-file", default=None)
    parser.add_argument("--time-budget-s", type=float, default=3600.0)
    parser.add_argument("--min-examples", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--reserve-gb", type=float, default=1.5,
                        help="GPU memory left for the desktop and other programs")
    parser.add_argument("--need-gb", type=float, default=3.0,
                        help="GPU memory this run must have beyond the reserve")
    parser.add_argument("--hash-weights", action="store_true")
    parser.set_defaults(selftest_admission=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv[:1] == ["selftest"]:
            mode = argv[1] if len(argv) > 1 else "sft"
            if mode not in REQUIRED:
                return emit({"status": "refused", "reason": "selftest takes sft or dpo"})
            return emit(selftest(mode))
        args = build_parser().parse_args(argv)
        if not 1 <= args.accumulation <= 256 or not 1 <= args.rank <= 128 \
                or not 0 < args.learning_rate < 1 or not 1 <= args.epochs <= 20 \
                or not 60 <= args.time_budget_s <= 86_400 or args.max_length < 64:
            return emit({"status": "refused", "reason": "a training setting is out of bounds"})
        return emit(run(args))
    except Refused as refused:
        return emit({"status": "refused", "reason": str(refused)})
    except Exception as exc:                  # noqa: BLE001 - one bounded report
        return emit({"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"})


if __name__ == "__main__":
    raise SystemExit(main())
