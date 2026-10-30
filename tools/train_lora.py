"""
QLoRA fine-tuning of ORION's local brain (stage 3, step 2).

    %USERPROFILE%\\ORION\\neural-env\\Scripts\\python.exe -m tools.train_lora --check
    %USERPROFILE%\\ORION\\neural-env\\Scripts\\python.exe -m tools.train_lora --dataset latest --preset llama3.2-3b
    .neural-debug-env\\Scripts\\python.exe -m tools.train_lora selftest        # CPU, random tiny model

What it does: loads a LOCAL copy of an instruction-tuned base model in 4-bit
(NF4), attaches a LoRA adapter (rank 16 on every linear layer), and trains it
on a dataset built by ``tools/train_build_dataset.py`` - learning only from the
assistant's own tokens (its tool calls and spoken replies) - at batch size 1
with gradient accumulation and gradient checkpointing, sequences up to 2048
tokens. That is sized for an 8 GB card: a 3-4B base is comfortable; a 7-8B base
is slow and may not fit beside the desktop (see docs/LOCAL_BRAIN_TRAINING.md).
The result is a CANDIDATE directory (adapter + a training receipt + hashes)
that ``tools/train_export.py`` and ``tools/evaluate_planner.py`` take from here.

What it refuses to do: install anything, download anything, or run when
something it needs is missing. It checks first - packages, base model, dataset,
GPU, whether ORION or a loaded Ollama model already holds the card - and
prints exactly what is missing, what to run and roughly how big it is, then
exits with status 3 having changed nothing.

It is a separate program on purpose. It runs in its own environment (never
ORION's), imports nothing from the app but a few standard-library-only helper
modules, takes the exclusive GPU for its duration, and writes only a new
candidate folder. A candidate does nothing until a person promotes it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion_core.training import candidate as candidate_mod  # noqa: E402  (stdlib only)
from orion_core.training import encode as encode_mod  # noqa: E402
from orion_core.training import files as files_mod  # noqa: E402
from tools import train_planner_adapter as base  # noqa: E402  (stdlib at import; shared guards)

EXIT_OK, EXIT_REFUSED, EXIT_FAILED = base.EXIT_OK, base.EXIT_REFUSED, base.EXIT_FAILED
Refused = base.Refused
GIB = base.GIB
ENV_TRAINING_DIR = "ORION_TRAINING_DIR"
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MIN_ROWS = 150


# ── what can be trained, and what it needs ────────────────────────────────────

@dataclass(frozen=True)
class Preset:
    key: str
    hf_id: str                  # the Hugging Face repository to take the weights from
    download_gb: float          # approximate size of the bf16 safetensors download
    vram_gb: float              # estimated peak VRAM for QLoRA, max_seq 2048, batch 1 (NOT measured here)
    ollama_base: str            # the Ollama tag the adapter is applied to afterwards
    ollama_adapter: str         # how Ollama takes the adapter: 'safetensors' or 'gguf' (needs llama.cpp)
    licence: str
    note: str


PRESETS: dict[str, Preset] = {p.key: p for p in (
    Preset("llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", 6.4, 4.5, "llama3.2:3b", "safetensors",
           "Llama 3.2 Community Licence (accept it on the model's Hugging Face page first)",
           "Comfortable on 8 GB. Ollama applies its LoRA directly."),
    Preset("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", 6.2, 4.5, "qwen2.5:3b", "gguf",
           "Qwen Research Licence (non-commercial use)",
           "Comfortable on 8 GB; strong at tool calling. Its LoRA must be converted to GGUF for Ollama."),
    Preset("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", 15.2, 7.0, "qwen2.5:7b", "gguf", "Apache-2.0",
           "Tight on 8 GB with the desktop running, slow. Better on a 16-24 GB card."),
    Preset("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", 16.1, 7.2, "llama3.1:latest", "safetensors",
           "Llama 3.1 Community Licence (accept it on the model's Hugging Face page first)",
           "The model ORION runs today; the adapter drops straight onto it. Tight on 8 GB, slow."),
)}
DEFAULT_PRESET = "llama3.2-3b"

#: (import name, install line, approximate download). Versions follow the project's own
#: docs/neural_reference/requirements.in.
PACKAGES: dict[str, tuple[str, str]] = {
    "torch": ('pip install torch --index-url https://download.pytorch.org/whl/cu128   '
              '(choose the line for your driver at https://pytorch.org/get-started/locally)',
              "about 3 GB"),
    "transformers": ('pip install "transformers>=4.57,<5"', "about 10 MB"),
    "peft": ('pip install "peft>=0.17,<0.19"', "under 1 MB"),
    "accelerate": ('pip install "accelerate>=1.0,<2"', "under 1 MB"),
    "bitsandbytes": ('pip install "bitsandbytes>=0.48,<0.50"', "about 70 MB"),
    "safetensors": ("pip install safetensors", "under 1 MB"),
    "unsloth": ("pip install unsloth", "large dependency tree; not needed - the PEFT route is the "
                                       "supported one on Windows"),
}
BACKEND_PACKAGES = {"peft": ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors"),
                    "unsloth": ("torch", "transformers", "peft", "bitsandbytes", "safetensors", "unsloth")}


@dataclass
class Missing:
    kind: str            # package | model | dataset | gpu | runtime
    what: str
    fix: str
    size: str = ""

    def line(self) -> str:
        size = f" ({self.size})" if self.size else ""
        return f"  - {self.what}{size}\n      {self.fix}"


def default_training_root(environ: dict[str, str] | None = None, home: Path | None = None) -> Path:
    """Where ``tools.train_build_dataset`` writes by default: $ORION_TRAINING_DIR, else the
    installed app's ``~/ORION/config/training`` if it exists, else ``~/ORION/training``."""
    env = os.environ if environ is None else environ
    given = str(env.get(ENV_TRAINING_DIR, "") or "").strip()
    if given:
        return Path(given).expanduser()
    house = (home or Path.home()) / "ORION"
    installed = house / "config" / "training"
    return installed if installed.is_dir() else house / "training"


def choose_backend(requested: str, find_spec: Callable[[str], Any] = importlib.util.find_spec) -> str:
    if requested != "auto":
        return requested
    return "unsloth" if find_spec("unsloth") is not None else "peft"


def installed_torch_version() -> str:
    """The installed torch distribution's version ('2.8.0+cpu', '2.8.0+cu128'), read
    without importing torch; '' when it is not installed."""
    from importlib import metadata
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return ""


def cpu_only_torch(version: str, platform: str = sys.platform) -> bool:
    """True for a PyTorch build that cannot use an NVIDIA card: '+cpu', or - on
    Windows, where the plain PyPI wheel is CPU-only - anything without '+cu'."""
    return bool(version) and "+cu" not in version and (version.endswith("+cpu") or platform == "win32")


def model_dir_problem(path: Path) -> str:
    """'' when *path* looks like a complete local Hugging Face snapshot."""
    if not path.is_dir():
        return "the folder does not exist"
    if not (path / "config.json").is_file():
        return "it has no config.json"
    if not list(path.glob("*.safetensors")):
        return "it has no .safetensors weights"
    if not ((path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file()):
        return "it has no tokenizer"
    return ""


def preflight(args: argparse.Namespace, *, find_spec: Callable[[str], Any] = importlib.util.find_spec,
              torch_version: Callable[[], str] = installed_torch_version) -> list[Missing]:
    """Everything that stops a run, checked WITHOUT importing a machine-learning library."""
    preset = PRESETS[args.preset]
    problems: list[Missing] = []
    backend = choose_backend(args.backend, find_spec)
    if args.device == "cuda":
        for name in BACKEND_PACKAGES[backend]:
            if find_spec(name) is None:
                fix, size = PACKAGES[name]
                problems.append(Missing("package", f"the Python package '{name}'", fix, size))
        version = torch_version() if find_spec("torch") is not None else ""
        if cpu_only_torch(version):
            fix, size = PACKAGES["torch"]
            problems.append(Missing("package", f"a CUDA build of PyTorch (this environment has torch "
                                    f"{version}, which cannot use the GPU)", fix, size))
    else:
        for name in ("torch", "transformers", "peft"):
            if find_spec(name) is None:
                fix, size = PACKAGES[name]
                problems.append(Missing("package", f"the Python package '{name}'", fix, size))
    folder = Path(args.base_model)
    why = model_dir_problem(folder)
    if why:
        problems.append(Missing(
            "model", f"the base model {preset.hf_id} at {folder}: {why}",
            f"huggingface-cli download {preset.hf_id} --local-dir \"{folder}\"   "
            f"(needs `pip install huggingface_hub`, and a Hugging Face login for gated models)",
            f"about {preset.download_gb:g} GB"))
    if args.dataset:
        try:
            data = files_mod.resolve_dataset(args.dataset, Path(args.training_root))
            found = files_mod.verify_dataset(data) if data.is_dir() else [f"{data} does not exist"]
        except ValueError as why_not:
            data, found = Path(args.training_root), [str(why_not)]
        if found:
            problems.append(Missing("dataset", f"a verified dataset at {data}: {found[0]}",
                                    "python -m tools.train_build_dataset"))
    else:
        problems.append(Missing("dataset", "no dataset was given", "python -m tools.train_build_dataset, "
                                "then --dataset latest"))
    return problems


def explain(problems: list[Missing], args: argparse.Namespace) -> str:
    preset = PRESETS[args.preset]
    lines = [f"Cannot train yet: {len(problems)} thing(s) are missing. Nothing was installed, "
             "downloaded or changed.", f"Python: {sys.executable}", f"Base model preset: {preset.key} "
             f"({preset.hf_id}; {preset.note})", ""]
    lines += [p.line() for p in problems]
    packages = [p for p in problems if p.kind == "package"]
    if packages:
        env = Path.home() / "ORION" / "neural-env"
        lines += ["", "Install these into a SEPARATE environment, never ORION's own, and outside the "
                  "project and OneDrive (it is several GB), for example:",
                  f"  py -3.13 -m venv {env}",
                  f"  {env}\\Scripts\\python.exe -m pip install --upgrade pip",
                  f"  (then each line above, with {env}\\Scripts\\python.exe -m before 'pip')",
                  "The existing .neural-debug-env has a CPU-only PyTorch and no bitsandbytes: fine for "
                  "`selftest`, not for training."]
    lines += ["", "Full guide: docs/LOCAL_BRAIN_TRAINING.md"]
    return "\n".join(lines)


# ── the run ───────────────────────────────────────────────────────────────────

def _versions() -> dict[str, str]:
    return base._versions()


def load_model(args: argparse.Namespace, backend: str) -> tuple[Any, Any]:
    """(model with a fresh LoRA adapter attached, tokenizer). Local files only."""
    folder = str(args.base_model)
    modules = list(args.target_modules)
    if args.device == "cpu":                                   # plumbing check only
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, dtype=torch.float32)
        model.config.use_cache = False
        lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                          target_modules=modules, bias="none", task_type="CAUSAL_LM")
        return get_peft_model(model, lora), tokenizer

    if backend == "unsloth":
        from unsloth import FastLanguageModel                    # must precede transformers
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=folder, max_seq_length=args.max_seq, dtype=None, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(
            model, r=args.rank, target_modules=modules, lora_alpha=args.alpha,
            lora_dropout=0.0, bias="none", use_gradient_checkpointing="unsloth",
            random_state=args.seed)
        FastLanguageModel.for_training(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True,
                                                 quantization_config=quant, device_map={"": 0},
                                                 dtype=dtype)
    model.config.use_cache = False
    # Not prepare_model_for_kbit_training: it up-casts the embeddings and output
    # layer to float32, which costs gigabytes an 8 GB card does not have.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=modules, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, lora), tokenizer


def make_optimiser(params: list[Any], args: argparse.Namespace) -> Any:
    import torch
    if args.device == "cuda":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=args.learning_rate, weight_decay=0.0)
    return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.0)


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    """Learning-rate multiplier: linear warm-up, then a cosine decay to 10%."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(model: Any, train_items: list[Any], val_items: list[Any], args: argparse.Namespace,
          guard: Any, work: Path) -> dict[str, Any]:
    import torch
    device = next(model.parameters()).device
    def autocast() -> Any:
        if args.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = make_optimiser(params, args)
    steps_per_epoch = math.ceil(len(train_items) / args.accumulation)
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = max(1, int(total * args.warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: cosine_with_warmup(step, total, warmup))

    def forward(item: Any) -> Any:
        ids = torch.tensor([item.input_ids], device=device)
        labels = torch.tensor([item.labels], device=device)
        with autocast():
            return model(input_ids=ids, labels=labels).loss

    def validation() -> float | None:
        if not val_items:
            return None
        model.eval()
        weighted, tokens = 0.0, 0
        with torch.no_grad():
            for item in val_items:
                weighted += float(forward(item)) * item.trainable
                tokens += item.trainable
        model.train()
        return weighted / max(1, tokens)

    metrics: dict[str, Any] = {"epochs": [], "optimiser_steps": 0, "examples": len(train_items)}
    if val_items and args.eval_base:
        with model.disable_adapter():
            metrics["val_loss_base"] = validation()
    rng = random.Random(args.seed)
    best: float | None = None
    steps = 0
    first_loss = last_loss = None
    model.train()
    stop = False
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_items)))
        rng.shuffle(order)
        running, counted = 0.0, 0
        for position, index in enumerate(order, 1):
            guard.check()
            loss = forward(train_items[index])
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError("the loss became non-finite; nothing was saved")
            (loss / args.accumulation).backward()
            running, counted = running + value, counted + 1
            first_loss = value if first_loss is None else first_loss
            last_loss = value
            if position % args.accumulation == 0 or position == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                steps += 1
                if steps % args.log_every == 0:
                    print(f"  epoch {epoch} step {steps}/{total} loss {running / counted:.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e}", file=sys.stderr, flush=True)
                if args.max_steps and steps >= args.max_steps:
                    stop = True
                    break
        val = validation()
        metrics["epochs"].append({"epoch": epoch, "train_loss": round(running / max(1, counted), 5),
                                  "val_loss": None if val is None else round(val, 5)})
        if val is None or best is None or val < best:
            best = val if val is not None else best
            model.save_pretrained(str(work / "best"), safe_serialization=True)
        if stop:
            break
    model.save_pretrained(str(work / "last"), safe_serialization=True)
    metrics.update({"optimiser_steps": steps, "train_loss_first": first_loss, "train_loss_last": last_loss,
                    "val_loss_best": best})
    if args.device == "cuda":
        metrics["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
    return metrics


def tidy_adapter(adapter_dir: Path, base_model_id: str) -> None:
    """Remove the model card and replace the local base-model PATH (which holds
    the user name) with the model's identity."""
    (adapter_dir / "README.md").unlink(missing_ok=True)
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = base_model_id
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    problems = preflight(args)
    if problems:
        print(explain(problems, args), file=sys.stderr)
        raise Refused(f"{len(problems)} requirement(s) missing: "
                      + "; ".join(p.what for p in problems)[:300])
    preset = PRESETS[args.preset]
    backend = choose_backend(args.backend)
    if args.device == "cpu" and not args.cpu_plumbing_check:
        raise Refused("CPU training is refused: it cannot produce a useful adapter. Use `selftest` "
                      "to prove the pipeline on a random tiny model.")
    dataset = files_mod.resolve_dataset(args.dataset, Path(args.training_root))
    manifest = files_mod.read_manifest(dataset)
    if manifest.get("contaminated_routing"):
        print("WARNING: this dataset trained on the routing evaluation's held-out half; any routing "
              "score measured afterwards is inflated and the export step will say so.", file=sys.stderr)
    drop = set(args.exclude_sources)
    train_rows = [r for r in files_mod.load_rows(dataset / "train.jsonl") if r.get("source") not in drop]
    val_rows = [r for r in files_mod.load_rows(dataset / "val.jsonl") if r.get("source") not in drop]
    if len(train_rows) < args.min_rows:
        raise Refused(f"only {len(train_rows)} training rows (at least {args.min_rows} are needed); "
                      "build a larger dataset or lower --min-rows knowingly")
    lease_file = Path(args.lease_file) if args.lease_file else None
    admission = ({"admitted_by": "selftest", "ticket": ""} if args.selftest_admission
                 else base.admit(lease_file, device=args.device))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with base.Lock(out_root):
        if backend == "unsloth" and args.device == "cuda":
            import unsloth  # noqa: F401  (must be imported before torch and transformers)
        import torch
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise Refused("CUDA is not available in this environment (a CPU-only PyTorch wheel?)")
            if not torch.cuda.is_bf16_supported():
                raise Refused("this GPU has no bfloat16 support (RTX 30-series or newer is needed)")
            memory = base.gate_gpu(args.reserve_gb, args.need_gb or preset.vram_gb)
        else:
            memory = {}
        guard = base.Guard(lease_file, time.monotonic() + float(args.time_budget_s))
        work = Path(tempfile.mkdtemp(prefix="orion-lora-", dir=str(out_root)))
        try:
            model, tokenizer = load_model(args, backend)
            train_items, train_counts = encode_mod.encode_rows(tokenizer, train_rows, args.max_seq)
            val_items, val_counts = encode_mod.encode_rows(tokenizer, val_rows, args.max_seq)
            if len(train_items) < args.min_rows:
                raise Refused(f"only {len(train_items)} rows fit {args.max_seq} tokens after encoding; "
                              "rebuild the dataset with a smaller --max-tokens/--tools-char-budget")
            try:
                metrics = train(model, train_items, val_items, args, guard, work)
            except base.Stop as stop:
                return {"status": "deferred", "reason": f"training stopped: {stop}"}
            chosen = work / ("best" if args.select == "best" and (work / "best").is_dir() else "last")
            tidy_adapter(chosen, args.base_model_id)
            receipt = {
                "tool": "tools/train_lora.py", "backend": backend, "preset": preset.key,
                "base_model_id": args.base_model_id,
                "base_model_fingerprint": base.fingerprint(Path(args.base_model)),
                "dataset": {"id": manifest.get("dataset_id"), "data_hash": manifest.get("data_hash"),
                            "counts": manifest.get("counts"), "contaminated_routing":
                            bool(manifest.get("contaminated_routing")),
                            "tool_surface_sha256": manifest.get("tool_surface_sha256"),
                            "system_prompt_sha256": manifest.get("system_prompt_sha256"),
                            "excluded_sources": sorted(drop)},
                "lora": {"r": args.rank, "alpha": args.alpha, "dropout": args.dropout,
                         "target_modules": list(args.target_modules)},
                "training": {"epochs": args.epochs, "learning_rate": args.learning_rate,
                             "accumulation": args.accumulation, "max_seq": args.max_seq,
                             "warmup": args.warmup, "seed": args.seed, "device": args.device,
                             "quantisation": "nf4" if args.device == "cuda" else "none",
                             "select": args.select, "seconds": round(time.time() - started, 1)},
                "rows": {"train": train_counts, "val": val_counts},
                "metrics": metrics, "gpu_memory": memory, "versions": _versions(),
                "admission": admission,
                "ollama": {"base_tag": preset.ollama_base, "adapter_format": preset.ollama_adapter},
            }
            path = candidate_mod.write_candidate(
                out_root / "candidates", chosen, base_model_id=args.base_model_id,
                data_hash=str(manifest.get("data_hash") or ""), receipt=receipt,
                dataset_manifest=manifest, plumbing_only=args.device == "cpu",
                extra={"preset": preset.key, "ollama_base_tag": preset.ollama_base,
                       "ollama_adapter_format": preset.ollama_adapter,
                       "contaminated_routing": bool(manifest.get("contaminated_routing"))})
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return {"status": "candidate", "candidate_id": path.name, "path": str(path),
            "plumbing_only": args.device == "cpu", "metrics": metrics,
            "next": f"python -m tools.train_export modelfile --candidate \"{path}\""}


# ── the plumbing self-test (CPU, random tiny model, nothing downloaded) ────────

SELFTEST_TEMPLATE = (
    "{% if tools %}{{ '<tools> ' ~ (tools | tojson) ~ ' </tools> ' }}{% endif %}"
    "{% for m in messages %}"
    "{% if m['role'] == 'assistant' and m.get('tool_calls') %}"
    "{{ '<assistant> ' ~ (m['tool_calls'][0]['function'] | tojson) ~ ' <end> ' }}"
    "{% else %}{{ '<' ~ m['role'] ~ '> ' ~ m['content'] ~ ' <end> ' }}{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}")


def selftest() -> dict[str, Any]:
    """Build a random one-layer model and a tiny dataset in a temporary folder
    and run the real code path on CPU. It proves the plumbing - template,
    masking, loop, adapter, candidate - and nothing about quality."""
    missing = [n for n in ("torch", "transformers", "peft", "tokenizers") if importlib.util.find_spec(n) is None]
    if missing:
        raise Refused(f"selftest needs {', '.join(missing)} (the CPU-only .neural-debug-env has them)")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    words = ["[UNK]", "[PAD]", "[EOS]", "[BOS]", "<system>", "<user>", "<assistant>", "<tool>", "<end>",
             "<tools>", "</tools>", "open", "notepad", "weather", "leeds", "search", "news", "{", "}",
             "ok", "done", "name", "arguments", "true", "false", "result"]
    raw = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]",
                                        eos_token="[EOS]", bos_token="[BOS]")
    tokenizer.chat_template = SELFTEST_TEMPLATE
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, eos_token_id=2, pad_token_id=1,
        bos_token_id=3))
    schema = [{"type": "function", "function": {"name": "open_app", "description": "open",
                                                "parameters": {"type": "object", "properties": {}}}}]
    rows = []
    for i in range(12):
        rows.append({"id": f"r{i}", "source": "selftest", "kind": "turn", "tools": schema, "messages": [
            {"role": "system", "content": "ok"}, {"role": "user", "content": "open notepad"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "open_app",
                                                  "arguments": {"app_name": "notepad"}}}]},
            {"role": "tool", "name": "open_app", "content": "ok done"},
            {"role": "assistant", "content": "done"}]})
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="orion-lora-selftest-") as scratch:
        root = Path(scratch)
        (root / "base").mkdir()
        model.save_pretrained(str(root / "base"), safe_serialization=True)
        tokenizer.save_pretrained(str(root / "base"))
        data = root / "data"
        data.mkdir()
        manifest_files = {}
        for name, chunk in (("train.jsonl", rows[:10]), ("val.jsonl", rows[10:]), ("preferences.jsonl", [])):
            text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in chunk)
            (data / name).write_text(text, encoding="utf-8")
            manifest_files[name] = {"rows": len(chunk), "sha256": files_mod.sha256_text(text)}
        manifest = {"format": files_mod.DATASET_FORMAT, "files": manifest_files, "dataset_id": "selftest",
                    "data_hash": files_mod.sha256_text(files_mod.canonical(
                        {k: v["sha256"] for k, v in manifest_files.items()})), "counts": {}}
        (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        args = build_parser().parse_args([
            "--dataset", str(data), "--base-model", str(root / "base"), "--base-model-id",
            "selftest/tiny-random-qwen3", "--out-root", str(root / "out"), "--device", "cpu",
            "--cpu-plumbing-check", "--epochs", "1", "--accumulation", "2", "--max-seq", "512",
            "--min-rows", "2", "--rank", "4", "--alpha", "8", "--learning-rate", "1e-3",
            "--target-modules", "q_proj", "v_proj", "--time-budget-s", "300", "--log-every", "1"])
        args.selftest_admission = True
        result = run(args)
        if result.get("status") != "candidate":
            return result
        record = json.loads((Path(result["path"]) / "candidate.json").read_text(encoding="utf-8"))
        candidate_mod.make_writable(Path(result["path"]))
        return {"status": "selftest", "plumbing_only": record["plumbing_only"],
                "adapter_files": sorted(f for f in record["files"] if f.startswith("adapter/")),
                "metrics": result["metrics"], "seconds": round(time.monotonic() - started, 2),
                "note": "A random one-layer model on CPU: this proves the pipeline runs, not that "
                        "anything was learned."}


# ── command line ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.train_lora",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="latest",
                        help="a dataset folder from tools.train_build_dataset, or 'latest'")
    parser.add_argument("--training-root", default=str(default_training_root()),
                        help="where datasets live and candidates are written "
                             f"(default ~/ORION/training, or ${ENV_TRAINING_DIR})")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--base-model", default=None,
                        help="a LOCAL folder holding the base model's Hugging Face files "
                             "(default <training root>/models/<preset>)")
    parser.add_argument("--base-model-id", default=None, help="recorded identity (default: the preset's)")
    parser.add_argument("--backend", choices=("auto", "peft", "unsloth"), default="auto")
    parser.add_argument("--out-root", default=None, help="default: the training root")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-plumbing-check", action="store_true")
    parser.add_argument("--check", action="store_true", help="only report what is missing; change nothing")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=list(ALL_LINEAR))
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--accumulation", type=int, default=16, help="batch size is 1; this many per step")
    parser.add_argument("--warmup", type=float, default=0.05, help="fraction of steps spent warming up")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = every epoch in full")
    parser.add_argument("--select", choices=("best", "last"), default="best",
                        help="keep the epoch with the lowest validation loss, or the last one")
    parser.add_argument("--eval-base", action="store_true",
                        help="also measure the validation loss of the untouched base model")
    parser.add_argument("--exclude-sources", nargs="*", default=[], help="dataset sources to leave out")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--time-budget-s", type=float, default=4 * 3600.0)
    parser.add_argument("--reserve-gb", type=float, default=1.0,
                        help="GPU memory left for the desktop and other programs")
    parser.add_argument("--need-gb", type=float, default=0.0,
                        help="override the preset's estimated VRAM need (0 = use the estimate)")
    parser.add_argument("--lease-file", default=None)
    parser.add_argument("--log-every", type=int, default=5)
    parser.set_defaults(selftest_admission=False)
    return parser


def finish_args(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    root = Path(args.training_root)
    args.base_model = str(args.base_model) if args.base_model else str(root / "models" / preset.key)
    args.base_model_id = args.base_model_id or preset.hf_id
    args.out_root = str(args.out_root) if args.out_root else str(root)
    return args


def _bounds_problem(args: argparse.Namespace) -> str:
    if not (1 <= args.accumulation <= 256 and 1 <= args.rank <= 128 and 0 < args.learning_rate < 1
            and 1 <= args.epochs <= 20 and 60 <= args.time_budget_s <= 86_400 and args.max_seq >= 256
            and 0 <= args.warmup < 0.5 and args.alpha >= 1):
        return "a training setting is out of bounds"
    return ""


# ── folding the adapter into its base, for Ollama ────────────────────────────────

MERGED_FORMAT = "orion-merged-model-v1"


def merge_candidate(candidate_dir: Path, base_model: Path, out_dir: Path, *,
                    device: str = "auto") -> dict[str, Any]:
    """Fold a candidate's LoRA adapter into its base weights and save a plain model folder.

    Ollama 0.40 refuses ``ADAPTER`` ("LoRA adapters are no longer supported", 29 Oct
    2026), so a trained candidate reaches Ollama as a model of its own, imported from
    this folder by ``tools.train_export register --merged``. The candidate is not
    touched; *out_dir* must not exist yet, and its merge.json binds it to the adapter."""
    candidate_dir, base_model, out_dir = Path(candidate_dir), Path(base_model), Path(out_dir)
    try:
        record = json.loads((candidate_dir / "candidate.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Refused(f"{candidate_dir} is not a candidate (no candidate.json)") from None
    if record.get("plumbing_only"):
        raise Refused("a plumbing-only candidate is never merged")
    adapter_files = {n: d for n, d in (record.get("files") or {}).items() if n.startswith("adapter/")}
    if not adapter_files:
        raise Refused("the candidate lists no adapter files")
    for name, digest in adapter_files.items():
        if not (candidate_dir / name).is_file() or candidate_mod.sha256_file(candidate_dir / name) != digest:
            raise Refused(f"the candidate's {name} has changed since it was trained")
    why = model_dir_problem(base_model)
    if why:
        raise Refused(f"the base model at {base_model}: {why}")
    if out_dir.exists():
        raise Refused(f"{out_dir} already exists")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    model = AutoModelForCausalLM.from_pretrained(str(base_model), local_files_only=True, dtype=torch.bfloat16,
                                                 device_map={"": 0} if use_cuda else {"": "cpu"},
                                                 low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(candidate_dir / "adapter")).merge_and_unload()
    staging = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    model.save_pretrained(str(staging), safe_serialization=True, max_shard_size="2GB")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        if (base_model / name).is_file() and not (staging / name).is_file():
            shutil.copy2(base_model / name, staging / name)
    files = {p.name: candidate_mod.sha256_file(p) for p in sorted(staging.iterdir()) if p.is_file()}
    (staging / "merge.json").write_text(json.dumps(
        {"format": MERGED_FORMAT, "candidate_id": record.get("candidate_id"),
         "adapter_sha256": record.get("adapter_sha256"), "base_model_id": record.get("base_model_id"),
         "device": "cuda" if use_cuda else "cpu", "files": files}, indent=2, sort_keys=True),
        encoding="utf-8")
    staging.rename(out_dir)
    return {"status": "merged", "out": str(out_dir), "files": len(files),
            "next": f"python -m tools.train_export register --candidate {candidate_dir} --merged {out_dir}"}


def merge_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.train_lora merge")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--training-root", default=str(default_training_root()))
    parser.add_argument("--base-model", default=None, help="default <training root>/models/<preset>")
    parser.add_argument("--out", default=None, help="default <training root>/merged/<candidate id>")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    root, candidate = Path(args.training_root), Path(args.candidate)
    try:
        record = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return base.emit({"status": "refused", "reason": f"{candidate} is not a candidate"})
    preset = PRESETS.get(str(record.get("preset") or ""), PRESETS[DEFAULT_PRESET])
    base_dir = Path(args.base_model) if args.base_model else root / "models" / preset.key
    out = Path(args.out) if args.out else root / "merged" / str(record.get("candidate_id"))
    result = merge_candidate(candidate, base_dir, out, device=args.device)
    base.emit(result)
    return EXIT_OK                       # emit() only knows the trainer's own statuses


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv[:1] == ["selftest"]:
            return base.emit(selftest())
        if argv[:1] == ["merge"]:
            return merge_main(argv[1:])
        args = finish_args(build_parser().parse_args(argv))
        why = _bounds_problem(args)
        if why:
            return base.emit({"status": "refused", "reason": why})
        if args.check:
            problems = preflight(args)
            if problems:
                print(explain(problems, args), file=sys.stderr)
                return base.emit({"status": "refused", "reason": f"{len(problems)} requirement(s) missing"})
            return base.emit({"status": "selftest", "note": "everything this run needs is present"})
        return base.emit(run(args))
    except Refused as refused:
        return base.emit({"status": "refused", "reason": str(refused)})
    except Exception as exc:                       # noqa: BLE001 - one bounded report
        return base.emit({"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"})


if __name__ == "__main__":
    raise SystemExit(main())
