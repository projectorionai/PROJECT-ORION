# Training the local brain on ORION's own conversations

## Where it stands (29 Oct 2026)

Trained: candidate `loc-20261029T115812Z-4a5faa77-dffaa0` from the identical Llama
3.2 3B weights published ungated as `unsloth/Llama-3.2-3B-Instruct` (Llama 3.2
Community Licence; no sign-in needed), 890 rows, 3 epochs, 91 minutes, peak 5.08 GiB,
validation loss 0.188 -> 0.129. Merged into its base with
`%USERPROFILE%\ORION\neural-env\Scripts\python.exe -m tools.train_lora merge --candidate <dir>`
(into `%USERPROFILE%\ORION\config\training\merged\<candidate id>`).

Not yet in Ollama. Ollama 0.40.11 refuses `ADAPTER` ("LoRA adapters are no longer
supported") and on Windows imports safetensors only through an MLX runtime it does
not ship. Left: convert the merged folder to GGUF with llama.cpp's converter (source
of release v0.8.0 is in `%USERPROFILE%\ORION\toolchain\llama.cpp-v0.8.0`):

```
%USERPROFILE%\ORION\neural-env\Scripts\python.exe %USERPROFILE%\ORION\toolchain\llama.cpp-v0.8.0\convert_hf_to_gguf.py <merged folder> --outtype q8_0 --outfile <merged folder>\model-q8_0.gguf
```

then `python -m tools.train_export register --candidate <dir> --merged <merged folder>`
(it imports the GGUF with `llama3.2:3b`'s own template) and `evaluate`.

### 28 Oct 2026

Approved and done: the training environment `%USERPROFILE%\ORION\neural-env`
(PyTorch 2.8.0+cu128, which sees the RTX 3060 Ti; bitsandbytes 0.49.2,
transformers 4.57.6, peft 0.18.1), the Ollama model `llama3.2:3b`, and the
training set `%USERPROFILE%\ORION\config\training\ds-20261028T010051Z-c82de1`
(887 rows: 817 train, 70 validation; 68 tools). `train_lora --check` reports
only the base model missing.

Left, needing your Hugging Face sign-in: accept the Llama 3.2 licence, run
`%USERPROFILE%\ORION\neural-env\Scripts\hf.exe auth login`, close ORION, then
`reports\mark_xl_4_release_2026_10_28\train_local_brain.ps1` downloads the base
model, trains, registers the adapter with Ollama and scores it. Nothing becomes
ORION's local brain until you run `train_export activate` yourself.

Baseline, before any training (`tools.eval_local_agent`, dry-run dispatcher,
28 Oct; the 8B row is the 25 Oct run):

| Model | Routing top-1 (394) | Shortlist recall | High-risk bypasses | Multi-step | Turn p50 / p95 |
|---|---|---|---|---|---|
| llama3.2:3b, untrained | 63.4% | 85.5% | 0 of 29 | 40% | 0.69 / 1.28 s |
| llama3.1 8B (ORION's local model now) | 73.6% | 85.8% | 0 of 29 | 50% | 1.77 / 3.56 s |

The 3B model answers 2.6 times faster and needs less than half the memory;
the training run has to close the 10-point routing gap for the promotion
gate to call it eligible.

The code installs nothing and downloads nothing by itself.

| Step | Command | Needs a download? |
|---|---|---|
| 1. Build the training set | `python -m tools.train_build_dataset` | No |
| 2. Fine-tune a LoRA adapter | `<env>\Scripts\python.exe -m tools.train_lora --dataset latest` | **Yes, see below** |
| 3. Hand it to Ollama, evaluate, activate | `python -m tools.train_export modelfile / register / evaluate / activate` | **Yes, one Ollama model** |

## What training still needs (all unapproved)

For the default preset, `llama3.2-3b` (Llama 3.2 3B Instruct, sized for the
RTX 3060 Ti 8 GB):

| # | What | Size | Why |
|---|---|---|---|
| 1 | **CUDA build of PyTorch**: `torch==2.8.0` from the `cu128` index | about 3 GB | `.neural-debug-env` has `2.8.0+cpu`, which cannot use the GPU. Driver 610.88 is new enough for CUDA 12.8. |
| 2 | **bitsandbytes** `>=0.48,<0.50` | about 70 MB | 4-bit (NF4) loading and the paged 8-bit optimiser that make a 3B model fit in 8 GB |
| 3 | transformers `>=4.57,<5`, peft `>=0.17,<0.19`, accelerate `>=1.0,<2`, safetensors, huggingface_hub `<1` | about 15 MB | Already in `.neural-debug-env`; needed again in the new environment |
| 4 | **Base model** `meta-llama/Llama-3.2-3B-Instruct` (Hugging Face, safetensors) | about 6.4 GB | The weights the adapter is trained on. This model is gated: you accept the Llama 3.2 Community Licence on its Hugging Face page and sign in with `huggingface-cli login` yourself. |
| 5 | **Ollama tag** `llama3.2:3b` | about 2.0 GB | Ollama applies the adapter to this. Only `llama3.1:latest` is installed now. |

That comes to about 11.5 GB. C: has 158 GB free.

Not needed for this preset: llama.cpp (only the Qwen presets need it, to
convert their adapter to GGUF), unsloth, and a new Python (the installed 3.13
has wheels for all of the above).

### The commands, once you approve them

Put the environment **outside the project folder**. The project folder is in
OneDrive, which would upload 3+ GB, and git does not ignore `.neural-env`.

```
py -3.13 -m venv %USERPROFILE%\ORION\neural-env
%USERPROFILE%\ORION\neural-env\Scripts\python.exe -m pip install --upgrade pip
%USERPROFILE%\ORION\neural-env\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
%USERPROFILE%\ORION\neural-env\Scripts\python.exe -m pip install "transformers>=4.57,<5" "peft>=0.17,<0.19" "accelerate>=1.0,<2" "bitsandbytes>=0.48,<0.50" safetensors "huggingface_hub<1"
```

Build the dataset first (step 1), because the default training folder becomes
`%USERPROFILE%\ORION\config\training` once it exists. Then:

```
%USERPROFILE%\ORION\neural-env\Scripts\huggingface-cli login
%USERPROFILE%\ORION\neural-env\Scripts\huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir "%USERPROFILE%\ORION\config\training\models\llama3.2-3b"
ollama pull llama3.2:3b
```

`%USERPROFILE%\ORION\neural-env\Scripts\python.exe -m tools.train_lora --check`
lists anything still missing, changes nothing, and exits.

## What the training set is (dry run, 27 Oct 2026)

`python -m tools.train_build_dataset --dry-run --config-dir %USERPROFILE%\ORION\config`
reads the stores read-only and prints counts only:

- 885 rows (815 train, 70 validation) and 10 preference pairs, covering 67 tools.
- 63 rows come from your own turns, 2 from your corrections and 15 from the
  reflex ledger. The rest are hand-written phrasings, the development half of
  the routing sets, generated results, safety examples, multi-step flows, chat,
  and the planner's JSON format.
- Stores read: 4,687 conversation rows, 277 tool choices, 4 jobs, 27 reflex phrases.
- The held-out routing half and every evaluation sentence (308) are kept out,
  so the promotion gate still measures something.

## Privacy

The dataset and any adapter trained on it are copies of your conversations,
redacted but still private. They are written only outside the project folder
and outside OneDrive (the builder refuses anywhere else). Never commit, share
or upload them.

The stores are opened read-only and their database files are not changed.
Because they are in WAL mode, SQLite may leave its own empty `-wal`/`-shm`
files next to them, as ORION does every time it opens them.

## Before the first real run

- Only the answer to a row's last request is learned
  (`orion_core/training/encode.py`): an earlier assistant message - ORION's
  original wrong reply in a correction row, or borrowed history - is context.
- The builder ranks each row's tool shortlist by words only, because the
  sentence encoder is not loaded from the command line. The live agent also
  ranks by meaning, so the offered tools can differ a little.
- Peak VRAM for the 3B preset is an estimate (about 4.5 GB), not a measurement.
  Close ORION, and anything else using Ollama, before training. The trainer
  takes the GPU to itself.
