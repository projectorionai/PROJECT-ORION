"""Offline candidate training. Never import this from a request handler.

    python -m docs.neural_reference.train_dpo MODEL_DIR TRAIN.jsonl VALID.jsonl OUTPUT_DIR

JSONL rows contain prompt/chosen/rejected conversational message arrays. Both
answers must address the same prompt. Curate and split by whole task/session.
This script starts a NEW adapter from the immutable base on each learning cycle.
It deliberately does not deploy the resulting adapter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .training_data import validate_preferences


def training_config(output_dir, max_prompt, max_length, *, use_cpu=False):
    """CPU is for tiny-model integration checks, not 4-bit production training."""
    from trl import DPOConfig

    return DPOConfig(
        output_dir=str(output_dir),
        beta=0.1,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        max_length=max_length,
        max_prompt_length=max_prompt,
        fp16=not use_cpu,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch" if use_cpu else "paged_adamw_8bit",
        use_cpu=use_cpu,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=42,
    )


def train(model_dir: str, train_file: str, valid_file: str, output_dir: str):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOTrainer

    if Path(output_dir).exists():
        raise ValueError(
            "Use a new candidate directory; never overwrite an active adapter"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    data = load_dataset(
        "json", data_files={"train": train_file, "validation": valid_file}
    )
    max_length = int(os.environ.get("ORION_DPO_MAX_LENGTH", "1024"))
    max_prompt = int(os.environ.get("ORION_DPO_MAX_PROMPT", "768"))
    validate_preferences(data, tokenizer, max_prompt, max_length)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this quantised training command")
    # Atomically reserve the directory; two trainers must not share a candidate.
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    config = training_config(output_dir, max_prompt, max_length)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        processing_class=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
    )
    # In the pinned TRL version, a fresh PEFT adapter uses the adapter-disabled
    # immutable base as reference. Do not reuse this recipe for an existing
    # trained adapter without explicitly defining the frozen reference policy.
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("Expected MODEL_DIR TRAIN.jsonl VALID.jsonl NEW_OUTPUT_DIR")
    train(*sys.argv[1:])
