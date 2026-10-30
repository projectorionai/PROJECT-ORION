"""Validate preference data before allocating GPU memory or creating output."""

from __future__ import annotations

import hashlib
import json


def validate_preferences(data, tokenizer, max_prompt: int, max_length: int):
    if not 0 < max_prompt < max_length:
        raise ValueError("Require 0 < prompt budget < total budget")
    if set(data) != {"train", "validation"} or any(
        len(data[name]) == 0 for name in data
    ):
        raise ValueError("Non-empty train and validation splits are required")
    prompt_sets = {}
    for name, split in data.items():
        prompts, pairs = set(), set()
        for example in split:
            for key in ("prompt", "chosen", "rejected"):
                messages = example.get(key)
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"{key} must be a non-empty conversation")
                for message in messages:
                    if (
                        not isinstance(message, dict)
                        or message.get("role")
                        not in {"system", "user", "assistant", "tool"}
                        or not isinstance(message.get("content"), str)
                        or not message["content"].strip()
                    ):
                        raise ValueError("Invalid conversational message")
                if key != "prompt" and (
                    len(messages) != 1 or messages[0]["role"] != "assistant"
                ):
                    raise ValueError("Each completion must be one assistant response")
            if example["chosen"] == example["rejected"]:
                raise ValueError("Identical preference answers")
            encoded_prompt = tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=True,
                add_generation_prompt=True,
            )
            encoded_completions = [
                tokenizer.apply_chat_template(
                    example["prompt"] + example[key],
                    tokenize=True,
                )
                for key in ("chosen", "rejected")
            ]
            if (
                len(encoded_prompt) > max_prompt
                or max(map(len, encoded_completions)) > max_length
            ):
                raise ValueError(
                    "Preference exceeds budget; curate or increase limits, do not truncate"
                )
            prompt_hash = hashlib.sha256(
                json.dumps(encoded_prompt).encode()
            ).hexdigest()
            pair_hash = hashlib.sha256(
                json.dumps(encoded_completions).encode()
            ).hexdigest()
            if pair_hash in pairs:
                raise ValueError("Duplicate preference pair")
            if encoded_completions[0] == encoded_completions[1]:
                raise ValueError("Preference answers are identical after tokenisation")
            prompts.add(prompt_hash)
            pairs.add(pair_hash)
        prompt_sets[name] = prompts
    if prompt_sets["train"] & prompt_sets["validation"]:
        raise ValueError("Prompt leakage between training and validation")
