"""
Rows -> token ids and loss masks, with no machine-learning library imported.

The trainer hands this module a tokenizer (anything with Hugging Face's
``apply_chat_template``) and gets back, for each row, the ids of the whole
conversation and a label list in which only the ASSISTANT's tokens are
trainable - the tool calls it makes and the sentences it speaks - and
everything else (the system prompt, the tool schemas, the owner's words, tool
results) is -100, ignored by the loss.

How the assistant's span is found: for each assistant message i, render
``messages[:i]`` with a generation prompt, and ``messages[:i+1]``; the tokens
between the two renderings are that message. Both must be prefixes of the
rendering of the whole conversation. If a chat template breaks that (some
reflow earlier turns), ``TemplateMismatch`` is raised - the alternative is a
mask in the wrong place, which trains on the prompt and teaches nothing, and
looks fine until the evaluation.

Only the ANSWER is learned: the assistant messages after the row's last
user message (the tool call, and the reply after its result). Assistant
messages before it are context - in a correction row that is ORION's original
wrong reply, and in a row given some history it is an earlier reply nobody
checked - and learning them would teach exactly what the row corrects. A
message may say otherwise with ``"train": True`` / ``False``.

Rows that do not fit ``max_len`` are DROPPED and counted, never truncated: a
cut tool call is a malformed one.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

IGNORE = -100


class TemplateMismatch(Exception):
    """The chat template does not render a conversation prefix-consistently."""


@dataclass(frozen=True)
class Encoded:
    id: str
    input_ids: list[int]
    labels: list[int]

    @property
    def trainable(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE)


def as_ids(value: Any) -> list[int]:
    """Ids from whatever ``apply_chat_template`` returned (a list, or a mapping
    with ``input_ids``, or a BatchEncoding)."""
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if value and isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple)):
        value = value[0]                                  # a batch of one
    return [int(v) for v in list(value)]


def _render(tokenizer: Any, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None,
            generation_prompt: bool) -> list[int]:
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": generation_prompt}
    if tools:
        kwargs["tools"] = list(tools)
    return as_ids(tokenizer.apply_chat_template(list(messages), **kwargs))


def encode_row(tokenizer: Any, row: dict[str, Any], max_len: int) -> Encoded | None:
    """The row's ids and labels, or None if it is longer than *max_len*."""
    messages = row["messages"]
    tools = row.get("tools") or None
    full = _render(tokenizer, messages, tools, False)
    if len(full) > max_len:
        return None
    labels = [IGNORE] * len(full)
    found = False
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if not message.get("train", i > last_user):
            continue
        before = _render(tokenizer, messages[:i], tools, True)
        upto = _render(tokenizer, messages[:i + 1], tools, False)
        if upto[:len(before)] != before or full[:len(upto)] != upto:
            raise TemplateMismatch(
                "the chat template does not render a conversation prefix-consistently, so the "
                "assistant's tokens cannot be located safely; use a base model whose template "
                "supports tools and tool calls (Llama 3.1/3.2 or Qwen 2.5 Instruct)")
        for j in range(len(before), len(upto)):
            labels[j] = full[j]
            found = True
    if not found:
        raise TemplateMismatch("a row has no assistant message to learn from")
    return Encoded(str(row.get("id") or ""), full, labels)


def encode_rows(tokenizer: Any, rows: Iterable[dict[str, Any]], max_len: int
                ) -> tuple[list[Encoded], dict[str, int]]:
    """(encoded rows, counts). ``too_long`` rows are dropped, not cut."""
    out: list[Encoded] = []
    counts = {"rows": 0, "kept": 0, "too_long": 0, "trainable_tokens": 0, "total_tokens": 0}
    for row in rows:
        counts["rows"] += 1
        item = encode_row(tokenizer, row, max_len)
        if item is None:
            counts["too_long"] += 1
            continue
        out.append(item)
        counts["kept"] += 1
        counts["trainable_tokens"] += item.trainable
        counts["total_tokens"] += len(item.input_ids)
    return out, counts


__all__ = ["Encoded", "IGNORE", "TemplateMismatch", "as_ids", "encode_row", "encode_rows"]
