"""
The shapes a training example takes - the same ones ``LocalAgent`` uses at run
time - and the checks that keep a bad call out of the data.

Nothing here is invented for training. The system prompt is
``local_agent.SYSTEM_PROMPT`` (plus its date line), the tool schemas are
``local_agent.to_ollama_tool``, the tool-result envelope is the JSON object
``local_agent._tool_message`` sends back, and the question "may the model call
this at all, and is the call high-risk?" is ``local_agent.offerable`` and
``local_agent.classify_risk``. If the agent's rules change, the next dataset
changes with them, and the manifest records a hash of what it was built against
so a stale adapter can be recognised.

Chat format. A row's ``messages`` follow the OpenAI / Hugging Face convention
that both the Llama 3.1 and the Qwen 2.5 chat templates accept:

    {"role": "system", "content": ...}
    {"role": "user", "content": ...}
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": ..., "arguments": {...}}}]}
    {"role": "tool", "name": ..., "content": "{\\"ok\\": true, \\"result\\": ...}"}
    {"role": "assistant", "content": "<the spoken reply>"}

One tool call per assistant message: the Llama 3.1 template accepts no more,
and ``LocalAgent`` runs several rounds anyway.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any, Iterable, Sequence

#: Tool results are cut to this before the model sees them (local_agent.RESULT_CHARS).
DEFAULT_RESULT_CHARS = 1500
#: A conservative characters-per-token for JSON-heavy chat text; the trainer
#: measures with the real tokenizer.
CHARS_PER_TOKEN = 3.0


def agent_module() -> Any:
    """``orion_core.local_agent``, imported on first use."""
    from .. import local_agent
    return local_agent


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


def system_prompt(when: datetime) -> str:
    """The agent's system message, exactly as ``LocalAgent._messages`` builds it."""
    return agent_module().SYSTEM_PROMPT + f"\nIt is {when.strftime('%A %d %B %Y, %H:%M')}."


def system_prompt_hash() -> str:
    return sha256_text(agent_module().SYSTEM_PROMPT)


def system_message(when: datetime) -> dict[str, Any]:
    return {"role": "system", "content": system_prompt(when)}


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant_text(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def assistant_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function",
                            "function": {"name": name, "arguments": dict(args)}}]}


def tool_result(name: str, ok: bool, text: str, *, limit: int = DEFAULT_RESULT_CHARS) -> dict[str, Any]:
    """A tool message, in the envelope ``LocalAgent`` returns results in."""
    return {"role": "tool", "name": name,
            "content": json.dumps({"ok": bool(ok), "result": str(text or "")[:limit]},
                                  ensure_ascii=False)}


def calls_in(messages: Sequence[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Every (tool, arguments) the assistant messages propose, in order."""
    found = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            found.append((str(function.get("name") or ""), dict(function.get("arguments") or {})))
    return found


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    and math.isfinite(float(v)),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class ToolBook:
    """The tools a local agent may be offered, with their schemas.

    ``declarations`` defaults to ORION's real ``TOOL_DECLARATIONS``; tests pass
    a small list. A tool is *offerable* when ``local_agent.offerable`` says so
    (never the ones that run other tools or code), and the local-only brain
    mode's cloud-AI exclusions apply.
    """

    def __init__(self, declarations: Sequence[dict[str, Any]] | None = None) -> None:
        la = agent_module()
        if declarations is None:
            from ..dispatch_schema import TOOL_DECLARATIONS
            declarations = TOOL_DECLARATIONS
        self._la = la
        self.declarations: dict[str, dict[str, Any]] = {}
        for decl in declarations:
            name = str(decl.get("name") or "")
            if name and la.offerable(name, local_only=True):
                self.declarations[name] = decl
        self.all_names: frozenset[str] = frozenset(str(d.get("name") or "") for d in declarations)
        self._schema: dict[str, dict[str, Any]] = {}
        self._chars: dict[str, int] = {}

    # ── lookup ───────────────────────────────────────────────────────────────

    def names(self) -> list[str]:
        return sorted(self.declarations)

    def has(self, name: str) -> bool:
        return name in self.declarations

    def schema(self, name: str) -> dict[str, Any]:
        if name not in self._schema:
            self._schema[name] = self._la.to_ollama_tool(self.declarations[name])
        return self._schema[name]

    def chars(self, name: str) -> int:
        if name not in self._chars:
            self._chars[name] = len(json.dumps(self.schema(name), ensure_ascii=False))
        return self._chars[name]

    def properties(self, name: str) -> dict[str, Any]:
        """The properties the model is shown (approval-carrying ones removed)."""
        schema = self.schema(name)["function"]["parameters"]
        return dict(schema.get("properties") or {})

    def required(self, name: str) -> list[str]:
        return list(self.schema(name)["function"]["parameters"].get("required") or [])

    def fingerprint(self) -> str:
        """A hash of every offerable tool's schema - what the surface looked like."""
        return sha256_text(canonical({n: self.schema(n) for n in self.names()}))

    # ── checks ───────────────────────────────────────────────────────────────

    def risk(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        """(read | act | high, why) exactly as the live agent classifies this call."""
        return self._la.classify_risk(name, args, builtin=self.all_names)

    def validate(self, name: str, args: Any, *, allow_dangerous: bool = False) -> str:
        """'' when the live agent would accept this call as written, else why not."""
        if not self.has(name):
            return f"{name!r} is not a tool the local agent offers"
        if not isinstance(args, dict):
            return "arguments are not an object"
        try:
            json.dumps(args, allow_nan=False)
        except (TypeError, ValueError):
            return "arguments are not plain JSON"
        for key in args:
            if self._la.is_approval_key(key):
                return f"carries an approval argument ({key!r})"
        if not allow_dangerous and self._la.dangerous_payload(args):
            return "carries a destructive system command"
        properties = self.properties(name)
        if properties:
            for key in args:
                if key not in properties:
                    return f"{key!r} is not a parameter of {name}"
        for key in self.required(name):
            value = args.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                return f"required parameter {key!r} is missing"
        for key, value in args.items():
            spec = properties.get(key) or {}
            kind = spec.get("type")
            check = _TYPE_CHECKS.get(str(kind))
            if check is not None and not check(value):
                return f"{key!r} should be a {kind}"
            enum = spec.get("enum")
            if enum and value not in enum:
                return f"{key!r} is not one of the allowed values"
        return ""

    # ── what to offer ────────────────────────────────────────────────────────

    def pack(self, targets: Iterable[str], ranked: Sequence[str], *, max_tools: int,
             char_budget: int) -> list[str]:
        """The tools a training example offers: every tool the answer needs,
        then the best-ranked others, within a size budget. The result is in
        rank order (the agent offers its shortlist best-first), so the right
        tool is not always first.

        Training offers fewer tools than the live shortlist (up to 8) because
        a full shortlist is ~2,000 tokens of schema and would not fit in the
        training sequence length; the distractors it does keep are the ones the
        agent's own ranking found most tempting.
        """
        chosen: list[str] = []
        for name in targets:
            if self.has(name) and name not in chosen:
                chosen.append(name)
        used = sum(self.chars(n) for n in chosen)
        for name in ranked:
            if len(chosen) >= max_tools:
                break
            if name in chosen or not self.has(name):
                continue
            if used + self.chars(name) > char_budget:
                continue
            chosen.append(name)
            used += self.chars(name)
        order = {name: i for i, name in enumerate(ranked)}
        return sorted(chosen, key=lambda n: (order.get(n, len(order)), n))


def tool_schemas(book: ToolBook, names: Iterable[str]) -> list[dict[str, Any]]:
    return [book.schema(n) for n in names if book.has(n)]


def failure_looks_like(text: str) -> bool:
    """Whether an assistant sentence reports that something did not work."""
    return bool(re.search(
        r"(?i)\b(?:couldn'?t|could not|can'?t|cannot|unable|failed|failure|error|sorry|"
        r"didn'?t work|not working|no luck|wasn'?t able)\b", str(text or "")))


__all__ = ["CHARS_PER_TOKEN", "DEFAULT_RESULT_CHARS", "ToolBook", "agent_module", "assistant_call",
           "assistant_text", "calls_in", "canonical", "estimate_tokens", "failure_looks_like",
           "sha256_text", "system_message", "system_prompt", "system_prompt_hash",
           "tool_result", "tool_schemas", "user_message"]
