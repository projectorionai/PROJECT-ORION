"""
Tool use for the TEXT path — so ORION can still act when Live is down.

Before this, the text fallback (``GenAILiveWorker._submit_text_fallback``) only
ever generated words and spoke them. Every request that needed the machine —
open a document, clear a task, show the research — got a promise instead of an
action, and the models, knowing they were an assistant with tools, wrote tool
calls anyway, in their own house styles:

    <tool_call>
    <function=dev_workbench>
    <parameter=command>list</parameter>
    </function>
    </tool_call>

which nothing parsed, so the markup was stored as his reply and nothing ran.
That is what "he can't open my Word documents" looked like whenever the voice
channel was on the text fallback (which, until the Live schema fix, was most
of the time).

This module gives the text path the same act -> observe -> answer loop the
Live path has, through the SAME dispatcher (so every guard — confirmation
tokens, protected paths, the spend gate — applies unchanged):

* a compact menu of the ~10 tools most relevant to the request (the resolver's
  hybrid word+meaning ranking), not all 167 — small free-tier models have
  token-per-minute budgets and choose worse from long lists;
* one protocol to ask for, and a parser that also understands the dialects
  models actually emit (Hermes JSON, Qwen XML parameters, fenced JSON,
  ``TOOL:``/``ARGS:`` lines);
* never speaks markup: whatever reaches the user has every tool block removed.

Pure apart from the injected ``generate`` and ``dispatch`` callables.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

#: Tool calls the loop will run for one request before it must answer.
MAX_STEPS = 4
#: Tools offered per request.
MENU_SIZE = 10
#: Characters of one tool result shown back to the model.
RESULT_CHARS = 1800

_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL | re.IGNORECASE)
_XML_FUNCTION = re.compile(r"<function\s*=\s*([\w.\-]+)\s*>(.*?)(?:</function>|$)",
                           re.DOTALL | re.IGNORECASE)
_XML_PARAM = re.compile(r"<parameter\s*=\s*([\w.\-]+)\s*>(.*?)(?:</parameter>|(?=<parameter)|$)",
                        re.DOTALL | re.IGNORECASE)
_FENCED = re.compile(r"```(?:tool_call|tool|json)?\s*(\{.*?\})\s*```", re.DOTALL)
_TOOL_LINE = re.compile(r"^\s*TOOL\s*[:=]\s*([A-Za-z0-9_.\-]+)\s*$", re.MULTILINE)
# Anything that must never be spoken, even half-formed. Blocks run to their
# closing tag or the END OF THE TEXT (\Z — a multiline "$" would stop a lazy
# match at the first line break and speak the rest of the block).
_MARKUP_BLOCKS = re.compile(
    r"<tool_call>.*?(?:</tool_call>|\Z)|</?tool_call>|<function\s*=.*?(?:</function>|\Z)"
    r"|```(?:tool_call|tool)\s*\{.*?\}\s*```",
    re.DOTALL | re.IGNORECASE)
_MARKUP_LINES = re.compile(r"^\s*(?:TOOL|ARGS)\s*[:=].*$", re.IGNORECASE | re.MULTILINE)


@dataclass
class ToolStep:
    name: str
    args: dict[str, Any]
    ok: bool
    result: str


@dataclass
class TextTurn:
    """The outcome of one text-path request."""

    reply: str
    steps: list[ToolStep] = field(default_factory=list)
    provider: Any = None


# ── the menu ──────────────────────────────────────────────────────────────────

def _first_sentence(text: str, limit: int = 180) -> str:
    text = " ".join(str(text or "").split())
    match = re.match(r"(.+?[.!?])(\s|$)", text)
    head = match.group(1) if match else text
    return head if len(head) <= limit else head[: limit - 1] + "…"


def _param_line(name: str, spec: Any, required: set[str]) -> str:
    kind = str((spec or {}).get("type") or "string").lower() if isinstance(spec, dict) else "string"
    desc = _first_sentence((spec or {}).get("description", "") if isinstance(spec, dict) else "", 90)
    mark = "*" if name in required else ""
    return f"{name}{mark}: {kind}" + (f" ({desc})" if desc else "")


def render_menu(declarations: Sequence[dict[str, Any]]) -> str:
    lines = []
    for decl in declarations:
        params = decl.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        sig = "; ".join(_param_line(k, v, required) for k, v in list(props.items())[:8])
        lines.append(f"- {decl.get('name')}({sig}) — {_first_sentence(decl.get('description', ''))}")
    return "\n".join(lines)


def choose_tools(query: str, declarations: Sequence[dict[str, Any]],
                 size: int = MENU_SIZE) -> list[dict[str, Any]]:
    """The declarations most relevant to *query* (word + meaning ranking)."""
    try:
        from . import tool_resolver as tr
        scorer = getattr(tr, "hybrid_scores", None) or tr.lexical_scores
        return tr.resolve(query, None, list(declarations), enabled=True,
                          max_tools=size, scorer=scorer)
    except Exception:
        return list(declarations)[:size]


PROTOCOL = (
    "You can act on this computer through the tools below. To use one, reply "
    "with ONLY this, and nothing else:\n"
    '<tool_call>{"name": "TOOL_NAME", "arguments": {"param": "value"}}</tool_call>\n'
    "You will be shown the result, then you may call another tool or answer. "
    "When the request is done, or needs no tool, reply in plain spoken English "
    "with no tool markup at all. Never say something has been done unless a "
    "tool result in this conversation confirms it; if a tool failed, say so "
    "plainly. Parameters marked * are required.\nTOOLS:\n"
)


# ── the parser ────────────────────────────────────────────────────────────────

def _value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if text[0] in "{[\"" or text in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", text):
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def _from_json(blob: str) -> tuple[str, dict[str, Any]] | None:
    try:
        data = json.loads(blob)
    except ValueError:
        # Tolerate a trailing comma / single quotes, the two commonest slips.
        try:
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", blob.replace("'", '"')))
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("function"), dict):          # OpenAI-shaped
        data = data["function"]
    name = str(data.get("name") or data.get("tool") or "").strip()
    if not name:
        return None
    args = data.get("arguments", data.get("args", data.get("parameters", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return name, dict(args) if isinstance(args, dict) else {}


def _from_xml(body: str) -> list[tuple[str, dict[str, Any]]]:
    calls = []
    for fn in _XML_FUNCTION.finditer(body):
        args: dict[str, Any] = {}
        for param in _XML_PARAM.finditer(fn.group(2)):
            args[param.group(1)] = _value(param.group(2))
        # A lone "arguments" parameter holding a JSON object is the model
        # wrapping the real arguments one level too deep: unwrap it.
        if set(args) == {"arguments"} and isinstance(args["arguments"], dict):
            args = args["arguments"]
        calls.append((fn.group(1), args))
    return calls


def parse_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Every tool call in *text*, in any dialect we have seen models use."""
    raw = str(text or "")
    calls: list[tuple[str, dict[str, Any]]] = []
    for block in _BLOCK.finditer(raw):
        body = block.group(1).strip()
        if "<function" in body.lower():
            calls.extend(_from_xml(body))
        else:
            parsed = _from_json(body)
            if parsed:
                calls.append(parsed)
    if not calls and "<function" in raw.lower():
        calls.extend(_from_xml(raw))
    if not calls:
        for fenced in _FENCED.finditer(raw):
            parsed = _from_json(fenced.group(1))
            if parsed:
                calls.append(parsed)
    if not calls:
        for match in _TOOL_LINE.finditer(raw):
            tail = raw[match.end():]
            args_match = re.match(r"\s*ARGS\s*[:=]\s*(\{.*?\})\s*(?:\n|$)", tail, re.DOTALL)
            args = {}
            if args_match:
                parsed = _from_json('{"name":"x","arguments":' + args_match.group(1) + "}")
                args = parsed[1] if parsed else {}
            calls.append((match.group(1), args))
    if not calls:
        stripped = raw.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            parsed = _from_json(stripped)
            if parsed and ("arguments" in stripped or "args" in stripped):
                calls.append(parsed)
    return calls


def strip_markup(text: str) -> str:
    """The reply with every tool block removed — what may be spoken."""
    cleaned = _MARKUP_LINES.sub("", _MARKUP_BLOCKS.sub("", str(text or "")))
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# ── the loop ──────────────────────────────────────────────────────────────────

Generate = Callable[[str, str], Awaitable[tuple[Any, str]]]
Dispatch = Callable[[str, dict[str, Any]], Awaitable[Any]]


def _result_text(result: Any) -> tuple[bool, str]:
    ok = bool(getattr(result, "ok", True))
    text = str(getattr(result, "text", result) or "").strip() or "(no output)"
    if len(text) > RESULT_CHARS:
        text = text[:RESULT_CHARS] + " …[truncated]"
    return ok, text


async def run(request: str, declarations: Sequence[dict[str, Any]],
              generate: Generate, dispatch: Dispatch, *,
              log: Callable[[str], None] | None = None,
              max_steps: int = MAX_STEPS) -> TextTurn:
    """Answer *request*, running tools the model asks for along the way.

    *generate(prompt, system_extra)* returns (provider, text); *dispatch(name,
    args)* returns a ToolResult-like object. A name the menu did not offer is
    still dispatched if it is a real tool — the model may know the right one
    better than the ranking did — and the dispatcher refuses unknown names.
    """
    menu = choose_tools(request, declarations)
    known = {d.get("name") for d in declarations}
    system_extra = PROTOCOL + render_menu(menu)
    transcript: list[str] = []
    steps: list[ToolStep] = []
    provider: Any = None
    for step in range(max_steps + 1):
        prompt = request if not transcript else (
            f"{request}\n\n" + "\n".join(transcript)
            + ("\n\nNow answer the user in plain spoken English — no more tool calls."
               if step == max_steps else ""))
        provider, reply = await generate(prompt, system_extra)
        calls = parse_calls(reply)
        if not calls or step == max_steps:
            final = strip_markup(reply)
            if not final and steps:
                last = steps[-1]
                final = (last.result if last.ok else f"That did not work: {last.result}")[:600]
            return TextTurn(final or "I'm not sure how to help with that.", steps, provider)
        for name, args in calls[:3]:
            if name not in known:
                observation = (f"[{name}] no such tool. Choose from the list.")
                transcript.append(f"TOOL RESULT {observation}")
                steps.append(ToolStep(name, args, False, "no such tool"))
                continue
            if log:
                log(f"TOOL: {name} requested (text path).")
            try:
                ok, text = _result_text(await dispatch(name, dict(args)))
            except Exception as exc:          # a tool fault is a result, not a crash
                ok, text = False, f"{name} failed: {exc}"
            steps.append(ToolStep(name, args, ok, text))
            transcript.append(
                f"TOOL RESULT [{name}({json.dumps(args, default=str)[:200]})] "
                f"{'' if ok else 'FAILED: '}{text}")
    return TextTurn("I could not finish that.", steps, provider)


__all__ = ["MAX_STEPS", "PROTOCOL", "TextTurn", "ToolStep", "choose_tools",
           "parse_calls", "render_menu", "run", "strip_markup"]
