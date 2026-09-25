"""
The Live tool gateway — everyday tools direct, everything else one hop away.

MEASURED on the real Gemini Live endpoint (Sep 25 2026), one trivial turn:

    all 167 tool declarations   42,329 prompt tokens   first audio 1.34 s
    10 declarations              2,807 prompt tokens   first audio 0.84 s

Tool schemas sit in the session context, so every spoken turn carried ~40k
input tokens and ~half a second of prefill just to describe tools most turns
never touch — and a model choosing among 167 chooses worse than among 45.

So the Live session gets the CORE set — what a voice conversation reaches for
day to day (~10.5k tokens) — plus two gateway tools:

* ``find_tool(need)`` searches the FULL registry (built-in, MCP, forged,
  plugins) by meaning and words, and returns the best matches with their
  parameters;
* ``use_tool(name, arguments)`` runs any of them through the same dispatcher,
  so every confirmation token, spend gate and plugin gate still applies.

Nothing becomes unreachable; a rare tool costs one extra hop. The text path is
unaffected (it already offers a ranked menu per request). ORION_LIVE_TOOLS=full
restores the whole list in the Live session.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .data import ToolResult

#: Direct in the Live session. Chosen by what a spoken session reaches for
#: every day; the heavy specialist suites (chess, perception, study, finance,
#: security labs, marketing, plugins ...) are one find_tool away.
LIVE_CORE_TOOLS: frozenset[str] = frozenset("""
interface_control open_app close_app window_control media_control desktop_control
find_files process_file file_controller fetch_url clipboard_operate capture_screen vision_analyse
sound_sense web_search browser_control web_control open_news save_memory
query_intelligence recall_conversation awareness reminder telephony messaging
outlook_mail notion_workspace morning_briefing research capabilities audio_devices
system_notify shutdown_orion restart_orion execute_plan mission globe aviation
catch_up agent_dispatch resource_status undo document_export protocol
find_tool use_tool
""".split())

GATEWAY_NOTE = (
    "TOOLS: your direct tools cover everyday actions. For anything else — "
    "chess, study, finance, security, marketing, plugins, MCP services such as "
    "Notion or search servers, diagnostics, and more — call find_tool with what "
    "you need, then use_tool with the exact name and arguments it returns. Never "
    "tell the user you cannot do something before checking find_tool. Fetch a "
    "web URL with fetch_url; read a local file with process_file (find_files "
    "first if its path is unknown).")


def gated() -> bool:
    return os.getenv("ORION_LIVE_TOOLS", "core").strip().lower() not in {
        "full", "all", "0", "off"}


def live_declarations(declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The declarations the Live session carries (core + gateway, or all)."""
    if not gated():
        return list(declarations)
    kept = [d for d in declarations if d.get("name") in LIVE_CORE_TOOLS]
    names = {d.get("name") for d in kept}
    if "find_tool" not in names or "use_tool" not in names:
        # A bare/test list without the gateway: never strand the rest.
        return list(declarations)
    return kept


def _first_sentence(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    for stop in (". ", "! ", "? "):
        cut = text.find(stop)
        if 0 < cut < limit:
            return text[:cut + 1]
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _signature(decl: dict[str, Any]) -> str:
    params = decl.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    parts = []
    for name, spec in list(props.items())[:10]:
        kind = str((spec or {}).get("type", "string")).lower() if isinstance(spec, dict) else "string"
        hint = _first_sentence((spec or {}).get("description", "") if isinstance(spec, dict) else "", 80)
        parts.append(f"{name}{'*' if name in required else ''} ({kind}){': ' + hint if hint else ''}")
    return "; ".join(parts) or "no arguments"


def find_tool(dispatcher: Any, args: dict[str, Any]) -> ToolResult:
    need = str(args.get("need") or args.get("query") or args.get("text") or "").strip()
    if not need:
        return ToolResult("Say what you need a tool for.", ok=False)
    declarations = [d for d in (getattr(dispatcher, "TOOL_DECLARATIONS", None) or [])
                    if d.get("name") not in {"find_tool", "use_tool"}]
    try:
        from . import tool_resolver as tr
        scorer = getattr(tr, "hybrid_scores", None) or tr.lexical_scores
        ranked = tr.resolve(need, None, declarations, enabled=True, max_tools=9, scorer=scorer)
        ranked = [d for d in ranked if d.get("name") not in tr.ALWAYS_INCLUDE] or ranked
    except Exception:
        ranked = declarations[:6]
    lines = [f"Tools for '{need}' (call use_tool with name and arguments as JSON):"]
    for decl in ranked[:6]:
        lines.append(f"- {decl['name']}: {_first_sentence(decl.get('description', ''))} "
                     f"Arguments: {_signature(decl)}")
    return ToolResult("\n".join(lines))


async def use_tool(dispatcher: Any, args: dict[str, Any]) -> ToolResult:
    name = str(args.get("name") or args.get("tool") or "").strip()
    raw = args.get("arguments", args.get("args", {}))
    if isinstance(raw, str):
        text = raw.strip()
        try:
            raw = json.loads(text) if text else {}
        except ValueError:
            return ToolResult("use_tool 'arguments' must be a JSON object, e.g. "
                              '{"action": "status"}.', ok=False)
    if not isinstance(raw, dict):
        return ToolResult("use_tool 'arguments' must be a JSON object.", ok=False)
    if not name:
        return ToolResult("use_tool needs the tool's name (from find_tool).", ok=False)
    if name in {"find_tool", "use_tool"}:
        return ToolResult("Call that tool directly, not through use_tool.", ok=False)
    known = {d.get("name") for d in (getattr(dispatcher, "TOOL_DECLARATIONS", None) or [])}
    if known and name not in known:
        hint = find_tool(dispatcher, {"need": name.replace("_", " ")}).text
        return ToolResult(f"There is no tool called '{name}'. {hint}", ok=False)
    return await dispatcher.dispatch_chain(name, raw)


__all__ = ["GATEWAY_NOTE", "LIVE_CORE_TOOLS", "find_tool", "gated", "live_declarations",
           "use_tool"]
