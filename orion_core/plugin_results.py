"""Preserve explicit plugin outcomes instead of turning failures into success."""
from __future__ import annotations

from collections.abc import Mapping

from .data import ToolResult


def plugin_result(value: object) -> ToolResult:
    if isinstance(value, ToolResult):
        if type(value.ok) is not bool or not isinstance(value.text, str):
            return ToolResult("Plugin returned an invalid outcome: expected boolean ok and text result.", ok=False)
        if len(value.text) > 131_072:
            return ToolResult("Plugin result exceeded its text limit; completion and side effects are unverified.", ok=False)
        return value
    if isinstance(value, Mapping):
        ok = value.get("ok")
        text = value.get("result", value.get("text"))
        if type(ok) is not bool or not isinstance(text, str):
            return ToolResult("Plugin returned an invalid outcome: expected boolean ok and text result.", ok=False)
        if len(text) > 131_072:
            return ToolResult("Plugin result exceeded its text limit; completion and side effects are unverified.", ok=False)
        receipt = value.get("receipt")
        if receipt is not None and not isinstance(receipt, Mapping):
            return ToolResult("Plugin returned an invalid receipt object; completion was not verified.", ok=False)
        return ToolResult(text, ok=ok, receipt=dict(receipt) if receipt is not None else None)
    if value is None:
        return ToolResult("Plugin returned no outcome; completion was not verified.", ok=False)
    if isinstance(value, str):
        if len(value) > 131_072:
            return ToolResult("Plugin result exceeded its text limit; completion and side effects are unverified.", ok=False)
        # Legacy plugins return text. Preserve compatibility, but never promote
        # their explicit failure messages to successful ToolResults.
        failed = value.lstrip().lower().startswith((
            "error:", "error in ", "an error occurred:", "failed to ",
            "failed:", "could not ",
        ))
        return ToolResult(value, ok=not failed)
    return ToolResult("Plugin returned an unsupported outcome; completion was not verified.", ok=False)
