"""Hosted Gemma 4 through Gemini's native GenerateContent API.

The ordinary chat provider API cannot execute a function.  This transport may
return function-call *proposals* to an authorised caller, but it never calls a
desktop tool or treats a proposal as proof that an action happened.
"""

from __future__ import annotations

import base64
import codecs
import json
import os
import re
import time
from typing import Any

from . import provider_transport as wire
from .gemini_schema import sanitise_declarations
from .provider_diagnostics import redacted_error_line


GEMMA_MODELS = frozenset({"gemma-4-26b-a4b-it", "gemma-4-31b-it"})
GEMMA_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MAX_BODY_BYTES = 2_097_152


def gemma_payload(req: wire.ChatRequest) -> dict[str, Any]:
    """Create the documented native request, keeping credentials out of JSON."""
    if req.model not in GEMMA_MODELS:
        raise ValueError("This provider only supports hosted Gemma 4 models")
    user_parts: list[dict[str, Any]] = [{"text": req.prompt}]
    if req.image_jpeg is not None:
        user_parts.append({"inlineData": {"mimeType": "image/jpeg",
                                          "data": base64.b64encode(req.image_jpeg).decode("ascii")}})
    contents: list[dict[str, Any]] = [{"role": "user", "parts": user_parts}]
    if req.continuation:
        contents.extend([
            {"role": "model", "parts": [{"text": req.continuation}]},
            {"role": "user", "parts": [{"text": wire.CONTINUE_REPLY}]},
        ])
    thinking = req.extra.get("thinking_level")
    if thinking is None:
        thinking = ("high" if os.getenv("ORION_MODEL_THINKING", "").strip().lower()
                    in {"1", "true", "on", "yes"} else "minimal")
    if thinking not in {"minimal", "high"}:
        raise ValueError("Gemma thinking level must be minimal or high")
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max(1, int(req.max_tokens)),
                             "temperature": float(req.temperature),
                             "thinkingConfig": {"thinkingLevel": thinking}},
    }
    if req.system:
        payload["systemInstruction"] = {"parts": [{"text": req.system}]}
    declarations = req.extra.get("function_declarations")
    if declarations is not None:
        if not isinstance(declarations, list):
            raise ValueError("Gemma function declarations must be a list")
        clean, dropped = sanitise_declarations(declarations)
        if dropped or not clean:
            raise ValueError("Gemma function declarations were invalid")
        payload["tools"] = [{"functionDeclarations": clean}]
    return payload


def _parse_candidate(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    if data.get("error"):
        raise RuntimeError("Gemma reported an error while answering")
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    parts = (candidate.get("content") or {}).get("parts") or []
    text: list[str] = []
    proposals: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str) and not part.get("thought"):
            text.append(part["text"])
        call = part.get("functionCall")
        if isinstance(call, dict):
            name = call.get("name")
            args = call.get("args")
            if not isinstance(name, str) or not isinstance(args, dict):
                raise ValueError("Gemma returned an invalid function proposal")
            proposals.append({"name": name, "args": args,
                              "id": str(call.get("id") or "")})
    return "".join(text), proposals, str(candidate.get("finishReason") or "")


def _usage(data: dict[str, Any]) -> dict[str, int]:
    raw = data.get("usageMetadata") or {}
    return {
        "prompt_tokens": int(raw.get("promptTokenCount") or 0),
        "completion_tokens": int(raw.get("candidatesTokenCount") or 0),
        "total_tokens": int(raw.get("totalTokenCount") or 0),
    }


class GemmaNativeTransport:
    """GenerateContent over pooled HTTP, with bounded JSON and SSE parsing."""

    kind = "gemma_native"

    def __init__(self, base_url: str = GEMMA_API_BASE) -> None:
        self.root = str(base_url or GEMMA_API_BASE).rstrip("/")
        if self.root != GEMMA_API_BASE:
            raise ValueError("Hosted Gemma requires the official Gemini API endpoint")

    def endpoint(self, model: str, stream: bool) -> str:
        if model not in GEMMA_MODELS:
            raise ValueError("This provider only supports hosted Gemma 4 models")
        action = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"{self.root}/models/{model}:{action}"

    async def chat(self, session: Any, req: wire.ChatRequest, *, timeout: Any,
                   trace: dict[str, Any]) -> wire.ChatResult:
        if not req.api_key.strip():
            raise ValueError("Gemma needs a Gemini API key")
        payload = gemma_payload(req)
        headers = {"Content-Type": "application/json", "x-goog-api-key": req.api_key}
        if req.stream:
            headers["Accept"] = "text/event-stream"
        started = time.monotonic()
        response = await wire._send(session, "POST", self.endpoint(req.model, req.stream),
                                    trace=trace, json=payload, headers=headers,
                                    timeout=timeout)
        try:
            if response.status >= 400:
                raw = (await response.content.read(16_384)).decode("utf-8", "replace")
                detail = redacted_error_line(wire._error_text(raw), 300)
                daily = " per-day quota" if re.search(r"(?i)per.?day|daily", raw) else ""
                retry = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', raw)
                delay = f' retryDelay: {retry.group(1)}s' if retry else ""
                raise RuntimeError(f"Gemma HTTP {response.status}: {detail}{daily}{delay}")
            if req.stream and "json" not in str(response.headers.get("Content-Type", "")).lower():
                result = await self._read_stream(response, req)
            else:
                raw = await wire._read_bounded(response,
                                               min(req.max_response_bytes or MAX_BODY_BYTES,
                                                   MAX_BODY_BYTES))
                data = json.loads(raw)
                text, proposals, reason = _parse_candidate(data)
                result = wire.ChatResult(text=text, usage=_usage(data), model=req.model,
                                         streamed=False, done_reason=reason,
                                         tool_calls=proposals)
                if req.stream and text:
                    result.first_token_ms = (time.monotonic() - started) * 1_000.0
                    wire._emit(req.on_delta, text)
            allowed = {decl["name"] for group in payload.get("tools", [])
                       for decl in group.get("functionDeclarations", [])}
            if result.tool_calls and (not allowed or any(
                    call["name"] not in allowed for call in result.tool_calls)):
                raise RuntimeError("Gemma proposed an undeclared function")
            if result.done_reason not in {"STOP", "MAX_TOKENS"}:
                raise RuntimeError("Gemma did not finish its response")
            if result.done_reason == "MAX_TOKENS" and not result.text.strip() \
                    and not result.tool_calls:
                raise RuntimeError("Gemma used its output budget before answering")
            return wire._timings(result, trace, started)
        except BaseException:
            response.close()
            raise
        finally:
            response.release()

    async def _read_stream(self, response: Any, req: wire.ChatRequest) -> wire.ChatResult:
        decoder = codecs.getincrementaldecoder("utf-8")()
        pending = ""
        lines: list[str] = []
        text: list[str] = []
        proposals: list[dict[str, Any]] = []
        reason = ""
        usage: dict[str, int] = {}
        total = 0
        first_token: float | None = None
        started = time.monotonic()

        def consume_event() -> None:
            nonlocal reason, usage, first_token
            if not lines:
                return
            raw = "\n".join(lines)
            lines.clear()
            if raw == "[DONE]":
                return
            data = json.loads(raw)
            piece, calls, ended = _parse_candidate(data)
            if piece:
                if first_token is None:
                    first_token = (time.monotonic() - started) * 1_000.0
                text.append(piece)
                wire._emit(req.on_delta, piece)
            proposals.extend(calls)
            if ended:
                reason = ended
            if data.get("usageMetadata"):
                usage = _usage(data)

        async for chunk in response.content.iter_chunked(8_192):
            total += len(chunk)
            if total > min(req.max_response_bytes or MAX_BODY_BYTES, MAX_BODY_BYTES):
                raise RuntimeError("Gemma streamed response exceeded the size limit")
            pending += decoder.decode(chunk)
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                line = line.rstrip("\r")
                if not line:
                    consume_event()
                elif line.startswith("data:"):
                    lines.append(line[5:].lstrip())
        pending += decoder.decode(b"", final=True)
        if pending:
            line = pending.rstrip("\r")
            if line.startswith("data:"):
                lines.append(line[5:].lstrip())
        consume_event()
        return wire.ChatResult(text="".join(text), usage=usage, model=req.model,
                               streamed=True, first_token_ms=first_token,
                               done_reason=reason, tool_calls=proposals)
