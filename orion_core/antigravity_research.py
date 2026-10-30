"""Bounded public research with Google's Antigravity managed agent.

Only a short public topic crosses this boundary. No local files, account data,
desktop tools, MCP servers or ORION memory are mounted in the remote sandbox.
The Interactions API is separate from Gemini's ordinary text generation API.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    import aiohttp

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
AGENT_ID = "antigravity-preview-09-2026"
TEXT_MODEL = "gemini-3.5-flash-lite"
MAX_TOTAL_TOKENS = 20_000
MAX_WAIT_SECONDS = 120.0
POLL_SECONDS = 5.0
MAX_OUTPUT_CHARS = 16_000

_PRIVATE_TOPIC = re.compile(
    r"(?i)(?:[A-Z]:[\\/]|(?:^|\s)/(?:users|home|etc|mnt|workspace)/|"
    r"localhost|127\.0\.0\.1|(?:\d{1,3}\.){3}\d{1,3}|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\b(?:my|our)\s+(?:account|email|messages|files|documents|computer|pc|"
    r"phone|password|api\s*key|token|calendar|contacts|inbox)\b|"
    r"\b(?:password|secret|api\s*key|access\s*token|private\s*key)\s*[:=]|"
    r"\b(?:AIza[\w-]{20,}|sk-[\w-]{16,})\b)"
)
_INTERACTION_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


class AntigravityError(Exception):
    """A concise user-facing failure that never includes the API key."""


@dataclass(frozen=True)
class ResearchResult:
    text: str
    status: str
    total_tokens: int | None


def validate_public_topic(raw: Any) -> str:
    """Reject obvious private content before constructing an agent request."""
    topic = str(raw or "").strip()
    if not 8 <= len(topic) <= 240 or "\n" in topic or "\r" in topic:
        raise AntigravityError("Give me a public research topic of 8–240 characters.")
    if _PRIVATE_TOPIC.search(topic):
        raise AntigravityError(
            "Antigravity research accepts public topics only. Keep private "
            "files, accounts and credentials out of this request."
        )
    return topic


def _http_failure(status: int) -> AntigravityError:
    if status in {401, 403}:
        return AntigravityError("The Gemini key cannot access Antigravity. Check its project access.")
    if status == 429:
        return AntigravityError("Antigravity's current rate or usage limit has been reached. Try later.")
    if status == 400:
        return AntigravityError("Antigravity rejected this preview API request. Its schema may have changed.")
    if status >= 500:
        return AntigravityError("Antigravity is temporarily unavailable. Try later.")
    return AntigravityError(f"Antigravity returned HTTP {status}.")


class AntigravityResearch:
    """Run one remote research interaction at a time, then discard its sandbox."""

    _global_lock = asyncio.Lock()

    def __init__(self, *, max_wait_seconds: float = MAX_WAIT_SECONDS,
                 poll_seconds: float = POLL_SECONDS) -> None:
        self.max_wait_seconds = max_wait_seconds
        self.poll_seconds = poll_seconds
        # Dispatch creates a fresh service for each request. Share the gate so
        # simultaneous calls cannot multiply remote agent spend.
        self._lock = self._global_lock

    async def _request(self, session: aiohttp.ClientSession, method: str,
                       url: str, key: str, payload: dict[str, Any] | None = None
                       ) -> dict[str, Any]:
        import aiohttp

        headers = {"x-goog-api-key": key, "Api-Revision": "2026-05-20"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        try:
            # The API key is in a custom header. Do not follow a provider or
            # intermediary redirect and forward it to another host.
            async with session.request(method, url, headers=headers, json=payload,
                                       allow_redirects=False) as response:
                if response.status >= 300:
                    # Provider error bodies may echo user text; never return them.
                    raise _http_failure(response.status)
                value = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            raise AntigravityError("The Antigravity connection failed. Try later.") from exc
        if not isinstance(value, dict):
            raise AntigravityError("Antigravity returned an unexpected response.")
        return value

    async def _cancel(self, session: aiohttp.ClientSession, key: str,
                      interaction_id: str) -> None:
        try:
            await asyncio.wait_for(self._request(
                session, "POST", f"{INTERACTIONS_URL}/{quote(interaction_id)}:cancel", key),
                timeout=8,
            )
        except (AntigravityError, asyncio.TimeoutError):
            pass  # Best effort: report the original timeout or cancellation.

    async def run(self, topic: Any, *, api_key: str) -> ResearchResult:
        import aiohttp

        public_topic = validate_public_topic(topic)
        if not api_key.strip():
            raise AntigravityError("Set the Gemini API key before using Antigravity research.")
        if self._lock.locked():
            raise AntigravityError("An Antigravity research task is already running.")
        async with self._lock:
            timeout = aiohttp.ClientTimeout(total=15, connect=8, sock_read=12)
            async with aiohttp.ClientSession(timeout=timeout,
                                             cookie_jar=aiohttp.DummyCookieJar()) as session:
                interaction_id = ""
                terminal = False
                try:
                    # The model may search public sources. It cannot call ORION
                    # tools or read this machine. A fresh sandbox is used once.
                    payload = {
                        "agent": AGENT_ID,
                        "input": (
                            "Research this public topic using public web sources. "
                            "Give a concise evidence-led answer with source URLs "
                            "and state uncertainty. Use UK English. Topic: "
                            + public_topic
                        ),
                        "environment": "remote",
                        "tools": [{"type": "google_search"}, {"type": "url_context"}],
                        "agent_config": {
                            "type": "antigravity",
                            "model": TEXT_MODEL,
                            "max_total_tokens": MAX_TOTAL_TOKENS,
                        },
                        "background": True,
                        "store": True,
                    }
                    async with asyncio.timeout(self.max_wait_seconds):
                        result = await self._request(session, "POST", INTERACTIONS_URL,
                                                     api_key, payload)
                        interaction_id = str(result.get("id") or "")
                        if not _INTERACTION_ID.fullmatch(interaction_id):
                            raise AntigravityError("Antigravity did not return a valid task ID.")
                        while str(result.get("status") or "") in {
                            "in_progress", "pending", "queued", "running"
                        }:
                            await asyncio.sleep(self.poll_seconds)
                            result = await self._request(
                                session, "GET",
                                f"{INTERACTIONS_URL}/{quote(interaction_id)}", api_key,
                            )
                        status = str(result.get("status") or "")
                        terminal = status in {"completed", "failed", "cancelled", "incomplete"}
                        usage = result.get("usage") or {}
                        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
                        tokens = int(tokens) if isinstance(tokens, (int, float)) else None
                        answer = str(result.get("output_text") or "").strip()
                        if len(answer) > MAX_OUTPUT_CHARS:
                            answer = answer[:MAX_OUTPUT_CHARS] + (
                                "\n[Antigravity output shortened for this tool response.]"
                            )
                        if status == "completed" and answer:
                            return ResearchResult(answer, status, tokens)
                        if status == "incomplete":
                            note = "Antigravity reached its 20,000-token budget."
                            if answer:
                                note += " Partial findings:\n" + answer
                            return ResearchResult(note, status, tokens)
                        if status == "failed":
                            raise AntigravityError("Antigravity could not complete this research task.")
                        raise AntigravityError("Antigravity ended without a usable answer.")
                except TimeoutError as exc:
                    raise AntigravityError(
                        "Antigravity took longer than two minutes; the task was cancelled."
                    ) from exc
                finally:
                    if interaction_id and not terminal:
                        await self._cancel(session, api_key, interaction_id)
