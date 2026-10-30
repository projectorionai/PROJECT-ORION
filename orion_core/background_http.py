"""Bounded HTTP text retrieval without a browser or extension subprocess."""
from __future__ import annotations

import asyncio
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from .data import ToolResult

TIMEOUT_S = 20.0
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_REDIRECTS = 3
_REDIRECT = frozenset({301, 302, 303, 307, 308})


class FetchError(ValueError):
    pass


def _url(value: str) -> str:
    value = str(value).strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise FetchError("Only HTTP(S) page URLs are supported")
    if parsed.username is not None or parsed.password is not None:
        raise FetchError("Credentials in a page URL are not supported")
    return value


class _PageText(HTMLParser):
    _HIDDEN = frozenset({"script", "style", "noscript", "template", "head"})
    _BLOCKS = frozenset({"p", "div", "li", "section", "article", "h1", "h2", "h3", "h4", "br", "tr"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._HIDDEN:
            self.hidden += 1
        if not self.hidden and tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._HIDDEN:
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        return "\n".join(line for raw in "".join(self.parts).splitlines()
                         if (line := " ".join(raw.split())))


async def _retrieve(url: str) -> tuple[str, str, int]:
    from aiohttp import ClientSession, ClientTimeout
    current = _url(url)
    async with ClientSession(timeout=ClientTimeout(total=TIMEOUT_S), trust_env=False) as session:
        for redirects in range(MAX_REDIRECTS + 1):
            async with session.get(current, allow_redirects=False,
                                   headers={"User-Agent": "ORION-PageReader/1.0"}) as response:
                if response.status in _REDIRECT:
                    if redirects == MAX_REDIRECTS:
                        raise FetchError("The page exceeded the redirect limit")
                    location = response.headers.get("Location", "")
                    if not location:
                        raise FetchError("The page redirected without a destination")
                    current = _url(urljoin(current, location))
                    continue
                if not 200 <= response.status < 300:
                    raise FetchError(f"The page returned HTTP {response.status}")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower().strip()
                if content_type and not (content_type.startswith("text/") or content_type in {
                        "application/xhtml+xml", "application/json"}):
                    raise FetchError("The response is not a supported text page")
                declared = response.content_length
                if declared is not None and declared > MAX_RESPONSE_BYTES:
                    raise FetchError("The page exceeded the response byte limit")
                body = bytearray()
                async for chunk in response.content.iter_chunked(16 * 1024):
                    if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise FetchError("The page exceeded the response byte limit")
                    body.extend(chunk)
                try:
                    text = body.decode(response.charset or "utf-8", errors="replace")
                except LookupError:
                    text = body.decode("utf-8", errors="replace")
                if content_type in {"text/html", "application/xhtml+xml"} or (
                        not content_type and text.lstrip().lower().startswith(("<!doctype html", "<html"))):
                    parser = _PageText()
                    parser.feed(text)
                    parser.close()
                    text = parser.text()
                return current, text, len(body)
    raise FetchError("The page could not be retrieved")


async def fetch_text(url: str, *, start_index: int = 0, max_length: int = 8000) -> ToolResult:
    receipt = {"browser_opened": False, "subprocess_started": False}
    try:
        url = _url(url)
        receipt["url"] = url
        start = max(0, int(start_index))
        length = max(1000, min(20_000, int(max_length)))
        async with asyncio.timeout(TIMEOUT_S):
            final_url, text, byte_count = await _retrieve(url)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return ToolResult("Page retrieval timed out.", ok=False, receipt={**receipt, "status": "failed", "error": "timeout"})
    except Exception as exc:
        detail = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
        return ToolResult(f"Page retrieval failed: {detail}", ok=False,
                          receipt={**receipt, "status": "failed", "error": type(exc).__name__})
    selected = text[start:start + length]
    end = start + len(selected)
    return ToolResult(selected or "No page text remains at this start index.", receipt={
        **receipt, "status": "completed", "final_url": final_url, "start_index": start,
        "max_length": length, "returned_chars": len(selected), "total_chars": len(text),
        "has_more": end < len(text), "next_start_index": end if end < len(text) else None,
        "bytes_read": byte_count}, evidence=[{"url": final_url}])


__all__ = ["fetch_text"]
