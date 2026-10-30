"""Exercise the actual native reader with fake aiohttp response transports."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from orion_core import background_http as reader


class Response:
    def __init__(self, body=b"Synthetic text", *, status=200, headers=None, declared=None, chunks=None):
        self.status = status
        self.headers = {"Content-Type": "text/plain", **(headers or {})}
        self.content_length = declared
        self.charset = "utf-8"
        self.chunks = chunks if chunks is not None else [body]
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def iter_chunked(self, _size):
        for chunk in self.chunks:
            yield chunk


def transport(monkeypatch, responses):
    import aiohttp
    calls = []
    queue = list(responses)
    class Session:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            return None
        def get(self, url, **kwargs):
            assert kwargs["allow_redirects"] is False
            calls.append(url)
            if not queue:
                pytest.fail("Unexpected additional HTTP request")
            return queue.pop(0)
    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    return calls


async def test_native_reader_extracts_html_and_keeps_start_length_continuation(monkeypatch):
    body = ("<html><head><title>Hidden</title><style>Hidden CSS</style></head><body>"
            "<script>Hidden script</script><p>" + "X" * 1500 + "</p><p>End &amp; text</p></body></html>").encode()
    calls = transport(monkeypatch, [Response(body, headers={"Content-Type": "text/html; charset=utf-8"})])
    result = await reader.fetch_text("https://example.invalid/start", start_index=20, max_length=1000)
    assert result.ok and result.text == "X" * 1000
    assert result.receipt["has_more"] is True and result.receipt["next_start_index"] == 1020
    assert result.receipt["browser_opened"] is False and result.receipt["subprocess_started"] is False
    assert calls == ["https://example.invalid/start"]


@pytest.mark.parametrize("kind", ["declared", "streamed"])
async def test_overlong_http_response_is_a_failure_without_reading_past_cap(monkeypatch, kind):
    cap = 16
    monkeypatch.setattr(reader, "MAX_RESPONSE_BYTES", cap)
    response = Response(declared=cap + 1) if kind == "declared" else Response(chunks=[b"X" * cap, b"!"])
    transport(monkeypatch, [response])
    result = await reader.fetch_text("https://example.invalid/too-large")
    assert not result.ok and "byte limit" in result.text
    assert result.receipt["status"] == "failed" and result.receipt["subprocess_started"] is False


@pytest.mark.parametrize("status,headers,expected", [(503, {}, "HTTP 503"),
    (200, {"Content-Type": "application/octet-stream"}, "not a supported text page"),
    (302, {"Location": "file:///synthetic.txt"}, "Only HTTP(S)"),
    (302, {}, "without a destination")])
async def test_http_failures_and_unsupported_redirects_never_fall_back_to_a_browser(monkeypatch, status, headers, expected):
    calls = transport(monkeypatch, [Response(status=status, headers=headers)])
    result = await reader.fetch_text("https://example.invalid/fail")
    assert not result.ok and expected in result.text and len(calls) == 1
    assert result.receipt["browser_opened"] is False


async def test_total_redirect_limit_covers_the_entire_read(monkeypatch):
    calls = transport(monkeypatch, [Response(status=302, headers={"Location": "/next"}) for _ in range(4)])
    result = await reader.fetch_text("https://example.invalid/start")
    assert not result.ok and "redirect limit" in result.text and len(calls) == 4


async def test_total_http_deadline_returns_typed_failure_and_cancellation_propagates(monkeypatch):
    async def never_returns(_url):
        await asyncio.Event().wait()
    monkeypatch.setattr(reader, "_retrieve", never_returns)
    monkeypatch.setattr(reader, "TIMEOUT_S", 0.02)
    result = await reader.fetch_text("https://example.invalid/slow")
    assert not result.ok and result.receipt["error"] == "timeout"
    task = asyncio.create_task(reader.fetch_text("https://example.invalid/cancel"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_credential_url_is_rejected_without_retaining_password(monkeypatch):
    calls = transport(monkeypatch, [])
    result = await reader.fetch_text("https://name:synthetic-password@example.invalid")
    assert not result.ok and not calls and "synthetic-password" not in repr(result.receipt)


async def test_background_fetch_dispatch_uses_native_reader_instead_of_mcp(monkeypatch):
    from orion_core import dispatch_web, execution_policy
    from orion_core.data import ToolResult
    calls = []
    async def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return ToolResult("Synthetic page")
    monkeypatch.setattr(reader, "fetch_text", fetch)
    host = dispatch_web.WebDispatchMixin()
    host.mcp_host = SimpleNamespace(call=lambda *_a: pytest.fail("Background retrieval started an MCP subprocess"))
    with execution_policy.background_scope("ordinary-tool"):
        result = await host.fetch_url({"url": "https://example.invalid", "start_index": 4, "max_length": 10000})
    assert result.ok and calls == [("https://example.invalid", {"start_index": 4, "max_length": 10000})]
