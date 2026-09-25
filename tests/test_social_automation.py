"""
Tests for SocialAutomationService — real-account TikTok/Instagram automation.

Hermetic: these never launch a browser. They cover input validation, the
security guard, narration, and the Outlook-style draft/confirm approval gate
for Instagram replies (all of which run and fail/succeed before any
Playwright work is attempted).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.social_automation import SocialAutomationService


class _RecordingSignal:
    def __init__(self) -> None:
        self.emitted: list = []

    def emit(self, *args) -> None:
        self.emitted.append(args[0] if len(args) == 1 else args)

    def connect(self, *a, **k) -> None:
        pass


class _FakeBus:
    def __init__(self) -> None:
        self.log = _RecordingSignal()
        self.browser_step = _RecordingSignal()
        self.banner = _RecordingSignal()


def _service() -> tuple[SocialAutomationService, _FakeBus]:
    bus = _FakeBus()
    return SocialAutomationService(bus), bus


# ── availability ─────────────────────────────────────────────────────────────

def test_available_is_a_bool():
    service, _ = _service()
    assert isinstance(service.available, bool)


# ── narration ────────────────────────────────────────────────────────────────

def test_narrate_emits_log_and_structured_step():
    service, bus = _service()
    service._narrate("Opening TikTok Studio", "detail here")
    assert any("Opening TikTok Studio" in str(line) for line in bus.log.emitted)
    step = bus.browser_step.emitted[-1]
    assert step["step"] == "Opening TikTok Studio"
    assert step["detail"] == "detail here"
    assert step["origin"] == "social"


# ── TikTok upload validation (fails before any browser work) ──────────────────

def test_tiktok_upload_requires_video_path():
    service, _ = _service()
    result = asyncio.run(service.tiktok_upload(video_path=""))
    assert isinstance(result, ToolResult) and not result.ok


def test_tiktok_upload_rejects_missing_file():
    service, _ = _service()
    result = asyncio.run(service.tiktok_upload(video_path="C:/definitely/not/a/real/video.mp4"))
    assert not result.ok
    assert "no file found" in result.text.lower()


def test_tiktok_upload_security_guard_blocks_destructive_caption():
    service, _ = _service()
    result = asyncio.run(
        service.tiktok_upload(video_path="C:/nope.mp4", caption="rm -rf /important")
    )
    assert not result.ok


# ── Instagram draft/confirm approval gate ─────────────────────────────────────

def test_instagram_draft_reply_requires_contact_and_message():
    service, _ = _service()
    result = asyncio.run(service.instagram_draft_reply(contact="", message=""))
    assert not result.ok


def test_instagram_draft_reply_returns_a_speakable_ref():
    service, _ = _service()
    result = asyncio.run(service.instagram_draft_reply(contact="alice", message="hey!"))
    assert result.ok
    assert "reply-1" in result.text
    pending = service.pending_replies()
    assert len(pending) == 1
    assert pending[0]["contact"] == "alice"
    assert pending[0]["message"] == "hey!"


def test_instagram_reply_refuses_without_confirm():
    service, _ = _service()
    asyncio.run(service.instagram_draft_reply(contact="alice", message="hey!"))
    result = asyncio.run(service.instagram_reply(reply_ref="reply-1", confirm=False))
    assert not result.ok
    assert "approval" in result.text.lower()
    # The draft must still be pending — refusing to send must not consume it.
    assert len(service.pending_replies()) == 1


def test_instagram_reply_unknown_ref_is_reported_cleanly():
    service, _ = _service()
    result = asyncio.run(service.instagram_reply(reply_ref="reply-999", confirm=True))
    assert not result.ok
    assert "no pending reply" in result.text.lower()


# ── graceful degradation when Playwright is unavailable ───────────────────────

def test_unavailable_message_is_actionable(monkeypatch):
    service, _ = _service()
    monkeypatch.setattr(type(service), "available", property(lambda self: False))
    assert not service.available
    message = service._unavailable()
    assert not message.ok
    assert "playwright" in message.text.lower()
