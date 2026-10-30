"""An explicit phone screen question uses the fresh JPEG, not text-only chat."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from orion_core.remote import RemoteGateway


class _Router:
    def __init__(self):
        self.image = None

    async def generate_vision(self, image, prompt, **kwargs):
        self.image = image
        return SimpleNamespace(name="vision"), "The screen shows Settings."


def test_phone_screen_question_routes_fresh_frame_to_vision_model():
    router = _Router()
    frame = SimpleNamespace(jpeg=b"\xff\xd8phone\xff\xd9")
    viewport = SimpleNamespace(latest_frame=lambda device: frame if device == "owner" else None)
    gateway = SimpleNamespace(router=router, mobile_viewport=viewport)

    reply, source = asyncio.run(RemoteGateway._answer(
        gateway, "What is on my phone screen?", device_id="owner"))

    assert (reply, source) == ("The screen shows Settings.", "phone-vision")
    assert router.image == frame.jpeg
