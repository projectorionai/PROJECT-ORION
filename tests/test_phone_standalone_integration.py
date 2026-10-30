"""Native phone provisioning and desktop-mode routing stay explicit."""

import json
from types import SimpleNamespace

from aiohttp import web

from orion_core.remote import RemoteGateway
from orion_core.phone_contracts import translate, valid_command


def test_popular_videos_uses_a_bounded_phone_command():
    assert translate("show_trending_videos", {"action": "show_trending_videos"}) == (
        "device.show_trending_videos", {})
    assert valid_command("device.show_trending_videos", {})
    assert not valid_command("device.show_trending_videos", {"url": "https://example.org"})


async def test_phone_command_skips_pc_task_fallback():
    gateway = object.__new__(RemoteGateway)
    gateway.mobile_viewport = None
    gateway.router = None
    gateway._grounding = lambda _message: ""
    gateway._phone_capabilities = lambda _device: "Phone controls available"

    class DesktopBrain:
        async def try_task(self, _message):
            raise AssertionError("A phone command reached the desktop task path")

    class Gate:
        def __init__(self):
            self.steps = None

        async def dispatch_plan(self, steps):
            self.steps = steps
            return SimpleNamespace(text="Phone app opened", evidence=[])

    gateway.local_brain = DesktopBrain()
    gateway.tool_gate = Gate()
    answer, source = await gateway._answer("Open YouTube", device_id="paired-phone")
    assert (answer, source) == ("Phone app opened", "remote-task")
    assert gateway.tool_gate.steps == [{"tool": "phone_device", "args": {
        "action": "open_app", "name": "YouTube"}}]


async def test_ai_profile_requires_the_exact_approved_pairing(tmp_path):
    gateway = object.__new__(RemoteGateway)
    gateway.auth = SimpleNamespace(config_dir=tmp_path,
                                   verify_access=lambda token: "paired-phone" if token == "valid" else None)
    gateway._auth_rate_ok = lambda _key: True
    gateway._web = web
    provider = SimpleNamespace(enabled=True)
    gateway.router = SimpleNamespace(settings=SimpleNamespace(providers={"gemini": provider}),
                                     active_key=lambda _provider: "sealed-in-test-only")
    request = SimpleNamespace(headers={"Authorization": "Bearer valid"},
                              secure=True, remote="192.0.2.5")

    denied = await gateway._handle_mobile_ai_profile(request)
    assert denied.status == 403
    assert b"sealed-in-test-only" not in denied.body

    (tmp_path / "mobile_ai_permissions.json").write_text(
        json.dumps({"version": 1, "devices": ["paired-phone"]}), encoding="utf-8")
    request.secure = False
    insecure = await gateway._handle_mobile_ai_profile(request)
    assert insecure.status == 403
    assert b"sealed-in-test-only" not in insecure.body
    request.secure = True
    approved = await gateway._handle_mobile_ai_profile(request)
    payload = json.loads(approved.body)
    assert approved.status == 200
    assert approved.headers["Cache-Control"].startswith("no-store")
    assert payload["api_key"] == "sealed-in-test-only"
    assert payload["requires_encrypted_storage"] is True
    assert payload["model"] == "gemini-3.5-flash-lite"
    assert payload["voice_model"] == "gemini-3.8-flash-tts"

    request.secure = False
    request.remote = "127.0.0.1"  # HTTPS proxy or tunnel terminates locally.
    proxied = await gateway._handle_mobile_ai_profile(request)
    assert proxied.status == 200

    gateway.auth.verify_access = lambda _token: None
    unauthorised = await gateway._handle_mobile_ai_profile(request)
    assert unauthorised.status == 401
    assert b"sealed-in-test-only" not in unauthorised.body
