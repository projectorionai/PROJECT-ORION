"""Display safety and real Qt acknowledgements for JARVIS capability integration."""
import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication, QLabel

from orion_core.bus import OrionBus
from orion_core.dispatch_desktop import DesktopDispatchMixin
from orion_core.gui.content_panel import ContentPanel
from orion_core.gui.core_window import OrionCoreWindow as CoreWindow
from orion_core.presentation import prepare, sanitise_markup, public_url, image_data


def test_active_markup_and_private_media_are_excluded():
    markup, images = sanitise_markup('<script>alert(1)</script><iframe src="http://127.0.0.1"></iframe>'
        '<p onclick="alert(2)">Useful information</p><a href="javascript:alert(3)">Link</a>'
        '<img src="file:///C:/private.txt"><table><tr><td colspan="2">42</td></tr></table>')
    assert "script" not in markup and "iframe" not in markup and "onclick" not in markup
    assert "javascript" not in markup and "file:" not in markup
    assert "Useful information" in markup and '<td colspan="2">42</td>' in markup
    assert not images
    for url in ("http://100.64.0.1/", "https://127.0.0.1/", "https://[::1]/", "http://localhost/",
                "https://foo.local/", "https://user:password@example.com/"):
        with pytest.raises(ValueError):
            public_url(url)


def test_images_are_decodable_bounded_raster_data():
    stream = BytesIO()
    Image.new("RGB", (16, 16), "red").save(stream, format="PNG")
    data = image_data(stream.getvalue())
    assert data.startswith("data:image/png;base64,")
    assert base64.b64decode(data.split(",")[1]) == stream.getvalue()
    with pytest.raises(ValueError):
        image_data(b"\x89PNG\r\n\x1a\nnot an actual picture")
    stream = BytesIO()
    Image.new("RGB", (1500, 1500)).save(stream, format="PNG")
    with pytest.raises(ValueError, match="megapixel"):
        image_data(stream.getvalue())


def test_public_media_dns_cannot_rebind_to_loopback(monkeypatch):
    from orion_core.presentation import fetch_image
    import socket
    async def run():
        async def resolve(*args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
        with pytest.raises(ValueError, match="Private"):
            await fetch_image("https://example.com/image.png")
    asyncio.run(run())


def test_display_commands_wait_for_real_widget_acknowledgements():
    app = QApplication.instance() or QApplication([])
    panel = ContentPanel()
    bus = OrionBus()
    applied = []
    host = SimpleNamespace(
        bus=bus, face=SimpleNamespace(set_presentation=applied.append),
        hud=SimpleNamespace(rail=QLabel("Metrics"), log=SimpleNamespace(clear=lambda: applied.append("clear"),
                                setVisible=lambda visible: applied.append(visible), show=lambda: None)),
        _ensure_content_panel=lambda: panel, _content_panel=panel,
        apply_accent=lambda accent: applied.append(accent) or True,
        set_face_form=lambda form: applied.append(form))
    def command(payload):
        try:
            CoreWindow._apply_presentation(host, payload["presentation"])
            receipt = {"request_id": payload["request_id"], "ok": True}
        except Exception as exc:
            receipt = {"request_id": payload["request_id"], "ok": False, "error": str(exc)}
        bus.gui_result.emit(receipt)
    bus.gui_command.connect(command)
    tool = DesktopDispatchMixin()
    tool.bus = bus
    async def run():
        display = await tool.presentation_tool("display", {"title": "Actual result", "html": "<h2>Telemetry</h2><p>FINAL WORDS.</p>"})
        assert display.ok and panel.count == 1
        assert panel.titles() == ["Actual result"]
        assert display.evidence[0]["presentation"]["html"].endswith("FINAL WORDS.</p>")
        for operation, args in (("ui_theme", {"accent": "#25bce8"}),
                                ("ui_reactor", {"shape": "torus"}),
                                ("ui_orbit", {"images": []}),
                                ("ui_chrome", {"telemetry": False}),
                                ("ui_screen", {"target": "panels"}),
                                ("ui_reset", {})):
            result = await tool.presentation_tool(operation, args)
            assert result.ok, result.text
        assert panel.count == 0
        assert any(isinstance(item, dict) and item["operation"] == "ui_reactor" for item in applied)
    try:
        asyncio.run(run())
        app.processEvents()
    finally:
        panel.close()
        host.hud.rail.close()
        bus.gui_command.disconnect(command)


@pytest.mark.parametrize("tool,args", [("ui_reactor", {"scale": float("nan")}),
                                       ("ui_chrome", {"telemetry": "false"}),
                                       ("ui_effect", {"kind": "arbitrary"})])
def test_invalid_visual_controls_are_rejected(tool, args):
    with pytest.raises(ValueError):
        asyncio.run(prepare(tool, args))
