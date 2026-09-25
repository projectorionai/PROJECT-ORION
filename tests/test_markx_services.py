import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QCoreApplication

from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.vision import VisionAgent, VolatileScreenGrabber, LocalFileIntelligence
from orion_core.peripherals import PeripheralController
from orion_core.messaging import MessagingGateway
from orion_core.gaming import GamingClientService
from orion_core.entertainment import EntertainmentService


def test_service_types_and_graceful_fallbacks():
    app = QCoreApplication.instance() or QCoreApplication([])
    bus = OrionBus()
    grabber = VolatileScreenGrabber(bus)
    file_intel = LocalFileIntelligence(bus)
    vision = VisionAgent(bus, grabber, file_intel)

    # The web layer is now the real visible-browser co-pilot (BrowserCopilot),
    # exercised in depth by tests/test_browser_copilot.py; the no-op
    # WebAutomationService stub it replaced was retired in Mark X.12 §2.1.

    peripheral = PeripheralController(bus)
    assert isinstance(peripheral.set_volume(0.2), ToolResult)
    assert isinstance(peripheral.toggle_mute(), ToolResult)
    assert isinstance(peripheral.shutdown(), ToolResult)

    messaging = MessagingGateway(bus)
    assert isinstance(messaging.send_text("whatsapp", "test", "demo"), ToolResult)
    assert isinstance(messaging.send_text("telegram", "test", "demo"), ToolResult)

    gaming = GamingClientService(bus)
    assert isinstance(gaming.index_installs(), ToolResult)

    entertainment = EntertainmentService(bus)
    assert isinstance(entertainment.summarise_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), ToolResult)

    assert hasattr(vision, "capture_live_frame")
    app.quit()
