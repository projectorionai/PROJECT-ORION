#!/usr/bin/env python3
"""
Integration test for JARVIS subsystems (Mark X.7+).

Verifies that all new subsystems (web automation, peripherals, messaging,
gaming, entertainment) are properly wired and can be dispatched.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

# Test imports
try:
    from orion_core.bus import OrionBus
    from orion_core.web_automation import WebAutomationService
    from orion_core.peripherals import PeripheralController
    from orion_core.messaging import MessagingGateway
    from orion_core.gaming import GamingClientService
    from orion_core.entertainment import EntertainmentService
    from orion_core.dispatcher import OrionDispatcher, TOOL_DECLARATIONS
    print("✓ All subsystem modules import successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Verify tool declarations
tool_names = {t["name"] for t in TOOL_DECLARATIONS}
expected_tools = {"web_automation", "peripherals", "messaging", "gaming", "entertainment"}
missing = expected_tools - tool_names
if missing:
    print(f"✗ Missing tools in TOOL_DECLARATIONS: {missing}")
    exit(1)
print(f"✓ All {len(expected_tools)} JARVIS tools registered in TOOL_DECLARATIONS")

# Verify tool declarations have required fields
for tool in TOOL_DECLARATIONS:
    if tool["name"] in expected_tools:
        if "description" not in tool or "parameters" not in tool:
            print(f"✗ Tool '{tool['name']}' missing description or parameters")
            exit(1)
print("✓ All JARVIS tools have complete declarations")

# Test service instantiation
bus = OrionBus()
services = {
    "web_automation": WebAutomationService(bus),
    "peripherals": PeripheralController(bus),
    "messaging": MessagingGateway(bus),
    "gaming": GamingClientService(bus),
    "entertainment": EntertainmentService(bus),
}
print(f"✓ All {len(services)} subsystem services instantiate successfully")

# Verify dispatcher accepts the services
dispatcher = OrionDispatcher(
    bus=bus,
    memory=None,
    grabber=None,
    file_intel=None,
    desktop=None,
    vision=None,
    outlook=None,
    notion=None,
    agent_manager=None,
    briefing=None,
)
dispatcher.web_automation = services["web_automation"]
dispatcher.peripherals = services["peripherals"]
dispatcher.messaging = services["messaging"]
dispatcher.gaming = services["gaming"]
dispatcher.entertainment = services["entertainment"]

# Verify handlers exist in dispatcher
handlers_to_check = [
    "web_automation_tool",
    "peripherals_tool",
    "messaging_tool",
    "gaming_tool",
    "entertainment_tool",
]
for handler_name in handlers_to_check:
    if not hasattr(dispatcher, handler_name):
        print(f"✗ Dispatcher missing handler: {handler_name}")
        exit(1)
    if not callable(getattr(dispatcher, handler_name)):
        print(f"✗ Dispatcher handler not callable: {handler_name}")
        exit(1)
print(f"✓ All {len(handlers_to_check)} dispatcher handlers are callable")

# Test graceful degradation when services are None
async def test_degradation():
    dispatcher.web_automation = None
    result = await dispatcher.web_automation_tool({})
    if result.ok:
        print("✗ Dispatcher should return error when service is None")
        exit(1)
    dispatcher.web_automation = services["web_automation"]
    print("✓ Dispatcher gracefully degrades when services unavailable")

asyncio.run(test_degradation())

print("\n" + "="*70)
print("✓✓✓ All JARVIS subsystem integration tests passed ✓✓✓")
print("="*70)
print(f"\nJARVIS subsystems ready:")
print(f"  • web_automation  — Browser automation (Chrome, Edge, Firefox)")
print(f"  • peripherals     — Hardware control (volume, power, network)")
print(f"  • messaging       — WhatsApp & Telegram alerts")
print(f"  • gaming          — Steam & Epic Games launcher")
print(f"  • entertainment   — YouTube media discovery & trends")
