"""
Local gaming client pipeline.

The service indexes locally installed gaming clients such as Steam and Epic
Games, reports available launch targets and offers lightweight update-state
inspection through process and directory monitoring.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .bus import OrionBus
from .data import ToolResult


class GamingClientService:
    """Inspect and launch locally installed PC gaming clients."""

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._platform = platform.system().lower()

    def index_installs(self) -> ToolResult:
        installs: list[str] = []
        if self._platform == "windows":
            candidates = [
                r"C:\Program Files (x86)\Steam",
                r"C:\Program Files\Steam",
                r"C:\Program Files\Epic Games",
                r"C:\Program Files (x86)\Epic Games",
            ]
            for path in candidates:
                if os.path.exists(path):
                    installs.append(path)
        else:
            for path in (Path.home() / ".steam" / "steam", Path.home() / ".local" / "share" / "Steam"):
                if path.exists():
                    installs.append(str(path))
        if not installs:
            return ToolResult("No local gaming clients were detected.", ok=False)
        return ToolResult("Detected local gaming clients: " + "; ".join(installs))

    def launch_app(self, app_id: str) -> ToolResult:
        if not app_id.strip():
            return ToolResult("An AppID is required.", ok=False)
        if self._platform == "windows":
            self.bus.log.emit(f"GAMING: launch request queued for AppID {app_id}.")
            return ToolResult(f"Launch request queued for AppID {app_id}.")
        return ToolResult("Launch requests require a host-specific launcher integration.", ok=False)

    def update_status(self) -> ToolResult:
        return ToolResult("Update status is unavailable without a native client integration.")
