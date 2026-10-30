"""
Local gaming client pipeline.

The service indexes locally installed gaming clients such as Steam and Epic
Games, reports available launch targets and offers lightweight update-state
inspection through process and directory monitoring.
"""

from __future__ import annotations

import platform
import re
import subprocess
from urllib.parse import quote

from .bus import OrionBus
from .data import ToolResult


class GamingClientService:
    """Inspect and launch locally installed PC gaming clients."""

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._platform = platform.system().lower()

    def index_installs(self) -> ToolResult:
        from .jv_plugins import legacy_game_updater as clients
        steam = clients._find_steam_path()
        games = clients._games(steam) if steam else []
        epic = clients._epic_games()
        rows = [f"Steam: {g['name']} (AppID {g['id']})" for g in games]
        rows.extend(f"Epic: {g['name']} (AppID epic:{g['id']})" for g in epic)
        if not rows:
            return ToolResult("No installed Steam or Epic games were found in the client manifests.", ok=False)
        return ToolResult("\n".join(rows), evidence=[{"platform": "steam", "id": g["id"], "name": g["name"]} for g in games]
                          + [{"platform": "epic", "id": g["id"], "name": g["name"]} for g in epic])

    def launch_app(self, app_id: str) -> ToolResult:
        from .execution_policy import require_foreground
        blocked = require_foreground("game_application_launch")
        if blocked is not None:
            return blocked
        app_id = app_id.strip()
        if not app_id:
            return ToolResult("An AppID is required.", ok=False)
        from .jv_plugins import legacy_game_updater as clients
        if app_id.startswith("epic:"):
            identifier = app_id[5:]
            game = next((g for g in clients._epic_games() if g["id"] == identifier), None)
            exe = clients._find_epic_exe()
            uri = "com.epicgames.launcher://apps/" + quote(identifier, safe="") + "?action=launch&silent=true"
        else:
            if not re.fullmatch(r"[0-9]{1,12}", app_id):
                return ToolResult("Choose a Steam AppID or an epic:AppID from the installed-game index.", ok=False)
            path = clients._find_steam_path()
            game = next((g for g in clients._games(path) if g["id"] == app_id), None) if path else None
            exe = clients._steam_exe(path) if path else None
            uri = "steam://rungameid/" + app_id
        if game is None:
            return ToolResult("That game was not found in the installed-game manifests. Use the updater to install it first.", ok=False)
        try:
            blocked = require_foreground("game_application_launch")
            if blocked is not None:
                return blocked
            if self._platform == "windows":
                if exe is None or not exe.is_file():
                    return ToolResult("The gaming client executable could not be found.", ok=False)
                subprocess.Popen([str(exe), uri], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                subprocess.Popen(["open" if self._platform == "darwin" else "xdg-open", uri])
        except OSError as exc:
            return ToolResult(f"The launcher could not be started ({type(exc).__name__}).", ok=False)
        self.bus.log.emit(f"GAMING: sent launch request for {game['name']}.")
        return ToolResult(f"Sent {game['name']}'s launch request to its gaming client. The client may still require sign-in.")

    def update_status(self) -> ToolResult:
        from .jv_plugins import legacy_game_updater as clients
        steam = clients._find_steam_path()
        if steam is None:
            return ToolResult("Steam was not detected. Epic does not publish a reliable download-progress counter.", ok=False)
        return ToolResult(clients._download_status(steam) + " Epic download progress is not available from its manifests.")
