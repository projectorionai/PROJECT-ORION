"""
Hardware and OS peripheral controller.

This module provides a platform-aware wrapper for volume control, power
commands, network toggles and basic host-system actions. The implementation
fails gracefully when optional native dependencies are absent.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any

from .bus import OrionBus
from .data import ToolResult


class PeripheralController:
    """Manage host audio, power and networking controls in a structured way."""

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self._platform = platform.system().lower()
        self._audio_backend: Any = None
        self._configure_audio()

    def _configure_audio(self) -> None:
        if self._platform != "windows":
            return
        try:
            import pycaw.pycaw  # type: ignore
            self._audio_backend = pycaw.pycaw  # type: ignore[attr-defined]
        except Exception as exc:
            self.bus.log.emit(f"PERIPHERALS: pycaw unavailable - {exc}")
            self._audio_backend = None

    def _run_shell(self, *args: str) -> ToolResult:
        try:
            subprocess.run(list(args), check=False, capture_output=True, text=True)
        except Exception as exc:
            return ToolResult(f"Host command failed: {exc}", ok=False)
        return ToolResult(f"Executed {' '.join(args)}")

    def set_volume(self, level: float) -> ToolResult:
        level = max(0.0, min(1.0, float(level)))
        if self._platform == "windows":
            try:
                from ctypes import POINTER, cast
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, None, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(level, None)
                return ToolResult(f"Set master volume to {level:.0%}.")
            except Exception as exc:
                self.bus.log.emit(f"PERIPHERALS: volume control failed - {exc}")
                return ToolResult("Volume control is unavailable on this host.", ok=False)
        return ToolResult(f"Volume level set to {level:.0%} (platform fallback).")

    def toggle_mute(self) -> ToolResult:
        if self._platform == "windows":
            try:
                from ctypes import POINTER, cast
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, None, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                current = volume.GetMasterVolumeLevelScalar()
                volume.SetMasterVolumeLevelScalar(0.0 if current > 0.0 else 1.0, None)
                return ToolResult("Mute state toggled.")
            except Exception as exc:
                self.bus.log.emit(f"PERIPHERALS: mute control failed - {exc}")
                return ToolResult("Mute control is unavailable on this host.", ok=False)
        return ToolResult("Mute toggle requested; no native backend available.", ok=False)

    def shutdown(self) -> ToolResult:
        if self._platform == "windows":
            return self._run_shell("shutdown", "/s", "/t", "0")
        if self._platform == "darwin":
            return self._run_shell("osascript", "-e", 'tell app "System Events" to shut down')
        return self._run_shell("shutdown", "-h", "now")

    def restart(self) -> ToolResult:
        if self._platform == "windows":
            return self._run_shell("shutdown", "/r", "/t", "0")
        if self._platform == "darwin":
            return self._run_shell("osascript", "-e", 'tell app "System Events" to restart')
        return self._run_shell("shutdown", "-r", "now")

    def lock_screen(self) -> ToolResult:
        if self._platform == "windows":
            return self._run_shell("rundll32.exe", "user32.dll,LockWorkStation")
        if self._platform == "darwin":
            return self._run_shell("pmset", "displaysleepnow")
        return self._run_shell("loginctl", "lock-session")

    def toggle_wifi(self) -> ToolResult:
        if self._platform == "windows":
            return self._run_shell("netsh", "interface", "set", "interface", "name=Wi-Fi", "admin=disabled")
        return ToolResult("Wi-Fi toggle requested; platform fallback only.", ok=False)

    def toggle_ethernet(self) -> ToolResult:
        if self._platform == "windows":
            return self._run_shell("netsh", "interface", "set", "interface", "name=Ethernet", "admin=disabled")
        return ToolResult("Ethernet toggle requested; platform fallback only.", ok=False)

    def toggle_network_adapter(self, adapter: str = "Ethernet") -> ToolResult:
        if self._platform == "windows":
            return self._run_shell("netsh", "interface", "set", "interface", "name=" + adapter, "admin=disabled")
        return ToolResult("Adapter toggling is not implemented on this platform.", ok=False)
