"""
Hardware and OS peripheral controller.

This module provides a platform-aware wrapper for volume control, power
commands, network toggles and basic host-system actions. The implementation
fails gracefully when optional native dependencies are absent.
"""

from __future__ import annotations

import os
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
        from .execution_policy import require_foreground
        blocked = require_foreground("host_system_command")
        if blocked is not None:
            return blocked
        try:
            done = subprocess.run(
                list(args), check=False, capture_output=True, text=True, errors="replace", timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as exc:
            return ToolResult(f"Host command failed: {exc}", ok=False)
        if done.returncode != 0:
            # "Executed shutdown /s" used to be reported even when Windows
            # refused it (access denied, a shutdown already pending).
            reason = (done.stderr or done.stdout or "").strip().splitlines()
            detail = reason[0][:160] if reason else f"exit code {done.returncode}"
            return ToolResult(f"Host command failed: {' '.join(args)} - {detail}", ok=False)
        return ToolResult(f"Executed {' '.join(args)}")

    # ── volume ────────────────────────────────────────────────────────────────

    def _endpoint(self) -> Any:
        """The default speakers' IAudioEndpointVolume, across pycaw releases.

        pycaw 20251023 returns an AudioDevice wrapper exposing .EndpointVolume;
        older releases returned the raw IMMDevice, which needed .Activate().
        The code only knew the old form, so on the real runtime every volume,
        mute and pinch-gesture call raised AttributeError and reported
        "Volume control is unavailable on this host"."""
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        device = AudioUtilities.GetSpeakers()
        endpoint = getattr(device, "EndpointVolume", None)
        if endpoint is not None:
            return endpoint
        from ctypes import POINTER, cast
        interface = device.Activate(IAudioEndpointVolume._iid_, 1 | 2 | 4 | 16, None)
        return cast(interface, POINTER(IAudioEndpointVolume))

    def volume_state(self) -> tuple[float, bool] | None:
        """(level 0.0-1.0, muted) of the default speakers, or None."""
        if self._platform != "windows":
            return None
        try:
            endpoint = self._endpoint()
            return float(endpoint.GetMasterVolumeLevelScalar()), bool(endpoint.GetMute())
        except Exception as exc:
            self.bus.log.emit(f"PERIPHERALS: volume read failed - {exc}")
            return None

    def describe_volume(self) -> ToolResult:
        state = self.volume_state()
        if state is None:
            return ToolResult("I can't read the volume on this host.", ok=False)
        level, muted = state
        return ToolResult(f"Volume is at {level:.0%}" + (", muted." if muted else "."))

    def set_volume(self, level: float, *, unmute: bool = True) -> ToolResult:
        """Set the master volume. Asking for a level means wanting to hear it,
        so a muted output is unmuted too (unmute=False leaves the mute alone)."""
        level = max(0.0, min(1.0, float(level)))
        if self._platform == "windows":
            try:
                endpoint = self._endpoint()
                endpoint.SetMasterVolumeLevelScalar(level, None)
                if unmute and level > 0 and endpoint.GetMute():
                    endpoint.SetMute(0, None)
                return ToolResult(f"Volume set to {level:.0%}.")
            except Exception as exc:
                self.bus.log.emit(f"PERIPHERALS: volume control failed - {exc}")
                return ToolResult("Volume control is unavailable on this host.", ok=False)
        return ToolResult(f"Volume level set to {level:.0%} (platform fallback).")

    def change_volume(self, delta: float) -> ToolResult:
        """Nudge the volume by delta (+0.1 = ten points louder)."""
        state = self.volume_state()
        if state is None:
            return ToolResult("Volume control is unavailable on this host.", ok=False)
        level, muted = state
        target = max(0.0, min(1.0, level + float(delta)))
        if delta > 0 and level >= 0.995 and not muted:
            return ToolResult("Volume is already at 100%.")
        if delta < 0 and level <= 0.005:
            return ToolResult("Volume is already at 0%.")
        result = self.set_volume(target)
        if result.ok:
            return ToolResult(f"Volume {'up' if delta > 0 else 'down'} to {target:.0%}.")
        return result

    def set_mute(self, muted: bool | None = None) -> ToolResult:
        """Mute (True), unmute (False) or toggle (None).

        This used to "mute" by setting the volume to 0 and "unmute" by setting
        it to 100%, so unmuting after a quiet evening blasted full volume and
        the user's level was lost. The endpoint's own mute flag keeps it."""
        if self._platform != "windows":
            return ToolResult("Mute toggle requested; no native backend available.", ok=False)
        try:
            endpoint = self._endpoint()
            current = bool(endpoint.GetMute())
            target = (not current) if muted is None else bool(muted)
            if target == current:
                return ToolResult("Already muted." if current else "Sound is already on.")
            endpoint.SetMute(int(target), None)
            if target:
                return ToolResult("Muted.")
            level = float(endpoint.GetMasterVolumeLevelScalar())
            return ToolResult(f"Unmuted - volume is at {level:.0%}.")
        except Exception as exc:
            self.bus.log.emit(f"PERIPHERALS: mute control failed - {exc}")
            return ToolResult("Mute control is unavailable on this host.", ok=False)

    def toggle_mute(self) -> ToolResult:
        return self.set_mute(None)

    # ── brightness ────────────────────────────────────────────────────────────

    def _wmi_brightness(self) -> tuple[Any, list] | None:
        try:
            import win32com.client
            wmi = win32com.client.GetObject(r"winmgmts:\\.\root\wmi")
            methods = list(wmi.ExecQuery("SELECT * FROM WmiMonitorBrightnessMethods"))
        except Exception:
            return None
        return (wmi, methods) if methods else None

    def brightness_level(self) -> float | None:
        """Current brightness 0.0-1.0 of the first controllable display."""
        if self._platform != "windows":
            return None
        wmi = self._wmi_brightness()
        if wmi is not None:
            try:
                current = list(wmi[0].ExecQuery("SELECT CurrentBrightness FROM WmiMonitorBrightness"))
                if current:
                    return int(current[0].CurrentBrightness) / 100.0
            except Exception:
                pass
        levels = _ddc_brightness()
        return levels[0] if levels else None

    def set_brightness(self, level: float) -> ToolResult:
        """Set display brightness, 0.0-1.0, on Windows.

        Laptop panels answer WMI (WmiMonitorBrightnessMethods). External
        monitors over HDMI/DisplayPort do not - this used to stop there with
        "this monitor likely needs DDC/CI", which was every desktop. They are
        now driven over DDC/CI (VCP code 0x10) through dxva2, which ships with
        Windows. A monitor with DDC/CI switched off in its menu still says so."""
        level = max(0.0, min(1.0, float(level)))
        if self._platform != "windows":
            return ToolResult("Brightness control is only available on Windows.", ok=False)
        target = int(round(level * 100))
        wmi = self._wmi_brightness()
        if wmi is not None:
            try:
                for method in wmi[1]:
                    method.WmiSetBrightness(1, target)
                return ToolResult(f"Brightness set to {target}%.")
            except Exception as exc:
                self.bus.log.emit(f"PERIPHERALS: WMI brightness failed - {exc}")
        changed = _ddc_set_brightness(level)
        if changed:
            return ToolResult(f"Brightness set to {target}%"
                              + (f" on {changed} monitors." if changed > 1 else "."))
        return ToolResult(
            "I couldn't change the brightness - no display here accepts software "
            "control (if it's an external monitor, DDC/CI may be switched off in "
            "its on-screen menu).", ok=False)

    def change_brightness(self, delta: float) -> ToolResult:
        current = self.brightness_level()
        if current is None:
            return ToolResult(
                "I couldn't read the brightness - no display here accepts software control.",
                ok=False)
        target = max(0.0, min(1.0, current + float(delta)))
        result = self.set_brightness(target)
        if result.ok:
            return ToolResult(f"Brightness {'up' if delta > 0 else 'down'} to {target:.0%}.")
        return result

    def describe_brightness(self) -> ToolResult:
        level = self.brightness_level()
        if level is None:
            return ToolResult("I can't read the brightness of this display.", ok=False)
        return ToolResult(f"Brightness is at {level:.0%}.")

    def shutdown(self, confirm: bool = False) -> ToolResult:
        """Power off the HOST computer.  Safe by default: this is a destructive,
        irreversible action, so it refuses to run unless an explicit
        ``confirm=True`` is passed.  The confirmation is issued by the
        deterministic SystemActionGuard only after the user approves the
        on-screen prompt — never by model text alone."""
        if not confirm:
            return ToolResult(
                "Host shutdown requires explicit confirmation and was not executed.",
                ok=False,
            )
        if self._platform == "windows":
            return self._run_shell("shutdown", "/s", "/t", "0")
        if self._platform == "darwin":
            return self._run_shell("osascript", "-e", 'tell app "System Events" to shut down')
        return self._run_shell("shutdown", "-h", "now")

    def restart(self, confirm: bool = False) -> ToolResult:
        """Restart the HOST computer.  Safe by default — requires ``confirm=True``."""
        if not confirm:
            return ToolResult(
                "Host restart requires explicit confirmation and was not executed.",
                ok=False,
            )
        if self._platform == "windows":
            return self._run_shell("shutdown", "/r", "/t", "0")
        if self._platform == "darwin":
            return self._run_shell("osascript", "-e", 'tell app "System Events" to restart')
        return self._run_shell("shutdown", "-r", "now")

    def logout(self, confirm: bool = False) -> ToolResult:
        """Log the current user out.  Requires explicit confirmation."""
        if not confirm:
            return ToolResult("Log out requires explicit confirmation and was not executed.", ok=False)
        if self._platform == "windows":
            return self._run_shell("shutdown", "/l")
        if self._platform == "darwin":
            return self._run_shell("osascript", "-e",
                                   'tell application "System Events" to log out')
        return self._run_shell("loginctl", "terminate-user", "")

    def sleep(self, confirm: bool = False) -> ToolResult:
        """Put the host to sleep (suspend).  Requires explicit confirmation."""
        if not confirm:
            return ToolResult("Sleep requires explicit confirmation and was not executed.", ok=False)
        if self._platform == "windows":
            return self._run_shell("rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0")
        if self._platform == "darwin":
            return self._run_shell("pmset", "sleepnow")
        return self._run_shell("systemctl", "suspend")

    def hibernate(self, confirm: bool = False) -> ToolResult:
        """Hibernate the host.  Requires explicit confirmation."""
        if not confirm:
            return ToolResult("Hibernate requires explicit confirmation and was not executed.", ok=False)
        if self._platform == "windows":
            return self._run_shell("shutdown", "/h")
        if self._platform == "darwin":
            return ToolResult("Hibernate is not available on macOS.", ok=False)
        return self._run_shell("systemctl", "hibernate")

    def lock_screen(self) -> ToolResult:
        if self._platform == "windows":
            return self._run_shell("rundll32.exe", "user32.dll,LockWorkStation")
        if self._platform == "darwin":
            return self._run_shell("pmset", "displaysleepnow")
        return self._run_shell("loginctl", "lock-session")

    # ── network adapters ──────────────────────────────────────────────────────
    # These were "toggles" that only ever disabled, against adapters named
    # "Wi-Fi" and "Ethernet" - so "turn the wifi back on" could not be done and
    # "Ethernet 2"/"Ethernet 3" were never found. Switching an adapter needs
    # administrator rights; when Windows refuses, the settings page is opened.

    def network_adapters(self) -> list[dict[str, Any]]:
        """[{name, enabled, connected}] from `netsh interface show interface`."""
        if self._platform != "windows":
            return []
        try:
            done = subprocess.run(
                ["netsh", "interface", "show", "interface"], capture_output=True,
                text=True, errors="replace", timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            return []
        adapters = []
        for line in done.stdout.splitlines():
            parts = line.split(None, 3)
            if len(parts) != 4 or parts[0].startswith("-") or parts[0].lower() == "admin":
                continue
            adapters.append({"name": parts[3].strip(),
                             "enabled": parts[0].lower() == "enabled",
                             "connected": parts[1].lower() == "connected"})
        return adapters

    @staticmethod
    def _adapter_kind(name: str) -> str:
        lowered = name.lower()
        if any(word in lowered for word in ("wi-fi", "wifi", "wlan", "wireless")):
            return "wifi"
        if lowered.startswith("ethernet") or "lan" in lowered.split():
            return "ethernet"
        return "other"

    def set_network(self, kind: str, enabled: bool | None = None) -> ToolResult:
        """Switch the Wi-Fi or Ethernet adapters on (True), off (False) or to
        the opposite of their current state (None)."""
        label = "Wi-Fi" if kind == "wifi" else "Ethernet"
        if self._platform != "windows":
            return ToolResult(f"{label} switching is only available on Windows.", ok=False)
        if kind == "wifi":
            # The radio first: no administrator rights needed (see set_radio).
            wired = any(a["connected"] for a in self.network_adapters()
                        if self._adapter_kind(a["name"]) == "ethernet")
            done = self.set_radio("WiFi", enabled)
            if done is not None:
                if done.ok and done.text.endswith("turned off.") and not wired:
                    done = ToolResult(done.text + " That takes this PC offline, so I won't "
                                      "hear you until it's back.")
                return done
        adapters = [a for a in self.network_adapters() if self._adapter_kind(a["name"]) == kind]
        if not adapters:
            return ToolResult(f"This PC has no {label} adapter.", ok=False)
        if enabled is None:
            enabled = not any(a["enabled"] for a in adapters)
        targets = [a for a in adapters if a["enabled"] != enabled]
        if not targets:
            return ToolResult(f"{label} is already {'on' if enabled else 'off'}.")
        others_online = any(a["connected"] for a in self.network_adapters()
                            if a not in targets and self._adapter_kind(a["name"]) != "other")
        failures = []
        for adapter in targets:
            result = self._run_shell("netsh", "interface", "set", "interface",
                                     "name=" + adapter["name"],
                                     "admin=" + ("enabled" if enabled else "disabled"))
            if not result.ok:
                failures.append(result.text)
        if failures:
            if any("elevation" in text.lower() or "administrator" in text.lower() for text in failures):
                page = "ms-settings:network-wifi" if kind == "wifi" else "ms-settings:network-ethernet"
                from .execution_policy import require_foreground
                blocked = require_foreground("network_settings_window")
                if blocked is not None:
                    return blocked
                try:
                    os.startfile(page)  # type: ignore[attr-defined]
                except Exception:
                    pass
                return ToolResult(
                    f"Windows needs administrator rights to switch {label} "
                    f"{'on' if enabled else 'off'}, so I've opened the {label} settings for you.",
                    ok=False)
            return ToolResult(failures[0], ok=False)
        note = "" if enabled or others_online else " That takes this PC offline, so I won't hear you until it's back."
        return ToolResult(f"{label} turned {'on' if enabled else 'off'}.{note}")

    # ── radios: Wi-Fi and Bluetooth without administrator rights ─────────────
    # Disabling an ADAPTER (netsh, above) needs elevation, so "turn the Wi-Fi
    # off" ended in a Settings page. Windows' Radio API switches the RADIO,
    # which is what the taskbar's own Wi-Fi and Bluetooth buttons do, as the
    # signed-in user. Bluetooth had no control at all.

    _RADIO_SCRIPT = (
        "Add-Type -AssemblyName System.Runtime.WindowsRuntime;"
        "$g=([System.WindowsRuntimeSystemExtensions].GetMethods()|?{$_.Name -eq 'AsTask' -and "
        "$_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq "
        "'IAsyncOperation`1'})[0];"
        "Function A($t,$r){$x=$g.MakeGenericMethod($r).Invoke($null,@($t));$x.Wait(-1)|Out-Null;$x.Result};"
        "[Windows.Devices.Radios.Radio,Windows.System.Devices,ContentType=WindowsRuntime]|Out-Null;"
        "$a=A ([Windows.Devices.Radios.Radio]::RequestAccessAsync()) ([Windows.Devices.Radios.RadioAccessStatus]);"
        "'access='+$a;"
        "$rs=A ([Windows.Devices.Radios.Radio]::GetRadiosAsync()) "
        "([System.Collections.Generic.IReadOnlyList[Windows.Devices.Radios.Radio]]);"
        "$k=$env:ORION_RADIO_KIND;$s=$env:ORION_RADIO_STATE;"
        "foreach($r in $rs){ if($k -and $s -and [string]$r.Kind -eq $k){"
        "'set='+(A ($r.SetStateAsync($s)) ([Windows.Devices.Radios.RadioAccessStatus])) };"
        "'radio='+$r.Kind+'|'+$r.State+'|'+$r.Name }"
    )

    def _radio_call(self, kind: str = "", state: str = "") -> list[str]:
        """Run the radio script; its output lines (empty on any failure)."""
        env = dict(os.environ, ORION_RADIO_KIND=kind, ORION_RADIO_STATE=state)
        try:
            done = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-Command", self._RADIO_SCRIPT],
                check=False, capture_output=True, text=True, errors="replace", timeout=20, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            return []
        return [line.strip() for line in (done.stdout or "").splitlines() if line.strip()]

    def radios(self) -> dict[str, str]:
        """{'WiFi': 'On', 'Bluetooth': 'Off', ...} for the radios this PC has."""
        if self._platform != "windows":
            return {}
        out: dict[str, str] = {}
        for line in self._radio_call():
            if line.startswith("radio="):
                kind, state, *_ = line[6:].split("|") + ["", ""]
                out.setdefault(kind, state)
        return out

    def set_radio(self, kind: str, enabled: bool | None) -> ToolResult | None:
        """Switch the 'WiFi' or 'Bluetooth' radio. None when this PC has no
        such radio or the API refused, so the caller can fall back."""
        label = "Wi-Fi" if kind == "WiFi" else kind
        if self._platform != "windows":
            return None
        current = self.radios()
        if kind not in current:
            return None
        if enabled is None:
            enabled = current[kind] != "On"
        if current[kind] == ("On" if enabled else "Off"):
            return ToolResult(f"{label} is already {'on' if enabled else 'off'}.")
        lines = self._radio_call(kind, "On" if enabled else "Off")
        result = next((line[4:] for line in lines if line.startswith("set=")), "")
        after = next((line[6:].split("|")[1] for line in lines
                      if line.startswith(f"radio={kind}|")), "")
        if result != "Allowed":
            return None
        if after and after != ("On" if enabled else "Off"):
            return ToolResult(f"Windows accepted the request but {label} still reads {after}.",
                              ok=False)
        return ToolResult(f"{label} turned {'on' if enabled else 'off'}.")

    def set_bluetooth(self, enabled: bool | None = None) -> ToolResult:
        if self._platform != "windows":
            return ToolResult("Bluetooth switching is only available on Windows.", ok=False)
        done = self.set_radio("Bluetooth", enabled)
        if done is not None:
            return done
        if "Bluetooth" not in self.radios():
            return ToolResult("This PC has no Bluetooth radio, so there's nothing to switch.",
                              ok=False)
        from .execution_policy import require_foreground
        blocked = require_foreground("bluetooth_settings_window")
        if blocked is not None:
            return blocked
        try:
            os.startfile("ms-settings:bluetooth")  # type: ignore[attr-defined]
        except Exception:
            pass
        return ToolResult("Windows wouldn't let me switch Bluetooth, so I've opened the "
                          "Bluetooth settings for you.", ok=False)

    def screen_off(self) -> ToolResult:
        """Put the monitors to sleep; any mouse movement or key wakes them.
        Posted, not sent: a broadcast SendMessage waits on every window and a
        hung one would hang this call."""
        from .execution_policy import require_foreground
        blocked = require_foreground("screen_off")
        if blocked is not None:
            return blocked
        if self._platform != "windows":
            return ToolResult("Turning the screen off is only available on Windows.", ok=False)
        try:
            import ctypes
            HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER = 0xFFFF, 0x0112, 0xF170
            ok = ctypes.windll.user32.PostMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)
        except Exception as exc:
            return ToolResult(f"I couldn't turn the screen off: {exc}", ok=False)
        if not ok:
            return ToolResult("Windows didn't accept the request to turn the screen off.", ok=False)
        return ToolResult("Screen off — move the mouse or press a key to wake it.")

    def toggle_wifi(self) -> ToolResult:
        return self.set_network("wifi")

    def toggle_ethernet(self) -> ToolResult:
        return self.set_network("ethernet")

    def toggle_network_adapter(self, adapter: str = "Ethernet") -> ToolResult:
        return self.set_network(self._adapter_kind(adapter) if self._adapter_kind(adapter) != "other" else "ethernet")


# ── level parsing ─────────────────────────────────────────────────────────────

_LEVEL_WORDS = {
    "max": 1.0, "maximum": 1.0, "full": 1.0, "full blast": 1.0, "all the way up": 1.0,
    "half": 0.5, "halfway": 0.5, "half way": 0.5, "a half": 0.5,
    "quarter": 0.25, "a quarter": 0.25, "three quarters": 0.75,
    "min": 0.0, "minimum": 0.0, "zero": 0.0, "nothing": 0.0, "none": 0.0,
}
_UNITS_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS_WORDS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def _number_words(text: str) -> float | None:
    words = text.replace("-", " ").split()
    if words == ["a", "hundred"] or words == ["one", "hundred"] or words == ["hundred"]:
        return 100.0
    total, seen = 0, False
    for word in words:
        if word in _TENS_WORDS:
            total += _TENS_WORDS[word]
        elif word in _UNITS_WORDS:
            total += _UNITS_WORDS[word]
        elif word == "and":
            continue
        else:
            return None
        seen = True
    return float(total) if seen else None


def parse_level(value: Any) -> float | None:
    """0.0-1.0 from what a model or a person actually supplies.

    The schema asks for 0.0-1.0 but models send 70 for "70%", and the old
    clamp turned that into 100% - "set the volume to 70" blasted full volume.
    `level or 0.5` also turned an explicit 0 into 50%. Accepted: 0.4, 40,
    "40%", "40 percent", "forty", "half", "max"; None for nothing usable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().lower().rstrip(".")
        text = text.removesuffix("percent").removesuffix("per cent").removesuffix("%").strip()
        if not text:
            return None
        if text in _LEVEL_WORDS:
            return _LEVEL_WORDS[text]
        try:
            number = float(text)
        except ValueError:
            number = _number_words(text)
            if number is None:
                return None
            return max(0.0, min(1.0, number / 100.0))
    if number != number:          # NaN
        return None
    if number > 1.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


# ── DDC/CI (external monitors) ────────────────────────────────────────────────

def _physical_monitors() -> list[tuple[Any, Any, int]]:
    """[(dxva2, PHYSICAL_MONITOR array, count)] for every display - the caller
    must DestroyPhysicalMonitors each array."""
    import ctypes
    from ctypes import wintypes

    class PHYSICAL_MONITOR(ctypes.Structure):
        _fields_ = [("hPhysicalMonitor", wintypes.HANDLE),
                    ("szPhysicalMonitorDescription", wintypes.WCHAR * 128)]

    user32, dxva2 = ctypes.windll.user32, ctypes.windll.dxva2
    handles: list[Any] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                              ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def collect(hmonitor, _hdc, _rect, _param):
        handles.append(hmonitor)
        return True

    user32.EnumDisplayMonitors(None, None, proc(collect), 0)
    found = []
    for hmonitor in handles:
        count = wintypes.DWORD()
        if not dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR(hmonitor, ctypes.byref(count)):
            continue
        if not count.value:
            continue
        array = (PHYSICAL_MONITOR * count.value)()
        if dxva2.GetPhysicalMonitorsFromHMONITOR(hmonitor, count.value, array):
            found.append((dxva2, array, count.value))
    return found


def _ddc_brightness() -> list[float]:
    """Brightness 0.0-1.0 of each DDC/CI monitor that answers (VCP 0x10)."""
    import ctypes
    from ctypes import wintypes
    levels: list[float] = []
    try:
        monitors = _physical_monitors()
    except Exception:
        return levels
    for dxva2, array, count in monitors:
        for monitor in array:
            kind, current, maximum = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD()
            if dxva2.GetVCPFeatureAndVCPFeatureReply(
                    monitor.hPhysicalMonitor, 0x10, ctypes.byref(kind),
                    ctypes.byref(current), ctypes.byref(maximum)) and maximum.value:
                levels.append(current.value / maximum.value)
        dxva2.DestroyPhysicalMonitors(count, array)
    return levels


def _ddc_set_brightness(level: float) -> int:
    """Set every DDC/CI monitor's brightness; returns how many accepted it."""
    import ctypes
    from ctypes import wintypes
    changed = 0
    try:
        monitors = _physical_monitors()
    except Exception:
        return 0
    for dxva2, array, count in monitors:
        for monitor in array:
            kind, current, maximum = wintypes.DWORD(), wintypes.DWORD(), wintypes.DWORD()
            if not dxva2.GetVCPFeatureAndVCPFeatureReply(
                    monitor.hPhysicalMonitor, 0x10, ctypes.byref(kind),
                    ctypes.byref(current), ctypes.byref(maximum)) or not maximum.value:
                continue
            value = int(round(level * maximum.value))
            if dxva2.SetVCPFeature(monitor.hPhysicalMonitor, 0x10, value):
                changed += 1
        dxva2.DestroyPhysicalMonitors(count, array)
    return changed
