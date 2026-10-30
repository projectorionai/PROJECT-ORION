"""
Robot tools: robot_control (drive a robot over ORION Robot Link) and
robot_firmware (write, compile and flash its code).

Owner, 28 Oct 2026: ORION "must be ready to push into a robot with C and
C++". The protocol and the C/C++ library are in robot/; this is ORION's side.

Safety:
  * stop and the emergency stop are always allowed, from anywhere;
  * motion (drive, turn, servo) runs only on the owner's own live turn - a
    job, plugin, agent or the phone never moves the robot;
  * flashing a board is the owner's live turn too, and is compiled first.
"""

from __future__ import annotations

import time
from typing import Any

from .data import ToolResult

#: Default speed (percent of full power) and times for the simple moves.
MOVE_SPEED_PCT = 40
MOVE_SECONDS = 1.0
TURN_SECONDS = 0.5


def _number(args: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(args.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number")
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low:g} and {high:g}")
    return value


class RobotDispatchMixin:
    _robot: Any = None

    def _robot_log(self, line: str) -> None:
        bus = getattr(self, "bus", None)
        if bus is not None:
            try:
                bus.log.emit(line)
            except RuntimeError:
                pass

    # ── robot_control ────────────────────────────────────────────────────────

    async def robot_control_tool(self, args: dict[str, Any]) -> ToolResult:
        """Off ORION's loop: a command waits up to 0.6 s for its
        acknowledgement and connecting up to 2 s. asyncio.to_thread carries
        the owner-turn context, so the motion check still sees whose turn it is."""
        import asyncio
        return await asyncio.to_thread(self._robot_control, dict(args or {}))

    async def robot_firmware_tool(self, args: dict[str, Any]) -> ToolResult:
        import asyncio
        return await asyncio.to_thread(self._robot_firmware, dict(args or {}))

    def _robot_control(self, args: dict[str, Any]) -> ToolResult:
        from . import robot_link
        from .execution_policy import require_foreground

        action = str(args.get("action") or "status").strip().lower().replace(" ", "_")
        link = getattr(self, "_robot", None)
        try:
            if action == "connect":
                address = str(args.get("address") or "sim").strip()
                if link is not None:
                    link.close()
                self._robot = link = robot_link.RobotLink(address, log=self._robot_log)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and not link.connected:
                    time.sleep(0.05)
                if not link.connected:
                    link.close()
                    self._robot = None
                    return ToolResult(f"No robot answered on {address}.", ok=False)
                return ToolResult(f"Connected to {link.info.name or 'the robot'} on {address}: "
                                  f"{', '.join(link.info.capabilities()) or 'no parts declared'}. "
                                  f"{link.sensors.describe().capitalize()}.",
                                  receipt={"robot": link.info.name, "address": address})
            if link is None:
                return ToolResult("No robot is connected. Connect first (address 'sim' for the "
                                  "software robot, 'serial:COM5' or 'tcp:host:port').", ok=False)
            if action == "disconnect":
                link.close()
                self._robot = None
                return ToolResult("Robot stopped and disconnected.")
            if action == "status":
                return ToolResult(link.status(), ok=link.connected)
            if action == "stop":
                return self._robot_result("Stop", link.stop())
            if action in ("estop", "emergency_stop"):
                return self._robot_result("Emergency stop engaged", link.estop(True))
            if action in ("release", "release_estop"):
                return self._robot_result("Emergency stop released", link.estop(False))

            # Everything below moves something.
            blocked = require_foreground("robot_motion")
            if blocked is not None:
                return blocked
            if action == "drive":
                left = _number(args, "left", 0, -100, 100)
                right = _number(args, "right", 0, -100, 100)
                seconds = _number(args, "seconds", MOVE_SECONDS, 0, robot_link.MAX_DRIVE_MS / 1000)
                return self._robot_result(f"Driving left {left:g}%, right {right:g}% for {seconds:g} s",
                                          link.drive(round(left * 10), round(right * 10), round(seconds * 1000)))
            if action in ("forward", "backward", "turn_left", "turn_right"):
                speed = _number(args, "speed", MOVE_SPEED_PCT, 1, 100) * 10
                seconds = _number(args, "seconds", TURN_SECONDS if action.startswith("turn") else MOVE_SECONDS,
                                  0.05, robot_link.MAX_DRIVE_MS / 1000)
                left, right = {"forward": (speed, speed), "backward": (-speed, -speed),
                               "turn_left": (-speed, speed), "turn_right": (speed, -speed)}[action]
                return self._robot_result(f"{action.replace('_', ' ').capitalize()} for {seconds:g} s",
                                          link.drive(round(left), round(right), round(seconds * 1000)))
            if action == "servo":
                index = int(_number(args, "index", 0, 0, 15))
                return self._robot_result(f"Servo {index} to {args.get('degrees')} degrees",
                                          link.servo(index, _number(args, "degrees", 90, 0, 180)))
            if action == "led":
                colour = [int(_number(args, key, 0, 0, 255)) for key in ("red", "green", "blue")]
                return self._robot_result("LED set", link.led(*colour))
            if action == "say":
                return self._robot_result("Said on the robot", link.say(str(args.get("text") or "")))
        except (ValueError, OSError) as exc:
            return ToolResult(f"Robot: {exc}.", ok=False)
        return ToolResult(f"Unknown robot action '{action}'.", ok=False)

    @staticmethod
    def _robot_result(done: str, result: str) -> ToolResult:
        if result == "ok":
            return ToolResult(f"{done}.", receipt={"robot_result": result})
        why = {"e-stopped": "the emergency stop is engaged - release it first",
               "rejected": "the robot refused it as out of range",
               "unsupported": "this robot does not have that part",
               "no answer": "the robot did not acknowledge it"}.get(result, result)
        return ToolResult(f"Not done: {why}.", ok=False, receipt={"robot_result": result})

    # ── robot_firmware ───────────────────────────────────────────────────────

    def _robot_firmware(self, args: dict[str, Any]) -> ToolResult:
        from . import robot_firmware as fw
        from .execution_policy import require_foreground

        action = str(args.get("action") or "boards").strip().lower()
        try:
            if fw.cli_path() is None:
                return ToolResult("Arduino CLI is not installed in ORION's toolchain folder.", ok=False)
            if action == "boards":
                found = fw.connected_boards()
                if not found:
                    return ToolResult("No boards are plugged in.")
                return ToolResult("Plugged in: " + "; ".join(
                    f"{b['port']} ({', '.join(b['boards']) or b['kind'] or 'unknown board'})" for b in found))
            name = str(args.get("name") or "").strip()
            fqbn, package = fw.board(str(args.get("board") or "uno"))
            folder = fw.sketch_dir(name)
            if action == "write":
                path = fw.write_sketch(name, str(args.get("code") or ""))
                action = "compile"                   # always checked straight away
                self._robot_log(f"ROBOT: wrote {path.name}.")
            if action in ("compile", "flash", "upload"):
                if not folder.is_dir():
                    return ToolResult(f"There is no sketch called '{name}'. Write it first.", ok=False)
                note = fw.ensure_core(package)
                ok, output = fw.compile_sketch(folder, fqbn)
                if not ok:
                    return ToolResult(f"The sketch did not compile for {fqbn}. Compiler output:\n{output}",
                                      ok=False, receipt={"compiled": False})
                if action == "compile":
                    return ToolResult(f"Compiled for {fqbn}" + (f" ({note})" if note else "") +
                                      f". {output.splitlines()[-2] if len(output.splitlines()) > 1 else output}",
                                      receipt={"compiled": True, "fqbn": fqbn})
                blocked = require_foreground("robot_firmware_flash")
                if blocked is not None:
                    return blocked
                port = str(args.get("port") or "").strip()
                if not port:
                    return ToolResult("Which port is the board on? (robot_firmware boards lists them.)", ok=False)
                ok, output = fw.upload_sketch(folder, fqbn, port)
                return ToolResult((f"Flashed {name} to the {fqbn} on {port}." if ok
                                   else f"Flashing failed:\n{output}"), ok=ok)
        except (ValueError, OSError, RuntimeError) as exc:
            return ToolResult(f"Robot firmware: {exc}.", ok=False)
        return ToolResult(f"Unknown firmware action '{action}'.", ok=False)


__all__ = ["RobotDispatchMixin"]
