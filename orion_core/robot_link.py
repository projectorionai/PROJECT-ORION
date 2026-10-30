"""
ORION's side of ORION Robot Link (ORL) v1 - see robot/orion_link/orion_link.h.

The robot's controller runs the C library; this is the same wire format in
Python, plus a connection that keeps ORION's heartbeat going (a robot that
stops hearing it stops its motors within half a second), waits for every
command's acknowledgement, and remembers the robot's latest sensors.

Connections:
    "sim"                    the software robot (robot/sim/orl_sim.c), built
                             with the toolchain in %USERPROFILE%\\ORION\\toolchain
    "serial:COM5" or "serial:COM5@115200"   a board on USB (needs pyserial)
    "tcp:192.168.1.40:7878"  a robot on the network (Raspberry Pi, ESP32)
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

VERSION = 1
SYNC = b"\xa5\x5a"
MAX_PAYLOAD = 160
HELLO, SENSORS, ACK, EVENT = 0x01, 0x02, 0x03, 0x04
HEARTBEAT, DRIVE, SERVO, STOP, ESTOP, LED, SAY = 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16
OK, REJECTED, UNSUPPORTED, ESTOPPED = 0, 1, 2, 3
RESULTS = {OK: "ok", REJECTED: "rejected", UNSUPPORTED: "unsupported", ESTOPPED: "e-stopped"}
CAPS = {0x01: "drive", 0x02: "servo", 0x04: "distance", 0x08: "battery", 0x10: "imu",
        0x20: "speaker", 0x40: "led"}
STATUS_ESTOPPED, STATUS_WATCHDOG, STATUS_LOW_BATTERY = 0x01, 0x02, 0x04
DRIVE_LIMIT = 1000
#: The longest single drive ORION sends (ms); longer moves are several commands.
MAX_DRIVE_MS = 10_000
HEARTBEAT_S = 0.1
ACK_TIMEOUT_S = 0.6
ROBOT_DIR = Path(__file__).resolve().parents[1] / "robot"
TOOLCHAIN = Path(os.path.expandvars(os.getenv("ORION_TOOLCHAIN", r"%USERPROFILE%\ORION\toolchain")))


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE, as orl_crc16."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode(kind: int, seq: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload of {len(payload)} bytes is over {MAX_PAYLOAD}")
    body = struct.pack("<BBBH", VERSION, kind & 0xFF, seq & 0xFF, len(payload)) + bytes(payload)
    return SYNC + body + struct.pack("<H", crc16(body))


@dataclass
class Frame:
    kind: int
    seq: int
    payload: bytes


class Parser:
    """A byte-at-a-time parser with the same resynchronisation as the C one."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.crc_errors = 0

    def feed(self, data: bytes) -> list[Frame]:
        self._buffer += data
        frames: list[Frame] = []
        while True:
            start = self._buffer.find(SYNC)
            if start < 0:
                del self._buffer[:-1 if self._buffer.endswith(b"\xa5") else len(self._buffer)]
                return frames
            del self._buffer[:start]
            if len(self._buffer) < 7:
                return frames
            version, kind, seq, length = struct.unpack_from("<BBBH", self._buffer, 2)
            if version != VERSION or length > MAX_PAYLOAD:
                del self._buffer[:1]
                continue
            total = 7 + length + 2
            if len(self._buffer) < total:
                return frames
            body = bytes(self._buffer[2:7 + length])
            (crc,) = struct.unpack_from("<H", self._buffer, 7 + length)
            del self._buffer[:total]
            if crc != crc16(body):
                self.crc_errors += 1
                continue
            frames.append(Frame(kind, seq, body[5:]))


def drive_payload(left: int, right: int, duration_ms: int = 0) -> bytes:
    return struct.pack("<hhH", int(left), int(right), int(duration_ms))


def servo_payload(index: int, degrees: float) -> bytes:
    return struct.pack("<BH", int(index), int(round(float(degrees) * 10)))


@dataclass
class Sensors:
    battery_mv: int = 0
    distance_mm: int | None = None
    accel_mg: tuple[int, int, int] = (0, 0, 0)
    gyro_z_dps: float = 0.0
    bumpers: int = 0
    status: int = 0
    at: float = 0.0

    @classmethod
    def unpack(cls, payload: bytes) -> "Sensors":
        battery, distance, ax, ay, az, gz, bumpers, status = struct.unpack("<HHhhhhBB", payload)
        return cls(battery, None if distance == 0xFFFF else distance, (ax, ay, az), gz / 10.0,
                   bumpers, status, time.monotonic())

    def describe(self) -> str:
        parts = []
        if self.battery_mv:
            parts.append(f"battery {self.battery_mv / 1000:.2f} V")
        if self.distance_mm is not None:
            parts.append(f"{self.distance_mm} mm clear ahead")
        if self.status & STATUS_ESTOPPED:
            parts.append("emergency stop engaged")
        if self.status & STATUS_WATCHDOG:
            parts.append("stopped by its watchdog (lost ORION's heartbeat)")
        if self.status & STATUS_LOW_BATTERY:
            parts.append("battery low")
        return ", ".join(parts) or "no readings yet"


@dataclass
class RobotInfo:
    name: str = ""
    caps: int = 0
    servos: int = 0

    def capabilities(self) -> list[str]:
        return [name for bit, name in CAPS.items() if self.caps & bit]


# ── transports ───────────────────────────────────────────────────────────────

class _Transport:
    """read() blocks briefly: bytes, None when nothing arrived yet, b"" when
    the connection has closed."""

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def read(self) -> bytes | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _ProcessTransport(_Transport):
    def __init__(self, command: list[str]) -> None:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, creationflags=flags, bufsize=0)

    def write(self, data: bytes) -> None:
        if self.proc.stdin is None:
            raise OSError("the simulator's input is closed")
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def read(self) -> bytes | None:
        stream = self.proc.stdout
        if stream is None:
            return b""
        return stream.read1(256) if hasattr(stream, "read1") else stream.read(1)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


class _SocketTransport(_Transport):
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=3.0)
        self.sock.settimeout(0.2)

    def write(self, data: bytes) -> None:
        self.sock.sendall(data)

    def read(self) -> bytes | None:
        try:
            return self.sock.recv(256)               # b"": the robot closed the connection
        except socket.timeout:
            return None

    def close(self) -> None:
        self.sock.close()


class _SerialTransport(_Transport):
    def __init__(self, port: str, baud: int) -> None:
        import serial  # pyserial
        self.port = serial.Serial(port, baud, timeout=0.1)

    def write(self, data: bytes) -> None:
        self.port.write(data)

    def read(self) -> bytes | None:
        return self.port.read(256) or None          # a serial port only times out

    def close(self) -> None:
        self.port.close()


def simulator_path() -> Path:
    return ROBOT_DIR / "sim" / ("orl_sim.exe" if os.name == "nt" else "orl_sim")


def build_simulator() -> Path:
    """Compile the software robot with the toolchain's C compiler (once)."""
    target = simulator_path()
    sources = [ROBOT_DIR / "sim" / "orl_sim.c", ROBOT_DIR / "orion_link" / "orion_link.c"]
    if target.exists() and target.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return target
    zig = zig_path()
    if zig is None:
        raise FileNotFoundError("no C compiler: the toolchain is not in ORION_TOOLCHAIN")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run([str(zig), "cc", "-std=c99", "-O2", "-I", str(ROBOT_DIR / "orion_link"),
                    *map(str, sources), "-o", str(target)], check=True, capture_output=True,
                   timeout=180, creationflags=flags)
    return target


def zig_path() -> Path | None:
    for candidate in (TOOLCHAIN / "zig" / "ziglang" / "zig.exe", TOOLCHAIN / "zig" / "ziglang" / "zig"):
        if candidate.is_file():
            return candidate
    return None


def open_transport(address: str) -> _Transport:
    address = str(address or "").strip()
    if address == "sim":
        return _ProcessTransport([str(build_simulator())])
    kind, _, rest = address.partition(":")
    if kind == "tcp":
        host, _, port = rest.rpartition(":")
        return _SocketTransport(host, int(port))
    if kind == "serial":
        port, _, baud = rest.partition("@")
        return _SerialTransport(port, int(baud or 115200))
    raise ValueError("connect to 'sim', 'serial:COM5[@115200]' or 'tcp:host:port'")


# ── the link ─────────────────────────────────────────────────────────────────

class RobotLink:
    """One robot: heartbeat, acknowledged commands, latest sensors."""

    def __init__(self, address: str, *, log: Callable[[str], Any] | None = None,
                 transport: _Transport | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.address = address
        self.log = log or (lambda _line: None)
        self.clock = clock
        self._transport = transport or open_transport(address)
        self._parser = Parser()
        self._lock = threading.Lock()
        self._seq = 0
        self._acks: dict[int, tuple[threading.Event, list[int]]] = {}
        self.info = RobotInfo()
        self.sensors = Sensors()
        self.events: list[tuple[int, str]] = []
        self.last_heard = 0.0
        self._started = clock()
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, name="orion-robot-read", daemon=True)
        self._heart = threading.Thread(target=self._heartbeat_loop, name="orion-robot-heartbeat", daemon=True)
        self._reader.start()
        self._heart.start()

    # sending
    def _send(self, kind: int, payload: bytes = b"") -> int:
        with self._lock:
            seq = self._seq
            self._seq = (self._seq + 1) & 0xFF
            self._transport.write(encode(kind, seq, payload))
        return seq

    def heartbeat(self) -> None:
        now_ms = int((self.clock() - self._started) * 1000) & 0xFFFFFFFF
        self._send(HEARTBEAT, struct.pack("<I", now_ms))

    def command(self, kind: int, payload: bytes = b"", timeout: float = ACK_TIMEOUT_S) -> str:
        """Send one command and wait for its acknowledgement: 'ok', 'rejected',
        'unsupported', 'e-stopped' or 'no answer'."""
        event, box = threading.Event(), []
        with self._lock:
            seq = self._seq
            self._acks[seq] = (event, box)
            self._seq = (self._seq + 1) & 0xFF
            self._transport.write(encode(kind, seq, payload))
        answered = event.wait(timeout)
        with self._lock:
            self._acks.pop(seq, None)
        return RESULTS.get(box[0], f"result {box[0]}") if answered and box else "no answer"

    def drive(self, left: int, right: int, duration_ms: int = 0) -> str:
        left, right, duration_ms = int(left), int(right), int(duration_ms)
        if not (-DRIVE_LIMIT <= left <= DRIVE_LIMIT and -DRIVE_LIMIT <= right <= DRIVE_LIMIT):
            raise ValueError(f"wheel power must be -{DRIVE_LIMIT}..{DRIVE_LIMIT} per mille")
        if not 0 <= duration_ms <= MAX_DRIVE_MS:
            raise ValueError(f"a single drive lasts 0..{MAX_DRIVE_MS} ms")
        return self.command(DRIVE, drive_payload(left, right, duration_ms))

    def stop(self) -> str:
        return self.command(STOP)

    def estop(self, engaged: bool = True) -> str:
        return self.command(ESTOP, bytes([1 if engaged else 0]))

    def servo(self, index: int, degrees: float) -> str:
        if not 0.0 <= float(degrees) <= 180.0:
            raise ValueError("a servo angle is 0..180 degrees")
        return self.command(SERVO, servo_payload(index, degrees))

    def led(self, red: int, green: int, blue: int) -> str:
        return self.command(LED, bytes(max(0, min(255, int(v))) for v in (red, green, blue)))

    def say(self, text: str) -> str:
        data = str(text).encode("utf-8")[:128]
        if not data:
            raise ValueError("nothing to say")
        return self.command(SAY, data)

    # receiving
    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._transport.read()
            except Exception as exc:
                self.log(f"ROBOT: link read failed - {exc}")
                return
            if data is None:
                continue
            if not data:
                self.log(f"ROBOT: the link to {self.address} closed.")
                return
            for frame in self._parser.feed(data):
                self._on_frame(frame)

    def _on_frame(self, frame: Frame) -> None:
        self.last_heard = self.clock()
        if frame.kind == ACK and len(frame.payload) == 2:
            with self._lock:
                waiter = self._acks.get(frame.payload[0])
            if waiter is not None:
                waiter[1].append(frame.payload[1])
                waiter[0].set()
        elif frame.kind == SENSORS and len(frame.payload) == 14:
            self.sensors = Sensors.unpack(frame.payload)
        elif frame.kind == HELLO and len(frame.payload) >= 2:
            self.info = RobotInfo(frame.payload[2:].decode("utf-8", "replace"),
                                  frame.payload[0], frame.payload[1])
            self.log(f"ROBOT: {self.info.name or 'a robot'} connected "
                     f"({', '.join(self.info.capabilities()) or 'no parts declared'}).")
        elif frame.kind == EVENT and frame.payload:
            self.events.append((frame.payload[0], frame.payload[1:].decode("utf-8", "replace")))
            del self.events[:-50]

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            try:
                self.heartbeat()
            except Exception as exc:
                self.log(f"ROBOT: heartbeat failed - {exc}")
                return

    @property
    def connected(self) -> bool:
        return bool(self.info.caps or self.info.name) and self.clock() - self.last_heard < 2.0

    def status(self) -> str:
        if not self.connected:
            return f"No answer from the robot on {self.address}."
        return f"{self.info.name or 'The robot'} on {self.address}: {self.sensors.describe()}."

    def close(self) -> None:
        """Stop the motors, then the heartbeat, then the connection."""
        try:
            self._send(STOP)
        except Exception:
            pass
        self._stop.set()
        self._transport.close()


__all__ = ["ACK", "DRIVE", "ESTOP", "HEARTBEAT", "HELLO", "LED", "Parser", "RobotLink", "SAY",
           "SENSORS", "SERVO", "STOP", "Sensors", "build_simulator", "crc16", "encode",
           "open_transport", "zig_path"]
