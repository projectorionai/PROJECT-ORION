"""
ORION writes robot firmware: sketches compiled with Arduino CLI against the
ORION Robot Link library (robot/orion_link), checked before anything is
flashed.

Arduino CLI and its board packages live in %USERPROFILE%\\ORION\\toolchain
(the owner approved the download on 28 Oct 2026), never in the app.
Sketches ORION writes go to CONFIG_DIR/robot/sketches/<name>/<name>.ino.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR
from .robot_link import ROBOT_DIR, TOOLCHAIN

SKETCHES = CONFIG_DIR / "robot" / "sketches"
LIBRARY = ROBOT_DIR / "orion_link"
#: Boards by a name a person would say, and the board package each needs.
BOARDS = {
    "uno": ("arduino:avr:uno", "arduino:avr"),
    "nano": ("arduino:avr:nano", "arduino:avr"),
    "mega": ("arduino:avr:mega", "arduino:avr"),
    # ATmega32U4: Serial is the USB Serial_ class, not HardwareSerial.
    "leonardo": ("arduino:avr:leonardo", "arduino:avr"),
    "micro": ("arduino:avr:micro", "arduino:avr"),
    "esp32": ("esp32:esp32:esp32", "esp32:esp32"),
}
ESP32_INDEX = "https://espressif.github.io/arduino-esp32/package_esp32_index.json"
_NAME = re.compile(r"[^A-Za-z0-9_]+")


def cli_path() -> Path | None:
    for name in ("arduino-cli.exe", "arduino-cli"):
        candidate = TOOLCHAIN / "arduino-cli" / name
        if candidate.is_file():
            return candidate
    return None


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"ARDUINO_DIRECTORIES_DATA": str(TOOLCHAIN / "arduino-data"),
                "ARDUINO_DIRECTORIES_USER": str(TOOLCHAIN / "arduino-user"),
                "ARDUINO_DIRECTORIES_DOWNLOADS": str(TOOLCHAIN / "arduino-downloads"),
                "ARDUINO_BOARD_MANAGER_ADDITIONAL_URLS": ESP32_INDEX})
    return env


def run_cli(*args: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    cli = cli_path()
    if cli is None:
        raise FileNotFoundError("Arduino CLI is not installed in ORION's toolchain folder")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run([str(cli), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=_env(), creationflags=flags)


def board(name: str) -> tuple[str, str]:
    """(FQBN, board package) for 'uno', 'esp32'... or a full FQBN."""
    key = str(name or "uno").strip().lower()
    if key in BOARDS:
        return BOARDS[key]
    if key.count(":") == 2:
        vendor, arch, _ = key.split(":")
        return key, f"{vendor}:{arch}"
    raise ValueError(f"unknown board '{name}' (try {', '.join(BOARDS)} or a full FQBN)")


def ensure_core(package: str) -> str:
    """Install a board package the first time it is needed."""
    installed = run_cli("core", "list", "--format", "json", timeout=120)
    if package in installed.stdout:
        return ""
    run_cli("core", "update-index", timeout=300)
    result = run_cli("core", "install", package, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"could not install the {package} board package: "
                           + (result.stderr or result.stdout).strip()[-400:])
    return f"installed the {package} board package"


def sketch_dir(name: str) -> Path:
    clean = _NAME.sub("_", str(name or "").strip()).strip("_")[:40]
    if not clean:
        raise ValueError("a sketch needs a name")
    return SKETCHES / clean


def write_sketch(name: str, code: str) -> Path:
    if not str(code or "").strip():
        raise ValueError("no code to write")
    folder = sketch_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{folder.name}.ino"
    target.write_text(str(code), encoding="utf-8")
    return target


def compile_sketch(folder: Path, fqbn: str) -> tuple[bool, str]:
    """Compile with every warning on. Returns (ok, the compiler's last words)."""
    result = run_cli("compile", "--fqbn", fqbn, "--library", str(LIBRARY), "--warnings", "all",
                     str(folder), timeout=900)
    output = (result.stdout + "\n" + result.stderr).strip()
    lines = [line for line in output.splitlines() if line.strip()]
    return result.returncode == 0, "\n".join(lines[-40:])


def upload_sketch(folder: Path, fqbn: str, port: str) -> tuple[bool, str]:
    result = run_cli("upload", "--fqbn", fqbn, "--port", str(port), str(folder), timeout=300)
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, "\n".join(output.splitlines()[-20:])


def connected_boards() -> list[dict[str, Any]]:
    result = run_cli("board", "list", "--format", "json", timeout=60)
    try:
        data = json.loads(result.stdout or "{}")
    except ValueError:
        return []
    ports = data.get("detected_ports", data) if isinstance(data, dict) else data
    out = []
    for entry in ports or []:
        port = (entry.get("port") or {}) if isinstance(entry, dict) else {}
        boards = [b.get("name", "") for b in entry.get("matching_boards") or []] if isinstance(entry, dict) else []
        out.append({"port": port.get("address", ""), "kind": port.get("protocol_label", ""),
                    "boards": [b for b in boards if b]})
    return out


__all__ = ["BOARDS", "board", "cli_path", "compile_sketch", "connected_boards", "ensure_core",
           "sketch_dir", "upload_sketch", "write_sketch"]
