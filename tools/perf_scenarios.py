"""Repeatable performance scenarios for ORION.

    python tools/perf_scenarios.py startup --warm-runs 3 --idle 20 --out startup.json
    python tools/perf_scenarios.py local-command --samples 200 --out routing.json
    python tools/perf_scenarios.py retrieval --out retrieval.json
    python tools/perf_scenarios.py shutdown --runs 20 --phone-every 2 --out shutdown.json

Every scenario records what it measured, how, on what machine, and what it
could NOT measure, so a number is never read without its conditions.

Isolation. ``startup`` boots the real application from this checkout against a
fresh temporary config directory with a dummy local provider (nothing answers
on 127.0.0.1:9), so no personal data, keys or paid provider are touched.
``ORION_NO_CAPTURE=1`` keeps the microphone and camera closed; the
``microphone_ready`` milestone is therefore reported as not measured.
Offscreen Qt is the default: it measures the boot's own work, not GPU
presentation, so frame times need ``--visible`` on a real desktop session.

"Cold" here means the first launch against a new config directory (first-run
seeding, empty caches). The operating system's file cache cannot be flushed
without administrator rights, so a truly cold disk is not claimed.

``shutdown`` boots the same isolated ORION, waits for command readiness, and
then asks for a normal shutdown exactly as the Quit button does
(``bus.request_shutdown.emit()`` on the event-loop thread; nothing is killed).
It times request to process exit and reads ``last_shutdown.txt`` from the
temporary config. ``ORION_START_MUTED=1`` keeps the farewell code path
(compose, speak_text, the wait for speech to end) while playing nothing, so
the goodbye's own speaking time is not in the number. With ``--phone-every N``
every Nth run also starts the phone gateway on a free 127.0.0.1 port and
holds a paired phone's live event stream and uplink websocket open across
the shutdown; nothing leaves localhost. It refuses to start while any ORION
holds the single-instance lock.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


# ── shared helpers ────────────────────────────────────────────────────────────

def _machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 2**30, 2)
        info["physical_cpus"] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    for module in ("PyQt6.QtCore", "numpy", "aiohttp"):
        try:
            mod = __import__(module, fromlist=["_"])
            info[module] = getattr(mod, "PYQT_VERSION_STR", None) or getattr(mod, "__version__", "?")
        except Exception:
            info[module] = None
    return info


def _pct(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 3)


def _write(result: dict[str, Any], out: str | None) -> None:
    text = json.dumps(result, indent=2)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    print(text)


# ── startup + idle ────────────────────────────────────────────────────────────

_READY_RE = re.compile(r"process start to ready: ([\d.]+) s")
_PHASE_RE = re.compile(r"^\s+([\d.]+)s\s+\(\+\s*(\d+) ms\)\s+(.+?)(\s+<- slow)?$")
_READINESS_RE = re.compile(r"(\w+) (?:([\d.]+)s|-)")


def _dummy_provider_config(config_dir: Path) -> None:
    """A provider set with nothing real in it: one local profile pointing at a
    port nothing listens on, so the first-run key dialog does not block and no
    cloud provider is ever called."""
    saved = {k: os.environ.pop(k) for k in list(os.environ)
             if k.startswith("ORION_") and ("KEY" in k or "BASE_URL" in k)}
    try:
        sys.path.insert(0, str(ROOT))
        from orion_core.providers import _default_provider_payload
        payload = _default_provider_payload("")
    finally:
        os.environ.update(saved)
    for profile in payload.get("providers", {}).values():
        if isinstance(profile, dict):
            profile["enabled"] = False
            if "api_key" in profile:
                profile["api_key"] = ""
    # A generic OpenAI-compatible profile, not the Ollama one: an Ollama
    # profile would have ORION try to start the user's Ollama server.
    local = payload.get("providers", {}).get("local_lm_studio")
    if not isinstance(local, dict):
        raise RuntimeError("the provider payload has no local_lm_studio profile")
    local.update(enabled=True, base_url="http://127.0.0.1:9/v1")
    (config_dir / "api_keys.json").write_text(json.dumps(payload, indent=2),
                                              encoding="utf-8")


def _child_env(config_dir: Path, visible: bool) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("ORION_") and ("KEY" in k or "TOKEN" in k))}
    env.update(ORION_CONFIG_DIR=str(config_dir), ORION_NO_CAPTURE="1",
               ORION_START_MUTED="1", PYTHONUNBUFFERED="1")
    if not visible:
        env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def _tree(proc: Any) -> list[Any]:
    try:
        return [proc] + proc.children(recursive=True)
    except Exception:
        return [proc]


def _tree_memory(proc: Any) -> dict[str, Any]:
    """Private and resident memory for the process AND its children — the
    WebEngine renderer and GPU processes are ORION's cost too."""
    private = rss = 0
    names: dict[str, int] = {}
    for p in _tree(proc):
        try:
            info = p.memory_info()
            private += getattr(info, "private", 0) or getattr(info, "vms", 0)
            rss += info.rss
            names[p.name()] = names.get(p.name(), 0) + 1
        except Exception:
            continue
    return {"private_mb": round(private / 2**20, 1), "rss_mb": round(rss / 2**20, 1),
            "processes": names}


def _parse_boot(text: str) -> dict[str, Any]:
    ready = _READY_RE.search(text)
    phases = []
    readiness: dict[str, float | None] = {}
    for line in text.splitlines():
        m = _PHASE_RE.match(line)
        if m:
            phases.append({"at_s": float(m.group(1)), "ms": int(m.group(2)),
                           "phase": m.group(3).strip()})
        if line.strip().startswith("readiness:"):
            for name, at in _READINESS_RE.findall(line.split(":", 1)[1]):
                readiness[name] = float(at) if at else None
    slowest = sorted(phases, key=lambda p: -p["ms"])[:6]
    return {"ready_s": float(ready.group(1)) if ready else None,
            "readiness_s": readiness, "slowest_phases": slowest,
            "phase_count": len(phases)}


def _one_boot(config_dir: Path, *, idle_s: float, visible: bool,
              timeout_s: float) -> dict[str, Any]:
    import psutil

    boot_file = config_dir / "diagnostics" / "last_boot.txt"
    if boot_file.exists():
        boot_file.unlink()
    log = config_dir / "perf_console.txt"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8", errors="replace") as sink:
        child = subprocess.Popen([str(PYTHON), "orion.py"], cwd=ROOT,
                                 env=_child_env(config_dir, visible),
                                 stdout=sink, stderr=subprocess.STDOUT)
    proc = psutil.Process(child.pid)
    result: dict[str, Any] = {}
    try:
        while not boot_file.exists():
            if child.poll() is not None:
                return {"error": f"exited with {child.returncode} before ready",
                        "console_tail": log.read_text(errors="replace")[-1500:]}
            if time.perf_counter() - started > timeout_s:
                return {"error": f"not ready after {timeout_s:.0f}s"}
            time.sleep(0.1)
        result["wall_to_report_s"] = round(time.perf_counter() - started, 2)
        # The milestones that arrive after the report rewrite it; give them a
        # moment, then sample the idle process tree.
        samples_cpu: list[float] = []
        # cpu_percent() measures since the previous call ON THE SAME OBJECT;
        # a fresh Process from children() always reports 0.0. Keep one object
        # per pid across samples.
        tracked: dict[int, Any] = {}

        def _refresh() -> None:
            for p in _tree(proc):
                if p.pid not in tracked:
                    tracked[p.pid] = p
                    try:
                        p.cpu_percent(None)
                    except Exception:
                        pass

        _refresh()
        memory_series = []
        end = time.perf_counter() + idle_s
        while time.perf_counter() < end:
            time.sleep(1.0)
            total = 0.0
            for p in list(tracked.values()):
                try:
                    total += p.cpu_percent(None)
                except Exception:
                    tracked.pop(p.pid, None)
            _refresh()
            samples_cpu.append(total / (os.cpu_count() or 1))
            memory_series.append(_tree_memory(proc)["private_mb"])
        result.update(_parse_boot(boot_file.read_text(encoding="utf-8", errors="replace")))
        result["idle_seconds"] = idle_s
        result["idle_cpu_pct_of_machine"] = {
            "mean": round(statistics.fmean(samples_cpu), 2) if samples_cpu else None,
            "p95": _pct(samples_cpu, 0.95), "samples": len(samples_cpu)}
        result["memory_after_idle"] = _tree_memory(proc)
        result["private_mb_min_max"] = ([min(memory_series), max(memory_series)]
                                        if memory_series else None)
    finally:
        for p in reversed(_tree(proc)):
            try:
                p.kill()
            except Exception:
                pass
        try:
            child.wait(timeout=15)
        except Exception:
            pass
    return result


def scenario_startup(args: argparse.Namespace) -> dict[str, Any]:
    running = [p for p in _orion_processes()]
    if running:
        return {"error": "ORION is already running; the single-instance guard "
                         "would refuse a measured boot", "pids": running}
    work = Path(tempfile.mkdtemp(prefix="orion-perf-"))
    config_dir = work / "config"
    config_dir.mkdir()
    _dummy_provider_config(config_dir)
    runs = []
    try:
        for index in range(1 + args.warm_runs):
            kind = "cold" if index == 0 else "warm"
            run = _one_boot(config_dir, idle_s=args.idle, visible=args.visible,
                            timeout_s=args.timeout)
            run["kind"] = kind
            runs.append(run)
            time.sleep(2.0)       # let the killed tree release the mutex
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    warm = [r["ready_s"] for r in runs if r.get("kind") == "warm" and r.get("ready_s")]
    return {
        "scenario": "startup", "machine": _machine(),
        "method": ("real orion.py from this checkout; fresh temporary config; dummy "
                   "local provider; ORION_NO_CAPTURE=1; "
                   + ("visible desktop" if args.visible else "offscreen Qt")),
        "not_measured": ["microphone_ready (capture disabled)",
                         "provider_ready (no real provider)"]
                        + ([] if args.visible else ["GPU presentation / frame times"]),
        "runs": runs,
        "warm_ready_s": {"median": statistics.median(warm) if warm else None,
                         "max": max(warm) if warm else None, "n": len(warm)},
        "target_s": 5.0,
    }


def _orion_processes() -> list[int]:
    try:
        import psutil
    except Exception:
        return []
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if p.info.get("name", "").lower() == "orion.exe" or cmd.endswith("orion.py"):
                found.append(p.info["pid"])
        except Exception:
            continue
    return found


# ── shutdown ──────────────────────────────────────────────────────────────────

_MUTEX_NAME = "Global\\ORION-single-instance-9f3c"
_COMMAND_READY_RE = re.compile(r"command_ready [\d.]+s")
_REQUESTED = "perf_shutdown_requested.txt"   # written by the child when it asks
_GO = "perf_shutdown_go.txt"                 # written by the parent: ask now


def _orion_seen_by_cim() -> bool:
    """Any ORION.exe, or orion.py run as the app, per Win32_Process. psutil
    and Get-Process can miss the installed app from a sandboxed shell; CIM
    does not. Cannot tell -> True, so nothing is started on a guess."""
    query = ("(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'ORION.exe' -or "
             "($_.CommandLine -like '*orion.py*' -and $_.CommandLine -notlike '*-host*' -and "
             "$_.CommandLine -notlike '*--mcp-server*' -and $_.CommandLine -notlike "
             "'*--forge-verify*') } | Measure-Object).Count")
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
                              capture_output=True, text=True, timeout=60,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return done.stdout.strip() not in {"0"}
    except Exception:
        return True


def _orion_holds_lock() -> bool:
    """Whether some ORION is running: its single-instance mutex exists (a
    handle is opened and closed, never acquired, so a running ORION is not
    affected) or Win32_Process lists one."""
    if os.name != "nt":
        return bool(_orion_processes())
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = wintypes.HANDLE
    handle = kernel32.OpenMutexW(0x00100000, False, _MUTEX_NAME)   # SYNCHRONIZE
    if handle:
        kernel32.CloseHandle(handle)
        return True
    if ctypes.get_last_error() == 5:
        return True
    return bool(_orion_processes()) or _orion_seen_by_cim()


def _child_asks_for_shutdown(config_dir: Path) -> None:
    """In the child, on a daemon thread: once ORION reports command readiness
    and the parent says go, request a normal shutdown the way the Quit button
    does - bus.request_shutdown.emit() on the event-loop thread."""
    import asyncio
    import gc

    boot = config_dir / "diagnostics" / "last_boot.txt"
    go = config_dir / _GO
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            if go.exists() and _COMMAND_READY_RE.search(
                    boot.read_text(encoding="utf-8", errors="replace")):
                break
        except OSError:
            pass
        time.sleep(0.1)
    else:
        return
    bus = next((o for o in gc.get_objects() if type(o).__name__ == "OrionBus"), None)
    loop = next((o for o in gc.get_objects()
                 if isinstance(o, asyncio.AbstractEventLoop) and o.is_running()), None)
    if bus is None or loop is None:
        return
    (config_dir / _REQUESTED).write_text(repr(time.time()), encoding="utf-8")
    loop.call_soon_threadsafe(bus.request_shutdown.emit)


def _shutdown_child() -> int:
    """`perf_scenarios.py _shutdown-child`: the real orion.py, run as __main__,
    plus the thread above. Started only by the shutdown scenario."""
    import runpy
    import threading

    config_dir = Path(os.environ["ORION_CONFIG_DIR"])
    threading.Thread(target=_child_asks_for_shutdown, args=(config_dir,),
                     name="perf-shutdown-request", daemon=True).start()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / "orion.py")]
    runpy.run_path(str(ROOT / "orion.py"), run_name="__main__")
    return 0


def _free_port() -> int:
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _paired_phone(config_dir: Path) -> str:
    """Pair a pretend phone in the temporary config; its access token."""
    sys.path.insert(0, str(ROOT))
    from orion_core.remote import RemoteAuthManager
    auth = RemoteAuthManager(config_dir=config_dir)
    device_id, refresh = auth.complete_pairing(auth.begin_pairing(), "perf-phone")
    token, _expires = auth.issue_access(device_id, refresh)
    return token


class _PhoneStreams:
    """A paired phone on localhost: its live event stream and its uplink
    websocket, held open on a thread of their own until close()."""

    def __init__(self, port: int, token: str) -> None:
        import asyncio
        import threading
        self.port, self.token = port, token
        self.opened: list[str] = []
        self.error = ""
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._stop: Any = None
        self._thread = threading.Thread(target=self._run, name="perf-phone", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import asyncio
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._hold())

    async def _hold(self) -> None:
        import asyncio
        import aiohttp
        self._stop = asyncio.Event()
        base = f"http://127.0.0.1:{self.port}"
        headers = {"Authorization": f"Bearer {self.token}"}
        held = []
        async with aiohttp.ClientSession() as session:
            try:
                for _ in range(100):              # the gateway starts beside the boot
                    try:
                        async with session.get(f"{base}/api/health") as probe:
                            if probe.status < 500:
                                break
                    except aiohttp.ClientError:
                        await asyncio.sleep(0.2)
                events = await session.get(f"{base}/api/events?token={self.token}")
                await asyncio.wait_for(events.content.readline(), 10)
                held.append(events)
                self.opened.append("/api/events")
                uplink = await session.ws_connect(f"{base}/v1/device/uplink", headers=headers,
                                                  autoclose=False, autoping=False, heartbeat=None)
                held.append(uplink)
                self.opened.append("/v1/device/uplink")
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self._ready.set()
            await self._stop.wait()
            for item in held:
                try:
                    closing = item.close()
                    if hasattr(closing, "__await__"):
                        await asyncio.wait_for(closing, 2)
                except Exception:
                    pass

    def wait_open(self, timeout: float) -> bool:
        return self._ready.wait(timeout) and not self.error

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5)


def _one_shutdown(config_dir: Path, *, phone: bool, timeout_s: float,
                  exit_timeout_s: float) -> dict[str, Any]:
    import psutil

    diagnostics = config_dir / "diagnostics"
    for stale in (diagnostics / "last_boot.txt", diagnostics / "last_shutdown.txt",
                  config_dir / _REQUESTED, config_dir / _GO):
        if stale.exists():
            stale.unlink()
    env = _child_env(config_dir, visible=False)
    # The owner's environment turns the phone gateway on at port 8765: never
    # here, unless this run asks for it, and then on a free loopback port.
    env.update(ORION_REMOTE_ACCESS="0", ORION_SELF_IMPROVE="0")
    for name in ("ORION_REMOTE_EXPOSURE", "ORION_REMOTE_PUBLIC_URL", "ORION_REMOTE_DNS_NAME"):
        env.pop(name, None)
    port = token = None
    if phone:
        port, token = _free_port(), _paired_phone(config_dir)
        env.update(ORION_REMOTE_ACCESS="1", ORION_REMOTE_HOST="127.0.0.1",
                   ORION_REMOTE_PORT=str(port), ORION_REMOTE_TLS="0", ORION_REMOTE_FIREWALL="0")
    log = config_dir / "perf_console.txt"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8", errors="replace") as sink:
        child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "_shutdown-child"],
                                 cwd=ROOT, env=env, stdout=sink, stderr=subprocess.STDOUT)
    result: dict[str, Any] = {"phone": phone}
    streams = None
    try:
        boot = diagnostics / "last_boot.txt"
        while True:
            if child.poll() is not None:
                return {**result, "error": f"exited with {child.returncode} before ready",
                        "console_tail": log.read_text(errors="replace")[-1200:]}
            if time.perf_counter() - started > timeout_s:
                return {**result, "error": f"not command-ready after {timeout_s:.0f}s"}
            try:
                if _COMMAND_READY_RE.search(boot.read_text(encoding="utf-8", errors="replace")):
                    break
            except OSError:
                pass
            time.sleep(0.2)
        result["ready_s"] = round(time.perf_counter() - started, 2)
        time.sleep(2.0)                           # let the tail of the boot settle
        if phone:
            streams = _PhoneStreams(port, token)
            streams.wait_open(30)
            result["phone_streams_open"] = list(streams.opened)
            if streams.error:
                result["phone_error"] = streams.error
        (config_dir / _GO).write_text("go", encoding="utf-8")
        requested = config_dir / _REQUESTED
        wait_until = time.perf_counter() + 30
        while not requested.exists() and child.poll() is None:
            if time.perf_counter() > wait_until:
                return {**result, "error": "the child never asked for shutdown"}
            time.sleep(0.05)
        try:
            child.wait(timeout=exit_timeout_s)
            exited_at = time.time()
        except subprocess.TimeoutExpired:
            return {**result, "error": f"still running {exit_timeout_s:.0f}s after the request"}
        asked = float(requested.read_text(encoding="utf-8"))
        result["request_to_exit_s"] = round(exited_at - asked, 2)
        result["exit_code"] = child.returncode
        report = (diagnostics / "last_shutdown.txt")
        text = report.read_text(encoding="utf-8", errors="replace") if report.exists() else ""
        lines = text.splitlines()
        result["report_first_line"] = lines[0] if lines else "(no last_shutdown.txt)"
        result["clean"] = bool(lines) and "Clean shutdown" in lines[0] and "FORCED" not in lines[0]
        result["slowest"] = [line.strip() for line in lines[2:7]]
        result["gave_up"] = [line.strip() for line in lines if ": gave up after" in line]
        result["loop_blocked"] = [line.strip() for line in
                                  lines[lines.index("Event loop blocked during teardown:") + 1:]
                                  if line.startswith("  ")][:3] \
            if "Event loop blocked during teardown:" in lines else []
        console = log.read_text(encoding="utf-8", errors="replace")
        result["console_says_clean"] = "ORION has shutdown cleanly." in console
        result["destroyed_pending_tasks"] = console.count("Task was destroyed but it is pending")
        return result
    finally:
        if streams is not None:
            streams.close()
        if child.poll() is None:              # only ever after a failure above
            try:
                for p in reversed(_tree(psutil.Process(child.pid))):
                    p.kill()
            except Exception:
                pass
            try:
                child.wait(timeout=15)
            except Exception:
                pass


def scenario_shutdown(args: argparse.Namespace) -> dict[str, Any]:
    if _orion_holds_lock():
        return {"scenario": "shutdown",
                "error": "ORION is running (the single-instance lock is held); a measured "
                         "shutdown needs a window when ORION is closed"}
    work = Path(tempfile.mkdtemp(prefix="orion-perf-shutdown-"))
    config_dir = work / "config"
    config_dir.mkdir()
    _dummy_provider_config(config_dir)
    runs: list[dict[str, Any]] = []
    try:
        for index in range(args.runs):
            if index and _orion_holds_lock():
                runs.append({"error": "another ORION started; stopping here"})
                break
            phone = bool(args.phone_every) and index % args.phone_every == args.phone_every - 1
            run = _one_shutdown(config_dir, phone=phone, timeout_s=args.timeout,
                                exit_timeout_s=args.exit_timeout)
            run["run"] = index + 1
            runs.append(run)
            print(json.dumps(run), flush=True)
            if "error" in run:
                break
            time.sleep(1.0)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    times = [r["request_to_exit_s"] for r in runs if "request_to_exit_s" in r]
    return {
        "scenario": "shutdown", "machine": _machine(),
        "method": ("real orion.py from this checkout run as __main__; fresh temporary config; "
                   "dummy local provider; ORION_NO_CAPTURE=1; ORION_START_MUTED=1 (farewell "
                   "path kept, nothing played); offscreen Qt; shutdown requested with "
                   "bus.request_shutdown.emit() on the event-loop thread after command "
                   "readiness; time = request to process exit"),
        "not_measured": ["the goodbye's speaking time (voice muted)",
                         "a real phone over Wi-Fi/Tailscale (localhost streams only)"],
        "runs": runs,
        "clean": sum(1 for r in runs if r.get("clean")),
        "request_to_exit_s": {"median": statistics.median(times) if times else None,
                              "max": max(times) if times else None, "n": len(times)},
        "target_s": 8.0,
    }


# ── warm local routing ────────────────────────────────────────────────────────

_COMMANDS = (
    "open notepad", "what's the time", "mute", "volume up", "pause the music",
    "run diagnostics", "search the web for rust async runtimes",
    "remind me to call the dentist at 3pm", "what's the weather in Birmingham",
    "play some lo-fi", "show me the chess board", "find my tax return pdf",
    "take a screenshot", "read my latest email", "start a focus session",
    "what's on my calendar tomorrow", "translate hello into French",
    "research solid-state batteries", "set a timer for ten minutes",
    "summarise this page",
)


def scenario_local_command(args: argparse.Namespace) -> dict[str, Any]:
    """Text → routing decision, with nothing executed: the reflex fast path,
    then the tool shortlist that would be offered to a model. Execution is
    deliberately excluded — measuring it would mean performing the action."""
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.reflex import match_reflex
    from orion_core import tool_resolver

    scorer = tool_resolver.lexical_scores
    uses_meaning = False
    if not args.lexical_only:
        try:
            from orion_core import semantic as _semantic
            _semantic.ENCODER.warm(allow_download=False)   # never download here
            if tool_resolver.warm_semantic(TOOL_DECLARATIONS):
                scorer, uses_meaning = tool_resolver.hybrid_scores, True
        except Exception:
            pass
    for text in _COMMANDS:                       # warm every cache first
        match_reflex(text)
        tool_resolver.resolve(text, None, TOOL_DECLARATIONS, scorer=scorer)

    def _time(texts: list[str]) -> tuple[list[float], int]:
        timings, reflexes = [], 0
        for text in texts:
            t0 = time.perf_counter()
            if match_reflex(text) is None:
                tool_resolver.resolve(text, None, TOOL_DECLARATIONS, scorer=scorer)
            else:
                reflexes += 1
            timings.append((time.perf_counter() - t0) * 1000.0)
        return timings, reflexes

    # Repeated phrasings hit the query-embedding cache; novel ones pay for a
    # fresh encoder pass, which is what a new sentence from the user costs.
    repeated, reflexes = _time([_COMMANDS[i % len(_COMMANDS)] for i in range(args.samples)])
    novel, _ = _time([f"{_COMMANDS[i % len(_COMMANDS)]} for me, request {i}"
                      for i in range(args.samples)])
    summary = lambda t: {"p50": _pct(t, 0.5), "p95": _pct(t, 0.95),   # noqa: E731
                         "max": round(max(t), 3)}
    return {
        "scenario": "local-command", "machine": _machine(),
        "method": (f"{args.samples} routing decisions per variant over {len(_COMMANDS)} "
                   f"fixed commands after warm-up; scorer="
                   f"{'hybrid (ONNX semantic + lexical)' if uses_meaning else 'lexical'}; "
                   "no tool executed"),
        "repeated_ms": summary(repeated),
        "novel_ms": summary(novel),
        "reflex_share": round(reflexes / max(1, args.samples), 3),
        "target_p95_ms": 250.0,
    }


# ── retrieval scale ───────────────────────────────────────────────────────────

def scenario_retrieval(args: argparse.Namespace) -> dict[str, Any]:
    bench = ROOT / "tools" / "bench_retrieval.py"
    if not bench.exists():
        return {"scenario": "retrieval", "error": "tools/bench_retrieval.py is not present"}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "retrieval.json"
        done = subprocess.run([str(PYTHON), str(bench), "--out", str(out)], cwd=ROOT,
                              capture_output=True, text=True, timeout=3600)
        if done.returncode != 0 or not out.exists():
            return {"scenario": "retrieval", "error": done.stderr[-1500:]}
        return json.loads(out.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_shutdown-child"]:          # the shutdown scenario's own child
        return _shutdown_child()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="scenario", required=True)
    s = sub.add_parser("startup", help="cold + warm boots with idle sampling")
    s.add_argument("--warm-runs", type=int, default=2)
    s.add_argument("--idle", type=float, default=15.0, help="idle seconds sampled after ready")
    s.add_argument("--timeout", type=float, default=120.0)
    s.add_argument("--visible", action="store_true", help="real windows instead of offscreen")
    s.add_argument("--keep", action="store_true", help="keep the temporary config for inspection")
    s.add_argument("--out")
    c = sub.add_parser("local-command", help="warm local routing latency")
    c.add_argument("--samples", type=int, default=200)
    c.add_argument("--lexical-only", action="store_true")
    c.add_argument("--out")
    r = sub.add_parser("retrieval", help="memory retrieval at 1k/10k/50k records")
    r.add_argument("--out")
    d = sub.add_parser("shutdown", help="normal shutdowns: request to process exit")
    d.add_argument("--runs", type=int, default=20)
    d.add_argument("--phone-every", type=int, default=2,
                   help="every Nth run holds a localhost phone's streams open (0 = never)")
    d.add_argument("--timeout", type=float, default=180.0, help="seconds to command readiness")
    d.add_argument("--exit-timeout", type=float, default=40.0,
                   help="seconds from the request to process exit before it counts as hung")
    d.add_argument("--keep", action="store_true", help="keep the temporary config for inspection")
    d.add_argument("--out")
    args = parser.parse_args(argv)
    runner = {"startup": scenario_startup, "local-command": scenario_local_command,
              "retrieval": scenario_retrieval, "shutdown": scenario_shutdown}[args.scenario]
    _write(runner(args), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
