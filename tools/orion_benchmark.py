"""ORION benchmark: settled, interleaved A/B boots of two source trees.

    python tools/orion_benchmark.py --a <tree A> --b <tree B> [--b-env NAME=VALUE ...]
                                    [--boots 5] [--window 60] [--out results.json]

Why this exists alongside tools/perf_scenarios.py
-------------------------------------------------
perf_scenarios' "warm" boots reuse one temporary config but kill each run 20 s
after it reports ready. ORION's first-run work - building the 50 MiB knowledge
corpus, training the intent network, the embedding backfill - takes longer
than that, never finishes, and so restarts on every "warm" boot: those numbers
measure first-run seeding forever, not the steady state the owner lives in.

Here each tree first gets ONE settle boot that runs until its process tree has
been quiet for 10 s (capped), so first-run work completes and is cached. Then
the measured boots alternate A, B, A, B: whatever else the machine is doing
lands on both sides instead of biasing one.

Per measured boot, against a fresh process each time (same settled config):
  first_paint_s / command_ready_s   ORION's own readiness milestones
  wall_ready_s                      launch -> last_boot.txt written
  cpu_s_after_ready                 CPU seconds the whole tree burns in the
                                    first 30 s after ready (post-boot warm-up)
  idle_cpu_pct                      steady idle: % of the machine over the
                                    second 30 s
  private_mb / rss_mb / processes   at the end of the window
  loop_stalls / worst_stall_ms      event-loop freezes ORION's own watchdog
                                    logged during the boot and window
  semantic_ready_s                  when meaning-based recall/routing was ready

Isolation is perf_scenarios': fresh temporary config, a dummy local provider on
127.0.0.1:9, ORION_NO_CAPTURE=1 (no microphone/camera), ORION_START_MUTED=1,
offscreen Qt. Offscreen means GPU presentation is NOT measured, and a WebGL
face renders in software there - a face comparison made this way overstates
its CPU, and the report says so.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

PYTHON = Path(sys.executable)
SETTLE_QUIET_PCT = 1.5      # % of the machine
SETTLE_QUIET_S = 10
SETTLE_CAP_S = 300


def _helpers(tree: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"perf_{abs(hash(tree))}",
                                                  tree / "tools" / "perf_scenarios.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return []


class _CpuMeter:
    """Total CPU seconds of a process tree, including children that come and go."""

    def __init__(self, root: psutil.Process) -> None:
        self.root = root
        self.last: dict[int, float] = {}
        self.total = 0.0
        self.started = False

    def sample(self) -> float:
        """CPU seconds since the FIRST call (which is the baseline and adds
        nothing). A process that appears later was born inside the window, so
        all of its CPU counts."""
        for p in _tree(self.root):
            try:
                t = p.cpu_times()
                now = t.user + t.system
            except psutil.Error:
                continue
            previous = self.last.get(p.pid)
            if previous is None:
                self.total += now if self.started else 0.0
            else:
                self.total += max(0.0, now - previous)
            self.last[p.pid] = now
        self.started = True
        return self.total


def _kill(proc: subprocess.Popen) -> None:
    try:
        root = psutil.Process(proc.pid)
        for p in reversed(_tree(root)):
            try:
                p.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        pass


def _launch(tree: Path, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen([str(PYTHON), "orion.py"], cwd=tree, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_ready(boot_file: Path, proc: subprocess.Popen, timeout: float = 120.0) -> float:
    start = time.perf_counter()
    while not boot_file.exists():
        if proc.poll() is not None:
            raise RuntimeError(f"exited with {proc.returncode} before ready")
        if time.perf_counter() - start > timeout:
            raise RuntimeError("not ready in time")
        time.sleep(0.05)
    return time.perf_counter() - start


class Side:
    def __init__(self, name: str, tree: Path, extra_env: dict[str, str], work: Path) -> None:
        self.name, self.tree = name, tree
        self.helpers = _helpers(tree)
        self.config = work / name / "config"
        self.config.mkdir(parents=True)
        self.helpers._dummy_provider_config(self.config)
        self.env = self.helpers._child_env(self.config, False)
        self.env.update(extra_env)
        self.boot_file = self.config / "diagnostics" / "last_boot.txt"
        self.runs: list[dict[str, Any]] = []
        self.settle: dict[str, Any] = {}

    def settle_boot(self) -> None:
        self.boot_file.unlink(missing_ok=True)
        proc = _launch(self.tree, self.env)
        try:
            ready = _wait_ready(self.boot_file, proc, timeout=240)
            meter = _CpuMeter(psutil.Process(proc.pid))
            meter.sample()
            cores = psutil.cpu_count() or 1
            quiet, start, last = 0, time.perf_counter(), meter.total
            while time.perf_counter() - start < SETTLE_CAP_S and quiet < SETTLE_QUIET_S:
                time.sleep(1.0)
                total = meter.sample()
                pct = (total - last) / cores * 100.0
                last = total
                quiet = quiet + 1 if pct < SETTLE_QUIET_PCT else 0
            self.settle = {"ready_s": round(ready, 2),
                           "settled_after_s": round(time.perf_counter() - start, 1),
                           "settled": quiet >= SETTLE_QUIET_S,
                           "cpu_s_to_settle": round(meter.total, 1)}
        finally:
            _kill(proc)
            time.sleep(3)

    def _log_metrics(self, offset: int) -> dict[str, Any]:
        """What ORION's own session.log said during this boot: event-loop
        stalls its watchdog caught (count, worst, total ms) and when
        meaning-based recall became ready."""
        log = self.config / "diagnostics" / "session.log"
        try:
            with log.open("rb") as stream:
                stream.seek(offset)
                text = stream.read().decode("utf-8", errors="replace")
        except OSError:
            return {}
        stalls = [int(m) for m in re.findall(r"PERF: event loop blocked (\d+) ms", text)]
        ready = re.search(r"SEMANTIC: meaning-based recall ready .*?, (\d+) s after launch", text)
        return {"loop_stalls": len(stalls), "worst_stall_ms": max(stalls, default=0),
                "stall_ms_total": sum(stalls),
                "semantic_ready_s": float(ready.group(1)) if ready else None}

    def measured_boot(self, window: float) -> dict[str, Any]:
        self.boot_file.unlink(missing_ok=True)
        log = self.config / "diagnostics" / "session.log"
        log_offset = log.stat().st_size if log.exists() else 0
        started = time.perf_counter()
        proc = _launch(self.tree, self.env)
        run: dict[str, Any] = {}
        try:
            run["wall_ready_s"] = round(_wait_ready(self.boot_file, proc), 2)
            meter = _CpuMeter(psutil.Process(proc.pid))
            meter.sample()
            half = window / 2.0
            time.sleep(half)
            first_half = meter.sample()
            time.sleep(half)
            total = meter.sample()
            cores = psutil.cpu_count() or 1
            run["cpu_s_after_ready"] = round(first_half, 2)
            run["idle_cpu_pct"] = round((total - first_half) / half / cores * 100.0, 2)
            memory = self.helpers._tree_memory(psutil.Process(proc.pid))
            run["private_mb"] = memory.get("private_mb")
            run["rss_mb"] = memory.get("rss_mb")
            run["processes"] = sum((memory.get("processes") or {}).values())
            parsed = self.helpers._parse_boot(self.boot_file.read_text(encoding="utf-8",
                                                                      errors="replace"))
            run["first_paint_s"] = parsed["readiness_s"].get("first_paint")
            run["command_ready_s"] = parsed["readiness_s"].get("command_ready")
            run["slowest_phases"] = parsed["slowest_phases"][:4]
            run.update(self._log_metrics(log_offset))
        except Exception as exc:
            run["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            run["elapsed_s"] = round(time.perf_counter() - started, 1)
            _kill(proc)
            time.sleep(3)
        self.runs.append(run)
        return run


METRICS = ("first_paint_s", "command_ready_s", "wall_ready_s", "cpu_s_after_ready",
           "idle_cpu_pct", "private_mb", "rss_mb", "processes", "loop_stalls",
           "worst_stall_ms", "stall_ms_total", "semantic_ready_s")


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in METRICS:
        values = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        if values:
            out[key] = {"median": round(statistics.median(values), 2),
                        "min": round(min(values), 2), "max": round(max(values), 2),
                        "n": len(values)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--a-env", action="append", default=[])
    parser.add_argument("--b-env", action="append", default=[])
    parser.add_argument("--boots", type=int, default=5)
    parser.add_argument("--window", type=float, default=60.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    def env_pairs(items: list[str]) -> dict[str, str]:
        return dict(item.split("=", 1) for item in items)

    work = Path(tempfile.mkdtemp(prefix="orion-bench-"))
    try:
        sides = [Side("A", args.a.resolve(), env_pairs(args.a_env), work),
                 Side("B", args.b.resolve(), env_pairs(args.b_env), work)]
        for side in sides:
            side.settle_boot()
            print(json.dumps({"settled": side.name, **side.settle}), flush=True)
        for index in range(args.boots):
            for side in (sides if index % 2 == 0 else list(reversed(sides))):
                run = side.measured_boot(args.window)
                print(json.dumps({"boot": index + 1, "side": side.name,
                                  **{k: run.get(k) for k in (*METRICS, "error")}}), flush=True)
        result = {
            "method": __doc__.split("Per measured boot")[0].strip().splitlines()[-1],
            "machine": sides[0].helpers._machine(),
            "sides": {s.name: {"tree": str(s.tree), "env": env_pairs(
                args.a_env if s.name == "A" else args.b_env),
                "settle": s.settle, "summary": summarise(s.runs), "runs": s.runs}
                for s in sides},
            "not_measured": ["GPU presentation (offscreen Qt)", "microphone (capture off)",
                             "real AI provider latency (dummy provider)"],
        }
        if args.out:
            args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        for name in ("A", "B"):
            print(name, json.dumps(result["sides"][name]["summary"]), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
