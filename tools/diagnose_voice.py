"""Summarise an existing session log without capturing audio or calling providers.

Usage: python tools/diagnose_voice.py --log PATH [--output report.json]
Only recognised diagnostic counters leave this reader. Conversation text, device
names, filesystem paths, provider error bodies and credentials are never copied.
The report describes observations; it cannot establish audible speech quality.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path


_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+")
_STALL = re.compile(r"^PERF: event loop blocked (\d{1,9}(?:\.\d{1,3})?) ms\b")
_CACHE = re.compile(r"^VOICE: Charon offline cache - (\d{1,9}) of (\d{1,9})\b")
_MOVE = re.compile(r"^AUDIO: the desktop's sound moved from (.+) to (.+) \(Windows default output\)\.$")
_REOPEN = re.compile(r"^AUDIO: output renderer moved to (?:\[\d+\] )?(.+?)\.$")


def _bounded_lines(stream, limit=8192):
    """Discard overlong tails without allocating a whole malformed log line."""
    while True:
        head = stream.readline(limit)
        if not head:
            return
        if len(head) == limit and not head.endswith("\n"):
            while True:
                tail = stream.readline(limit)
                if not tail or tail.endswith("\n"):
                    break
        yield head


def summarise(lines):
    """Read once, retain at most 2,048 stall timings, and select the last session."""
    def fresh():
        return Counter(), deque(maxlen=2048), Counter(), None, None, 0.0

    counters, stalls, workloads, cache, pending_move, maximum = fresh()
    sessions = 0
    for raw in lines:
        # App logs are short lines. Bound parsing even on a malformed input.
        line = _STAMP.sub("", raw[:8192].rstrip("\r\n"))
        if line.startswith("===== session "):
            counters, stalls, workloads, cache, pending_move, maximum = fresh()
            sessions += 1
            continue
        counters["lines"] += 1
        stall = _STALL.match(line)
        if stall:
            duration = float(stall[1])
            maximum = max(maximum, duration)
            stalls.append(duration)
            counters["stalls"] += 1
            for name in ("intent", "semantic", "import", "local-voice"):
                if f"thread 'orion-{name}" in line:
                    workloads[name] += 1
        cache_match = _CACHE.match(line)
        if cache_match:
            cache = {"available_lines": int(cache_match[1]), "total_lines": int(cache_match[2])}
        if line.startswith("VOICE: Charon warm-up request refused - quota:"):
            counters["quota_refusals"] += 1
        if line.startswith("VOICE: the reply stopped mid-sentence"):
            counters["continuation_requests"] += 1
        if line.startswith("AUDIO: input status") and "overflow" in line:
            counters["input_overflows"] += 1
        if line.startswith("AUDIO: microphone pipeline initialised"):
            counters["microphone_starts"] += 1
        move = _MOVE.match(line)
        if move:
            counters["default_output_changes"] += 1
            pending_move = (move[1].casefold(), move[2].casefold())
        reopen = _REOPEN.match(line)
        if reopen and pending_move:
            actual = reopen[1].casefold()
            before, after = pending_move
            # MME's device labels can be truncated. Only report positive old
            # identity evidence, rather than guessing from an unknown label.
            if len(actual) >= 8 and before.startswith(actual) and not after.startswith(actual):
                counters["stale_output_reopens"] += 1
            pending_move = None
    ordered = sorted(stalls)
    p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)] if ordered else None
    findings = []
    if maximum >= 100:
        findings.append({"code": "scheduling_stalls", "detail":
                         "Scheduling stalls were recorded; compare their timing with speech and playback telemetry."})
    if counters["stale_output_reopens"]:
        findings.append({"code": "stale_default_output", "detail":
                         "A renderer reopen still named the previous Windows default output."})
    if counters["quota_refusals"]:
        findings.append({"code": "voice_quota", "detail":
                         "Charon cache warming was refused for quota; inspect the actual speech fallback separately."})
    if counters["input_overflows"]:
        findings.append({"code": "capture_overflows", "detail":
                         "The capture device reported dropped input; inspect scheduling and device transport."})
    return {"schema": 1, "scope": "latest_session", "sessions_seen": sessions,
            "lines_examined": counters["lines"],
            "stalls": {"count": counters["stalls"], "max_ms": maximum,
                       "recent_p95_ms": p95, "timings_retained": len(stalls),
                       "coincident_workers": dict(workloads)},
            "audio": {key: counters[key] for key in
                      ("input_overflows", "microphone_starts", "default_output_changes", "stale_output_reopens")},
            "speech": {"quota_refusals": counters["quota_refusals"],
                       "continuation_requests": counters["continuation_requests"], "cache": cache},
            "findings": findings,
            "limitation": "Log observations do not prove audible quality or the cause of a stutter. "
                          "No logged underflow does not establish an underflow-free session."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Existing session.log to read")
    parser.add_argument("--output", type=Path, help="New JSON report; existing files are preserved")
    args = parser.parse_args(argv)
    try:
        with args.log.open("r", encoding="utf-8-sig", errors="replace") as stream:
            result = summarise(_bounded_lines(stream))
        payload = json.dumps(result, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        else:
            print(payload, end="")
    except OSError as exc:
        # OSError messages can contain the private input path. Report its class.
        print(json.dumps({"ok": False, "error": type(exc).__name__}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
