"""
What ORION actually loaded, said out loud at boot.

ORION starts about a hundred and forty tools, a plugin registry, an audio
stack and a provider router, and until now the console said nothing about any
of it. A silent boot is indistinguishable from a broken one: when a tool was
quarantined by the forge, or a plugin was rejected, or the microphone opened
on a host API that moves no audio, the first sign was the capability simply
not working later.

So this prints a roll-call. Every line names a real thing that was really
loaded, and the source file is read from the function object rather than
guessed from the name — a tool whose handler lives somewhere unexpected says
so, which is exactly the case worth seeing.

Printing under pythonw
----------------------
ORION's shipped launcher runs under ``pythonw.exe``, which has no console and
where ``sys.stdout`` is ``None``. A bare ``print()`` there raises
``AttributeError`` on the first line of the roll-call and takes the boot down
with it. Everything here goes through :func:`say`, which survives having
nowhere to write.
"""

from __future__ import annotations

import inspect
import os
import sys
from collections import Counter
from typing import Any, Iterable

#: Printed at the start of every line so the boot log can be read at a glance,
#: and grepped. Matches the shape of the rest of ORION's console output.
TOOLS = "[Tools]"
PLUGINS = "[Plugins]"
AUDIO = "[Audio]"
CORE = "[ORION]"


def say(line: str, bus: Any = None) -> None:
    """Write one line to the console, and to the activity log if there is one.

    Never raises. Under ``pythonw`` there is no console at all and
    ``sys.stdout`` is ``None``; a boot must not die because it had nothing to
    print to.
    """
    stream = getattr(sys, "stdout", None)
    if stream is not None:
        try:
            print(line, flush=True)
        except Exception:
            pass
    if bus is not None:
        try:
            bus.log.emit(line)
        except Exception:
            pass


def _source_file(handler: Any) -> str:
    """The file a tool's handler is really defined in.

    Read from the function object, not inferred from the tool's name. A tool
    living in a module nobody expects is worth seeing in the log rather than
    being quietly mislabelled as the module it ought to be in.
    """
    function = getattr(handler, "__func__", handler)
    try:
        path = inspect.getsourcefile(function) or inspect.getfile(function)
        if path:
            return os.path.basename(path)
    except (TypeError, OSError):
        pass
    module = getattr(function, "__module__", "") or ""
    return f"{module.rsplit('.', 1)[-1]}.py" if module else "unknown"


def tool_lines(dispatcher: Any) -> list[str]:
    """One line per tool ORION can actually route to, plus a total.

    Built from ``handler_table()`` — the set that can be CALLED — rather than
    from the schema, which is the set the model is merely TOLD about. Where
    those two disagree the router is the truth, and a tool declared but
    unreachable is precisely the fault this is meant to expose.
    """
    try:
        table = dispatcher.handler_table()
    except Exception as exc:
        return [f"{TOOLS} Tool discovery FAILED: "
                f"{type(exc).__name__}: {exc}"]
    if not isinstance(table, dict):
        return [f"{TOOLS} Tool discovery returned no routing table."]

    lines = []
    for name in sorted(table):
        lines.append(f"{TOOLS} Tool loaded: {name} ({_source_file(table[name])})")
    lines.append(f"{TOOLS} Tool discovery complete: {len(table)} active.")
    return lines


def tool_summary(dispatcher: Any) -> str:
    """Where the tools came from, as one line under the roll-call.

    A hundred and forty individual lines scroll past; this is the part a
    person actually reads.
    """
    try:
        table = dispatcher.handler_table()
    except Exception:
        return ""
    counts: Counter[str] = Counter(_source_file(h) for h in table.values())
    parts = ", ".join(f"{file} {count}" for file, count in counts.most_common())
    return f"{TOOLS} Routed from: {parts}." if parts else ""


def plugin_line(registry: Any) -> str:
    """Plugins found, accepted and refused.

    The rejected count is the point. A plugin that fails its capability audit
    is silently absent otherwise, and "it stopped working" is a much harder
    thing to debug than "it was rejected at boot".
    """
    active = rejected = total = 0
    try:
        for attribute, names in (
            ("active", ("active", "loaded", "enabled")),
            ("rejected", ("rejected", "refused", "failed")),
            ("total", ("total", "discovered", "all")),
        ):
            for candidate in names:
                value = getattr(registry, candidate, None)
                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        continue
                if value is None:
                    continue
                count = len(value) if hasattr(value, "__len__") else int(value)
                if attribute == "active":
                    active = count
                elif attribute == "rejected":
                    rejected = count
                else:
                    total = count
                break
    except Exception:
        pass
    total = max(total, active + rejected)
    return (f"{PLUGINS} Plugin discovery complete: {active} active, "
            f"{rejected} rejected, {total} total.")


def audio_lines(devices: Any = None) -> list[str]:
    """Which host API the microphone and speakers actually landed on.

    ORION has been caught opening an output on a DirectSound device that
    reports success and moves no audio — a silent assistant with a healthy
    log. Naming the host API at boot makes that visible before it is puzzled
    over.
    """
    if devices is None:
        try:
            from . import audio_devices as devices  # type: ignore[no-redef]
        except Exception:
            return []
    lines: list[str] = []
    for kind in ("input", "output"):
        try:
            usable = devices.usable_devices(kind)
        except Exception:
            continue
        try:
            host = devices.preferred_host_api(kind)
        except Exception:
            host = ""
        count = len(usable) if usable is not None else 0
        if host:
            lines.append(f"{AUDIO} {kind}: using {host} ({count} device"
                         f"{'s' if count != 1 else ''})")
        else:
            lines.append(f"{AUDIO} {kind}: {count} device"
                         f"{'s' if count != 1 else ''} available")
    return lines


def report(dispatcher: Any = None, *, registry: Any = None, bus: Any = None,
           devices: Any = None, tools: bool = True) -> list[str]:
    """Print the whole roll-call. Returns the lines, for tests.

    Nothing here is allowed to stop a boot, so every section is independently
    guarded: a broken plugin registry must not cost you the tool list.
    """
    lines: list[str] = []
    if dispatcher is not None and tools:
        try:
            lines.extend(tool_lines(dispatcher))
            summary = tool_summary(dispatcher)
            if summary:
                lines.append(summary)
        except Exception as exc:
            lines.append(f"{TOOLS} roll-call failed: {type(exc).__name__}: {exc}")
    if registry is not None:
        try:
            lines.append(plugin_line(registry))
        except Exception as exc:
            lines.append(f"{PLUGINS} roll-call failed: {type(exc).__name__}: {exc}")
    try:
        lines.extend(audio_lines(devices))
    except Exception:
        pass

    for line in lines:
        say(line, bus)
    return lines


def banner(name: str = "O.R.I.O.N.", subtitle: str = "") -> list[str]:
    """The two lines before the roll-call, so a boot has a beginning."""
    lines = [f"{CORE} {name} starting…"]
    if subtitle:
        lines.append(f"{CORE} {subtitle}")
    return lines


__all__ = [
    "AUDIO", "CORE", "PLUGINS", "TOOLS",
    "audio_lines", "banner", "plugin_line", "report", "say",
    "tool_lines", "tool_summary",
]
