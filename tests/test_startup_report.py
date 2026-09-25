"""
The boot roll-call.

ORION starts about a hundred and forty tools, a plugin registry and an audio
stack, and said nothing about any of it. A silent boot is indistinguishable
from a broken one: a tool the forge quarantined, a plugin rejected by its
capability audit, or a microphone opened on a host API that moves no audio all
showed up first as a capability that mysteriously did not work later.

Two things here are load-bearing beyond the formatting. The source file for
each tool is read from the function object rather than inferred from the name,
so a handler living somewhere unexpected says so. And nothing in the roll-call
may stop a boot — including the case where there is no console to print to,
which is how ORION's shipped launcher actually runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.startup_report import (  # noqa: E402
    AUDIO,
    PLUGINS,
    TOOLS,
    audio_lines,
    banner,
    plugin_line,
    report,
    say,
    tool_lines,
    tool_summary,
)


class _Dispatcher:
    """A dispatcher with a routing table and nothing else."""

    def __init__(self, table: dict | None = None) -> None:
        self._table = {"web_search": self.web_search,
                       "flight_search": self.flight_search} \
            if table is None else table

    def handler_table(self) -> dict:
        return self._table

    def web_search(self, args):  # pragma: no cover - never called
        return None

    def flight_search(self, args):  # pragma: no cover - never called
        return None


# ── the tool roll-call ────────────────────────────────────────────────────────

def test_every_tool_gets_a_line():
    lines = tool_lines(_Dispatcher())
    assert any("web_search" in line for line in lines)
    assert any("flight_search" in line for line in lines)


def test_the_line_names_the_file_the_handler_is_really_in():
    """Read from the function object, not inferred from the tool's name.

    A tool whose handler lives in an unexpected module is worth seeing in the
    log rather than being quietly mislabelled as the module it ought to be in.
    """
    line = next(line for line in tool_lines(_Dispatcher())
                if "web_search" in line)
    assert "test_startup_report.py" in line, (
        f"the file was guessed, not read: {line}")


def test_it_counts_what_it_listed():
    lines = tool_lines(_Dispatcher())
    listed = sum(1 for line in lines if "Tool loaded:" in line)
    assert f"{TOOLS} Tool discovery complete: {listed} active." == lines[-1]


def test_the_lines_are_ordered():
    """A roll-call you have to search is not a roll-call."""
    names = [line.split("Tool loaded: ")[1].split(" (")[0]
             for line in tool_lines(_Dispatcher()) if "Tool loaded:" in line]
    assert names == sorted(names)


def test_a_dispatcher_that_raises_is_reported_not_fatal():
    class _Broken:
        def handler_table(self):
            raise RuntimeError("routing table is not built yet")

    lines = tool_lines(_Broken())
    assert len(lines) == 1 and "FAILED" in lines[0]


def test_a_dispatcher_with_no_table_is_reported():
    class _Empty:
        def handler_table(self):
            return None

    assert "no routing table" in tool_lines(_Empty())[0]


def test_the_summary_says_where_the_tools_came_from():
    """A hundred and forty individual lines scroll past; this is the line a
    person actually reads."""
    summary = tool_summary(_Dispatcher())
    assert "test_startup_report.py 2" in summary


# ── plugins ───────────────────────────────────────────────────────────────────

def test_the_plugin_line_reports_all_three_counts():
    class _Registry:
        active = [1, 2]
        rejected = [3]
        total = 3

    line = plugin_line(_Registry())
    assert line == (f"{PLUGINS} Plugin discovery complete: 2 active, "
                    "1 rejected, 3 total.")


def test_rejected_plugins_are_counted_in_the_total():
    """The rejected count is the point. A plugin that fails its capability
    audit is silently absent otherwise, and "it stopped working" is far harder
    to debug than "it was rejected at boot"."""
    class _Registry:
        active = [1]
        rejected = [2, 3]

    assert "1 active, 2 rejected, 3 total" in plugin_line(_Registry())


def test_a_registry_it_cannot_read_still_produces_a_line():
    class _Hostile:
        def __getattr__(self, name):
            raise RuntimeError("no")

    assert "Plugin discovery complete" in plugin_line(_Hostile())


# ── audio ─────────────────────────────────────────────────────────────────────

def test_audio_names_the_host_api():
    """ORION has opened an output on a DirectSound device that reports
    success and moves no audio — a silent assistant with a healthy log."""
    class _Devices:
        @staticmethod
        def usable_devices(kind):
            return [1, 2, 3] if kind == "input" else [1]

        @staticmethod
        def preferred_host_api(kind):
            return "wasapi"

    lines = audio_lines(_Devices())
    assert f"{AUDIO} input: using wasapi (3 devices)" in lines
    assert f"{AUDIO} output: using wasapi (1 device)" in lines, (
        "one device should not be called '1 devices'")


def test_audio_that_cannot_be_probed_is_simply_absent():
    class _Devices:
        @staticmethod
        def usable_devices(kind):
            raise OSError("PortAudio is not initialised")

    assert audio_lines(_Devices()) == []


# ── nothing here may stop a boot ──────────────────────────────────────────────

def test_printing_survives_having_nowhere_to_print(monkeypatch):
    """The trap this module exists to avoid.

    ORION's shipped launcher runs under pythonw.exe, which has no console and
    where sys.stdout is None. A bare print() raises AttributeError on the
    first line of the roll-call and takes the whole boot down with it.
    """
    monkeypatch.setattr(sys, "stdout", None)
    say("this must not raise")          # the assertion is that it returns


def test_printing_survives_a_stdout_that_raises(monkeypatch):
    class _Broken:
        def write(self, _text):
            raise OSError("the pipe is closed")

        def flush(self):
            raise OSError("the pipe is closed")

    monkeypatch.setattr(sys, "stdout", _Broken())
    say("this must not raise either")


def test_a_broken_bus_does_not_stop_the_console_line(capsys):
    class _Bus:
        class log:
            @staticmethod
            def emit(_line):
                raise RuntimeError("the bus is gone")

    say("still printed", _Bus())
    assert "still printed" in capsys.readouterr().out


def test_a_broken_plugin_registry_does_not_cost_the_tool_list():
    """Every section is independently guarded. Losing the plugin line must
    not lose the hundred and forty lines above it."""
    class _Hostile:
        def __getattr__(self, name):
            raise RuntimeError("no")

    lines = report(_Dispatcher(), registry=_Hostile())
    assert any("web_search" in line for line in lines)


def test_report_returns_what_it_printed(capsys):
    lines = report(_Dispatcher())
    printed = capsys.readouterr().out
    for line in lines:
        assert line in printed


def test_the_banner_opens_the_log():
    assert banner()[0].startswith("[ORION]")
    assert "subtitle" in banner("X", "subtitle")[1]


# ── it is actually wired in ───────────────────────────────────────────────────

def test_the_roll_call_runs_at_boot():
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "from .startup_report import report" in source
    assert "report(dispatcher, bus=bus)" in source


def test_the_plugin_line_is_printed_when_loading_finishes():
    """It cannot be printed with the tool list — plugins load in the
    background and their counts are not true yet at that point."""
    source = (ROOT / "orion_core" / "plugin_manifest.py").read_text(
        encoding="utf-8")
    assert "Plugin discovery complete" in source
    assert "loadable" in source and "skipped" in source


def test_the_real_dispatcher_produces_a_real_roll_call():
    """The one test that would catch handler_table changing shape."""
    import inspect

    from orion_core.bus import OrionBus
    from orion_core.dispatcher import OrionDispatcher

    pytest.importorskip("PyQt6")
    names = [p for p in inspect.signature(OrionDispatcher.__init__).parameters
             if p != "self"]
    dispatcher = OrionDispatcher(
        **{name: (OrionBus() if name == "bus" else None) for name in names})
    lines = tool_lines(dispatcher)
    assert len(lines) > 100, "ORION routes well over a hundred tools"
    assert all(line.startswith(TOOLS) for line in lines)
    assert "dispatch_" in tool_summary(dispatcher)
