"""Regression coverage for quiet Sentinel process monitoring."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.sentinel import is_routine_process  # noqa: E402


def test_normal_desktop_processes_are_quiet_by_name_or_executable():
    for name in (
        "msedge.exe", "Claude.exe", "ChatGPT.exe", "obs64.exe",
        "Discord.exe", "Battle.net.exe", "C:/Program Files/Steam/steam.exe",
    ):
        assert is_routine_process(name), name


def test_unfamiliar_processes_remain_eligible_for_sentinel_attention():
    assert is_routine_process("unfamiliar-heavy-worker.exe") is False
