"""A durable scheduled updater entry point for source and frozen builds."""

from __future__ import annotations


def main() -> int:
    from .legacy_game_updater import game_updater

    try:
        result = game_updater({"action": "update", "platform": "both", "shutdown_when_done": False})
    except (OSError, ValueError, RuntimeError):
        return 1
    # GUI builds may have no stdout. A successful scheduler exit still cannot
    # mean a download finished: the launcher owns the asynchronous operation.
    import sys
    if sys.stdout is not None:
        print(result)
    return int(any(word in result.casefold() for word in ("failed", "not installed", "i found neither")))
