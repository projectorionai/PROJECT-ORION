"""Retire exact obsolete tasks while ORION is closed.

Dry-run by default. --apply persists local removal; --archive-notion also
archives exact matching remote tasks. No API credential is printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace


class QuietSignal:
    def emit(self, *args):
        pass


async def run(args):
    from orion_core.cognition import CognitiveStateManager
    from orion_core.constants import API_CONFIG_PATH
    from orion_core.notion import NotionService

    bus = SimpleNamespace(log=QuietSignal(), dashboard_event=QuietSignal())
    state = CognitiveStateManager(bus, SimpleNamespace())
    with state._lock:
        state._load_locked()
    notion = None
    if args.archive_notion:
        from orion_core.key_seal import load_json
        config = load_json(API_CONFIG_PATH)
        notion = NotionService(bus, (config.get("integrations") or {}).get("notion"))
    results = []
    for title in args.title:
        row = {"title": title, "local": "planned removal"}
        if args.apply:
            state.dismiss_task(title)
            row["local"] = "dismissed; persists across restarts"
            if notion:
                method = notion.archive_overdue_tasks_exact if args.overdue_only else notion.archive_task_exact
                result = await method(title)
                row["remote"] = result.text
                row["remote_ok"] = result.ok
        results.append(row)
    print(json.dumps(results, indent=2))
    return 1 if any(r.get("remote_ok") is False for r in results) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", action="append", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--archive-notion", action="store_true")
    parser.add_argument("--overdue-only", action="store_true", help="Retire duplicate exact titles only before now")
    args = parser.parse_args()
    if args.archive_notion and not args.apply:
        parser.error("--archive-notion requires --apply")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
