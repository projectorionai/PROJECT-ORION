"""
Build the local-brain training set (stage 3, step 1).

    python -m tools.train_build_dataset --dry-run
    python -m tools.train_build_dataset --dry-run --config-dir C:\\Users\\you\\ORION\\config
    python -m tools.train_build_dataset --config-dir C:\\Users\\you\\ORION\\config

``--dry-run`` reads ORION's stores (read-only) and reports counts - how many
turns each source yields, what was dropped and why, how many tools are covered
- and writes NOTHING. Without it, the dataset is written to
``<training root>/ds-<time>-<id>/`` (train.jsonl, val.jsonl, preferences.jsonl,
manifest.json) and ``latest.txt`` is updated.

The training root is ``<config dir>/training`` - unless that is inside the
project folder or a OneDrive folder (a source checkout's config folder is),
in which case it is ``~/ORION/training``. ``--out-dir`` or ORION_TRAINING_DIR
choose another place; one inside the project or a synced folder is refused.

The report never prints any of the owner's text: counts and names of tools
only. The dataset itself is private - see docs/LOCAL_BRAIN_TRAINING.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion_core.training.dataset import ALL_SOURCES, BuildConfig, build  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.train_build_dataset",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--config-dir", default=None,
                        help="ORION's config directory to mine, read-only (default: the one this "
                             "checkout uses; the installed app's is ~/ORION/config)")
    parser.add_argument("--out-dir", default=None, help="training root to write under")
    parser.add_argument("--dry-run", action="store_true", help="report counts only; write nothing")
    parser.add_argument("--sources", default=",".join(ALL_SOURCES),
                        help="comma list of: " + ", ".join(ALL_SOURCES))
    parser.add_argument("--routing-split", choices=("dev", "all"), default="dev",
                        help="dev: train on the routing sets' development half only (recommended; "
                             "the held-out half stays an honest score). all: train on every case "
                             "- the routing score is then contaminated and the manifest says so")
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--max-per-tool", type=int, default=24)
    parser.add_argument("--augment", type=int, default=2,
                        help="extra phrasings per routing phrase (politeness, casing)")
    parser.add_argument("--max-tools", type=int, default=8)
    parser.add_argument("--tools-char-budget", type=int, default=4200,
                        help="schema characters offered per example (a full 8-tool shortlist is "
                             "~7,000 and would not fit a 2048-token sequence)")
    parser.add_argument("--max-tokens", type=int, default=1900,
                        help="drop examples estimated longer than this")
    parser.add_argument("--no-generic-args", action="store_true",
                        help="only use argument rules written for a specific tool")
    parser.add_argument("--redact-term", action="append", default=[],
                        help="a name or word to replace with <USER> (repeatable): a partner, an "
                             "employer. The operating-system user name is always redacted.")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--allow-synced-output", action="store_true",
                        help="write even inside the project or a synced folder (not recommended)")
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    return parser


def summary(manifest: dict) -> str:
    c = manifest["counts"]
    lines = [
        ("DRY RUN - nothing was written." if manifest.get("dry_run") else "Dataset written."),
        f"rows {c['rows']} (train {c['train']}, val {c['val']}); preference pairs {c['preferences']}; "
        f"candidates collected {c['candidates']}; tools covered {c['tools_covered']}",
        "by source: " + ", ".join(f"{k} {v}" for k, v in sorted(c["by_source"].items())),
        "by kind:   " + ", ".join(f"{k} {v}" for k, v in sorted(c["by_kind"].items())),
        "by risk:   " + ", ".join(f"{k} {v}" for k, v in sorted(c["by_risk"].items())),
        "arguments: " + ", ".join(f"{k} {v}" for k, v in sorted(c["by_args"].items())),
        "dropped:   " + (", ".join(f"{k} {v}" for k, v in sorted(manifest["dropped"].items())) or "none"),
        "skipped:   " + (", ".join(f"{k} {v}" for k, v in sorted(manifest["skipped"].items())) or "none"),
        "redacted:  " + (", ".join(f"{k} {v}" for k, v in sorted(manifest["redactions"].items())) or "nothing"),
        f"evaluation guard: {manifest['evaluation_guard'].get('strings', 0)} evaluation sentences "
        f"checked; missing: {manifest['evaluation_guard'].get('missing') or 'none'}",
        f"routing split: {manifest['routing_split']}"
        + ("  (CONTAMINATED: the routing score will be inflated)" if manifest["contaminated_routing"] else ""),
        f"stores: " + ", ".join(f"{k} {'%d rows' % v['rows'] if v['found'] else 'not found'}"
                                for k, v in manifest.get("stores", {}).items()),
        f"estimated tokens per row: mean {manifest['tokens_estimate']['mean']}, "
        f"max {manifest['tokens_estimate']['max']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chosen = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    unknown = set(chosen) - set(ALL_SOURCES)
    if unknown:
        print(f"Unknown source(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(ALL_SOURCES)}",
              file=sys.stderr)
        return 2
    config = BuildConfig(
        config_dir=Path(args.config_dir) if args.config_dir else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        project_dir=ROOT, sources=chosen, routing_split=args.routing_split,
        val_fraction=args.val_fraction, max_per_tool=args.max_per_tool, augment=args.augment,
        max_tools=args.max_tools, tools_char_budget=args.tools_char_budget,
        max_tokens_estimate=args.max_tokens, generic_args=not args.no_generic_args,
        seed=args.seed, redact_terms=tuple(args.redact_term), dry_run=args.dry_run,
        allow_synced_output=args.allow_synced_output)
    try:
        result = build(config)
    except ValueError as refused:
        print(f"Refused: {refused}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.manifest, indent=2, sort_keys=True))
    else:
        print(summary(result.manifest))
        if result.path is not None:
            print(f"Written: {result.path}")
            print("Next: python -m tools.train_lora --dataset " + str(result.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
