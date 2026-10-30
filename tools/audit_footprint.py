"""Read-only disk inventory. Excludes links/reparse points; reads no file contents.

Run with a standard Python interpreter; no ORION or third-party imports needed.
Logical sizes are reported, not allocated/OneDrive-download sizes. Inaccessible
directories are reported explicitly so a partial scan cannot appear complete.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import stat
from collections import defaultdict
from pathlib import Path


def inventory(root: Path, top: int = 30) -> dict:
    root_info = root.lstat()
    if (stat.S_ISLNK(root_info.st_mode)
            or getattr(root_info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
        raise ValueError("Inventory root must not be a link or reparse point")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Inventory root must be a directory")
    totals: dict[str, int] = defaultdict(int)
    largest: list[tuple[int, str]] = []
    errors: list[dict[str, str]] = []
    skipped: list[str] = []
    file_count = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = Path(entry.path).relative_to(root)
                    try:
                        info = entry.stat(follow_symlinks=False)
                        if (stat.S_ISLNK(info.st_mode)
                                or getattr(info, "st_file_attributes", 0)
                                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
                            skipped.append(str(relative))
                            continue
                        if stat.S_ISDIR(info.st_mode):
                            stack.append(Path(entry.path))
                        elif stat.S_ISREG(info.st_mode):
                            size = info.st_size
                            file_count += 1
                            totals["."] += size
                            # A two-level breakdown distinguishes runtime from
                            # retained builds without an enormous output file.
                            for depth in range(1, min(len(relative.parts), 2) + 1):
                                totals[str(Path(*relative.parts[:depth]))] += size
                            item = (size, str(relative))
                            if top > 0 and len(largest) < top:
                                heapq.heappush(largest, item)
                            elif top > 0 and item > largest[0]:
                                heapq.heapreplace(largest, item)
                    except OSError as exc:
                        errors.append({"path": str(relative), "error": str(exc)})
        except OSError as exc:
            errors.append({"path": str(directory.relative_to(root)), "error": str(exc)})
    return {
        "root": str(root), "size_kind": "logical_bytes", "files": file_count,
        "total_bytes": totals.get(".", 0), "complete": not errors and not skipped,
        "subtrees": dict(sorted(totals.items(), key=lambda item: (-item[1], item[0]))),
        "largest_files": [{"bytes": size, "path": path}
                          for size, path in sorted(largest, reverse=True)],
        "errors": errors, "skipped_links_or_reparse_points": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    report = inventory(args.root, max(0, args.top))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['total_bytes'] / 1024 ** 3:.2f} GiB logical; {report['files']:,} files")
    print(f"Complete: {report['complete']}; errors: {len(report['errors'])}; "
          f"links/reparse points skipped: {len(report['skipped_links_or_reparse_points'])}")
    for name, size in report["subtrees"].items():
        if name != "." and len(Path(name).parts) == 1:
            print(f"{size / 1024 ** 3:9.3f} GiB  {name}")


if __name__ == "__main__":
    main()
