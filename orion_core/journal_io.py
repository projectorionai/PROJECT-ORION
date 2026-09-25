"""Bounded reads of recent log lines, without loading an entire journal."""
from __future__ import annotations

from pathlib import Path


def tail_lines(path: Path, limit: int, *, max_bytes: int = 256 * 1024) -> list[str]:
    """Read complete recent lines within a byte budget; missing logs are empty.

    A single oversized line can be omitted. A partial final line is returned
    for callers to validate (for example, with json.loads).
    """
    if limit <= 0 or max_bytes <= 0:
        return []
    blocks: list[bytes] = []
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            position = stream.tell()
            remaining = max_bytes
            newlines = 0
            while position > 0 and remaining > 0 and newlines <= limit:
                size = min(position, remaining, 16 * 1024)
                position -= size
                stream.seek(position)
                block = stream.read(size)
                blocks.append(block)
                newlines += block.count(b"\n")
                remaining -= size
    except OSError:
        return []
    data = b"".join(reversed(blocks))
    if position > 0:
        # The first line may begin before the bounded read, including midway
        # through a UTF-8 codepoint. Never expose that fragment as a log entry.
        _, separator, data = data.partition(b"\n")
        if not separator:
            return []
    return [line.decode("utf-8", errors="replace") for line in data.splitlines()[-limit:]]
