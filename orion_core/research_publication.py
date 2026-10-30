"""Bounded exact-file publication evidence; recovery never moves or replays work."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any
import uuid

from .atomic_io import atomic_write_text

MAX_MANIFEST_BYTES = 262144
MAX_ARTIFACTS = 128
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_CURRENT: ContextVar[Any] = ContextVar("research_publication", default=None)


def manifest_name(run_id: str) -> str:
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest() + "/manifest.json"


def _ordinary(path: Path, *, directory: bool = False):
    """Refuse links/reparse points along the exact recorded path."""
    for ancestor in reversed((path, *path.parents)):
        info = ancestor.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Linked research publication path")
        if ancestor != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError("Invalid research publication ancestor")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError("Unexpected research publication path type")
    return info


def _identity(info) -> list[int]:
    return [info.st_dev, info.st_ino]


def _fingerprint(path: Path) -> dict[str, Any]:
    before = _ordinary(path)
    if before.st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Research artifact exceeds recovery limit")
    sha = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise ValueError("Research artifact changed while opening")
        while chunk := handle.read(min(1024 * 1024, before.st_size - count + 1)):
            count += len(chunk)
            if count > MAX_ARTIFACT_BYTES:
                raise ValueError("Research artifact exceeds recovery limit")
            if count > before.st_size:
                raise ValueError("Research artifact grew during verification")
            sha.update(chunk)
        after = os.fstat(handle.fileno())
    current = _ordinary(path)
    if (count != before.st_size or _identity(after) != _identity(before)
            or (current.st_size, current.st_mtime_ns, *_identity(current)) !=
               (before.st_size, before.st_mtime_ns, *_identity(before))):
        raise ValueError("Research artifact changed during verification")
    return {"identity": _identity(before), "bytes": count, "sha256": sha.hexdigest()}


def _path(raw: Any) -> Path:
    if not isinstance(raw, str) or not 0 < len(raw) <= 2048:
        raise ValueError("Invalid research publication path")
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts or str(path.absolute()) != str(path):
        raise ValueError("Noncanonical research publication path")
    return path


def _signed(data: dict[str, Any], key: str) -> str:
    return hmac.new(bytes.fromhex(key), json.dumps(data, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode("utf-8"), hashlib.sha256).hexdigest()


def _read(path: Path, run_id: str, key: str) -> dict[str, Any]:
    before = _ordinary(path)
    if before.st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Research publication manifest exceeds its limit")
    with path.open("rb") as handle:
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise ValueError("Research publication manifest changed while opening")
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    after = _ordinary(path)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Research publication manifest exceeds its limit")
    if (after.st_size, after.st_mtime_ns, *_identity(after)) != (before.st_size, before.st_mtime_ns, *_identity(before)):
        raise ValueError("Research publication manifest changed while reading")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or set(data) != {"version", "run_id", "artifacts", "signature"}:
        raise ValueError("Invalid research publication manifest")
    signature = data.pop("signature")
    if (not isinstance(signature, str) or len(signature) != 64 or
            not hmac.compare_digest(signature, _signed(data, key))):
        raise ValueError("Research publication manifest integrity mismatch")
    if (type(data["version"]) is not int or data["version"] != 1 or data["run_id"] != run_id
            or not isinstance(data["artifacts"], list) or len(data["artifacts"]) > MAX_ARTIFACTS):
        raise ValueError("Invalid research publication manifest")
    seen = set()
    for item in data["artifacts"]:
        if (not isinstance(item, dict) or set(item) != {"id", "folder", "temporary", "destination", "stage", "expected"}
                or not isinstance(item["id"], str) or len(item["id"]) != 32
                or item["id"] in seen or item["stage"] not in {"writing", "serialized", "reserved", "published"}):
            raise ValueError("Invalid research publication entry")
        seen.add(item["id"])
        folder, temporary = _path(item["folder"]), _path(item["temporary"])
        if temporary.parent != folder or not temporary.name.startswith(".paper-") or temporary.suffix != ".tmp":
            raise ValueError("Research temporary escaped its recorded folder")
        if item["destination"] is not None and _path(item["destination"]).parent != folder:
            raise ValueError("Research destination escaped its recorded folder")
        expected = item["expected"]
        if item["stage"] != "writing":
            if (not isinstance(expected, dict) or set(expected) != {"identity", "bytes", "sha256"}
                    or not isinstance(expected["identity"], list) or len(expected["identity"]) != 2
                    or any(type(v) is not int or v < 0 for v in expected["identity"])
                    or type(expected["bytes"]) is not int or not 0 <= expected["bytes"] <= MAX_ARTIFACT_BYTES
                    or not isinstance(expected["sha256"], str) or len(expected["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in expected["sha256"])):
                raise ValueError("Invalid research artifact fingerprint")
    return data


class PublicationJournal:
    def __init__(self, path: Path, run_id: str, key: str, guard: Any, lock: threading.RLock):
        self.path, self.run_id, self.guard, self.lock = path, run_id, guard, lock
        self._key = key
        self._last_signature = None
        self.observed_paths: list[str] = []
        self.data = {"version": 1, "run_id": run_id, "artifacts": []}
        if path.exists():
            self.data = _read(path, run_id, key)
            self._last_signature = _signed(self.data, key)

    def _save(self):
        self.guard()
        existing = self.path.parent
        while not existing.exists():
            existing = existing.parent
        _ordinary(existing, directory=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _ordinary(self.path.parent, directory=True)
        text = json.dumps({**self.data, "signature": _signed(self.data, self._key)}, ensure_ascii=False)
        if len(text.encode("utf-8")) > MAX_MANIFEST_BYTES:
            raise ValueError("Research publication manifest exceeds its limit")
        if self.path.exists():
            _ordinary(self.path)
            previous = _read(self.path, self.run_id, self._key)
            if self._last_signature is None or _signed(previous, self._key) != self._last_signature:
                raise OSError("Research publication manifest changed outside its writer")
        elif self._last_signature is not None:
            raise OSError("Research publication manifest was removed outside its writer")
        atomic_write_text(self.path, text, encoding="utf-8")
        self._last_signature = _signed(self.data, self._key)

    def begin(self, folder: Path, temporary: Path) -> str:
        with self.lock:
            if len(self.data["artifacts"]) >= MAX_ARTIFACTS:
                raise ValueError("Research run publication limit reached")
            _ordinary(folder, directory=True)
            _ordinary(temporary)
            key = uuid.uuid4().hex
            self.data["artifacts"].append({"id": key, "folder": str(folder),
                "temporary": str(temporary), "destination": None, "stage": "writing", "expected": None})
            self._save()
            return key

    def stage(self, key: str, stage: str, *, temporary: Path | None = None, destination: Path | None = None):
        with self.lock:
            item = next(row for row in self.data["artifacts"] if row["id"] == key)
            if temporary is not None:
                item["expected"] = _fingerprint(temporary)
            if destination is not None:
                item["destination"] = str(destination)
            item["stage"] = stage
            self._save()

    def observed(self, path: Path) -> None:
        """A completed rename is useful even if its final receipt save fails."""
        with self.lock:
            value = str(path.absolute())
            if value not in self.observed_paths:
                self.observed_paths.append(value)


@contextmanager
def scope(journal: PublicationJournal | None):
    token = _CURRENT.set(journal)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current() -> PublicationJournal | None:
    return _CURRENT.get()


def reconcile(path: Path, run_id: str, *, key: str, budget: list[int] | None = None) -> dict[str, Any]:
    """Inspect only known files, with bounded reads; never rename or delete."""
    if not path.exists():
        return {"paths": [], "published_count": 0, "serialized_partial_count": 0, "uncertain_count": 0}
    data = _read(path, run_id, key)
    paths = []
    published = partial = uncertain = 0
    budget = budget if budget is not None else [128 * 1024 * 1024]
    limited = False
    for item in data["artifacts"]:
        expected = item["expected"]
        if item["stage"] == "writing" or expected is None:
            uncertain += 1
            continue
        candidate = item["destination"] if item["stage"] in {"reserved", "published"} else None
        # A rename preserves the temporary file identity. A populated/changed
        # reservation is never accepted merely because it occupies the name.
        choices = [(candidate, True), (item["temporary"], False)]
        found = False
        for raw, final in choices:
            if not raw:
                continue
            file = _path(raw)
            try:
                info = _ordinary(file)
                if _identity(info) != expected["identity"] or info.st_size != expected["bytes"]:
                    continue
                if info.st_size > budget[0]:
                    limited = True
                    continue
                budget[0] -= info.st_size
                if _fingerprint(file) != expected:
                    continue
            except (OSError, ValueError):
                continue
            if expected["bytes"] == 0:
                continue
            paths.append(str(file))
            published += int(final)
            partial += int(not final)
            found = True
            break
        if not found:
            uncertain += 1
    return {"paths": paths, "published_count": published,
            "serialized_partial_count": partial, "uncertain_count": uncertain,
            "verification_limited": limited}
