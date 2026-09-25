"""
Safe file-cleanup assistant for GitHub preparation (Section 9).

ORION may autonomously inspect a project and RECOMMEND unnecessary files, but it
must never delete anything without explicit, informed confirmation.  This module
provides that analysis plus a deletion path fenced by every safeguard the spec
requires:

* Detects regenerable artefacts, caches, temp/log/coverage output, IDE/OS
  metadata, local env files, oversized files, and secret-bearing files.
* Recommends ``.gitignore`` entries; secret-bearing files are excluded from Git,
  never displayed — their contents are NEVER read or printed.
* Defaults to preserving anything ambiguous, and never proposes deleting source
  code, documents, databases, migrations, credentials, config, or model files.
* Deletion requires an explicit confirmation that names the selected files (a
  token bound to the exact planned set), a dry-run preview, path-traversal and
  symlink-escape blocking, restriction to approved roots, revalidation of each
  path immediately before deletion, rejection if a file changed after approval,
  recoverable deletion (recycle bin) where possible — otherwise a second
  explicit confirmation — and an audit record.

Pure and deterministic (filesystem access aside), so the safeguards are tested
directly with temporary directories and fakes.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class RiskLevel(str, Enum):
    LOW = "low"        # regenerable, safe to remove
    MEDIUM = "medium"  # probably removable, review first
    HIGH = "high"      # sensitive or valuable — never auto-remove


class ProposedAction(str, Enum):
    SUGGEST_DELETE = "suggest_delete"
    SUGGEST_GITIGNORE = "suggest_gitignore"
    REVIEW = "review"
    PRESERVE = "preserve"


# Names/dirs that are pure regenerable artefacts → safe to suggest deleting.
_CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
               ".cache", ".tox", "htmlcov"}
_BUILD_DIRS = {"build", "dist"}
_ENV_DIRS = {".venv", "venv", "env", "node_modules"}
_IDE_DIRS = {".idea", ".vs"}
_OS_META = {".ds_store", "thumbs.db", "desktop.ini"}
_ARTEFACT_SUFFIX = {".pyc", ".pyo", ".log", ".tmp", ".temp", ".bak", ".swp",
                    ".coverage", ".orig"}
# Suffixes/names that are SENSITIVE — exclude from Git, never delete/display.
_SECRET_SUFFIX = {".pem", ".key", ".pfx", ".p12", ".keystore"}
_SECRET_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "credentials.json",
                 "service_account.json", ".env"}
_SECRET_SUBSTR = ("api_key", "apikey", "secret", "credential", "token")
# Valuable/never-auto-delete categories.
_PROTECTED_SUFFIX = {".py", ".db", ".sqlite3", ".sqlite", ".onnx", ".pt",
                     ".gguf", ".safetensors", ".bin", ".docx", ".pdf", ".xlsx"}
_PROTECTED_DIRS = {"migrations", "alembic"}

_DEFAULT_LARGE_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass
class CleanupCandidate:
    path: str                      # absolute, resolved
    relpath: str                   # relative to its approved root
    file_type: str                 # human category
    size_bytes: int
    reason: str
    risk: RiskLevel
    tracked_by_git: bool
    proposed_action: ProposedAction
    is_secret_bearing: bool = False
    gitignore_suggestion: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "relpath": self.relpath,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "reason": self.reason,
            "risk": self.risk.value,
            "tracked_by_git": self.tracked_by_git,
            "proposed_action": self.proposed_action.value,
            "is_secret_bearing": self.is_secret_bearing,
            "gitignore_suggestion": self.gitignore_suggestion,
        }


@dataclass
class CleanupReport:
    candidates: list[CleanupCandidate] = field(default_factory=list)
    gitignore_suggestions: list[str] = field(default_factory=list)

    def deletable(self) -> list[CleanupCandidate]:
        return [c for c in self.candidates
                if c.proposed_action is ProposedAction.SUGGEST_DELETE]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.as_dict() for c in self.candidates],
            "gitignore_suggestions": list(self.gitignore_suggestions),
            "summary": {
                "total": len(self.candidates),
                "deletable": len(self.deletable()),
                "secret_bearing": sum(1 for c in self.candidates if c.is_secret_bearing),
            },
        }


@dataclass
class DeletionPlan:
    token: str
    entries: list[dict[str, Any]]      # {path, size, mtime, sha_hint}
    created_at: float
    ttl: float = 300.0

    def paths(self) -> list[str]:
        return [e["path"] for e in self.entries]


@dataclass
class DeletionOutcome:
    deleted: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    recoverable: bool = True
    audit: list[dict[str, Any]] = field(default_factory=list)


class CleanupError(Exception):
    pass


def _is_within(child: Path, parent: Path) -> bool:
    """True when *child* (already realpath-resolved) is inside *parent*."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


class CleanupAssistant:
    """Analyse a project for removable files and delete only under confirmation."""

    def __init__(self, roots: list[str | Path], *,
                 git_tracked: set[str] | None = None,
                 large_file_bytes: int = _DEFAULT_LARGE_BYTES,
                 trash: Callable[[str], None] | None = None,
                 clock: Callable[[], float] | None = None,
                 logger: Callable[[str], None] | None = None,
                 audit_sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        # Approved roots are the ONLY places deletion may touch.  Resolve to real
        # paths up front so symlink escapes are caught by containment checks.
        self.roots = [Path(r).resolve() for r in roots]
        # Absolute, resolved paths Git already tracks (never suggested for delete
        # unless they are pure artefacts erroneously committed → still gitignore).
        self._git_tracked = {str(Path(p).resolve()) for p in (git_tracked or set())}
        self.large_file_bytes = int(large_file_bytes)
        self._trash = trash if trash is not None else _default_trash()
        self._clock = clock or time.time
        self._log = logger or (lambda _m: None)
        self._audit_sink = audit_sink
        self._plans: dict[str, DeletionPlan] = {}

    # ── analysis ──────────────────────────────────────────────────────────────

    def scan(self) -> CleanupReport:
        report = CleanupReport()
        gitignore: set[str] = set()
        for root in self.roots:
            if not root.exists():
                continue
            for path in self._walk(root):
                candidate = self._classify(path, root)
                if candidate is None:
                    continue
                report.candidates.append(candidate)
                if candidate.gitignore_suggestion:
                    gitignore.add(candidate.gitignore_suggestion)
        report.gitignore_suggestions = sorted(gitignore)
        return report

    def _walk(self, root: Path):
        skip = _CACHE_DIRS | _ENV_DIRS | {".git"}
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            # Yield artefact directories themselves (as a single candidate), and
            # do not descend into caches/venvs we already flag.
            pruned = []
            for d in list(dirnames):
                low = d.lower()
                if low in _CACHE_DIRS or low in _BUILD_DIRS or low in _ENV_DIRS or low in _IDE_DIRS:
                    yield here / d
                    pruned.append(d)
                elif low == ".git":
                    pruned.append(d)
            for d in pruned:
                dirnames.remove(d)
            for name in filenames:
                yield here / name

    def _tracked(self, resolved: Path) -> bool:
        return str(resolved) in self._git_tracked

    def _classify(self, path: Path, root: Path) -> CleanupCandidate | None:
        try:
            resolved = path.resolve()
        except Exception:
            return None
        name = path.name
        low = name.lower()
        suffix = path.suffix.lower()
        try:
            size = self._dir_size(path) if path.is_dir() else path.stat().st_size
        except Exception:
            size = 0
        rel = self._relpath(resolved, root)
        tracked = self._tracked(resolved)

        secret = self._looks_secret(low, suffix)
        if secret:
            # Never delete or read; recommend gitignore + review.  No content.
            return CleanupCandidate(
                path=str(resolved), relpath=rel, file_type="secret-bearing file",
                size_bytes=size,
                reason=("May contain credentials/secrets — must be excluded from Git, "
                        "not committed. Contents were not read."),
                risk=RiskLevel.HIGH, tracked_by_git=tracked,
                proposed_action=ProposedAction.SUGGEST_GITIGNORE,
                is_secret_bearing=True,
                gitignore_suggestion=self._gitignore_for(low, suffix),
            )

        if path.is_dir():
            if low in _CACHE_DIRS:
                return self._mk(resolved, rel, "cache directory", size,
                                "Regenerable cache — safe to remove.", RiskLevel.LOW,
                                tracked, ProposedAction.SUGGEST_DELETE, low + "/")
            if low in _BUILD_DIRS:
                return self._mk(resolved, rel, "build output", size,
                                "Build artefacts are regenerated by the build.", RiskLevel.LOW,
                                tracked, ProposedAction.SUGGEST_DELETE, low + "/")
            if low in _ENV_DIRS:
                return self._mk(resolved, rel, "local environment", size,
                                "Local virtualenv / dependencies — should not be in Git.",
                                RiskLevel.MEDIUM, tracked, ProposedAction.SUGGEST_GITIGNORE, low + "/")
            if low in _IDE_DIRS:
                return self._mk(resolved, rel, "IDE metadata", size,
                                "Editor metadata — usually not shared.", RiskLevel.MEDIUM,
                                tracked, ProposedAction.SUGGEST_GITIGNORE, low + "/")
            return None

        # Files.
        if low in _OS_META:
            return self._mk(resolved, rel, "OS metadata", size,
                            "Operating-system metadata — safe to remove.", RiskLevel.LOW,
                            tracked, ProposedAction.SUGGEST_DELETE, name)
        if suffix in _ARTEFACT_SUFFIX or low.endswith("~"):
            return self._mk(resolved, rel, "regenerable artefact", size,
                            f"{suffix or 'temp'} artefact — regenerated automatically.",
                            RiskLevel.LOW, tracked, ProposedAction.SUGGEST_DELETE,
                            "*" + suffix if suffix else None)
        if self._is_protected(path, suffix):
            return None  # source/db/migrations/models/docs — never suggested
        if size >= self.large_file_bytes and not tracked:
            return self._mk(resolved, rel, "large file", size,
                            f"Large ({size // (1024*1024)} MB) and untracked — may be "
                            "unsuitable for Git. Review before removing.",
                            RiskLevel.HIGH, tracked, ProposedAction.REVIEW, None)
        return None

    def _mk(self, resolved: Path, rel: str, ftype: str, size: int, reason: str,
            risk: RiskLevel, tracked: bool, action: ProposedAction,
            gitignore: str | None) -> CleanupCandidate:
        # A tracked file that is really an artefact should be un-tracked via
        # gitignore rather than silently deleted from the working tree.
        if tracked and action is ProposedAction.SUGGEST_DELETE:
            action = ProposedAction.REVIEW
            reason += " (currently tracked by Git — review before removing.)"
        return CleanupCandidate(
            path=str(resolved), relpath=rel, file_type=ftype, size_bytes=size,
            reason=reason, risk=risk, tracked_by_git=tracked,
            proposed_action=action, gitignore_suggestion=gitignore)

    @staticmethod
    def _looks_secret(low: str, suffix: str) -> bool:
        if suffix in _SECRET_SUFFIX:
            return True
        if low in _SECRET_NAMES or low.startswith(".env"):
            return True
        return any(s in low for s in _SECRET_SUBSTR)

    @staticmethod
    def _gitignore_for(low: str, suffix: str) -> str:
        if suffix in _SECRET_SUFFIX:
            return "*" + suffix
        if low.startswith(".env"):
            return ".env*"
        return low

    @staticmethod
    def _is_protected(path: Path, suffix: str) -> bool:
        if suffix in _PROTECTED_SUFFIX:
            return True
        return any(part.lower() in _PROTECTED_DIRS for part in path.parts)

    @staticmethod
    def _dir_size(path: Path) -> int:
        total = 0
        for dp, _dn, fn in os.walk(path):
            for f in fn:
                try:
                    total += (Path(dp) / f).stat().st_size
                except Exception:
                    continue
        return total

    def _relpath(self, resolved: Path, root: Path) -> str:
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            return str(resolved)

    # ── deletion (fenced) ─────────────────────────────────────────────────────

    def _validate_path(self, raw: str) -> Path:
        """Resolve *raw* and confirm it lives inside an approved root, blocking
        path traversal and symlink escapes."""
        resolved = Path(raw).resolve()
        real = Path(os.path.realpath(resolved))
        if not any(_is_within(real, r) or real == r for r in self.roots):
            raise CleanupError(f"refused: '{raw}' is outside the approved project roots")
        # A symlink whose real target escapes the roots is rejected above; a
        # symlink pointing inside is still not something we delete blindly.
        if resolved.is_symlink():
            raise CleanupError(f"refused: '{raw}' is a symlink")
        return resolved

    @staticmethod
    def _fingerprint(path: Path) -> str:
        st = path.stat()
        # size+mtime is enough to detect post-approval change without reading a
        # (possibly secret) file's contents.
        return hashlib.sha256(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:16]

    def plan_deletion(self, selected: list[str]) -> DeletionPlan:
        """Validate the explicitly-selected paths and return a plan bound to a
        single-use token.  Only paths the scan marked SUGGEST_DELETE (never
        secret-bearing, protected or merely REVIEW) are accepted."""
        if not selected:
            raise CleanupError("no files were selected for deletion")
        deletable = {c.path: c for c in self.scan().candidates
                     if c.proposed_action is ProposedAction.SUGGEST_DELETE}
        entries: list[dict[str, Any]] = []
        for raw in selected:
            resolved = self._validate_path(raw)
            key = str(resolved)
            candidate = deletable.get(key)
            if candidate is None:
                raise CleanupError(
                    f"refused: '{raw}' is not an approved deletion candidate "
                    "(it may be protected, secret-bearing, or only flagged for review)")
            if not resolved.exists():
                raise CleanupError(f"refused: '{raw}' no longer exists")
            entries.append({
                "path": key,
                "size": candidate.size_bytes,
                "fingerprint": self._fingerprint(resolved) if resolved.is_file() else "dir",
            })
        plan = DeletionPlan(token=secrets.token_urlsafe(16), entries=entries,
                            created_at=self._clock())
        self._plans[plan.token] = plan
        self._log(f"CLEANUP: deletion plan prepared for {len(entries)} item(s), awaiting confirmation.")
        return plan

    def dry_run(self, plan: DeletionPlan) -> list[dict[str, Any]]:
        """Preview exactly what a confirmed deletion would do — nothing removed."""
        return [{"path": e["path"], "size": e["size"], "action": "would delete (recoverable)"
                 if self._trash is not None else "would delete (PERMANENT — no recycle bin)"}
                for e in plan.entries]

    def confirm_and_delete(self, token: str, *, second_confirm: bool = False) -> DeletionOutcome:
        """Execute a planned deletion after the human has confirmed.

        Revalidates every path immediately before removal and rejects any that
        changed since approval.  Uses the recycle bin when available; otherwise
        requires ``second_confirm=True`` because deletion would be permanent."""
        plan = self._plans.pop(token, None)
        if plan is None:
            raise CleanupError("no matching deletion plan — it may have expired or been used")
        if self._clock() - plan.created_at > plan.ttl:
            raise CleanupError("this deletion approval has expired; please re-scan and confirm again")
        recoverable = self._trash is not None
        if not recoverable and not second_confirm:
            raise CleanupError(
                "No recycle bin is available, so deletion would be PERMANENT. "
                "Re-confirm with second_confirm=True to proceed irreversibly.")

        outcome = DeletionOutcome(recoverable=recoverable)
        for entry in plan.entries:
            raw = entry["path"]
            try:
                resolved = self._validate_path(raw)  # re-check containment/symlink
            except CleanupError as exc:
                outcome.skipped.append({"path": raw, "reason": str(exc)})
                continue
            if not resolved.exists():
                outcome.skipped.append({"path": raw, "reason": "vanished before deletion"})
                continue
            # Reject if the file changed after approval.
            current = self._fingerprint(resolved) if resolved.is_file() else "dir"
            if entry["fingerprint"] != current:
                outcome.skipped.append({"path": raw, "reason": "changed after approval — not deleted"})
                continue
            try:
                if recoverable:
                    self._trash(str(resolved))
                else:
                    self._permanent_delete(resolved)
                outcome.deleted.append(raw)
                self._emit_audit(outcome, raw, recoverable, entry["size"])
            except Exception as exc:
                outcome.skipped.append({"path": raw, "reason": f"delete failed: {exc}"})
        self._log(f"CLEANUP: removed {len(outcome.deleted)}, skipped {len(outcome.skipped)} "
                  f"({'recycle bin' if recoverable else 'permanent'}).")
        return outcome

    def cancel_plan(self, token: str) -> None:
        self._plans.pop(token, None)

    def _emit_audit(self, outcome: DeletionOutcome, path: str, recoverable: bool, size: int) -> None:
        record = {
            "at": self._clock(),
            "path": path,
            "size": size,
            "recoverable": recoverable,
            "action": "recycled" if recoverable else "permanently_deleted",
        }
        outcome.audit.append(record)
        if self._audit_sink is not None:
            try:
                self._audit_sink(record)
            except Exception:
                pass

    @staticmethod
    def _permanent_delete(path: Path) -> None:
        import shutil
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _default_trash() -> Callable[[str], None] | None:
    """Recoverable deletion via send2trash when installed; None otherwise (the
    caller then requires a second confirmation for permanent deletion)."""
    try:
        from send2trash import send2trash  # type: ignore
        return lambda p: send2trash(p)
    except Exception:
        return None
