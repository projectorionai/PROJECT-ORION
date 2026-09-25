"""
The suite must not be able to stop existing.

What happened
-------------
tools/prepare_public_source.py wrote a publication export to
release/validation-2026-09-19/ORION/, which contains a complete copy of tests/.
pytest then tried to import two different files as `test_memory.py`, hit the
import-file-mismatch error, and aborted:

    !!!!!!!! Interrupted: 626 errors during collection !!!!!!!!
    626 errors in 15.61s

Zero tests ran. Not one failure was reported, because nothing was collected to
fail — the safety net did not go red, it went away. Anything committed in that
window would have looked exactly as green as a clean run.

The export was correctly gitignored, which is why this was easy to miss:
.gitignore governs what git tracks and has no bearing whatever on what pytest
walks.

These tests are cheap and exist only so that a generated copy of the project
can never again silently delete the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ignored() -> list[str]:
    import conftest

    return list(getattr(conftest, "collect_ignore_glob", []))


@pytest.mark.parametrize("directory", ["release", "dist", "build", ".venv"])
def test_generated_copies_of_the_project_are_not_collected(directory):
    """Each of these can hold a full tests/ tree with the same basenames."""
    assert f"{directory}/*" in _ignored(), (
        f"{directory}/ is not excluded from collection; a copy of the project "
        f"landing there will abort the whole run"
    )


def test_the_export_tool_writes_somewhere_that_is_excluded():
    """Pin the two together. If the export path moves and the ignore list does
    not follow, the suite dies again — and dies quietly."""
    source = (ROOT / "tools" / "prepare_public_source.py").read_text(encoding="utf-8")
    if "release" not in source:
        pytest.skip("the export tool no longer mentions a release directory")
    assert "release/*" in _ignored()


def test_a_real_test_tree_is_still_collected():
    """The exclusions must not be so broad that they take the suite with them."""
    assert (ROOT / "tests").is_dir()
    collected = list((ROOT / "tests").glob("test_*.py"))
    assert len(collected) > 100, f"only {len(collected)} test modules found"


def test_no_stray_copy_of_the_suite_exists_right_now():
    """A direct check of the failure itself, rather than only of the guard.

    Walks for a second tests/ directory containing this very file's siblings.
    Cheap: it stops at the first offender and skips the excluded roots.
    """
    excluded = {"release", "dist", "build", ".venv", ".git", "node_modules",
                # ORION writes whole projects of his own here, each with its
                # own tests; conftest excludes them from collection and so
                # must this.
                "projects",
                "android", "__pycache__"}
    duplicates = []
    for candidate in ROOT.rglob("tests"):
        if not candidate.is_dir() or candidate == ROOT / "tests":
            continue
        if any(part in excluded for part in candidate.relative_to(ROOT).parts):
            continue
        if any(candidate.glob("test_*.py")):
            duplicates.append(str(candidate.relative_to(ROOT)))
    assert not duplicates, (
        "a second copy of the test suite is present and not excluded: "
        + ", ".join(duplicates[:3])
    )
