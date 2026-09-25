"""
requirements.txt must describe reality.

Dependency lists rot quietly. A package gets added for a feature, the feature is
rewritten, and the line stays — carrying a comment that is now simply untrue.
Nothing fails, the install just gets heavier and the next person reads a
confident explanation that is wrong.

The audit that produced this file found exactly that: ``matplotlib`` was
declared as "charts", while ``reports.py`` states in its own module docstring
that visualisation "needs no matplotlib and works entirely offline" and renders
pure-Python inline SVG. It was really a **test fixture** — used once, to build a
valid PDF for the flashcard-reader test, with a graceful skip if absent. The
package was fine; the description was a lie.

So the rule enforced here is not "no unused packages" — several are legitimately
transitive (``librosa`` for resemblyzer, ``webrtcvad-wheels`` for its prebuilt
wheel, ``onnxruntime`` pulled by rapidocr). The rule is: **every declared package
is either imported somewhere, or carries a comment saying why it is there.**

Module names come from installed distribution metadata rather than a
hand-maintained alias map, because a hand-maintained map is the same kind of
thing that rots. ``pywin32`` really does provide ``win32com`` and ``pythoncom``;
guessing "pywin32 -> win32" is how an audit reports a false positive.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIREMENTS = ROOT / "requirements.txt"

#: Where a dependency could legitimately be used from.
SEARCH_ROOTS = (ROOT / "orion_core", ROOT / "tests", ROOT / "config" / "custom_tools")
SEARCH_FILES = (ROOT / "orion.py",)


def _declared() -> list[tuple[str, str, int]]:
    """(package, its explanation, line number) for each requirement.

    The explanation is the inline comment OR an immediately preceding comment
    block — the file uses both, and a longer rationale (why webrtcvad-wheels
    rather than webrtcvad) legitimately does not fit on one line. A test that
    only accepted inline comments would be enforcing a style, not honesty.
    """
    out: list[tuple[str, str, int]] = []
    preceding: list[str] = []
    for number, raw in enumerate(REQUIREMENTS.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            preceding = []
            continue
        if line.startswith("#"):
            preceding.append(line.lstrip("# ").strip())
            continue
        if line.startswith("-"):
            continue
        spec, _, comment = line.partition("#")
        name = re.split(r"[<>=!\[;]", spec.strip())[0].strip()
        if name:
            explanation = comment.strip() or " ".join(preceding)
            out.append((name, explanation, number))
        preceding = []
    return out


_DISTRIBUTION_MODULES: dict[str, set[str]] | None = None


def _modules_of(package: str) -> set[str]:
    """The top-level modules a distribution actually installs.

    packages_distributions() walks every installed distribution, so it is built
    once and inverted — calling it per package took a minute on a 2.4 GB
    site-packages.
    """
    global _DISTRIBUTION_MODULES
    if _DISTRIBUTION_MODULES is None:
        import importlib.metadata as metadata
        table: dict[str, set[str]] = {}
        for module, distributions in metadata.packages_distributions().items():
            if module.startswith("_") or module == "__pycache__":
                continue
            for dist in distributions:
                table.setdefault(dist.lower(), set()).add(module)
        _DISTRIBUTION_MODULES = table
    return (_DISTRIBUTION_MODULES.get(package.lower())
            or {package.lower().replace("-", "_")})


def _corpus() -> str:
    parts: list[str] = []
    for root in SEARCH_ROOTS:
        if root.exists():
            for path in root.rglob("*.py"):
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
    for path in SEARCH_FILES:
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


CORPUS = _corpus()


def _is_imported(package: str) -> bool:
    for module in _modules_of(package):
        if re.search(r"(?:^|\n)\s*(?:import|from)\s+" + re.escape(module) + r"\b", CORPUS):
            return True
        # lazily resolved imports count too
        if re.search(r"(?:import_module|find_spec)\(\s*[\"']" + re.escape(module) + r"\b",
                     CORPUS):
            return True
        if re.search(r"LazyModule\(\s*[\"']" + re.escape(module) + r"\b", CORPUS):
            return True
    return False


# ── the file parses ─────────────────────────────────────────────────────────

def test_requirements_exists_and_declares_packages():
    assert REQUIREMENTS.is_file()
    assert len(_declared()) >= 20, "the requirements file looks truncated"


def test_no_package_is_declared_twice():
    names = [p.lower() for p, _c, _n in _declared()]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, "declared more than once: %s" % sorted(duplicates)


# ── the rule ────────────────────────────────────────────────────────────────

def test_every_dependency_is_used_or_explained():
    """Not "no unused packages" — several are legitimately transitive. The rule
    is that a package nobody imports must say why it is there."""
    unexplained = []
    for package, explanation, number in _declared():
        if _is_imported(package):
            continue
        if explanation:
            continue
        unexplained.append("%s (line %d)" % (package, number))
    assert not unexplained, (
        "these packages are never imported and carry no comment explaining why "
        "they are installed — either drop them or say what they are for: "
        + ", ".join(unexplained))


def test_the_check_can_actually_tell_used_from_unused():
    """Guards the guard. If _is_imported returned True for everything the test
    above would pass vacuously."""
    assert _is_imported("PyQt6"), "a package ORION plainly imports read as unused"
    assert not _is_imported("orion-nonexistent-package-xyz")


def test_module_names_come_from_real_metadata_not_guesswork():
    """A hand-written alias map is the same kind of thing that rots. Guessing
    'pywin32 -> win32' is how the first pass of this audit produced a false
    positive: the real modules are win32com and pythoncom."""
    modules = _modules_of("pywin32")
    if modules == {"pywin32"}:
        pytest.skip("pywin32 metadata unavailable on this platform")
    assert "win32com" in modules or "pythoncom" in modules


# ── the specific thing that prompted this file ──────────────────────────────

def test_matplotlib_is_described_as_what_it_actually_is():
    """It was declared as "charts" while reports.py says visualisation needs no
    plotting library. It is a test fixture, and the file now says so."""
    declared = {p.lower(): c for p, c, _n in _declared()}
    if "matplotlib" not in declared:
        pytest.skip("matplotlib is no longer declared")
    comment = declared["matplotlib"].lower()
    assert "test" in comment, (
        "matplotlib's comment does not say it is a test-only fixture: %r" % comment)


def test_orion_itself_never_imports_matplotlib():
    """The claim the comment now makes has to stay true."""
    offenders = []
    for path in (ROOT / "orion_core").rglob("*.py"):
        src = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"(?:^|\n)\s*(?:import|from)\s+matplotlib\b", src):
            offenders.append(path.name)
    assert not offenders, (
        "reports.py claims charts need no plotting library, but these import "
        "matplotlib: " + ", ".join(offenders))


def test_the_report_renderer_still_has_no_plotting_dependency():
    src = (ROOT / "orion_core" / "reports.py").read_text(encoding="utf-8")
    assert "svg" in src.lower(), "the pure-Python SVG path has gone"


# ── heavy packages must stay off the startup path ───────────────────────────

HEAVY = ("torch", "sklearn", "matplotlib", "mediapipe", "resemblyzer", "whisper")


def test_no_heavy_package_is_imported_at_module_scope():
    """Startup was measured at 466 ms with only numpy pulled in. A heavy import
    added at module scope would undo that silently."""
    offenders = []
    for path in (ROOT / "orion_core").rglob("*.py"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line[0] in " \t":
                continue                     # indented == inside a function
            for package in HEAVY:
                if re.match(r"(?:import|from)\s+" + package + r"\b", line):
                    offenders.append("%s: %s" % (path.name, line.strip()))
    assert not offenders, (
        "these put a heavy package on ORION's startup path: " + "; ".join(offenders))
