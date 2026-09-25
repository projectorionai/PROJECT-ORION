"""
Startup import budget — what may be imported before the window appears.

Importing ``orion_core.app`` used to pull in 133 of ORION's ~292 modules and
cost ~353 ms, because the composition root listed every service it would later
construct as a module-level import.  None of that is needed to put a window on
screen, so every launch waited for the antivirus scanner, the breach monitor,
the cyber curriculum and the forge before the user saw anything at all.

Those imports now live inside ``run_application()``, after the shell has
painted.  The total work is unchanged — Python caches modules either way — but
it happens while ORION is visible instead of before he exists.

This is exactly the kind of win that decays silently: one convenient
``from .forge import ...`` at the top of app.py restores the whole cost, no test
fails, and nobody notices until launch feels slow again months later.  So the
budget is asserted here.

The check runs in a SUBPROCESS on purpose.  By the time this test executes, the
pytest session has already imported most of ORION, so measuring
``sys.modules`` in-process would measure the test suite rather than a launch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Ceiling on orion_core modules loaded by ``import orion_core.app``.
#: Measured at 33 after the deferral (was 133).  The headroom absorbs ordinary
#: growth in the window's own dependencies; a jump past it means a service
#: import has been added back to the module header.
MAX_STARTUP_MODULES = 55

#: Services with no business loading before the shell is on screen.  Each was
#: verified to be imported by app.py's header before the deferral, and to be
#: used only after ``window.show()``.
MUST_NOT_LOAD = (
    "orion_core.antivirus",
    "orion_core.breach_monitor",
    "orion_core.cyber_curriculum",
    "orion_core.cyber_knowledge",
    "orion_core.browser_copilot",
    "orion_core.audio_studio",
    "orion_core.backup_manager",
    "orion_core.entertainment",
    "orion_core.forge",
    "orion_core.debugger",
    "orion_core.social_automation",
    "orion_core.security_recon",
    "orion_core.dispatcher",
)


def _startup_modules() -> set[str]:
    """orion_core modules present after a bare ``import orion_core.app``."""
    code = (
        "import sys, json; import orion_core.app; "
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith('orion_core'))))"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        pytest.fail(f"importing orion_core.app failed:\n{proc.stderr[-2000:]}")
    import json
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.fixture(scope="module")
def startup_modules() -> set[str]:
    return _startup_modules()


def test_startup_stays_under_the_module_budget(startup_modules):
    assert len(startup_modules) <= MAX_STARTUP_MODULES, (
        f"{len(startup_modules)} orion_core modules load before the window "
        f"appears (budget {MAX_STARTUP_MODULES}). A service import has "
        f"probably been added to app.py's module header — move it into the "
        f"deferred block in run_application()."
    )


@pytest.mark.parametrize("module", MUST_NOT_LOAD)
def test_heavy_services_do_not_load_before_the_window(startup_modules, module):
    assert module not in startup_modules, (
        f"{module} is imported before the shell is painted. It is only used "
        f"after window.show(), so it belongs in the deferred import block in "
        f"run_application()."
    )


def test_the_window_itself_still_loads_eagerly(startup_modules):
    """The deferral must not have pushed the window off the startup path."""
    assert "orion_core.gui.core_window" in startup_modules
    assert "orion_core.bus" in startup_modules


def test_the_deferred_block_sits_after_the_first_paint():
    """Order is the whole point: show, paint, then import.

    Asserted on the source rather than at runtime because run_application()
    needs a QApplication and a real event loop, which the suite deliberately
    does not stand up (importing app.py in-process has been observed to break
    later Qt tests).
    """
    src = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8").split("\n")

    def line_of(needle: str) -> int:
        for i, line in enumerate(src, 1):
            if needle in line:
                return i
        pytest.fail(f"could not find {needle!r} in app.py")

    show = line_of("window.show()")
    paint = line_of("await asyncio.sleep(0)")
    block = line_of("deferred service imports")
    assert show < paint < block, (
        f"expected show({show}) < paint({paint}) < deferred imports({block}); "
        f"the deferred block must come after Qt has painted the shell."
    )


def test_app_header_has_no_service_imports():
    """Guard the header itself, so a regression is caught at source level."""
    import ast

    src = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    allowed = {
        "bus", "constants", "display", "gui", "gui.core_window", "memory",
        "providers", "telemetry", "startup_budget", "dynamic_loader",
        "dependencies", "sandbox", "console_hygiene",
    }
    offenders = [
        f"line {n.lineno}: from .{n.module} import "
        + ", ".join(a.name for a in n.names)
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
        and n.level == 1
        and n.module
        and n.module not in allowed
        and n.lineno < line_number_of_run_application(tree)
    ]
    assert not offenders, (
        "module-level service imports found in app.py's header:\n  "
        + "\n  ".join(offenders)
        + "\nMove them into the deferred block inside run_application()."
    )


def line_number_of_run_application(tree) -> int:
    import ast

    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_application":
            return node.lineno
    raise AssertionError("run_application not found in app.py")
