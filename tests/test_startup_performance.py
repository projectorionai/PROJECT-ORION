"""
Startup latency policy (Mark XXVI) — why ORION's window appears quickly.

Importing ``orion_core.app`` measured **1780 ms** before this pass, and every
millisecond was spent before the window existed. Three third-party packages were
78% of it:

    google.genai   ~1280 ms   (live_worker)
    aiohttp         ~178 ms   (utils, providers and 9 others)
    sounddevice     ~162 ms   (audio)

None of them is touched at import time, so all three are now behind the proxies
in ``orion_core.lazy_import``. Measured after: **~465 ms — 3.8x faster.**

The second half of the policy matters as much as the first. Deferring an import
MOVES its cost; ``google.genai`` is first needed inside ``connect()``, which runs
on the qasync loop, so a naive deferral would have traded a slow start for a
freeze mid-conversation. ``warm()`` imports all three on a daemon thread the
moment the window is visible, so no foreground path ever pays.

These tests defend both halves: the imports stay deferred, the proxies stay
transparent, and the warm list cannot silently drift from what was deferred.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.lazy_import import (  # noqa: E402
    STARTUP_DEFERRED,
    LazyModule,
    lazy_attr,
    warm,
    warmed,
)


def _in_fresh_process(code: str) -> str:
    """Run code in a clean interpreter — the only honest way to test imports."""
    preamble = (
        "import sys, os\n"
        "sys.path.insert(0, r'" + str(ROOT) + "')\n"
        "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(code)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


# ── the proxies are transparent ──────────────────────────────────────────────

def test_lazy_module_does_not_import_until_touched():
    proxy = LazyModule("json")
    assert proxy.loaded is False
    assert proxy.dumps({"a": 1}) == '{"a": 1}'
    assert proxy.loaded is True


def test_lazy_module_reaches_submodule_attributes():
    proxy = LazyModule("email.utils")
    assert callable(proxy.formatdate)


def test_lazy_module_repr_shows_state():
    proxy = LazyModule("base64")
    assert "deferred" in repr(proxy)
    proxy.b64encode(b"x")
    assert "loaded" in repr(proxy)


def test_lazy_module_supports_dir_and_setattr():
    proxy = LazyModule("json")
    assert "dumps" in dir(proxy)
    proxy.__orion_probe__ = 1              # delegates to the real module
    import json
    assert json.__orion_probe__ == 1
    del json.__orion_probe__


def test_lazy_module_can_target_an_attribute_of_a_module():
    proxy = LazyModule("json", "JSONDecoder")
    assert proxy.__name__ == "JSONDecoder"


def test_lazy_callable_constructs_the_real_class():
    import fractions
    fraction = lazy_attr("fractions", "Fraction")
    assert fraction.loaded is False
    value = fraction(3, 4)
    assert str(value) == "3/4"
    assert fraction.loaded is True
    assert isinstance(value, fractions.Fraction)


def test_lazy_callable_repr_and_attribute_passthrough():
    import decimal as decimal_module
    proxy = lazy_attr("decimal", "Decimal")
    assert "deferred" in repr(proxy)
    assert proxy("1.5") + 1 == decimal_module.Decimal("2.5")
    assert "loaded" in repr(proxy)


def test_a_missing_module_fails_at_the_point_of_use_not_at_import():
    proxy = LazyModule("orion_definitely_not_a_real_module")
    with pytest.raises(ImportError):
        proxy.anything


# ── the deferral actually holds (fresh interpreter) ──────────────────────────

def test_importing_the_app_does_not_pull_the_heavy_packages():
    out = _in_fresh_process(
        """
        import orion_core.app
        pulled = [m for m in ('google.genai', 'aiohttp', 'sounddevice')
                  if m in sys.modules]
        print(','.join(pulled) or 'none')
        """
    )
    assert out == "none", "these are back on the startup path: " + out


def test_the_app_imports_well_inside_the_startup_budget():
    """Loose ceiling: the point is that it is hundreds of ms, not seconds."""
    out = _in_fresh_process(
        """
        import time
        t0 = time.perf_counter()
        import orion_core.app
        print(round((time.perf_counter() - t0) * 1000))
        """
    )
    elapsed = float(out)
    assert elapsed < 1400, (
        "importing orion_core.app costs %.0f ms — the eager third-party imports "
        "have crept back (1780 ms before the lazy pass, ~465 after)" % elapsed)


def test_live_worker_still_produces_real_genai_objects():
    out = _in_fresh_process(
        """
        import orion_core.live_worker as lw
        c = lw.types.Content(role='user', parts=[lw.types.Part(text='hi')])
        print(type(c).__name__, c.parts[0].text, callable(lw.genai.Client))
        """
    )
    assert out == "Content hi True"


def test_the_aiohttp_proxies_build_real_aiohttp_objects():
    out = _in_fresh_process(
        """
        from orion_core.utils import ClientTimeout
        from orion_core.providers import ClientSession
        t = ClientTimeout(total=7)
        print(type(t).__module__.split('.')[0], t.total, callable(ClientSession))
        """
    )
    assert out == "aiohttp 7 True"


def test_audio_still_reaches_the_real_sounddevice():
    out = _in_fresh_process(
        """
        import orion_core.audio as audio
        print(callable(audio.sd.query_devices), audio.sd._resolve().__name__)
        """
    )
    assert out == "True sounddevice"


# ── no module may re-introduce an eager heavy import ─────────────────────────

EAGER = re.compile(
    r"^(?:import\s+(?:aiohttp|sounddevice)"
    r"|from\s+(?:aiohttp|sounddevice|google\.genai)\s+import"
    r"|from\s+google\s+import\s+genai)"
)


def test_no_module_eagerly_imports_the_deferred_packages():
    offenders = []
    for path in sorted((ROOT / "orion_core").rglob("*.py")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if EAGER.match(line):
                offenders.append(path.name + ": " + line.strip())
    assert not offenders, (
        "these put a heavy package back on the startup path (import it inside "
        "the function, or use lazy_import): " + "; ".join(offenders))


# ── warming: the other half of the policy ────────────────────────────────────

def test_warm_imports_everything_it_is_given():
    warm("json", "base64").join(timeout=60)
    assert warmed("json") and warmed("base64")


def test_warm_survives_a_module_that_cannot_be_imported():
    warm("orion_definitely_not_a_real_module", "csv").join(timeout=60)
    assert warmed("csv"), "one bad module must not abort the warm-up"


def test_warm_reports_what_it_did():
    seen: list[str] = []
    warm("hashlib", log=seen.append).join(timeout=60)
    assert any("hashlib" in line for line in seen)


def test_warm_survives_a_logger_that_raises():
    def boom(_msg):
        raise RuntimeError("logger is broken")
    warm("colorsys", log=boom).join(timeout=60)
    assert warmed("colorsys")


def test_warm_is_idempotent_while_running():
    first = warm("zoneinfo")
    second = warm("zoneinfo")
    first.join(timeout=60)
    assert second is first or not second.is_alive()


def test_the_startup_deferred_list_matches_what_is_actually_deferred():
    """If someone defers a fourth package and forgets to warm it, its cost lands
    on the event loop instead — this is the check that catches that."""
    deferred: set[str] = set()
    for path in sorted((ROOT / "orion_core").rglob("*.py")):
        if path.name == "lazy_import.py":
            continue                 # its docstrings demonstrate the API
        src = path.read_text(encoding="utf-8", errors="replace")
        deferred |= set(re.findall(r'LazyModule\("([\w.]+)"', src))
        deferred |= set(re.findall(r'lazy_attr\("([\w.]+)"', src))
    warmed_roots = {name.split(".")[0] for name in STARTUP_DEFERRED}
    missing = {name.split(".")[0] for name in deferred} - warmed_roots
    assert not missing, (
        "deferred but never warmed, so the cost lands on the event loop: "
        + ", ".join(sorted(missing)))


def test_app_warms_the_imports_once_the_window_is_visible():
    src = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    window = src.index('budget.mark("core window visible")')
    warm_call = src.index("_warm_imports(")
    assert warm_call > window, "warming must start AFTER the window is shown"
    assert "lazy_import import warm" in src


def test_the_measured_justification_stays_in_the_source():
    src = (ROOT / "orion_core" / "lazy_import.py").read_text(encoding="utf-8")
    assert "1.28 s" in src and "qasync" in src


# ── ORION's own modules are off the pre-window path too ──────────────────────
#
# Once the third-party packages were deferred, what was left blocking first
# paint was ORION's own tree: app.py imported live_worker (~103 ms, which pulls
# audio, which pulls numpy), agents and gui.globe (~36 ms, which pulls
# QtWebEngineCore) at module scope, and constructed none of them until well
# after the window was visible. Measured: ~445 ms to import app before this,
# ~330 ms after.

LOCAL_DEFERRED = ("orion_core.live_worker", "orion_core.agents",
                  "orion_core.gui.globe")


def test_importing_the_app_does_not_pull_orions_own_heavy_modules():
    out = _in_fresh_process(
        """
        import orion_core.app
        names = ('orion_core.live_worker', 'orion_core.agents',
                 'orion_core.gui.globe', 'numpy')
        print(','.join(n for n in names if n in sys.modules) or 'none')
        """
    )
    assert out == "none", "back on the pre-window startup path: " + out


@pytest.mark.parametrize("module", LOCAL_DEFERRED)
def test_each_locally_deferred_module_still_imports(module):
    """Deferring must not hide a module that no longer imports cleanly."""
    out = _in_fresh_process(
        f"""
        import importlib
        print(importlib.import_module({module!r}).__name__)
        """
    )
    assert out == module


def test_the_local_deferrals_are_all_warmed():
    from orion_core.lazy_import import STARTUP_DEFERRED_LOCAL
    assert set(LOCAL_DEFERRED) <= set(STARTUP_DEFERRED_LOCAL), (
        "deferred out of app.py but never warmed — the cost just moves onto "
        "the qasync loop during boot")


def test_orions_own_modules_are_warmed_before_the_slow_third_party_ones():
    """One warm thread imports in order. google.genai is ~1.28 s; anything
    queued behind it is still cold when boot asks for it a moment later.

    The recorder does NOT perform the imports. ``orion_core.gui.globe`` is on
    the warm list, and test_gui_import_hygiene forbids any test importing it
    in-process: it pulls QtWebEngine, which corrupts Qt for every real-widget
    test that runs afterwards. The ordering contract is what is under test
    here, and it can be observed without importing anything.
    """
    from orion_core.lazy_import import STARTUP_DEFERRED, STARTUP_DEFERRED_LOCAL
    import orion_core.lazy_import as lazy

    recorded: list[str] = []
    real = lazy.importlib.import_module
    lazy.importlib.import_module = lambda name: recorded.append(name)
    previous = lazy._warm_thread
    try:
        lazy._warm_thread = None
        lazy.warm().join(timeout=60)
    finally:
        lazy.importlib.import_module = real
        lazy._warm_thread = previous
    assert recorded, "warm() imported nothing"
    order = {name: index for index, name in enumerate(recorded)}
    latest_local = max(order[name] for name in STARTUP_DEFERRED_LOCAL)
    assert order["google.genai"] > latest_local, (
        "google.genai is warmed before ORION's own modules: " + ", ".join(recorded))
    assert set(STARTUP_DEFERRED) <= set(recorded)


def test_the_warm_list_does_not_import_webengine_into_the_test_process():
    """A guard on the guard: if another QtWebEngine module joins the warm list,
    whoever adds it must confront this rule rather than discover it as a
    tracebackless interpreter death in an unrelated test."""
    from orion_core.lazy_import import STARTUP_DEFERRED_LOCAL
    assert "orion_core.gui.face3d" not in STARTUP_DEFERRED_LOCAL
    for name in STARTUP_DEFERRED_LOCAL:
        assert name in sys.modules or "webengine" not in name.lower()


def test_annotation_only_imports_stay_behind_type_checking():
    """dispatcher and the widget dashboard take an AgentManager but never touch
    it at import time; importing it eagerly is what dragged agents back in."""
    for relative in ("dispatcher.py", "gui/dashboard.py"):
        src = (ROOT / "orion_core" / relative).read_text(encoding="utf-8")
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith(("from .agents import", "from ..agents import")):
                assert line.startswith("    "), (
                    f"{relative}: agents imported at module scope again — it "
                    "belongs under `if TYPE_CHECKING:`")
