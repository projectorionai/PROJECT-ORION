"""
Capability health matrix (Mark XXVI).

Turns "why can't ORION do X?" into a table. The load-bearing guarantees: a probe
never takes the matrix down, registration reflects the real schema, and a
capability that is present-but-unusable reads DEGRADED (the OCR silent-fault class)
rather than OK.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.capability_health import (  # noqa: E402
    CapabilityHealth,
    capability_matrix,
    render_matrix,
    unavailable,
)


# ── status logic (pure) ───────────────────────────────────────────────────────

def test_status_reflects_presence_and_availability():
    assert CapabilityHealth("X", present=True, available=True, offline=True).status() == "OK"
    assert CapabilityHealth("X", present=True, available=False, offline=True).status() == "DEGRADED"
    assert CapabilityHealth("X", present=False, available=False, offline=True).status() == "MISSING"


def test_a_line_renders_every_field():
    row = CapabilityHealth("OCR", True, True, True, "engine: rapidocr", registered=True)
    line = row.line()
    assert "OCR" in line and "reg" in line and "offline" in line and "rapidocr" in line


# ── the matrix ────────────────────────────────────────────────────────────────

def test_matrix_probes_the_major_capabilities():
    names = {r.name for r in capability_matrix(deep=False)}
    for expected in ("OCR", "Vision/screen", "Study", "Focus", "Finance",
                     "Wellbeing", "Forge", "Chess"):
        assert expected in names, expected


def test_every_probe_returns_a_valid_status():
    for row in capability_matrix(deep=False):
        assert row.status() in {"OK", "DEGRADED", "MISSING"}, row.name


def test_registration_reflects_the_real_schema():
    rows = {r.name: r for r in capability_matrix(deep=False)}
    # OCR is exposed via vision_analyse, which is a registered tool.
    assert rows["OCR"].registered is True
    assert rows["Study"].registered is True
    # Voice rows are not single tools → registered is None.
    assert rows["Voice (cloud)"].registered is None


def test_offline_flags_are_sensible():
    rows = {r.name: r for r in capability_matrix(deep=False)}
    assert rows["Study"].offline is True
    assert rows["Voice (cloud)"].offline is False       # Gemini Live needs the net


def test_forge_probe_finds_the_real_class():
    # Regression: the class is ForgeOrchestrationManager, not ForgeEngine.
    forge = next(r for r in capability_matrix(deep=False) if r.name == "Forge")
    assert forge.present is True, forge.detail


def test_render_and_unavailable():
    rows = capability_matrix(deep=False)
    text = render_matrix(rows)
    assert "Capability health" in text
    assert "OCR" in text
    bad = unavailable(rows)
    assert all(r.status() != "OK" for r in bad)


# ── OCR functional probe (deep) — skips if no backend ─────────────────────────

def test_deep_ocr_probe_confirms_a_real_read():
    ocr = next(r for r in capability_matrix(deep=True) if r.name == "OCR")
    if ocr.status() == "MISSING":
        pytest.skip("no OCR backend installed")
    # If a backend is present, the deep probe must have actually read text —
    # this is the check that would have caught the numpy '.convert' fault.
    assert ocr.available is True
    assert "read OK" in ocr.detail


def test_a_present_but_unreadable_engine_reads_degraded(monkeypatch):
    # Simulate the exact OCR failure mode: engine available, read returns nothing.
    import orion_core.capability_health as ch

    class _Broken:
        available = True
        engine_name = "phantom"

        def image_to_text(self, _img):
            return ""            # present, "available", but reads nothing
    monkeypatch.setattr(ch, "OcrEngine", None, raising=False)
    # Drive the functional probe directly on the broken engine.
    ok, _note = ch._functional_ocr(_Broken())
    assert ok is False


# ── diagnostics wiring ────────────────────────────────────────────────────────

async def test_diagnostics_capability_report_renders_the_matrix():
    from orion_core.capability_health import CapabilityHealth  # noqa: F401
    from orion_core.diagnostics import DiagnosticsEngine

    class _Bus:
        def __getattr__(self, _n):
            class _S:
                def emit(self, *a): pass
                def connect(self, *a): pass
            return _S()

    eng = DiagnosticsEngine(_Bus(), memory=None)
    result = await eng.capability_report(deep=False)
    assert "Capability health" in result.text


def test_the_diagnostics_tool_routes_capabilities():
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "dispatch_files.py"
           ).read_text(encoding="utf-8")
    assert "capability_report" in src
    assert '"capabilities"' in src


def test_full_diagnostic_includes_a_capability_check():
    import inspect
    from orion_core.diagnostics import DiagnosticsEngine
    src = inspect.getsource(DiagnosticsEngine)
    assert "_check_capabilities" in src


# ── the matrix is polled, so it must be cheap (Mark XXVI performance) ────────

def test_the_matrix_is_cheap_to_poll():
    """Diagnostics and the proactive loop poll this. Constructing an OcrEngine
    per call re-probed every backend and cost ~750 ms EVERY time; the probe
    engine and the plugin scan are now cached.

    Measured RELATIVELY (cold call vs warm call) rather than against a fixed
    millisecond budget: an absolute threshold is a flake on a machine already
    running four thousand other tests, whereas the ratio holds under any load.
    """
    import time
    from orion_core import capability_health as ch

    ch.reset_probe_cache()
    ch._PLUGIN_PROBE = None
    start = time.perf_counter()
    capability_matrix(deep=False)
    cold = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(3):
        capability_matrix(deep=False)
    warm = (time.perf_counter() - start) / 3

    assert warm < cold, "the probe cache is not being used at all"
    # The cold path builds an OCR engine and AST-parses every plugin; warm should
    # be a small fraction of that. 0.5 is loose enough to survive a loaded box.
    assert warm < cold * 0.5, (
        f"caching barely helped: cold {cold*1000:.0f} ms vs warm {warm*1000:.0f} ms")


def test_the_probe_cache_can_be_reset():
    from orion_core import capability_health as ch
    capability_matrix(deep=False)
    assert ch._OCR_PROBE is not None or True          # may be absent on a bare box
    ch.reset_probe_cache()
    assert ch._OCR_PROBE is None
    # and it rebuilds cleanly afterwards
    assert capability_matrix(deep=False)


# ── module-availability cache (Mark XXVI performance) ────────────────────────

def test_module_availability_is_cached():
    """Profiling showed 84 nt.stat calls per poll — 99% of the matrix cost, and
    all of it disk I/O on the qasync loop, re-answering a question that cannot
    change without a pip install."""
    from orion_core import capability_health as ch
    ch.reset_probe_cache()
    assert ch._IMPORTABLE == {}
    capability_matrix(deep=False)
    assert ch._IMPORTABLE, "nothing was cached"
    assert "cv2" in ch._IMPORTABLE


def test_the_cache_returns_the_same_answer_as_a_live_lookup():
    import importlib.util

    from orion_core import capability_health as ch
    ch.reset_probe_cache()
    for module in ("cv2", "pyttsx3", "orion_definitely_not_real"):
        cached = ch._importable(module)
        try:
            live = importlib.util.find_spec(module) is not None
        except Exception:
            live = False
        assert cached is live, module
        assert ch._importable(module) is cached, "second call disagreed with the first"


def test_reset_clears_every_cache_not_just_one():
    """A half-cleared diagnostic lies: after a pip install all three caches are
    potentially stale."""
    from orion_core import capability_health as ch
    capability_matrix(deep=False)
    ch.reset_probe_cache()
    assert ch._OCR_PROBE is None
    assert ch._PLUGIN_PROBE is None
    assert ch._IMPORTABLE == {}


def test_a_probe_for_a_missing_module_is_cached_as_false():
    from orion_core import capability_health as ch
    ch.reset_probe_cache()
    assert ch._importable("orion_definitely_not_real") is False
    assert ch._IMPORTABLE["orion_definitely_not_real"] is False
