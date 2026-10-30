"""Memory ORION commits for nothing at start-up.

MEASURED 29 Oct 2026 on the owner's 12-thread Ryzen (source boot, temporary
config, 40 s after ready): the main process committed 1,240 MB private with
622 MB resident; with the two fixes below, 841 MB and 502 MB.

* OpenBLAS (numpy's, and scipy's own copy) commits a ~32 MB buffer per thread
  when it loads, one thread per logical core: `import numpy` cost +363 MB
  private with +11 MB resident. orion.py caps it at four threads (+106 MB),
  which ran ORION's BLAS work as fast as twelve.
* The voice-activity gate imported torch (2.1 s of import beside the GUI,
  +144 MB) before finding that silero_vad was not installed at all.

Offline: no QApplication, no audio device, no model.
"""

from __future__ import annotations

import builtins
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.audio import SileroVADGatekeeper  # noqa: E402


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _Bus:
    def __getattr__(self, name):
        return _Signal()


def test_openblas_threads_are_capped_before_anything_imports_numpy():
    source = (ROOT / "orion.py").read_text(encoding="utf-8")
    cap = source.find('os.environ.setdefault("OPENBLAS_NUM_THREADS"')
    assert cap != -1, "orion.py no longer caps OpenBLAS's per-thread buffers"
    first_import = re.search(r"^\s*(import numpy|from numpy|import orion_core|from orion_core)",
                             source, re.MULTILINE)
    assert first_import is not None and cap < first_import.start()


def test_no_torch_import_when_silero_is_not_installed(monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "silero_vad" else real_find_spec(name, *a))
    real_import = builtins.__import__
    asked: list[str] = []

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            asked.append(name)
            raise AssertionError("torch imported although silero_vad is absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    gate = SileroVADGatekeeper(_Bus(), threshold=0.5, defer=True)
    assert gate.ensure_ready() is False          # the local acoustic gate
    assert asked == []
    assert gate.confidence(b"\x00\x10" * 256) >= 0.0
