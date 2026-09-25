"""
Whatever core_window calls on the face, the face must actually implement.

The trap
--------
`core_window._to_face` dispatches by name and guards every call with hasattr:

    def _to_face(self, method, *args):
        face = getattr(self, "face", None)
        ...

That guard is right — it lets several very different faces share one window —
but it means a method the face does NOT have is not an error. The call
succeeds, nothing happens, and nothing anywhere reports it.

That is precisely how the banner glance arrived: HoloHead.glance was written
and wired to bus.banner, but it lived on the RENDERER inside the panel rather
than on the panel itself, so `_to_face("glance", ...)` found no attribute and
the eyes never once moved. Same shape as bus.viseme having no producer: a
finished feature that silently was not there.

So the protocol is derived from the call sites and asserted against the face
that actually ships.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _protocol() -> list[str]:
    """Every method name core_window dispatches through _to_face."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    names = sorted(set(re.findall(r"_to_face\(\s*[\"'](\w+)[\"']", source)))
    assert names, "no _to_face call sites found — has core_window moved?"
    return names


def test_the_protocol_is_discoverable():
    names = _protocol()
    for expected in ("set_state", "set_amplitude", "set_speaking", "set_viseme"):
        assert expected in names, f"{expected} is no longer routed to the face"


@pytest.mark.parametrize("method", _protocol())
def test_the_default_face_implements_every_routed_method(method):
    """HoloHeadPanel is what _build_face returns, so a gap here is a feature
    that silently does nothing."""
    from orion_core.gui.holo_head import HoloHeadPanel

    assert hasattr(HoloHeadPanel, method), (
        f"core_window calls face.{method}() but the default face has no such "
        f"method — the call will be swallowed by _to_face's hasattr guard"
    )
    assert callable(getattr(HoloHeadPanel, method))


def test_a_renderer_only_method_does_not_count():
    """The specific mistake: implementing it on HoloHead, which lives INSIDE
    the panel, and assuming the panel therefore has it."""
    from orion_core.gui.holo_head import HoloHead, HoloHeadPanel

    for method in _protocol():
        if hasattr(HoloHead, method) and not hasattr(HoloHeadPanel, method):
            pytest.fail(
                f"{method} exists on the renderer but not on the panel; "
                f"_to_face only ever sees the panel"
            )


def test_the_glance_is_actually_wired_to_something():
    """A glance nothing triggers is the same as no glance at all."""
    source = (ROOT / "orion_core" / "gui" / "core_window.py").read_text(encoding="utf-8")
    assert '_to_face("glance"' in source, "nothing asks the face to look anywhere"


def test_the_fallback_faces_degrade_rather_than_crash():
    """The voxel face has no glance, and that is allowed — _to_face's guard is
    what makes several faces interchangeable. What must NOT happen is a crash."""
    from orion_core.gui.face import HologramFace

    for method in _protocol():
        attribute = getattr(HologramFace, method, None)
        assert attribute is None or callable(attribute), (
            f"HologramFace.{method} exists but is not callable"
        )
