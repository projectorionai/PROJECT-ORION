"""
3-D face — presence & motion (Mark XXVI, §10) and the embedded-JS integrity guard.

These are SOURCE checks by necessity: importing face3d pulls in QtWebEngine, which
(a) is unavailable headless and (b) corrupts Qt for later tests. But the embedded
Three.js can still be validated statically — bracket balance catches the class of
syntax error that would blank the WebGL face (which the Python test suite otherwise
cannot see), and the presence evolution can be confirmed by inspection.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SRC = (Path(__file__).resolve().parents[1] / "orion_core" / "gui"
        / "face3d.py").read_text(encoding="utf-8")
_HTML = re.search(r'FACE_HTML\s*=\s*r"""(.*?)"""', _SRC, re.S).group(1)


def test_embedded_threejs_brackets_are_balanced():
    # A blanked WebGL face from an unbalanced bracket is invisible to the Python
    # suite; this is the cheapest net that would still catch it.
    for op, cl in (("(", ")"), ("{", "}"), ("[", "]")):
        assert _HTML.count(op) == _HTML.count(cl), f"unbalanced '{op}{cl}' in FACE_HTML"


def test_idle_motion_is_subtle_and_organic_not_a_metronome():
    # The old idle yaw was a single ±0.22rad sine — a visible metronome swing.
    assert "Math.sin(clock*0.32)*0.22" not in _HTML, "the metronome idle sway is back"
    # It is now two incommensurate slow frequencies at a small amplitude.
    assert "Math.sin(clock*0.19)*0.6+Math.sin(clock*0.37)*0.4" in _HTML


def test_the_head_settles_while_listening():
    # Attention reads as stillness: idle drift scales down by a 'calm' factor
    # driven by the listening state.
    assert "const calm=1-listen" in _HTML
    assert "*calm" in _HTML


def test_the_new_expressions_have_eye_posture_body_language():
    # The scalar geometry reaches the 3-D face via orionEmotion; the eye/head
    # posture (pupil/gaze/blink/tilt) comes from the EXPR table, which must now
    # cover the nine Mark XXVI expressions too — not just the original eight.
    expr_block = _HTML[_HTML.find("const EXPR="):_HTML.find("const exState=")]
    for name in ("concentrating", "curious", "amused", "confused", "uncertain",
                 "disappointed", "proud", "reassuring", "empathetic"):
        assert re.search(rf"\b{name}\s*:", expr_block), f"{name} missing from EXPR"


def test_presence_still_preserves_the_existing_signals():
    # Evolve, don't remove: speech nods, emotional posture, listening tilt,
    # breathing and torso lag must all still be present.
    assert "nodP" in _HTML                       # speech nods
    assert "exState.headX" in _HTML and "exState.headZ" in _HTML   # emotional posture
    assert "listen*0.04" in _HTML                # listening roll tilt
    assert "torsoRig.rotation" in _HTML          # torso lag
