"""
ORION is red and black.

This keeps being re-established and keeps drifting back, because the cyan does
not live in one place. It has turned up as a renderer's own class defaults
(`#39b6ff`), as a hand-written voxel ramp that never referenced the palette at
all, and as stale comments describing an "electric-cyan Fresnel rim" on
something that had already been recoloured.

The pattern behind all three: a colour written as a literal instead of taken
from `C`. So these tests check the values a renderer actually starts with,
rather than checking that some method exists to change them later — the face
spent a long release cyan precisely because the recolouring method was only
ever called from one code path.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Files that draw ORION himself. A stray cyan anywhere else is a bug; a stray
#: cyan in one of these is the bug this module is about.
FACE_FILES = (
    "orion_core/gui/holo_head.py",
    "orion_core/gui/face.py",
    "orion_core/gui/hud.py",
    "orion_core/gui/hud_shell.py",
    "orion_core/gui/native_face.py",
)


def _is_cyan(red: int, green: int, blue: int) -> bool:
    """Blue or green clearly dominant, and the colour is not near-grey.

    Near-greys are excluded because ORION's surfaces are graphite and his
    accent is a silver-white, both of which have a slight cool cast by design.
    """
    if max(red, green, blue) - min(red, green, blue) < 40:
        return False                      # grey, graphite, silver — allowed
    return (blue > red + 30) or (green > red + 30)


def _code_only(path: Path) -> str:
    """The file with its comments removed.

    Tokenised rather than stripped at the first `#`, because a colour literal
    IS a `#` followed by hex and naive stripping would delete exactly what
    this is looking for. Docstrings are left in — a colour quoted in one is
    still prose, but comments are where the old values get recorded, and a
    note saying "this used to be #39b6ff" must not read as a relapse.
    """
    import io
    import tokenize

    source = path.read_text(encoding="utf-8")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    return "\n".join(token.string for token in tokens
                     if token.type == tokenize.STRING
                     or token.type == tokenize.NAME
                     or token.type == tokenize.NUMBER
                     or token.type == tokenize.OP)


def _hex_colours(text: str) -> list[tuple[str, tuple[int, int, int]]]:
    found = []
    for match in re.finditer(r"#([0-9a-fA-F]{6})\b", text):
        raw = match.group(1)
        found.append((match.group(0),
                      (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))))
    return found


@pytest.mark.parametrize("path", FACE_FILES)
def test_no_cyan_is_written_into_the_things_that_draw_him(path):
    """The literal-colour check.

    Every one of these has carried a hard-coded cyan at some point, and each
    time it survived because the palette was applied somewhere else and nobody
    looked at the default.
    """
    file = ROOT / path
    if not file.exists():
        pytest.skip(f"{path} is not part of this build")
    offenders = [
        f"{text} in {path}"
        for text, rgb in _hex_colours(_code_only(file))
        if _is_cyan(*rgb)
    ]
    assert offenders == [], f"cyan is back: {offenders}"


def test_the_voxel_face_ramp_is_crimson():
    """The 2-D fallback face was navy → blue → pale cyan, and referenced no
    palette constant at all, so nothing could recolour it."""
    from orion_core.gui.face import SKIN_BRIGHT, SKIN_DARK, SKIN_MID

    for name, rgb in (("SKIN_DARK", SKIN_DARK), ("SKIN_MID", SKIN_MID),
                      ("SKIN_BRIGHT", SKIN_BRIGHT)):
        red, green, blue = rgb
        assert red >= green and red >= blue, f"{name}={rgb} is not red-led"
        assert not _is_cyan(*rgb), f"{name}={rgb} is cyan"


def test_the_ramp_still_goes_dark_to_light():
    """Recolouring must not flatten the sculpt — the three stops carry the
    shading, and their relative lightness is what makes it read as a face."""
    from orion_core.gui.face import SKIN_BRIGHT, SKIN_DARK, SKIN_MID

    assert sum(SKIN_DARK) < sum(SKIN_MID) < sum(SKIN_BRIGHT)


_PROBE = r'''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, r"{root}")
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from orion_core.constants import C
from orion_core.gui.holo_head import HoloHeadPanel

head = HoloHeadPanel()
print("@@" + json.dumps({{
    "primary": head._primary.name(),
    "accent": head._accent.name(),
    "background": head._background.name(),
    "PRI": C.PRI,
    "ACCENT": C.ACCENT,
}}))
'''


@pytest.fixture(scope="module")
def head() -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(root=str(ROOT).replace("\\", "/"))],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail(f"probe failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    pytest.fail("probe produced nothing")


def test_the_head_starts_in_orions_colours(head):
    """Not "can be recoloured" — starts.

    It was recoloured only from `apply_accent`, which only ran if the user
    opened Preferences and chose one. Every other path — the overlay orb, a
    renderer attached after boot, a test — got the renderer's cyan.
    """
    assert head["primary"].lower() == head["PRI"].lower()
    assert head["accent"].lower() == head["ACCENT"].lower()


def test_the_head_sits_on_the_void_black(head):
    red, green, blue = (int(head["background"][i:i + 2], 16) for i in (1, 3, 5))
    assert red + green + blue < 90, "the backdrop should be near-black"
