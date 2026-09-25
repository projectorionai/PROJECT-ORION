"""No surface gets a palette of its own.

The stylesheet was never the whole interface. Several widgets paint themselves
with QPainter, so no stylesheet has ever reached them, and each had quietly
grown its own colours: the operational status strip on top of every deck page
was cyan, green and amber; the cognition deck had a fourth set; the plugin deck
kept the exact green and yellow that ``test_status_colour_consolidation``
removed everywhere else; the workflow canvas was a blue theme inside a crimson
one; and ORION's own orb carried a saturated cyan rim.

None of that was visible in a stylesheet diff, and none of it raised anything.
It was found by rendering surfaces and looking at them. This test is the part
that does not need somebody to remember to look.

Offline: reads source, builds no QApplication.
"""

from __future__ import annotations

import colorsys
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GUI = ROOT / "orion_core" / "gui"

#: Where a hue encodes DATA or a rendered OBJECT, and one colour genuinely is
#: not an option — you cannot tell six token types apart with one hue, and a
#: circuit board is green. Each entry is a reason, not a permission slip: a
#: new name here should be arguable out loud.
HUE_IS_MEANING = {
    "token_graph.py": "categorical series on a graph",
    "knowledge_graph_view.py": "node categories and cluster colours",
    "globe.py": "map and terrain data",
    "face3d.py": "the rendered face",
    "holo_head.py": "the rendered face",
    "electronics_workbench.py": "a photographed circuit board",
    "chess_view.py": "board squares and piece sides",
    "theming.py": "the worked example of recolouring, in its own tests",
}


def _hues(source: str) -> list[tuple[str, int]]:
    """Saturated, mid-lightness colours that are not in the crimson family."""
    found: list[tuple[str, tuple[int, int, int]]] = []
    for match in re.finditer(r"QColor\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})",
                             source):
        found.append((match.group(0), tuple(int(x) for x in match.groups())))
    for match in re.finditer(r'"#([0-9a-fA-F]{6})"', source):
        raw = match.group(1)
        found.append((f"#{raw}", tuple(int(raw[i:i + 2], 16) for i in (0, 2, 4))))

    stray = []
    for text, (red, green, blue) in found:
        hue, light, saturation = colorsys.rgb_to_hls(red / 255, green / 255, blue / 255)
        degrees = hue * 360
        if saturation <= 0.25 or not 0.12 < light < 0.92:
            continue                      # a grey, or too dark/light to read as a hue
        if 340 <= degrees <= 360 or degrees <= 8:
            continue                      # the crimson family
        stray.append((text, round(degrees)))
    return stray


def test_no_widget_paints_itself_a_second_colour():
    offenders: dict[str, list[tuple[str, int]]] = {}
    for path in sorted(GUI.glob("*.py")):
        if path.name in HUE_IS_MEANING:
            continue
        stray = _hues(path.read_text(encoding="utf-8", errors="replace"))
        if stray:
            offenders[path.name] = stray
    assert not offenders, (
        "these custom-painted surfaces carry a hue that is not crimson:\n" +
        "\n".join(f"  {name}: {values}" for name, values in offenders.items()) +
        "\n\nIf the colour encodes data or a rendered object, add the file to "
        "HUE_IS_MEANING with the reason. Otherwise take it from C.")


def test_the_exemptions_all_still_exist():
    """An exemption for a deleted file is a hole nobody can see."""
    missing = [name for name in HUE_IS_MEANING if not (GUI / name).exists()]
    assert not missing, f"stale exemptions: {missing}"


def test_no_dialog_carries_a_private_copy_of_the_stylesheet():
    """Three dialogs each defined their own QDialog/QLineEdit/QPushButton
    rules: the same palette tokens in the old idiom, with a hover that turned
    every button crimson.

    Nothing was wrong with them as QSS, so nothing complained — they simply
    aged a little further behind the rest of the interface every time the
    shared sheet moved. A window may style its OWN objects (#hint, #excerpt);
    it may not restyle the base widget classes.
    """
    offenders = []
    for path in sorted(GUI.glob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        for call in re.finditer(r"setStyleSheet\(", source):
            chunk = source[call.end():call.end() + 2000]
            depth = 1
            body = chunk
            for index, char in enumerate(chunk):
                depth += (char == "(") - (char == ")")
                if depth == 0:
                    body = chunk[:index]
                    break
            if "APP_STYLESHEET" in body or "hud_stylesheet" in body:
                continue
            if re.search(r'["\s]QDialog\s*\{', body):
                line = source[:call.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line}")
    assert not offenders, (
        "these dialogs restyle QDialog themselves instead of using the "
        f"application's sheet: {offenders}")


def test_the_status_strip_takes_its_colours_from_the_palette():
    """It sits above every page on the deck, so it was the most-seen surface
    with a private palette."""
    from orion_core.constants import C
    from orion_core.gui import status_strip

    assert status_strip._OK.name().lower() == C.GOOD.lower()
    assert status_strip._WARN.name().lower() == C.WARN.lower()
    assert status_strip._INK.name().lower() == C.WHITE.lower()


def test_a_failure_does_not_look_like_being_offline():
    """The one line meant to tell them apart painted both the same grey."""
    from orion_core.gui.status_strip import _health_colour
    from orion_core.operational_status import OpStatus

    failing = _health_colour(OpStatus(online=True, health="FAILING"))
    offline = _health_colour(OpStatus(online=False))
    assert failing.name() != offline.name()
