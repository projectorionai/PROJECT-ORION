"""Export ORION's Screen Face for the Android app.

The phone wears the desktop's current face: orion_core/gui/screen_face.py
(the QPainter "robot display" face, Sentinel design by default). The page is a
Canvas 2-D port of that renderer, tools/phone_face_template.html, and this
script fills it with the face's own tables read from the PC source - every
expression channel, state family, emotion preset and alias, timing constant,
easing time, idle flourish, design and the HUD colours - plus the sentiment
lexicon and emotion mapping of orion_core/emotion.py, so the phone reads
feelings the way the PC does. Nothing is imported (no Qt on a build host):
the sources are parsed with ``ast``.

    python tools/export_phone_avatar.py           # write the Android asset
    python tools/export_phone_avatar.py --check   # fail if it is out of date

tests/test_phone_avatar_export.py runs the check, so the phone's face cannot
quietly drift from the desktop's.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FACE_SOURCE = ROOT / "orion_core" / "gui" / "screen_face.py"
FEEL_SOURCE = ROOT / "orion_core" / "emotion.py"
COLOUR_SOURCE = ROOT / "orion_core" / "constants.py"
MOTION_SOURCE = ROOT / "orion_core" / "appearance.py"
TEMPLATE = ROOT / "tools" / "phone_face_template.html"
DESTINATION = ROOT / "android" / "app" / "src" / "main" / "assets" / "orion_face.html"

FACE_MARK = "/*@FACE_TABLES@*/null"
FEEL_MARK = "/*@FEEL_TABLES@*/null"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignments(body: list[ast.stmt]) -> dict[str, ast.expr]:
    """name -> value node for plain, annotated and tuple assignments."""
    found: dict[str, ast.expr] = {}
    for node in body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            found[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value
                elif (isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple)
                      and len(target.elts) == len(node.value.elts)):
                    for name, value in zip(target.elts, node.value.elts):
                        if isinstance(name, ast.Name):
                            found[name.id] = value
    return found


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise RuntimeError(f"class {name} not found")


def _value(node: ast.expr, designs: dict[str, Any] | None = None) -> Any:
    """A literal, a frozenset({...}), a FaceDesign(...) or re.compile(...)."""
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else "")
        if name == "frozenset":
            return sorted(_value(node.args[0])) if node.args else []
        if name == "compile":
            return _value(node.args[0])
        if name == "FaceDesign" and designs is not None:
            design = dict(designs)
            design["key"] = _value(node.args[0])
            for keyword in node.keywords:
                design[keyword.arg] = _value(keyword.value)
            return design
        raise RuntimeError(f"unsupported call {name!r} in a face table")
    value = ast.literal_eval(node)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return [list(v) if isinstance(v, tuple) else v for v in value]
    if isinstance(value, dict):
        return {k: (_jsonable(v)) for k, v in value.items()}
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return value


def face_tables() -> dict[str, Any]:
    module = _parse(FACE_SOURCE)
    top = _assignments(module.body)
    design_class = _class(module, "FaceDesign")
    defaults = {node.target.id: ast.literal_eval(node.value)
                for node in design_class.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.value is not None}
    rig = _assignments(_class(module, "ScreenFace").body)
    timing_names = {
        "BLINK_MIN": "_BLINK_MIN", "BLINK_MAX": "_BLINK_MAX",
        "DOUBLE_BLINK_CHANCE": "_DOUBLE_BLINK_CHANCE", "BLINK_CLOSE": "_BLINK_CLOSE",
        "BLINK_HOLD": "_BLINK_HOLD", "BLINK_OPEN": "_BLINK_OPEN",
        "SACCADE_IDLE": "_SACCADE_IDLE", "SACCADE_LISTENING": "_SACCADE_LISTENING",
        "SACCADE_SPEAKING": "_SACCADE_SPEAKING", "SACCADE_THINKING": "_SACCADE_THINKING",
        "BREATH_PERIOD": "_BREATH_PERIOD", "VARIETY_EVERY": "_VARIETY_EVERY",
        "VARIETY_MIN_AMBIENT": "_VARIETY_MIN_AMBIENT", "ONSET_TAU": "_ONSET_TAU",
        "BOB_PERIOD": "_BOB_PERIOD", "BOB_TILT": "_BOB_TILT",
        "COCK_PERIOD": "_COCK_PERIOD", "COCK_TILT": "_COCK_TILT",
        "BOOT_SECONDS": "BOOT_SECONDS", "DOZE_AFTER": "_DOZE_AFTER",
        "VISEME_FRESH": "_VISEME_FRESH", "TAU_DEFAULT": "_TAU_DEFAULT",
        "LAG_SPLIT": "_LAG_SPLIT",
    }
    timing = {key: _value(top[name]) for key, name in timing_names.items()}
    timing["BAR_SEGMENTS"] = _value(rig["_BAR_SEGMENTS"])
    variety = {name: [duration, offsets] for name, (duration, offsets)
               in ast.literal_eval(top["VARIETY"]).items()}
    designs = {}
    for key_node, value_node in zip(top["DESIGNS"].keys, top["DESIGNS"].values):
        designs[ast.literal_eval(key_node)] = _value(value_node, defaults)
    return {
        "timing": timing,
        "NEUTRAL": _value(top["NEUTRAL"]),
        "TAU": _value(top["_TAU"]),
        "SINGLE_LAG": _value(top["_SINGLE_LAG"]),
        "MODES": _value(top["MODES"]),
        "STATE_MODES": _value(top["_STATE_MODES"]),
        "EMOTIONS": _value(top["EMOTIONS"]),
        "EMOTION_ALIASES": _value(top["_EMOTION_ALIASES"]),
        "ONSET": _value(top["_ONSET"]),
        "VARIETY": variety,
        "VARIETY_SIDED": _value(top["_VARIETY_SIDED"]),
        "VARIETY_WITH_EMOTION": _value(top["_VARIETY_WITH_EMOTION"]),
        "DESIGNS": designs,
        "DEFAULT_DESIGN": _value(top["DEFAULT_DESIGN"]),
        "MAX_RENDER_PIXELS": _value(top["MAX_RENDER_PIXELS"]),
        "PALETTE": palette(),
        "REDUCED_AMBIENT": reduced_ambient(),
    }


def palette() -> dict[str, str]:
    """ORION's crimson, his silver accent and the void, from constants.C."""
    colours = _assignments(_class(_parse(COLOUR_SOURCE), "C").body)
    return {"primary": _value(colours["PRI"]), "accent": _value(colours["ACCENT"]),
            "background": _value(colours["BG"])}


def reduced_ambient() -> float:
    """The Reduced-motion profile's ambient scale (appearance.MOTION_PROFILES)."""
    for node in ast.walk(_parse(MOTION_SOURCE)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "MotionProfile" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "reduced"):
            return float(ast.literal_eval(node.args[4]))
    raise RuntimeError("the Reduced motion profile was not found")


def feel_tables() -> dict[str, Any]:
    module = _parse(FEEL_SOURCE)
    top = _assignments(module.body)
    analyser = _assignments(_class(module, "SentimentAnalyser").body)
    manager = _assignments(_class(module, "EmotionStateManager").body)
    lexicon = {name: [[pattern, weight] for pattern, weight in rules]
               for name, rules in ast.literal_eval(analyser["_LEXICON"]).items()}
    return {
        "LEXICON": lexicon,
        "POSITIVE": _value(analyser["_POSITIVE"]),
        "NEGATOR": _value(analyser["_NEGATOR"]),
        "SENTIMENT_EMOTION": _value(top["_SENTIMENT_EMOTION"]),
        "USER_SENTIMENT_EMOTION": _value(top["_USER_SENTIMENT_EMOTION"]),
        "STATE_EMOTION": _value(top["_STATE_EMOTION"]),
        "SENTIMENT_HOLD_S": _value(manager["SENTIMENT_HOLD_S"]),
        "MIN_CONFIDENCE": _value(manager["MIN_CONFIDENCE"]),
        "EXCITED_AT": _value(manager["EXCITED_AT"]),
    }


def _js(value: Any) -> str:
    # "</" never appears inside the script element, whatever a table holds.
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")


def render_phone_page() -> str:
    page = TEMPLATE.read_text(encoding="utf-8")
    for mark in (FACE_MARK, FEEL_MARK):
        if page.count(mark) != 1:
            raise RuntimeError(f"The phone face template must hold {mark} exactly once.")
    page = page.replace(FACE_MARK, _js(face_tables()), 1)
    return page.replace(FEEL_MARK, _js(feel_tables()), 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the Android page is out of date")
    args = parser.parse_args(argv)
    expected = render_phone_page()
    if args.check:
        if not DESTINATION.is_file() or DESTINATION.read_text(encoding="utf-8") != expected:
            print("Android face page is out of date; run tools/export_phone_avatar.py", file=sys.stderr)
            return 1
        print("Android face page matches the desktop Screen Face.")
        return 0
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(expected.encode("utf-8"))
    print(f"Exported {DESTINATION} ({len(expected.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
