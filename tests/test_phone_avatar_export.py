"""Keep the phone's face the desktop's current face: the Screen Face.

Owner, 29 Oct 2026: the app should "just have his current face on the PC ...
with the same emotions" instead of the Three.js orb and face. The Android
page is generated from orion_core/gui/screen_face.py and emotion.py by
tools/export_phone_avatar.py; these tests fail if the bundled page drifts from
those sources or brings Three.js back.
"""

import ast
import json
import re
from pathlib import Path

from tools.export_phone_avatar import (DESTINATION, FACE_SOURCE, face_tables,
                                       feel_tables, render_phone_page)

ROOT = Path(__file__).resolve().parents[1]


def _embedded(page: str, name: str) -> dict:
    match = re.search(rf"const {name}=(\{{.*?\}});\n", page)
    assert match, f"{name} tables missing from the phone page"
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_phone_face_is_the_current_desktop_screen_face():
    page = render_phone_page()
    assert DESTINATION.read_text(encoding="utf-8") == page
    face = _embedded(page, "F")
    assert face == face_tables()
    from orion_core.gui import screen_face
    assert face["NEUTRAL"] == screen_face.NEUTRAL
    assert face["MODES"] == screen_face.MODES
    assert face["EMOTIONS"] == screen_face.EMOTIONS
    assert face["EMOTION_ALIASES"] == screen_face._EMOTION_ALIASES
    assert face["STATE_MODES"] == screen_face._STATE_MODES
    assert face["DEFAULT_DESIGN"] == screen_face.DEFAULT_DESIGN == "sentinel"
    for key, design in screen_face.DESIGNS.items():
        assert face["DESIGNS"][key] == {
            field: getattr(design, field) for field in design.__dataclass_fields__}
    assert face["timing"]["BOOT_SECONDS"] == screen_face.BOOT_SECONDS
    assert face["timing"]["BAR_SEGMENTS"] == screen_face.ScreenFace._BAR_SEGMENTS
    from orion_core.constants import C
    assert face["PALETTE"] == {"primary": C.PRI, "accent": C.ACCENT, "background": C.BG}


def test_phone_reads_feelings_with_the_desktop_lexicon_and_mapping():
    page = render_phone_page()
    feel = _embedded(page, "S")
    assert feel == feel_tables()
    from orion_core import emotion
    assert feel["SENTIMENT_EMOTION"] == emotion._SENTIMENT_EMOTION
    assert feel["USER_SENTIMENT_EMOTION"] == emotion._USER_SENTIMENT_EMOTION
    assert feel["STATE_EMOTION"] == emotion._STATE_EMOTION
    assert feel["NEGATOR"] == emotion.SentimentAnalyser._NEGATOR.pattern
    assert set(feel["LEXICON"]) == set(emotion.SentimentAnalyser._LEXICON)
    assert feel["MIN_CONFIDENCE"] == emotion.EmotionStateManager.MIN_CONFIDENCE


def test_phone_face_has_the_app_bridge_and_no_three_js():
    page = render_phone_page()
    for api in ("window.orionStep=", "window.orionState=", "window.orionAmplitude=",
                "window.orionSpeaking=", "window.orionEmotion=", "window.orionFeel=",
                "window.orionLink=", "window.orionDesign=", "window.orionPalette="):
        assert api in page
    for three in ("three.module", "THREE.", "importmap", "WebGLRenderer",
                  'type="module"', "import(", "getContext('webgl"):
        assert three not in page
    assert "getContext('2d')" in page
    assert not re.search(r"https?://", page)                  # nothing fetched
    assert "/*@" not in page                                  # template fully filled


def test_android_build_no_longer_bundles_three_js():
    gradle = (ROOT / "android" / "app" / "build.gradle").read_text(encoding="utf-8")
    assert "assets/three" not in gradle
    source = (ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "orion" /
              "companion" / "PhoneAvatarView.kt").read_text(encoding="utf-8")
    assert "orion_face.html" in source
    assert "orionMorph(" not in source            # no orb/face morph any more


def test_exporter_reads_the_source_without_importing_qt():
    tree = ast.parse((ROOT / "tools" / "export_phone_avatar.py").read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert "PyQt6" not in imported and "orion_core" not in imported
    assert FACE_SOURCE.name == "screen_face.py"
