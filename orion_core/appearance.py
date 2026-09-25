"""
Which form ORION wears on screen.

ORION's default face is a rendered human one, and a human face is not always
what you want in the room. Shown to somebody who is not expecting it — a
friend leaning over the desk, a video call, anybody who did not ask to be
looked at by a face — it lands somewhere between striking and unsettling.

So the face has an alternative that is unmistakably a machine: the orb. Same
ORION, same voice, same states; a glowing sphere rather than a person. This
module remembers which one the user asked for.

Why it is a stored preference rather than a switch in the window
---------------------------------------------------------------
Because the reason for choosing the orb usually outlasts the session. Somebody
who turned the face off because it unsettles a housemate does not want it back
the next time ORION restarts, and having to turn it off again each time is the
kind of small friction that ends with the feature unused.

Reading order is the same as everywhere else in ORION: an environment variable
for a one-off, then the stored choice, then the default.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

#: The forms ORION can take. "face" is the rendered face; "orb" is the sphere.
FACE_FORMS: tuple[str, ...] = ("face", "orb")

#: What he wears when nobody has said otherwise.
DEFAULT_FORM = "face"

APPEARANCE_PATH = CONFIG_DIR / "appearance.json"


def _load() -> dict[str, Any]:
    try:
        data = json.loads(APPEARANCE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict[str, Any]) -> bool:
    try:
        APPEARANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(APPEARANCE_PATH, json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def normalise(form: Any) -> str:
    """The canonical name for *form*, or "" if it names neither.

    Generous about phrasing on purpose: this is reached by voice as often as
    by a menu, and "go into your orb form" should not fail on the word "form".
    """
    text = str(form or "").strip().lower()
    if not text:
        return ""
    for word in ("orb", "sphere", "ball"):
        if word in text:
            return "orb"
    for word in ("face", "human", "avatar", "person"):
        if word in text:
            return "face"
    return ""


def face_form() -> str:
    """Which form ORION should be wearing."""
    override = normalise(os.getenv("ORION_FACE_FORM", ""))
    if override:
        return override
    stored = normalise(_load().get("face_form"))
    return stored or DEFAULT_FORM


def set_face_form(form: Any) -> str:
    """Remember *form*. Returns the form now in effect, or "" if unrecognised.

    An unrecognised value changes nothing rather than falling back to the
    default: silently putting the face back on somebody who asked for the orb
    is the one outcome to avoid here.
    """
    chosen = normalise(form)
    if not chosen:
        return ""
    data = _load()
    data["face_form"] = chosen
    data.setdefault("schema", "orion.appearance.v1")
    _save(data)
    return chosen


def describe() -> str:
    """One line for diagnostics and for ORION to say out loud."""
    current = face_form()
    if current == "orb":
        return "I'm in orb form — no face, just the sphere."
    return "I'm wearing my face."


__all__ = [
    "APPEARANCE_PATH", "DEFAULT_FORM", "FACE_FORMS",
    "describe", "face_form", "normalise", "set_face_form",
]
