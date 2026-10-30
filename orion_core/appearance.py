"""
Which form ORION wears on screen.

THIS BUILD OFFERS ONE (Mark XXXVII): the owner chose the Screen Face and wants
nothing else shown, so ``ENABLED_FORMS`` is ``("screen",)``. The classic face
and the orb are still defined here, their code paths still live in the tree
(gui/face3d.py, gui/hud.py, the WebGL morph in core_window), and nothing was
deleted - they are simply not offered: no button, no menu entry, no jump-list
task, no orion:// link, no voice switch. A stored "classic" / "orb" / legacy
"face" therefore wears the screen face. ``ORION_FACE_FORM`` can still name a
dormant form for development, and that form then counts as enabled for that
process only.

The three that exist:

  * ``screen``  — the default. A robot display (gui/screen_face.py): a dark
    glass visor, two glowing eyes and a small mouth, Cozmo / EVE style. It is
    the owner's choice, made because the rendered human face unsettled him:
    "almost a person" is where the uncanny valley lives, and a screen face
    never tries to be one. It is also the cheapest form by far — no WebEngine
    page and no Chromium face process at startup.
  * ``classic`` — the rendered quantum human (gui/face3d.py), exactly as it
    was. Still there for anyone who prefers it.
  * ``orb``     — unmistakably a machine: a glowing sphere, no face at all.

Same ORION, same voice, same states in all three. This module remembers which
one the user asked for.

A stored "face" from before the screen face existed now means the screen
face. That is deliberate: "face" was simply "the default face", and the owner
chose the new default.

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
from dataclasses import dataclass
from typing import Any

from .constants import CONFIG_DIR
from .atomic_io import atomic_write_text

#: Every form ORION's code can draw: the screen face (default), the classic
#: rendered face, and the orb. Defined, not necessarily offered - see
#: ENABLED_FORMS.
FACE_FORMS: tuple[str, ...] = ("screen", "classic", "orb")

#: The forms THIS BUILD offers. Mark XXXVII: the Screen Face only. Widening
#: this tuple is all it takes to bring the others back to every menu, button,
#: jump-list task, link and voice command.
ENABLED_FORMS: tuple[str, ...] = ("screen",)

#: What he wears when nobody has said otherwise (and for any stored choice
#: that is no longer offered). Must be in ENABLED_FORMS.
DEFAULT_FORM = "screen"

#: What ORION says when somebody asks for a form this build does not offer.
SCREEN_ONLY_REPLY = "I only have my Screen Face in this version."

#: How each form is named on screen (Title Case, as the menu shows it).
FORM_LABELS: dict[str, str] = {
    "screen": "Screen Face",
    "classic": "Classic Face",
    "orb": "Orb",
}

# Words that name each form. Checked orb first, then classic, then screen, so
# "classic face" is classic (it contains "face") and "orb face" is an orb.
_ORB_WORDS = ("orb", "sphere", "ball")
_CLASSIC_WORDS = ("classic", "quantum", "realistic", "human", "3d", "3-d",
                  "person", "real face", "old face", "original")
_SCREEN_WORDS = ("screen", "robot", "cute", "companion", "eyes", "face",
                 "avatar", "display", "default")

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
    """The canonical name for *form* ("screen", "classic" or "orb"), or ""
    if it names none of them.

    Generous about phrasing on purpose: this is reached by voice as often as
    by a menu, and "go into your orb form" should not fail on the word "form".
    The legacy value "face" — what this module stored before the screen face
    existed — normalises to "screen", the new default face.
    """
    text = str(form or "").strip().lower().replace("_", " ")
    if not text:
        return ""
    if text in FACE_FORMS:
        return text
    for words, form_name in ((_ORB_WORDS, "orb"), (_CLASSIC_WORDS, "classic"),
                             (_SCREEN_WORDS, "screen")):
        for word in words:
            if word in text:
                return form_name
    return ""


def enabled_forms() -> tuple[str, ...]:
    """The forms offered right now: the build's ENABLED_FORMS, plus the one
    form ORION_FACE_FORM names (development re-enable, this process only)."""
    forms = list(ENABLED_FORMS)
    override = normalise(os.getenv("ORION_FACE_FORM", ""))
    if override and override not in forms:
        forms.append(override)
    return tuple(forms)


def form_enabled(form: Any) -> bool:
    """Whether *form* names a form this build currently offers."""
    return normalise(form) in enabled_forms()


def form_unavailable(form: Any) -> bool:
    """True when *form* names a real form that this build does not offer
    ("switch to your orb"): ORION answers it politely, it is not gibberish."""
    named = normalise(form)
    return bool(named) and named not in enabled_forms()


def face_form() -> str:
    """Which form ORION should be wearing.

    Always an enabled one: a stored "classic", "orb" or legacy "face" from an
    older build wears the screen face (the stored value is left alone, so
    re-enabling a form later restores the owner's old choice).
    """
    override = normalise(os.getenv("ORION_FACE_FORM", ""))
    if override:
        return override
    stored = normalise(_load().get("face_form"))
    return stored if stored in enabled_forms() else DEFAULT_FORM


def set_face_form(form: Any) -> str:
    """Remember *form*. Returns the form now in effect, or "" if unrecognised
    or not offered by this build.

    An unrecognised value changes nothing rather than falling back to the
    default: silently putting the face back on somebody who asked for the orb
    is the one outcome to avoid here. A form this build does not offer is
    refused the same way - callers say SCREEN_ONLY_REPLY (form_unavailable).
    """
    chosen = normalise(form)
    if not chosen or chosen not in enabled_forms():
        return ""
    data = _load()
    data["face_form"] = chosen
    data.setdefault("schema", "orion.appearance.v1")
    _save(data)
    return chosen


def form_label(form: Any) -> str:
    """The on-screen name of *form* ("Screen Face", "Classic Face", "Orb")."""
    return FORM_LABELS.get(normalise(form), "")


def describe() -> str:
    """One line for diagnostics and for ORION to say out loud."""
    current = face_form()
    if current == "orb":
        return "I'm in orb form — no face, just the sphere."
    if current == "classic":
        return "I'm wearing my classic face — the rendered 3-D one."
    return (f"I'm wearing my screen face — the {design_label(face_design())} design.")


# ── the Screen Face's designs (Mark XXXIX) ───────────────────────────────────
#
# Owner, 27 Oct 2026: "ORION's facial features must be better suited to the
# current design and ORION must have more facial designs". The Screen Face
# had one look - round cartoon eyes with blush strokes and sparkles - on a
# crimson-and-silver HUD that is anything but cartoon. Sentinel (the new
# default) is drawn for that HUD; Companion is the old look, kept; Arc, Visor
# and Minimal are new. The drawing lives in gui/screen_face.DESIGNS; this is
# only the choice, so nothing here imports Qt.

FACE_DESIGNS: tuple[str, ...] = ("sentinel", "arc", "visor", "minimal", "companion")
DEFAULT_FACE_DESIGN = "sentinel"
DESIGN_LABELS: dict[str, str] = {
    "sentinel": "Sentinel", "arc": "Arc", "visor": "Visor", "minimal": "Minimal",
    "companion": "Companion",
}
#: Words that name each design, checked in this order (whole words).
_DESIGN_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("companion", ("companion", "cute", "cartoon", "bubbly", "old", "original", "previous",
                   "round eyes", "classic screen")),
    ("arc", ("arc", "ring", "rings", "reactor", "circle", "circles", "circular", "halo")),
    ("visor", ("visor", "slit", "slits", "band", "stealth", "scanner", "narrow")),
    ("minimal", ("minimal", "minimalist", "simple", "dots", "dot", "plain", "clean")),
    ("sentinel", ("sentinel", "professional", "sleek", "sharp", "default", "standard",
                  "normal", "usual", "hud", "new")),
)


def normalise_design(name: Any) -> str:
    """The design *name* names ('the arc one', 'your cute face' -> 'companion'),
    or '' when it names none."""
    text = " " + " ".join(str(name or "").lower().replace("_", " ").replace("-", " ").split()) + " "
    if text.strip() in FACE_DESIGNS:
        return text.strip()
    for key, words in _DESIGN_WORDS:
        if any(f" {word} " in text for word in words):
            return key
    return ""


def face_design() -> str:
    """The Screen Face design in effect (ORION_FACE_DESIGN, stored, default)."""
    for candidate in (os.getenv("ORION_FACE_DESIGN", ""), _load().get("face_design")):
        key = normalise_design(candidate)
        if key:
            return key
    return DEFAULT_FACE_DESIGN


def set_face_design(name: Any) -> str:
    """Remember a design; the design now in effect, or '' if *name* is none."""
    key = normalise_design(name)
    if not key:
        return ""
    data = _load()
    data["face_design"] = key
    data.setdefault("schema", "orion.appearance.v1")
    _save(data)
    return key


def design_label(name: Any) -> str:
    return DESIGN_LABELS.get(normalise_design(name) or DEFAULT_FACE_DESIGN, "Sentinel")


def designs_sentence() -> str:
    return ", ".join(DESIGN_LABELS[key] for key in FACE_DESIGNS)


# ── motion and performance ───────────────────────────────────────────────────
#
# One visible choice instead of three environment variables nobody finds.
# What it may and may not change was decided from measurements, not taste:
#
#   * The ACTIVE frame rate is never lowered. A face that moves at 30 fps
#     while speaking reads as stutter, and stutter is the one thing the face
#     must never do; the governor already drops to an even 30 when nothing
#     moves, which was measured to save most of a core invisibly.
#   * "balanced" lowers the render resolution cap. That is GPU fill work,
#     which is what a weaker graphics card actually runs out of.
#   * "reduced" is for people who find motion uncomfortable (or the system's
#     own "show animations" setting is off): the head's idle drift and breath
#     are made smaller and the deck stops sliding between pages. Speech, lip
#     movement and expressions stay — they carry meaning.

@dataclass(frozen=True)
class MotionProfile:
    name: str
    label: str
    idle_fps: int            # the face's frame rate when nothing moves
    pixel_cap: float         # device-pixel-ratio ceiling for the face
    ambient: float           # 0..1 scale on idle drift and breathing
    deck_transitions: bool   # slide/fade between deck pages


MOTION_PROFILES: dict[str, MotionProfile] = {
    "full": MotionProfile("full", "Full", 30, 1.75, 1.0, True),
    "balanced": MotionProfile("balanced", "Balanced — lighter on the graphics card",
                              30, 1.25, 1.0, True),
    "reduced": MotionProfile("reduced", "Reduced motion", 30, 1.75, 0.3, False),
}

DEFAULT_MOTION = "full"

_motion_listeners: list[Any] = []


def _system_prefers_reduced_motion() -> bool:
    """Windows' "Show animations in Windows" switch, read once per call."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        enabled = ctypes.c_int(1)
        SPI_GETCLIENTAREAANIMATION = 0x1042
        if ctypes.windll.user32.SystemParametersInfoW(
                SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(enabled), 0):
            return not bool(enabled.value)
    except Exception:
        pass
    return False


def normalise_motion(name: Any) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return ""
    if "reduc" in text or "less motion" in text or "calm" in text:
        return "reduced"
    if "balanc" in text or "light" in text or "perform" in text or "save" in text:
        return "balanced"
    if "full" in text or "normal" in text or "default" in text:
        return "full"
    return ""


def system_prefers_reduced_motion() -> bool:
    """Whether Windows' own animation setting is off — offered as a hint in
    Preferences, never applied on its own. That switch is just as often part
    of "adjust for best performance" as a motion preference, and changing how
    ORION's face moves is the user's decision, not an inference."""
    return _system_prefers_reduced_motion()


def motion_profile() -> MotionProfile:
    """The profile in effect: ORION_MOTION_PROFILE, then the stored choice,
    then full."""
    chosen = (normalise_motion(os.getenv("ORION_MOTION_PROFILE", ""))
              or normalise_motion(_load().get("motion"))
              or DEFAULT_MOTION)
    return MOTION_PROFILES[chosen]


def set_motion_profile(name: Any) -> str:
    """Remember a profile and tell every live surface. "" if unrecognised."""
    chosen = normalise_motion(name)
    if not chosen:
        return ""
    data = _load()
    data["motion"] = chosen
    data.setdefault("schema", "orion.appearance.v1")
    _save(data)
    profile = MOTION_PROFILES[chosen]
    for listener in list(_motion_listeners):
        try:
            listener(profile)
        except Exception:
            pass
    return chosen


def on_motion_change(listener: Any) -> None:
    """Call *listener(profile)* whenever the profile changes."""
    _motion_listeners.append(listener)


def remove_motion_listener(listener: Any) -> None:
    try:
        _motion_listeners.remove(listener)
    except ValueError:
        pass


__all__ = [
    "APPEARANCE_PATH", "DEFAULT_FORM", "ENABLED_FORMS", "FACE_FORMS", "FORM_LABELS",
    "DEFAULT_FACE_DESIGN", "DESIGN_LABELS", "FACE_DESIGNS",
    "DEFAULT_MOTION", "MOTION_PROFILES", "MotionProfile", "SCREEN_ONLY_REPLY",
    "describe", "design_label", "designs_sentence", "enabled_forms", "face_design",
    "face_form", "form_enabled", "form_label", "normalise_design", "set_face_design",
    "form_unavailable", "motion_profile", "normalise",
    "normalise_motion",
    "on_motion_change", "remove_motion_listener", "set_face_form",
    "set_motion_profile", "system_prefers_reduced_motion",
]
