"""
Recolouring ORION without rebuilding him.

``APP_STYLESHEET`` is an f-string built once at import from the constants in
``C``, so the crimson identity is baked into a few thousand characters of Qt
stylesheet the moment the module loads. That is the right shape for a fixed
palette and the wrong shape for a colour the user can change, and it is why
"live theming" was listed as missing even though every widget already reads its
colours from one place.

Rather than making the palette mutable — which would mean auditing every module
that captured a colour into a local, and would still not repaint anything —
this derives a new stylesheet from the existing one by substitution. One pass,
no global state, and the result is a string the caller applies with
``setStyleSheet``. Qt repaints the whole tree itself.

A hue is a family, not a colour
-------------------------------
ORION's crimson is four related values, not one:

    PRI      #ff1a3c   the identity colour
    PRI_DIM  #991024   pressed, borders, filled buttons
    PRI_HI   #ff4d68   hover and highlight
    CORE     #ff6478   the incandescent orb centre

Swapping only PRI gives a button whose hover state is still crimson, which
looks like a bug rather than a theme. So a chosen colour is expanded back into
the same family by moving lightness while keeping the hue and saturation the
user picked — the relationships between the four are preserved because they are
measured from the originals rather than invented.

The graphite surfaces, the silver accent and the status triad (good, warn, bad)
are deliberately NOT touched. Backgrounds are what make text readable, and a
red "bad" badge means bad in every theme.
"""

from __future__ import annotations

from ..constants import C

#: The crimson family, in the order a substitution must not confuse. Longest
#: first is irrelevant here (all are seven characters), but the mapping is
#: explicit so a future fifth member cannot be missed silently.
FAMILY: tuple[str, ...] = (C.PRI, C.PRI_DIM, C.PRI_HI, C.CORE)

#: Lightness of each family member relative to the identity colour, measured
#: from the originals rather than chosen. PRI_DIM is markedly darker, PRI_HI
#: and CORE progressively lighter.
_RELATIVE_LIGHTNESS: dict[str, float] = {
    C.PRI: 1.00,
    C.PRI_DIM: 0.58,
    C.PRI_HI: 1.22,
    C.CORE: 1.34,
}


def _to_rgb(value: str) -> tuple[int, int, int]:
    raw = (value or "").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        raise ValueError(f"not a colour: {value!r}")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(
        f"{max(0, min(255, int(round(channel)))):02x}" for channel in rgb)


def _scaled(value: str, factor: float) -> str:
    """Move a colour's lightness by *factor*, keeping its hue.

    Scaling the channels directly would desaturate as it brightens — a light
    red drifts toward white rather than staying red — so brightening moves each
    channel a fraction of the way to its maximum instead, which holds the hue.
    """
    red, green, blue = _to_rgb(value)
    if factor <= 1.0:
        return _to_hex((red * factor, green * factor, blue * factor))
    reach = min(1.0, factor - 1.0)
    return _to_hex((
        red + (255 - red) * reach,
        green + (255 - green) * reach,
        blue + (255 - blue) * reach,
    ))


def derive_family(primary: str) -> dict[str, str]:
    """Map each crimson value to its equivalent in *primary*'s family.

    Returns the identity mapping for an unusable colour, so a bad hex cannot
    produce an unreadable interface.
    """
    try:
        _to_rgb(primary)
    except (ValueError, AttributeError, TypeError):
        return {original: original for original in FAMILY}
    return {
        original: _scaled(primary, _RELATIVE_LIGHTNESS.get(original, 1.0))
        for original in FAMILY
    }


def themed_stylesheet(primary: str, base: str | None = None) -> str:
    """The application stylesheet, recoloured to *primary*.

    Surfaces, the silver accent and the status triad are untouched: a
    background is what makes text readable, and a red "bad" badge has to mean
    bad whatever the theme.
    """
    from .style import APP_STYLESHEET

    sheet = APP_STYLESHEET if base is None else base
    # A non-string reaches here whenever a caller passes a QColor, a None from
    # an empty setting, or anything else; returning the untouched sheet is the
    # right answer for all of them and costs nothing to check.
    if not isinstance(primary, str) or not primary:
        return sheet
    if primary.lower() == C.PRI.lower():
        return sheet
    mapping = derive_family(primary)
    # Case-insensitively, because a stylesheet written by hand may not match
    # the constants' casing.
    for original, replacement in mapping.items():
        sheet = sheet.replace(original, replacement)
        sheet = sheet.replace(original.upper(), replacement)
    return sheet


def is_readable(primary: str) -> bool:
    """Whether *primary* has enough contrast against ORION's background.

    Not a hard gate — it is the user's interface — but a colour that vanishes
    into the void black is worth declining rather than silently applying.
    """
    try:
        red, green, blue = _to_rgb(primary)
        back_r, back_g, back_b = _to_rgb(C.BG)
    except (ValueError, AttributeError, TypeError):
        return False

    def luminance(rgb: tuple[int, int, int]) -> float:
        channels = []
        for channel in rgb:
            value = channel / 255.0
            channels.append(value / 12.92 if value <= 0.03928
                            else ((value + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    lighter = max(luminance((red, green, blue)),
                  luminance((back_r, back_g, back_b)))
    darker = min(luminance((red, green, blue)),
                 luminance((back_r, back_g, back_b)))
    return (lighter + 0.05) / (darker + 0.05) >= 3.0


__all__ = ["FAMILY", "derive_family", "is_readable", "themed_stylesheet"]
