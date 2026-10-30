"""Render ORION's emblem and write every icon the desktop and phone apps use.

    .venv\\Scripts\\python.exe tools\\make_brand_icons.py

The emblem is a lens: a graphite bezel holding an obsidian glass dome, with a
crimson orbit of light inside it and the three belt stars of Orion on one line
through the centre — Alnilam is the hot core, Alnitak and Mintaka sit on the
orbit. It is drawn procedurally (numpy + Pillow, no source artwork), in linear
light with supersampling, so it can be regenerated exactly and changed by
editing numbers rather than pixels.

Two things drive the structure:

* A lit, glossy render stops reading below about 40 px: the bezel's
  reflections, the star glints and the orbit blur into one grey-pink smudge.
  16, 20, 24 and 32 px are therefore drawn by a separate, flat renderer whose
  radii are chosen per size so the ring, core and stars land on whole pixels.
* The palette is the desktop's (orion_core.constants.C): true neutrals and one
  crimson. Nothing here introduces a second hue.

Outputs (paths relative to the repository root):
  assets/orion.ico                          Windows icon, 16-256 px, every build reads it
  assets/brand/orion-emblem-1024.png        master artwork
  android/app/src/main/ic_launcher-playstore.png           512 px store icon
  android/app/src/main/res/mipmap-*/ic_launcher*.webp      adaptive layers + legacy icons
  android/app/src/main/res/mipmap-anydpi-v26/ic_launcher*.xml
  android/app/src/main/res/drawable/ic_launcher_monochrome.xml  themed-icon layer
  android/app/src/main/res/drawable/ic_stat_orion.xml       status-bar icon
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "android" / "app" / "src" / "main" / "res"

# ── palette (orion_core.constants.C), converted to linear light ─────────────


def _linear(hex_colour: str) -> np.ndarray:
    c = np.array([int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)], np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


PRI = _linear("#ff2038")       # crimson, the only hue
PRI_HI = _linear("#ff5a6b")
CORE = _linear("#ff7a86")
WHITE = np.ones(3, np.float32)

# Orion's belt runs lower-left (Alnitak) to upper-right (Mintaka).
BELT = math.radians(22.0)
ORBIT = 0.52                   # orbit radius, as a fraction of the glass radius
GLASS = 0.84                   # glass radius, as a fraction of the bezel radius


def _smooth(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit(x: float, y: float, z: float) -> tuple[float, float, float]:
    n = math.sqrt(x * x + y * y + z * z)
    return x / n, y / n, z / n


# ── studio lighting ──────────────────────────────────────────────────────────
# A softbox up and to the left (the reflection that makes glass read as
# glass), a sky glow above, and a strip light behind and below right that
# draws the cold rim along the lower edge.
KEY = _unit(-0.42, 0.55, 0.72)
RIM = _unit(0.55, -0.62, -0.30)
SKY = _unit(0.0, 1.0, 0.12)
_E1 = np.array(_unit(KEY[2], 0.0, -KEY[0]))
_E2 = np.cross(KEY, _E1)
_TILT = math.radians(-35)


def _environment(rx, ry, rz, key_gain: float):
    ambient = 0.018 + 0.05 * np.clip(ry * 0.5 + 0.5, 0, 1) ** 2
    facing = np.maximum(rx * KEY[0] + ry * KEY[1] + rz * KEY[2], 1e-3)
    a0 = rx * _E1[0] + ry * _E1[1] + rz * _E1[2]
    b0 = rx * _E2[0] + ry * _E2[1] + rz * _E2[2]
    a = (a0 * math.cos(_TILT) - b0 * math.sin(_TILT)) / facing
    b = (a0 * math.sin(_TILT) + b0 * math.cos(_TILT)) / facing
    window = ((np.abs(a) / 0.42) ** 4 + (np.abs(b) / 0.24) ** 4) ** 0.25
    grade = 0.30 + 0.70 * _smooth(-1.0, 0.8, b / 0.24)       # brighter at the top
    key = key_gain * (1 - _smooth(0.70, 1.05, window)) * grade
    sky = 0.55 * _smooth(0.55, 0.98, rx * SKY[0] + ry * SKY[1] + rz * SKY[2])
    rim = 2.6 * _smooth(0.72, 0.94, rx * RIM[0] + ry * RIM[1] + rz * RIM[2])
    return ambient + key + sky + rim


def _harmonics(seed: int, count: int, kmin: int, kmax: int):
    """Fixed-seed sine sets: fine texture that is identical on every run."""
    rng = np.random.default_rng(seed)
    ks = rng.integers(kmin, kmax, count)
    phase = rng.uniform(0, 2 * math.pi, count)
    amp = rng.uniform(0.4, 1.0, count) / np.sqrt(ks / kmin)
    return [(int(k), float(p), float(w)) for k, p, w in zip(ks, phase, amp)]


_FILAMENT = _harmonics(7, 16, 18, 160)
_BRUSHING = _harmonics(11, 24, 40, 900)
_FIELD = np.random.default_rng(3).uniform(-0.9, 0.9, (18, 2))


def _texture(angle_or_radius, terms):
    total = sum(w for _, _, w in terms)
    out = np.zeros_like(angle_or_radius)
    for k, p, w in terms:
        out += w * np.sin(k * angle_or_radius + p)
    return out / total


# ── the full render (40 px and up) ───────────────────────────────────────────


def _interior(qx, qy, px_glass: float, detail: dict):
    """What is seen through the glass, in units of the glass radius (HDR)."""
    rho = np.sqrt(qx * qx + qy * qy)
    theta = np.arctan2(qy, qx)
    width = max(0.030, 0.55 / px_glass)
    lit_side = 0.80 + 0.20 * np.cos(theta - math.atan2(-0.62, 0.55))
    ring = np.exp(-((rho - ORBIT) / width) ** 2)
    ring = ring * (1 + detail["filament"] * _texture(theta, _FILAMENT))
    glow = np.exp(-np.abs(rho - ORBIT) / 0.045)
    out = PRI * ((1.35 * ring + 0.20 * glow) * lit_side)[..., None]
    out += CORE * (0.55 * np.exp(-((rho - ORBIT) / (width * 0.45)) ** 2) * lit_side)[..., None]
    out += PRI * (0.015 * np.exp(-(rho / 0.40) ** 2))[..., None]

    # Alnilam, the core.
    core = max(0.030, 0.65 / px_glass)
    out += WHITE * (4.0 * np.exp(-(rho / core) ** 2))[..., None]
    out += CORE * (0.55 * np.exp(-(rho / (core * 3.2)) ** 2) + 0.10 * np.exp(-rho / 0.12))[..., None]

    # Alnitak and Mintaka, on the orbit; Alnitak is the brighter of the two.
    star = max(0.018, detail["star_px"] / px_glass)
    for side, brightness in ((-1, 1.8), (1, 1.45)):
        dx = qx - side * ORBIT * math.cos(BELT)
        dy = qy - side * ORBIT * math.sin(BELT)
        d2 = dx * dx + dy * dy
        v = brightness * detail["star_gain"] * (np.exp(-d2 / star ** 2)
                                                + 0.22 * np.exp(-d2 / (3.2 * star) ** 2))
        if detail["spikes"]:
            v += 0.25 * brightness * (np.exp(-(dx / (star * 0.25)) ** 2) * np.exp(-np.abs(dy) / 0.035)
                                      + np.exp(-(dy / (star * 0.25)) ** 2) * np.exp(-np.abs(dx) / 0.035))
        out += WHITE * v[..., None]
    if detail["spikes"]:
        v = 0.30 * (np.exp(-(qx / (core * 0.22)) ** 2) * np.exp(-np.abs(qy) / 0.045)
                    + np.exp(-(qy / (core * 0.22)) ** 2) * np.exp(-np.abs(qx) / 0.045))
        out += WHITE * v[..., None]

    # A faint star field behind, only where it can resolve.
    if detail["field"]:
        speck = max(0.006, 0.45 / px_glass)
        rng = np.random.default_rng(5)
        for fx, fy in _FIELD:
            r = math.hypot(fx, fy)
            brightness = float(rng.uniform(0.05, 0.22))
            if r * r > 0.85 or abs(r - ORBIT) < 0.1 or r < 0.15:
                continue
            out += WHITE * (brightness * np.exp(-((qx - fx) ** 2 + (qy - fy) ** 2) / speck ** 2))[..., None]
    return out


def _shade(u, v, px_r0: float, detail: dict):
    """HDR colour of the lens at bezel-radius coordinates (u, v), y up."""
    r = np.sqrt(u * u + v * v)
    ux, uy = u / (r + 1e-9), v / (r + 1e-9)
    col = np.zeros(u.shape + (3,), np.float32)

    # Bezel: a flat-topped rounded profile, circularly brushed graphite.
    s = 2 * np.clip((r - GLASS) / (1 - GLASS), 0, 1) - 1
    nr = np.sign(s) * np.abs(s) ** 2.2
    nz = np.sqrt(np.clip(1 - nr * nr, 0, 1))
    nx, ny = nr * ux, nr * uy
    metal = _environment(2 * nz * nx, 2 * nz * ny, 2 * nz * nz - 1, key_gain=2.6) * 0.42
    off_key = np.angle(np.exp(1j * (np.arctan2(v, u) - math.atan2(KEY[1], KEY[0]))))
    sheen = 0.40 * np.exp(-(off_key / 0.42) ** 2) + 0.20 * np.exp(-((np.abs(off_key) - math.pi) / 0.35) ** 2)
    metal = metal + sheen * (1 - np.abs(nr)) ** 1.5
    metal = metal * (1 + 0.21 * _texture(r * 1.0, _BRUSHING))
    if detail["groove"]:                       # a machined step, only where it resolves
        t = (s + 1) / 2
        metal = metal * (1 - 0.35 * np.exp(-((t - 0.30) / 0.03) ** 2))
        metal = metal + 0.10 * np.exp(-((t - 0.34) / 0.018) ** 2) * _smooth(-0.3, 0.8, 2 * nz * ny)
    bezel = (r > GLASS) & (r <= 1.0)
    col[bezel] = (metal + 0.012)[bezel][:, None] * WHITE

    # Glass: a dome that steepens at the edge, so Fresnel reflection gathers
    # there, and a slight magnification of what lies beneath it.
    q = np.clip(r / GLASS, 0, 0.9995)
    slope = -0.30 * q ** 1.2 * (1 - q ** 2.2) ** (1 / 2.2 - 1)
    length = np.sqrt(slope * slope + 1)
    gx, gy, gz = -slope * ux / length, -slope * uy / length, 1 / length
    fresnel = 0.04 + 0.96 * (1 - gz) ** 5
    reflection = _environment(2 * gz * gx, 2 * gz * gy, 2 * gz * gz - 1, key_gain=6.0)
    inside = _interior(u / GLASS - 0.22 * gx, v / GLASS - 0.22 * gy, px_r0 * GLASS, detail)
    absorb = np.exp(-0.35 * (1 / np.maximum(gz, 0.2) - 1))
    shade = 1 - 0.75 * _smooth(0.78, 1.0, q) * (0.6 + 0.4 * _smooth(-1, 1, u * KEY[0] + v * KEY[1]))
    glass = inside * (absorb * shade * (1 - fresnel))[..., None] + (reflection * fresnel)[..., None]
    glass += (0.35 * np.exp(-((q - 1.0) / 0.012) ** 2))[..., None]
    lens = r <= GLASS
    col[lens] = glass[lens]
    return col, (r <= 1.0)


def _aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def _logistic(x):
    return 1 / (1 + np.exp(-3.4 * np.clip(x, -20, 20)))


def _box(img, rad: int, axis: int):
    pad = [(0, 0)] * img.ndim
    pad[axis] = (rad + 1, rad)
    csum = np.cumsum(np.pad(img, pad, mode="edge"), axis=axis)
    n = img.shape[axis]
    hi = np.take(csum, np.arange(2 * rad + 1, 2 * rad + 1 + n), axis=axis)
    lo = np.take(csum, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * rad + 1)


def _blur(img, sigma: float):
    """Three box passes ≈ a Gaussian; enough for a bloom."""
    rad = max(1, int(round(sigma * 0.95)))
    for _ in range(3):
        img = _box(_box(img, rad, 0), rad, 1)
    return img


def render_lens(size: int, *, r0_frac: float = 0.43, cy_frac: float = 0.47,
                shadow: bool = True, ss: int | None = None):
    """Premultiplied display-linear RGB + alpha of the lit lens on transparency."""
    ss = ss or (3 if size >= 256 else 4 if size >= 96 else 6)
    r0 = size * r0_frac
    cx, cy = size / 2.0, size * cy_frac
    detail = {
        "filament": float(np.clip((r0 - 20) / 80, 0.15, 0.55)),
        "star_px": 0.75 if r0 < 40 else 0.9,
        "star_gain": 1.0 if r0 > 60 else 1.5,
        "spikes": r0 >= 50,
        "groove": r0 >= 40,
        "field": r0 >= 150,
    }
    prem = np.zeros((size, size, 3), np.float32)
    alpha = np.zeros((size, size), np.float32)
    cover = np.zeros((size, size), np.float32)
    bright = np.zeros((size, size, 3), np.float32)
    xs = ((np.arange(size * ss) + 0.5) / ss).astype(np.float32)
    strip = max(1, 96 // ss)
    for y0 in range(0, size, strip):
        rows = min(strip, size - y0)
        ys = (y0 + (np.arange(rows * ss) + 0.5) / ss).astype(np.float32)
        X, Y = np.meshgrid(xs, ys)
        u, v = (X - cx) / r0, (cy - Y) / r0
        hdr, obj = _shade(u, v, r0, detail)
        a = obj.astype(np.float32)
        rgb = _aces(hdr)
        if shadow:
            drop = 0.55 * _logistic((1.0 - np.sqrt(u * u + ((v + 0.10) / 0.97) ** 2)) / 0.085)
            contact = 0.30 * _logistic((0.99 - np.sqrt(u * u + (v + 0.03) ** 2)) / 0.02)
            shade = np.clip(drop + contact, 0, 0.85)
            a_all = a + shade * (1 - a)
        else:
            a_all = a
        premul = rgb * a[..., None]            # the shadow is black: adds alpha only

        def down(x):
            return x.reshape(rows, ss, size, ss, *x.shape[2:]).mean(axis=(1, 3))

        prem[y0:y0 + rows] = down(premul)
        alpha[y0:y0 + rows] = down(a_all)
        cover[y0:y0 + rows] = down(a)
        bright[y0:y0 + rows] = down(np.clip(hdr - 1.0, 0, None) * a[..., None])
    bloom = np.stack([_blur(bright[..., c], 0.03 * r0) for c in range(3)], axis=-1)
    prem = np.minimum(prem + 0.55 * (1 - np.exp(-bloom)) * cover[..., None], alpha[..., None])
    return prem, alpha


# ── the small render (16-32 px) ──────────────────────────────────────────────
# Per size: bezel radius, bezel width, orbit radius, orbit width, core radius,
# star radius (0 = no stars; at 16 px they would only fuse with the orbit).
SMALL = {
    16: (7.55, 1.30, 3.55, 1.30, 1.00, 0.00),
    20: (9.50, 1.55, 4.45, 1.45, 1.15, 0.72),
    24: (11.45, 1.80, 5.35, 1.60, 1.35, 0.82),
    32: (15.20, 2.35, 7.10, 1.90, 1.70, 1.00),
}


def render_small(size: int, ss: int = 16):
    bezel_r, bezel_w, orbit_r, orbit_w, core_r, star_r = SMALL[size]
    n = size * ss
    grid = ((np.arange(n) + 0.5) / ss - size / 2.0).astype(np.float32)
    x, y = np.meshgrid(grid, -grid)
    r = np.hypot(x, y)
    glass_r = bezel_r - bezel_w
    light = np.clip((x * -0.6 + y * 0.8) / bezel_r, -1, 1)[..., None] * 0.5 + 0.5
    col = _linear("#4c4c4c") * (1 - light ** 1.2) + _linear("#d9d9d9") * light ** 1.2
    glass = r <= glass_r
    col[glass] = _linear("#090909")
    # The softbox reflection, flattened to one soft shape.
    a = math.radians(-35)
    hx, hy = x / glass_r + 0.30, y / glass_r - 0.42
    pu = hx * math.cos(a) - hy * math.sin(a)
    pv = hx * math.sin(a) + hy * math.cos(a)
    window = ((pu / 0.46) ** 2 + (pv / 0.20) ** 2 <= 1) & glass & (size >= 20)
    col += (window * (0.34 if size >= 24 else 0.24)
            * (0.4 + 0.6 * np.clip(pv / 0.40 + 0.5, 0, 1)))[..., None]
    if size >= 32:
        spill = np.exp(-np.maximum(np.abs(r - orbit_r) - orbit_w / 2, 0) / 0.45) * glass * 0.22
        col += PRI * spill[..., None]
        edge = x * 0.6 - y * 0.8
        lip = (r > glass_r - 0.55) & glass & (edge > 0)
        col[lip] += _linear("#8a8a8a") * np.clip(edge[lip] / glass_r, 0, 1)[:, None]
    col[(np.abs(r - orbit_r) <= orbit_w / 2) & glass] = PRI
    if size >= 24:
        col[(np.abs(r - orbit_r) <= orbit_w * 0.18) & glass] = PRI_HI
    col[(r > core_r) & (r <= core_r + 0.45) & glass] = CORE * 0.8
    col[r <= core_r] = WHITE
    for side in (-1, 1):
        if star_r:
            sx, sy = side * orbit_r * math.cos(BELT), side * orbit_r * math.sin(BELT)
            col[np.hypot(x - sx, y - sy) <= star_r] = WHITE
    col[(r > bezel_r - 0.55) & (r <= bezel_r)] *= 0.45    # a dark outline on light backgrounds
    obj = (r <= bezel_r).astype(np.float32)
    prem = (col * obj[..., None]).reshape(size, ss, size, ss, 3).mean(axis=(1, 3))
    alpha = obj.reshape(size, ss, size, ss).mean(axis=(1, 3))
    return prem, alpha


def emblem(size: int) -> Image.Image:
    """The desktop emblem at one size, choosing the renderer that reads there."""
    if size in SMALL:
        return to_image(*render_small(size))
    return to_image(*render_lens(size))


# ── conversion ───────────────────────────────────────────────────────────────


def to_image(prem, alpha) -> Image.Image:
    rgb = prem / np.maximum(alpha, 1e-6)[..., None]
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    # No dither: the brushed bezel already breaks up banding, and dither
    # noise tripled the master's size.
    out = np.dstack([srgb, alpha]) * 255.0
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8), "RGBA")


def over(top: Image.Image, bottom: Image.Image) -> Image.Image:
    return Image.alpha_composite(bottom.convert("RGBA"), top.convert("RGBA"))


# ── Android ──────────────────────────────────────────────────────────────────
DENSITIES = {"mdpi": 1.0, "hdpi": 1.5, "xhdpi": 2.0, "xxhdpi": 3.0, "xxxhdpi": 4.0}
LENS_DP = 29.0          # bezel radius inside the 108 dp canvas (safe zone is 33 dp)


def adaptive_background(size: int) -> Image.Image:
    """Near-black with a soft pool of light under the lens and faint crimson spill."""
    g = ((np.arange(size) + 0.5) / size).astype(np.float32)
    x, y = np.meshgrid(g, g)
    d = np.hypot(x - 0.5, (y - 0.42) * 1.1)
    lum = _linear("#181818") * np.exp(-(d / 0.42) ** 2)[..., None] + _linear("#060606")
    ring = np.hypot(x - 0.5, y - 0.49) - LENS_DP / 108
    lum += PRI * (0.018 * np.exp(-np.maximum(ring, 0) / 0.05) * (ring > -0.02))[..., None]
    return to_image(lum, np.ones((size, size), np.float32))


def adaptive_foreground(size: int) -> Image.Image:
    return to_image(*render_lens(size, r0_frac=LENS_DP / 108, cy_frac=0.49))


def _crop_centre(img: Image.Image, fraction: float, out: int) -> Image.Image:
    w = img.width
    side = w * fraction
    box = tuple(round(v) for v in ((w - side) / 2, (w - side) / 2, (w + side) / 2, (w + side) / 2))
    return img.crop(box).resize((out, out), Image.LANCZOS)


def _mask(size: int, rounded: bool) -> Image.Image:
    from PIL import ImageDraw
    ss = 8
    m = Image.new("L", (size * ss, size * ss), 0)
    inset = size * ss / 24
    box = (inset, inset, size * ss - inset, size * ss - inset)
    draw = ImageDraw.Draw(m)
    if rounded:
        draw.rounded_rectangle(box, radius=size * ss * 0.20, fill=255)
    else:
        draw.ellipse(box, fill=255)
    return m.resize((size, size), Image.LANCZOS)


def legacy_icon(size: int, rounded: bool) -> Image.Image:
    """Pre-adaptive icon: the visible 72 dp of the layers, masked."""
    canvas = round(size * 108 / 72)
    full = over(adaptive_foreground(canvas), adaptive_background(canvas))
    img = _crop_centre(full, 72 / 108, size)
    img.putalpha(_mask(size, rounded))
    return img


def _ring_arcs(r: float, gap_deg: float) -> str:
    """The orbit as two arcs, broken where Alnitak and Mintaka sit on it.

    A single-colour glyph cannot show a white star on a crimson ring; a gap
    around each star keeps all three stars readable in themed icons and in
    the status bar."""
    parts = []
    for start in (math.degrees(BELT), math.degrees(BELT) + 180):
        a0 = math.radians(start + gap_deg)
        a1 = math.radians(start + 180 - gap_deg)
        x0, y0 = 54 + r * math.cos(a0), 54 - r * math.sin(a0)
        x1, y1 = 54 + r * math.cos(a1), 54 - r * math.sin(a1)
        parts.append(f"M{x0:.2f},{y0:.2f} A{r:.2f},{r:.2f} 0 0,0 {x1:.2f},{y1:.2f}")
    return " ".join(parts)


def _circle(cx: float, cy: float, r: float) -> str:
    return f"M{cx - r:.2f},{cy:.2f} a{r:.2f},{r:.2f} 0 1,0 {2 * r:.2f},0 a{r:.2f},{r:.2f} 0 1,0 {-2 * r:.2f},0"


def glyph_vector(width_dp: int, scale: float, colour: str, note: str) -> str:
    """The single-colour emblem as a VectorDrawable (108-unit viewport)."""
    orbit = 11.0 * scale
    stars = " ".join(_circle(54 + side * orbit * math.cos(BELT), 54 - side * orbit * math.sin(BELT),
                             2.2 * scale) for side in (-1, 1))
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!-- {note}
     Written by tools/make_brand_icons.py; edit the script, not this file. -->
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="{width_dp}dp"
    android:height="{width_dp}dp"
    android:viewportWidth="108"
    android:viewportHeight="108">
    <path
        android:fillColor="#00000000"
        android:pathData="{_circle(54, 54, 21.0 * scale)}"
        android:strokeColor="{colour}"
        android:strokeWidth="{4.2 * scale:.2f}" />
    <path
        android:fillColor="#00000000"
        android:pathData="{_ring_arcs(orbit, 19)}"
        android:strokeColor="{colour}"
        android:strokeLineCap="round"
        android:strokeWidth="{2.6 * scale:.2f}" />
    <path
        android:fillColor="{colour}"
        android:pathData="{_circle(54, 54, 3.4 * scale)} {stars}" />
</vector>
"""


ADAPTIVE = """<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@mipmap/ic_launcher_background" />
    <foreground android:drawable="@mipmap/ic_launcher_foreground" />
    <monochrome android:drawable="@drawable/ic_launcher_monochrome" />
</adaptive-icon>
"""


def _save_webp(img: Image.Image, path: Path) -> None:
    # Lossless: the launcher shows these at full size, and lossy WebP blocks
    # visibly in the dark glass. Still well under the equivalent PNGs.
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "WEBP", lossless=True, quality=100, method=6)


def write_android(written: list[Path]) -> None:
    for name, scale in DENSITIES.items():
        folder = RES / f"mipmap-{name}"
        layer = round(108 * scale)
        for stem, img in (("ic_launcher_foreground", adaptive_foreground(layer)),
                          ("ic_launcher_background", adaptive_background(layer)),
                          ("ic_launcher", legacy_icon(round(48 * scale), rounded=True)),
                          ("ic_launcher_round", legacy_icon(round(48 * scale), rounded=False))):
            path = folder / f"{stem}.webp"
            _save_webp(img, path)
            written.append(path)
    for stem in ("ic_launcher", "ic_launcher_round"):
        path = RES / "mipmap-anydpi-v26" / f"{stem}.xml"
        path.write_text(ADAPTIVE, encoding="utf-8", newline="\n")
        written.append(path)
    vectors = {
        "ic_launcher_monochrome": glyph_vector(
            108, 1.0, "#FFFFFFFF",
            "Themed-icon layer (Android 13+): the emblem as one colour, which the\n"
            "     launcher tints. Kept inside the central 48 dp."),
        "ic_stat_orion": glyph_vector(
            24, 2.2, "#FFFFFFFF",
            "Status-bar icon. Android draws notification icons from their alpha\n"
            "     alone, so the full-colour launcher icon showed as a blank disc."),
    }
    for stem, xml in vectors.items():
        path = RES / "drawable" / f"{stem}.xml"
        path.write_text(xml, encoding="utf-8", newline="\n")
        written.append(path)
    # Google Play: 512 px, full-bleed square (the store applies its own mask).
    canvas = round(512 * 108 / 84)
    store = over(adaptive_foreground(canvas), adaptive_background(canvas))
    path = ROOT / "android" / "app" / "src" / "main" / "ic_launcher-playstore.png"
    _crop_centre(store, 84 / 108, 512).convert("RGB").save(path, optimize=True)
    written.append(path)


# ── desktop ──────────────────────────────────────────────────────────────────
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def write_desktop(written: list[Path]) -> None:
    frames = [emblem(s) for s in ICO_SIZES]
    ico = ROOT / "assets" / "orion.ico"
    # Every size is supplied, so Pillow stores these exact frames instead of
    # resampling the largest one (which is what blurred the small sizes).
    frames[-1].save(ico, format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                    append_images=frames[:-1])
    written.append(ico)
    master = ROOT / "assets" / "brand" / "orion-emblem-1024.png"
    master.parent.mkdir(parents=True, exist_ok=True)
    to_image(*render_lens(1024, ss=3)).save(master, optimize=True)
    written.append(master)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=("desktop", "android"))
    args = parser.parse_args()
    written: list[Path] = []
    if args.only in (None, "desktop"):
        write_desktop(written)
    if args.only in (None, "android"):
        write_android(written)
    for path in written:
        print(f"{path.stat().st_size:>9,}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
