"""
ORION's emblem, as rendered by tools/make_brand_icons.py.

Two regressions are easy to make and invisible until someone looks closely at a
taskbar or a launcher: an .ico whose small sizes are resampled from the 256 px
render (which turns the lens into a grey-pink smudge at 16-24 px), and an
adaptive icon that references a layer missing from one density.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from tools import make_brand_icons as brand  # noqa: E402

RES = ROOT / "android" / "app" / "src" / "main" / "res"


def test_the_ico_carries_every_size_windows_asks_for():
    with Image.open(ROOT / "assets" / "orion.ico") as ico:
        sizes = set(ico.ico.sizes())
    assert sizes == {(s, s) for s in brand.ICO_SIZES}


@pytest.mark.parametrize("size", [16, 24])
def test_small_frames_are_the_hand_tuned_ones_not_resamples(size):
    with Image.open(ROOT / "assets" / "orion.ico") as ico:
        stored = np.asarray(ico.ico.getimage((size, size)).convert("RGBA"))
    assert np.array_equal(stored, np.asarray(brand.emblem(size)))


def test_every_adaptive_layer_exists_in_every_density():
    for stem in ("ic_launcher", "ic_launcher_round"):
        xml = (RES / "mipmap-anydpi-v26" / f"{stem}.xml").read_text(encoding="utf-8")
        assert "@mipmap/ic_launcher_foreground" in xml
        assert "@mipmap/ic_launcher_background" in xml
        assert "@drawable/ic_launcher_monochrome" in xml
    for density, scale in brand.DENSITIES.items():
        for stem, side in (("ic_launcher_foreground", 108), ("ic_launcher_background", 108),
                           ("ic_launcher", 48), ("ic_launcher_round", 48)):
            path = RES / f"mipmap-{density}" / f"{stem}.webp"
            with Image.open(path) as im:
                assert im.size == (round(side * scale),) * 2, path
    assert (RES / "drawable" / "ic_launcher_monochrome.xml").is_file()
    assert (RES / "drawable" / "ic_stat_orion.xml").is_file()


def test_the_status_bar_glyph_is_a_single_colour_silhouette():
    """Android draws notification icons from alpha alone; any second colour is lost."""
    xml = (RES / "drawable" / "ic_stat_orion.xml").read_text(encoding="utf-8")
    colours = {part.split('"')[1] for part in xml.split("Color=")[1:]}
    assert colours == {"#FFFFFFFF", "#00000000"}
