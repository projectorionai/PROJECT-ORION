"""Deterministic synthetic PCB scenes; no real-board accuracy claim or API calls."""
import asyncio

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import pytest

from orion_core.electronics_inspection import inspect_frame, prepare_frame


def board():
    image = Image.new("RGB", (800, 600), (24, 80, 44))
    draw = ImageDraw.Draw(image)
    for x in range(50, 751, 45):
        draw.line((x, 30, x, 570), fill=(185, 170, 90), width=3)
    for x in range(100, 700, 100):
        draw.rectangle((x, 220, x + 60, 320), fill=(15, 15, 15), outline=(160, 160, 160), width=3)
        draw.text((x + 4, 242), "U1 123", fill=(220, 220, 220))
    return image


def test_blur_and_low_light_are_measured_against_the_same_scene():
    image = board()
    _, _, sharp = prepare_frame(image)
    _, _, blurred = prepare_frame(image.filter(ImageFilter.GaussianBlur(8)))
    _, _, dark = prepare_frame(ImageEnhance.Brightness(image).enhance(.1))
    assert blurred["sharpness"] < sharp["sharpness"]
    assert any("edge detail" in warning for warning in blurred["warnings"])
    assert any("dark" in warning for warning in dark["warnings"])


def test_local_glare_is_reported_even_when_whole_frame_is_not_overexposed():
    image = board()
    ImageDraw.Draw(image).rectangle((250, 180, 450, 380), fill="white")
    _, _, quality = prepare_frame(image)
    assert quality["brightness"] < .87
    assert quality["clipped_fraction"] > .025
    assert any("glare" in warning for warning in quality["warnings"])


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_rotations_and_unreadable_markings_do_not_invent_identifications(angle):
    image = board().rotate(angle, expand=True)
    result = asyncio.run(inspect_frame(image, ocr_reader=lambda image: ""))
    report = result.evidence[0]
    assert result.ok and result.media["data"]
    assert report["observations"] == [] and report["markings"] == []
    assert (report["quality"]["width"], report["quality"]["height"]) == image.size
    assert any("no readable markings" in note for note in report["limitations"])
