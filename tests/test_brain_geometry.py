"""
ORION's mind at the centre of the swarm (Mark XXIII).

His face moved to the Core Window, which freed the middle of the graph to be
what the graph is actually about. What matters is that it reads as a BRAIN and
not as a brain-shaped blob, and that it costs the renderer nothing new: same
vertex format, same shader, same draw call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from orion_core.swarm.render.brain_geometry import (  # noqa: E402
    BRAIN_RADIUS,
    FISSURE_DEPTH,
    build_brain,
    fold_relief,
    region_counts,
    surface_radius,
)
from orion_core.swarm.render.face_geometry import (  # noqa: E402
    FACE_FLOATS,
    FEATURE_BROW,
    FEATURE_EYE_L,
    FEATURE_EYE_R,
    FEATURE_HALO,
    FEATURE_MOUTH,
    FEATURE_SKIN,
)


@pytest.fixture(scope="module")
def brain():
    return build_brain()


# ── it costs the renderer nothing new ────────────────────────────────────────

def test_it_is_the_same_vertex_format_as_the_face(brain):
    """Same shader, same buffer, same draw call — swapping the model at the
    centre of the scene is one upload, not a second pipeline."""
    assert brain.shape[1] == FACE_FLOATS
    assert brain.dtype == np.float32
    assert brain.flags["C_CONTIGUOUS"]


def test_it_never_wears_a_face_tag(brain):
    """The face shader animates eyes, mouth and brows off tags 1-4. A brain
    carrying those would blink and mouth its way through speech."""
    tags = set(np.unique(brain[:, 7]).tolist())
    assert tags <= {FEATURE_SKIN, FEATURE_HALO}
    for facial in (FEATURE_EYE_L, FEATURE_EYE_R, FEATURE_MOUTH, FEATURE_BROW):
        assert facial not in tags


def test_every_point_carries_a_unit_normal(brain):
    """Without normals the folds cannot be lit, and unlit folds are no folds."""
    lengths = np.linalg.norm(brain[:, 9:12], axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-3)


def test_it_stays_within_one_upload(brain):
    assert brain.nbytes < 3 * 1024 * 1024


def test_it_is_identical_on_every_launch():
    assert np.array_equal(build_brain(cortex_points=800, halo_points=100),
                          build_brain(cortex_points=800, halo_points=100))


# ── it is a BRAIN, not a brain-shaped blob ───────────────────────────────────

def test_the_hemispheres_are_split_by_a_fissure():
    """The deep midline groove. Nothing else in the shape says "brain" as
    immediately, and without it the two hemispheres are one lump.

    Measured across a BAND, not at two points: the gyri are deep enough that a
    single sample beside the midline can land in a sulcus and report no
    fissure at all. What the eye reads is the midline being systematically
    lower than the crowns either side."""
    from orion_core.swarm.render.brain_geometry import fissure_relief
    assert fissure_relief() > FISSURE_DEPTH * 0.25


def test_the_fissure_is_a_groove_on_top_not_a_slice_through_him():
    """It should not cut the underside, where the stem and cerebellum attach."""
    under_mid = surface_radius(0.0, -1.0, 0.0)
    under_beside = surface_radius(0.25, -1.0, 0.0)
    assert abs(under_beside - under_mid) < 0.04


def test_the_cortex_is_folded_rather_than_smooth():
    """Gyri are the whole reason it reads as a cortex. A smooth ellipsoid at
    this size is an egg."""
    assert fold_relief() > 0.08


def test_the_folds_are_a_displacement_the_light_can_find():
    """Normals must vary across neighbouring points, or the ridges are
    geometry nobody can see."""
    brain = build_brain(cortex_points=6000, cerebellum_points=0,
                        stem_points=0, halo_points=0, jitter=0.0)
    upper = brain[brain[:, 1] > BRAIN_RADIUS * 0.2]
    radial = upper[:, 0:3] / np.linalg.norm(upper[:, 0:3], axis=1, keepdims=True)
    alignment = np.einsum("ij,ij->i", upper[:, 9:12], radial)
    # A purely radial normal would sit at 1.0 everywhere.
    assert alignment.min() < 0.9


def test_he_is_longer_front_to_back_than_he_is_wide(brain):
    """The reverse of the head, which is what stops the two silhouettes
    reading as the same object."""
    body = brain[brain[:, 7] != FEATURE_HALO][:, 0:3]
    assert np.ptp(body[:, 2]) > np.ptp(body[:, 0])


def test_the_cerebrum_is_wider_than_it_is_tall(brain):
    """Measured on the cerebrum, not the whole assembly: the stem descends
    well below everything else, so with it included he is legitimately taller
    than he is wide and the claim would be about the stem, not the shape."""
    body = brain[brain[:, 7] != FEATURE_HALO][:, 0:3]
    cerebrum = body[body[:, 1] > -BRAIN_RADIUS * 0.35]
    assert np.ptp(cerebrum[:, 0]) > np.ptp(cerebrum[:, 1])


def test_the_cerebellum_sits_low_and_behind(brain):
    counts = region_counts(brain)
    assert counts["cerebellum"] > 500
    body = brain[brain[:, 7] != FEATURE_HALO]
    cere = body[(body[:, 1] < -BRAIN_RADIUS * 0.30)
                & (body[:, 2] < -BRAIN_RADIUS * 0.35)]
    assert len(cere) > 0
    assert cere[:, 1].mean() < 0        # below the midline
    assert cere[:, 2].mean() < 0        # behind the midline


def test_the_stem_descends_below_everything_else(brain):
    body = brain[brain[:, 7] != FEATURE_HALO][:, 0:3]
    lowest = body[body[:, 1] < np.percentile(body[:, 1], 2)]
    # Whatever is lowest must be narrow and central — that is the stem, not a
    # lobe hanging off one side.
    assert np.abs(lowest[:, 0]).max() < BRAIN_RADIUS * 0.35


def test_removing_the_stem_raises_his_lowest_point():
    with_stem = build_brain(cortex_points=4000, cerebellum_points=800,
                            stem_points=1200, halo_points=0)
    without = build_brain(cortex_points=4000, cerebellum_points=800,
                          stem_points=0, halo_points=0)
    assert with_stem[:, 1].min() < without[:, 1].min()


def test_the_halo_surrounds_the_mass_rather_than_sitting_on_it(brain):
    body = np.linalg.norm(brain[brain[:, 7] != FEATURE_HALO][:, 0:3], axis=1)
    halo = np.linalg.norm(brain[brain[:, 7] == FEATURE_HALO][:, 0:3], axis=1)
    assert halo.mean() > body.mean()


def test_the_scatter_never_fills_the_sulci_back_in():
    """Jitter breaks the Fibonacci regularity, but a jitter deeper than the
    folds erases the very thing it is texturing — the mistake that made the
    first native face a fog ball."""
    import inspect

    from orion_core.swarm.render import brain_geometry as bg
    jitter = inspect.signature(bg.build_brain).parameters["jitter"].default
    assert jitter * BRAIN_RADIUS < bg.FOLD_DEPTH * BRAIN_RADIUS


# ── colour ───────────────────────────────────────────────────────────────────

def test_he_takes_the_colourway_the_rest_of_him_uses():
    crimson = build_brain(cortex_points=1500, cerebellum_points=200,
                          stem_points=100, halo_points=100, colourway="crimson")
    blue = build_brain(cortex_points=1500, cerebellum_points=200,
                       stem_points=100, halo_points=100, colourway="blue")
    assert crimson[:, 3].mean() > crimson[:, 5].mean()
    assert blue[:, 5].mean() > blue[:, 3].mean()


def test_no_channel_sits_on_the_clipping_cliff(brain):
    """A channel reaching exactly 1.0 came back as ~0 through this renderer's
    readback path."""
    assert brain[:, 3:6].max() < 1.0


# ── degenerate requests ──────────────────────────────────────────────────────

def test_asking_for_nothing_gives_an_empty_cloud_not_a_crash():
    empty = build_brain(cortex_points=0, cerebellum_points=0,
                        stem_points=0, halo_points=0)
    assert empty.shape == (0, FACE_FLOATS)


def test_it_scales_down_without_losing_a_region():
    counts = region_counts(build_brain(cortex_points=3000, cerebellum_points=600,
                                       stem_points=300, halo_points=200))
    for region in ("cortex", "cerebellum", "stem", "halo"):
        assert counts[region] > 0, region
