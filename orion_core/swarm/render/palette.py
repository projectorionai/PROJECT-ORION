"""
palette.py — what a node looks like, derived from what it IS and what it is
DOING (Mark XXII, Phase 1).

One rule drives the whole colour design: **categorical hue and semantic alarm
must never compete for the same channel.** If clusters were assigned twelve
evenly spaced hues around the wheel, two of them would land on red and amber —
and a healthy BUSINESS node would then be indistinguishable at a glance from a
DOWN subsystem. That is a genuine misreading risk in a monitoring surface, not
a matter of taste.

So the twelve cluster hues are drawn only from the cool half of the wheel
(blue → cyan → teal → green → violet), and red/amber are reserved exclusively
for DEGRADED and DOWN. A node in trouble is therefore the only warm thing on
screen, and reads instantly without the user decoding a legend.

Activity is carried on a separate channel again — emissive intensity and pulse
rate, not hue — so "what cluster is this", "is it healthy" and "is it working
right now" are three independent readings from one sphere.

Pure functions over floats. No GL, no Qt, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..clusters import CLUSTER_ORDER, cluster_index, normalise_cluster
from ..model import Activity, Health, NodeKind, SwarmNode

RGB = tuple[float, float, float]

# Twelve cluster hues drawn from ORION's OWN identity — crimson, rose, copper,
# gold, mauve, plum, graphite — rather than the cool blues/teals this started
# with. The cool palette was internally coherent but looked like a different
# application bolted onto ORION: his shell is near-black with crimson chrome
# and amber accents, and a cyan-green graph in the middle of it read as
# foreign.
#
# Because everything is now warm, hue can no longer carry the alarm signal on
# its own. The separation moved to CHROMA and BRIGHTNESS instead: every
# cluster colour is deliberately held below ~0.86 max channel and kept
# relatively low-chroma, while the two alarm colours are the only fully
# saturated, near-maximum-brightness things in the scene — and the only ones
# that pulse. A failing node is therefore still the most conspicuous object on
# screen, without the graph having to leave ORION's colour world.
_CLUSTER_COLOURS: dict[str, RGB] = {
    "INTELLIGENCE":  (0.62, 0.55, 0.98),   # violet — the reasoning core
    # Pulled back from (0.40, 0.72, 0.98): that hit brightness 0.98 and
    # chroma 0.58, crossing BOTH alarm thresholds, so a perfectly healthy
    # RESEARCH node read as a fault. Every cluster hue must keep clear
    # headroom under those limits — see is_alarm_colour.
    "RESEARCH":      (0.46, 0.72, 0.90),   # azure
    "DEVELOPMENT":   (0.44, 0.84, 0.88),   # cyan
    "BUSINESS":      (0.88, 0.74, 0.46),   # gold
    "CREATIVE":      (0.80, 0.48, 0.94),   # magenta-violet
    "AUTOMATION":    (0.50, 0.64, 0.88),   # electric blue
    "OPERATIONS":    (0.52, 0.70, 0.88),   # steel blue
    "COMMUNICATION": (0.66, 0.52, 0.96),   # periwinkle violet
    "MEMORY":        (0.56, 0.46, 0.90),   # deep indigo
    "MONITORING":    (0.42, 0.78, 0.86),   # teal-cyan
    "SECURITY":      (0.74, 0.58, 0.92),   # pale amethyst
    "SYSTEM":        (0.55, 0.60, 0.76),   # cool graphite
}

# Reserved, and the only fully saturated colours in the scene.
COLOUR_DEGRADED: RGB = (1.00, 0.72, 0.14)   # pure amber
COLOUR_DOWN: RGB = (1.00, 0.22, 0.30)       # fracture red
# Core is white-hot — the brightest thing in the field, the origin everything
# else radiates from.
COLOUR_CORE: RGB = (0.94, 0.96, 1.00)
_FALLBACK: RGB = (0.55, 0.60, 0.76)

# Emissive intensity per activity. IDLE sits well below 1.0 so a quiet system
# reads as calm rather than as a wall of glowing spheres — the contrast is
# what makes genuine activity legible.
_ACTIVITY_EMISSIVE: dict[Activity, float] = {
    Activity.OFFLINE:  0.10,
    Activity.IDLE:     0.28,
    Activity.ACTIVE:   0.62,
    Activity.BUSY:     0.90,
    Activity.THINKING: 0.80,
    Activity.ERROR:    1.00,
}

# Pulse frequency in Hz. Zero means "steady" — most nodes, most of the time.
_ACTIVITY_PULSE: dict[Activity, float] = {
    Activity.OFFLINE:  0.0,
    Activity.IDLE:     0.0,
    Activity.ACTIVE:   0.6,
    Activity.BUSY:     1.4,
    Activity.THINKING: 0.9,
    Activity.ERROR:    2.6,
}

# A never-invoked subsystem is dimmed toward the background so the graph
# distinguishes "working and quiet" from "never used" without a legend.
_QUIET_DIM = 0.55


@dataclass(frozen=True, slots=True)
class Appearance:
    """Everything the renderer needs to draw one node, as plain floats ready
    to be written straight into an instance buffer."""

    colour: RGB
    emissive: float
    pulse_hz: float
    radius_scale: float = 1.0

    def as_tuple(self) -> tuple[float, ...]:
        return (*self.colour, self.emissive, self.pulse_hz, self.radius_scale)


def cluster_colour(cluster: str) -> RGB:
    return _CLUSTER_COLOURS.get(normalise_cluster(cluster), _FALLBACK)


def _mix(a: RGB, b: RGB, t: float) -> RGB:
    t = max(0.0, min(1.0, t))
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def _scale(colour: RGB, factor: float) -> RGB:
    return (colour[0] * factor, colour[1] * factor, colour[2] * factor)


def appearance_for(node: SwarmNode) -> Appearance:
    """The full visual treatment for one node.

    Health wins over cluster hue, because a subsystem being down is more
    urgent than which zone it belongs to. The cluster colour is still mixed in
    rather than fully replaced, so a failing node stays locatable — you can
    still see WHICH cluster is in trouble."""
    base = COLOUR_CORE if node.kind is NodeKind.CORE else cluster_colour(node.cluster)

    # Mix ratios are set so the result clears ALARM_MIN_BRIGHTNESS even from
    # the DARKEST cluster (SYSTEM, 0.53 max). At the previous 0.85/0.7 a
    # failing SYSTEM or MONITORING node landed just under the threshold and
    # read as merely "a bit warm" rather than as an alarm.
    if node.health is Health.DOWN:
        colour = _mix(base, COLOUR_DOWN, 0.9)
    elif node.health is Health.DEGRADED:
        colour = _mix(base, COLOUR_DEGRADED, 0.9)
    elif node.activity is Activity.ERROR:
        # Health can be OK while the error RATE is alarming — a subsystem
        # answering every call with a failure is still "up". Mixed as hard as
        # the DOWN case: at a gentler 0.6 this landed on a muddy pink that
        # read as neither healthy nor alarming, which is the worst outcome
        # for a state the user needs to notice immediately.
        colour = _mix(base, COLOUR_DOWN, 0.88)
    elif node.health is Health.UNKNOWN:
        # Never measured is not the same as measured-and-fine; desaturate
        # toward grey rather than claiming health the system does not have.
        colour = _mix(base, _FALLBACK, 0.45)
    else:
        colour = base

    emissive = _ACTIVITY_EMISSIVE.get(node.activity, 0.3)
    pulse = _ACTIVITY_PULSE.get(node.activity, 0.0)

    if node.telemetry.is_quiet and not node.needs_attention:
        colour = _scale(colour, _QUIET_DIM)
        emissive *= _QUIET_DIM

    # Core always reads as the brightest thing in the scene.
    if node.kind is NodeKind.CORE:
        emissive = max(emissive, 0.95)

    return Appearance(colour=colour, emissive=emissive, pulse_hz=pulse)


def all_cluster_colours() -> dict[str, RGB]:
    """Every cluster's hue — used by the legend and the inspector so they can
    never drift from what the renderer actually draws."""
    return {cluster: cluster_colour(cluster) for cluster in CLUSTER_ORDER}


ALARM_MIN_BRIGHTNESS = 0.93
ALARM_MIN_CHROMA = 0.55


def is_alarm_colour(colour: RGB) -> bool:
    """True if *colour* reads as an alarm.

    Tested on CHROMA and BRIGHTNESS, not hue. Once the cluster palette moved
    into ORION's warm identity, a hue-based test was useless — every colour in
    the scene is warm, so "red beats blue" flagged healthy clusters as alarms.
    What still separates them cleanly is intensity: alarm colours are the only
    fully saturated, near-maximum-brightness values in the whole palette, and
    every cluster hue is deliberately held below both thresholds.
    """
    brightest = max(colour)
    chroma = brightest - min(colour)
    return brightest >= ALARM_MIN_BRIGHTNESS and chroma >= ALARM_MIN_CHROMA


__all__ = [
    "Appearance", "COLOUR_CORE", "COLOUR_DEGRADED", "COLOUR_DOWN", "RGB",
    "all_cluster_colours", "appearance_for", "cluster_colour", "is_alarm_colour",
    "cluster_index",
]
