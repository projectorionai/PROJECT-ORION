"""
What ORION is, what he is running on, and — the part that was missing — what he
cannot do.

Why this is assembled rather than written
-----------------------------------------
``identity.py`` carries who ORION is: his name, his register, his manner. That
is the right place for it and none of it belongs here. What it cannot carry is
anything that depends on the machine or the build, because a sentence written
into a prompt goes stale the moment the world moves. Install a plugin and a
hardcoded list still claims the old abilities; run on a different OS and a
hardcoded "you are on Windows" is simply wrong.

So this block is generated from the live system at session start: the real
platform, the tools actually registered in the dispatcher, the count of what is
available. Add a tool and ORION knows he gained it; remove one and he stops
claiming it.

The limits are the important half
---------------------------------
ORION's persona told him at length who he was and never once told him what he
could not do. A model with no stated limits does not decline gracefully — it
improvises, because improvising is what a language model does in the absence of
a rule. The failure mode is specific and damaging: asked to do something outside
its reach, it produces a confident description of having done it.

So the limits are stated as plainly as the abilities:

  * his sight is a frame captured on demand, not a live feed he is watching;
  * he acts on THIS machine, and cannot reach a device he has no tool for;
  * anything outside his tool list should be said plainly rather than mimed.

This is deliberately short. A long list of prohibitions reads as distrust and
crowds out the rest of the prompt; three honest sentences about the shape of
his reach do the work.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Iterable

#: Tools whose names do not make their limitation obvious, and the limitation.
#: Only entries that change what ORION would otherwise claim are worth the
#: tokens — this is not documentation, it is a guard against improvising.
_NOTABLE_LIMITS: dict[str, str] = {
    "vision_analyse": "sight is a single frame captured when you ask for one, "
                      "not a live feed you are watching",
    "screen_read": "you see the screen only at the moment you capture it",
    "web_automation": "you drive a real visible browser, so the user can see "
                      "every step and it competes with them for the mouse",
    "flight_search": "you read fares off the search page rather than from "
                     "an airline, so quote them as what the page showed "
                     "just now — and you cannot book or pay for anything",
}


#: The rule is about the SILENCE, not about which tool is slow.
#:
#: Some replies open with several seconds of nothing, and usually not because a
#: tool is slow. The tool is instant; what takes the time is ORION composing the
#: call, or writing a long answer. From the user's side those are the same
#: thing: an assistant that has stopped responding.
#:
#: Naming the tools that "take a while" does not work — the list is wrong the
#: moment anything changes, and it misses the composing case entirely. Phrasing
#: the rule around the GAP covers both without a list to maintain.
SILENCE_RULE = (
    "WHEN A GAP WOULD FORM: if answering will take more than a moment — "
    "because a tool has to run, OR simply because the answer is long to write "
    "— say one short sentence first naming what you are about to do, then do "
    "it. \"Let me check the calendar.\" \"I'll pull those figures together.\" "
    "One sentence, in the user's own language, and then silence is fine. Do "
    "not do this for quick answers: a running commentary is worse than a pause."
)


def machine_summary() -> str:
    """The real platform, read at session start rather than assumed."""
    try:
        system = platform.system() or "an unknown system"
        release = platform.release() or ""
        machine = platform.machine() or ""
        python = ".".join(str(p) for p in sys.version_info[:3])
        parts = [f"{system} {release}".strip()]
        if machine:
            parts.append(machine)
        parts.append(f"Python {python}")
        return ", ".join(p for p in parts if p)
    except Exception:
        return "an unidentified machine"


def tool_names(dispatcher: Any = None,
               declarations: Iterable[dict[str, Any]] | None = None) -> list[str]:
    """The tools ORION actually has, from the live registry.

    Prefers whatever the dispatcher is really routing, because that is the set
    that can be CALLED; the declarations are the set the model is TOLD about,
    and where they disagree the router is the truth.
    """
    names: list[str] = []
    registry = getattr(dispatcher, "_routes", None) or getattr(dispatcher, "routes", None)
    if isinstance(registry, dict) and registry:
        names = [str(k) for k in registry]
    elif declarations is not None:
        names = [str(d.get("name")) for d in declarations if d.get("name")]
    else:
        try:
            from .dispatch_schema import TOOL_DECLARATIONS

            names = [str(d.get("name")) for d in TOOL_DECLARATIONS if d.get("name")]
        except Exception:
            names = []
    return sorted({n for n in names if n})


def capability_block(dispatcher: Any = None,
                     declarations: Iterable[dict[str, Any]] | None = None,
                     *, name: str = "Orion") -> str:
    """The generated self-knowledge block, ready to append to the persona."""
    names = tool_names(dispatcher, declarations)
    available = set(names)

    lines = [
        "WHAT YOU ARE RUNNING ON (assembled live this session, not written "
        f"into this prompt): you are {name}, running on {machine_summary()}. "
        f"You currently have {len(names)} tools registered."
        if names else
        "WHAT YOU ARE RUNNING ON (assembled live this session): you are "
        f"{name}, running on {machine_summary()}. No tools are registered in "
        "this session, so you can only converse.",
    ]

    limits = [
        "you act on THIS machine and through these tools only — you cannot "
        "reach a device, account or service you have no tool for",
        "if something is outside your tools, say so plainly and offer the "
        "nearest thing you can actually do; never describe having done "
        "something you did not do",
    ]
    for tool, limitation in _NOTABLE_LIMITS.items():
        if tool in available:
            limits.append(limitation)

    lines.append(
        "WHAT YOU CANNOT DO — state these plainly rather than improvising "
        "around them: " + "; ".join(limits) + "."
    )
    lines.append(SILENCE_RULE)
    # The user's language, if ORION has heard enough to be sure. Absent until
    # then, because a wrong guess here would have him greeting an English
    # speaker in Portuguese every morning.
    try:
        from .language_memory import LanguageMemory

        spoken = LanguageMemory().prompt_line()
        if spoken:
            lines.append(spoken)
    except Exception:
        pass
    return "\n".join(lines)


__all__ = ["SILENCE_RULE", "capability_block", "machine_summary", "tool_names"]
