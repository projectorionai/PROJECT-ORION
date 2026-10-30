"""
Reflex fast-path (Mark XXVI) — instant local understanding of clear commands.

Almost every spoken/typed turn currently pays a full model round-trip, even for
unambiguous imperatives ("start a 25 minute focus block", "what's my runway",
"quiz me"). This layer recognises a small, curated set of HIGH-PRECISION command
forms locally and maps them straight to a tool call — so ORION acts in tens of
milliseconds instead of a network turn, and never mis-selects a tool for the ones
that matter most.

Design rules that keep it safe:

  * **Precision over recall.** A rule fires only on an unambiguous imperative /
    interrogative shape (anchored, specific verbs). Anything conversational,
    hedged, or open-ended returns None and flows to the model unchanged. The
    tests enforce a corpus of "must NOT match" conversational lines.
  * **Only safe, known-argument tools.** Read-mostly or trivially-reversible
    intents (focus/study/catch_up/finance/diagnostics/token_usage). No
    destructive, no ambiguous-argument, no open-domain commands.
  * **Pure.** ``match_reflex(text) -> ReflexMatch | None`` is a free function with
    no I/O; the worker does the dispatching. Fully unit-testable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ReflexMatch:
    tool: str
    args: dict
    why: str            # which reflex fired (for the log / trace)


def _focus_start(m: re.Match) -> tuple[str, dict, str]:
    args: dict = {"action": "start"}
    if m.group("min"):
        args["minutes"] = int(m.group("min"))
    preset = (m.group("preset") or "").lower()
    for key in ("pomodoro", "long", "sprint"):
        if key in preset:
            args["preset"] = key
            break
    else:
        if "deep" in preset:
            args["preset"] = "deep"
    label = (m.group("label") or "").strip()
    args["label"] = label or "focus"
    return "focus", args, "focus.start"


_LEVEL = (r"(?P<lvl>\d{1,3}(?:\.\d+)?\s*(?:%|percent|per cent)?|half(?:way)?|max(?:imum)?|full|zero|"
          r"a quarter|three quarters)")
_AMOUNT = (r"(?:\s+(?P<amt>a\s+(?:bit|little|touch|notch|tad)|slightly|a\s+lot|loads|"
           r"(?:by\s+)?\d{1,2}\s*(?:%|percent)?))?")
#: ORION's three forms as people say them: "your screen face", "the classic
#: face", "your robot face", "the orb", "your face" (the screen face). The
#: matched words go to appearance.normalise unchanged.
_FORM = (r"(?:(?:screen|robot|cute|classic|quantum|realistic|human|3-?d|old|original)"
         r"\s+face|orb|sphere|face|human|classic|robot)")
#: The Screen Face's designs as people say them (appearance.normalise_design
#: reads the matched words).
_DESIGN = (r"(?:sentinel|arc|ring|reactor|visor|slit|minimal|minimalist|simple|companion|"
           r"cute|cartoon|sleek|professional|old|original|previous|default|normal|usual)")
def _place_kinds() -> str:
    from .places import _CUISINES, _TAGS
    kinds = sorted(set(_TAGS) | {f"{c} (?:place|restaurant|takeaway)" for c in _CUISINES}
                   | {f"{c}s?" for c in ("restaurant", "pub", "cafe", "shop")}, key=len, reverse=True)
    words = [k if "(" in k or "?" in k else re.escape(k) + "s?" for k in kinds]
    return "(?:" + "|".join(words) + ")"


_PLACE_KINDS = _place_kinds()

_TIMER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15,
                "twenty": 20, "twenty five": 25, "thirty": 30, "forty": 40, "forty five": 45,
                "fifty": 50, "sixty": 60, "ninety": 90}


#: A timer given no unit is in minutes ("set a timer for 10") - but only up to
#: this many; "a timer for 400" is not obviously anything, so it is asked about.
UNITLESS_TIMER_MAX_MINUTES = 180


def _timer_args(m: re.Match) -> dict | None:
    half = (m.group("half") or "").lower()
    if half:
        return {"action": "timer", "minutes": {"half an hour": 30, "an hour": 60,
                                               "an hour and a half": 90}[half]}
    raw = (m.group("n1") or m.group("n2") or "").lower()
    n = int(raw) if raw.isdigit() else _TIMER_WORDS.get(raw, 0)
    unit = (m.group("u1") or m.group("u2") or "").lower()
    if not unit and not 0 < n <= UNITLESS_TIMER_MAX_MINUTES:
        return None
    key = "seconds" if unit.startswith("s") else "hours" if unit.startswith("h") else "minutes"
    return {"action": "timer", key: n}


_SMALL_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                  "eight": 8, "nine": 9, "ten": 10, "fifteen": 15}
_LEVEL_WORDS = {"half": 0.5, "halfway": 0.5, "max": 1.0, "maximum": 1.0, "full": 1.0,
                "zero": 0.0, "a quarter": 0.25, "three quarters": 0.75}


def _level_reflex(m: re.Match) -> tuple[str, dict, str]:
    what = (m.group("what") or m.group("what2") or "").lower()
    raw = (m.group("lvl") or m.group("lvl2") or "").lower()
    number = re.match(r"\d{1,3}(?:\.\d+)?", raw)
    if number:
        value = float(number.group(0))
        # Spoken levels are percentages ("volume to 1" is 1%); only a
        # written fraction like 0.5 is a fraction.
        level = value if ("." in number.group(0) and value <= 1) else value / 100.0
    else:
        level = _LEVEL_WORDS.get(raw.strip(), 0.5)
    if level > 1.0:
        # "Volume to 300" was quietly clamped to 100% - full blast on a level
        # that cannot mean anything. Let the model ask what was meant.
        return None
    action = "brightness" if "bright" in what else "volume"
    return "peripherals", {"action": action, "level": level}, f"peripherals.{action}"


def _step_reflex(m: re.Match) -> tuple[str, dict, str]:
    what = (m.group("what") or m.group("what2") or m.group("what3") or "").lower()
    direction = (m.group("dir") or m.group("dir2") or m.group("dir3") or "").lower()
    amount = (m.group("amt") or m.group("amt2") or "").lower()
    args: dict = {"action": ("brightness_" if "bright" in what else "volume_") + direction}
    number = re.search(r"\d{1,2}", amount)
    if number:
        args["level"] = int(number.group(0)) / 100.0
    elif amount in {"a lot", "loads"}:
        args["level"] = 0.25
    elif amount:
        args["level"] = 0.05
    return "peripherals", args, "peripherals." + args["action"]


def _sing(m: re.Match) -> tuple[str, dict, str]:
    from .singing import sing_args
    return "sing", sing_args(m.group(0)), "sing"


# Each rule: (compiled anchored pattern, builder(match) -> (tool, args, why)).
# Patterns are deliberately strict — they describe a COMMAND, not a conversation.
# Every rule accepts any of . ? ! as its terminator: the punctuation on a spoken
# command is chosen by the transcriber, not the speaker, so treating "log my
# mood?" as a different utterance from "log my mood." would drop real commands.
_RULES: list[tuple[re.Pattern, Callable[[re.Match], tuple[str, dict, str]]]] = [
    # focus — "start a [25 minute] {pomodoro|deep work|focus|long|sprint} [block] [on X]"
    (re.compile(
        r"^(?:start|begin|kick off)\s+(?:a\s+|an\s+)?"
        r"(?:(?P<min>\d{1,3})\s*(?:min|mins|minute|minutes|m)\s+)?"
        r"(?P<preset>pomodoro|deep[- ]?work(?:\s+(?:block|session))?|"
        r"deep\s+(?:block|session)|long\s+(?:block|session)|"
        r"focus(?:\s+(?:block|session))?|sprint)"
        r"(?:\s+(?:on|for|about)\s+(?P<label>.+?))?[?.!]?$", re.I), _focus_start),
    (re.compile(r"^(?:how (?:long|much time)\s+(?:is\s+)?(?:left|remaining)"
                r"(?:\s+(?:on\s+)?(?:my\s+)?focus)?|focus status)[?.!]?$", re.I),
     lambda m: ("focus", {"action": "status"}, "focus.status")),
    (re.compile(r"^(?:end|finish|stop)\s+(?:my\s+|the\s+)?focus(?:\s+(?:block|session))?[?.!]?$", re.I),
     lambda m: ("focus", {"action": "done"}, "focus.done")),

    # study — "quiz me", "review my cards", "what's due"
    (re.compile(r"^(?:quiz me|test me|review(?:\s+my)?\s+(?:cards|flashcards)|"
                r"start (?:a\s+)?review)[?.!]?$", re.I),
     lambda m: ("study", {"action": "review"}, "study.review")),
    (re.compile(r"^(?:what'?s\s+due(?:\s+for\s+review)?|study (?:stats|progress)|"
                r"how many (?:cards|flashcards) (?:are\s+)?due)[?.!]?$", re.I),
     lambda m: ("study", {"action": "stats"}, "study.stats")),

    # daily brief
    (re.compile(r"^(?:catch me up|situation report|brief me|"
                r"where do (?:things|we) stand|what'?s (?:new|the situation))[?.!]?$", re.I),
     lambda m: ("catch_up", {"action": "report"}, "catch_up")),

    # finance — runway / balance
    (re.compile(r"^(?:what'?s\s+)?(?:my\s+)?runway(?:\s+(?:like|now))?[?.!]?$", re.I),
     lambda m: ("finance", {"action": "runway"}, "finance.runway")),
    (re.compile(r"^how (?:much|long) (?:is\s+)?(?:my\s+)?runway[?.!]?$", re.I),
     lambda m: ("finance", {"action": "runway"}, "finance.runway")),
    (re.compile(r"^(?:what'?s\s+)?(?:my\s+)?(?:cash|balance|liquid cash)(?:\s+(?:left|now))?[?.!]?$", re.I),
     lambda m: ("finance", {"action": "balance"}, "finance.balance")),

    # capability health — "what can you do", "why can't you…"
    (re.compile(r"^(?:what can you (?:actually\s+)?do|what are you capable of|"
                r"capabilit(?:y|ies)(?:\s+health)?|run (?:a\s+)?capability check)[?.!]?$", re.I),
     lambda m: ("diagnostics", {"action": "capabilities"}, "diagnostics.capabilities")),

    # token usage
    (re.compile(r"^(?:token usage|how many tokens(?:\s+(?:have\s+)?i\s+used)?|"
                r"how much (?:usage|is left|do i have left))[?.!]?$", re.I),
     lambda m: ("token_usage", {}, "token_usage")),

    # ── Mark XXVI expansion ──────────────────────────────────────────────────
    # Each of these previously cost a full model round-trip (1-3 s) to reach a
    # tool call that was never in doubt. The patterns stay strictly anchored:
    # a reflex that fires on a conversational sentence is far worse than one
    # that misses, because the model fall-through is always correct and only
    # slower.

    # wellbeing
    (re.compile(r"^(?:how(?:'?s| is| am i)\s+(?:my\s+)?(?:wellbeing|sleeping|sleep|mood)"
                r"(?:\s+(?:lately|recently|been|doing|going))?|"
                r"wellbeing (?:trend|report|summary))[?.!]?$", re.I),
     lambda m: ("wellbeing", {"action": "trend"}, "wellbeing.trend")),
    (re.compile(r"^(?:wellbeing (?:check ?in|today)|"
                r"(?:log|record) (?:my |a )?(?:check ?in|mood))[?.!]?$", re.I),
     lambda m: ("wellbeing", {"action": "checkin"}, "wellbeing.checkin")),

    # decisions — the calibration loop
    (re.compile(r"^(?:(?:what|which) decisions (?:are\s+)?due|decisions due|"
                r"any decisions to (?:resolve|score))[?.!]?$", re.I),
     lambda m: ("decision", {"action": "due"}, "decision.due")),
    (re.compile(r"^(?:(?:my |show (?:me )?(?:my )?)?calibration(?:\s+(?:score|curve|report))?|"
                r"how (?:well )?calibrated am i|(?:my )?brier score)[?.!]?$", re.I),
     lambda m: ("decision", {"action": "calibration"}, "decision.calibration")),

    # plugins
    (re.compile(r"^(?:(?:list|show) (?:my |the )?plugins|what plugins"
                r"(?:\s+(?:do you have|are (?:installed|there)))?)[?.!]?$", re.I),
     lambda m: ("plugin", {"action": "list"}, "plugin.list")),
    (re.compile(r"^(?:plugin (?:doctor|health|check)|"
                r"(?:are|is) (?:my |the )?plugins? (?:ok|healthy|working))[?.!]?$", re.I),
     lambda m: ("plugin", {"action": "doctor"}, "plugin.doctor")),

    # responsiveness — the new latency view
    (re.compile(r"^(?:why (?:were|are) you (?:so\s+)?(?:slow|laggy|lagging|stuttering)|"
                r"(?:are you|were you) lagging|"
                r"(?:show (?:me )?(?:your )?)?(?:latency|performance) (?:report|stats)|"
                r"latency report)[?.!]?$", re.I),
     lambda m: ("diagnostics", {"action": "latency"}, "diagnostics.latency")),

    # the black box — "why did you crash?" reads the crash reports back
    (re.compile(r"^(?:orion[,\s]+)?(?:why did you (?:just\s+)?(?:crash|close|shut down unexpectedly)|"
                r"(?:did|have) you (?:just\s+)?crash(?:ed)?|"
                r"(?:show (?:me )?(?:your )?)?crash (?:reports?|log))[?.!]?$", re.I),
     lambda m: ("diagnostics", {"action": "crashes"}, "diagnostics.crashes")),

    # media — only when the MUSIC is named: a bare "pause", "stop" or "mute"
    # is ORION's own voice and microphone, handled before any reflex.
    (re.compile(r"^(?:orion[,\s]+)?(?:pause|stop) (?:the |my )?(?:music|song|track|spotify|playback)[?.!]?$", re.I),
     lambda m: ("media_control", {"action": "pause"}, "media_control.pause")),
    (re.compile(r"^(?:orion[,\s]+)?(?:resume|unpause|continue) (?:the |my )?(?:music|song|track|spotify|playback)[?.!]?$|"
                r"^(?:orion[,\s]+)?play (?:some |my |the )?music[?.!]?$", re.I),
     lambda m: ("media_control", {"action": "play"}, "media_control.play")),
    (re.compile(r"^(?:orion[,\s]+)?(?:skip|next)(?: (?:this|the))? (?:song|track)[?.!]?$|"
                r"^(?:orion[,\s]+)?skip (?:this|it)[?.!]?$", re.I),
     lambda m: ("media_control", {"action": "next"}, "media_control.next")),
    (re.compile(r"^(?:orion[,\s]+)?(?:previous|last) (?:song|track)[?.!]?$|"
                r"^(?:orion[,\s]+)?go back a (?:song|track)[?.!]?$", re.I),
     lambda m: ("media_control", {"action": "previous"}, "media_control.previous")),
    (re.compile(r"^(?:orion[,\s]+)?what(?:'s| is) (?:playing|on spotify|playing (?:right )?now|"
                r"this (?:track|song) (?:that's |that is )?playing)[?.!]?$", re.I),
     lambda m: ("media_control", {"action": "now_playing"}, "media_control.now_playing")),

    # the everyday camera: a photo, a page, a code (camera_tools.py)
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:take|snap|grab)\s+(?:a\s+|me\s+a\s+)?"
                r"(?:photo|picture|pic|selfie|snapshot)(?:\s+(?:with|on)\s+(?:the\s+)?(?:camera|webcam))?"
                r"(?:\s+in\s+(?P<delay>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
                r"fifteen)\s+seconds?)?"
                r"(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("vision_analyse",
                {"action": "photo", **({"delay": _SMALL_NUMBERS.get(m.group("delay").lower(), None)
                                        or int(m.group("delay"))} if m.group("delay") else {})},
                "vision.photo")),
    # "scan" only: "read this page / my letter" already means the window
    # on screen (vision_analyse read_window), not a page held to the camera.
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?scan\s+(?:this|the|my|a)\s+"
                r"(?:document|page|letter|paper|receipt|form)(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("vision_analyse", {"action": "document"}, "vision.document")),
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:read|scan)\s+(?:this|the|that)\s+"
                r"(?:qr\s*code|qr|barcode|bar\s+code)(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("vision_analyse", {"action": "read_code"}, "vision.read_code")),

    # something NAMED, on a named service — or the user's own playlist
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:play|put on|put)\s+(?P<q>.{2,80}?)\s+on\s+"
                r"(?P<svc>spotify|youtube)(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("media_control", {"action": "play", "query": m.group("q").strip(),
                                  "service": m.group("svc").lower()}, "media_control.play_named")),
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:play|put on)\s+my\s+(?P<q>.{1,60}?)\s+playlist"
                r"(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("media_control", {"action": "play", "query": f"my {m.group('q').strip()} playlist"},
                "media_control.play_playlist")),

    # the PC's volume and screen brightness — only when the volume, sound or
    # brightness is NAMED: "louder" alone may be about ORION's own voice.
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:set|put|change|turn|bring)\s+(?:the\s+|my\s+)?"
                r"(?P<what>volume|sound|brightness|screen brightness)\s+(?:up\s+|down\s+)?(?:to|at)\s+"
                + _LEVEL + r"(?:\s*,?\s*please)?[?.!]?$|"
                r"^(?:orion[,\s]+)?(?P<what2>volume|brightness)\s+(?:to\s+)?" + _LEVEL.replace("lvl", "lvl2")
                + r"[?.!]?$", re.I),
     lambda m: _level_reflex(m)),
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:turn|put)\s+(?:the\s+|my\s+)?(?P<what>volume|sound|brightness)\s+"
                r"(?P<dir>up|down)" + _AMOUNT + r"(?:\s*,?\s*please)?[?.!]?$|"
                r"^(?:orion[,\s]+)?(?:please\s+)?turn\s+(?P<dir2>up|down)\s+(?:the\s+|my\s+)?(?P<what2>volume|sound|brightness)"
                + _AMOUNT.replace("amt", "amt2") + r"(?:\s*,?\s*please)?[?.!]?$|"
                r"^(?:orion[,\s]+)?(?P<what3>volume|brightness)\s+(?P<dir3>up|down)[?.!]?$", re.I),
     lambda m: _step_reflex(m)),
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:(?:dim|darken)\s+(?:the\s+|my\s+)?(?:screen|display|monitor)|"
                r"make\s+(?:the\s+|my\s+)?(?:screen|display|monitor)\s+(?P<adj>brighter|dimmer|darker))"
                r"(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("peripherals",
                {"action": "brightness_up" if (m.group("adj") or "").lower() == "brighter"
                 else "brightness_down"}, "peripherals.brightness_step")),
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?P<verb>mute|unmute)\s+(?:the\s+|my\s+)?"
                r"(?:sound|volume|speakers?|audio|pc|computer|laptop)(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("peripherals", {"action": m.group("verb").lower()}, "peripherals." + m.group("verb").lower())),
    (re.compile(r"^(?:orion[,\s]+)?(?:what(?:'s| is) (?:the |my )?(?P<what>volume|brightness)(?: (?:at|set to|on))?|"
                r"how (?:loud|high) is (?:the |my )?volume(?: set)?|how bright is (?:the |my )?screen)[?.!]?$", re.I),
     lambda m: ("peripherals",
                {"action": "brightness_status" if "bright" in m.group(0).lower() else "volume_status"},
                "peripherals.status")),

    # weather — the plain questions, answered from the forecast with no model
    # turn. Anchored: an optional place ("in Leeds") and an optional time
    # ("tomorrow"), nothing else, so "I love this weather" or "what's the
    # weather like in your opinion" stay conversation. The whole question
    # goes along as `when`, so "do I need a coat" leads with the advice.
    (re.compile(r"^(?:orion[,\s]+)?(?:"
                r"(?:what(?:'s|s| is)|how(?:'s|s| is))\s+the\s+weather(?:\s+(?:like|looking|doing))?"
                r"|(?:what(?:'s|s| is)|give me)\s+the\s+(?:weather\s+)?forecast"
                r"|what(?:'s|s| is)\s+the\s+temperature"
                r"|(?:is\s+it|will\s+it|is\s+it\s+going\s+to|it\s+is\s+going\s+to)\s+(?:be\s+)?"
                r"(?:rain(?:ing)?|snow(?:ing)?|cold|hot|warm|sunny|windy|wet|dry)"
                r"|(?:do|will|should)\s+i\s+(?:need|take|bring|wear)\s+(?:a|an|my)\s+"
                r"(?:coat|jacket|umbrella|brolly|raincoat|jumper)"
                r")"
                r"(?:\s+(?:for\s+)?(?P<when1>today|tonight|tomorrow|later(?: on)?|now|right now|outside|"
                r"out there|this (?:morning|afternoon|evening|weekend|week)|the weekend|"
                r"(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)))?"
                r"(?:\s+(?:in|for|at|over)\s+(?!(?:your|my|general|here|which|what|that|the moment|today|"
                r"tonight|tomorrow|later|this|next|the weekend|(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b)"
                r"(?P<place>[a-z][a-z .'-]{1,40}?))?"
                r"(?:\s+(?:for\s+)?(?P<when2>today|tonight|tomorrow|later(?: on)?|now|this (?:morning|afternoon|"
                r"evening|weekend|week)|the weekend|(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|"
                r"saturday|sunday)))?"
                r"(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("geo", {"action": "weather", "query": (m.group("place") or "").strip(),
                        "when": m.group(0)}, "geo.weather")),

    # nearby places — only kinds places.py maps to OpenStreetMap tags, so
    # "find me a job nearby" or "where's the nearest exit" stay conversation.
    (re.compile(r"^(?:orion[,\s]+)?(?:"
                r"where(?:'s| is) (?:the\s+)?(?:nearest|closest)\s+(?P<w1>" + _PLACE_KINDS + r")|"
                r"(?:find|show)(?:\s+me)?\s+(?:a|an|the|some)?\s*(?:nearby\s+|local\s+|nearest\s+|closest\s+)?"
                r"(?P<w2>" + _PLACE_KINDS + r")"
                r"(?:\s+(?:nearby|near me|near here|around here|close by))?|"
                r"is there an?\s+(?P<w3>" + _PLACE_KINDS + r")\s+(?:nearby|near me|near here|around here|close by)"
                r")(?:\s+(?:that's\s+)?open(?:\s+now)?)?[?.!]?$", re.I),
     lambda m: ("geo", {"action": "places",
                        "what": (m.group("w1") or m.group("w2") or m.group("w3")).strip()},
                "geo.places")),

    # timers — "set a timer for ten minutes", "a 5 minute timer", "timer 30 seconds".
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:(?:set|start|put on|give me)\s+)?(?:an?\s+|the\s+)?"
                r"(?:(?P<n1>\d{1,3}|" + "|".join(sorted(_TIMER_WORDS, key=len, reverse=True)) + r")"
                r"[\s-]+(?P<u1>seconds?|secs?|minutes?|mins?|hours?|hrs?)[\s-]+timer|"
                r"timer\s+(?:for\s+)?(?P<n2>\d{1,3}|" + "|".join(sorted(_TIMER_WORDS, key=len, reverse=True))
                + r")(?:\s+(?P<u2>seconds?|secs?|minutes?|mins?|hours?|hrs?))?|"
                r"timer\s+for\s+(?P<half>half an hour|an hour and a half|an hour))"
                r"(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("reminder", _timer_args(m), "reminder.timer")),

    # the markets in one answer (morning_briefing section=...), not a briefing.
    (re.compile(r"^(?:orion[,\s]+)?"
                # "what are markets" / "what is crypto" ask for a definition
                r"(?!what(?:'s|s| is| are)\s+(?:the\s+)?(?:stock\s*)?(?:markets?|stocks|shares|"
                r"crypto(?:currency)?|bitcoin)[?.!]?$)"
                r"(?:how(?:'s|s| is| are)|what(?:'s|s| is| are))\s+(?:the\s+)?"
                r"(?P<what>stock\s*market|stock\s*markets|markets?|shares|stocks|ftse(?:\s*100)?|"
                r"s\s*(?:and|&)\s*p(?:\s*500)?|nasdaq|dow(?:\s*jones)?|crypto(?:currency)?|bitcoin|"
                r"price of bitcoin|bitcoin price)"
                r"(?:\s+(?:doing|looking|at|trading))?(?:\s+(?:today|right now|now|this morning))?[?.!]?$", re.I),
     lambda m: ("morning_briefing",
                {"section": "crypto" if re.search(r"crypto|bitcoin", m.group("what"), re.I)
                 else "stocks" if re.search(r"ftse|s\s*(?:and|&)\s*p|nasdaq|dow|shares|stocks",
                                            m.group("what"), re.I)
                 else "markets"},
                "briefing.markets")),

    # sunrise / sunset — from the same forecast (weather.sun_times).
    (re.compile(r"^(?:orion[,\s]+)?(?:"
                r"what time (?:is|'s) (?:the\s+)?(?:sun ?rise|sun ?set)|when(?:'s| is) (?:the\s+)?(?:sun ?rise|sun ?set)|"
                r"(?:what time|when) does (?:it get (?:dark|light)|the sun (?:go down|come up|set|rise))|"
                r"what time does the sun (?:go down|come up|set|rise)"
                r")(?:\s+(?P<when>today|tonight|tomorrow|on (?:monday|tuesday|wednesday|thursday|friday|"
                r"saturday|sunday)))?"
                r"(?:\s+(?:in|at)\s+(?!your\b|my\b|here\b)(?P<place>[a-z][a-z .'-]{1,40}?))?"
                r"(?:\s+(?P<when2>today|tonight|tomorrow))?[?.!]?$", re.I),
     lambda m: ("geo", {"action": "weather", "query": (m.group("place") or "").strip(),
                        "when": m.group(0)}, "geo.sun")),

    # radios and the screen — "turn the wifi off", "switch bluetooth on",
    # "turn off the screen". The on/off word is required, so "is the wifi
    # working" or "the screen is flickering" stay with the model.
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:can you\s+|could you\s+)?(?:turn|switch|put)\s+"
                r"(?:(?P<s1>on|off)\s+(?:the\s+|my\s+)?(?P<r1>wi-?fi|wireless|bluetooth)|"
                r"(?:the\s+|my\s+)?(?P<r2>wi-?fi|wireless|bluetooth)\s+(?P<s2>on|off))"
                r"(?:\s+for me)?(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("peripherals",
                {"action": "bluetooth" if "blue" in (m.group("r1") or m.group("r2")).lower() else "wifi",
                 "state": (m.group("s1") or m.group("s2")).lower()},
                "peripherals.radio")),
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:can you\s+|could you\s+)?(?:"
                r"(?:turn|switch|put)\s+off\s+(?:the\s+|my\s+)?(?:screens?|monitors?|displays?)|"
                r"(?:turn|switch|put)\s+(?:the\s+|my\s+)?(?:screens?|monitors?|displays?)\s+off|"
                r"(?:put|send)\s+(?:the\s+|my\s+)?(?:screens?|monitors?|displays?)\s+to\s+sleep)"
                r"(?:\s+for me)?(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("peripherals", {"action": "screen_off"}, "peripherals.screen_off")),

    # the capability-gap ledger — what has been asked for and not delivered
    (re.compile(r"^(?:orion[,\s]+)?(?:what (?:can't|cannot|can not) you do(?: yet)?|"
                r"what (?:have i|did i) asked? (?:for )?(?:that )?you (?:couldn't|could not|can't) do|"
                r"what (?:are|were) you unable to do|(?:show (?:me )?)?(?:your )?capability gaps)[?.!]?$", re.I),
     lambda m: ("diagnostics", {"action": "gaps"}, "diagnostics.gaps")),

    # appearance — "go into your orb form", "switch to your classic face".
    # Mark XXXVII offers only the Screen Face, but these phrases are still
    # Mark XXXX.1: how long he waits before answering. "I want to be able to
    # finish my sentences" (owner, 27 Oct 2026). One rule, both directions.
    (re.compile(r"^(?:orion[,\s]+)?(?:please\s+)?(?:(?P<slow>give me (?:a (?:bit|little) )?(?:more|some more) time"
                r"(?: to finish(?: my sentences?)?| before you (?:reply|answer|respond))?"
                r"|(?:stop|don'?t|do not) (?:cut(?:ting)? me off|jump(?:ing)? in)"
                r"|wait (?:a (?:bit|little) )?longer before (?:you )?(?:reply|replying|answer|answering|respond|responding)"
                r"|let me finish(?: my sentences?)?(?: before you (?:reply|answer))?)"
                r"|(?P<fast>(?:reply|answer|respond) (?:faster|quicker|more quickly)"
                r"|be (?:faster|quicker) to (?:reply|answer|respond)))(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("elevenlabs_voice", {"action": "reply_pause",
                                     "pause": "patient" if m.group("slow") else "brisk"},
                "voice.reply_pause")),
    # Mark XXXX security: the audit, the phone master switch, Downloads checked.
    (re.compile(r"^(?:orion[,\s]+)?(?:how (?:secure|safe|protected) is (?:my|this) "
                r"(?:computer|pc|laptop|machine|system)|(?:run|do) (?:a )?security (?:audit|check)"
                r"|security audit)[?.!]?$", re.I),
     lambda m: ("security_watch", {"action": "audit"}, "security.audit")),
    (re.compile(r"^(?:orion[,\s]+)?(?:(?P<off>stop everything (?:from|on) (?:my|the) phone"
                r"|(?:turn|switch) (?:off )?phone control(?: off)?|phone control off"
                r"|block (?:my|the) phone)"
                r"|(?P<on>(?:turn|switch) phone control (?:back )?on"
                r"|phone control on|allow (?:my|the) phone again))[?.!]?$", re.I),
     lambda m: ("security_watch", {"action": "phone_off" if m.group("off") else "phone_on"},
                "security.phone_off" if m.group("off") else "security.phone_on")),
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:scan|check) my downloads(?: folder)?(?: for (?:viruses|malware))?)"
                r"[?.!]?$", re.I),
     lambda m: ("antivirus", {"action": "downloads"}, "antivirus.downloads")),
    # Mark XXXX: "sing me a song", "can you sing happy birthday to Sam". A model
    # answering it recited the lyrics in its speaking voice (27 Oct 2026); the
    # sing tool is what carries a melody.
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+(?:please\s+)?|please\s+)?"
                r"sing\b(?!\s+along)(?:\s+.{0,70})?$", re.I),
     _sing),
    # caught here ON PURPOSE: interface_control answers a form this version
    # does not have with "I only have my Screen Face in this version." rather
    # than the sentence reaching a model that might promise an orb.
    # Asked for when somebody else has just walked in, so it must not wait on
    # a model round-trip.
    # Mark XXXIX: the Screen Face's DESIGNS - "switch your face to arc", "use
    # your visor face", "go back to your old face" (the Companion look). Before
    # the form switch, which reads "old face" as the retired classic face.
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+|please\s+)?(?:"
                r"(?:change|switch|set|turn)\s+your\s+face(?:\s+design)?\s+(?:back\s+)?to\s+"
                r"(?:the\s+|your\s+)?(?P<a>" + _DESIGN + r")(?:\s+(?:one|design|face|look))?"
                r"|(?:use|wear|try|put on|go to|go back to|switch to|change to|show me)\s+"
                r"(?:your\s+|the\s+|a\s+)?(?P<b>" + _DESIGN + r")\s+(?:face|design|look|eyes)"
                r"(?:\s+design)?"
                # Mark XXXX.1: "next face design", "try another look", "change your face".
                r"|(?:show me |try |use |wear |switch to )?(?:your |a )?(?P<n>next|another|different)"
                r" (?:face(?: design)?|design|look)|(?P<c>change) your (?:face|look)"
                r")(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("interface_control",
                {"action": "face_design",
                 "target": (m.group("a") or m.group("b") or ("next" if m.group("n") or m.group("c")
                                                             else "")).lower()},
                "interface.face_design")),
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+|please\s+)?(?:"
                r"(?:go|switch|change|turn|morph|transform|shift)(?:\s+back)?\s+(?:in\s*)?to\s+(?:your\s+)?(?:the\s+)?(?P<a>" + _FORM + r")(?:\s+(?:form|mode|shape|version))?"
                r"|(?:use|be|become|wear)\s+(?:your\s+|an?\s+|the\s+)?(?P<d>" + _FORM + r")(?:\s+(?:form|mode|shape))?"
                r"|(?P<b>" + _FORM + r")\s+(?:form|mode)"
                r"|(?:show|give)\s+me\s+(?:your\s+)?(?P<c>" + _FORM + r")(?:\s+(?:form|mode))?"
                r")(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("interface_control",
                {"action": "face_form",
                 "target": (m.group("a") or m.group("b") or m.group("c")
                            or m.group("d") or "").lower()},
                "interface.face_form")),

    # the smart home (smart_home.py) — "turn off the bedroom lights", "lights
    # on", "dim the lights to 30 percent", "goodnight routine".
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+|please\s+)?"
                r"(?:turn|switch|put)\s+(?P<s>on|off)\s+(?:the\s+|my\s+)?(?P<t>[a-z ]{0,30}?)"
                r"\s*(?:lights?|lamps?)(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("smart_home", {"action": m.group("s").lower(),
                               "target": (m.group("t").strip() + " lights").strip()
                               if m.group("t").strip() else "all lights"},
                "home.power")),
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+|please\s+)?"
                r"(?:turn|switch|put)\s+(?:the\s+|my\s+)?(?P<t>[a-z ]{0,30}?)\s*(?:lights?|lamps?)\s+"
                r"(?P<s>on|off)(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("smart_home", {"action": m.group("s").lower(),
                               "target": (m.group("t").strip() + " lights").strip()
                               if m.group("t").strip() else "all lights"},
                "home.power")),
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+|please\s+)?"
                r"(?:dim|set|turn)\s+(?:the\s+|my\s+)?(?P<t>[a-z ]{0,30}?)\s*(?:lights?|lamps?)\s+"
                r"(?:down\s+)?to\s+(?P<v>\d{1,3})\s*(?:%|percent)(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("smart_home", {"action": "brightness", "value": m.group("v"),
                               "target": (m.group("t").strip() + " lights").strip()
                               if m.group("t").strip() else "all lights"},
                "home.brightness")),
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+|please\s+)?"
                r"(?:(?:run|start|do|activate)\s+(?:the\s+|my\s+)?)?(?P<r>goodnight|good night|morning|"
                r"movie|away|bedtime)\s+routine(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("smart_home", {"action": "routine", "target": m.group("r").lower()},
                "home.routine")),

    # how he sounds (voice_settings) — "go back to your normal voice", "sound
    # professional again". Offline AND online voice change together.
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+|please\s+)?(?:"
                r"(?:use|switch to|change to|go back to|go to|put on|speak in|talk in|be in|"
                r"try|give me)\s+(?:your\s+|the\s+|a\s+|my\s+)?(?P<w>(?:professional|formal|serious|"
                r"classic|normal|default|original|usual|old|deep)(?:\s+(?:male|boy|female|girl))?)\s+voice"
                r"|sound\s+(?P<n>normal|professional|like yourself|like before)(?:\s+again)?"
                r"|(?:change|switch|put|turn)\s+your\s+voice\s+(?P<b>back)(?:\s+to\s+"
                r"(?:normal|how it was|what it was|the old one))?"
                r")(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("elevenlabs_voice",
                {"action": "voice_style",
                 "style": ("classic" if m.group("n") or m.group("b")
                           else m.group(0).lower())},       # apply_style reads the words
                "voice.style")),

    # the dock — he is not on the screen edge at start; this is how he is asked
    # for ("show your dock") and asked to go again ("hide your dock").
    (re.compile(r"^(?:orion[,\s]+)?(?:(?:can|could|would|will) you\s+|please\s+)?(?:"
                r"(?P<s>(?:show|bring up|bring out|open|put up|display|summon)\s+(?:me\s+)?(?:your\s+|the\s+|my\s+)?dock(?:ed (?:screen )?face)?"
                r"|dock yourself|dock your (?:screen )?face)"
                r"|(?P<h>(?:hide|close|dismiss|put away|take away|remove|get rid of)\s+(?:your\s+|the\s+|my\s+)?dock(?:ed (?:screen )?face)?"
                r"|undock yourself)"
                r")(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("interface_control",
                {"action": "dock", "target": "hide" if m.group("h") else "show"},
                "interface.dock")),

    # camera — "show me the camera" is a request to SEE, not to be told.
    # It must be matched BEFORE the page rule below, which would otherwise
    # swallow "show me the camera lab" as ordinary navigation. Note what is
    # deliberately absent: anything resembling "what can you see" or "what's
    # in front of me", which are questions for vision_analyse and belong to
    # the model, not to a regex.
    (re.compile(r"^(?:(?:can you\s+)?(?:show|bring|pull|put|open)"
                r"(?:\s+me)?(?:\s+up)?\s+(?:the\s+|my\s+)?(?:live\s+)?"
                r"(?:camera|webcam)(?:\s+(?:lab|feed|view))?|"
                r"camera\s+lab|let me see the camera|"
                r"(?:can you\s+|could you\s+|please\s+)?(?:turn|switch|put)\s+on\s+(?:the\s+|my\s+)?(?:camera|webcam)|"
                r"(?:can you\s+|could you\s+|please\s+)?(?:turn|switch|put)\s+(?:the\s+|my\s+)?(?:camera|webcam)\s+on"
                r")(?:\s+for me)?(?:\s*,?\s*please)?[?.!]?$", re.I),
     lambda m: ("interface_control", {"action": "camera"}, "interface.camera")),

    # deck navigation — the trailing "page"/"deck"/"tab" is what keeps this
    # anchored. Without it "show me the memory" is a request for a memory,
    # not for a page, and a reflex firing on a conversational sentence is far
    # worse than one that misses. The tool validates the name against the real
    # page list, so a misheard target is reported rather than silently ignored.
    (re.compile(r"^(?:open|go to|show me|switch to|take me to|bring up)\s+"
                r"(?:the\s+)?(?P<name>[a-z0-9][a-z0-9 ]{1,28}?)\s+"
                r"(?:page|deck|tab)[?.!]?$", re.I),
     lambda m: ("interface_control",
                {"action": "page", "target": m.group("name").strip()},
                "interface.page")),

    # his chess rating — the only Elo he has. Asked "what's your Elo?" he used
    # to answer as if the word meant nothing; the figures live in the chess
    # tool, so the question goes straight there. Anchored at both ends and
    # limited to HIS rating: "rate this film" or "what's the Elo system?" are
    # conversation, not this.
    (re.compile(r"^(?:(?:hey\s+)?orion[,]?\s+)?(?:"
                r"(?:what(?:'s|s| is)|tell me)\s+(?:your|orion'?s)\s+(?:current\s+)?"
                r"(?:chess\s+)?(?:e\.?\s?l\.?\s?o\.?(?:\s+rating)?|rating)(?:\s+now)?"
                r"|(?:what|which)\s+(?:e\.?\s?l\.?\s?o\.?|rating)\s+are\s+you(?:\s+at\s+chess)?(?:\s+now)?"
                r"|how\s+(?:good|strong)\s+are\s+you\s+at\s+chess(?:\s+now)?"
                r"|(?:your|orion'?s)\s+(?:chess\s+)?(?:e\.?\s?l\.?\s?o\.?|rating)"
                r")\s*[?.!]?$", re.I),
     lambda m: ("chess", {"action": "rating"}, "chess.rating")),

    # self-diagnostic
    (re.compile(r"^(?:run (?:a )?(?:self[- ])?diagnostics?(?:\s+check)?|"
                r"(?:self[- ])?diagnostics?|system check|check yourself)[?.!]?$", re.I),
     lambda m: ("diagnostics", {"action": "full"}, "diagnostics.full")),
]


#: How transcribers actually hear the name at the start of a command: "Ryan,
#: pause the music", "Orin, ..." (which correct_identity then writes as
#: "O.R.I.O.N., ..."). The rules accept "orion," - so the variants become it;
#: before this, an addressed command matched no reflex at all.
_VOCATIVE = re.compile(
    r"^(?:o\.r\.i\.o\.n\.|(?:orion|o'?ryan|oh ryan|ryan|orin|orien|orian|oreon|arion)\b)[,.]?\s+",
    re.I)


def match_reflex(text: str) -> ReflexMatch | None:
    """The first high-precision reflex the text matches, or None (→ model)."""
    raw = (text or "").strip()
    t = _VOCATIVE.sub("orion, ", raw, count=1)
    if not t or len(t) > 90:            # a long utterance is a conversation, not a command
        return None
    # Addressed or not, the command is the same: rules without an "orion,"
    # prefix of their own still see "Ryan, turn the camera on" as the command.
    candidates = [t]
    if t != raw or t.lower().startswith("orion, "):
        bare = _VOCATIVE.sub("", raw, count=1).strip()
        if bare:
            candidates.append(bare)
    for candidate in candidates:
        for pattern, build in _RULES:
            m = pattern.match(candidate)
            if m:
                built = build(m)
                if built is None:            # a rule can decline (common sense)
                    continue
                tool, args, why = built
                if args is None:
                    continue
                return ReflexMatch(tool, args, why)
    for candidate in reversed(candidates):          # the bare expression first
        if _is_exact_calculation(candidate):
            return ReflexMatch("calculate", {"expression": candidate}, "calculate.exact")
    return None


_WORDS = re.compile(r"[a-z]+")


def match_reflex_forgiving(text: str) -> ReflexMatch | None:
    """match_reflex, and - when nothing matched - the same command with its
    typing slips put right ("colse spotify", "pasue the music"; spelling.py).

    A corrected match is taken only when every corrected word is a COMMAND
    word - one the rule needs spelt just so ("pause", "volume"). A word the
    rule would take ANY word in place of is an argument - a song, a name, a
    search - and may be a real word ORION has never seen ("plya stormzy" is
    not "play stormy"); acting on that guess is what this layer must never
    do, so the turn goes to the model. Measured on the reflex tests' own
    commands with one slip each: 1/48 handled locally before, 40/48 after,
    no wrong argument, no conversational line newly matched."""
    found = match_reflex(text)
    if found is not None:
        return found
    try:
        from .spelling import correct
        fixed = correct(text)
    except Exception:
        return None
    if fixed == text:
        return None
    found = match_reflex(fixed)
    if found is None:
        return None
    corrected = set(_WORDS.findall(fixed.lower())) - set(_WORDS.findall(text.lower()))
    return None if any(_takes_any_word(fixed, word) for word in corrected) else found


#: A token no rule's command vocabulary contains.
_PROBE = "zqxjv"


def _takes_any_word(text: str, word: str) -> bool:
    """True when the rule matching *text* still matches with *word* replaced
    by nonsense, and carries the nonsense into its arguments - i.e. *word*
    sits in a free-text argument rather than in the command itself."""
    probe = match_reflex(re.sub(rf"\b{re.escape(word)}\b", _PROBE, text, flags=re.I))
    return probe is not None and _PROBE in " ".join(str(v) for v in probe.args.values()).lower()


#: Pure arithmetic: digits and operators only (after an optional "what's").
_ARITHMETIC = re.compile(
    r"^(?:what(?:'s| is)\s+|calculate\s+|work out\s+)?"
    r"[\d.,\s()]*\d[\d.,\s()]*(?:[-+*/x×÷^%]|\s(?:plus|minus|times|divided by)\s)"
    r"[\d.,\s()+\-*/x×÷^%a-z]*[?=]?$", re.I)


def _is_exact_calculation(text: str) -> bool:
    """A unit conversion the calculator recognises, or plain arithmetic it can
    evaluate — answered exactly with no model turn. Anything else, however
    numeric it looks ("what happened in 1066"), goes to the model."""
    try:
        from .calculator import CalcError, chance, convert, evaluate, world_time
        if chance(text):
            return True
        converted = convert(text)
        if converted and not converted.startswith("I can't"):
            return True
        if world_time(text):
            return True
        from .dates import answer as date_answer
        if date_answer(text):
            return True
        from .currency import parse as currency_request
        if currency_request(text) is not None:
            return True
        if _ARITHMETIC.match(text):
            evaluate(text)
            return True
    except (CalcError, ArithmeticError, ValueError, TypeError):
        return False
    except Exception:
        return False
    return False


def reflex_enabled() -> bool:
    """On by default (the point is instant response); disable with ORION_REFLEX=0.
    Precision is enforced by tests, and the model fall-through is always intact."""
    return os.getenv("ORION_REFLEX", "1").strip().lower() not in {"0", "false", "off", "no"}


__all__ = ["ReflexMatch", "match_reflex", "match_reflex_forgiving", "reflex_enabled"]
