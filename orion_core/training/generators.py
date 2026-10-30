"""
Generated read-only turns: request -> call -> result -> spoken answer.

For a read-only tool the live agent hands the result back to the model and the
model speaks the answer (action tools finish without another generation).
That second step needs examples too, or a model fine-tuned on mostly "request
-> call" rows would get worse at it. These generators make many of them, with
varied cities, numbers and topics, so the model learns the habit - answer from
the result, in one or two spoken sentences, add nothing the result does not
say - and not a handful of memorised sentences.

Every number in a reply is computed from the number in the result it follows.
The tool results are written in the style the tools use but are invented: no
real weather, file or person is described.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Generated:
    text: str
    tool: str
    args: dict[str, Any]
    result: str
    reply: str


def _num(value: float) -> str:
    """12 not 12.0, 1.5 not 1.50."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


_CITIES = ("Leeds", "Manchester", "Bristol", "Glasgow", "Cardiff", "Newcastle", "Sheffield", "Liverpool",
           "Nottingham", "Edinburgh", "Norwich", "Exeter", "Brighton", "York", "Oxford", "Cambridge")
_SKIES = (("light rain", 8, 13, True), ("sunny spells", 14, 21, False), ("overcast", 7, 14, False),
          ("showers", 9, 14, True), ("clear skies", 3, 9, False), ("heavy rain", 7, 12, True),
          ("light cloud", 12, 18, False))


def weather(rng: random.Random) -> Generated:
    city = rng.choice(_CITIES)
    sky, low, high, wet = rng.choice(_SKIES)
    temp = rng.randint(low, high)
    wind = rng.choice(("light", "moderate", "breezy"))
    tomorrow = rng.random() < 0.3
    asks = (f"what's the weather like in {city}", f"weather in {city}", f"how's the weather in {city}",
            f"is it cold in {city}", f"give me the forecast for {city}") if not tomorrow else (
        f"what's the weather in {city} tomorrow", f"what will it be like in {city} tomorrow",
        f"forecast for {city} tomorrow")
    args: dict[str, Any] = {"action": "weather", "query": city}
    when = "tomorrow" if tomorrow else "now"
    if tomorrow:
        args["when"] = "tomorrow"
    result = f"{city} {when}: {temp}C, {sky}, wind {wind}."
    lead = f"Tomorrow in {city} it'll be" if tomorrow else f"It's"
    reply = f"{lead} {temp} degrees with {sky}" + (" in " + city if not tomorrow else "") + "."
    if wet:
        reply += rng.choice((" Take a coat.", " You'll want an umbrella.", " Best take a waterproof."))
    elif temp <= 5:
        reply += " It's cold, so wrap up."
    return Generated(rng.choice(asks), "geo", args, result, reply)


def arithmetic(rng: random.Random) -> Generated:
    kind = rng.choice(("percent", "times", "divide", "plus"))
    if kind == "percent":
        a, b = rng.choice((5, 10, 12, 15, 20, 25, 30, 35)), rng.choice((80, 120, 150, 240, 360, 500, 750))
        value = a * b / 100
        ask = rng.choice((f"what's {a} percent of {b}", f"work out {a} percent of {b}",
                          f"{a} percent of {b}, what is that"))
        return Generated(ask, "calculate", {"expression": f"{a}% of {b}"},
                         f"{a}% of {b} is {_num(value)}.", rng.choice(
                             (f"That's {_num(value)}.", f"It comes to {_num(value)}.")))
    if kind == "times":
        a, b = rng.randint(6, 29), rng.randint(6, 29)
        return Generated(rng.choice((f"what is {a} times {b}", f"multiply {a} by {b}")), "calculate",
                         {"expression": f"{a} * {b}"}, f"{a} * {b} = {a * b}.", f"That's {a * b}.")
    if kind == "divide":
        a, b = rng.choice((90, 144, 200, 360, 1000, 75)), rng.choice((4, 5, 6, 8, 3))
        value = round(a / b, 2)
        return Generated(rng.choice((f"what's {a} divided by {b}", f"divide {a} by {b}")), "calculate",
                         {"expression": f"{a} / {b}"}, f"{a} / {b} = {_num(value)}.",
                         f"That comes to {_num(value)}.")
    a, b = rng.randint(120, 980), rng.randint(120, 980)
    return Generated(rng.choice((f"add {a} and {b}", f"what's {a} plus {b}")), "calculate",
                     {"expression": f"{a} + {b}"}, f"{a} + {b} = {a + b}.", f"That's {a + b}.")


def resources(rng: random.Random) -> Generated:
    total = rng.choice((8, 16, 32))
    mem = rng.randint(28, 88)
    used = round(total * mem / 100, 1)
    cpu, disk = rng.randint(4, 63), rng.randint(40, 90)
    ask = rng.choice(("how much memory am I using", "what's my memory usage", "how loaded is the computer",
                      "is my ram getting full", "how's the cpu and memory looking"))
    result = f"CPU {cpu}%, memory {mem}% ({used} of {total} GB), disk {disk}% full."
    reply = (f"You're using about {used} of your {total} gigabytes of memory, {mem} percent, "
             f"and the CPU is at {cpu} percent.")
    return Generated(ask, "resource_status", {}, result, reply)


_TOPICS = ("the best budget laptops", "how to descale a kettle", "the next solar eclipse",
           "cheap train tickets to London", "how to bleed a radiator", "the history of the bicycle",
           "where to buy school shoes", "things to do in York", "how to repot a houseplant",
           "the safest way to carry a bike on a car")


def search(rng: random.Random) -> Generated:
    topic = rng.choice(_TOPICS)
    ask = rng.choice((f"search the web for {topic}", f"google {topic}", f"look up {topic} online"))
    result = (f"1. {topic[:1].upper() + topic[1:]} - example.org: A clear overview of {topic}.\n"
              f"2. A guide to {topic} - example.com: Practical advice and recent updates.")
    reply = rng.choice((
        "I found an overview on example.org and a guide on example.com. Want me to open one?",
        "There's an overview on example.org and a practical guide on example.com.",
        "Two good results: an overview on example.org and a guide on example.com. Shall I open the first?"))
    return Generated(ask, "web_search", {"query": topic}, result, reply)


_FILES = (("invoice", "pdf", "pdf"), ("cv", "docx", "word"), ("budget", "xlsx", "spreadsheet"),
          ("project plan", "docx", "word"), ("lecture notes", "pdf", "pdf"), ("tax return", "pdf", "pdf"))


def files(rng: random.Random) -> Generated:
    name, ext, kind = rng.choice(_FILES)
    ask = rng.choice((f"find my {name}", f"where's my {name}", f"locate the {name} {kind}"))
    stem = name.replace(" ", "_")
    result = f"Found 2 files: D:/Docs/{stem}_2026.{ext}; D:/Docs/old/{stem}_2025.{ext}"
    reply = rng.choice((f"I found two: the 2026 {name} in Docs and an older 2025 copy. Shall I open the newer one?",
                        f"There's a 2026 {name} in Docs and an older one from 2025."))
    return Generated(ask, "find_files", {"query": name, "type": kind}, result, reply)


def reminders_list(rng: random.Random) -> Generated:
    things = rng.sample(("ring the plumber", "send the invoice", "take the bins out", "book the dentist",
                         "pick up the parcel", "call Mum"), 2)
    times = rng.sample(("09:00", "12:30", "15:00", "17:45", "20:00"), 2)
    spoken = {"09:00": "nine this morning", "12:30": "half past twelve", "15:00": "three this afternoon",
              "17:45": "quarter to six", "20:00": "eight this evening"}
    result = f"You have 2 reminders: {things[0]} at {times[0]}; {things[1]} at {times[1]}."
    reply = (f"You've got two: {things[0]} at {spoken[times[0]]}, and {things[1]} at {spoken[times[1]]}.")
    return Generated(rng.choice(("what reminders have I got", "list my reminders", "do I have any reminders")),
                     "reminder", {"action": "list"}, result, reply)


def tokens(rng: random.Random) -> Generated:
    used = rng.randrange(12_000, 480_000, 400)
    result = f"Today: {used:,} tokens used of a 1,000,000 allowance."
    reply = f"You've used about {used // 1000} thousand tokens today, out of a million."
    return Generated(rng.choice(("how many tokens have I used", "what's my token usage today")),
                     "token_usage", {}, result, reply)


def recall(rng: random.Random) -> Generated:
    topic, amount = rng.choice((("budget", rng.choice((300, 450, 800, 1200))), ("deposit", rng.choice((500, 750, 1500)))))
    result = f"[2026-09-14] You: we agreed the {topic} is {amount} pounds."
    reply = f"You said the {topic} was {amount} pounds."
    return Generated(rng.choice((f"what did we say about the {topic}", f"remind me what we agreed on the {topic}")),
                     "recall_conversation", {"query": topic}, result, reply)


def monitors(rng: random.Random) -> Generated:
    count = rng.choice((1, 2, 3))
    sizes = ["2560x1440", "1920x1080", "1920x1080"][:count]
    result = f"{count} monitor{'s' if count > 1 else ''}: " + ", ".join(
        f"{s}{' (primary)' if i == 0 else ''}" for i, s in enumerate(sizes))
    word = {1: "one", 2: "two", 3: "three"}[count]
    reply = (f"You have {word} monitor{'s' if count > 1 else ''}, "
             + ("the main one at " + sizes[0] + "." if count == 1 else
                f"the main one at {sizes[0]} and the others at 1920 by 1080."))
    return Generated(rng.choice(("how many monitors do I have", "what resolution are my screens")),
                     "display_info", {}, result, reply)


GENERATORS: tuple[Callable[[random.Random], Generated], ...] = (
    weather, arithmetic, resources, search, files, reminders_list, tokens, recall, monitors)

#: How many rows each generator contributes at most (before de-duplication).
WEIGHTS = {"weather": 22, "arithmetic": 18, "resources": 10, "search": 10, "files": 8,
           "reminders_list": 6, "tokens": 5, "recall": 6, "monitors": 4}


def generate(rng: random.Random) -> list[Generated]:
    out: list[Generated] = []
    for fn in GENERATORS:
        out += [fn(rng) for _ in range(WEIGHTS[fn.__name__])]
    return out


__all__ = ["GENERATORS", "Generated", "generate"]
