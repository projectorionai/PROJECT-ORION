"""
Plausible tool arguments for a spoken request.

Most of ORION's recorded experience says WHICH tool answered a request, not
with what arguments: the resolver's shadow log keeps the words and the tool,
the reflex ledger keeps a phrase and an action. The routing cases are the same.
To turn those into complete training turns, the arguments are built here from
the owner's own words, by rules, and then checked against the real tool schema
(``ToolBook.validate``).

The rules are conservative on purpose. They only ever copy or lightly reshape
what the owner said - "open spotify" gives ``app_name: "Spotify"``, never a
path or a name the sentence did not contain - and where a required value is
not in the words (a flight with no date, a call with nobody named) they return
``None`` and the example is left out rather than guessed. ORION's own prompt
tells the model "never invent paths, names, numbers or recipients"; training
data must not teach the opposite.

A generic fallback picks an ``action`` word the sentence contains, from the
action names the tool's own schema lists, and fills a required free-text
parameter (``query``, ``topic``...) with the request. Examples built that way
are marked ``generic`` so they can be counted and, if wanted, excluded.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .toolcalls import ToolBook

Recipe = Callable[[str], "dict[str, Any] | None"]

_FILLER_PREFIX = re.compile(
    r"(?i)^(?:(?:hey|ok|okay|right|so|well|um+|er+)[, ]+)*(?:orion[, ]+)?"
    r"(?:(?:please|just|quickly)[, ]+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|i\s+(?:want|need)\s+you\s+to\s+|"
    r"go\s+ahead\s+and\s+|i'?d\s+like\s+you\s+to\s+|i'?d\s+like\s+to\s+)?"
    r"(?:(?:please|just|quickly)[, ]+)*")
_FILLER_SUFFIX = re.compile(
    r"(?i)(?:[, ]+(?:for me|please|thanks|thank you|now|right now|orion|if you can|will you))+[ .!?]*$")

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
    "half an": 0.5, "half a": 0.5,
}


#: Speech-to-text and typing slips seen in the routing cases. The utterance in
#: an example keeps its slip - ORION really hears them - but the arguments are
#: built from the corrected words, as a good assistant would pass them on.
SLIP_FIXES = {
    "wether": "weather", "spotfy": "spotify", "musik": "music", "pasue": "pause", "inbx": "inbox",
    "chekc": "check", "remnd": "remind", "serch": "search", "sumarise": "summarise",
    "raiting": "rating", "munny": "money", "dwn": "down", "pls": "please", "fer": "for",
}
_SLIP_RE = re.compile(r"(?i)\b(" + "|".join(map(re.escape, SLIP_FIXES)) + r")\b")


def fix_slips(text: str) -> str:
    return _SLIP_RE.sub(lambda m: SLIP_FIXES[m.group(1).lower()], str(text or ""))


_CLAUSE_SPLIT = re.compile(
    r"(?i)\s*(?:,\s*)?\b(?:and then|then|after that|afterwards|and also)\b\s*"
    r"|\s+and\s+(?=(?:email|text|message|remind|tell|send|open|search|check|make|put|add)\b)")


def first_clause(text: str) -> str:
    """The first step of a multi-part request ("find my cv and email it" -> "find my cv")."""
    head = _CLAUSE_SPLIT.split(str(text or ""), maxsplit=1)[0].strip(" ,.")
    return head or str(text or "")


def strip_filler(text: str) -> str:
    """The request without the politeness around it ("could you please ... for me")."""
    out = " ".join(str(text or "").replace("’", "'").split())
    for _ in range(2):
        out = _FILLER_PREFIX.sub("", out)
    out = _FILLER_SUFFIX.sub("", out)
    return out.strip(" ,.;:!?")


def _number(word: str) -> float | None:
    word = word.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", word):
        value = float(word)
        return int(value) if value.is_integer() else value
    value = _NUMBER_WORDS.get(word)
    if value is None:
        return None
    return int(value) if float(value).is_integer() else value


def _titled(name: str) -> str:
    words = []
    for word in name.split():
        words.append(word if (any(c.isupper() for c in word) or any(c.isdigit() for c in word))
                     else word[:1].upper() + word[1:])
    return " ".join(words)


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


def _has(text: str, *words: str) -> bool:
    low = text.lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in words)


# ── recipes ──────────────────────────────────────────────────────────────────

RECIPES: dict[str, Recipe] = {}


def recipe(*tools: str) -> Callable[[Recipe], Recipe]:
    def register(fn: Recipe) -> Recipe:
        for tool in tools:
            RECIPES[tool] = fn
        return fn
    return register


_URL = re.compile(r"(?i)\bhttps?://[^\s,;]+|\b(?:[a-z0-9-]+\.)+(?:com|co\.uk|org\.uk|org|net|io|uk|gov\.uk|dev|app|ai)\b(?:/[^\s,;]*)?")


@recipe("open_app")
def _open_app(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = (re.match(r"(?i)^(?:open|launch|start|fire up|run|bring up|pull up|load)\s+(?:up\s+)?"
                      r"(?:the\s+|my\s+)?(?P<x>.+)$", t)
             or re.match(r"(?i)^get\s+(?:the\s+|my\s+)?(?P<x>.+?)\s+(?:open|up|running)\b.*$", t))
    if not match:
        return None
    name = match.group("x").strip(" .,")
    name = re.sub(r"(?i)\s+(?:app|application|program|window)$", "", name)
    if (not name or re.search(r"(?i)\b(?:and|then|also)\b|[,;]", name) or len(name.split()) > 5
            or re.search(r"(?i)\b(?:in|on|for|to)\b", name) and len(name.split()) > 3):
        return None
    return {"app_name": _titled(name)}


@recipe("close_app")
def _close_app(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = re.match(r"(?i)^(?:close|kill|quit|exit|shut|end|terminate)\s+(?:down\s+)?"
                     r"(?:the\s+|my\s+)?(?P<x>[\w .+-]+?)(?:\s+(?:down|window|app|application))*$", t)
    if not match:
        return None
    name = match.group("x").strip()
    if not name or len(name.split()) > 4 or re.search(r"(?i)\b(?:and|then|it|this|that)\b", name):
        return None
    return {"app_name": _titled(name)}


@recipe("web_search")
def _web_search(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = (re.match(r"(?i)^(?:search|google|look)\b(?:\s+(?:online|the\s+web|the\s+internet|"
                      r"for|up|this\s+up|google|on\s+the\s+internet|on\s+google))*[:,]?\s*(?P<q>.+)$", t)
             or re.match(r"(?i)^(?:find\s+out|look\s+into)\s+(?P<q>.+)$", t))
    query = (match.group("q") if match else t).strip(" .,?")
    query = re.sub(r"(?i)^(?:for|about)\s+", "", query)
    if len(query) < 3 or len(query.split()) > 18:
        return None
    return {"query": query}


@recipe("fetch_url")
def _fetch_url(text: str) -> dict[str, Any] | None:
    found = _URL.search(text)
    if not found:
        return None
    url = found.group(0).rstrip(".,;)")
    return {"url": url if re.match(r"(?i)https?://", url) else "https://" + url}


@recipe("calculate")
def _calculate(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    t = re.sub(r"(?i)^(?:what(?:'s|\s+is)|work\s+out|calculate|how\s+much\s+is|what\s+does)\s+", "", t)
    t = t.strip(" ?.")
    if len(t) < 3 or not re.search(r"\d", t):
        return None
    return {"expression": t}


@recipe("media_control")
def _media_control(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    play = re.match(r"(?i)^play\s+(?:me\s+)?(?:some\s+)?(?P<q>.+)$", t)
    if play and not re.fullmatch(r"(?i)(?:music|something|the music|it|again)", play.group("q")):
        return {"action": "play", "query": play.group("q").strip(" .")}
    if _has(t, "skip", "next", "forward"):
        return {"action": "next"}
    if _has(t, "previous", "back", "last song", "go back"):
        return {"action": "previous"}
    if _has(t, "unpause", "resume", "play"):
        return {"action": "play"}
    if _has(t, "pause", "pasue", "hold"):
        return {"action": "pause"}
    if _has(t, "stop"):
        return {"action": "stop"}
    if _has(t, "mute"):
        return {"action": "mute"}
    if re.search(r"(?i)\bwhat(?:'s| is)\s+(?:this\s+)?(?:playing|song|track)", t):
        return {"action": "now_playing"}
    if re.search(r"(?i)\b(?:turn\s+(?:it\s+)?up|louder|volume\s+up|up\s+a\s+bit)", t):
        return {"action": "volume_up"}
    if re.search(r"(?i)\b(?:turn\s+(?:it\s+)?down|quieter|volume\s+down|down\s+a\s+(?:bit|couple))", t):
        return {"action": "volume_down"}
    return None


@recipe("peripherals")
def _peripherals(text: str) -> dict[str, Any] | None:
    t = strip_filler(text).lower()
    if re.search(r"\bunmute\b", t):
        return {"action": "unmute"}
    if re.search(r"\bmute\b", t):
        return {"action": "mute"}
    bright = re.search(r"\b(?:bright(?:ness)?|screen|display|dim)\b", t)
    if bright and re.search(r"\b(?:too bright|dim|lower|down|darker|hurt)", t):
        return {"action": "brightness_down"}
    if bright and re.search(r"\b(?:brighter|raise|up|too dark|dark)", t):
        return {"action": "brightness_up"}
    if re.search(r"\b(?:lock)\b", t):
        return {"action": "lock"}
    level = re.search(r"\bvolume\b.*?\b(?:to|at)\s+(\d{1,3})\s*(?:%|percent)?", t)
    if level:
        return {"action": "volume", "level": round(min(100, int(level.group(1))) / 100.0, 2)}
    if re.search(r"\b(?:louder|turn (?:it )?up|volume up|raise the volume|up a (?:bit|notch|little))\b", t):
        return {"action": "volume_up"}
    if re.search(r"\b(?:quieter|softer|turn (?:it )?down|volume down|lower the volume|"
                 r"down a (?:bit|notch|little|couple))\b", t):
        return {"action": "volume_down"}
    if re.search(r"\bvolume\b", t):
        return {"action": "volume_status"}
    return None


@recipe("organise_files")
def _organise_files(text: str) -> dict[str, Any] | None:
    folder = re.search(r"(?i)\b(downloads?|desktop|documents?)\b", text)
    if folder and re.search(r"(?i)\b(?:sort|tidy|organi[sz]e|clean)\b", text):
        names = {"download": "Downloads", "downloads": "Downloads", "desktop": "Desktop",
                 "document": "Documents", "documents": "Documents"}
        return {"action": "preview", "folder": names[folder.group(1).lower()]}
    return None


@recipe("reminder")
def _reminder(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    timer = re.search(
        r"(?i)\b(?:timer|alarm)\b.*?\b(?:for\s+)?(?P<n>\d+(?:\.\d+)?|half an?|an?|one|two|three|four|five|"
        r"six|seven|eight|nine|ten|fifteen|twenty|thirty|forty five|forty|fifty|sixty)[- ]"
        r"(?P<u>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", t) or re.search(
        r"(?i)\b(?:set|start)\s+(?:a|an)\s+(?P<n>\d+(?:\.\d+)?|half an?|one|two|three|four|five|six|"
        r"seven|eight|nine|ten|fifteen|twenty|thirty|forty five|forty|fifty|sixty)[- ]"
        r"(?P<u>seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+timer", t)
    if timer:
        value = _number(timer.group("n"))
        if value is None:
            return None
        unit = timer.group("u").lower()
        key = "seconds" if unit.startswith("sec") else "hours" if unit.startswith(("hour", "hr")) \
            else "minutes"
        return {"action": "timer", key: value}
    if re.search(r"(?i)\b(?:what|which|any|list|show)\b.*\breminders?\b", t):
        return {"action": "list"}
    if re.search(r"(?i)\bremind(?:er)?\b|\bwake me\b|\bnudge me\b", t):
        return {"action": "add", "phrase": t}
    return None


_WEATHER = re.compile(r"(?i)\b(?:weather|forecast|rain(?:ing)?|umbrella|coat|jacket|temperature|"
                      r"cold|warm|hot|snow(?:ing)?|sunny|windy|wet|degrees|freezing|pouring)\b")
_PLACE = re.compile(
    r"(?:\b(?:in|for|at|around)\s+(?P<p>[A-Za-z][A-Za-z' .-]{1,40}?))"
    r"(?=\s+(?:today|tonight|tomorrow|this|next|on|at|right|now|please|later|later on)\b|[?.!,]|$)")


@recipe("geo")
def _geo(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    low = t.lower()
    near = re.search(r"(?i)\b(?:find|any|nearest|closest|where(?:'s| is) (?:the|a)|show me)\s+(?:some\s+|a\s+|the\s+)?"
                     r"(?P<what>[a-z][a-z ]{2,30}?)\s+(?:near(?:by| to)?|around|close to)\s*(?P<near>.*)$", t)
    if near and not _WEATHER.search(low):
        args: dict[str, Any] = {"action": "places", "what": near.group("what").strip()}
        place = near.group("near").strip(" .?!")
        if place and not re.fullmatch(r"(?i)(?:me|here|us)", place):
            args["near"] = place
        return args
    if re.search(r"(?i)\bnearest\s+(?P<w>[a-z ]+)", t):
        found = re.search(r"(?i)\bnearest\s+(?P<w>[a-z ]{3,25}?)(?:\s+(?:to|from|near)\b|[?.!]|$)", t)
        if found:
            return {"action": "places", "what": found.group("w").strip()}
    direct = re.search(r"(?i)\b(?:directions?|route|how do i get|how can i get|take me)\s+(?:to\s+)?(?P<d>.+)$", t)
    if direct:
        return {"action": "directions", "query": direct.group("d").strip(" .?!")}
    far = re.search(r"(?i)\bhow far (?:is it )?(?:to|is)\s+(?P<d>.+)$", t)
    if far:
        return {"action": "distance", "query": far.group("d").strip(" .?!")}
    if _WEATHER.search(low):
        args = {"action": "weather"}
        place = _PLACE.search(t)
        if place:
            name = place.group("p").strip()
            if not re.fullmatch(r"(?i)(?:the|my|this|that|case|a|an|it|time|general)", name):
                args["query"] = _titled(name)
        when = re.search(r"(?i)\b(today|tonight|tomorrow|this weekend|the weekend|this week|"
                         r"next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t)
        if when:
            args["when"] = when.group(1).lower()
        elif re.search(r"(?i)\b(?:coat|umbrella|jacket)\b", t) and len(t.split()) <= 8:
            args["when"] = t
        return args
    return None


_FILE_TYPES = {"pdf": "pdf", "pdfs": "pdf", "word": "word", "spreadsheet": "spreadsheet",
               "spreadsheets": "spreadsheet", "excel": "spreadsheet", "presentation": "presentation",
               "photo": "photo", "photos": "photo", "picture": "image", "pictures": "image",
               "image": "image", "images": "image", "video": "video", "videos": "video",
               "folder": "folder", "song": "audio", "audio": "audio", "document": "document",
               "documents": "document"}


@recipe("find_files")
def _find_files(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = re.match(r"(?i)^(?:find|locate|where(?:'s| is| are)|look for|search for|show me|bring up|"
                     r"i lost)\s+(?:all\s+)?(?:my\s+|the\s+|that\s+|those\s+)?(?P<x>.+)$", t)
    if not match:
        return None
    rest = match.group("x").strip(" .?!")
    args: dict[str, Any] = {}
    when = re.search(r"(?i)\b(today|yesterday|this week|last week|this month|last month|last \d+ days)\b", rest)
    if when:
        args["modified"] = when.group(1).lower()
        rest = (rest[:when.start()] + rest[when.end():]).strip(" ,.")
        rest = re.sub(r"(?i)\b(?:i|that i)\s+(?:downloaded|made|saved|edited|changed|opened)\b", "", rest).strip()
    for word in re.findall(r"[a-z]+", rest.lower()):
        if word in _FILE_TYPES:
            args["type"] = _FILE_TYPES[word]
            rest = re.sub(r"(?i)\b" + word + r"\b", "", rest).strip()
            break
    rest = re.sub(r"(?i)\b(?:files?|that|i|downloaded|made|saved|with|in|it)\b", " ", rest)
    rest = " ".join(rest.split()).strip(" ,.")
    if rest:
        args["query"] = rest
    return args if args else None


@recipe("save_memory")
def _save_memory(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = re.match(r"(?i)^(?:remember|note(?:\s+down)?|make a note(?:\s+of)?|jot down|keep in mind|"
                     r"don't forget|save)\s+(?:that\s+)?(?P<f>.+)$", t)
    if not match:
        return None
    fact = match.group("f").strip(" .")
    words = [w for w in re.findall(r"[a-z0-9]+", fact.lower())
             if w not in {"my", "the", "a", "an", "is", "are", "was", "to", "of", "that", "i", "me",
                          "in", "on", "at", "for", "and", "it", "its", "be", "am", "have", "has"}]
    if len(fact) < 6 or len(words) < 2:
        return None
    return {"key": "_".join(words[:4]), "value": _sentence(fact)}


@recipe("recall_conversation", "conversation_recall")
def _recall(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = re.search(r"(?i)\b(?:about|regarding|on|said about|told you about|discuss(?:ed)?)\s+(?P<q>.+)$", t)
    if not match:
        return None
    query = match.group("q").strip(" .?!")
    query = re.sub(r"(?i)^(?:the\s+)?(?:last time|earlier)\b.*", "", query).strip()
    query = re.sub(r"(?i)\s+(?:before|already|earlier|recently|lately)$", "", query).strip()
    return {"query": query} if len(query) >= 3 else None


@recipe("query_intelligence")
def _query_intelligence(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = re.search(r"(?i)\b(?:about|regarding|on|for)\s+(?P<q>.+)$", t)
    if not match:
        return None
    query = match.group("q").strip(" .?!")
    return {"query": query} if len(query) >= 3 else None


@recipe("flight_search")
def _flight_search(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = re.search(r"(?i)\bfrom\s+(?P<o>[A-Za-z' ]{3,30}?)\s+to\s+(?P<d>[A-Za-z' ]{3,30}?)"
                      r"(?:\s+(?:in|on|for|next|this)\s+(?P<w>.+?))?[?.!]*$", t)
    if not match or not match.group("w"):
        return None
    when = re.search(r"(?i)(?:in|on|for|next|this)\s+" + re.escape(match.group("w")) + r"[?.!]*$", t)
    depart = when.group(0).strip(" ?.!") if when else match.group("w")
    depart = re.sub(r"(?i)^(?:in|on|for)\s+", "", depart)
    return {"origin": _titled(match.group("o").strip()), "destination": _titled(match.group("d").strip()),
            "depart": depart}


@recipe("focus")
def _focus(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    if re.search(r"(?i)\b(?:streak|stats|in a row|how many days)\b", t):
        return {"action": "stats"}
    block = re.search(r"(?i)\b(\d{1,3})\s*(?:minutes?|mins?)\s+of\s+(?:no distractions|focus|deep work)", t)
    if block:
        return {"action": "start", "minutes": int(block.group(1)), "label": "focus session"}
    if re.search(r"(?i)\b(?:how long|time left|status|how much time)\b", t):
        return {"action": "status"}
    if re.search(r"(?i)\b(?:i'?m done|finished|end my|stop my|that's it)\b", t):
        return {"action": "done"}
    if re.search(r"(?i)\b(?:distracted|interrupted)\b", t):
        return {"action": "interrupt"}
    if re.search(r"(?i)\b(?:streak|stats)\b", t):
        return {"action": "stats"}
    if re.search(r"(?i)\b(?:start|begin|pomodoro|deep work|focus (?:session|block)|concentrate)\b", t):
        args: dict[str, Any] = {"action": "start"}
        mins = re.search(r"(?i)\b(\d{1,3})[- ]?(?:minute|min)", t)
        if mins:
            args["minutes"] = int(mins.group(1))
        elif re.search(r"(?i)\bpomodoro\b", t):
            args["preset"] = "pomodoro"
        about = re.search(r"(?i)\b(?:on|for)\s+(?P<l>[a-z][\w ]{2,40}?)$", t)
        args["label"] = about.group("l").strip() if about else "focus session"
        return args
    return None


@recipe("finance")
def _finance(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    amount = re.search(r"(?:£|\bgbp\s*)?(\d+(?:[.,]\d+)?)\s*(?:pounds?|quid|gbp|£)?", t)
    spend = re.search(r"(?i)\b(?:spent|paid|bought|log(?:ged)? (?:an? )?expense|record an expense)\b", t)
    income = re.search(r"(?i)\b(?:got paid|earned|received|income|log(?:ged)? (?:some )?income)\b", t)
    if (spend or income) and amount:
        number = float(amount.group(1).replace(",", ""))
        args: dict[str, Any] = {"action": "spend" if spend else "income",
                                "amount": int(number) if number.is_integer() else number}
        cat = re.search(r"(?i)\b(?:on|for)\s+(?:the\s+|my\s+)?(?P<c>[a-z][a-z ]{2,25}?)(?:[.,!?]|$)", t)
        if cat:
            args["category"] = cat.group("c").strip()
        return args
    if re.search(r"(?i)\brunway\b", t):
        return {"action": "runway"}
    if re.search(r"(?i)\b(?:balance|how much (?:money|cash)|left|have i got)\b", t):
        return {"action": "balance"}
    if re.search(r"(?i)\b(?:renew|upcoming|due soon)\b", t):
        return {"action": "upcoming"}
    return None


@recipe("wellbeing")
def _wellbeing(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    mood = re.search(r"(?i)\bmood\b.*?\b(great|good|ok|okay|low|bad|fine|rubbish)\b", t)
    if mood:
        label = mood.group(1).lower()
        return {"action": "checkin", "mood": "ok" if label in {"okay", "fine"} else
                "bad" if label == "rubbish" else label}
    energy = re.search(r"(?i)\benergy\b.*?\b([1-5])\b", t)
    if energy:
        return {"action": "checkin", "energy": int(energy.group(1))}
    if re.search(r"(?i)\b(?:trend|lately|been|over the (?:last|past))\b", t):
        return {"action": "trend"}
    if re.search(r"(?i)\bhow am i doing\b", t):
        return {"action": "today"}
    return None


@recipe("study")
def _study(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    quiz = re.search(r"(?i)\b(?:quiz|test)\s+me\s+on\s+(?P<d>.+)$", t)
    if quiz:
        return {"action": "review", "deck": quiz.group("d").strip(" .?!")}
    if re.search(r"(?i)\b(?:due|review|revision)\b", t):
        return {"action": "review"}
    if re.search(r"(?i)\b(?:stats|how is my revision|progress)\b", t):
        return {"action": "stats"}
    return None


@recipe("window_control")
def _window_control(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    low = t.lower()
    action = ("minimise" if re.search(r"\bminimi[sz]e\b", low) else
              "maximise" if re.search(r"\bmaximi[sz]e\b|full size|full screen", low) else
              "focus" if re.search(r"\b(?:bring|switch to|focus|front)\b", low) else
              "close" if re.search(r"\bclose\b", low) else
              "list" if re.search(r"\b(?:list|which|what) windows\b", low) else "")
    if not action:
        return None
    if action == "list":
        return {"action": "list"}
    if re.search(r"\b(?:everything|all (?:my )?windows)\b", low):
        return {"action": action, "title": "everything"}
    match = re.search(r"(?i)\b(?:minimi[sz]e|maximi[sz]e|close|focus|bring|switch to)\s+(?:up\s+)?"
                      r"(?:the\s+|my\s+)?(?P<w>[\w .]+?)(?:\s+window)?(?:\s+to the front)?$", t)
    if match:
        title = match.group("w").strip()
        title = re.sub(r"(?i)\s+(?:window|to the front)$", "", title)
        if title and len(title.split()) <= 3 and not re.fullmatch(r"(?i)(?:this|that|it)", title):
            return {"action": action, "title": title}
    return {"action": action} if action in {"maximise", "minimise"} and re.search(r"\b(?:this|it)\b", low) \
        else None


@recipe("screen_read")
def _screen_read(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    return {"prompt": t} if len(t) >= 6 else None


@recipe("resource_status")
def _resource_status(text: str) -> dict[str, Any] | None:
    low = text.lower()
    if re.search(r"\b(?:storage|disk|drive|space)\b", low):
        return {"action": "storage"}
    if re.search(r"\bbattery\b", low):
        return {"action": "battery"}
    return {}


@recipe("awareness")
def _awareness(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    add = re.search(r"(?i)\badd\s+(?P<x>.+?)\s+to\s+my\s+(?:priorities|priority list)", t)
    if add:
        return {"action": "add_priority", "text": add.group("x").strip()}
    task = re.search(r"(?i)\badd\s+(?P<x>.+?)\s+to\s+my\s+(?:tasks?|to-?do)", t)
    if task:
        return {"action": "add_task", "text": task.group("x").strip()}
    if re.search(r"(?i)\b(?:list|show|what are)\b.*\btasks\b", t):
        return {"action": "list_tasks"}
    if re.search(r"(?i)\b(?:priorit|situation|aware|on my plate|deadlines?)\w*", t):
        return {"action": "situation"}
    return None


@recipe("outlook_mail")
def _outlook(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    draft = re.match(r"(?i)^(?:draft|write|compose|prepare)\s+(?:an?\s+)?(?:email|reply|response|message)?"
                     r"\s*(?:reply\s+)?(?:to\s+)(?P<to>[A-Za-z][\w .'-]{1,30}?)\s+"
                     r"(?:saying|that says|telling (?:him|her|them)|to say|about)\s+(?P<b>.+)$", t)
    if draft:
        return {"action": "draft", "to": _titled(draft.group("to").strip()),
                "body": _sentence(draft.group("b").strip(" ."))}
    if re.search(r"(?i)\b(?:inbox|new emails?|any emails?|unread|check (?:my )?(?:mail|email))\b", t):
        return {"action": "read_inbox"}
    if re.search(r"(?i)\bpriority\b|\bimportant emails?\b", t):
        return {"action": "priority"}
    return None


@recipe("messaging")
def _messaging(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = re.match(
        r"(?i)^(?P<v>whatsapp|message|msg|dm|discord|telegram|text|send (?:a )?(?:message|whatsapp) to)\s+"
        r"(?P<c>[A-Za-z][\w .'-]{1,30}?)\s+(?:on\s+(?P<p>whatsapp|discord|telegram)\s+)?"
        r"(?:saying|that|to say|telling (?:him|her|them))\s+(?P<m>.+)$", t)
    if not match:
        return None
    verb = match.group("v").lower()
    platform = (match.group("p") or (verb if verb in {"whatsapp", "discord", "telegram"} else "")).lower()
    contact = match.group("c").strip()
    contact = re.sub(r"(?i)^(?:my\s+)", "", contact)
    args: dict[str, Any] = {"action": "send", "contact": contact, "message": _sentence(match.group("m").strip(" ."))}
    if platform:
        args["platform"] = platform
    return args


@recipe("telephony")
def _telephony(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    call = re.match(r"(?i)^(?:ring|call|phone|dial)\s+(?:up\s+)?(?P<c>[A-Za-z][\w .'-]{1,30}?)"
                    r"(?:\s+(?:and\s+(?:tell|say)\b.*|to\s+say\b.*|about\b.*))?$", t)
    if call:
        who = re.sub(r"(?i)^my\s+", "", call.group("c").strip())
        if re.match(r"(?i)^(?:me|us|myself)\b", who):
            return None                     # "call me in ten minutes" is a reminder
        return {"action": "call", "to": who}
    return None


@recipe("notion_workspace")
def _notion(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    add = re.match(r"(?i)^add\s+(?P<x>.+?)\s+to\s+(?:my\s+)?(?:notion\s+)?(?:tasks?|to-?do)(?:\s+list)?$", t)
    if add:
        return {"action": "create_task", "title": add.group("x").strip(" .")}
    if re.search(r"(?i)\b(?:calendar|events?|schedule|meetings?)\b", t):
        return {"action": "upcoming_events"}
    if re.search(r"(?i)\bprojects?\b", t):
        return {"action": "projects"}
    if re.search(r"(?i)\btasks?\b|\bto-?do\b", t):
        return {"action": "list_tasks"}
    return None


@recipe("system_notify")
def _system_notify(text: str) -> dict[str, Any] | None:
    match = re.search(r"(?i)\b(?:notify|alert|ping)\s+me\s+(?:that|about|with|saying)?\s*(?P<m>.+)$", text)
    return {"message": _sentence(match.group("m").strip(" ."))} if match and len(match.group("m")) > 2 else None


@recipe("process_governor")
def _process_governor(text: str) -> dict[str, Any] | None:
    if re.search(r"(?i)\b(?:using|hogging|eating|taking up|slow|crawling|top|biggest|most)\b", text):
        return {"action": "top"}
    return None


@recipe("browser_control")
def _browser(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    found = _URL.search(t)
    if found and re.search(r"(?i)\b(?:open|go to|take me|browse|visit)\b", t):
        url = found.group(0).rstrip(".,;)")
        return {"action": "go_to", "url": url if re.match(r"(?i)https?://", url) else "https://" + url}
    site = re.match(r"(?i)^(?:open|go to|take me to|visit)\s+(?P<s>youtube|gmail|github|reddit|amazon|netflix|twitter|bbc)\b", t)
    if site:
        return {"action": "go_to", "url": f"https://{site.group('s').lower()}.com"}
    return None


@recipe("clipboard_operate")
def _clipboard(text: str) -> dict[str, Any] | None:
    if re.search(r"(?i)\b(?:what(?:'s| is)|read|show|check)\b.*\bclipboard\b", text):
        return {"action": "read"}
    return None


@recipe("research")
def _research(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    match = re.match(r"(?i)^(?:research|dig into|look into|investigate|do a deep dive (?:on|into)|"
                     r"deep dive (?:on|into)|write me a (?:proper )?(?:report|paper) on)\s+(?P<t>.+)$", t)
    if not match:
        return None
    topic = re.sub(r"(?i)(?:\s+(?:for me|in depth|in detail|please))+$", "", match.group("t")).strip(" .?!")
    topic = re.split(r"(?i)\s+and\s+(?:then\s+)?(?:write|make|put|turn)\b", topic)[0].strip()
    if len(topic) < 3:
        return None
    paper = re.search(r"(?i)\b(?:report|paper|thesis|write[- ]?up)\b", t)
    return {"action": "paper" if paper else "start", "topic": topic}


@recipe("docker")
def _docker(text: str) -> dict[str, Any] | None:
    logs = re.search(r"(?i)\blogs?\b.*?\b(?:for|of)\s+(?:my\s+|the\s+)?(?P<c>[\w.-]+?)(?:\s+container)?[?.!]*$", text)
    if logs and logs.group("c").lower() not in {"container", "containers", "it", "docker"}:
        return {"action": "logs", "container": logs.group("c")}
    if re.search(r"(?i)\bimages\b", text):
        return {"action": "images"}
    if re.search(r"(?i)\bcontainers?\b", text):
        return {"action": "list"}
    if re.search(r"(?i)\bdocker\b.*\b(?:running|available|installed|working)\b", text):
        return {"action": "status"}
    return None


@recipe("skill")
def _skill(text: str) -> dict[str, Any] | None:
    return {"action": "list"} if re.search(r"(?i)\b(?:which|what|list|show)\b.*\bskills?\b", text) else None


@recipe("plugin")
def _plugin(text: str) -> dict[str, Any] | None:
    return {"action": "list"} if re.search(r"(?i)\b(?:which|what|list|show)\b.*\bplugins?\b", text) else None


@recipe("backup")
def _backup(text: str) -> dict[str, Any] | None:
    return {"action": "list"} if re.search(r"(?i)\b(?:show|list|what|which)\b.*\bbackups?\b", text) else None


@recipe("chess")
def _chess(text: str) -> dict[str, Any] | None:
    low = fix_slips(text).lower()
    if re.search(r"\b(?:rating|elo)\b", low):
        return {"action": "rating"}
    if re.search(r"\b(?:analy[sz]e)\b.*\b(?:last|that|my)\b.*\bgame\b", low):
        return {"action": "analyse_game"}
    if re.search(r"\b(?:game of chess|play (?:a )?(?:game of )?chess|new game|fancy a game)\b", low):
        return {"action": "new_game"}
    return None


@recipe("rewind")
def _rewind(text: str) -> dict[str, Any] | None:
    t = strip_filler(text)
    when = re.search(r"(?i)\bwhen did (?:i|we)\s+(?:first\s+)?(?:bring up|mention|talk about|discuss)\s+(?P<q>.+)$", t)
    if when:
        return {"action": "search", "query": when.group("q").strip(" .?!")}
    day = re.search(r"(?i)\b(?:recap|summar\w+|what did we (?:talk about|discuss|say))\b.*\b(today|yesterday)\b", t)
    if day:
        return {"action": "recap", "day": day.group(1).lower()}
    return None


@recipe("system_startup")
def _system_startup(text: str) -> dict[str, Any] | None:
    low = text.lower()
    if re.search(r"\bwhich browser|what browser\b", low):
        return {"action": "browser"}
    if re.search(r"\b(?:launch|start|run)\b.*\b(?:when i turn|when (?:windows|my pc|the pc) starts?|on startup|at startup|at boot)\b", low):
        return {"action": "enable"}
    if re.search(r"\b(?:do you|are you)\b.*\b(?:start|launch)\b.*\b(?:automatically|on startup|at boot)\b", low):
        return {"action": "status"}
    return None


@recipe("interface_control")
def _interface_control(text: str) -> dict[str, Any] | None:
    low = text.lower()
    form = re.search(r"\b(?:switch|change|go|turn)\b.*\b(?:to\s+)?(?:your\s+)?(orb|face)\b", low)
    if form:
        return {"action": "face_form", "target": "orb" if form.group(1) == "orb" else "screen"}
    return None


@recipe("diagnostics")
def _diagnostics(text: str) -> dict[str, Any] | None:
    if re.search(r"(?i)\b(?:health check|diagnostic|self[- ]?test|check yourself|run a check)\b", text):
        return {}
    return None


@recipe("display_info")
def _display_info(text: str) -> dict[str, Any] | None:
    if re.search(r"(?i)\b(?:resolution|how many (?:monitors|screens|displays)|monitors? (?:do i|have i)|"
                 r"what (?:monitors|screens|displays))\b", text):
        return {}
    return None


# Tools with nothing to say: calling them with {} is the whole call.
NO_ARGUMENTS = frozenset({
    "capture_screen", "token_usage", "proactive_check", "commerce_hub", "morning_briefing",
    "catch_up", "patch_notes", "self_changes", "workflow_patterns", "navigation_trace",
    "briefing", "shutdown_orion", "restart_orion",
})

_ACTION_STOP = frozenset({"or", "and", "the", "a", "an", "default", "omit", "optional", "confirmation",
                          "gated", "when", "for", "to", "with", "see", "use"})
#: Parameters the generic fallback may fill with the request itself.
_FREE_TEXT = frozenset({"query", "question"})
#: Words that, matched literally, stand for an action in the generic fallback.
_LISTING: dict[str, tuple[str, ...]] = {}


def action_names(description: str) -> list[tuple[str, bool]]:
    """(name, is_default) pairs from an ``action`` parameter's description, e.g.
    "report (default), account, spend, or income." Empty if it is not a plain list."""
    head = re.split(r"(?<=[a-z)])\.\s", str(description or "").strip(), maxsplit=1)[0]
    head = head.replace("—", ",").replace(" - ", ", ")
    pieces = [p.strip() for p in re.split(r",", head) if p.strip()]
    found: list[tuple[str, bool]] = []
    for piece in pieces:
        default = "(default)" in piece.lower()
        piece = re.sub(r"\([^)]*\)", " ", piece)
        piece = re.sub(r"(?i)^or\s+", "", piece).strip(" .;:'\"")
        token = piece.split()[0] if piece.split() else ""
        token = token.strip("'\"")
        if re.fullmatch(r"[a-z][a-z_]{1,24}", token) and token not in _ACTION_STOP:
            found.append((token, default))
    return found if len(found) >= 2 else []


def _generic(book: ToolBook, tool: str, text: str) -> dict[str, Any] | None:
    """A last resort: an action word the request literally contains, and the
    request itself for a required ``query``. Read-only calls only (checked by the caller)."""
    properties = book.properties(tool)
    required = book.required(tool)
    t = strip_filler(text)
    if len(t.split()) > 12:
        return None
    low = t.lower()
    args: dict[str, Any] = {}
    action_spec = properties.get("action")
    if action_spec:
        names = [n for n, _d in action_names(str(action_spec.get("description") or ""))]
        if action_spec.get("enum"):
            names = [str(v) for v in action_spec["enum"]]
        chosen, best = "", 10 ** 6
        for name in names:
            words = {name.replace("_", " "), *_LISTING.get(name, ())}
            for word in words:
                found = re.search(r"\b" + re.escape(word) + r"\b", low)
                if found and found.start() < best:
                    best, chosen = found.start(), name
        if chosen:
            args["action"] = chosen
        elif "action" in required:
            return None
    for key in required:
        if key in args:
            continue
        spec = properties.get(key) or {}
        if spec.get("type") == "string" and key in _FREE_TEXT:
            args[key] = t.strip(" .?!")
        else:
            return None
    if not required and not args:
        return None
    return args


def synthesise(book: ToolBook, tool: str, text: str, *, generic: bool = True) -> tuple[dict[str, Any] | None, str]:
    """(arguments, how). *how* is 'rule', 'generic' or 'none'. Arguments are
    returned only when the real schema accepts them; the generic fallback
    additionally requires the call to be read-only."""
    if not book.has(tool):
        return None, "none"
    text = fix_slips(text)
    if tool in RECIPES:
        args, how = RECIPES[tool](text), "rule"
    elif tool in NO_ARGUMENTS:
        args, how = {}, "rule"
    elif generic:
        args, how = _generic(book, tool, text), "generic"
    else:
        return None, "none"
    if args is None or book.validate(tool, args):
        return None, "none"
    if how == "generic" and book.risk(tool, args)[0] != "read":
        return None, "none"
    return args, how


__all__ = ["NO_ARGUMENTS", "RECIPES", "SLIP_FIXES", "action_names", "first_clause", "fix_slips",
           "strip_filler", "synthesise"]
