"""
Meaning-based routing for the specialist agents (Mark XXXX.7).

The keyword router (BaseAgent.score) only fired on the domain's own words:
measured on 103 labelled requests (28 Oct 2026) it sent the right specialist
31 times in 88 and no specialist at all 50 times - "What's filling up my C
drive?", "Why did your voice go silent earlier?" and "Help me revise the
stages of sleep" reached nobody.

Here each specialist is described by what people actually ask it, and
ORION's own sentence encoder (semantic.py, the model his memory already uses)
compares a request with those descriptions by meaning. ORION himself is one
of the candidates ("" - small talk, the weather, music, timers): a request
that is most like those stays with him. A specialist takes a request only
when it is clearly the best fit; a strong keyword match still wins, and with
no encoder the keyword router decides exactly as before.
"""

from __future__ import annotations

import re
import threading
from typing import Any

#: What people ask each specialist ("" = ORION answers himself). Written from
#: each agent's remit; none of the labelled test requests is repeated here.
EXEMPLARS: dict[str, tuple[str, ...]] = {
    "marketing": (
        "grow my brand's followers and engagement on social media",
        "write ad copy and captions that convert",
        "plan a paid advertising campaign and budget",
        "my sales are flat, how do I get more customers",
        "build an email marketing sequence for a product launch",
        "improve search rankings for my online shop",
        "come up with viral content ideas for short videos",
        "position my product against competitors and pick a price point",
        "landing page that turns visitors into buyers",
    ),
    "coding": (
        "fix this bug in my code",
        "this program throws an exception, what's wrong",
        "write a function that parses this file",
        "my application crashes with a stack trace",
        "review my pull request and suggest cleaner code",
        "optimise this algorithm, it is too slow",
        "how should I organise the folders and modules of my software project",
        "git rebase went wrong and I lost my commits",
        "compile error in my program",
        "what does this error message mean",
        "write a shell script that automates a daily job",
        "rewrite this code using a different language feature",
        "a regular expression to match these lines",
        "set up a web page layout with html and css",
    ),
    "research": (
        "compare these products and tell me which is better value",
        "find out what the evidence says about this topic",
        "weigh up the advantages and disadvantages of this decision",
        "do a deep dive on this market and summarise the findings",
        "which option should I choose, analyse them for me",
        "look into the history and causes of this event",
        "gather sources and write a structured report",
        "rank these choices against my criteria",
        "find the cheapest way to do this and compare the prices",
        "what does published research conclude about this",
        "plan the best route and compare travel options",
    ),
    "neuroscience": (
        "how do neurons fire and communicate",
        "explain this brain region and what it does",
        "neurotransmitters and how drugs affect the brain",
        "memory formation and learning in the brain",
        "help me study for my psychology and neuroscience lecture",
        "how does sleep affect the brain and cognition",
        "behavioural psychology experiments and learning theories",
        "the nervous system, senses and perception",
    ),
    "aiml": (
        "train a neural network on my data",
        "explain how large language models work",
        "my machine learning model's accuracy is poor",
        "choose hyperparameters and a loss function",
        "fine-tune an open model on my own dataset",
        "vector search and semantic similarity for documents",
        "how much compute and GPU memory does training need",
        "evaluate a model with precision, recall and validation sets",
        "explain a machine learning concept like overfitting or regularisation",
        "reinforcement learning with agents, rewards and policies",
        "classification versus regression models",
    ),
    "security": (
        "is this message a scam or a hacking attempt",
        "how do I keep my accounts safe from being hacked",
        "secure my home network and router",
        "explain how this cyber attack works and how to defend against it",
        "my account was accessed by someone else",
        "protect my computer from malware and viruses",
        "strong passwords, passkeys and multi-factor login",
        "assess the risks and attack surface of my system",
        "explain a web vulnerability and how to prevent it",
        "a suspicious link or email that might be phishing",
    ),
    "reliability": (
        "you stopped responding, what went wrong",
        "why are you so slow today",
        "the program froze and Windows said it was not responding",
        "diagnose why you crashed",
        "you couldn't hear me properly earlier",
        "look through your diagnostics and logs for faults",
        "something in your interface keeps breaking",
        "how stable have you been, any errors recently",
        "you are using too much memory or processor",
        "you take too long to start up",
        "your voice keeps cutting out or stuttering",
    ),
    "integrations": (
        "connect this service to you through its API",
        "the third-party API returns authentication errors",
        "set up an MCP server or plugin connection",
        "sync data between two apps automatically",
        "handle API limits, retries and paging through results",
        "my access token or API key stopped working",
        "send data from one web service to another with a webhook",
        "link my smart home or workspace account",
        "connect my music, email or calendar account to you",
        "automate a workflow between two online services",
    ),
    "continuity": (
        "what was I doing last time",
        "remind me where this project stands",
        "carry on with the work from the previous session",
        "what's still outstanding on this piece of work",
        "write up a handover of progress and next steps",
        "get me back into this task after a break",
        "recap what we decided and what remains",
    ),
    "files": (
        "my disk is nearly full, what's using the space",
        "clean up and sort the files on my computer",
        "move these files into a folder",
        "find files of a certain type across my drive",
        "remove old downloads and things I no longer need",
        "which files are taking the most storage",
        "rename and arrange my documents",
        "collect my pictures into one place",
    ),
    "triage": (
        "what have you been unable to do for me",
        "which of your features break most often",
        "what should I add to you next",
        "what are your weaknesses and missing abilities",
        "how could you become more helpful",
        "things you refused or failed at lately",
        "your biggest gaps and what to improve",
    ),
    "": (
        "hello there",
        "how's it going",
        "what's the weather like",
        "put some music on",
        "pause the song",
        "set an alarm for seven",
        "start a countdown",
        "what's the time",
        "open an app for me",
        "tell me something funny",
        "thank you",
        "good night",
        "remind me to do something later",
        "what is twelve times eight",
        "turn the brightness down",
        "what do I have on today",
        "who are you",
        "make it louder",
        "capture my screen",
        "lock the computer",
        "close this window",
        "read me the headlines",
        "how are you feeling",
        "next song",
    ),
}

_NUMBER_WORDS = ("zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
                 "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
                 "sixty|seventy|eighty|ninety|hundred|thousand|million|half|quarter")
_QUANTITY = rf"(?:[-+]?\d[\d,]*(?:\.\d+)?|(?:(?:{_NUMBER_WORDS})\b[\s-]*(?:and\s+)?)+)"

#: Shapes in a request that point at one specialist whatever the wording:
#: code and error names, ORION asked about his own running, and a named
#: online service with something to connect. Each adds to that candidate's
#: similarity, so meaning still decides between close candidates.
CUES: dict[str, tuple[Any, float]] = {
    "coding": (re.compile(
        r"\b\w+(Error|Exception)\b|\w\(\)|=>|[{};]|\.(py|js|ts|cpp|java|cs|html|css)\b|"
        r"\b(python|javascript|typescript|java|c\+\+|rust|bash|powershell|regex|css|html|"
        r"async|await|function|class|compile|syntax|script)\b", re.I), 0.035),
    # ORION by name is "you" too: "ORION keeps crashing after startup".
    "reliability": (re.compile(
        r"\b(you|your|you're|yourself|orion|orion's)\b.*\b(slow|lag\w*|freez\w*|froze|crash\w*|hang\w*|"
        r"memory|cpu|processor|start(ing)? ?up|boot\w*|cut(ting)? out|stutter\w*|errors?|logs?|"
        r"glitch\w*|stuck|unresponsive|deaf|silent)\b", re.I), 0.035),
    "integrations": (re.compile(
        r"\b(spotify|gmail|notion|zapier|ifttt|slack|github|gitlab|discord|dropbox|trello|"
        r"outlook|google (calendar|drive|sheets)|hue|home assistant|webhooks?|apis?|oauth)\b", re.I),
        0.03),
    # A bare sum is ORION's own (his calculator), never a researcher's:
    # "What's 15 percent of 240?" sat closest to the research specialist.
    # The whole request must be the sum, so "why is 0.1 + 0.2 wrong in
    # python" stays with the coder.
    "": (re.compile(
        rf"^\s*(?:(?:what(?:'s| is)|how much is|calculate|work out|compute)\s+)?(?:the\s+)?"
        rf"(?:square root of\s+)?{_QUANTITY}(?:\s*(?:%|percent|per cent|times|x|\*|plus|\+|minus|-|"
        rf"divided by|over|/|multiplied by|to the power of)\s*(?:of\s+)?{_QUANTITY})*\s*[?.!]*\s*$",
        re.I), 0.10),
}
#: A specialist needs at least this similarity (bge-small cosine) ...
MIN_SIMILARITY = 0.60
#: ... this lead over ORION himself ...
MIN_MARGIN = 0.015
#: ... and must beat every other specialist (any lead at all). Two
#: specialists close together, both well ahead of ORION, is a request for
#: one of them: "Why does my React component render twice?" was coding
#: 0.607 / reliability 0.598 and went to nobody under one shared margin.
#: An exact tie still goes to nobody.
SPECIALIST_MARGIN = 0.0
#: A keyword score this high (two primary signals) decides by itself.
STRONG_KEYWORDS = 6

#: ORION_AGENT_MEANING=0 routes by keywords alone (the pre-28 Oct router).
ENABLED = __import__("os").getenv("ORION_AGENT_MEANING", "1").strip().lower() not in {"0", "false", "no", "off"}

_lock = threading.Lock()
_matrix: dict[int, tuple[list[str], Any]] = {}


def _labels_and_vectors(encoder: Any) -> tuple[list[str], Any] | None:
    """Every exemplar's vector with its label, encoded once per encoder."""
    key = id(encoder)
    with _lock:
        cached = _matrix.get(key)
    if cached is not None:
        return cached
    labels = [label for label, texts in EXEMPLARS.items() for _ in texts]
    texts = [text for items in EXEMPLARS.values() for text in items]
    from . import accelerator
    vectors = encoder.encode(texts, gpu=not accelerator.on_gui_thread())
    if vectors is None:
        return None
    with _lock:
        _matrix[key] = (labels, vectors)
    return labels, vectors


def similarities(request: str, encoder: Any = None) -> dict[str, float] | None:
    """Each candidate's similarity to *request*: the mean of its two closest
    exemplars. None when the encoder is not ready (nothing is loaded here)."""
    if encoder is None:
        from . import semantic
        encoder = semantic.ENCODER
    try:
        if not encoder.ready():
            return None
        table = _labels_and_vectors(encoder)
        question = encoder.encode_one(str(request or "")[:600])
    except Exception:
        return None
    if table is None or question is None:
        return None
    labels, vectors = table
    scores = vectors @ question
    best: dict[str, list[float]] = {}
    for label, score in zip(labels, scores):
        best.setdefault(label, []).append(float(score))
    result = {label: sum(sorted(values)[-2:]) / min(2, len(values)) for label, values in best.items()}
    for label, (pattern, bonus) in CUES.items():
        if label in result and pattern.search(str(request or "")):
            result[label] += bonus
    return result


def choose(request: str, keyword_pick: tuple[Any, int], agents: dict[str, Any],
           encoder: Any = None, *, keyword_threshold: int = 3) -> tuple[Any, float, str]:
    """(agent or None, confidence, how it was decided).

    *keyword_pick* is the keyword router's (best agent, weighted score); it
    counts only at *keyword_threshold* or above, as before."""
    kw_agent, kw_score = keyword_pick
    if kw_score < keyword_threshold:
        kw_agent = None
    if kw_agent is not None and kw_score >= STRONG_KEYWORDS:
        return kw_agent, float(kw_score), "keywords"
    scores = similarities(request, encoder)
    if not scores:
        return (kw_agent, float(kw_score), "keywords") if kw_agent is not None else (None, 0.0, "none")
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    top, top_score = ranked[0]
    orion = scores.get("", 0.0)
    rival = max((score for label, score in ranked[1:] if label), default=0.0)
    if (top and top in agents and top_score >= MIN_SIMILARITY and top_score - orion >= MIN_MARGIN
            and top_score - rival > SPECIALIST_MARGIN):
        return agents[top], top_score, "meaning"
    if kw_agent is not None:
        return kw_agent, float(kw_score), "keywords"
    return None, top_score, "none"


__all__ = ["CUES", "EXEMPLARS", "MIN_MARGIN", "MIN_SIMILARITY", "SPECIALIST_MARGIN", "choose",
           "similarities"]
