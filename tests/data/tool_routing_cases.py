"""
Tool-routing evaluation set — the phrases a person actually says, and the tool
each one needs.

Written by hand against ORION's registered tool list, deliberately in ordinary
speech rather than in the vocabulary the schema happens to use: the whole point
is to catch the gap between how a capability is DESCRIBED and how it is ASKED
FOR.

Split deterministically into DEV and HOLDOUT by index. Only DEV may be looked at
while tuning the resolver or its vocabulary. HOLDOUT exists so the reported
recall means something — a scorer tuned until it passes the cases used to tune
it has measured nothing but its own tuning.
"""

from __future__ import annotations

#: (utterance, the tool that should handle it)
CASES: list[tuple[str, str]] = [
    # study / spaced repetition
    ("what's due for review today", "study"),
    ("quiz me on neuroscience", "study"),
    ("test me on my flashcards", "study"),
    ("add a card about the hippocampus", "study"),
    ("how is my revision going", "study"),
    ("I want to memorise these terms", "study"),
    # focus
    ("start a deep work block on revision", "focus"),
    ("how long is left on my focus", "focus"),
    ("begin a pomodoro", "focus"),
    ("I need to concentrate for an hour", "focus"),
    ("end my session", "focus"),
    # finance
    ("what's my runway looking like", "finance"),
    ("log that I spent 40 pounds on groceries", "finance"),
    ("how much money do I have left", "finance"),
    ("record an expense for the train", "finance"),
    ("what did I spend this month", "finance"),
    # wellbeing
    ("how have I been sleeping", "wellbeing"),
    ("log my mood as pretty low", "wellbeing"),
    ("how has my energy been lately", "wellbeing"),
    ("track how I'm feeling today", "wellbeing"),
    # search and research
    ("search the web for transformer architectures", "web_search"),
    ("google the price of a used thinkpad", "web_search"),
    ("look up the latest on protein folding", "research"),
    ("do a deep dive on the UK rental market", "research"),
    ("write me a proper report on this topic", "research"),
    # vision
    ("take a screenshot of my screen", "capture_screen"),
    ("what's on my screen right now", "capture_screen"),
    ("read the text in this image", "vision_analyse"),
    ("what does that error message say", "vision_analyse"),
    ("describe what you can see", "vision_analyse"),
    # desktop control
    ("click the submit button", "desktop_control"),
    ("type my email address into that box", "desktop_control"),
    ("scroll down the page", "desktop_control"),
    # plugins / diagnostics
    ("what plugins do I have", "plugin"),
    ("disable the discord plugin", "plugin"),
    ("run a self diagnostic", "diagnostics"),
    ("is everything working properly", "diagnostics"),
    ("why can't you do that", "diagnostics"),
    # usage
    ("how many tokens have I used", "token_usage"),
    ("am I close to my limit", "token_usage"),
    # chess
    ("play me a game of chess", "chess"),
    ("what's the best move here", "chess"),
    # geo / place
    ("what's the weather in Birmingham", "geo"),
    ("how far is it to Manchester", "geo"),
    ("what's the postcode for that address", "geo"),
    # files
    ("find the file called budget spreadsheet", "find_files"),
    ("where did I save that document", "find_files"),
    ("summarise this document for me", "process_file"),
    ("read this pdf and tell me what it says", "process_file"),
    # memory / recall
    ("what did we talk about yesterday", "recall_conversation"),
    ("remind me what I said about the business", "recall_conversation"),
    ("what do you remember about my project", "second_brain"),
    # forge
    ("build me a tool that renames files", "forge"),
    ("can you write yourself a new capability", "forge"),
    # security
    ("scan my network for open ports", "security_recon"),
    ("check if my email was in a breach", "breach_check"),
    # language
    ("teach me some spanish vocabulary", "language_tutor"),
    ("how do you say thank you in french", "language_tutor"),
    # decisions
    ("record a decision about the business", "decision"),
    ("how well calibrated am I", "decision"),
    # briefing / news
    ("what's happening in the news", "briefing"),
    ("give me my morning briefing", "briefing"),
    # system
    ("how is the system performing", "resource_status"),
    ("how much memory is being used", "resource_status"),
    ("what can you actually do", "capabilities"),
    # awareness
    ("what have you been up to", "awareness"),
    ("what's changed recently", "self_changes"),
    # ── whose voice, and whether to answer it ────────────────────────────────
    #
    # Placed with care. The vocabulary was tuned against three of these —
    # "who is speaking", "only listen to my voice" and "was that a man or a
    # woman" — so they belong in DEV by definition; a phrase used to fix the
    # scorer cannot also be used to score it. The split is by index, so the
    # order below is what puts each one on the right side of it, and the
    # HOLDOUT entries are phrasings that were never looked at while tuning.
    # The first entry below sits at an odd index, so each tuned phrase is
    # placed SECOND in its pair to land on an even one. Check with
    # dev_cases(), not by eye — getting this backwards silently launders a
    # tuned phrase into the honest score.
    #
    # The pair also guards a real confusion: speaker_id is a pitch-based guess
    # at gender and voice_speaker_id compares a trained voiceprint against
    # enrolled people. speaker_id's own description used to claim it answered
    # "who was just talking?", which it cannot, and it out-scored the tool that
    # can on exactly the question people ask most.
    ("whose voice was that", "voice_speaker_id"),               # unseen
    ("who is speaking", "voice_speaker_id"),                    # tuned → DEV
    ("stop responding to other people", "voice_speaker_id"),    # unseen
    ("only listen to my voice", "voice_speaker_id"),            # tuned → DEV
    ("did that sound male or female", "speaker_id"),            # unseen
    ("was that a man or a woman", "speaker_id"),                # tuned → DEV
    ("go back to answering everyone", "voice_speaker_id"),      # unseen
    ("remember my voice so you know me", "voice_speaker_id"),   # unseen
]


def dev_cases() -> list[tuple[str, str]]:
    """Even indices — the half that may be inspected while tuning."""
    return [c for i, c in enumerate(CASES) if i % 2 == 0]


def holdout_cases() -> list[tuple[str, str]]:
    """Odd indices — never inspected while tuning; the honest score."""
    return [c for i, c in enumerate(CASES) if i % 2 == 1]


__all__ = ["CASES", "dev_cases", "holdout_cases"]
