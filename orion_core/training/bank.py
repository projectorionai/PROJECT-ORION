"""
The hand-written part of the training set: everyday requests with their exact
calls, a few of them with the tool's result and ORION's spoken answer.

Written fresh for training. None of these sentences comes from the routing
evaluation sets (``tests/data``) or from ``tools/planner_eval_set.json``; the
dataset builder still checks every row against them and drops near-copies, so
an accident here cannot leak an evaluation case into training.

Style of the answers, taken from ORION's own system prompt: British English,
one or two short sentences, spoken (no lists, no markdown), and never a claim
that something was done unless the result in front of it says so.

Every call here is validated against the real tool schemas by the test suite
(``tests/test_training_dataset.py``), so a renamed parameter fails a test
rather than silently training the model on a call the agent would reject.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Phrase:
    """A request, the call that answers it, and optionally what came back."""

    text: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result: str = ""          # a successful tool result; with ``reply`` it makes a full turn
    reply: str = ""
    fail_result: str = ""     # what the tool says when it fails; with ``fail_reply``
    fail_reply: str = ""


def P(text: str, tool: str, args: dict[str, Any] | None = None, result: str = "", reply: str = "",
      fail: tuple[str, str] = ("", "")) -> Phrase:
    return Phrase(text, tool, dict(args or {}), result, reply, fail[0], fail[1])


PHRASEBANK: tuple[Phrase, ...] = (
    # ── apps and windows ──────────────────────────────────────────────────────
    P("open notepad", "open_app", {"app_name": "Notepad"}, "Opened Notepad.", "Notepad's open.",
      ("Could not find an application called Notepad.", "I couldn't find Notepad on this computer.")),
    P("can you launch chrome for me", "open_app", {"app_name": "Chrome"}),
    P("start up the calculator", "open_app", {"app_name": "Calculator"}),
    P("pull up file explorer", "open_app", {"app_name": "File Explorer"}),
    P("open my downloads folder", "open_app", {"app_name": "Downloads"}),
    P("get steam going", "open_app", {"app_name": "Steam"}),
    P("close notepad", "close_app", {"app_name": "Notepad"}),
    P("quit steam please", "close_app", {"app_name": "Steam"}),
    P("minimise the browser", "window_control", {"action": "minimise", "title": "the browser"}),
    P("maximise word", "window_control", {"action": "maximise", "title": "word"}),
    P("bring spotify to the front", "window_control", {"action": "focus", "title": "spotify"}),
    P("which windows have I got open", "window_control", {"action": "list"}),
    # ── the web ───────────────────────────────────────────────────────────────
    P("search the web for the best budget mechanical keyboards", "web_search",
      {"query": "best budget mechanical keyboards"},
      "1. Best budget mechanical keyboards - example.org: A comparison of keyboards under fifty pounds.\n"
      "2. Budget keyboard buying guide - example.com: What to look for in switches and layout.",
      "I found a comparison of keyboards under fifty pounds on example.org, plus a buying guide on "
      "example.com. Want me to open the first one?",
      ("web_search failed: no network connection.", "I couldn't search just now because there's no connection.")),
    P("google when the next bank holiday is", "web_search", {"query": "next bank holiday UK"},
      "1. UK bank holidays - gov.uk: The next bank holiday in England and Wales is Monday 25 December.",
      "The next one is Monday 25 December, according to gov.uk."),
    P("look up the opening times for Birmingham library", "web_search",
      {"query": "Birmingham library opening times"}),
    P("what's the latest on the Premier League", "web_search", {"query": "latest Premier League news"}),
    P("find out who won the Grand National this year", "web_search", {"query": "Grand National winner this year"}),
    P("search online for how to reset a router", "web_search", {"query": "how to reset a router"}),
    P("what does the page at bbc.co.uk/weather say", "fetch_url", {"url": "https://bbc.co.uk/weather"}),
    P("go to bbc.co.uk/sport", "browser_control", {"action": "go_to", "url": "https://bbc.co.uk/sport"}),
    P("search youtube for sourdough tutorials", "browser_control",
      {"action": "search", "query": "sourdough tutorials youtube"}),
    # ── sums and facts ────────────────────────────────────────────────────────
    P("what's 18 percent of 250", "calculate", {"expression": "18% of 250"}, "18% of 250 is 45.",
      "That comes to 45."),
    P("convert 10 miles to kilometres", "calculate", {"expression": "10 miles in km"},
      "10 miles is 16.09 km.", "Ten miles is about 16.1 kilometres."),
    P("how many pounds is 70 kilos", "calculate", {"expression": "70 kg in pounds"},
      "70 kg is 154.32 lb.", "Seventy kilos is about 154 pounds."),
    P("what is 15 times 12", "calculate", {"expression": "15 * 12"}, "15 * 12 = 180.", "That's 180."),
    P("split 84 pounds between four of us", "calculate", {"expression": "84 / 4"}, "84 / 4 = 21.",
      "That's 21 pounds each."),
    # ── sound and media ───────────────────────────────────────────────────────
    P("play some Arctic Monkeys", "media_control", {"action": "play", "query": "Arctic Monkeys"}),
    P("pause", "media_control", {"action": "pause"}),
    P("next track please", "media_control", {"action": "next"}),
    P("go back a song", "media_control", {"action": "previous"}),
    P("what's playing right now", "media_control", {"action": "now_playing"},
      "Spotify is playing 'Do I Wanna Know?' by Arctic Monkeys.",
      "Spotify's playing 'Do I Wanna Know?' by Arctic Monkeys."),
    P("play my chill playlist on spotify", "media_control",
      {"action": "play", "query": "my chill playlist", "service": "spotify"}),
    P("set the volume to forty percent", "peripherals", {"action": "volume", "level": 0.4}),
    P("mute the computer", "peripherals", {"action": "mute"}),
    P("it's far too loud", "peripherals", {"action": "volume_down"}),
    P("make the screen brighter", "peripherals", {"action": "brightness_up"}),
    P("lock the screen", "peripherals", {"action": "lock"}),
    # ── reminders and time ────────────────────────────────────────────────────
    P("remind me at 5pm to ring the plumber", "reminder",
      {"action": "add", "phrase": "remind me at 5pm to ring the plumber"}),
    P("set a five minute timer", "reminder", {"action": "timer", "minutes": 5}),
    P("timer for 90 seconds", "reminder", {"action": "timer", "seconds": 90}),
    P("what reminders do I have", "reminder", {"action": "list"},
      "You have 2 reminders: ring the plumber at 17:00; send the invoice tomorrow at 09:00.",
      "You've got two: ringing the plumber at five this evening, and sending the invoice tomorrow at nine."),
    P("remind me tomorrow morning to send the invoice", "reminder",
      {"action": "add", "phrase": "remind me tomorrow morning to send the invoice"}),
    # ── places and weather ────────────────────────────────────────────────────
    P("what's the weather in Manchester tomorrow", "geo",
      {"action": "weather", "query": "Manchester", "when": "tomorrow"},
      "Manchester tomorrow: 11C, cloudy with showers from mid-morning, wind light.",
      "Tomorrow in Manchester it's about 11 degrees and cloudy, with showers from mid-morning. Take a coat.",
      ("geo failed: weather service unavailable.", "I couldn't get the forecast just now.")),
    P("will I need an umbrella this afternoon", "geo",
      {"action": "weather", "when": "will I need an umbrella this afternoon"},
      "Here this afternoon: 13C, dry until about 4pm, then light rain.",
      "It stays dry until about four, then there's light rain, so take one if you're out late."),
    P("how far is Cardiff", "geo", {"action": "distance", "query": "Cardiff"},
      "Cardiff is about 98 miles from you, roughly 1 hour 50 minutes by car.",
      "Cardiff is about 98 miles away, roughly an hour and fifty minutes by car."),
    P("find a chemist near me", "geo", {"action": "places", "what": "chemist"}),
    P("is it going to snow this weekend", "geo", {"action": "weather", "when": "this weekend"}),
    # ── files ─────────────────────────────────────────────────────────────────
    P("find my latest invoice pdf", "find_files", {"query": "invoice", "type": "pdf", "sort": "newest"},
      "Found 1 file: D:/Work/Invoices/invoice_0412.pdf (modified yesterday).",
      "The newest one is invoice 0412 in your Work folder, from yesterday. Shall I open it?"),
    P("where's the budget spreadsheet", "find_files", {"query": "budget", "type": "spreadsheet"}),
    P("show me photos from last month", "find_files", {"type": "photo", "modified": "last month"}),
    P("open my cv", "find_files", {"query": "cv", "action": "open"}),
    # ── memory ────────────────────────────────────────────────────────────────
    P("remember that I prefer dark mode in every app", "save_memory",
      {"key": "dark_mode_preference", "value": "The user prefers dark mode in every app."},
      "Saved.", "Noted - dark mode everywhere."),
    P("keep in mind my sister's birthday is the third of June", "save_memory",
      {"key": "sister_birthday", "value": "The user's sister's birthday is on the 3rd of June."}),
    P("remember the wifi router lives in the hallway cupboard", "save_memory",
      {"key": "router_location", "value": "The wifi router is in the hallway cupboard."}),
    P("what did we say about the holiday budget", "recall_conversation", {"query": "holiday budget"}),
    P("did we ever talk about the boiler", "recall_conversation", {"query": "boiler"}),
    P("what do you know about my allotment", "query_intelligence", {"query": "allotment"}),
    P("rewind to what we talked about yesterday", "rewind", {"action": "recap", "day": "yesterday"}),
    # ── planning, money, wellbeing, focus ─────────────────────────────────────
    P("find flights from Birmingham to Edinburgh on Friday 14 November", "flight_search",
      {"origin": "Birmingham", "destination": "Edinburgh", "depart": "Friday 14 November"}),
    P("cheapest flights from Manchester to Faro in July", "flight_search",
      {"origin": "Manchester", "destination": "Faro", "depart": "July"}),
    P("start a 50 minute deep work block on my dissertation", "focus",
      {"action": "start", "minutes": 50, "label": "my dissertation"}),
    P("how much focus time is left", "focus", {"action": "status"}),
    P("I'm done focusing", "focus", {"action": "done"}),
    P("log 25 pounds on lunch", "finance", {"action": "spend", "amount": 25, "category": "lunch"}),
    P("what's my runway", "finance", {"action": "runway"}),
    P("I got paid 1800 pounds today", "finance", {"action": "income", "amount": 1800}),
    P("check in, energy 3, mood ok", "wellbeing", {"action": "checkin", "energy": 3, "mood": "ok"}),
    P("log that I slept six hours", "wellbeing", {"action": "checkin", "sleep_hours": 6}),
    P("quiz me on anatomy", "study", {"action": "review", "deck": "anatomy"}),
    P("what's on my plate today", "awareness", {"action": "situation"}),
    P("add finish the report to my priorities", "awareness",
      {"action": "add_priority", "text": "finish the report"}),
    P("add book the dentist to my tasks", "awareness", {"action": "add_task", "text": "book the dentist"}),
    P("what's on my calendar tomorrow", "notion_workspace", {"action": "upcoming_events", "days": 1}),
    P("add call the bank to my notion tasks", "notion_workspace",
      {"action": "create_task", "title": "call the bank"}),
    # ── the screen and the machine ────────────────────────────────────────────
    P("take a screenshot", "capture_screen", {}),
    P("what does this dialog box want from me", "screen_read",
      {"prompt": "what does this dialog box want from me"}),
    P("what's in front of the camera", "vision_analyse", {"action": "camera"}),
    P("how much RAM am I using", "resource_status", {},
      "CPU 18%, memory 54% (8.6 of 16 GB), disk 66% full.",
      "You're using about 8.6 of your 16 gigabytes of memory, so 54 percent."),
    P("how much space have I got left on my drives", "resource_status", {"action": "storage"},
      "C: 212 GB free of 476 GB; D: 1.4 TB free of 1.8 TB.",
      "C has 212 gigabytes free and D has about 1.4 terabytes."),
    P("what's my battery at", "resource_status", {"action": "battery"},
      "Battery 72%, on battery power, about 2 hours 40 minutes left.",
      "You're at 72 percent, with about two hours forty minutes left."),
    P("what's eating all my memory", "process_governor", {"action": "top", "metric": "ram"}),
    P("how many tokens have I used today", "token_usage", {}),
    P("how many screens have I got plugged in", "display_info", {}),
    P("run diagnostics on yourself", "diagnostics", {}),
    P("what's new in this version of you", "patch_notes", {}),
    P("give me my morning briefing", "morning_briefing", {}),
    P("what's on my clipboard", "clipboard_operate", {"action": "read"}),
    P("is docker running", "docker", {"action": "status"}),
    P("is windows defender switched on", "antivirus", {"action": "status"}),
    P("scan for viruses", "antivirus", {"action": "scan"}),
    P("flash a notification saying the oven is ready", "system_notify", {"message": "The oven is ready."}),
    # ── people, post and phone ────────────────────────────────────────────────
    P("any unread emails", "outlook_mail", {"action": "read_inbox", "unread_only": True}),
    P("show me my priority emails", "outlook_mail", {"action": "priority"}),
    P("draft an email to Priya saying the meeting has moved to Thursday", "outlook_mail",
      {"action": "draft", "to": "Priya", "body": "The meeting has moved to Thursday."}),
    P("ring the dentist", "telephony", {"action": "call", "to": "dentist"}),
    # ── knowledge and research ────────────────────────────────────────────────
    P("what's the difference between a list and a tuple in python", "programming_knowledge",
      {"query": "difference between a list and a tuple in Python"}),
    P("what is a man in the middle attack", "cyber_knowledge", {"query": "man in the middle attack"}),
    P("how does long term potentiation work", "neuro_knowledge", {"query": "long term potentiation"}),
    P("do a deep dive on solid state batteries", "research",
      {"action": "start", "topic": "solid state batteries"}),
    P("write a paper on urban heat islands", "research", {"action": "paper", "topic": "urban heat islands"}),
    P("what can you do with spreadsheets", "capabilities", {"query": "spreadsheets"}),
    P("let's play chess", "chess", {"action": "new_game"}),
    P("what's my chess rating at the moment", "chess", {"action": "rating"}),
)


# ── follow-ups: the previous turn is in context ───────────────────────────────

@dataclass(frozen=True)
class FollowUp:
    prev_user: str
    prev_reply: str
    text: str
    tool: str
    args: dict[str, Any]


FOLLOWUPS: tuple[FollowUp, ...] = (
    FollowUp("what's the weather in Leeds", "It's 12 degrees in Leeds with light rain.",
             "and tomorrow", "geo", {"action": "weather", "query": "Leeds", "when": "tomorrow"}),
    FollowUp("what's the weather in Bristol today", "Bristol's dry today, about 15 degrees.",
             "what about Cardiff", "geo", {"action": "weather", "query": "Cardiff", "when": "today"}),
    FollowUp("play some Radiohead", "Playing Radiohead.",
             "turn it up a bit", "peripherals", {"action": "volume_up"}),
    FollowUp("play some Radiohead", "Playing Radiohead.",
             "skip this one", "media_control", {"action": "next"}),
    FollowUp("open spotify", "Spotify's open.", "now pause it", "media_control", {"action": "pause"}),
    FollowUp("search the web for cheap standing desks", "I found several, starting around 120 pounds.",
             "search for ones with a memory function", "web_search",
             {"query": "standing desks with memory function"}),
    FollowUp("find my invoice pdf", "There's one called invoice 0412 in your Work folder.",
             "open it", "find_files", {"query": "invoice", "type": "pdf", "action": "open"}),
    FollowUp("remind me at 6 to start dinner", "Done - I'll remind you at six.",
             "actually what reminders do I have", "reminder", {"action": "list"}),
    FollowUp("how much memory am I using", "About 54 percent of your 16 gigabytes.",
             "and how much space is left on my drives", "resource_status", {"action": "storage"}),
    FollowUp("start a 25 minute focus session", "Focus block started for 25 minutes.",
             "how long is left", "focus", {"action": "status"}),
    FollowUp("what's 15 percent of 240", "That comes to 36.",
             "and 20 percent", "calculate", {"expression": "20% of 240"}),
    FollowUp("close notepad", "Notepad is closed.", "open it again", "open_app", {"app_name": "Notepad"}),
)


# ── multi-step flows, written out in full ─────────────────────────────────────

@dataclass(frozen=True)
class Step:
    tool: str
    args: dict[str, Any]
    result: str
    ok: bool = True


@dataclass(frozen=True)
class Flow:
    text: str
    steps: tuple[Step, ...]
    reply: str


FLOWS: tuple[Flow, ...] = (
    Flow("open notepad and then tell me how much memory I'm using",
         (Step("open_app", {"app_name": "Notepad"}, "Opened Notepad."),
          Step("resource_status", {}, "CPU 21%, memory 57% (9.1 of 16 GB), disk 66% full.")),
         "Notepad's open, and you're using about 9.1 of your 16 gigabytes of memory."),
    Flow("check the weather in Leeds and set a timer for ten minutes",
         (Step("geo", {"action": "weather", "query": "Leeds"}, "Leeds now: 9C, overcast, light wind."),
          Step("reminder", {"action": "timer", "minutes": 10}, "Timer set for 10 minutes.")),
         "It's 9 degrees and overcast in Leeds, and your ten minute timer is running."),
    Flow("pause the music then remind me in half an hour to stretch",
         (Step("media_control", {"action": "pause"}, "Paused."),
          Step("reminder", {"action": "add", "phrase": "remind me in half an hour to stretch"},
               "Reminder set for 30 minutes from now.")),
         "I've paused the music and I'll remind you to stretch in half an hour."),
    Flow("look for my cv and then search the web for interview advice",
         (Step("find_files", {"query": "cv"}, "Found 1 file: C:/Users/Owner/Documents/CV_2026.docx"),
          Step("web_search", {"query": "interview advice"},
               "1. Interview advice - example.org: Prepare three stories and ask two questions at the end.")),
         "Your CV is in Documents as CV 2026. For interviews, example.org says to prepare three stories and "
         "ask two questions at the end."),
    Flow("how much storage is left and what's on my calendar tomorrow",
         (Step("resource_status", {"action": "storage"}, "C: 198 GB free of 476 GB."),
          Step("notion_workspace", {"action": "upcoming_events", "days": 1},
               "Tomorrow: 10:00 team stand-up; 14:30 dentist.")),
         "You've got 198 gigabytes free on C. Tomorrow you have the team stand-up at ten and the dentist "
         "at half past two."),
    Flow("grab a screenshot and work out why my script is failing",
         (Step("capture_screen", {}, "Screenshot captured (2560x1440)."),
          Step("screen_read", {"prompt": "what is wrong on this screen"},
               "The screen shows a Python traceback: ModuleNotFoundError for 'requests'.")),
         "There's a Python error on screen: the requests module isn't installed."),
    Flow("look up the weather in Bristol and remind me to pack a coat in two hours",
         (Step("geo", {"action": "weather", "query": "Bristol"}, "Bristol now: 8C, clear, getting colder tonight."),
          Step("reminder", {"action": "add", "phrase": "remind me in two hours to pack a coat"},
               "Reminder set for 2 hours from now.")),
         "It's 8 degrees and getting colder in Bristol, and I'll remind you about the coat in two hours."),
    Flow("launch spotify, then lower the volume a little",
         (Step("open_app", {"app_name": "Spotify"}, "Opened Spotify."),
          Step("peripherals", {"action": "volume_down"}, "Volume lowered to 30%.")),
         "Spotify's open and the volume is down to thirty percent."),
    Flow("check my inbox and then read out my calendar for tomorrow",
         (Step("outlook_mail", {"action": "read_inbox"}, "3 unread: Priya (Meeting moved); HR (Payslip); "
                                                         "Amazon (Dispatch note)."),
          Step("notion_workspace", {"action": "upcoming_events", "days": 1}, "Tomorrow: 09:30 design review.")),
         "You have three unread, from Priya, HR and Amazon. Tomorrow you've got a design review at half nine."),
    Flow("open the calculator and then work out 15 percent of 240",
         (Step("open_app", {"app_name": "Calculator"}, "Opened Calculator."),
          Step("calculate", {"expression": "15% of 240"}, "15% of 240 is 36.")),
         "The calculator's open, and fifteen percent of 240 is 36."),
    # partial failures: the reply must say which part did not work
    Flow("open notepad and then check the weather in Leeds",
         (Step("open_app", {"app_name": "Notepad"}, "Opened Notepad."),
          Step("geo", {"action": "weather", "query": "Leeds"}, "geo failed: weather service unavailable.",
               False)),
         "Notepad's open, but I couldn't get the Leeds forecast because the weather service isn't responding."),
    Flow("search the web for train times to Derby and remind me to leave at 4",
         (Step("web_search", {"query": "train times to Derby"}, "web_search failed: no network connection.",
               False),
          Step("reminder", {"action": "add", "phrase": "remind me to leave at 4"},
               "Reminder set for 16:00.")),
         "I couldn't search for train times because there's no connection, but I've set your reminder for four."),
    Flow("find my cv and tell me how much memory I'm using",
         (Step("find_files", {"query": "cv"}, "No files matched 'cv'.", False),
          Step("resource_status", {}, "CPU 14%, memory 49% (7.8 of 16 GB), disk 63% full.")),
         "I couldn't find a file called CV, but you're using about 7.8 of your 16 gigabytes of memory."),
)


__all__ = ["FLOWS", "FOLLOWUPS", "Flow", "FollowUp", "P", "PHRASEBANK", "Phrase", "Step"]
