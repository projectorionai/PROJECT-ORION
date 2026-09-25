"""
Tool vocabulary — the words people actually say, mapped to the tool they mean.

Shadow-evaluating the tool resolver against 67 real phrases gave 76.5% recall:
it would have removed the tool the model needed on roughly one turn in four.
Diagnosing the misses found a single cause, and it was not the scoring
algorithm. The words were simply not there:

    "weather"     appears nowhere in the geo tool's schema
    "quiz"        appears nowhere in the study tool's schema
    "summarise"   appears nowhere in process_file's schema
    "yesterday"   appears nowhere in recall_conversation's schema

No lexical scorer can match a term that does not exist in the document. Tool
descriptions are written in capability language — precise, technical, aimed at a
model reading a schema — and people ask in ordinary speech. This module is the
bridge: a curated set of invoking phrases per tool. With it, recall on the
held-out half of the evaluation set went from 72.7% to 100%.

**This is document expansion, not a synonym hack.** It is the standard
information-retrieval answer to a vocabulary mismatch between queries and
documents, applied at index time. The terms are chosen from what each tool *is*,
not from the phrases in any test set — the evaluation is split into a
development half and a held-out half precisely so that distinction can be
checked rather than asserted.

**It costs the model nothing.** These terms are used only by ORION's own local
scorers; they are never added to the schema sent to a provider, so the 31,600
token tool surface does not grow by a byte.

**One hard rule, enforced by a test.** A term here must never be another tool's
whole name. Vocabulary is weighted like a tool's own name, so putting "study" in
the research tool's vocabulary made the literal query "study" rank research
above study — a capability stolen by its neighbour. Twelve such collisions
existed on the first draft; ``collisions()`` finds them and
``test_tool_vocabulary`` fails on any that return. For a multi-word tool name
like ``morning_briefing`` this costs nothing: the name already tokenises to
"morning" and "briefing".

Otherwise, adding a term here is cheap and safe. Leaving one out costs recall.
When in doubt, include the word — the risk of over-inclusion is a slightly less
precise ranking, and the risk of omission is a capability quietly disappearing.
"""

from __future__ import annotations

from typing import Iterable

#: tool name -> the ordinary words and phrases someone uses to ask for it.
#: Written per tool from its purpose. Kept as plain strings; the resolver
#: tokenises them the same way it tokenises a description.
VOCABULARY: dict[str, str] = {
    # ── learning and recall ──────────────────────────────────────────────────
    "study": (
        "quiz test me flashcard flashcards revise revision memorise memorize "
        "by heart practice cards deck due review studying revising exam "
        "remember facts drill"
    ),
    "language_tutor": (
        "spanish french german italian portuguese language vocabulary translate "
        "how do you say phrase lesson speak fluent conjugate word for"
    ),
    "expand_mind": (
        "explain teach knowledge corpus topic subject look up in your material "
        "what do you know about"
    ),
    "neuro_knowledge": (
        "neuroscience brain neural neuron cortex synapse psychology cognition "
        "neuroengineering nervous system"
    ),
    "programming_knowledge": (
        "programming code algorithm data structure complexity software "
        "engineering pattern syntax"
    ),
    "literature_vault": (
        "paper papers academic journal citation pdf article literature review"
    ),

    # ── work rhythm ──────────────────────────────────────────────────────────
    "focus": (
        "concentrate deep work pomodoro session block timer stretch "
        "distraction free head down crack on get stuck in"
    ),
    "momentum": (
        "finish ship next action stuck progress momentum move forward"
    ),
    "wellbeing": (
        "sleep sleeping slept mood feeling energy tired stress stressed "
        "wellness health check in how am i doing rested"
    ),
    "decision": (
        "decide judgement judgment calibration calibrated brier prediction "
        "forecast confidence bet call it right wrong"
    ),

    # ── money ────────────────────────────────────────────────────────────────
    "finance": (
        "money spend spent expense expenses cost budget runway cash balance "
        "income revenue afford pounds bank account financial savings bills paid"
    ),
    "commerce_hub": (
        "product opportunity ecommerce store selling margin supplier dropship"
    ),
    "product_research": (
        "winning products dropshipping niche score opportunity"
    ),

    # ── looking things up ────────────────────────────────────────────────────
    "flight_search": (
        "flight flights fly flying plane aeroplane airplane airline airfare "
        "fare ticket tickets airport departure arrival cheapest flight "
        "book a flight how much to fly get to by air direct nonstop layover "
        "stopover return one way round trip business class economy travel to"
    ),
    "web_search": (
        "search google look up online web find on the internet what is the "
        "price of who is"
    ),
    "research": (
        "deep dive investigate thesis paper write me a report in depth "
        "thorough findings sources"
    ),
    "recall_conversation": (
        "remember talked about said discussed conversation earlier yesterday "
        "last time previously we spoke chat history what did i say"
    ),
    "second_brain": (
        "remember about know about my project knowledge graph what do you know "
        "stored facts"
    ),
    "rewind": (
        "verbatim exact words replay what was said earlier scroll back"
    ),
    "query_intelligence": (
        "memory search stored remember lookup fact what have you learned about me "
        "what do you know about me what did i tell you"
    ),
    "save_memory": (
        "remember this note that store keep in mind memorise for later"
    ),
    "learn": (
        "teach you remember permanently absorb take this in"
    ),

    # ── place ────────────────────────────────────────────────────────────────
    "geo": (
        "weather temperature forecast rain where is location place city town "
        "country postcode post code address distance how far map coordinates "
        "latitude longitude nearby directions"
    ),
    "globe": (
        "earth world map fly to zoom show me on the spin"
    ),
    "aviation": (
        "plane planes aircraft airplane aeroplane jet jets airliner helicopter "
        "overhead flying over above us in the sky up there what plane is that "
        "air traffic radar flight tracker flightradar live flights callsign "
        "transponder adsb altitude track a flight where is flight"
    ),

    # ── files ────────────────────────────────────────────────────────────────
    "find_files": (
        "find file find folder where did i save locate document search my "
        "computer look for a file where is my bring up pull up show me the "
        "open the document proposal report biggest files largest folders "
        "taking up space disk space pdfs downloaded last week made in "
        "modified created size over gigabytes"
    ),
    "process_file": (
        "fetch local file read this summarise summarize document pdf spreadsheet csv "
        "what does this say tell me what is in analyse this file contents"
    ),
    "fetch_url": (
        "fetch url web link webpage page online file read the link website "
        "download article markdown retrieve remote document"
    ),
    "read_documents": (
        "read these documents across all of them folder of files go through"
    ),
    "file_controller": (
        "file folder path directory copy move rename delete create folder"
    ),
    "organise_files": (
        "organise organize tidy sort clean up my folder messy desktop declutter"
    ),
    "ingest": (
        "import add to knowledge base index this folder absorb"
    ),

    # ── seeing ───────────────────────────────────────────────────────────────
    "sound_sense": (
        "hear listen what was that noise sound what song is this music playing "
        "what is making that sound is that music or talking tempo bpm key of the song "
        "beat pitch frequency hz hum buzz whine beep alarm siren doorbell dog barking "
        "what can you hear audio recording soundtrack identify the sound"
    ),
    "vision_analyse": (
        "look see read the text image picture screenshot what does it say "
        "describe what you see camera photo ocr error message on screen "
        "what am i typing read my notepad read this page read the web page "
        "what is in my word document read the chat what does this window say "
        "exact text in the game on my screen "
        "pcb circuit board electronics inspect scan components solder markings "
        # The words people actually point a camera and say. Without them
        # "why is this motherboard dead" scored as a system-diagnostics
        # question and never reached the eyes that could look at it.
        "motherboard mainboard chip ic capacitor resistor diode transistor "
        "connector header pin trace fuse relay burnt scorched"
    ),
    "capture_screen": (
        "screenshot screen grab capture my screen what is on my screen picture "
        "of the screen"
    ),
    "screen_read": (
        "what should i do about this on my screen right now help me with this"
    ),
    "vision_verify": (
        "buttons controls elements ui what can i click on this window"
    ),
    "perception": (
        "watch camera webcam see me keep an eye continuous vision detect objects "
        "what objects yolo computer vision opencv edges colours colors motion "
        "movement grid of numbers pose body hand raised gesture mediapipe when you "
        "see trigger rule automation overlay"
    ),

    # ── doing things on the machine ──────────────────────────────────────────
    "desktop_control": (
        "click type press key scroll mouse cursor drag hotkey keyboard tab "
        "enter button move the mouse control my computer"
    ),
    "window_control": (
        "window minimise maximise close bring up switch to"
    ),
    "open_app": (
        "open launch start run app application program"
    ),
    "close_app": (
        "close quit exit kill shut down the app stop the program"
    ),
    "media_control": (
        "play pause skip next track volume louder quieter mute music"
    ),
    "peripherals": (
        "volume brightness wifi bluetooth screen off lock sleep hardware"
    ),
    "clipboard_operate": (
        "clipboard copy paste copied"
    ),
    "process_governor": (
        "processes cpu hog memory hog task manager kill process what is slowing"
    ),

    # ── the web ──────────────────────────────────────────────────────────────
    "web_automation": (
        "browser browse website navigate click on the page fill in the form "
        "log in to online account show me doing it"
    ),
    "web_control": (
        "cookies popup tab navigate accept reject banner"
    ),
    "browser_control": (
        "open url open website open link in browser"
    ),

    # ── ORION about himself ──────────────────────────────────────────────────
    "diagnostics": (
        "diagnostic health check working properly broken faulty everything ok "
        "self test why can you not what is wrong problems lagging slow "
        "performance latency"
    ),
    "capabilities": (
        "what can you do abilities capable features what are you able to"
    ),
    "find_tool": (
        "find a tool which tool can do search tools is there a way to "
        "look for a capability"
    ),
    "use_tool": (
        "run the tool call that tool use the tool named execute tool by name"
    ),
    "resource_status": (
        "cpu memory ram usage load system performance resources how hard are "
        "you working machine pressure disk"
    ),
    "token_usage": (
        "tokens quota limit budget how much left allowance credits spent "
        "how long have i got"
    ),
    "self_changes": (
        "what changed what did you change recently updated modified your code "
        "new since"
    ),
    "patch_notes": (
        "updates changelog release what is new version"
    ),
    "code_changes": (
        "your code source files functions you edited"
    ),
    "awareness": (
        "what are you aware of what have you been doing thinking noticing "
        "up to right now"
    ),
    "catch_up": (
        "catch me up brief me situation report where do things stand what is new"
    ),
    "self_repair": (
        "broken fix yourself incident error report a bug something went wrong"
    ),
    "restart_orion": "restart reboot yourself start again",
    "shutdown_orion": "shut down goodbye power off turn yourself off",

    # ── extending ORION ──────────────────────────────────────────────────────
    "forge": (
        "build a tool make yourself write a new capability create a tool new "
        "ability teach yourself to"
    ),
    "plugin": (
        "plugins extension add on installed enable disable doctor"
    ),
    "skill": "skills pack template prompt pack",
    "workflow": "automate repeatable routine chain of steps",
    "protocol": "macro named sequence one command routine",
    "mcp": "server connector external tools model context",

    # ── communication ────────────────────────────────────────────────────────
    "messaging": (
        "text message whatsapp telegram send a message tell them message him "
        "message her notify"
    ),
    "outlook_mail": (
        "email inbox mail outlook unread reply draft send an email"
    ),
    "notion_workspace": (
        "notion task tasks todo to do calendar event schedule my list"
    ),
    # Ringing a real phone, as opposed to handing the user's own handset a
    # pre-filled dialler. Deliberately does NOT claim "text" or "message" on
    # its own: messaging owns those, and telephony only sends an SMS when
    # somebody explicitly asks for one.
    # These two are easy to confuse and were confusing the router: each had
    # claimed the other's words. The distinction is WHO dials.
    #
    #   telephony     ORION rings YOU, from a Twilio number. Costs money.
    #   phone_action  hands YOUR OWN paired handset a pre-filled dialler or
    #                 message for you to tap. Costs nothing, sends nothing.
    "telephony": (
        "call me ring me phone me give me a call call my mobile "
        "ring me back call me back speak to me on the phone"
    ),
    "phone_action": (
        "on my phone from my phone open the dialler on my handset "
        "android mobile hand off to my phone"
    ),
    "reminder": (
        "remind alarm alert me wake me at nudge me later do not let me forget"
    ),

    # ── news and briefing ────────────────────────────────────────────────────
    "briefing": (
        "news headlines what is happening current events today world"
    ),
    "morning_briefing": (
        "daily brief start my day"
    ),
    "open_news": (
        "open the story read that article link to the news"
    ),

    # ── security ─────────────────────────────────────────────────────────────
    "security_recon": (
        "scan network ports nmap pentest penetration test vulnerability recon "
        "my network open ports"
    ),
    "breach_check": (
        "leaked password pwned data breach compromised my email exposed"
    ),
    "security_watch": (
        "security posture listening ports suspicious processes"
    ),
    "undo": (
        "undo revert take that back put it back reverse restore that "
        "cancel that change unwrite undelete never mind that"
    ),
    "cyber_knowledge": (
        "cybersecurity concept attack defence explain threat"
    ),
    "antivirus": (
        "defender malware virus protection scan my pc"
    ),
    "privacy_guard": (
        "safe to send redact private sensitive leak check this text"
    ),

    # ── play ─────────────────────────────────────────────────────────────────
    "chess": (
        "game play move board opening checkmate best move white black piece pawn"
    ),
    "gaming": (
        "steam epic library installed games play a game"
    ),
    "entertainment": (
        "youtube video watch summarise a video music"
    ),

    # ── work and business ────────────────────────────────────────────────────
    "agent_dispatch": (
        "specialist expert advice consult ask an agent second opinion"
    ),
    "business_advisor": (
        "business advice clients pricing offer agency grow revenue"
    ),
    "brand_growth": (
        "brand growth positioning marketing channel audience"
    ),
    "creator_intel": (
        "content tiktok reels shorts hook views creator posting"
    ),
    "mission": (
        "long running endeavour big goal objective"
    ),
    "executive": (
        "priorities what should i do first urgent important"
    ),
    "reason": (
        "think hard difficult question work it out carefully deeply consider"
    ),
    "strategy": (
        "options alternatives space explore possibilities trade offs"
    ),

    # ── voice and presence ───────────────────────────────────────────────────
    "audio_devices": (
        "microphone mic speaker headphones audio device input output sound"
    ),
    "elevenlabs_voice": (
        "voice accent how you sound speech synthesis elevenlabs"
    ),
    "emotion": (
        "face expression look mood on your face"
    ),
    "avatar": (
        "face tracking your face head"
    ),
    "voice_tone": (
        "how do i sound my voice tone stressed calm loud"
    ),
    "speaker_id": (
        # Gender only. It said "who is speaking", which it cannot answer, and
        # so it out-scored voice_speaker_id — the tool that compares a real
        # voiceprint — on exactly the question people ask most.
        "man or woman male female voice sounded gender of the speaker"
    ),
    "interface_control": (
        "switch page open the deck your interface panel tab show me the"
    ),
    "cursor_overlay": (
        "halo pointer highlight show where you click"
    ),
    "system_notify": (
        "alert banner notify me on screen warning"
    ),
    "backup": (
        "back up save a copy archive settings restore"
    ),
    "transcript": (
        "export this conversation save the chat log"
    ),

    # ── remaining surface (added after the first coverage pass) ──────────────
    "ai_mode": "online offline cloud local which brain are you using mode",
    "audio_studio": "audio track mix vocal stem master song production podcast",
    "autoplan": "do all of this by yourself plan it out multi step autonomously",
    "execute_plan": "run these steps in order carry out the plan step by step",
    "campaign_pipeline": "kanban client pipeline board deliverable agency stage",
    "cleanup_review": "tidy the repo unnecessary files github prep removable clutter",
    "codebase_copilot": "repository codebase whole project dependency graph hotspots refactor",
    "dev_workbench": "engineering workbench read the code run the tests repo",
    "debugger": "step through breakpoint debug this script pdb inspect variables",
    "docker": "container containers image images compose service",
    "community_share": "share export import pack publish community network",
    "companion": "continuity what was i doing pick up where we left off recap",
    "competitor_intel": "competitor rival their store their offer benchmark against",
    "conversation_recall": "time scoped what did we discuss on which day we spoke about",
    "cyber_curriculum": "training curriculum modules course syllabus red blue team",
    "pentest_lab": "hands on lab guided exercise practise attacking walkthrough",
    "display_info": "monitor monitors screens resolution dpi scaling how many displays",
    "document_export": "docx word document deck slides export a brief produce a file",
    "draft_report": "draft write up structured professional markdown html",
    "founder_knowledge": "hormozi vaynerchuk blakely bezos founder operator playbook",
    "gesture_control": "hand gestures wave webcam control with my hands",
    "instagram_intel": "instagram influencer partnership reel niche discovery",
    "tiktok_intel": "tiktok shop virality trend assessment",
    "job": "background dont wait run it in the background long running while we talk",
    "knowledge_pack": "pack packs offline knowledge installed corpus dropshipping",
    "max_zoom_in_globe": "zoom right in closest maximum street level surface",
    "muscle_memory": "record once replay repeat that task again same steps",
    "navigation_trace": "targeting history what did you click how did you find it",
    "proactive_check": "survey now check on things anything i should know",
    "proactive_report": "scheduled report daily business weekly progress generate a report",
    "sentinel": "ambient watch host health warn me proactively watching",
    "social_media": "post to social my account publish a post logged in real account",
    "standing_questions": "keep watching keep an eye on tell me when something new",
    "system_startup": "start with windows autostart boot on login default browser",
    "voice_speaker_id": "who is this who is speaking who said that enrol my voice recognise know it is me by name specific person only listen to me ignore other voices other people talking tell my voice apart from others voiceprint answer only my voice",
    "workflow_patterns": "keep repeating same sequence worth saving pattern i always",
    "workspace_control": "desktop layout arrange windows save my workspace restore layout",
}


def terms_for(tool: str) -> str:
    """The curated invoking vocabulary for *tool* ("" if it has none)."""
    return VOCABULARY.get(tool, "")


def coverage(tool_names: Iterable[str]) -> tuple[int, int, list[str]]:
    """How many of *tool_names* have vocabulary. Returns (have, total, missing)."""
    names = list(tool_names)
    missing = sorted(n for n in names if n not in VOCABULARY)
    return len(names) - len(missing), len(names), missing


def collisions(tool_names: Iterable[str]) -> list[tuple[str, str]]:
    """Vocabulary terms that are another tool's whole name — always a defect.

    Vocabulary is weighted like a tool's own name, so a term that IS another
    tool's name lets one capability outrank the tool actually being asked for.
    Only single-word tool names can collide: a multi-word name tokenises into
    ordinary words that are legitimately shared.
    """
    names = set(tool_names)
    single = {n for n in names if "_" not in n}
    found: list[tuple[str, str]] = []
    for tool, phrases in VOCABULARY.items():
        for term in set(phrases.split()):
            if term in single and term != tool:
                found.append((tool, term))
    return sorted(found)


__all__ = ["VOCABULARY", "collisions", "coverage", "terms_for"]
