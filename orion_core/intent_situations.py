"""
Common sense for routing: what people SAY when they need a tool.

Every tool describes itself by what it does ("sets display brightness",
"reports system load"). People rarely ask that way. They describe their
situation and expect the obvious help: "the screen is blinding me" means
brightness, "everything is lagging" means look at the machine's load, "I
always forget my tablets" means a reminder. A word matcher cannot make that
leap and the intent network only ever saw the tools' own wording, so these
requests ranked the wrong tool first.

This is that leap, written down: for each tool, the problems and needs it is
the natural answer to, in everyday British speech. The intent network trains
on them alongside each tool's own phrases (intent_brain._extra_phrases), so
it learns the situation -> tool mapping and generalises it to wordings it has
not seen. Nothing here is copied from the routing evaluation set
(tests/data/intent_routing_cases.py); a check keeps it that way.
"""

from __future__ import annotations

SITUATIONS: dict[str, tuple[str, ...]] = {
    "peripherals": (
        "the screen is blinding me", "the display is far too bright tonight",
        "it is too dark to see the screen", "everything is way too loud",
        "the sound is barely audible", "my ears are ringing from the volume",
        "it's late so make it quieter", "the glare is giving me a headache",
        "lock the computer i'm stepping away",
    ),
    "process_governor": (
        "some program is devouring the processor", "which program is hogging the memory",
        "an app has frozen and won't respond", "find what's making the fans spin",
        "stop the program that's stuck",
    ),
    "geo": (
        "will it rain later on", "should i bring an umbrella",
        "is it cold outside right now", "what should i wear today given the weather",
        "is it going to be sunny this afternoon", "how chilly is it out there",
        "good day for drying washing outdoors",
        "i need a chemist that's open", "where can i get cash round here",
        "i'm starving, anywhere to eat close by", "my car needs fuel, where's the closest",
        "how long would it take to drive to york", "is it far to bristol from here",
        "draw the drive to cardiff on the map", "what time does it go dark tonight",
    ),
    "reminder": (
        "my medication keeps slipping my mind", "don't let me forget the meeting",
        "nudge me in an hour", "i need to remember to call mum later",
        "ping me when it's time to leave", "make sure i don't miss the bin day",
        "i'll forget unless you tell me",
        "count down eight minutes for the eggs", "give me a shout when the pasta's done in twelve minutes",
    ),
    "calculate": (
        "split the bill between four of us", "how much is twenty percent off fifty pounds",
        "what's fifteen percent of the total", "how many grams is half a pound",
        "how many sleeps till my holiday", "what's twenty quid in euros",
        "toss a coin to decide for me", "how long ago was the first of june",
    ),
    "resource_status": (
        "everything is lagging", "the pc feels sluggish today",
        "the fans are roaring", "my computer keeps freezing up",
        "it takes forever for anything to load", "the machine is struggling badly",
        "is the computer overloaded", "why is everything so laggy",
        "am i running out of room on the hard drive", "is my ssd nearly full",
    ),
    "organise_files": (
        "my desktop is cluttered", "my files are all over the place",
        "everything in my folders is disorganised", "sort out this jumble of files",
        "my pictures folder is chaos", "group my documents into sensible folders",
    ),
    "finance": (
        "money is tight this month", "can i afford a holiday this year",
        "how much have i spent lately", "i'm worried about paying my bills",
        "am i overspending", "will i have enough left at the end of the month",
        "i want to save more money",
    ),
    "wellbeing": (
        "i'm feeling really stressed", "i've been anxious all week",
        "i can't sleep properly", "i'm feeling low today",
        "i think i'm burnt out", "i need to look after myself better",
        "i've been overwhelmed lately",
    ),
    "focus": (
        "i keep getting distracted", "help me concentrate for a while",
        "i can't get anything done", "i need some deep work time",
        "keep me on task for the next hour",
    ),
    "study": (
        "i have an exam coming up", "help me revise for my test",
        "quiz me on what i've been learning", "i keep forgetting what i studied",
        "make me some flashcards",
    ),
    "goals": (
        "i want to be able to run a half marathon by spring",
        "i did a 10k this morning", "i'm trying to save for a house deposit",
        "am i going to make my deadline", "i keep setting targets and dropping them",
        "my new year's resolution is slipping", "i put another hundred quid in savings",
        "i managed three hours of practice today", "i want to lose a stone",
    ),
    "voice_speaker_id": (
        "you keep answering other people", "only respond to my voice",
        "you mistake my friends for me", "ignore voices that aren't mine",
        "the tv keeps setting you off",
    ),
    "media_control": (
        "it's too quiet in here put something on", "i need some background music",
        "this song is annoying me", "i'm bored of this track",
        "the music is too loud for this call",
    ),
    "telephony": (
        "let my mum know i'm on my way", "give my brother a ring",
        "i need to phone the doctor's surgery", "call my partner for me",
    ),
    "catch_up": (
        "i've been away, what happened", "bring me up to speed",
        "what did i miss while i was out", "anything new since this morning",
    ),
    "morning_briefing": (
        "start my day for me", "what does today look like",
        "give me the rundown for this morning",
    ),
    "system_notify": (
        "put a message up on the screen", "show a banner reminding me to stretch",
    ),
    "companion": (
        "where did i leave off", "pick up where we stopped last time",
        "what was i doing yesterday", "remind me what i was working on",
    ),
    "query_intelligence": (
        "do you remember what i told you about my car", "what do you know about my family",
        "what's the name of my doctor", "you know my birthday don't you",
    ),
    "save_memory": (
        "keep in mind that i'm allergic to nuts", "remember i work late on fridays",
        "don't forget that i prefer the window seat", "make a note that my car is blue",
    ),
    "decision": (
        "i can't make up my mind", "help me choose between two options",
        "which should i go for",
    ),
    "reason": (
        "consider it deeply before you reply", "what are the pros and cons",
        "i need a careful considered answer",
    ),
    "executive": (
        "what should i focus on today", "sort out my priorities",
        "turn this mess of tasks into a plan",
    ),
    "flight_search": (
        "i want to fly somewhere cheap", "cheapest way to fly to rome",
        "find me a cheap getaway flight",
    ),
    "antivirus": (
        "i think my computer has a virus", "something dodgy is running on my pc",
        "scan for anything malicious",
    ),
    "breach_check": (
        "has my email been leaked anywhere", "was my password in a data breach",
    ),
    "backup": (
        "i'm scared of losing my work", "make a copy of my important files",
    ),
    "cleanup_review": (
        "my drive is nearly full", "free up some disk space",
        "get rid of the junk files",
    ),
    "audio_devices": (
        "your voice is not coming through", "you can't hear me properly",
        "use my headphones instead", "switch to the other microphone",
    ),
    "window_control": (
        "i lost the window i was using", "bring that window back to the front",
        "make that window bigger", "tuck everything away so i can see the desktop",
    ),
    "web_search": (
        "look this up online for me", "search the internet for",
        "find out online what people say about",
    ),
    "open_news": (
        "what's in the headlines", "what's happening in the world today",
    ),
    "momentum": (
        "where am i up to with my projects", "what's the next step on my project",
    ),
    "rewind": (
        "when did we last talk about the boiler", "how long ago did we discuss the holiday",
    ),
    "conversation_recall": (
        "what were we just talking about", "recap our chat so far",
    ),
    "awareness": (
        "what's on your mind right now", "what are you keeping an eye on",
    ),
    "self_repair": (
        "you keep getting this wrong, fix yourself", "something in you is broken, repair it",
    ),
    "diagnostics": (
        "are you working properly", "something seems off with you",
    ),
    "capabilities": (
        "what are you able to do", "can you do things like this",
    ),
    "outlook_mail": (
        "have i got any new emails", "did anyone reply to my email",
    ),
    "proactive_check": (
        "is there anything i should know about", "anything urgent i need to sort",
    ),
}


#: Everyday requests for the tools that had none: the core voice tools first.
#: Same rule as above - written fresh, never taken from the routing
#: evaluation set, with a four-word-overlap check before measuring.
EVERYDAY: dict[str, tuple[str, ...]] = {
    "open_app": (
        "get word up for me", "i want to listen to something on spotify, start it",
        "fire up the calculator app", "bring up my email program", "load steam",
        "i need excel running", "start a new browser window",
    ),
    "close_app": (
        "i'm done with discord, shut it", "quit the music player",
        "get rid of excel", "kill that frozen app", "close everything except chrome",
    ),
    "find_files": (
        "where did i put the invoice from march", "i can't find my passport scan",
        "dig out the photos from my holiday folder", "which folder has my coursework in",
        "locate the spreadsheet with the budget", "hunt down that pdf the bank sent",
    ),
    "process_file": (
        "go through this document and tell me the main points", "read the attached report",
        "what does this contract actually say", "pull the numbers out of this spreadsheet",
        "give me the gist of the file on my desktop",
    ),
    "file_controller": (
        "make a new folder called receipts", "rename the file to final version",
        "move the notes into the archive folder", "put a text file on my desktop with my list",
        "delete that old draft",
    ),
    "fetch_url": (
        "what's on this web page", "read this link out to me", "grab the article from that url",
        "pull the text off this website",
    ),
    "clipboard_operate": (
        "paste what i just copied", "what's sitting on my clipboard",
        "put this phone number on the clipboard", "copy that for me",
    ),
    "capture_screen": (
        "grab an image of my screen", "save a picture of what's on my monitor",
        "screenshot this for me please",
    ),
    "vision_analyse": (
        "what's in front of the camera", "peek through the camera for me",
        "can you see what i'm holding", "read the text on this label for me",
        "describe what you can see", "read what's in my notepad window",
    ),
    "sound_sense": (
        "what song is playing right now", "what's that beeping noise",
        "is someone at the door, i heard something", "listen to this tune and name it",
        "how fast is this track",
    ),
    "web_search": (
        "find out what time the shop opens", "look up reviews for that phone",
        "what are people saying online about the update", "check the internet for the answer",
    ),
    "browser_control": (
        "open the bbc website", "go to youtube in the browser", "load my bank's website",
        "take me to the weather site", "open a new tab with google",
    ),
    "web_control": (
        "agree to the cookie banner", "close this annoying popup",
        "switch to the other tab", "fill in this form for me", "download the file on this page",
    ),
    "desktop_control": (
        "click the blue button", "type my email address in there", "scroll down a bit",
        "press enter", "drag that window to the left", "move the cursor to the search box",
    ),
    "window_control": (
        "i lost the window i was using", "bring that window back to the front",
        "make that window bigger", "tuck everything away so i can see the desktop",
        "put this window on my second screen", "hide all my windows",
    ),
    "recall_conversation": (
        "what did i tell you about my landlord", "did we ever talk about holidays",
        "remind me what i said about the car", "what was the name i mentioned earlier",
    ),
    "telephony": (
        "let my mum know i'm on my way", "give my brother a ring",
        "i need to phone the doctor's surgery", "call my partner for me",
        "text her that i'm stuck in traffic", "ring me back in five minutes",
    ),
    "messaging": (
        "drop the group chat a message", "send a whatsapp to dad",
        "tell kieran i'm running late on telegram", "message my friend on discord",
    ),
    "notion_workspace": (
        "stick bread on the shopping list in notion", "what's on my schedule this week",
        "add a meeting on friday to my calendar", "show me my notion tasks",
        "tick off the shopping task",
    ),
    "agent_dispatch": (
        "get a second opinion from one of your specialists", "ask the coding expert about this",
        "have the marketing agent weigh in", "let a specialist handle this question",
        "have the files specialist clear the junk off my disk",
        "get your file operator to tidy my downloads properly",
        "ask the triage specialist what you still can't do",
        "have the reliability specialist find out why you froze",
    ),
    "execute_plan": (
        "do these three things one after the other", "first open chrome then check my mail",
        "run this sequence of steps for me", "work through this list in order",
    ),
    "undo": (
        "undo that", "put it back how it was", "undo your last change",
        "i didn't mean that, take it back", "restore the file you deleted",
    ),
    "document_export": (
        "turn this into a word document", "build slides out of my notes",
        "export a brief i can send", "produce a powerpoint about the project",
    ),
    "research": (
        "do a deep dive on electric cars", "write me a proper report on sleep science",
        "thoroughly look into how the internet began", "research this topic in depth",
    ),
    "mission": (
        "how's my big project going", "show the mission board", "add a task to the orion mission",
        "what's left to do on the game",
    ),
    "globe": (
        "spin the earth to japan", "show me new york on the globe", "zoom into london",
        "turn on the rain radar", "show where the earthquakes are",
    ),
    "aviation": (
        "what plane is flying over us", "which helicopter is circling above",
        "track the flight from dubai", "how many aircraft are around here",
    ),
    "interface_control": (
        "show me the command deck", "become the sphere again", "go back to your face",
        "open the chess page", "bring up the camera view",
    ),
    "protocol": (
        "run my morning routine", "start the wind down sequence",
        "set up a sequence for movie night", "list my protocols",
    ),
    "shutdown_orion": (
        "go to sleep orion", "that's all for tonight, power down", "turn yourself off",
        "shut down orion",
    ),
    "restart_orion": (
        "give yourself a fresh restart", "restart orion please", "turn yourself off and back on",
    ),
    "screen_read": (
        "decode this error for me", "help me with what's on my screen",
        "which button should i click here", "explain this dialog box",
    ),
    "chess": (
        "fancy a chess match", "what's the best move here",
        "how good are you at chess", "review the game i just played",
    ),
    "perception": (
        "keep an eye on the room", "tell me if my phone leaves the desk",
        "what objects can you see", "watch the door while i'm away",
    ),
    "display_info": (
        "how many monitors have i got", "what resolution is this screen",
        "name my left monitor the gaming screen",
    ),
    "system_startup": (
        "start yourself when windows boots", "stop launching at login",
        "what browser opens my links",
    ),
    "ai_mode": (
        "are you online or offline", "what brain is running you",
        "pick the least expensive model",
        "switch to your local brain", "stop using the cloud entirely",
        "think with your own model from now on", "go back to the cloud brain",
        "use hybrid mode for a while",
    ),
    "language_tutor": (
        "teach me some spanish", "how do you say thank you in french",
        "give me a german lesson", "test me on italian words",
    ),
    "dev_workbench": (
        "run my project's tests", "show me the start of main.py",
        "make a new python project", "analyse my code repo",
    ),
    "debugger": (
        "step through my script", "pause my code at line ten", "inspect the variable while it runs",
    ),
    "docker": (
        "which containers are up", "restart the database container", "show the container logs",
    ),
    "codebase_copilot": (
        "where is this function defined", "review my whole project for bugs",
        "what depends on this module",
    ),
    "programming_knowledge": (
        "explain recursion simply", "lists versus tuples, explain",
        "when would i use a hash map",
    ),
    "cyber_knowledge": (
        "what is phishing", "explain how ransomware works", "how does two factor login help",
    ),
    "neuro_knowledge": (
        "how do neurons talk to each other", "what does the hippocampus do",
        "explain long term potentiation",
    ),
    "product_research": (
        "is this product worth selling", "how crowded is the pet toy market",
        "score this dropshipping idea",
    ),
    "strategy": (
        "weigh up every possible option", "compare all the ways to launch this",
    ),
    "standing_questions": (
        "alert me to any tesla news", "keep watching for updates on the election",
    ),
}


#: The rest of the tools: specialist suites and ORION's own settings.
EVERYDAY.update({
    "antigravity_research": ("send a specialist agent to investigate this publicly",
                             "have an outside agent gather evidence on this"),
    "audio_studio": ("clean up my vocal recording", "level the volume on my podcast take",
                     "bundle the processed stems for the producer"),
    "autoplan": ("sort this whole job out yourself step by step", "take care of all of it on your own",
                 "plan and carry out this task without me"),
    "avatar": ("follow me with your eyes", "stop tracking my face", "what state is your face in"),
    "brand_growth": ("how do i get more people to know my brand", "help me grow the agency's audience",
                     "set us apart from the big agencies"),
    "briefing": ("give me the latest news briefing", "headlines from around the globe please",
                 "evening update on everything"),
    "business_advisor": ("what should i charge new clients", "ideas to expand the side business",
                         "advise me on my offer"),
    "campaign_pipeline": ("move this client to the next stage", "show the campaign board",
                          "log how the latest campaign performed"),
    "code_changes": ("what did you change in your own code", "explain your latest code edits",
                     "list the functions you rewrote"),
    "commerce_hub": ("show my ecommerce overview", "which products should i stock",
                     "open the store dashboard"),
    "community_share": ("package my research to send to a friend", "load the pack my mate sent me",
                        "export my notes as a shareable pack"),
    "competitor_intel": ("break down a rival's online shop", "how does my competitor sell",
                         "compare my offer with theirs"),
    "creator_intel": ("rate my video hook", "check my script before i film",
                      "what's trending for creators this week"),
    "cursor_overlay": ("show the glow around the pointer", "hide the cursor halo",
                       "let me see where you're clicking"),
    "cyber_curriculum": ("what's next in my security course", "which certification should i go for",
                         "mark this cyber module as done", "teach me cybersecurity code",
                         "start my cyber learning path", "quiz me on secure coding",
                         "teach me electronics fundamentals", "how do i defend against destructive hardware",
                         "explain signed firmware updates", "teach me in small steps",
                         "i lose focus when learning", "make this lesson simpler",
                         "pause my lesson", "resume my lesson", "repeat this learning step"),
    "display": ("show that as a card on screen", "draw a visual summary for me",
                "present the answer graphically"),
    "draft_report": ("format this as a proper written report", "turn my notes into a structured report",
                     "draft a report with a chart"),
    "elevenlabs_voice": ("change how your voice sounds", "use a deeper voice",
                         "which voice preset are you on"),
    "emotion": ("smile for me", "make a surprised face", "put your expression back to normal"),
    "entertainment": ("sum up this youtube video", "popular videos on youtube at the moment",
                      "what has this channel posted lately"),
    "expand_mind": ("search your study library", "what do your offline notes say about stoicism",
                    "how many books are in your study corpus"),
    "forge": ("build yourself a new tool for this", "code a new ability for bus timetables",
              "make a capability that does this"),
    "founder_knowledge": ("what would hormozi say about pricing", "give me a founder playbook",
                          "how did bezos think about customers"),
    "gaming": ("launch a game from my steam library", "list my steam games",
               "start rocket league"),
    "gesture_control": ("let me control things with my hands", "turn on hand gestures",
                        "switch off the gesture tracking"),
    "ingest": ("bring this directory into your library", "absorb these pdfs", "index my notes folder"),
    "instagram_intel": ("find influencers in the fitness niche", "who should i partner with on instagram",
                        "weekly instagram report"),
    "job": ("carry on with that out of sight", "what jobs are running", "cancel the background task"),
    "knowledge_pack": ("which knowledge packs do you have", "consult the marketing pack",
                       "remove the dropshipping pack"),
    "learn": ("learn this for good", "take this in permanently", "forget what i taught you about that"),
    "literature_vault": ("add this paper to my library", "summarise the citations in my papers",
                         "what mechanisms do my papers mention"),
    "max_zoom_in_globe": ("zoom right in to street level", "get as close to the ground as you can"),
    "mcp": ("which external servers are connected", "reconnect the notion server",
            "add a new mcp connector"),
    "muscle_memory": ("watch me do this once and repeat it", "record these steps for later",
                      "replay the routine you learned"),
    "navigation_trace": ("which clicks went wrong today", "how did you find that button"),
    "patch_notes": ("what's new in your latest version", "show your changelog"),
    "pentest_lab": ("start a hacking practice lab", "give me a hint for this exercise",
                    "walk me through the next step of the lab"),
    "phone_action": ("open maps on my phone", "send this to my mobile", "set an alarm on my phone"),
    "plugin": ("which plugins are installed", "turn off that plugin", "make me a new plugin"),
    "privacy_guard": ("is this text safe to share", "remove anything private from this",
                      "check this for leaked keys"),
    "proactive_report": ("generate this week's progress report", "produce the growth numbers for the month"),
    "read_documents": ("read all the documents in that directory", "answer questions across these documents"),
    "second_brain": ("how are my projects linked", "what does your knowledge graph know about james",
                     "show the timeline for this project"),
    "security_recon": ("scan my raspberry pi's ports", "look up this cve", "scan the network real quick",
                       "what devices are connected to my wifi",
                       "authorise my test machine as a target"),
    "security_watch": ("is my firewall switched on", "which ports are listening on this pc",
                       "anything suspicious running"),
    "self_changes": ("which of your files changed recently", "what's different since you started"),
    "semantic_compare": ("which of these phrases is closest in meaning", "rank these sentences by similarity"),
    "sentinel": ("alert me when the charge gets low", "turn on background monitoring",
                 "keep an eye on the pc's health"),
    "conversation_recording": ("record this conversation", "stop recording our conversation",
                               "show my recorded conversations", "read the meeting transcript"),
    "sing": ("give us a song", "sing happy birthday to my mum", "can you sing me to sleep",
             "make up a little tune about my cat and sing it", "sing that one again",
             "keep a copy of the song you just sang"),
    "skill": ("which skills have you got", "add the skill pack from that directory"),
    "social_media": ("post this video to tiktok", "reply to my instagram messages"),
    "speaker_id": ("was that a man or a woman", "guess the age of whoever just spoke"),
    "tiktok_intel": ("is this product trending on tiktok", "predict whether this will go viral"),
    "token_usage": ("how many tokens have i used", "remaining allowance on my plan"),
    "transcript": ("save this conversation", "export our chat log"),
    "ui_theme": ("change the colours of the dashboard", "make the interface purple"),
    "ui_reset": ("put the visuals back to normal",),
    "vision_verify": ("what can i click in this window", "list the buttons on screen"),
    "voice_tone": ("how do i sound today", "do i sound stressed"),
    "web_automation": ("browse the shop and add it to my basket while i watch",
                       "log in to the site and check my orders"),
    "workflow": ("create a workflow that emails me the summary", "run my reporting workflow"),
    "workflow_patterns": ("spot routines i keep repeating", "which tasks do i always do together"),
    "workspace_control": ("set up my coding workspace", "arrange my windows for work"),
})


#: More for the tools that had the fewest (two to five): routing still missed
#: most often there. Added to what each already has; same rules as above.
#: Measured 6 Oct 2026 (development half, words only, 5 seeds): top-3
#: 85.8% -> 88.5%; with the sentence encoder warm, top-1 77.7% -> 79.4%.
for _tool, _more in {
    # Mark XXXVII (23 Oct 2026), from the owner's sessions: each of these got
    # the wrong tool or none (resource_status for "your updates", "not quite
    # equipped" for a song, an unsupported close_window, nothing marked done).
    "patch_notes": (
        "tell me about your updates", "what's changed with you since the update",
        "go through the patch notes", "what improvements came in the new version",
    ),
    "self_changes": (
        "what got changed in you today", "did anything about you change since yesterday",
    ),
    "code_changes": (
        "talk me through the code you changed", "what did the last update do to your code",
    ),
    "sing": (
        "sing something for me", "could you sing a little song", "sing us a tune",
    ),
    "web_search": (
        "what events are on at my university", "anything happening on campus this week",
        "what's the newest ai model out", "which laptop is best for ai right now",
        "find the latest reviews online",
    ),
    "window_control": (
        "close that window", "shut this window", "get that window off my screen",
    ),
    "awareness": (
        "i've done all those tasks", "i finished everything on that list",
        "tick them all off, they're done", "those jobs are all done now",
    ),
    "workflow": (
        "run my morning check every weekday at half seven",
        "do that routine for me automatically every friday evening",
        "what did my automation turn up", "stop that routine running on mondays",
    ),
    "query_intelligence": (
        "what do you have saved about my mum", "remind me what my sister's husband is called",
        "do you know my blood type", "what did i say my wifi password was",
        "who's my gp", "what's my landlord's name again", "what car do i drive",
        "have i told you where i work",
    ),
    "conversation_recall": (
        "what did you just say", "go back over what we discussed",
        "give me the gist of our conversation", "what were the main points you made",
        "repeat your last answer", "summarise the chat we've been having",
    ),
    "rewind": (
        "when was the first time i mentioned the wedding", "how long ago did i bring up moving house",
        "find when we spoke about the dentist", "what day did i ask about the boiler",
        "go back to what i said last tuesday", "when did i last mention my exam",
    ),
    "awareness": (
        "what are you watching out for", "what have you noticed lately",
        "anything you're tracking for me right now", "what are you doing behind the scenes",
        "what are you paying attention to",
    ),
    "executive": (
        "tear my business idea apart", "challenge my plan before i commit",
        "what could go wrong with this decision", "be my chief of staff for the day",
        "red team my strategy", "give me a brutally honest assessment of this plan",
    ),
    "gesture_control": (
        "stop tracking my hands", "let me wave to skip songs", "use my hands as a mouse",
        "disable the hand controls", "pinch to click mode on",
    ),
    "self_repair": (
        "you're broken, sort yourself out", "debug yourself", "that feature of yours keeps failing, mend it",
        "repair whatever went wrong in you", "patch your own mistake", "you crashed again, fix the cause",
    ),
    "capabilities": (
        "are you able to make spreadsheets", "could you edit a photo for me",
        "do you know how to send texts", "what kinds of things can you help with",
        "is that something you can manage", "list your skills",
    ),
    "voice_speaker_id": (
        "you're listening to the telly instead of me", "ignore everyone else's voice",
        "train on my voice so you know it's me", "my partner keeps triggering you",
        "recognise who is talking", "you replied to my flatmate",
    ),
    "codebase_copilot": (
        "where in the code is the login handled", "which file defines this class",
        "search my project for where the config is loaded", "find all uses of this function",
        "explain how this repo is structured", "what calls this method",
    ),
    "programming_knowledge": (
        "what is time complexity", "how does a linked list work", "teach me about closures in python",
        "difference between a class and an object", "when is a dictionary better than a list",
        "what's dependency injection", "explain the singleton design pattern",
    ),
    "instagram_intel": (
        "which reels are blowing up in my niche", "what's selling on instagram right now",
        "find instagram accounts in the beauty space", "how are fitness creators growing on insta",
        "spot trending products on instagram",
    ),
    "creator_intel": (
        "give feedback on my video idea", "is my thumbnail any good",
        "critique my youtube script", "how can i make my intro more gripping",
        "what should my next video be about",
    ),
    "finance": (
        "i'm skint until payday", "will i have enough to pay the bills",
        "how's my budget looking", "can i cover the rent this month",
        "where is all my money going",
    ),
    "organise_files": (
        "tidy up my downloads", "my documents folder is chaos",
        "sort these files into folders by type", "clean up the mess on my desktop",
        "group my photos into folders",
    ),
    "flight_search": (
        "find me flights to barcelona in may", "how much to fly to new york in the summer",
        "cheapest dates to fly to tenerife", "i need a return flight to glasgow",
        "look up budget flights for a weekend away",
    ),
    "open_news": (
        "open the top story", "read me the third headline", "show me that news article",
        "open the latest tech news", "pull up the bbc headlines",
    ),
    "browser_control": (
        "take me to the netflix website", "go to my bank's site",
        "navigate to wikipedia", "visit the argos website",
    ),
    "web_automation": (
        "fill in this form on the website for me", "go on the site and search for a cheap tent",
        "click through the checkout while i watch", "book it on the website step by step",
        "find a used car on autotrader while i watch you click",
    ),
    "workflow_patterns": (
        "notice what i do every morning", "what habits do i repeat on my computer",
        "find patterns in how i work", "what do i always open together",
        "suggest automations from my routines",
    ),
    "reason": (
        "consider every angle first", "weigh up both options before answering",
        "give me a well reasoned answer", "take your time and reason it out",
        "should i do it, think hard", "work through the logic step by step",
    ),
    "file_controller": (
        "list the contents of my pictures folder", "show me the files in that folder",
        "delete the old copy", "copy this file to my desktop",
        "what's inside the projects folder",
    ),
    "resource_status": (
        "my pc is running slow","why is my fan so loud",
        "how much memory is free", "is something hogging the cpu",
    ),
    "reminder": (
        "nudge me to stretch every hour", "make sure i don't forget my keys",
        "prompt me to take a break later", "help me remember to ring the garage",
    ),
}.items():
    EVERYDAY[_tool] = tuple(dict.fromkeys((*EVERYDAY.get(_tool, ()), *_more)))
del _tool, _more


def phrases() -> dict[str, list[str]]:
    merged = {tool: list(items) for tool, items in SITUATIONS.items()}
    for tool, items in EVERYDAY.items():
        merged.setdefault(tool, [])
        merged[tool].extend(p for p in items if p not in merged[tool])
    return merged


__all__ = ["SITUATIONS", "phrases"]
