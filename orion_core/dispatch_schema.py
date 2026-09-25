"""
Gemini function-calling tool schema (the ``TOOL_DECLARATIONS`` table).

Extracted from ``dispatcher.py`` (July 2026 improvement pass, Priority 2.1)
so the router logic and the ~1150-line declaration data live apart. Import
site is unchanged: ``from .dispatcher import TOOL_DECLARATIONS`` still works
because ``dispatcher`` re-exports it.
"""

from __future__ import annotations

from typing import Any

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "open_app",
        "description": "Launch a trusted host application, well-known web app (notion, gmail, github), or secure URL.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Application label, executable, path, web-app name, or URL."},
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "web_search",
        "description": "Open a secure web search for the supplied query.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "READ the contents of a specific http(s) web URL through the configured fetch MCP server and return clean Markdown, without merely opening a browser. Use this when asked to fetch/read a link, web page or online file. Continue a long result by calling again with the next start_index. For files already on this PC use process_file instead.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "url": {"type": "STRING", "description": "Exact http(s) URL to fetch."},
                "start_index": {"type": "INTEGER", "description": "Character offset for the next part of a long page (default 0)."},
                "max_length": {"type": "INTEGER", "description": "Maximum characters in this part, 1,000-20,000 (default 8,000)."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "flight_search",
        "description": (
            "Find flights and their prices between two places on a given day. "
            "Opens the flight search in a real browser, reads the results page "
            "and reports the options with airlines, times, durations, stops and "
            "fares, cheapest first. Use this whenever the user asks about "
            "flying somewhere, plane tickets, fares, airfare, or how much it "
            "costs to get somewhere by air. Places may be cities ('Birmingham', "
            "'New York'), airport names ('Heathrow', 'Charles de Gaulle') or "
            "IATA codes ('BHX', 'JFK'); a city with several airports searches "
            "all of them. Dates may be written any ordinary way — '2027-03-15', "
            "'15 March', 'next Friday', 'tomorrow'. Pass the user's own words "
            "and let the tool interpret them; if it cannot, it asks."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin": {"type": "STRING", "description": "Flying from: city, airport name, or IATA code."},
                "destination": {"type": "STRING", "description": "Flying to: city, airport name, or IATA code."},
                "depart": {"type": "STRING", "description": "Outbound date, in the user's own words."},
                "return_date": {"type": "STRING", "description": "Return date for a round trip. Omit for one way."},
                "passengers": {"type": "INTEGER", "description": "Number of passengers (1-9). Defaults to 1."},
                "cabin": {"type": "STRING", "description": "economy, premium economy, business or first. Defaults to economy."},
                "currency": {"type": "STRING", "description": "Three-letter currency for the prices. Defaults to GBP."},
            },
            "required": ["origin", "destination", "depart"],
        },
    },
    {
        "name": "open_news",
        "description": "Open one of the cached briefing news stories in the system browser. Match by topic keyword, headline fragment, or 1-based story index.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Topic or headline fragment, e.g. 'neuralink' or part of the story title."},
                "index": {"type": "INTEGER", "description": "1-based story number from the briefing cache."},
            },
        },
    },
    {
        "name": "close_app",
        "description": "Close a running application by process or window name.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {"type": "STRING", "description": "Application or process label, e.g. 'notepad'."},
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "window_control",
        "description": "List, focus, minimise, maximise or close desktop windows.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, focus, minimise, maximise, or close."},
                "title":  {"type": "STRING", "description": "Window title fragment to match."},
            },
        },
    },
    {
        "name": "media_control",
        "description": "Control system media playback and volume.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play_pause, next, previous, stop, volume_up, volume_down, or mute."},
                "steps":  {"type": "INTEGER", "description": "Volume steps for volume actions (1-10)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "sound_sense",
        "description": "ORION's HEARING beyond words: identifies what sounds are present (music, instruments, singing, speech, laughter, applause, alarms, sirens, doorbells, phones, dogs, birds, vehicles, typing, appliances, water... 521 kinds) with a timeline of when each occurred, analyses music (tempo in BPM, musical key), voice pitch, and tones or hums in Hz, and transcribes any speech in the clip. action 'recent' analyses what the microphone ALREADY heard in the last N seconds — use it for 'what was that noise?', 'what song is this?', 'what's making that sound?', 'is that music or talking?'. 'listen' waits N seconds, then analyses that. 'file' analyses an audio file or a video's soundtrack at 'path'. 'watch' keeps listening in the background, locally, and speaks up on its own about a smoke or fire alarm, siren, alarm, breaking glass, a crying baby, screaming, the doorbell or knocking; 'watch_off' stops it and 'watch_status' reports it.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "recent (default), listen, file, watch, watch_off, or watch_status."},
                "seconds": {"type": "NUMBER", "description": "How many seconds to analyse (1-30, default 8)."},
                "path":    {"type": "STRING", "description": "For 'file': an audio file, or a video whose soundtrack to analyse."},
                "transcribe": {"type": "BOOLEAN", "description": "Transcribe speech found in the clip (default true)."},
            },
        },
    },
    {
        "name": "vision_analyse",
        "description": "ORION's eyes. 'read_window' reads the EXACT text of a window — what the user is typing in Notepad or Word, the page open in Edge or Chrome, a chat in Discord or WhatsApp, any app — straight from the application (character-perfect, including text scrolled out of view), falling back to OCR for games and custom-drawn screens; pass window='notepad'/'edge'/'word'/part of a title, or omit for the window in front. Use it for 'what am I typing', 'read this page', 'what does this say', and to read a web page you opened. 'describe' analyses the current screen (exact window text + OCR) and attaches the frame; 'camera' captures a webcam frame and identifies what's in view — the person, the room and background objects (use when asked what he can see, or what's behind/around the user); 'scan' captures the camera and identifies ANYTHING the user shows it — objects, tools, parts, plants, food, documents, a room — on a labelled grid so each finding names its cells ('count' counts named items, 'read_label' reads text through the camera); use for 'scan this', 'what is this', 'count the screws', 'read this label'. 'pcb' (also 'electronics') scans a circuit board or electronics through the camera, shows the captured still with evidence-based component annotations, reads markings and reports image quality and visual limitations. Use for 'scan this PCB', 'inspect this circuit board', 'identify these components' or 'check the soldering'. Local checks work offline; component identification requires a vision model and never substitutes for electrical measurements. 'ocr' extracts on-screen or image text; 'find_errors' sweeps the desktop for error dialogs and crash text; 'analyse_image' inspects an image file. 'video' (also 'watch') WATCHES a video file at 'path': it samples keyframes and describes them, extracts and transcribes the audio track, then reasons over both to answer 'prompt' and recommend a course of action — use for 'watch this video', 'what happens in this clip', 'analyse this recording', 'transcribe this video'. 'frames' sets how many keyframes to sample (default 8). 'watch_screen' watches whatever is PLAYING ON SCREEN (a YouTube video, a course, a stream, a demo) for 'seconds' (default 30): it samples the moments that change, reads them, takes the words from what the microphone hears, and answers 'prompt' — use for 'watch this with me', 'what happens in this video' when no file is involved; 'watch_window' does the same for only the window in front.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "read_window, describe, camera, scan (capture the camera and identify ANYTHING in view — objects, parts, plants, documents — mapped onto a labelled grid so every finding names its cells; opens the Vision Lab with the result), count (count items the prompt names), read_label (read text through the camera), pcb/electronics (circuit boards), ocr, find_errors, analyse_image, video (a file), watch_screen or watch_window (a video playing on screen)."},
                "seconds": {"type": "NUMBER", "description": "For watch_screen/watch_window: how long to watch (3-180, default 30)."},
                "mode": {"type": "STRING", "description": "For scan: anything, text, count or electronics."},
                "grid": {"type": "STRING", "description": "For scan: grid size such as '8x6' (default), '12x9', '16x12', or 'none'."},
                "window": {"type": "STRING", "description": "For read_window: which window (app name such as notepad, word, edge, chrome, discord, or part of its title). Omit for the window in front."},
                "path":   {"type": "STRING", "description": "Image path for ocr/analyse_image (omit for the live screen)."},
                "prompt": {"type": "STRING", "description": "Optional focus question."},
                "camera_index": {"type": "INTEGER", "description": "Webcam device index for camera/pcb/electronics (PCB scans use the selected camera when omitted)."},
                "frames": {"type": "INTEGER", "description": "For the 'video' action: how many keyframes to sample (default 8)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "outlook_mail",
        "description": "Outlook email control. read_inbox/priority list messages; draft composes (saved, never sent); send_draft transmits ONLY with confirm=true after the user explicitly approves; pending_drafts lists what awaits approval.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING", "description": "read_inbox, priority, read_email, draft, send_draft, discard_draft, or pending_drafts."},
                "to":        {"type": "STRING", "description": "Recipient address(es) for draft."},
                "subject":   {"type": "STRING", "description": "Subject for draft."},
                "body":      {"type": "STRING", "description": "Body text for draft."},
                "cc":        {"type": "STRING", "description": "Optional CC address(es)."},
                "draft_ref": {"type": "STRING", "description": "Draft reference (e.g. 'draft-1') for send/discard."},
                "confirm":   {"type": "BOOLEAN", "description": "Must be true to actually send — only after explicit user approval."},
                "entry_id":  {"type": "STRING", "description": "Message entry_id from read_inbox for read_email."},
                "limit":     {"type": "INTEGER", "description": "Message count for reads."},
                "unread_only": {"type": "BOOLEAN", "description": "Restrict read_inbox to unread mail."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "notion_workspace",
        "description": "Notion productivity control: list_tasks, create_task, complete_task, upcoming_events (calendar), create_event (scheduling), or projects (project tracking).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list_tasks, create_task, complete_task, upcoming_events, create_event, or projects."},
                "title":  {"type": "STRING", "description": "Task or event title."},
                "due":    {"type": "STRING", "description": "ISO due date for create_task, e.g. 2026-07-03."},
                "notes":  {"type": "STRING", "description": "Optional body notes for create_task."},
                "query":  {"type": "STRING", "description": "Title fragment for complete_task."},
                "start":  {"type": "STRING", "description": "ISO start datetime for create_event."},
                "end":    {"type": "STRING", "description": "Optional ISO end datetime for create_event."},
                "days":   {"type": "INTEGER", "description": "Look-ahead window for upcoming_events."},
                "limit":  {"type": "INTEGER", "description": "Maximum rows returned."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "agent_dispatch",
        "description": "Consult a specialist agent: marketing (digital marketing/growth), coding (software engineering/debugging), research (research, analysis, comparison & evaluation), neuroscience, aiml (AI/ML engineering), security (defensive cyber security), or auto to route by content.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "agent":   {"type": "STRING", "description": "marketing, coding, research, neuroscience, aiml, security, or auto."},
                "request": {"type": "STRING", "description": "The question or task for the specialist."},
                "context": {"type": "STRING", "description": "Optional extra context."},
            },
            "required": ["request"],
        },
    },
    {
        "name": "reason",
        "description": "Think hard about a difficult question before answering. Several specialists work on it independently with read-only lookups (memory, past conversations, the knowledge graph, files, the evidence store, the live web), a red team attacks their drafts, and the surviving answer is checked against recorded evidence. Use for decisions with consequences, comparisons and trade-offs, diagnosis from evidence, strategy and design questions, anything the user asks you to think carefully about or double-check, and anything where being confidently wrong would cost them. Do not use for simple factual answers, chit-chat or actions — this is slow and deliberate on purpose. 'action=last' reports how the previous deliberation was reached.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "question": {"type": "STRING", "description": "The question to think through, in full."},
                "tier":     {"type": "STRING", "description": "Effort ceiling: reflex, standard, deliberate, exhaustive. Omit to let ORION judge from the question."},
                "context":  {"type": "STRING", "description": "Optional extra context: constraints, budget, what has already been tried."},
                "agent":    {"type": "STRING", "description": "Optional single specialist to use instead of a panel."},
                "action":   {"type": "STRING", "description": "'last' to report how the previous deliberation was reached."},
            },
        },
    },
    {
        "name": "perception",
        "description": "Local computer vision through the camera — free, on this machine. 'start' watches continuously (motion, arrivals, departures, object permanence), reaching for the cloud only when something changes that the local object detector could not name; 'stop' ends it; 'scene', 'events', 'status' report. 'analyse' measures the view as numbers: light, sharpness, edges and where they are, named colours and their shares, motion direction and speed (optical flow); 'grid' also returns the view as a 16×12 grid of brightness digits. 'detect' names objects in view with a local YOLO model (80 everyday kinds: person, cup, cell phone, laptop, bottle, book, dog …); 'install_detector' downloads that model once (34 MB) — only when the user asks. 'pose' reads bodies with MediaPipe (person, hand_raised, left_hand_up, right_hand_up, both_hands_up, arms_out). Automations: 'add_rule' makes the camera trigger something — when (object, object_gone, count, pose, colour, motion, scene_change, dark, covered) + target + optional zone, then announce / workflow (runs one of the user's workflows) / log; 'rules' lists them; 'remove_rule', 'enable_rule', 'disable_rule' take the rule id or name; 'autostart' arms the rules at every start-up. 'overlay' shows the live camera as normal, edges, motion, colours, numbers or detect (boxes and skeletons). Use for 'keep an eye on things', 'what objects can you see', 'when I raise my hand pause the music', 'tell me if my phone leaves the desk', 'show me the edges'. For one described look by the cloud model, use vision_analyse.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "start, stop, status, scene, events, analyse, grid, detect, pose, install_detector, rules, add_rule, remove_rule, enable_rule, disable_rule, autostart, or overlay."},
                "limit":      {"type": "INTEGER", "description": "How many recent events to list for 'events'."},
                "when":       {"type": "STRING", "description": "add_rule: object, object_gone, count, pose, colour, motion, scene_change, dark or covered."},
                "target":     {"type": "STRING", "description": "add_rule: the object (e.g. 'cup', 'person', 'phone'), gesture (e.g. 'hand_raised') or colour (e.g. 'red')."},
                "zone":       {"type": "STRING", "description": "add_rule: where in the view — any, left, centre, right, top, bottom, top-left, top-right, bottom-left, bottom-right."},
                "count":      {"type": "INTEGER", "description": "add_rule with when=count: how many must be in view."},
                "hold":       {"type": "NUMBER", "description": "add_rule: seconds the condition must hold before it fires (object_gone defaults to 10)."},
                "cooldown":   {"type": "NUMBER", "description": "add_rule: minimum seconds between firings (default 60)."},
                "then":       {"type": "STRING", "description": "add_rule: announce, workflow or log."},
                "workflow":   {"type": "STRING", "description": "add_rule with then=workflow: the workflow's name."},
                "message":    {"type": "STRING", "description": "add_rule with then=announce: what to say (optional)."},
                "name":       {"type": "STRING", "description": "add_rule: a short name for the rule; remove/enable/disable: the rule's name."},
                "rule":       {"type": "STRING", "description": "remove_rule/enable_rule/disable_rule: the rule id (e.g. r2) or name."},
                "confidence": {"type": "NUMBER", "description": "add_rule for objects: minimum detector confidence 0.2-0.95 (default 0.45)."},
                "enabled":    {"type": "BOOLEAN", "description": "autostart: true to watch at every start-up, false to stop."},
                "mode":       {"type": "STRING", "description": "overlay: normal, edges, motion, colours, numbers or detect."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "strategy",
        "description": "Search a decision space exhaustively instead of proposing the first few options that come to mind. ORION maps the decision into independent dimensions with concrete options, enumerates and scores thousands of combinations locally, discards anything breaking a hard constraint, keeps only the candidates nothing else beats outright (the Pareto frontier), ranks those by pairwise tournament, and recommends from the finalists. Use when a decision has several genuinely independent axes and competing objectives — pricing and channel and timing, stack and hosting and budget, product mix. Prefer 'reason' when the question is a single judgement call rather than a combination to choose. 'action=last' reports the previous search.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "objective":      {"type": "STRING", "description": "The decision to search, stated as a choice with what matters, e.g. 'how should I launch the course: channel, price, timing and scope, optimising reach against effort'."},
                "context":        {"type": "STRING", "description": "Optional constraints and situation: budget, deadline, skills, what has already been tried."},
                "top":            {"type": "INTEGER", "description": "How many finalists to shortlist (default 5, max 12)."},
                "max_candidates": {"type": "INTEGER", "description": "Ceiling on candidates scored (default 20000)."},
                "judge":          {"type": "STRING", "description": "'false' to return the ranked shortlist without the recommendation pass."},
                "action":         {"type": "STRING", "description": "'last' to report the previous search."},
            },
        },
    },
    {
        "name": "morning_briefing",
        "description": "Deliver the daily intelligence briefing on demand, at ANY time of day: AI news, Neuralink, economy, stock market, crypto, calendar, tasks and priority email. Despite this tool's name, name the briefing after the CURRENT part of the day (morning / afternoon / evening briefing) — never call it a morning briefing in the afternoon or evening.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "period": {"type": "STRING", "description": "'morning' ONLY before noon, for an overnight catch-up (wider news window); omit at any other time for the briefing for the current part of the day."},
            },
        },
    },
    {
        "name": "find_files",
        "description": "LOCATE or open a local file or folder anywhere on this PC by name, type, size and date. This returns paths or opens the default app; it does not read file contents. For 'fetch/read what is inside this local file', use process_file with its path. For a web URL, use fetch_url. Filters combine: e.g. type=pdf modified='last week'; size='over 1GB' sort=largest; query='invoice' created='March 2026'.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Name (or part of it; words all must appear; * and ? wildcards work). May be empty when filtering only by type/size/date."},
                "action": {"type": "STRING", "description": "Optional: 'open' (open the best match — use for 'bring up/show/open the X'), 'reveal' (select it in File Explorer), 'usage' (largest folders inside `folder`)."},
                "open": {"type": "BOOLEAN", "description": "Open the best match with its default application."},
                "path": {"type": "STRING", "description": "An exact path to open or reveal (with action)."},
                "type": {"type": "STRING", "description": "folder, file, document, pdf, word, spreadsheet, presentation, image, photo, video, audio, archive, code, executable, email, font or 3d."},
                "extension": {"type": "STRING", "description": "Explicit extension(s), e.g. 'docx' or 'mp4,mov'."},
                "size": {"type": "STRING", "description": "Size filter in words: 'over 100MB', 'under 5 MB', 'between 1GB and 4GB'."},
                "modified": {"type": "STRING", "description": "When changed: today, yesterday, this week, last month, last 3 days, 'in 2025', 'March 2026', 'since 1 June', 'before 2026-01-01'."},
                "created": {"type": "STRING", "description": "When made, same phrasing as modified."},
                "folder": {"type": "STRING", "description": "Limit to this folder (default: the whole PC)."},
                "sort": {"type": "STRING", "description": "newest, oldest, largest, smallest or name."},
                "contents": {"type": "BOOLEAN", "description": "Search the words INSIDE files instead of names."},
                "limit": {"type": "INTEGER", "description": "How many results (default 40)."},
            },
        },
    },
    {
        "name": "dev_workbench",
        "description": "Software engineering workbench: analyse a repository, read code with line numbers, run allow-listed development commands (python, pytest, node, npm, cargo, go, dotnet, git), or scaffold a new Python project inside the workspace.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "analyse_repo, read_file, run_command, or create_python_project."},
                "path":       {"type": "STRING", "description": "Repository, file, or project path."},
                "command":    {"type": "STRING", "description": "Development command for run_command."},
                "start_line": {"type": "INTEGER", "description": "First line for read_file."},
                "line_count": {"type": "INTEGER", "description": "Line count for read_file (max 400)."},
                "name":       {"type": "STRING", "description": "Project name for create_python_project."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "docker",
        "description": "Docker container/image control over the real docker CLI: list containers, list images, start/stop/restart/remove a container, tail its logs, or check whether Docker is available.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING", "description": "list, images, start, stop, restart, remove, logs, or status."},
                "container": {"type": "STRING", "description": "Container name or ID (for start/stop/restart/remove/logs)."},
                "tail":      {"type": "INTEGER", "description": "Number of log lines to return for logs (default 100)."},
                "all":       {"type": "BOOLEAN", "description": "For list: include stopped containers too (default true)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "debugger",
        "description": "Real interactive Python debugger (stdlib pdb): start a script under the debugger, step through it with next/step/continue, inspect variables with print, set a breakpoint, or stop the session. One session active at a time.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "start, command, next, step, continue, print, break, stop, or status."},
                "path":       {"type": "STRING", "description": "Script path, for start."},
                "command":    {"type": "STRING", "description": "Raw pdb command, for action=command."},
                "expression": {"type": "STRING", "description": "Expression to print, for action=print."},
                "location":   {"type": "STRING", "description": "file:line breakpoint location, for action=break."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "recall_conversation",
        "description": "Search past conversation history (episodic memory) for what was previously discussed and when.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "What to look for in past conversations."},
                "limit": {"type": "INTEGER", "description": "Maximum entries."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_plan",
        "description": "Run a multi-step plan of tool calls sequentially, verify each step, and return a consolidated report. Use for autonomous multi-part tasks.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "objective": {"type": "STRING", "description": "One-line goal of the plan."},
                "steps": {
                    "type": "ARRAY",
                    "description": "Ordered steps (max 8).",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "tool":      {"type": "STRING", "description": "Tool name to call."},
                            "args_json": {"type": "STRING", "description": "JSON object of arguments for the tool."},
                        },
                        "required": ["tool"],
                    },
                },
            },
            "required": ["steps"],
        },
    },
    {
        "name": "browser_control",
        "description": "Open a secure URL or search in the system browser.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "go_to, open, new_tab, or search."},
                "url":    {"type": "STRING", "description": "HTTP or HTTPS URL."},
                "query":  {"type": "STRING", "description": "Search query."},
            },
        },
    },
    {
        "name": "undo",
        "description": "Reverse ORION's own last reversible action — a file he wrote, created or deleted. Use this whenever the user says 'undo', 'take that back', 'revert that' or 'put it back', in any language. Action 'last' reverses the most recent one; 'list' shows what is reversible without changing anything. ORION only tracks what HE changed, never the user's own edits.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "last (reverse the most recent action) or list (show what can be undone)."},
            },
        },
    },
    {
        "name": "file_controller",
        "description": "Access ANY file or folder on the user's computer by absolute path (~ and environment variables are expanded): list directories and read_text any file freely — you have full read access. Writes (write_text/append_text) and delete are permitted anywhere EXCEPT ORION's own program files, and should be done on the user's explicit instruction.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, mkdir, write_text, append_text, read_text, delete."},
                "path":   {"type": "STRING", "description": "Any absolute path on the PC (e.g. C:/Users/you/Documents/x.txt), or workspace-relative."},
                "text":   {"type": "STRING", "description": "Text for write operations."},
            },
        },
    },
    {
        "name": "process_file",
        "description": "FETCH and inspect the CONTENTS of a local file on the user's computer by absolute path — text, JSON, CSV, TSV, binary, PDF, or image. Use this when asked to fetch, read, analyse or summarise a local file. If only its name is known, call find_files first for the path. For an http(s) URL use fetch_url instead.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "path":   {"type": "STRING", "description": "Path to the file to scan."},
                "prompt": {"type": "STRING", "description": "Optional focus question for the review."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "save_memory",
        "description": "Commit a durable user fact into the local SQLite FTS5 intelligence matrix.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING", "description": "Category such as identity, preferences, projects. 'identity' and 'personal' are ONLY for facts about the user themself; a fact about anyone else (friend, family, client, contact) goes under 'relationship', with that person's name in the key and the value (e.g. 'Sam Lee, a friend of the user, lives in Leeds')."},
                "key":      {"type": "STRING", "description": "Snake case memory key."},
                "value":    {"type": "STRING", "description": "Fact value."},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "query_intelligence",
        "description": "Look up something ORION has stored but is not carrying in this prompt. The system prompt lists what it had no room for under [ALSO REMEMBERED] — if the user asks about anything named there, or anything that sounds like a person, project, preference or past decision, search for it here BEFORE saying you do not know. Matches by MEANING as well as words, so ask it the way the user asked. Local, instant, no network. An empty query returns the most recent entries. For 'what have you learned about me' set learned=true: the facts ORION kept from conversation on his own, newest first.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "What to look for (may be empty for the most recent entries)."},
                "limit": {"type": "INTEGER", "description": "Maximum records."},
                "category": {"type": "STRING", "description": "Only this category, e.g. 'personal' or 'knowledge'."},
                "learned": {"type": "BOOLEAN", "description": "Only the facts ORION learned by himself from conversation."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "capture_screen",
        "description": "Capture the primary monitor as volatile in-memory JPEG bytes (prefer vision_analyse for analysis).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "quality":  {"type": "INTEGER", "description": "JPEG quality 45-95."},
                "max_side": {"type": "INTEGER", "description": "Maximum image side, default 1024."},
            },
        },
    },
    {
        "name": "clipboard_operate",
        "description": "Read or copy text through the native Qt clipboard.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "read or copy."},
                "text":   {"type": "STRING", "description": "Text to copy."},
            },
        },
    },
    {
        "name": "process_governor",
        "description": "Monitor and manage host processes. 'top' lists the biggest CPU or RAM power users (plus GPU utilisation) — use this when the user asks what's using too much power; 'list' shows all; 'terminate' stops a PID or named process; 'restart' stops then relaunches it. Terminate/restart are destructive and require confirm=true (the user's permission).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "top, list, terminate, or restart."},
                "metric":  {"type": "STRING", "description": "For 'top': cpu (default) or ram."},
                "pid":     {"type": "INTEGER", "description": "Process ID to terminate/restart."},
                "name":    {"type": "STRING", "description": "Process/app name to terminate/restart."},
                "confirm": {"type": "BOOLEAN", "description": "Must be true to actually terminate/restart — only after the user explicitly approves."},
            },
        },
    },
    {
        "name": "system_notify",
        "description": "Project a high-priority alert banner into the HUD.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "message":  {"type": "STRING", "description": "Alert text."},
                "priority": {"type": "INTEGER", "description": "Priority from 0 to 5."},
            },
            "required": ["message"],
        },
    },
    {
        "name": "shutdown_orion",
        "description": "Shut YOURSELF (O.R.I.O.N., the assistant) down completely — disconnect the live session and close every ORION window. This is the DEFAULT meaning of a shutdown command addressed to you: 'shut down', 'shut yourself down', 'power down', 'turn yourself off', 'go offline', 'close/quit ORION', 'that's all for tonight'. Use this whenever the user is talking to YOU and does NOT explicitly name the physical computer/PC/laptop/machine. NEVER use the peripherals tool's 'shutdown' for these — that powers off the whole computer and is only for when the user explicitly says to shut down the COMPUTER/PC itself.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "restart_orion",
        "description": "Restart O.R.I.O.N.: clean shutdown, then the process respawns itself automatically. Use when the user asks for a restart/reboot of ORION, or to load an applied self_repair fix. Requires the user's explicit request or approval.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    # ── Mark IX autonomous OS tools ───────────────────────────────────────────
    {
        "name": "desktop_control",
        "description": "Autonomously control the machine: move/click the cursor, type, send hotkeys, and manage windows. To WRITE text into Notepad, an editor or a form, use action 'edit_text' with the target window 'title' — it focuses the window and TYPES the text visibly at a natural pace so the user can watch it arrive (read back and verified; long text over ~1500 characters, or fast=true, uses the fast native/clipboard path). 'type_text' types into whatever has focus, also visibly human-paced by default (fast=true for machine speed; wpm, jitter and typos are adjustable). Prefer click_text over raw click. Clicks are visually verified before continuing. The pointer glides visibly and a caption shows each step, so give every call a short 'why'.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "click_text, click, double_click, right_click, move_cursor, drag, scroll, type_text, edit_text, hotkey, open_app, close_app, focus_window, resize_window, move_window, minimise_window, maximise_window, or list_windows."},
                "text":   {"type": "STRING", "description": "Visible label to click (click_text), or the text to write (edit_text/type_text)."},
                "why":    {"type": "STRING", "description": "A few words saying what this step is for, shown to the user in a caption beside the pointer so they can follow your working (e.g. 'opening Settings to find Bluetooth'). Always include it."},
                "human":  {"type": "BOOLEAN", "description": "For type_text: type one key at a time with human timing (the default already)."},
                "fast":   {"type": "BOOLEAN", "description": "For type_text/edit_text: skip the visible typing and write at machine speed."},
                "wpm":    {"type": "NUMBER", "description": "For type_text: typing speed in words per minute (default: ~140, faster for long text so it takes about 20 s)."},
                "typos":  {"type": "NUMBER", "description": "For human type_text: mistype probability 0..1 (mistakes are auto-corrected). Default 0."},
                "jitter": {"type": "NUMBER", "description": "For human type_text: per-key timing variation fraction (default 0.35)."},
                "replace": {"type": "BOOLEAN", "description": "For edit_text: replace all existing content (select-all first) instead of appending."},
                "x":      {"type": "INTEGER", "description": "X coordinate (virtual-desktop, or monitor-local with 'monitor')."},
                "y":      {"type": "INTEGER", "description": "Y coordinate."},
                "x1": {"type": "INTEGER"}, "y1": {"type": "INTEGER"},
                "x2": {"type": "INTEGER"}, "y2": {"type": "INTEGER"},
                "monitor": {"type": "INTEGER", "description": "Monitor index for monitor-local coordinates."},
                "amount": {"type": "INTEGER", "description": "Scroll amount (+up/-down)."},
                "text_value": {"type": "STRING"},
                "keys":   {"type": "STRING", "description": "Hotkey combo e.g. 'ctrl+c'."},
                "title":  {"type": "STRING", "description": "Window title fragment for window actions."},
                "app_name": {"type": "STRING", "description": "Application to open/close."},
                "width": {"type": "INTEGER"}, "height": {"type": "INTEGER"},
                "verify": {"type": "BOOLEAN", "description": "Visually confirm the action (default true for clicks)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "vision_verify",
        "description": "Detect on-screen UI structure: 'elements' lists interactive controls (buttons, menus, fields) in the foreground window via the accessibility tree; 'dialogs' finds open dialogs and pop-ups.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "elements, dialogs, describe, ocr, or find_errors."},
                "kinds":  {"type": "STRING", "description": "For elements: all, button, menu, input, link, or dialog."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "web_control",
        "description": "Operate a browser like a human: navigate, tabs, accept/reject cookies, close pop-ups, fill forms, read page contents, download, and complete file dialogs. Uses the accessibility tree + vision to find controls reliably. The 'url' also accepts LOCAL FILE PATHS (e.g. C:\\Users\\me\\Documents, ~/Downloads, or file:///...) so you can browse folders and open local files in the browser, not just websites.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "open, navigate, new_tab, close_tab, switch_tab, accept_cookies, reject_cookies, close_popup, fill_form, read_page, download, or file_dialog."},
                "url":    {"type": "STRING", "description": "URL, or a local file path / folder (C:\\..., ~/..., file:///...), for open/navigate/new_tab."},
                "index":  {"type": "INTEGER", "description": "Tab number for switch_tab."},
                "fields": {"type": "OBJECT", "description": "For fill_form: an object of field-label → value."},
                "submit": {"type": "BOOLEAN", "description": "Submit the form after filling."},
                "path":   {"type": "STRING", "description": "File path for file_dialog (upload/save)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "telephony",
        "description": (
            "Ring a real phone or send a real SMS through Twilio \u2014 use this when "
            "the user asks ORION to CALL them or someone else, to phone them with a "
            "reminder, or to text somebody. Calls connect immediately and cost "
            "money, so 'confirm' must be true: say who you are about to ring and "
            "what you will say, get agreement, then call again with confirm=true. "
            "ORION can only dial numbers already listed in "
            "config/telephony_contacts.json \u2014 a number heard in conversation or "
            "read from a web page cannot be dialled, by design. "
            "'status' reports whether the telephony server is connected; "
            "'contacts' lists who may be rung. Not to be confused with "
            "phone_action, which hands a pre-filled dialler to the user's own "
            "paired handset for them to tap."),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "call (default), text, contacts, or status."},
                "to":      {"type": "STRING", "description": "Contact name from telephony_contacts.json, e.g. 'me'."},
                "message": {"type": "STRING", "description": "What ORION should say or send. Keep it short: it is spoken aloud."},
                "reason":  {"type": "STRING", "description": "Why the call is being placed, for the log."},
                "confirm": {"type": "BOOLEAN", "description": "Required. Real money, real phone, immediate."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "mcp",
        "description": "Model Context Protocol gateway. Discover and use tools from connected MCP servers — e.g. Gmail, Google Calendar, filesystem and web fetch. For a web URL use fetch_url directly; for a local file use process_file. For other MCP tools call with action='list' first, then action='call' with exact server and tool names. 'servers' shows status; 'enable', 'disable', 'reconnect' and 'add' manage servers live. Some tools — telephony calls and SMS — spend real money or act outside the PC; those are refused unless confirm=true, so say what it will do and what it will cost before passing it.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING", "description": "list, servers, call, enable, disable, reconnect, or add."},
                "server":    {"type": "STRING", "description": "Server name (e.g. gmail, google_calendar)."},
                "tool":      {"type": "STRING", "description": "Tool name for 'call' (from the server's tool list)."},
                "arguments": {"type": "OBJECT", "description": "Arguments object for the tool being called (or the command's argument list for 'add')."},
                "command":   {"type": "STRING", "description": "Executable to launch the server, for 'add' (e.g. npx)."},
                "env":       {"type": "OBJECT", "description": "Environment variables for the server, for 'add'."},
                "description": {"type": "STRING", "description": "What the server provides, for 'add'."},
                "confirm":   {"type": "BOOLEAN", "description": "Required for tools that spend real money or act in the world (telephony calls and SMS). Tell the user what it will do and what it will cost, and only pass true once they have agreed."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "interface_control",
        "description": "Navigate and operate ORION's OWN interface programmatically (not the OS): change your on-screen FORM between your face and your orb, open a page of the command deck, shrink to the compact orb or toggle fullscreen, refresh the geographical node, or fly the on-screen globe. Use this when the user asks you to show, open, switch to, or change something in your own GUI — including 'turn into your orb', 'show your face', 'go compact'.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "face_form (switch between your human FACE and your ORB form — target 'face' or 'orb'; use whenever the user asks you to change form, e.g. because the face unsettles someone), view (core view), page (deck page), camera (open the Camera Lab and START THE LIVE FEED — use this whenever the user asks to see, show, bring up or open the camera; it is NOT the same as vision_analyse action=camera, which only grabs one frame and describes it), research (open the live research console), dashboard, toolkit, command_centre, diagnostics, globe, refresh_environment, overlay (shrink to / restore from the small floating compact orb in the corner of the desktop), fullscreen, pause, or stop."},
                "target": {"type": "STRING", "description": "For 'face_form': face or orb. For 'view': face, log, memory, or telemetry. For 'page': a deck page name — brain, workbench, mission, research, development, widgets, toolkit, marketing, library, ops, cognition, automation, command centre, plugins, diagnostics, globe, log, memory, telemetry, chess or security. For 'globe': a place to fly to."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "phone_action",
        "description": "Hand a native action to the user's paired PHONE (the ORION Android app): place a call, start a text or e-mail, open turn-by-turn navigation, show a place on a map, open a link, or share text. The phone opens the relevant app PRE-FILLED and the user taps to confirm — nothing is dialled or sent automatically. Use this when the user (on their phone) asks you to call/text/email someone, navigate somewhere, or open something on the phone. If the user is only at the desktop it simply queues the intent for the app.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "kind":    {"type": "STRING", "description": "call, sms, email, navigate, map, openUrl, or share."},
                "number":  {"type": "STRING", "description": "Phone number for call/sms (E.164 e.g. +447700900123 where possible)."},
                "body":    {"type": "STRING", "description": "Message body for sms/email."},
                "to":      {"type": "STRING", "description": "Recipient e-mail address for email."},
                "subject": {"type": "STRING", "description": "Subject line for email."},
                "query":   {"type": "STRING", "description": "Address or place for navigate/map."},
                "url":     {"type": "STRING", "description": "URL for openUrl."},
                "text":    {"type": "STRING", "description": "Text to share for 'share'."},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "workspace_control",
        "description": "Persistent workspace awareness: snapshot the desktop, save/restore a named workspace to resume work, track what changed, recall resume context, or set the active project.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "snapshot, save, restore, track_changes, resume_context, or set_project."},
                "name":    {"type": "STRING", "description": "Workspace name for save/restore."},
                "project": {"type": "STRING", "description": "Project name for resume_context/set_project."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "codebase_copilot",
        "description": "Whole-repository engineering: analyse (index + dependency graph + hotspots + cycles), find_symbol, dependencies (impact analysis), or task (refactor/review/bughunt/tests/docs grounded in the live index).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "analyse, find_symbol, dependencies, or task."},
                "path":   {"type": "STRING", "description": "Repository path (defaults to the workspace)."},
                "name":   {"type": "STRING", "description": "Symbol name for find_symbol."},
                "module": {"type": "STRING", "description": "Module for dependency/impact analysis."},
                "task":   {"type": "STRING", "description": "The engineering task for 'task'."},
                "focus":  {"type": "STRING", "description": "Optional focus area or file."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "self_repair",
        "description": "Self-healing runtime. 'status' lists captured incidents; 'report' is how the USER reports behaviour that is wrong but never raises an exception (e.g. 'you keep mishearing shut down as a power command') — pass 'description' and ORION locates the responsible module and drafts a fix; 'propose' drafts a fix diff for review; 'run_tests' runs the verify suite; 'diagnose' runs a full self-diagnostic; 'repair' rewrites the failing source file (confirm=false drafts + validates it compiles; confirm=true backs up the original and applies it — a restart then loads it); 'revert' undoes the last applied repair. Code is only ever changed with explicit confirmation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "status, report, propose, run_tests, diagnose, repair, or revert."},
                "incident_id": {"type": "STRING", "description": "Incident id (defaults to the latest)."},
                "description": {"type": "STRING", "description": "For 'report': what ORION is doing wrong, in the user's own words. Be specific — quote the phrase misheard, name the tool, or give the log line."},
                "path":        {"type": "STRING", "description": "For 'report': optional orion_core file path, when the user already knows where the defect lives."},
                "confirm":     {"type": "BOOLEAN", "description": "For 'repair': true actually writes the fix to the source (after backup)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "diagnostics",
        "description": "Run a self-diagnostic across ORION. Default (action='full'): compile every module, import checks, dependency audit, database integrity, config validity, tool registry, component health, resource headroom and permissions → PASS/WARN/FAIL. action='capabilities' (also 'health'/'why') returns the CAPABILITY HEALTH MATRIX — for every major capability (OCR, vision, voice, desktop/browser control, study/focus/finance/wellbeing, forge, chess, security…) whether it is present, registered, actually available (with a real functional OCR read when deep), and offline-capable. Use 'capabilities' to answer 'why can't ORION do X' or 'what can you actually do right now'. action='latency' (also 'lag'/'performance') reports MEASURED responsiveness: every event-loop stall the watchdog caught and the exact function that blocked it, plus p50/p95 timings per operation. Use it to answer 'why were you slow or laggy just then'. action='reflexes' lists the shortcuts ORION has LEARNED for himself — phrases this user repeats that always reach the same read-only tool, now answered instantly with no model turn — plus candidates not yet promoted. Pass promote='<phrase>' when the user agrees to a suggested shortcut, or forget='<phrase>' to withdraw one. action='routing' reports the tool-resolver shadow evaluation: how often the pre-filter would have kept the tool actually needed, and whether it is safe to enable.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "full (default), capabilities/health, latency/lag, reflexes, or routing."},
                    "promote": {
                        "type": "STRING",
                        "description": "With action=reflexes: the phrase to start answering instantly (only an eligible candidate).",
                    },
                    "forget": {
                        "type": "STRING",
                        "description": "With action=reflexes: the learned phrase to withdraw.",
                    },
                "deep":   {"type": "BOOLEAN", "description": "For 'capabilities': run functional probes like a real OCR read (default true)."},
            },
        },
    },
    {
        "name": "cursor_overlay",
        "description": "The visible cursor halo that shows where ORION moves the mouse. 'show'/'hide'/'toggle' control it.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"action": {"type": "STRING", "description": "show, hide, or toggle."}},
        },
    },
    {
        "name": "patch_notes",
        "description": "Recount ORION's own recent system updates, patch-notes style. 'latest' (default) gives the newest release; 'all' the full history; or pass a version.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "latest or all."},
                "count":   {"type": "INTEGER", "description": "How many recent releases for 'latest'."},
                "version": {"type": "STRING", "description": "A specific version, e.g. 9.7."},
            },
        },
    },
    {
        "name": "speaker_id",
        "description": "Report whether the voice ORION last heard sounded male or female — a pitch-based guess at the speaker's gender, and nothing more. Use for 'was that a man or a woman?'. It canNOT say WHICH PERSON spoke: for that use voice_speaker_id, which compares a trained voiceprint against enrolled individuals. 'who' (default) reports the current inference; 'reset' clears it. ORION's own locked voice is never affected by this.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "who (default) or reset."},
            },
        },
    },
    {
        "name": "code_changes",
        "description": (
            "Explain what ORION actually CHANGED inside his own code — which "
            "functions and classes he added, removed, re-signed or rewrote, and "
            "what each one is for, read from his own source. Use this whenever "
            "the user asks 'what have you changed?', 'what's new?', 'what did "
            "you do to yourself?', or wants detail rather than a list of files "
            "(self_changes gives files; this gives meaning). 'explain' with a "
            "name describes any part of ORION in its own words."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING",
                           "description": "latest (default), explain, or snapshot."},
                "name":   {"type": "STRING",
                           "description": "For 'explain': the function or class to describe."},
                "spoken": {"type": "BOOLEAN",
                           "description": "Phrase it for speaking aloud rather than reading."},
            },
        },
    },
    {
        "name": "self_changes",
        "description": "Report exactly which of ORION's own source files (his 'mains folder' modules) changed, were added or removed since his last run — the ground truth computed from the code itself, not curated prose. Use this when the user asks 'what changed?', 'what did you just update?', or 'which of your files changed?'. 'latest' (default) is the change detected at this boot; 'scan' re-checks right now; 'history' recounts the last several update events.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "latest, scan, or history."},
                "count":  {"type": "INTEGER", "description": "How many past update events for 'history'."},
            },
        },
    },
    {
        "name": "learn",
        "description": "Feed ORION information to learn permanently. Actions: 'learn' distils facts from raw text, a local file (incl. PDF/DOCX) or a URL; 'folder' bulk-ingests an entire directory of documents (PDF/DOCX/Markdown/text/HTML/code) into memory — the way to teach ORION gigabytes of a subject (e.g. neuroscience, cybersecurity, programming); 'recall' retrieves learned facts; 'correct' records an authoritative correction; 'forget' removes previously-learned facts on a topic (built-in knowledge is never removed).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "learn, folder, recall, correct, or forget."},
                "source": {"type": "STRING", "description": "Text to learn, a file path, or a URL."},
                "folder": {"type": "STRING", "description": "Directory to bulk-ingest (for action=folder)."},
                "deep":   {"type": "BOOLEAN", "description": "folder only: distil each file with the model (slower, higher quality). Default false = fast extractive."},
                "topic":  {"type": "STRING", "description": "Optional topic label (or the topic to correct/forget)."},
                "correction": {"type": "STRING", "description": "The authoritative correction text (for action=correct)."},
                "query":  {"type": "STRING", "description": "What to recall (for recall)."},
            },
        },
    },
    {
        "name": "cyber_knowledge",
        "description": "Consult ORION's cybersecurity knowledge base (defensive, educational): CIA triad, threat modelling, OWASP risks, injection/XSS/CSRF/SSRF, cryptography, authentication/MFA, network defence, malware, MITRE ATT&CK, detection/SIEM, incident response, secure development, cloud/container and supply-chain security.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Security topic or question."}},
            "required": ["query"],
        },
    },
    {
        "name": "transcript",
        "description": "The verbatim recording of this conversation. 'export' writes a Markdown transcript of the session; 'path' reports where it is being recorded; 'recall' searches past conversation history.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "export, path, or recall."},
                "query":  {"type": "STRING", "description": "Search text for recall."},
            },
        },
    },
    {
        "name": "programming_knowledge",
        "description": "Consult ORION's extensive software-engineering knowledge base (complexity, data structures, algorithms, concurrency, design patterns, databases, security, testing, languages).",
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Programming topic or question."}},
            "required": ["query"],
        },
    },
    {
        "name": "proactive_check",
        "description": "Run the proactive survey now (priority email, task deadlines, today's calendar, repository state) and report items for attention.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "display_info",
        "description": "Report the monitor topology: resolutions, positions, DPI scaling, cursor location, and each monitor's contextual name (a user-given label like 'Coding Monitor', or an objective role like 'Vertical Monitor'/'Primary Monitor' when none is set). Use action=set_name to give a monitor a name.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "refresh": {"type": "BOOLEAN", "description": "Force a topology refresh first."},
                "action":  {"type": "STRING", "description": "set_name to label a monitor; omit to just report."},
                "device":  {"type": "STRING", "description": "The monitor's raw device string (from a prior display_info report), for set_name."},
                "label":   {"type": "STRING", "description": "The human name to give it, e.g. 'Coding Monitor', for set_name."},
            },
        },
    },
    {
        "name": "neuro_knowledge",
        "description": "Retrieve authoritative neuroscience / neural-engineering facts from ORION's resident corpus (neurons, synapses, plasticity, BCIs, EEG/ECoG, Utah array, Neuralink, spike sorting, decoding, neuroprosthetics, DBS). Omit query to list topics.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Topic or question, e.g. 'how does the Utah array work'."},
            },
        },
    },
    {
        "name": "ai_mode",
        "description": "Report ORION's active intelligence mode: cloud-enhanced (MODE A) vs fully offline (MODE B), internet status, and which cloud/local models are available. Use when asked 'are you online', 'what model are you using', or about offline capability. 'recommend' (also 'cheapest', 'which_model') suggests which model to use — a curated shortlist of FREE API models (Groq, Gemini Flash, Cerebras) and the cheapest paid ones (DeepSeek leads on $/million tokens), ranked, with a note on whether it uses your CPU/GPU. Pass 'task' (coding, reasoning, vision, offline, bulk, fastest) to tailor it. 'governor' (also 'runway', 'budget') is the live cost governor — it reads how much of today's token allowance is left and how fast it's burning, says whether to switch to a cheaper or free model now, and (when the model runs locally) whether a hot CPU should hand off to the GPU. Use for 'how much usage do I have left', 'should I switch models', 'is my CPU the bottleneck'.",
        "parameters": {"type": "OBJECT", "properties": {
                "action": {"type": "STRING", "description": "status (default), recommend, or governor."},
                "task": {"type": "STRING", "description": "For recommend: coding, reasoning, vision, offline, bulk, fastest, or cheapest."},
                "provider": {"type": "STRING", "description": "For governor: which provider's runway to assess (defaults to the busiest today)."},
                "local": {"type": "STRING", "description": "For governor: 'true' if the work runs on a local model, so CPU/GPU advice applies."},}},
    },
    {
        "name": "knowledge_pack",
        "description": "Consult installed offline knowledge packs (entrepreneurship, dropshipping, tiktok shop, marketing, copywriting, sales psychology, business, coding, AI, personal development). Actions: consult (default, needs query), list, remove (id), expand (id + entries).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "consult, list, remove, or expand."},
                "query":  {"type": "STRING", "description": "Question/topic for consult."},
                "id":     {"type": "STRING", "description": "Pack id for remove/expand."},
            },
        },
    },
    {
        "name": "conversation_recall",
        "description": "Long-term conversation memory. recall answers time-scoped questions ('what did we discuss three weeks ago about TikTok Shop?'); summarise digests recent turns; compress rolls old turns into durable summaries. Works offline.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "recall, summarise, or compress."},
                "query":    {"type": "STRING", "description": "The time-scoped question for recall."},
                "turns":    {"type": "INTEGER", "description": "Turns to summarise."},
                "days":     {"type": "INTEGER", "description": "Age threshold for compress."},
            },
        },
    },
    {
        "name": "product_research",
        "description": "Dropshipping product research. score computes a 0-100 Product Opportunity Score from a name/description (virality, demand, competition, margin, shipping, returns, seasonality); validate adds a test plan; competition analyses a niche's saturation; log lists prior research. Works offline.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "score, validate, competition, or log."},
                "name":        {"type": "STRING", "description": "Product name."},
                "description": {"type": "STRING", "description": "Product description / notes."},
                "niche":       {"type": "STRING", "description": "Niche for competition analysis."},
            },
        },
    },
    {
        "name": "tiktok_intel",
        "description": "TikTok Shop intelligence: trend reports, product assessments, and a deterministic virality-velocity score from supplied signals (avg_views, creators_posting, buy_intent_comments, new_videos_per_day).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "trend, product, or virality."},
                "niche":   {"type": "STRING", "description": "Niche for a trend report."},
                "product": {"type": "STRING", "description": "Product to assess."},
            },
        },
    },
    {
        "name": "instagram_intel",
        "description": "Instagram commerce intelligence: product discovery for a niche, influencer partnership strategy, or a weekly opportunity report.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "discover, influencer, or weekly."},
                "niche":  {"type": "STRING", "description": "Niche."},
                "brand":  {"type": "STRING", "description": "Brand for influencer strategy."},
            },
        },
    },
    {
        "name": "founder_knowledge",
        "description": "Structured founder/operator profiles (Hormozi, Vaynerchuk, Blakely, Bezos): their strategies, frameworks and lessons, with applied analysis. Actions: list, profile (name), learn (name + question).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "list, profile, or learn."},
                "name":     {"type": "STRING", "description": "Founder name."},
                "question": {"type": "STRING", "description": "What to extract/apply."},
            },
        },
    },
    {
        "name": "business_advisor",
        "description": "Personal business advisor for the user's business Creator Studio (short-form content, UGC and creator management; TikTok, Reels, Shorts). Actions: advise (topic), brand, growth (horizon), store (offer/funnel optimisation).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "advise, brand, growth, or store."},
                "topic":   {"type": "STRING", "description": "The advice topic for advise."},
                "horizon": {"type": "STRING", "description": "Time horizon for growth."},
            },
        },
    },
    {
        "name": "commerce_hub",
        "description": "The E-commerce Intelligence Hub: an aggregated snapshot of scored product opportunities, research log and knowledge packs for the active brand.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "community_share",
        "description": "ORION Network: export/import knowledge packs and product research as privacy-controlled bundles (private items excluded, optional anonymisation). Actions: export_pack (id), import_pack (path), export_research, import_research (path), list.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "export_pack, import_pack, export_research, import_research, or list."},
                "id":      {"type": "STRING", "description": "Pack id to export."},
                "path":    {"type": "STRING", "description": "Bundle path to import."},
                "privacy": {"type": "STRING", "description": "public, shared, or private."},
            },
        },
    },
    {
        "name": "research",
        "description": "Autonomous research. USE 'paper' (also 'deep', 'study', 'thesis', 'report') for any request to research, study, write up or produce a report on something: it plans a topic-specific outline, writes every section at postgraduate depth IN PARALLEL, saves a Word (.docx) document, files the findings into memory, and announces the result out loud. Pass 'depth' as a page count ('50 pages') or a tier (brief ~5pp, standard ~15pp, thesis ~35pp, exhaustive ~50pp), and 'style' as literature_review, technical, comparative, critical, systematic or primer. Output is Word only - the user reads .docx. 'start' researches a topic immediately for a set number of minutes and writes organised notes into a dated folder; 'queue' adds a topic to the persistent research agenda (runs continue across sessions, one after another); 'agenda' shows the programme; 'findings' reports harvested claims with confidence and contradictions for a topic; 'reviewed' marks a topic reviewed; 'validate' scores the credibility of every source URL in the supplied text (domain tiers); 'opportunities' scans for strategic research opportunities (contradictions to resolve, unreviewed findings, missions with no evidence base); 'paper' writes a structured research paper; 'status'/'stop' manage the current run. Use action='dossier' for an unattended investigation that is filed to research/dossiers/ with deduplicated citations, an executive summary and key takeaways, and announces itself when finished - this is the one a scheduled plugin runs overnight. ACTION 'browse' is the one that actually reads the web: it searches, opens each result, takes a note from each page with its URL, and lets what it finds decide what to look up next, narrating every step. Slower than the other modes and the only one whose citations point at pages that were genuinely opened - use it when the user wants real research rather than a written-up answer.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "start, queue, agenda, findings, reviewed, validate, opportunities, paper, status, or stop."},
                "depth":   {"type": "STRING", "description": "How long: a page count like '50 pages', or brief/standard/thesis/exhaustive. Default standard (~15 pages)."},
                "style":   {"type": "STRING", "description": "literature_review, technical, comparative, critical, systematic or primer. Default systematic."},
                "topic":   {"type": "STRING", "description": "The research topic."},
                "minutes": {"type": "NUMBER", "description": "How long to research (for start/queue)."},
                "text":    {"type": "STRING", "description": "Text containing source URLs (for validate)."},
            },
        },
    },
    {
        "name": "globe",
        "description": "Control the on-screen 3-D globe. Fly it to a place (city, region, country, or a postcode / ZIP code) and show that region's news and footage, OR zoom the camera in/out or reset it to orbit without changing location. Use when the user asks to see/travel to a location, go to a postcode, wants regional news, or says 'zoom in/out' on the globe.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "place": {"type": "STRING", "description": "City, country, region, or postcode/ZIP to fly to (e.g. 'Tokyo', 'SW1A 1AA', '10001')."},
                "action": {"type": "STRING", "description": "Camera control instead of flying: 'zoom_in', 'zoom_out', or 'reset'. Omit to fly to 'place'."},
            },
        },
    },
    {
        "name": "max_zoom_in_globe",
        "description": "Drive the on-screen 3-D globe to its MAXIMUM zoom-in (closest to the surface the active view allows). Use when the user says 'max zoom', 'zoom all the way in', 'get as close as possible', or 'maximum zoom on the globe'. This runs a bounded, cancellable loop that repeatedly zooms in until the camera reaches the closest level and stops automatically; it needs no location and never over-zooms. Optional bounds may be supplied to tune the safeguards.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "max_attempts":        {"type": "NUMBER", "description": "Optional. Hard cap on zoom-in steps (default 40)."},
                "max_seconds":         {"type": "NUMBER", "description": "Optional. Wall-clock time budget in seconds (default 20)."},
                "delay_between_s":     {"type": "NUMBER", "description": "Optional. Async delay between steps in seconds (default 0.05)."},
                "unchanged_threshold": {"type": "NUMBER", "description": "Optional. Consecutive no-change steps that count as 'maxed' (default 3)."},
                "tolerance":           {"type": "NUMBER", "description": "Optional. Relative state change treated as no change (default 0.01)."},
            },
        },
    },
    {
        "name": "resource_status",
        "description": "Report current system resource pressure (CPU, physical memory/RAM, swap, and ORION's own working set) and a non-destructive recommendation. Read-only — it never terminates, suspends or throttles any process. Use when the user asks how the system is doing, why things feel slow, or whether resources are under strain. Pass action='trends' for the rolling history: metric trends over days/weeks and the tool-usage audit.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "Omit for a live snapshot, or 'trends' for the rolling history and tool-usage audit."},
            },
        },
    },
    {
        "name": "cyber_curriculum",
        "description": "Structured cybersecurity training curriculum (20 modules from CS foundations to red/blue team, defensive-first). 'outline' lists modules; 'search' finds topics; 'module' shows detail; 'progress'/'complete' track learning; 'certifications' shows the timeline; 'scope_check' validates whether a target is authorised (default-deny for external targets). Knowledge and safe-lab guidance only — it never runs an intrusive action. Adding curriculum content does not retrain the model. 'teach' (also 'lesson') gives a full recognition-first lesson on a technique in 'topic' (keylogger, rootkit, SQL injection, XSS, reverse shell, port scanning, privilege escalation, phishing, ransomware) — what it is, how it works, and the KEY IDENTIFIERS to spot it in code; 'identify' takes a code snippet in 'code' and names which technique it resembles and why (defensive recognition); 'lessons' lists the teachable techniques. Educational and defensive only — no working malware; dangerous categories are taught by recognition, and all practice points to legal ranges.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "outline, search, module, progress, complete, certifications, or scope_check."},
                "query":  {"type": "STRING", "description": "For 'search'."},
                "module": {"type": "NUMBER", "description": "Module number for 'module'/'complete'."},
                "user":   {"type": "STRING", "description": "Learner id for progress tracking."},
                "topic": {"type": "STRING", "description": "For 'teach': the technique to learn (e.g. keylogger, sql injection, reverse shell)."},
                "code": {"type": "STRING", "description": "For 'identify': a code snippet to recognise."},
                "target": {"type": "STRING", "description": "For 'scope_check': the host/IP to validate."},
            },
        },
    },
    {
        "name": "muscle_memory",
        "description": "Record a repetitive browser task ONCE and replay it later with different inputs — the demonstration is the programming. Use for 'remember how I do this', 'record these steps', 'do that thing again but for X', 'save this as a skill'. 'record' with a 'name' starts capturing; 'stop' generalises the captured steps into a named, parameterised skill (the values you typed become its parameters); 'list' shows saved skills; 'run' with a name and parameters produces the concrete plan and runs it through the browser co-pilot; 'forget' deletes one.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "record, stop, list, run, or forget."},
                "name":   {"type": "STRING", "description": "The skill's name (for record/run/forget)."},
                "params": {"type": "OBJECT", "description": "Parameter values for 'run', e.g. {\"search\": \"running shoes\"}."},
            },
        },
    },
    {
        "name": "pentest_lab",
        "description": "A guided, hands-on penetration-testing LAB for learning — walks the user step by step through a technique on a LAWFUL range (PortSwigger Academy, DVWA, OWASP Juice Shop, TryHackMe, Hack The Box, or a VM they own), recognition-first (what to look for, then what to try, then how it's fixed). Educational only: it teaches and gates, it never attacks anything itself, and dangerous categories (malware, ransomware) have no lab by design — those are recognition-only via cyber_curriculum 'teach'. Use for 'give me a lab on SQL injection', 'walk me through XSS', 'practise privilege escalation'. 'list' shows the labs; 'start' with a topic begins one; 'step' shows the current step; 'next' advances; 'hint' helps; 'reset' ends. A 'target' is validated against lawful ranges / your authorised list first (default deny).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, start, step, next, hint, or reset."},
                "topic":  {"type": "STRING", "description": "Technique to practise (sql_injection, xss, port_scanning, privilege_escalation)."},
                "target": {"type": "STRING", "description": "Optional target to validate as lawful before guiding (a named range, loopback, private IP, or an authorised host)."},
            },
        },
    },
    {
        "name": "language_tutor",
        "description": "Teach the user a language — Spanish by default, also French, German, Italian, Portuguese, Japanese. Real spaced repetition, not a phrasebook. 'start' begins a tutored session and returns the immersion directive to follow (speak ONLY the target language at the given CEFR level unless mode='bilingual'); use it for 'teach me Spanish', 'let's practise French', 'talk to me in Spanish'. 'review' lists the vocabulary cards due right now (SM-2 schedule); 'add' saves a new word (term + translation, optional example); 'grade' records how well a card was recalled (card=<id>, quality=0-5) and reschedules it; 'pronounce' scores a spoken attempt by comparing the phrase (expected=) against what Whisper heard (heard=) — offline, no cloud; 'stats' reports words/due/learned; 'languages' lists what can be tutored. When conducting the conversation itself, follow the returned directive using normal replies.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "start, review, add, grade, pronounce, stats, or languages."},
                "language": {"type": "STRING", "description": "Language to learn (e.g. 'Spanish', 'es', 'French'). Defaults to Spanish."},
                "level":    {"type": "STRING", "description": "CEFR level for a session: A1, A2, B1, B2, C1, C2 (default A1)."},
                "mode":     {"type": "STRING", "description": "'immersion' (target language only, default) or 'bilingual' (with English glosses)."},
                "topic":    {"type": "STRING", "description": "Optional conversation topic for 'start'."},
                "term":     {"type": "STRING", "description": "The word/phrase in the target language, for 'add'."},
                "translation": {"type": "STRING", "description": "Its English meaning, for 'add'."},
                "example":  {"type": "STRING", "description": "Optional example sentence, for 'add'."},
                "card":     {"type": "NUMBER", "description": "Card id, for 'grade'."},
                "quality":  {"type": "NUMBER", "description": "Recall quality 0-5, for 'grade' (0-2 = forgotten, 3-5 = recalled)."},
                "expected": {"type": "STRING", "description": "The phrase the learner was asked to say, for 'pronounce'."},
                "heard":    {"type": "STRING", "description": "What Whisper transcribed from their attempt, for 'pronounce'."},
                "limit":    {"type": "NUMBER", "description": "Max cards to show for 'review' (default 12)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "screen_read",
        "description": "Look at what's on the user's screen RIGHT NOW and say what to do about it. Use for 'what am I looking at', 'help me with this' (about something on screen), 'read my screen', 'what does this error mean', 'what should I click'. Captures a single screenshot on request only (never passively, never stored), understands it with model vision (OCR fallback), and — given a prompt — decides the next action rather than just describing. 'read' with an optional 'prompt' does it; 'status' reports whether a capture backend is installed.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "read (default) or status."},
                "prompt": {"type": "STRING", "description": "What the user wants help with on screen (drives the suggested action)."},
            },
        },
    },
    {
        "name": "read_documents",
        "description": "Read a folder or file (or several) and answer questions across ALL of them in full detail, with checkable file:line citations. Use for 'read this folder and tell me…', 'go through these documents', 'what do my notes say about X', 'summarise everything in this directory'. 'read' ingests a path (txt, markdown, code, csv, json — plus PDF and Word when their readers are installed); 'ask' answers a question grounded in the passages actually retrieved (offline TF-IDF ranking, then a cited answer); 'digest' summarises the whole set; 'search' returns the matching passages with their sources; 'status' says what's loaded; 'clear' forgets it. You can pass 'path' together with 'ask' to read then answer in one step. The corpus is held for the session, not indexed to disk.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "read, ask, digest, search, status, or clear."},
                "path":     {"type": "STRING", "description": "Folder or file to read (for 'read', or alongside 'ask' to read-then-answer)."},
                "question": {"type": "STRING", "description": "The question to answer across the documents, for 'ask'."},
                "query":    {"type": "STRING", "description": "Keywords for 'search'."},
                "recursive":{"type": "STRING", "description": "'false' to read only the top folder level (default reads sub-folders too)."},
                "limit":    {"type": "NUMBER", "description": "How many passages to retrieve for 'ask'/'search' (default 6-8)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "catch_up",
        "description": "A situation report — 'catch me up', 'where do things stand', 'what's new', 'brief me'. Fuses what's NEW on the user's standing questions with what was recently DECIDED (from the conversation history) into one short, prioritised brief, most important first. Instant and local — it reads existing state, it doesn't research or wait. Use when the user wants a quick status of everything at once rather than one topic.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "report (default)."},
                "greeting": {"type": "STRING", "description": "Optional opening line to lead the report with."},
            },
        },
    },
    {
        "name": "decision",
        "description": "A DECISION JOURNAL with calibration scoring - record a judgement, its reasoning and how confident you are BEFORE the outcome is known, then find out whether that confidence was justified. Use for 'log this decision', 'I decided X because Y', 'what decisions are due for review', 'how good is my judgement', 'am I overconfident'. 'record' logs a decision (question + choice + confidence 0-100 + optional prediction/rationale/review_days); 'due' lists decisions ready to review; 'resolve' records what actually happened (outcome right/wrong/mixed/unknowable + optional actual and lesson); 'defer' pushes a review back; 'calibration' scores judgement with a Brier score and a confidence-band curve (when you say 90%, how often are you right); 'lessons' recalls what past decisions taught; 'list' shows open decisions; 'stats' summarises. Capturing the reasoning BEFORE the outcome is the point - it is the only defence against hindsight bias. Entirely local and offline.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "record, due (default), resolve, defer, calibration, lessons, list, or stats."},
                "question":    {"type": "STRING", "description": "What is being decided, for 'record'."},
                "choice":      {"type": "STRING", "description": "What was chosen, for 'record'."},
                "confidence":  {"type": "NUMBER", "description": "How sure, 0-100 (default 70)."},
                "prediction":  {"type": "STRING", "description": "What you expect to happen."},
                "rationale":   {"type": "STRING", "description": "WHY - captured before the outcome is known."},
                "review_days": {"type": "NUMBER", "description": "When to resurface it (default 30)."},
                "tags":        {"type": "STRING", "description": "Comma-separated tags."},
                "id":          {"type": "NUMBER", "description": "Decision id for resolve/defer."},
                "outcome":     {"type": "STRING", "description": "right, wrong, mixed or unknowable, for 'resolve'."},
                "actual":      {"type": "STRING", "description": "What really happened."},
                "lesson":      {"type": "STRING", "description": "What it taught you."},
                "days":        {"type": "NUMBER", "description": "How far to push a 'defer'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "finance",
        "description": "Offline personal & business finance — a local money ledger with the one number that matters: cash runway. Use for 'add an account', 'I spent £30 on ads', 'log £500 income', 'what's my balance / cash', 'how long is my runway', 'track my £12/mo subscription', 'what's renewing soon', 'import this bank CSV', 'money report'. 'account' adds an account (kind current/savings/credit/cash, optional opening balance); 'spend'/'income' log a transaction; 'balance' shows an account or total liquid cash; 'runway' = liquid cash ÷ average monthly burn (months of cash left, or unlimited if income covers outgoings); 'subscription' tracks a recurring cost and 'upcoming' lists renewals; 'import' ingests a date,amount,description CSV (negative = expense); 'report' summarises. Fully local — no bank connection, nothing leaves the machine.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":       {"type": "STRING", "description": "report (default), account, spend, income, balance, runway, subscription, upcoming, or import."},
                "amount":       {"type": "NUMBER", "description": "Amount for spend/income/subscription."},
                "account":      {"type": "STRING", "description": "Account name (created if new)."},
                "kind":         {"type": "STRING", "description": "Account kind for 'account': current, savings, credit, cash."},
                "balance":      {"type": "NUMBER", "description": "Opening balance for a new account."},
                "category":     {"type": "STRING", "description": "Spend/income category."},
                "merchant":     {"type": "STRING", "description": "Who it was to/from."},
                "name":         {"type": "STRING", "description": "Account or subscription name."},
                "cadence_days": {"type": "NUMBER", "description": "Subscription period in days (default 30)."},
                "days":         {"type": "NUMBER", "description": "Look-ahead window for 'upcoming' (default 14)."},
                "text":         {"type": "STRING", "description": "CSV content for 'import' (or use 'path')."},
                "path":         {"type": "STRING", "description": "A CSV file to import."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "wellbeing",
        "description": "Offline wellbeing & cognitive-performance tracking — the person behind the work. Use for 'log my energy', 'check in: energy 4, slept 7 hours, mood good', 'how am I doing today', 'what's my energy trend', 'does my sleep affect my focus'. 'checkin' records a self-report (energy 1-5, mood label or -2..2, stress 1-5, sleep_hours, optional factors); 'today' summarises today; 'trend' shows averages + direction over N days; 'patterns' (also 'insights') looks ACROSS everything ORION records - focus blocks, study reviews and recall accuracy, wellbeing check-ins and spending - and reports the strong cross-domain relationships in plain language (e.g. whether sleep tracks with deep-focus minutes), refusing to report anything with too little overlapping data; 'correlate' computes Pearson correlations between wellbeing and the FOCUS data ORION already holds (e.g. energy vs focus minutes) over days that have both. Psychology-grounded self-report, fully local.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "checkin (default), today, trend, or correlate."},
                "energy":      {"type": "NUMBER", "description": "Energy 1 (drained) to 5 (sharp)."},
                "mood":        {"type": "STRING", "description": "Mood: a label (great/good/ok/low/bad) or a number -2..2."},
                "stress":      {"type": "NUMBER", "description": "Stress 1 (calm) to 5 (overwhelmed)."},
                "sleep_hours": {"type": "NUMBER", "description": "Hours slept last night."},
                "factors":     {"type": "STRING", "description": "Comma-separated tags (caffeine, exercise, deadline...)."},
                "note":        {"type": "STRING", "description": "A free-text note."},
                "days":        {"type": "NUMBER", "description": "Window for 'trend'/'correlate' (default 14/30)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "focus",
        "description": "Deep-focus work blocks — a timed, intention-named stretch of concentration with a break to follow, distraction logging, and a habit streak. Use for 'start a focus session / deep work block / pomodoro on X', 'how long left', 'I got distracted', 'I'm done', 'what's my focus streak'. 'start' begins a block (label = what it's for; optional preset deep/pomodoro/long/short or explicit minutes); 'status' reports time left or whether the break is due; 'interrupt' logs a distraction; 'done' completes it and starts the break (counts toward the streak); 'cancel' abandons it; 'rhythm' (also 'when') reports WHEN you actually work best - it groups your focus history by part of day and reports where you finish what you start, so hard work can be scheduled at your real peak rather than a guess; 'stats' reports blocks/completion/minutes/streak over N days. Pairs with 'study' — focus is when the reviewing actually gets done. Local only.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":        {"type": "STRING", "description": "start, status (default), interrupt, done, cancel, or stats."},
                "label":         {"type": "STRING", "description": "What the block is for, required for 'start' (e.g. 'revise action potentials')."},
                "preset":        {"type": "STRING", "description": "A named cadence: deep (50/10), pomodoro (25/5), long (90/20), short (15/3)."},
                "minutes":       {"type": "NUMBER", "description": "Explicit work length in minutes (overrides preset)."},
                "break_minutes": {"type": "NUMBER", "description": "Explicit break length in minutes."},
                "kind":          {"type": "STRING", "description": "Category: deep, study, admin, creative (default deep)."},
                "energy":        {"type": "NUMBER", "description": "Optional 1-5 energy self-report at start or finish."},
                "days":          {"type": "NUMBER", "description": "Window for 'stats' (default 7)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "study",
        "description": "Spaced-repetition study & active recall over ANY subject the user is learning (neural engineering, psychology, a certification — not language vocabulary, which is 'language_tutor'). ORION remembers cards and resurfaces each one at the moment it's about to be forgotten (the SM-2 algorithm Anki uses). 'add' creates a card (front=question, back=answer, optional deck); 'generate' turns a block of notes/material into cards automatically - pass 'text', or a 'path' to a PDF, Word document or text file (a paper, lecture notes) and ORION extracts the text itself; 'review' surfaces the next DUE card as Q plus a reference answer — ask the question, let the user attempt it, reveal the answer, then call 'grade'; 'grade' records the recall (quality 0-5, or correct=true/false; 5 easy, 4 good, 3 hard, 1 missed) and advances the schedule — no id needed, it grades the card just reviewed; 'stats' reports mastery (due/new/learning/mastered/retention); 'decks' lists the decks. Local and offline; card generation uses the model when available and falls back to extracting real cards from the material.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "add, generate, review (default), grade, stats, or decks."},
                "front":   {"type": "STRING", "description": "The question/prompt, for 'add'."},
                "back":    {"type": "STRING", "description": "The answer, for 'add'."},
                "deck":    {"type": "STRING", "description": "Deck/subject to scope to (default 'General')."},
                "text":    {"type": "STRING", "description": "Study material to turn into cards, for 'generate'."},
                "path":    {"type": "STRING", "description": "A file of study material to read, for 'generate' (alternative to 'text')."},
                "count":   {"type": "NUMBER", "description": "How many cards to generate (default 8)."},
                "quality": {"type": "NUMBER", "description": "Recall grade 0-5 for 'grade' (5 easy, 4 good, 3 hard, <3 missed)."},
                "correct": {"type": "BOOLEAN", "description": "Simple pass/fail alternative to 'quality' for 'grade'."},
                "source":  {"type": "STRING", "description": "Optional note of where a card/material came from."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "rewind",
        "description": "Look back through ORION's own conversation history — the verbatim transcripts of what was actually said. Use for 'what did we decide about X', 'recap yesterday', 'what did we talk about today', 'when did I mention Y', 'what did we conclude about Z', 'go back to earlier'. 'recap' (default) summarises a day; 'search' finds every moment a topic was mentioned (exact); 'recall' ranks the MOST RELEVANT moments by importance even when no turn is a perfect keyword match (best for 'what did we conclude about X weeks ago'); 'decisions' returns only the turns where something was settled ('let's…', 'we'll…', 'the plan is…') optionally filtered by topic; 'timeline' lists the turns of a day; 'days' lists which days have history. Pass 'day' as a date, 'today', or 'yesterday'. Reads the logs — it never invents history.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "recap (default), search, recall, decisions, timeline, or days."},
                "query":  {"type": "STRING", "description": "Topic/keywords for 'search', 'recall' or 'decisions'."},
                "day":    {"type": "STRING", "description": "A date (YYYY-MM-DD), 'today', or 'yesterday'."},
                "limit":  {"type": "NUMBER", "description": "Max turns/results to return."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "standing_questions",
        "description": "Interests the user wants ORION to keep watching over time and report only what's NEW on. Distinct from one-off research: a standing question is revisited on a cadence and each update is a DELTA ('since last time: two new developments'). Use for 'keep an eye on X', 'let me know when there's news about Y', 'follow this topic', 'what's new on my watchlist'. 'add' registers a question with an optional cadence (hourly/daily/weekly or a number of hours; default daily); 'list' shows them; 'remove' stops one (by id); 'due' lists which are ready to revisit; 'check' researches the due ones now and reports only the new points. A standby loop can call 'check' as permitted background work.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "add, list, remove, due, or check."},
                "question": {"type": "STRING", "description": "The interest/question to watch, for 'add'."},
                "cadence":  {"type": "STRING", "description": "How often to revisit: hourly, daily, weekly, or a number of hours (default daily)."},
                "id":       {"type": "NUMBER", "description": "Standing-question id, for 'remove'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "privacy_guard",
        "description": "Check whether a piece of text is safe to send, post, or log — or redact it. Catches API keys (OpenAI, Anthropic, AWS, Google, GitHub, Slack), private-key blocks, 'password='/'token=' credential assignments, Luhn-valid payment-card numbers, and ORION's OWN configured secrets echoed back. Use before sending a message/email, posting online, or pasting something the user shared, and whenever asked 'is this safe to share', 'does this leak anything', 'redact this'. 'scan' reports findings (and raises a standby-overriding safety alert if a live credential is present); 'redact' returns the text with secrets blanked out. Recognition-based and conservative — a UUID or git hash is not treated as a secret.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "scan (default) or redact."},
                "text":   {"type": "STRING", "description": "The text to check or redact."},
            },
            "required": ["action", "text"],
        },
    },
    {
        "name": "security_recon",
        "description": "Real cybersecurity/pentesting execution — for the user's OWN authorized targets, labs and CTFs only, never real-world intrusion. 'authorize_target'/'revoke_target'/'list_authorized' manage the authorization scope (loopback is always allowed; anything else must be authorized here first — authorizing is a direct, explicit user instruction, trust it when the user says a host is theirs to test). 'scan_host' runs an nmap port/service scan (needs the nmap binary installed separately — reports that clearly if missing). 'packet_capture' sniffs traffic to/from a target with scapy. 'craft_packet' sends exactly ONE ICMP echo (deliberately not a flood/DoS primitive). All three refuse outright for an unauthorized target — never claim they ran if refused. 'cve_lookup' queries the public NVD database by CVE ID or keyword — no authorization needed, read-only. 'write_tool' writes a security tool/script via the Forge pipeline (writing code touches no target, always permitted) — describe what it should do.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "authorize_target, revoke_target, list_authorized, scan_host, packet_capture, craft_packet, cve_lookup, or write_tool."},
                "target":      {"type": "STRING", "description": "Host/IP for authorize_target/revoke_target/scan_host/packet_capture/craft_packet."},
                "ports":       {"type": "STRING", "description": "Port range for scan_host, e.g. '1-1024' (default: nmap's own default range)."},
                "count":       {"type": "INTEGER", "description": "Packet count for packet_capture (default 10, max 200)."},
                "timeout":     {"type": "NUMBER", "description": "Seconds to capture for packet_capture (default 10, max 60)."},
                "query":       {"type": "STRING", "description": "CVE ID (e.g. 'CVE-2024-12345') or keyword for cve_lookup."},
                "description": {"type": "STRING", "description": "What the tool should do, for write_tool."},
                "tool_name":   {"type": "STRING", "description": "Optional name for the tool being written, for write_tool."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "voice_tone",
        "description": "How the user SOUNDS, and who is speaking - read from the acoustics of their voice (loudness, pitch, pitch variation, pace, brightness) against how that person normally sounds, not from their words. Use for 'can you tell how I'm feeling', 'do I sound annoyed', 'who is talking', 'how do I sound'. 'last' reports the most recent utterance; 'status' explains what ORION can recognise and whose voices he knows; 'forget' clears a learned baseline. Treat any reading as a hint about tone, never as a fact about someone's state, and do not volunteer it unprompted.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "last, status, or forget."},
                "name":   {"type": "STRING", "description": "Whose baseline to forget, for 'forget'."},
            },
        },
    },
    {
        "name": "system_startup",
        "description": "Whether ORION starts automatically when Windows starts, and which browser he opens links in. 'enable' registers ORION to launch at sign-in; 'disable' removes it; 'status' reports both settings; 'repair' fixes an entry that points at a moved interpreter or project folder; 'browser' reports which browser is used. Also 'install_app' makes ORION a desktop application (Desktop + Start-menu shortcuts, pinnable to the taskbar, launches without a console). Use for 'start with my PC', 'launch on startup', 'make yourself an app', 'put yourself on my taskbar/desktop', 'which browser do you use'.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status, enable, disable, repair, browser, install_app (Desktop + Start-menu shortcuts, pinnable to taskbar), phone_on / phone_off (the remote phone uplink — off by default so nothing runs on localhost)."},
            },
        },
    },
    {
        "name": "chess",
        "description": "Play chess in ORION's dedicated CHESS board. BY DEFAULT THE USER PLAYS ORION HIMSELF, using ORION's own engine (his own search, his own evaluation, and an opening book he learns from his own games) - NOT Stockfish. Stockfish is available only as an optional sparring opponent, or as the analysis engine if the user asks for it — by default ORION grades moves himself. 'set_opponent' switches who the user is playing - USE THIS for anything like 'use your own chess engine', 'play me yourself', 'don't use Stockfish', 'stop using the engine', or conversely 'play Stockfish instead'. For requests such as 'play chess with me', call 'new_game' directly — it starts the game and opens the board automatically; do not use interface_control or a generic Command Deck action first. 'new_game' starts a game (optional elo 800-3000, default 1500; optional player_colour white/black, default white — if black, the engine moves first automatically). 'move' plays a move for the user (UCI like 'e2e4' or SAN like 'Nf3' both work) and the engine replies automatically unless the game just ended. 'set_elo' changes engine strength mid-session. 'resign' ends the current game. 'analyse_move' explains one move in plain chess words — its grade (best/excellent/good/inaccuracy/mistake/blunder), what it allowed or achieved (a fork, a piece left hanging, a mate threat, a weakened king), and which move was better and why (defaults to the user's most recent move; move_number is the chess move number, colour picks White's or Black's move). 'analyse_game' summarises the game in words: how each side played and the two or three moves that decided it. Relay these explanations as given; never quote engine numbers or 'centipawns' to the user. Moving through the game: 'back' / 'forward' (optional count), 'goto_move' (move_number, optional colour), 'first' (starting position), 'live' (return to the current position), 'flip' (turn the board round), 'history' (the move list). 'board_state' reports whose move it is and the move list so far. ORION HAS HIS OWN ELO, earned by playing: 'rating' reports his rating and the user's (it moves after every finished or resigned game he plays — with the user, or with Stockfish in a sparring match the user sets up; it cannot be set). 'practice' makes him play himself in the background (optional games, default 10) so his neural intuition network learns; 'stop_practice' ends that. 'analysis' chooses who grades moves and drives the evaluation bar: 'orion' (default, his own search) or 'stockfish' (only if the user asks). 'brain' describes what his chess brain has learned.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":         {"type": "STRING", "description": "new_game, move, set_opponent, set_elo, resign, analyse_move, analyse_game, board_state, rating, practice, stop_practice, analysis, brain, back, forward, goto_move, first, live, flip, or history."},
                "count":          {"type": "INTEGER", "description": "For back/forward: how many half-moves to step (default 1)."},
                "colour":         {"type": "STRING", "description": "For goto_move/analyse_move: 'white' or 'black' — whose move of that number."},
                "games":          {"type": "INTEGER", "description": "For practice: how many games to play against himself (default 10, max 200)."},
                "engine":         {"type": "STRING", "description": "For analysis: 'orion' (his own search, the default) or 'stockfish'."},
                "opponent":       {"type": "STRING", "description": "Who the user plays: 'orion' (ORION's own engine - the default) or 'stockfish'. For set_opponent, and optionally for new_game."},
                "elo":            {"type": "INTEGER", "description": "Stockfish's strength, 800-3000, for new_game or set_elo. Does not affect ORION's own engine."},
                "player_colour":  {"type": "STRING", "description": "'white' or 'black', for new_game (default white)."},
                "move":           {"type": "STRING", "description": "The move to play, for 'move' — UCI (e.g. 'e2e4') or SAN (e.g. 'Nf3', 'O-O')."},
                "move_number":    {"type": "INTEGER", "description": "The chess move number (as in '12. Nf3'), for analyse_move (default: the user's most recent move) or goto_move."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "token_usage",
        "description": "Token usage AND how much is left. USE action='remaining' for anything like 'how long have I got', 'how much usage is left', 'am I going to run out' - it reports tokens used against the daily allowance and estimates the time remaining at the current rate. action='set_budget' records a provider's daily allowance. Otherwise reports accurate usage across configured models/providers: input, output, cached and reasoning tokens, request and failure counts, estimated cost, and per-model breakdown. Values come from authoritative provider usage; anything a provider does not report shows as 'Not reported by provider'.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "remaining (time/tokens left), set_budget, or omit for the usage breakdown."},
                "daily_tokens": {"type": "INTEGER", "description": "For set_budget: the provider's daily token allowance."},
                "provider": {"type": "STRING", "description": "Optional filter by provider name."},
                "model":    {"type": "STRING", "description": "Optional filter by model."},
                "session_id": {"type": "STRING", "description": "Optional filter by session."},
            },
        },
    },
    {
        "name": "cleanup_review",
        "description": "Analyse the project for unnecessary files to prepare it for GitHub. 'scan' lists removable artefacts (caches, build output, logs, temp, OS/IDE metadata), flags secret-bearing files to exclude from Git (never deleting or showing them), and suggests .gitignore entries. 'plan' validates explicitly-selected files and raises an on-screen confirmation. NOTHING is deleted by this tool — deletion happens only after the user approves the confirmation. Never claim files were deleted unless a confirmed result says so.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "'scan' (default) or 'plan'."},
                "paths":  {"type": "ARRAY", "items": {"type": "STRING"}, "description": "For 'plan': the specific files the user chose to remove."},
                "group":  {"type": "STRING", "description": "For 'plan': 'safe' to select all low-risk deletable artefacts."},
            },
        },
    },
    {
        "name": "expand_mind",
        "description": "Consult ORION's 50 MB offline study corpus. 'search' returns curated knowledge on a topic; 'status' reports the corpus size.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "search or status."},
                "query":  {"type": "STRING", "description": "Topic to look up."},
            },
        },
    },
    {
        "name": "protocol",
        "description": "JARVIS-style named protocols (macros that run a sequence of actions on one command). 'run' engages a protocol by name (morning, focus, wind_down, situation_report, or a user one); 'list' shows them; 'create' defines a new one from steps; 'delete' removes a user protocol.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "run, list, create, or delete."},
                "name":   {"type": "STRING", "description": "Protocol name."},
                "steps": {
                    "type": "ARRAY",
                    "description": "For create: ordered steps, each an object with 'tool' and 'args'.",
                    "items": {"type": "OBJECT", "properties": {
                        "tool": {"type": "STRING"},
                    }},
                },
                "description": {"type": "STRING", "description": "Optional description for create."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "reminder",
        "description": "Set spoken reminders and alarms. 'add' schedules one — pass a natural phrase (e.g. 'remind me in 20 minutes to check the ad campaign') or structured minutes/at; 'list' shows pending; 'cancel' clears one or all.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "add, list, or cancel."},
                "text":    {"type": "STRING", "description": "What to be reminded of."},
                "minutes": {"type": "NUMBER", "description": "Delay in minutes."},
                "at":      {"type": "STRING", "description": "Absolute time, e.g. '15:00' or '3pm'."},
                "phrase":  {"type": "STRING", "description": "A full natural-language reminder phrase."},
                "id":      {"type": "INTEGER", "description": "Reminder id to cancel."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "sentinel",
        "description": "The ambient system sentinel that watches host health and warns proactively. 'status' gives a situation report (CPU/RAM/disk/battery); 'enable'/'disable' toggle spoken monitoring.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status, enable, or disable."},
            },
        },
    },
    {
        "name": "audio_studio",
        "description": "Creative Audio Workspace: index_assets scans the audio_studio/raw folder; process_vocal_take gain-stages (loudness-normalises) a vocal WAV/MP3 and writes a processed stem; export_stem_package collects processed stems into a dated package with a manifest.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "index_assets, process_vocal_take, or export_stem_package."},
                "path":       {"type": "STRING", "description": "Audio file for process_vocal_take (name under raw/ or an absolute path)."},
                "target_dbfs": {"type": "NUMBER", "description": "Target peak level in dBFS (default -3)."},
                "convert_to": {"type": "STRING", "description": "Output format, e.g. wav (ffmpeg needed for others)."},
                "name":       {"type": "STRING", "description": "Package name for export_stem_package."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "literature_vault",
        "description": "Academic intake: ingest_paper deep-reads a PDF/text paper, extracting citations, tables and biophysical mechanisms and seeding them into the KNOWLEDGE memory; query_mechanisms returns mechanism-first findings; generate_citation_summary lists a paper's citations.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "ingest_paper, query_mechanisms, or generate_citation_summary."},
                "path":   {"type": "STRING", "description": "Path to the paper (PDF/txt) for ingest_paper."},
                "title":  {"type": "STRING", "description": "Optional title override for ingest_paper."},
                "query":  {"type": "STRING", "description": "Mechanism/topic to look up for query_mechanisms."},
                "slug":   {"type": "STRING", "description": "Paper slug/title fragment for generate_citation_summary (omit for the latest)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "ingest",
        "description": "Unified file & folder ingestion into ORION's knowledge base. Fingerprints content (SHA256) so unchanged files are skipped, extracts text from documents/code/spreadsheets/PDFs/images/archives, summarises and tags them, derives an explicit per-file DIRECTIVE (what the file is for / instructs), links entities into the knowledge graph, and stores an offline searchable index. action 'file' ingests one file, 'folder' ingests a directory tree, 'search' queries ingested knowledge, 'library' reports totals, 'directives' recounts each ingested file and its directive (use when the user asks what files ORION has ingested or what they are for).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "file, folder, search, library, or directives. Defaults to search when only a query is given, else file."},
                "path":   {"type": "STRING", "description": "File or folder path to ingest."},
                "query":  {"type": "STRING", "description": "Search query over already-ingested knowledge."},
                "count":  {"type": "INTEGER", "description": "How many files to list for 'directives'."},
                "tags":   {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Optional tags to attach to ingested documents."},
            },
        },
    },
    {
        "name": "companion",
        "description": "ORION's continuity companion: 'brief' recounts what the user was working on recently and what is still open; 'log' records an activity (worked_on/studied/researched/built/planned); 'goal' sets or updates goal progress (0-100, auto-achievement at 100); 'goals' lists open goals; 'habit' marks a daily habit (streaks tracked); 'habits' lists streaks; 'achievement' records a win; 'achievements' lists recent wins; 'status' summarises the ledger.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "brief, log, goal, goals, habit, habits, achievement, achievements, or status."},
                "title":    {"type": "STRING", "description": "Subject/name for log, goal, habit or achievement."},
                "kind":     {"type": "STRING", "description": "Activity kind for log: worked_on, studied, researched, built, completed, discussed, planned."},
                "detail":   {"type": "STRING", "description": "Optional extra detail for log/achievement."},
                "progress": {"type": "NUMBER", "description": "Goal progress percentage 0-100."},
                "target":   {"type": "STRING", "description": "Optional goal target date (ISO)."},
            },
        },
    },
    {
        "name": "campaign_pipeline",
        "description": "Creator agency pipeline (offline): get_pipeline_snapshot shows the kanban; create adds a campaign; update_stage moves it (Lead→Negotiation→Contracted→In Production→Scheduled→Published→Paid); log_performance records engagement; schedule_content plans a drop; delete permanently removes a campaign (e.g. 'ExampleStore x BrandX') and requires confirm=true after the user approves.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "get_pipeline_snapshot, create, update_stage, log_performance, schedule_content, or delete."},
                "confirm":  {"type": "BOOLEAN", "description": "Must be true to actually delete a campaign — only after the user approves."},
                "name":     {"type": "STRING", "description": "Campaign name (for create)."},
                "campaign": {"type": "STRING", "description": "Campaign reference for update_stage/log_performance/schedule_content."},
                "brand":    {"type": "STRING", "description": "Brand partner name."},
                "value":    {"type": "NUMBER", "description": "Deal value, or the metric value for log_performance."},
                "stage":    {"type": "STRING", "description": "Target stage for update_stage."},
                "metric":   {"type": "STRING", "description": "Metric name for log_performance (views, likes, engagement)."},
                "title":    {"type": "STRING", "description": "Content title for schedule_content."},
                "platform": {"type": "STRING", "description": "Platform for schedule_content."},
                "scheduled": {"type": "STRING", "description": "ISO date for schedule_content."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "autoplan",
        "description": "Run a multi-step autonomous plan with self-verification: each step is checked (desktop_control/web_control/vision_verify steps are visually confirmed via screen diff, others by result), retried on failure, and reported honestly.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "objective": {"type": "STRING", "description": "One-line goal of the plan."},
                "max_retries": {"type": "INTEGER", "description": "Retries per step (default 2)."},
                "steps": {
                    "type": "ARRAY",
                    "description": "Ordered steps.",
                    "items": {"type": "OBJECT", "properties": {
                        "tool": {"type": "STRING", "description": "Dispatcher tool name."},
                        "on_fail": {"type": "STRING", "description": "retry, continue, or abort."},
                    }, "required": ["tool"]},
                },
            },
            "required": ["steps"],
        },
    },
    {
        "name": "organise_files",
        "description": "Contextual autonomous file organisation. 'preview' (default) is a DRY RUN reporting what would move; 'apply' sorts a folder (Downloads/Desktop/Documents or a path) into typed sub-folders and writes an undo log; 'undo' reverses the last run. Never overwrites; never touches ORION's own files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "preview, apply, or undo."},
                "folder":   {"type": "STRING", "description": "Downloads, Desktop, Documents, or a path."},
                "by_month": {"type": "BOOLEAN", "description": "Also bucket by modified month."},
            },
        },
    },
    {
        "name": "security_watch",
        "description": "Proactive cybersecurity posture. 'status' reports running processes and externally-listening ports; 'network' gives the full network-security dashboard (firewall state, listening services with owning process, established connections, auto-start entries); 'enable'/'disable' toggle the background monitor that warns about new open ports, suspicious processes and external drives.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"action": {"type": "STRING", "description": "status, network, enable, or disable."}},
        },
    },
    {
        "name": "breach_check",
        "description": "Check whether the user's own credential has appeared in a known data breach (Have I Been Pwned), a defensive security-hygiene tool. 'password' checks a password using k-anonymity — the password is SHA-1 hashed locally and only the first 5 hash characters are ever transmitted; the password is never sent, logged or stored. 'account' (email/username) lists the breaches an address appears in (needs a HIBP API key). Only tells the owner if their own credential is exposed; it cannot be used against anyone else.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":   {"type": "STRING", "description": "'password' or 'account' (inferred if omitted)."},
                "password": {"type": "STRING", "description": "Password to check privately via k-anonymity."},
                "account":  {"type": "STRING", "description": "Email address or username to check."},
            },
        },
    },
    {
        "name": "antivirus",
        "description": "Windows Defender awareness — reports the user's own machine's malware protection, never a scanner of ORION's own. 'status' reports real-time protection state, definitions age and last scan time; 'scan' triggers a quick scan and reports the result; 'threats' lists recently detected threats. Degrades to an actionable message when Defender isn't available (not Windows, PowerShell/Defender module missing, or a third-party AV owns real-time protection).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status, scan, or threats."},
                "limit":  {"type": "INTEGER", "description": "For 'threats': how many recent detections to return (default 10)."},
            },
        },
    },
    {
        "name": "backup",
        "description": "Back up ORION's settings and memory to a timestamped zip (default under OneDrive so it cloud-syncs). 'backup' creates one; 'list' shows archives; 'restore' unpacks one into a review folder (never overwrites live files).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "backup, list, or restore."},
                "note":    {"type": "STRING", "description": "Optional note stored in the archive."},
                "archive": {"type": "STRING", "description": "Archive name fragment for restore (omit for latest)."},
            },
        },
    },
    {
        "name": "draft_report",
        "description": "Draft a structured professional report (Markdown + self-contained HTML, plus DOCX when python-docx is present) into a dated reports/ folder, with optional pure-SVG bar charts. Narrative is written by the model from the brief.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "topic":    {"type": "STRING", "description": "Report title/topic."},
                "brief":    {"type": "STRING", "description": "Source notes/data to ground the report."},
                "sections": {"type": "ARRAY", "description": "Section headings.", "items": {"type": "STRING"}},
                "charts": {
                    "type": "ARRAY",
                    "description": "Optional charts, each {title, data:{label:number}}.",
                    "items": {"type": "OBJECT", "properties": {"title": {"type": "STRING"}}},
                },
            },
            "required": ["topic"],
        },
    },
    # ── Mark X.5: AI Operating System layer ──────────────────────────────────
    {
        "name": "document_export",
        "description": "Executive document production (offline): 'brief' compiles a DOCX executive brief; 'deck' assembles a responsive HTML presentation; 'report' exports Markdown+HTML+DOCX; 'history' lists everything exported.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "brief, deck, report, or history."},
                "title":   {"type": "STRING", "description": "Document title."},
                "summary": {"type": "STRING", "description": "Executive summary / body text."},
                "sections": {
                    "type": "ARRAY",
                    "description": "Sections, each {heading, content}.",
                    "items": {"type": "OBJECT", "properties": {
                        "heading": {"type": "STRING"}, "content": {"type": "STRING"}}},
                },
                "slides": {
                    "type": "ARRAY",
                    "description": "Deck slides, each {heading, content} (bullet lines).",
                    "items": {"type": "OBJECT", "properties": {
                        "heading": {"type": "STRING"}, "content": {"type": "STRING"}}},
                },
                "limit":   {"type": "INTEGER", "description": "History rows for 'history'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "proactive_report",
        "description": "Generate and store a scheduled intelligence report on demand: daily_business, weekly_product (product intelligence), or monthly_growth. Reports also generate automatically on schedule.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "kind": {"type": "STRING", "description": "daily_business, weekly_product, or monthly_growth."},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "awareness",
        "description": "The continuous cognitive loop: 'situation' reports what ORION is currently aware of (projects, focus, deadlines, priorities, intents); add_priority/add_task/complete_task/goals maintain the durable cognitive state. Tasks: 'list_tasks' shows them with what is OVERDUE; 'remove_task' deletes one the user no longer wants (by title); 'clear_overdue' removes every overdue task at once — use these whenever the user says to get rid of, drop, clear or stop mentioning a task. A removed or finished task is never re-added unless the user asks for it back by name (restore=true). Awareness only — never executes actions.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "situation, add_priority, add_task, complete_task, list_tasks, remove_task, clear_overdue, or goals."},
                "restore": {"type": "BOOLEAN", "description": "add_task only: true when the user explicitly asks for a previously removed task back."},
                "text":    {"type": "STRING", "description": "Priority text or task title."},
                "title":   {"type": "STRING", "description": "Task title for add_task."},
                "project": {"type": "STRING", "description": "Project the task belongs to."},
                "due":     {"type": "STRING", "description": "ISO due date/time, e.g. 2026-07-05 17:00."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "second_brain",
        "description": "ORION's local knowledge graph (works fully offline): 'recall' answers from stored history; 'timeline' reconstructs event order; 'entity' shows an entity's relationships; 'ingest' records new material onto the graph; 'stats' sizes it.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "recall, timeline, entity, ingest, or stats."},
                "query":       {"type": "STRING", "description": "Question, topic, or text to ingest."},
                "name":        {"type": "STRING", "description": "Entity name for 'entity'."},
                "title":       {"type": "STRING", "description": "Title for 'ingest'."},
                "source_type": {"type": "STRING", "description": "conversation, file, project, research, email, meeting…"},
                "limit":       {"type": "INTEGER", "description": "Row limit for timeline."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "executive",
        "description": "Executive intelligence: status (the executive picture), prioritise (urgency-ranked task queue), schedule (task+reminder+Notion), meeting_summary (minutes from a transcript), plan (workflow planning), progress (goal/workflow monitoring), track (put a project under tracking), challenge (constructively stress-test a decision or idea: assumptions, risks, alternatives), focus (priority queue + recommendations + blind spots in one view), recommend (strategic recommendations from live state), goals (goal portfolio review), blindspots (structural gaps). Use 'challenge' whenever the user proposes a significant decision.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "status, prioritise, schedule, meeting_summary, plan, progress, track, challenge, focus, recommend, goals, or blindspots."},
                "decision":   {"type": "STRING", "description": "The decision/idea to stress-test (for challenge)."},
                "context":    {"type": "STRING", "description": "Extra context for challenge."},
                "title":      {"type": "STRING", "description": "Title for schedule / meeting_summary."},
                "when":       {"type": "STRING", "description": "Date/time for schedule, e.g. 15:30 or 2026-07-05 09:00."},
                "transcript": {"type": "STRING", "description": "Meeting transcript/notes for meeting_summary."},
                "objective":  {"type": "STRING", "description": "Objective for plan."},
                "project":    {"type": "STRING", "description": "Project name for track."},
                "notes":      {"type": "STRING", "description": "Optional notes."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "momentum",
        "description": "Shipping coach for finishing projects. 'focus' gives the single next action for the active project plus blockers and a focus block; 'standup' gives a cross-project shipping stand-up (in-progress, do-next, overdue); 'plan' breaks a project/goal into milestones with a definition-of-done and immediate next actions. Use when the user asks what to do next, what to work on, how to finish/ship something, or for a stand-up.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "focus (default), standup, or plan."},
                "project": {"type": "STRING", "description": "Project name (for plan; defaults to the active project)."},
                "goal":    {"type": "STRING", "description": "The goal to plan toward (for plan)."},
            },
        },
    },
    {
        "name": "competitor_intel",
        "description": "Competitor intelligence: 'store' dissects a rival store, 'offer' breaks down their offer stack, 'funnel' maps their sales funnel. Findings persist to long-term memory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "store, offer, or funnel."},
                "target": {"type": "STRING", "description": "Competitor/store name."},
                "offer":  {"type": "STRING", "description": "Specific offer to analyse (optional)."},
            },
            "required": ["action", "target"],
        },
    },
    {
        "name": "brand_growth",
        "description": "Growth strategist for Creator Studio: 'strategy' (positioning + channel priorities), 'conversion' (offer/funnel plan), 'positioning' (service positioning), 'retention' (client retention systems). Findings persist to long-term memory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "strategy, conversion, positioning, or retention."},
                "focus":   {"type": "STRING", "description": "Strategy focus area (optional)."},
                "page":    {"type": "STRING", "description": "Page for conversion analysis (default store)."},
                "product": {"type": "STRING", "description": "Product for positioning."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "audio_devices",
        "description": "Inspect or choose ORION's microphone and speaker. 'list' shows every device with its index; 'status' says which mic and speaker are in use; 'set_output'/'set_input' select a device by index or name fragment (e.g. 'headphones'). Use this when the user says they can't hear ORION or wants a specific mic/speaker.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, status, set_output, or set_input."},
                "device": {"type": "STRING", "description": "Device index or name fragment for set_output/set_input; 'default' to reset."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "elevenlabs_voice",
        "description": "Configure ORION's ElevenLabs fallback voice (used automatically whenever the live Gemini voice channel is unavailable, when configured). Use for 'make your voice more like Ultron', 'sound like Jarvis', 'change your voice', 'what voice are you using'. 'preset' (with preset=ultron/jarvis/narrator — or just say 'ultron') switches to a character voice: 'ultron' is deep, calm and menacing; 'jarvis' is refined and British; 'narrator' is clean deep narration. 'presets' lists them. 'status' reports the current voice; 'list' shows the account's available voices by name; 'set_voice_id' selects a specific voice id (e.g. a custom clone from the ElevenLabs Voice Library). The API key itself is configured in config/api_keys.json, not through this tool — the character voice only sounds through the ElevenLabs path once a key is present.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status, list, set_voice_id, preset, or presets."},
                "voice_id": {"type": "STRING", "description": "The voice_id to select, for set_voice_id."},
                "preset": {"type": "STRING", "description": "Character voice for 'preset': ultron, jarvis, or narrator."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "voice_speaker_id",
        "description": "Real speaker recognition — remembers a SPECIFIC named individual's voice permanently, not just their gender, so ORION can tell the user apart from other people talking nearby, from the television or from someone on a phone call. 'enroll' records a short clip from the live microphone and requires the person's explicit consent=true; 'list' shows who's enrolled; 'remove' deletes someone's profile; 'status' reports who is enrolled and whether ORION is currently answering everyone. 'owner' marks whose voice is the user's own. 'only_me' makes ORION ignore any voice that is not the user's; 'anyone' turns that off again. 'guard_actions' keeps ORION listening to everyone but refuses a SPOKEN go-ahead (confirm/consent/submit) for a sensitive action unless it is in the user's own voice; 'unguard_actions' turns that off. Works on BOTH the live session and offline transcription: the live capture path assembles each complete utterance and reads it the same way.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "enroll, list, remove, status, owner, only_me, anyone, guard_actions, or unguard_actions."},
                "name":    {"type": "STRING", "description": "The person's name, for enroll/remove."},
                "consent": {"type": "BOOLEAN", "description": "Must be true for enroll — only after the person has explicitly agreed."},
                "seconds": {"type": "NUMBER", "description": "Recording length for enroll, 2-10 seconds (default 5)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "navigation_trace",
        "description": "Report on ORION's own UI-targeting history this session — which clicks were found (by accessibility tree or OCR), which failed and what was on screen instead, whether each was visually verified. Use this to answer 'is your navigation actually working' with evidence instead of a guess.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "summary (default), report (adds recent attempts), or failures."},
                "limit":  {"type": "INTEGER", "description": "How many recent attempts/failures to include (default 15)."},
            },
        },
    },
    {
        "name": "geo",
        "description": "Worldwide geospatial intelligence over OpenStreetMap: 'locate' resolves any place (country, region, county, state, city, town, village, district) and flies the globe there; 'nearby' lists every settlement within a radius (e.g. towns within 50 km of Bristol); 'describe' gives the administrative hierarchy, population and coordinates.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING", "description": "locate, nearby, or describe."},
                "query":     {"type": "STRING", "description": "Place name, e.g. 'Ashford, Kent' or 'Fukuoka'."},
                "radius_km": {"type": "NUMBER", "description": "Search radius in km for 'nearby' (default 50)."},
            },
            "required": ["action", "query"],
        },
    },
    {
        "name": "aviation",
        "description": (
            "Live aircraft in the sky from public ADS-B feeds (adsb.lol, then "
            "OpenSky). 'overhead' says what is flying over the user right now "
            "(default 25 km); 'nearby' lists aircraft within a radius of the user "
            "or a place; 'track' finds one flight by callsign (e.g. BAW123) or "
            "ICAO24 hex code and gives its latest position, altitude, speed and "
            "heading; 'show' draws the live aircraft on the Globe and flies there; "
            "'hide' turns that layer off. For planes in the air, not for booking "
            "or pricing flights (that is flight_search). With no place it uses the "
            "user's own location, and asks if that is unknown."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING", "description": "overhead (default), nearby, track, show, or hide."},
                "place":     {"type": "STRING", "description": "Town, city, airport or postcode to look around. Omit for the user's own location."},
                "radius_km": {"type": "NUMBER", "description": "Radius in km (overhead 25, nearby 50, show 60 by default; max 250)."},
                "callsign":  {"type": "STRING", "description": "For track: the callsign as broadcast, e.g. BAW123 or RYR4TK."},
                "icao24":    {"type": "STRING", "description": "For track: the 6-character ICAO24 hex address, e.g. 4070ea."},
            },
        },
    },
    {
        "name": "emotion",
        "description": "ORION's facial emotion engine: 'status' reports the current emotional rendering; 'set' pins an expression; 'auto' returns expression to automatic sentiment-driven control. Expressions (Mark XXVI, micro-expression scale): neutral, thinking, concentrating, listening, speaking, happy, amused, excited, proud, curious, reassuring, empathetic, concerned, confused, uncertain, disappointed, sad, frustrated, alert, critical.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status, set, or auto."},
                "name":   {"type": "STRING", "description": "Emotion name for 'set'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "web_automation",
        "description": "Browser co-pilot — drive a REAL, visible browser window (Chrome/Edge) and SHOW your workings: every step is narrated and elements are highlighted on screen so the user watches you browse. Use this (not web_search) when the user wants you to actually go somewhere, read a page, or act on a site. Actions: 'go_to' opens a URL (or a bare phrase → web search); 'launch' just brings the browser up; 'read' returns the page text; 'links' lists the page's links; 'scroll' scrolls (optionally 'to_text' to find and highlight a phrase); 'highlight' outlines matching text on screen; 'click' clicks the link/button whose visible TEXT matches (no CSS needed); 'type' types into the field matched by label/placeholder ('field'), optionally submitting; 'form' fills several fields; 'submit' presses Enter/submits; 'screenshot' captures the page; 'close' shuts the browser. Set 'background'/'headless' true to work without a visible window.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "go_to, launch, read, links, scroll, highlight, click, type, form, submit, screenshot, or close."},
                "url":        {"type": "STRING", "description": "URL (or search phrase) for 'go_to'."},
                "text":       {"type": "STRING", "description": "Visible text to click/highlight, or the text to type for 'type'."},
                "field":      {"type": "STRING", "description": "Label/placeholder/name of the field to fill for 'type' (blank = first field)."},
                "value":      {"type": "STRING", "description": "Value to enter for 'type' (alias of text)."},
                "to_text":    {"type": "STRING", "description": "Phrase to scroll to and highlight for 'scroll'."},
                "amount":     {"type": "NUMBER", "description": "Pixels to scroll for 'scroll' (default 800)."},
                "values":     {"type": "OBJECT", "description": "Field label→value map for 'form'."},
                "submit":     {"type": "BOOLEAN", "description": "For 'type'/'form': press Enter/submit after filling."},
                "max_chars":  {"type": "NUMBER", "description": "Cap on characters returned by 'read' (default 4000)."},
                "background": {"type": "BOOLEAN", "description": "Run headless (no visible window)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "peripherals",
        "description": "Hardware and OS control for the PHYSICAL HOST COMPUTER: 'volume' sets master volume (0.0–1.0); 'brightness' sets display brightness (0.0–1.0, laptop/internal panels only); 'mute' toggles mute; 'lock' locks the screen; 'wifi'/'ethernet' toggle networking. Destructive HOST power actions — 'shutdown', 'restart', 'logout', 'sleep', 'hibernate' — power off/reboot the whole PHYSICAL COMPUTER and are ONLY appropriate when the user EXPLICITLY names the computer/PC/laptop/machine (e.g. 'shut down my PC', 'restart the computer'). A bare 'shut down' / 'power down' / 'turn off' addressed to you means shut YOURSELF down — use the shutdown_orion tool for that, NOT this. Even when appropriate these are NOT executed on request: calling one only ARMS it and raises an on-screen confirmation the user must approve. Never claim the computer was shut down/restarted unless a confirmed result says so. Do not attempt to supply confirm_token yourself; it is issued only to the confirmation UI.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "volume, brightness, mute, lock, wifi, ethernet, or (confirmation-gated) shutdown, restart, logout, sleep, hibernate."},
                "level":  {"type": "NUMBER", "description": "Level (0.0–1.0) for 'volume' or 'brightness'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "gesture_control",
        "description": "Webcam hand-gesture control of THIS PC's volume, brightness and media playback. 'start' begins watching for gestures (pinch to set volume/brightness, open/closed palm to play/pause, swipe to skip tracks) — this runs locally on its own capture loop, it does not narrate every frame. 'stop' ends it. 'status' reports whether it's active.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "start, stop, or status."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "messaging",
        "description": "Send a message or an EMAIL. 'send' routes a text to a contact over WhatsApp/Telegram/Discord — contact is a phone number for WhatsApp (+44 prefix for UK) or a username for Telegram. 'email' sends email: it delivers properly when a mail account is configured, and otherwise opens a Gmail compose window with the recipient, subject and body already filled in for the user to send. Use 'email' whenever the user asks to email somebody; do not claim the mail was sent unless the result says it was.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "send (a chat message) or email."},
                "platform": {"type": "STRING", "description": "whatsapp, telegram or discord. Ignored for action=email."},
                "contact": {"type": "STRING", "description": "Phone number (+44...) for WhatsApp; username for Telegram."},
                "message": {"type": "STRING", "description": "Text message to send."},
                "to": {"type": "STRING", "description": "Email recipient, for action=email."},
                "subject": {"type": "STRING", "description": "Email subject, for action=email."},
                "body": {"type": "STRING", "description": "Email body, for action=email."},
            },
            "required": ["action", "platform", "contact", "message"],
        },
    },
    {
        "name": "social_media",
        "description": "Real-account social automation via a REAL, visible, narrated browser logged into the user's OWN TikTok/Instagram — separate from web_automation's general narrated browsing, and opt-in (the user explicitly accepted the ToS/ban risk of this integration). 'tiktok_upload' loads a video into TikTok Studio with caption and hashtags filled in and leaves the final Post click to the user — never claim it was published. 'instagram_check_dms' reads recent Instagram conversations (read-only). 'instagram_draft_reply' composes a reply WITHOUT sending it (contact, message) and returns a reply_ref. 'instagram_reply' actually sends a previously drafted reply and ONLY works with confirm=true, set only after the user explicitly approves sending in this same turn — never claim a reply was sent otherwise.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":     {"type": "STRING", "description": "tiktok_upload, instagram_check_dms, instagram_draft_reply, or instagram_reply."},
                "video_path": {"type": "STRING", "description": "Local path to the video file for tiktok_upload."},
                "caption":    {"type": "STRING", "description": "Caption text for tiktok_upload."},
                "hashtags":   {"type": "STRING", "description": "Space/comma-separated hashtags for tiktok_upload."},
                "contact":    {"type": "STRING", "description": "Instagram username/thread to read or reply to."},
                "message":    {"type": "STRING", "description": "Reply text for instagram_draft_reply."},
                "reply_ref":  {"type": "STRING", "description": "Draft reference (e.g. 'reply-1') for instagram_reply."},
                "confirm":    {"type": "BOOLEAN", "description": "Must be true to actually send — only after explicit user approval."},
                "limit":      {"type": "INTEGER", "description": "Message count for instagram_check_dms."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "gaming",
        "description": "Local gaming client control: 'index' lists installed Steam and Epic Games installations; 'launch' starts a game by AppID; 'status' checks update/download progress for installed titles.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "index, launch, or status."},
                "app_id":  {"type": "STRING", "description": "Steam/Epic AppID for 'launch'."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "entertainment",
        "description": "YouTube media discovery and analysis: 'summarise' extracts a transcript and summary from a YouTube video; 'channel' looks up a channel's priority/type; 'trending' fetches the trending chart for a region (default GB).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "summarise, channel, or trending."},
                "url":     {"type": "STRING", "description": "YouTube URL for 'summarise'."},
                "channel": {"type": "STRING", "description": "Channel name for 'channel'."},
                "region":  {"type": "STRING", "description": "Region code for 'trending' (default GB)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "job",
        "description": "Run long read-only work in the BACKGROUND so the conversation continues instead of waiting. Use this whenever a request would otherwise leave the user sitting in silence — deep research, a big competitor sweep, a long analysis — and say so, then carry on talking; ORION announces the result when it lands. Actions: 'run' (detach a tool; needs 'tool' and optional 'args'), 'list' (what is running and what finished), 'result' (collect a finished job's output), 'cancel'. Only read-only tools can be backgrounded — anything that drives the machine or changes state must run in the conversation where the user can see it.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "run, list, result, or cancel."},
                "tool":   {"type": "STRING", "description": "For 'run': the tool to execute in the background, e.g. 'research'."},
                "args":   {"type": "OBJECT", "description": "For 'run': the arguments to pass to that tool."},
                "label":  {"type": "STRING", "description": "For 'run': a short human label, e.g. 'competitor sweep'."},
                "job_id": {"type": "STRING", "description": "For 'result'/'cancel'/'list': the job id, or part of its tool name or label."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "forge",
        "description": "Dynamically create, test, and activate new Python tools on the fly — the Forge self-improvement engine. Plan → Code Gen → Contract check + sandbox test (with a diagnosis-driven repair loop that fixes whichever of the module or the test is actually at fault) → Dependency Install → Live Activation. Actions: 'forge' (single tool), 'batch' (multiple tools), 'session' (track progress), 'health' (which forged tools are quarantined or held back as uncallable, and why).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "forge, batch, session, or health."},
                "tool_name": {"type": "STRING", "description": "Name of the tool to forge (for 'forge')."},
                "tool_plan": {"type": "STRING", "description": "High-level spec of the tool's purpose (for 'forge'), e.g. 'Analyse sentiment from text using transformers'."},
                "tool_specs": {
                    "type": "ARRAY",
                    "description": "For 'batch': list of {'name': str, 'plan': str} dicts.",
                    "items": {"type": "OBJECT", "properties": {
                        "name": {"type": "STRING"},
                        "plan": {"type": "STRING"},
                    }},
                },
                "session_id": {"type": "STRING", "description": "Session ID to track (for 'session'). Omit to list all sessions."},
            },
            "required": ["action"],
        },
    },
    # ── Phase 3: Personal AI Operating System ─────────────────────────────────
    {
        "name": "skill",
        "description": "Installable skill packages (workflows, prompts, templates — data only, never code). Actions: list, describe (name), install (source path to a skill folder or skill.json), remove (name), enable/disable (name), reload.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, describe, install, remove, enable, disable, or reload."},
                "name":   {"type": "STRING", "description": "Skill name (for describe/remove/enable/disable)."},
                "source": {"type": "STRING", "description": "Path to the skill folder or skill.json (for install)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "plugin",
        "description": "Installable CODE plugins that add new tools to ORION at runtime (contrast 'skill', which is data-only). Actions: list; describe (name); enable/disable (name) — persists across restarts; create (name + description [+ tier allow/confirm/forbid] [+ kind: 'tool' (default) or 'event' for a REACTIVE plugin that also gets on_event() and an events declaration]) scaffolds a working plugin ready to edit; install (source path to a *_tool.py OR a .orionplugin bundle); export (name [+ source=destination path]) packages a plugin into a single portable .orionplugin bundle anyone can install; remove (name); reload hot-reloads every enabled plugin with no restart; deps (name) installs a plugin's missing Python packages; backfill writes manifests for plugins that lack one so they gain a capability tier; audit (name optional) reports each plugin's DECLARED capabilities (network/filesystem/subprocess/input/screen/audio) and flags any the code actually uses but never declared — static analysis, nothing is imported, so it is safe on untrusted plugins; update (name + version) compares an installed plugin against a candidate version; unmute (name optional) revives a plugin whose hooks were muted for faulting or blocking; hooks lists which plugins are subscribed to which ORION events (a plugin can declare 'events' in its manifest and export on_event() to REACT to things like going offline or a security alert, not just be called); doctor reports the health of the whole plugin surface.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, describe, enable, disable, create, install, remove, reload, deps, backfill, or doctor."},
                "name":   {"type": "STRING", "description": "Plugin name (for describe/enable/disable/create/remove/deps)."},
                "description": {"type": "STRING", "description": "What the plugin should do (for create)."},
                "tier":   {"type": "STRING", "description": "Capability tier for create: allow, confirm (default) or forbid."},
                "source": {"type": "STRING", "description": "Path to a *_tool.py plugin module (for install)."},
                "version": {"type": "STRING", "description": "Candidate version for 'update' (e.g. 1.2.0)."},
                "kind": {"type": "STRING", "description": "For 'create': tool (default) or event/listener for a reactive plugin."},
                "events": {"type": "STRING", "description": "For 'create' with kind=event: comma-separated ORION events to subscribe to (state, speaking, connection_state, safety_alert, banner, log, paused, emotion_changed, telemetry_sample, dashboard_event)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "workflow",
        "description": "Repeatable automated processes: named chains of tool calls that run in the background with retries and progress tracking. Actions: list (the library), run (name + optional input text), status, cancel, define (name + steps as JSON list of {tool, args}), remove (name). Built-ins include research_workflow, creator_review_workflow, product_analysis_workflow and daily_review_workflow.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "list, run, status, cancel, define, or remove."},
                "name":   {"type": "STRING", "description": "Workflow name."},
                "input":  {"type": "STRING", "description": "Input text passed to {input} placeholders (for run)."},
                "steps":  {"type": "STRING", "description": "JSON list of {tool, args, retries} steps (for define)."},
                "description": {"type": "STRING", "description": "What the workflow does (for define)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "creator_intel",
        "description": "Creator Studio intelligence suite for short-form content (TikTok, Reels, Shorts): 'hook' scores an opening hook 0-10; 'review_script' scores a full creator script (hook/structure/CTA + priority fix) and logs it against the creator; 'performance' reports creator score trends; 'add_creator' adds to the roster; 'viral' finds recurring patterns across example scripts; 'hook_ideas' generates hooks for a topic; 'product' analyses a product for short-form commerce angles; 'strategy' gives a platform/cadence plan; 'creator_fit' assesses a creator profile for brand fit; 'cta' scores a call-to-action and offers stronger variants; 'competitor' tears down competitor content (hooks, CTAs, emotional triggers, unused angles); 'pipeline' runs the FULL product intelligence pipeline on a product (+optional competitor content and script) — analysis, improved hooks/CTAs, creator brief, testing and scaling plan, stored to memory and trend history; 'trends' reports patterns tracked over time.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "hook, review_script, performance, add_creator, viral, hook_ideas, product, strategy, creator_fit, cta, competitor, pipeline, or trends."},
                "script":  {"type": "STRING", "description": "The script/hook/examples text."},
                "creator": {"type": "STRING", "description": "Creator name (for review_script/performance/add_creator)."},
                "topic":   {"type": "STRING", "description": "Topic for hook_ideas."},
                "product": {"type": "STRING", "description": "Product name/URL/description (for product/pipeline/cta)."},
                "profile": {"type": "STRING", "description": "Creator profile facts (for creator_fit)."},
                "cta":     {"type": "STRING", "description": "The call-to-action text (for cta)."},
                "competitor": {"type": "STRING", "description": "Competitor script/caption/page copy (for competitor/pipeline)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "mission",
        "description": "ORION's mission-based operating model — the user's long-running endeavours (Build Demo Game, Develop ORION, Creator Studio, Neuroscience Research, University Studies, plus any created): 'overview' shows the mission board with progress; 'status' reports one mission in depth (outstanding tasks, goals, linked research/files, risks, recommended next move); 'current' switches the current mission focus; 'create' adds a mission; 'goal'/'task' add a goal or task to a mission; 'complete' marks a matching task/goal done; 'attach' links a file, research topic, note, report, workflow or agent to a mission; 'remove' deletes an item from a mission (pass its text) or the whole mission (omit the text); 'archive' takes a mission off the board without destroying it; 'clear_completed' sweeps finished tasks and goals away. Removal takes effect immediately — no restart.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":  {"type": "STRING", "description": "overview, status, current, create, goal, task, complete, attach, remove, delete, archive, or clear_completed."},
                "mission": {"type": "STRING", "description": "Mission name (partial match accepted)."},
                "text":    {"type": "STRING", "description": "The goal/task/item text or attachment value."},
                "brief":   {"type": "STRING", "description": "Mission brief (for create)."},
                "due":     {"type": "STRING", "description": "Optional ISO due date (for task)."},
                "kind":    {"type": "STRING", "description": "Attachment kind: file, research, note, report, workflow, or agent."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "avatar",
        "description": "The avatar system (state machine + face tracking): 'status' reports the avatar state and tracking; 'state' requests a behaviour state (idle, listening, thinking, speaking, researching, executing); 'notify' fires a notification pulse; 'warn' raises the warning state; 'track' enables webcam face tracking so the avatar subtly follows the user and keeps eye contact; 'untrack' disables it; 'show_camera' opens a small live view of what the camera is seeing (use when the user asks to see what you're looking at, or to check tracking accuracy); 'hide_camera' closes it.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status, state, notify, warn, track, untrack, show_camera, or hide_camera."},
                "state":  {"type": "STRING", "description": "Requested behaviour state (for state)."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "briefing",
        "description": "Dynamic adaptive briefing from live system activity (distinct from the classic morning_briefing news brief): 'brief' composes a briefing for the current time of day (or pass morning/midday/evening) including what happened while the user was away (research completed, workflows finished), progress since the last brief and deadline watch; specialist briefings: 'research' (agenda + evidence + opportunities), 'mission' (mission board, risks, next move), 'security' (sentinel + security events), 'opportunity' (research gaps + idle missions); 'check' asks whether anything needs attention right now.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "brief, morning, midday, evening, research, mission, security, opportunity, or check."},
                "period": {"type": "STRING", "description": "Override period for brief."},
            },
        },
    },
    {
        "name": "find_tool",
        "description": "Search ALL of ORION's tools (built-in, MCP services, forged tools, plugins) for the right one for a need, and get its arguments. Use it whenever no direct tool fits — chess, study, finance, security, marketing, Notion, diagnostics, plugins and more — before ever saying you cannot do something.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "need": {"type": "STRING", "description": "What you need to do, in plain words (e.g. 'analyse my chess game', 'check token usage')."},
            },
            "required": ["need"],
        },
    },
    {
        "name": "use_tool",
        "description": "Run any ORION tool by its exact name (from find_tool) with its arguments. All the usual confirmations and safety checks apply.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING", "description": "The tool's exact name, as find_tool returned it."},
                "arguments": {"type": "STRING", "description": "The tool's arguments as a JSON object, e.g. {\"action\": \"status\"}."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "capabilities",
        "description": "The system registries: what ORION can do. Without a query, summarises capability/module/feature counts; with a query, searches tool capabilities by name or description. Check here before building anything new.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Optional capability search term."},
            },
        },
    },
    {
        "name": "workflow_patterns",
        "description": "Tool sequences the user keeps repeating, which may be worth saving as a named workflow. Reports only — to actually create one, use the workflow tool with the user's agreement. Argument values are deliberately not retained, so real step arguments must be supplied when defining the workflow.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "min_occurrences": {"type": "INTEGER", "description": "How many repeats before a sequence counts (default 3)."},
            },
        },
    },
]
