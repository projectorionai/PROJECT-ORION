# What ORION can and cannot do offline

Written 29 Oct 2026 for the companion app 3.17 and the PC's home Wi-Fi link.

There are three different kinds of "offline". They are not the same:

| Situation | What it means |
|---|---|
| **PC has no internet** | ORION on the PC cannot reach Gemini or the web. The phone can still reach the PC on the home Wi-Fi if the router is on. |
| **Phone has no internet** | No Wi-Fi with internet and no mobile data. The phone's hosted AI (Gemini or the provider chosen in Options) cannot be reached. |
| **PC unreachable from the phone** | The PC is off, asleep, or on another network, and no Tailscale or tunnel route works. The phone keeps working on its own. |

The home Wi-Fi link needs only the router, not the internet. A phone on the same
Wi-Fi as the PC can use Desktop mode, PC controls and the PC's face even while
the internet connection is down. The PC then answers with its local model.

## On the PC

### Does not work without internet

| Feature | Why |
|---|---|
| Gemini Live voice conversation | Needs Gemini Live (Google's servers). |
| Cloud text models (Gemini, Groq, OpenRouter and the other router providers) | Need their APIs. |
| Charon cloud voice | Needs Gemini text-to-speech. Speech cached earlier still plays. |
| Web search, live and deep research, news, reading web pages | Need the web. |
| Fresh weather, prices and markets, flights, sport scores | Need live web data. |
| E-mail and Gmail, calendar sync, Telegram, phone calls through the telephony gateway, social posting | Need those services. |
| Installing software, self-update, downloading models, voices or plugins | Need downloads. |
| Forge and research agents that use a cloud model | Need the cloud model. With a local model configured they use that instead. |
| Online map imagery on the Globe and in research | Needs map tile servers. |
| The phone reaching the PC away from home (Tailscale Serve, Funnel or the quick tunnel) | Needs internet at both ends. |

### Still works without internet

- The Screen Face, with every expression and emotion.
- Conversation through the local model when its server is running (`--offline`, or the automatic fallback), plus the rule-based local commands with no model at all.
- Local speech: Kokoro, Piper or Windows SAPI voices. Local transcription with an installed Whisper or Vosk model.
- Memory, notes, reminders and timers, time and date, arithmetic.
- Local files, applications, window and system controls (with the usual confirmations).
- Camera, OCR and QR/barcode reading on the PC.
- Document, workbook and data tools that work on supplied files.
- The robot link and other local or LAN devices.
- **The phone on the home Wi-Fi**: Desktop mode, PC controls, PC voice transcription for the phone and the PC's face in the app, through the pinned HTTPS link.

See also [Running ORION offline](OFFLINE_OPERATION_2026-10-20.md).

## On the phone

### Does not work with no internet on the phone

| Feature | Why | What happens instead |
|---|---|---|
| General questions and AI-planned phone actions ("text Mum I'm late") | Need the hosted AI (Gemini or the chosen provider). | The request is saved (up to 5, for 30 minutes, in memory only) and sent in order when the connection returns. Actions in it still ask for the normal review slide. Time, date and battery are answered on the phone at once. |
| Mic transcription through Gemini | Needs Gemini. | The Mic button hands over to Android's own speech recogniser with its on-device model. This works only if an offline speech language is installed on the phone (Android settings). Typing always works. |
| Background listening (always-on, hands-free) | Its transcription needs Gemini. | Use the Mic button or type. |
| Charon voice (Live or Direct TTS) | Needs Gemini speech. | Replies are read by the phone's own voice. |
| Phone MCP tools on a hosted service | Need that service. | None. |
| Downloading offline map packs or BRouter road regions | Need a download (Wi-Fi). | Maps, places and car routes already downloaded keep working. |
| Gmail sending through Accessibility | Gmail needs its connection. | Drafts can still be prepared on the phone. |

### Does not work when the PC is unreachable

| Feature | Why |
|---|---|
| Desktop mode (the PC's own ORION page) | It is served by the PC uplink. |
| PC tools, desktop memory, PC power controls | They run on the PC. |
| The face following the PC's state and emotions | It is the PC's live event stream. The face follows the phone instead. |
| First-time key provisioning from the PC | It is a request to the PC. A key can be typed on the phone instead. |
| PC-side voice transcription for the phone and a phone MCP hosted on the PC | They run on the PC. |
| Reaching the PC away from home | Needs Tailscale or a tunnel. On the home Wi-Fi no Tailscale is needed. |

### Still works on the phone with no internet and no PC

- ORION's face: the PC's Screen Face is bundled in the app. Its design and colours are the ones the PC last sent, and it shows the phone's own listening, thinking and speaking. It reads the feeling in what you type and in its answers with the PC's own sentiment rules. It dims to the offline face when the phone has no connection.
- A clear status: Options shows whether the PC is linked, whether there is internet, and how many requests are waiting.
- Time, date and battery answers.
- Saved requests, sent when the connection returns.
- Phone controls opened by hand (calls and SMS use the mobile network, not the internet), the share sheet, alarms and settings.
- The offline UK map, saved places, BRouter car routes and spoken turn guidance from downloaded data (see [offline car routes](OFFLINE_CAR_ROUTES_2026-09-30.md)).
- The encrypted conversation history on the phone.
- The Sentinel overlay, screenshots and screen-sharing consent. Sending the shared screen needs the PC.
