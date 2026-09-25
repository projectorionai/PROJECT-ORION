# ORION MCP host — connecting Gmail, Google Calendar and more

ORION is now an **MCP host**: it can connect to Model Context Protocol servers
(small programs that expose tools like your email, calendar or filesystem) and
call their tools on your behalf. This is how ORION reads your Gmail and manages
your Google Calendar.

Servers are declared in **`config/mcp_servers.json`** (a template is written on
first run). Each entry is a command ORION launches. When ORION starts, it
discovers tools for enabled servers; a server with a cached tool list can stay
paused until its first call. Its tools remain available through the `mcp` tool.

---

## Prerequisites

Most ready-made MCP servers (Gmail, Google Calendar, filesystem) are distributed
as Node packages run with `npx`. Install **Node.js 18+** from <https://nodejs.org>
and confirm it's on your PATH:

```
node --version
npx --version
```

(If you prefer not to use Node, any MCP server reachable by a command works —
including Python ones run with `python -m ...`.)

---

## Enabling a server

Edit `config/mcp_servers.json`, set `"enabled": true` on the server you want, and
restart ORION. Example (filesystem — no credentials needed, good first test):

```json
"filesystem": {
  "enabled": true,
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\you\\Documents"],
  "env": {}
}
```

After restart, ask ORION: *"List your MCP tools"* — he'll call `mcp` with
`action=list` and read them back.

---

## Gmail + Google Calendar (needs your Google OAuth)

These access your private data, so **you** create the Google credentials — ORION
(and I) never handle your password or tokens directly.

1. **Google Cloud Console** → create a project (or reuse one):
   <https://console.cloud.google.com/>
2. **APIs & Services → Enable APIs**: enable **Gmail API** and **Google
   Calendar API**.
3. **APIs & Services → OAuth consent screen**: configure it (External is fine
   for personal use), and add your own Google account under **Test users**.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID →
   Desktop app**. Download the JSON — this is your OAuth client secret file.
5. Point the server at that file. For the calendar entry, set the path in `env`:

```json
"google_calendar": {
  "enabled": true,
  "command": "npx",
  "args": ["-y", "@cocal/google-calendar-mcp"],
  "env": { "GOOGLE_OAUTH_CREDENTIALS": "C:\\Users\\you\\.config\\gcp-oauth.keys.json" }
}
```

```json
"gmail": {
  "enabled": true,
  "command": "npx",
  "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
  "env": {}
}
```

6. **First run does a one-time browser sign-in.** The Gmail/Calendar servers
   open a Google consent page the first time; approve it with your account. The
   token is cached by the server for future runs. (The exact auth step differs
   per server — check that package's README; both above self-open the consent
   flow.)

Once enabled and authorised, ORION links the two: he can read a booking email in
Gmail and add it to your Google Calendar, or check your day and draft replies.

> Package names above are popular community servers and may change. Any
> Gmail/Calendar MCP server works — just set its `command`, `args` and `env`.
> ORION allows the shipped servers' known read-only tools without a prompt.
> Sending email, changing mail or calendar data, and newly added tools require
> on-screen confirmation before ORION runs them.

---

## Telephony — letting ORION place real phone calls & SMS

ORION has **two** ways to reach a phone:

1. **Through your paired Android phone** (`phone_action`, kind=`call`/`sms`) — the
   ORION app opens the dialer/messages **pre-filled** and *you tap to send*.
   Nothing dials automatically. No extra setup, no cost, always safe.
2. **Directly, over the network** via the **`twilio` MCP server** — ORION places
   the call or sends the SMS *itself*, to any number, without your phone. This is
   what to enable if you want ORION to actually call people on your behalf.

### Enabling the Twilio server

1. Create a [Twilio](https://www.twilio.com/) account and buy a phone number.
2. Create an **API key** (Console → Account → API keys & tokens) — note the
   **SID**, **Key** and **Secret**.
3. In `config/mcp_servers.json`, under `servers.twilio`, fill:
   - `TWILIO_ACCOUNT_SID`, `TWILIO_API_KEY`, `TWILIO_API_SECRET`
   - `TWILIO_FROM_NUMBER` (your Twilio number, E.164 e.g. `+441234567890`)
   then set `"enabled": true`.
4. The `command`/`args` shown (`npx -y @twilio-alpha/mcp`) are Twilio's community
   MCP; package name and exact args can change — check the server's current README
   if it fails to launch. Any stdio telephony MCP works in the same shape.

Once enabled, its tools appear as `mcp__twilio__…` and ORION can call them via the
`mcp` tool.

> ⚠️ **Real calls dial immediately and cost money.** Unlike `phone_action`, there is
> no tap-to-confirm on the phone. Treat every `mcp__twilio__` call/SMS as an
> outward action: confirm the number and intent before placing it, and keep the
> server disabled unless you actively want autonomous calling. Credentials live in
> `config/`, never in source.

---

## How it works internally

- `orion_core/mcp_host.py` is a self-contained MCP client (JSON-RPC 2.0 over
  stdio) — no extra Python dependency.
- On startup `app.py` runs `mcp_host.connect_all()` in the background, so a slow
  or missing server never delays launch.
- The `mcp` dispatcher tool exposes `action=list` (browse servers/tools) and
  `action=call` (run a tool with `server`, `tool`, `arguments`).
- A server that fails to launch or handshake is logged and skipped — ORION
  always starts.

---

## Servers added 2026-09-20 (all shipped disabled)

Three more entries in the template. They are present and off rather than
documented-only, because the template is how anyone discovers what ORION can be
connected to — a line in a document is not a thing you can switch on.

### Brave Search — `brave_search`

A web index ORION can query directly instead of driving a browser, which is
slower and more fragile. There is a free tier (2,000 queries a month at the time
of writing).

1. Get an API key from <https://brave.com/search/api/>.
2. Put it in `BRAVE_API_KEY` under the `brave_search` entry in
   `config/mcp_servers.json`.
3. Set `"enabled": true`.

### GitHub — `github`

*(Replaced 2026-09-25: the npm `@modelcontextprotocol/server-github` is
deprecated.)* GitHub's own server, `github/github-mcp-server`, ships as a
release binary. Install it in your own ORION MCP directory and point the local
`config/mcp_servers.json` entry there. The installation reviewed for this guide
used v1.12.2 with its SHA-256 checked against the release. Toolsets: `context, repos,
issues, pull_requests, actions`.

1. Create a **fine-grained** personal access token at
   <https://github.com/settings/personal-access-tokens/new>, limited to the
   repositories you want ORION to see. Permissions: Metadata (read), Contents,
   Issues, Pull requests, Actions (read, or read/write for anything ORION
   should change).
2. Put it in `GITHUB_PERSONAL_ACCESS_TOKEN`, set `"enabled": true`.

A classic token with full scopes hands ORION every repository you can see,
which is a great deal more than this needs. Fine-grained, least privilege.
If `GITHUB_TOKEN` or `GITHUB_PERSONAL_ACCESS_TOKEN` is set in the environment,
ORION fills the entry from it, so make sure that one is fine-grained too.
Merging, pushing or deleting files, creating or forking repos and triggering
workflows ask on screen first (`confirm_tools`).

### Home Assistant — `home_assistant`

*(Replaced 2026-09-25: this now uses Home Assistant's own MCP server instead
of the third-party `homeassistant-mcp` npm package.)*

1. In Home Assistant: **Settings → Devices & services → Add integration →
   Model Context Protocol Server**. Choose which entities it exposes (Settings →
   Voice assistants → Expose).
2. Profile → Security → **Long-lived access tokens** → create one. Put it in
   `API_ACCESS_TOKEN`.
3. Change the last `args` entry if your instance is not at
   `http://homeassistant.local:8123` (keep `/api/mcp` on the end).
4. Set `"enabled": true`.

The server speaks Streamable HTTP, and ORION's host speaks stdio, so
`mcp-proxy` bridges the two, as Home Assistant's docs recommend. It is held
below `mcp` 2.x because 0.12.0 fails to import against it. It runs at
`--log-level WARNING` because at INFO it logs every request's headers,
including the bearer token.

**That token can do anything you can**, including unlocking whatever your Home
Assistant unlocks. So the entry is `"confirm": true`: every tool asks on screen
first, except the read-only ones in `confirm_except` (`GetLiveContext`,
`GetDateTime`, and so on).

If you would rather not run a bridge for this, the
`homeassistant_call_service` plugin covers the same ground over plain REST with
no extra dependency — see `docs/PLUGINS.md`.

## Telephony on Windows

`twilio` places **real** calls and sends **real** SMS. They connect immediately
and they cost money, so every call and SMS tool is treated as a confirmed
action — ORION asks before dialling, and `CallAuthority` is consulted
regardless of how the request reached him.

ORION's own telephony server (`orion_core/telephony_server.py`) is the other
half: it answers inbound calls and bridges the audio. On Windows it runs as an
ordinary process rather than a systemd unit — `python -m orion_core.telephony_server`
— and Twilio needs a public HTTPS webhook pointing at it. Either expose the
port through a tunnel, or run the server on the VPS (`deploy/` has the units)
and point the webhook there. `X-Twilio-Signature` is verified on every request,
so a webhook URL that does not match the one Twilio was configured with will be
rejected rather than answered.

## The rest of the free integrations (2026-09-20)

All shipped disabled in `config/mcp_servers.json`. Set `"enabled": true`, fill
in whatever credentials the entry names, then `mcp reconnect <server>`.

| server | what it gives ORION | cost | credential |
|---|---|---|---|
| `duckduckgo` | web search | free | **none** |
| `brave_search` | better-ranked web search | free tier | API key |
| `github` | repos, issues, PRs | free | fine-grained PAT |
| `obsidian` | your notes vault | free | none (vault path) |
| `sqlite` | query a local `.db` | free | none (file path) |
| `postgres` | query a database | free | database, user, password |
| `matrix` | messaging | free | access token |
| `calcom` | bookings others make with you | free tier | API key |
| `home_assistant` | lights, climate, automations | free | long-lived token |
| `twilio` | **real** calls and SMS | pay per use | account SID + API key |

Three of those deserve a word of warning rather than a row in a table.

**`obsidian`** gives ORION read and write access to the entire vault. For most
people that is years of private notes, including things written precisely
because nobody else was going to read them. A vault ORION should not read is
one to leave out of the config.

**`postgres`** — point it at a **read-only role**. A model composing SQL
against a writable production database is a bad afternoon waiting to happen,
and the MCP server cannot prevent it; the database's own permissions can.

**`twilio`** costs real money and connects immediately. It is marked
`"confirm": true`, so `mcp` refuses its call and SMS tools unless the model
passes `confirm=true` — and ORION's own `telephony` tool requires the same.
See below.

## Servers added 2026-09-25

Every entry was tested through ORION's own `MCPServerConn` (a real handshake
plus a real read-only call where no credential was needed). The tool schemas
were also checked to build a valid Gemini `LiveConnectConfig`. MCP tools are
not in the Live core set; the voice session reaches them through
`find_tool`/`use_tool`.

| server | package | on? | what you supply |
|---|---|---|---|
| `filesystem` | `@modelcontextprotocol/server-filesystem` | **on** | nothing. Folders: Desktop, Documents, Downloads and the OneDrive Desktop (never ORION's config) |
| `playwright` | `@playwright/mcp` (Microsoft) | **on** | nothing. Headless, `--isolated` (no saved logins) |
| `memory` | `@modelcontextprotocol/server-memory` | **on** | nothing. Graph in `ORION/config/mcp_memory.jsonl` |
| `sqlite` | `@toolbox-sdk/server --prebuilt=sqlite` (Google MCP Toolbox) | **on** | nothing. Its own `ORION/config/mcp_sqlite.db` |
| `git` | `mcp-server-git` (reference server) | **on** | nothing. Scoped to the ORION Main repository |
| `context7` | `@upstash/context7-mcp` | **on** | nothing. Optional key for higher limits |
| `markitdown` | `markitdown-mcp` (Microsoft) | **on** | nothing |
| `github` | `github-mcp-server` release binary | off | fine-grained PAT (see above) |
| `home_assistant` | Home Assistant's `/api/mcp` via `mcp-proxy` | off | integration + token + URL (see above) |
| `postgres` | `@toolbox-sdk/server --prebuilt=postgres/data` | off | `POSTGRES_DATABASE`, `POSTGRES_USER`, `POSTGRES_PASSWORD` (read-only role) |
| `google_drive` | `workspace-mcp` 1.29.0, `--tools drive --read-only` | off | Google OAuth desktop client ID + secret |

**Google Drive setup.** Google's own Drive MCP server is remote-only and
limited to the Workspace Developer Preview, so the entry uses `workspace-mcp`
instead. It runs locally and works with a personal account.

1. <https://console.cloud.google.com/>: pick a project and enable the
   **Google Drive API**.
2. Set up the OAuth consent screen (External, and add yourself as a test user).
3. Create an **OAuth client ID** of type **Desktop app**.
4. Put its ID and secret in `GOOGLE_OAUTH_CLIENT_ID` and
   `GOOGLE_OAUTH_CLIENT_SECRET`, then set `"enabled": true`.
5. The first Drive request returns a Google sign-in link. The token is cached
   in `ORION/config/mcp_google_drive/`.

**Asking before acting.** `"confirm": true` still gates every tool on a server.
Two new fields make it finer:

- `"confirm_tools": [...]` gates only the tools listed. Exact names or
  wildcards, case-insensitive.
- `"confirm_except": [...]` exempts tools from a `"confirm": true`.

`"confirm_reason"` is the sentence the on-screen prompt shows. Shipped gates:

- **filesystem:** `write_file`, `edit_file` and `move_file` (reading,
  searching and creating folders do not ask).
- **git:** staging, commit, reset, checkout and branch creation.
- **github:** merge, pushing/creating/deleting files, creating or forking
  repos, updating a PR branch, and triggering workflows.
- **playwright:** the RCE-equivalent `browser_run_code_unsafe`, plus the two
  tools that hand a local file to a web page.
- **home_assistant:** everything except the state readers.

A money server (`twilio`) cannot be exempted by `confirm_except`.

**Memory cost.** Each stdio server is a small process tree. Measured, all ten
enabled servers together use about 0.9–1.0 GB of private memory. The largest
are `markitdown` at about 165 MB and `playwright` at about 100–140 MB before it
opens Chrome. Disable what you do not use with `mcp disable <server>`.

**Idle servers are paused.** A server nobody has called for 30 minutes is
stopped and its tool list kept, so the model still sees its tools. The next
call restarts it first (measured: `memory` freed 176 MB and came back in
1.5 s). When the resource governor reports memory pressure (85% used by
default), the limit drops to 2 minutes and idle servers are paused at once.
`ORION_MCP_IDLE_PARK_S` sets the everyday limit in seconds; `0` keeps every
server running.

## Ringing your phone

ORION can place a real call and speak to you. Three things stand between a
sentence and your phone ringing, and all three are deliberate:

1. **The number must already be listed.** Copy
   `config/telephony_contacts.example.json` to `telephony_contacts.json` and
   put your number in it. Default-deny: a number ORION heard in conversation,
   or read off a web page, cannot be dialled. This is the control that matters
   most, because it is the one that cannot be talked around.
2. **The message is screened**, and screening fails **closed** — if the
   spillage guard cannot check the text, nothing is said. A phone call cannot
   be un-sent, is often recorded at the far end, and may be heard by somebody
   who is not you.
3. **Confirmation.** `confirm=true` is required, so ORION says who he is about
   to ring and what he will say before he does it.

Setup:

```jsonc
// config/mcp_servers.json
"twilio": {
  "enabled": true,
  "env": {
    "TWILIO_ACCOUNT_SID": "AC...",
    "TWILIO_API_KEY": "SK...",
    "TWILIO_API_SECRET": "...",
    "TWILIO_FROM_NUMBER": "+44..."
  }
}
```

```jsonc
// config/telephony_contacts.json
{"contacts": [{"name": "me", "number": "07700 900123"}]}
```

Then `telephony action=status` reports whether the server is connected and
which numbers he is allowed to ring.

A note on what this is *not*: `phone_action` hands a pre-filled dialler to your
own paired handset for you to tap. It costs nothing and sends nothing. The
`telephony` tool dials from a Twilio number and bills you. They are easy to
confuse, which is why the tool descriptions say so explicitly.
