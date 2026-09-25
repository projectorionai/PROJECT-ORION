"""
mcp_host.py — ORION as a Model Context Protocol (MCP) host.

The user asked for ORION to "use all relevant MCPs" and to link his calendar to
Gmail through one.  MCP (modelcontextprotocol.io) is the open standard for
exposing tools/data to an assistant; a *host* connects to MCP *servers* (small
programs — Gmail, Google Calendar, filesystem, web search, …) and makes their
tools callable.  This module makes ORION that host.

It is a self-contained MCP client — a minimal JSON-RPC 2.0 implementation over
the stdio transport (newline-delimited messages), so it adds NO new Python
dependency and cannot break the rest of ORION if the ``mcp`` SDK is absent.
Servers are declared in ``config/mcp_servers.json`` (a template is written on
first run); each is a command ORION launches as a subprocess.  Nothing here
ever raises to the caller — a missing command, a bad handshake or a server
crash is logged and that server is simply skipped, so the app always starts.

Model integration (widened in the Mark XXI pass, Track E1): each connected
server's tools are ALSO registered as first-class, namespaced dispatcher
tools ("mcp__<server>__<tool>", see OrionDispatcher.register_mcp_tools) —
callable directly, the same as any native ORION tool, not only through the
generic ``mcp`` list/call indirection (which still exists, for browsing).
The generic tool remains useful for discovery ("what can this server do")
and as a fallback if a namespaced call is ever mistyped.

Reconnection (Track E2): a dropped or crashed server no longer stays gone
until the next full app restart — MCPHost.supervise() watches every
enabled server and reconnects with bounded backoff, re-registering its
tools the moment it's back; per-server health is reported through the
attached Telemetry, when one is given.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import shutil
import time
from typing import Any

from .constants import BASE_DIR, CONFIG_DIR
from .atomic_io import atomic_write_text

MCP_CONFIG_PATH = CONFIG_DIR / "mcp_servers.json"
#: Each server's tool list from its last successful start, so the next boot
#: can offer the tools without launching the process (see _cached_tools).
MCP_CATALOGUE_PATH = CONFIG_DIR / "mcp_catalogue.json"
_PROTOCOL_VERSION = "2024-11-05"
_CALL_TIMEOUT = 45.0
#: Long enough for the FIRST launch, when npx/uvx download the server package
#: before it can answer. 25 s failed every first run on a home connection and
#: the server was then reported as broken rather than as still installing.
_HANDSHAKE_TIMEOUT = 120.0
#: A server nobody has called for this long is paused (its process stopped,
#: its tool list kept) and restarts on its next call. Ten servers held about
#: a gigabyte between them while doing nothing. ORION_MCP_IDLE_PARK_S=0 keeps
#: every server running.
_IDLE_PARK_S = 1800.0

#: Bumped when migrate_config learns a new fix; recorded in the config so a
#: server the user later switches off is not switched back on.
CONFIG_REVISION = "mark31"

#: Env names that hold a credential. An enabled server with one of these empty
#: cannot work, so it is reported as waiting for that key — not launched,
#: failed, retried and reported as broken every thirty seconds.
_SECRET_RE = __import__("re").compile(
    r"(?i)(key|token|secret|sid|password|credential|auth)")

# Per-tool confirmation lists for the shipped entries (see requires_confirmation).
# Exact tool names, so a tool a server adds later is judged when it is added
# rather than swept in or out by a pattern.
#: mcp-server-git's tools that only read. The server is "confirm": true with
#: these excepted rather than listing the writes: its tool set is small and
#: stable, a tool it adds later asks until judged, and an ORION build that
#: predates confirm_except still gates every git tool instead of none.
_GIT_READ_TOOLS = ("git_status", "git_diff_unstaged", "git_diff_staged", "git_diff",
                   "git_log", "git_show", "git_branch")
#: github-mcp-server: landing, deleting or publishing code, and running
#: workflows. Issues, comments and pull-request text can be edited afterwards.
_GITHUB_CONFIRM_TOOLS = ("merge_pull_request", "push_files", "create_or_update_file",
                         "delete_file", "create_repository", "fork_repository",
                         "update_pull_request_branch", "actions_run_trigger")
#: @playwright/mcp: the RCE-equivalent code runner, and the two tools that can
#: hand a local file to a web page (the way a prompt-injected page would
#: exfiltrate one).
_PLAYWRIGHT_CONFIRM_TOOLS = ("browser_run_code_unsafe", "browser_file_upload",
                             "browser_drop")
#: Home Assistant's Assist API tools that only read. Everything else it
#: exposes — every intent, every script — acts on the house, so it asks.
#: Filesystem tools that change a file. write_file replaces a whole file with
#: no undo, and the allowed folders can include ORION's own source.
_FILESYSTEM_CONFIRM_TOOLS = ("write_file", "edit_file", "move_file")
_HOME_ASSISTANT_READ_TOOLS = ("GetLiveContext", "GetDateTime", "HassTimerStatus",
                              "todo_get_items", "calendar_get_events")


def _default_config() -> dict[str, Any]:
    """The starting configuration. The recommended servers (fetch, DuckDuckGo,
    Notion — see _RECOMMENDED) ship ON because they need nothing ORION does
    not already have; everything needing an account ships disabled with
    empty credentials. Gmail/Calendar need Google OAuth the user sets up
    themselves — see docs/MCP_SETUP.md."""
    template = _default_template()
    template["servers"].update(json.loads(json.dumps(_RECOMMENDED)))
    template["_revision"] = CONFIG_REVISION
    return template


def _default_template() -> dict[str, Any]:
    return {
        "_comment": (
            "ORION MCP servers. Set 'enabled': true and fill credentials to "
            "activate one. Node servers run through npx and Python ones through "
            "uvx (ORION finds both even when they are not on PATH). A server "
            "with an empty credential is shown as waiting for it rather than "
            "launched. See docs/MCP_SETUP.md for Gmail/Calendar OAuth."
        ),
        "servers": {
            "gmail": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
                "env": {},
                "description": "Gmail: read, search, send and manage email.",
            },
            "google_calendar": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@cocal/google-calendar-mcp"],
                "env": {"GOOGLE_OAUTH_CREDENTIALS": ""},
                "description": "Google Calendar: list, create and update events; links to your Gmail account.",
            },
            # Named folders, never the whole profile: the home directory holds
            # ORION's own config (and its secrets) and every app's data.
            "filesystem": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem",
                         *(os.path.join(os.path.expanduser("~"), folder)
                           for folder in ("Desktop", "Documents", "Downloads"))],
                "env": {},
                "confirm_tools": list(_FILESYSTEM_CONFIRM_TOOLS),
                "confirm_reason": "This changes or overwrites a file on your computer.",
                "description": ("Local filesystem: read, write and search files "
                                "in the listed folders only."),
            },
            # ── SEARCH: a web index ORION can query without scraping ────────
            # Brave's Search API has a free tier (2,000 queries a month at the
            # time of writing) and an official MCP server. It answers the
            # "look this up" half of research without ORION driving a browser,
            # which is slower and more fragile.
            # Setup: an API key from brave.com/search/api, then enabled:true.
            "brave_search": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@brave/brave-search-mcp-server"],
                "env": {"BRAVE_API_KEY": ""},
                "description": ("Web search: query Brave's index directly, "
                                "with a free tier, instead of driving a "
                                "browser."),
            },
            # ── GITHUB: repositories, issues, pull requests ──────────────────
            # GitHub's own server (github/github-mcp-server); the old npm
            # @modelcontextprotocol/server-github is deprecated. It ships as a
            # release binary, not a package. A fine-grained personal access
            # token scoped to the repositories you want is the right
            # credential — a classic token with full scopes hands ORION every
            # repository you can see. Changes that land code, delete it or
            # run workflows need an on-screen yes (confirm_tools).
            "github": {
                "enabled": False,
                "command": os.path.join(os.path.expanduser("~"), "ORION", "mcp",
                                        "github-mcp-server", "github-mcp-server.exe"),
                "args": ["stdio"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "",
                        "GITHUB_TOOLSETS": "context,repos,issues,pull_requests,actions"},
                "confirm_tools": list(_GITHUB_CONFIRM_TOOLS),
                "confirm_reason": ("changes code, branches or workflows on GitHub, "
                                   "under your name"),
                "description": ("GitHub: repositories, commits, issues, pull "
                                "requests and Actions. Binary from "
                                "github.com/github/github-mcp-server/releases."),
            },
            # ── HOME ASSISTANT: the house ────────────────────────────────────
            # Home Assistant's OWN MCP server (the "Model Context Protocol
            # Server" integration, at /api/mcp) — not a third-party package.
            # It speaks Streamable HTTP, so mcp-proxy bridges it to stdio, as
            # Home Assistant's docs recommend. mcp-proxy is held below mcp 2
            # (0.12.0 does not import against it) and kept at WARNING: at INFO
            # it logs every request's headers, bearer token included, into the
            # stderr tail a failed handshake quotes.
            #
            # API_ACCESS_TOKEN is a long-lived access token from your own
            # profile page, and it can do anything you can — including
            # unlocking whatever your Home Assistant unlocks. So every tool
            # asks first except the ones that only read state.
            "home_assistant": {
                "enabled": False,
                "confirm": True,
                "confirm_except": list(_HOME_ASSISTANT_READ_TOOLS),
                "confirm_reason": "controls a real device or service in your home",
                "command": "uvx",
                "args": ["--with", "mcp>=1.17,<2", "mcp-proxy@0.12.0",
                         "--log-level", "WARNING", "--transport=streamablehttp",
                         "--stateless", "http://homeassistant.local:8123/api/mcp"],
                "env": {"API_ACCESS_TOKEN": ""},
                "description": ("Home Assistant: read and control lights, "
                                "climate, media, timers and scripts. Needs the "
                                "MCP Server integration enabled in Home "
                                "Assistant; edit the URL in args."),
            },
            # ── SEARCH, without an account ───────────────────────────────────
            # DuckDuckGo needs no key at all, which makes it the one to try
            # first: an integration that requires signing up is an integration
            # that stays off. Brave above is better ranked and rate-limited by
            # a key; this is the zero-friction fallback.
            "duckduckgo": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "duckduckgo-mcp-server"],
                "env": {},
                "description": ("Web search with no API key and no account, "
                                "for when Brave's free tier runs out."),
            },
            # ── OBSIDIAN: the notes vault ────────────────────────────────────
            # Reads and writes the markdown files directly, so it works
            # whether or not Obsidian itself is running. Point it at the vault
            # FOLDER — and be aware this gives ORION read/write access to
            # everything in it, which for most people is years of private
            # notes. A vault ORION should not read is one to leave out.
            "obsidian": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "mcp-obsidian", "/path/to/your/vault"],
                "env": {},
                "description": ("Obsidian vault: search, read and write notes "
                                "as plain markdown files."),
            },
            # ── DATABASES: ask questions of your own data ────────────────────
            # Google's MCP Toolbox for Databases serves both (the reference
            # SQLite and Postgres servers are archived, and the Postgres one's
            # read-only guard could be escaped). SQLite gets a database of its
            # OWN — not one of ORION's stores, which a stray UPDATE would
            # corrupt while ORION holds them open.
            "sqlite": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@toolbox-sdk/server", "--prebuilt=sqlite",
                         "--stdio", "--disable-version-check"],
                "env": {"SQLITE_DATABASE": str(CONFIG_DIR / "mcp_sqlite.db")},
                "description": ("SQLite: a scratch database of ORION's own for "
                                "tables he builds — list tables, run SQL."),
            },
            # Use a READ-ONLY role: a model composing SQL against a writable
            # production database is a bad afternoon waiting to happen, and
            # execute_sql cannot tell a SELECT from a DROP; the role can.
            "postgres": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@toolbox-sdk/server", "--prebuilt=postgres/data",
                         "--stdio", "--disable-version-check",
                         "--defer-source-connect"],
                "env": {"POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432",
                        "POSTGRES_DATABASE": "", "POSTGRES_USER": "",
                        "POSTGRES_PASSWORD": ""},
                "description": ("PostgreSQL: list schemas and tables and run "
                                "SQL. Point it at a READ-ONLY role."),
            },
            # ── GIT: this repository, read freely, changed only on a yes ─────
            "git": {
                "enabled": False,
                "command": "uvx",
                "args": ["mcp-server-git", "--repository", str(BASE_DIR)],
                "env": {},
                "confirm": True,
                "confirm_except": list(_GIT_READ_TOOLS),
                "confirm_reason": "changes the repository (index, commits or branches)",
                "description": ("Git: status, diffs, log and history of the "
                                "ORION repository."),
            },
            # ── BROWSER AUTOMATION: operate websites ─────────────────────────
            # Microsoft's server. Headless, because the browser co-pilot is
            # ORION's visible browser; isolated, so it carries none of your
            # logins. The RCE-equivalent code runner and the two tools that
            # can hand a local file to a web page ask first.
            "playwright": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@playwright/mcp", "--headless", "--isolated",
                         "--browser", "chrome", "--image-responses", "omit",
                         "--idle-timeout", "300000", "--file-paths", "absolute",
                         "--output-dir", os.path.join(os.path.expanduser("~"), "ORION",
                                                      "mcp", "playwright-output"),
                         "--output-max-size", "52428800"],
                "env": {},
                "confirm_tools": list(_PLAYWRIGHT_CONFIRM_TOOLS),
                "confirm_reason": ("runs arbitrary code or hands a local file "
                                   "to a web page"),
                "description": ("Operate websites in a headless browser: "
                                "navigate, read the page, click, type, fill forms."),
            },
            # ── MEMORY: a knowledge graph of entities and relations ──────────
            "memory": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
                "env": {"MEMORY_FILE_PATH": str(CONFIG_DIR / "mcp_memory.jsonl")},
                "description": ("Knowledge graph: entities, relations and "
                                "observations kept across sessions."),
            },
            # ── GOOGLE DRIVE ─────────────────────────────────────────────────
            # Google's own Drive MCP server is remote-only and needs the
            # Workspace Developer Preview; workspace-mcp runs locally and works
            # with a personal account. Pinned, read-only, Drive tools only.
            "google_drive": {
                "enabled": False,
                "command": "uvx",
                "args": ["workspace-mcp@1.29.0", "--single-user", "--tools",
                         "drive", "--read-only"],
                "env": {"GOOGLE_OAUTH_CLIENT_ID": "", "GOOGLE_OAUTH_CLIENT_SECRET": "",
                        "OAUTHLIB_INSECURE_TRANSPORT": "1",
                        "WORKSPACE_MCP_CREDENTIALS_DIR": str(CONFIG_DIR / "mcp_google_drive")},
                "description": ("Google Drive: search, list and read your files "
                                "(read-only). Needs a Google OAuth desktop client."),
            },
            # ── DOCUMENTATION + CONVERSION ───────────────────────────────────
            "context7": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp"],
                "env": {},
                "description": ("Current, version-specific documentation and "
                                "code examples for libraries and frameworks."),
            },
            "markitdown": {
                "enabled": False,
                "command": "uvx",
                "args": ["markitdown-mcp"],
                "env": {},
                "description": ("Convert PDF, Word, PowerPoint, Excel, EPUB, "
                                "HTML and more to Markdown from a file or URL."),
            },
            # ── MATRIX: messaging on a server you choose ─────────────────────
            # Free homeservers exist and you can run your own, which is the
            # difference from the other chat integrations. The access token is
            # a full login — it can read every room the account is in.
            "matrix": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "matrix-mcp-server"],
                "env": {"MATRIX_HOMESERVER": "https://matrix.org",
                        "MATRIX_USER_ID": "@you:matrix.org",
                        "MATRIX_ACCESS_TOKEN": ""},
                "description": ("Matrix: read and send messages in rooms, on "
                                "any homeserver including your own."),
            },
            # ── CAL.COM: scheduling ──────────────────────────────────────────
            # Open-source scheduling with a free tier. Useful mostly if people
            # book time with you; if you only keep your own calendar, the
            # google_calendar entry above is the simpler thing.
            "calcom": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@calcom/mcp"],
                "env": {"CALCOM_API_KEY": ""},
                "description": ("Cal.com: list and manage bookings other "
                                "people have made with you."),
            },
            # ── TELEPHONY: let ORION place REAL outbound phone calls + SMS ──────
            # Twilio's MCP server. Unlike phone_action (which pre-fills the dialer
            # on your paired Android phone for you to tap), this DIALS DIRECTLY via
            # your Twilio account — real calls cost money and connect immediately,
            # so treat every call/SMS tool here as a CONFIRMED action.
            # Setup: create a Twilio account + an API key, then fill the three env
            # values and set enabled:true. Exact args can vary by package version —
            # see docs/MCP_SETUP.md (Telephony). Alternatives: any telephony MCP
            # that speaks stdio works here in the same shape.
            # NOTE: launching this needs npx, i.e. Node. A Windows machine
            # without Node has none, and then this server can never start
            # whatever the credentials say — which is exactly how outbound
            # calling appeared broken for months. ORION no longer depends on
            # it: telephony_direct.py reaches Twilio's REST API with nothing
            # but the credentials, and `python tools/setup_twilio.py` sets
            # those in one step. This server is still preferred when it IS
            # connected, because it does more than dial.
            "twilio": {
                "enabled": False,
                # Enforced, not merely described: mcp 'call' refuses tools on
                # a server marked this way unless confirm=true is passed.
                "confirm": True,
                "command": "npx",
                "args": ["-y", "@twilio-alpha/mcp"],
                "env": {
                    "TWILIO_ACCOUNT_SID": "",
                    "TWILIO_API_KEY": "",
                    "TWILIO_API_SECRET": "",
                    "TWILIO_FROM_NUMBER": "",
                },
                "description": ("Telephony: place REAL outbound phone calls and send "
                                "SMS to any number via your Twilio account. Calls dial "
                                "immediately and cost money — confirm before calling."),
            },
        },
    }


def _tool_dirs() -> list[str]:
    """Where Node, npx and uvx land on Windows, whether or not PATH knows yet.

    A tool installed while ORION is running is not on THIS process's PATH
    until he restarts, and a winget/MSI install may not have updated PATH at
    all for a process started earlier. Looking in the standard places means
    the servers start the moment the tools exist.
    """
    env = os.environ
    candidates = [
        os.path.join(env.get("ProgramFiles", r"C:\Program Files"), "nodejs"),
        os.path.join(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "nodejs"),
        os.path.join(env.get("APPDATA", ""), "npm"),
        os.path.join(env.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links"),
        os.path.join(os.path.expanduser("~"), ".local", "bin"),
        os.path.join(os.path.expanduser("~"), ".cargo", "bin"),
        os.path.join(env.get("LOCALAPPDATA", ""), "Programs", "nodejs"),
    ]
    return [path for path in candidates if path and os.path.isdir(path)]


def resolve_command(command: str) -> str:
    """The full path of *command*, or "" if it cannot be found.

    On Windows ``npx`` is ``npx.cmd``. asyncio's subprocess launcher calls
    CreateProcess, which appends only ``.exe`` to a bare name — so "npx" was
    never found even on a machine with Node installed, and every Node server
    failed to launch. The resolved path carries the right extension.
    """
    command = str(command or "").strip()
    if not command:
        return ""
    if os.path.isabs(command):
        return command if os.path.exists(command) else ""
    found = shutil.which(command)
    if found:
        return found
    search = os.pathsep.join(_tool_dirs())
    return shutil.which(command, path=search) or ""


def missing_credentials(spec: dict[str, Any]) -> list[str]:
    """The credential env vars an enabled server still needs filled in."""
    env = spec.get("env") or {}
    return [str(k) for k, v in env.items()
            if _SECRET_RE.search(str(k)) and not str(v or "").strip()]


def _tool_listed(tool: str, patterns: Any) -> bool:
    """Whether *tool* is named in *patterns* — exact names or shell wildcards
    ("git_*"), ignoring case. A lone string counts as a list of one."""
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, (list, tuple)):
        return False
    tool = str(tool or "").strip().lower()
    return any(fnmatch.fnmatchcase(tool, str(p).strip().lower())
               for p in patterns if str(p or "").strip())


def configured_confirmation(server: str, tool: str, spec: dict[str, Any]) -> str:
    """Why a server's OWN config says *tool* must be confirmed, or "".

    * ``"confirm": true`` — every tool, except any named in ``"confirm_except"``
      (for a server where nearly everything acts: Home Assistant, whose
      state-reading tools need not ask).
    * ``"confirm_tools": [...]`` — only the named tools (for a server that is
      mostly reading: git, whose commit and checkout should ask and whose
      status and log should not).
    * ``"confirm_reason"`` — what the person is approving, in words; shown in
      the prompt instead of the bare "marked as requiring confirmation".
    """
    if not isinstance(spec, dict):
        return ""
    why = str(spec.get("confirm_reason") or "").strip()
    if spec.get("confirm"):
        if _tool_listed(tool, spec.get("confirm_except")):
            return ""
        return (f"{server}.{tool} {why}" if why
                else f"'{server}' is marked as requiring confirmation")
    if _tool_listed(tool, spec.get("confirm_tools")):
        return (f"{server}.{tool} {why}" if why
                else f"'{server}.{tool}' is marked as requiring confirmation")
    return ""


def autofill_credentials(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Fill credentials ORION already holds, so they are not asked for twice.

    The Notion token is in api_keys.json's integrations (ORION's own Notion
    tool uses it); Brave/Tavily/GitHub keys may be in the environment. Values
    the user has typed into the server's own env always win.
    """
    spec = dict(spec)
    env = dict(spec.get("env") or {})
    known: dict[str, str] = {}
    try:
        from .providers import read_provider_settings

        notion = read_provider_settings().integration("notion")
        if str(notion.get("token") or "").strip():
            known["NOTION_TOKEN"] = str(notion["token"]).strip()
    except Exception:
        pass
    for var in ("BRAVE_API_KEY", "TAVILY_API_KEY", "GITHUB_PERSONAL_ACCESS_TOKEN",
                "GOOGLE_MAPS_API_KEY", "SLACK_BOT_TOKEN"):
        if os.getenv(var, "").strip():
            known[var] = os.environ[var].strip()
    if os.getenv("GITHUB_TOKEN", "").strip() and "GITHUB_PERSONAL_ACCESS_TOKEN" not in known:
        known["GITHUB_PERSONAL_ACCESS_TOKEN"] = os.environ["GITHUB_TOKEN"].strip()
    for key, value in list(env.items()):
        if not str(value or "").strip() and key in known:
            env[key] = known[key]
    spec["env"] = env
    return spec


#: Servers worth having on out of the box: useful, and needing nothing ORION
#: does not already have. Anything else stays off until the user turns it on.
_RECOMMENDED: dict[str, dict[str, Any]] = {
    "fetch": {
        "enabled": True,
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": {},
        "description": ("Fetch any web page as clean text or Markdown — ORION "
                        "reads sources with it when researching."),
    },
    "duckduckgo": {
        "enabled": True,
        "command": "uvx",
        "args": ["duckduckgo-mcp-server"],
        "env": {},
        "description": ("Web search with no API key and no account, plus page "
                        "fetching — a second search engine for research."),
    },
    "notion": {
        "enabled": True,
        "command": "npx",
        "args": ["-y", "@notionhq/notion-mcp-server"],
        "env": {"NOTION_TOKEN": ""},
        "description": ("Notion: search, read and update any page or database "
                        "the integration can see (token taken from ORION's "
                        "Notion settings)."),
    },
}


def migrate_config(data: dict[str, Any]) -> bool:
    """Repair known-broken entries and add the recommended servers. Returns
    whether anything changed. Runs once per CONFIG_REVISION, so a server the
    user switches off afterwards stays off."""
    servers = data.setdefault("servers", {})
    if not isinstance(servers, dict):
        return False
    changed = False
    # Always-safe repairs: these could never have started as written.
    fetch = servers.get("fetch")
    if isinstance(fetch, dict) and fetch.get("command") == "npx" and \
            "mcp-server-fetch" in " ".join(map(str, fetch.get("args") or [])):
        # mcp-server-fetch is a PYTHON package; there is no npm one.
        fetch["command"], fetch["args"] = "uvx", ["mcp-server-fetch"]
        changed = True
    brave = servers.get("brave_search")
    if isinstance(brave, dict) and "@modelcontextprotocol/server-brave-search" in \
            " ".join(map(str, brave.get("args") or [])):
        brave["args"] = ["-y", "@brave/brave-search-mcp-server"]   # the old one is deprecated
        changed = True
    ddg = servers.get("duckduckgo")
    if isinstance(ddg, dict) and ddg.get("command") == "npx":
        # The npm package of that name is a stale stub; the maintained server
        # is the Python one.
        ddg["command"], ddg["args"] = "uvx", ["duckduckgo-mcp-server"]
        changed = True
    if data.get("_revision") != CONFIG_REVISION:
        for name, spec in _RECOMMENDED.items():
            existing = servers.get(name)
            if not isinstance(existing, dict):
                servers[name] = json.loads(json.dumps(spec))
            else:
                existing["enabled"] = True
                existing.setdefault("description", spec["description"])
        data["_revision"] = CONFIG_REVISION
        changed = True
    return changed


def load_config() -> dict[str, Any]:
    try:
        data = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("servers"), dict):
            if migrate_config(data):
                try:
                    atomic_write_text(MCP_CONFIG_PATH, json.dumps(data, indent=2),
                                      encoding="utf-8")
                except OSError:
                    pass
            return data
    except FileNotFoundError:
        # First run — drop a template so the user can see the shape and enable one.
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_text(MCP_CONFIG_PATH, json.dumps(_default_config(), indent=2), encoding="utf-8")
        except OSError:
            pass
    except (OSError, ValueError):
        pass
    return _default_config()


#: How long a server gets to exit on its own, and again after being forced.
_EXIT_GRACE_S = 2.0


class MCPProtocolError(RuntimeError):
    """A JSON-RPC error reply: the server refused the request itself (unknown
    method, invalid params), as opposed to a tool that ran and reported an
    error in its result."""

    def __init__(self, error: Any) -> None:
        err = error if isinstance(error, dict) else {"message": str(error)}
        self.code = err.get("code")
        self.message = str(err.get("message") or "request rejected")
        self.data = err.get("data")
        detail = self.message if self.code is None else f"{self.message} (JSON-RPC {self.code})"
        if self.data:
            detail += f": {str(self.data)[:300]}"
        super().__init__(detail)


class MCPReply(str):
    """What ``MCPHost.call`` returns: the reply text, plus how it failed.

    A ``str`` so every caller that treats replies as text keeps working, but
    one that carries its outcome, so failure is not guessed from wording (a
    successful read of a file that happens to begin "(tool error)" is not a
    failure). ``kind`` is "" on success, else one of ``ERROR_KINDS``.
    """

    ERROR_KINDS = ("tool", "protocol", "timeout", "unavailable")
    kind: str

    def __new__(cls, text: str, kind: str = "") -> "MCPReply":
        reply = super().__new__(cls, text)
        reply.kind = kind
        return reply

    @property
    def is_error(self) -> bool:
        return bool(self.kind)


async def _kill_tree(proc: Any) -> None:
    """Force a server down together with anything it started.

    On Windows the process we hold is often a launcher (uvx.exe, or cmd.exe
    running npx.cmd) and the server proper is its child; terminating only the
    launcher orphans the child. taskkill /T takes the whole tree.
    """
    if os.name == "nt" and getattr(proc, "pid", None):
        import subprocess

        def _taskkill() -> int:
            return subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True, creationflags=0x08000000,   # CREATE_NO_WINDOW
                timeout=5).returncode

        try:
            if await asyncio.to_thread(_taskkill) == 0:
                return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


class MCPServerConn:
    """One MCP server subprocess and the JSON-RPC conversation with it."""

    def __init__(self, name: str, spec: dict[str, Any], bus: Any) -> None:
        self.name = name
        self.command = str(spec.get("command") or "")
        self.args = [str(a) for a in (spec.get("args") or [])]
        self.env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
        self.description = str(spec.get("description") or "")
        self.bus = bus
        self.proc: asyncio.subprocess.Process | None = None
        self.tools: list[dict[str, Any]] = []
        self._id = 0
        self._lock = asyncio.Lock()          # one in-flight request at a time
        self.ready = False
        self.last_error = ""
        self._stderr_tail: list[str] = []
        self._stderr_task: Any = None
        #: When a tool was last called (monotonic), for idle parking.
        self.last_used = time.monotonic()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        executable = resolve_command(self.command)
        if not executable:
            hint = (" — install Node.js (winget install OpenJS.NodeJS.LTS)"
                    if self.command in {"npx", "node", "npm"} else
                    " — install uv (winget install astral-sh.uv)"
                    if self.command in {"uvx", "uv"} else "")
            self.last_error = f"'{self.command}' is not installed{hint}"
            self._log(f"{self.last_error}; skipped.")
            return False
        # The server's own directory on PATH too, so npx can find node.
        env = {**os.environ, **self.env}
        extra = os.pathsep.join([os.path.dirname(executable), *_tool_dirs()])
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")
        flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
        try:
            self.proc = await asyncio.create_subprocess_exec(
                executable, *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                creationflags=flags,
                # A chatty server's stderr (npx download progress) must never
                # fill the pipe and stall it mid-handshake.
                limit=4 * 1024 * 1024,
            )
        except Exception as exc:
            self.last_error = f"failed to launch - {exc}"
            self._log(self.last_error)
            return False
        self._drain_stderr()
        try:
            await asyncio.wait_for(self._handshake(), timeout=_HANDSHAKE_TIMEOUT)
        except Exception as exc:
            tail = " | ".join(self._stderr_tail[-3:])
            self.last_error = f"handshake failed - {exc or type(exc).__name__}" + (
                f" ({tail[:200]})" if tail else "")
            self._log(self.last_error)
            await self.stop()
            return False
        self.ready = True
        self.last_error = ""
        self.last_used = time.monotonic()
        self._log(f"connected — {len(self.tools)} tool(s) available.")
        return True

    def _drain_stderr(self) -> None:
        """Read the server's stderr continuously, keeping the last few lines.

        Nothing read it before, so a server that logged enough (npx prints
        download progress there) filled the OS pipe buffer and blocked — and
        when a handshake failed there was no clue why. The tail is what the
        failure message quotes.
        """
        proc = self.proc
        if proc is None or proc.stderr is None:
            return

        async def _pump() -> None:
            try:
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        return
                    text = line.decode("utf-8", "replace").strip()
                    if text:
                        self._stderr_tail.append(text[:240])
                        del self._stderr_tail[:-12]
            except Exception:
                return

        try:
            self._stderr_task = asyncio.get_running_loop().create_task(_pump())
        except RuntimeError:
            pass

    async def _handshake(self) -> None:
        result = await self._request("initialize", {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "O.R.I.O.N.", "version": "1.0"},
        })
        if not isinstance(result, dict):
            raise RuntimeError("no initialize result")
        await self._notify("notifications/initialized")
        listed = await self._request("tools/list", {})
        self.tools = list((listed or {}).get("tools") or [])

    async def stop(self) -> None:
        proc = self.proc
        self.proc = None
        self.ready = False
        task = self._stderr_task
        self._stderr_task = None
        if task is not None:
            task.cancel()
        if proc is None:
            return
        # The MCP stdio shutdown order: close the server's stdin and let it
        # exit, and only then force it. The order matters on Windows: uvx and
        # npx run the real server as a CHILD that shares the pipes, so
        # terminating the launcher left wait() blocked on pipes the child
        # still held — the full timeout, per server, at every shutdown
        # (measured: 12.0 s of a 14.3 s exit for three servers).
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=_EXIT_GRACE_S)
        except Exception:
            await _kill_tree(proc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=_EXIT_GRACE_S)
            except Exception:
                pass
        # Close the pipes explicitly. Left to the garbage collector, each one
        # raised "I/O operation on closed pipe" from its __del__ at exit.
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass

    def is_alive(self) -> bool:
        """Mark XXI, Track E2: True only while the subprocess is genuinely
        still running — a crashed server (non-None returncode) or one that
        was never started is not, regardless of the last-known `ready`
        flag, so a supervisor can tell the difference between "healthy"
        and "handshake succeeded once but has since died"."""
        return self.ready and self.proc is not None and self.proc.returncode is None

    # ── JSON-RPC over stdio (newline-delimited) ────────────────────────────────

    async def _send(self, message: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("server not running")
        self.proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        async with self._lock:
            self._id += 1
            req_id = self._id
            msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                msg["params"] = params
            await self._send(msg)
            # Read replies until the one matching our id arrives; server-initiated
            # requests/notifications (sampling, logging) are ignored — ORION uses
            # servers only as tool providers.
            assert self.proc is not None and self.proc.stdout is not None
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    self.ready = False   # Track E2: detected here, not only on next probe
                    raise RuntimeError("server closed the connection")
                try:
                    obj = json.loads(line.decode("utf-8").strip())
                except ValueError:
                    continue
                if obj.get("id") == req_id:
                    if "error" in obj:
                        raise MCPProtocolError(obj["error"])
                    return obj.get("result")

    def busy(self) -> bool:
        """A request is in flight, so the process must not be stopped."""
        return self._lock.locked()

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> "MCPReply":
        self.last_used = time.monotonic()
        result = await asyncio.wait_for(
            self._request("tools/call", {"name": tool, "arguments": arguments or {}}),
            timeout=_CALL_TIMEOUT,
        )
        return self._render(result)

    @staticmethod
    def _render(result: Any) -> "MCPReply":
        """Flatten an MCP tool result's content blocks to text."""
        if not isinstance(result, dict):
            return MCPReply(str(result))
        parts: list[str] = []
        for block in result.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(f"[{block.get('type', 'content')}]")
        text = "\n".join(p for p in parts if p).strip()
        if result.get("isError"):
            # The tool ran and failed: its own message is what lets the model
            # correct the arguments, so it is kept whole.
            return MCPReply(f"(tool error) {text or 'the MCP server reported an error'}", "tool")
        return MCPReply(text or "(the tool returned no content)")

    def _log(self, message: str) -> None:
        try:
            self.bus.log.emit(f"MCP[{self.name}]: {message}")
        except Exception:
            pass


def _spec_fingerprint(spec: dict[str, Any]) -> str:
    """A digest of what launches a server; a changed command, argument or
    credential means its cached tool list can no longer be trusted. Only the
    digest is stored, never the credential itself."""
    import hashlib
    material = json.dumps({"command": spec.get("command"), "args": spec.get("args"),
                           "env": spec.get("env")}, sort_keys=True, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_catalogue() -> dict[str, Any]:
    try:
        data = json.loads(MCP_CATALOGUE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _lazy_start_enabled(spec: dict[str, Any]) -> bool:
    """Servers start on first use unless ORION_MCP_LAZY=0 or the server's
    own config says "eager": true."""
    if os.getenv("ORION_MCP_LAZY", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return not spec.get("eager")


class MCPHost:
    """Connects to the configured MCP servers and routes tool calls to them."""

    # Mark XXI, Track E2: bounded backoff for reconnect attempts — a server
    # that's genuinely down (bad credentials, package removed) must not be
    # hammered every supervision tick forever.
    _RECONNECT_BACKOFF_S = (5.0, 15.0, 60.0, 300.0)

    def __init__(self, bus: Any, telemetry: Any | None = None) -> None:
        self.bus = bus
        self.telemetry = telemetry
        self.servers: dict[str, MCPServerConn] = {}
        self._specs: dict[str, dict[str, Any]] = {}    # last-known enabled specs, for reconnect
        self._reconnect_attempts: dict[str, int] = {}
        self._next_reconnect_at: dict[str, float] = {}
        self._supervising = False
        #: Why each enabled server is not live, in words ("waiting for
        #: TWILIO_AUTH_TOKEN", "'npx' is not installed", a handshake error).
        self.status: dict[str, str] = {}
        #: Servers stopped while idle. They stay in ``servers`` so their tools
        #: are still listed, and ``call`` restarts one before using it.
        self.parked: set[str] = set()
        self._wake_locks: dict[str, asyncio.Lock] = {}
        self._on_reconnect: Any = None
        self._pressure_park_s: float | None = None
        # App startup connects optional servers in the background so the GUI
        # paints immediately.  A first user request can therefore arrive in
        # the small gap before ``connect_all`` publishes a live connection.
        # The call path waits briefly for that owned task before reporting a
        # missing server; disabled or already-finished hosts remain instant.
        self._startup_task: asyncio.Task | None = None
        try:
            self.idle_park_s = float(os.getenv("ORION_MCP_IDLE_PARK_S", _IDLE_PARK_S))
        except ValueError:
            self.idle_park_s = _IDLE_PARK_S

    def _prepare(self, name: str, spec: dict[str, Any]) -> dict[str, Any] | None:
        """The spec to launch, or None when it cannot work yet (and why)."""
        spec = autofill_credentials(name, spec)
        missing = missing_credentials(spec)
        if missing:
            self.status[name] = "waiting for " + ", ".join(missing)
            self._beat(name, "NEEDS_KEY", self.status[name])
            return None
        return spec

    async def connect_all(self) -> None:
        config = load_config()
        specs = config.get("servers") or {}
        enabled = {n: s for n, s in specs.items() if isinstance(s, dict) and s.get("enabled")}
        self._specs = enabled
        if not enabled:
            try:
                self.bus.log.emit(
                    "MCP: host ready — no servers enabled yet "
                    "(edit config/mcp_servers.json; see docs/MCP_SETUP.md)."
                )
            except Exception:
                pass
            return

        catalogue = _load_catalogue()

        async def _one(name: str, spec: dict[str, Any]) -> None:
            ready = self._prepare(name, spec)
            if ready is None:
                self._log_host(f"'{name}' is enabled but {self.status[name]} — "
                               f"not started (add it in config/mcp_servers.json).")
                return
            conn = MCPServerConn(name, ready, self.bus)
            cached = catalogue.get(name) or {}
            if (_lazy_start_enabled(spec) and cached.get("tools")
                    and cached.get("fingerprint") == _spec_fingerprint(ready)):
                # Ten servers held about a gigabyte from the moment ORION
                # started, on a machine already near its limit — which is when
                # the face flashed black. Their tool lists are known from the
                # last run, so offer those now and start each process only
                # when one of its tools is actually called (call -> _wake).
                conn.tools = list(cached["tools"])
                self.servers[name] = conn
                self.parked.add(name)
                self.status[name] = "idle - starts when first used"
                self._beat(name, "OK", f"{len(conn.tools)} tool(s), starts on first use")
                return
            ok = await conn.start()
            if ok:
                self.servers[name] = conn
                self._reconnect_attempts[name] = 0
                self.status[name] = "live"
                self._remember_tools(name, ready, conn)
                self._beat(name, "OK", f"{len(conn.tools)} tool(s)")
            else:
                self.status[name] = conn.last_error or "did not start"
                self._beat(name, "DOWN", self.status[name])

        # Concurrently: a first launch downloads each server's package, and
        # doing that one after another made the last server wait minutes.
        await asyncio.gather(*(_one(n, s) for n, s in enabled.items()),
                             return_exceptions=True)
        live = [n for n in enabled if self.status.get(n) == "live"]
        waiting = [n for n in enabled if n in self.parked]
        self._log_host(f"{len(live)} of {len(enabled)} enabled server(s) live"
                       + (f": {', '.join(live)}" if live else "")
                       + (f"; {len(waiting)} start on first use: {', '.join(waiting)}"
                          if waiting else "") + ".")

    def _remember_tools(self, name: str, spec: dict[str, Any], conn: "MCPServerConn") -> None:
        """Save a server's tool list for the next boot's lazy start."""
        if not conn.tools:
            return
        try:
            catalogue = _load_catalogue()
            catalogue[name] = {"fingerprint": _spec_fingerprint(spec),
                               "description": conn.description,
                               "tools": conn.tools}
            atomic_write_text(MCP_CATALOGUE_PATH, json.dumps(catalogue, indent=1),
                              encoding="utf-8")
        except Exception:
            pass

    # ── health (Track E2) ────────────────────────────────────────────────────

    def _beat(self, name: str, status: str, detail: str = "") -> None:
        if self.telemetry is None:
            return
        component = f"mcp.{name}"
        try:
            self.telemetry.health.register(component)
            self.telemetry.health.beat(component, status, detail)
        except Exception:
            pass

    def health_snapshot(self) -> dict[str, str]:
        """{server_name: "OK"|"DOWN"} for every server that has ever been
        enabled this session — including one currently disconnected and
        awaiting reconnect, so a dashboard can show it as a real problem
        rather than it simply vanishing from view."""
        out: dict[str, str] = {}
        for name in self._specs:
            conn = self.servers.get(name)
            if name in self.parked:
                out[name] = "IDLE"        # paused to save memory, not broken
            elif conn is not None and conn.is_alive():
                out[name] = "OK"
            elif str(self.status.get(name, "")).startswith("waiting for"):
                out[name] = "NEEDS_KEY"   # not broken — unconfigured
            else:
                out[name] = "DOWN"
        return out

    # ── reconnect + supervision (Track E2) ──────────────────────────────────

    async def reconnect(self, name: str) -> bool:
        """Tear down and re-establish one server's connection from its
        last-known enabled spec — re-reading config/mcp_servers.json first,
        so a credential/command fix the user just made takes effect without
        a full app restart. Returns whether it's alive afterwards."""
        config = load_config()
        spec = (config.get("servers") or {}).get(name)
        if not isinstance(spec, dict) or not spec.get("enabled"):
            # The user disabled or removed it — stop tracking it as a live
            # target rather than retrying a server that's meant to be off.
            self._specs.pop(name, None)
            old = self.servers.pop(name, None)
            if old is not None:
                await old.stop()
            return False
        self._specs[name] = spec
        old = self.servers.pop(name, None)
        if old is not None:
            await old.stop()
        ready = self._prepare(name, spec)
        if ready is None:
            return False
        conn = MCPServerConn(name, ready, self.bus)
        ok = await conn.start()
        if ok:
            self.servers[name] = conn
            self._reconnect_attempts[name] = 0
            self.status[name] = "live"
            self._remember_tools(name, ready, conn)
            self._beat(name, "OK", f"reconnected — {len(conn.tools)} tool(s)")
        else:
            self.status[name] = conn.last_error or "reconnect attempt failed"
            self._beat(name, "DOWN", "reconnect attempt failed")
        return ok

    def _due_for_reconnect(self, name: str, now: float) -> bool:
        return now >= self._next_reconnect_at.get(name, 0.0)

    def _schedule_next_reconnect(self, name: str, now: float) -> None:
        attempt = self._reconnect_attempts.get(name, 0)
        delay = self._RECONNECT_BACKOFF_S[min(attempt, len(self._RECONNECT_BACKOFF_S) - 1)]
        self._reconnect_attempts[name] = attempt + 1
        self._next_reconnect_at[name] = now + delay

    async def supervise(self, interval_s: float = 30.0,
                        on_reconnect: Any | None = None) -> None:
        """Run forever (until cancelled): every *interval_s*, check each
        enabled server that isn't currently alive and reconnect it with
        bounded backoff. *on_reconnect(name, conn)* — a plain sync
        callable, e.g. re-registering the dispatcher's first-class MCP
        tools (Track E1) — fires after each successful reconnect. Never
        raises out of the loop; one server's trouble never stops the
        supervisor watching the others."""
        self._supervising = True
        self._on_reconnect = on_reconnect
        try:
            while self._supervising:
                await asyncio.sleep(interval_s)
                try:
                    await self.park_idle()
                except Exception as exc:
                    self._log_host(f"idle parking skipped - {exc}")
                now = asyncio.get_running_loop().time()
                for name in list(self._specs):
                    if name in self.parked:
                        continue              # asleep on purpose; woken by a call
                    conn = self.servers.get(name)
                    if conn is not None and conn.is_alive():
                        continue
                    if not self._due_for_reconnect(name, now):
                        continue
                    if str(self.status.get(name, "")).startswith("waiting for") and \
                            missing_credentials(autofill_credentials(name, self._specs[name])):
                        # Still no key: retrying cannot help, and every retry
                        # used to be logged as a fresh failure.
                        self._schedule_next_reconnect(name, now)
                        continue
                    self._schedule_next_reconnect(name, now)
                    try:
                        ok = await self.reconnect(name)
                    except Exception as exc:
                        self._log_host(f"supervised reconnect of '{name}' raised - {exc}")
                        continue
                    if ok and on_reconnect is not None:
                        try:
                            on_reconnect(name, self.servers[name])
                        except Exception as exc:
                            self._log_host(f"on_reconnect callback for '{name}' failed - {exc}")
        except asyncio.CancelledError:
            pass

    def stop_supervising(self) -> None:
        self._supervising = False

    # ── idle parking ────────────────────────────────────────────────────────

    def set_pressure_parking(self, seconds: float | None) -> None:
        """Park sooner while memory is short (the resource governor's lever);
        None goes back to the everyday limit."""
        self._pressure_park_s = seconds

    def _park_limit(self) -> float | None:
        limits = [v for v in (self.idle_park_s, self._pressure_park_s)
                  if v is not None and v > 0]
        return min(limits) if limits else None

    async def park_idle(self, now: float | None = None) -> list[str]:
        """Stop every live server unused for longer than the limit.

        Its tool list is kept, so the model and the dispatcher still see the
        tools; the process (and its memory) is what goes.
        """
        limit = self._park_limit()
        if limit is None:
            return []
        now = time.monotonic() if now is None else now
        parked: list[str] = []
        for name, conn in list(self.servers.items()):
            if name in self.parked or not conn.is_alive() or conn.busy():
                continue
            if now - conn.last_used < limit:
                continue
            self.parked.add(name)
            self.status[name] = "paused while idle (restarts on its next use)"
            await conn.stop()
            parked.append(name)
        if parked:
            self._log_host(f"paused idle server(s) to free memory: "
                           f"{', '.join(parked)} - each restarts when next used.")
        return parked

    async def _wake(self, name: str) -> bool:
        lock = self._wake_locks.setdefault(name, asyncio.Lock())
        async with lock:
            if name not in self.parked:
                conn = self.servers.get(name)
                return conn is not None and conn.is_alive()
            ok = await self.reconnect(name)
            self.parked.discard(name)       # failed: the supervisor takes over
            if ok and self._on_reconnect is not None:
                try:
                    self._on_reconnect(name, self.servers[name])
                except Exception as exc:
                    self._log_host(f"on_reconnect callback for '{name}' failed - {exc}")
            return ok

    def _log_host(self, message: str) -> None:
        try:
            self.bus.log.emit(f"MCP: {message}")
        except Exception:
            pass

    def catalogue(self) -> dict[str, Any]:
        """Every connected server and its tools — for the model to browse."""
        out: dict[str, Any] = {}
        for name, conn in self.servers.items():
            out[name] = {
                "description": conn.description,
                "tools": [
                    {"name": t.get("name"), "description": t.get("description", "")}
                    for t in conn.tools
                ],
            }
        return out

    def describe(self) -> str:
        cat = self.catalogue()
        if not cat:
            return ("No MCP servers are connected. Enable one in "
                    "config/mcp_servers.json (see docs/MCP_SETUP.md).")
        lines: list[str] = []
        for server, info in cat.items():
            lines.append(f"• {server} — {info['description']}")
            for tool in info["tools"]:
                lines.append(f"    - {tool['name']}: {tool['description'][:100]}")
        return "\n".join(lines)

    #: Servers whose tools act in the world at the user's expense, even if
    #: their spec predates the "confirm" flag. Twilio dials immediately and
    #: bills immediately; there is no draft state to inspect afterwards.
    CONFIRM_SERVERS = {"twilio"}

    #: Tool names that spend money or leave the machine, on any server. Matched
    #: as substrings, because servers name these differently — create_call,
    #: send_sms, make_outbound_call — and the cost is the same whichever.
    CONFIRM_WORDS = ("call", "sms", "dial", "fax", "purchase", "buy",
                     "payment", "charge", "transfer")

    def requires_confirmation(self, server: str, tool: str) -> str:
        """Why this call should be confirmed first, or "" if it need not be.

        A reason rather than a boolean: the person is being asked to approve
        something, and "this needs confirmation" does not tell them what they
        are approving.
        """
        server = str(server or "").strip().lower()
        shown = str(tool or "").strip()
        tool = shown.lower()
        spec = (load_config().get("servers") or {}).get(server)
        reason = configured_confirmation(server, shown, spec)
        if reason:
            return reason
        if server in self.CONFIRM_SERVERS:
            hit = next((w for w in self.CONFIRM_WORDS if w in tool), "")
            if hit in ("call", "dial"):
                return (f"{server}.{tool} places a real phone call that "
                        f"connects immediately and costs money")
            if hit in ("sms", "fax"):
                return (f"{server}.{tool} sends a real {hit.upper()} and "
                        f"costs money")
            if hit:
                return f"{server}.{tool} spends real money"
            return f"'{server}' acts at your expense"
        return ""

    async def call(self, server: str, tool: str, arguments: dict[str, Any]) -> MCPReply:
        startup = self._startup_task
        if (server not in self.servers and server not in self.parked
                and startup is not None and not startup.done()):
            try:
                # Two seconds covers the normal local handshake without
                # turning a slow first package download into a voice stall.
                await asyncio.wait_for(asyncio.shield(startup), timeout=2.0)
            except asyncio.CancelledError:
                # The startup task being cancelled is fine; THIS call being
                # cancelled (a cancelled turn, shutdown) must not be swallowed.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
            except Exception:
                pass
        if server in self.parked and not await self._wake(server):
            return MCPReply(
                f"MCP server '{server}' was paused to save memory and could "
                f"not restart: {self.status.get(server) or 'unknown error'}.", "unavailable")
        conn = self.servers.get(server)
        if conn is None:
            avail = ", ".join(self.servers) or "none"
            return MCPReply(
                f"No connected MCP server named '{server}'. Available: {avail}.", "unavailable")
        if not any(t.get("name") == tool for t in conn.tools):
            names = ", ".join(t.get("name", "?") for t in conn.tools) or "none"
            return MCPReply(
                f"Server '{server}' has no tool '{tool}'. Its tools: {names}.", "protocol")
        try:
            return await conn.call_tool(tool, arguments)
        except asyncio.TimeoutError:
            return MCPReply(f"MCP tool '{server}.{tool}' timed out.", "timeout")
        except MCPProtocolError as exc:
            # The server refused the call itself — almost always arguments
            # that do not match the tool's input schema. Say so, so the model
            # fixes the arguments rather than retrying the same call.
            return MCPReply(
                f"MCP tool '{server}.{tool}' failed: the server rejected the request "
                f"- {exc}. Check the arguments against the tool's input schema.", "protocol")
        except Exception as exc:
            return MCPReply(f"MCP tool '{server}.{tool}' failed: {exc}", "unavailable")

    # ── runtime server management ───────────────────────────────────────────
    # Enabling a server used to mean editing config/mcp_servers.json by hand
    # and restarting ORION.  These make the whole MCP surface manageable while
    # he is running — the config file stays the single source of truth, it is
    # just written for the user and applied live.

    def list_servers(self) -> str:
        """Every DECLARED server (not merely the connected ones) with its
        enabled and live state, so a server that is off — or enabled but
        failing to start — is visible instead of simply absent."""
        config = load_config()
        specs = config.get("servers") or {}
        if not specs:
            return "No MCP servers are declared in config/mcp_servers.json."
        lines: list[str] = []
        for name in sorted(specs):
            spec = specs[name] if isinstance(specs[name], dict) else {}
            enabled = bool(spec.get("enabled"))
            conn = self.servers.get(name)
            live = conn is not None and conn.is_alive()
            reason = self.status.get(name, "")
            if name in self.parked:
                reason = (f"idle, paused to save memory ({len(conn.tools) if conn else 0}"
                          f" tool(s); restarts on its next use)")
            if not reason and enabled and not live:
                missing = missing_credentials(autofill_credentials(name, spec))
                if missing:
                    reason = "waiting for " + ", ".join(missing)
                elif not resolve_command(str(spec.get("command") or "")):
                    reason = f"'{spec.get('command')}' is not installed"
            state = "live" if live else (
                f"enabled — {reason}" if enabled and reason else
                "enabled (not connected)" if enabled else "disabled")
            tools = f", {len(conn.tools)} tool(s)" if live and conn else ""
            lines.append(f"• {name} — {state}{tools}")
            description = str(spec.get("description") or "").strip()
            if description:
                lines.append(f"    {description[:120]}")
        lines.append("")
        lines.append("Use mcp enable/disable <server> to change this, then "
                     "mcp reconnect <server> if needed.")
        return "\n".join(lines)

    def _write_config(self, config: dict[str, Any]) -> bool:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_text(MCP_CONFIG_PATH, json.dumps(config, indent=2), encoding="utf-8")
            return True
        except OSError:
            return False

    async def set_enabled(self, name: str, enabled: bool) -> str:
        """Enable or disable one declared server and apply it immediately."""
        server = str(name or "").strip()
        if not server:
            return "Which MCP server, sir?"
        config = load_config()
        specs = config.get("servers") or {}
        if server not in specs or not isinstance(specs[server], dict):
            known = ", ".join(sorted(specs)) or "none"
            return f"No MCP server named '{server}'. Declared: {known}."
        specs[server]["enabled"] = bool(enabled)
        config["servers"] = specs
        if not self._write_config(config):
            return f"Could not write {MCP_CONFIG_PATH.name}."
        if enabled:
            ok = await self.reconnect(server)
            return (f"MCP server '{server}' enabled and connected."
                    if ok else
                    f"MCP server '{server}' enabled, but it did not start — "
                    "check its command and credentials in "
                    f"{MCP_CONFIG_PATH.name}.")
        conn = self.servers.pop(server, None)
        if conn is not None:
            try:
                await conn.stop()
            except Exception:
                pass
        self._specs.pop(server, None)
        self._beat(server, "DOWN", "disabled by the user")
        return f"MCP server '{server}' disabled and disconnected."

    def add_server(self, name: str, command: str, args: list[str] | None = None,
                   env: dict[str, str] | None = None,
                   description: str = "") -> str:
        """Declare a NEW MCP server. It is written disabled, so nothing is
        launched until the user has filled in credentials and enabled it."""
        server = str(name or "").strip()
        if not server:
            return "Give the server a name, sir."
        if not str(command or "").strip():
            return "Give the server a command to run (e.g. npx)."
        config = load_config()
        specs = config.get("servers") or {}
        if server in specs:
            return (f"An MCP server called '{server}' is already declared. "
                    "Edit it in config/mcp_servers.json, or pick another name.")
        specs[server] = {
            "enabled": False,
            "command": str(command).strip(),
            "args": [str(a) for a in (args or [])],
            "env": {str(k): str(v) for k, v in (env or {}).items()},
            "description": str(description or "").strip(),
        }
        config["servers"] = specs
        if not self._write_config(config):
            return f"Could not write {MCP_CONFIG_PATH.name}."
        return (f"Declared MCP server '{server}' (disabled). Fill in any "
                f"credentials in {MCP_CONFIG_PATH.name}, then: mcp enable {server}.")

    async def close(self) -> None:
        # Together, not in turn: each stop can wait out its grace period.
        await asyncio.gather(*(conn.stop() for conn in list(self.servers.values())),
                             return_exceptions=True)
        self.servers.clear()
        self.parked.clear()
