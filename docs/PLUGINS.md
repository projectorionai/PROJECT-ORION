# Writing an ORION plugin

A **plugin** is Python code that adds a new tool to ORION at runtime — no restart,
no editing `dispatcher.py`. (Contrast a **skill**, which is data only: prompts,
templates, workflows.)

The fastest way to start is to let ORION write the skeleton:

```
plugin create name=lamp description="Flash my desk lamp"
```

That writes two files into `config/custom_tools/` and they already work.

---

## The contract

A plugin is **one module** plus **one manifest**, side by side:

```
config/custom_tools/
    lamp_tool.py        <- the code   (must end in _tool.py)
    lamp.plugin.json    <- the manifest
```

### The module

Two top-level functions are required:

```python
"""Flash my desk lamp."""

def get_tool_schema() -> dict:
    """What the model sees."""
    return {
        "name": "lamp",
        "description": "Turn the desk lamp on or off.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "state": {"type": "STRING", "description": "on or off"},
            },
            "required": ["state"],
        },
    }


def run(**kwargs) -> str:
    """Do the work. Return a short string for ORION to speak or show."""
    return f"Lamp turned {kwargs.get('state', 'on')}."
```

`run()` is called with the arguments the model supplied. Return a **string**.
Raising is safe — the failure is recorded against the plugin's health rather than
propagating into ORION.

### The manifest

```json
{
  "name": "lamp",
  "description": "Turn the desk lamp on or off.",
  "module": "lamp_tool.py",
  "version": "1.0.0",
  "tier": "confirm",
  "permissions": ["network"],
  "events": [],
  "requires": ["requests"],
  "parameters": { "type": "OBJECT", "properties": {} }
}
```

| field | meaning |
| --- | --- |
| `tier` | how a **paired remote device** may call it: `allow` (unattended), `confirm` (asks first), `forbid` (never remotely). Default `confirm`. |
| `permissions` | what the plugin needs to touch — see below. |
| `events` | ORION events it wants to react to (see *Reactive plugins*). |
| `requires` | Python packages. A plugin whose dependency is missing is **skipped with an actionable log**, never crash-loaded. `plugin deps <name>` installs them. |
| `version` | compared against a candidate by `plugin update <name> version=…`. |

---

## Permissions — an honest disclosure

Declare the coarse capabilities the plugin uses:

`network` · `filesystem` · `subprocess` · `input` · `screen` · `audio`

ORION **statically** reads the module (never imports it) and compares what the
code actually does against what the manifest claims. Anything used but not
declared shows up as **UNDECLARED**:

```
plugin audit
```

```
  lamp (v1.0.0, tier=confirm) — declares: network
      ⚠ uses undeclared: subprocess
```

This is a *disclosure* mechanism, not a sandbox — ORION does not block the call,
it makes the truth visible. Two things are surfaced automatically:

* at **load time**, in the ordinary log, and
* in `plugin doctor`, along with a warning if a plugin runs **unattended**
  (`tier: allow`) while touching `subprocess`, `input`, `network` or `filesystem`.

---

## Reactive plugins — react, don't just get called

A plugin can also *watch* ORION. Declare the events and export `on_event`:

```
plugin create name=lamp description="Flash on state changes" kind=event
```

```json
"events": ["state", "speaking", "connection_state"]
```

```python
def on_event(event: str, payload=None) -> None:
    """Called when a subscribed event fires."""
    if event == "state" and payload == "SPEAKING":
        ...
```

Subscribable events (observational only — a plugin cannot hook command
execution or confirmations):

`state` · `speaking` · `log` · `banner` · `safety_alert` · `connection_state` ·
`emotion_changed` · `telemetry_sample` · `dashboard_event` · `paused`

> ⚠ **A hook runs on ORION's own thread.** Keep `on_event` well under **50 ms** and
> hand slow work to a background job. A plugin that keeps blocking is **muted**
> for the session, as is one that keeps raising. `plugin hooks` shows delivery
> counts, timings, faults and mutes.

---

## Managing plugins

| command | what it does |
| --- | --- |
| `plugin list` | every plugin, its tier and live status |
| `plugin describe <name>` | version, what it can touch, health, call count |
| `plugin audit [name]` | declared vs actual capabilities |
| `plugin doctor` | one health pass over everything |
| `plugin enable/disable <name>` | persists across restarts; disabling stops its hooks at once |
| `plugin reload` | hot-reload every enabled plugin, re-binding hooks |
| `plugin deps <name>` | install its missing Python packages |
| `plugin hooks` | which plugins are subscribed to what |
| `plugin unmute [name]` | revive a plugin whose hooks were muted for faulting or blocking |
| `plugin backfill` | write manifests for plugins that lack one |
| `plugin export <name>` | package it as a portable `.orionplugin` bundle |
| `plugin install <path>` | install a `*_tool.py` **or** a `.orionplugin` bundle |
| `plugin remove <name>` | delete it |

---

## Sharing a plugin

```
plugin export name=lamp
```

produces a single `lamp.orionplugin` file (module + manifest). Anyone can add it:

```
plugin install source=C:\path\to\lamp.orionplugin
```

Bundles are validated before anything is written — only a plain `*_tool.py` /
`*.plugin.json` at the archive root is accepted, so a bundle cannot write outside
the plugin directory.

---

## How it works internally

* `orion_core/plugin_manifest.py` — parses/validates the manifest, dependency
  checks it, and loads the plan into the live dispatcher.
* `orion_core/plugin_registry.py` — discovery, enable/disable state
  (`config/plugins.json`), scaffolding, install/remove, static validation,
  capability auditing, bundles, and `doctor`.
* `orion_core/plugin_events.py` — the event bridge, with fault and slow-hook
  muting.
* `orion_core/dynamic_loader.py` — the shared loader (also used by the Forge), so
  a plugin that hangs at import is **timed out and quarantined** rather than
  blocking start-up.
* The PLUGINS page on the Command Deck shows all of it, including the capability
  disclosure.

---

## Plugins added 2026-09-20

Three, chosen because none of them needs a paid account and all three do
something ORION could not do before. Each is a manifest plus a module in
`config/custom_tools/`, exposing `run(**kwargs) -> str`.

### `open_meteo_weather` — weather, with no key at all

```
"what's the weather in Birmingham"
"weather for B15 2TT for the next three days"
```

Open-Meteo is free for non-commercial use and requires no account. A weather
tool that first demands an API key is a weather tool nobody turns on.

Places resolve through Open-Meteo's geocoder, except UK postcodes, which it
handles poorly — those go to postcodes.io, which ORION already uses elsewhere
for the same reason. WMO weather codes are translated into words, because
"code 61" is not an answer to "what's the weather".

Nothing to configure.

### `push_notify` — a notification on your phone

```
"send that to my phone"
"remind me on my phone when the build finishes"
```

Three services behind one tool — ntfy, Gotify and Pushover. They do the same
job and all have a free route to it, so they are backends rather than three
separate plugins: a tool list with three entries that each mean "tell me
something" is one the model chooses from badly, and ORION already has a great
many tools competing for attention.

Whichever is configured is used; configure several and `service` picks one.

**ntfy** is the quickest and needs no account at all:

1. Install the ntfy app, or open <https://ntfy.sh/app>.
2. Choose an **unguessable** topic name and subscribe to it.
3. `{"ntfy": {"topic": "orion-<something-random>"}}` in
   `config/messaging.json`, or set `NTFY_TOPIC`.

**Gotify** — self-hosted, properly authenticated: `GOTIFY_URL` +
`GOTIFY_TOKEN`.

**Pushover** — one-off app purchase, then the API is free, and the most
reliable delivery of the three on iOS: `PUSHOVER_TOKEN` + `PUSHOVER_USER`.

**On ntfy's public server a topic is only as private as its name.** Anyone who
learns it can read what is published there and publish to it themselves. Use
it for nudges, not secrets; Gotify self-hosted and Pushover are authenticated.

### `philips_hue` — lights, over your own network

```
"turn the kitchen light off"
"dim the lounge to 30%"
```

The bridge answers HTTP directly on the LAN, so nothing touches Philips' cloud,
nothing needs an account, and it works with no internet connection at all.

Pair once: ask ORION to pair, press the round button on top of the bridge, ask
again within thirty seconds. That button is the entire security model, and a
good one — you have to be in the room. The "username" it returns is a bearer
token and lives in `config/philips_hue.json`.

Redundant if you run Home Assistant, which reaches the same bulbs.

### `ifttt_webhook` — the long tail

```
"trigger my kettle applet"
```

One HTTP request becomes whatever applet you built on the other side, which
makes it the catch-all for devices ORION will never have a dedicated tool for.

`IFTTT_KEY` from <https://ifttt.com/maker_webhooks> → Documentation.

It is **fire-and-forget**: IFTTT acknowledges the request, not the outcome. The
tool says "asked IFTTT to run X" rather than claiming the applet succeeded,
because it cannot know that and saying otherwise would be a confident lie.

### `homeassistant_call_service` — the house

```
"turn the kitchen light off"
"what lights are on"
```

Home Assistant's REST API is the whole integration — no SDK, no cloud account.

1. `HASS_URL`, e.g. `http://homeassistant.local:8123`.
2. `HASS_TOKEN` — a long-lived access token from your own profile page.

Either as environment variables or in `config/homeassistant.json`:

```json
{ "url": "http://homeassistant.local:8123", "token": "ey..." }
```

Three actions: `entities` lists what exists, `state` reads one, `call` acts.
The lister matters more than it looks — entity ids are not guessable
(`light.kitchen` and `light.kitchen_ceiling` are different things, and yours
may be called `light.0x00158d0004a1b2c3`), and a tool that can only act is one
you cannot use without already knowing the answer.

Tiered `confirm`, and deliberately so: `lock`, `cover`, `alarm_control_panel`
and `climate` services move things in the physical world.

**The token is the keys to the house.** It lives in `config/` or the
environment and never in a source file.

## Messaging now delivers rather than composing

`messaging` opened a `wa.me` or `t.me` link in the browser. That is a fallback,
not delivery: it hands you a pre-filled compose window and waits for you to
press send, so "text Mum I'll be late" ended as a browser tab.

When a bot token is configured it now sends through the corresponding plugin
and the message arrives. When one is not, the link opens exactly as before.

| platform | with a bot token | without |
|---|---|---|
| Telegram | sent via the Bot API | `t.me` compose link |
| Discord | sent via bot or webhook | refused, with what is missing |
| WhatsApp | — | `wa.me` compose link |

WhatsApp is link-only on purpose: it has no free bot API for messaging
arbitrary contacts, so the compose link genuinely is the best available.

Tokens go under `telegram` / `discord` in `config/messaging.json`, or in
`TELEGRAM_BOT_TOKEN` / `DISCORD_BOT_TOKEN`. A dry run never becomes a real
send: `delivery_suppressed()` is checked before either path, and it is true for
the whole test suite.
