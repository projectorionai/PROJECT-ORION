# Letting ORION message people

ORION can reach people three ways. They are genuinely different, and it is worth
knowing which one you are using.

| route | what it does | confirmation |
| --- | --- | --- |
| `phone_action` (kind `call`/`sms`) | opens the dialer/messages **pre-filled** on your paired Android phone | you tap send |
| `messaging` tool | opens a **pre-filled WhatsApp/Telegram chat in a browser** | you press send |
| **messaging plugins** (this page) | **actually delivers** the message over the platform's API | none — so ORION confirms with you first |
| `mcp__twilio__…` | real phone calls / SMS over Twilio | none — see [MCP_SETUP.md](MCP_SETUP.md) |

The plugins below are ordinary ORION plugins (see [PLUGINS.md](PLUGINS.md)): they
declare `network` + `filesystem`, ship at capability tier **`confirm`** (so a
paired remote device must confirm before using them), and hold **no credentials
in their source**.

---

## Where credentials live

`config/messaging.json` — never in source, never in the plugin file:

```json
{
  "discord": {
    "bot_token": "",
    "webhook_url": "",
    "default_channel_id": "",
    "contacts": {
      "dave": { "user_id": "111111111111111111" },
      "team": { "channel_id": "222222222222222222" }
    }
  },
  "telegram": {
    "bot_token": "",
    "default_chat_id": "",
    "contacts": { "mum": "123456789" }
  }
}
```

Environment variables override the file, which is handy for a machine you do not
want the token written on:
`DISCORD_BOT_TOKEN`, `DISCORD_WEBHOOK_URL`, `DISCORD_CHANNEL_ID`,
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

---

## Discord

Two mechanisms — pick either, or set both:

**A webhook** (simplest; posts to one channel, no bot):
1. Discord → Server Settings → Integrations → Webhooks → New Webhook.
2. Copy the URL into `discord.webhook_url`.

**A bot** (needed to DM a person, or post to several channels):
1. <https://discord.com/developers/applications> → New Application → Bot.
2. Copy the token into `discord.bot_token`.
3. Invite the bot to your server with the *Send Messages* permission.
4. Enable Developer Mode in Discord (Settings → Advanced) so you can
   right-click → **Copy ID** for a channel or a person.

Then:

```
"Message Dave on Discord and say I'm running ten minutes late"
"Post that to the team Discord channel"
```

`discord_message status` reports what is configured; `discord_message contacts`
lists who ORION can reach.

> A bot can only DM someone who shares a server with it and has DMs open — if
> Discord refuses, ORION tells you exactly that rather than pretending it sent.

---

## Telegram

1. Message **@BotFather** on Telegram → `/newbot` → copy the token into
   `telegram.bot_token`.
2. **Message your bot once from each account that should receive messages** —
   Telegram will not let a bot open a conversation first.
3. Get the chat id (e.g. message @userinfobot, or read
   `https://api.telegram.org/bot<TOKEN>/getUpdates`) and save it under
   `telegram.contacts`.

```
"Text mum on Telegram that I'll call after dinner"
```

Unlike the older browser route, this is *delivered* — nothing opens on screen.

---

## Adding another platform

Slack, Matrix, Signal-CLI and most others are the same shape: an HTTPS POST with
a token. Copy `config/custom_tools/telegram_message_tool.py`, change the endpoint
and the config section, and drop a manifest beside it — or start from a scaffold:

```
plugin create name=slack_message description="Send a Slack message" tier=confirm
```

Then `plugin audit slack_message` will tell you whether what you wrote matches
what you declared.
