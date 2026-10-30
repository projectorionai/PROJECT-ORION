# Let your friends play ORION on Lichess

ORION can take chess challenges from your friends over the internet through
**Lichess** (lichess.org, free). Your friends play from any browser or phone;
ORION plays with his own chess engine, the same one you play on the Chess page.

**How it connects:** ORION only makes *outgoing* connections to lichess.org,
the same way a web browser does. Nothing on your PC is opened to the
internet: no ports, no router settings, and your friends never connect to
your computer. They connect to Lichess, and ORION talks to Lichess.

**What you need:** ORION running on your PC while the games happen, and about
10 minutes for the one-off setup below.

---

## 1. Create a Lichess account just for ORION

1. Sign out of your own Lichess account, or use a private browser window.
2. Go to <https://lichess.org/signup> and create a **new** account, for
   example `OrionBot`.
3. **Don't play any games on it.** The next step only works on an account
   that has never played.

> Use a new account, never your own. Step 3 turns the account into a
> computer-player (BOT) account, and that change is permanent.

## 2. Create a token for ORION

1. While signed in as the new account, open
   <https://lichess.org/account/oauth/token/create>.
2. Give it a description such as `ORION`.
3. Tick **only** "Play bot moves" (`bot:play`). ORION needs
   nothing else.
4. Click **Create** and copy the token. It starts with `lip_`.
   Treat it like a password.

## 3. Turn the account into a BOT account (one time, permanent)

Lichess only lets computer players use its Bot API from accounts marked as
BOTs. Conversion is **permanent** and requires an account that has **never
played a game**. Keep your personal playing account unchanged and use the new
dedicated account from step 1.

Open **PowerShell** in the ORION project folder. First check which account the
token belongs to. Paste the token only into the helper's hidden prompt:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\setup_lichess_bot.ps1
```

This command only verifies the account; it does not convert it. If it already
reports **BOT**, continue to step 4.

To convert the new account, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\setup_lichess_bot.ps1 -Upgrade
```

The helper identifies the authenticated account, warns about permanent
conversion and asks you to type its **exact username**. Press Enter to cancel.
It refuses a known nonzero game count before offering conversion. Where the
account response omits that count, Lichess checks eligibility itself.

Success is reported only after a fresh account lookup confirms **BOT** for the
same username. If the result is uncertain, run the first verification command
again before attempting another upgrade. Each HTTPS request has a 15-second
timeout; redirects are refused. Tokens, headers and server error bodies are
not displayed or saved. The token exists briefly in process memory so the
authenticated request can be made; the helper does not write ORION's key store.

The helper uses Lichess's official [account lookup](https://lichess.org/api#tag/Account/operation/accountMe)
and [BOT upgrade](https://lichess.org/api#tag/Bot/operation/botAccountUpgrade) endpoints.

Before you run it, know what it does:

- It **cannot be undone**.
- It only works if the account has **never played a game**.
- The account can no longer play in the normal lobby or with the Lichess
  website's play buttons. It can only accept or send challenges, which is
  exactly what ORION needs.

## 4. Give ORION the token

ORION keeps his keys in one key store, `api_keys.json`. For the installed app
it lives at `C:\Users\<you>\ORION\config\api_keys.json`. Open it in a text
editor and add a `lichess` entry inside `"integrations"`:

```json
{
  "integrations": {
    "lichess": { "bot_token": "lip_YOUR_TOKEN" }
  }
}
```

If `"integrations"` is already there, add only the `"lichess": { ... }` line,
with a comma after the entry before it. Preserve every existing provider and
integration key; do not replace the whole file with the example above. Save
the file and restart ORION.

ORION sends the token only to lichess.org, in the request header. It is never
written to his logs, shown on screen, or said aloud. Don't paste the token
into a chat with ORION; the file is the right place for it.

*(Alternative: set the environment variable `ORION_LICHESS_BOT_TOKEN`.)*

## 5. Choose who may challenge ORION

By default **nobody** can challenge ORION until you name them. Tell him:

- "Let **sam_chess** challenge you on Lichess." Use your friend's exact
  Lichess username.
- "Remove **sam_chess** from your Lichess list."
- "Accept Lichess challenges from anyone." To go back: "…only from my
  friends".

The list is saved in `config/chess_lichess.json` inside the ORION folder.

## 6. Play

- **"Start taking chess challenges."** ORION goes online on Lichess and tells
  you the link your friends need: `https://lichess.org/@/OrionBot`.
- Your friend opens that link, clicks **Challenge to a game**, chooses
  **Standard** chess with a real-time clock of **3 minutes or more** (for
  example 5+3 or 10+0), and sends it. ORION accepts and the game starts at
  once.
- Or ORION challenges them: **"Challenge sam_chess to a 10 minute game."**
  You can add "with 5 seconds increment", "rated", or "you play white".
- **"How's my Lichess game going?"** or **"Lichess status"** tells you who
  he's playing and the link to watch live. The Chess page in the Command Deck
  also shows a Lichess status line and his games' key moments.
- **"Stop taking challenges."** ORION stops accepting new games and still
  finishes a game in progress. **"Resign the Lichess game"** ends it now.

### What ORION accepts (the safe defaults)

| Setting (in `config/chess_lichess.json`) | Default | Meaning |
|---|---|---|
| `allow` | empty | Lichess usernames that may challenge him |
| `accept_anyone` | `false` | `true` = anyone may challenge him |
| `accept_rated` | `true` | allow rated games as well as casual ones |
| `min_initial_seconds` | `180` | clocks faster than 3 minutes are declined; he thinks up to 1.5 s a move |
| `max_games` | `1` | one game at a time (a real 1v1) |
| `move_seconds` | `1.5` | his longest think per move (the strength he was measured at) |
| `greeting` | a short hello | sent once in the game chat; `""` turns it off |

Correspondence and unlimited games are always declined. They can last days,
and ORION would lose them whenever your PC is off.

## Good to know

- **Keep the PC awake during a game.** If it sleeps or ORION closes mid-game,
  his clock runs out on Lichess and he loses.
- He plays with **his own engine**, never Stockfish, and learns from every
  finished game, just like the games you play him at home.
- How strong he is: ORION measures this against Stockfish at known strengths.
  Ask him "what's your chess rating?"
- **Troubleshooting**
  - "Isn't a BOT account yet": step 3 hasn't been done on the account the
    token belongs to.
  - "Rejected my bot token": the token was deleted or lacks `bot:play`.
    Make a new one (step 2) and replace it in the key store.
  - A friend's challenge is declined: check their exact username is on the
    list (step 5), and that the challenge is standard chess, 3+ minutes, and
    not correspondence.
