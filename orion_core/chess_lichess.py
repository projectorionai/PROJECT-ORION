"""
Lichess BOT bridge — ORION plays your friends on Lichess, from your own PC.

  "I want him to remotely go against some of my friends in chess ... I'll need
   to give them some type of access to go against him — ORION will be running
   on my computer at the time, so a 1v1 should work."

How it works
------------
ORION's Lichess account (a BOT account — see docs/CHESS_LICHESS_BOT.md) is the
access your friends get: they challenge it on lichess.org like any player,
from any browser or phone. This module is an OUTBOUND-ONLY client of the
Lichess Bot API — it opens long-lived HTTPS streams to lichess.org and posts
moves back. Nothing listens on the user's PC; no port is opened, no router is
touched, and a friend never connects to the PC itself.

  events     GET /api/stream/event (ndjson): challenges and game starts. On
             connect Lichess re-sends every open challenge and game in
             progress, so a dropped connection loses nothing — it is simply
             reopened with exponential backoff.
  policy     A challenge is accepted only from a username on the allow-list
             (or anyone, if the user turned that on), for standard chess at a
             clock ORION can keep up with (default 3+0 or slower: he thinks up
             to 1.5 s a move), and only while he is not already playing
             (default: one game at a time). Everything else is declined with
             the reason Lichess shows the challenger.
  games      GET /api/bot/game/stream/{id} (ndjson); ORION's own engine
             (chess_brain.py — never Stockfish) chooses each move on a worker
             thread, with a think time taken from his clock, and posts it.
             A greeting goes to the chat once; a draw offer is taken only when
             he is clearly worse; a finished game is learned from like any
             other.

The token
---------
Read from ORION's key store (``integrations.lichess.bot_token`` in
api_keys.json) or the ORION_LICHESS_BOT_TOKEN environment variable. It is
sent only in the Authorization header to lichess.org and never logged,
echoed, or written anywhere else.

The allow-list and preferences are not secrets and live in
config/chess_lichess.json, editable by voice ("let Sam challenge you").
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

try:
    import chess
    CHESS_AVAILABLE = True
except Exception:  # pragma: no cover
    chess = None  # type: ignore
    CHESS_AVAILABLE = False

from .constants import CONFIG_DIR
from .utils import first_line

LICHESS_URL = "https://lichess.org"
SETTINGS_PATH = CONFIG_DIR / "chess_lichess.json"
TOKEN_ENV = "ORION_LICHESS_BOT_TOKEN"

#: Lichess usernames: 2-30 letters, digits, '_' or '-'. Checked before a name
#: goes into a URL path, so a spoken name can never become a different request.
_USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,29}$")
_GAME_OVER = {"aborted", "mate", "resign", "stalemate", "timeout", "draw", "outoftime",
              "cheat", "noStart", "unknownFinish", "variantEnd", "insufficientMaterialClaim"}


def read_token() -> str:
    """ORION's Lichess bot token from the key store, or '' if there is none."""
    env = os.getenv(TOKEN_ENV, "").strip()
    if env:
        return env
    try:
        from .constants import API_CONFIG_PATH
        from .key_seal import load_json
        data = load_json(API_CONFIG_PATH)
    except (OSError, ValueError, ImportError):
        return ""
    integrations = data.get("integrations") if isinstance(data, dict) else None
    if not isinstance(integrations, dict):
        return ""
    raw = integrations.get("lichess")
    if isinstance(raw, dict):
        token = raw.get("bot_token") or raw.get("token")
    elif isinstance(raw, str):
        token = raw
    else:
        token = integrations.get("lichess_bot_token")
    return str(token or "").strip()


def valid_username(name: str) -> bool:
    return bool(_USERNAME.match(str(name or "").strip()))


# ── preferences ───────────────────────────────────────────────────────────────

@dataclass
class LichessSettings:
    """Who may challenge ORION, and on what terms. Safe by default: nobody
    until named, casual and rated both allowed, one game at a time, and no
    clock faster than ORION can think."""

    allow: list[str] = field(default_factory=list)       # lowercase Lichess usernames
    accept_anyone: bool = False
    accept_rated: bool = True
    min_initial_seconds: int = 180                        # 3+0 blitz and slower
    max_initial_seconds: int = 3 * 3600
    max_games: int = 1
    move_seconds: float = 1.5                             # his usual think time
    greeting: str = "Hi! I'm ORION, playing with my own chess engine. Good luck and have fun!"

    @staticmethod
    def load(path: Path) -> "LichessSettings":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return LichessSettings()
        settings = LichessSettings()
        for key, value in (data or {}).items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        settings.allow = sorted({str(n).strip().lower() for n in settings.allow
                                 if valid_username(str(n))})
        return settings

    def save(self, path: Path) -> None:
        from .atomic_io import atomic_write_text
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(Path(path), json.dumps(asdict(self), indent=2), encoding="utf-8")


def challenge_decision(challenge: dict[str, Any], settings: LichessSettings,
                       games_in_progress: int) -> tuple[bool, str]:
    """(accept?, Lichess decline reason or '') for an incoming challenge."""
    challenger = challenge.get("challenger") or {}
    name = str(challenger.get("id") or challenger.get("name") or "").lower()
    variant = (challenge.get("variant") or {}).get("key", "standard")
    if variant != "standard":
        return False, "standard"
    if not settings.accept_anyone and name not in settings.allow:
        return False, "generic"
    control = challenge.get("timeControl") or {}
    if control.get("type") != "clock":
        return False, "timeControl"          # correspondence/unlimited: ORION may be off
    limit = int(control.get("limit") or 0)
    if limit < int(settings.min_initial_seconds):
        return False, "tooFast"
    if limit > int(settings.max_initial_seconds):
        return False, "tooSlow"
    if challenge.get("rated") and not settings.accept_rated:
        return False, "casual"
    if games_in_progress >= int(settings.max_games):
        return False, "later"
    return True, ""


def think_time(state: dict[str, Any], white: bool, cap: float) -> float:
    """Seconds to think this move, from the clock: about a fortieth of what is
    left plus most of the increment, never more than *cap* (ORION's usual
    1.5 s — the strength he was measured at) and never so much that a short
    clock runs out."""
    remaining = float(state.get("wtime" if white else "btime") or 0) / 1000.0
    increment = float(state.get("winc" if white else "binc") or 0) / 1000.0
    if remaining <= 0:
        return min(cap, 1.0)                  # untimed state: play at his usual pace
    budget = remaining / 40.0 + increment * 0.7
    if remaining < 15:
        budget = min(budget, 0.2)
    return max(0.05, min(cap, budget))


# ── the HTTP client (outbound only) ──────────────────────────────────────────

class LichessError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Lichess said {status}: {message}")
        self.status = status


class LichessAuthError(LichessError):
    pass


class LichessRateLimited(LichessError):
    pass


class LichessStopUnconfirmed(RuntimeError):
    """A stop request failed without evidence that the remote game ended."""


async def iter_ndjson(lines: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Objects from a newline-delimited JSON stream. Blank lines are Lichess's
    keep-alives (every ~7 s); an undecodable line is skipped, not fatal."""
    async for raw in lines:
        text = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


class LichessClient:
    """The few Bot API calls ORION needs, over aiohttp. The token goes only in
    the Authorization header; errors carry Lichess's message, never it."""

    def __init__(self, token: str, base_url: str = LICHESS_URL) -> None:
        self._token = token
        self.base_url = base_url.rstrip("/")
        self._session: Any = None

    async def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._token}",
                         "User-Agent": "ORION-chess-bot (python-chess; aiohttp)"})
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    @staticmethod
    async def _raise_for(resp: Any) -> None:
        if 300 <= resp.status < 400:
            raise LichessError(resp.status, "Bot API redirect refused")
        if resp.status < 400:
            return
        try:
            body = await resp.text()
            detail = json.loads(body).get("error") or body
        except Exception:
            detail = resp.reason or "error"
        detail = first_line(str(detail), 160)
        if resp.status in (401, 403):
            raise LichessAuthError(resp.status, detail)
        if resp.status == 429:
            raise LichessRateLimited(resp.status, detail)
        raise LichessError(resp.status, detail)

    async def _request(self, method: str, path: str, data: dict | None = None) -> Any:
        import aiohttp
        session = await self._get_session()
        async with session.request(method, self.base_url + path, data=data,
                                   allow_redirects=False,
                                   timeout=aiohttp.ClientTimeout(total=30)) as resp:
            await self._raise_for(resp)
            try:
                return await resp.json(content_type=None)
            except Exception:
                return {}

    async def _stream(self, path: str, on_open: Callable[[], None] | None = None
                      ) -> AsyncIterator[dict[str, Any]]:
        import aiohttp
        session = await self._get_session()
        # No total timeout — these streams are meant to stay open — but a
        # silent socket (no keep-alive for 60 s) is a dead one: reconnect.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=60)
        async with session.get(self.base_url + path, timeout=timeout,
                               allow_redirects=False) as resp:
            await self._raise_for(resp)
            if on_open is not None:
                on_open()
            async for obj in iter_ndjson(resp.content):
                yield obj

    # the API
    async def account(self) -> dict[str, Any]:
        return await self._request("GET", "/api/account")

    def stream_events(self, on_open: Callable[[], None] | None = None
                      ) -> AsyncIterator[dict[str, Any]]:
        return self._stream("/api/stream/event", on_open)

    def stream_game(self, game_id: str) -> AsyncIterator[dict[str, Any]]:
        return self._stream(f"/api/bot/game/stream/{game_id}")

    async def accept_challenge(self, challenge_id: str) -> None:
        await self._request("POST", f"/api/challenge/{challenge_id}/accept")

    async def decline_challenge(self, challenge_id: str, reason: str = "generic") -> None:
        await self._request("POST", f"/api/challenge/{challenge_id}/decline",
                            {"reason": reason or "generic"})

    async def make_move(self, game_id: str, uci: str) -> None:
        await self._request("POST", f"/api/bot/game/{game_id}/move/{uci}")

    async def chat(self, game_id: str, text: str, room: str = "player") -> None:
        await self._request("POST", f"/api/bot/game/{game_id}/chat",
                            {"room": room, "text": text[:140]})

    async def accept_draw(self, game_id: str) -> None:
        await self._request("POST", f"/api/bot/game/{game_id}/draw/yes")

    async def abort(self, game_id: str) -> None:
        await self._request("POST", f"/api/bot/game/{game_id}/abort")

    async def resign(self, game_id: str) -> None:
        await self._request("POST", f"/api/bot/game/{game_id}/resign")

    async def challenge(self, username: str, limit: int, increment: int, rated: bool,
                        colour: str) -> dict[str, Any]:
        return await self._request("POST", f"/api/challenge/{username}", {
            "rated": "true" if rated else "false", "clock.limit": str(int(limit)),
            "clock.increment": str(int(increment)), "color": colour,
            "variant": "standard"})


# ── the bridge ────────────────────────────────────────────────────────────────

Think = Callable[[Any, float], Awaitable[tuple[Any, Optional[int]]]]


@dataclass
class LichessGame:
    game_id: str
    opponent: str = "?"
    colour: str = "white"              # ORION's colour
    rated: bool = False
    clock: str = ""
    moves: str = ""
    status: str = "started"
    last_score: Optional[int] = None   # ORION's view, from his last search
    started: float = field(default_factory=time.time)


class LichessBridge:
    """ORION's presence on Lichess: accept friends' challenges, play them with
    his own engine, and report to the chess page. One instance per ChessService."""

    BACKOFF_START = 1.0
    BACKOFF_MAX = 60.0
    RATE_LIMIT_WAIT = 65.0

    def __init__(self, bus: Any, think: Think,
                 settings_path: Path | None = None,
                 client_factory: Callable[[str], Any] | None = None,
                 token_reader: Callable[[], str] = read_token,
                 on_finish: Callable[[str, str, LichessGame], None] | None = None) -> None:
        self.bus = bus
        self.think = think
        self.settings_path = Path(settings_path) if settings_path is not None else SETTINGS_PATH
        self.settings = LichessSettings.load(self.settings_path)
        self._client_factory = client_factory or (lambda token: LichessClient(token))
        self._token_reader = token_reader
        self._on_finish = on_finish
        self.client: Any = None
        self.username = ""
        self.running = False
        self.connected = False
        self.last_error = ""
        self.games: dict[str, LichessGame] = {}
        self.finished: list[str] = []
        self._events_task: Any = None
        self._game_tasks: dict[str, Any] = {}
        self._backoff = self.BACKOFF_START

    # ── reporting ────────────────────────────────────────────────────────────

    def _say(self, text: str, **extra: Any) -> None:
        try:
            self.bus.log.emit(f"LICHESS: {text}")
        except Exception:
            pass
        try:
            self.bus.dashboard_event.emit("chess", {"phase": "lichess", "text": text, **extra})
        except Exception:
            pass

    def status_text(self) -> str:
        if not self.running:
            state = "not taking challenges"
            if self.last_error:
                state += f" (last problem: {self.last_error})"
        else:
            state = ("online and taking challenges" if self.connected
                     else "reconnecting to Lichess")
            state = f"@{self.username} is {state}" if self.username else state
        who = ("anyone" if self.settings.accept_anyone
               else (", ".join(self.settings.allow) or "nobody yet — add a friend first"))
        lines = [f"Lichess: {state}. Who may challenge me: {who}."]
        for game in self.games.values():
            ply = len(game.moves.split()) if game.moves else 0
            lines.append(f"Playing {game.opponent} as {game.colour} ({game.clock}, "
                         f"{'rated' if game.rated else 'casual'}), move {ply // 2 + 1}: "
                         f"https://lichess.org/{game.game_id}")
        if self.finished:
            lines.append("Recent: " + "; ".join(self.finished[-3:]))
        return "\n".join(lines)

    # ── preferences ──────────────────────────────────────────────────────────

    def allow(self, username: str) -> str:
        name = str(username or "").strip().lstrip("@").lower()
        if not valid_username(name):
            return f"'{username}' isn't a valid Lichess username."
        if name not in self.settings.allow:
            self.settings.allow = sorted(set(self.settings.allow) | {name})
            self.settings.save(self.settings_path)
        return f"{name} can challenge me on Lichess now."

    def disallow(self, username: str) -> str:
        name = str(username or "").strip().lstrip("@").lower()
        if name in self.settings.allow:
            self.settings.allow = [n for n in self.settings.allow if n != name]
            self.settings.save(self.settings_path)
            return f"{name} is off my Lichess allow-list."
        return f"{name} wasn't on my Lichess allow-list."

    def set_accept_anyone(self, value: bool) -> str:
        self.settings.accept_anyone = bool(value)
        self.settings.save(self.settings_path)
        return ("I'll accept a challenge from anyone on Lichess (still standard chess, "
                "3 minutes or slower, one game at a time)." if value
                else "Only people on my allow-list can challenge me on Lichess now.")

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> tuple[bool, str]:
        if self.running:
            return True, self.status_text()
        token = self._token_reader()
        if not token:
            return False, ("I don't have a Lichess bot token yet. Put it in my key store as "
                           "integrations → lichess → bot_token (see docs/CHESS_LICHESS_BOT.md).")
        if self.client is None:          # still open if a game outlived a stop()
            self.client = self._client_factory(token)
        try:
            account = await self.client.account()
        except LichessAuthError:
            await self._close_client()
            self.last_error = "Lichess rejected the token"
            return False, ("Lichess rejected my bot token — it may be revoked, or missing the "
                           "bot:play scope.")
        except Exception as exc:
            await self._close_client()
            self.last_error = first_line(exc, 120)
            return False, f"I couldn't reach Lichess ({self.last_error})."
        if str(account.get("title") or "").upper() != "BOT":
            await self._close_client()
            return False, (f"The Lichess account @{account.get('username', '?')} isn't a BOT "
                           "account yet, so the Bot API won't let it play. The setup guide "
                           "(docs/CHESS_LICHESS_BOT.md) shows the one-time upgrade.")
        self.username = str(account.get("username") or account.get("id") or "")
        self.running = True
        self.last_error = ""
        self._events_task = asyncio.get_running_loop().create_task(
            self._event_loop(), name="orion-lichess-events")
        self._say(f"online as @{self.username}", online=True)
        return True, (f"I'm on Lichess as @{self.username}, taking challenges from "
                      f"{'anyone' if self.settings.accept_anyone else ', '.join(self.settings.allow) or 'nobody yet'}."
                      f" Friends challenge https://lichess.org/@/{self.username} — "
                      "3 minutes or slower, standard chess.")

    async def stop(self) -> str:
        """Stop taking challenges. A game in progress is played to the end —
        walking out of it would lose on time and be rude to a friend."""
        self.running = False
        self.connected = False
        task = self._events_task
        self._events_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if not self._game_tasks:
            await self._close_client()
        self._say("stopped taking challenges", online=False)
        if self._game_tasks:
            return ("I've stopped taking new challenges on Lichess; I'll finish the game in "
                    "progress first.")
        return "I've stopped taking challenges on Lichess."

    def shutdown(self) -> None:
        """ORION is closing: cancel everything without waiting (a game in
        progress is then lost on time on Lichess — resign first if it matters)."""
        self.running = False
        for task in [self._events_task, *self._game_tasks.values()]:
            if task is not None and not task.done():
                try:
                    task.cancel()
                except Exception:
                    pass
        self._events_task = None
        self._game_tasks.clear()
        client, self.client = self.client, None
        if client is not None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass                          # no loop left: the process is exiting
            else:
                # Owned: a discarded task could be collected before the
                # HTTP session closed ("Unclosed client session").
                from . import background
                background.spawn(client.close(), name="lichess: close client")

    async def _close_client(self) -> None:
        if self.client is not None:
            try:
                await self.client.close()
            except Exception:
                pass
        self.client = None

    # ── actions ──────────────────────────────────────────────────────────────

    async def challenge(self, username: str, minutes: float = 5, increment: int = 3,
                        rated: bool = False, colour: str = "random") -> tuple[bool, str]:
        name = str(username or "").strip().lstrip("@")
        if not valid_username(name):
            return False, f"'{username}' isn't a valid Lichess username."
        if not self.running:
            ok, text = await self.start()
            if not ok:
                return False, text
        # Lichess takes whole minutes here (multiples of 60 s); 3 is ORION's floor.
        limit = max(180, min(10_800, int(round(float(minutes))) * 60))
        increment = max(0, min(60, int(increment)))
        colour = colour if colour in ("white", "black", "random") else "random"
        try:
            await self.client.challenge(name, limit, increment, bool(rated), colour)
        except Exception as exc:
            return False, f"Lichess wouldn't send that challenge ({first_line(exc, 120)})."
        # The friend has to be able to challenge back / rematch too.
        if not self.settings.accept_anyone and name.lower() not in self.settings.allow:
            self.allow(name)
        self._say(f"challenged {name} ({limit // 60}+{increment})")
        return True, (f"Challenge sent to {name}: {limit // 60}+{increment}, "
                      f"{'rated' if rated else 'casual'}, I play {colour}. The game starts "
                      "as soon as they accept on lichess.org.")

    async def resign_all(self) -> str:
        if not self.games:
            return "I'm not playing anyone on Lichess."
        confirmed, failed = 0, 0
        for game_id in list(self.games):
            try:
                await self.client.resign(game_id)
            except Exception:
                try:
                    await self.client.abort(game_id)
                except Exception:
                    failed += 1
                    continue
            confirmed += 1
        if failed:
            # The HTTP exception or body can contain private details. Report
            # counts only, and let the caller return a failed tool receipt.
            raise LichessStopUnconfirmed(
                f"Lichess could not confirm stopping {failed} game(s); "
                f"{confirmed} stop request(s) confirmed. Check the remote game status before retrying.")
        return f"Lichess confirmed {confirmed} stop request(s). Waiting for the final game state."

    # ── the event stream ─────────────────────────────────────────────────────

    def _on_open(self) -> None:
        self.connected = True
        self._backoff = self.BACKOFF_START

    async def _event_loop(self) -> None:
        while self.running:
            try:
                from contextlib import aclosing
                async with aclosing(self.client.stream_events(self._on_open)) as events:
                    async for event in events:
                        if not self.running:
                            break
                        await self._on_event(event)
            except asyncio.CancelledError:
                raise
            except LichessAuthError:
                self.last_error = "Lichess rejected the token"
                self._say("the token was rejected — stopping", online=False)
                self.running = False
                break
            except LichessRateLimited:
                self.connected = False
                self._say("rate-limited by Lichess — waiting a minute")
                await asyncio.sleep(self.RATE_LIMIT_WAIT)
                continue
            except Exception as exc:
                self.last_error = first_line(exc, 120)
            self.connected = False
            if not self.running:
                break
            await asyncio.sleep(self._backoff)
            self._backoff = min(self.BACKOFF_MAX, self._backoff * 2)

    async def _on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "challenge":
            await self._on_challenge(event.get("challenge") or {})
        elif kind == "gameStart":
            game = event.get("game") or {}
            game_id = str(game.get("gameId") or game.get("id") or "")
            if game_id and game_id not in self._game_tasks:
                self._game_tasks[game_id] = asyncio.get_running_loop().create_task(
                    self._run_game(game_id), name=f"orion-lichess-{game_id}")
        elif kind in ("challengeCanceled", "challengeDeclined"):
            challenge = event.get("challenge") or {}
            if kind == "challengeDeclined" and challenge.get("direction", "out") == "out":
                who = (challenge.get("destUser") or {}).get("name", "they")
                self._say(f"{who} declined my challenge")

    async def _on_challenge(self, challenge: dict[str, Any]) -> None:
        challenge_id = str(challenge.get("id") or "")
        challenger = challenge.get("challenger") or {}
        if not challenge_id or str(challenger.get("id") or "").lower() == self.username.lower():
            return                                  # our own outgoing challenge
        ok, reason = challenge_decision(challenge, self.settings, len(self._game_tasks))
        name = challenger.get("name") or challenger.get("id") or "someone"
        control = (challenge.get("timeControl") or {}).get("show", "")
        try:
            if ok:
                await self.client.accept_challenge(challenge_id)
                self._say(f"accepted {name}'s challenge ({control})")
            else:
                await self.client.decline_challenge(challenge_id, reason)
                self._say(f"declined {name}'s challenge ({reason})")
        except Exception as exc:
            self._say(f"couldn't answer {name}'s challenge ({first_line(exc, 100)})")

    # ── a game ───────────────────────────────────────────────────────────────

    async def _run_game(self, game_id: str) -> None:
        """Play one game to the end. A dropped game stream is reopened with
        backoff — Lichess re-sends the full game on reconnect — because a
        stream that is simply abandoned leaves ORION's clock running until he
        loses on time."""
        game = LichessGame(game_id)
        self.games[game_id] = game
        context = {"initial_fen": chess.STARTING_FEN, "greeted": False, "answered": -1}
        delay = self.BACKOFF_START
        try:
            for _attempt in range(12):
                try:
                    if await self._play_stream(game, context):
                        return                          # the game is over
                except asyncio.CancelledError:
                    raise
                except LichessAuthError:
                    self._say("the token was rejected mid-game", game_id=game_id)
                    return
                except Exception as exc:
                    self._say(f"lost the stream for game {game_id} ({first_line(exc, 100)}) "
                              "— reconnecting", game_id=game_id)
                await asyncio.sleep(delay)
                delay = min(10.0, delay * 2)
            self._say(f"gave up reconnecting to game {game_id}", game_id=game_id)
        finally:
            self.games.pop(game_id, None)
            self._game_tasks.pop(game_id, None)
            if not self.running and not self._game_tasks:
                await self._close_client()

    async def _play_stream(self, game: LichessGame, context: dict[str, Any]) -> bool:
        """Follow the game stream, answering each position that is ORION's to
        move. True when the game has finished."""
        from contextlib import aclosing
        game_id = game.game_id
        async with aclosing(self.client.stream_game(game_id)) as events:
            async for event in events:
                kind = event.get("type")
                if kind == "gameFull":
                    white = event.get("white") or {}
                    black = event.get("black") or {}
                    ours_white = str(white.get("id") or "").lower() == self.username.lower()
                    game.colour = "white" if ours_white else "black"
                    opponent = black if ours_white else white
                    game.opponent = str(opponent.get("name") or opponent.get("id")
                                        or ("Lichess computer level %s" % opponent.get("aiLevel")
                                            if opponent.get("aiLevel") else "?"))
                    game.rated = bool(event.get("rated"))
                    clock = event.get("clock") or {}
                    if clock:
                        game.clock = (f"{int(clock.get('initial', 0)) // 60000}+"
                                      f"{int(clock.get('increment', 0)) // 1000}")
                    fen = event.get("initialFen") or "startpos"
                    context["initial_fen"] = chess.STARTING_FEN if fen == "startpos" else fen
                    state = event.get("state") or {}
                    if not context["greeted"]:
                        context["greeted"] = True
                        self._say(f"game started against {game.opponent} "
                                  f"({game.clock or 'no clock'}, "
                                  f"{'rated' if game.rated else 'casual'}), I'm {game.colour}",
                                  game_id=game_id)
                        if self.settings.greeting and not state.get("moves"):
                            try:
                                await self.client.chat(game_id, self.settings.greeting)
                            except Exception:
                                pass
                elif kind == "gameState":
                    state = event
                elif kind == "opponentGone":
                    if event.get("gone"):
                        self._say(f"{game.opponent} has left the board", game_id=game_id)
                    continue
                else:
                    continue                        # chat lines etc.
                game.moves = str(state.get("moves") or "")
                game.status = str(state.get("status") or "started")
                if game.status in _GAME_OVER:
                    self._finish(game, context["initial_fen"], state)
                    return True
                board = chess.Board(context["initial_fen"])
                for uci in game.moves.split():
                    board.push_uci(uci)
                self._emit_position(game, board)
                ours = (board.turn == chess.WHITE) == (game.colour == "white")
                opponent_offers = state.get("bdraw" if game.colour == "white" else "wdraw")
                if opponent_offers and self._should_accept_draw(game, board):
                    try:
                        await self.client.accept_draw(game_id)
                        continue
                    except Exception:
                        pass
                if not ours or board.is_game_over() or len(board.move_stack) == context["answered"]:
                    continue
                context["answered"] = len(board.move_stack)
                seconds = think_time(state, game.colour == "white", self.settings.move_seconds)
                move, score = await self.think(board, seconds)
                if move is None:
                    continue
                game.last_score = score
                if not await self._send_move(game, move):
                    context["answered"] = -1        # answer the next state instead
        return False

    async def _send_move(self, game: LichessGame, move: Any) -> bool:
        """Post a move, retrying a transient failure briefly — every second
        spent here is on ORION's clock. A move Lichess calls illegal (400) is
        not retried."""
        for delay in (0.0, 0.5, 1.5):
            if delay:
                await asyncio.sleep(delay)
            try:
                await self.client.make_move(game.game_id, move.uci())
                return True
            except LichessAuthError:
                raise
            except LichessError as exc:
                if exc.status == 400:
                    self._say(f"Lichess refused my move {move.uci()} ({exc})",
                              game_id=game.game_id)
                    return False
            except Exception:
                pass
        self._say(f"couldn't send my move {move.uci()}", game_id=game.game_id)
        return False

    def _emit_position(self, game: LichessGame, board: Any) -> None:
        try:
            self.bus.dashboard_event.emit("chess", {
                "phase": "lichess_position", "game_id": game.game_id, "fen": board.fen(),
                "opponent": game.opponent, "colour": game.colour})
        except Exception:
            pass

    @staticmethod
    def _should_accept_draw(game: LichessGame, board: Any) -> bool:
        """Take a draw when ORION is clearly worse, or there is nothing left
        to play for; otherwise play on."""
        if board.is_insufficient_material():
            return True
        return game.last_score is not None and game.last_score <= -150

    def _finish(self, game: LichessGame, initial_fen: str, state: dict[str, Any]) -> None:
        winner = state.get("winner")
        if winner is None:
            result = "1/2-1/2" if game.status not in ("aborted", "noStart") else "*"
        else:
            result = "1-0" if winner == "white" else "0-1"
        if result == "*":
            summary = f"game against {game.opponent} was aborted"
        elif winner is None:
            summary = f"drew with {game.opponent} ({game.status})"
        else:
            won = winner == game.colour
            summary = f"{'beat' if won else 'lost to'} {game.opponent} ({game.status})"
        self.finished.append(summary)
        self.finished = self.finished[-10:]
        self._say(summary, game_id=game.game_id, finished=True)
        if self._on_finish is not None and result != "*" and initial_fen == chess.STARTING_FEN:
            try:
                self._on_finish(game.moves, result, game)
            except Exception:
                pass


__all__ = ["LICHESS_URL", "LichessAuthError", "LichessBridge", "LichessClient",
           "LichessError", "LichessGame", "LichessRateLimited", "LichessSettings", "LichessStopUnconfirmed",
           "SETTINGS_PATH", "TOKEN_ENV", "challenge_decision", "iter_ndjson",
           "read_token", "think_time", "valid_username"]
