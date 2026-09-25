"""
Tests for OrionDispatcher.chess_tool — the routing layer between the
model-facing 'chess' tool and ChessService.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.dispatcher import OrionDispatcher


class _Signal:
    def __init__(self) -> None:
        self.emitted: list[tuple] = []

    def emit(self, *a, **k) -> None:
        self.emitted.append((a, k))

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __init__(self) -> None:
        self.gui_command = _Signal()

    def __getattr__(self, name):
        return _Signal()


class _StubBrain:
    """ORION's own engine (chess_brain.py): the board state names his rating
    when he is the opponent, which is the default."""

    def rating_text(self) -> str:
        return "1200 (provisional)"

    def describe(self) -> str:
        return "brain"


class _StubChess:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.opponent = "orion"
        self.brain = _StubBrain()

    async def new_game(self, elo=None, player_colour="white"):
        self.calls.append(("new_game", (elo, player_colour)))
        return ToolResult(f"New game, elo={elo}, colour={player_colour}")

    async def make_move(self, move):
        self.calls.append(("make_move", (move,)))
        return ToolResult(f"played {move}")

    async def set_elo(self, elo):
        self.calls.append(("set_elo", (elo,)))
        return ToolResult(f"elo set to {elo}")

    async def resign(self):
        self.calls.append(("resign", ()))
        return ToolResult("resigned")

    async def analyse_move(self, move_number=None):
        self.calls.append(("analyse_move", (move_number,)))
        return ToolResult("move analysis")

    async def analyse_game(self):
        self.calls.append(("analyse_game", ()))
        return ToolResult("game analysis")

    def board_state(self):
        self.calls.append(("board_state", ()))
        return {
            "available": True, "active": True, "turn": "white", "elo": 1500,
            "history_san": ["e4", "e5"], "status": "in progress",
        }

    # Navigation, for "go back three moves" / "show move 12" / "flip the board".
    def step(self, delta):
        self.calls.append(("step", (delta,)))
        return ToolResult(f"stepped {delta}")

    def goto_move(self, number, colour=None):
        self.calls.append(("goto_move", (number, colour)))
        return ToolResult(f"move {number}")

    def goto_ply(self, ply):
        self.calls.append(("goto_ply", (ply,)))
        return ToolResult(f"ply {ply}")

    def resume_live(self):
        self.calls.append(("resume_live", ()))
        return ToolResult("live")

    def flip(self):
        self.calls.append(("flip", ()))
        return ToolResult("flipped")


def _dispatcher(chess=None) -> OrionDispatcher:
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _StubBus()
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)
    d.chess = chess
    return d


def test_unavailable_when_service_not_wired():
    d = _dispatcher(chess=None)
    result = asyncio.run(d.chess_tool({"action": "board_state"}))
    assert not result.ok


def test_routes_new_game_with_elo_and_colour():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "new_game", "elo": 1800, "player_colour": "black"}))
    assert chess.calls == [("new_game", (1800, "black"))]
    assert d.bus.gui_command.emitted == [(({"action": "chess", "target": ""},), {})]


def test_new_game_defaults_elo_to_none_and_colour_to_white():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "new_game"}))
    assert chess.calls == [("new_game", (None, "white"))]


def test_routes_move():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "move", "move": "e2e4"}))
    assert chess.calls == [("make_move", ("e2e4",))]


def test_move_without_a_move_argument_is_refused():
    chess = _StubChess()
    d = _dispatcher(chess)
    result = asyncio.run(d.chess_tool({"action": "move"}))
    assert not result.ok
    assert chess.calls == []


def test_routes_set_elo():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "set_elo", "elo": 2200}))
    assert chess.calls == [("set_elo", (2200,))]


def test_set_elo_without_a_value_is_refused():
    chess = _StubChess()
    d = _dispatcher(chess)
    result = asyncio.run(d.chess_tool({"action": "set_elo"}))
    assert not result.ok


def test_routes_resign():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "resign"}))
    assert chess.calls == [("resign", ())]


def test_routes_analyse_move_with_move_number():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "analyse_move", "move_number": 3}))
    assert chess.calls == [("analyse_move", (3,))]


def test_routes_analyse_move_without_move_number():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "analyse_move"}))
    assert chess.calls == [("analyse_move", (None,))]


def test_routes_analyse_game():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "analyse_game"}))
    assert chess.calls == [("analyse_game", ())]


def test_routes_board_state():
    chess = _StubChess()
    d = _dispatcher(chess)
    result = asyncio.run(d.chess_tool({"action": "board_state"}))
    assert result.ok
    assert "white to move" in result.text.lower()
    assert "e4" in result.text


def test_board_state_with_no_active_game_reports_cleanly():
    class _NoGameChess(_StubChess):
        def board_state(self):
            return {"available": False}

    d = _dispatcher(_NoGameChess())
    result = asyncio.run(d.chess_tool({"action": "board_state"}))
    assert not result.ok


def test_unknown_action_reports_cleanly():
    d = _dispatcher(_StubChess())
    result = asyncio.run(d.chess_tool({"action": "not_a_real_action"}))
    assert not result.ok
    assert "chess" in result.text.lower()


def test_default_action_is_board_state():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({}))
    assert chess.calls == [("board_state", ())]


# ── moving through the game by voice ────────────────────────────────────────

def test_routes_step_back_and_forward_with_a_count():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "back", "count": 3}))
    asyncio.run(d.chess_tool({"action": "forward"}))
    assert chess.calls == [("step", (-3,)), ("step", (1,))]


def test_routes_goto_move_live_first_and_flip():
    chess = _StubChess()
    d = _dispatcher(chess)
    asyncio.run(d.chess_tool({"action": "goto_move", "move_number": 12, "colour": "black"}))
    asyncio.run(d.chess_tool({"action": "first"}))
    asyncio.run(d.chess_tool({"action": "live"}))
    asyncio.run(d.chess_tool({"action": "flip"}))
    assert chess.calls == [("goto_move", (12, "black")), ("goto_ply", (0,)),
                           ("resume_live", ()), ("flip", ())]


def test_goto_without_a_number_asks_which():
    chess = _StubChess()
    result = asyncio.run(_dispatcher(chess).chess_tool({"action": "goto_move"}))
    assert not result.ok and chess.calls == []


def test_a_garbled_count_steps_once_rather_than_failing():
    chess = _StubChess()
    asyncio.run(_dispatcher(chess).chess_tool({"action": "back", "count": "three"}))
    assert chess.calls == [("step", (-1,))]


def test_the_schema_promises_words_not_centipawns():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    schema = next(s for s in TOOL_DECLARATIONS if s["name"] == "chess")
    text = schema["description"]
    assert "average centipawn loss" not in text.lower()
    assert "preferred line" not in text
    for action in ("back", "forward", "goto_move", "live", "flip"):
        assert action in schema["parameters"]["properties"]["action"]["description"]
