"""
Tests for the chess Command Deck page — a plain QPainter board (no
QWebEngineView/CDN dependency) with click-to-select/click-to-move, plus the
ChessPanel that wraps it with move entry, autonomous-play controls, an eval
bar and game-review navigation.

Headless (offscreen Qt) — no display required.

Known gap versus the pre-rewrite version of this file: the old
ChessBoardWidget supported drag-and-drop, a slide animation between
positions, and a flip() board-orientation toggle. Those did not survive an
in-place rewrite of chess_view.py and could not be recovered (no backup of
the original existed) — see the project history around 2026-08-02. The
board still supports click-to-select/click-to-move with server-side legality
checking (illegal attempts are simply refused by ChessService, not
pre-filtered client-side). Restoring drag/animation/flip is a follow-up, not
covered here.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
import chess
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from orion_core.gui import chess_view as chess_view_module
from orion_core.gui.chess_view import ChessBoardWidget, ChessPanel, EvalBarWidget
from orion_core import chess_engine as chess_engine_module
from orion_core.chess_engine import ChessService


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _Signal:
    def __init__(self) -> None:
        self._slots = []

    def emit(self, *args) -> None:
        for slot in self._slots:
            slot(*args)

    def connect(self, slot) -> None:
        self._slots.append(slot)


class _StubBus:
    def __init__(self) -> None:
        self.dashboard_event = _Signal()

    def __getattr__(self, name):
        return _Signal()


class _FakeEngine:
    """Always plays the first legal move; analyse() reports a flat score so
    every move grades as 'best' — commentary/quality text isn't the point
    of these tests, wiring is."""

    def configure(self, options):
        pass

    def play(self, board, limit, **kw):
        move = next(iter(board.legal_moves), None)

        class _Result:
            pass
        result = _Result()
        result.move = move
        return result

    def analyse(self, board, limit, **kw):
        move = next(iter(board.legal_moves), None)

        class _Score:
            def score(self, mate_score=100_000):
                return 0

        class _PovScore:
            def white(self):
                return _Score()

            def black(self):
                return _Score()

        return {"pv": [move] if move is not None else [], "score": _PovScore()}

    def quit(self):
        pass


def _service(monkeypatch, tmp_path) -> ChessService:
    """A real ChessService with the engine layer faked out — same pattern as
    test_chess_engine.py, so ChessPanel is exercised against its actual
    collaborator rather than a hand-maintained stub that can drift out of
    sync with ChessService's real shape."""
    engine_path = tmp_path / "stockfish.exe"
    engine_path.write_bytes(b"fake")
    monkeypatch.setattr(chess_engine_module, "ENGINE_PATH", engine_path)
    service = ChessService(_StubBus())
    fake_engine = _FakeEngine()

    async def _fake_ensure_engine():
        if service._engine is None:
            service._engine = fake_engine
            await service._apply_elo(service.elo)
        return service._engine

    service._ensure_engine = _fake_ensure_engine
    # The fake engine plays and grades, so replies land inside _drain(). By
    # default ORION now answers and grades with his own search, on a thread
    # and on the clock — under a loaded full-suite run his reply sometimes
    # arrived after the test had read the position ("assert 2 == 0").
    service.opponent = "engine"
    service.analyst = "stockfish"
    return service


async def _drain() -> None:
    """Let fire-and-forget asyncio.create_task() calls (the panel's action
    handlers) actually run before assertions inspect their effects. A real
    (if tiny) sleep, not sleep(0): several of these paths cross
    asyncio.to_thread, which needs actual wall-clock time for the executor
    thread to finish and post its result back to the loop."""
    for _ in range(10):
        await asyncio.sleep(0.02)


def _click(widget: ChessBoardWidget, square_name: str) -> None:
    file_index = "abcdefgh".index(square_name[0])
    rank_index = 8 - int(square_name[1])
    size = widget._square_size()
    pos = QPointF((file_index + 0.5) * size, (rank_index + 0.5) * size)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, pos, pos,
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(event)


# ── ChessBoardWidget: rendering + interaction ────────────────────────────────

def test_set_position_loads_the_position(_app):
    widget = ChessBoardWidget()
    widget.resize(400, 400)
    widget.set_position(chess.STARTING_FEN)
    assert widget._fen == chess.STARTING_FEN
    widget.grab()   # must paint without raising


def test_click_click_emits_a_move_request(_app):
    widget = ChessBoardWidget()
    widget.resize(400, 400)
    widget.set_position(chess.STARTING_FEN)
    moves: list[str] = []
    widget.move_requested.connect(moves.append)
    _click(widget, "e2")
    assert widget._selected == "e2"
    _click(widget, "e4")
    assert moves == ["e2e4"]
    assert widget._selected is None   # selection clears after committing


def test_clicking_the_same_square_twice_deselects(_app):
    widget = ChessBoardWidget()
    widget.resize(400, 400)
    widget.set_position(chess.STARTING_FEN)
    _click(widget, "e2")
    assert widget._selected == "e2"
    _click(widget, "e2")
    assert widget._selected is None


def test_promotion_move_auto_queens(_app):
    widget = ChessBoardWidget()
    widget.resize(400, 400)
    # White pawn on a7, a8 empty and unopposed — both kings well clear.
    widget.set_position("8/P7/8/8/8/8/7k/K7 w - - 0 1")
    moves: list[str] = []
    widget.move_requested.connect(moves.append)
    _click(widget, "a7")
    _click(widget, "a8")
    assert moves == ["a7a8q"]


def test_set_interactive_false_disables_selection(_app):
    widget = ChessBoardWidget()
    widget.resize(400, 400)
    widget.set_position(chess.STARTING_FEN)
    widget.set_interactive(False)
    _click(widget, "e2")
    assert widget._selected is None


def test_paint_does_not_crash_when_chess_package_unavailable(_app, monkeypatch):
    """Regression test: a real crash — the 'chess' pip package was installed
    into the dev .venv but not into the Python that actually runs the app,
    so CHESS_AVAILABLE was False at runtime. paintEvent must degrade to a
    message, never raise."""
    monkeypatch.setattr(chess_view_module, "CHESS_AVAILABLE", False)
    monkeypatch.setattr(chess_view_module, "chess", None)
    widget = chess_view_module.ChessBoardWidget()
    widget.resize(400, 400)
    widget.grab()   # must paint without raising


# ── eval bar ────────────────────────────────────────────────────────────────

def test_format_eval_label_shows_signed_pawns():
    assert chess_view_module._format_eval_label(250, None) == "+2.5"
    assert chess_view_module._format_eval_label(-130, None) == "-1.3"
    assert chess_view_module._format_eval_label(None, None) == "0.0"


def test_format_eval_label_shows_mate_distance():
    assert chess_view_module._format_eval_label(None, 3) == "M3"
    assert chess_view_module._format_eval_label(None, -2) == "M2"


def test_eval_bar_clamps_fraction_and_paints_without_crashing(_app):
    bar = EvalBarWidget()
    bar.resize(24, 300)
    bar.set_evaluation(1.4, "+9.9")
    assert bar._white_fraction == 1.0
    bar.set_evaluation(-0.2, "-9.9")
    assert bar._white_fraction == 0.0
    bar.grab()   # must paint without raising


def test_eval_bar_set_flipped_updates_state(_app):
    bar = EvalBarWidget()
    assert bar._flipped is False
    bar.set_flipped(True)
    assert bar._flipped is True


# ── ChessPanel: construction, actions, review navigation ────────────────────

def test_panel_with_no_service_shows_a_disabled_placeholder(_app):
    panel = ChessPanel(_StubBus(), None)
    assert "unavailable" in panel.status_chip.text().lower()
    assert panel.move_input.isEnabled() is False


def test_panel_constructs_and_shows_live_status(monkeypatch, tmp_path, _app):
    service = _service(monkeypatch, tmp_path)
    panel = ChessPanel(_StubBus(), service)
    assert "LIVE" in panel.status_chip.text()


def test_new_game_button_starts_a_game_with_selected_sides_and_elo(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        panel = ChessPanel(_StubBus(), service)
        panel.white_combo.setCurrentText("user")
        panel.black_combo.setCurrentText("engine")
        panel.elo_spin.setValue(1200)
        panel._new_game()
        await _drain()
        assert service.sides == {"white": "user", "black": "engine"}
        assert service.elo == 1200
        assert "New game" in panel.commentary.toPlainText()

    asyncio.run(_run())


def test_submitting_a_move_plays_it_and_refreshes_the_move_list(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        panel = ChessPanel(_StubBus(), service)
        panel._new_game()
        await _drain()
        panel.move_input.setText("e4")
        panel._submit_move_from_input()
        await _drain()
        assert panel.move_list.count() >= 1   # start position + at least one ply
        assert len(service.history) >= 1

    asyncio.run(_run())


def test_board_click_submits_a_move_through_the_panel(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        panel = ChessPanel(_StubBus(), service)
        panel._new_game()
        await _drain()
        _click(panel.board, "e2")
        _click(panel.board, "e4")
        await _drain()
        assert any(r.uci.startswith("e2e4") for r in service.history)

    asyncio.run(_run())


def test_resign_button_ends_the_game(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        panel = ChessPanel(_StubBus(), service)
        panel._new_game()
        await _drain()
        panel._resign()
        await _drain()
        assert service._active is False
        assert "Resigned" in panel.commentary.toPlainText()

    asyncio.run(_run())


def test_analyse_buttons_append_commentary(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        panel = ChessPanel(_StubBus(), service)
        panel._new_game()
        await _drain()
        panel.move_input.setText("e4")
        panel._submit_move_from_input()
        await _drain()
        before = panel.commentary.toPlainText()
        panel._analyse_last_move()
        await _drain()
        panel._analyse_game()
        await _drain()
        assert panel.commentary.toPlainText() != before

    asyncio.run(_run())


def test_history_navigation_updates_the_view_cursor_and_disables_input(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        panel = ChessPanel(_StubBus(), service)
        panel._new_game()
        await _drain()
        panel.move_input.setText("e4")
        panel._submit_move_from_input()
        await _drain()
        live_ply = service.view_ply
        panel._step(-1)
        await _drain()
        assert service.view_ply == live_ply - 1
        assert service.is_live is False
        assert panel.move_input.isEnabled() is False
        panel._go_live()
        await _drain()
        assert service.is_live is True
        assert panel.move_input.isEnabled() is True

    asyncio.run(_run())


def test_dashboard_event_on_chess_channel_triggers_refresh(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        bus = _StubBus()
        panel = ChessPanel(bus, service)
        panel._new_game()
        await _drain()
        # A move made elsewhere (autonomous task, dispatcher tool call) still
        # reaches this panel purely through the bus, with no direct call.
        await service.make_move("e4")
        bus.dashboard_event.emit("chess", {"phase": "move"})
        await _drain()
        assert panel.move_list.count() >= 1

    asyncio.run(_run())


# ── Sep 2026: navigating the board ───────────────────────────────────────────
#
# "Navigating in the UI such as in Chess ... must be fixed." The deck turned
# pages on Left/Right, the board had no keys at all, never turned round for
# Black, let the user pick up the opponent's pieces, and started a Stockfish
# game from NEW GAME whatever had been chosen.

from PyQt6.QtGui import QKeyEvent  # noqa: E402


def _key(widget, key) -> bool:
    event = QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    return widget.handle_navigation_key(event)


def test_arrow_keys_step_through_the_game(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        panel = ChessPanel(_StubBus(), service)
        await service.new_game(white="user", black="user")
        for san in ("e4", "e5", "Nf3"):
            await service.make_move(san)
        panel.refresh_from_service()
        assert _key(panel, Qt.Key.Key_Left) and service.view_ply == 2
        assert _key(panel, Qt.Key.Key_Home) and service.view_ply == 0
        assert _key(panel, Qt.Key.Key_Right) and service.view_ply == 1
        assert _key(panel, Qt.Key.Key_End) and service.is_live
        assert "VIEWING" not in panel.status_chip.text()
        assert not _key(panel, Qt.Key.Key_Q)            # not ours: the deck may have it

    asyncio.run(_run())


def test_f_flips_the_board_and_black_starts_flipped(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        panel = ChessPanel(_StubBus(), service)
        await service.new_game(player_colour="black")
        panel.refresh_from_service()
        assert panel.board.flipped and panel.eval_bar._flipped
        assert _key(panel, Qt.Key.Key_F)
        assert not panel.board.flipped

    asyncio.run(_run())


def test_a_flipped_board_maps_clicks_to_the_right_squares(_app):
    widget = ChessBoardWidget()
    widget.resize(400, 400)
    widget.set_position(chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").fen())
    widget.set_flipped(True)
    moves: list[str] = []
    widget.move_requested.connect(moves.append)
    # Flipped, e7 sits where d2 is on an unflipped board.
    _click(widget, "d2")
    _click(widget, "d4")               # unflipped d4 is e5 when flipped
    assert moves == ["e7e5"]


def test_only_your_own_pieces_can_be_picked_up(_app):
    widget = ChessBoardWidget()
    widget.resize(400, 400)
    widget.set_position(chess.STARTING_FEN)
    moves: list[str] = []
    widget.move_requested.connect(moves.append)
    _click(widget, "e7")               # Black's pawn, White to move
    assert widget._selected is None
    _click(widget, "e4")               # an empty square
    assert widget._selected is None
    _click(widget, "g1")
    _click(widget, "b1")               # changed their mind: the other knight
    assert widget._selected == "b1" and moves == []
    assert widget._targets() == {"a3", "c3"}   # legal destinations are shown


def test_the_board_is_locked_while_it_is_not_your_turn(monkeypatch, tmp_path, _app):
    async def _run():
        service = _service(monkeypatch, tmp_path)
        panel = ChessPanel(_StubBus(), service)
        await service.new_game(player_colour="white")
        panel.refresh_from_service()
        assert panel.board.interactive
        service._thinking = True
        panel.refresh_from_service()
        assert not panel.board.interactive
        assert "thinking" in panel.status_chip.text()
        service._thinking = False

    asyncio.run(_run())


def test_new_game_combos_default_to_orion_not_stockfish(_app):
    from orion_core.chess_engine import ChessService as _Service
    panel = ChessPanel(_StubBus(), _Service(_StubBus()))
    assert panel.black_combo.currentText() == "orion"


def test_a_review_lands_in_the_commentary_when_it_arrives(monkeypatch, tmp_path, _app):
    service = _service(monkeypatch, tmp_path)
    bus = _StubBus()
    panel = ChessPanel(bus, service)
    bus.dashboard_event.emit("chess", {"phase": "review",
                                       "comment": "3... Nf6 — a blunder. It allows 4. Qxf7#."})
    assert "Qxf7#" in panel.commentary.toPlainText()


def test_the_deck_lets_the_chess_page_have_the_arrow_keys():
    """The Command Deck asks the current page before turning the page."""
    import inspect
    from orion_core.gui.unified_dashboard import UnifiedDashboard
    source = inspect.getsource(UnifiedDashboard.keyPressEvent)
    assert "handle_navigation_key" in source
    assert source.index("handle_navigation_key") < source.index("next_page")
