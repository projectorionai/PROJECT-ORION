"""
ChessPanel — the chess board, move list and game-review controls.

Every widget is fed by ``ChessService``
directly for one-shot reads (board FEN, history, status) and stays live via
``OrionBus.dashboard_event`` for anything that changes off its own thread —
most importantly the autonomous background game, which pushes moves from an
``asyncio`` task the panel never has to poll.

Game review (chess.com-style "step back through the moves") is a read-only
*view cursor* (``ChessService.view_ply``) that is completely separate from
the live game state: browsing history repaints the board from a stored FEN
without touching ``ChessService.board``, and the move-submission controls
disable themselves whenever the cursor isn't at the live tip — so a user
scrubbing back to move 12 can never accidentally play a move into the
middle of the game. The status chip at the top makes which mode you're in
unambiguous at a glance (LIVE vs VIEWING HISTORY).

Navigation is keyboard-first as well as clickable: Left/Right step back and
forward a half-move, Home/End jump to the start and back to the live
position, F flips the board, and the move list follows the arrow keys. The
Command Deck turns pages with Left/Right too, so it hands those keys to this
page first (``handle_navigation_key``) and only turns the page when the chess
board declines them. The board turns round by itself when the user plays
Black, and the user can only pick up their own pieces, on their own turn.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from .. import chess_sound
from ..chess_engine import CHESS_AVAILABLE, ChessService

if CHESS_AVAILABLE:
    import chess
else:  # pragma: no cover - python-chess is a hard requirement in practice
    chess = None  # type: ignore

from ..constants import C
from .. import background
from .widgets import describe_control

_UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

_FILES = "abcdefgh"


def _heading(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("panelHeading")
    return lab


# ──────────────────────────────────────────────────────────────────────────────
# BOARD
# ──────────────────────────────────────────────────────────────────────────────

class ChessBoardWidget(QWidget):
    """
    Custom-painted 8x8 board. Renders whatever FEN it is given — the live
    position or any historical ply — and, only when ``interactive`` is True
    (the panel sets this to the live/view-live state), turns two clicks
    (origin square, destination square) into a UCI move request via
    ``move_requested``. Painting itself never depends on whose turn it is
    being editable — the panel is the single source of truth for that.
    """

    move_requested = pyqtSignal(str)  # UCI move, e.g. "e2e4"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._fen: Optional[str] = chess.STARTING_FEN if CHESS_AVAILABLE else None
        self._last_move: tuple[str, str] | None = None   # (from_square, to_square)
        self._selected: Optional[str] = None               # algebraic square, e.g. "e2"
        self.interactive = True
        #: Black at the bottom. The user playing Black used to look at the
        #: game upside down, with no way to turn it round.
        self.flipped = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Perf: the position is parsed ONCE per move, not once per repaint. The
        # old paintEvent built a fresh chess.Board(self._fen) every single frame,
        # which on a busy Command Deck (many live pages sharing the GUI thread)
        # made every board repaint needlessly heavy. Cache the parsed board and
        # the piece font (rebuilt only when the square size changes).
        self._board: Any = None
        self._parse_fen()
        self._piece_font: Any = None
        self._piece_font_size: int = -1

    def _parse_fen(self) -> None:
        if not CHESS_AVAILABLE or not self._fen:
            self._board = None
            return
        try:
            self._board = chess.Board(self._fen)
        except ValueError:
            self._board = None

    def set_position(self, fen: Optional[str], last_move_uci: Optional[str] = None) -> None:
        self._fen = fen
        self._parse_fen()
        self._last_move = (last_move_uci[:2], last_move_uci[2:4]) if last_move_uci else None
        self._selected = None
        self.update()

    def set_interactive(self, interactive: bool) -> None:
        self.interactive = interactive
        if not interactive:
            self._selected = None
            self.update()

    def set_flipped(self, flipped: bool) -> None:
        if bool(flipped) != self.flipped:
            self.flipped = bool(flipped)
            self._selected = None
            self.update()

    def _display(self, file_index: int, rank: int) -> tuple[int, int]:
        """Board file (0 = a) and rank (1..8) -> on-screen column and row."""
        if self.flipped:
            return 7 - file_index, rank - 1
        return file_index, 8 - rank

    def _square_at(self, column: int, row: int) -> str:
        """On-screen column and row -> square name."""
        if self.flipped:
            return f"{_FILES[7 - column]}{row + 1}"
        return f"{_FILES[column]}{8 - row}"

    def _targets(self) -> set[str]:
        """Where the selected piece can legally go — shown as dots."""
        if self._selected is None or self._board is None:
            return set()
        try:
            origin = chess.parse_square(self._selected)
        except ValueError:
            return set()
        return {chess.square_name(m.to_square)
                for m in self._board.legal_moves if m.from_square == origin}

    # ── painting ─────────────────────────────────────────────────────────────

    #: Fraction of the widget reserved for the rank/file labels. A board with
    #: no coordinates is unreadable the moment you want to talk about it —
    #: "the knight on f3" needs an f and a 3 you can actually see.
    MARGIN_FRACTION = 0.055

    def _margin(self) -> float:
        return min(self.width(), self.height()) * self.MARGIN_FRACTION

    def _square_size(self) -> float:
        margin = self._margin()
        return (min(self.width(), self.height()) - margin) / 8.0

    def _square_rect(self, file_index: int, rank_index: int) -> tuple[float, float, float]:
        """rank_index 0 = rank 8 (top row) so White's home row paints at the bottom.

        Offset by the margin so the board sits to the right of the rank labels
        and above the file labels.
        """
        size = self._square_size()
        margin = self._margin()
        return margin + file_index * size, rank_index * size, size

    def _paint_coordinates(self, painter: "QPainter", size: float) -> None:
        """Files a-h along the bottom, ranks 1-8 up the left — standard notation."""
        margin = self._margin()
        font = QFont("Consolas", max(7, int(margin * 0.52)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(C.ACCENT_DIM))
        board_bottom = 8 * size
        for index in range(8):
            file_letter = _FILES[7 - index if self.flipped else index].upper()
            rank_number = str(index + 1 if self.flipped else 8 - index)
            x, _y, _s = self._square_rect(index, 0)
            painter.drawText(int(x), int(board_bottom), int(size), int(margin),
                             Qt.AlignmentFlag.AlignCenter, file_letter)
            painter.drawText(0, int(index * size), int(margin), int(size),
                             Qt.AlignmentFlag.AlignCenter, rank_number)

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = self._square_size()
        painter.fillRect(self.rect(), QColor(C.BG))
        if not CHESS_AVAILABLE or not self._fen:
            painter.setPen(QColor(C.MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                              "python-chess is not installed.")
            return
        board = self._board                 # parsed once per move, not per paint
        if board is None:
            painter.setPen(QColor(C.MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Invalid position.")
            return
        light, dark = QColor("#2a2a36"), QColor("#151520")
        highlight = QColor(C.ACCENT_DEEP)
        select_pen = QPen(QColor(C.ACCENT), 2)
        # Rebuild the piece font only when the square size actually changes —
        # constructing a QFont every frame is pure waste.
        font_size = max(10, int(size * 0.55))
        if self._piece_font is None or font_size != self._piece_font_size:
            self._piece_font = QFont("Segoe UI Symbol", font_size)
            self._piece_font_size = font_size
        piece_font = self._piece_font
        targets = self._targets()
        for rank in range(1, 9):
            for file_index in range(8):      # 0 = file a .. 7 = file h
                square_name = f"{_FILES[file_index]}{rank}"
                column, row = self._display(file_index, rank)
                x, y, s = self._square_rect(column, row)
                is_light = (file_index + rank) % 2 == 0   # a1 is dark
                painter.fillRect(int(x), int(y), int(s) + 1, int(s) + 1,
                                  light if is_light else dark)
                if self._last_move and square_name in self._last_move:
                    painter.fillRect(int(x), int(y), int(s) + 1, int(s) + 1, highlight)
                if self._selected == square_name:
                    painter.setPen(select_pen)
                    painter.drawRect(int(x) + 1, int(y) + 1, int(s) - 2, int(s) - 2)
                piece = board.piece_at(chess.parse_square(square_name))
                if piece is not None:
                    glyph = _UNICODE_PIECES.get(piece.symbol(), "")
                    painter.setPen(QColor(C.WHITE) if piece.color == chess.WHITE else QColor("#ff5c73"))
                    painter.setFont(piece_font)
                    painter.drawText(int(x), int(y), int(s), int(s),
                                      Qt.AlignmentFlag.AlignCenter, glyph)
                if square_name in targets:
                    dot = max(4.0, s * 0.16)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(C.ACCENT))
                    painter.drawEllipse(int(x + s / 2 - dot / 2), int(y + s / 2 - dot / 2),
                                        int(dot), int(dot))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
        self._paint_coordinates(painter, size)

    # ── interaction ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if not self.interactive or not CHESS_AVAILABLE or not self._fen:
            return
        size = self._square_size()
        if size <= 0:
            return
        # Subtract the coordinate margin before converting to a square, or every
        # click lands one file to the left of where it was aimed.
        column = int((event.position().x() - self._margin()) // size)
        row = int(event.position().y() // size)
        if not (0 <= column < 8 and 0 <= row < 8):
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        square_name = self._square_at(column, row)
        piece = None
        if self._board is not None:
            piece = self._board.piece_at(chess.parse_square(square_name))
        own = piece is not None and piece.color == self._board.turn
        if self._selected is None:
            # Only the side to move's own pieces can be picked up — a click
            # on an empty square or an enemy piece used to "select" it, and
            # the next click then sent a nonsense move.
            if own:
                self._selected = square_name
                self.update()
            return
        if self._selected == square_name:
            self._selected = None
            self.update()
            return
        if own:
            self._selected = square_name        # changed their mind: pick this one up
            self.update()
            return
        origin, self._selected = self._selected, None
        uci = origin + square_name
        # Default under-promotion-free UX: auto-queen when a pawn reaches the
        # back rank. The service still validates the move regardless.
        if square_name[1] in ("1", "8"):
            try:
                board = chess.Board(self._fen)
                piece = board.piece_at(chess.parse_square(origin))
                if piece is not None and piece.piece_type == chess.PAWN:
                    uci += "q"
            except ValueError:
                pass
        self.update()
        self.move_requested.emit(uci)


def _format_eval_label(cp: Optional[int], mate: Optional[int]) -> str:
    """Short eval-bar caption: "+2.5" (pawns, White-positive), "M3" for a
    forced mate in 3, or "0.0" when there's nothing to show yet."""
    if mate is not None:
        return f"M{abs(int(mate))}"
    if cp is None:
        return "0.0"
    return f"{cp / 100.0:+.1f}"


class EvalBarWidget(QWidget):
    """A thin vertical bar showing White's live win fraction (0..1) — "who's
    better right now?" at a glance, fed by ``ChessService.evaluate_current``."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(18)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._white_fraction = 0.5
        self._label = "0.0"
        self._flipped = False

    def set_evaluation(self, white_fraction: float, label: str) -> None:
        self._white_fraction = max(0.0, min(1.0, float(white_fraction)))
        self._label = label
        self.update()

    def set_flipped(self, flipped: bool) -> None:
        self._flipped = bool(flipped)
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151520"))
        white_h = self.height() * self._white_fraction
        top = 0 if self._flipped else (self.height() - white_h)
        painter.fillRect(0, int(top), self.width(), int(white_h) + 1, QColor("#e8e8ee"))
        painter.setPen(QColor(C.MUTED))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                          self._label)


# ──────────────────────────────────────────────────────────────────────────────
# PANEL
# ──────────────────────────────────────────────────────────────────────────────

class ChessPanel(QFrame):
    """Assembles the board with move entry, autonomous-play controls, and a
    move-list/commentary review pane. Construct with the shared
    ``ChessService`` (``dispatcher.chess``) — pass ``None`` only for a
    disabled placeholder (e.g. a UI smoke test with no service wired up)."""

    def __init__(self, bus: OrionBus, service: Optional[ChessService] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.service = service
        # Wooden move sounds, synthesised on first use (chess_sound.py). Kept on
        # the panel rather than the board widget so the preference and the
        # effect cache live with the rest of the game's state.
        self.sounds = chess_sound.MoveSounds()
        self._last_sounded: Optional[str] = None
        self.setObjectName("panelFrame")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(_heading("CHESS"))
        self.status_chip = QLabel("—")
        self.status_chip.setObjectName("mutedLabel")
        header.addStretch(1)
        header.addWidget(self.status_chip)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._build_left_column(), 3)
        body.addWidget(self._build_right_column(), 2)
        outer.addLayout(body)

        # After both columns exist, so the toggles are there to restore into.
        self._restore_board_options()

        if self.service is not None:
            self.bus.dashboard_event.connect(self._on_dashboard_event)
            if getattr(self.service, "_active", False):
                self._sync_sides()
            self.refresh_from_service()
        else:
            self._set_enabled(False)
            self.status_chip.setText("chess subsystem unavailable")

    # ── layout builders ──────────────────────────────────────────────────────

    def _build_left_column(self) -> QWidget:
        col = QWidget()
        layout = QVBoxLayout(col)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        board_row = QHBoxLayout()
        self.board = ChessBoardWidget()
        self.board.move_requested.connect(self._submit_move_uci)
        self.eval_bar = EvalBarWidget()
        board_row.addWidget(self.board, 1)
        board_row.addWidget(self.eval_bar, 0)
        layout.addLayout(board_row, 1)

        # The evaluation bar is a choice. Some players want the engine's verdict
        # in their peripheral vision; others find that it plays the game for
        # them. Both preferences are remembered between sessions.
        options_row = QHBoxLayout()
        self.eval_toggle = QCheckBox("Evaluation bar")
        self.eval_toggle.setToolTip(
            "Show the engine's running assessment beside the board.")
        self.eval_toggle.toggled.connect(self._on_eval_toggled)
        self.sound_toggle = QCheckBox("Move sound")
        self.sound_toggle.setToolTip("A wooden click when a piece is played.")
        self.sound_toggle.toggled.connect(self._on_sound_toggled)
        options_row.addWidget(self.eval_toggle)
        options_row.addWidget(self.sound_toggle)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        move_row = QHBoxLayout()
        self.move_input = QLineEdit()
        self.move_input.setPlaceholderText("Move (e.g. Nf3 or g1f3) — Enter to play")
        self.move_input.returnPressed.connect(self._submit_move_from_input)
        submit_btn = QPushButton("PLAY")
        submit_btn.clicked.connect(self._submit_move_from_input)
        self.live_btn = QPushButton("RETURN TO LIVE")
        self.live_btn.clicked.connect(self._go_live)
        move_row.addWidget(self.move_input, 1)
        move_row.addWidget(submit_btn)
        move_row.addWidget(self.live_btn)
        layout.addLayout(move_row)

        new_row = QHBoxLayout()
        self.white_combo = QComboBox()
        self.black_combo = QComboBox()
        # Default to the user against the service's own opponent — ORION. The
        # combos used to default Black to "engine", so NEW GAME started a
        # Stockfish game whatever the user had asked for by voice.
        opponent = getattr(self.service, "opponent", "orion") or "orion"
        for combo, default in ((self.white_combo, "user"), (self.black_combo, opponent)):
            combo.addItems(["user", "orion", "engine"])
            combo.setCurrentText(default)
        self.elo_spin = QSpinBox()
        self.elo_spin.setRange(400, 3000)
        self.elo_spin.setSingleStep(50)
        self.elo_spin.setValue(1500)
        self.elo_spin.setToolTip("Engine strength (ELO) for the side(s) it plays.")
        new_btn = QPushButton("NEW GAME")
        new_btn.clicked.connect(self._new_game)
        self.resign_btn = QPushButton("RESIGN")
        self.resign_btn.clicked.connect(self._resign)
        new_row.addWidget(QLabel("White")); new_row.addWidget(self.white_combo)
        new_row.addWidget(QLabel("Black")); new_row.addWidget(self.black_combo)
        new_row.addWidget(QLabel("ELO")); new_row.addWidget(self.elo_spin)
        new_row.addWidget(new_btn)
        new_row.addWidget(self.resign_btn)
        layout.addLayout(new_row)

        auto_row = QHBoxLayout()
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.2, 30.0)
        self.delay_spin.setValue(2.0)
        self.delay_spin.setSuffix("s")
        self.start_btn = QPushButton("START AUTONOMOUS")
        self.start_btn.setToolTip("Both sides must be 'orion' and/or 'engine' — "
                                  "e.g. 'play against the engine' or 'train against yourself'.")
        self.start_btn.clicked.connect(self._start_autonomous)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.clicked.connect(self._stop_autonomous)
        auto_row.addWidget(QLabel("Delay")); auto_row.addWidget(self.delay_spin)
        auto_row.addWidget(self.start_btn)
        auto_row.addWidget(self.stop_btn)
        layout.addLayout(auto_row)
        return col

    # ── board options ────────────────────────────────────────────────────────

    def _chess_settings(self) -> Any:
        from PyQt6.QtCore import QSettings
        return QSettings("ORION", "Chess")

    def _restore_board_options(self) -> None:
        settings = self._chess_settings()
        show_eval = settings.value("eval_bar", True, type=bool)
        play_sound = settings.value("move_sound", True, type=bool)
        # setChecked fires toggled, which applies and re-saves the same value —
        # harmless, and it means one code path applies the preference.
        self.eval_toggle.setChecked(bool(show_eval))
        self.sound_toggle.setChecked(bool(play_sound))
        self._on_eval_toggled(bool(show_eval))
        self._on_sound_toggled(bool(play_sound))

    def _on_eval_toggled(self, shown: bool) -> None:
        self.eval_bar.setVisible(bool(shown))
        self._chess_settings().setValue("eval_bar", bool(shown))

    def _on_sound_toggled(self, enabled: bool) -> None:
        self.sounds.set_enabled(bool(enabled))
        self._chess_settings().setValue("move_sound", bool(enabled))
        if enabled:
            self.sounds.prewarm()

    def _play_move_sound(self, last_move_uci: Optional[str]) -> None:
        """Sound the move that has just appeared on the board.

        Driven by the position actually changing rather than by the input path,
        so ORION's moves, the engine's moves and the user's own all sound —
        typing a move and clicking one behave identically.
        """
        if not last_move_uci or last_move_uci == self._last_sounded:
            return
        self._last_sounded = last_move_uci
        kind = "move"
        service = self.service
        try:
            if CHESS_AVAILABLE and service is not None and service.view_ply > 0:
                before = chess.Board(service.position_fen(service.view_ply - 1))
                kind = chess_sound.classify(before, chess.Move.from_uci(last_move_uci))
        except Exception:
            kind = "move"
        self.sounds.play(kind)

    def _build_right_column(self) -> QWidget:
        col = QWidget()
        layout = QVBoxLayout(col)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(_heading("MOVES — click, or ←/→ to step"))
        self.move_list = QListWidget()
        self.move_list.itemDoubleClicked.connect(self._on_history_item)
        # A single click or the arrow keys in the list show that position.
        self.move_list.currentRowChanged.connect(self._on_history_row)
        self.move_list.installEventFilter(self)
        layout.addWidget(self.move_list, 1)

        nav_row = QHBoxLayout()
        self.first_btn = describe_control(
            QPushButton("|<"), "First move", "Jump to the start of the game")
        self.back_btn = describe_control(
            QPushButton("<"), "Previous move", "Step one move back")
        self.fwd_btn = describe_control(
            QPushButton(">"), "Next move", "Step one move forward")
        self.last_btn = describe_control(
            QPushButton(">|"), "Latest position", "Return to the live position")
        self.flip_btn = describe_control(
            QPushButton("⇅"), "Flip board", "Turn the board round (F)")
        self.first_btn.clicked.connect(lambda: self._goto(0))
        self.back_btn.clicked.connect(lambda: self._step(-1))
        self.fwd_btn.clicked.connect(lambda: self._step(1))
        self.last_btn.clicked.connect(self._go_live)
        self.flip_btn.clicked.connect(self._flip)
        for b in (self.first_btn, self.back_btn, self.fwd_btn, self.last_btn, self.flip_btn):
            nav_row.addWidget(b)
        layout.addLayout(nav_row)

        analyse_row = QHBoxLayout()
        self.analyse_move_btn = QPushButton("ANALYSE LAST MOVE")
        self.analyse_move_btn.clicked.connect(self._analyse_last_move)
        self.analyse_game_btn = QPushButton("ANALYSE GAME")
        self.analyse_game_btn.clicked.connect(self._analyse_game)
        analyse_row.addWidget(self.analyse_move_btn)
        analyse_row.addWidget(self.analyse_game_btn)
        layout.addLayout(analyse_row)

        layout.addWidget(_heading("COMMENTARY"))
        self.commentary = QPlainTextEdit()
        self.commentary.setReadOnly(True)
        self.commentary.setMaximumHeight(160)
        layout.addWidget(self.commentary)
        return col

    # ── service actions ──────────────────────────────────────────────────────

    def _new_game(self) -> None:
        if self.service is None:
            return

        async def _run() -> None:
            result = await self.service.new_game(
                white=self.white_combo.currentText(), black=self.black_combo.currentText(),
                elo=self.elo_spin.value(),
            )
            self.commentary.appendPlainText(result.text)
            self.refresh_from_service()

        background.spawn(_run())

    def _resign(self) -> None:
        if self.service is None:
            return

        async def _run() -> None:
            result = await self.service.resign()
            self.commentary.appendPlainText(result.text)
            self.refresh_from_service()

        background.spawn(_run())

    def _analyse_last_move(self) -> None:
        if self.service is None:
            return

        async def _run() -> None:
            result = await self.service.analyse_move()
            self.commentary.appendPlainText(result.text)

        background.spawn(_run())

    def _analyse_game(self) -> None:
        if self.service is None:
            return

        async def _run() -> None:
            result = await self.service.analyse_game()
            self.commentary.appendPlainText(result.text)

        background.spawn(_run())

    def _submit_move_from_input(self) -> None:
        text = self.move_input.text().strip()
        if not text:
            return
        self.move_input.clear()
        self._submit_move_uci(text)

    def _submit_move_uci(self, move_text: str) -> None:
        if self.service is None:
            return

        async def _run() -> None:
            result = await self.service.make_move(move_text)
            self.commentary.appendPlainText(result.text)
            self.refresh_from_service()

        background.spawn(_run())

    def _start_autonomous(self) -> None:
        if self.service is None:
            return
        result = self.service.start_autonomous(self.delay_spin.value())
        self.commentary.appendPlainText(result.text)
        self.refresh_from_service()

    def _stop_autonomous(self) -> None:
        if self.service is None:
            return
        result = self.service.stop_autonomous()
        self.commentary.appendPlainText(result.text)

    def _goto(self, ply: int) -> None:
        if self.service is None:
            return
        self.service.goto_ply(ply)
        self.refresh_from_service()

    def _step(self, delta: int) -> None:
        if self.service is None:
            return
        self.service.step(delta)
        self.refresh_from_service()

    def _go_live(self) -> None:
        if self.service is None:
            return
        self.service.resume_live()
        self.refresh_from_service()

    def _flip(self) -> None:
        if self.service is None:
            return
        self.service.flip()
        self.refresh_from_service()

    def _on_history_item(self, item: QListWidgetItem) -> None:
        ply = item.data(Qt.ItemDataRole.UserRole)
        if ply is not None:
            self._goto(int(ply))

    def _on_history_row(self, row: int) -> None:
        if self.service is not None and row >= 0 and row != self.service.view_ply:
            self._goto(row)

    # ── keyboard ─────────────────────────────────────────────────────────────

    def handle_navigation_key(self, event: Any) -> bool:
        """Left/Right: a half-move back or forward. Home/End: the start, or
        back to the live position. F: flip the board. Returns True when the
        key was used — the Command Deck asks this before turning its page
        with the same arrows."""
        if self.service is None or event.modifiers() & (
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            return False
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._step(-1)
        elif key == Qt.Key.Key_Right:
            self._step(1)
        elif key == Qt.Key.Key_Home:
            self._goto(0)
        elif key == Qt.Key.Key_End:
            self._go_live()
        elif key == Qt.Key.Key_F:
            self._flip()
        else:
            return False
        return True

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        if self.handle_navigation_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt override
        # The move list would otherwise swallow Left/Right/Home/End for its
        # own cursor; in this list they mean "step through the game".
        if obj is self.move_list and event.type() == QEvent.Type.KeyPress \
                and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right,
                                    Qt.Key.Key_Home, Qt.Key.Key_End, Qt.Key.Key_F):
            return self.handle_navigation_key(event)
        return super().eventFilter(obj, event)

    # ── bus / refresh ─────────────────────────────────────────────────────────

    def _on_dashboard_event(self, channel: str, payload: object) -> None:
        if channel != "chess":
            return
        # A coaching review arrives after the reply, in the background: show
        # it when it lands rather than making the reply wait for it.
        if isinstance(payload, dict) and payload.get("phase") == "review" \
                and payload.get("comment"):
            self.commentary.appendPlainText(str(payload["comment"]))
        if isinstance(payload, dict) and payload.get("phase") in ("new_game", "sides"):
            self._sync_sides()
        self.refresh_from_service()

    def _sync_sides(self) -> None:
        """Show who holds each side of the game in progress. Only when a game
        starts or the opponent changes — syncing on every refresh would undo
        the user's choice for their NEXT game while they were making it."""
        svc = self.service
        if svc is None:
            return
        for combo, colour in ((self.white_combo, "white"), (self.black_combo, "black")):
            combo.blockSignals(True)
            combo.setCurrentText(svc.sides.get(colour, combo.currentText()))
            combo.blockSignals(False)

    def refresh_from_service(self) -> None:
        """Re-render everything from ``ChessService`` — safe to call after
        any service mutation or bus notification; never mutates the
        service itself, so it's also safe to call purely to redraw."""
        if self.service is None:
            return
        svc = self.service
        fen = svc.position_fen(svc.view_ply)
        last_move_uci = None
        if svc.view_ply > 0 and svc.view_ply <= len(svc.history):
            last_move_uci = svc.history[svc.view_ply - 1].uci
        self.board.set_position(fen, last_move_uci)
        self._play_move_sound(last_move_uci)
        flipped = svc.board_flipped()
        self.board.set_flipped(flipped)
        self.eval_bar.set_flipped(flipped)

        is_live = svc.is_live
        # Pieces can be picked up only on the user's own turn: never while
        # ORION is thinking, and never for the side he or Stockfish plays.
        self.board.set_interactive(svc.users_turn())
        self.move_input.setEnabled(is_live and not svc.thinking)
        self.live_btn.setEnabled(not is_live)
        if not is_live:
            where = (svc.history[svc.view_ply - 1].label if svc.view_ply
                     else "the start")
            self.status_chip.setText(f"VIEWING {where} — ←/→ to step, End for live")
            self.status_chip.setStyleSheet(f"color: {C.WARN};")
        elif svc.thinking:
            self.status_chip.setText("● LIVE — thinking…")
            self.status_chip.setStyleSheet(f"color: {C.ACCENT};")
        elif svc.users_turn():
            self.status_chip.setText("● LIVE — your move")
            self.status_chip.setStyleSheet(f"color: {C.GOOD};")
        else:
            self.status_chip.setText("● LIVE")
            self.status_chip.setStyleSheet(f"color: {C.GOOD};")

        self.move_list.blockSignals(True)
        self.move_list.clear()
        start_item = QListWidgetItem("Start position")
        start_item.setData(Qt.ItemDataRole.UserRole, 0)
        self.move_list.addItem(start_item)
        for record in svc.history_list():
            label = f"{record.get('label') or record['san']} ({record['actor']})"
            if record.get("quality"):
                label += f" — {record['quality']}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, record["ply"])
            self.move_list.addItem(item)
        if 0 <= svc.view_ply < self.move_list.count():
            self.move_list.setCurrentRow(svc.view_ply)
        self.move_list.blockSignals(False)

        async def _refresh_eval() -> None:
            info = await svc.evaluate_current()
            self.eval_bar.set_evaluation(
                info["white_fraction"], _format_eval_label(info["cp"], info["mate"]))

        try:
            loop_running = asyncio.get_running_loop() is not None
        except RuntimeError:
            loop_running = False
        if loop_running:
            background.spawn(_refresh_eval())
        # else: constructed before an event loop is running — the next
        # bus-driven refresh (once the loop is up) will catch up.

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.board, self.move_input, self.white_combo, self.black_combo,
                       self.elo_spin, self.resign_btn, self.analyse_move_btn,
                       self.analyse_game_btn, self.start_btn, self.stop_btn,
                       self.first_btn, self.back_btn, self.fwd_btn, self.last_btn,
                       self.flip_btn, self.live_btn):
            widget.setEnabled(enabled)
