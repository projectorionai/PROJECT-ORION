"""Remote board: latest-state mailbox, frame coalescing and bounded Qt lifetime."""
from __future__ import annotations

import asyncio
import math

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class RemoteChessWidget(QWidget):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        from .chess_view import BoardViewport, ChessBoardWidget, EvalBarWidget
        self.service = service
        self._latest = None
        self._snapshot = {}
        self._tasks = set()
        self.board = ChessBoardWidget(self)
        self.board.set_interactive(False)
        self.bar = EvalBarWidget(self)
        self.label = QLabel("Remote chess idle", self)
        self.label.setWordWrap(True)
        self.training_status = QLabel("Checking training…", self)
        self.training_status.setWordWrap(True)
        self.connect_button = QPushButton("Connect Lichess", self)
        self.disconnect_button = QPushButton("Disconnect", self)
        self.username = QLineEdit(self)
        self.username.setPlaceholderText("Opponent's Lichess username")
        self.username.setMaxLength(30)
        self.challenge_button = QPushButton("Authorise and Challenge", self)
        self.elo = QSpinBox(self)
        self.elo.setRange(400, 3000)
        self.elo.setValue(2604)
        self.elo.setPrefix("Casual Handicap ")
        self.abort = QPushButton("Abort / Resign Remote Game", self)
        self.train = QPushButton("Queue NNUE Training", self)
        self.swarm = QPushButton("Start Training Swarm", self)
        self.stop = QPushButton("Stop Training Swarm", self)
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.addWidget(BoardViewport(self.board, self.bar), 3)
        controls = QVBoxLayout()
        controls.addWidget(QLabel("Remote Game"))
        controls.addWidget(self.label)
        connection = QHBoxLayout()
        connection.addWidget(self.connect_button)
        connection.addWidget(self.disconnect_button)
        controls.addLayout(connection)
        controls.addWidget(self.username)
        controls.addWidget(self.challenge_button)
        controls.addWidget(self.elo)
        controls.addWidget(self.abort)
        controls.addSpacing(18)
        controls.addWidget(QLabel("Training"))
        controls.addWidget(self.training_status)
        for widget in (self.train, self.swarm, self.stop):
            controls.addWidget(widget)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setMaximumBlockCount(80)
        self.history.setMaximumHeight(180)
        self.history.setAccessibleName("Remote chess and training action results")
        controls.addWidget(self.history)
        controls.addStretch(1)
        layout.addLayout(controls, 2)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)  # coalesce paint requests to at most ~60 Hz
        self._timer.timeout.connect(self._render)
        self._training_timer = QTimer(self)
        self._training_timer.setInterval(2000)
        self._training_timer.timeout.connect(self._refresh_training)
        self.connect_button.clicked.connect(lambda: self._launch(
            self.service.lichess_start(), button=self.connect_button))
        self.disconnect_button.clicked.connect(lambda: self._launch(
            self.service.lichess_stop(), button=self.disconnect_button))
        self.challenge_button.clicked.connect(self._challenge)
        self.abort.clicked.connect(lambda: self._launch(self.service.lichess.abort_all(),
                                                       "Remote games stopped.", self.abort))
        self.elo.valueChanged.connect(lambda value: self._launch(self.service.lichess.set_handicap(value)))
        self.train.clicked.connect(lambda: self._launch(self.service.grandmaster.force_training(),
                                                       "Training queued.", self.train))
        self.swarm.clicked.connect(self._start_swarm)
        self.stop.clicked.connect(lambda: self._launch(self.service.grandmaster.stop_training(),
                                                      "Training stopped.", self.stop))
        self.destroyed.connect(self._cancel)

    def _start_swarm(self):
        from ..grandmaster.book import DEFAULT_SOURCES
        self._launch(self.service.grandmaster.start(pgn_sources=DEFAULT_SOURCES),
                     "Training swarm started.", self.swarm)

    def _challenge(self):
        username = self.username.text().strip()
        from ..chess_lichess import valid_username
        if not valid_username(username):
            self.label.setText("Enter a valid Lichess username before challenging.")
            return
        async def challenge():
            await self.service.lichess.authorise(username, username, elo=self.elo.value())
            return await self.service.lichess_challenge(username, rated=False)
        self._launch(challenge(), button=self.challenge_button)

    def _cancel(self, *_args):
        for task in tuple(self._tasks):
            task.cancel()

    def _launch(self, coroutine, success="Action completed.", button=None):
        try:
            if len(self._tasks) >= 4:
                coroutine.close()
                self.label.setText("Busy - try again in a moment")
                return
            task = asyncio.get_running_loop().create_task(coroutine)
        except RuntimeError:
            coroutine.close()
            self.label.setText("Not ready yet")
            return
        self._tasks.add(task)
        self.label.setText("Working…")
        if button is not None:
            button.setEnabled(False)
        def done(completed):
            self._tasks.discard(completed)
            try:
                if button is not None:
                    button.setEnabled(True)
                if completed.cancelled():
                    text = "Action cancelled."
                elif completed.exception():
                    text = f"Action failed: {str(completed.exception())[:180]}"
                else:
                    result = completed.result()
                    if hasattr(result, "text"):
                        text = ("" if getattr(result, "ok", True) else "Failed: ") + str(result.text)
                    elif isinstance(result, tuple) and len(result) == 2:
                        text = ("" if result[0] else "Failed: ") + str(result[1])
                    elif isinstance(result, dict):
                        text = f"{result.get('status', '')}: {result.get('reason', '')}".strip(": ")
                        self.training_status.setText(text[:600])
                    else:
                        text = str(result or success)
                self.label.setText(text[:600])
                self.history.appendPlainText(text[:1200])
                self._refresh_training()
            except RuntimeError:
                pass  # The owning Qt page was destroyed during shutdown.
        task.add_done_callback(done)

    def _refresh_training(self):
        grandmaster = getattr(self.service, "grandmaster", None)
        if grandmaster is None or not hasattr(grandmaster, "status"):
            return
        try:
            state = grandmaster.status()
            self.training_status.setText(f"{state.get('status', '')}: {state.get('reason', '')}".strip(": "))
        except Exception as exc:
            self.training_status.setText(f"Training status unavailable: {str(exc)[:180]}")

    def consume(self, payload):
        if not isinstance(payload, dict):
            return
        if payload.get("phase") == "swarm":
            self.training_status.setText(str(payload.get("status", "")) + ": " + str(payload.get("reason", "")))
            return
        if payload.get("fen"):
            # Exactly one pending state; an event burst cannot grow a render queue.
            if payload.get("game_id") != self._snapshot.get("game_id"):
                self._snapshot = {}
            snapshot = {**self._snapshot, **payload}
            # A move acknowledgement can arrive after a newer gameState. Its
            # evaluation describes the search root, never a replacement board.
            if payload.get("phase_detail") == "evaluation" and self._snapshot.get("fen"):
                snapshot["fen"] = self._snapshot["fen"]
            self._snapshot = self._latest = snapshot
            if self.isVisible() and not self._timer.isActive():
                self._timer.start()

    def _render(self):
        if self._latest is None:
            self._timer.stop()
            return
        payload, self._latest = self._latest, None
        self.board.set_position(payload["fen"])
        if payload.get("colour") in ("white", "black"):
            self.board.set_flipped(payload["colour"] == "black")
            self.bar.set_flipped(payload["colour"] == "black")
        cp = payload.get("cp")
        if cp is not None:
            value = max(-20000, min(20000, float(cp)))
            fraction = 1 / (1 + math.exp(-value / 350))
            self.bar.set_evaluation(fraction, f"{value / 100:+.2f}")
        if "depth" in payload:
            self.label.setText(f"Depth {payload['depth']} · {payload.get('nodes', 0):,} nodes · "
                               f"{payload.get('elapsed', 0):.2f}s · {payload.get('version', 'classical')[:18]}")

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_training()
        self._training_timer.start()
        if self._latest is not None:
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        self._training_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event):
        self._timer.stop()
        self._training_timer.stop()
        self._cancel()
        super().closeEvent(event)
