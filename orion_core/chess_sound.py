"""
The sound of a wooden piece being set down.

Synthesised rather than shipped as an asset: a wav file would be one more
binary in the repo, one more thing to license, and one more thing to go
missing. A wood knock is a short, well-understood sound — a sharp transient
followed by a few resonant partials decaying fast — so ORION can make his own.

Deliberately NOT played through ORION's voice renderer. That path is a single
shared output stream carrying his speech, and pushing a click into it would
mean a move sound could interrupt him mid-sentence, or arrive late behind a
queued reply. Qt's own QSoundEffect owns a separate, tiny output and is exactly
the right tool for a UI click.

Failure is always silent: a machine with no working audio output should get a
chess board that works, not an error.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR

SOUND_DIR = CONFIG_DIR / "sounds"
SAMPLE_RATE = 44_100


def _wood_knock(move_kind: str = "move") -> bytes:
    """One wooden click as 16-bit mono PCM.

    Three ingredients, which is what makes it read as WOOD rather than as a
    beep or a tick:

      • a very fast attack and a ~90 ms exponential decay — wood does not ring;
      • two low resonant partials, slightly inharmonic (a solid body, not a
        tuned bar);
      • a burst of noise at the very front for the impact itself, decaying much
        faster than the body.
    """
    if move_kind == "capture":
        # A capture is two pieces meeting: lower, louder, a touch longer.
        base, decay, noise_gain, length = 168.0, 26.0, 0.55, 0.16
    elif move_kind == "castle":
        base, decay, noise_gain, length = 196.0, 30.0, 0.40, 0.13
    elif move_kind == "check":
        base, decay, noise_gain, length = 320.0, 22.0, 0.30, 0.20
    else:
        base, decay, noise_gain, length = 220.0, 34.0, 0.42, 0.11

    count = int(SAMPLE_RATE * length)
    samples = []
    # A cheap deterministic noise source — no RNG state, identical every run.
    seed = 12345
    for index in range(count):
        t = index / SAMPLE_RATE
        envelope = math.exp(-decay * t)
        body = (math.sin(2 * math.pi * base * t) * 0.6
                + math.sin(2 * math.pi * base * 2.76 * t) * 0.25
                + math.sin(2 * math.pi * base * 5.4 * t) * 0.10)
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        noise = ((seed / 0x7FFFFFFF) * 2.0 - 1.0)
        # The impact transient dies far faster than the body.
        strike = noise * noise_gain * math.exp(-260.0 * t)
        value = (body * envelope + strike) * 0.42
        samples.append(max(-1.0, min(1.0, value)))

    # A short fade-out so the buffer never ends on a discontinuity (a click).
    fade = min(400, count // 4)
    for index in range(fade):
        samples[count - fade + index] *= 1.0 - (index / fade)

    return b"".join(struct.pack("<h", int(s * 32767)) for s in samples)


def sound_path(move_kind: str = "move") -> Path | None:
    """Path to the wav for *move_kind*, generating it once. None if unwritable."""
    path = SOUND_DIR / f"chess_{move_kind}.wav"
    if path.is_file() and path.stat().st_size > 1000:
        return path
    try:
        SOUND_DIR.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(_wood_knock(move_kind))
        return path
    except (OSError, wave.Error):
        return None


def classify(board_before: Any, move: Any) -> str:
    """Which sound this move deserves. Never raises — falls back to 'move'."""
    try:
        if board_before.is_capture(move):
            return "capture"
        if board_before.is_castling(move):
            return "castle"
        copy = board_before.copy(stack=False)
        copy.push(move)
        if copy.is_check():
            return "check"
    except Exception:
        pass
    return "move"


class MoveSounds:
    """Plays the piece sounds, and can be switched off.

    Effects are cached per kind because QSoundEffect loads its source
    asynchronously — building a new one per move would drop the first sound
    every time and leak an object per move.
    """

    def __init__(self, enabled: bool = True, volume: float = 0.35) -> None:
        self.enabled = bool(enabled)
        self.volume = max(0.0, min(1.0, float(volume)))
        self._effects: dict[str, Any] = {}
        self._unavailable = False

    def _effect(self, kind: str) -> Any:
        if kind in self._effects:
            return self._effects[kind]
        try:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtMultimedia import QSoundEffect
        except Exception:
            self._unavailable = True
            return None
        path = sound_path(kind)
        if path is None:
            self._unavailable = True
            return None
        effect = QSoundEffect()
        effect.setSource(QUrl.fromLocalFile(str(path)))
        effect.setVolume(self.volume)
        self._effects[kind] = effect
        return effect

    def play(self, kind: str = "move") -> bool:
        """Play the sound for *kind*. Returns whether anything was played."""
        if not self.enabled or self._unavailable:
            return False
        effect = self._effect(kind if kind in
                              ("move", "capture", "castle", "check") else "move")
        if effect is None:
            return False
        try:
            effect.setVolume(self.volume)
            effect.play()
            return True
        except Exception:
            return False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def prewarm(self) -> None:
        """Generate the wavs up front so the first move is not silent."""
        for kind in ("move", "capture", "castle", "check"):
            sound_path(kind)


__all__ = ["MoveSounds", "SAMPLE_RATE", "SOUND_DIR", "classify", "sound_path"]
