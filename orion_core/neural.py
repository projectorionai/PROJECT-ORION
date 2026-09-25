"""
ORION's neural engine — small networks that learn from his own experience.

The request was a brain that learns, "much like the fly brain project". The
honest reading of that project is not "simulate 140,000 neurons" — a connectome
replayed in software does not make an assistant any wiser about its user. What
it showed is that behaviour worth having comes from connections shaped by
experience. That is what this engine gives ORION: networks whose weights are
written by what actually happens to him, persisted, and used the next time.

Deliberately small and dependency-light (numpy only, CPU, milliseconds per
step), because it trains *while he works*, on a desktop, from the handful of
examples one person generates — not in a datacentre.

    MLP             dense layers (ReLU/tanh hidden, linear/tanh/softmax out),
                    Adam, L2, gradient clipping; sparse-input fast paths
    ReplayMemory    a bounded, weighted store of experience, sampled in
                    minibatches — learning from the past, not only the latest
    save / load     one .npz per network, written atomically

Used by: the chess intuition network (chess_brain.py) and the language→action
network that learns which tool you mean from how YOU phrase things
(intent_brain.py).
"""

from __future__ import annotations

import io
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .atomic_io import atomic_write_bytes


def _np():
    import numpy as np
    return np


@dataclass
class TrainStats:
    steps: int = 0
    samples: int = 0
    last_loss: float = 0.0
    ema_loss: float = 0.0

    def note(self, loss: float, batch: int) -> None:
        self.steps += 1
        self.samples += batch
        self.last_loss = float(loss)
        self.ema_loss = loss if self.steps == 1 else 0.98 * self.ema_loss + 0.02 * loss


class MLP:
    """A multi-layer perceptron with Adam, for small online problems.

    ``output`` picks the head and its loss: "linear" (MSE), "tanh" (MSE on
    [-1, 1] targets) or "softmax" (cross-entropy on class indices).
    """

    def __init__(self, sizes: Sequence[int], *, hidden: str = "relu",
                 output: str = "linear", lr: float = 1e-3, l2: float = 1e-5,
                 seed: int = 0) -> None:
        np = _np()
        if len(sizes) < 2:
            raise ValueError("an MLP needs at least an input and an output size")
        if output not in {"linear", "tanh", "softmax"}:
            raise ValueError(f"unknown output '{output}'")
        self.sizes = [int(s) for s in sizes]
        self.hidden = hidden
        self.output = output
        self.lr = float(lr)
        self.l2 = float(l2)
        rng = np.random.default_rng(seed)
        self.W: list[Any] = []
        self.b: list[Any] = []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            scale = np.sqrt(2.0 / fan_in) if hidden == "relu" else np.sqrt(1.0 / fan_in)
            self.W.append((rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32))
            self.b.append(np.zeros(fan_out, dtype=np.float32))
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0
        self.stats = TrainStats()

    # ── forward ──────────────────────────────────────────────────────────────

    def _act(self, z: Any) -> Any:
        np = _np()
        return np.maximum(z, 0.0) if self.hidden == "relu" else np.tanh(z)

    def _head(self, z: Any) -> Any:
        np = _np()
        if self.output == "tanh":
            return np.tanh(z)
        if self.output == "softmax":
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        return z

    def forward(self, x: Any) -> Any:
        np = _np()
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = self._head(z) if i == len(self.W) - 1 else self._act(z)
        return a

    def forward_sparse(self, active: Sequence[int], values: Sequence[float] | None = None) -> Any:
        """One sample whose input is mostly zeros: only the rows of the first
        layer that are switched on are summed (the NNUE trick). Returns the
        output vector."""
        np = _np()
        idx = np.asarray(active, dtype=np.int64)
        if values is None:
            z = self.W[0][idx].sum(axis=0) + self.b[0]
        else:
            z = (self.W[0][idx] * np.asarray(values, dtype=np.float32)[:, None]).sum(axis=0) \
                + self.b[0]
        a = self._head(z[None, :]) if len(self.W) == 1 else self._act(z)[None, :]
        for i in range(1, len(self.W)):
            z = a @ self.W[i] + self.b[i]
            a = self._head(z) if i == len(self.W) - 1 else self._act(z)
        return a[0]

    # ── learning ─────────────────────────────────────────────────────────────

    def train_batch(self, x: Any, y: Any, weights: Any = None, clip: float = 5.0) -> float:
        """One Adam step on a minibatch. Returns the (weighted) loss."""
        np = _np()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        n = x.shape[0]
        w = np.ones(n, dtype=np.float32) if weights is None else np.asarray(weights, np.float32)
        w = w / max(1e-8, float(w.sum()))
        acts = [x]
        pre = []
        a = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            pre.append(z)
            a = self._head(z) if i == len(self.W) - 1 else self._act(z)
            acts.append(a)
        out = acts[-1]
        if self.output == "softmax":
            labels = np.asarray(y, dtype=np.int64).reshape(-1)
            probs = np.clip(out[np.arange(n), labels], 1e-9, 1.0)
            loss = float(-(np.log(probs) * w).sum())
            delta = out.copy()
            delta[np.arange(n), labels] -= 1.0
        else:
            target = np.asarray(y, dtype=np.float32).reshape(out.shape)
            err = out - target
            loss = float(((err ** 2).mean(axis=1) * w).sum())
            delta = 2.0 * err / out.shape[1]
            if self.output == "tanh":
                delta = delta * (1.0 - out ** 2)
        delta = delta * w[:, None]
        grads_W: list[Any] = [None] * len(self.W)
        grads_b: list[Any] = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            grads_W[i] = acts[i].T @ delta + self.l2 * self.W[i]
            grads_b[i] = delta.sum(axis=0)
            if i:
                delta = delta @ self.W[i].T
                if self.hidden == "relu":
                    delta = delta * (pre[i - 1] > 0)
                else:
                    delta = delta * (1.0 - acts[i] ** 2)
        grads = grads_W + grads_b
        norm = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads)))
        if norm > clip:
            grads = [g * (clip / norm) for g in grads]
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = self.W + self.b
        for k, (p, g) in enumerate(zip(params, grads)):
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            m_hat = self._m[k] / (1 - b1 ** self._t)
            v_hat = self._v[k] / (1 - b2 ** self._t)
            p -= (self.lr * m_hat / (np.sqrt(v_hat) + eps)).astype(p.dtype)
        self.stats.note(loss, n)
        return loss

    # ── persistence ──────────────────────────────────────────────────────────

    def to_bytes(self, meta: dict[str, Any] | None = None) -> bytes:
        np = _np()
        arrays = {f"W{i}": W for i, W in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        header = {"sizes": self.sizes, "hidden": self.hidden, "output": self.output,
                  "lr": self.lr, "l2": self.l2, "t": self._t,
                  "stats": vars(self.stats), "meta": meta or {}}
        arrays["header"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple["MLP", dict[str, Any]]:
        np = _np()
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            header = json.loads(bytes(archive["header"]).decode("utf-8"))
            net = cls(header["sizes"], hidden=header["hidden"], output=header["output"],
                      lr=header["lr"], l2=header["l2"])
            net.W = [archive[f"W{i}"].astype(np.float32) for i in range(len(net.W))]
            net.b = [archive[f"b{i}"].astype(np.float32) for i in range(len(net.b))]
        net._m = [np.zeros_like(p) for p in net.W + net.b]
        net._v = [np.zeros_like(p) for p in net.W + net.b]
        net._t = int(header.get("t", 0))
        for key, value in (header.get("stats") or {}).items():
            if hasattr(net.stats, key):
                setattr(net.stats, key, value)
        return net, header.get("meta") or {}

    def save(self, path: Path, meta: dict[str, Any] | None = None) -> bool:
        try:
            atomic_write_bytes(Path(path), self.to_bytes(meta))
            return True
        except OSError:
            return False

    @classmethod
    def load(cls, path: Path) -> tuple["MLP", dict[str, Any]] | None:
        try:
            return cls.from_bytes(Path(path).read_bytes())
        except (OSError, ValueError, KeyError):
            return None

    @property
    def parameters(self) -> int:
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))


class ReplayMemory:
    """Experience, kept and revisited.

    Learning only from the latest example makes a network forget everything
    else it knew (catastrophic forgetting). A bounded memory sampled in
    minibatches is what lets a small net keep what it learned last week while
    it learns today.
    """

    def __init__(self, capacity: int = 5000, seed: int = 0) -> None:
        self.items: deque[tuple[Any, Any, float]] = deque(maxlen=int(capacity))
        self._rng = random.Random(seed)

    def add(self, x: Any, y: Any, weight: float = 1.0) -> None:
        self.items.append((x, y, float(weight)))

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, k: int) -> list[tuple[Any, Any, float]]:
        if not self.items:
            return []
        k = min(k, len(self.items))
        return self._rng.sample(list(self.items), k)


__all__ = ["MLP", "ReplayMemory", "TrainStats"]
