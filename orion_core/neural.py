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
    SparseMLP       hashed sparse features -> hidden -> softmax, trained
                    row-sparse so a step costs what the batch used
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
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from zipfile import BadZipFile

from .atomic_io import atomic_write_bytes


def _np():
    import numpy as np
    return np


def _checkpoint_array(archive: Any, key: str, shape: tuple[int, ...]) -> Any:
    """Reject corrupt parameters before publishing a loaded network."""
    np = _np()
    value = archive[key].astype(np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid checkpoint parameter: {key}")
    return value


def _batch_weights(weights: Any, n: int) -> Any:
    np = _np()
    w = np.ones(n, dtype=np.float32) if weights is None else np.asarray(weights, np.float32)
    if (n == 0 or w.shape != (n,) or not np.isfinite(w).all()
            or (w < 0).any() or not 0 < float(w.sum()) < float("inf")):
        raise ValueError("Training needs a nonempty batch and finite positive total weight")
    return w / w.sum()


def _optimizer_step(header: dict[str, Any]) -> int:
    step = header.get("t", 0)
    if type(step) is not int or not 0 <= step <= 2**53:
        raise ValueError("Invalid optimizer step")
    return step


def _moment(archive: Any, key: str, shape: tuple[int, ...], *, variance=False) -> Any:
    value = _checkpoint_array(archive, key, shape)
    if variance and (value < 0).any():
        raise ValueError("Negative optimizer variance")
    return value


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


def _checkpoint_header(archive: Any) -> dict[str, Any]:
    """Validate JSON metadata before allocating or publishing a network."""
    header = json.loads(bytes(archive["header"]).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("checkpoint header must be an object")
    sizes = header.get("sizes")
    if (not isinstance(sizes, list) or len(sizes) < 2
            or any(type(size) is not int or size <= 0 for size in sizes)):
        raise ValueError("checkpoint dimensions must be positive integers")

    def finite_number(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    for name in ("lr", "l2"):
        value = header.get(name)
        if not finite_number(value) or value < 0:
            raise ValueError(f"invalid checkpoint hyperparameter: {name}")
    if header.get("kind") != "sparse" and header.get("hidden") not in ("relu", "tanh"):
        raise ValueError("invalid checkpoint hidden activation")
    for name in ("stats", "meta"):
        value = header.get(name)
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint {name} must be an object")
        header[name] = value
    for name, value in header["stats"].items():
        if name not in TrainStats.__dataclass_fields__:
            raise ValueError(f"unknown checkpoint statistic: {name}")
        if name in ("steps", "samples"):
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid checkpoint counter: {name}")
        elif not finite_number(value):
            raise ValueError(f"invalid checkpoint statistic: {name}")
    return header


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
        w = _batch_weights(weights, n)
        if x.ndim != 2 or x.shape[1] != self.sizes[0] or not np.isfinite(x).all():
            raise ValueError("Invalid training inputs")
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("Gradient clip must be positive")
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
            if labels.shape != (n,) or (labels < 0).any() or (labels >= out.shape[1]).any():
                raise ValueError("Invalid class labels")
            probs = np.clip(out[np.arange(n), labels], 1e-9, 1.0)
            loss = float(-(np.log(probs) * w).sum())
            delta = out.copy()
            delta[np.arange(n), labels] -= 1.0
        else:
            target = np.asarray(y, dtype=np.float32).reshape(out.shape)
            if not np.isfinite(target).all():
                raise ValueError("Non-finite targets")
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

    def to_bytes(self, meta: dict[str, Any] | None = None, *, include_optimizer=True) -> bytes:
        np = _np()
        arrays = {f"W{i}": W for i, W in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        header = {"sizes": self.sizes, "hidden": self.hidden, "output": self.output,
                  "lr": self.lr, "l2": self.l2, "t": self._t,
                  "stats": vars(self.stats), "meta": meta or {}}
        if include_optimizer:
            header["optimizer"] = "adam-v1"
            arrays.update({f"m{i}": m for i, m in enumerate(self._m)})
            arrays.update({f"v{i}": v for i, v in enumerate(self._v)})
        arrays["header"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple["MLP", dict[str, Any]]:
        np = _np()
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            header = _checkpoint_header(archive)
            net = cls(header["sizes"], hidden=header["hidden"], output=header["output"],
                      lr=header["lr"], l2=header["l2"])
            net.W = [_checkpoint_array(archive, f"W{i}", p.shape)
                     for i, p in enumerate(net.W)]
            net.b = [_checkpoint_array(archive, f"b{i}", p.shape)
                     for i, p in enumerate(net.b)]
            if header.get("optimizer") is not None:
                if header["optimizer"] != "adam-v1":
                    raise ValueError("Unknown optimizer format")
                net._t = _optimizer_step(header)
                net._m = [_moment(archive, f"m{i}", p.shape) for i, p in enumerate(net.W + net.b)]
                net._v = [_moment(archive, f"v{i}", p.shape, variance=True)
                          for i, p in enumerate(net.W + net.b)]
        # Legacy files have no moments: the constructor's zero clock is correct.
        net.stats = TrainStats(**header["stats"])
        return net, header["meta"]

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
        except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError):
            return None

    @property
    def parameters(self) -> int:
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))


class SparseMLP:
    """A classifier over hashed sparse features: a lookup table, one ReLU
    hidden layer, a softmax.

    A sample switches on a few dozen of the table's rows and the hidden layer
    is their (scaled) sum — the same arithmetic as a dense first layer over a
    one-hot input, without multiplying thousands of zeros. Training touches
    only the rows a batch used: the gradient comes from a batch x touched-rows
    matrix and Adam updates just those rows (the "lazy" variant), so a step
    costs what the batch used, not the table's full size.
    """

    def __init__(self, features: int, hidden: int, outputs: int, *,
                 lr: float = 3e-3, l2: float = 1e-6, seed: int = 0) -> None:
        np = _np()
        rng = np.random.default_rng(seed)
        self.sizes = [int(features), int(hidden), int(outputs)]
        self.lr = float(lr)
        self.l2 = float(l2)
        self.table = (rng.standard_normal((features, hidden)) * 0.05).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, outputs)) * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(outputs, dtype=np.float32)
        self._reset_moments()
        self.stats = TrainStats()

    def _reset_moments(self) -> None:
        np = _np()
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self._dense().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._dense().items()}
        self._tm = np.zeros_like(self.table)
        self._tv = np.zeros_like(self.table)
        self._row_steps = np.zeros(self.table.shape[0], dtype=np.int64)

    def _dense(self) -> dict[str, Any]:
        return {"b1": self.b1, "W2": self.W2, "b2": self.b2}

    # ── forward ──────────────────────────────────────────────────────────────

    def _pre(self, rows: Sequence[Any]) -> Any:
        np = _np()
        z = np.zeros((len(rows), self.b1.shape[0]), dtype=np.float32)
        for i, active in enumerate(rows):
            if len(active):
                z[i] = self.table[active].sum(axis=0) * (1.0 / np.sqrt(len(active)))
        return z + self.b1

    def forward(self, rows: Sequence[Any]) -> Any:
        """Class probabilities, one row per sample (each a list of active indices)."""
        np = _np()
        h = np.maximum(self._pre(rows), 0.0)
        o = h @ self.W2 + self.b2
        o -= o.max(axis=1, keepdims=True)
        e = np.exp(o)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, active: Sequence[int]) -> Any:
        """Probabilities for one sample."""
        np = _np()
        return self.forward([np.asarray(active, dtype=np.int64)])[0]

    # ── learning ─────────────────────────────────────────────────────────────

    def train_batch(self, rows: Sequence[Any], labels: Sequence[int], weights: Any = None,
                    clip: float = 5.0, away: Sequence[bool] | None = None,
                    live_outputs: int | None = None) -> float:
        """One Adam step on a minibatch. Each row's indices must be distinct
        (features() returns a sorted set). Returns the (weighted) loss.

        *away* marks rows whose label is a MISTAKE: those rows are trained to
        make the label less likely (unlikelihood, -log(1 - p)) instead of more,
        which is how a correction becomes a negative lesson. *live_outputs*
        (the first N outputs) is where that probability goes: a head with spare,
        unallocated outputs would otherwise hand it to actions that do not exist."""
        np = _np()
        rows = [np.unique(np.asarray(r, dtype=np.int64)) for r in rows]
        n = len(rows)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        w = _batch_weights(weights, n)
        if (y.shape != (n,) or (y < 0).any() or (y >= self.sizes[-1]).any()
                or any((r < 0).any() or (r >= self.sizes[0]).any() for r in rows)):
            raise ValueError("Invalid sparse training batch")
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("Gradient clip must be positive")
        neg = np.zeros(n, dtype=bool) if away is None else np.asarray(away, dtype=bool).reshape(-1)
        if neg.shape != (n,):
            raise ValueError("away needs one flag per row")
        pre = self._pre(rows)
        h = np.maximum(pre, 0.0)
        o = h @ self.W2 + self.b2
        o -= o.max(axis=1, keepdims=True)
        p = np.exp(o)
        p /= p.sum(axis=1, keepdims=True)
        py = p[np.arange(n), y].copy()
        per_row = -np.log(np.clip(py, 1e-9, 1.0))
        d = p
        d[np.arange(n), y] -= 1.0
        if neg.any():
            # d/do of -log(1 - p_y) is p_y * (onehot_y - q), q being the softmax
            # over every OTHER label. Written that way it stays exact when the
            # net is certain (p_y rounds to 1.0, where 1 - p_y is 0).
            rows_neg = np.flatnonzero(neg)
            others = o[rows_neg].astype(np.float64)
            others[np.arange(len(rows_neg)), y[rows_neg]] = -np.inf
            if live_outputs is not None and 1 < int(live_outputs) < others.shape[1]:
                others[:, int(live_outputs):] = -np.inf
            top = others.max(axis=1, keepdims=True)
            q = np.exp(others - top)
            mass = q.sum(axis=1, keepdims=True)
            q /= mass
            full = np.log(np.exp(o[rows_neg].astype(np.float64)).sum(axis=1))
            per_row[rows_neg] = full - (np.log(mass[:, 0]) + top[:, 0])   # -log(1 - p_y)
            grad = -q * py[rows_neg, None]
            grad[np.arange(len(rows_neg)), y[rows_neg]] = py[rows_neg]
            d[rows_neg] = grad.astype(d.dtype)
        loss = float((per_row * w).sum())
        d *= w[:, None]
        grads = {"W2": h.T @ d + self.l2 * self.W2, "b2": d.sum(axis=0)}
        dh = (d @ self.W2.T) * (pre > 0)
        grads["b1"] = dh.sum(axis=0)
        # The table: a gradient only for the rows this batch switched on.
        used = [r for r in rows if len(r)]
        touched = np.unique(np.concatenate(used)) if used else np.zeros(0, dtype=np.int64)
        table_grad = None
        if len(touched):
            x = np.zeros((n, len(touched)), dtype=np.float32)
            for i, r in enumerate(rows):
                if len(r):
                    x[i, np.searchsorted(touched, r)] = 1.0 / np.sqrt(len(r))
            table_grad = x.T @ dh + self.l2 * self.table[touched]
        norm = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads.values())
                             + (float((table_grad ** 2).sum()) if table_grad is not None else 0.0)))
        scale = clip / norm if norm > clip else 1.0
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1, c2 = 1 - b1 ** self._t, 1 - b2 ** self._t
        for key, param in self._dense().items():
            g = grads[key] * scale
            self._m[key] = b1 * self._m[key] + (1 - b1) * g
            self._v[key] = b2 * self._v[key] + (1 - b2) * (g * g)
            param -= (self.lr * (self._m[key] / c1) / (np.sqrt(self._v[key] / c2) + eps)).astype(np.float32)
        if table_grad is not None:
            g = table_grad * scale
            m = b1 * self._tm[touched] + (1 - b1) * g
            v = b2 * self._tv[touched] + (1 - b2) * (g * g)
            self._tm[touched], self._tv[touched] = m, v
            # Lazy rows need their own bias-correction clock. A rare/new word
            # must not inherit thousands of updates made to unrelated words.
            self._row_steps[touched] += 1
            row_c1 = (1 - b1 ** self._row_steps[touched])[:, None]
            row_c2 = (1 - b2 ** self._row_steps[touched])[:, None]
            self.table[touched] -= (self.lr * (m / row_c1) / (np.sqrt(v / row_c2) + eps)).astype(np.float32)
        self.stats.note(loss, n)
        return loss

    def reset_momentum(self, active: Sequence[int] | None = None) -> None:
        """Drop Adam's first moments (the dense layers', and the table rows in
        *active*). Before a correction: the lesson just taught was the mistake,
        and its momentum would carry the net further into it for a dozen
        steps (measured: 0.95 -> 0.97 before turning) before the correction
        could pull it back."""
        np = _np()
        for key in self._m:
            self._m[key][...] = 0.0
        if active is not None:
            rows = np.unique(np.asarray(active, dtype=np.int64))
            rows = rows[(rows >= 0) & (rows < self.sizes[0])]
            self._tm[rows] = 0.0

    # ── persistence ──────────────────────────────────────────────────────────

    def to_bytes(self, meta: dict[str, Any] | None = None, *, include_optimizer=True) -> bytes:
        np = _np()
        arrays = {"table": self.table, "b1": self.b1, "W2": self.W2, "b2": self.b2}
        header = {"kind": "sparse", "sizes": self.sizes, "lr": self.lr, "l2": self.l2,
                  "stats": vars(self.stats), "meta": meta or {}}
        if include_optimizer:
            header.update(optimizer="lazy-adam-v1", t=self._t)
            arrays.update({"tm": self._tm, "tv": self._tv, "row_steps": self._row_steps})
            arrays.update({"m_" + k: v for k, v in self._m.items()})
            arrays.update({"v_" + k: v for k, v in self._v.items()})
        arrays["header"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple["SparseMLP", dict[str, Any]]:
        np = _np()
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            header = _checkpoint_header(archive)
            if header.get("kind") != "sparse":
                raise ValueError("not a sparse network")
            features, hidden, outputs = header["sizes"]
            net = cls(features, hidden, outputs, lr=header["lr"], l2=header["l2"])
            net.table = _checkpoint_array(archive, "table", net.table.shape)
            net.b1 = _checkpoint_array(archive, "b1", net.b1.shape)
            net.W2 = _checkpoint_array(archive, "W2", net.W2.shape)
            net.b2 = _checkpoint_array(archive, "b2", net.b2.shape)
            if header.get("optimizer") is not None:
                if header["optimizer"] != "lazy-adam-v1":
                    raise ValueError("Unknown sparse optimizer format")
                net._t = _optimizer_step(header)
                net._m = {k: _moment(archive, "m_" + k, p.shape) for k, p in net._dense().items()}
                net._v = {k: _moment(archive, "v_" + k, p.shape, variance=True)
                          for k, p in net._dense().items()}
                net._tm = _moment(archive, "tm", net.table.shape)
                net._tv = _moment(archive, "tv", net.table.shape, variance=True)
                counts = archive["row_steps"]
                if (counts.shape != (features,) or counts.dtype.kind not in "iu"
                        or (counts < 0).any() or (counts > net._t).any()):
                    raise ValueError("Invalid sparse optimizer clocks")
                net._row_steps = counts.astype(np.int64)
        net.stats = TrainStats(**header["stats"])
        return net, header["meta"]

    def save(self, path: Path, meta: dict[str, Any] | None = None) -> bool:
        try:
            atomic_write_bytes(Path(path), self.to_bytes(meta))
            return True
        except OSError:
            return False

    @classmethod
    def load(cls, path: Path) -> tuple["SparseMLP", dict[str, Any]] | None:
        try:
            return cls.from_bytes(Path(path).read_bytes())
        except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError):
            return None

    @property
    def parameters(self) -> int:
        return int(self.table.size + self.b1.size + self.W2.size + self.b2.size)


class ReplayMemory:
    """Experience, kept and revisited.

    Learning only from the latest example makes a network forget everything
    else it knew (catastrophic forgetting). A bounded memory sampled in
    minibatches is what lets a small net keep what it learned last week while
    it learns today.
    """

    def __init__(self, capacity: int = 5000, seed: int = 0) -> None:
        self.items: deque[tuple[Any, Any, float]] = deque(maxlen=int(capacity))
        #: Experience that is never evicted (pin): what the network must
        #: never forget, however much new experience arrives. The intent
        #: network pins the tools' own phrases - in one FIFO memory of 12,000
        #: they were the OLDEST rows, so from the 2,086th lesson on every new
        #: lesson pushed one of them out (9,915 of them, 29 Oct 2026).
        self.pinned: list[tuple[Any, Any, float]] = []
        self._rng = random.Random(seed)

    def add(self, x: Any, y: Any, weight: float = 1.0) -> None:
        self.items.append((x, y, float(weight)))

    def pin(self, x: Any, y: Any, weight: float = 1.0) -> None:
        self.pinned.append((x, y, float(weight)))

    def clear(self) -> None:
        self.items.clear()
        self.pinned.clear()

    def __len__(self) -> int:
        return len(self.pinned) + len(self.items)

    def sample(self, k: int) -> list[tuple[Any, Any, float]]:
        """*k* rows drawn uniformly from the pinned and the recent rows,
        without copying the memory (it copied all 12,000 rows per call)."""
        pinned, items = self.pinned, self.items
        total = len(pinned) + len(items)
        if not total:
            return []
        first = len(pinned)
        picks = self._rng.sample(range(total), min(k, total))
        return [pinned[i] if i < first else items[i - first] for i in picks]


__all__ = ["MLP", "ReplayMemory", "SparseMLP", "TrainStats"]
