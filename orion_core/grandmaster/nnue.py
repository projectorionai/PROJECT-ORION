"""Worker-only PyTorch HalfKP 40960x128 -> 256 -> 32 -> 32 -> 1.

Targets: y = 0.8 sigmoid(q_stm / 300) + 0.2 (wdl_stm + 1)/2.
Loss: BCEWithLogits(z, y). Inference score: 300 z centipawns.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from torch import nn

from .dataset import Dataset
from .weights import FEATURES, SCALE_CP, WIDTH, export_weights, features, read_weights


class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0.0, max=1.0)


class HalfKP(nn.Module):
    def __init__(self, *, factorized=True, material_init=True):
        super().__init__()
        # Sparse INPUT: gather <=30 active rows per perspective, never create
        # Bx40960 one-hot tensors. Dense gradients keep AdamW well-defined.
        self.feature = nn.Embedding(FEATURES, WIDTH, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(WIDTH))
        self.hidden1 = nn.Linear(2 * WIDTH, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        self.clip = ClippedReLU()
        nn.init.normal_(self.feature.weight, mean=0, std=0.01)
        with torch.no_grad():
            self.feature.weight[0].zero_()
        # Share piece-square lessons across all 64 king contexts during
        # training. Export folds this factor into the existing HalfKP table,
        # so playing costs exactly the same and old weights remain readable.
        self.shared = nn.Embedding(641, WIDTH, padding_idx=0)
        nn.init.zeros_(self.shared.weight)
        self.shared.weight.requires_grad_(factorized)
        if material_init:
            self._init_material()
        self._material_anchors = []
        if material_init:
            for name, index in self._material_indices():
                self._material_anchors.append((name, index, self.get_parameter(name)[index].detach().clone()))

    @staticmethod
    def _material_indices():
        return (("feature.weight", (slice(None), slice(0, 2))),
                ("shared.weight", (slice(None), slice(0, 2))),
                ("hidden1.weight", slice(0, 2)), ("hidden1.bias", slice(0, 2)),
                ("hidden2.weight", slice(0, 2)), ("hidden2.bias", slice(0, 2)),
                ("output.weight", (slice(None), slice(0, 2))))

    def constrain_material(self, *, gradients=False):
        """Preserve the material prior; train the remaining positional path."""
        with torch.no_grad():
            for name, index, anchor in self._material_anchors:
                parameter = self.get_parameter(name)
                if gradients:
                    if parameter.grad is not None:
                        parameter.grad[index].zero_()
                else:
                    parameter[index].copy_(anchor.to(parameter.device))

    def _init_material(self):
        """Start with exact material balance, leaving capacity for residuals.

        Two nonnegative channels carry our/their material through the clipped
        layers. The remaining channels learn positional corrections. This
        gives new networks useful gradients and known chess structure before
        any training, without adding work or tensors to exported inference.
        """
        scale = 20000.0  # safely above legal material totals, including promotions
        values = torch.tensor([100, 320, 330, 500, 900], dtype=torch.float32)
        with torch.no_grad():
            self.feature.weight[:, :2].zero_()
            blocks = self.feature.weight[1:].view(64, 10, 64, WIDTH)
            blocks[:, :5, :, 0] = values[None, :, None] / scale
            blocks[:, 5:, :, 1] = values[None, :, None] / scale
            self.hidden1.weight[:2].zero_()
            self.hidden1.weight[0, 0] = 1
            self.hidden1.weight[1, 1] = 1
            self.hidden1.bias.zero_()
            self.hidden2.weight[:2].zero_()
            self.hidden2.weight[0, 0] = 1
            self.hidden2.weight[1, 1] = 1
            self.hidden2.bias.zero_()
            self.output.weight.zero_()
            self.output.bias.zero_()
            self.output.weight[0, 0] = scale / SCALE_CP
            self.output.weight[0, 1] = -scale / SCALE_CP

    def accumulate(self, active):
        shared_ids = torch.where(active == 0, 0, (active - 1) % 640 + 1)
        return self.clip((self.feature(active) + self.shared(shared_ids)).sum(dim=1) + self.bias)

    def forward(self, us, them):
        first = self.accumulate(us)
        second = self.accumulate(them)
        x = torch.cat((first, second), dim=1)
        return self.output(self.clip(self.hidden2(self.clip(self.hidden1(x))))).squeeze(-1)

    def load_nnue(self, path):
        # Copy read-only mmap arrays; autograd must never modify a live mapping.
        state = {key: torch.from_numpy(value.copy()) for key, value in read_weights(path).items()}
        state["shared.weight"] = torch.zeros_like(self.shared.weight)
        self.load_state_dict(state, strict=True)
        # Keep old arbitrary NNUE files compatible. Only anchor models that
        # actually contain the exact material path, including resumed exports.
        if not all(torch.equal(self.get_parameter(name)[index], anchor)
                   for name, index, anchor in self._material_anchors):
            self._material_anchors = []


def batch(rows, device="cpu"):
    import chess
    us, them, cp, wdl = [], [], [], []
    for fen, value, outcome in rows:
        board = chess.Board(fen)
        if not board.is_valid() or outcome not in (-1, 0, 1) or not math.isfinite(value):
            raise ValueError("Invalid training row")
        ours, theirs = features(board, board.turn), features(board, not board.turn)
        if len(ours) > 30 or len(theirs) > 30:
            raise ValueError("Invalid HalfKP position")
        us.append(list(ours) + [0] * (30 - len(ours)))
        them.append(list(theirs) + [0] * (30 - len(theirs)))
        cp.append(max(-20000, min(20000, value)))
        wdl.append((outcome + 1) / 2)
    q = torch.tensor(cp, dtype=torch.float32, device=device)
    outcomes = torch.tensor(wdl, dtype=torch.float32, device=device)
    target = 0.8 * torch.sigmoid(q / SCALE_CP) + 0.2 * outcomes
    return (torch.tensor(us, dtype=torch.long, device=device),
            torch.tensor(them, dtype=torch.long, device=device), target)


def cuda_budget(free: int, total: int, *, cap=1280 * 1024**2, reserve=1536 * 1024**2) -> int:
    """Fail closed on unknown/insufficient VRAM. Never budget beyond an 8 GiB card."""
    usable_total = min(total, 8 * 1024**3)
    allowance = min(cap, usable_total - reserve, free - reserve)
    return allowance if allowance >= 768 * 1024**2 else 0


def train_pass(directory: str, initial: str | None, stop, result_pipe,
               steps=128, batch_size=128, device="auto", seed=20260927, factorized=True,
               material_init=True, quiet_only=False):
    """A short-lived process owns ALL CUDA allocations, including its context.
    Exit releases the context; empty_cache alone cannot promise that.
    """
    directory = Path(directory)
    dataset = None
    label_policy = "quiet-v2" if quiet_only else "all-positions-v1"

    def report(result):
        result_pipe.send({**result, "label_policy": label_policy})

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if not 1 <= steps <= 4096 or not 1 <= batch_size <= 256:
            raise ValueError("Unbounded training pass")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            if not torch.cuda.is_available():
                report({"status": "deferred", "reason": "CUDA unavailable"})
                return
            free, total = torch.cuda.mem_get_info()
            budget = cuda_budget(free, total)
            if not budget:
                report({"status": "deferred", "reason": "VRAM reserve unavailable"})
                return
            # Caching allocator cap plus global free-memory reserve. Context and
            # non-Torch consumers are covered by the reserve, not this fraction.
            torch.cuda.set_per_process_memory_fraction(budget / total)
        elif device != "cpu":
            raise ValueError("Unsupported training device")
        if stop.is_set():
            return
        dataset = Dataset(directory / "positions.sqlite3")
        if dataset.count() < 1024:
            report({"status": "deferred", "reason": "Need 1024 terminal-labelled positions"})
            return
        rng = random.Random(seed)
        torch.manual_seed(seed)
        heldout = dataset.validation_rows(1024, random.Random(seed), quiet_only=quiet_only)
        if len(heldout) < 32:
            report({"status": "deferred", "reason": "Need held-out complete games"})
            return
        model = HalfKP(factorized=factorized, material_init=material_init)
        if initial:
            model.load_nnue(initial)
        model.to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
        loss_function = nn.BCEWithLogitsLoss()
        validation = batch(heldout, device)
        with torch.no_grad():
            before = loss_function(model(*validation[:2]), validation[2]).item()
        final_loss = before
        best_loss, best_state, best_step, stale = before, None, 0, 0
        completed = 0
        for step in range(steps):
            if stop.is_set():
                return
            if device == "cuda":
                free, _total = torch.cuda.mem_get_info()
                if free < 1024**3:
                    report({"status": "deferred", "reason": "GPU pressure"})
                    return
            rows = dataset.sample(batch_size, rng, quiet_only=quiet_only)
            if len(rows) != batch_size:
                raise ValueError("Insufficient training split")
            ours, theirs, targets = batch(rows, device)
            optimiser.zero_grad(set_to_none=True)
            for group in optimiser.param_groups:
                group["lr"] = 2e-4 * (0.1 + 0.9 * (1 + math.cos(math.pi * step / steps)) / 2)
            loss = loss_function(model(ours, theirs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            model.constrain_material(gradients=True)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            model.constrain_material()
            # Bounds accumulator weights and keeps the padding row exact.
            with torch.no_grad():
                model.feature.weight.clamp_(-1.0, 1.0)
                model.feature.weight[0].zero_()
                model.shared.weight.clamp_(-1.0, 1.0)
                model.shared.weight[0].zero_()
            final_loss = float(loss.detach().cpu())   # batch completion checkpoint
            completed = step + 1
            if completed % 32 == 0 or completed == steps:
                model.eval()
                with torch.no_grad():
                    measured = loss_function(model(*validation[:2]), validation[2]).item()
                if not math.isfinite(measured):
                    raise ValueError("Non-finite validation loss")
                if measured < best_loss - 1e-7:
                    best_loss, best_step, stale = measured, completed, 0
                    best_state = {key: tensor.detach().cpu().clone()
                                  for key, tensor in model.state_dict().items()}
                else:
                    stale += 1
                model.train()
                if stale >= 4:
                    break
        if stop.is_set():
            return
        after = best_loss
        if best_state is None:
            report({"status": "rejected", "before": before, "after": after})
            return
        # Unique candidate: never overwrite an active/mapped file on Windows.
        import uuid
        path = directory / ("candidate-" + uuid.uuid4().hex + ".nnue")
        digest = export_weights(best_state, path)
        report({"status": "candidate", "path": str(path), "sha256": digest,
                          "loss": final_loss, "before": before, "after": after,
                          "steps": completed, "best_step": best_step, "seed": seed,
                          "validation_positions": len(heldout), "device": device,
                          "factorized": factorized, "material_init": material_init})
    except Exception as exc:  # noqa: BLE001 - report CUDA/native failures at the process boundary
        report({"status": "error", "reason": type(exc).__name__ + ": " + str(exc)[:180]})
    finally:
        if dataset is not None:
            dataset.close()
        result_pipe.close()


def training_main(*args, **kwargs):
    train_pass(*args, **kwargs)
