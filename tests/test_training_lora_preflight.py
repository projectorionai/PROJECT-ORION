"""
tools/train_lora.py's --check: what a run needs, found without importing a
machine-learning library, installing or downloading anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import train_lora  # noqa: E402


def args_for(tmp_path: Path, **overrides):
    argv = ["--training-root", str(tmp_path), "--check"]
    args = train_lora.finish_args(train_lora.build_parser().parse_args(argv))
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def every_package(name: str) -> object:
    return object() if name != "unsloth" else None


def test_a_cpu_only_pytorch_is_reported_for_a_gpu_run(tmp_path):
    problems = train_lora.preflight(args_for(tmp_path), find_spec=every_package,
                                    torch_version=lambda: "2.8.0+cpu")
    torch = [p for p in problems if p.kind == "package"]
    assert len(torch) == 1 and "CUDA build of PyTorch" in torch[0].what and "2.8.0+cpu" in torch[0].what
    assert "download.pytorch.org/whl/cu" in torch[0].fix


def test_a_cuda_pytorch_passes_and_a_cpu_plumbing_check_needs_none(tmp_path):
    cuda = train_lora.preflight(args_for(tmp_path), find_spec=every_package, torch_version=lambda: "2.8.0+cu128")
    assert [p for p in cuda if p.kind == "package"] == []
    cpu = train_lora.preflight(args_for(tmp_path, device="cpu"), find_spec=every_package,
                               torch_version=lambda: "2.8.0+cpu")
    assert [p for p in cpu if p.kind == "package"] == []


def test_which_builds_count_as_cpu_only():
    assert train_lora.cpu_only_torch("2.8.0+cpu", "linux")
    assert train_lora.cpu_only_torch("2.8.0", "win32")           # the plain PyPI wheel on Windows
    assert not train_lora.cpu_only_torch("2.8.0", "linux")
    assert not train_lora.cpu_only_torch("2.8.0+cu128", "win32")
    assert not train_lora.cpu_only_torch("", "win32")             # not installed is reported elsewhere
