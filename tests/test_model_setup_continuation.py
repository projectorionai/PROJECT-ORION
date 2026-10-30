"""Setup failures must reach model callers through the documented error type."""
import io
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orion_core.model_store import ModelDownloadError, download_verified


def test_unwritable_model_directory_uses_download_error_before_network(tmp_path, monkeypatch):
    directory = tmp_path / "unavailable"
    original = Path.mkdir
    opened = []
    def mkdir(path, *args, **kwargs):
        if path == directory:
            raise PermissionError("synthetic model directory unavailable")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(ModelDownloadError, match="model directory unavailable"):
        download_verified("synthetic", directory / "model.bin",
                          opener=lambda *_args, **_kwargs: opened.append(True))
    assert opened == []
    assert not directory.exists()


@pytest.mark.parametrize("limits", [{"total_timeout": "bad"}, {"total_timeout": None},
                                    {"total_timeout": True}, {"timeout": 0},
                                    {"timeout": float("inf")}, {"timeout": float("nan")}])
def test_invalid_network_or_total_deadline_never_starts_a_transfer(tmp_path, limits):
    opened = []
    target = tmp_path / "model.bin"
    target.write_bytes(b"existing model")
    def opener(*_, **__):
        opened.append(True)
        return io.BytesIO(b"new model")
    with pytest.raises(ModelDownloadError):
        download_verified("synthetic", target, opener=opener, **limits)
    assert opened == []
    assert target.read_bytes() == b"existing model"
