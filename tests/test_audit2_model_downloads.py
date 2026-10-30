"""Concurrent downloads must own their temporary files through publication."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading

import pytest

from orion_core.model_store import ModelDownloadError, download_verified


def test_overlapping_downloads_to_one_target_both_publish_verified_bytes(tmp_path):
    barrier = threading.Barrier(2, timeout=10)
    body = b"complete model bytes"
    class Response:
        headers = {"Content-Length": str(len(body))}
        def __enter__(self):
            self.index = 0
            return self
        def __exit__(self, *_):
            return False
        def read(self, _):
            barrier.wait()  # both calls have opened their staging files
            self.index += 1
            return body if self.index == 1 else b""
    target = tmp_path / "model.bin"
    def download():
        return download_verified("unused", target, sha256=hashlib.sha256(body).hexdigest(),
                                 opener=lambda *_, **__: Response())
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(download) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=15))
            except Exception as exc:
                outcomes.append((type(exc).__name__, str(exc)))
    assert outcomes == [len(body), len(body)]
    assert target.read_bytes() == body
    assert list(tmp_path.iterdir()) == [target]


def test_rejected_overlap_cannot_mutate_an_already_verified_model(tmp_path):
    opened = threading.Barrier(2, timeout=10)
    published = threading.Event()
    good = b"verified model"
    bad = b"tampered model"
    target = tmp_path / "model.bin"
    class Response:
        headers = {}
        def __init__(self, is_good):
            self.is_good, self.index = is_good, 0
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self, _):
            self.index += 1
            if self.index > 1:
                return b""
            opened.wait()
            if not self.is_good:
                assert published.wait(10)
            return good if self.is_good else bad
    def download(is_good):
        try:
            return download_verified("unused", target, sha256=hashlib.sha256(good).hexdigest(),
                                     opener=lambda *_, **__: Response(is_good))
        finally:
            if is_good:
                published.set()
    with ThreadPoolExecutor(max_workers=2) as executor:
        accepted = executor.submit(download, True)
        rejected = executor.submit(download, False)
        assert accepted.result(timeout=15) == len(good)
        try:
            rejected.result(timeout=15)
        except ModelDownloadError:
            pass
        else:
            raise AssertionError("Invalid checksum was accepted")
    assert target.read_bytes() == good


def test_staging_failure_preserves_existing_model_and_error_contract(tmp_path, monkeypatch):
    import orion_core.model_store as models
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous model")
    def full_disk(**_):
        raise OSError("disk full")
    monkeypatch.setattr(models.tempfile, "mkstemp", full_disk)
    with pytest.raises(ModelDownloadError, match="disk full"):
        download_verified("unused", target)
    assert target.read_bytes() == b"previous model"


def test_model_publication_retries_a_transient_windows_sharing_lock(tmp_path, monkeypatch):
    import io
    import orion_core.model_store as models
    original = models.os.replace
    attempts = []
    def transient(source, target):
        attempts.append(source)
        if len(attempts) == 1:
            raise PermissionError("transient sharing lock")
        return original(source, target)
    monkeypatch.setattr(models.os, "replace", transient)
    target = tmp_path / "model.bin"
    assert download_verified("unused", target, size=5, opener=lambda *_, **__: io.BytesIO(b"model")) == 5
    assert len(attempts) == 2 and target.read_bytes() == b"model"
    assert list(tmp_path.iterdir()) == [target]


def test_an_unannounced_oversized_response_never_replaces_the_model(tmp_path):
    import io
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous model")
    with pytest.raises(ModelDownloadError, match="larger"):
        download_verified("synthetic", target, max_bytes=8,
                          opener=lambda *_, **__: io.BytesIO(b"ninebytes"))
    assert target.read_bytes() == b"previous model"
    assert list(tmp_path.iterdir()) == [target]


def test_a_slow_read_cannot_publish_after_the_total_deadline(tmp_path, monkeypatch):
    import io
    import orion_core.model_store as models
    clock = iter((0.0, 0.0, 5.0))
    monkeypatch.setattr(models.time, "monotonic", lambda: next(clock))
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous model")
    with pytest.raises(ModelDownloadError, match="time budget"):
        download_verified("synthetic", target, total_timeout=1,
                          opener=lambda *_, **__: io.BytesIO(b"model"))
    assert target.read_bytes() == b"previous model"


@pytest.mark.parametrize("limits", [{"total_timeout": float("nan")}, {"total_timeout": float("inf")},
                                    {"total_timeout": 0}, {"max_bytes": 0}])
def test_download_limits_cannot_disable_the_resource_bound(tmp_path, limits):
    with pytest.raises(ModelDownloadError, match="finite and positive"):
        download_verified("synthetic", tmp_path / "model.bin", **limits)


def test_a_late_end_of_stream_cannot_publish_after_the_deadline(tmp_path, monkeypatch):
    import io
    import orion_core.model_store as models
    clock = iter((0.0, 0.0, 0.0, 0.0, 5.0))
    monkeypatch.setattr(models.time, "monotonic", lambda: next(clock))
    target = tmp_path / "model.bin"
    with pytest.raises(ModelDownloadError, match="time budget"):
        download_verified("synthetic", target, total_timeout=1,
                          opener=lambda *_, **__: io.BytesIO(b"model"))
    assert not target.exists()


def test_a_directory_cannot_be_used_as_a_model(tmp_path):
    from orion_core.model_store import usable
    assert not usable(tmp_path)
