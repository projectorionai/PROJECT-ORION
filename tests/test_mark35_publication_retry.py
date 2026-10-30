"""Only Windows rename-sharing failures retry; serialization never repeats."""

import errno
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import research_docx


def _denied(code=5):
    error = PermissionError(errno.EACCES, "synthetic Windows publication sharing denial")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_denial_retries_complete_temp_without_repeating_writer(tmp_path, monkeypatch, code):
    real_replace = research_docx.os.replace
    writes, calls, sleeps = [], [], []

    def writer(path):
        writes.append(path)
        path.write_bytes(b"complete serialized paper")

    def replace(source, target):
        calls.append((Path(source), Path(target)))
        assert Path(source).read_bytes() == b"complete serialized paper"
        if len(calls) == 1:
            raise _denied(code)
        return real_replace(source, target)

    monkeypatch.setattr(research_docx.os, "replace", replace)
    monkeypatch.setattr(research_docx, "_WINDOWS", True, raising=False)
    monkeypatch.setattr(research_docx, "time", SimpleNamespace(sleep=sleeps.append), raising=False)
    result = research_docx._publish_new(tmp_path, "paper", ".docx", writer)
    assert result.read_bytes() == b"complete serialized paper"
    assert len(writes) == 1 and len(calls) == 2
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 0.1
    assert list(tmp_path.iterdir()) == [result]


def test_permanent_windows_denial_is_bounded_and_cleans_only_own_files(tmp_path, monkeypatch):
    original = tmp_path / "paper.docx"
    original.write_bytes(b"prior report")
    calls, writes, sleeps = [], [], []

    def writer(path):
        writes.append(path)
        path.write_bytes(b"new report")

    def replace(source, target):
        calls.append((source, target))
        assert Path(source).read_bytes() == b"new report"
        raise _denied()

    monkeypatch.setattr(research_docx.os, "replace", replace)
    monkeypatch.setattr(research_docx, "_WINDOWS", True)
    monkeypatch.setattr(research_docx, "time", SimpleNamespace(sleep=sleeps.append))
    with pytest.raises(PermissionError):
        research_docx._publish_new(tmp_path, "paper", ".docx", writer)
    assert len(calls) == 5 and len(writes) == 1
    assert len(sleeps) == 4 and 0.175 <= sum(sleeps) <= 0.35
    assert list(tmp_path.iterdir()) == [original]
    assert original.read_bytes() == b"prior report"


@pytest.mark.parametrize("changed", ["populated", "replaced-empty", "replaced-populated"])
def test_changed_reservation_is_never_retried_overwritten_or_deleted(tmp_path, monkeypatch, changed):
    real_replace = research_docx.os.replace
    calls, sleeps = [], []

    def replace(source, target):
        calls.append((source, target))
        if changed == "populated":
            Path(target).write_bytes(b"someone else's data")
        else:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"" if changed == "replaced-empty" else b"someone else's data")
            real_replace(replacement, target)
        raise _denied()

    monkeypatch.setattr(research_docx.os, "replace", replace)
    monkeypatch.setattr(research_docx, "_WINDOWS", True)
    monkeypatch.setattr(research_docx, "time", SimpleNamespace(sleep=sleeps.append))
    with pytest.raises(PermissionError):
        research_docx._publish_new(tmp_path, "paper", ".docx", lambda p: p.write_bytes(b"new paper"))
    assert len(calls) == 1 and sleeps == []
    result = tmp_path / "paper.docx"
    assert result.read_bytes() == (b"" if changed == "replaced-empty" else b"someone else's data")
    assert list(tmp_path.iterdir()) == [result]


@pytest.mark.parametrize("windows,code", [(True, None), (True, 87), (False, 5)])
def test_generic_or_non_windows_errors_do_not_retry(tmp_path, monkeypatch, windows, code):
    calls, sleeps = [], []
    error = _denied(code) if code is not None else PermissionError(errno.EACCES, "generic denial")

    def replace(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(research_docx.os, "replace", replace)
    monkeypatch.setattr(research_docx, "_WINDOWS", windows)
    monkeypatch.setattr(research_docx, "time", SimpleNamespace(sleep=sleeps.append))
    with pytest.raises(PermissionError) as raised:
        research_docx._publish_new(tmp_path, "paper", ".docx", lambda p: p.write_bytes(b"new paper"))
    assert raised.value is error and len(calls) == 1 and sleeps == []
    assert list(tmp_path.iterdir()) == []


def test_success_has_one_write_one_replace_and_no_sleep(tmp_path, monkeypatch):
    real_replace = research_docx.os.replace
    calls, writes, sleeps = [], [], []

    def writer(path):
        writes.append(path)
        path.write_bytes(b"complete paper")

    def replace(*args):
        calls.append(args)
        return real_replace(*args)

    monkeypatch.setattr(research_docx.os, "replace", replace)
    monkeypatch.setattr(research_docx, "time", SimpleNamespace(sleep=sleeps.append))
    path = research_docx._publish_new(tmp_path, "paper", ".docx", writer)
    assert path.read_bytes() == b"complete paper"
    assert len(calls) == len(writes) == 1 and sleeps == []


def test_writer_error_preserves_prior_report_and_cleans_owned_temp(tmp_path, monkeypatch):
    previous = tmp_path / "paper.docx"
    previous.write_bytes(b"old paper")
    calls = []

    def writer(path):
        path.write_bytes(b"partial serialization")
        raise RuntimeError("synthetic writer error")

    monkeypatch.setattr(research_docx.os, "replace", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="writer error"):
        research_docx._publish_new(tmp_path, "paper", ".docx", writer)
    assert calls == [] and list(tmp_path.iterdir()) == [previous]
    assert previous.read_bytes() == b"old paper"


def test_writer_replaced_temp_is_preserved_when_writer_fails(tmp_path):
    from orion_core.atomic_io import _replace
    replacement_path = []

    def writer(path):
        other = tmp_path / "other"
        other.write_bytes(b"replacement has different identity")
        # This is fixture setup, not the publication under test. A transient
        # Windows reader must not prevent reaching the intended identity case.
        _replace(str(other), path)
        replacement_path.append(path)
        raise RuntimeError("synthetic writer error")

    with pytest.raises(RuntimeError, match="writer error"):
        research_docx._publish_new(tmp_path, "paper", ".docx", writer)
    assert list(tmp_path.iterdir()) == replacement_path
    assert replacement_path[0].read_bytes() == b"replacement has different identity"


def test_failed_initial_identity_read_preserves_original_error(tmp_path, monkeypatch):
    error = OSError("synthetic identity read failure")

    def fail(_fd):
        raise error

    monkeypatch.setattr(research_docx.os, "fstat", fail)
    with pytest.raises(OSError) as raised:
        research_docx._publish_new(tmp_path, "paper", ".docx", lambda _p: pytest.fail("writer started"))
    assert raised.value is error
    # Without an identity the temporary path is preserved, not blindly deleted.
    assert len(list(tmp_path.glob(".paper-*.tmp"))) == 1
