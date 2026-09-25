"""
One ORION at a time.

Nothing prevented a second ORION from starting — not a lock, not a mutex, not
a check of any kind. Every launch started a whole new assistant: its own audio
capture on the same microphone, its own live session spending its own tokens,
its own writers against the same SQLite files. Launch it again because the
first one seemed slow to appear and you have two. Again and you have three.

The property that matters most here is the one a lock *file* cannot give: the
lock must be released when the process dies, however it dies. A crash, a kill
or a power cut must not leave ORION unable to start, because refusing to start
ever is a worse failure than starting twice.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.single_instance import Claim, claim  # noqa: E402


@pytest.fixture(autouse=True)
def _private_lock_name(monkeypatch):
    """A lock name of this test's own. The real one is machine-global, so
    tests run in parallel workers (pytest -n) took it from each other, and a
    running ORION would take it from every test. Child processes apply the
    same suffix from ORION_TEST_LOCK_TAG."""
    import uuid

    from orion_core import single_instance as si

    tag = "-test-" + uuid.uuid4().hex[:12]
    monkeypatch.setattr(si, "MUTEX_NAME", si.MUTEX_NAME + tag)
    monkeypatch.setattr(si, "SOCKET_NAME", si.SOCKET_NAME + tag)
    monkeypatch.setenv("ORION_TEST_LOCK_TAG", tag)
    yield
    si.release()


#: Prepended to every child script so it claims the same private name.
_TAGGED = '''
import os
import orion_core.single_instance as _si
_tag = os.environ.get("ORION_TEST_LOCK_TAG", "")
_si.MUTEX_NAME += _tag
_si.SOCKET_NAME += _tag
'''

#: Run in a child process, so the lock is claimed by a genuinely separate
#: process rather than by a second call in this one.
_CHILD = '''
import sys
sys.path.insert(0, r"{root}")
{tagged}
from orion_core.single_instance import claim
result = claim()
print("GRANTED" if result.granted else "REFUSED")
'''


def _child_claim() -> str:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.format(root=str(ROOT).replace("\\", "/"), tagged=_TAGGED)],
        capture_output=True, text=True, timeout=120)
    return (proc.stdout or proc.stderr).strip().splitlines()[-1] if (
        proc.stdout or proc.stderr) else ""


def _release(held: Claim) -> None:
    handle = held.handle
    if handle is None:
        return
    import os

    if os.name == "nt":
        import ctypes

        ctypes.WinDLL("kernel32").CloseHandle(int(handle))
    else:
        try:
            handle.close()
        except Exception:
            pass


def test_the_first_orion_is_granted_the_lock():
    held = claim()
    try:
        assert held.granted is True
        assert bool(held) is True
    finally:
        _release(held)


def test_a_second_orion_is_refused():
    """The whole point. Launching again while one runs must not start another
    assistant on the same microphone, tokens and databases."""
    held = claim()
    try:
        assert held.granted is True
        assert _child_claim() == "REFUSED"
    finally:
        _release(held)


def test_the_lock_is_released_when_the_holder_ends():
    """The property a lock FILE cannot give.

    A file has to be deleted to be released, so a crash leaves it claiming
    ORION is running when he is not — and then he never starts again, which
    is worse than the problem being solved. A kernel object is reclaimed by
    the operating system however the process ends.
    """
    held = claim()
    assert held.granted is True
    _release(held)
    assert _child_claim() == "GRANTED", (
        "the lock outlived its holder; ORION would refuse to start after a "
        "crash")


def test_a_refused_claim_says_why():
    held = claim()
    try:
        refused = claim()
        if refused.granted:
            pytest.skip("this platform's lock is not re-entrant-safe in-process")
        assert refused.reason
    finally:
        _release(held)


def test_a_refused_claim_does_not_keep_the_lock_alive():
    """The bug that made ORION unstartable, rather than startable twice.

    Found by launching the real executable: it appeared and vanished in 0.4 s
    with exit code 0 — the signature of a refused claim — while no ORION was
    running anywhere on the machine. The holders were test processes hung in
    teardown, each of which had been refused a claim once and had kept the
    handle. A named mutex lives while ANY handle to it is open, so a refusal
    propped the object up after its owner had gone and every launch after that
    was turned away in silence.

    So: be refused, then release the real holder, and a fresh process must
    still be granted the lock.
    """
    held = claim()
    assert held.granted is True
    refused = claim()
    if refused.granted:
        pytest.skip("this platform's lock is not re-entrant-safe in-process")
    assert refused.handle is None, (
        "a refused claim is holding a handle to the lock; it will outlive the "
        "holder and ORION will never start again")
    _release(held)
    assert _child_claim() == "GRANTED", (
        "the refused claim kept the lock alive after its owner released it")


def test_a_claim_is_truthy_in_the_obvious_way():
    """`if claim():` has to mean what it looks like it means."""
    assert bool(Claim(True)) is True
    assert bool(Claim(False)) is False


def test_failing_to_check_never_blocks_a_launch(monkeypatch):
    """Starting twice is a bad day. An assistant that refuses to start at all
    because a lock could not be created is a worse one, and much harder to
    diagnose."""
    import orion_core.single_instance as module

    monkeypatch.setattr(module, "_claim_windows", lambda: None)
    monkeypatch.setattr(module, "_claim_posix", lambda: None)
    monkeypatch.setattr(module, "_claim_lockfile",
                        lambda directory=None: Claim(True, "not enforcing"))
    assert module.claim().granted is True


def test_a_lock_directory_that_cannot_be_written_still_starts(tmp_path,
                                                              monkeypatch):
    import orion_core.single_instance as module

    blocked = tmp_path / "file-not-a-directory"
    blocked.write_text("x", encoding="utf-8")
    result = module._claim_lockfile(blocked / "nested")
    assert result.granted is True, (
        "an unwritable lock location must not stop ORION starting")


# ── it is wired in, as early as possible ──────────────────────────────────────

def test_the_launcher_claims_before_anything_else():
    """Before the GUI, before a microphone is opened, before a live session is
    dialled. A second instance that gets as far as opening the microphone has
    already taken it from the first."""
    source = (ROOT / "orion.py").read_text(encoding="utf-8")
    assert "_claim_single_instance" in source
    claim_at = source.index("_INSTANCE_CLAIM = _claim_single_instance()")
    for later in ("from orion_core.app import main",
                  "from orion_core.server import main"):
        if later in source:
            assert source.index(later) > claim_at, (
                f"{later} runs before the lock is claimed")


def test_the_claim_is_held_for_the_life_of_the_process():
    """Letting it be garbage-collected would release the lock while ORION is
    still running, which is the same as not having one."""
    source = (ROOT / "orion.py").read_text(encoding="utf-8")
    assert "_INSTANCE_CLAIM" in source, "the claim is not kept anywhere"


def test_a_restart_releases_the_lock_before_spawning():
    """Otherwise the successor claims it, finds it held by a process that is
    still exiting, and refuses to start — a restart that quietly becomes a
    shutdown.

    This test used to assert the release happened through `_INSTANCE_CLAIM`,
    which is how it was done and why it did not work: reaching that name meant
    `import orion`, and orion.py is the entry script, so the import re-ran the
    guard and raised SystemExit before the respawn. The test encoded the bug.
    """
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    start = source.index("if restart_requested:")
    body = source[start:start + 2500]
    assert "release()" in body
    # The CALL, not the word: a comment above it explains why Popen is used
    # rather than os.execv, and matching that put the "spawn" before the
    # release and failed a correct implementation.
    assert body.index("release()") < body.index(
        "subprocess.Popen([sys.executable]")


def test_a_frozen_restart_does_not_pass_itself_as_an_argument():
    """Frozen, sys.executable IS ORION.exe and argv[0] is ORION.exe too, so
    the old form launched him with his own path as a positional argument."""
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    assert "sys.argv[1:] if getattr(sys, \"frozen\", False)" in source


# ── restarting ───────────────────────────────────────────────────────────────
#
# The lock broke ORION's restart, and it broke it in the worst available way:
# silently, by turning it into a shutdown.
#
# app.py released the lock before respawning by doing `import orion` and
# reading _INSTANCE_CLAIM off it. But orion.py is the ENTRY script, so it is
# __main__, and importing it by name builds a SECOND copy of the module —
# which re-runs the guard, finds the mutex held by this very process, and
# calls `raise SystemExit(0)`.
#
# SystemExit inherits from BaseException, not Exception, so it walked straight
# through the `except Exception: pass` that was meant to make the release
# best-effort, out of the restart block, and ended the process several lines
# BEFORE subprocess.Popen. Nothing logged a reason, because exiting is what
# SystemExit is for.


def test_releasing_lets_the_successor_start():
    """The whole point of releasing before a respawn."""
    held = claim()
    assert held.granted is True
    assert _child_claim() == "REFUSED"

    from orion_core.single_instance import release

    assert release() is True
    assert _child_claim() == "GRANTED", (
        "the successor was refused, so the restart is a shutdown")


def test_the_module_knows_what_it_granted():
    """Releasing must not depend on reaching whichever module called claim().
    That dependency is what broke the restart."""
    from orion_core.single_instance import held, release

    got = claim()
    try:
        assert held() is not None
    finally:
        release()
    assert held() is None


def test_releasing_twice_is_harmless():
    from orion_core.single_instance import release

    claim()
    assert release() is True
    assert release() is False       # nothing left to close, and no exception


def test_a_refused_claim_is_not_registered_as_held():
    """Otherwise releasing would close the refused handle and leave the real
    lock in place — which is the original bug wearing a different hat."""
    from orion_core.single_instance import held, release

    first = claim()
    try:
        second = claim()
        if second.granted:
            pytest.skip("this platform's lock is not re-entrant-safe in-process")
        assert held() is first, "the refused claim replaced the real one"
    finally:
        release()


def test_the_restart_does_not_import_the_entry_script():
    """`import orion` re-runs the guard and raises SystemExit out of a place
    that cannot catch it."""
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    start = source.index("if restart_requested:")
    body = source[start:start + 2000]
    assert "import orion\n" not in body, (
        "the restart path imports the entry script again, which re-runs the "
        "single-instance guard and exits before the respawn")
    assert "from .single_instance import release" in body


def test_the_restart_catches_what_can_actually_be_raised():
    """SystemExit is a BaseException. `except Exception` does not catch it,
    which is precisely how the restart became a silent shutdown."""
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    start = source.index("if restart_requested:")
    body = source[start:start + 2000]
    release_at = body.index("release()")
    guard = body[release_at:release_at + 400]
    assert "except BaseException" in guard, (
        "a BaseException raised while releasing the lock would skip the "
        "respawn entirely")


def test_the_respawn_still_happens_after_the_release():
    source = (ROOT / "orion_core" / "app.py").read_text(encoding="utf-8")
    start = source.index("if restart_requested:")
    body = source[start:start + 2500]
    assert body.index("release()") < body.index("subprocess.Popen(["), (
        "the lock is released after the respawn, so the successor is refused")


def test_release_works_through_enforce_not_just_claim():
    """orion.py calls enforce(), not claim().

    Testing only the convenient entry point is how the original defect
    survived: the restart path went through code no test exercised.
    """
    from orion_core.single_instance import enforce, held, release

    got = enforce()
    assert got.granted is True
    assert held() is not None, "enforce() did not register what it granted"
    assert _child_claim() == "REFUSED"
    assert release() is True
    assert _child_claim() == "GRANTED"


# -- the refusal notice belongs on the diagnostic stream ---------------------

_ENFORCE_CHILD = """
import json, sys
sys.path.insert(0, r"{root}")
{tagged}
from orion_core.single_instance import enforce
got = enforce()
print(json.dumps({{"granted": got.granted}}))
"""


def test_a_refused_launch_leaves_stdout_clean():
    """Whoever launched the second ORION may be READING its stdout.

    The desktop launcher's preflight does exactly that: it runs a subprocess
    that imports orion.py and parses one line of JSON back. Announcing the
    refusal on stdout put an English sentence in front of that JSON, the parse
    failed, and a perfectly healthy launcher reported a broken one -- but only
    while ORION happened to be open, which is why it read as five unrelated
    flaky tests rather than as one bug.
    """
    from orion_core.single_instance import enforce, release

    mine = enforce()
    assert mine.granted, "could not take the lock to test being refused"
    try:
        child = subprocess.run(
            [sys.executable, "-c",
             _ENFORCE_CHILD.format(root=str(ROOT).replace("\\", "/"), tagged=_TAGGED)],
            capture_output=True, text=True, timeout=120)
        # The child was refused, and said so -- on stderr.
        assert json.loads(child.stdout.strip()) == {"granted": False}, (
            f"stdout was not parseable JSON: {child.stdout!r}")
        assert "already running" in child.stderr, (
            "the refusal was never reported anywhere")
    finally:
        release()


def test_the_notice_survives_having_no_console():
    """Under pythonw there is no console and sys.stderr is None. A refused
    launch must not turn into a traceback."""
    from orion_core import single_instance

    saved = sys.stderr
    sys.stderr = None
    try:
        single_instance._notify_refused("[ORION] anything")
    finally:
        sys.stderr = saved


def test_importing_the_entry_script_does_not_claim():
    """Importing orion.py is not starting a second ORION.

    The desktop launcher's preflight imports it in a subprocess to check
    dependencies, and diagnostics do the same. With the claim at module
    scope, every one of those raised SystemExit(0) the moment a real ORION
    was already running -- so the preflight produced no output, and a healthy
    launcher reported a broken one. It surfaced as five unrelated flaky tests
    because it only happened while ORION was open.
    """
    body = (ROOT / "orion.py").read_text(encoding="utf-8")
    assert '_claim_single_instance() if __name__ == "__main__"' in body


def test_the_claim_still_happens_before_the_app_is_imported():
    """A refused second instance must exit before building the whole
    application graph, not after."""
    body = (ROOT / "orion.py").read_text(encoding="utf-8")
    assert body.index("_INSTANCE_CLAIM = _claim_single_instance()") < \
        body.index("from orion_core.app import main")


def test_a_real_launch_is_still_refused_while_one_is_running():
    """The whole point of the guard, and the thing the fix must not undo."""
    from orion_core.single_instance import enforce, release

    mine = enforce()
    assert mine.granted
    try:
        assert _child_claim() == "REFUSED"
    finally:
        release()
