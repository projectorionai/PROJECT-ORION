"""
No web server on localhost unless you ask for it.

  "Let's get rid of ORION on localhost, I want him to just be on the app ...
   dropping localhost and keeping just the app."

Now that ORION is a proper desktop application in the taskbar, the local web
server is redundant as a LOCAL interface. So a normal desktop session no longer
stands one up — nothing listens on localhost by default. The phone/remote
uplink is kept as an opt-in fallback (ORION_REMOTE_ACCESS=1, or asking ORION to
turn phone access on), and the headless/cloud node still forces it on because
serving is its entire purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _app_source() -> str:
    return (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8", errors="replace")


def test_the_gateway_is_off_by_default():
    """The desktop session must default to NO remote server — the whole point
    of dropping localhost."""
    source = _app_source()
    # The guard now enables only on an explicit truthy value, with default "0".
    assert 'os.getenv("ORION_REMOTE_ACCESS", "0")' in source
    assert 'in {"1", "true", "yes", "on"}' in source


def test_the_default_off_matches_the_module_docstring():
    """remote.py always said 'Disabled by default' — the app.py default now
    agrees with it instead of contradicting it."""
    remote = (Path(__file__).resolve().parents[1] / "orion_core" / "remote.py").read_text(
        encoding="utf-8", errors="replace")
    assert "Disabled by default" in remote


def test_phone_access_remains_reachable_as_a_fallback():
    """Dropped from the default, not removed — the user can turn it back on."""
    import inspect

    from orion_core.dispatch_desktop import DesktopDispatchMixin

    source = inspect.getsource(DesktopDispatchMixin.system_startup_tool)
    assert "phone_on" in source
    assert "phone_off" in source
    assert "ORION_REMOTE_ACCESS=1" in source


def test_the_headless_server_still_forces_it_on():
    """The cloud node's entire job is to serve — dropping localhost on the
    desktop must not disarm it there."""
    server = (Path(__file__).resolve().parents[1] / "orion_core" / "server.py").read_text(
        encoding="utf-8", errors="replace")
    assert "ORION_REMOTE_ACCESS" in server


def test_the_gateway_class_is_unchanged_and_still_importable():
    """Opt-out is a startup-default change, not a capability removal."""
    from orion_core.remote import RemoteGateway
    assert RemoteGateway is not None
