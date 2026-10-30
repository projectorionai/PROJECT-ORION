"""Optional diagnostics must not invalidate durable user operations."""
import asyncio
from types import SimpleNamespace

import pytest

import orion_core.file_organiser as organiser
from orion_core.reminders import ReminderService


def broken(*_, **__):
    raise RuntimeError("synthetic telemetry unavailable")


def telemetry(**overrides):
    methods = {name: (lambda *_: None) for name in ("incr", "gauge", "register", "beat")}
    methods.update(overrides)
    return SimpleNamespace(metrics=SimpleNamespace(**methods), health=SimpleNamespace(**methods))


def bus():
    speech, log = [], []
    signals = {name: SimpleNamespace(emit=lambda *_: None) for name in
               ("banner", "dashboard_event", "phone_action")}
    signals["speak_request"] = SimpleNamespace(emit=lambda *args: speech.append(args))
    signals["log"] = SimpleNamespace(emit=lambda *args: log.append(args))
    return SimpleNamespace(**signals), speech


def test_saved_reminder_returns_success_when_added_counter_fails(tmp_path):
    events, _ = bus()
    path = tmp_path / "reminders.json"
    service = ReminderService(events, telemetry(incr=broken), path=path)
    assert service.add("check oven", minutes=1).ok
    assert [r.text for r in ReminderService(events, path=path).active()] == ["check oven"]


def test_fired_counter_failure_does_not_block_spoken_delivery(tmp_path):
    events, speech = bus()
    service = ReminderService(events, path=tmp_path / "reminders.json")
    service.add("check oven", minutes=1)
    service.telemetry = telemetry(incr=broken)
    service._fire(service.active()[0])
    assert speech == [("A reminder: check oven.",)]


def test_failed_health_registration_does_not_wedge_runner_ownership(tmp_path):
    events, _ = bus()
    service = ReminderService(events, telemetry(register=broken), path=tmp_path / "reminders.json")
    service.stop()
    asyncio.run(service.run())
    assert service._running is False


@pytest.mark.parametrize("hook", ["beat", "gauge"])
def test_failed_periodic_telemetry_does_not_abort_delivery_loop(tmp_path, hook):
    events, speech = bus()
    path = tmp_path / "reminders.json"
    service = ReminderService(events, path=path)
    service.add("check oven", minutes=1)
    service.active()[0].due_at = 0
    service.telemetry = telemetry(**{hook: broken})
    def spoken(*args):
        speech.append(args)
        service.stop()
    events.speak_request.emit = spoken
    asyncio.run(service.run())
    assert len(speech) == 1
    assert service._running is False
    assert not ReminderService(events, path=path).active()


def test_organiser_returns_completed_move_evidence_when_metrics_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(organiser, "UNDO_DIR", tmp_path / "journal")
    target = tmp_path / "selected"
    target.mkdir()
    (target / "note.txt").write_text("keep these bytes", encoding="utf-8")
    events, _ = bus()
    service = organiser.FileOrganiser(events, telemetry(incr=broken))
    monkeypatch.setattr(service, "_resolve_target", lambda *_: target)
    result = asyncio.run(service.organise(str(target), apply=True))
    assert result.ok and len(result.evidence) == 1
    assert (target / "Documents" / "note.txt").read_text() == "keep these bytes"
    assert asyncio.run(service.undo()).ok
    assert (target / "note.txt").read_text() == "keep these bytes"


def test_reminder_diagnostics_log_once_per_failed_hook_without_chat_spam(tmp_path, caplog):
    events, _ = bus()
    service = ReminderService(events, telemetry(incr=broken), path=tmp_path / "reminders.json")
    assert service.add("first", minutes=1).ok
    assert service.add("second", minutes=1).ok
    assert sum("Optional reminder telemetry" in r.message for r in caplog.records) == 1


def test_completed_report_returns_its_receipt_when_metrics_fail(tmp_path, monkeypatch):
    import orion_core.reports as reports
    monkeypatch.setattr(reports, "BASE_DIR", tmp_path)
    monkeypatch.setattr(reports, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(reports.ReportDrafter, "_write_docx", lambda *_: False)
    drafter = reports.ReportDrafter(None, SimpleNamespace(has_text_fallback=lambda: False), telemetry(incr=broken))
    result = asyncio.run(drafter.draft("Synthetic telemetry check", brief="Only synthetic notes", sections=["Notes"]))
    assert result.ok
    folders = list(reports.REPORTS_DIR.iterdir())
    assert len(folders) == 1
    assert "Only synthetic notes" in (folders[0] / "report.md").read_text(encoding="utf-8")
    assert (folders[0] / "report.html").is_file()
