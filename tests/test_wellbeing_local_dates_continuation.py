"""Wellbeing summaries and focus correlations share the user's local dates."""
from datetime import datetime, timedelta, timezone

from orion_core.focus import FocusEngine, FocusStore
from orion_core.time_service import TIME
from orion_core.wellbeing import WellbeingEngine, WellbeingStore


def test_today_excludes_prior_local_day_even_when_both_are_same_utc_day(tmp_path):
    zone = timezone(timedelta(hours=1))
    store = WellbeingStore(tmp_path / "wellbeing.db")
    engine = WellbeingEngine(store)
    try:
        engine.checkin(energy=1, at=datetime(2026, 10, 6, 23, 50, tzinfo=zone))
        engine.checkin(energy=5, at=datetime(2026, 10, 7, 0, 10, tzinfo=zone))
        assert engine.today(now=datetime(2026, 10, 7, 0, 30, tzinfo=zone)) == {
            "checkins": 1, "avg_energy": 5.0, "logged": True}
    finally:
        store.close()


def test_default_today_uses_orions_configured_clock(tmp_path, monkeypatch):
    now = datetime(2026, 10, 7, 0, 30, tzinfo=timezone(timedelta(hours=1)))
    monkeypatch.setattr(TIME, "_clock", lambda: now)
    store = WellbeingStore(tmp_path / "wellbeing.db")
    engine = WellbeingEngine(store)
    try:
        engine.checkin(energy=4, at=now)
        assert engine.today()["checkins"] == 1
    finally:
        store.close()


def test_correlations_pair_checkin_and_focus_on_local_day(tmp_path):
    zone = timezone(timedelta(hours=1))
    wb_store, focus_store = WellbeingStore(tmp_path / "wellbeing.db"), FocusStore(tmp_path / "focus.db")
    wellbeing, focus = WellbeingEngine(wb_store), FocusEngine(focus_store)
    try:
        for day, energy, minutes in ((4, 1, 10), (5, 3, 30), (6, 5, 50)):
            wellbeing.checkin(energy=energy, at=datetime(2026, 10, day, 0, 10, tzinfo=zone))
            start = datetime(2026, 10, day, 12, 0, tzinfo=zone)
            focus.start("synthetic study", minutes=minutes, now=start)
            focus.complete(now=start + timedelta(minutes=minutes))
        results = wellbeing.correlate(focus=focus, now=datetime(2026, 10, 7, 12, tzinfo=zone))
        correlation = next(row for row in results if row["pair"] == "energy vs focus minutes")
        assert correlation["n"] == 3
        assert correlation["coefficient"] == 1.0
    finally:
        wb_store.close()
        focus_store.close()
