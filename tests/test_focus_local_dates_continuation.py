"""User-facing focus day reports must use the supplied local day."""
from datetime import datetime, timedelta, timezone

from orion_core.focus import FocusEngine, FocusStore


def test_completed_block_after_local_midnight_counts_as_today(tmp_path):
    zone = timezone(timedelta(hours=1))
    store = FocusStore(tmp_path / "focus.db")
    engine = FocusEngine(store)
    try:
        engine.start("late study", minutes=10, now=datetime(2026, 10, 7, 0, 10, tzinfo=zone))
        engine.complete(now=datetime(2026, 10, 7, 0, 25, tzinfo=zone))
        report = engine.today(now=datetime(2026, 10, 7, 0, 30, tzinfo=zone))
        assert report == {"sessions": 1, "completed": 1, "focus_minutes": 15}
    finally:
        store.close()


def test_consecutive_local_days_keep_streak_across_midnight_offset(tmp_path):
    zone = timezone(timedelta(hours=1))
    store = FocusStore(tmp_path / "focus.db")
    engine = FocusEngine(store)
    try:
        for day, hour in ((5, 0), (6, 23)):
            start = datetime(2026, 10, day, hour, 10, tzinfo=zone)
            engine.start("study", minutes=10, now=start)
            engine.complete(now=start + timedelta(minutes=15))
        assert engine.streak(now=datetime(2026, 10, 7, 0, 30, tzinfo=zone)) == 2
    finally:
        store.close()
