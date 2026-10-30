"""Regression coverage for independent focus, study, decision and export fixes."""

from __future__ import annotations

import asyncio
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from orion_core.decisions import calibration_bins, describe_calibration
from orion_core.focus import FocusEngine, FocusStore, resolve_cadence
from orion_core.reports import ReportDrafter
from orion_core.study import StudyEngine, StudyStore
from orion_core.wellbeing import Checkin, WellbeingEngine, WellbeingStore, pearson

NOW = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)


@pytest.fixture
def focus(tmp_path):
    store = FocusStore(tmp_path / "focus.db")
    yield FocusEngine(store)
    store.close()


@pytest.fixture
def study(tmp_path):
    store = StudyStore(tmp_path / "study.db")
    yield StudyEngine(store)
    store.close()


@pytest.fixture
def wellbeing(tmp_path):
    store = WellbeingStore(tmp_path / "wellbeing.db")
    yield WellbeingEngine(store)
    store.close()


@pytest.mark.parametrize("minutes", [None, 25, 0])
def test_focus_honours_explicit_zero_break(minutes):
    work, rest = resolve_cadence("deep", minutes=minutes, break_minutes=0)
    assert rest == 0
    assert work == (50 if minutes is None else max(1, minutes))


def test_invalid_replacement_keeps_focus_running(focus):
    original = focus.start("revision", now=NOW)
    with pytest.raises(ValueError):
        focus.start("replacement", minutes="invalid", now=NOW)
    assert focus.active().id == original.id
    assert len(focus.store.all()) == 1


def test_failed_focus_insert_rolls_back_previous_finish(focus):
    original = focus.start("revision", now=NOW)
    focus.store._db.execute(
        "CREATE TRIGGER fail_session BEFORE INSERT ON sessions "
        "BEGIN SELECT RAISE(ABORT, 'injected write failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        focus.start("replacement", now=NOW + timedelta(minutes=10))
    assert focus.active().id == original.id
    assert not focus.store._db.in_transaction


def test_parallel_focus_starts_leave_one_active(focus):
    with ThreadPoolExecutor(max_workers=4) as pool:
        sessions = list(pool.map(
            lambda n: focus.start(f"block {n}", now=NOW), range(12)))
    assert len({s.id for s in sessions}) == 12
    assert sum(not s.ended_at for s in focus.store.all()) == 1


def test_focus_replacement_is_atomic_across_connections(focus):
    other = FocusStore(focus.store.path)
    try:
        engines = [focus, FocusEngine(other)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: engines[i % 2].start(f"block {i}", now=NOW), range(12)))
        assert len(focus.store.all()) == 12
        assert sum(not s.ended_at for s in focus.store.all()) == 1
    finally:
        other.close()


def test_parallel_interruptions_preserve_the_count(focus):
    focus.start("work", now=NOW)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: focus.interrupt(), range(20)))
    assert focus.active().interruptions == 20


def test_study_failure_keeps_schedule_and_pending_review(study):
    card = study.add("Question", "Answer")
    study.next_due(now=NOW)
    study.store._db.execute(
        "CREATE TRIGGER fail_review BEFORE INSERT ON reviews "
        "BEGIN SELECT RAISE(ABORT, 'injected write failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        study.grade(5, now=NOW)
    assert study.store.get(card.id).reviews == 0
    assert study._current.reviews == 0
    assert not study.store._db.in_transaction
    study.store._db.execute("DROP TRIGGER fail_review")
    assert study.grade(5, now=NOW).reviews == 1
    assert study.store._db.execute("SELECT count(*) FROM reviews").fetchone()[0] == 1


def test_deleted_pending_card_is_not_graded(study):
    card = study.add("Question", "Answer")
    study.next_due(now=NOW)
    study.store.delete(card.id)
    assert study.grade(5, now=NOW) is None
    assert study._current is None
    assert study.store._db.execute("SELECT count(*) FROM reviews").fetchone()[0] == 0


def test_pending_card_does_not_overwrite_newer_schedule(study):
    card = study.add("Question", "Answer")
    study.next_due(now=NOW)
    second_engine = StudyEngine(study.store)
    second_engine.grade(5, card_id=card.id, now=NOW)
    assert study.grade(5, now=NOW).reviews == 2
    assert study.store._db.execute("SELECT count(*) FROM reviews").fetchone()[0] == 2


@pytest.mark.parametrize(("quality", "expected"), [(-10, 0), (10, 5)])
def test_review_log_uses_same_grade_as_scheduler(study, quality, expected):
    card = study.add("Question", "Answer")
    study.grade(quality, card_id=card.id, now=NOW)
    assert study.store._db.execute("SELECT quality FROM reviews").fetchone()[0] == expected


def test_parallel_study_grades_preserve_every_review(study):
    card = study.add("Question", "Answer")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: study.grade(1, card_id=card.id, now=NOW), range(20)))
    assert study.store.get(card.id).reviews == 20
    assert study.store._db.execute("SELECT count(*) FROM reviews").fetchone()[0] == 20


def test_study_reviews_are_atomic_across_connections(study):
    card = study.add("Question", "Answer")
    other = StudyStore(study.store.path)
    try:
        engines = [study, StudyEngine(other)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: engines[i % 2].grade(1, card_id=card.id, now=NOW),
                          range(20)))
        assert study.store.get(card.id).reviews == 20
        assert study.store._db.execute("SELECT count(*) FROM reviews").fetchone()[0] == 20
    finally:
        other.close()


@pytest.mark.parametrize("deck", [None, "General"])
def test_study_retention_excludes_future_reviews(study, deck):
    card = study.add("Question", "Answer")
    study.grade(5, card_id=card.id, now=NOW - timedelta(days=1))
    study.grade(0, card_id=card.id, now=NOW + timedelta(days=1))
    assert study.store.retention(deck, now=NOW) == 100.0


def test_calibration_counts_low_confidence_and_uses_actual_predictions():
    bins = calibration_bins([(0.0, 0.0), (0.2, 0.0), (0.49, 0.5), (1.0, 1.0)])
    assert sum(b["n"] for b in bins) == 4
    certain = next(b for b in bins if b["band"] == "90-100%")
    assert certain["stated"] == 100 and certain["gap"] == 0


def test_calibration_verdict_weights_samples():
    pairs = [(0.5, 1.0)] * 50 + [(0.5, 0.0)] * 50 + [(0.9, 0.0)] * 2
    assert "Well calibrated" in describe_calibration(calibration_bins(pairs))


def test_opposite_calibration_errors_do_not_cancel():
    pairs = [(0.1, 0.9)] * 10 + [(0.9, 0.1)] * 10
    verdict = describe_calibration(calibration_bins(pairs))
    assert "Well calibrated" not in verdict
    assert "varies" in verdict.lower()


def test_failed_checkin_does_not_leak_into_later_commit(wellbeing):
    store = wellbeing.store
    store._db.execute(
        "CREATE TRIGGER fail_factor BEFORE INSERT ON factors "
        "WHEN NEW.tag='bad' BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    checkin = Checkin(at=NOW.isoformat(), factors=["first", "bad"])
    with pytest.raises(sqlite3.IntegrityError):
        store.add(checkin)
    assert checkin.id is None
    assert not store._db.in_transaction
    wellbeing.checkin(energy=4, at=NOW)
    assert len(store.all()) == 1
    assert store._db.execute("SELECT count(*) FROM factors").fetchone()[0] == 0


def test_wellbeing_history_uses_bounded_queries(wellbeing):
    for i in range(30):
        wellbeing.checkin(energy=3, factors=[f"tag {i}", "shared"], at=NOW)
    queries = []
    wellbeing.store._db.set_trace_callback(queries.append)
    try:
        history = wellbeing.store.since(NOW - timedelta(days=1))
    finally:
        wellbeing.store._db.set_trace_callback(None)
    assert len(history) == 30
    assert history[0].factors == ["tag 0", "shared"]
    assert history[-1].factors == ["tag 29", "shared"]
    assert sum(q.lstrip().upper().startswith("SELECT") for q in queries) <= 2


@pytest.mark.parametrize("sleep", [float("inf"), float("-inf"), float("nan"), -1, 25])
def test_invalid_sleep_is_missing_not_a_misleading_average(wellbeing, sleep):
    assert wellbeing.checkin(sleep_hours=sleep, at=NOW).sleep_hours is None
    assert wellbeing.trend(now=NOW)["avg_sleep"] is None


def test_non_finite_optional_scores_do_not_crash(wellbeing):
    checkin = wellbeing.checkin(energy=math.inf, stress=-math.inf, mood=math.inf)
    assert (checkin.energy, checkin.stress, checkin.mood_valence) == (None, None, None)
    assert pearson([1.0, math.inf], [1.0, 2.0]) is None


def test_large_finite_correlation_does_not_overflow():
    assert pearson([-1e308, 0.0, 1e308], [-1e308, 0.0, 1e308]) == 1.0


def test_old_invalid_sleep_data_is_ignored_without_rewriting_it(wellbeing):
    checkin = wellbeing.store.add(Checkin(at=NOW.isoformat(), sleep_hours=math.inf))
    assert wellbeing.trend(now=NOW)["avg_sleep"] is None
    raw = wellbeing.store._db.execute(
        "SELECT sleep_hours FROM checkins WHERE id=?", (checkin.id,)).fetchone()[0]
    assert math.isinf(raw)


def test_single_factor_string_is_preserved(wellbeing):
    checkin = wellbeing.checkin(factors="exercise", at=NOW)
    assert checkin.factors == ["exercise"]
    assert wellbeing.store.all()[0].factors == ["exercise"]


def test_wellbeing_windows_exclude_future_checkins(wellbeing):
    wellbeing.checkin(energy=2, at=NOW)
    wellbeing.checkin(energy=5, at=NOW + timedelta(days=1))
    assert wellbeing.today(now=NOW)["checkins"] == 1
    assert wellbeing.trend(now=NOW)["avg_energy"] == 2


@pytest.fixture
def reports(tmp_path, monkeypatch):
    import orion_core.reports as module
    monkeypatch.setattr(module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(ReportDrafter, "_write_docx", lambda *args: False)
    router = SimpleNamespace(has_text_fallback=lambda: False)
    return ReportDrafter(None, router), tmp_path / "reports"


async def test_parallel_report_exports_never_overwrite(reports):
    drafter, root = reports
    results = await asyncio.gather(*[
        drafter.draft("Repeated topic", brief=f"Unique notes {i}") for i in range(6)])
    assert all(r.ok for r in results)
    documents = list(root.glob("*/report.md"))
    assert len(documents) == 6
    for i in range(6):
        assert sum(f"Unique notes {i}" in p.read_text(encoding="utf-8")
                   for p in documents) == 1


async def test_report_write_failure_is_an_unsuccessful_tool_result(reports, monkeypatch):
    drafter, _ = reports
    def fail(*args):
        raise OSError("simulated disk full")
    monkeypatch.setattr(drafter, "_write_files", fail)
    result = await drafter.draft("A report")
    assert not result.ok
    assert "disk full" in result.text


async def test_a_reports_folder_outside_orions_own_is_named_in_full(reports, monkeypatch, tmp_path):
    """relative_to(BASE_DIR) raised AFTER the files were written when the
    reports folder lay elsewhere, so a saved report was reported as a crash."""
    import orion_core.reports as module
    elsewhere = tmp_path / "elsewhere" / "reports"
    elsewhere.mkdir(parents=True)
    monkeypatch.setattr(module, "BASE_DIR", tmp_path / "orion")
    monkeypatch.setattr(module, "REPORTS_DIR", elsewhere)
    drafter, _ = reports
    result = await drafter.draft("Quarterly figures")
    assert result.ok and str(elsewhere) in result.text


async def test_empty_chart_does_not_create_broken_figure(reports):
    drafter, root = reports
    assert (await drafter.draft("A report", charts=[{"data": {}}])).ok
    assert not list(root.glob("*/chart_*.svg"))
    assert "## Figures" not in next(root.glob("*/report.md")).read_text(encoding="utf-8")


@pytest.mark.parametrize("data", [
    {"loss": -100, "gain": 10, "zero": 0},
    {"first": -100, "second": -10},
    {"huge": 1e308, "negative": -1e308},
    {"zero": 0},
])
def test_report_chart_geometry_stays_inside_plot(reports, data):
    drafter, _ = reports
    svg = ElementTree.fromstring(drafter._bar_chart({"data": data}))
    rects = svg.findall("{http://www.w3.org/2000/svg}rect")[1:]
    assert len(rects) == len(data)
    for rect in rects:
        x, width = float(rect.attrib["x"]), float(rect.attrib["width"])
        assert math.isfinite(x) and math.isfinite(width)
        assert 130 <= x <= 460 and 0 <= width <= 330
        assert x + width <= 460.1
    zero_bars = [rect for rect, value in zip(rects, data.values()) if value == 0]
    assert all(float(rect.attrib["width"]) == 0 for rect in zero_bars)


def test_report_chart_skips_non_finite_and_boolean_values(reports):
    drafter, _ = reports
    chart = {"data": {"nan": math.nan, "inf": math.inf, "bool": True, "too_big": 10**400}}
    assert drafter._bar_chart(chart) == ""


def test_report_chart_keeps_fractional_values_and_escapes_labels(reports):
    drafter, _ = reports
    svg = ElementTree.fromstring(drafter._bar_chart(
        {"title": "<unsafe>", "data": {"<script>": 0.25}}))
    texts = [node.text for node in svg.findall("{http://www.w3.org/2000/svg}text")]
    assert "0.25" in texts and "<script>" in texts and "<unsafe>" in texts
    assert not svg.findall("{http://www.w3.org/2000/svg}script")
