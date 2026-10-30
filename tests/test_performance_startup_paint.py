"""Core-shell readiness observes real offscreen Qt paint dispatch, not show."""
import ast
from pathlib import Path
import threading

from PyQt6.QtCore import QEvent, QThread
from PyQt6.QtWidgets import QApplication, QWidget


def test_first_paint_waits_for_real_widget_paint_and_records_once():
    from orion_core.app import _FirstShellPaint
    from orion_core.startup_budget import StartupBudget
    application = QApplication.instance() or QApplication([])
    policy = application.quitOnLastWindowClosed()
    application.setQuitOnLastWindowClosed(False)
    owner, events = threading.get_ident(), []

    class Shell(QWidget):
        def paintEvent(self, event):
            events.append((threading.get_ident(), QThread.currentThread()))
            return super().paintEvent(event)

    shell = Shell()
    budget = StartupBudget()
    reported = []
    budget.on_milestone(lambda name, at: reported.append((name, threading.get_ident())))
    shell._observer = _FirstShellPaint(shell, budget)
    try:
        shell.setUpdatesEnabled(False)
        shell.show()
        application.processEvents()
        assert shell.isVisible() and not events and "first_paint" not in budget.milestones
        # Other startup/window events are not evidence of painted content.
        QApplication.sendEvent(shell, QEvent(QEvent.Type.UpdateRequest))
        assert "first_paint" not in budget.milestones
        shell.setUpdatesEnabled(True)
        shell.update()
        application.processEvents()
        assert events and events[0] == (owner, shell.thread())
        assert reported == [("first_paint", owner)]
        shell.update()
        application.processEvents()
        assert len(events) >= 2 and reported == [("first_paint", owner)]
    finally:
        shell.hide()
        shell.deleteLater()
        application.processEvents()
        application.setQuitOnLastWindowClosed(policy)


def test_startup_installs_observer_before_show_without_manual_paint_claim():
    source = (Path(__file__).parents[1] / "orion_core" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    boot = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "run_application")
    calls = [n for n in ast.walk(boot) if isinstance(n, ast.Call)]
    observe = next(n for n in calls if isinstance(n.func, ast.Name)
                   and n.func.id == "_FirstShellPaint")
    show = next(n for n in calls if isinstance(n.func, ast.Attribute)
                and n.func.attr == "show" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "window")
    assert observe.lineno < show.lineno
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr == "milestone"
                   and n.args and isinstance(n.args[0], ast.Constant)
                   and n.args[0].value == "first_paint" for n in calls)
    assert 'budget.mark("core window visible")' in source


def test_optional_corpus_is_queued_after_command_readiness_and_stays_requestable():
    source = (Path(__file__).parents[1] / "orion_core" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    boot = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "run_application")
    calls = [n for n in ast.walk(boot) if isinstance(n, ast.Call)]
    ready = next(n for n in calls if isinstance(n.func, ast.Attribute)
                 and n.func.attr == "milestone" and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == "command_ready")
    corpus = next(n for n in calls if isinstance(n.func, ast.Attribute)
                  and n.func.attr == "to_thread" and n.args
                  and isinstance(n.args[0], ast.Attribute) and n.args[0].attr == "build"
                  and isinstance(n.args[0].value, ast.Name) and n.args[0].value.id == "corpus")
    assert ready.lineno < corpus.lineno
    assert "dispatcher.corpus = corpus" in source
    assert 'name="orion-knowledge-corpus"' in source


def test_only_plain_startup_work_is_awaited_on_workers():
    source = (Path(__file__).parents[1] / "orion_core" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    boot = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "run_application")
    workers = [n for n in ast.walk(boot) if isinstance(n, ast.Await)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == "construct_startup_resource"]
    assert {n.value.args[0].id for n in workers} == {
        "_make_startup_memory", "IdentityManager", "TokenUsageLedger", "KnowledgeGraphEngine",
        "AutonomousControlLayer", "SkillManager"}
    # COM-importing peripheral setup and all QWidget construction stay on Qt's
    # owner thread. Worker helpers must not quietly expand to these factories.
    assert not any(n.value.args[0].id in {"PeripheralController", "OrionCoreWindow",
                                        "CommandCentreWindow", "DisplayTopologyManager"}
                   for n in workers)
