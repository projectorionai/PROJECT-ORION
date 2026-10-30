"""Exercise the real Qt panel in a subprocess, preserving GUI import hygiene."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cognition_reflections_work_without_token_animation_or_duplicate_ops():
    project = Path(__file__).resolve().parents[1]
    script = r'''
from PyQt6.QtWidgets import QApplication
from orion_core.bus import OrionBus
from orion_core.gui.cognition_deck import CognitionDeckView
from orion_core.gui.ops_deck import OperationsDeckView, InnerVoicePanel

app = QApplication([])
bus = OrionBus()
bus.dashboard_event.emit('reflections', {'entries': [
    {'id': n, 'text': f'Previous note {n}', 'at': '09:00:00'} for n in range(60)
]})
view = CognitionDeckView(bus)
panel = view.reflections_panel
assert panel.notes.count() == 50
assert panel.notes.item(0).text().endswith('Previous note 59')
assert not panel.notes.isHidden()
assert panel.empty_label.isHidden()
assert bus.receivers(bus.thought_delta) == 0

for n in range(1000):
    bus.thought_delta.emit({'id': 90, 'phase': 'delta', 'text': 'uncommitted fragment'})
assert panel.notes.count() == 50
bus.thought.emit({'id': 90, 'text': 'An action was requested.', 'at': '09:01:00',
    'observed_at': '2026-09-30T08:01:00+00:00', 'origin': 'observation',
    'context': {'focus': 'Writing'}, 'evidence': ['Desktop · open Calculator']})
assert panel.notes.count() == 50
assert 'Activity note' in panel.notes.item(0).text()
assert 'Focus: Writing' in panel.details.text()
assert 'Recorded evidence:' in panel.details.text()
assert 'Desktop · open Calculator' in panel.details.text()
bus.thought.emit({'id': 90, 'text': 'An action was requested.'})
assert panel.notes.count() == 50

commands = []
bus.gui_command.connect(commands.append)
panel.tasks_button.click()
assert commands == [{'action': 'page', 'target': 'TASKS'}]
panel.clear_button.click()
assert panel.notes.count() == 0
assert panel.notes.isHidden()
assert panel.details.isHidden()
assert not panel.empty_label.isHidden()
assert not panel.clear_button.isEnabled()
assert len(bus.last_value('dashboard:reflections')['entries']) == 60

bus.thought.emit({'id': 91, 'text': 'Fresh note after clearing.'})
assert panel.notes.count() == 1
assert not panel.notes.isHidden()
assert not hasattr(panel, '_type_timer')
ops = OperationsDeckView(bus)
assert not hasattr(ops, 'inner_voice_panel')
assert hasattr(ops, 'environment_panel')
compatibility_panel = InnerVoicePanel(bus)
assert compatibility_panel.history.entries
view._timer.stop()
ops._timer.stop()
view.close()
ops.close()
print('reflection-widget-ok')
'''
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", ORION_REMOTE_ACCESS="0")
    result = subprocess.run([sys.executable, "-c", script], cwd=project, env=env,
                            capture_output=True, text=True, timeout=45, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reflection-widget-ok" in result.stdout
