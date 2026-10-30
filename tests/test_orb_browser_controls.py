"""Exercise the orb's actual pointer and hand JavaScript without a WebGL device."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_mouse_input_survives_camera_samples_and_camera_shutdown():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the browser control block")
    source = (Path(__file__).resolve().parents[1] / "orion_core/gui/face3d.py").read_text(
        encoding="utf-8")
    controls = "const orbCtl=" + source.split("const orbCtl=", 1)[1].split(
        "window.orionEmotion=", 1)[0]
    targets = "orbCtl.tx=orbClamp(orbCtl.mouseX" + source.split(
        "orbCtl.tx=orbClamp(orbCtl.mouseX", 1)[1].split("for(const [now,target]", 1)[0]
    runtime = f"""
const vm = require('node:vm');
const assert = require('node:assert/strict');
const handlers = {{}};
let now = 1000;
const canvas = {{style: {{}}, addEventListener: (name, fn) => handlers[name] = fn,
  setPointerCapture: () => {{}}, getBoundingClientRect: () => ({{width: 400, height: 400}})}};
const context = {{renderer: {{domElement: canvas}},
  document: {{getElementById: () => ({{style: {{}}}})}},
  performance: {{now: () => now}}, morphTarget: 0, morphT: 0,
  clamp01: value => Math.max(0, Math.min(1, Number(value) || 0)), window: {{}}}};
vm.createContext(context);
vm.runInContext({json.dumps(controls)}, context);
const orb = vm.runInContext('orbCtl', context);
context.window.orionOrbCameraEnabled(true);
context.window.orionOrbHands({{hands: 2, x: 0.4, y: 0, spread: 0.8, twist: 0.2}});
assert.equal(orb.handSpread, 0.8);
handlers.wheel({{deltaY: -500, preventDefault() {{}}}});
assert(orb.mouseSpread > 0.5);
now = 1200;
context.window.orionOrbHands({{hands: 0}});
assert.equal(orb.handSpread, 0.8);
now = 1400;
context.window.orionOrbHands({{hands: 0}});
assert.equal(orb.handSpread, 0);
assert(orb.mouseSpread > 0.5);
context.window.orionOrbHands({{hands: 1, x: 0.4, y: 0}});
handlers.pointerdown({{pointerId: 1, clientX: 100, clientY: 100,
  button: 0, shiftKey: false, preventDefault() {{}}}});
handlers.pointermove({{pointerId: 1, clientX: 180, clientY: 100,
  preventDefault() {{}}}});
assert(orb.mouseX > 0 && orb.handX > 0);
vm.runInContext({json.dumps(targets)}, context);
assert.equal(orb.tx, orb.mouseX + orb.handX);
assert.equal(orb.ts, Math.max(orb.mouseSpread, orb.handSpread));
context.window.orionOrbCameraEnabled(false);
assert.equal(orb.handX, 0);
assert(orb.mouseX > 0 && orb.mouseSpread > 0);
vm.runInContext({json.dumps(targets)}, context);
assert.equal(orb.tx, orb.mouseX);
assert.equal(orb.ts, orb.mouseSpread);
context.morphTarget = 1;
const before = orb.mouseX;
handlers.pointermove({{pointerId: 1, clientX: 280, clientY: 100,
  preventDefault() {{}}}});
assert.equal(orb.mouseX, before);
"""
    result = subprocess.run([node, "-e", runtime], capture_output=True, text=True,
                            timeout=10, check=False)
    assert result.returncode == 0, result.stderr
