"""
SwarmDeckView — a navigable 3D map of ORION himself (Mark XXI, Track B).

The user asked for "a 3D web like infrastructure... an AI agent swarm that
I can navigate through" — a live graph of ORION's core, his six
specialists, connected MCP servers, defined workflows and memory tiers,
all in one scene rather than scattered across separate panels. This is not
decoration: every node is backed by data an earlier Mark XXI track already
built and gave no visual surface —

    agents         AgentManager.describe()                (Phase 3)
    subsystems     SystemRegistries.modules.all_described  (Track G)
    MCP servers    MCPHost.catalogue()/health_snapshot()   (Track E1/E2)
    workflows      WorkflowEngine.definitions               (Mark XX)
    memory tiers   MemoryAgent.tiers_snapshot()             (existing)

Rendered via Three.js/WebGL in a QWebEngineView — the exact same technique
already proven in face3d.py (the avatar) and globe.py (the intelligence
globe): an inline HTML page, Three.js loaded from the same pinned
cdn.jsdelivr.net import map face3d.py uses, so this introduces no new CDN
dependency. Node positions use a deterministic ring layout (grouped by
kind, cached by node id so nodes don't visually jump between refreshes)
rather than an N-body force simulation — this backend's own relationships
are a small, flat "core connects to everything" graph plus a handful of
module-dependency edges, not a large organic network a physics simulation
would earn its keep on; deterministic is also something reviewable and
correct by inspection, which a hand-written physics sim run only inside a
browser is not.

JS -> Python click routing reuses globe.py's own proven technique — a
QWebEnginePage subclass intercepting a prefixed console.log message —
rather than introducing QWebChannel as a second bridging mechanism.

Degrades to a plain list view when QWebEngineView is unavailable
(WEBENGINE_OK False), matching face3d.py's own contract exactly.

QtWebEngineWidgets/QtWebEngineCore are NOT imported at module level, unlike
face3d.py/globe.py's own top-level try/except (which nothing in the test
suite ever imports, so it never mattered there). Qt requires
QtWebEngineWidgets to be imported — and AA_ShareOpenGLContexts set —
*before* the process's QApplication is constructed (app.py does exactly
this at real startup: see its "import-order requirement" comment). Test
files build their own ad-hoc QApplication with neither precondition met,
so the first-ever import of this module during pytest collection
destabilised OpenGL for every *other* real-widget test that ran after it
(discovered as a native STATUS_STACK_BUFFER_OVERRUN crash deep into the
full suite, in an unrelated avatar-rendering test). The import is instead
deferred to _ensure_built() — reached only via a real showEvent, which no
test in this file triggers — and WEBENGINE_OK is computed via
importlib.util.find_spec (module lookup only, no import), the same
lazy-availability convention already used elsewhere in this codebase
for cv2/mediapipe.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import APP_MARK
from ..swarm.model import NodeKind


def _webengine_available() -> bool:
    return (
        importlib.util.find_spec("PyQt6.QtWebEngineWidgets") is not None
        and importlib.util.find_spec("PyQt6.QtWebEngineCore") is not None
    )


WEBENGINE_OK = _webengine_available()


def gl_renderer_enabled() -> bool:
    """Whether to use the native GPU renderer (Mark XXII).

    Now the DEFAULT, having been verified against a live OpenGL 4.1 core
    context on real hardware: nodes, edges, animated packets, orbit camera and
    picking all render. It reached parity with the WebEngine view in Phase 3
    and is strictly better on the things that matter — no Chromium process, no
    CDN dependency (the old view needed internet to load Three.js and rendered
    blank without it), and no full scene rebuild per tick.

    ORION_SWARM_GL=0 forces the old WebEngine renderer back, and the GL path
    degrades to it automatically if the widget cannot be created at all (old
    driver, software rasteriser, remote desktop) — see _build_gl_view."""
    if os.getenv("ORION_SWARM_GL", "").strip().lower() in {"0", "false", "off", "no"}:
        return False
    return importlib.util.find_spec("PyQt6.QtOpenGLWidgets") is not None

# Structural buffers are cached and telemetry is patched incrementally, so the
# swarm can update its state every second without recreating the neural map.
_REFRESH_MS = 1000
_EVENT_PREFIX = "ORION_SWARM_EVENT:"

_KIND_LABELS = {
    "core": "ORION",
    "agent": "Specialist Agents",
    "module": "Subsystems",
    "mcp": "MCP Servers",
    "workflow": "Workflows",
    "memory": "Memory Tiers",
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA SOURCE — pure, independently testable
# ──────────────────────────────────────────────────────────────────────────────

def compose_snapshot(
    core_state: str = "STANDBY",
    agents: Any | None = None,
    registries: Any | None = None,
    telemetry: Any | None = None,
    mcp_host: Any | None = None,
    workflow_engine: Any | None = None,
    memory: Any | None = None,
) -> dict[str, Any]:
    """A {"nodes": [...], "edges": [...]} snapshot of ORION's live
    subsystems. Every backend is optional and independently defensive —
    one subsystem being absent or raising never blanks the others."""
    nodes: list[dict[str, Any]] = [
        {"id": "core", "kind": "core", "label": "ORION", "state": str(core_state or "STANDBY")}
    ]
    edges: list[dict[str, str]] = []

    if agents is not None:
        try:
            for row in agents.describe():
                node_id = f"agent:{row['name']}"
                nodes.append({
                    "id": node_id, "kind": "agent",
                    "label": str(row.get("title") or row["name"]),
                    "calls": int(row.get("calls") or 0),
                    "detail": str(row.get("focus") or ""),
                })
                edges.append({"from": "core", "to": node_id})
        except Exception:
            pass

    if registries is not None:
        try:
            described = registries.modules.all_described(telemetry)
            module_ids = {f"module:{name}" for name in described}
            for name, record in described.items():
                nodes.append({
                    "id": f"module:{name}", "kind": "module", "label": name,
                    "health": str(record.get("health") or "UNKNOWN"),
                    "detail": str(record.get("role") or ""),
                })
            for name, record in described.items():
                for dep in record.get("dependencies") or []:
                    dep_id = f"module:{dep}"
                    if dep_id in module_ids:
                        edges.append({"from": f"module:{name}", "to": dep_id})
        except Exception:
            pass

    if mcp_host is not None:
        try:
            catalogue = mcp_host.catalogue()
            health = mcp_host.health_snapshot()
            for server, info in catalogue.items():
                node_id = f"mcp:{server}"
                nodes.append({
                    "id": node_id, "kind": "mcp", "label": server,
                    "health": str(health.get(server) or "UNKNOWN"),
                    "detail": f"{len(info.get('tools') or [])} tool(s)",
                })
                edges.append({"from": "core", "to": node_id})
        except Exception:
            pass

    if workflow_engine is not None:
        try:
            for name, definition in workflow_engine.definitions.items():
                steps = definition.get("steps") or []
                node_id = f"workflow:{name}"
                nodes.append({
                    "id": node_id, "kind": "workflow", "label": name,
                    "detail": f"{len(steps)} step(s)",
                })
                edges.append({"from": "core", "to": node_id})
        except Exception:
            pass

    if memory is not None:
        try:
            for tier, count in memory.tiers_snapshot().items():
                node_id = f"memory:{tier}"
                nodes.append({
                    "id": node_id, "kind": "memory", "label": tier,
                    "detail": f"{count} row(s)",
                })
                edges.append({"from": "core", "to": node_id})
        except Exception:
            pass

    return {"nodes": nodes, "edges": edges}


# ──────────────────────────────────────────────────────────────────────────────
# INLINE THREE.JS SCENE
# ──────────────────────────────────────────────────────────────────────────────

SWARM_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
  html,body{margin:0;height:100%;background:#0b0f14;overflow:hidden;font-family:'Segoe UI',Arial}
  #tip{position:absolute;left:12px;bottom:10px;color:#7fb2d9;font-size:11px;
    letter-spacing:2px;text-transform:uppercase;opacity:.65}
</style></head><body>
<div id="tip">drag to orbit &middot; scroll to zoom &middot; click a node</div>
<script type="importmap">{"imports":{
  "three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const HEALTH_COLOR = {OK:0x5cc98f, DEGRADED:0xe8b34a, DOWN:0xe0655c, UNKNOWN:0x63758a};
const KIND_COLOR = {core:0x7fb2d9, agent:0xa9cdec, module:0x6ea8d8, mcp:0xc99bf0,
  workflow:0x8fd0c9, memory:0xd8b06e};
const RING_RADIUS = {core:0, agent:9, module:16, mcp:23, workflow:30, memory:37};
const RING_HEIGHT = {core:0, agent:1.5, module:-1.5, mcp:3, workflow:-3, memory:0};

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 500);
camera.position.set(0, 22, 46);
const renderer = new THREE.WebGLRenderer({antialias:true, alpha:false});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(2, devicePixelRatio||1));
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.minDistance = 6; controls.maxDistance = 140;

scene.add(new THREE.AmbientLight(0x334455, 1.4));
const key = new THREE.PointLight(0x9fd0ff, 2.2, 200); key.position.set(20, 30, 30);
scene.add(key);

const nodeGroup = new THREE.Group(); scene.add(nodeGroup);
const edgeGroup = new THREE.Group(); scene.add(edgeGroup);

let nodeData = {};           // id -> full node record (for click payloads)
let nodeMeshes = {};         // id -> THREE.Mesh
let layoutCache = {};        // id -> {angle}
let ringSlotCount = {};      // kind -> how many ids already placed this session

function ringSlot(id, kind){
  if(layoutCache[id] !== undefined) return layoutCache[id];
  const n = (ringSlotCount[kind] = (ringSlotCount[kind]||0) + 1);
  const angle = n * 2.399963;   // golden-angle spacing — even, never repeats exactly
  layoutCache[id] = angle;
  return angle;
}

function positionFor(node){
  const r = RING_RADIUS[node.kind] ?? 20;
  if(r === 0) return new THREE.Vector3(0,0,0);
  const a = ringSlot(node.id, node.kind);
  const h = (RING_HEIGHT[node.kind] ?? 0) + Math.sin(a*1.7) * 1.2;
  return new THREE.Vector3(Math.cos(a)*r, h, Math.sin(a)*r);
}

function clearGroup(g){ while(g.children.length){ const c = g.children.pop();
  c.geometry && c.geometry.dispose(); c.material && c.material.dispose(); } }

window.orionSwarmUpdate = (jsonStr) => {
  let snap; try{ snap = JSON.parse(jsonStr); } catch(e){ return; }
  clearGroup(nodeGroup); clearGroup(edgeGroup);
  nodeData = {}; nodeMeshes = {};
  const positions = {};
  const seen = new Set();
  for(const node of (snap.nodes||[])){
    seen.add(node.id);
    nodeData[node.id] = node;
    const pos = positionFor(node);
    positions[node.id] = pos;
    const isCore = node.kind === 'core';
    const size = isCore ? 2.2 : 1.05;
    const color = node.health ? (HEALTH_COLOR[node.health] ?? KIND_COLOR[node.kind]) : (KIND_COLOR[node.kind] ?? 0x7fb2d9);
    const geo = new THREE.SphereGeometry(size, 24, 18);
    const mat = new THREE.MeshStandardMaterial({
      color, emissive: color, emissiveIntensity: isCore ? 0.9 : 0.55,
      metalness: 0.25, roughness: 0.35});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    mesh.userData.id = node.id;
    nodeGroup.add(mesh);
    nodeMeshes[node.id] = mesh;
  }
  // Prune cached angles for nodes that no longer exist, so a slot is
  // eventually reused rather than the rings growing forever.
  for(const id of Object.keys(layoutCache)) if(!seen.has(id)) delete layoutCache[id];
  const lineMat = new THREE.LineBasicMaterial({color:0x3d5872, transparent:true, opacity:0.55});
  for(const edge of (snap.edges||[])){
    const a = positions[edge.from], b = positions[edge.to];
    if(!a || !b) continue;
    const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
    edgeGroup.add(new THREE.Line(geo, lineMat.clone()));
  }
};

window.orionSwarmPulse = (fromId, toId) => {
  const flash = (id) => { const m = nodeMeshes[id]; if(!m) return;
    const orig = m.material.emissiveIntensity;
    m.material.emissiveIntensity = 1.6;
    setTimeout(()=>{ if(m.material) m.material.emissiveIntensity = orig; }, 650);
  };
  flash(fromId); flash(toId);
};

// ── click routing: raycast, then hand the node's data back to Python ──────
const ray = new THREE.Raycaster();
const ptr = new THREE.Vector2();
renderer.domElement.addEventListener('pointerdown', (ev) => {
  const rect = renderer.domElement.getBoundingClientRect();
  ptr.x = ((ev.clientX-rect.left)/rect.width)*2-1;
  ptr.y = -((ev.clientY-rect.top)/rect.height)*2+1;
  ray.setFromCamera(ptr, camera);
  const hits = ray.intersectObjects(nodeGroup.children);
  if(hits.length){
    const id = hits[0].object.userData.id;
    const node = nodeData[id];
    if(node) console.log('__EVENT_PREFIX__' + JSON.stringify(node));
  }
}, {passive:true});

// ── render-loop gating ──────────────────────────────────────────────────────
// This loop used to run for ever, at full frame rate, for the whole session.
// The SWARM and the GLOBE are BOTH QWebEngineView pages, each with its own
// Chromium render process and its own WebGL context. With the swarm still
// rendering Three.js at 60 fps behind the deck, opening the globe put two
// live WebGL contexts on the GPU at once — and Cesium is by far the heavier
// of the two. The globe's render process then loses its context or is killed
// outright, which is exactly "load the SWARM, then the GLOBE, and the globe
// no longer works". The globe already paused itself when hidden; the swarm
// never did, so the globe was the one that died.
//
// Same contract as the globe now: Python drives orionSetActive() from
// showEvent/hideEvent, and a hidden page costs the GPU nothing.
let __orionActive = true;
let __orionFrame = null;

function animate(){
  __orionFrame = requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

window.orionSetActive = function(on){
  const next = !!on;
  if(next === __orionActive) return;
  __orionActive = next;
  if(next){
    animate();
  } else if(__orionFrame !== null){
    cancelAnimationFrame(__orionFrame);
    __orionFrame = null;
  }
};

// The browser also stops firing rAF when the page is genuinely hidden; this
// keeps our own flag honest if that happens without Python telling us.
document.addEventListener('visibilitychange', () => {
  if(document.hidden) window.orionSetActive(false);
});

animate();
addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script></body></html>"""

SWARM_HTML = SWARM_HTML.replace("__EVENT_PREFIX__", _EVENT_PREFIX)


# ──────────────────────────────────────────────────────────────────────────────
# QT WIDGET
# ──────────────────────────────────────────────────────────────────────────────

def _route_console_message(on_event: Any, message: str) -> None:
    """The actual click-routing logic behind _SwarmPage.javaScriptConsoleMessage
    — a plain function so it's testable without ever importing Qt WebEngine
    (see the module docstring). Mirrors globe.py's _GlobePage technique for
    JS -> Python signalling, rather than a second bridging mechanism
    (QWebChannel)."""
    text = str(message or "")
    if not text.startswith(_EVENT_PREFIX) or on_event is None:
        return
    try:
        payload = json.loads(text[len(_EVENT_PREFIX):])
    except (ValueError, TypeError):
        return
    try:
        on_event(payload)
    except Exception:
        pass


_SwarmPageCls: Any = None


def _swarm_page_class() -> Any:
    """Builds (once) and caches the real QWebEnginePage subclass. Only ever
    called from _ensure_built() — importing QtWebEngineCore here, not at
    module level, is the whole point (see module docstring)."""
    global _SwarmPageCls
    if _SwarmPageCls is None:
        from PyQt6.QtWebEngineCore import QWebEnginePage

        class _SwarmPage(QWebEnginePage):
            def __init__(self, parent: Any = None, on_event: Any = None) -> None:
                super().__init__(parent)
                self._on_event = on_event

            def javaScriptConsoleMessage(self, level: Any, message: str,  # type: ignore[override]
                                         line: int, source: str) -> None:
                _route_console_message(self._on_event, message)

        _SwarmPageCls = _SwarmPage
    return _SwarmPageCls


def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panelFrame")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("panelHeading")
    layout.addWidget(heading)
    return frame, layout


class SwarmDeckView(QWidget):
    """The SWARM page: a navigable 3D map of ORION's live subsystems, plus
    a side inspector that shows what's actually selected and offers a real
    next action (jump to the agent's workspace, run the workflow, ...)."""

    def __init__(
        self,
        bus: OrionBus,
        agents: Any | None = None,
        registries: Any | None = None,
        telemetry: Any | None = None,
        mcp_host: Any | None = None,
        workflow_engine: Any | None = None,
        memory: Any | None = None,
        on_open_agent: Any | None = None,
        on_open_zone: Any | None = None,
        on_open_page: Any | None = None,
        deck_pages: dict[str, str] | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.bus = bus
        self.agents = agents
        self.registries = registries
        self.telemetry = telemetry
        self.mcp_host = mcp_host
        self.workflow_engine = workflow_engine
        self.memory = memory
        self._on_open_agent = on_open_agent
        # Mark XXII: opening a cluster anchor navigates the Command Deck to
        # that zone, so every zone is reachable from inside the swarm.
        self._on_open_zone = on_open_zone
        # ...and every deck PAGE is itself a node, so the swarm contains all
        # of ORION rather than only his agents. deck_pages maps page name to
        # zone name; attach_deck_pages can supply it after construction, since
        # the deck is built after the swarm page it contains.
        self._on_open_page = on_open_page
        self.deck_pages: dict[str, str] = dict(deck_pages or {})
        self._core_state = "STANDBY"
        self._last_snapshot: dict[str, Any] = {"nodes": [], "edges": []}
        self._built = False
        # Is this page on screen? Drives the render loop, so a swarm behind the
        # deck stops competing with the globe for the GPU (see _set_active).
        self._active = False
        self._page_loaded = False
        self.view: Any = None
        self.gl_view: Any = None
        # Mark XXII, Phase 1. Opt-in while the native renderer reaches parity
        # with the WebEngine one (edges are Phase 3), so the default path is
        # unchanged and this can be exercised on real hardware without
        # risking the shipped page. ORION_SWARM_GL=1 switches it on.
        self._gl_mode = gl_renderer_enabled()
        # Mark XXII Phase 2/3: cached composition + measured edge traffic.
        # Constructed unconditionally (both are pure, allocate nothing until
        # used, and import no Qt) so the WebEngine path can adopt them later
        # without a second code path.
        from ..swarm.bridges import BridgeLedger
        from ..swarm.composer import SnapshotComposer
        from ..swarm.traffic import TrafficLedger

        self._composer = SnapshotComposer()
        self.traffic = TrafficLedger()
        # Commands in flight. Held here rather than on the GL view so a
        # dispatch that happens before the renderer is built (startup does
        # exactly that) is still recorded, and so the ledger survives the
        # renderer being torn down and rebuilt.
        self.bridges = BridgeLedger()
        self._build_ui()
        try:
            self.bus.state.connect(self._on_bus_state)
        except Exception:
            pass
        # ORION's own voice drives his face. bus.amplitude is emitted by the
        # speech pipeline for every audio chunk, so the mouth follows what he
        # is actually saying rather than a canned talking animation.
        try:
            self.bus.amplitude.connect(self._on_amplitude)
            # The spectral profile shapes the mouth into syllables, not just a
            # pulsing hole — vowels open it, sibilants tighten it.
            self.bus.voice_spectrum.connect(self._on_voice_spectrum)
        except Exception:
            pass
        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        title = QLabel("SWARM — ORION AND EVERY SUBSYSTEM HE'S BUILT FROM")
        title.setText(f"SWARM — {APP_MARK.upper()} NEURAL SUBSYSTEM MAP")
        title.setObjectName("titleLabel")
        outer.addWidget(title)
        navigation = QLabel(
            "NAVIGATE  Drag to orbit  ·  Scroll to zoom  ·  Right-drag to pan  ·  "
            "Click to inspect  ·  Double-click to open  ·  F focus  ·  Home reset"
        )
        navigation.setObjectName("mutedLabel")
        navigation.setWordWrap(True)
        outer.addWidget(navigation)
        # Held so portrait mode can strip the chrome: the compact orb is
        # ORION's face and nothing else.
        self._title_label = title
        self._outer_layout = outer

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter = splitter

        if self._gl_mode:
            # Mark XXII, Phase 1: the native GPU renderer. Container only —
            # the widget itself is built lazily in _ensure_built, because
            # creating a GL context at construction time would put it on the
            # startup path and (as with WebEngine) into pytest collection.
            self._placeholder = QLabel("◉  The swarm assembles when you open this page.")
            self._placeholder.setObjectName("mutedLabel")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._webengine_container = QWidget()
            container_layout = QVBoxLayout(self._webengine_container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.addWidget(self._placeholder)
            splitter.addWidget(self._webengine_container)
        elif WEBENGINE_OK:
            self._placeholder = QLabel("◉  The swarm assembles when you open this page.")
            self._placeholder.setObjectName("mutedLabel")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._webengine_container = QWidget()
            container_layout = QVBoxLayout(self._webengine_container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.addWidget(self._placeholder)
            splitter.addWidget(self._webengine_container)
        else:
            self._placeholder = None
            self.list_view = QListWidget()
            self.list_view.currentItemChanged.connect(self._on_list_selection)
            splitter.addWidget(self.list_view)

        self._inspector_panel = self._build_inspector_panel()
        splitter.addWidget(self._inspector_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

    def _build_inspector_panel(self) -> QFrame:
        frame, layout = _panel("INSPECTOR")
        self.inspector_label = QLabel("Select a node to inspect it.")
        self.inspector_label.setObjectName("mutedLabel")
        self.inspector_label.setWordWrap(True)
        layout.addWidget(self.inspector_label)
        self.inspector_detail = QLabel("")
        self.inspector_detail.setWordWrap(True)
        layout.addWidget(self.inspector_detail, 1)
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        self.inspector_focus = QPushButton("Focus")
        self.inspector_focus.setToolTip("Centre the neural map on the selected subsystem")
        self.inspector_focus.setEnabled(False)
        self.inspector_focus.clicked.connect(self._on_focus_selected)
        actions.addWidget(self.inspector_focus)
        self.inspector_action = QPushButton("Open")
        self.inspector_action.setEnabled(False)
        self.inspector_action.clicked.connect(self._on_inspector_action)
        actions.addWidget(self.inspector_action)
        layout.addLayout(actions)
        return frame

    # ── lazy WebEngine build (parity with QuantumFace3D) ────────────────────

    def _clear_placeholder(self) -> None:
        if self._placeholder is None:
            return
        self._placeholder.hide()
        self._webengine_container.layout().removeWidget(self._placeholder)
        self._placeholder.deleteLater()
        self._placeholder = None

    def _ensure_built(self) -> None:
        if self._built or not (self._gl_mode or WEBENGINE_OK):
            return
        self._built = True
        self._clear_placeholder()
        if self._gl_mode:
            self._build_gl_view()
        else:
            from PyQt6.QtWebEngineWidgets import QWebEngineView

            self.view = QWebEngineView()
            self.view.setPage(
                _swarm_page_class()(self.view, on_event=self._on_swarm_event))
            # Only drive the page once it has actually loaded. Calling
            # runJavaScript on a QWebEngineView that has just been constructed
            # crashes the process natively — no Python traceback, the
            # interpreter simply stops. Found when the render-loop gating below
            # started calling it from showEvent/hideEvent.
            self._page_loaded = False
            try:
                self.view.loadFinished.connect(self._on_load_finished)
            except Exception:
                pass
            self.view.setHtml(SWARM_HTML)
            self._webengine_container.layout().addWidget(self.view)
        self._push_snapshot()

    def _build_gl_view(self) -> None:
        """Construct the native renderer, degrading to the WebEngine/list
        surface if the GL widget cannot be created (old driver, software
        rasteriser, remote desktop). A missing GPU must never cost ORION the
        page entirely."""
        try:
            from ..swarm.render.gl_view import SwarmGLView

            self.gl_view = SwarmGLView(self)
            self.gl_view.attach_status_sink(self._log)
            self.gl_view.node_selected.connect(self._on_gl_node_selected)
            self.gl_view.node_activated.connect(self._on_gl_node_activated)
            # Commands already in flight keep drawing across the handover.
            self.gl_view.attach_bridges(self.bridges)
            self._webengine_container.layout().addWidget(self.gl_view)
        except Exception as exc:
            self.gl_view = None
            self._gl_mode = False
            self._log(f"SWARM: native renderer unavailable ({exc}); using fallback.")
            notice = QLabel("Native swarm renderer unavailable on this display.")
            notice.setObjectName("mutedLabel")
            notice.setWordWrap(True)
            notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._webengine_container.layout().addWidget(notice)

    def attach_centre_face(self, face: Any, fraction: float = 0.34) -> None:
        """Place ORION's face at the CENTRE of the network.

        The graph is meant to read as a brain with ORION at the middle of it
        and everything radiating outward, rather than his face being a small
        panel off to one side. The layout keeps a keep-out sphere clear at the
        centre (CORE_KEEPOUT_RADIUS) and the core node is drawn at radius 0,
        so this overlay lands in genuinely empty space and every edge that
        terminates at core appears to terminate at him.

        The face is a child of the swarm page, positioned in resizeEvent —
        not added to a layout — because it has to float ON TOP of the GL
        surface rather than take space beside it.
        """
        self.centre_face = face
        self._centre_face_fraction = max(0.15, min(0.6, float(fraction)))
        face.setParent(self)
        face.raise_()
        self._layout_centre_face()
        face.show()

    def _layout_centre_face(self) -> None:
        face = getattr(self, "centre_face", None)
        if face is None:
            return
        # Square, centred on the GL surface's own area rather than the whole
        # page, so the inspector panel does not push him off-centre.
        host = getattr(self, "_webengine_container", None)
        area = host.geometry() if host is not None else self.rect()
        size = int(min(area.width(), area.height()) * self._centre_face_fraction)
        size = max(200, size)
        face.setGeometry(
            area.x() + (area.width() - size) // 2,
            area.y() + (area.height() - size) // 2,
            size, size)
        face.raise_()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout_centre_face()

    def _log(self, message: str) -> None:
        try:
            self.bus.log.emit(str(message))
        except Exception:
            pass

    def _on_gl_node_selected(self, node_id: str) -> None:
        """Bridge a GL pick into the same inspector the other surfaces use, so
        all three paths converge on one code path."""
        node = self.gl_view.scene.node(node_id) if self.gl_view else None
        if node is not None:
            self._on_swarm_event(node.as_dict())

    def _on_gl_node_activated(self, node_id: str) -> None:
        """Double-click: select, then perform the node's action.

        For a cluster anchor that means jumping straight to its Command Deck
        zone — navigating the swarm and navigating ORION become the same
        gesture."""
        self._on_gl_node_selected(node_id)
        self._on_inspector_action()

    def _on_load_finished(self, ok: bool) -> None:
        self._page_loaded = bool(ok)
        # Apply whatever visibility state we settled on while it was loading.
        if self._page_loaded:
            self._set_active(self._active)

    def _set_active(self, on: bool) -> None:
        """Run the swarm's render loop only while the page is on screen.

        The SWARM and the GLOBE are both WebEngine pages with their own WebGL
        contexts. Leaving this one rendering behind the deck is what starved
        the globe's Cesium context and killed its render process — see the
        render-loop gating note in SWARM_HTML.
        """
        self._active = on
        view = getattr(self, "view", None)
        if view is not None and getattr(self, "_page_loaded", False):
            try:
                view.page().runJavaScript(
                    f"window.orionSetActive&&orionSetActive({'true' if on else 'false'})")
            except Exception:
                pass
        gl_view = getattr(self, "gl_view", None)
        if gl_view is not None:
            # The native GL path has no JS bridge; stopping its timer is the
            # equivalent, and an offscreen GL widget redrawing is the same waste.
            for attr in ("set_active", "setActive", "pause"):
                fn = getattr(gl_view, attr, None)
                if callable(fn):
                    try:
                        fn(on)
                    except TypeError:
                        pass
                    break

    def showEvent(self, event: Any) -> None:
        self._ensure_built()
        self._set_active(True)
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:
        # Stop rendering the moment the page leaves the screen, so opening the
        # globe does not have to share the GPU with a swarm nobody is looking at.
        self._set_active(False)
        super().hideEvent(event)

    # ── live data ─────────────────────────────────────────────────────────────

    def _on_bus_state(self, state: Any) -> None:
        self._core_state = str(state or "STANDBY")
        self._on_bus_state_cognition(state)

    def _compose(self) -> dict[str, Any]:
        return compose_snapshot(
            core_state=self._core_state, agents=self.agents, registries=self.registries,
            telemetry=self.telemetry, mcp_host=self.mcp_host,
            workflow_engine=self.workflow_engine, memory=self.memory,
        )

    def _refresh(self) -> None:
        if not self.isVisible():
            return
        self._push_snapshot()
        # Keep the inspector showing live numbers for whatever is selected,
        # rather than a frozen copy from the moment it was clicked.
        self.refresh_inspector()

    # ── persistent view state (Mark XXII, Phase 4) ───────────────────────────
    #
    # "Changing tabs should never destroy objects. Views should persist.
    #  Camera positions should persist. Node selections should persist."
    #
    # The deck already satisfies the first two: UnifiedDashboard keeps every
    # page alive in a QStackedWidget and only hides them, so this widget, its
    # scene, its GPU buffers and its camera all survive a tab switch intact —
    # verified by test rather than assumed. What follows lets that state be
    # captured and restored explicitly, which is what makes it survive a
    # renderer rebuild (GL context loss) or a restart too.

    def view_state(self) -> dict[str, Any]:
        """A serialisable snapshot of how the user has set this page up."""
        state: dict[str, Any] = {"selected": getattr(self, "_selected_id", "")}
        if self.gl_view is not None:
            camera = self.gl_view.camera
            state["camera"] = {
                "azimuth": float(camera.azimuth),
                "elevation": float(camera.elevation),
                "distance": float(camera.distance),
                "target": [float(v) for v in camera.target],
            }
        return state

    def restore_view_state(self, state: dict[str, Any]) -> None:
        """Reapply a previously captured view state. Tolerant of partial or
        malformed input — a bad saved state must never break the page."""
        if not isinstance(state, dict):
            return
        camera_state = state.get("camera")
        if isinstance(camera_state, dict) and self.gl_view is not None:
            camera = self.gl_view.camera
            try:
                camera.focus_on(tuple(camera_state.get("target") or (0.0, 0.0, 0.0)),
                                distance=float(camera_state.get("distance") or 78.0))
                camera.orbit(float(camera_state.get("azimuth") or 0.0) - camera.azimuth,
                             float(camera_state.get("elevation") or 0.0) - camera.elevation)
                camera.snap_to_goal()
            except Exception:
                pass
        selected = str(state.get("selected") or "")
        if selected and self.gl_view is not None:
            self.gl_view.select(selected)
            self._selected_id = selected
            self.refresh_inspector()

    def compose_typed(self) -> Any:
        """The Mark XXII typed snapshot — clusters, real telemetry, MCP tools
        as their own nodes, and edges carrying measured traffic.

        Uses the caching composer (Phase 2), so a tick where only telemetry
        moved reuses every node object rather than reallocating ~3,600
        dataclasses. The traffic ledger is applied last, over the composed
        edges, and likewise returns unchanged edges by identity so the diff
        engine can skip them."""
        from dataclasses import replace

        snapshot = self._composer.compose(
            core_state=self._core_state, agents=self.agents,
            registries=self.registries, telemetry=self.telemetry,
            mcp_host=self.mcp_host, workflow_engine=self.workflow_engine,
            memory=self.memory, deck_pages=self.deck_pages,
        )
        return replace(snapshot, edges=self.traffic.apply(snapshot.edges))

    def attach_deck_pages(self, pages: dict[str, str]) -> None:
        """Tell the swarm which Command Deck pages exist and which zone each
        belongs to, so every one becomes a node.

        Late-attached because the deck is constructed AFTER the swarm page it
        contains — the swarm cannot be handed a deck that does not exist yet."""
        self.deck_pages = dict(pages or {})
        self._composer.invalidate()
        if self.isVisible():
            self._push_snapshot()

    def _route_for(self, tool: str, args: dict[str, Any] | None = None) -> list[str]:
        """The path through ORION this command really takes.

        A memory tool gets the constellation route instead of the generic one:
        a recall reaches outward through the tiers ORION actually holds, so
        the illumination shows which memory answered rather than lighting the
        whole cluster indiscriminately."""
        from ..swarm.cognition_state import memory_recall_path, path_for_tool

        if tool in ("memory", "remember", "recall", "search_memory") and self.memory:
            try:
                from ..swarm.memory_field import describe_tiers

                tiers = describe_tiers(self.memory.tiers_snapshot())
                if tiers:
                    return memory_recall_path(
                        tiers, str((args or {}).get("query") or ""))
            except Exception:
                pass
        return path_for_tool(tool, args)

    def set_portrait_mode(self, enabled: bool) -> None:
        """Fill this view with ORION's face — the compact overlay orb.

        The orb is not a second avatar and not a second widget. It is this
        renderer, framed on the one ORION it already draws. Building the orb a
        face of its own is precisely what left it blank: the widget it had was
        the 2-D placeholder that app.py hides whenever the native renderer
        owns the face.
        """
        enabled = bool(enabled)
        self._portrait = enabled
        self._ensure_built()
        # Chrome off: a title bar and an inspector in a 380px orb would leave
        # almost nothing for ORION himself.
        for widget in (self._title_label, self._inspector_panel):
            if widget is not None:
                widget.setVisible(not enabled)
        margin = 0 if enabled else 10
        self._outer_layout.setContentsMargins(margin, margin, margin, margin)
        if self.gl_view is not None:
            self.gl_view.set_portrait_mode(enabled)

    @property
    def portrait_mode(self) -> bool:
        return bool(getattr(self, "_portrait", False))

    def _engage_cognition(self, tool: str, args: dict[str, Any] | None = None) -> None:
        """Light the whole route a command travelled, not just its endpoint.

        core -> reasoning -> the responsible cluster -> the specific node, so
        the user can follow the decision. Driven by a real dispatched tool
        call, which is what separates this from an animation."""
        if self.gl_view is None:
            return
        try:
            self.gl_view.scene.cognition.engage_path(self._route_for(tool, args))
        except Exception:
            pass

    # ── commands in flight ───────────────────────────────────────────────────

    def launch_bridge(self, tool: str, args: dict[str, Any] | None = None) -> Any:
        """Open a packet for a tool call that is ABOUT to run.

        Called before the handler rather than after it, which is the whole
        point: the packet is visibly in flight, and then visibly waiting, for
        exactly as long as the real work takes. Resolving it afterwards is
        what makes the return leg mean "this finished" instead of "some time
        passed"."""
        try:
            return self.bridges.launch(self._route_for(tool, args), label=tool)
        except Exception:
            return None

    def resolve_bridge(self, flight: Any, ok: bool = True) -> None:
        """The work finished — send the result home, or flare where it broke."""
        if flight is None:
            return
        try:
            self.bridges.resolve(flight, ok=bool(ok))
        except Exception:
            pass

    def _on_amplitude(self, amplitude: Any) -> None:
        """Speech amplitude -> the native face's mouth."""
        if self.gl_view is None:
            return
        try:
            self.gl_view.set_speech_amplitude(float(amplitude))
        except Exception:
            pass

    def _on_voice_spectrum(self, bands: Any) -> None:
        """Voice spectral profile -> the native face's mouth articulation."""
        if self.gl_view is None:
            return
        try:
            self.gl_view.set_speech_spectrum(bands)
        except Exception:
            pass

    def _on_bus_state_cognition(self, state: Any) -> None:
        """Mirror ORION's own broadcast state onto the field's posture, so
        LISTENING / THINKING / SPEAKING / STANDBY are visible in how the whole
        network moves rather than only in a text label."""
        if self.gl_view is None:
            return
        try:
            cognition = self.gl_view.scene.cognition
            cognition.observe_bus_state(str(state or ""))
            # The same posture drives ORION's expression, so his face and his
            # network always agree about what he is doing — thinking narrows
            # his eyes at the moment the reasoning cluster brightens.
            self.gl_view.set_face_mode(cognition.mode)
        except Exception:
            pass

    def record_activity(self, tool: str, args: dict[str, Any] | None = None) -> None:
        """Note one real dispatch event so the edge it travelled lights up.

        Called from the dispatcher (via SwarmDeckView.pulse) rather than
        polled, so intensity reflects actual tool traffic instead of a guess
        derived from counters."""
        from ..swarm.traffic import edge_for_tool_call

        try:
            source, target, kind = edge_for_tool_call(tool, args)
            self.traffic.record(source, target, kind)
        except Exception:
            pass

    def _push_snapshot(self) -> None:
        if self._gl_mode and self.gl_view is not None:
            # No JSON, no bridge, no full-scene rebuild: the scene diffs this
            # against the previous frame and writes only what changed.
            self.gl_view.apply_snapshot(self.compose_typed())
            return
        self._last_snapshot = self._compose()
        if WEBENGINE_OK and self.view is not None:
            payload = json.dumps(self._last_snapshot)
            self._js(f"window.orionSwarmUpdate && orionSwarmUpdate({json.dumps(payload)})")
        elif not WEBENGINE_OK and not self._gl_mode:
            self._render_list_fallback()

    def _js(self, code: str) -> None:
        if self.view is not None:
            try:
                self.view.page().runJavaScript(code)
            except Exception:
                pass

    def pulse(self, from_id: str, to_id: str, tool: str = "",
              args: dict[str, Any] | None = None,
              flight: Any = None, ok: bool = True) -> None:
        """Called by the dispatcher when a tool call finishes.

        Three effects, deliberately independent: the WebEngine path flashes the
        two nodes (Mark XXI, Track B5); the traffic ledger records the event so
        the EDGE it travelled lights and animates a packet (Mark XXII, Phase
        3); and the in-flight bridge opened at dispatch is resolved, which is
        what sends its packet home or flares it where it failed. Every extra
        argument is optional so the older two-argument call site keeps working
        unchanged.
        """
        self.resolve_bridge(flight, ok)
        if tool:
            self.record_activity(tool, args)
            self._engage_cognition(tool, args)
        self._js(f"window.orionSwarmPulse && orionSwarmPulse({json.dumps(from_id)}, {json.dumps(to_id)})")

    # ── fallback list (no WebEngine) ─────────────────────────────────────────

    def _render_list_fallback(self) -> None:
        self.list_view.clear()
        for node in self._last_snapshot.get("nodes", []):
            kind = _KIND_LABELS.get(node["kind"], node["kind"])
            extra = node.get("health") or node.get("detail") or ""
            item = QListWidgetItem(f"[{kind}] {node['label']}" + (f" — {extra}" if extra else ""))
            item.setData(Qt.ItemDataRole.UserRole, node)
            self.list_view.addItem(item)

    def _on_list_selection(self, current: QListWidgetItem, _previous: QListWidgetItem) -> None:
        if current is None:
            return
        self._on_swarm_event(current.data(Qt.ItemDataRole.UserRole))

    # ── inspector (Track B4) ─────────────────────────────────────────────────

    def _on_swarm_event(self, node: dict[str, Any]) -> None:
        """Render the full inspector for a selected node.

        Formatting lives in swarm/inspect.py (pure, tested) so the GL surface,
        the WebEngine bridge and the list fallback all render identically
        instead of each growing its own rules."""
        if not isinstance(node, dict):
            return
        from ..swarm.inspect import report_from_dict

        self._selected_node = node
        self._selected_id = str(node.get("id") or "")
        report = report_from_dict(
            node, can_open_agent=self._on_open_agent is not None)
        self._render_report(report)

    def _render_report(self, report: Any) -> None:
        self._report = report
        self.inspector_label.setText(f"{report.title}  ·  {report.subtitle}")
        lines = []
        for row in report.rows:
            marker = "! " if row.severity == "alarm" else "  "
            lines.append(f"{marker}{row.label:<16}{row.value}")
        if report.neighbours:
            shown = ", ".join(report.neighbours[:6])
            if len(report.neighbours) > 6:
                shown += f"  (+{len(report.neighbours) - 6} more)"
            lines.append("")
            lines.append(f"  Connected       {shown}")
        self.inspector_detail.setText("\n".join(lines) or "No further detail available.")
        self.inspector_action.setText(report.action_label)
        self.inspector_action.setEnabled(report.action_target is not None)
        self.inspector_focus.setEnabled(self.gl_view is not None)

    def refresh_inspector(self) -> None:
        """Re-render the selected node from the CURRENT snapshot.

        Without this the inspector would keep showing whatever was true at the
        moment of the click — exactly the stale-telemetry problem a live
        monitoring surface must not have."""
        node_id = getattr(self, "_selected_id", "")
        if not node_id or self.gl_view is None:
            return
        from ..swarm.inspect import build_report

        node = self.gl_view.scene.node(node_id)
        if node is None:
            return
        self._render_report(build_report(
            node, self.gl_view.scene.snapshot,
            can_open_agent=self._on_open_agent is not None))

    def _on_focus_selected(self) -> None:
        """Centre the native neural map on the selected subsystem."""
        if self.gl_view is not None and getattr(self, "_selected_id", ""):
            self.gl_view.focus_selected()

    def _on_inspector_action(self) -> None:
        report = getattr(self, "_report", None)
        if report is None or report.action_target is None:
            return
        node = getattr(self, "_selected_node", None) or {}
        node_id = str(node.get("id") or "")
        kind = str(node.get("kind") or "")

        # A cluster anchor opens its Command Deck zone — this is what makes
        # the swarm the navigation surface rather than a picture of one.
        if node_id.startswith("cluster:") and self._on_open_zone is not None:
            try:
                self._on_open_zone(report.action_target)
            except Exception:
                pass
            return

        if kind == NodeKind.PAGE.value and self._on_open_page is not None:
            try:
                self._on_open_page(report.action_target)
            except Exception:
                pass
            return

        if kind == NodeKind.AGENT.value and self._on_open_agent is not None:
            try:
                self._on_open_agent(report.action_target)
            except Exception:
                pass
            return

        # Workflow/MCP actions are named honestly in the button but have no
        # handler yet; they must not silently pretend to have run.
        self._log(f"SWARM: '{report.action_label}' is not wired up yet.")
