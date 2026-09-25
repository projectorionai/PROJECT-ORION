"""
Object detection — naming what is in view, locally, with a YOLO-family model.

The perception loop knows THAT something arrived; this says WHAT it is ("a
person", "a cup", "a laptop") on this machine, for free, in tens of
milliseconds. That is what makes camera-triggered automation practical: a
workflow can fire when a specific object appears without a paid cloud call
per frame, and the loop can skip its cloud description entirely when every
new arrival has already been named locally.

It runs on ONNX Runtime, not PyTorch. ONNX Runtime is already in the
standalone app (the OCR and voiceprint models use it); torch is not, and adds
half a gigabyte. So the detector works in the .exe as well as from source.

Models, from ``CONFIG_DIR/models``:

* the default — YOLOX-S from OpenCV's official model zoo (Apache-2.0, 80
  everyday COCO classes, 34.2 MB). It is only downloaded when asked for
  (``install_model``), never at start-up, and the download is checked against
  a pinned SHA-256 before it is used.
* any Ultralytics export you drop in yourself (``yolov8*.onnx``,
  ``yolo11*.onnx``, ``yolov5*.onnx``) — the output layout is recognised from
  the model's shape, so no configuration is needed. A file you add wins over
  the default.

Detection is synchronous and CPU-bound: callers run it in a worker thread.
"""

from __future__ import annotations

import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .constants import CONFIG_DIR
from .model_store import ModelDownloadError, download_verified, usable
from .vision_lab import zone_of

MODEL_DIR = CONFIG_DIR / "models"

COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

#: What people call things, mapped onto the class names the model uses.
LABEL_ALIASES = {
    "phone": "cell phone", "mobile": "cell phone", "mobile phone": "cell phone",
    "smartphone": "cell phone", "iphone": "cell phone", "mug": "cup",
    "monitor": "tv", "screen": "tv", "television": "tv", "tv monitor": "tv",
    "sofa": "couch", "plant": "potted plant", "table": "dining table",
    "desk": "dining table", "bag": "backpack", "rucksack": "backpack",
    "people": "person", "someone": "person", "anyone": "person", "human": "person",
    "man": "person", "woman": "person", "kid": "person", "child": "person",
    "ball": "sports ball", "football": "sports ball", "remote control": "remote",
    "glass": "wine glass", "controller": "remote", "puppy": "dog", "kitten": "cat",
    "hairdryer": "hair drier", "hair dryer": "hair drier", "fridge": "refrigerator",
}


def canonical_label(name: str) -> str | None:
    """The model's class for a spoken name, or None if it cannot detect it."""
    text = " ".join(str(name or "").strip().lower().split())
    if text.endswith("s") and text[:-1] in COCO_LABELS:
        text = text[:-1]                        # "cups" -> "cup"
    text = LABEL_ALIASES.get(text, text)
    return text if text in COCO_LABELS else None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    size: int
    sha256: str
    title: str


DEFAULT_MODEL = ModelSpec(
    key="yolox",
    filename="object_detection_yolox_2022nov.onnx",
    url=("https://huggingface.co/opencv/object_detection_yolox/resolve/main/"
         "object_detection_yolox_2022nov.onnx"),
    size=35_858_002,
    sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    title="YOLOX-S (OpenCV model zoo, Apache-2.0)",
)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]      # normalised x1, y1, x2, y2

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def zone(self) -> str:
        return zone_of(*self.centre)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2


def describe(detections: Sequence[Detection]) -> str:
    """'2 persons (left, centre), a laptop (bottom)' — or that nothing was found."""
    if not detections:
        return "I can't pick out any objects I know."
    groups: dict[str, list[Detection]] = {}
    for detection in detections:
        groups.setdefault(detection.label, []).append(detection)
    parts = []
    for label, items in sorted(groups.items(), key=lambda kv: -max(d.confidence for d in kv[1])):
        where = ", ".join(sorted({d.zone for d in items}))
        name = label if len(items) == 1 else f"{len(items)} {label}s"
        article = "" if len(items) > 1 else ("an " if label[0] in "aeiou" else "a ")
        parts.append(f"{article}{name} ({where})")
    return "I can see " + ", ".join(parts) + "."


# ── pre- and post-processing, pure numpy (no model needed to test) ──────────

def letterbox(frame: Any, size: int, *, centre: bool) -> tuple[Any, float, float, float]:
    """Resize keeping aspect ratio and pad with grey 114 to size×size.

    YOLOX pads bottom/right; Ultralytics centres. Returns the canvas, the
    scale and the (x, y) padding so boxes can be mapped back."""
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2 if centre else 0
    pad_y = (size - new_h) // 2 if centre else 0
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, float(pad_x), float(pad_y)


def nms(boxes: Any, scores: Any, classes: Any, iou: float = 0.45) -> list[int]:
    """Class-aware greedy non-maximum suppression. Boxes are x1,y1,x2,y2."""
    import numpy as np
    keep: list[int] = []
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        idx = idx[np.argsort(-scores[idx])]
        while idx.size:
            best = idx[0]
            keep.append(int(best))
            if idx.size == 1:
                break
            rest = idx[1:]
            xx1 = np.maximum(boxes[best, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[best, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[best, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[best, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_best = (boxes[best, 2] - boxes[best, 0]) * (boxes[best, 3] - boxes[best, 1])
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            overlap = inter / np.maximum(area_best + area_rest - inter, 1e-9)
            idx = rest[overlap < iou]
    return sorted(keep, key=lambda i: -scores[i])


def output_layout(shape: Sequence[int], input_size: int = 640) -> str:
    """Which YOLO family produced an output tensor, from its shape alone.

        (1, 4+C, N)   Ultralytics v8 / 11: channels first, no objectness
        (1, N, 5+C)   YOLOX (raw grid offsets, N = anchor-free grid cells)
                      or YOLOv5 (decoded boxes, N = 3 anchors per cell)
    """
    if len(shape) != 3:
        raise ValueError(f"unexpected detector output shape {tuple(shape)}")
    _, a, b = shape
    grid = sum((input_size // s) ** 2 for s in (8, 16, 32))
    if a < b and a >= 5:
        return "v8"
    if a == grid:
        return "yolox"
    if a == 3 * grid:
        return "v5"
    raise ValueError(f"unrecognised detector output shape {tuple(shape)}")


def _yolox_grids(input_size: int) -> tuple[Any, Any]:
    import numpy as np
    grids, strides = [], []
    for stride in (8, 16, 32):
        cells = input_size // stride
        xv, yv = np.meshgrid(np.arange(cells), np.arange(cells))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((cells * cells, 1), stride))
    return np.concatenate(grids), np.concatenate(strides)


def decode(output: Any, layout: str, input_size: int, min_confidence: float,
           num_classes: int = len(COCO_LABELS)) -> tuple[Any, Any, Any]:
    """Raw model output -> (boxes x1y1x2y2 in input pixels, scores, class ids),
    already filtered by confidence and suppressed."""
    import numpy as np
    out = np.asarray(output, dtype=np.float32)[0]
    if layout == "v8":
        out = out.T                                   # (N, 4 + C)
        centres = out[:, :4]
        class_scores = out[:, 4:4 + num_classes]
    elif layout == "yolox":
        grids, strides = _yolox_grids(input_size)
        centres = out[:, :4].copy()
        centres[:, :2] = (centres[:, :2] + grids) * strides
        centres[:, 2:4] = np.exp(centres[:, 2:4]) * strides
        class_scores = out[:, 4:5] * out[:, 5:5 + num_classes]
    elif layout == "v5":
        centres = out[:, :4]
        class_scores = out[:, 4:5] * out[:, 5:5 + num_classes]
    else:
        raise ValueError(layout)
    classes = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(classes)), classes]
    keep = scores >= min_confidence
    centres, scores, classes = centres[keep], scores[keep], classes[keep]
    boxes = np.empty_like(centres)
    boxes[:, 0] = centres[:, 0] - centres[:, 2] / 2
    boxes[:, 1] = centres[:, 1] - centres[:, 3] / 2
    boxes[:, 2] = centres[:, 0] + centres[:, 2] / 2
    boxes[:, 3] = centres[:, 1] + centres[:, 3] / 2
    order = nms(boxes, scores, classes) if len(scores) else []
    return boxes[order], scores[order], classes[order]


# ── the detector ────────────────────────────────────────────────────────────

class ObjectDetector:
    """Lazily loads one ONNX model and runs it on BGR frames."""

    INPUT_SIZE = 640
    THREADS = 2          # leave the rest of the CPU to everything else ORION does

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
        self._session: Any = None
        self._layout = ""
        self._input_name = ""
        self._input_size = self.INPUT_SIZE
        self._model_path: Path | None = None
        self._lock = threading.Lock()
        self.error = ""

    # ── availability ─────────────────────────────────────────────────────────

    @staticmethod
    def runtime_available() -> bool:
        import importlib.util
        return importlib.util.find_spec("onnxruntime") is not None

    def model_path(self) -> Path | None:
        """A model you added beats the default; the default if installed."""
        if not self.model_dir.is_dir():
            return None
        own = sorted(p for p in self.model_dir.glob("*.onnx")
                     if p.name.lower().startswith(("yolov8", "yolo11", "yolov5", "yolo_")))
        if own:
            return own[0]
        default = self.model_dir / DEFAULT_MODEL.filename
        return default if usable(default, size=DEFAULT_MODEL.size) else None

    @property
    def available(self) -> bool:
        return self.runtime_available() and self.model_path() is not None

    def describe_state(self) -> str:
        if not self.runtime_available():
            return "object detection needs onnxruntime, which is not installed"
        path = self.model_path()
        if path is None:
            return ("object detection is not installed — say 'install the object "
                    f"detector' to download {DEFAULT_MODEL.title} "
                    f"({DEFAULT_MODEL.size / 1024**2:.1f} MB, once)")
        return f"object detection ready ({path.name})"

    # ── installing the default model ─────────────────────────────────────────

    def install_model(self, progress: Callable[[str], None] | None = None,
                      opener: Callable[..., Any] = urllib.request.urlopen) -> str:
        """Download and verify the default model. Returns a sentence. Blocking."""
        target = self.model_dir / DEFAULT_MODEL.filename
        if usable(target, size=DEFAULT_MODEL.size):
            return f"The object detector is already installed ({target.name})."
        if progress:
            progress(f"downloading {DEFAULT_MODEL.title}, {DEFAULT_MODEL.size / 1024**2:.1f} MB")
        try:
            received = download_verified(DEFAULT_MODEL.url, target, size=DEFAULT_MODEL.size,
                                         sha256=DEFAULT_MODEL.sha256, opener=opener)
        except ModelDownloadError as exc:
            return f"I couldn't install the object detector: {exc}."
        self._session = None                 # pick the new model up on next use
        return (f"Object detector installed — {DEFAULT_MODEL.title}, "
                f"{received / 1024**2:.1f} MB, checksum verified.")

    # ── inference ────────────────────────────────────────────────────────────

    def _ensure_session(self) -> bool:
        path = self.model_path()
        if path is None:
            self.error = "no model installed"
            return False
        if self._session is not None and self._model_path == path:
            return True
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.THREADS
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            model_input = session.get_inputs()[0]
            size = model_input.shape[-1]
            self._input_size = size if isinstance(size, int) and size > 0 else self.INPUT_SIZE
            self._input_name = model_input.name
            self._layout = output_layout(session.get_outputs()[0].shape
                                         if all(isinstance(d, int) for d in session.get_outputs()[0].shape)
                                         else self._probe_shape(session), self._input_size)
            self._session = session
            self._model_path = path
            self.error = ""
            return True
        except Exception as exc:
            self.error = f"could not load {path.name}: {exc}"
            self._session = None
            return False

    def _probe_shape(self, session: Any) -> tuple[int, ...]:
        import numpy as np
        blank = np.zeros((1, 3, self._input_size, self._input_size), np.float32)
        return tuple(session.run(None, {session.get_inputs()[0].name: blank})[0].shape)

    def detect(self, frame: Any, min_confidence: float = 0.4,
               max_results: int = 25) -> list[Detection]:
        """Objects in a BGR frame. Empty (never raises) if anything is missing."""
        if frame is None or getattr(frame, "ndim", 0) != 3:
            return []
        with self._lock:
            if not self._ensure_session():
                return []
            import numpy as np
            layout, size = self._layout, self._input_size
            # YOLOX was trained on raw BGR 0-255 padded bottom-right;
            # Ultralytics on RGB 0-1 letterboxed in the centre.
            yolox = layout == "yolox"
            canvas, scale, pad_x, pad_y = letterbox(frame, size, centre=not yolox)
            if yolox:
                blob = canvas.astype(np.float32)
            else:
                blob = canvas[..., ::-1].astype(np.float32) / 255.0
            blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
            try:
                output = self._session.run(None, {self._input_name: blob})[0]
            except Exception as exc:
                self.error = f"inference failed: {exc}"
                return []
        boxes, scores, classes = decode(output, layout, size, min_confidence)
        height, width = frame.shape[:2]
        found = []
        for box, score, cls in zip(boxes[:max_results], scores, classes):
            x1 = (box[0] - pad_x) / scale / width
            y1 = (box[1] - pad_y) / scale / height
            x2 = (box[2] - pad_x) / scale / width
            y2 = (box[3] - pad_y) / scale / height
            label = COCO_LABELS[int(cls)] if int(cls) < len(COCO_LABELS) else f"class {int(cls)}"
            found.append(Detection(label, float(score), (
                float(min(max(x1, 0.0), 1.0)), float(min(max(y1, 0.0), 1.0)),
                float(min(max(x2, 0.0), 1.0)), float(min(max(y2, 0.0), 1.0)))))
        return found


__all__ = [
    "COCO_LABELS", "DEFAULT_MODEL", "Detection", "ObjectDetector", "canonical_label",
    "decode", "describe", "letterbox", "nms", "output_layout",
]
