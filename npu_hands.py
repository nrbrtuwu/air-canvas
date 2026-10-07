"""MediaPipe's hand landmarker pipeline on ONNX Runtime, for NPUs: Qualcomm
(Snapdragon, via onnxruntime-qnn) and Intel Core Ultra (via onnxruntime-openvino).

Runs the same two models as MediaPipe (palm detector + hand landmark model, converted
to ONNX) and reproduces the steps MediaPipe's HandLandmarker graph does around them:

  palm detection (only while fewer than num_hands are tracked):
    letterbox to 192x192 -> SSD anchors -> sigmoid scores -> weighted NMS
    -> rotated rect from wrist / middle-finger keypoints, shifted and scaled 2.6x
  landmarks, per rect:
    rotated 224x224 crop -> landmark model -> project the 21 points back
    -> hand presence check -> rect for the next frame from the palm landmarks
       (rotation-aligned bounds, scaled 2.0x), so detection is skipped while tracking

The steps and constants follow MediaPipe's calculators of the same names
(SsdAnchors, TensorsToDetections, NonMaxSuppression WEIGHTED, DetectionsToRects,
RectTransformation, HandLandmarksToRect, HandAssociation). On the NPU each model
call takes under 1 ms on a Snapdragon X, against ~15-30 ms on one CPU core.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import cv2
import numpy as np

PALM_SIZE, LM_SIZE = 192, 224
# Palm-detector keypoints used for the rect rotation (wrist, middle-finger base).
PALM_WRIST, PALM_MIDDLE = 0, 2
# Landmarks MediaPipe uses for the tracking rect: wrist, thumb base and the
# first two joints of each finger (fingertips are left out so the rect does
# not jump when a finger bends).
TRACK_POINTS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]


BACKENDS = ("qnn-npu", "openvino-npu", "openvino-gpu", "openvino-cpu", "cpu")


def _qnn_npu_devices():
    """Qualcomm NPU devices (Snapdragon), via the onnxruntime-qnn plugin; [] if none."""
    try:
        import onnxruntime as ort
        import onnxruntime_qnn as qnn
    except ImportError:
        return []
    try:
        ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
    except Exception:  # already registered in this process
        pass
    return [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"
            and d.device.type == ort.OrtHardwareDeviceType.NPU]


def openvino_devices() -> list[str]:
    """OpenVINO devices (e.g. CPU, GPU, NPU on Intel Core Ultra); [] without
    onnxruntime-openvino + openvino."""
    try:
        import onnxruntime as ort
        if "OpenVINOExecutionProvider" not in ort.get_available_providers():
            return []
        if os.name == "nt":  # Windows: the OpenVINO DLLs come from the openvino package
            import onnxruntime.tools.add_openvino_win_libs as ov_libs
            ov_libs.add_openvino_libs_to_path()
        import openvino
        return list(openvino.Core().available_devices)
    except Exception:
        return []


def create_session(path: str, device: str = "auto"):
    """ONNX Runtime session for one model. device: "auto"/"npu" (Qualcomm NPU, then
    Intel NPU), one of BACKENDS, or "cpu". Returns (session, description)."""
    import onnxruntime as ort

    def options():
        so = ort.SessionOptions()
        # The models have a symbolic batch dimension; NPUs need static shapes.
        for name in ("N", "batch"):
            so.add_free_dimension_override_by_name(name, 1)
        return so

    tried = []
    order = {"auto": ("qnn-npu", "openvino-npu"), "npu": ("qnn-npu", "openvino-npu")}.get(device, (device,))
    for backend in order:
        try:
            if backend == "qnn-npu":
                import onnxruntime_qnn as qnn
                npu = _qnn_npu_devices()
                if not npu:
                    raise RuntimeError("no Qualcomm NPU found")
                so = options()
                # Fail instead of quietly running unsupported layers on the CPU.
                so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                so.add_provider_for_devices(npu, {
                    "backend_path": qnn.get_qnn_htp_path(),
                    "enable_htp_fp16_precision": "1",   # measured: <0.1 px from fp32
                    "htp_performance_mode": "burst",
                })
                return ort.InferenceSession(path, so), "Qualcomm NPU"
            if backend.startswith("openvino-"):
                kind = backend.split("-", 1)[1].upper()
                if kind not in openvino_devices():
                    raise RuntimeError(f"OpenVINO has no {kind} device")
                session = ort.InferenceSession(path, options(), providers=[
                    ("OpenVINOExecutionProvider", {"device_type": kind})])
                # If the OpenVINO plugin fails to load (e.g. openvino version does not
                # match onnxruntime-openvino), ONNX Runtime quietly uses the CPU instead.
                if session.get_providers()[0] != "OpenVINOExecutionProvider":
                    raise RuntimeError("OpenVINO plugin failed to load (openvino version mismatch?)")
                return session, f"Intel {kind} (OpenVINO)"
            if backend == "cpu":
                return ort.InferenceSession(path, options(), providers=["CPUExecutionProvider"]), "CPU"
            raise ValueError(f"unknown backend {backend!r}")
        except Exception as error:
            tried.append(f"{backend}: {error}")
    if device in ("auto",):
        return ort.InferenceSession(path, options(), providers=["CPUExecutionProvider"]), "CPU"
    raise RuntimeError("no usable backend (" + "; ".join(tried) + ")")


def ssd_anchors() -> np.ndarray:
    """Anchor centres (2016, 2) of the 192x192 palm detector: strides 8, 16, 16, 16,
    two anchors per layer and cell (aspect ratio 1 + interpolated scale), fixed size."""
    strides = [8, 16, 16, 16]
    centres, layer = [], 0
    while layer < len(strides):
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            last += 1
        per_cell = 2 * (last - layer)
        cells = math.ceil(PALM_SIZE / strides[layer])
        for y in range(cells):
            for x in range(cells):
                centres.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * per_cell)
        layer = last
    return np.asarray(centres, np.float32)


def weighted_nms(boxes: np.ndarray, scores: np.ndarray, keypoints: np.ndarray,
                 threshold: float, limit: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """MediaPipe's WEIGHTED non-max suppression: each kept detection is the
    score-weighted mean of all detections overlapping it (IoU > threshold), which
    is steadier than keeping only the single best box. boxes are (x0, y0, x1, y1)."""
    order = list(np.argsort(-scores))
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    out = []
    while order and len(out) < limit:
        top = order[0]
        rest = np.array(order)
        x0 = np.maximum(boxes[top, 0], boxes[rest, 0]); y0 = np.maximum(boxes[top, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[top, 2], boxes[rest, 2]); y1 = np.minimum(boxes[top, 3], boxes[rest, 3])
        inter = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
        iou = inter / (area[top] + area[rest] - inter + 1e-9)
        group = rest[iou > threshold]
        w = scores[group][:, None]
        out.append(((boxes[group] * w).sum(0) / w.sum(),
                    (keypoints[group] * w[:, :, None]).sum(0) / w.sum(),
                    float(scores[top])))
        order = [i for i in rest[iou <= threshold]]
    return out


def normalize_radians(angle: float) -> float:
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


class Rect:
    """Rotated rect in pixels: centre, width, height, rotation (rad, clockwise in image)."""
    __slots__ = ("cx", "cy", "w", "h", "rot")

    def __init__(self, cx, cy, w, h, rot):
        self.cx, self.cy, self.w, self.h, self.rot = cx, cy, w, h, rot

    def transformed(self, scale: float, shift_y: float) -> "Rect":
        """RectTransformation with square_long: shift along the rect's own y axis
        (towards the fingers), make it square on the long side, then scale."""
        c, s = math.cos(self.rot), math.sin(self.rot)
        cx = self.cx - self.h * shift_y * s
        cy = self.cy + self.h * shift_y * c
        side = max(self.w, self.h) * scale
        return Rect(cx, cy, side, side, self.rot)

    def crop_matrix(self, size: int) -> np.ndarray:
        """2x3 affine from crop pixels (0..size) to image pixels."""
        c, s = math.cos(self.rot), math.sin(self.rot)
        kx, ky = self.w / size, self.h / size
        return np.array([[c * kx, -s * ky, self.cx - (c * self.w - s * self.h) / 2],
                         [s * kx, c * ky, self.cy - (s * self.w + c * self.h) / 2]], np.float32)

    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned extent ignoring rotation (what MediaPipe's rect IoU uses)."""
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def rect_from_landmarks(pts: np.ndarray) -> Rect:
    """HandLandmarksToRect + RectTransformation(scale 2.0, shift_y -0.1): the rect for
    the next frame, from the palm landmarks, measured along the hand's own rotation."""
    p = pts[TRACK_POINTS]
    wrist = p[0]
    # Direction: wrist -> middle of (index MCP + ring MCP)/2 and middle MCP.
    target = ((p[4] + p[8]) / 2 + p[6]) / 2
    rot = normalize_radians(math.pi / 2 - math.atan2(-(target[1] - wrist[1]), target[0] - wrist[0]))
    centre = (p.min(0) + p.max(0)) / 2
    c, s = math.cos(-rot), math.sin(-rot)
    d = p - centre
    proj = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], 1)
    lo, hi = proj.min(0), proj.max(0)
    mid = (lo + hi) / 2
    c, s = math.cos(rot), math.sin(rot)
    cx = mid[0] * c - mid[1] * s + centre[0]
    cy = mid[0] * s + mid[1] * c + centre[1]
    return Rect(cx, cy, float(hi[0] - lo[0]), float(hi[1] - lo[1]), rot).transformed(2.0, -0.1)


class NpuHandLandmarker:
    """Drop-in for MediaPipe's HandLandmarker in VIDEO mode, as used by camera.py."""

    def __init__(self, palm_model: str, landmark_model: str, num_hands: int = 2,
                 min_detection: float = 0.6, min_presence: float = 0.6,
                 min_tracking: float = 0.6, device: str = "auto"):
        self.palm, palm_dev = create_session(palm_model, device)
        self.lm, lm_dev = create_session(landmark_model, device)
        self.description = palm_dev if palm_dev == lm_dev else f"palm {palm_dev}, landmarks {lm_dev}"
        self.palm_in = self.palm.get_inputs()[0].name
        self.lm_in = self.lm.get_inputs()[0].name
        names = [o.name for o in self.palm.get_outputs()]
        self.reg_i, self.cls_i = names.index("regressors"), names.index("classificators")
        names = [o.name for o in self.lm.get_outputs()]
        self.pts_i, self.pres_i, self.hand_i = (names.index("screen_landmarks"),
                                                names.index("presence"), names.index("handedness"))
        self.anchors = ssd_anchors()
        self.num_hands = num_hands
        self.min_detection, self.min_presence, self.min_tracking = min_detection, min_presence, min_tracking
        self.tracked: list[Rect] = []
        # The first NPU run of each model compiles its graph.
        self.palm.run(None, {self.palm_in: np.zeros((1, 3, PALM_SIZE, PALM_SIZE), np.float32)})
        self.lm.run(None, {self.lm_in: np.zeros((1, 3, LM_SIZE, LM_SIZE), np.float32)})

    @staticmethod
    def _tensor(rgb: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray((rgb.astype(np.float32) * (1 / 255.0)).transpose(2, 0, 1)[None])

    def detect_palms(self, rgb: np.ndarray) -> list[Rect]:
        h, w = rgb.shape[:2]
        scale = PALM_SIZE / max(w, h)
        pad_x, pad_y = (PALM_SIZE - w * scale) / 2, (PALM_SIZE - h * scale) / 2
        letterbox = cv2.warpAffine(rgb, np.float32([[scale, 0, pad_x], [0, scale, pad_y]]),
                                   (PALM_SIZE, PALM_SIZE), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = self.palm.run(None, {self.palm_in: self._tensor(letterbox)})
        raw, logits = out[self.reg_i][0], out[self.cls_i][0, :, 0]
        scores = 1 / (1 + np.exp(-np.clip(logits, -100, 100)))
        keep = np.flatnonzero(scores >= self.min_detection)
        if keep.size == 0:
            return []
        raw, anchors, scores = raw[keep] / PALM_SIZE, self.anchors[keep], scores[keep]
        centre = raw[:, 0:2] + anchors
        half = raw[:, 2:4] / 2
        boxes = np.concatenate([centre - half, centre + half], 1)
        keypoints = raw[:, 4:18].reshape(-1, 7, 2) + anchors[:, None, :]
        rects = []
        for box, kps, _ in weighted_nms(boxes, scores, keypoints, 0.3, self.num_hands):
            # letterbox (0..1 of 192) -> image pixels
            box = (box.reshape(2, 2) * PALM_SIZE - (pad_x, pad_y)) / scale
            kps = (kps * PALM_SIZE - (pad_x, pad_y)) / scale
            (x0, y0), (x1, y1) = kps[PALM_WRIST], kps[PALM_MIDDLE]
            rot = normalize_radians(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
            (bx0, by0), (bx1, by1) = box
            rects.append(Rect((bx0 + bx1) / 2, (by0 + by1) / 2, bx1 - bx0, by1 - by0, rot)
                         .transformed(2.6, -0.5))
        return rects

    def landmarks(self, rgb: np.ndarray, rect: Rect):
        """(points (21, 2) in image pixels, presence, right-hand probability)."""
        m = rect.crop_matrix(LM_SIZE)
        crop = cv2.warpAffine(rgb, m, (LM_SIZE, LM_SIZE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        out = self.lm.run(None, {self.lm_in: self._tensor(crop)})
        local = np.asarray(out[self.pts_i], np.float32).reshape(21, 3)[:, :2]
        pts = local @ m[:, :2].T + m[:, 2]
        return pts, float(np.ravel(out[self.pres_i])[0]), float(np.ravel(out[self.hand_i])[0])

    def detect(self, rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Hands in an RGB frame as (label, points normalised to 0..1), tracked
        hands first, like MediaPipe's HandLandmarker in VIDEO mode."""
        h, w = rgb.shape[:2]
        rects = list(self.tracked)
        if len(rects) < self.num_hands:
            for rect in self.detect_palms(rgb):
                # HandAssociation: a palm overlapping a tracked hand is that hand.
                if all(iou(rect.bounds(), t.bounds()) <= 0.5 for t in rects):
                    rects.append(rect)
        hands, boxes, self.tracked = [], [], []
        for rect in rects[: self.num_hands]:
            pts, presence, right = self.landmarks(rgb, rect)
            if presence < self.min_presence:
                continue
            box = (*pts.min(0), *pts.max(0))
            if any(iou(box, other) > 0.5 for other in boxes):   # two rects on one hand
                continue
            boxes.append(box)
            hands.append(("Right" if right > 0.5 else "Left", pts / (w, h)))
            self.tracked.append(rect_from_landmarks(pts))
        return hands


def available(intel: bool = False) -> bool:
    """True if the ONNX models and a supported NPU are present (cheap: no model is
    loaded). Qualcomm NPUs count by default; Intel NPUs only with intel=True,
    because a fast desktop CPU may beat them (see scripts/bench_hands.py)."""
    if not all(os.path.isfile(p) for p in default_models()):
        return False
    return bool(_qnn_npu_devices()) or (intel and "NPU" in openvino_devices())


def default_models(here: Optional[str] = None) -> tuple[str, str]:
    here = here or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    return (os.path.join(here, "palm_detection_full_Nx3x192x192.onnx"),
            os.path.join(here, "hand_landmark_full_Nx3x224x224.onnx"))
