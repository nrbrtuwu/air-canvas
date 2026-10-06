"""Hand-tracked drawing canvas with stable two-hand gestures.

Requirements: pip install opencv-python mediapipe numpy
Place ``hand_landmarker.task`` next to this file (or pass --model PATH).
Keys: S = save, C = clear, [ / ] = smaller/bigger brush or eraser,
      F = toggle FPS/frametime overlay, Q/Esc = quit.

Tools: hold your pinky over a color or the eraser in the column at the bottom
left to select it. Slide your pinky along the slider at the bottom right to set the
brush (or eraser) size. With the index finger extended, the active tool draws or erases.
A single fist no longer erases; fists are only used for two-hand undo/redo
(left fist = undo, right fist = redo).

Background: the BG switch at the bottom right (hold your pinky on it for about 0.3 s, or
press B) replaces the camera image with a beige page. Drawing, erasing, the hand
skeleton and the toolbar all work exactly the same on top of it. With the switch on,
S saves the drawing on the beige page instead of on black.

Eraser size (v8): with the eraser selected, the number of raised fingers sets its
size: index only = smallest, + middle = bigger, + ring = bigger, + pinky = biggest.
The eraser is centred on the raised fingertips. [ and ] change the base size.

What changed in v6 (drawing feel + drawing speed):
  * Cursor smoothing is now a One Euro filter instead of a 10-frame median +
    EMA + weighted trail. The old chain delayed the pen by roughly ten frames;
    the One Euro filter is heavy only while your finger is nearly still (to
    remove jitter) and almost lag-free as soon as you move.
  * Compositing only touches the bounding box of the ink instead of blending
    the whole 1280x720 frame on every frame once anything was drawn. That
    full-frame blend was the reason FPS dropped as soon as you started drawing.
  * Eraser strokes use non-antialiased lines: faster, and they no longer leave
    a faint ghost outline in the mask.
  * A small ring is drawn at the pen tip so you can see exactly where it is.
  * The overlay also shows "Paint" time (everything after detection).

What changed in v11:
  * The preview is 1920x1080 by default (--width / --height). The hand detector
    still runs on a 400 px wide copy (--detect-width), so tracking cost is unchanged.
    If the camera cannot deliver 1080p, its frames are upscaled to the preview size.
  * The MediaPipe GPU delegate is no longer attempted on Windows. The pip wheels
    for Windows are built with GPU support disabled, so it always failed and fell
    back to CPU. OpenCL is still used by OpenCV where it helps.
  * MediaPipe/TFLite log noise is silenced (set KAMERA_VERBOSE=1 to see it again).

Tuning:
  --min-cutoff  lower = steadier when still, but more lag on slow strokes (default 1.0)
  --beta        higher = less lag when moving fast, but more jitter (default 0.025)
  INK_OPACITY   1.0 = solid paint (fastest); 0.88 = old look with 12% see-through
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from typing import Optional

# Quiet MediaPipe / TFLite startup warnings. Must be set before mediapipe is imported.
if not os.environ.get("KAMERA_VERBOSE"):
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe landmark indices
WRIST, THUMB_TIP = 0, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

PALETTE = [
    (255, 90, 40), (20, 200, 60), (40, 60, 255), (0, 220, 255),
    (230, 80, 220), (255, 255, 255), (30, 30, 30), (0, 165, 255),
]
MIN_BRUSH, MAX_BRUSH = 3, 48
# Eraser size follows how many fingers are raised (index only = smallest, index +
# middle + ring + pinky = biggest). ``eraser_size`` is the index-only diameter and
# the other levels are multiples of it; [ and ] change that base size.
MIN_ERASER, MAX_ERASER = 12, 80
ERASER_SCALES = (1.0, 1.7, 2.5, 3.4)   # diameter multiplier for 1, 2, 3, 4 fingers up
LEVEL_CONFIRM_FRAMES = 2               # a new finger count must persist this many frames
ERASER_EASING = 0.4                    # 0..1, how fast the eraser grows/shrinks per frame
ERASER_SLOT = len(PALETTE)  # the eraser button sits right after the last color
SWITCH_SLOT = ERASER_SLOT + 1  # the beige-background switch sits right after the eraser
SWITCH_HALF_WIDTH = 34         # pinky hit zone (px) either side of the switch centre
SWITCH_DWELL = 0.30            # seconds the pinky must rest on the switch to flip it
BG_COLOR = (196, 222, 235)     # beige page (BGR), shown instead of the camera when the switch is on


SLOT_STEP = 58                 # vertical distance between buttons in the color column
COLUMN_X = 45                  # horizontal centre of the color column (bottom left)
PANEL_W, PANEL_H = 520, 80     # bottom-right panel: BG switch + size slider
SLOT_HIT = 25                  # pinky must be this close (px) to a color/eraser button


class Layout:
    """Screen positions of the on-screen controls for one frame size.

    Bottom left: a vertical column, colors on top and the eraser at the bottom.
    Bottom right: the background switch and the size slider."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.col_rect = (COLUMN_X - 37, self.slot_y(0) - 35, COLUMN_X + 37, h - 8)
        self.panel_rect = (w - PANEL_W - 8, h - PANEL_H - 8, w - 8, h - 8)
        self.cy = h - 8 - PANEL_H // 2          # vertical centre of the right panel
        self.switch_x = self.panel_rect[0] + 50
        self.slider_x0 = self.switch_x + 90
        self.slider_x1 = w - 30

    def slot_y(self, slot: int) -> int:
        """Vertical centre of a column button (slot 0 at the top, eraser at the bottom)."""
        return self.h - 45 - (ERASER_SLOT - slot) * SLOT_STEP

    def slider_x(self, frac: float) -> int:
        return int(round(self.slider_x0 + frac * (self.slider_x1 - self.slider_x0)))

    @staticmethod
    def _inside(p, rect, margin: int = 0) -> bool:
        x0, y0, x1, y1 = rect
        return x0 - margin <= p[0] <= x1 + margin and y0 - margin <= p[1] <= y1 + margin

    def hit(self, p) -> Optional[tuple]:
        """What a point is over: ("slot", i), ("switch", None), ("size", 0..1),
        ("panel", None) for empty space on a panel, or None when off the panels."""
        if self._inside(p, self.col_rect, 10):
            slot = min(range(ERASER_SLOT + 1), key=lambda i: abs(p[1] - self.slot_y(i)))
            if abs(p[1] - self.slot_y(slot)) < SLOT_HIT:
                return ("slot", slot)
            return ("panel", None)
        if self._inside(p, self.panel_rect, 10):
            if abs(p[0] - self.switch_x) < SWITCH_HALF_WIDTH:
                return ("switch", None)
            if self.slider_x0 - 20 <= p[0] <= self.slider_x1 + 20:
                frac = (p[0] - self.slider_x0) / (self.slider_x1 - self.slider_x0)
                return ("size", float(np.clip(frac, 0.0, 1.0)))
            return ("panel", None)
        return None

# 1.0 = paint is opaque (one cheap masked copy). Below 1.0 the paint is blended
# with the camera image (the old look used 0.88) at the cost of an extra pass.
INK_OPACITY = 1.0

HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                    (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

FINGERS = [(INDEX_MCP, INDEX_PIP, INDEX_TIP), (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
           (RING_MCP, RING_PIP, RING_TIP), (PINKY_MCP, PINKY_PIP, PINKY_TIP)]


# --------------------------------------------------------------------------
# Geometry helpers (all operate on a (21, 2) float32 array of pixel points)
# --------------------------------------------------------------------------
def hand_points(lm, w: int, h: int) -> np.ndarray:
    return np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC, in degrees; straight fingers have a large angle."""
    u, v = a - b, c - b
    den = np.linalg.norm(u) * np.linalg.norm(v)
    if den < 1e-6:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / den, -1, 1))))


def finger_extended(pts: np.ndarray, mcp: int, pip: int, tip_i: int) -> bool:
    """Rotation-tolerant finger state from joint angle and reach."""
    base, joint, tip_p = pts[mcp], pts[pip], pts[tip_i]
    return angle(base, joint, tip_p) > 145 and distance(tip_p, base) > distance(joint, base) * 1.45


def is_fist(pts: np.ndarray) -> bool:
    """Very strict fist test: all four fingers and the thumb must be folded."""
    wrist = pts[WRIST]
    palm = max(distance(wrist, pts[MIDDLE_MCP]), 20.0)
    for mcp, pip, tip_i in FINGERS:
        # Every finger is required; accepting 3/4 caused half-open hands to erase.
        if angle(pts[mcp], pts[pip], pts[tip_i]) >= 112:
            return False
        if distance(pts[tip_i], wrist) >= palm * 1.72:
            return False
    # Thumb tip must be over the palm, not extended sideways.
    return (distance(pts[THUMB_TIP], pts[MIDDLE_MCP]) < palm * 1.05
            and distance(pts[THUMB_TIP], wrist) < palm * 1.75)


def draw_hand_landmarks(frame: np.ndarray, pts: np.ndarray) -> None:
    ipts = [(int(x), int(y)) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, ipts[a], ipts[b], (40, 220, 80), 2, cv2.LINE_AA)
    for point in ipts:
        cv2.circle(frame, point, 4, (40, 70, 255), cv2.FILLED, cv2.LINE_AA)


class Hand:
    __slots__ = ("label", "pts", "fist")

    def __init__(self, label: str, pts: np.ndarray, fist: bool) -> None:
        self.label, self.pts, self.fist = label, pts, fist


# --------------------------------------------------------------------------
# Threaded camera
# --------------------------------------------------------------------------
class CameraStream:
    """Grabs frames in a background thread so the main loop never blocks on I/O."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame: Optional[np.ndarray] = None
        self.seq = 0
        self.last_seq = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            with self.cond:
                if not ok:
                    self.running = False
                else:
                    self.frame = frame
                    self.seq += 1
                self.cond.notify_all()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Return the newest unseen frame (older frames are dropped), or None."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != self.last_seq or not self.running, timeout)
            if self.seq == self.last_seq:
                return None
            self.last_seq = self.seq
            return self.frame

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)

class Acceleration:
    """Select the fastest available OpenCV path without requiring GPU hardware."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.cuda = mode != "off" and hasattr(cv2, "cuda")
        self.cuda = self.cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.opencl = mode != "off" and cv2.ocl.haveOpenCL()
        if self.opencl:
            cv2.ocl.setUseOpenCL(True)
        if mode == "on" and not (self.cuda or self.opencl):
            raise RuntimeError("GPU acceleration was requested, but no CUDA/OpenCL backend is available")

    @property
    def media_pipe_delegate(self):
        """GPU delegate is attempted only when acceleration was requested/available.

        The MediaPipe pip wheels for Windows are compiled with GPU disabled
        (MEDIAPIPE_DISABLE_GPU), so the GPU delegate can never start there.
        Skip it instead of failing and falling back every launch."""
        if sys.platform == "win32":
            return python.BaseOptions.Delegate.CPU
        if self.mode == "off" or not (self.cuda or self.opencl):
            return python.BaseOptions.Delegate.CPU
        return python.BaseOptions.Delegate.GPU

    def prepare(self, frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Resize and convert on CUDA/OpenCL, then download for MediaPipe."""
        if self.cuda:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(frame)
            if (frame.shape[1], frame.shape[0]) != size:
                gpu = cv2.cuda.resize(gpu, size, interpolation=cv2.INTER_AREA)
            gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2RGB)
            return gpu.download()
        if self.opencl:
            image = cv2.UMat(frame)
            if (frame.shape[1], frame.shape[0]) != size:
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).get()
        small = frame if (frame.shape[1], frame.shape[0]) == size else cv2.resize(
            frame, size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# Performance statistics
# --------------------------------------------------------------------------
class PerfStats:
    """FPS (0.5 s window), smoothed frametime, worst frametime, detect and paint time."""

    def __init__(self) -> None:
        now = time.perf_counter()
        self.last = now
        self.window_start = now
        self.window_frames = 0
        self.window_worst = 0.0
        self.fps = 0.0
        self.frametime = 0.0      # smoothed, ms
        self.worst = 0.0          # worst frametime in last window, ms
        self.infer = 0.0          # smoothed hand-detection time, ms
        self.paint = 0.0          # smoothed gesture + drawing + compositing time, ms

    def tick(self, infer_ms: float, paint_ms: float) -> None:
        now = time.perf_counter()
        dt_ms = (now - self.last) * 1000.0
        self.last = now
        self.frametime = dt_ms if self.frametime == 0 else 0.9 * self.frametime + 0.1 * dt_ms
        self.infer = infer_ms if self.infer == 0 else 0.9 * self.infer + 0.1 * infer_ms
        self.paint = paint_ms if self.paint == 0 else 0.9 * self.paint + 0.1 * paint_ms
        self.window_frames += 1
        self.window_worst = max(self.window_worst, dt_ms)
        elapsed = now - self.window_start
        if elapsed >= 0.5:
            self.fps = self.window_frames / elapsed
            self.worst = self.window_worst
            self.window_start, self.window_frames, self.window_worst = now, 0, 0.0


# --------------------------------------------------------------------------
# Cursor smoothing (One Euro filter) and painter
# --------------------------------------------------------------------------
class OneEuroCursor:
    """Adaptive low-pass filter: smooth when slow, nearly lag-free when fast.

    The cutoff frequency rises with the filtered speed of the point, so jitter
    is removed while the finger is still, but fast strokes follow the finger
    almost immediately. Uses real timestamps, so it behaves the same at any FPS.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.025, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: Optional[np.ndarray] = None
        self.dx = np.zeros(2, np.float32)
        self.t = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, raw: np.ndarray, t: float) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        if self.x is None:
            self.x, self.t = raw.copy(), t
            self.dx[:] = 0.0
            return self.x.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (raw - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1.0 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.hypot(self.dx[0], self.dx[1]))
        a = self._alpha(cutoff, dt)
        self.x = a * raw + (1.0 - a) * self.x
        return self.x.copy()

    def reset(self) -> None:
        self.x = None
        self.dx[:] = 0.0


class Painter:
    def __init__(self, w: int, h: int, min_cutoff: float = 1.0, beta: float = 0.025):
        self.w, self.h = w, h
        self.canvas = np.zeros((h, w, 3), np.uint8)
        self.mask = np.zeros((h, w), np.uint8)
        # Bounding box (x0, y0, x1, y1) that contains all ink; None when empty.
        # Compositing only touches this region.
        self.ink_box: Optional[tuple[int, int, int, int]] = None
        self._erased = False
        self.tip: Optional[tuple[int, int]] = None  # current pen/eraser tip (for the ring marker)
        self.tip_radius = 6
        self.undo_stack: list[tuple] = []
        self.redo_stack: list[tuple] = []
        self.tool = "draw"        # "draw" or "erase"
        self.color = PALETTE[0]
        self.brush = 10
        self.eraser_size = 48     # base diameter in px (index finger only)
        self.eraser_level = 1     # committed finger count, 1..4
        self._level_pending = 1
        self._level_votes = 0
        self.eraser_diam = float(self.eraser_size)  # eased diameter actually used
        self.bg_on = False                          # beige background switch
        self.switch_progress = 0.0                  # 0..1 dwell progress on the switch (for the UI)
        self._switch_since: Optional[float] = None
        self._switch_armed = True                   # must leave the switch before it can flip again
        self.cursor = OneEuroCursor(min_cutoff, beta)
        self.previous: Optional[tuple[int, int]] = None
        self.status, self.status_until = "Draw", 0.0
        self.last_action = 0.0
        self.shake_samples: deque = deque(maxlen=9)

    def set_status(self, text: str, seconds: float = 0.45) -> None:
        self.status, self.status_until = text, time.monotonic() + seconds

    # ---- ink bounding box ----
    def _grow_box(self, a: tuple[int, int], b: tuple[int, int], width: int) -> None:
        r = width // 2 + 3
        x0 = max(0, min(a[0], b[0]) - r)
        y0 = max(0, min(a[1], b[1]) - r)
        x1 = min(self.w, max(a[0], b[0]) + r + 1)
        y1 = min(self.h, max(a[1], b[1]) + r + 1)
        if x1 <= x0 or y1 <= y0:
            return
        if self.ink_box is None:
            self.ink_box = (x0, y0, x1, y1)
        else:
            bx0, by0, bx1, by1 = self.ink_box
            self.ink_box = (min(bx0, x0), min(by0, y0), max(bx1, x1), max(by1, y1))

    def _recompute_box(self) -> None:
        x, y, bw, bh = cv2.boundingRect(self.mask)
        self.ink_box = None if bw == 0 or bh == 0 else (x, y, x + bw, y + bh)

    # ---- history ----
    # A state only stores the ink's bounding box, because everything outside it
    # is empty. That is far cheaper than copying the whole 1280x720 canvas each
    # time a stroke starts (which used to cause a visible hitch).
    def _capture(self) -> tuple:
        if self.ink_box is None:
            return (None, None, None)
        x0, y0, x1, y1 = self.ink_box
        return (self.ink_box, self.canvas[y0:y1, x0:x1].copy(), self.mask[y0:y1, x0:x1].copy())

    def _restore(self, state: tuple) -> None:
        box, canvas_crop, mask_crop = state
        self.canvas.fill(0)
        self.mask.fill(0)
        if box is not None:
            x0, y0, x1, y1 = box
            self.canvas[y0:y1, x0:x1] = canvas_crop
            self.mask[y0:y1, x0:x1] = mask_crop
        self.ink_box = box

    def snapshot(self) -> None:
        self.undo_stack.append(self._capture())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self) -> None:
        if self.undo_stack:
            self.redo_stack.append(self._capture())
            self._restore(self.undo_stack.pop())
            self.set_status("UNDO", 1.0)

    def redo(self) -> None:
        if self.redo_stack:
            self.undo_stack.append(self._capture())
            self._restore(self.redo_stack.pop())
            self.set_status("REDO", 1.0)

    def clear(self) -> None:
        self.snapshot()
        self.canvas.fill(0)
        self.mask.fill(0)
        self.ink_box = None
        self.previous = None
        self.tip = None
        self.set_status("CLEAR", 1.0)

    # ---- tools ----
    @property
    def erasing(self) -> bool:
        return self.tool == "erase"

    def select_slot(self, slot: int) -> None:
        """Pick a toolbar slot: 0..len(PALETTE)-1 are colors, the next one is the eraser."""
        if slot == ERASER_SLOT:
            if self.tool != "erase":
                self.tool = "erase"
                self.set_status("ERASER selected")
        else:
            if self.tool != "draw" or self.color != PALETTE[slot]:
                self.tool = "draw"
                self.color = PALETTE[slot]
                self.set_status("COLOR selected")

    def toggle_bg(self) -> None:
        self.bg_on = not self.bg_on
        self.set_status("BEIGE BACKGROUND ON" if self.bg_on else "CAMERA BACKGROUND", 1.0)

    def update_switch_hover(self, over: bool) -> None:
        """Call once per frame. Flips the switch after the pinky rests on it for
        SWITCH_DWELL seconds; it must then leave the switch before it can flip again,
        so sweeping across the bar or hovering never makes it flicker."""
        if not over:
            self._switch_since, self._switch_armed, self.switch_progress = None, True, 0.0
            return
        if not self._switch_armed:
            return
        now = time.monotonic()
        if self._switch_since is None:
            self._switch_since = now
        held = now - self._switch_since
        self.switch_progress = min(1.0, held / SWITCH_DWELL)
        if held >= SWITCH_DWELL:
            self.toggle_bg()
            self._switch_since, self._switch_armed, self.switch_progress = None, False, 0.0

    def eraser_target(self, level: Optional[int] = None) -> float:
        """Eraser diameter in px for a finger count (default: the committed one)."""
        level = self.eraser_level if level is None else level
        return self.eraser_size * ERASER_SCALES[int(np.clip(level, 1, len(ERASER_SCALES))) - 1]

    def observe_eraser_level(self, raised: int) -> int:
        """Feed this frame's raised-finger count; returns the committed level.

        A change only counts after LEVEL_CONFIRM_FRAMES identical frames, so a
        finger flickering for one frame does not make the eraser jump.
        """
        raised = int(np.clip(raised, 1, len(ERASER_SCALES)))
        if raised == self.eraser_level:
            self._level_votes = 0
        elif raised == self._level_pending:
            self._level_votes += 1
            if self._level_votes >= LEVEL_CONFIRM_FRAMES:
                self.eraser_level, self._level_votes = raised, 0
        else:
            self._level_pending, self._level_votes = raised, 1
            if LEVEL_CONFIRM_FRAMES <= 1:
                self.eraser_level, self._level_votes = raised, 0
        return self.eraser_level

    def adjust_size(self, direction: int) -> None:
        """[ and ] keys: shrink/grow whichever tool is active."""
        if self.erasing:
            self.eraser_size = int(np.clip(self.eraser_size + 4 * direction, MIN_ERASER, MAX_ERASER))
            lo, hi = self.eraser_target(1), self.eraser_target(len(ERASER_SCALES))
            self.set_status(f"Eraser {lo:.0f}-{hi:.0f}px", 0.8)
        else:
            self.brush = int(np.clip(self.brush + 2 * direction, MIN_BRUSH, MAX_BRUSH))
            self.set_status(f"Brush {self.brush}px", 0.8)

    def size_fraction(self) -> float:
        """Active tool size as 0..1, for the slider."""
        if self.erasing:
            return (self.eraser_size - MIN_ERASER) / (MAX_ERASER - MIN_ERASER)
        return (self.brush - MIN_BRUSH) / (MAX_BRUSH - MIN_BRUSH)

    def set_size_fraction(self, frac: float) -> None:
        """Pinky on the slider: set the active tool size from its position (0..1)."""
        if self.erasing:
            self.eraser_size = int(round(MIN_ERASER + frac * (MAX_ERASER - MIN_ERASER)))
            lo, hi = self.eraser_target(1), self.eraser_target(len(ERASER_SCALES))
            self.set_status(f"Eraser {lo:.0f}-{hi:.0f}px", 0.8)
        else:
            self.brush = int(round(MIN_BRUSH + frac * (MAX_BRUSH - MIN_BRUSH)))
            self.set_status(f"Brush {self.brush}px", 0.8)

    # ---- drawing ----
    def draw(self, raw: np.ndarray) -> None:
        """Extend the current stroke with the active tool (brush or eraser)."""
        erase = self.erasing
        p = self.cursor.update(raw, time.perf_counter())
        current = (int(round(float(p[0]))), int(round(float(p[1]))))
        if self.previous is None:
            self.snapshot()
            if erase:  # start the stroke at the right size instead of easing from a stale one
                self.eraser_diam = self.eraser_target()
        elif erase:
            # The eraser eases toward the size for the current finger count, so it
            # grows and shrinks smoothly instead of snapping.
            target = self.eraser_target()
            self.eraser_diam += (target - self.eraser_diam) * ERASER_EASING
            if abs(target - self.eraser_diam) < 0.5:
                self.eraser_diam = target
            width = max(1, int(round(self.eraser_diam)))
            # No anti-aliasing here: AA would leave partially-erased mask
            # pixels (a ghost outline) and costs more on a wide eraser.
            cv2.line(self.canvas, self.previous, current, (0, 0, 0), width, cv2.LINE_8)
            cv2.line(self.mask, self.previous, current, 0, width, cv2.LINE_8)
            self._erased = True
        else:
            # Short gaps are bridged naturally by a continuous stabilized segment.
            cv2.line(self.canvas, self.previous, current, self.color, self.brush, cv2.LINE_AA)
            cv2.line(self.mask, self.previous, current, 255, self.brush, cv2.LINE_AA)
            self._grow_box(self.previous, current, self.brush)
        self.previous = current
        self.tip = current
        self.tip_radius = max(1, int(round(self.eraser_diam)) // 2) if erase else max(4, self.brush // 2 + 3)

    def end_stroke(self) -> None:
        self.previous = None
        self.tip = None
        self.cursor.reset()
        if self._erased:
            # Shrink the ink box again after erasing (runs once per erase stroke).
            self._erased = False
            self._recompute_box()

    def composite(self, frame: np.ndarray) -> None:
        """Paint the ink onto ``frame`` in place, touching only the ink's bounding box."""
        if self.ink_box is None:
            return
        x0, y0, x1, y1 = self.ink_box
        dst = frame[y0:y1, x0:x1]
        src = self.canvas[y0:y1, x0:x1]
        m = self.mask[y0:y1, x0:x1]
        if INK_OPACITY >= 0.999:
            cv2.copyTo(src, m, dst)
        else:
            blended = cv2.addWeighted(dst, 1.0 - INK_OPACITY, src, INK_OPACITY, 0)
            cv2.copyTo(blended, m, dst)

    def maybe_shake_clear(self, raw: np.ndarray) -> None:
        self.shake_samples.append(raw)
        if len(self.shake_samples) == self.shake_samples.maxlen:
            xs = np.asarray(self.shake_samples)[:, 0]
            changes = np.count_nonzero(np.diff(np.sign(np.diff(xs))))
            span = xs.max() - xs.min()
            if changes >= 4 and span > 180:
                self.clear()
                self.shake_samples.clear()


def eraser_step(painter: Painter, pts: np.ndarray) -> int:
    """Erase for one frame. Size comes from the raised fingers (index is already up):
    index = smallest, +middle, +ring, +pinky = biggest. The eraser is centred on the
    raised fingertips, so it covers what your fingers point at. Returns the level."""
    raised = 1 + sum(finger_extended(pts, *f) for f in FINGERS[1:])
    level = painter.observe_eraser_level(raised)
    tips = np.asarray([pts[f[2]] for f in FINGERS[:level]], dtype=np.float32)
    painter.draw(tips.mean(axis=0))
    return level


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def put_text_shadow(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1) -> None:
    """Text with a dark outline so it stays readable on the camera and on beige."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_switch(frame: np.ndarray, painter: Painter, layout: Layout) -> None:
    """Pill-shaped on/off switch for the beige background (bottom-right panel)."""
    sx, cy, r = layout.switch_x, layout.cy, 13
    x0, x1 = sx - 26, sx + 26
    track = (110, 190, 100) if painter.bg_on else (90, 90, 90)
    cv2.circle(frame, (x0, cy), r, track, -1, cv2.LINE_AA)
    cv2.circle(frame, (x1, cy), r, track, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, cy - r), (x1, cy + r), track, -1)
    cv2.circle(frame, (x1 if painter.bg_on else x0, cy), r - 3, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.putText(frame, "BG", (x1 + r + 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1, cv2.LINE_AA)
    if painter.switch_progress > 0:  # dwell progress bar under the switch
        end = x0 - r + int((x1 - x0 + 2 * r) * painter.switch_progress)
        cv2.line(frame, (x0 - r, cy + 24), (end, cy + 24), (255, 255, 255), 3)


def draw_ui(frame: np.ndarray, painter: Painter, layout: Layout, stats: PerfStats, show_stats: bool) -> None:
    h, w = frame.shape[:2]

    # ---- bottom left: vertical color column, eraser at the bottom ----
    x0, y0, x1, y1 = layout.col_rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), (25, 25, 25), -1)
    for i, col in enumerate(PALETTE):
        cy = layout.slot_y(i)
        cv2.circle(frame, (COLUMN_X, cy), 18, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (COLUMN_X, cy), 18, (110, 110, 110), 1, cv2.LINE_AA)  # keeps black visible
        if not painter.erasing and col == painter.color:
            cv2.circle(frame, (COLUMN_X, cy), 22, (255, 255, 255), 2, cv2.LINE_AA)
    ex, ey = COLUMN_X, layout.slot_y(ERASER_SLOT)
    cv2.rectangle(frame, (ex - 17, ey - 12), (ex + 17, ey + 12), (225, 225, 225), -1)
    cv2.rectangle(frame, (ex - 17, ey - 12), (ex - 3, ey + 12), (170, 120, 255), -1)
    cv2.rectangle(frame, (ex - 17, ey - 12), (ex + 17, ey + 12), (90, 90, 90), 1, cv2.LINE_AA)
    if painter.erasing:
        cv2.rectangle(frame, (ex - 23, ey - 18), (ex + 23, ey + 18), (255, 255, 255), 2, cv2.LINE_AA)

    # ---- bottom right: background switch + size slider ----
    px0, py0, px1, py1 = layout.panel_rect
    cv2.rectangle(frame, (px0, py0), (px1, py1), (25, 25, 25), -1)
    draw_switch(frame, painter, layout)
    sx0, sx1, cy = layout.slider_x0, layout.slider_x1, layout.cy
    cv2.rectangle(frame, (sx0, cy - 8), (sx1, cy + 8), (75, 75, 75), -1)
    kx = layout.slider_x(painter.size_fraction())
    fill = (170, 170, 170) if painter.erasing else painter.color
    cv2.rectangle(frame, (sx0, cy - 8), (kx, cy + 8), fill, -1)
    cv2.circle(frame, (kx, cy), 12, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(frame, (kx, cy), 12, (60, 60, 60), 1, cv2.LINE_AA)
    if painter.erasing:
        label = f"Eraser {painter.eraser_level}/{len(ERASER_SCALES)}  {painter.eraser_target():.0f}px"
    else:
        label = f"Brush {painter.brush}px"
    put_text_shadow(frame, label, (sx0, cy - 20), 0.5)

    # ---- status line, top centre ----
    idle = ("Erase: index up, more fingers = bigger | Pinky on panels: tools, size & BG" if painter.erasing
            else "Draw: index finger | Pinky on panels: colors, eraser, size & BG")
    text = painter.status if time.monotonic() < painter.status_until else idle
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    put_text_shadow(frame, text, ((w - tw) // 2, 40), 0.62, (255, 255, 255), 2)

    if show_stats:
        fps_color = (80, 255, 80) if stats.fps >= 24 else (0, 200, 255) if stats.fps >= 15 else (60, 60, 255)
        put_text_shadow(frame, f"FPS: {stats.fps:5.1f}", (16, 80), 0.6, fps_color)
        put_text_shadow(frame, f"Frame: {stats.frametime:5.1f} ms (worst {stats.worst:.0f})", (16, 104))
        put_text_shadow(frame, f"Detect: {stats.infer:5.1f} ms", (16, 126))
        put_text_shadow(frame, f"Paint: {stats.paint:5.1f} ms", (16, 148))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920, help="preview / canvas width")
    ap.add_argument("--height", type=int, default=1080, help="preview / canvas height")
    ap.add_argument("--detect-width", type=int, default=400,
                    help="width of the image given to the hand detector (0 = full size). Lower = faster.")
    ap.add_argument("--min-cutoff", type=float, default=1.0,
                    help="pen smoothing when the finger is still. Lower = steadier, higher = more responsive.")
    ap.add_argument("--beta", type=float, default=0.025,
                    help="how quickly smoothing relaxes as you move. Higher = less lag on fast strokes.")
    ap.add_argument("--gpu", choices=("auto", "on", "off"), default="auto",
                    help="GPU acceleration mode: auto detects CUDA/OpenCL, on requires it, off disables it.")
    args = ap.parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Hand model not found: {args.model}")

    acceleration = Acceleration(args.gpu)
    backends = []
    if acceleration.cuda:
        backends.append(f"CUDA ({cv2.cuda.getCudaEnabledDeviceCount()} device)")
    if acceleration.opencl:
        backends.append("OpenCL")
    print(f"OpenCV acceleration: {', '.join(backends) if backends else 'CPU'}")
    if sys.platform == "win32":
        print("MediaPipe: CPU (the Windows pip build has no GPU delegate)")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    # MJPG usually unlocks 30 fps at 1080p (raw YUYV at 1080p is often capped at 5 fps).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Cannot read camera")
    cam_h, cam_w = first.shape[:2]
    w, h = args.width, args.height
    # If the camera delivers a different size, every frame is resized to the preview size.
    upscale = (cam_w, cam_h) != (w, h)
    print(f"Camera: {cam_w}x{cam_h}  Preview: {w}x{h}" + ("  (resized)" if upscale else ""))

    if args.detect_width and args.detect_width < w:
        det_w = args.detect_width
        det_h = max(1, round(h * det_w / w))
    else:
        det_w, det_h = w, h

    painter = Painter(w, h, args.min_cutoff, args.beta)
    layout = Layout(w, h)
    stats = PerfStats()
    show_stats = True
    window_title = "Hand Paint - S save | C clear | B background | [ ] size | F stats | Q quit"
    # WINDOW_NORMAL lets the view scale when you resize or maximize it.
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    initial_w = min(1600, max(960, w))
    cv2.resizeWindow(window_title, initial_w, max(540, round(initial_w * h / w)))
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=args.model,
            delegate=acceleration.media_pipe_delegate,
        ),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.60, min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60,
    )
    cam = CameraStream(cap)
    timestamp_ms = 0
    # Per-side transition state prevents repeated undo/redo while a hand stays closed.
    action_armed = {"Left": True, "Right": True}
    bg_img = np.full((h, w, 3), BG_COLOR, np.uint8)  # beige page, copied over the frame when the switch is on
    try:
        try:
            landmarker_context = vision.HandLandmarker.create_from_options(options)
        except (RuntimeError, ValueError) as error:
            if acceleration.media_pipe_delegate != python.BaseOptions.Delegate.GPU:
                raise
            print(f"MediaPipe GPU delegate unavailable ({error}); using CPU delegate")
            options.base_options = python.BaseOptions(
                model_asset_path=args.model,
                delegate=python.BaseOptions.Delegate.CPU,
            )
            landmarker_context = vision.HandLandmarker.create_from_options(options)
        with landmarker_context as landmarker:
            while True:
                frame = cam.read()
                if frame is None:
                    if not cam.running:
                        break
                    continue
                frame = cv2.flip(frame, 1)

                # ---- hand detection (on a downscaled copy) ----
                # Detection reads the camera frame before any upscaling, so the
                # tracking input stays the same size whatever the preview size is.
                # Landmarks are normalized (0..1), so they map onto the preview directly.
                t0 = time.perf_counter()
                rgb = acceleration.prepare(frame, (det_w, det_h))
                if upscale:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(timestamp_ms + 1, int(time.monotonic() * 1000))
                result = landmarker.detect_for_video(image, timestamp_ms)
                t1 = time.perf_counter()
                infer_ms = (t1 - t0) * 1000.0

                hands: dict[str, Hand] = {}
                for lm, handed in zip(result.hand_landmarks, result.handedness):
                    # Tasks labels assume a mirrored/selfie input, which is exactly what
                    # we supply above; keep the user's natural Left/Right labels.
                    label = handed[0].category_name
                    pts = hand_points(lm, w, h)
                    hands[label] = Hand(label, pts, is_fist(pts))  # fist computed once

                over_switch = False  # pinky resting on the background switch this frame
                ui_point = None      # pinky position while it is on a panel (drawn as a ring)

                # Two-hand-only undo/redo. Require a short closed/open state before transition.
                if len(hands) == 2:
                    closed_sides = [side for side, hand in hands.items() if hand.fist]
                    # A single closing hand creates one action.  Two simultaneous fists
                    # intentionally do nothing, avoiding ambiguous undo+redo sequences.
                    if len(closed_sides) == 1:
                        side = closed_sides[0]
                        if action_armed[side] and time.monotonic() - painter.last_action > 0.55:
                            (painter.undo if side == "Left" else painter.redo)()
                            action_armed[side] = False
                            painter.last_action = time.monotonic()
                    for side, hand in hands.items():
                        if not hand.fist:
                            action_armed[side] = True
                    painter.end_stroke()
                else:
                    # Restore gesture arming when the other hand leaves view.
                    for side in action_armed:
                        if side not in hands or not hands[side].fist:
                            action_armed[side] = True
                    if hands:
                        hand = next(iter(hands.values()))
                        pts = hand.pts
                        index = pts[INDEX_TIP]
                        pinky = pts[PINKY_TIP]
                        # Panels only react when no stroke is running.
                        zone = layout.hit(pinky) if painter.previous is None else None
                        if hand.fist:
                            # A single fist is just "pen up" now; fists only matter
                            # for two-hand undo/redo.
                            painter.end_stroke()
                            painter.shake_samples.clear()
                        elif zone is not None:
                            # Pinky on a panel: colors/eraser (left column), BG switch or
                            # size slider (bottom right). A stroke in progress is never
                            # interrupted, so drawing near the panels stays safe.
                            painter.end_stroke()
                            painter.shake_samples.clear()
                            ui_point = (int(pinky[0]), int(pinky[1]))
                            kind, value = zone
                            if kind == "switch":
                                over_switch = True
                            elif kind == "slot":
                                painter.select_slot(value)
                            elif kind == "size":
                                painter.set_size_fraction(value)
                        elif finger_extended(pts, INDEX_MCP, INDEX_PIP, INDEX_TIP):
                            # The active tool works exclusively while the index is extended.
                            if painter.erasing:
                                level = eraser_step(painter, pts)
                                painter.set_status(f"ERASE {level}/{len(ERASER_SCALES)}")
                            else:
                                painter.draw(index)
                                painter.maybe_shake_clear(index)
                                painter.set_status("DRAW: index extended")
                        else:
                            painter.end_stroke()
                            painter.shake_samples.clear()
                    else:
                        painter.end_stroke()
                        painter.shake_samples.clear()

                painter.update_switch_hover(over_switch)

                # ---- background: with the switch on, the camera image is replaced by beige ----
                # Detection above already used the camera frame, so it is safe to overwrite it.
                if painter.bg_on:
                    np.copyto(frame, bg_img)

                # ---- compositing: only the ink's bounding box is touched ----
                # The frame is not needed afterwards, so draw on it directly (no copy).
                painter.composite(frame)
                # The skeleton, eraser radius and pen tip are drawn last, so paint never hides them.
                for hand in hands.values():
                    draw_hand_landmarks(frame, hand.pts)
                if painter.tip is not None:
                    # Light rings disappear on beige, so use dark ones there.
                    if painter.erasing:  # eraser outline shows exactly what will be wiped
                        ring = (70, 70, 70) if painter.bg_on else (210, 210, 210)
                        cv2.circle(frame, painter.tip, painter.tip_radius, ring, 2, cv2.LINE_AA)
                    else:
                        ring = (50, 50, 50) if painter.bg_on else (255, 255, 255)
                        cv2.circle(frame, painter.tip, painter.tip_radius, ring, 1, cv2.LINE_AA)
                paint_ms = (time.perf_counter() - t1) * 1000.0

                stats.tick(infer_ms, paint_ms)
                draw_ui(frame, painter, layout, stats, show_stats)
                if ui_point is not None:  # show where the pinky is on the panels
                    cv2.circle(frame, ui_point, 10, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, ui_point, 12, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    painter.clear()
                if key == ord("f"):
                    show_stats = not show_stats
                if key == ord("["):
                    painter.adjust_size(-1)
                if key == ord("]"):
                    painter.adjust_size(+1)
                if key == ord("b"):
                    painter.toggle_bg()
                if key == ord("s"):
                    name = f"drawing_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    if painter.bg_on:
                        # Save what you see: the drawing on the beige page.
                        page = bg_img.copy()
                        cv2.copyTo(painter.canvas, painter.mask, page)
                        cv2.imwrite(name, page)
                    else:
                        cv2.imwrite(name, painter.canvas)
                    painter.set_status(f"SAVED {name}", 1.5)
    finally:
        cam.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
