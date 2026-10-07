"""

Hand-tracked drawing canvas with two-hand gestures (MediaPipe + OpenCV).

Needs hand_landmarker.task next to this file (or --model PATH).
Keys: S - save, C - clear, B - background, [  ] - brush size, F - FPS overlay, Q/Esc - quit.

"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp_process
import os
import sys
import threading
import time
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
SWITCH_DWELL = 0.30            # seconds the pinky must rest on the switch to flip it
CLEAR_DWELL = 1.2              # seconds the pinky must rest on Clear (long, so it is never an accident)
PEN_JUMP = 0.2                 # a hand this far (fraction of frame width) from a pen's last spot is a new hand
LOW_LIGHT = 55                 # mean camera brightness (0..255) below which a "too dark" warning shows
BG_COLOR = (196, 222, 235)     # beige page (BGR), shown instead of the camera when the switch is on
# A picture with one of these names next to camera.py replaces the beige page.
BG_IMAGE_NAMES = ("background.png", "background.jpg", "background.jpeg")


# Left: color column, vertically centred.
SLOT_STEP = 68                 # vertical distance between buttons in the color column
COLUMN_X = 62                  # horizontal centre of the color column
COLUMN_HALF_W = 50             # half width of the column background
SLOT_R = 26                    # radius of a color button
SLOT_HIT = 33                  # pinky must be this close (px) to a color/eraser button
# Right: BG switch on top, vertical size slider, Clear button; vertically centred.
PANEL_W, PANEL_H = 124, 670
SWITCH_Y = 75                  # switch centre, from the panel top
SWITCH_HALF_LEN, SWITCH_R = 26, 22   # vertical pill: half distance between end centres, radius
SLIDER_TOP, SLIDER_BOTTOM = 190, 470  # slider ends, from the panel top (top = biggest)
SLIDER_HALF_W, KNOB_R = 12, 22
CLEAR_Y, CLEAR_HALF_H = 610, 34       # Clear button centre (from the panel top) and half height
# All UI sizes above are in screen pixels on a 1080p monitor. They are scaled to the
# monitor resolution, not to the preview resolution. The UI is drawn after the frame
# has been resized to the window, so it is never stretched (and never pixelated).
UI_REFERENCE_HEIGHT = 1080


def enable_dpi_awareness() -> None:
    """Stop Windows from bitmap-stretching the window on scaled (125%/150%) displays."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def screen_height() -> int:
    """Height of the primary monitor in pixels (1080 if it cannot be found)."""
    if sys.platform == "win32":
        try:
            import ctypes
            return int(ctypes.windll.user32.GetSystemMetrics(1)) or UI_REFERENCE_HEIGHT
        except (AttributeError, OSError):
            pass
        return UI_REFERENCE_HEIGHT
    try:  # Linux / macOS: Tk is part of most Python installs (Debian/Ubuntu: python3-tk)
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        height = int(root.winfo_screenheight())
        root.destroy()
        return height or UI_REFERENCE_HEIGHT
    except Exception:
        return UI_REFERENCE_HEIGHT


def display_size(window: str, w: int, h: int) -> tuple[int, int]:
    """Largest size with the frame's aspect ratio that fits the window's image area."""
    try:
        _, _, dw, dh = cv2.getWindowImageRect(window)
    except cv2.error:
        dw = dh = 0
    if dw <= 0 or dh <= 0:
        return w, h
    f = min(dw / w, dh / h)
    return max(1, round(w * f)), max(1, round(h * f))


def load_background(w: int, h: int) -> tuple[np.ndarray, bool]:
    """The page shown when the BG switch is on: background.png/.jpg next to this file,
    scaled to cover the frame (centre-cropped, never stretched), or plain beige.
    Also returns whether the page is light, so markers drawn on it stay visible."""
    folder = os.path.dirname(os.path.abspath(__file__))
    for name in BG_IMAGE_NAMES:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        # imdecode instead of imread: imread cannot open non-ASCII paths on Windows.
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Background: could not read {name}, using beige")
            break
        ih, iw = image.shape[:2]
        f = max(w / iw, h / ih)
        sw, sh = max(w, round(iw * f)), max(h, round(ih * f))
        scaled = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        x0, y0 = (sw - w) // 2, (sh - h) // 2
        page = np.ascontiguousarray(scaled[y0:y0 + h, x0:x0 + w])
        print(f"Background: {name} ({iw}x{ih})")
        return page, float(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).mean()) > 128
    return np.full((h, w, 3), BG_COLOR, np.uint8), True


def ui_scale(screen_h: int, view_h: int) -> float:
    """Display pixels per UI pixel. Follows the monitor resolution; only shrinks when
    the window is too short for the side panels."""
    tallest = max(ERASER_SLOT * SLOT_STEP + 2 * COLUMN_HALF_W, PANEL_H)
    return float(np.clip(screen_h / UI_REFERENCE_HEIGHT, 0.4, max(0.4, (view_h - 40) / tallest)))


class Layout:
    """Screen positions of the on-screen controls for one frame size and UI scale.

    Left, vertically centred: a column with the colors on top and the eraser at the bottom.
    Right, vertically centred: the background switch, the size slider and Clear."""

    def __init__(self, w: int, h: int, s: float = 1.0) -> None:
        self.w, self.h, self.s = w, h, s
        px = self.px
        self.column_x = px(COLUMN_X)
        self.slot_step = px(SLOT_STEP)
        self.slot_hit = px(SLOT_HIT)
        self.col_rect = (self.column_x - px(COLUMN_HALF_W), self.slot_y(0) - px(COLUMN_HALF_W),
                         self.column_x + px(COLUMN_HALF_W), self.slot_y(ERASER_SLOT) + px(COLUMN_HALF_W))
        pad = px(8)
        top = h // 2 - px(PANEL_H) // 2
        self.panel_rect = (w - pad - px(PANEL_W), top, w - pad, top + px(PANEL_H))
        self.panel_x = (self.panel_rect[0] + self.panel_rect[2]) // 2   # horizontal centre
        self.switch_y = top + px(SWITCH_Y)
        self.slider_y0 = top + px(SLIDER_TOP)       # biggest size
        self.slider_y1 = top + px(SLIDER_BOTTOM)    # smallest size
        self.clear_y = top + px(CLEAR_Y)

    def px(self, v: float) -> int:
        """A UI size in frame pixels (never below 1)."""
        return max(1, int(round(v * self.s)))

    def font(self, size: float) -> float:
        return size * self.s

    def slot_y(self, slot: int) -> int:
        """Vertical centre of a column button (slot 0 at the top, eraser at the bottom)."""
        return self.h // 2 + int(round((slot - ERASER_SLOT / 2) * self.slot_step))

    def slider_y(self, frac: float) -> int:
        """Slider position for a size fraction (0 = bottom/smallest, 1 = top/biggest)."""
        return int(round(self.slider_y1 - frac * (self.slider_y1 - self.slider_y0)))

    @staticmethod
    def _inside(p, rect, margin: int = 0) -> bool:
        x0, y0, x1, y1 = rect
        return x0 - margin <= p[0] <= x1 + margin and y0 - margin <= p[1] <= y1 + margin

    def hit(self, p) -> Optional[tuple]:
        """What a point is over: ("slot", i), ("switch", None), ("size", 0..1),
        ("clear", None), ("panel", None) for empty space on a panel, or None."""
        if self._inside(p, self.col_rect, self.px(10)):
            slot = min(range(ERASER_SLOT + 1), key=lambda i: abs(p[1] - self.slot_y(i)))
            if abs(p[1] - self.slot_y(slot)) < self.slot_hit:
                return ("slot", slot)
            return ("panel", None)
        if self._inside(p, self.panel_rect, self.px(10)):
            if abs(p[1] - self.switch_y) < self.px(SWITCH_HALF_LEN + SWITCH_R + 12):
                return ("switch", None)
            if self.slider_y0 - self.px(20) <= p[1] <= self.slider_y1 + self.px(20):
                frac = (self.slider_y1 - p[1]) / (self.slider_y1 - self.slider_y0)
                return ("size", float(np.clip(frac, 0.0, 1.0)))
            if abs(p[1] - self.clear_y) < self.px(CLEAR_HALF_H + 6):
                return ("clear", None)
            return ("panel", None)
        return None

# Brush stroke smoothing. Each stroke point is replaced by a Gaussian-weighted average
# of its neighbours on BOTH sides, which removes hand tremor without the lag a normal
# (one-sided) filter has. A point can only be finalised once STROKE_SMOOTH_RADIUS newer
# points exist, so the newest bit of the stroke is drawn as a live preview that follows
# the finger and is replaced by the smoothed ink a moment later.
STROKE_SMOOTH_RADIUS = 2       # neighbours on each side (detections, ~25 ms apart)
STROKE_SMOOTH_SIGMA = 1.6      # Gaussian width in points; higher = smoother, rounder corners
SUBPIXEL_BITS = 4              # strokes are drawn with 1/16 px precision (cv2 "shift")
_STROKE_WEIGHTS = np.exp(-0.5 * (np.arange(-STROKE_SMOOTH_RADIUS, STROKE_SMOOTH_RADIUS + 1)
                                 / STROKE_SMOOTH_SIGMA) ** 2)


def subpixel(points) -> np.ndarray:
    """Float points -> int32 points for cv2 drawing with SUBPIXEL_BITS of precision."""
    return np.round(np.asarray(points, np.float64) * (1 << SUBPIXEL_BITS)).astype(np.int32)


def quad_curve(a: np.ndarray, c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Points along the quadratic Bezier a -> b with control point c (~3 px apart)."""
    length = float(np.hypot(*(c - a)) + np.hypot(*(b - c)))
    t = np.linspace(0.0, 1.0, max(2, int(length / 3) + 2))[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


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


def draw_hand_landmarks(frame: np.ndarray, pts: np.ndarray, layout: "Layout") -> None:
    ipts = [(int(x), int(y)) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, ipts[a], ipts[b], (40, 220, 80), layout.px(2), cv2.LINE_AA)
    for point in ipts:
        cv2.circle(frame, point, layout.px(4), (40, 70, 255), cv2.FILLED, cv2.LINE_AA)


class Hand:
    __slots__ = ("label", "pts", "fist")

    def __init__(self, label: str, pts: np.ndarray, fist: bool) -> None:
        self.label, self.pts, self.fist = label, pts, fist


# --------------------------------------------------------------------------
# Threaded camera
# --------------------------------------------------------------------------
def open_camera(index: int, w: int, h: int) -> Optional[cv2.VideoCapture]:
    """Open and configure the camera; None if it cannot deliver a frame."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    # MJPG usually unlocks 30 fps at 1080p (raw YUYV at 1080p is often capped at 5 fps).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def find_cameras() -> list[tuple[int, str]]:
    """(index, device name) of every camera. The index already selects the backend the
    names came from (e.g. 700 = DirectShow camera 0), so it can go straight to VideoCapture."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        backend = {"win32": cv2.CAP_DSHOW, "linux": cv2.CAP_V4L2}.get(sys.platform, cv2.CAP_ANY)
        cameras = [(c.backend + c.index, c.name) for c in enumerate_cameras(backend)]
        if cameras:
            return cameras
        print("Could not read camera names on this system, showing numbers instead.")
    except ImportError:
        print("Camera names need the cv2_enumerate_cameras package: pip install -r requirements.txt")
    except Exception as e:
        print(f"Could not read camera names ({e}), showing numbers instead.")
    return [(index, f"Camera {index}") for index in probe_cameras()]


def probe_cameras(max_index: int = 6) -> list[int]:
    """Indices that deliver a frame; fallback when cv2_enumerate_cameras is missing."""
    try:
        previous = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # missing indices warn
    except AttributeError:
        previous = None
    found = []
    try:
        for index in range(max_index):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(index)
            cap.release()
    finally:
        if previous is not None:
            cv2.utils.logging.setLogLevel(previous)
    return found


def ask_camera() -> Optional[int]:
    """Ask in the terminal which camera to use; None if no camera was found."""
    print("Looking for cameras...")
    cameras = find_cameras()
    if not cameras:
        print("No camera found.")
        return None
    for number, (_, name) in enumerate(cameras, 1):
        print(f"  [{number}] {name}")
    while True:
        try:
            answer = input("Which camera do you want to use? [1]: ").strip()
        except EOFError:
            answer = ""
        if not answer:
            answer = "1"
        if answer.isdigit() and 1 <= int(answer) <= len(cameras):
            index, name = cameras[int(answer) - 1]
            print(f"Using {name}.")
            return index
        print(f"Please enter a number from 1 to {len(cameras)}.")


class CameraStream:
    """Grabs frames in a background thread so the main loop never blocks on I/O.
    Frames are resized to the preview size and mirrored here, off the main thread.
    Published frames are shared between threads, so treat them as read-only.
    If the camera stops delivering (cable pulled, capture card hiccup), it is
    reopened every second until it works again."""

    def __init__(self, cap: cv2.VideoCapture, size: tuple[int, int], reopen):
        self.cap: Optional[cv2.VideoCapture] = cap
        self.size = size
        self.reopen = reopen              # () -> Optional[VideoCapture]
        self.cond = threading.Condition()
        self.frame: Optional[np.ndarray] = None
        self.seq = 0
        self.connected = True
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.running:
            if self.cap is None:
                time.sleep(1.0)
                self.cap = self.reopen()
                if self.cap is not None:
                    print("Camera reconnected")
                continue
            ok, frame = self.cap.read()
            if not ok:
                print("Camera lost, reconnecting...")
                self.connected = False
                self.cap.release()
                self.cap = None
                continue
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)
            frame = cv2.flip(frame, 1)
            with self.cond:
                self.frame = frame
                self.seq += 1
                self.connected = True
                self.cond.notify_all()

    def read(self, after: int, timeout: float = 1.0) -> tuple[int, Optional[np.ndarray]]:
        """Wait for a frame newer than sequence number ``after`` (older frames are
        dropped). Returns (seq, frame), or (after, None) on timeout."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != after or not self.running, timeout)
            if self.seq == after:
                return after, None
            return self.seq, self.frame

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


def detector_process(conn, model_path: str, use_gpu: bool, num_hands: int = 2) -> None:
    """Child process: runs MediaPipe hand detection. It lives in its own process
    because MediaPipe needs the GIL to hand work between its threads, and sharing
    the GIL with the display loop made detection several times slower.
    While fewer than num_hands are tracked, MediaPipe re-runs palm detection on every
    frame to look for the missing one, so num_hands=1 roughly halves detection time."""
    def create(delegate):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path, delegate=delegate),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=0.60, min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        return vision.HandLandmarker.create_from_options(options)

    try:
        try:
            landmarker = create(python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU)
            delegate = "GPU" if use_gpu else "CPU"
        except Exception as error:
            if not use_gpu:
                raise
            print(f"MediaPipe GPU delegate unavailable ({error}); using CPU delegate")
            landmarker = create(python.BaseOptions.Delegate.CPU)
            delegate = "CPU (GPU failed)"
    except BaseException as error:
        conn.send(("error", repr(error)))
        return
    conn.send(("ready", delegate))
    with landmarker:
        while True:
            message = conn.recv()
            if message is None:
                break
            timestamp_ms, rgb = message
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
            # Tasks labels assume a mirrored/selfie input, which is exactly what we
            # supply; keep the user's natural Left/Right labels. Points are 0..1.
            conn.send(("hands", [
                (handed[0].category_name, np.array([(q.x, q.y) for q in lm], np.float32))
                for lm, handed in zip(result.hand_landmarks, result.handedness)
            ]))


def default_detectors() -> int:
    """MediaPipe's Python CPU path runs one detection on a single core, so several
    detector processes working on alternate frames multiply detections per second
    (measured: near-linear up to 4 on a 12-core Snapdragon X). Leave cores for the
    camera and display threads."""
    cores = os.cpu_count() or 1
    return 3 if cores >= 8 else 2 if cores >= 4 else 1


class HandTracker:
    """Feeds the newest camera frames to detector processes from background
    threads, so the display loop never waits for detection (the slowest step).
    Each detector has its own thread; a thread claims a frame no other detector
    has taken, and only results for frames newer than the last shown are kept."""

    def __init__(self, cam: CameraStream, model_path: str, acceleration: "Acceleration",
                 det_size: tuple[int, int], w: int, h: int, detectors: int = 1, num_hands: int = 2):
        self.cam, self.acceleration = cam, acceleration
        self.det_size, self.scale = det_size, np.array([w, h], np.float32)
        self.lock = threading.Lock()
        self.hands: list[Hand] = []
        self.infer_ms = 0.0
        self.seq = 0                      # bumps once per finished detection
        self.claimed = 0                  # newest camera frame handed to a detector
        self.shown = 0                    # camera frame of the published hands
        self.error: Optional[BaseException] = None
        use_gpu = acceleration.media_pipe_delegate == python.BaseOptions.Delegate.GPU
        self.workers: list[tuple] = []
        for _ in range(max(1, detectors)):
            conn, child_conn = mp_process.Pipe()
            process = mp_process.Process(target=detector_process, daemon=True,
                                         args=(child_conn, model_path, use_gpu, num_hands))
            process.start()
            self.workers.append((conn, process))
        for conn, _ in self.workers:      # wait until every model is loaded
            kind, detail = conn.recv()
            if kind == "error":
                self.stop_processes()
                raise RuntimeError(f"Hand detector failed to start: {detail}")
        print(f"MediaPipe: {detail}, {len(self.workers)} detector process(es)")
        self.running = True
        self.threads = [threading.Thread(target=self._run, args=(conn,), daemon=True)
                        for conn, _ in self.workers]
        for thread in self.threads:
            thread.start()

    def _run(self, conn) -> None:
        timestamp_ms = 0
        try:
            while self.running:
                frame_seq, frame = self.cam.read(self.claimed)
                if frame is None:
                    continue
                with self.lock:
                    if frame_seq <= self.claimed:   # another detector took it
                        continue
                    self.claimed = frame_seq
                t0 = time.perf_counter()
                rgb = self.acceleration.prepare(frame, self.det_size)
                timestamp_ms = max(timestamp_ms + 1, int(time.monotonic() * 1000))
                conn.send((timestamp_ms, rgb))
                _, found = conn.recv()
                hands = []
                for label, norm in found:
                    pts = norm * self.scale
                    hands.append(Hand(label, pts, is_fist(pts)))  # fist computed once
                with self.lock:
                    if frame_seq > self.shown:      # a later frame may finish first
                        self.hands, self.seq, self.shown = hands, self.seq + 1, frame_seq
                        self.infer_ms = (time.perf_counter() - t0) * 1000.0
        except BaseException as error:  # surfaced in the main loop
            self.error = error

    def latest(self) -> tuple[int, list[Hand], float]:
        with self.lock:
            return self.seq, self.hands, self.infer_ms

    def stop(self) -> None:
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2.0)
        self.stop_processes()

    def stop_processes(self) -> None:
        for conn, _ in self.workers:
            try:
                conn.send(None)
            except OSError:
                pass
        for _, process in self.workers:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

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
        """MediaPipe's GPU delegate (OpenGL) is tried unless --gpu off; the detector
        falls back to CPU if it cannot start. It does not depend on OpenCL/CUDA.

        The MediaPipe pip wheels for Windows are compiled with GPU disabled
        (MEDIAPIPE_DISABLE_GPU), so the GPU delegate can never start there.
        Skip it instead of failing and falling back every launch."""
        if sys.platform == "win32" or self.mode == "off":
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
        self.window_detections = 0
        self.detect_fps = 0.0     # finished hand detections per second (own thread)

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
            self.detect_fps = self.window_detections / elapsed
            self.worst = self.window_worst
            self.window_start, self.window_frames, self.window_worst = now, 0, 0.0
            self.window_detections = 0

    def detection(self) -> None:
        self.window_detections += 1


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


class Dwell:
    """A pinky-held button: fires once after resting on it for ``seconds``, then the
    pinky must leave before it can fire again, so hovering never repeats it."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.progress = 0.0           # 0..1, for the UI
        self._since: Optional[float] = None
        self._armed = True

    def update(self, over: bool) -> bool:
        """Call once per detection; True on the update the button fires."""
        if not over:
            self._since, self._armed, self.progress = None, True, 0.0
            return False
        if not self._armed:
            return False
        now = time.monotonic()
        if self._since is None:
            self._since = now
        held = now - self._since
        self.progress = min(1.0, held / self.seconds)
        if held < self.seconds:
            return False
        self._since, self._armed, self.progress = None, False, 0.0
        return True


class Pen:
    """Stroke state for one hand, so two people can draw at the same time. Tool,
    color and size are shared (on the Painter); a stroke keeps the color and width
    it started with, so changing them never recolors someone's stroke halfway."""

    def __init__(self, min_cutoff: float, beta: float) -> None:
        self.cursor = OneEuroCursor(min_cutoff, beta)
        self.pos: Optional[np.ndarray] = None       # last palm position, to recognise the hand
        self.vel = np.zeros(2)                      # palm movement per detection (smoothed)
        self.previous: Optional[tuple[int, int]] = None  # last point; None = no stroke running
        self.tip: Optional[tuple[int, int]] = None  # current pen/eraser tip (for the ring marker)
        self.tip_radius = 6
        self.color = PALETTE[0]
        self.width = 10
        # Brush stroke in progress: filtered finger points, the smoothed points already
        # turned into ink, and where that ink currently ends (see STROKE_SMOOTH_RADIUS).
        self.stroke_raw: list[np.ndarray] = []
        self.stroke_smooth: list[np.ndarray] = []
        self.stroke_end: Optional[np.ndarray] = None
        self.eraser_level = 1     # committed finger count, 1..4
        self._level_pending = 1
        self._level_votes = 0
        self.eraser_diam = 0.0    # eased diameter actually used

    @property
    def active(self) -> bool:
        return self.previous is not None


class Painter:
    def __init__(self, w: int, h: int, min_cutoff: float = 1.0, beta: float = 0.025):
        self.w, self.h = w, h
        self.canvas = np.zeros((h, w, 3), np.uint8)
        self.mask = np.zeros((h, w), np.uint8)
        # Bounding box (x0, y0, x1, y1) that contains all ink; None when empty.
        # Compositing only touches this region.
        self.ink_box: Optional[tuple[int, int, int, int]] = None
        self._erased = False
        self.tool = "draw"        # "draw" or "erase"
        self.color = PALETTE[0]
        self.brush = 10
        self.eraser_size = 48     # base diameter in px (index finger only)
        self.bg_on = False                          # background switch (beige or background.png)
        self.switch = Dwell(SWITCH_DWELL)           # BG switch (pinky held on it)
        self.clear_button = Dwell(CLEAR_DWELL)      # Clear button (pinky held on it)
        self.pens = [Pen(min_cutoff, beta), Pen(min_cutoff, beta)]  # up to two hands draw at once
        self.status, self.status_until = "Draw", 0.0

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

    def clear(self) -> None:
        self.canvas.fill(0)
        self.mask.fill(0)
        self.ink_box = None
        for pen in self.pens:  # strokes in progress restart from scratch
            pen.stroke_raw, pen.stroke_smooth, pen.stroke_end = [], [], None
            pen.previous = None
            pen.cursor.reset()
        self.set_status("CLEAR", 1.0)

    # ---- hands -> pens ----
    def assign_pens(self, hands: list[Hand]) -> list[tuple[Pen, Hand]]:
        """Match this detection's hands to pens by position, so each person keeps their
        own stroke even though both may be labelled "Right". Pens without a hand end
        their stroke; a hand that appears far from a pen's last spot starts fresh."""
        jump = PEN_JUMP * self.w
        centres = [(hand.pts[WRIST] + hand.pts[MIDDLE_MCP]) / 2 for hand in hands]

        def cost(pen: Pen, centre: np.ndarray) -> float:
            # Compare with where the hand is heading, so two hands that cross keep their pens.
            return jump if pen.pos is None else min(distance(pen.pos + pen.vel, centre), 2 * jump)

        # At most two hands and two pens: try both ways round, keep the cheaper one.
        best, best_cost = None, float("inf")
        for order in ((0, 1), (1, 0)):
            pairs = list(zip(order, range(len(hands))))
            total = sum(cost(self.pens[p], centres[i]) for p, i in pairs)
            if total < best_cost:
                best, best_cost = pairs, total
        matched = []
        used = set()
        for p, i in best or []:
            pen = self.pens[p]
            if pen.pos is not None and distance(pen.pos + pen.vel, centres[i]) > jump:
                self.end_stroke(pen)  # a different hand: do not join the two with a line
                pen.pos = None
            pen.vel = np.zeros(2) if pen.pos is None else 0.5 * pen.vel + 0.5 * (centres[i] - pen.pos)
            pen.pos = centres[i]
            matched.append((pen, hands[i]))
            used.add(p)
        for p, pen in enumerate(self.pens):
            if p not in used:
                self.end_stroke(pen)
                pen.pos, pen.vel = None, np.zeros(2)
        return matched

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
        self.set_status("BACKGROUND ON" if self.bg_on else "CAMERA BACKGROUND", 1.0)

    def eraser_target(self, level: int = 1) -> float:
        """Eraser diameter in px for a raised-finger count."""
        return self.eraser_size * ERASER_SCALES[int(np.clip(level, 1, len(ERASER_SCALES))) - 1]

    @staticmethod
    def observe_eraser_level(pen: Pen, raised: int) -> int:
        """Feed this frame's raised-finger count; returns the committed level.

        A change only counts after LEVEL_CONFIRM_FRAMES identical frames, so a
        finger flickering for one frame does not make the eraser jump.
        """
        raised = int(np.clip(raised, 1, len(ERASER_SCALES)))
        if raised == pen.eraser_level:
            pen._level_votes = 0
        elif raised == pen._level_pending:
            pen._level_votes += 1
            if pen._level_votes >= LEVEL_CONFIRM_FRAMES:
                pen.eraser_level, pen._level_votes = raised, 0
        else:
            pen._level_pending, pen._level_votes = raised, 1
            if LEVEL_CONFIRM_FRAMES <= 1:
                pen.eraser_level, pen._level_votes = raised, 0
        return pen.eraser_level

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

    # ---- smoothed brush strokes ----
    @staticmethod
    def _smoothed(pen: Pen, i: int) -> np.ndarray:
        """Stroke point i averaged with its neighbours on both sides (fewer at the ends)."""
        r = STROKE_SMOOTH_RADIUS
        lo, hi = max(0, i - r), min(len(pen.stroke_raw), i + r + 1)
        weights = _STROKE_WEIGHTS[lo - i + r:hi - i + r]
        return (np.asarray(pen.stroke_raw[lo:hi]) * weights[:, None]).sum(axis=0) / weights.sum()

    def _ink(self, pen: Pen, points: np.ndarray) -> None:
        """Draw a polyline of float points into the canvas and mask (anti-aliased, subpixel)."""
        pts = subpixel(points)
        cv2.polylines(self.canvas, [pts], False, pen.color, pen.width, cv2.LINE_AA, SUBPIXEL_BITS)
        cv2.polylines(self.mask, [pts], False, 255, pen.width, cv2.LINE_AA, SUBPIXEL_BITS)
        lo, hi = points.min(axis=0), points.max(axis=0)
        self._grow_box((int(lo[0]), int(lo[1])), (int(hi[0]) + 1, int(hi[1]) + 1), pen.width)

    def _commit(self, pen: Pen, point: np.ndarray) -> None:
        """Turn the next smoothed point into ink. The ink runs through the midpoints
        between smoothed points with quadratic curves, so it has no corners."""
        pen.stroke_smooth.append(point)
        if len(pen.stroke_smooth) == 1:
            pen.stroke_end = point
            self._ink(pen, np.array([point, point]))  # a tap still leaves a dot
            return
        control = pen.stroke_smooth[-2]
        end = (control + point) / 2
        self._ink(pen, quad_curve(pen.stroke_end, control, end))
        pen.stroke_end = end

    def _brush_step(self, pen: Pen, point: np.ndarray) -> None:
        pen.stroke_raw.append(point)
        # Point i is final once it has STROKE_SMOOTH_RADIUS newer neighbours.
        while len(pen.stroke_smooth) < len(pen.stroke_raw) - STROKE_SMOOTH_RADIUS:
            self._commit(pen, self._smoothed(pen, len(pen.stroke_smooth)))

    def _finish_brush_stroke(self, pen: Pen) -> None:
        """Pen up: smooth the remaining points with what is known and ink them."""
        while len(pen.stroke_smooth) < len(pen.stroke_raw):
            self._commit(pen, self._smoothed(pen, len(pen.stroke_smooth)))
        if len(pen.stroke_smooth) >= 2:
            # Finish where the finger stopped: the last averages lean back into the stroke.
            self._ink(pen, quad_curve(pen.stroke_end, pen.stroke_smooth[-1], pen.stroke_raw[-1]))
        pen.stroke_raw, pen.stroke_smooth, pen.stroke_end = [], [], None

    def draw_pending(self, frame: np.ndarray) -> None:
        """Live preview of the not-yet-smoothed end of each stroke, from the end of the
        ink to the finger, so the brush never lags behind (drawn on the frame only)."""
        for pen in self.pens:
            if pen.stroke_end is None or len(pen.stroke_raw) <= len(pen.stroke_smooth):
                continue
            tail = [pen.stroke_end, pen.stroke_smooth[-1]] + pen.stroke_raw[len(pen.stroke_smooth):]
            cv2.polylines(frame, [subpixel(tail)], False, pen.color, pen.width, cv2.LINE_AA, SUBPIXEL_BITS)

    # ---- drawing ----
    def draw(self, pen: Pen, raw: np.ndarray, eraser_level: int = 1) -> None:
        """Extend the hand's current stroke with the active tool (brush or eraser)."""
        erase = self.erasing
        p = pen.cursor.update(raw, time.perf_counter())
        current = (int(round(float(p[0]))), int(round(float(p[1]))))
        if pen.previous is None:
            # The stroke keeps the color and width it starts with.
            pen.color, pen.width = self.color, self.brush
            if erase:  # start the stroke at the right size instead of easing from a stale one
                pen.eraser_diam = self.eraser_target(eraser_level)
        elif erase:
            # The eraser eases toward the size for the current finger count, so it
            # grows and shrinks smoothly instead of snapping.
            target = self.eraser_target(eraser_level)
            pen.eraser_diam += (target - pen.eraser_diam) * ERASER_EASING
            if abs(target - pen.eraser_diam) < 0.5:
                pen.eraser_diam = target
            width = max(1, int(round(pen.eraser_diam)))
            # No anti-aliasing here: AA would leave partially-erased mask
            # pixels (a ghost outline) and costs more on a wide eraser.
            cv2.line(self.canvas, pen.previous, current, (0, 0, 0), width, cv2.LINE_8)
            cv2.line(self.mask, pen.previous, current, 0, width, cv2.LINE_8)
            self._erased = True
        if not erase:  # the brush draws through the smoothing pipeline (also the first point)
            self._brush_step(pen, np.asarray(p, np.float64))
        pen.previous = current
        pen.tip = current
        pen.tip_radius = max(1, int(round(pen.eraser_diam)) // 2) if erase else max(4, pen.width // 2 + 3)

    def end_stroke(self, pen: Pen) -> None:
        if pen.stroke_raw:
            self._finish_brush_stroke(pen)
        pen.previous = None
        pen.tip = None
        pen.cursor.reset()
        if self._erased and not any(other.active for other in self.pens if other is not pen):
            # Shrink the ink box again after erasing (once nobody is mid-stroke).
            self._erased = False
            self._recompute_box()

    def composite(self, frame: np.ndarray) -> None:
        """Paint the ink (and the live stroke preview) onto ``frame`` in place,
        touching only the ink's bounding box."""
        if self.ink_box is not None:
            x0, y0, x1, y1 = self.ink_box
            dst = frame[y0:y1, x0:x1]
            src = self.canvas[y0:y1, x0:x1]
            m = self.mask[y0:y1, x0:x1]
            if INK_OPACITY >= 0.999:
                cv2.copyTo(src, m, dst)
            else:
                blended = cv2.addWeighted(dst, 1.0 - INK_OPACITY, src, INK_OPACITY, 0)
                cv2.copyTo(blended, m, dst)
        self.draw_pending(frame)


def eraser_step(painter: Painter, pen: Pen, pts: np.ndarray) -> int:
    """Erase for one frame. Size comes from the raised fingers (index is already up):
    index = smallest, +middle, +ring, +pinky = biggest. The eraser is centred on the
    raised fingertips, so it covers what your fingers point at. Returns the level."""
    raised = 1 + sum(finger_extended(pts, *f) for f in FINGERS[1:])
    level = painter.observe_eraser_level(pen, raised)
    tips = np.asarray([pts[f[2]] for f in FINGERS[:level]], dtype=np.float32)
    painter.draw(pen, tips.mean(axis=0), level)
    return level


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text_shadow(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1) -> None:
    """Text with a dark outline so it stays readable on the camera and on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def shade_box(img: np.ndarray, rect: tuple[int, int, int, int], radius: int, alpha: float = 0.55) -> None:
    """Darken a rounded rectangle in place (a see-through shadow box behind text)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, rect[0]), max(0, rect[1]), min(w, rect[2]), min(h, rect[3])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    bw, bh = x1 - x0, y1 - y0
    r = max(0, min(radius, bw // 2, bh // 2))
    mask = np.zeros((bh, bw), np.uint8)
    cv2.rectangle(mask, (r, 0), (bw - 1 - r, bh - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (bw - 1, bh - 1 - r), 255, -1)
    for cx, cy in ((r, r), (bw - 1 - r, r), (r, bh - 1 - r), (bw - 1 - r, bh - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    cv2.copyTo(cv2.convertScaleAbs(roi, alpha=1.0 - alpha), mask, roi)


def text_box(img: np.ndarray, lines: list[tuple[str, tuple]], x: int, y: int,
             layout: "Layout", scale: float, thickness: int, center: bool = False,
             from_bottom: bool = False) -> int:
    """Draw lines of (text, color) on a shadow box whose top edge is at ``y`` (bottom
    edge with ``from_bottom``). ``x`` is the left edge, or the centre when ``center``
    is set; centred boxes also centre each line. Returns the opposite edge's y."""
    pad, gap = layout.px(10), layout.px(7)
    sizes = [cv2.getTextSize(text, FONT, scale, thickness) for text, _ in lines]
    box_w = max(tw for (tw, _), _ in sizes) + 2 * pad
    line_h = max(th + base for (_, th), base in sizes)
    box_h = len(lines) * line_h + (len(lines) - 1) * gap + 2 * pad
    x0 = x - box_w // 2 if center else x
    top = y - box_h if from_bottom else y
    shade_box(img, (x0, top, x0 + box_w, top + box_h), layout.px(8))
    ty = top + pad
    for (text, color), ((tw, th), _) in zip(lines, sizes):
        tx = x0 + (box_w - tw) // 2 if center else x0 + pad
        cv2.putText(img, text, (tx, ty + th), FONT, scale, color, thickness, cv2.LINE_AA)
        ty += line_h + gap
    return top if from_bottom else top + box_h


def draw_switch(frame: np.ndarray, painter: Painter, layout: Layout) -> None:
    """Vertical pill switch for the background page (top of the right panel);
    the knob is up when the page (beige or background.png) is on."""
    px = layout.px
    cx, cy, r = layout.panel_x, layout.switch_y, px(SWITCH_R)
    y0, y1 = cy - px(SWITCH_HALF_LEN), cy + px(SWITCH_HALF_LEN)
    track = (110, 190, 100) if painter.bg_on else (90, 90, 90)
    cv2.circle(frame, (cx, y0), r, track, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, y1), r, track, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (cx - r, y0), (cx + r, y1), track, -1)
    cv2.circle(frame, (cx, y0 if painter.bg_on else y1), r - px(4), (245, 245, 245), -1, cv2.LINE_AA)
    label_y = y1 + r + px(26)
    (tw, _), _ = cv2.getTextSize("BG", FONT, layout.font(0.6), px(1))
    cv2.putText(frame, "BG", (cx - tw // 2, label_y), FONT, layout.font(0.6), (235, 235, 235),
                px(1), cv2.LINE_AA)
    if painter.switch.progress > 0:  # dwell progress bar under the label
        half = px(PANEL_W) // 2 - px(14)
        end = cx - half + int(2 * half * painter.switch.progress)
        cv2.line(frame, (cx - half, label_y + px(12)), (end, label_y + px(12)), (255, 255, 255), px(4))


def draw_clear_button(frame: np.ndarray, painter: Painter, layout: Layout) -> None:
    """Clear button (bottom of the right panel); fills up while the pinky is held on it."""
    px = layout.px
    cx, cy = layout.panel_x, layout.clear_y
    half_w, half_h = px(PANEL_W) // 2 - px(12), px(CLEAR_HALF_H)
    x0, y0, x1, y1 = cx - half_w, cy - half_h, cx + half_w, cy + half_h
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 150), -1)
    if painter.clear_button.progress > 0:
        fill_x = x0 + int((x1 - x0) * painter.clear_button.progress)
        cv2.rectangle(frame, (x0, y0), (fill_x, y1), (60, 60, 230), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 230), px(1), cv2.LINE_AA)
    label = "HOLD" if painter.clear_button.progress > 0 else "CLEAR"
    (tw, th), _ = cv2.getTextSize(label, FONT, layout.font(0.6), px(2))
    cv2.putText(frame, label, (cx - tw // 2, cy + th // 2), FONT, layout.font(0.6), (255, 255, 255),
                px(2), cv2.LINE_AA)


def draw_ui(frame: np.ndarray, painter: Painter, layout: Layout, stats: PerfStats, show_stats: bool,
            warnings: list[str]) -> None:
    h, w = frame.shape[:2]
    px = layout.px

    # ---- left, vertically centred: color column, eraser at the bottom ----
    x0, y0, x1, y1 = layout.col_rect
    cx = layout.column_x
    cv2.rectangle(frame, (x0, y0), (x1, y1), (25, 25, 25), -1)
    for i, col in enumerate(PALETTE):
        cy = layout.slot_y(i)
        cv2.circle(frame, (cx, cy), px(SLOT_R), col, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), px(SLOT_R), (110, 110, 110), px(1), cv2.LINE_AA)  # keeps black visible
        if not painter.erasing and col == painter.color:
            cv2.circle(frame, (cx, cy), px(SLOT_R + 6), (255, 255, 255), px(3), cv2.LINE_AA)
    ey = layout.slot_y(ERASER_SLOT)
    ew, eh = px(25), px(17)
    cv2.rectangle(frame, (cx - ew, ey - eh), (cx + ew, ey + eh), (225, 225, 225), -1)
    cv2.rectangle(frame, (cx - ew, ey - eh), (cx - px(4), ey + eh), (170, 120, 255), -1)
    cv2.rectangle(frame, (cx - ew, ey - eh), (cx + ew, ey + eh), (90, 90, 90), px(1), cv2.LINE_AA)
    if painter.erasing:
        cv2.rectangle(frame, (cx - ew - px(7), ey - eh - px(7)), (cx + ew + px(7), ey + eh + px(7)),
                      (255, 255, 255), px(3), cv2.LINE_AA)

    # ---- right, vertically centred: background switch, size slider below ----
    px0, py0, px1, py1 = layout.panel_rect
    cv2.rectangle(frame, (px0, py0), (px1, py1), (25, 25, 25), -1)
    draw_switch(frame, painter, layout)
    draw_clear_button(frame, painter, layout)
    sx, sy0, sy1, sw = layout.panel_x, layout.slider_y0, layout.slider_y1, px(SLIDER_HALF_W)
    cv2.rectangle(frame, (sx - sw, sy0), (sx + sw, sy1), (75, 75, 75), -1)
    ky = layout.slider_y(painter.size_fraction())
    fill = (170, 170, 170) if painter.erasing else painter.color
    cv2.rectangle(frame, (sx - sw, ky), (sx + sw, sy1), fill, -1)
    cv2.circle(frame, (sx, ky), px(KNOB_R), (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(frame, (sx, ky), px(KNOB_R), (60, 60, 60), px(1), cv2.LINE_AA)
    if painter.erasing:
        lines = ["Eraser", f"{painter.eraser_target(1):.0f}-{painter.eraser_target(len(ERASER_SCALES)):.0f}px"]
    else:
        lines = ["Brush", f"{painter.brush}px"]
    ty = sy1 + px(KNOB_R) + px(20)
    for line in lines:  # centred under the slider
        (tw, _), _ = cv2.getTextSize(line, FONT, layout.font(0.5), px(1))
        put_text_shadow(frame, line, (sx - tw // 2, ty), layout.font(0.5), thickness=px(1))
        ty += px(20)

    # ---- status line, top centre (on a shadow box) ----
    idle = ("Erase: index up, more fingers = bigger | Pinky on panels: tools, size, BG & clear"
            if painter.erasing else "Draw: index finger | Pinky on panels: colors, eraser, size, BG & clear")
    text = painter.status if time.monotonic() < painter.status_until else idle
    text_box(frame, [(text, (255, 255, 255))], w // 2, px(14), layout,
             layout.font(0.62), px(2), center=True)

    # ---- stats, bottom centre (on a shadow box, lines centred) ----
    above = h - px(14)  # warnings go above the stats box
    if show_stats:
        fps_color = (80, 255, 80) if stats.fps >= 24 else (0, 200, 255) if stats.fps >= 15 else (60, 60, 255)
        white = (255, 255, 255)
        above = text_box(frame, [
            (f"FPS: {stats.fps:.1f}", fps_color),
            (f"Frame: {stats.frametime:.1f} ms (worst {stats.worst:.0f})", white),
            (f"Detect: {stats.infer:.1f} ms ({stats.detect_fps:.0f}/s)", white),
            (f"Paint: {stats.paint:.1f} ms", white),
        ], w // 2, above, layout, layout.font(0.55), px(1), center=True, from_bottom=True) - px(10)

    # ---- warnings (camera lost, too dark), bottom centre above the stats ----
    if warnings:
        lines = [(text, (0, 200, 255)) for text in warnings]
        text_box(frame, lines, w // 2, above, layout, layout.font(0.7), px(2), center=True, from_bottom=True)


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
    ap.add_argument("--detectors", type=int, default=default_detectors(),
                    help="hand detector processes working on alternate frames. More = more "
                         "detections per second on many-core CPUs (default: by core count).")
    ap.add_argument("--hands", type=int, choices=(1, 2), default=2,
                    help="hands to track. 1 = one person drawing alone, about twice as fast "
                         "detection; 2 = two hands / two people.")
    ap.add_argument("--min-cutoff", type=float, default=1.0,
                    help="pen smoothing when the finger is still. Lower = steadier, higher = more responsive.")
    ap.add_argument("--beta", type=float, default=0.025,
                    help="how quickly smoothing relaxes as you move. Higher = less lag on fast strokes.")
    ap.add_argument("--gpu", choices=("auto", "on", "off"), default="auto",
                    help="GPU acceleration mode: auto detects CUDA/OpenCL and tries MediaPipe's GPU "
                         "delegate (not on Windows), on requires CUDA/OpenCL, off disables both.")
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

    # OpenCV's own thread pool (resize, flip, ...) would otherwise use every core in
    # bursts and starve the detector process; 2 threads measured fastest overall.
    cv2.setNumThreads(2)

    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w, h = args.width, args.height
    # If the camera delivers a different size, every frame is resized to the preview size.
    resized = (cam_w, cam_h) != (w, h)
    print(f"Camera: {cam_w}x{cam_h}  Preview: {w}x{h}" + ("  (resized)" if resized else ""))

    if args.detect_width and args.detect_width < w:
        det_w = args.detect_width
        det_h = max(1, round(h * det_w / w))
    else:
        det_w, det_h = w, h

    painter = Painter(w, h, args.min_cutoff, args.beta)
    enable_dpi_awareness()  # before the window exists, so sizes are real screen pixels
    screen_h = screen_height()
    view_w, view_h, view_f = w, h, 1.0  # size the frame is shown at, and frame -> view factor
    layout = Layout(w, h, ui_scale(screen_h, h))
    stats = PerfStats()
    show_stats = True
    window_title = "Hand Paint - S save | C clear | B background | [ ] size | F stats | Q quit"
    # WINDOW_NORMAL lets the view scale when you resize or maximize it.
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    initial_w = min(1600, max(960, w))
    cv2.resizeWindow(window_title, initial_w, max(540, round(initial_w * h / w)))
    cam = CameraStream(cap, (w, h), lambda: open_camera(args.camera, w, h))
    # Page shown instead of the camera when the switch is on (background.png or beige).
    bg_img, bg_light = load_background(w, h)
    try:
        with HandTracker(cam, args.model, acceleration, (det_w, det_h), w, h,
                         args.detectors, args.hands) as tracker:
            frame_seq = det_seq = 0
            hands: list[Hand] = []
            ui_points: list[tuple[int, int]] = []
            brightness = 128.0    # smoothed camera brightness, for the "too dark" warning
            frame = None
            while True:
                frame_seq, new_frame = cam.read(frame_seq, timeout=0.25)
                if tracker.error is not None:
                    raise tracker.error
                if new_frame is None:
                    if frame is None or cam.connected:
                        continue
                    # Camera lost: keep the window alive (and Q working) while it reconnects.
                    view = cv2.resize(cv2.convertScaleAbs(frame, alpha=0.35), (view_w, view_h))
                    draw_ui(view, painter, layout, stats, show_stats,
                            ["Camera disconnected - reconnecting..."])
                    cv2.imshow(window_title, view)
                    if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                        break
                    continue
                frame = new_frame
                t1 = time.perf_counter()

                # The UI lives in window pixels; rebuild it when the window is resized.
                size = display_size(window_title, w, h)
                if size != (view_w, view_h):
                    view_w, view_h = size
                    view_f = view_w / w
                    layout = Layout(view_w, view_h, ui_scale(screen_h, view_h))

                # ---- gestures: run once per finished detection, not once per frame ----
                # Detection runs in its own thread; the display uses its newest result.
                seq, latest_hands, infer_ms = tracker.latest()
                if seq != det_seq:
                    det_seq, hands = seq, latest_hands
                    stats.detection()
                    # Cheap brightness estimate on every 16th pixel, smoothed over ~1 s.
                    brightness = 0.95 * brightness + 0.05 * float(frame[::16, ::16].mean())
                    over_switch = over_clear = False
                    ui_points = []  # pinky positions while on a panel (drawn as rings)
                    # Every hand is its own pen, so two people can draw at once.
                    for pen, hand in painter.assign_pens(hands):
                        pts = hand.pts
                        pinky = pts[PINKY_TIP] * view_f  # the panels are in window pixels
                        # Panels only react when this hand is not mid-stroke.
                        zone = layout.hit(pinky) if not pen.active else None
                        if hand.fist:
                            painter.end_stroke(pen)  # a fist is "pen up"
                        elif zone is not None:
                            # Pinky on a panel: colors/eraser (left column), BG switch,
                            # size slider or Clear (right). A stroke in progress is never
                            # interrupted, so drawing near the panels stays safe.
                            painter.end_stroke(pen)
                            ui_points.append((int(pinky[0]), int(pinky[1])))
                            kind, value = zone
                            if kind == "switch":
                                over_switch = True
                            elif kind == "clear":
                                over_clear = True
                            elif kind == "slot":
                                painter.select_slot(value)
                            elif kind == "size":
                                painter.set_size_fraction(value)
                        elif finger_extended(pts, INDEX_MCP, INDEX_PIP, INDEX_TIP):
                            # The active tool works exclusively while the index is extended.
                            if painter.erasing:
                                level = eraser_step(painter, pen, pts)
                                painter.set_status(f"ERASE {level}/{len(ERASER_SCALES)}")
                            else:
                                painter.draw(pen, pts[INDEX_TIP])
                                painter.set_status("DRAW: index extended")
                        else:
                            painter.end_stroke(pen)
                    if painter.switch.update(over_switch):
                        painter.toggle_bg()
                    if painter.clear_button.update(over_clear):
                        painter.clear()

                # ---- background + ink (on a private copy: the camera frame is shared
                # with the detection thread, so it is never drawn on) ----
                out = bg_img.copy() if painter.bg_on else frame.copy()
                # Compositing only touches the ink's bounding box.
                painter.composite(out)
                # Resize to the window now, so everything drawn below is sharp on screen.
                view = out if (view_w, view_h) == (w, h) else cv2.resize(
                    out, (view_w, view_h), interpolation=cv2.INTER_LINEAR)
                # The skeleton, eraser radius and pen tip are drawn last, so paint never hides them.
                for hand in hands:
                    draw_hand_landmarks(view, hand.pts * view_f, layout)
                # Light rings disappear on a light page, so use dark ones there.
                dark_ring = painter.bg_on and bg_light
                for pen in painter.pens:
                    if pen.tip is None:
                        continue
                    tip = (int(pen.tip[0] * view_f), int(pen.tip[1] * view_f))
                    radius = max(1, int(pen.tip_radius * view_f))
                    if painter.erasing:  # eraser outline shows exactly what will be wiped
                        ring = (70, 70, 70) if dark_ring else (210, 210, 210)
                        cv2.circle(view, tip, radius, ring, layout.px(2), cv2.LINE_AA)
                    else:
                        ring = (50, 50, 50) if dark_ring else (255, 255, 255)
                        cv2.circle(view, tip, radius, ring, layout.px(1), cv2.LINE_AA)
                paint_ms = (time.perf_counter() - t1) * 1000.0

                stats.tick(infer_ms, paint_ms)
                warnings = ["Too dark - hand tracking needs more light"] if brightness < LOW_LIGHT else []
                draw_ui(view, painter, layout, stats, show_stats, warnings)
                for point in ui_points:  # show where each pinky is on the panels
                    cv2.circle(view, point, layout.px(10), (255, 255, 255), layout.px(2), cv2.LINE_AA)
                    cv2.circle(view, point, layout.px(12), (0, 0, 0), layout.px(1), cv2.LINE_AA)
                cv2.imshow(window_title, view)
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
                    # Save what you see, without the UI: the drawing on the background
                    # page, or on the current camera frame (full preview resolution).
                    page = bg_img.copy() if painter.bg_on else frame.copy()
                    painter.composite(page)
                    cv2.imwrite(name, page)
                    painter.set_status(f"SAVED {name}", 1.5)
    finally:
        cam.stop()
        cv2.destroyAllWindows()


CRASH_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log")


def log_crash(text: str) -> None:
    with open(CRASH_LOG, "a", encoding="utf-8") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{text}\n")


def run_app() -> None:
    """Run the app once; Python errors are written to crash.log (and still raised),
    native crashes (inside OpenCV/MediaPipe) via faulthandler."""
    import faulthandler
    import traceback
    fault_log = open(CRASH_LOG, "a", encoding="utf-8")
    faulthandler.enable(fault_log)
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        log_crash(traceback.format_exc())
        raise


def supervise() -> None:
    """Run the app in a child process and restart it when it crashes, so it keeps
    running unattended. Quitting with Q/Esc ends it. Gives up after 5 crashes in a
    minute (e.g. no camera plugged in), so a broken setup does not loop forever."""
    import subprocess
    command = [sys.executable, os.path.abspath(__file__), "--worker", *sys.argv[1:]]
    crashes: list[float] = []
    while True:
        try:
            code = subprocess.call(command)
        except KeyboardInterrupt:
            return
        if code == 0:
            return
        now = time.monotonic()
        crashes = [t for t in crashes if now - t < 60] + [now]
        if len(crashes) >= 5:
            log_crash(f"App exited with code {code}; 5 crashes within a minute, giving up.")
            print("Crashed 5 times within a minute, giving up. See crash.log.")
            return
        log_crash(f"App exited with code {code}; restarting.")
        print(f"App crashed (exit code {code}), restarting in 2 s. Details in crash.log")
        time.sleep(2)


if __name__ == "__main__":
    # Default: supervised (auto-restart). --worker is the supervised child;
    # --no-restart runs the app directly, e.g. while developing.
    # The camera is asked for once here (not in the worker), so restarts reuse the choice.
    if ("--worker" not in sys.argv and sys.stdin.isatty()
            and not any(a == "--camera" or a.startswith("--camera=") for a in sys.argv)):
        try:
            chosen = ask_camera()
        except KeyboardInterrupt:
            sys.exit(0)
        if chosen is not None:
            sys.argv += ["--camera", str(chosen)]
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        run_app()
    elif "--no-restart" in sys.argv:
        sys.argv.remove("--no-restart")
        run_app()
    else:
        supervise()
