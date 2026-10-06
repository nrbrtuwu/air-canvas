# Camera Drawer

Draw in the air with your finger. A webcam tracks your hands (MediaPipe) and your index finger becomes a brush. Two people can draw at the same time.

## Setup

```bash
pip install -r requirements.txt
python camera.py
```

`hand_landmarker.task` (the hand-tracking model) must sit next to `camera.py`. It is included in the repo.

## How to draw

| Gesture | Does |
|---|---|
| Index finger up | Draw (or erase, with the eraser selected) |
| Eraser + more fingers up | Bigger eraser |
| Fist / index finger down | Pen up |
| Pinky on a color or the eraser (left) | Select it |
| Pinky on the slider (right) | Brush / eraser size (up = bigger) |
| Pinky held on **BG** | Toggle the background |
| Pinky held on **CLEAR** (~1 s) | Clear the canvas |

## Keys

| Key | Does |
|---|---|
| `S` | Save a screenshot (`drawing_<time>.png`, without the UI) |
| `C` | Clear |
| `B` | Toggle the background |
| `[` / `]` | Smaller / bigger brush or eraser |
| `F` | Show / hide stats |
| `Q` / `Esc` | Quit |

## Custom background

Put `background.png` (or `.jpg` / `.jpeg`) next to `camera.py` and restart. It replaces the beige page when BG is on. Without one, the beige page is used.

## Options

| Option | Default | |
|---|---|---|
| `--camera N` | `0` | Which camera to use |
| `--width`, `--height` | `1920`, `1080` | Preview / canvas size (try `1280 720` on slow machines) |
| `--gpu auto\|on\|off` | `auto` | GPU acceleration (MediaPipe GPU works on Linux only) |
| `--no-restart` | | Don't auto-restart after a crash (for development) |

By default the app restarts itself after a crash and writes the error to `crash.log`. It gives up after 5 crashes within a minute. If the camera disconnects, it reconnects on its own.

## Linux notes

- If the window doesn't open (Wayland): `QT_QPA_PLATFORM=xcb python camera.py`
- `sudo apt install python3-tk` lets the UI detect your screen size (otherwise it assumes 1080p).
