# air-canvas

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
| `--camera N` | asked at start | Which camera to use. Without it, the app lists the cameras by name every time it starts and asks in the terminal |
| `--width`, `--height` | `1920`, `1080` | Preview / canvas size (try `1280 720` on slow machines) |
| `--engine auto\|npu\|mediapipe` | `auto` | Hand detection engine. `auto` uses the Snapdragon NPU when available (Windows on ARM setup), otherwise MediaPipe |
| `--detectors N` | `1` on the NPU, else `3` on 8+ cores | Hand detector processes working on alternate frames. Each detection uses one core, so more = more detections per second |
| `--hands 1\|2` | `2` | Hands to track. `1` (drawing alone) roughly halves detection time, so the pen lags less |
| `--gpu auto\|on\|off` | `auto` | GPU acceleration (MediaPipe GPU works on Linux only) |
| `--no-restart` | | Don't auto-restart after a crash (for development) |

By default the app restarts itself after a crash and writes the error to `crash.log`. It gives up after 5 crashes within a minute. If the camera disconnects, it reconnects on its own.

## Linux notes

- If the window doesn't open (Wayland): `QT_QPA_PLATFORM=xcb python camera.py`
- `sudo apt install python3-tk` lets the UI detect your screen size (otherwise it assumes 1080p).

## Windows on ARM (beta)

Runs natively on ARM64 Python (e.g. Snapdragon X laptops) instead of x64 emulation. MediaPipe ships an ARM64 wheel, but OpenCV doesn't, so it is compiled once (15-30 min). Needs git, an ARM64 Python 3.12 from python.org, and Visual Studio 2022 Build Tools with **Desktop development with C++**, **MSVC ARM64 build tools** and a **Windows 11 SDK**.

```powershell
py -3.12-arm64 -m venv .venv-arm64
.\scripts\build-opencv-arm64.ps1
.venv-arm64\Scripts\pip install -r requirements-arm64.txt
.venv-arm64\Scripts\pip install mediapipe --no-deps
.venv-arm64\Scripts\python camera.py
```

pip warns that mediapipe needs `opencv-contrib-python`. That's expected: the OpenCV built by the script replaces it.

### NPU hand detection

On Snapdragon, hand detection runs on the NPU by default (`--engine auto`): the same MediaPipe palm and landmark models, as ONNX files in `models/`, through ONNX Runtime's Qualcomm plugin (`npu_hands.py` reproduces MediaPipe's pipeline around them). Detection takes ~5 ms instead of ~25-45 ms on a CPU core, and its points stay within ~1 px of MediaPipe's. The first start takes a few seconds while the models are compiled for the NPU. `--engine mediapipe` switches back.

Virtual cameras that only ship an x64 driver (e.g. OBS Virtual Camera) show up in the list but can't be opened from ARM64. Use a real webcam, or the x64 setup for those.


## Disclaimer

- This project was made for a High School show-off day, so keep in mind that this project probably will not be updated in the future.
- **This project was mainly vibe-coded, so do not expect the best ux...**
