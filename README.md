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
| `--engine auto\|npu\|mediapipe` | `auto` | Hand detection engine (beta). `auto` uses an NPU when one is set up (see [NPU hand detection](#npu-hand-detection-beta)), otherwise MediaPipe |
| `--detectors N` | `1` on the NPU, else `3` on 8+ cores | Hand detector processes working on alternate frames. Each detection uses one core, so more = more detections per second |
| `--hands 1\|2` | `2` | Hands to track. `1` (drawing alone) roughly halves detection time, so the pen lags less |
| `--gpu auto\|on\|off` | `auto` | GPU acceleration (MediaPipe GPU works on Linux only) |
| `--no-restart` | | Don't auto-restart after a crash (for development) |

By default the app restarts itself after a crash and writes the error to `crash.log`. It gives up after 5 crashes within a minute. If the camera disconnects, it reconnects on its own.

## Linux notes

- If the window doesn't open (Wayland): `QT_QPA_PLATFORM=xcb python camera.py`
- `sudo apt install python3-tk` lets the UI detect your screen size (otherwise it assumes 1080p).

## NPU hand detection (beta)

If your PC has an NPU (the AI chip in Intel Core Ultra and Snapdragon X processors), hand detection can run on it instead of the CPU. It's the same MediaPipe hand models (ONNX copies in `models/`), so tracking quality stays the same, but it's much faster and leaves the CPU free:

| | Detect time in the app |
|---|---|
| MediaPipe on the CPU (default without an NPU) | ~25-45 ms |
| Intel NPU (Core Ultra 5 245KF) | ~8-10 ms |
| Qualcomm NPU (Snapdragon X Elite X1E-78-100) | ~5 ms |

Once set up, `python camera.py` uses the NPU on its own. Check the terminal line at start:

```
Hand detection: NPU pipeline (Intel NPU (OpenVINO)), 1 detector process(es)
```

If it says `Hand detection: CPU, ...` instead, it's running MediaPipe. `--engine npu` forces the NPU (and tells you what's missing if it can't), `--engine mediapipe` switches back. The first start takes a few extra seconds while the models are compiled for the NPU.

### Intel Core Ultra (x64 Windows)

Needs **Python 3.11-3.13** (the Intel package doesn't support 3.14 yet; `py install 3.13` or `winget install Python.Python.3.13` adds it next to 3.14) and Intel's NPU driver (Task Manager > Performance should show **NPU**; it comes with Windows Update or Intel's driver page).

```powershell
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-intel-npu.txt
python camera.py
```

`camera.py` switches to the `.venv` next to it by itself (on ARM64 it prefers `.venv-arm64`), so plain `python camera.py` uses the venv's Python and its NPU packages. Set `KAMERA_NO_VENV=1` to turn that off.

## *I have not tested it on any AMD CPU's that has an NPU, so results may vairy*

### Snapdragon X (Windows on ARM)

This runs natively on ARM64 Python instead of x64 emulation. MediaPipe ships an ARM64 wheel, but OpenCV doesn't, so it is compiled once (15-30 min). Needs git, an **ARM64 Python 3.12** from python.org, and Visual Studio 2022 Build Tools with **Desktop development with C++**, **MSVC ARM64 build tools** and a **Windows 11 SDK**.

```powershell
py -3.12-arm64 -m venv .venv-arm64
.\scripts\build-opencv-arm64.ps1
.venv-arm64\Scripts\pip install -r requirements-arm64.txt
.venv-arm64\Scripts\pip install mediapipe --no-deps
python camera.py
```

pip warns that mediapipe needs `opencv-contrib-python`. That's expected: the OpenCV built by the script replaces it. The NPU part (`onnxruntime-qnn`) is included in `requirements-arm64.txt`.

**OBS Virtual Camera on ARM64:** it shows up in the camera list but only opens once OBS's ARM64 driver is registered instead of the x64 one. Close OBS, then in an **admin** terminal, from OBS's `data\obs-plugins\win-dshow` folder (the ARM64 file is in the ARM64 build of OBS):

```powershell
regsvr32.exe /i /u obs-virtualcam-module64.dll
regsvr32.exe /i obs-virtualcam-module-arm64.dll
```

x64 apps can't use the OBS virtual camera while the ARM64 driver is registered ([OBS docs](https://obsproject.com/kb/windows-on-arm)).

### Benchmark

To see how every engine performs on your machine (hold a hand up for a real frame):

```powershell
.venv\Scripts\python scripts\bench_hands.py --camera 0
```


## Disclaimer

- This project was made for a High School show-off day, so keep in mind that this project probably will not be updated in the future.
- **This project was mainly vibe-coded, so do not expect the best ux...**
