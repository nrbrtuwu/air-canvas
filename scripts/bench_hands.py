"""Benchmark hand detection backends on this machine.

Times the palm and landmark models on every backend that is installed (Qualcomm
NPU, Intel NPU / GPU / CPU via OpenVINO, plain ONNX Runtime CPU), checks that each
gives the same results as the CPU, then times the whole pipeline per frame next
to MediaPipe. Run it from the repo root, with the venv you run the app with:

    python scripts\\bench_hands.py               (synthetic test image)
    python scripts\\bench_hands.py --camera 0    (a real frame; hold a hand up)

Intel NPU needs:  pip install onnxruntime-openvino openvino
(and Intel's NPU driver; Task Manager shows "NPU" under Performance when it's there).
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import npu_hands  # noqa: E402


def timed(fn, n: int) -> tuple[float, float]:
    for _ in range(10):
        fn()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 90))


def test_frame(camera: int | None) -> np.ndarray:
    """RGB frame at the app's detection size (400 wide)."""
    if camera is not None:
        cap = cv2.VideoCapture(camera)
        frame = None
        for _ in range(15):          # let exposure settle
            ok, got = cap.read()
            frame = got if ok else frame
        cap.release()
        if frame is None:
            sys.exit(f"Camera {camera} delivered no frame")
        frame = cv2.flip(frame, 1)
    else:  # smooth noise: makes the palm detector do real work
        rng = np.random.default_rng(0)
        frame = cv2.GaussianBlur(rng.integers(0, 255, (1080, 1920, 3), np.uint8), (0, 0), 3)
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (400, round(h * 400 / w)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None, help="use a frame from this camera")
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    print(f"{platform.processor() or platform.machine()} | Python {platform.python_version()} "
          f"({platform.machine()}) | OpenVINO devices: {npu_hands.openvino_devices() or 'none'}")
    palm_path, lm_path = npu_hands.default_models()
    rgb = test_frame(args.camera)
    palm_x = npu_hands.NpuHandLandmarker._tensor(cv2.resize(rgb, (192, 192)))
    lm_x = npu_hands.NpuHandLandmarker._tensor(cv2.resize(rgb, (224, 224)))

    # ---- the two models on their own, per backend ----
    print("\nModel speed (ms per run, mean / p90) and difference from CPU results:")
    reference, usable = None, []
    for backend in npu_hands.BACKENDS:
        try:
            start = time.perf_counter()
            palm, desc = npu_hands.create_session(palm_path, backend)
            lm, _ = npu_hands.create_session(lm_path, backend)
            load = time.perf_counter() - start
        except Exception as error:
            print(f"  {backend:13} not available ({str(error).splitlines()[0][:90]})")
            continue
        pi, li = palm.get_inputs()[0].name, lm.get_inputs()[0].name
        p_mean, p90 = timed(lambda: palm.run(None, {pi: palm_x}), args.runs)
        l_mean, l90 = timed(lambda: lm.run(None, {li: lm_x}), args.runs)
        outputs = (palm.run(None, {pi: palm_x}), lm.run(None, {li: lm_x}))
        if backend == "cpu":
            reference = outputs
        usable.append((backend, desc, outputs))
        print(f"  {backend:13} palm {p_mean:6.2f} / {p90:6.2f}   landmarks {l_mean:6.2f} / {l90:6.2f}"
              f"   (load {load:.1f}s, {desc})")
    if reference is not None:
        for backend, _, (p_out, l_out) in usable:
            if backend == "cpu":
                continue
            palm_diff = max(float(np.abs(a - b).max()) for a, b in zip(p_out, reference[0]))
            lm_diff = float(np.abs(l_out[0] - reference[1][0]).max())   # landmark coords, px of 224
            verdict = "OK" if lm_diff < 2.0 else "CHECK: differs from CPU"
            print(f"  {backend:13} max difference: palm outputs {palm_diff:.3f}, "
                  f"landmarks {lm_diff:.2f} px  -> {verdict}")

    # ---- whole pipeline per frame, like the app ----
    print("\nWhole pipeline per frame (ms, mean / p90), 2 hands:")
    for backend, desc, _ in usable:
        tracker = npu_hands.NpuHandLandmarker(palm_path, lm_path, num_hands=2, device=backend)
        mean, p90 = timed(lambda: tracker.detect(rgb), args.runs)
        print(f"  {backend:13} {mean:6.2f} / {p90:6.2f}   hands found: {len(tracker.detect(rgb))}   ({desc})")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        task = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hand_landmarker.task")
        landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=task), running_mode=vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = [0]

        def detect():
            ts[0] += 33
            return landmarker.detect_for_video(image, ts[0])
        mean, p90 = timed(detect, args.runs)
        print(f"  {'mediapipe':13} {mean:6.2f} / {p90:6.2f}   hands found: {len(detect().hand_landmarks)}"
              f"   (what the app uses by default)")
    except ImportError:
        print("  mediapipe     not installed")
    print("\nSend me this whole output.")


if __name__ == "__main__":
    main()
