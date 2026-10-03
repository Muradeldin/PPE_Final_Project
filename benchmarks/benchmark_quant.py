"""
Speed benchmark: FP32 vs FP16 vs INT8 versions of the same YOLOv8n model, on the Raspberry Pi.

Times end-to-end inference (pre-processing + model + NMS, like the real pipeline) at 320 px on the same
frames of the test video, and prints a table plus one JSON line to paste back. Accuracy of the same files
was measured separately on the dataset's test split (benchmarks/accuracy_test.json).

Run on the Pi, from the project folder, with detection stopped (it would compete for the CPU):
    python benchmarks/benchmark_quant.py
"""
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "media" / "cctv_test.mp4"
IMGSZ = 320
WARMUP = 10
FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

MODELS = [
    ("NCNN FP32", ROOT / "models" / "quant" / "best_yolo8_fp32_ncnn_model"),
    ("NCNN FP16 (deployed)", ROOT / "models" / "best_yolo8_ncnn_model_half"),
    ("TFLite FP32", ROOT / "models" / "quant" / "best_yolo8_fp32.tflite"),
    ("TFLite FP16", ROOT / "models" / "quant" / "best_yolo8_fp16.tflite"),
    ("TFLite INT8 full PTQ", ROOT / "models" / "quant" / "best_yolo8_int8_full.tflite"),
]


def cpu_temp():
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except (OSError, ValueError):
        return None


def size_mb(path):
    if path.is_file():
        return round(path.stat().st_size / 1e6, 1)
    return round(sum(f.stat().st_size for f in path.iterdir() if f.is_file()) / 1e6, 1)


def load_frames(n):
    cap = cv2.VideoCapture(str(VIDEO))
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:                       # loop the short test video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(cv2.resize(frame, (640, 480)))
    cap.release()
    return frames


def tflite_shim():
    """Older Ultralytics versions look for `tflite_runtime`; the current runtime package is `ai-edge-litert`."""
    try:
        import tflite_runtime.interpreter  # noqa: F401
    except ImportError:
        try:
            import types
            from ai_edge_litert import interpreter
            sys.modules["tflite_runtime"] = types.ModuleType("tflite_runtime")
            sys.modules["tflite_runtime.interpreter"] = interpreter
        except ImportError:
            print("Note: TFLite runtime not installed (pip install ai-edge-litert) - TFLite rows will fail.\n")


def main():
    tflite_shim()
    from ultralytics import YOLO

    frames = load_frames(WARMUP + FRAMES)
    print(f"Device: {platform.machine()} | Python {platform.python_version()} | {os.cpu_count()} CPU cores | "
          f"{FRAMES} timed frames at {IMGSZ} px | CPU temp at start: {cpu_temp()} °C\n")

    results = []
    for name, path in MODELS:
        if not path.exists():
            print(f"{name:26} skipped: {path.name} not found"); continue
        try:
            model = YOLO(str(path), task="detect")
            for f in frames[:WARMUP]:
                model.predict(f, imgsz=IMGSZ, conf=0.10, verbose=False)
            times = []
            for f in frames[WARMUP:]:
                t = time.perf_counter()
                model.predict(f, imgsz=IMGSZ, conf=0.10, verbose=False)
                times.append((time.perf_counter() - t) * 1000)
            ms = statistics.median(times)
            row = dict(name=name, size_mb=size_mb(path), ms_median=round(ms, 1), fps=round(1000 / ms, 1),
                       ms_p90=round(sorted(times)[int(len(times) * 0.9)], 1), temp_after=cpu_temp())
            results.append(row)
            print(f"{name:26} {row['size_mb']:5} MB  {row['ms_median']:7.1f} ms/frame  {row['fps']:5.1f} FPS"
                  f"  (p90 {row['ms_p90']} ms, CPU {row['temp_after']} °C)", flush=True)
        except Exception as e:  # keep going if one format can't load on this device
            print(f"{name:26} FAILED: {type(e).__name__}: {e}"[:200], flush=True)

    print("\nCopy this line back:")
    print(json.dumps({"device": platform.machine(), "frames": FRAMES, "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
