import cv2
import time
import threading
import yaml
from collections import deque
from queue import Queue, Empty, Full
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# Input / Output
VIDEO_PATH = Path(__file__).parent / "media" / "cctv_test_2.mp4"
MODEL_PATH = str(Path(__file__).parent / "models" / "best_yolo8_ncnn_model_half")  # Exported at imgsz=320
CROPS_DIR = Path("worker_crops")
CROPS_DIR.mkdir(exist_ok=True)

# Performance & Display
HEADLESS = False
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
INFERENCE_SIZE = 320

# PPE Logic & Thresholds
HELMET_CONF_THRESH = 0.25   # Min confidence for helmet / no-helmet boxes
VEST_THRESH = 0.15          # Lowered for challenging lighting & lime green vests
DETECT_CONF = 0.10          # Model cutoff; must be <= the thresholds above (0.10 also feeds ByteTrack's low-score matching)
HELMET_MARGIN = 0.20
IOA_THRESHOLD = 0.30        # Lowered to allow bulky vests to overhang outside the person box
TOP_EDGE_PX = 10            # Person box this close to the top edge -> head may be cut off
BOTTOM_EDGE_PX = 2          # Person box this close to the bottom edge -> body is cut off
VEST_MAX_BOTTOM = 0.80      # Worn vest ends within the top 80% of the person box; a vest held in the hands hangs lower
VEST_CENTER_X_RANGE = (0.10, 0.90)  # Vest centre must sit over this person's torso, not a neighbour's

# Timing & Throttling
EVAL_INTERVAL_SECONDS = 0.5
VIOLATION_WINDOW_SECONDS = 2.0  # Look-back window for deciding a violation is real
VIOLATION_RATIO = 0.6       # Share of checks in the window that must be non-compliant (tolerates flicker)
MIN_CHECKS = 3              # Checks needed in the window before an alert can fire
MIN_VIOLATION_SPAN = 1.5    # Seconds a worker must be observed before alerting (filters short-lived ghost boxes)
ALERT_COOLDOWN_SECONDS = 10.0  # Per-worker gap between alerts (prevents SD card spam)
STALE_WORKER_SECONDS = 30.0    # Forget workers not seen for this long (bounds memory)

# Class IDs (Must match your trained dataset)
PERSON_ID = 6
HELMET_ID = 0
VEST_ID = 2
NO_HELMET_ID = 7

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def calculate_ioa_and_position(person_box, ppe_box, ppe_class_id, feet_in_frame=True):
    """
    Calculates Intersection over Area using the minimum bounding area to handle
    edge-of-frame crops. Enforces spatial positioning to prevent 'helmet in hand'
    and 'vest in hand' bypasses, and vests borrowed from a neighbour.
    """
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = ppe_box

    ix1, iy1 = max(px1, hx1), max(py1, hy1)
    ix2, iy2 = min(px2, hx2), min(py2, hy2)

    # No overlap
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, False

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    ppe_area = (hx2 - hx1) * (hy2 - hy1)
    person_area = (px2 - px1) * (py2 - py1)

    # Use the smaller of the two areas to handle edge-of-frame crops (e.g., chest-up views)
    denominator = min(ppe_area, person_area)
    ioa = 0.0 if denominator <= 0 else intersection_area / denominator

    if ioa < IOA_THRESHOLD:
        return ioa, False

    person_height = py2 - py1
    person_width = px2 - px1
    if person_height <= 0 or person_width <= 0:
        return ioa, False

    if ppe_class_id in [HELMET_ID, NO_HELMET_ID]:
        ppe_center_y = (hy1 + hy2) / 2.0
        relative_y = (ppe_center_y - py1) / person_height
        valid_position = relative_y <= 0.35  # Helmet must be in top 35%
        return ioa, valid_position

    if ppe_class_id == VEST_ID:
        relative_x = ((hx1 + hx2) / 2.0 - px1) / person_width
        if not (VEST_CENTER_X_RANGE[0] <= relative_x <= VEST_CENTER_X_RANGE[1]):
            return ioa, False

        # Only judge how low the vest hangs when the whole body is visible
        relative_bottom = (hy2 - py1) / person_height
        if feet_in_frame and relative_bottom > VEST_MAX_BOTTOM:
            return ioa, False

    return ioa, True


def evaluate_person_ppe(person_xyxy, ppe_dets, head_in_frame=True, feet_in_frame=True):
    """
    Associates PPE detections with a person box using IoA and spatial logic.
    Returns {"helmet": True/False/None, "vest": bool}. helmet is None (unknown)
    only when the head may be cut off by the frame edge and there is no helmet
    evidence either way.
    """
    best = {HELMET_ID: 0.0, NO_HELMET_ID: 0.0, VEST_ID: 0.0}

    for d in ppe_dets:
        cls_id = d["cls"]
        conf = d["conf"]
        ppe_box = d["xyxy"]

        min_conf = VEST_THRESH if cls_id == VEST_ID else HELMET_CONF_THRESH
        if conf >= min_conf:
            ioa, valid_pos = calculate_ioa_and_position(person_xyxy, ppe_box, cls_id, feet_in_frame)
            if valid_pos and conf > best.get(cls_id, 0.0):
                best[cls_id] = conf

    helmet_conf = best[HELMET_ID]
    nohelmet_conf = best[NO_HELMET_ID]
    vest_conf = best[VEST_ID]

    # Helmet Decision (no helmet evidence counts as a fail, unless the head is off-camera)
    if helmet_conf == 0.0 and nohelmet_conf == 0.0:
        helmet = False if head_in_frame else None
    else:
        helmet = (helmet_conf - nohelmet_conf) > HELMET_MARGIN

    # Vest Decision
    vest = vest_conf >= VEST_THRESH

    return {"helmet": helmet, "vest": vest}


def load_person_tracker():
    """ByteTrack instance that only ever sees person boxes, so PPE boxes skip tracking."""
    with open(check_yaml("bytetrack.yaml"), encoding="utf-8") as f:
        cfg = IterableSimpleNamespace(**yaml.safe_load(f))
    return BYTETracker(args=cfg)

# ==============================================================================
# PIPELINE CLASS
# ==============================================================================

class EdgePPEPipeline:
    def __init__(self, source, model_path):
        self.source = source if isinstance(source, int) else str(source)  # int = camera index
        self.model = YOLO(model_path, task="detect")
        self.tracker = load_person_tracker()

        self.frame_queue = Queue(maxsize=2)
        self.io_queue = Queue(maxsize=15)

        self.running = False
        self.last_checked = {}      # {worker_id: timestamp}
        self.last_status = {}       # {worker_id: {"helmet": bool/None, "vest": bool}}
        self.check_history = {}     # {worker_id: deque[(timestamp, helmet_bad, vest_bad)]}
        self.last_alert = {}        # {worker_id: {"no_helmet"/"no_vest": timestamp}}
        self.last_seen = {}         # {worker_id: timestamp}
        self.last_prune = 0.0

    def _camera_worker(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

        # Play video files at their real frame rate, like a live camera, so the time-based
        # alert rules behave the same on a fast PC as on the Pi
        is_file = isinstance(self.source, str) and Path(self.source).is_file()
        video_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
        start = time.time()
        frame_idx = 0

        while self.running and cap.isOpened():
            if video_fps > 0:
                time.sleep(max(0.0, start + frame_idx / video_fps - time.time()))  # wait until this frame is due
                while (time.time() - start) * video_fps > frame_idx + 1 and cap.grab():  # too slow: skip frames
                    frame_idx += 1

            ret, frame = cap.read()
            frame_idx += 1
            if not ret:
                break

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

            try:
                self.frame_queue.put(frame, timeout=0.5)
            except Full:
                continue

        cap.release()

    def _io_worker(self):
        while self.running or not self.io_queue.empty():
            try:
                task = self.io_queue.get(timeout=0.5)
                file_path, image = task
                cv2.imwrite(str(file_path), image)
                self.io_queue.task_done()
            except Empty:
                continue

    def _forget_stale_workers(self, now):
        """Drops state for track IDs ByteTrack has discarded, so the dicts don't grow forever."""
        if (now - self.last_prune) < 5.0:
            return
        self.last_prune = now

        stale = [wid for wid, t in self.last_seen.items() if (now - t) >= STALE_WORKER_SECONDS]
        for wid in stale:
            for state in (self.last_checked, self.last_status, self.check_history,
                          self.last_alert, self.last_seen):
                state.pop(wid, None)

    def run(self):
        self.running = True

        cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        io_thread = threading.Thread(target=self._io_worker, daemon=True)
        cam_thread.start()
        io_thread.start()

        print(f"[INFO] Pipeline active on {self.source}")
        print(f"[INFO] Model: {MODEL_PATH} | Headless: {HEADLESS} | ImgSz: {INFERENCE_SIZE}")

        prev_time = time.time()
        fps = 0.0
        last_print_time = time.time()

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                if not cam_thread.is_alive():
                    self.running = False
                    break
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            if HEADLESS and (now - last_print_time) >= 2.0:
                print(f"[INFO] Pipeline running at {fps:.1f} FPS")
                last_print_time = now

            # Detect everything at a low cutoff; per-class thresholds are applied later.
            # (model.track() would only return boxes that became tracks, dropping weak PPE boxes.)
            res = self.model.predict(
                frame,
                conf=DETECT_CONF,
                imgsz=INFERENCE_SIZE,
                classes=[PERSON_ID, HELMET_ID, VEST_ID, NO_HELMET_ID],
                verbose=False
            )[0]

            boxes = res.boxes.cpu().numpy()
            is_person = boxes.cls.astype(int) == PERSON_ID

            # Only people are tracked. Update every frame, even when empty, so lost tracks age out.
            tracks = self.tracker.update(boxes[is_person], frame)

            # Collect PPE detections
            ppe_boxes = boxes[~is_person]
            ppe_dets = [
                {"cls": int(c), "conf": float(s), "xyxy": b}
                for c, s, b in zip(ppe_boxes.cls, ppe_boxes.conf, ppe_boxes.xyxy)
            ]

            h, w = frame.shape[:2]

            for track in tracks:
                worker_id = int(track[4])
                x1, y1, x2, y2 = map(int, track[:4])

                # Clamp to frame dimensions
                x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                self.last_seen[worker_id] = now

                should_update = (
                    (worker_id not in self.last_checked) or
                    ((now - self.last_checked[worker_id]) >= EVAL_INTERVAL_SECONDS)
                )

                if should_update:
                    self.last_checked[worker_id] = now
                    status = evaluate_person_ppe(
                        (x1, y1, x2, y2), ppe_dets,
                        head_in_frame=(y1 > TOP_EDGE_PX),
                        feet_in_frame=(y2 < h - BOTTOM_EDGE_PX)
                    )

                    # Head off-camera with no helmet evidence: keep the last known helmet state
                    if status["helmet"] is None:
                        prev = self.last_status.get(worker_id)
                        status["helmet"] = prev["helmet"] if prev else None

                    self.last_status[worker_id] = status
                    noncompliant = (status["helmet"] is False) or (not status["vest"])

                    history = self.check_history.setdefault(worker_id, deque())
                    history.append((now, status["helmet"] is False, not status["vest"]))
                    while (now - history[0][0]) > VIOLATION_WINDOW_SECONDS:
                        history.popleft()

                    if noncompliant:
                        # Alert only when most recent checks agree on an item
                        n_checks = len(history)
                        observed_long_enough = n_checks >= MIN_CHECKS and (now - history[0][0]) >= MIN_VIOLATION_SPAN
                        reasons = []
                        if sum(c[1] for c in history) / n_checks >= VIOLATION_RATIO: reasons.append("no_helmet")
                        if sum(c[2] for c in history) / n_checks >= VIOLATION_RATIO: reasons.append("no_vest")

                        # Cooldown is per worker AND per reason, so a new kind of violation still alerts
                        alerted = self.last_alert.setdefault(worker_id, {})
                        new_reason = any((now - alerted.get(r, 0.0)) >= ALERT_COOLDOWN_SECONDS for r in reasons)

                        if observed_long_enough and new_reason:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                            out_path = CROPS_DIR / f"worker_{worker_id}_{'_'.join(reasons)}_{ts}.jpg"
                            crop = frame[y1:y2, x1:x2].copy()

                            try:
                                self.io_queue.put_nowait((out_path, crop))
                                alerted.update({r: now for r in reasons})
                                print(f"[ALERT] Violation! Crop saved to: {out_path.name}")
                            except Full:
                                pass

                # UI Rendering
                if not HEADLESS:
                    ppe = self.last_status.get(worker_id)
                    if ppe is None:
                        color, label = (0, 255, 255), f"ID {worker_id} Scanning..."
                    else:
                        if ppe["helmet"] is False or not ppe["vest"]:
                            color = (0, 0, 255)
                        elif ppe["helmet"] is None:
                            color = (0, 255, 255)  # Vest OK, helmet not visible yet
                        else:
                            color = (0, 255, 0)
                        h_str = {True: "H:OK", False: "H:NO", None: "H:?"}[ppe["helmet"]]
                        v_str = "V:OK" if ppe["vest"] else "V:NO"
                        label = f"ID {worker_id} [{h_str} {v_str}]"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            self._forget_stale_workers(now)

            if not HEADLESS:
                cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Edge PPE Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

            time.sleep(0.001)

        if not HEADLESS:
            cv2.destroyAllWindows()
        cam_thread.join(timeout=1.0)
        io_thread.join(timeout=2.0)
        print("[INFO] Pipeline shut down cleanly.")

if __name__ == "__main__":
    pipeline = EdgePPEPipeline(source=VIDEO_PATH, model_path=MODEL_PATH)
    pipeline.run()
