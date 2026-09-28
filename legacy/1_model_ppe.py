import cv2
import time
import threading
from queue import Queue, Empty, Full
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# Input / Output
VIDEO_PATH = Path(__file__).parent / "media" / "cctv_test.mp4"
MODEL_PATH = "best_yolo11_ncnn_model"  # Path to exported NCNN directory
CROPS_DIR = Path("worker_crops")
CROPS_DIR.mkdir(exist_ok=True)

# Performance & Display
HEADLESS = False            # Set to True on headless Pi (saves X11/CPU cycles)
TARGET_WIDTH = 640          # Camera stream frame width
TARGET_HEIGHT = 480         # Camera stream frame height
INFERENCE_SIZE = 416        # MUST match your NCNN export resolution to prevent tracker crashes

# PPE Logic & Thresholds
TRACK_CONF = 0.25           # Tracker threshold (prevents noise/0-pixel boxes from crashing Kalman filter)
PPE_CONF_THRESH = 0.10      # Base detector threshold for PPE inside the person box
VEST_THRESH = 0.25          # Acceptance cutoff for safety vests
HELMET_MARGIN = 0.20        # Margin: (helmet_conf - no_helmet_conf) > HELMET_MARGIN
IOA_THRESHOLD = 0.55        # Intersection-over-Area required to associate PPE to person

# Throttling & Violations
SAVE_EVERY_SECONDS = 0.5    # Evaluation interval per tracked worker (saves CPU)
VIOLATION_SECONDS = 2.0     # Time worker must continuously violate rules before alerting
CROP_SAVE_COOLDOWN = 3.0    # Cooldown between consecutive crop saves for same ID

# Class IDs (Must match your trained dataset)
PERSON_ID = 6
HELMET_ID = 0
VEST_ID = 2
NO_HELMET_ID = 7

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def calculate_ioa(person_box, ppe_box):
    """
    Calculates Intersection over Area (IoA) relative to the PPE bounding box.
    Returns the percentage (0.0 to 1.0) of the PPE box that falls inside the person box.
    """
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = ppe_box

    ix1, iy1 = max(px1, hx1), max(py1, hy1)
    ix2, iy2 = min(px2, hx2), min(py2, hy2)

    if ix2 < ix1 or iy2 < iy1:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    ppe_area = (hx2 - hx1) * (hy2 - hy1)

    return 0.0 if ppe_area == 0 else intersection_area / ppe_area


def evaluate_person_ppe(person_xyxy, ppe_dets):
    """
    Associates PPE detections with a person box using IoA and applies margin logic.
    Returns: {"helmet": bool, "vest": bool} or None if no valid PPE detected.
    """
    best = {HELMET_ID: 0.0, NO_HELMET_ID: 0.0, VEST_ID: 0.0}

    for d in ppe_dets:
        cls_id = d["cls"]
        conf = d["conf"]
        ppe_box = d["xyxy"]

        # Discard detections that don't overlap sufficiently with the person
        if conf >= PPE_CONF_THRESH and calculate_ioa(person_xyxy, ppe_box) >= IOA_THRESHOLD:
            if conf > best.get(cls_id, 0.0):
                best[cls_id] = conf

    helmet_conf = best[HELMET_ID]
    nohelmet_conf = best[NO_HELMET_ID]
    vest_conf = best[VEST_ID]

    # No PPE evidence detected for this worker
    if helmet_conf == 0.0 and nohelmet_conf == 0.0 and vest_conf == 0.0:
        return None

    # Helmet Decision (Margin Competition)
    if helmet_conf == 0.0 and nohelmet_conf == 0.0:
        helmet = False
    else:
        helmet = (helmet_conf - nohelmet_conf) > HELMET_MARGIN

    # Vest Decision
    vest = vest_conf >= VEST_THRESH

    return {"helmet": helmet, "vest": vest}

# ==============================================================================
# PIPELINE CLASS
# ==============================================================================

class EdgePPEPipeline:
    def __init__(self, source, model_path):
        self.source = str(source)
        self.model = YOLO(model_path, task="detect")
        
        # Thread communication queues
        self.frame_queue = Queue(maxsize=3)  # Limits RAM usage
        self.io_queue = Queue(maxsize=20)    # Buffers disk writes

        # State tracking
        self.running = False
        self.last_checked = {}      # {worker_id: timestamp}
        self.last_status = {}       # {worker_id: {"helmet": bool, "vest": bool}}
        self.violation_since = {}   # {worker_id: timestamp}
        self.last_crop_saved = {}   # {worker_id: timestamp}

    def _camera_worker(self):
        """Thread 1: Dedicated frame capture and downscaling."""
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Guarantee fixed resolution for memory consistency
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

            try:
                self.frame_queue.put(frame, timeout=1.0)
            except Full:
                continue

        cap.release()

    def _io_worker(self):
        """Thread 3: Asynchronous non-blocking disk writes."""
        while self.running or not self.io_queue.empty():
            try:
                task = self.io_queue.get(timeout=0.5)
                file_path, image = task
                cv2.imwrite(str(file_path), image)
                self.io_queue.task_done()
            except Empty:
                continue

    def run(self):
        """Thread 2 (Main): Inference, Tracking, State Logic, and UI."""
        self.running = True

        cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        io_thread = threading.Thread(target=self._io_worker, daemon=True)
        cam_thread.start()
        io_thread.start()

        print(f"[INFO] Pipeline active. Target: {MODEL_PATH} | Headless: {HEADLESS}")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                if not cam_thread.is_alive():
                    self.running = False
                    break
                continue

            now = time.time()

            # YOLO11 NCNN Tracking
            # Switched to botsort to prevent singular matrix errors on edge devices
            results = self.model.track(
                frame,
                persist=True,
                tracker="botsort.yaml", 
                conf=TRACK_CONF,
                imgsz=INFERENCE_SIZE,
                classes=[PERSON_ID, HELMET_ID, VEST_ID, NO_HELMET_ID],
                verbose=False
            )

            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                if not HEADLESS:
                    cv2.imshow("PPE Pipeline (Pi 4)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                continue

            # Extract arrays
            xyxy_all = res.boxes.xyxy.cpu().numpy()
            cls_all = res.boxes.cls.cpu().numpy().astype(int)
            conf_all = res.boxes.conf.cpu().numpy()
            ids_all = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None

            # Separate PPE items (allow lower confidence boxes for evaluation)
            ppe_dets = [
                {"cls": cls_all[i], "conf": float(conf_all[i]), "xyxy": xyxy_all[i]}
                for i in range(len(cls_all)) if cls_all[i] != PERSON_ID
            ]

            if ids_all is not None:
                h, w = frame.shape[:2]

                for i in range(len(cls_all)):
                    if cls_all[i] != PERSON_ID:
                        continue

                    worker_id = int(ids_all[i])
                    x1, y1, x2, y2 = map(int, xyxy_all[i])

                    # Clamp coordinates to frame boundaries to prevent crash
                    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                    x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Periodic evaluation check
                    should_update = (
                        (worker_id not in self.last_checked) or 
                        ((now - self.last_checked[worker_id]) >= SAVE_EVERY_SECONDS)
                    )

                    if should_update:
                        self.last_checked[worker_id] = now
                        status = evaluate_person_ppe((x1, y1, x2, y2), ppe_dets)

                        if status is not None:
                            self.last_status[worker_id] = status
                            noncompliant = (not status["helmet"]) or (not status["vest"])

                            if noncompliant:
                                if worker_id not in self.violation_since:
                                    self.violation_since[worker_id] = now

                                # Check continuous violation threshold
                                if (now - self.violation_since[worker_id]) >= VIOLATION_SECONDS:
                                    last_save = self.last_crop_saved.get(worker_id, 0.0)
                                    if (now - last_save) >= CROP_SAVE_COOLDOWN:
                                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        out_path = CROPS_DIR / f"violation_id{worker_id}_{ts}.jpg"
                                        crop = frame[y1:y2, x1:x2].copy()

                                        # Offload I/O without blocking pipeline
                                        self.io_queue.put((out_path, crop))
                                        self.last_crop_saved[worker_id] = now
                            else:
                                self.violation_since.pop(worker_id, None)

                    # UI Drawing (Only if display is active)
                    if not HEADLESS:
                        ppe = self.last_status.get(worker_id)
                        if ppe is None:
                            color, label = (0, 255, 255), f"ID {worker_id} Scanning..."
                        else:
                            ok = ppe["helmet"] and ppe["vest"]
                            color = (0, 255, 0) if ok else (0, 0, 255)
                            h_str = "H:OK" if ppe["helmet"] else "H:NO"
                            v_str = "V:OK" if ppe["vest"] else "V:NO"
                            label = f"ID {worker_id} [{h_str} {v_str}]"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(20, y1 - 8)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if not HEADLESS:
                cv2.imshow("PPE Pipeline (Pi 4)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

            # Minimal yield to keep CPU thermals and scheduler balanced
            time.sleep(0.001)

        # Teardown
        if not HEADLESS:
            cv2.destroyAllWindows()
        cam_thread.join()
        io_thread.join()
        print("[INFO] Pipeline shut down cleanly.")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    pipeline = EdgePPEPipeline(source=VIDEO_PATH, model_path=MODEL_PATH)
    pipeline.run()