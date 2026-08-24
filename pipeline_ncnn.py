import cv2
import time
import threading
from queue import Queue, Empty, Full
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------
VIDEO_PATH = Path(__file__).parent / "media" / "cctv_test.mp4"
MODEL_PATH = "best_yolo11_ncnn_model"
CROPS_DIR = Path("worker_crops")
CROPS_DIR.mkdir(exist_ok=True)
SAVE_CROPS = True

OUTPUT_VIDEO_PATH = Path("annotated_output.mp4")
SAVE_ANNOTATED_VIDEO = True

# Detection & Tracking Thresholds
PPE_CONF_THRESH = 0.10
HELMET_THRESH = 0.25          # confidence required for hardhat
VEST_THRESH = 0.45            # confidence required for safety vest
HELMET_MARGIN = 0.10          # helmet vs no-helmet margin

# Temporal constraints (seconds)
VIOLATION_SECONDS = 2.0       # time a violation must persist before alerting
CROP_SAVE_COOLDOWN = 3.0      # cooldown between saves for the same worker

# Dataset Class IDs
PERSON_ID = 6
HELMET_ID = 0
VEST_ID = 2
NO_HELMET_ID = 7
# -----------------------------------------------


def _box_center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def ppe_for_person(person_xyxy, ppe_dets, worker_id=None):
    """
    Evaluates PPE compliance for a specific person bounding box.
    Strictly checks that the PPE item is located on this worker's body.
    """
    px1, py1, px2, py2 = person_xyxy
    ph = max(1.0, py2 - py1)

    # Vertical body regions
    helmet_y1 = py1 - 0.15 * ph
    helmet_y2 = py1 + 0.42 * ph
    vest_y1 = py1 + 0.18 * ph
    vest_y2 = py1 + 0.78 * ph

    best_helmet_conf = 0.0
    best_nohelmet_conf = 0.0
    best_vest_conf = 0.0

    for d in ppe_dets:
        cls_id = d["cls"]
        conf = d["conf"]
        cx, cy = _box_center_xyxy(d["xyxy"])

        # Strict horizontal containment: center must be within person's x-bounds
        if not (px1 <= cx <= px2):
            continue

        # Helmet / No-helmet check
        if cls_id in (HELMET_ID, NO_HELMET_ID):
            if helmet_y1 <= cy <= helmet_y2:
                if cls_id == HELMET_ID and conf > best_helmet_conf:
                    best_helmet_conf = conf
                elif cls_id == NO_HELMET_ID and conf > best_nohelmet_conf:
                    best_nohelmet_conf = conf

        # Safety vest check (torso region)
        elif cls_id == VEST_ID:
            if vest_y1 <= cy <= vest_y2:
                if conf > best_vest_conf:
                    best_vest_conf = conf

    # Helmet Decision
    if best_helmet_conf >= HELMET_THRESH:
        helmet = (best_helmet_conf - best_nohelmet_conf) >= HELMET_MARGIN or best_nohelmet_conf == 0.0
    else:
        helmet = False

    # Vest Decision
    vest = best_vest_conf >= VEST_THRESH

    return {"helmet": helmet, "vest": vest, "h_conf": best_helmet_conf, "v_conf": best_vest_conf}


class MultiThreadedNCNNppePipeline:
    """
    High-Performance Multi-Threaded PPE Monitoring Pipeline.
    - Thread 1 (Camera): Continuously ingests frames to prevent I/O blocking.
    - Thread 2 (Main Inference): Runs NCNN YOLO11 tracking & compliance checks.
    - Thread 3 (Async Disk I/O): Silently saves violation snapshots in the background.
    """

    def __init__(self, source=VIDEO_PATH, model_path=MODEL_PATH):
        self.source = str(source)
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path, task="detect")

        # Thread-safe queues
        self.frame_queue = Queue(maxsize=3)   # keeps RAM low & paces video
        self.io_queue = Queue(maxsize=20)     # background disk saving queue

        # State management
        self.running = False
        self.last_status = {}
        self.violation_since = {}
        self.last_crop_saved = {}

    def _camera_worker(self):
        """Thread 1: Reads video stream without stalling AI inference."""
        cap = cv2.VideoCapture(self.source)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                self.frame_queue.put(frame, timeout=1.0)
            except Full:
                continue

        cap.release()

    def _io_worker(self):
        """Thread 3: Asynchronously writes violation crops to disk."""
        while self.running or not self.io_queue.empty():
            try:
                task = self.io_queue.get(timeout=0.5)
                file_path, image = task
                cv2.imwrite(str(file_path), image)
                self.io_queue.task_done()
            except Empty:
                continue

    def run(self):
        """Main Thread: Runs NCNN tracking, UI rendering, and video recording."""
        self.running = True

        # Start worker threads
        cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        io_thread = threading.Thread(target=self._io_worker, daemon=True)
        cam_thread.start()
        io_thread.start()

        print(f"[INFO] Multi-Threaded NCNN PPE Pipeline started on: {self.source}")
        print("[INFO] Press 'q' in the display window to exit cleanly.")

        writer = None
        prev_time = time.time()
        fps_display = 0.0

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                if not cam_thread.is_alive():
                    self.running = False
                    break
                continue

            now = time.time()

            # Calculate live FPS
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

            # Initialize video writer if recording
            if SAVE_ANNOTATED_VIDEO and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, 24.0, (w, h))

            # Run NCNN YOLO11 tracking
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=PPE_CONF_THRESH,
                imgsz=416,
                classes=[PERSON_ID, HELMET_ID, VEST_ID, NO_HELMET_ID],
                verbose=False
            )

            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy_all = res.boxes.xyxy.cpu().numpy()
                cls_all = res.boxes.cls.cpu().numpy().astype(int)
                conf_all = res.boxes.conf.cpu().numpy()
                ids_all = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None

                # Extract PPE items
                ppe_dets = []
                for i in range(len(cls_all)):
                    c = cls_all[i]
                    if c in (HELMET_ID, VEST_ID, NO_HELMET_ID):
                        ppe_dets.append({"cls": c, "conf": float(conf_all[i]), "xyxy": xyxy_all[i]})

                # Process each tracked person
                if ids_all is not None:
                    for i in range(len(cls_all)):
                        if cls_all[i] != PERSON_ID:
                            continue

                        worker_id = int(ids_all[i])
                        x1, y1, x2, y2 = map(int, xyxy_all[i])
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                        x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Evaluate compliance
                        status = ppe_for_person((x1, y1, x2, y2), ppe_dets, worker_id)
                        self.last_status[worker_id] = status

                        noncompliant = (not status["helmet"]) or (not status["vest"])

                        if noncompliant:
                            if worker_id not in self.violation_since:
                                self.violation_since[worker_id] = now

                            # If non-compliant longer than threshold, queue background snapshot
                            if (now - self.violation_since[worker_id]) >= VIOLATION_SECONDS:
                                if SAVE_CROPS:
                                    last_t = self.last_crop_saved.get(worker_id, 0.0)
                                    if (now - last_t) >= CROP_SAVE_COOLDOWN:
                                        crop = frame[y1:y2, x1:x2].copy()
                                        if crop.size != 0:
                                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            reasons = []
                                            if not status["helmet"]:
                                                reasons.append("no_helmet")
                                            if not status["vest"]:
                                                reasons.append("no_vest")
                                            out_path = CROPS_DIR / f"worker_{worker_id}_{'_'.join(reasons)}_{ts}.jpg"

                                            # Non-blocking async queue
                                            try:
                                                self.io_queue.put_nowait((out_path, crop))
                                                self.last_crop_saved[worker_id] = now
                                            except Full:
                                                pass
                        else:
                            self.violation_since.pop(worker_id, None)

                        # Render status graphics
                        ok = status["helmet"] and status["vest"]
                        color = (0, 255, 0) if ok else (0, 0, 255)
                        label = f"ID {worker_id}, H:{status['helmet']} V:{status['vest']}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw live FPS on top-left corner
            cv2.putText(frame, f"FPS: {fps_display:.1f} (NCNN Multi-Threaded)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            cv2.imshow("NCNN Edge PPE Monitoring Pipeline", frame)
            if SAVE_ANNOTATED_VIDEO and writer is not None:
                writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break

        # Graceful shutdown
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()

        cam_thread.join(timeout=1.0)
        io_thread.join(timeout=2.0)
        print("[INFO] Pipeline shutdown complete. Output saved.")


if __name__ == "__main__":
    pipeline = MultiThreadedNCNNppePipeline(source=VIDEO_PATH, model_path=MODEL_PATH)
    pipeline.run()
