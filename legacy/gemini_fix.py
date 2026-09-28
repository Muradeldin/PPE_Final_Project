import cv2
import time
import threading
from queue import Queue, Empty, Full
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# ---------------- CONFIGURATION ----------------
# Adjust these paths for your environment
VIDEO_PATH = Path(__file__).parent / "media" / "cctv_test.mp4"
MODEL_PATH = "best_yolo8.pt"  
CROPS_DIR = Path("worker_crops")
CROPS_DIR.mkdir(exist_ok=True)

# System constraints (Tuned for Edge hardware)
SAVE_EVERY_SECONDS = 0.5      
VIOLATION_SECONDS = 2.0       
CROP_SAVE_COOLDOWN = 3.0      

# Optimization targets for memory bandwidth
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# PPE Class IDs (Match your trained dataset)
PERSON_ID = 6
HELMET_ID = 0
VEST_ID = 2
NO_HELMET_ID = 7
# -----------------------------------------------

def calculate_ioa(person_box, ppe_box):
    """
    Calculates what percentage of the PPE box overlaps with the person box.
    This makes the detection posture-independent.
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


def ppe_for_person(person_xyxy, ppe_dets):
    """Evaluates PPE compliance using IoA mapping."""
    IOA_THRESHOLD = 0.60 
    
    best_helmet_conf = 0.0
    best_nohelmet_conf = 0.0
    best_vest_conf = 0.0

    for d in ppe_dets:
        cls_id = d["cls"]
        conf = d["conf"]
        ppe_box = d["xyxy"]

        ioa = calculate_ioa(person_xyxy, ppe_box)
        if ioa < IOA_THRESHOLD:
            continue

        if cls_id == HELMET_ID and conf > best_helmet_conf:
            best_helmet_conf = conf
        elif cls_id == NO_HELMET_ID and conf > best_nohelmet_conf:
            best_nohelmet_conf = conf
        elif cls_id == VEST_ID and conf > best_vest_conf:
            best_vest_conf = conf

    if best_helmet_conf == 0.0 and best_nohelmet_conf == 0.0 and best_vest_conf == 0.0:
        return None

    # Independent Thresholds
    helmet = None
    if best_helmet_conf > 0.30:
        helmet = True
    elif best_nohelmet_conf > 0.30:
        helmet = False
    else:
        helmet = False 

    vest = best_vest_conf >= 0.40

    return {"helmet": helmet, "vest": vest}


class EdgePPEPipeline:
    def __init__(self, source, model_path):
        self.source = str(source)
        self.model = YOLO(model_path)
        
        # Threading Queues
        # maxsize=3 prevents RAM explosion and paces video files
        self.frame_queue = Queue(maxsize=3) 
        self.io_queue = Queue(maxsize=10)   

        # State Management
        self.running = False
        self.last_checked = {}
        self.last_status = {}
        self.violation_since = {}
        self.last_crop_saved = {}

    def _camera_worker(self):
        """Thread 1: Ingests frames, scales them down, and handles video pacing."""
        cap = cv2.VideoCapture(self.source)
        
        # Hardware-level request to reduce bandwidth 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Software fallback: guarantee the frame is downsampled
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            # Block with a timeout to pace .mp4 files and allow clean shutdown
            try:
                self.frame_queue.put(frame, timeout=1.0)
            except Full:
                continue
                
        cap.release()

    def _io_worker(self):
        """Thread 3: Handles disk writes asynchronously to prevent inference blocking."""
        while self.running or not self.io_queue.empty():
            try:
                # Wait up to 1 second for a task
                task = self.io_queue.get(timeout=1.0)
                file_path, image = task
                cv2.imwrite(str(file_path), image)
                self.io_queue.task_done()
            except Empty:
                continue

    def run(self):
        """Thread 2 (Main): Pulls frames, runs C++ backend inference, updates state."""
        self.running = True
        
        # Start background threads
        cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        io_thread = threading.Thread(target=self._io_worker, daemon=True)
        cam_thread.start()
        io_thread.start()

        print("[INFO] Edge Pipeline Initialized. Threads running.")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                # If the queue is empty AND the camera thread is dead, the video is over.
                if not cam_thread.is_alive():
                    self.running = False  # Signal all threads to shut down
                    break 
                continue

            now = time.time()
            
            # Run inference & tracking
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml", 
                conf=0.20, 
                imgsz=416, 
                classes=[PERSON_ID, HELMET_ID, VEST_ID, NO_HELMET_ID],
                verbose=False
            )
            
            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                cv2.imshow("Edge PPE Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                continue

            # Parse detections
            xyxy_all = res.boxes.xyxy.cpu().numpy()
            cls_all = res.boxes.cls.cpu().numpy().astype(int)
            conf_all = res.boxes.conf.cpu().numpy()
            ids_all = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None

            # Separate PPE items
            ppe_dets = [
                {"cls": cls_all[i], "conf": conf_all[i], "xyxy": xyxy_all[i]}
                for i in range(len(cls_all)) if cls_all[i] != PERSON_ID
            ]

            if ids_all is not None:
                for i in range(len(cls_all)):
                    if cls_all[i] != PERSON_ID:
                        continue
                    
                    worker_id = int(ids_all[i])
                    x1, y1, x2, y2 = map(int, xyxy_all[i])
                    h, w = frame.shape[:2]
                    
                    # Clamp bounding boxes to frame limits
                    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                    x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Throttle checks to save CPU cycles
                    should_update = (worker_id not in self.last_checked) or ((now - self.last_checked[worker_id]) >= SAVE_EVERY_SECONDS)

                    if should_update:
                        self.last_checked[worker_id] = now
                        status = ppe_for_person((x1, y1, x2, y2), ppe_dets)
                        
                        if status is not None:
                            self.last_status[worker_id] = status

                            noncompliant = (not status["helmet"]) or (not status["vest"])
                            if noncompliant:
                                if worker_id not in self.violation_since:
                                    self.violation_since[worker_id] = now

                                if (now - self.violation_since[worker_id]) >= VIOLATION_SECONDS:
                                    last_t = self.last_crop_saved.get(worker_id, 0.0)
                                    if (now - last_t) >= CROP_SAVE_COOLDOWN:
                                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        out_path = CROPS_DIR / f"worker_{worker_id}_{ts}.jpg"
                                        crop = frame[y1:y2, x1:x2].copy()
                                        
                                        # Offload disk I/O to background thread
                                        self.io_queue.put((out_path, crop))
                                        self.last_crop_saved[worker_id] = now
                            else:
                                self.violation_since.pop(worker_id, None)

                    # Draw graphics
                    ppe = self.last_status.get(worker_id)
                    if ppe is None:
                        color, label = (0, 255, 255), f"ID {worker_id} scan.."
                    else:
                        ok = ppe["helmet"] and ppe["vest"]
                        color = (0, 255, 0) if ok else (0, 0, 255)
                        label = f"ID {worker_id} H:{int(ppe['helmet'])} V:{int(ppe['vest'])}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Yield the thread for 1 millisecond so the CPU doesn't lock up
            time.sleep(0.001)

        # Cleanup
        cv2.destroyAllWindows()
        cam_thread.join()
        io_thread.join()
        print("[INFO] Pipeline Shutdown Cleanly.")

if __name__ == "__main__":
    pipeline = EdgePPEPipeline(source=VIDEO_PATH, model_path=MODEL_PATH)
    pipeline.run()