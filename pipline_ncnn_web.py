import cv2
import time
import threading
import requests
from queue import Queue, Empty, Full
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# Input / Output
VIDEO_PATH = Path(__file__).parent / "media" / "cctv_test.mp4"
MODEL_PATH = "best_yolo8_ncnn_model"  # Make sure this was exported at imgsz=320!

# REPLACE THIS WITH YOUR MAC'S TAILSCALE IP ADDRESS (e.g., "http://100.64.1.2:8080/api/upload-crop")
CLOUD_WEBHOOK_URL = "http://YOUR_MAC_TAILSCALE_IP:8080/api/upload-crop"

# Performance & Display
HEADLESS = True            
TARGET_WIDTH = 640          
TARGET_HEIGHT = 480         
INFERENCE_SIZE = 320        

# PPE Logic & Thresholds
TRACK_CONF = 0.25          
PPE_CONF_THRESH = 0.10      
VEST_THRESH = 0.15          # Lowered for challenging lighting & lime green vests
HELMET_MARGIN = 0.20        
IOA_THRESHOLD = 0.30        # Lowered to allow bulky vests to overhang outside the person box

# Timing & Throttling
EVAL_INTERVAL_SECONDS = 0.5 
GLOBAL_CROP_COOLDOWN = 3.0  # Prevents spamming webhooks

# Class IDs (Must match your trained dataset)
PERSON_ID = 6
HELMET_ID = 0
VEST_ID = 2
NO_HELMET_ID = 7

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def calculate_ioa_and_position(person_box, ppe_box, ppe_class_id):
    """
    Calculates Intersection over Area using the minimum bounding area to handle
    edge-of-frame crops. Enforces spatial positioning ONLY for helmets to 
    prevent 'helmet in hand' bypasses.
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

    # Spatial logic is now ONLY enforced for helmets
    if ppe_class_id in [HELMET_ID, NO_HELMET_ID]:
        person_height = py2 - py1
        if person_height <= 0:
            return ioa, False
            
        ppe_center_y = (hy1 + hy2) / 2.0
        relative_y = (ppe_center_y - py1) / person_height
        valid_position = relative_y <= 0.35  # Helmet must be in top 35%
        return ioa, valid_position

    # If it's a vest and passes the IoA check, accept it immediately
    return ioa, True


def evaluate_person_ppe(person_xyxy, ppe_dets):
    """
    Associates PPE detections with a person box using IoA and spatial logic.
    """
    best = {HELMET_ID: 0.0, NO_HELMET_ID: 0.0, VEST_ID: 0.0}

    for d in ppe_dets:
        cls_id = d["cls"]
        conf = d["conf"]
        ppe_box = d["xyxy"]

        if conf >= PPE_CONF_THRESH:
            ioa, valid_pos = calculate_ioa_and_position(person_xyxy, ppe_box, cls_id)
            if valid_pos and conf > best.get(cls_id, 0.0):
                best[cls_id] = conf

    helmet_conf = best[HELMET_ID]
    nohelmet_conf = best[NO_HELMET_ID]
    vest_conf = best[VEST_ID]

    # If NO valid PPE boxes were detected at all, fail them immediately
    if helmet_conf == 0.0 and nohelmet_conf == 0.0 and vest_conf == 0.0:
        return {"helmet": False, "vest": False}

    # Helmet Decision
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
    def __init__(self, source=VIDEO_PATH, model_path=MODEL_PATH):
        self.source = str(source)
        self.model = YOLO(model_path, task="detect")
        
        self.frame_queue = Queue(maxsize=2)
        self.io_queue = Queue(maxsize=15)

        self.running = False
        self.last_checked = {}      
        self.last_status = {}       
        self.last_global_save = 0.0 

    def _camera_worker(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

            try:
                self.frame_queue.put(frame, timeout=0.5)
            except Full:
                continue

        cap.release()

    def _io_worker(self):
        """Asynchronously sends violation crops to the Cloud Admin Dashboard via HTTP POST."""
        while self.running or not self.io_queue.empty():
            try:
                task = self.io_queue.get(timeout=0.5)
                worker_id, reasons_str, crop = task
                
                # Encode crop to JPEG in memory
                _, img_encoded = cv2.imencode('.jpg', crop)
                files = {'file': (f"worker_{worker_id}.jpg", img_encoded.tobytes(), 'image/jpeg')}
                data = {'worker_id': str(worker_id), 'reasons': reasons_str}

                # Push to cloud dashboard
                response = requests.post(CLOUD_WEBHOOK_URL, files=files, data=data, timeout=5)
                if response.status_code == 200:
                    print(f"[INFO] Successfully uploaded violation for Worker {worker_id}")
                else:
                    print(f"[WARNING] Cloud responded with status {response.status_code}")

                self.io_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Failed to send webhook to cloud: {e}")

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

            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml", 
                conf=TRACK_CONF,
                imgsz=INFERENCE_SIZE,
                classes=[PERSON_ID, HELMET_ID, VEST_ID, NO_HELMET_ID],
                verbose=False
            )

            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                if not HEADLESS:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow("Edge PPE Pipeline", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                continue

            xyxy_all = res.boxes.xyxy.cpu().numpy()
            cls_all = res.boxes.cls.cpu().numpy().astype(int)
            conf_all = res.boxes.conf.cpu().numpy()
            
            # Safely handle when ByteTrack drops IDs due to low FPS
            ids_all = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None

            # Collect PPE detections
            ppe_dets = [
                {"cls": cls_all[i], "conf": float(conf_all[i]), "xyxy": xyxy_all[i]}
                for i in range(len(cls_all)) if cls_all[i] != PERSON_ID
            ]

            h, w = frame.shape[:2]

            for i in range(len(cls_all)):
                if cls_all[i] != PERSON_ID:
                    continue

                # Generate a temporary ID if tracker drops frame to prevent skipping worker
                worker_id = int(ids_all[i]) if ids_all is not None else f"tmp_{i}"
                
                x1, y1, x2, y2 = map(int, xyxy_all[i])

                # Clamp to frame dimensions
                x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                should_update = (
                    (worker_id not in self.last_checked) or 
                    ((now - self.last_checked[worker_id]) >= EVAL_INTERVAL_SECONDS)
                )

                if should_update:
                    self.last_checked[worker_id] = now
                    status = evaluate_person_ppe((x1, y1, x2, y2), ppe_dets)

                    if status is not None:
                        # If worker touches top of the frame boundary, assume head is off-camera
                        if y1 <= 10: 
                            status["helmet"] = True
                            
                        self.last_status[worker_id] = status
                        noncompliant = (not status["helmet"]) or (not status["vest"])

                        if noncompliant:
                            # Trigger violation off global cooldown
                            if (now - self.last_global_save) >= GLOBAL_CROP_COOLDOWN:
                                reasons = []
                                if not status["helmet"]: reasons.append("no_helmet")
                                if not status["vest"]: reasons.append("no_vest")
                                reasons_str = "_".join(reasons)
                                
                                crop = frame[y1:y2, x1:x2].copy()

                                try:
                                    # Push data to asynchronous webhook queue
                                    self.io_queue.put_nowait((worker_id, reasons_str, crop))
                                    self.last_global_save = now
                                    print(f"[ALERT] Violation! Webhook queued for Worker ID: {worker_id}")
                                except Full:
                                    pass  

                # UI Rendering 
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