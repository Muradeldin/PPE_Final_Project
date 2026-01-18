import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
SOURCE = 0  # Mac camera
PERSON_MODEL_PATH = "yolov8n.pt"
PPE_MODEL_PATH = "ppe_model.pt"

CROPS_DIR = Path("worker_crops")
CROPS_DIR.mkdir(exist_ok=True)

SAVE_EVERY_SECONDS = 1.5
PPE_CONF_THRESH = 0.25

# PPE class IDs (set these to your model’s mapping)
HELMET_ID = 0
VEST_ID   = 2
NO_HELMET_ID = 7
# ----------------------------------------


person_model = YOLO(PERSON_MODEL_PATH)
ppe_model = YOLO(PPE_MODEL_PATH)

last_saved = {}
last_status = {}

def check_ppe(crop):
    r = ppe_model.predict(
        source=crop,
        imgsz=416,
        classes=[HELMET_ID, VEST_ID, NO_HELMET_ID],
        verbose=False
    )[0]

    # collect best confidence per class (above threshold)
    best = {}
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if conf < PPE_CONF_THRESH:
                continue
            best[cls_id] = max(best.get(cls_id, 0.0), conf)

    # If we got NOTHING above threshold, don't update status
    if not best:
        return None, False

    # Decide helmet: helmet vs no_helmet
    helmet_conf = best.get(HELMET_ID, 0.0)
    nohelmet_conf = best.get(NO_HELMET_ID, 0.0)
    helmet = helmet_conf > nohelmet_conf and helmet_conf > 0.0
    # Decide vest: only positive class exists in your mapping
    vest = best.get(VEST_ID, 0.0) > 0.0

    return {"helmet": helmet, "vest": vest}, True


# Track people from webcam with ByteTrack
for tracked in person_model.track(
    source=SOURCE,
    stream=True,
    classes=[0],               
    tracker="bytetrack.yaml",
    persist=True,
    verbose=False
):
    frame = tracked.orig_img
    now = time.time()

    if tracked.boxes is None or tracked.boxes.id is None:
        cv2.imshow("Track + PPE (Webcam)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    for box, tid_tensor in zip(tracked.boxes, tracked.boxes.id):
        worker_id = int(tid_tensor.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # clamp to frame bounds (prevents weird crops)
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # rate limit PPE checks + saving per worker
        should_check = (worker_id not in last_saved) or ((now - last_saved[worker_id]) >= SAVE_EVERY_SECONDS)

        if should_check:
            crop = frame[y1:y2, x1:x2]
            if crop.size != 0:
                new_ppe, updated = check_ppe(crop)
                if updated:
                    last_status[worker_id] = new_ppe
                    print(f"Worker {worker_id}: {new_ppe}")

                last_saved[worker_id] = now
                # overwrite latest crop
                out_path = CROPS_DIR / f"worker_id_{worker_id}.jpg"
                cv2.imwrite(str(out_path), crop)

        # ---- draw EVERY frame using last known status ----
        ppe = last_status.get(worker_id)
        if ppe is None:
            color = (0, 255, 255)  # yellow
            label = f"ID {worker_id} checking..."
        else:
            ok = ppe["helmet"] and ppe["vest"]
            color = (0, 255, 0) if ok else (0, 0, 255)
            label = f"ID {worker_id},  H: {bool(ppe['helmet'])},  V: {bool(ppe['vest'])}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        
    cv2.imshow("Track + PPE (Webcam)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
