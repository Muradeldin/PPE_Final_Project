import cv2
import time
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# ---------------- CONFIG ----------------
VIDEO_PATH = Path(__file__).parent / "media/cctv_test.mp4"
MODEL_PATH = "best_yolo11_ncnn_model"  

PPE_CONF_THRESH = 0.10
HELMET_THRESH = 0.25          # confidence required for helmet
VEST_THRESH = 0.45            # confidence required for safety vest
HELMET_MARGIN = 0.10          # helmet vs no-helmet margin

# Your dataset class IDs
PERSON_ID = 6
HELMET_ID = 0
VEST_ID = 2
NO_HELMET_ID = 7

# If you want to save crops
SAVE_CROPS = True
CROPS_DIR = Path("worker_crops")
CROPS_DIR.mkdir(exist_ok=True)

# Output annotated video
SAVE_ANNOTATED_VIDEO = True
OUTPUT_VIDEO_PATH = Path("annotated_output.mp4")

VIOLATION_SECONDS = 2.0          # must be non-compliant for this long before saving crop
CROP_SAVE_COOLDOWN = 3.0         # cooldown between saving images of the same worker
# ----------------------------------------


model = YOLO(MODEL_PATH, task="detect")


violation_since = {}             # {worker_id: first_time_noncompliant}
last_crop_saved = {}             # {worker_id: last_time_saved}
last_status = {}                 # {worker_id: {"helmet": bool, "vest": bool}}


def _box_center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def ppe_for_person(person_xyxy, ppe_dets, worker_id=None):
    """
    Evaluates PPE compliance for a specific person.
    Matches PPE detections strictly within the person's own bounding box.
    """
    px1, py1, px2, py2 = person_xyxy
    ph = max(1.0, py2 - py1)

    # Vertical regions within person box
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

        # Strict horizontal containment: vest/helmet center MUST be inside this person's box
        if not (px1 <= cx <= px2):
            continue

        # Helmet/no-helmet check
        if cls_id in (HELMET_ID, NO_HELMET_ID):
            if helmet_y1 <= cy <= helmet_y2:
                if cls_id == HELMET_ID and conf > best_helmet_conf:
                    best_helmet_conf = conf
                elif cls_id == NO_HELMET_ID and conf > best_nohelmet_conf:
                    best_nohelmet_conf = conf

        # Vest check (strictly on the torso)
        elif cls_id == VEST_ID:
            if vest_y1 <= cy <= vest_y2:
                if conf > best_vest_conf:
                    best_vest_conf = conf

    print(f"worker_id: {worker_id} -> helmet_conf: {best_helmet_conf:.2f}, nohelmet_conf: {best_nohelmet_conf:.2f}, vest_conf: {best_vest_conf:.2f}")

    # Helmet Decision
    if best_helmet_conf >= HELMET_THRESH:
        helmet = (best_helmet_conf - best_nohelmet_conf) >= HELMET_MARGIN or best_nohelmet_conf == 0.0
    else:
        helmet = False

    # Vest Decision
    vest = best_vest_conf >= VEST_THRESH

    return {"helmet": helmet, "vest": vest}


# --- Prepare video writer (init after first frame) ---
writer = None

# Track + detect in ONE call (same model)
for res in model.track(
    source=str(VIDEO_PATH),
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    conf=PPE_CONF_THRESH,
    imgsz=416,
    verbose=False,
    classes=[PERSON_ID, HELMET_ID, VEST_ID, NO_HELMET_ID],
):
    frame = res.orig_img

    # init writer once we know frame size/fps
    if SAVE_ANNOTATED_VIDEO and writer is None:
        cap_tmp = cv2.VideoCapture(str(VIDEO_PATH))
        fps = cap_tmp.get(cv2.CAP_PROP_FPS)
        cap_tmp.release()
        fps_out = fps if fps and fps > 0 else 30.0

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps_out, (w, h))

    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        cv2.imshow("Track + PPE (Single Model)", frame)
        if SAVE_ANNOTATED_VIDEO and writer is not None:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Convert detections to easy python lists
    xyxy_all = boxes.xyxy.cpu().numpy()
    cls_all = boxes.cls.cpu().numpy().astype(int)
    conf_all = boxes.conf.cpu().numpy()
    ids_all = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    # Separate PPE detections for this frame
    ppe_dets = []
    for i in range(len(cls_all)):
        c = int(cls_all[i])
        if c in (HELMET_ID, VEST_ID, NO_HELMET_ID):
            x1, y1, x2, y2 = map(float, xyxy_all[i])
            ppe_dets.append({"cls": c, "conf": float(conf_all[i]), "xyxy": (x1, y1, x2, y2)})

    # Process each tracked person in real time
    for i in range(len(cls_all)):
        if int(cls_all[i]) != PERSON_ID:
            continue
        if ids_all is None:
            continue

        worker_id = int(ids_all[i])
        now = time.time()

        x1, y1, x2, y2 = map(int, xyxy_all[i].tolist())
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # Evaluate compliance immediately for this frame
        status = ppe_for_person((x1, y1, x2, y2), ppe_dets, worker_id)
        last_status[worker_id] = status

        noncompliant = (not status["helmet"]) or (not status["vest"])

        if noncompliant:
            if worker_id not in violation_since:
                violation_since[worker_id] = now

            if (now - violation_since[worker_id]) >= VIOLATION_SECONDS:
                if SAVE_CROPS:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size != 0:
                        last_t = last_crop_saved.get(worker_id, 0.0)
                        if (now - last_t) >= CROP_SAVE_COOLDOWN:
                            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                            out_path = CROPS_DIR / f"worker_{worker_id}_{ts}.jpg"
                            cv2.imwrite(str(out_path), crop)
                            last_crop_saved[worker_id] = now
        else:
            violation_since.pop(worker_id, None)

        # Draw using current status with original format
        ppe = last_status.get(worker_id)
        if ppe is None:
            color = (0, 255, 255)
            label = f"ID {worker_id} checking..."
        else:
            ok = ppe["helmet"] and ppe["vest"]
            color = (0, 255, 0) if ok else (0, 0, 255)
            label = f"ID {worker_id}, H:{ppe['helmet']} V:{ppe['vest']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Track + PPE (Single Model)", frame)
    if SAVE_ANNOTATED_VIDEO and writer is not None:
        writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
if writer is not None:
    writer.release()
