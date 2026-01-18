import cv2
import time
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# ---------------- CONFIG ----------------
VIDEO_PATH = Path(__file__).parent / "media/cctv_test.mp4"

MODEL_PATH = "best_yolo8.pt"  

SAVE_EVERY_SECONDS = 0.5      # update each tracked worker status this often
PPE_CONF_THRESH = 0.20
VEST_THRESH = 0.40
HELMET_MARGIN = 0.40

# Your dataset class IDs (EDIT THESE!)
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

VIOLATION_SECONDS = 2.0          # must be false for this long before saving
CROP_SAVE_COOLDOWN = 3.0         # optional: avoid saving every frame after triggered
# ----------------------------------------


model = YOLO(MODEL_PATH)


violation_since = {}             # {worker_id: first_time_noncompliant}
last_crop_saved = {}             # {worker_id: last_time_saved}
last_checked = {}   # {worker_id: time}
last_status = {}    # {worker_id: {"helmet": bool, "vest": bool}}


def _box_center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def ppe_for_person(person_xyxy, ppe_dets, worker_id=None):
    """
    person_xyxy: (x1,y1,x2,y2) for person
    ppe_dets: list of dicts: {"cls": int, "conf": float, "xyxy": (x1,y1,x2,y2)}
    Returns: {"helmet": bool, "vest": bool} or None if no evidence.
    """
    px1, py1, px2, py2 = person_xyxy
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)

    # Regions within a person bbox
    # Helmet region: top 35% of person
    helmet_y2 = py1 + 0.35 * ph
    # Vest region: middle chunk (roughly torso)
    vest_y1 = py1 + 0.30 * ph
    vest_y2 = py1 + 0.75 * ph

    best = {}  # best confidence per class for this person

    for d in ppe_dets:
        cls_id = d["cls"]
        conf = d["conf"]
        x1, y1, x2, y2 = d["xyxy"]
        cx, cy = _box_center_xyxy((x1, y1, x2, y2))

        # Must be inside the person's bbox at least
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            continue

        # Helmet/no-helmet only if in helmet region
        if cls_id in (HELMET_ID, NO_HELMET_ID):
            if cy > helmet_y2:
                continue

        # Vest only if in vest region
        if cls_id == VEST_ID:
            if not (vest_y1 <= cy <= vest_y2):
                continue

        if conf >= PPE_CONF_THRESH and conf > best.get(cls_id, 0.0):
            best[cls_id] = conf

    if not best:
        return None

    helmet_conf = best.get(HELMET_ID, 0.0)
    nohelmet_conf = best.get(NO_HELMET_ID, 0.0)
    vest_conf = best.get(VEST_ID, 0.0)
    print(f"worker_id: {worker_id} -> ", end="")
    print(f"helmet_conf: {helmet_conf}, nohelmet_conf: {nohelmet_conf}, vest_conf: {vest_conf}")
    # If we have no helmet/nohelmet evidence at all, treat as unknown -> None
    if helmet_conf == 0.0 and nohelmet_conf == 0.0:
        helmet = False
    else:
        helmet = (helmet_conf - nohelmet_conf) > HELMET_MARGIN

    vest = vest_conf >= VEST_THRESH

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
    # Keep only the classes we care about to reduce clutter/work:
    classes=[PERSON_ID , HELMET_ID, VEST_ID, NO_HELMET_ID],
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

    # Separate PPE detections (helmet/vest/nohelmet) for this frame
    ppe_dets = []
    for i in range(len(cls_all)):
        c = int(cls_all[i])
        if c in (HELMET_ID, VEST_ID, NO_HELMET_ID):
            x1, y1, x2, y2 = map(float, xyxy_all[i])
            ppe_dets.append({"cls": c, "conf": float(conf_all[i]), "xyxy": (x1, y1, x2, y2)})

    # Process each tracked person
    for i in range(len(cls_all)):
        if int(cls_all[i]) != PERSON_ID:
            continue
        if ids_all is None:
            continue  # no tracking IDs yet

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

        should_update = (worker_id not in last_checked) or ((now - last_checked[worker_id]) >= SAVE_EVERY_SECONDS)

        if should_update:
            last_checked[worker_id] = now

            status = ppe_for_person((x1, y1, x2, y2), ppe_dets, worker_id)
            if status is not None:
                last_status[worker_id] = status

            if SAVE_CROPS:
                crop = frame[y1:y2, x1:x2]
                ppe = last_status.get(worker_id)

                # Only act if we have a status
                if ppe is not None:
                    noncompliant = (not ppe["helmet"]) or (not ppe["vest"])

                    if noncompliant:
                        # start timer if first time we see noncompliance
                        if worker_id not in violation_since:
                            violation_since[worker_id] = now

                        # if it stayed noncompliant long enough -> save crop (once / cooldown)
                        if (now - violation_since[worker_id]) >= VIOLATION_SECONDS:
                            if SAVE_CROPS:
                                crop = frame[y1:y2, x1:x2]
                                if crop.size != 0:
                                    # cooldown so it doesn't spam saves
                                    last_t = last_crop_saved.get(worker_id, 0.0)
                                    if (now - last_t) >= CROP_SAVE_COOLDOWN:
                                        ts = datetime.now().strftime("%Y-%m-%d_%H:%M")
                                        out_path = CROPS_DIR / f"worker_{worker_id}_{ts}.jpg"
                                        cv2.imwrite(str(out_path), crop)
                                        last_crop_saved[worker_id] = now
                    else:
                        # compliant again -> reset timer
                        violation_since.pop(worker_id, None)


        # draw using last known status
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
