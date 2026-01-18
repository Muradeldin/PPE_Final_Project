import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_PATH = Path(__file__).parent / "cctv_test.mp4"

PERSON_MODEL_PATH = "yolov8n.pt"
PPE_MODEL_PATH = "best.pt"

CROPS_DIR = Path("worker_crops")
CROPS_DIR.mkdir(exist_ok=True)

SAVE_EVERY_SECONDS = 0.5
PPE_CONF_THRESH = 0.2
VEST_THRESH = 0.4
HELMET_MARGIN = 0.4

# PPE class IDs (set these to your model’s mapping)
HELMET_ID = 0
VEST_ID = 2
NO_HELMET_ID = 7

# Output annotated video (optional but useful)
SAVE_ANNOTATED_VIDEO = True
OUTPUT_VIDEO_PATH = Path("annotated_output.mp4")
# ----------------------------------------


person_model = YOLO(PERSON_MODEL_PATH)
ppe_model = YOLO(PPE_MODEL_PATH)

last_saved = {}
last_status = {}


def check_ppe(crop):
    r = ppe_model.predict(
        source=crop,
        imgsz=416,  # trained on 416
        classes=[HELMET_ID, VEST_ID, NO_HELMET_ID],
        conf=PPE_CONF_THRESH,
        verbose=False,
    )[0]

    best = {}
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            if conf < PPE_CONF_THRESH:
                continue
            if conf > best.get(cls_id, 0.0):
                best[cls_id] = conf

    # If nothing above threshold, don't update status
    if not best:
        return None, False

    helmet_conf   = best.get(HELMET_ID, 0.0)
    nohelmet_conf = best.get(NO_HELMET_ID, 0.0)
    vest_conf     = best.get(VEST_ID, 0.0)

    # Helmet: choose stronger with margin
    if helmet_conf == 0.0 and nohelmet_conf == 0.0:
        return None, False

    print(f"helmet_conf: {helmet_conf}, nohelmet_conf: {nohelmet_conf}, vest_conf: {vest_conf}")
    helmet = (helmet_conf - nohelmet_conf) > HELMET_MARGIN
    vest = vest_conf >= VEST_THRESH


    return {"helmet": helmet, "vest": vest}, True




# --- Prepare video writer (we'll init it after we get first frame) ---
writer = None
fps_out = None
frame_size = None

# Track people from video with ByteTrack
for tracked in person_model.track(
    source=str(VIDEO_PATH),
    stream=True,
    classes=[0],
    tracker="bytetrack.yaml",
    persist=True,
    verbose=False
):
    frame = tracked.orig_img

    # init writer once we know frame size/fps
    if SAVE_ANNOTATED_VIDEO and writer is None:
        # Try to read FPS from the source video; fall back to 30 if unknown
        cap_tmp = cv2.VideoCapture(str(VIDEO_PATH))
        fps = cap_tmp.get(cv2.CAP_PROP_FPS)
        cap_tmp.release()
        fps_out = fps if fps and fps > 0 else 30.0

        h, w = frame.shape[:2]
        frame_size = (w, h)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps_out, frame_size)

    if tracked.boxes is None or tracked.boxes.id is None:
        cv2.imshow("Track + PPE (Video)", frame)
        if SAVE_ANNOTATED_VIDEO and writer is not None:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    for box, tid_tensor in zip(tracked.boxes, tracked.boxes.id):
        now = time.time()
        worker_id = int(tid_tensor.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

                # padded crop for PPE
        PAD = 0.0
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * PAD)
        py = int(bh * PAD)

        x1p = max(0, x1 - px) if PAD != 0 else x1
        y1p = max(0, y1 - py) if PAD != 0 else y1
        x2p = min(w - 1, x2 + px) if PAD != 0 else x2
        y2p = min(h - 1, y2 + py) if PAD != 0 else y2

        UPPER_BODY_RATIO = 0.70  # keep top 70% of person

        bh_p = y2p - y1p
        y2_upper = y1p + int(bh_p * UPPER_BODY_RATIO)

        # safety clamp
        y2_upper = min(y2_upper, h - 1)

        should_check = (worker_id not in last_saved) or ((now - last_saved[worker_id]) >= SAVE_EVERY_SECONDS)

        if should_check:
            last_saved[worker_id] = now

            crop = frame[y1p:y2_upper, x1p:x2p]
            if crop.size != 0:
                new_ppe, updated = check_ppe(crop)
                if updated:
                    last_status[worker_id] = new_ppe
                    print(f"Worker {worker_id}: {new_ppe}")

                out_path = CROPS_DIR / f"worker_id_{worker_id}.jpg"
                cv2.imwrite(str(out_path), crop)

        # draw using last known status
        ppe = last_status.get(worker_id)
        if ppe is None:
            color = (0, 255, 255)
            label = f"ID {worker_id} checking..."
        else:
            ok = ppe["helmet"] and ppe["vest"]
            color = (0, 255, 0) if ok else (0, 0, 255)
            label = f"ID {worker_id}, H: {bool(ppe['helmet'])}, V: {bool(ppe['vest'])}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Track + PPE (Video)", frame)

    if SAVE_ANNOTATED_VIDEO and writer is not None:
        writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
if writer is not None:
    writer.release()
