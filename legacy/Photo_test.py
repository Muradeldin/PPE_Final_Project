import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
IMG_PATH = Path(__file__).parent / "worker_test.jpg"  # change if needed

PERSON_MODEL_PATH = "best_yolo8.pt"
PPE_MODEL_PATH = "best_yolo8.pt"

CROPS_DIR = Path("worker_crops")
CROPS_DIR.mkdir(exist_ok=True)

SAVE_EVERY_SECONDS = 1.0

# PPE thresholds
PPE_CONF_THRESH = 0.50     # for helmet/no-helmeta
VEST_THRESH = 0.25         # for vest
HELMET_MARGIN = 0.10       # helmet_conf - nohelmet_conf must exceed this

# PPE class IDs (MAKE SURE THESE MATCH YOUR MODEL)
HELMET_ID = 0
VEST_ID = 2
NO_HELMET_ID = 7

PAD = 0.0  # padding around person box before making center crop (set 0.0 if you want none)

# Center-crop inside person box (crowd fix)
CENTER_LEFT = 0.333333333   # keep middle band: [0.30w .. 0.70w]
CENTER_RIGHT = 0.666666667
CENTER_TOP = 0.05    # trim top/bottom a bit: [0.05h .. 0.95h]
CENTER_BOTTOM = 0.95

# Debug: show crops for 5 seconds each PPE check
SHOW_DEBUG_CROPS = True
DEBUG_WAIT_MS = 5000

# Fake tracking loop settings
REPEAT_FRAMES = 120
DELAY_MS = 50

SAVE_ANNOTATED_IMAGE = True
OUTPUT_IMAGE_PATH = Path("annotated_output.jpg")
# ----------------------------------------

person_model = YOLO(PERSON_MODEL_PATH)
ppe_model = YOLO(PPE_MODEL_PATH)

last_saved = {}
last_status = {}

def check_ppe(crop):
    """
    PPE check on the given crop (already center-filtered).
    Returns: (status_dict_or_None, updated_bool)
    """
    # Helmet / No-Helmet
    r = ppe_model.predict(
        source=crop,
        imgsz=416,  # you trained on 416
        classes=[HELMET_ID, VEST_ID, NO_HELMET_ID],
        conf=min(PPE_CONF_THRESH, VEST_THRESH),
        verbose=False
    )[0]

    best = {}
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            best[cls_id] = max(best.get(cls_id, 0.0), conf)

    # If nothing detected at all, don't update
    if not best:
        return None, False

    helmet_conf = best.get(HELMET_ID, 0.0)
    nohelmet_conf = best.get(NO_HELMET_ID, 0.0)
    vest_conf = best.get(VEST_ID, 0.0)

    updates = {}

    # Update helmet only if we saw helmet or no-helmet with enough confidence
    if helmet_conf >= PPE_CONF_THRESH or nohelmet_conf >= PPE_CONF_THRESH:
        updates["helmet"] = (helmet_conf - nohelmet_conf) > HELMET_MARGIN

    # Update vest only if we saw vest with enough confidence
    if vest_conf >= VEST_THRESH:
        updates["vest"] = True

    if not updates:
        return None, False

    return updates, True


# Load photo
img = cv2.imread(str(IMG_PATH))
if img is None:
    raise RuntimeError(f"Could not read image: {IMG_PATH}")

# Fake tracking: feed the same image multiple times
for _ in range(REPEAT_FRAMES):

    tracked = person_model.track(
        source=img,
        classes=[0],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )[0]

    frame = tracked.orig_img.copy()
    h, w = frame.shape[:2]

    if tracked.boxes is None or tracked.boxes.id is None:
        cv2.imshow("Track + PPE (Photo)", frame)
        if cv2.waitKey(DELAY_MS) & 0xFF == ord("q"):
            break
        continue

    for box, tid_tensor in zip(tracked.boxes, tracked.boxes.id):
        now = time.time()
        worker_id = int(tid_tensor.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # clamp person box
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        
        # padded crop for PPE
        PAD = 0.10
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * PAD)
        py = int(bh * PAD)

        x1p = max(0, x1 - px)
        y1p = max(0, y1 - py)
        x2p = min(w - 1, x2 + px)
        y2p = min(h - 1, y2 + py)

        should_check = (worker_id not in last_saved) or ((now - last_saved[worker_id]) >= SAVE_EVERY_SECONDS)

        if should_check:
            last_saved[worker_id] = now

            # full person crop
            crop = frame[y1p:y2p, x1p:x2p]

            if crop.size != 0:
                if SHOW_DEBUG_CROPS:
                    cv2.imshow("CENTER CROP (Debug)", crop)
                    cv2.waitKey(DEBUG_WAIT_MS)
                    cv2.destroyWindow("CENTER CROP (Debug)")

                new_ppe, updated = check_ppe(crop)
                if updated and new_ppe is not None:
                    # merge updates (don't overwrite missing keys)
                    prev = last_status.get(worker_id, {})
                    merged = prev.copy()
                    merged.update(new_ppe)
                    last_status[worker_id] = merged
                    print(f"Worker {worker_id}: {last_status[worker_id]}")

                # optional debug save (disable on Pi for speed)
                out_path = CROPS_DIR / f"worker_id_{worker_id}.jpg"
                cv2.imwrite(str(out_path), crop)

        # draw using last known status
        ppe = last_status.get(worker_id)
        if ppe is None:
            color = (0, 255, 255)
            label = f"ID {worker_id} checking..."
        else:
            helmet_ok = bool(ppe.get("helmet", False))
            vest_ok = bool(ppe.get("vest", False))
            ok = helmet_ok and vest_ok
            color = (0, 255, 0) if ok else (0, 0, 255)
            label = f"ID {worker_id}, H:{helmet_ok}, V:{vest_ok}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Track + PPE (Photo)", frame)
    if cv2.waitKey(DELAY_MS) & 0xFF == ord("q"):
        break

if SAVE_ANNOTATED_IMAGE:
    cv2.imwrite(str(OUTPUT_IMAGE_PATH), frame)

cv2.destroyAllWindows()
