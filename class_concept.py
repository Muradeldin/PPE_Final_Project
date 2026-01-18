import cv2
import time
from pathlib import Path
from ultralytics import YOLO


class PPEPipeline:
    def __init__(
        self,
        source=0,
        person_model_path="yolov8n.pt",
        ppe_model_path="ppe_model.pt",
        crops_dir="worker_crops",
        save_every_seconds=1.5,
        ppe_conf_thresh=0.25,
    ):
        self.source = source
        self.save_every_seconds = save_every_seconds
        self.ppe_conf_thresh = ppe_conf_thresh

        self.helmet_id = 0
        self.vest_id = 2
        self.no_helmet_id = 7

        self.person_model = YOLO(person_model_path)
        self.ppe_model = YOLO(ppe_model_path)

        self.crops_dir = Path(crops_dir)
        self.crops_dir.mkdir(exist_ok=True)

        self.last_saved = {}   # worker_id -> last time we ran PPE/saved crop
        self.last_status = {}  # worker_id -> {"helmet": bool, "vest": bool}

    def _check_ppe(self, crop):
        r = self.ppe_model.predict(
            source=crop,
            classes=[self.helmet_id, self.vest_id, self.no_helmet_id],
            verbose=False
        )[0]

        best = {}
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if conf < self.ppe_conf_thresh:
                    continue
                best[cls_id] = max(best.get(cls_id, 0.0), conf)

        # No confident detections -> don't update
        if not best:
            return None, False

        helmet_conf = best.get(self.helmet_id, 0.0)
        nohelmet_conf = best.get(self.no_helmet_id, 0.0)
        helmet = (helmet_conf > nohelmet_conf) and (helmet_conf > 0.0)

        vest = best.get(self.vest_id, 0.0) > 0.0

        return {"helmet": helmet, "vest": vest}, True

    def run(self):
        for tracked in self.person_model.track(
            source=self.source,
            stream=True,
            classes=[0],               # person only
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        ):
            frame = tracked.orig_img
            now = time.time()

            if tracked.boxes is None or tracked.boxes.id is None:
                cv2.imshow("Track + PPE", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            h, w = frame.shape[:2]

            for box, tid_tensor in zip(tracked.boxes, tracked.boxes.id):
                worker_id = int(tid_tensor.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # clamp
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                should_check = (worker_id not in self.last_saved) or ((now - self.last_saved[worker_id]) >= self.save_every_seconds)

                if should_check:
                    self.last_saved[worker_id] = now  # throttle regardless

                    crop = frame[y1:y2, x1:x2]
                    if crop.size != 0:
                        new_ppe, updated = self._check_ppe(crop)
                        if updated:
                            self.last_status[worker_id] = new_ppe
                            print(f"Worker {worker_id}: {new_ppe}")

                        # overwrite latest crop
                        out_path = self.crops_dir / f"worker_id_{worker_id}.jpg"
                        cv2.imwrite(str(out_path), crop)

                # draw using last known status
                ppe = self.last_status.get(worker_id)
                if ppe is None:
                    color = (0, 255, 255)
                    label = f"ID {worker_id} checking..."
                else:
                    ok = ppe["helmet"] and ppe["vest"]
                    color = (0, 255, 0) if ok else (0, 0, 255)
                    label = f"ID {worker_id} H:{int(ppe['helmet'])}, V:{int(ppe['vest'])}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Track + PPE", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


pipeline = PPEPipeline()
pipeline.run()
