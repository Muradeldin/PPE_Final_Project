from ultralytics import YOLO

model = YOLO("models/best_yolo8.pt")
# imgsz must match INFERENCE_SIZE in your edge script
model.export(format="ncnn", imgsz=320, half=True)