from ultralytics import YOLO

# Load the exported NCNN model
model = YOLO("best_yolo11_ncnn_model", task="detect")

# Added save=True to export the resulting video
results = model.predict(source="media/cctv_test.mp4", show=True, imgsz=416, half=True, save=True)