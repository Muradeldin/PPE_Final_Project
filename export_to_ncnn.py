from ultralytics import YOLO

# Load your trained PyTorch model
model = YOLO("best_yolo11.pt")

# Export the model to NCNN format with FP16 (half-precision)
model.export(format="ncnn", half=True)