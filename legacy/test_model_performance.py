from ultralytics import YOLO


ppe_model = YOLO("best_yolo8.pt")

ppe_model.predict(source="test_photo.png", save=True)