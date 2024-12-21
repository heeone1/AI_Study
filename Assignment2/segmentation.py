from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.predict("image2.png", show=True, save=True, conf=0.5)
