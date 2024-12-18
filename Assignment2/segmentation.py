from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")

model.predict(0, show=True, save=True, conf=0.8)
