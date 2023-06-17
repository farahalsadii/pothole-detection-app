from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("train/POTHOLE.pt")

results = model.predict(source="video/Sections2.mp4", show=True)
print(results)








