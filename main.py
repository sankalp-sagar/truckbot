from ultralytics import YOLO
import cv2
import cv2
import numpy as np
import mss
import pyautogui
import time
import dxcam

camera = dxcam.create()
model = YOLO("runs/detect/train/weights/best.pt")
classes = model.names

sct = mss.mss()
monitor = sct.monitors[1]

def capture_screen_dxcam():
    frame = camera.grab()
    return frame

def detect_trucks(image_path):
    img = cv2.imread(image_path)
    results = model(img, save=False, conf=0.4)
    centers = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy))
    return centers

time.sleep(5)
frame = capture_screen_dxcam()
cv2.imshow("screen", frame)
cv2.waitKey(0)