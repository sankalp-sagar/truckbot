from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")
classes = model.names

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

centers = detect_trucks("screen.png")

print("Detected truck centers:")
print(centers)