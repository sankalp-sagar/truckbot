from ultralytics import YOLO
import cv2
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import pyautogui
import time

model = YOLO("runs/detect/train/weights/best.pt")
classes = model.names

def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def detect_trucks(img):
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

def find_refresh_icon(frame, template_path="templates/refresh.png", threshold=0.8):
    template = cv2.imread(template_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        h, w = template_gray.shape
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        return (center_x, center_y)
    return None

def crop_loot_panel(frame):
    frame = cv2.imread(frame)
    h, w, _ = frame.shape
    x1 = int(w * 0.39)
    x2 = int(w * 0.61)

    y1 = int(h * 0.64)
    y2 = int(h * 0.86)
    panel = frame[y1:y2, x1:x2]
    return panel

def crop_state_region(panel):
    h, w, _ = panel.shape
    x1 = int(w * 0.23)
    x2 = int(w * 0.8)
    y1 = int(h * 0.02)
    y2 = int(h * 0.25)
    state_region = panel[y1:y2, x1:x2] 
    text = pytesseract.image_to_string(state_region)
    text = text.split()
    timestamp = time.time()
    try:
        text = int(text[0][1:])
    except (IndexError, ValueError, TypeError):
        cv2.imwrite(f"badstatedetection/state_region_{text}_{timestamp}.png", state_region)
        cv2.imwrite(f"badstatedetection/state_region_{text}_{timestamp}_info_box.png", panel)
        text = 0
    return text

# time.sleep(5)
# frame = capture_screen()
# refresh_coords = find_refresh_icon(frame)
# trucks = detect_trucks(frame)

# for truck in trucks:
#     pyautogui.click(truck[0], truck[1])

frame = crop_loot_panel("screenbox.png")
state = crop_state_region(frame)
print(state)

# cv2.imshow("frane", state)
# cv2.waitKey(0)