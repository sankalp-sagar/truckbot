import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import pyautogui
import time
import os
from pathlib import Path
import configparser
model = YOLO("runs/detect/train/weights/best.pt")
classes = model.names
ocr = PaddleOCR(lang="en")

config = configparser.ConfigParser()
config.read("config.ini")

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

def parse_player_info(ocr_result):
    state = None
    level = None
    alliance = None
    power = None

    for block in ocr_result:
        for line in block:
            text = line[1][0]
            m = re.search(r"#(\d+)", text)
            if m and state is None:
                state = int(m.group(1))
            m = re.search(r"Lv\.?\s*(\d+)", text, re.IGNORECASE)
            if m and level is None:
                level = int(m.group(1))
            m = re.search(r"\[([A-Za-z0-9]+)\]", text)
            if m and alliance is None:
                alliance = m.group(1)
            m = re.search(r"(\d{1,3}(?:,\d{3})+)", text)
            if m and power is None:
                power = int(m.group(1).replace(",", ""))

    return state, level, alliance, power

def crop_loot_panel(frame, x1,y1,x2,y2):
    panel = frame[y1:y2, x1:x2]
    cv2.imwrite("panel.png", panel)
    return panel

def detect_template(img, template_path, threshold=0.8):
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    h, w = template.shape[:2]

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    detections = []
    for pt in zip(*loc[::-1]):
        detections.append(pt)

    detections.sort(key=lambda p: p[0])

    count = 0
    last_x = -9999
    for (x, y) in detections:
        if x - last_x > w * 0.8:
            count += 1
            last_x = x

    return count

def scroll(x1, y1, x2, y2, duration=0.2):
    pyautogui.moveTo(x1, y1)
    time.sleep(0.05)
    pyautogui.mouseDown()
    pyautogui.moveTo(x2, y2, duration=duration)
    pyautogui.mouseUp()

def count_fragments(
        info_box_path="panel.png",
        fragment_template="templates/fragment.png",
        mod_template="templates/mod_box.png",
        threshold=0.8
    ):
    img = cv2.imread(info_box_path, cv2.IMREAD_COLOR)

    fragment_count = detect_template(img, fragment_template, threshold)
    mod_count = detect_template(img, mod_template, threshold)

    return {
        "fragments": fragment_count,
        "mods": mod_count
    }

def crop_state_region(panel, x1, y1, x2, y2):
    state_region = panel[y1:y2, x1:x2] 
    cv2.imwrite("stateregion.png", state_region)
    result = ocr.ocr(state_region)
    state, level, alliance, power = parse_player_info(result)
    return state, level, alliance, power

x1 = int(config["screen_values"]["x1"])
y1 = int(config["screen_values"]["y1"])
x2 = int(config["screen_values"]["x2"])
y2 = int(config["screen_values"]["y2"])
truckinfotopx1 = int(config["screen_values"]["truckinfotopx1"])
truckinfotopy1 = int(config["screen_values"]["truckinfotopy1"])
truckinfobottomx2 = int(config["screen_values"]["truckinfobottomx2"])
truckinfobottomy2 = int(config["screen_values"]["truckinfobottomy2"])

stateboxx1 = int(config["screen_values"]["stateboxx1"])
stateboxy1 = int(config["screen_values"]["stateboxy1"])
stateboxx2 = int(config["screen_values"]["stateboxx2"])
stateboxy2 = int(config["screen_values"]["stateboxy2"])

scrollx1 = int(config["screen_values"]["scrollx1"])
scrolly1 = int(config["screen_values"]["scrolly1"])
scrollx2 = int(config["screen_values"]["scrollx2"])
scrollx2 = int(config["screen_values"]["scrolly2"])

refreshx = int(config["screen_values"]["refreshx"])
refreshy = int(config["screen_values"]["refreshy"])

world_chat_x = int(config["screen_values"]["firstchatx"])
world_chat_y = int(config["screen_values"]["firstchaty"])

secondchatx = int(config["screen_values"]["secondchatx"])
secondchaty = int(config["screen_values"]["secondchaty"])
thirdchatx = int(config["screen_values"]["thirdchatx"])
thirdchaty = int(config["screen_values"]["thirdchaty"])
fourthchatx = int(config["screen_values"]["fourthchatx"])
fourthchaty = int(config["screen_values"]["fourthchaty"])
fifthchatx = int(config["screen_values"]["fifthchatx"])
fifthchaty = int(config["screen_values"]["fifthchaty"])
sixthchatx = int(config["screen_values"]["sixthchatx"])
sixthchaty = int(config["screen_values"]["sixthchaty"])

# panel = crop_loot_panel("screenbox.png", truckinfotopx1, truckinfotopy1, truckinfobottomx2, truckinfobottomy2)
# state, level, alliance, power = crop_state_region("screenbox.png", stateboxx1, stateboxy1, stateboxx2, stateboxy2)
# print(f"\nState: {state}, Level: {level}, Alliance: {alliance}, Power: {power}")

if __name__ == "__main__":
    time.sleep(5)
    state_filter = 0
    #state_filter = int(input("Enter the state ID to filter (0 for no filter): "))
    while True:
        screen = capture_screen()
        trucks = detect_trucks(screen)
        for truck in trucks:
            pyautogui.click(truck[0], truck[1]) # Click on truck
            time.sleep(0.5)
            points = 0
            screen = capture_screen()
            info_box = crop_loot_panel(screen, truckinfotopx1, truckinfotopy1, truckinfobottomx2, truckinfobottomy2)
            state, level, alliance, power = crop_state_region(screen, stateboxx1, stateboxy1, stateboxx2, stateboxy2)
            print(f"\nState: {state}, Level: {level}, Alliance: {alliance}, Power: {power}")

        pyautogui.click(refreshx, refreshy) # Click on refresh







# time.sleep(5)
# frame = capture_screen()
# refresh_coords = find_refresh_icon(frame)
# trucks = detect_trucks(frame)

# for truck in trucks:
#     pyautogui.click(truck[0], truck[1])

# tests_folder = "tests"
# for test_file in Path(tests_folder).glob("*"):
#     if test_file.is_file():
#         frame = crop_loot_panel(str(test_file))
#         state, level, alliance, power = crop_state_region(frame)
#         print(f"\n{test_file.name} - State: {state}, Level: {level}, Alliance: {alliance}, Power: {power}")

# frame = crop_loot_panel("screenbox.png")
# state, level, alliance, power = crop_state_region(frame)
# print("State: ", state)
# print("Level: ", level)
# print("Alliance: ", alliance)
# print("Power: ", power)

# cv2.imshow("frane", state)
# cv2.waitKey(0)

# tests_filder = "tests"
# for test_file in Path(tests_filder).glob("*"):
#     if test_file.is_file():
#         frame = crop_loot_panel(str(test_file))
#         fragments = count_fragments()
#         print("Fragment is: ", fragments)
