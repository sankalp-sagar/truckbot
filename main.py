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
scrolly2 = int(config["screen_values"]["scrolly2"])

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

def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite("screen.png", frame)
    return frame

def detect_trucks(img):
    results = model(img, save=False, conf=0.7)
    centers = []
    offset = -20 
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2) + offset
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

    if state is None:
        state = 0

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

def send_to_world_chat():
    pyautogui.click(int(config["screen_values"]["sharebuttonx"]), int(config["screen_values"]["sharebuttony"]))
    time.sleep(0.5)
    pyautogui.click(world_chat_x, world_chat_y)
    time.sleep(0.5)
    pyautogui.click(int(config["screen_values"]["sharebuttonfinalx"]), int(config["screen_values"]["sharebuttonfinaly"]))
    time.sleep(1)

def send_to_group_chat():
    pyautogui.click(int(config["screen_values"]["sharebuttonx"]), int(config["screen_values"]["sharebuttony"]))
    time.sleep(0.5)
    pyautogui.click(fourthchatx, fourthchaty)
    time.sleep(0.5)
    pyautogui.click(int(config["screen_values"]["sharebuttonfinalx"]), int(config["screen_values"]["sharebuttonfinaly"]))
    time.sleep(1)

def send_to_second_chat():
    pyautogui.click(int(config["screen_values"]["sharebuttonx"]), int(config["screen_values"]["sharebuttony"]))
    time.sleep(0.5)
    pyautogui.click(fifthchatx, fifthchaty)
    time.sleep(0.5)
    pyautogui.click(int(config["screen_values"]["sharebuttonfinalx"]), int(config["screen_values"]["sharebuttonfinaly"]))
    time.sleep(1)

if __name__ == "__main__":
    state_filter = 0
    greedy_run = False
    group_run = False
    group_greedy_run = False
    second_mod_box_mode = False
    self_run = False
    mod_boxes = 0
    total_trucks_detected = 0
    run_mode = int(input("[+] Press 0 for normal mode.\n" \
                        "[+] Press 1 for greedy mode.\n" \
                        "[+] Press 2 for group run mode.\n" \
                        "[+] Press 3 for group greedy mode.\n" \
                        "[+] Press 4 for second chat mod box mode.\n"
                        "[+] Press 5 for send to myself mode\n"
                        "[+] Press 6 for custom points search mode\n"
                        "Input: "))
    state_filter = int(input("Enter the state ID to filter (0 for no filter): "))

    if run_mode == 1:
        greedy_run = True
        min_point = 2
    elif run_mode == 2:
        group_run = True
        min_point = 2
    elif run_mode == 3:
        group_greedy_run = True
        min_point = 3
    elif run_mode == 4:
        second_mod_box_mode = True
        min_point = 3
    elif run_mode == 5:
        self_run = True
        min_point = 3
    elif run_mode == 6:
        custom_group_run = True
        min_point = 3

    if run_mode == 0 or run_mode == 6:
        min_point = int(input("Enter the minimum number of points to be sent: "))

    detect_mod_boxes = int(input("Do you want to scroll as well? Might be slower.\nPress 0 for no and 1 for yes: "))
    time.sleep(5)

    while True:
        time.sleep(1)
        screen = capture_screen()
        trucks = detect_trucks(screen)
        for truck in trucks:
            pyautogui.click(truck[0], truck[1]) # Click on truck
            time.sleep(1)
            points = 0
            screen = capture_screen()
            info_box = crop_loot_panel(screen, truckinfotopx1, truckinfotopy1, truckinfobottomx2, truckinfobottomy2)
            state, level, alliance, power = crop_state_region(screen, stateboxx1, stateboxy1, stateboxx2, stateboxy2)
            print(f"\nState: {state}, Level: {level}, Alliance: {alliance}, Power: {power}")
            if run_mode == 0:
                if state_filter != 0 and state != state_filter:
                    print(f"[-] Skipping truck with state ID {state}")
                    if state != 0:
                        continue
            result = count_fragments()
            fragments = int(result["fragments"])
            mods = int(result["mods"])

            points += (fragments+mods)
            print(f"Detected {fragments} fragments, {mods} mods")
            if detect_mod_boxes and (second_mod_box_mode or points < min_point):
                scroll(scrollx2, scrolly2, scrollx1, scrolly1)
                time.sleep(0.5)
                screen = capture_screen()
                result = count_fragments()
                fragments = int(result["fragments"])
                mods = int(result["mods"])
                points += (fragments+mods)
                print(f"Detected {mods} mod boxes")
                scroll(scrollx1, scrolly1, scrollx2, scrolly2)
                time.sleep(0.5)
            total_trucks_detected += 1
            print(f"[+] Total trucks detected so far: {total_trucks_detected}")

            if greedy_run:
                if state_filter == 0:
                    # No state filter
                    if points == 2:
                        send_to_world_chat()
                    elif points > 2:
                        send_to_group_chat()
                else:
                    # State filter is ON
                    if points > 2:
                        send_to_group_chat()
                    elif state == state_filter and points == 2:
                        send_to_world_chat()
                    elif state == 0 and points > 1:
                        send_to_second_chat()
            elif group_run:
                # Group run ON
                if points > 1:
                    send_to_group_chat()
            elif group_greedy_run:
                if points >= min_point:
                    #send_to_group_chat()
                    send_to_group_chat()
            elif second_mod_box_mode:
                if mod_boxes >= 3:
                    send_to_second_chat()
            elif self_run:
                if points >= 3:
                    send_to_second_chat()
            elif custom_group_run:
                if points >= min_point:
                    send_to_group_chat()
            else:
                # Normal mode
                if state_filter == 0:
                    if points >= min_point:
                        send_to_world_chat()
                else:
                    if state == state_filter and points >= min_point:
                        send_to_world_chat()
                    elif state == 0 and points >= min_point:
                        send_to_second_chat()



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
