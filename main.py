from ultralytics import YOLO
import cv2
import cv2
import numpy as np
import pyautogui
import time
import win32gui
import win32ui
import win32con

model = YOLO("runs/detect/train/weights/best.pt")
classes = model.names

def find_window(title):
    hwnd = win32gui.FindWindow(None, title)
    if hwnd == 0:
        return None
    return hwnd

def list_windows():
    def callback(hwnd, windows):
        title = win32gui.GetWindowText(hwnd)
        if title:
            print(hwnd, ":", title)
    win32gui.EnumWindows(callback, None)

def capture_window(hwnd):
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(bitmap)
    saveDC.BitBlt(
        (0,0),
        (width,height),
        mfcDC,
        (0,0),
        win32con.SRCCOPY
    )
    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype='uint8')
    img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    win32gui.DeleteObject(bitmap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    return img

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

