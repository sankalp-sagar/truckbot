import cv2

drawing = False
ix, iy = -1, -1
rectangles = []
img = None
display = None
scale = 1.0


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        temp = display.copy()

        if drawing:
            cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)

        cv2.putText(
            temp,
            f"{int(x/scale)}, {int(y/scale)}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        cv2.imshow("Mapper", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(display, (ix, iy), (x, y), (0, 255, 0), 2)

        x1, y1 = int(ix/scale), int(iy/scale)
        x2, y2 = int(x/scale), int(y/scale)

        rectangles.append((x1, y1, x2, y2))
        print(f"Mapped area: ({x1}, {y1}) -> ({x2}, {y2})")


def map_coordinates(image_path="screenbox.png"):
    global img, display, scale

    img = cv2.imread(image_path)

    h, w = img.shape[:2]

    # limit window size so large screenshots fit
    max_width = 1200
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0

    display = img.copy()

    cv2.namedWindow("Mapper", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Mapper", mouse_callback)

    print("Instructions:")
    print("Drag mouse to create rectangle")
    print("Press r to reset")
    print("Press q to quit")

    while True:
        cv2.imshow("Mapper", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            display = img.copy()
            rectangles.clear()
            print("Reset")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return rectangles


areas = map_coordinates("sharebutton.png")
print("Final areas:", areas)