import cv2
import numpy as np
import math
import mss
import pyautogui
import sys
sys.stdout.reconfigure(encoding='utf-8')
# متغیر جهانی برای انتخاب ناحیه
selection_in_progress = False
selection_start = (0, 0)
selection_end = (0, 0)

def draw_rectangle(event, x, y, flags, param):
    global selection_in_progress, selection_start, selection_end
    if event == cv2.EVENT_LBUTTONDOWN:
        selection_in_progress = True
        selection_start = (x, y)
        selection_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and selection_in_progress:
        selection_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selection_in_progress = False
        selection_end = (x, y)

def extract_needle_angle(frame, center):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=5)
    best_angle = None
    max_len = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            dist1 = math.hypot(x1 - center[0], y1 - center[1])
            dist2 = math.hypot(x2 - center[0], y2 - center[1])
            if (dist1 < 30 or dist2 < 30) and length > max_len:
                max_len = length
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                best_angle = angle
    return best_angle

def draw_info(frame, angle, mm, center):
    cv2.putText(frame, f"Angle: {angle:.2f} deg", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Displacement: {mm:.3f} mm", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    return frame

def main():
    global selection_start, selection_end

    # گرفتن اسکرین‌شات از دسکتاپ برای انتخاب ناحیه
    screen = pyautogui.screenshot()
    screen_np = np.array(screen)
    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)

    clone = screen_np.copy()
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", draw_rectangle)

    print("با ماوس دور ناحیه گیج رو بکش. وقتی انتخاب تموم شد، کلید Enter رو بزن.")

    while True:
        temp = clone.copy()
        if selection_in_progress or selection_end != selection_start:
            cv2.rectangle(temp, selection_start, selection_end, (0, 255, 0), 2)
        cv2.imshow("Select Region", temp)
        key = cv2.waitKey(1)
        if key == 13:  # Enter
            break

    cv2.destroyWindow("Select Region")

    x1, y1 = selection_start
    x2, y2 = selection_end
    left, top = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)

    monitor_region = {"top": top, "left": left, "width": width, "height": height}
    center = (width // 2, height // 2)
    FULL_SCALE_MM = 1.0

    print(f"\nمنطقه انتخاب شد: {monitor_region}")
    print("شروع مانیتورینگ زنده...\n")

    sct = mss.mss()
    initial_angle = None

    while True:
        sct_img = sct.grab(monitor_region)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        angle = extract_needle_angle(frame, center)
        if angle is not None:
            angle = (angle + 360) % 360
            if initial_angle is None:
                initial_angle = angle

            delta_angle = (angle - initial_angle) % 360
            mm_moved = delta_angle / 360.0 * FULL_SCALE_MM

            # ✅ چاپ زنده در کنسول
            print(f"🔁 زاویه فعلی: {angle:.2f}° | جابجایی: {mm_moved:.3f} mm", end='\r')

            # نمایش در تصویر
            frame = draw_info(frame, angle, mm_moved, center)

        cv2.imshow("Live Gauge Reader", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
