import cv2
import numpy as np
import math
import mss
import pyautogui
import sys
import csv
from datetime import datetime
from filterpy.kalman import KalmanFilter

# پشتیبانی از چاپ فارسی
sys.stdout.reconfigure(encoding='utf-8')

# متغیرهای جهانی
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

def find_gauge_center(edges):
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=150,
        param1=100,
        param2=30,
        minRadius=50,
        maxRadius=200
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return (circles[0][0], circles[0][1])  # فقط اولین دایره
    return None

def kalman_filter_init():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=np.float32)  # Measurement function
    kf.P *= 1000.
    kf.R = np.array([[5, 0],
                     [0, 5]], dtype=np.float32)  # Measurement noise
    kf.Q = np.eye(4, dtype=np.float32) * 0.01  # Process noise
    return kf

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 75, 150)
    return edges

def detect_needle_tip(edges, center, radius):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
    best_point = None
    max_distance = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dist1 = math.hypot(x1 - center[0], y1 - center[1])
            dist2 = math.hypot(x2 - center[0], y2 - center[1])

            if dist1 > max_distance and abs(dist1 - radius) < 10:
                max_distance = dist1
                best_point = (x1, y1)
            if dist2 > max_distance and abs(dist2 - radius) < 10:
                max_distance = dist2
                best_point = (x2, y2)
    return best_point

def calculate_angle(center, tip):
    dx = float(tip[0] - center[0])
    dy = float(tip[1] - center[1])
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360

def calculate_nanometer_displacement(angle, initial_angle, divisions_per_circle, nm_per_division):
    delta_angle = (angle - initial_angle + 360) % 360
    angle_per_division = 360 / divisions_per_circle
    passed_divisions = delta_angle / angle_per_division
    displacement_nm = passed_divisions * nm_per_division
    return displacement_nm

def draw_info(frame, angle, displacement_nm, center, tip):
    cv2.putText(frame, f"Angle: {angle:.2f}°", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Displacement: {displacement_nm:.2f} nm", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # تضمین نوع صحیح برای OpenCV
    center_tuple = (int(center[0]), int(center[1]))
    tip_tuple = (int(tip[0]), int(tip[1]))

    cv2.circle(frame, center_tuple, 5, (0, 0, 255), -1)     # مرکز
    cv2.circle(frame, tip_tuple, 5, (0, 255, 255), -1)       # نوک عقربه
    return frame

def main():
    global selection_start, selection_end

    screen = pyautogui.screenshot()
    screen_np = np.array(screen)
    screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)

    clone = screen_np.copy()
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", draw_rectangle)

    print("با ماوس دور ناحیه دستکتاب نگاه را انتخاب کنید. پس از اتمام، کلید Enter را بزنید.")

    while True:
        temp = clone.copy()
        if selection_in_progress or selection_end != selection_start:
            cv2.rectangle(temp, selection_start, selection_end, (0, 255, 0), 2)
        cv2.imshow("Select Region", temp)
        key = cv2.waitKey(1)
        if key == 13:
            break

    cv2.destroyWindow("Select Region")

    x1, y1 = selection_start
    x2, y2 = selection_end
    left, top = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)

    monitor_region = {"top": top, "left": left, "width": width, "height": height}
    sct = mss.mss()

    # شناسایی مرکز فشارسنج
    while True:
        sct_img = sct.grab(monitor_region)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        edges = preprocess_frame(frame)
        center = find_gauge_center(edges)
        if center is not None:
            print(f"[✓] مرکز فشارسنج شناسایی شد: {center}")
            radius = int(min(width, height) // 2 * 0.9)
            break
        else:
            print("[!] منتظر شناسایی مرکز فشارسنج...")

    # تنظیمات محاسبه
    divisions_per_circle = 3600
    nm_per_division = 1
    initial_angle = None
    previous_displacement_nm = 0

    # فیلتر Kalman
    kf = kalman_filter_init()

    # فایل لاگ
    log_file = f"gauge_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "angle_deg", "displacement_nm"])

        while True:
            sct_img = sct.grab(monitor_region)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            edges = preprocess_frame(frame)
            tip = detect_needle_tip(edges, center, radius)

            if tip is not None:
                raw_angle = calculate_angle(center, tip)

                # فیلتر Kalman
                kf.predict()
                kf.update(np.array([tip[0], tip[1]]))
                x, y = kf.x[:2].astype(int)
                filtered_tip = (int(x), int(y))  # تبدیل دقیق به int
                angle = calculate_angle(center, filtered_tip)

                if initial_angle is None:
                    initial_angle = angle

                displacement_nm = calculate_nanometer_displacement(
                    angle, initial_angle, divisions_per_circle, nm_per_division
                )

                if displacement_nm > 3000 or displacement_nm < -3000:
                    displacement_nm = previous_displacement_nm
                previous_displacement_nm = displacement_nm

                timestamp = datetime.now().isoformat()
                writer.writerow([timestamp, angle, displacement_nm])

                print(f"\r🔄 زاویه: {angle:.2f}° | جابجایی: {displacement_nm:.2f} nm", end='')

                frame = draw_info(frame, angle, displacement_nm, center, filtered_tip)

            cv2.imshow("Live Gauge Reader", frame)
            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()