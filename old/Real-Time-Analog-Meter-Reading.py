import cv2
import numpy as np
import time

def calculate_angle(center, needle_tip):
    dx = needle_tip[0] - center[0]
    dy = center[1] - needle_tip[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    angle_deg = (angle_deg + 360) % 360
    return angle_deg

def map_angle_to_displacement(angle, min_angle, max_angle, min_disp, max_disp):
    angle_range = max_angle - min_angle
    disp_range = max_disp - min_disp
    displacement = ((angle - min_angle) / angle_range) * disp_range + min_disp
    return displacement

def detect_needle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines is None:
        return None

    max_len = 0
    needle = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > max_len:
            max_len = length
            needle = ((x1, y1), (x2, y2))
    return needle

def main():
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)

    min_angle = 45
    max_angle = 315
    min_disp = 0
    max_disp = 1000

    prev_time = time.time()
    fps_counter = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        center = (width // 2, height // 2)

        needle = detect_needle(frame)
        if needle:
            (x1, y1), (x2, y2) = needle
            tip = (x2, y2) if np.hypot(x2 - center[0], y2 - center[1]) > np.hypot(x1 - center[0], y1 - center[1]) else (x1, y1)
            angle = calculate_angle(center, tip)
            displacement_um = map_angle_to_displacement(angle, min_angle, max_angle, min_disp, max_disp)
            
            status = "مقدار عادی" if 200 < displacement_um < 800 else "هشدار"

            # رسم مرکز و عقربه
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.line(frame, center, tip, (0, 255, 0), 2)

            # چاپ اطلاعات به زبان فارسی در ترمینال (بدون ایموجی)
            print(f"زاویه: {angle:.2f} درجه | جابجایی: {displacement_um:.3f} میکرومتر | وضعیت: {status}", end='\r')

        # محاسبه FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            prev_time = current_time

        # نمایش FPS روی تصویر
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Analog Meter Reader", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
