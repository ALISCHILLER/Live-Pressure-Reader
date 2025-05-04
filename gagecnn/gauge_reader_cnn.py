# gauge_reader_cnn.py
import cv2
import numpy as np
import math
import mss
import pyautogui
import sys
import json
import time
from needle_detector_cnn import load_needle_model

sys.stdout.reconfigure(encoding='utf-8')

# متغیرهای جهانی برای انتخاب ناحیه
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

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

def detect_gauge_circle(image):
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=200
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]
        return (x, y), r
    return None, None

def detect_needle_cnn(frame, model):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (64, 64)) / 255.0
    img_input = np.expand_dims(np.expand_dims(img_resized, axis=-1), axis=0)

    pred = model.predict(img_input, verbose=0)[0]
    x_pred = int(pred[0] * frame.shape[1])
    y_pred = int(pred[1] * frame.shape[0])

    return (x_pred, y_pred)

def calculate_angle(center, tip):
    dx = tip[0] - center[0]
    dy = tip[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360

def calculate_micrometer_displacement(angle, initial_angle, divisions_per_circle, mm_per_division):
    delta_angle = (angle - initial_angle + 360) % 360
    angle_per_division = 360 / divisions_per_circle
    passed_divisions = delta_angle / angle_per_division
    displacement_mm = passed_divisions * mm_per_division
    return displacement_mm * 1000  # to micrometers

def draw_info(frame, angle, displacement_um, center, tip, status="OK"):
    cv2.putText(frame, f"Angle: {angle:.2f} deg", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Displacement: {displacement_um:.3f} µm", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Status: {status}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if status != "OK" else (0, 255, 0), 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    if tip is not None:
        cv2.circle(frame, tip, 5, (0, 255, 255), -1)
    return frame

def save_region(region, filename="region.json"):
    with open(filename, "w") as f:
        json.dump(region, f)

def load_region(filename="region.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main():
    global selection_start, selection_end

    # تنظیمات گیج
    divisions_per_circle = 200
    mm_per_division = 0.001  # 1 میکرومتر

    monitor_region = load_region()
    if monitor_region is None:
        screen = pyautogui.screenshot()
        screen_np = np.array(screen)
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
        clone = screen_np.copy()
        cv2.namedWindow("Select Region")
        cv2.setMouseCallback("Select Region", draw_rectangle)

        print("ناحیه گیج را انتخاب کنید...")
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
        save_region(monitor_region)

    sct = mss.mss()
    initial_angle = None
    previous_displacement_um = 0

    # بارگذاری مدل CNN
    model = load_needle_model()

    try:
        while True:
            start_time = time.time()
            sct_img = sct.grab(monitor_region)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            processed = preprocess_frame(frame)
            edges = cv2.Canny(processed, 50, 150)

            # تشخیص دایره و مرکز
            center, radius = detect_gauge_circle(processed)
            if center is None:
                continue

            # تشخیص عقربه با CNN
            tip = detect_needle_cnn(frame, model)

            status = "OK"
            if tip is not None:
                angle = calculate_angle(center, tip)

                if initial_angle is None:
                    initial_angle = angle

                displacement_um = calculate_micrometer_displacement(
                    angle, initial_angle, divisions_per_circle, mm_per_division
                )

                if abs(displacement_um - previous_displacement_um) > 5:
                    displacement_um = previous_displacement_um
                    status = "Displacement Limited"
                previous_displacement_um = displacement_um

                print(f"🔄 زاویه: {angle:.2f}° | جابجایی: {displacement_um:.3f} µm | وضعیت: {status}", end='\r')
                frame = draw_info(frame, angle, displacement_um, center, tip, status)
            else:
                status = "Needle Not Detected"
                frame = draw_info(frame, 0, previous_displacement_um, center, None, status)
                print(f"🔄 زاویه: N/A | جابجایی: {previous_displacement_um:.3f} µm | وضعیت: {status}", end='\r')

            cv2.imshow("Live Gauge Reader", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('r'):
                initial_angle = None
                print("\nکالیبراسیون ریست شد.")

            elapsed = time.time() - start_time
            if elapsed < 0.03:
                time.sleep(0.03 - elapsed)

    finally:
        cv2.destroyAllWindows()
        sct.close()

if __name__ == "__main__":
    main()