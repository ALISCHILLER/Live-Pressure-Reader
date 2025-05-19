import cv2
import numpy as np
import math
import mss
import pyautogui
import sys
import json
import time
from scipy.signal import savgol_filter  # برای صاف کردن داده‌ها

# تنظیمات چاپ فارسی
sys.stdout.reconfigure(encoding='utf-8')

# متغیرهای جهانی برای انتخاب ناحیه
selection_in_progress = False
selection_start = (0, 0)
selection_end = (0, 0)

def draw_rectangle(event, x, y, flags, param):
    """رسم مستطیل برای انتخاب ناحیه"""
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
    """پیش‌پردازش تصویر برای بهبود تشخیص عقربه"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # افزایش کنتراست با CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    return gray, edges

def detect_needle_line(edges, center, radius, min_line_length=30):
    """تشخیص خط عقربه با Hough Line Transform و استفاده از کانتور به عنوان مکمل"""
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=10)
    best_tip = None
    max_dist = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # فاصله دو سر خط تا مرکز
            dist1 = math.hypot(x1 - center[0], y1 - center[1])
            dist2 = math.hypot(x2 - center[0], y2 - center[1])
            # انتخاب نقطه‌ای که نزدیک به شعاع انتظار باشد و دورتر از مرکز باشد
            if abs(dist1 - radius) < 20 and dist1 > max_dist:
                max_dist = dist1
                best_tip = (x1, y1)
            if abs(dist2 - radius) < 20 and dist2 > max_dist:
                max_dist = dist2
                best_tip = (x2, y2)
    # در صورت عدم یافتن خط مناسب، تلاش با کانتور
    if best_tip is None:
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 30:
                continue
            for pt in cnt:
                pt = pt[0]
                dist = math.hypot(pt[0] - center[0], pt[1] - center[1])
                if abs(dist - radius) < 20 and dist > max_dist:
                    max_dist = dist
                    best_tip = (pt[0], pt[1])
    return best_tip

def calculate_angle(center, tip):
    """محاسبه زاویه عقربه"""
    dx = tip[0] - center[0]
    dy = tip[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360

def calculate_micrometer_displacement(angle, initial_angle, divisions_per_circle, mm_per_division):
    """محاسبه جابجایی به میکرومتر"""
    delta_angle = (angle - initial_angle + 360) % 360
    angle_per_division = 360 / divisions_per_circle
    passed_divisions = delta_angle / angle_per_division
    displacement_mm = passed_divisions * mm_per_division
    return displacement_mm * 1000  # تبدیل به میکرومتر

def draw_info(frame, angle, displacement_um, center, tip, status="OK"):
    """نمایش اطلاعات روی تصویر"""
    cv2.putText(frame, f"Angle: {angle:.2f} deg", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Displ.: {displacement_um:.3f} um", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Status: {status}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if status != "OK" else (0, 255, 0), 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)  # مرکز
    if tip is not None:
        cv2.circle(frame, tip, 5, (0, 255, 255), -1)  # نوک عقربه
    return frame

def save_region(region, filename="region.json"):
    """ذخیره ناحیه انتخاب‌شده"""
    with open(filename, "w") as f:
        json.dump(region, f)

def load_region(filename="region.json"):
    """بارگذاری ناحیه ذخیره‌شده"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# فیلتر Kalman ساده برای صاف کردن زاویه
class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-3, measurement_variance=1e-2, initial_value=0.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = initial_value
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        kalman_gain = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate
        return self.posteri_estimate

def main():
    global selection_start, selection_end

    # تنظیمات گیج: 200 تقسیم‌بندی، هر تقسیم = 0.001 میلی‌متر
    divisions_per_circle = 200
    mm_per_division = 0.001  # یعنی هر تقسیم 1 میکرومتر است

    # بررسی ناحیه ذخیره‌شده
    monitor_region = load_region()
    if monitor_region is None:
        # گرفتن اسکرین‌شات برای انتخاب ناحیه
        screen = pyautogui.screenshot()
        screen_np = np.array(screen)
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)

        clone = screen_np.copy()
        cv2.namedWindow("Select Region")
        cv2.setMouseCallback("Select Region", draw_rectangle)

        print("با ماوس ناحیه گیج را انتخاب کنید. پس از اتمام، کلید Enter را بزنید.")
        while True:
            temp = clone.copy()
            if selection_in_progress or selection_end != selection_start:
                cv2.rectangle(temp, selection_start, selection_end, (0, 255, 0), 2)
            cv2.imshow("Select Region", temp)
            key = cv2.waitKey(1)
            if key == 13:  # Enter
                break
        cv2.destroyWindow("Select Region")

        # محاسبه ناحیه
        x1, y1 = selection_start
        x2, y2 = selection_end
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        monitor_region = {"top": top, "left": left, "width": width, "height": height}
        save_region(monitor_region)

    center = (monitor_region["width"] // 2, monitor_region["height"] // 2)
    radius = min(monitor_region["width"], monitor_region["height"]) // 2

    print(f"\nمنطقه انتخاب شد: {monitor_region}")
    print("شروع مانیتورینگ زنده...\n")

    sct = mss.mss()
    initial_angle = None
    angle_history = []  # برای صاف کردن زاویه با Savitzky-Golay (اختیاری)
    kalman_filter = SimpleKalmanFilter(initial_value=0.0)  # استفاده از فیلتر Kalman
    previous_displacement_um = 0

    try:
        while True:
            start_time = time.time()
            # گرفتن تصویر زنده
            sct_img = sct.grab(monitor_region)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # پیش‌پردازش تصویر
            gray, edges = preprocess_frame(frame)

            # تشخیص عقربه
            tip = detect_needle_line(edges, center, radius, min_line_length=radius // 2)
            status = "OK"
            if tip is not None:
                angle = calculate_angle(center, tip)

                # صاف کردن زاویه با فیلتر Savitzky-Golay (اختیاری) و Kalman filter
                angle_history.append(angle)
                if len(angle_history) > 15:
                    angle_history.pop(0)
                if len(angle_history) >= 5:
                    smoothed_angles = savgol_filter(angle_history, window_length=5, polyorder=2)
                    angle = smoothed_angles[-1]
                # اعمال Kalman filter برای صاف‌تر شدن تغییرات
                angle = kalman_filter.update(angle)

                # کنترل پرش‌های ناگهانی زاویه (در صورت نیاز)
                if initial_angle is None:
                    initial_angle = angle
                else:
                    if len(angle_history) >= 2:
                        delta = (angle - angle_history[-2] + 360) % 360
                        if delta > 45 and delta < 315:  # پرش غیرمنطقی؛ استفاده از زاویه قبلی
                            angle = angle_history[-2]
                            status = "Angle Jump Detected"

                displacement_um = calculate_micrometer_displacement(
                    angle, initial_angle, divisions_per_circle, mm_per_division
                )

                # محدود کردن تغییرات جابجایی غیرمنطقی
                if abs(displacement_um - previous_displacement_um) > 50:
                    displacement_um = previous_displacement_um
                    status = "Displacement Limited"
                previous_displacement_um = displacement_um

                # چاپ اطلاعات
                print(f"🔄 زاویه: {angle:.2f}° | جابجایی: {displacement_um:.3f} um | وضعیت: {status}", end='\r')

                # نمایش اطلاعات روی تصویر
                frame = draw_info(frame, angle, displacement_um, center, tip, status)
            else:
                status = "Needle Not Detected"
                frame = draw_info(frame, 0, previous_displacement_um, center, None, status)
                print(f"🔄 زاویه: N/A | جابجایی: {previous_displacement_um:.3f} um | وضعیت: {status}", end='\r')

            # نمایش تصویر زنده
            cv2.imshow("Live Gauge Reader", frame)
            key = cv2.waitKey(1)
            if key == 27:  # خروج با ESC
                break
            elif key == ord('r'):
                initial_angle = None
                angle_history = []
                kalman_filter = SimpleKalmanFilter(initial_value=0.0)
                print("\nکالیبراسیون ریست شد.")

            # کنترل نرخ فریم
            elapsed = time.time() - start_time
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
    finally:
        cv2.destroyAllWindows()
        sct.close()

if __name__ == "__main__":
    main()
