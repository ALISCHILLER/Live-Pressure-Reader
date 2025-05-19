import cv2
import numpy as np
import math
import mss
import pyautogui
import sys

# اطمینان از پشتیبانی یونیکد برای چاپ فارسی
sys.stdout.reconfigure(encoding='utf-8')

# متغیرهای جهانی برای انتخاب ناحیه
selection_in_progress = False
selection_start = (0, 0)
selection_end = (0, 0)

def draw_rectangle(event, x, y, flags, param):
    """
    رسم مستطیل برای انتخاب ناحیه
    """
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
    """
    پیش‌پردازش تصویر برای بهبود دقت تشخیص عقربه
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)  # فیلتر بیشتر برای کاهش نویز
    edges = cv2.Canny(blur, 75, 150)          # آستانه‌های بهتر برای لبه‌یابی
    return edges

def detect_needle_tip(edges, center, radius):
    """
    تشخیص نوک عقربه با استفاده از Hough Transform
    """
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
    best_point = None
    max_distance = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # محاسبه فاصله از مرکز
            dist1 = math.hypot(x1 - center[0], y1 - center[1])
            dist2 = math.hypot(x2 - center[0], y2 - center[1])
            
            # انتخاب نقطه دورتر (نوک عقربه)
            if dist1 > max_distance and abs(dist1 - radius) < 10:  # تقریباً روی محیط دایره
                max_distance = dist1
                best_point = (x1, y1)
            if dist2 > max_distance and abs(dist2 - radius) < 10:
                max_distance = dist2
                best_point = (x2, y2)
    
    return best_point

def calculate_angle(center, tip):
    """
    محاسبه زاویه عقربه با استفاده از مختصات نوک عقربه
    """
    dx = tip[0] - center[0]
    dy = tip[1] - center[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360

def calculate_nanometer_displacement(angle, initial_angle, divisions_per_circle, nm_per_division):
    """
    محاسبه جابجایی نوک عقربه به نانومتر
    """
    delta_angle = (angle - initial_angle + 360) % 360
    angle_per_division = 360 / divisions_per_circle
    passed_divisions = delta_angle / angle_per_division
    displacement_nm = passed_divisions * nm_per_division
    return displacement_nm

def draw_info(frame, angle, displacement_nm, center, tip):
    """
    نمایش اطلاعات روی تصویر
    """
    cv2.putText(frame, f"Angle: {angle:.2f} deg", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Displacement: {displacement_nm:.2f} nm", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)  # مرکز دستکتاب
    cv2.circle(frame, tip, 5, (0, 255, 255), -1)  # نوک عقربه
    return frame

def main():
    global selection_start, selection_end

    # گرفتن اسکرین‌شات برای انتخاب ناحیه
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
        if key == 13:  # Enter
            break

    cv2.destroyWindow("Select Region")

    # محاسبه ناحیه انتخاب‌شده
    x1, y1 = selection_start
    x2, y2 = selection_end
    left, top = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)

    monitor_region = {"top": top, "left": left, "width": width, "height": height}
    center = (width // 2, height // 2)
    radius = min(width, height) // 2  # شعاع دایره

    # تنظیمات مقیاس نانومتری
    divisions_per_circle = 3600  # هر 0.1 درجه یک تقسیم
    nm_per_division = 1          # هر تقسیم = 1 نانومتر

    print(f"\nمنطقه انتخاب شد: {monitor_region}")
    print("شروع مانیتورینگ زنده...\n")

    sct = mss.mss()
    initial_angle = None
    previous_angle = None
    previous_displacement_nm = 0

    while True:
        # گرفتن تصویر زنده
        sct_img = sct.grab(monitor_region)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # پیش‌پردازش تصویر
        edges = preprocess_frame(frame)

        # تشخیذ نوک عقربه
        tip = detect_needle_tip(edges, center, radius)
        if tip is not None:
            angle = calculate_angle(center, tip)

            # فیلتر کردن نویز
            if previous_angle is not None:
                angle = (angle + previous_angle * 4) / 5  # میانگین‌گیری ساده
            previous_angle = angle

            if initial_angle is None:
                initial_angle = angle

            displacement_nm = calculate_nanometer_displacement(
                angle, initial_angle, divisions_per_circle, nm_per_division
            )

            # محدود کردن جابجایی‌های غیرمنطقی
            if displacement_nm > 3000 or displacement_nm < -3000:
                displacement_nm = previous_displacement_nm
            previous_displacement_nm = displacement_nm

            # چاپ زنده در کنسول
            print(f"🔄 زاویه فعلی: {angle:.2f}° | جابجایی: {displacement_nm:.2f} nm", end='\r')

            # نمایش اطلاعات روی تصویر
            frame = draw_info(frame, angle, displacement_nm, center, tip)

        # نمایش تصویر زنده
        cv2.imshow("Live Gauge Reader", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()