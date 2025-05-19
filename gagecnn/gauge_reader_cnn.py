import cv2
import numpy as np
import math
import mss
import pyautogui
import json
import time
from needle_detector_cnn import load_needle_model
import sys

sys.stdout.reconfigure(encoding='utf-8')


class GaugeReader:
    def __init__(self, divisions_per_circle=200, mm_per_division=0.001, region_file="region.json"):
        self.divisions_per_circle = divisions_per_circle
        self.mm_per_division = mm_per_division
        self.region_file = region_file

        self.selection_in_progress = False
        self.selection_start = (0, 0)
        self.selection_end = (0, 0)

        self.initial_angle = None
        self.previous_displacement_um = 0

        # بارگذاری مدل CNN سوزن
        self.model = load_needle_model()
        # بارگذاری ناحیه نمایش از فایل یا انتخاب ناحیه
        self.monitor_region = self.load_region()
        if self.monitor_region is None:
            self.select_region()

        # ایجاد شیء mss برای گرفتن اسکرین‌شات سریع
        self.sct = mss.mss()

    def draw_rectangle(self, event, x, y, flags, param):
        """تابع callback برای رسم مستطیل انتخاب ناحیه"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_in_progress = True
            self.selection_start = (x, y)
            self.selection_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.selection_in_progress:
            self.selection_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selection_in_progress = False
            self.selection_end = (x, y)

    def select_region(self):
        """اجازه انتخاب ناحیه گیج توسط کاربر"""
        screen = pyautogui.screenshot()
        screen_np = np.array(screen)
        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
        clone = screen_np.copy()

        cv2.namedWindow("Select Region")
        cv2.setMouseCallback("Select Region", self.draw_rectangle)

        print("ناحیه گیج را انتخاب کنید و Enter را فشار دهید...")

        while True:
            temp = clone.copy()
            if self.selection_in_progress or self.selection_end != self.selection_start:
                cv2.rectangle(temp, self.selection_start, self.selection_end, (0, 255, 0), 2)
            cv2.imshow("Select Region", temp)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break

        cv2.destroyWindow("Select Region")

        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        self.monitor_region = {"top": top, "left": left, "width": width, "height": height}
        self.save_region(self.monitor_region)

    def save_region(self, region):
        """ذخیره ناحیه انتخاب شده در فایل json"""
        with open(self.region_file, "w", encoding='utf-8') as f:
            json.dump(region, f, ensure_ascii=False)

    def load_region(self):
        """بارگذاری ناحیه ذخیره شده"""
        try:
            with open(self.region_file, "r", encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    @staticmethod
    def preprocess_frame(frame):
        """پیش‌پردازش تصویر برای تشخیص بهتر دایره گیج"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    @staticmethod
    def detect_gauge_circle(image):
        """تشخیص دایره گیج با روش HoughCircles"""
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

    def detect_needle_cnn(self, frame):
        """استفاده از مدل CNN برای تشخیص نوک سوزن"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img, (64, 64)) / 255.0
        img_input = np.expand_dims(np.expand_dims(img_resized, axis=-1), axis=0)

        pred = self.model.predict(img_input, verbose=0)[0]
        x_pred = int(pred[0] * frame.shape[1])
        y_pred = int(pred[1] * frame.shape[0])

        return (x_pred, y_pred)

    @staticmethod
    def calculate_angle(center, tip):
        """محاسبه زاویه نوک سوزن نسبت به مرکز دایره"""
        dx = tip[0] - center[0]
        dy = tip[1] - center[1]
        angle = math.degrees(math.atan2(dy, dx))
        return (angle + 360) % 360

    def calculate_micrometer_displacement(self, angle):
        """محاسبه جابجایی میکرومتری بر اساس زاویه"""
        delta_angle = (angle - self.initial_angle + 360) % 360
        angle_per_division = 360 / self.divisions_per_circle
        passed_divisions = delta_angle / angle_per_division
        displacement_mm = passed_divisions * self.mm_per_division
        return displacement_mm * 1000  # تبدیل میلی‌متر به میکرومتر

    @staticmethod
    def draw_info(frame, angle, displacement_um, center, tip, status="OK"):
        """رسم اطلاعات روی تصویر"""
        cv2.putText(frame, f"Angle: {angle:.2f} deg", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Displacement: {displacement_um:.3f} µm", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {status}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if status != "OK" else (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        if tip is not None:
            cv2.circle(frame, tip, 5, (0, 255, 255), -1)
        return frame

    def run(self):
        print("شروع خواندن گیج...")

        try:
            while True:
                start_time = time.time()
                # گرفتن اسکرین‌شات از ناحیه گیج
                sct_img = self.sct.grab(self.monitor_region)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                processed = self.preprocess_frame(frame)

                center, radius = self.detect_gauge_circle(processed)
                if center is None:
                    print("دایره گیج تشخیص داده نشد.", end='\r')
                    cv2.imshow("Live Gauge Reader", frame)
                    if cv2.waitKey(1) == 27:
                        break
                    continue

                tip = self.detect_needle_cnn(frame)

                status = "OK"
                if tip is not None:
                    angle = self.calculate_angle(center, tip)

                    if self.initial_angle is None:
                        self.initial_angle = angle
                        print("کالیبراسیون اولیه انجام شد.")

                    displacement_um = self.calculate_micrometer_displacement(angle)

                    # محدودیت تغییرات ناگهانی جابجایی
                    if abs(displacement_um - self.previous_displacement_um) > 5:
                        displacement_um = self.previous_displacement_um
                        status = "Displacement Limited"
                    self.previous_displacement_um = displacement_um

                    print(f"🔄 زاویه: {angle:.2f}° | جابجایی: {displacement_um:.3f} µm | وضعیت: {status}   ", end='\r')
                    frame = self.draw_info(frame, angle, displacement_um, center, tip, status)
                else:
                    status = "Needle Not Detected"
                    frame = self.draw_info(frame, 0, self.previous_displacement_um, center, None, status)
                    print(f"🔄 زاویه: N/A | جابجایی: {self.previous_displacement_um:.3f} µm | وضعیت: {status}   ", end='\r')

                cv2.imshow("Live Gauge Reader", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # کلید ESC برای خروج
                    break
                elif key == ord('r'):
                    self.initial_angle = None
                    print("\nکالیبراسیون ریست شد.")

                # کنترل فریم ریت به حدود ۳۰ فریم بر ثانیه
                elapsed = time.time() - start_time
                if elapsed < 1/30:
                    time.sleep(1/30 - elapsed)

        finally:
            cv2.destroyAllWindows()
            self.sct.close()


if __name__ == "__main__":
    reader = GaugeReader()
    reader.run()
