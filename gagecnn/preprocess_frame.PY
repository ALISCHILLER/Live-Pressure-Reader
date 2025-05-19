import numpy as np
import cv2
import tensorflow as tf
from needle_detector_cnn import build_model
import pygetwindow as gw
import pyautogui

IMAGE_WIDTH, IMAGE_HEIGHT = 64, 64

def load_model(path=r"E:\project-zar\custom\python\Live Pressure Reader\gagecnn\needle_model.keras"):
    try:
        model = tf.keras.models.load_model(path)
        print("مدل با موفقیت بارگذاری شد.")
    except Exception as e:
        print("خطا در بارگذاری مدل:", e)
        model = build_model()
    return model

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32) / 255.0
    input_arr = np.expand_dims(resized, axis=(0, -1))
    return input_arr

def main():
    model = load_model()

    prev_point = None

    print("شروع ضبط صفحه نمایش... (برای خروج کلید q را بزنید)")

    while True:
        # گرفتن اسکرین‌شات زنده (تمام صفحه)
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        height, width = frame.shape[:2]

        input_frame = preprocess_frame(frame)
        pred = model.predict(input_frame)[0]

        x_pixel = int(pred[0] * width)
        y_pixel = int(pred[1] * height)

        cv2.circle(frame, (x_pixel, y_pixel), 10, (0, 0, 255), -1)

        if prev_point is not None:
            dx = x_pixel - prev_point[0]
            dy = y_pixel - prev_point[1]

            if abs(dx) > 2 or abs(dy) > 2:
                direction = ""
                if dy < 0:
                    direction += "بالا "
                elif dy > 0:
                    direction += "پایین "
                if dx > 0:
                    direction += "راست"
                elif dx < 0:
                    direction += "چپ"
                cv2.putText(frame, f"حرکت عقربه: {direction.strip()}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"حرکت عقربه: {direction.strip()}")

        prev_point = (x_pixel, y_pixel)

        cv2.imshow("تحلیل زنده صفحه", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("خروج توسط کاربر")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
