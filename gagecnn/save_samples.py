import pyautogui
import cv2
import numpy as np
import os
import time
import re

folder = r"E:\project-zar\custom\python\Live Pressure Reader\gagecnn\samples"
os.makedirs(folder, exist_ok=True)

existing_files = os.listdir(folder)
pattern = re.compile(r"sample_(\d{3})\.jpg")
numbers = [int(pattern.match(f).group(1)) for f in existing_files if pattern.match(f)]
start_index = max(numbers) + 1 if numbers else 0

num_samples = 100  # تعداد نمونه‌هایی که میخوایم ذخیره کنیم
interval_sec = 0.1  # فاصله زمانی بین هر عکس (ثانیه) = 100 میلی‌ثانیه
max_samples_on_disk = 200  # حداکثر تعداد نمونه روی دیسک

def clean_old_samples(folder, max_files=200):
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    if len(files) <= max_files:
        return  # نیاز به حذف نیست

    # مرتب‌سازی فایل‌ها بر اساس زمان ایجاد (قدیمی‌ترین‌ها اول)
    files = sorted(files, key=lambda f: os.path.getctime(os.path.join(folder, f)))

    # تعداد فایل‌هایی که باید حذف شوند
    num_to_delete = len(files) - max_files

    for i in range(num_to_delete):
        try:
            os.remove(os.path.join(folder, files[i]))
            print(f"[REMOVED] {files[i]}")
        except Exception as e:
            print(f"[ERROR] حذف فایل {files[i]} امکان‌پذیر نبود: {e}")

print(f"[INFO] Starting from sample_{start_index:03d}.jpg")
print(f"[INFO] Capturing {num_samples} samples... One every {interval_sec*1000:.0f} milliseconds.")

for i in range(start_index, start_index + num_samples):
    start_time = time.time()

    screenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    filename = os.path.join(folder, f"sample_{i:03d}.jpg")
    cv2.imwrite(filename, img)

    print(f"[SAVED] {filename}")

    # حذف فایل‌های قدیمی در صورت زیاد بودن تعداد نمونه‌ها
    clean_old_samples(folder, max_files=max_samples_on_disk)

    elapsed = time.time() - start_time
    sleep_time = interval_sec - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

print("[INFO] Done.")
