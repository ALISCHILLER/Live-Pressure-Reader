# save_samples.py
import pyautogui
import cv2
# save_samples.py
import pyautogui
import cv2
import numpy as np 
import os
import time
import os
import time

# پوشه ذخیره تصاویر
os.makedirs("samples", exist_ok=True)
print("[INFO] Capturing 20 samples... One every 3 seconds.")
for i in range(20):
    screenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"samples/sample_{i:03d}.jpg", img)
    print(f"[SAVED] sample_{i:03d}.jpg")
    time.sleep(3)