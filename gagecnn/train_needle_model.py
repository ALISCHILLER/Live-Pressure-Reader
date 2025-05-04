# train_needle_model.py
import numpy as np
import cv2
from needle_detector_cnn import build_model

# بارگذاری داده‌ها
X_train = []
y_train = []

with open("labels.csv", "r") as f:
    lines = f.readlines()

for line in lines:
    filename, x_str, y_str = line.strip().split(',')
    img = cv2.imread(f"samples/{filename}", 0)
    img = cv2.resize(img, (64, 64)) / 255.0
    X_train.append(np.expand_dims(img, axis=-1))
    y_train.append([int(x_str), int(y_str)])

X_train = np.array(X_train)
y_train = np.array(y_train)

# نرمالایز کردن خروجی برای محدوده [0, 1]
y_train[:, 0] /= 640  # عرض تصویر
y_train[:, 1] /= 480  # ارتفاع تصویر

# ساخت و آموزش مدل
model = build_model()
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.2)
model.save("needle_detector.h5")
print("✅ مدل آموزش دیده و ذخیره شد!")