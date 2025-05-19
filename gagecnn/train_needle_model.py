import numpy as np
import cv2
import csv
import os
from needle_detector_cnn import build_model
import sys
import tensorflow as tf
import random

sys.stdout.reconfigure(encoding='utf-8')

IMAGE_WIDTH, IMAGE_HEIGHT = 64, 64
ORIG_WIDTH, ORIG_HEIGHT = 640, 480

samples_folder = r"E:\project-zar\custom\python\Live Pressure Reader\gagecnn\samples"

if not os.path.exists("labels.csv"):
    print("فایل labels.csv یافت نشد!")
    sys.exit(1)

if not os.path.exists(samples_folder):
    print(f"پوشه نمونه‌ها یافت نشد: {samples_folder}")
    sys.exit(1)

# تنظیم seed برای reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

X_train = []
y_train = []

print("شروع بارگذاری داده‌ها...")

with open("labels.csv", "r", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # رد کردن هدر
    for row in reader:
        filename, x_str, y_str = row
        img_path = os.path.join(samples_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"هشدار: تصویر '{filename}' بارگذاری نشد!")
            continue

        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32) / 255.0
        X_train.append(np.expand_dims(img, axis=-1))

        x_orig = float(x_str)
        y_orig = float(y_str)

        x_resized = x_orig * IMAGE_WIDTH / ORIG_WIDTH
        y_resized = y_orig * IMAGE_HEIGHT / ORIG_HEIGHT

        x_norm = x_resized / IMAGE_WIDTH
        y_norm = y_resized / IMAGE_HEIGHT

        y_train.append([x_norm, y_norm])

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

if len(X_train) == 0:
    print("هیچ نمونه آموزشی یافت نشد. لطفا داده‌ها را بررسی کنید.")
    sys.exit(1)

if len(X_train) != len(y_train):
    print("تعداد تصاویر و برچسب‌ها برابر نیست! لطفا داده‌ها را بررسی کنید.")
    sys.exit(1)

print(f"تعداد نمونه‌های بارگذاری شده: {len(X_train)}")
print(f"ابعاد داده‌های ورودی: {X_train.shape}")
print(f"ابعاد برچسب‌ها: {y_train.shape}")

print("در حال ساخت مدل...")
model = build_model()

print("شروع آموزش مدل...")

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = r"E:\project-zar\custom\python\Live Pressure Reader\gagecnn\best_model.keras"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor="val_loss",
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=8,
    validation_split=0.2,
    callbacks=[checkpoint]
)

print(f"مدل آموزش داده شد و بهترین نسخه آن در مسیر زیر ذخیره شد:\n{checkpoint_path}")

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save(r"E:\project-zar\custom\python\Live Pressure Reader\gagecnn\needle_model.keras")