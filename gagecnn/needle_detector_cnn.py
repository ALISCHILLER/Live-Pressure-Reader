# needle_detector_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def build_model(input_shape=(64, 64, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)  # خروجی: (x, y)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def load_needle_model(model_path="needle_detector.h5"):
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        print("❌ مدل پیدا نشد. یک مدل جدید ساخته می‌شود.")
        return build_model()