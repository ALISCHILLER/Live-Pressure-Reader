import tensorflow as tf
from tensorflow.keras import layers, models
import os
import logging
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

MODEL_PATH = r"E:\project-zar\custom\python\Live Pressure Reader\gagecnn\best_model.keras"
SAVE_PATH = r"E:\project-zar\custom\python\Live Pressure Reader\gagecnn\needle_model.keras"

def build_model(input_shape=(64, 64, 1), conv_filters=[32, 64], dense_units=128):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    for filters in conv_filters:
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(2))  # مختصات x, y

    model.compile(optimizer='adam', loss='mse')
    logging.info("مدل ساخته شد.")
    model.summary(print_fn=logging.info)
    return model

def load_needle_model(model_path=None):
    path = model_path or MODEL_PATH

    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            logging.info(f"مدل از مسیر '{path}' با موفقیت بارگذاری شد.")
            print(f"فایل مدل بارگذاری‌شده: {path}")
            return model
        except Exception as e:
            logging.error(f"خطا در بارگذاری مدل: {e}")
            logging.info("ساخت مدل جدید به جای مدل قبلی.")
            return build_model()
    else:
        logging.warning(f"مدل پیدا نشد در مسیر '{path}'، در حال ساخت مدل جدید.")
        return build_model()

if __name__ == "__main__":
    model = load_needle_model()
    dummy_input = tf.zeros((1, 64, 64, 1))
    prediction = model.predict(dummy_input)
    logging.info(f"پیش‌بینی مدل: x={prediction[0][0]:.2f}, y={prediction[0][1]:.2f}")

    try:
        model.save(SAVE_PATH)
        logging.info(f"مدل به مسیر '{SAVE_PATH}' ذخیره شد.")
    except Exception as e:
        logging.error(f"خطا در ذخیره‌سازی مدل: {e}")

    logging.info("برنامه پایان یافت.")