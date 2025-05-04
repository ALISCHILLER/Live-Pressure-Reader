# label_samples.py
import cv2
import os
import csv

sample_folder = "samples"
output_csv = "labels.csv"

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"📌 نقطه مشخص شد: ({x}, {y})")
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Labeling Tool", img)

files = [f for f in os.listdir(sample_folder) if f.endswith(".jpg")]
files.sort()

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "x", "y"])

    for filename in files:
        img_path = os.path.join(sample_folder, filename)
        img = cv2.imread(img_path)

        print(f"\n🖼️ لطفاً روی نوک عقربه در تصویر '{filename}' کلیک کنید.")
        cv2.imshow("Labeling Tool", img)
        cv2.setMouseCallback("Labeling Tool", click_event)

        while True:
            key = cv2.waitKey(1)
            if key == 13:  # Enter
                if len(points) > 0:
                    last_point = points[-1]
                    writer.writerow([filename, last_point[0], last_point[1]])
                    points.clear()
                    break
                else:
                    print("❗ لطفاً یک نقطه روی تصویر انتخاب کنید.")
            elif key == 27:  # Esc
                print("❌ انصراف از این تصویر.")
                points.clear()
                break

cv2.destroyAllWindows()
print("✅ تمامی برچسب‌ها ذخیره شدند.")