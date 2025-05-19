import cv2
import os
import csv
import sys
sys.stdout.reconfigure(encoding='utf-8')

sample_folder = r"E:\project-zar\custom\python\Live Pressure Reader\gagecnn\samples"
output_csv = "labels.csv"

point = None

def click_event(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point selected: ({x}, {y})")
        point = (x, y)

if not os.path.exists(sample_folder):
    print(f"Folder '{sample_folder}' not found!")
    exit(1)

files = [f for f in os.listdir(sample_folder) if f.endswith(".jpg")]
files.sort()

if len(files) == 0:
    print("No jpg images found in the samples folder!")
    exit(1)

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "x", "y"])

    total = len(files)
    for idx, filename in enumerate(files):
        img_path = os.path.join(sample_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not load image {filename}")
            continue

        point = None

        print(f"\nImage {idx + 1}/{total}: Please click on the needle tip in '{filename}'. Press Enter to confirm, Esc to skip.")

        cv2.imshow("Labeling Tool", img)
        cv2.setMouseCallback("Labeling Tool", click_event, param=img)

        while True:
            display_img = img.copy()
            if point is not None:
                cv2.circle(display_img, point, 5, (0, 0, 255), -1)
            
            cv2.putText(display_img, "Click needle tip, Enter=confirm, Esc=skip", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Labeling Tool", display_img)

            key = cv2.waitKey(20) & 0xFF
            if key == 13:  # Enter
                if point is not None:
                    writer.writerow([filename, point[0], point[1]])
                    print(f"Point saved: {point}")
                    break
                else:
                    print("Please select a point on the image before pressing Enter.")
            elif key == 27:  # Esc
                print("Skipping this image.")
                break

cv2.destroyAllWindows()
print("All labels have been saved.")
