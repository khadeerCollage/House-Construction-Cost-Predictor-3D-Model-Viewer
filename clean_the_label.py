# Remove images without corresponding label files
import os

image_folders = [
    "C:/Users/USER/Desktop/vit_project/yolov5/images/train",
    "C:/Users/USER/Desktop/vit_project/yolov5/images/val",
    "C:/Users/USER/Desktop/vit_project/yolov5/images/test"
]
label_folders = [
    "C:/Users/USER/Desktop/vit_project/yolov5/labels/train",
    "C:/Users/USER/Desktop/vit_project/yolov5/labels/val",
    "C:/Users/USER/Desktop/vit_project/yolov5/labels/test"
]

for img_folder, lbl_folder in zip(image_folders, label_folders):
    if not os.path.exists(img_folder) or not os.path.exists(lbl_folder):
        continue
    for img_file in os.listdir(img_folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(img_file)[0]
            label_path = os.path.join(lbl_folder, base + ".txt")
            img_path = os.path.join(img_folder, img_file)
            if not os.path.exists(label_path):
                print(f"Removing image without label: {img_path}")
                os.remove(img_path)