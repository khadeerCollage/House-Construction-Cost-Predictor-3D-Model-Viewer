import os
from pathlib import Path
import shutil
import random
from ultralytics import YOLO

# ⚙️ SETTINGS
random.seed(42)
base_dir = Path('C:/Users/USER/.cache/kagglehub/datasets/qmarva/cubicasa5k/versions/4/cubicasa5k/cubicasa5k')
output_base = Path('yolov5')  # Output folder

# 📁 Create output folder structure
for split in ['train', 'val', 'test']:
    (output_base / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output_base / 'labels' / split).mkdir(parents=True, exist_ok=True)

# 📦 Collect all images recursively
all_images = list(base_dir.rglob('*.png'))

# 🔀 Shuffle and split
random.shuffle(all_images)
n = len(all_images)
train_split = int(0.7 * n)
val_split = int(0.9 * n)

splits = {
    'train': all_images[:train_split],
    'val': all_images[train_split:val_split],
    'test': all_images[val_split:]
}

# Load YOLOv8 model (use your own weights if needed)
model = YOLO('yolov8n.pt')

# 🚀 Copy images and labels with unique names
for split, images in splits.items():
    for img_path in images:
        # Create a unique name based on relative path
        rel_path = img_path.relative_to(base_dir)
        unique_name = str(rel_path).replace(os.sep, '_')
        
        img_output_path = output_base / 'images' / split / unique_name
        shutil.copy2(img_path, img_output_path)

        # Label paths
        label_unique_name = unique_name.replace('.png', '.txt')
        label_output_path = output_base / 'labels' / split / label_unique_name

        # If label exists and is not empty, copy it; else, auto-label
        label_path = img_path.with_suffix('.txt')
        if label_path.exists() and label_path.stat().st_size > 0:
            shutil.copy2(label_path, label_output_path)
        else:
            # Auto-label using YOLO
            results = model(img_output_path)
            boxes = results[0].boxes
            if len(boxes) == 0:
                print(f"⚠️ No objects detected for {img_output_path}, label will be empty.")
            else:
                print(f"Auto-labeling {img_output_path} with {len(boxes)} boxes.")
            with open(label_output_path, 'w') as f:
                for box in boxes:
                    cls = int(box.cls)
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ Images and labels split into train/val/test folders with unique names and auto-labels!")




















