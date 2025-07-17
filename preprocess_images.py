# src/preprocessing/preprocess_images.py
import os
import sys
from PIL import Image

def find_plant_folder(data_root="data"):
    
    for entry in os.listdir(data_root):
        path = os.path.join(data_root, entry)
        if entry.lower().startswith("plant") and os.path.isdir(path):
            return path
    raise FileNotFoundError(f"No 'Plant_...' folder found in {data_root}")

def preprocess_images(src_dir: str, dst_dir: str, size: tuple = (224, 224)):
    # Make sure destination exists
    os.makedirs(dst_dir, exist_ok=True)

    # Each subfolder under src_dir is a disease class
    classes = [
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    ]
    if not classes:
        print(f"[Error] No class subfolders found in {src_dir}")
        sys.exit(1)

    for cls in classes:
        in_folder  = os.path.join(src_dir, cls)
        out_folder = os.path.join(dst_dir, cls)
        os.makedirs(out_folder, exist_ok=True)

        count = 0
        for fname in os.listdir(in_folder):
            src_path = os.path.join(in_folder, fname)
            try:
                img = Image.open(src_path).convert('RGB')
                img = img.resize(size)
                img.save(os.path.join(out_folder, fname))
                count += 1
            except Exception as e:
                print(f"[Warning] Skipping {src_path}: {e}")
        print(f"[Done] {count} images processed for class '{cls}'")

if __name__ == "__main__":
    # Step 1: auto‑find your Plant dataset folder
    src_folder = find_plant_folder(data_root="data")
    dst_folder = os.path.join("data", "processed", "disease_images_224")

    print(f"[Info] Source images: {src_folder}")
    print(f"[Info] Destination folder: {dst_folder}")

    preprocess_images(src_dir=src_folder, dst_dir=dst_folder)
    print("[Success] All disease images have been resized and saved.")
