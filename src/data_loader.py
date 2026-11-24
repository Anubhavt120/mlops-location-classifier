import os
import sys
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import requests
from PIL import Image, ImageDraw

import tensorflow as tf
from src.config import (
    CLASSES,
    IMG_SIZE,
    BATCH_SIZE,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    TEST_DATA_DIR,
    SAMPLE_IMAGES_DIR,
    CLASS_NAMES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co/datasets/sfarrukhm/intel-image-classification/resolve/main"

# Curated image filenames from the dataset for reliable downloading
SAMPLE_FILENAMES = {
    "buildings": [
        "data/seg_train/buildings/0.jpg", "data/seg_train/buildings/10006.jpg",
        "data/seg_train/buildings/1001.jpg", "data/seg_train/buildings/10014.jpg",
        "data/seg_train/buildings/10018.jpg", "data/seg_train/buildings/10029.jpg",
        "data/seg_train/buildings/10032.jpg", "data/seg_train/buildings/10056.jpg",
        "data/seg_train/buildings/1009.jpg", "data/seg_train/buildings/10103.jpg",
    ],
    "forest": [
        "data/seg_train/forest/10007.jpg", "data/seg_train/forest/10010.jpg",
        "data/seg_train/forest/10020.jpg", "data/seg_train/forest/1004.jpg",
        "data/seg_train/forest/10055.jpg", "data/seg_train/forest/10077.jpg",
        "data/seg_train/forest/10086.jpg", "data/seg_train/forest/10091.jpg",
        "data/seg_train/forest/10104.jpg", "data/seg_train/forest/10108.jpg",
    ],
    "glacier": [
        "data/seg_train/glacier/10.jpg", "data/seg_train/glacier/10003.jpg",
        "data/seg_train/glacier/10019.jpg", "data/seg_train/glacier/1002.jpg",
        "data/seg_train/glacier/10037.jpg", "data/seg_train/glacier/10051.jpg",
        "data/seg_train/glacier/10064.jpg", "data/seg_train/glacier/10070.jpg",
        "data/seg_train/glacier/10078.jpg", "data/seg_train/glacier/10088.jpg",
    ],
    "mountain": [
        "data/seg_train/mountain/10000.jpg", "data/seg_train/mountain/10002.jpg",
        "data/seg_train/mountain/10005.jpg", "data/seg_train/mountain/10012.jpg",
        "data/seg_train/mountain/10021.jpg", "data/seg_train/mountain/10025.jpg",
        "data/seg_train/mountain/10030.jpg", "data/seg_train/mountain/10041.jpg",
        "data/seg_train/mountain/10045.jpg", "data/seg_train/mountain/1005.jpg",
    ],
    "sea": [
        "data/seg_train/sea/10001.jpg", "data/seg_train/sea/10008.jpg",
        "data/seg_train/sea/10011.jpg", "data/seg_train/sea/10017.jpg",
        "data/seg_train/sea/10026.jpg", "data/seg_train/sea/10034.jpg",
        "data/seg_train/sea/10038.jpg", "data/seg_train/sea/10058.jpg",
        "data/seg_train/sea/1007.jpg", "data/seg_train/sea/10075.jpg",
    ],
    "street": [
        "data/seg_train/street/10013.jpg", "data/seg_train/street/10016.jpg",
        "data/seg_train/street/10024.jpg", "data/seg_train/street/10033.jpg",
        "data/seg_train/street/10044.jpg", "data/seg_train/street/10047.jpg",
        "data/seg_train/street/1006.jpg", "data/seg_train/street/10074.jpg",
        "data/seg_train/street/10087.jpg", "data/seg_train/street/10101.jpg",
    ],
}


def download_single_file(url: str, dest_path: Path) -> bool:
    """Download a single image file if not already present."""
    if dest_path.exists() and dest_path.stat().st_size > 500:
        return True
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        logger.debug(f"Failed to download {url}: {e}")
    return False


def create_synthetic_image(cls_name: str, width: int = 150, height: int = 150) -> Image.Image:
    """Create realistic synthetic category images if offline."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    np.random.seed(None)

    if cls_name == "glacier":
        # Icy blues and white peaks
        for y in range(height):
            c = int(180 + 70 * (y / height))
            b = int(220 + 35 * (y / height))
            draw.line([(0, y), (width, y)], fill=(c - 40, c - 10, min(255, b)))
        points = [(0, 100), (30, 40), (70, 70), (100, 30), (130, 60), (150, 90), (150, 150), (0, 150)]
        draw.polygon(points, fill=(235, 245, 255))
        draw.polygon([(20, 150), (50, 80), (80, 150)], fill=(190, 220, 245))

    elif cls_name == "mountain":
        # Gray rock mountains with sky
        draw.rectangle([(0, 0), (width, 80)], fill=(135, 206, 235))
        draw.polygon([(0, 150), (40, 50), (90, 150)], fill=(110, 110, 115))
        draw.polygon([(30, 62), (40, 50), (50, 62)], fill=(250, 250, 250)) # snow peak
        draw.polygon([(70, 150), (115, 60), (150, 150)], fill=(90, 90, 95))
        draw.polygon([(105, 75), (115, 60), (125, 75)], fill=(250, 250, 250))

    elif cls_name == "sea":
        # Deep blue ocean with light horizon sky
        draw.rectangle([(0, 0), (width, 60)], fill=(176, 226, 255))
        for y in range(60, height):
            blue = int(150 + 90 * ((y - 60) / (height - 60)))
            draw.line([(0, y), (width, y)], fill=(10, 80, min(255, blue)))
        # Waves
        for y in range(70, height, 12):
            draw.arc([10, y, 70, y + 10], 0, 180, fill=(200, 230, 255), width=2)
            draw.arc([80, y + 5, 140, y + 15], 0, 180, fill=(200, 230, 255), width=2)

    elif cls_name == "forest":
        # Green trees canopy
        draw.rectangle([(0, 0), (width, 50)], fill=(190, 225, 250))
        for x in range(10, width, 25):
            draw.polygon([(x, 140), (x + 10, 40), (x + 20, 140)], fill=(34, 139, 34))
            draw.polygon([(x - 5, 140), (x + 10, 60), (x + 25, 140)], fill=(20, 110, 20))
            draw.rectangle([(x + 8, 135), (x + 12, 150)], fill=(101, 67, 33))

    elif cls_name == "buildings":
        # Skyline buildings with windows
        draw.rectangle([(0, 0), (width, 150)], fill=(200, 220, 240))
        draw.rectangle([(10, 30), (50, 150)], fill=(70, 80, 95))
        draw.rectangle([(55, 60), (95, 150)], fill=(90, 100, 115))
        draw.rectangle([(100, 20), (140, 150)], fill=(60, 70, 85))
        # Windows
        for bx in [20, 35, 65, 80, 110, 125]:
            for by in range(40, 140, 15):
                draw.rectangle([(bx, by), (bx + 6, by + 8)], fill=(255, 255, 180))

    else:  # street
        # Road with asphalt and center lines
        draw.rectangle([(0, 0), (width, 70)], fill=(180, 190, 200))
        draw.polygon([(40, 70), (110, 70), (150, 150), (0, 150)], fill=(60, 60, 65))
        # Yellow dashed center line
        for y in range(75, 145, 15):
            draw.line([(75, y), (75, y + 8)], fill=(255, 215, 0), width=3)
        # Sidewalk
        draw.polygon([(0, 70), (40, 70), (0, 150)], fill=(160, 160, 160))
        draw.polygon([(110, 70), (150, 70), (150, 150)], fill=(160, 160, 160))

    # Add subtle random texture noise
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def setup_dataset(num_train_per_class: int = 50, num_val_per_class: int = 15, num_test_per_class: int = 15):
    """
    Downloads real Intel dataset images concurrently or falls back to synthetic data.
    Ensures train, val, test, and sample_images directories are populated.
    """
    logger.info("Initializing Location Dataset Setup...")

    # Save class_names.json for serving
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(CLASSES, f, indent=2)

    # First, fetch filenames list from HF Hub to download a real diverse subset
    available_files = {cls_name: [] for cls_name in CLASSES}
    try:
        resp = requests.get(
            "https://huggingface.co/api/datasets/sfarrukhm/intel-image-classification",
            timeout=8,
        )
        if resp.status_code == 200:
            all_siblings = [s["rfilename"] for s in resp.json().get("siblings", [])]
            for cls_name in CLASSES:
                train_prefix = f"data/seg_train/{cls_name}/"
                cls_files = [f for f in all_siblings if f.startswith(train_prefix) and f.endswith(".jpg")]
                available_files[cls_name] = cls_files
    except Exception as e:
        logger.warning(f"Could not query HF API directly: {e}")

    # Fallback to SAMPLE_FILENAMES if API didn't return
    for cls_name in CLASSES:
        if not available_files[cls_name]:
            available_files[cls_name] = SAMPLE_FILENAMES.get(cls_name, [])

    download_tasks = []
    total_needed = num_train_per_class + num_val_per_class + num_test_per_class

    with ThreadPoolExecutor(max_workers=10) as executor:
        for cls_name in CLASSES:
            cls_train_dir = TRAIN_DATA_DIR / cls_name
            cls_val_dir = VAL_DATA_DIR / cls_name
            cls_test_dir = TEST_DATA_DIR / cls_name
            cls_train_dir.mkdir(parents=True, exist_ok=True)
            cls_val_dir.mkdir(parents=True, exist_ok=True)
            cls_test_dir.mkdir(parents=True, exist_ok=True)

            files = available_files.get(cls_name, [])
            for i in range(total_needed):
                # Determine target directory
                if i < num_train_per_class:
                    target_dir = cls_train_dir
                elif i < num_train_per_class + num_val_per_class:
                    target_dir = cls_val_dir
                else:
                    target_dir = cls_test_dir

                target_file = target_dir / f"{cls_name}_{i:03d}.jpg"

                if i < len(files):
                    remote_file = files[i]
                    url = f"{HF_BASE_URL}/{remote_file}"
                    download_tasks.append((executor.submit(download_single_file, url, target_file), cls_name, target_file))
                else:
                    # Create synthetic image if remote file not indexed
                    if not target_file.exists():
                        syn_img = create_synthetic_image(cls_name)
                        syn_img.save(target_file)

        # Process download tasks
        downloaded = 0
        for future, cls_name, target_file in download_tasks:
            try:
                success = future.result()
                if success:
                    downloaded += 1
                else:
                    if not target_file.exists():
                        create_synthetic_image(cls_name).save(target_file)
            except Exception:
                if not target_file.exists():
                    create_synthetic_image(cls_name).save(target_file)

    logger.info(f"Dataset populated. Real images downloaded: {downloaded}.")

    # Generate sample images for testing
    for cls_name in CLASSES:
        sample_path = SAMPLE_IMAGES_DIR / f"{cls_name}.jpg"
        if not sample_path.exists():
            # Copy first image from test directory or generate
            src_test = TEST_DATA_DIR / cls_name / f"{cls_name}_000.jpg"
            if src_test.exists():
                sample_img = Image.open(src_test)
                sample_img.save(sample_path)
            else:
                create_synthetic_image(cls_name).save(sample_path)

    logger.info("Sample test images ready in data/sample_images/")


def get_data_generators(batch_size: int = BATCH_SIZE, img_size: tuple = IMG_SIZE):
    """
    Creates tf.data.Dataset pipelines for training, validation, and testing.
    Includes data augmentation for training.
    """
    # Load dataset from directories
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
    )

    # Data augmentation pipeline for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize datasets for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    setup_dataset()
    train_ds, val_ds, test_ds = get_data_generators()
    print("Train classes:", CLASSES)
    print("Dataset setup completed successfully!")
