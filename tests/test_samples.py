import sys
import shutil
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import predict_location
from src.config import CLASSES, TEST_DATA_DIR, SAMPLE_IMAGES_DIR

# Ensure sample_images has real test photos
SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
for cls_name in CLASSES:
    cls_test_dir = TEST_DATA_DIR / cls_name
    test_files = sorted(list(cls_test_dir.glob("*.jpg")))
    if test_files:
        dest = SAMPLE_IMAGES_DIR / f"{cls_name}.jpg"
        shutil.copyfile(test_files[0], dest)

print("\n" + "=" * 65)
print("     REAL TEST SAMPLES INFERENCE (MobileNetV2)")
print("=" * 65)

sample_files = sorted(list(SAMPLE_IMAGES_DIR.glob("*.jpg")))
correct = 0

for img_path in sample_files:
    true_label = img_path.stem
    res = predict_location(img_path, model_name="MobileNetV2")
    pred_label = res["predicted_class"]
    conf = res["confidence"]
    match = "MATCH" if true_label == pred_label else "DIFF"
    if match == "MATCH":
        correct += 1

    print(f"[{match:^5}] Actual: {true_label:<10} | Predicted: {pred_label:<10} | Confidence: {conf:6.1%}")

print("-" * 65)
print(f"Total Samples: {len(sample_files)} | Correct: {correct} / {len(sample_files)} ({correct/len(sample_files)*100:.1f}%)")
print("=" * 65 + "\n")
