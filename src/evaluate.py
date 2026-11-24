import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    CLASSES,
    CNN_MODEL_PATH,
    MOBILENET_MODEL_PATH,
    REPORTS_DIR,
)
from src.data_loader import get_data_generators
from src.utils import plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_single_model(model: tf.keras.Model, test_ds, model_name: str) -> dict:
    """Evaluates a single model on test dataset and generates full classification metrics."""
    logger.info(f"Evaluating {model_name}...")

    # Extract true labels and predictions
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs_list.append(preds)
        y_true_list.append(labels.numpy())

    y_true_onehot = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)

    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate metrics
    eval_loss, eval_acc = model.evaluate(test_ds, verbose=0)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix plot
    safe_name = model_name.lower().replace(" ", "_")
    cm_path = REPORTS_DIR / f"{safe_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, CLASSES, cm_path, title=f"{model_name} Confusion Matrix")

    results = {
        "model_name": model_name,
        "test_loss": float(eval_loss),
        "test_accuracy": float(eval_acc),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {cls_name: report[cls_name] for cls_name in CLASSES if cls_name in report},
    }
    return results


def run_evaluation() -> dict:
    """Runs evaluation on all available models and logs results."""
    logger.info("=== Running Model Evaluation Suite ===")
    _, _, test_ds = get_data_generators()

    all_results = {}

    if CNN_MODEL_PATH.exists():
        cnn_model = tf.keras.models.load_model(str(CNN_MODEL_PATH))
        all_results["Custom CNN"] = evaluate_single_model(cnn_model, test_ds, "Custom CNN Baseline")
    else:
        logger.warning(f"Custom CNN model not found at {CNN_MODEL_PATH}")

    if MOBILENET_MODEL_PATH.exists():
        mob_model = tf.keras.models.load_model(str(MOBILENET_MODEL_PATH))
        all_results["MobileNetV2"] = evaluate_single_model(mob_model, test_ds, "MobileNetV2 Transfer Learning")
    else:
        logger.warning(f"MobileNetV2 model not found at {MOBILENET_MODEL_PATH}")

    # Save overall evaluation report
    report_file = REPORTS_DIR / "evaluation_metrics.json"
    with open(report_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'MODEL':<30} | {'ACCURACY':<10} | {'MACRO F1':<10} | {'LOSS':<10}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<30} | {res['test_accuracy'] * 100:6.2f}%    | {res['macro_f1'] * 100:6.2f}%    | {res['test_loss']:6.4f}")
    print("=" * 65 + "\n")

    return all_results


if __name__ == "__main__":
    run_evaluation()
