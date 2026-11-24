import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from src.config import (
    EPOCHS,
    CNN_MODEL_PATH,
    MOBILENET_MODEL_PATH,
    REPORTS_DIR,
    MODELS_DIR,
)
from src.data_loader import setup_dataset, get_data_generators
from src.model import build_custom_cnn, build_mobilenet_v2
from src.utils import plot_training_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_models():
    """Trains both Custom CNN baseline and MobileNetV2 Transfer Learning models."""
    logger.info("=== STEP 1: Verifying Dataset ===")
    setup_dataset(num_train_per_class=40, num_val_per_class=12, num_test_per_class=12)
    train_ds, val_ds, test_ds = get_data_generators()

    # Callbacks
    callbacks_cnn = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CNN_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # --- 1. Train Custom CNN Baseline ---
    logger.info("=== STEP 2: Training Custom Deep CNN Baseline ===")
    cnn_model = build_custom_cnn()
    cnn_history = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_cnn,
    )

    # Save Custom CNN history & plot
    cnn_hist_dict = {k: [float(val) for val in v] for k, v in cnn_history.history.items()}
    with open(REPORTS_DIR / "cnn_history.json", "w") as f:
        json.dump(cnn_hist_dict, f, indent=2)
    plot_training_history(cnn_hist_dict, REPORTS_DIR / "cnn_training_curve.png")
    logger.info(f"Custom CNN training completed. Saved to: {CNN_MODEL_PATH}")

    # --- 2. Train MobileNetV2 Transfer Learning ---
    logger.info("=== STEP 3: Training MobileNetV2 Transfer Learning Model ===")
    mobilenet_model = build_mobilenet_v2(freeze_base=True)

    callbacks_mob = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MOBILENET_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    mob_history = mobilenet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_mob,
    )

    # Save MobileNetV2 history & plot
    mob_hist_dict = {k: [float(val) for val in v] for k, v in mob_history.history.items()}
    with open(REPORTS_DIR / "mobilenet_history.json", "w") as f:
        json.dump(mob_hist_dict, f, indent=2)
    plot_training_history(mob_hist_dict, REPORTS_DIR / "mobilenet_training_curve.png")
    logger.info(f"MobileNetV2 model training completed. Saved to: {MOBILENET_MODEL_PATH}")

    logger.info("=== Training for both models completed successfully! ===")


if __name__ == "__main__":
    train_models()
