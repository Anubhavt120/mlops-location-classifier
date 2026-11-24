import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CLASSES,
    IMG_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    CNN_MODEL_PATH,
    MOBILENET_MODEL_PATH,
    SAMPLE_IMAGES_DIR,
)
from src.model import build_custom_cnn, build_mobilenet_v2
from src.predict import predict_location
from src.utils import preprocess_image


def test_classes_configuration():
    """Verify standard Intel Location classes."""
    assert len(CLASSES) == 6
    assert set(CLASSES) == {"buildings", "forest", "glacier", "mountain", "sea", "street"}


def test_custom_cnn_architecture():
    """Verifies that the Custom CNN compiles and has correct output shape."""
    model = build_custom_cnn(input_shape=(150, 150, 3), num_classes=6)
    dummy_input = np.random.rand(2, 150, 150, 3).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    assert preds.shape == (2, 6)
    # Check that predictions sum to 1 (softmax)
    assert np.allclose(np.sum(preds, axis=1), [1.0, 1.0], atol=1e-4)


def test_mobilenet_v2_architecture():
    """Verifies that MobileNetV2 compiles and outputs 6 class probabilities."""
    # Test architecture without downloading external weights
    model = build_mobilenet_v2(input_shape=(150, 150, 3), num_classes=6, freeze_base=True, weights=None)
    dummy_input = np.random.rand(2, 150, 150, 3).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    assert preds.shape == (2, 6)
    assert np.allclose(np.sum(preds, axis=1), [1.0, 1.0], atol=1e-4)


def test_saved_model_inference():
    """Verifies that the trained and saved MobileNetV2 artifact loads and runs inference."""
    if MOBILENET_MODEL_PATH.exists():
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MOBILENET_MODEL_PATH))
        dummy_input = np.random.rand(1, 150, 150, 3).astype(np.float32)
        preds = model.predict(dummy_input, verbose=0)
        assert preds.shape == (1, 6)
        assert np.isclose(np.sum(preds), 1.0, atol=1e-3)


def test_image_preprocessing():
    """Verifies image preprocessing output shapes."""
    raw_img = Image.new("RGB", (300, 400), color=(100, 150, 200))
    resized_img, batch_arr = preprocess_image(raw_img, target_size=(150, 150))
    assert resized_img.size == (150, 150)
    assert batch_arr.shape == (1, 150, 150, 3)


def test_prediction_pipeline_on_sample():
    """Tests full end-to-end inference pipeline on a sample image."""
    # Create or test on an existing sample
    sample_img = Image.new("RGB", (150, 150), color=(180, 220, 255))
    res = predict_location(sample_img, model_name="Custom CNN")

    assert "predicted_class" in res
    assert res["predicted_class"] in CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 6
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-3
