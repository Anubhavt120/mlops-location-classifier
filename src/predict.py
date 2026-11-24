import sys
import json
import logging
from pathlib import Path
from typing import Union, Dict, Any, Tuple
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import tensorflow as tf
from src.config import (
    CLASSES,
    IMG_SIZE,
    CNN_MODEL_PATH,
    MOBILENET_MODEL_PATH,
    CLASS_NAMES_PATH,
)
from src.utils import (
    load_image_from_source,
    preprocess_image,
    check_is_animation,
    extract_geographic_location,
)

logger = logging.getLogger(__name__)

# Model cache to avoid reloading on each prediction
_LOADED_MODELS = {}


def get_model(model_name: str = "MobileNetV2") -> tf.keras.Model:
    """Loads and caches the requested model."""
    global _LOADED_MODELS

    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if model_name.lower().startswith("mobile"):
        target_path = MOBILENET_MODEL_PATH
        fallback_path = CNN_MODEL_PATH
    else:
        target_path = CNN_MODEL_PATH
        fallback_path = MOBILENET_MODEL_PATH

    if target_path.exists():
        model = tf.keras.models.load_model(str(target_path))
    elif fallback_path.exists():
        logger.warning(f"Requested {target_path.name} not found. Falling back to {fallback_path.name}")
        model = tf.keras.models.load_model(str(fallback_path))
    else:
        raise FileNotFoundError("No trained models found! Please run `python src/train.py` first.")

    _LOADED_MODELS[model_name] = model
    return model


def predict_location(
    image_source: Union[str, Path, Image.Image],
    model_name: str = "MobileNetV2",
    location_hint: str = "",
) -> Dict[str, Any]:
    """
    Predicts location scene for an input image (URL, path, or PIL image).

    Returns:
    {
        "predicted_class": "glacier",
        "confidence": 0.985,
        "probabilities": {"buildings": 0.01, ...},
        "model_used": "MobileNetV2",
        "image": <PIL.Image>
    }
    """
    # 1. Load image
    if isinstance(image_source, Image.Image):
        pil_img = image_source.convert("RGB")
    else:
        pil_img = load_image_from_source(image_source)

    # 2. Check if image is an animation / illustration
    is_anim, anim_reason = check_is_animation(pil_img)

    # 3. Preprocess
    resized_img, batch_arr = preprocess_image(pil_img, target_size=IMG_SIZE)

    # 4. Predict
    model = get_model(model_name)
    raw_probs = model.predict(batch_arr, verbose=0)[0]

    # 5. Parse results
    predicted_idx = int(np.argmax(raw_probs))
    predicted_class = CLASSES[predicted_idx]
    confidence = float(raw_probs[predicted_idx])

    probabilities = {CLASSES[i]: float(raw_probs[i]) for i in range(len(CLASSES))}
    # Sort probabilities descending
    sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    # 6. Extract geographical location / coordinates
    source_label = str(image_source) if not isinstance(image_source, Image.Image) else ""
    geo_location = extract_geographic_location(
        pil_img,
        source_str=source_label,
        predicted_class=predicted_class,
        location_hint=location_hint,
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": sorted_probs,
        "model_used": model_name,
        "image": pil_img,
        "is_animation": is_anim,
        "animation_reason": anim_reason,
        "geo_location": geo_location,
    }


if __name__ == "__main__":
    from src.config import SAMPLE_IMAGES_DIR
    test_sample = SAMPLE_IMAGES_DIR / "glacier.jpg"
    if test_sample.exists():
        res = predict_location(test_sample)
        print("Sample Prediction Result:")
        print(f"Predicted Class : {res['predicted_class']}")
        print(f"Confidence      : {res['confidence']:.2%}")
        print("Probabilities   :", res['probabilities'])
    else:
        print("No sample image found yet.")
