import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE


def build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES) -> tf.keras.Model:
    """
    Builds a custom Deep Convolutional Neural Network (CNN) baseline model.
    Architecture:
    - Input & Normalization (Rescaling 1/255)
    - 4 Conv2D + BatchNorm + MaxPool blocks
    - GlobalAveragePooling2D
    - Fully Connected Dense layer with Dropout regularization
    - Softmax Output
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Rescaling pixel values to [0, 1]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="custom_cnn_location_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_v2(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, freeze_base=True, weights="imagenet") -> tf.keras.Model:
    """
    Builds a Transfer Learning model using pre-trained MobileNetV2 backbone.
    Architecture:
    - Pre-trained MobileNetV2 feature extractor (trained on ImageNet)
    - Custom Classification Head tuned for 6 Location Scene classes
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # MobileNetV2 preprocessing maps inputs from [0, 255] to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Base pretrained model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = not freeze_base

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="mobilenet_gap")(x)
    x = layers.BatchNormalization(name="mobilenet_bn")(x)
    x = layers.Dropout(0.4, name="mobilenet_dropout")(x)
    x = layers.Dense(256, activation="relu", name="mobilenet_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_location_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    cnn = build_custom_cnn()
    cnn.summary()
    print("\n--- MobileNetV2 Model ---")
    mob = build_mobilenet_v2()
    mob.summary()
