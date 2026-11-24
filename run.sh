#!/bin/bash
# 1-Click Startup for Location Image Classifier
echo "======================================================="
echo "    🌍 Location Image Classifier (MLOps Project)"
echo "======================================================="

# Check Python
if ! command -v python &> /dev/null
then
    echo "[ERROR] Python could not be found."
    exit 1
fi

# Train models if not present
if [ ! -f "models/mobilenetv2_location.keras" ]; then
    echo "[INFO] Trained models not found. Running training pipeline..."
    python src/train.py
fi

# Run Streamlit Web App
echo "[INFO] Starting Streamlit Application on http://localhost:8501 ..."
streamlit run app.py
