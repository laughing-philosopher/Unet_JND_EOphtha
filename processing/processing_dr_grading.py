import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import efficientnet.tfkeras as efn  # Required to recognize the EfficientNet architecture
import sys, os

BASE_PATH = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_PATH, "models", "best_dr_model.keras")

def load_dr_model():
    """Builds the architecture and loads the fine-tuned TensorFlow/Keras DR model weights."""
    print("Building Architecture and Loading Diabetic Retinopathy Weights...")
    
    # 1. Rebuild the exact base architecture used on Kaggle
    base_model = efn.EfficientNetB5(weights=None, include_top=False, input_shape=(456, 456, 3))
    
    # 2. Add the custom regression head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='linear') 
    ])
    
    # 3. Load JUST the weights from the file, bypassing the corrupted architecture graph
    model.load_weights(model_path)
    
    return model

def predict_dr_severity(image_path, model):
    """Predicts the severity of Diabetic Retinopathy from a fundus image."""
    
    # 1. Image Preprocessing (Must match Kaggle exactly)
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found"
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (456, 456))  # 456x456 is the native size for EfficientNet-B5
    img = img / 255.0                  # Rescale pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension: shape becomes (1, 456, 456, 3)

    # 2. Get continuous prediction
    raw_val = model.predict(img, verbose=0)[0][0]

    # 3. Apply standard APTOS thresholds to round to discrete grades
    if raw_val < 0.5:
        severity = 0
    elif raw_val < 1.5:
        severity = 1
    elif raw_val < 2.5:
        severity = 2
    elif raw_val < 3.5:
        severity = 3
    else:
        severity = 4

    # 4. Map integer to clinical terminology
    severity_map = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR",
    }

    return severity_map.get(severity, "Unknown")