# processing/processing_odoc.py

import cv2
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
model_path = os.path.join(ROOT_DIR, "models", "retinet_9010.h5")

# Lazy model loader — only loads when first needed, not at import time
_model = None
def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(model_path)
    return _model


def processing(image_cv2, threshold, batch_size):
    """
    OD-OC segmentation processing function.
    """
    model = get_model()

    small = cv2.resize(image_cv2, (0, 0), fx=0.33, fy=0.33)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    mask_small = (gray > 128).astype(np.uint8)

    mask_full = cv2.resize(
        mask_small,
        (image_cv2.shape[1], image_cv2.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return mask_full