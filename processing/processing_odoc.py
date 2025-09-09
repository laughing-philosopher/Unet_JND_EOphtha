# processing/processing_odoc.py

import cv2
import numpy as np
import tensorflow as tf
from math import sqrt, ceil
from scipy.cluster.hierarchy import fclusterdata

import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../BTP/processing
ROOT_DIR = os.path.dirname(CURRENT_DIR)                    # .../BTP
model_path = os.path.join(ROOT_DIR, "models", "retinet_9010.h5")

# model.load_weights(model_path)

model = tf.keras.models.load_model(model_path)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (0, 0), fx=0.33, fy=0.33)
    return img

def run_od_oc_segmentation(img_path, output_path):
    img = preprocess_image(img_path)
    # 👇 Replace with your segmentation pipeline
    # TODO: refactor your detect_lines() logic here
    # (remove cv2.imshow parts, return processed image or path)

    # Dummy example: just saving grayscale copy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray)
    return output_path

def processing(image_cv2, threshold, batch_size):
    """
    Dummy OD-OC segmentation processing function.
    """
    # Resize for processing
    small = cv2.resize(image_cv2, (0, 0), fx=0.33, fy=0.33)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Normalize mask 0/1
    mask_small = (gray > 128).astype(np.uint8)

    # Resize mask back to original image size
    mask_full = cv2.resize(
        mask_small, 
        (image_cv2.shape[1], image_cv2.shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )

    return mask_full
