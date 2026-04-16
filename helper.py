import os
import numpy as np
import base64
import sys

def get_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return filename

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def overlay_mask_on_rgb(rgb_img, mask):
    out = rgb_img.copy()
    mask_bool = (mask > 0)
    out[mask_bool, 1] = 255
    return out