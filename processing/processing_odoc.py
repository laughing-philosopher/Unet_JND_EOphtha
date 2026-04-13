import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

# ==========================================
# CONFIGURATION
# ==========================================
# Ensure this path matches where your model is actually stored relative to app.py
MODEL_PATH = os.path.join("models", "best_unet_refuge2.pth")

IMG_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Global model variable to cache the model in memory (prevents reloading on every click)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

def get_model():
    """Loads and caches the model."""
    global _model
    if _model is None:
        _model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=3,
            activation=None,  # We apply softmax manually below
        )
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ODOC Model not found at: {MODEL_PATH}")
            
        state = torch.load(MODEL_PATH, map_location=device)
        _model.load_state_dict(state)
        _model.to(device)
        _model.eval()
    return _model

def to_probs(arr, axis=0):
    """Applies softmax along the specified axis to convert raw logits to probabilities."""
    exp = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
    return exp / (exp.sum(axis=axis, keepdims=True) + 1e-8)

def processing(image_rgb, threshold=0.5, batch_size=8):
    """
    Main inference function called by app.py.
    
    Args:
        image_rgb: NumPy array of the fundus image in RGB format (from Streamlit/PIL).
        threshold: Float probability cutoff.
        batch_size: Ignored for single image inference, kept for signature compatibility.
        
    Returns:
        NumPy array (RGB) representing the color-coded mask, sized to the original image.
    """
    orig_h, orig_w = image_rgb.shape[:2]
    
    # 1. Preprocess (No BGR2RGB needed, app.py already provides RGB)
    img_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img_norm = (img_resized - MEAN) / STD
    
    # Convert to PyTorch tensor shape (Batch, Channels, Height, Width)
    img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # 2. Inference
    net = get_model()
    with torch.no_grad():
        logits = net(img_t)
        
    # Remove batch dimension and move to CPU
    out_np = logits.squeeze().cpu().numpy()

    # 3. Softmax & Thresholding
    probs = to_probs(out_np, axis=0)
    p_od = probs[1]
    p_oc = probs[2]

    pred = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    pred[p_od >= threshold] = 1   # Optic Disc (OD)
    pred[p_oc >= threshold] = 2   # Optic Cup (OC) overrides OD if overlapping

    # 4. Color Mapping
    color_mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    color_mask[pred == 1] = (0, 255, 0)   # Green — Optic Disc
    color_mask[pred == 2] = (255, 0, 0)   # Red   — Optic Cup

    # 5. Calculate Vertical CDR
    cdr = 0.0
    # Get vertical heights of OD and OC masks
    od_rows = np.any(pred == 1, axis=1)
    oc_rows = np.any(pred == 2, axis=1)
    
    if np.any(od_rows):
        od_height = np.sum(od_rows)
        oc_height = np.sum(oc_rows) if np.any(oc_rows) else 0
        cdr = oc_height / od_height if od_height > 0 else 0.0

    # 6. Resize mask back to original
    color_mask_resized = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Return both the mask and the calculated CDR
    return color_mask_resized, round(cdr, 3)