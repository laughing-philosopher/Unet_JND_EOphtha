"""processing_odoc_basic.py
============================
Optic Disc / Cup segmentation using best_unet_refuge2.pth
(segmentation_models_pytorch ResNet34 U-Net, 3-class output, REFUGE2 dataset).

Class convention (matches friend's reference implementation):
  0 = background
  1 = optic disc rim  (disc minus cup)
  2 = optic cup

Output (H, W, 3) uint8 RGB:
  Green [0, 255, 0] — Optic Disc rim  (class 1)
  Blue  [0, 0, 255] — Optic Cup       (class 2)
  Black [0, 0, 0]   — Background      (class 0)
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Support both normal run and PyInstaller bundle
_BASE = (os.environ.get('AAKHI_BASE_PATH') or
         os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(_BASE, "models", "best_unet_refuge2.pth")

IMG_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model  = None
_device = None


def _load_model():
    global _model, _device
    if _model is not None:
        return _model

    import segmentation_models_pytorch as smp

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 3,
        activation      = None,
    )

    state = torch.load(MODEL_PATH, map_location=_device)
    # Strip DataParallel prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    _model.load_state_dict(state)
    _model.to(_device)
    _model.eval()

    print(f"[ODOC] Loaded ResNet34 U-Net (smp) from {MODEL_PATH}")
    return _model


def _to_probs(arr):
    """Softmax over channel axis (numpy)."""
    exp = np.exp(arr - np.max(arr, axis=0, keepdims=True))
    return exp / (exp.sum(axis=0, keepdims=True) + 1e-8)


def processing(image_rgb: np.ndarray, threshold: float = 0.5,
               batch_size: int = 1) -> np.ndarray:
    """
    Segment optic disc and cup from an RGB fundus image.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8 RGB
    threshold : per-class probability cutoff for binarisation
    batch_size: unused (kept for API compatibility)

    Returns
    -------
    (H, W, 3) uint8 colour-coded RGB
      Green = disc rim, Blue = cup, Black = background
    """
    model = _load_model()
    orig_h, orig_w = image_rgb.shape[:2]

    # Preprocess — match ImageNet normalisation used during REFUGE2 training
    img = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(_device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)           # (1, 3, 512, 512)

    out_np = logits.squeeze(0).cpu().numpy()   # (3, 512, 512)
    probs  = _to_probs(out_np)                  # softmax probabilities

    # Per-class threshold assignment (same as reference implementation)
    pred = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    pred[probs[1] >= threshold] = 1   # disc rim
    pred[probs[2] >= threshold] = 2   # cup (overwrites rim if both above threshold)

    # Scale back to original resolution
    pred_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Colour-code output
    out = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    out[pred_full == 1] = [0, 255,   0]   # disc rim → green
    out[pred_full == 2] = [0,   0, 255]   # cup      → blue

    return out
