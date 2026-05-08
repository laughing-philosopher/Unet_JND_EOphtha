import cv2
import numpy as np
import math

from model import build_UNet
from skimage.measure import label, regionprops

# Patch dimension details
Img_Width = 48
Img_Height = 48
Img_Channels = 1

import sys, os

BASE_PATH = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_PATH, "models", "UNet_JND_EOphtha.h5")

# Lazy model loader
_model = None
def get_model():
    global _model
    if _model is None:
        _model = build_UNet((Img_Width, Img_Height, Img_Channels), (Img_Width, Img_Height, Img_Channels))
        _model.load_weights(model_path)
    return _model


# ---------------------------------------------------------------------------
#  False-Positive Reduction (from Kaggle notebook: reduction-of-false-positive-from-ma)
#
#  Heuristic filters applied to each connected component in the baseline mask:
#    - Area < 8 px              → removed
#    - Circularity < 0.45       → removed  (4π·area / perimeter²)
#    - Mean confidence < 0.25   → removed
#
#  These thresholds are hardcoded based on the validated notebook results
#  (73.68% FP reduction, precision 0.66 → 0.82 on the EOptha test set).
# ---------------------------------------------------------------------------

# Hardcoded heuristic thresholds
_FP_MIN_AREA        = 8     # minimum component area in pixels
_FP_MIN_CIRCULARITY = 0.45  # minimum circularity (4π·area / perimeter²)
_FP_MIN_CONFIDENCE  = 0.25  # minimum mean probability inside the component


def fp_reduction(prob_map, binary_mask):
    """
    Post-process the baseline binary mask to remove likely false-positive
    MA candidates using shape and confidence heuristics.

    Parameters
    ----------
    prob_map : np.ndarray (H, W), float
        Normalised probability map from the UNet (values in [0, 1]).
    binary_mask : np.ndarray (H, W), uint8
        Baseline binary mask (1 = candidate, 0 = background).

    Returns
    -------
    dict with keys:
        refined_mask       – np.ndarray (H, W), uint8, filtered binary mask
        original_count     – int, total candidate components before filtering
        refined_count      – int, components kept after filtering
        removed_count      – int, components removed
        removed_locations  – list of (x, y) centroids of removed components
    """
    labels = label(binary_mask)
    refined = np.zeros_like(binary_mask)

    original_count = 0
    refined_count = 0
    removed_locations = []

    for region in regionprops(labels):
        original_count += 1

        area = region.area
        y, x = region.centroid

        component = (labels == region.label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        c = contours[0]
        perimeter = cv2.arcLength(c, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        conf = np.mean(prob_map[labels == region.label])

        # ---- Heuristic FP filter ----
        keep = True
        if area < _FP_MIN_AREA:
            keep = False
        if circularity < _FP_MIN_CIRCULARITY:
            keep = False
        if conf < _FP_MIN_CONFIDENCE:
            keep = False

        if keep:
            refined[labels == region.label] = 1
            refined_count += 1
        else:
            removed_locations.append((int(x), int(y)))

    removed_count = original_count - refined_count

    return {
        "refined_mask": refined,
        "original_count": original_count,
        "refined_count": refined_count,
        "removed_count": removed_count,
        "removed_locations": removed_locations,
    }


def processing(img, threshold, batch_size, progress_callback=None):
    """
    Run MA detection with the UNet model and return both the baseline
    (Advanced) and FP-reduced (Basic) results.

    Parameters
    ----------
    img : np.ndarray (H, W, 3), uint8, BGR/RGB fundus image.
    threshold : float — unused but kept for API compatibility.
    batch_size : int  — patches per prediction batch.
    progress_callback : callable(current_batch, total_batches) or None

    Returns
    -------
    dict with keys:
        prob_map          – np.ndarray (H, W), float32, normalised [0, 1]
        baseline_mask     – np.ndarray (H, W), uint8  (Advanced / all candidates)
        refined_mask      – np.ndarray (H, W), uint8  (Basic / FP-reduced)
        baseline_count    – int
        refined_count     – int
        removed_count     – int
        removed_locations – list of (x, y)
    """
    model = get_model()

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_g = img1[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(8, 8))
    img_c = clahe.apply(img_g)

    [m, n] = img_c.shape

    gt = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gt = gt / 255.0
    gt = (gt > 0.5).astype(np.float32)

    iarr, jarr = [], []
    step = 16
    for i in range(0, m, step):
        for j in range(0, n, step):
            if (i + Img_Height) > (m - 1) or (j + Img_Width) > (n - 1):
                continue
            iarr.append(i)
            jarr.append(j)

    tot = len(iarr)
    if tot == 0:
        empty = np.zeros((m, n), dtype=np.float32)
        return {
            "prob_map": empty,
            "baseline_mask": empty.astype(np.uint8),
            "refined_mask": empty.astype(np.uint8),
            "baseline_count": 0,
            "refined_count": 0,
            "removed_count": 0,
            "removed_locations": [],
        }

    final_res = np.zeros((m, n), dtype=np.float32)
    num_batches = math.ceil(tot / batch_size)

    for batch in range(num_batches):
        start = batch * batch_size
        end = min(start + batch_size, tot)
        current_batch_size = end - start

        patches_img = np.zeros((current_batch_size, Img_Height, Img_Width, 1), dtype=np.float32)
        patches_gt  = np.zeros((current_batch_size, Img_Height, Img_Width, 1), dtype=np.float32)

        for k in range(current_batch_size):
            itr = start + k
            patch_img = img_c[iarr[itr]:(iarr[itr] + Img_Height), jarr[itr]:(jarr[itr] + Img_Width)]
            patches_img[k] = np.expand_dims(patch_img, axis=-1)

            patch_gt = gt[iarr[itr]:(iarr[itr] + Img_Height), jarr[itr]:(jarr[itr] + Img_Width)]
            patches_gt[k] = np.expand_dims(patch_gt, axis=-1)

        inter_res, _ = model.predict([patches_img, patches_gt], verbose=0)

        for k in range(current_batch_size):
            itr = start + k
            final_res[iarr[itr]:(iarr[itr] + Img_Height), jarr[itr]:(jarr[itr] + Img_Width)] += np.squeeze(inter_res[k])

        del patches_img, patches_gt, inter_res

        # Fire progress callback if provided
        if progress_callback is not None:
            progress_callback(batch + 1, num_batches)

    mx = np.max(final_res)
    if mx <= 0:
        empty = np.zeros_like(final_res)
        return {
            "prob_map": empty,
            "baseline_mask": empty.astype(np.uint8),
            "refined_mask": empty.astype(np.uint8),
            "baseline_count": 0,
            "refined_count": 0,
            "removed_count": 0,
            "removed_locations": [],
        }

    # Normalised probability map
    prob_map = final_res / mx

    # Baseline binary mask (Advanced — all candidates, threshold > 0.1)
    baseline_mask = (prob_map > 0.1).astype(np.uint8)

    # FP-reduced mask (Basic)
    fp_result = fp_reduction(prob_map, baseline_mask)

    return {
        "prob_map": prob_map,
        "baseline_mask": baseline_mask,
        "refined_mask": fp_result["refined_mask"],
        "baseline_count": fp_result["original_count"],
        "refined_count": fp_result["refined_count"],
        "removed_count": fp_result["removed_count"],
        "removed_locations": fp_result["removed_locations"],
    }