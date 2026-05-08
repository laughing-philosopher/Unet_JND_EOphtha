import cv2
import numpy as np
import math

from model import build_UNet

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


def processing(img, threshold, batch_size, progress_callback=None):
    """
    progress_callback: optional callable(current_batch, total_batches)
                       called after each batch so the UI can update a progress bar.
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
    step = 8
    for i in range(0, m, step):
        for j in range(0, n, step):
            if (i + Img_Height) > (m - 1) or (j + Img_Width) > (n - 1):
                continue
            iarr.append(i)
            jarr.append(j)

    tot = len(iarr)
    if tot == 0:
        return np.zeros((m, n), dtype=np.float32)

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
        return np.zeros_like(final_res)

    psm_th2 = final_res / mx
    psm_th2 = (psm_th2 > 0.1).astype(float)

    return psm_th2