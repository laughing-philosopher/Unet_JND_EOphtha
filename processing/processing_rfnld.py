import cv2
import tensorflow as tf
import numpy as np
from math import sqrt, ceil
from scipy.cluster.hierarchy import fclusterdata
import sys
import os

# ── Path helpers ──────────────────────────────────────────────────────────── #
BASE_PATH = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_PATH, "models", "retinet_9010.h5")

# ── Lazy model loader (singleton) ─────────────────────────────────────────── #
_model = None

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(model_path)
    return _model


(cx, cy) = (-1, -1)
(rx, ry) = (-1, -1)

def removearray(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


def extract_bv(imag = None):
    (b, green_fundus, r) = cv2.split(imag)
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize = (5, 5))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    clahe = cv2.createCLAHE(clipLimit = 1, tileGridSize = (5, 5))
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations = 1)
    f4 = cv2.subtract(R3, contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)
    (ret, f6) = cv2.threshold(f5, 3, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype = 'uint8') * 255
    (contours, hierarchy) = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 150:
            cv2.drawContours(mask, [
                cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask = mask)
    (ret, fin) = cv2.threshold(im, 3, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(imag.shape[:2], dtype = 'uint8') * 255
    (xcontours, xhierarchy) = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = 'unidentified'
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = 'circle'
        else:
            shape = 'veins'
            if shape == 'circle':
                cv2.drawContours(xmask, [
                    cnt], -1, 0, -1)
            finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask = xmask)
            blood_vessels = cv2.bitwise_not(finimage)
            return (contrast_enhanced_green_fundus, blood_vessels)


def derive_coordinates_from_odoc(odoc_measurements):
    """
    Derive the optic disc center (coordinates) and a radius-defining point
    (coordinates2) from the ODOC segmentation measurements.

    Parameters
    ----------
    odoc_measurements : dict
        Measurements dict from the ODOC overlay module (overlay_odoc.py).
        Expected keys: disc_center (tuple), disc_radius_px, disc_vert_diam_px,
        disc_horiz_diam_px.

    Returns
    -------
    (coordinates, coordinates2) : tuple of dicts or (None, None)
        coordinates  = {'x': center_x, 'y': center_y}  — optic disc center
        coordinates2 = {'x': edge_x,   'y': edge_y}    — point on disc edge (defines radius)
    """
    # disc_center is stored as a (cx, cy) tuple by overlay_odoc.py
    disc_center = odoc_measurements.get('disc_center')
    if disc_center is None or len(disc_center) < 2:
        return None, None

    center_x, center_y = int(disc_center[0]), int(disc_center[1])

    # Use disc_radius_px directly if available, otherwise compute from diameters
    radius = odoc_measurements.get('disc_radius_px', 0)
    if radius <= 0:
        vert_diam  = odoc_measurements.get('disc_vert_diam_px', 0)
        horiz_diam = odoc_measurements.get('disc_horiz_diam_px', 0)
        radius = int((vert_diam + horiz_diam) / 4)  # diameter/2 averaged

    if radius <= 0:
        return None, None

    # coordinates = disc center, coordinates2 = a point on the disc edge
    coordinates  = {'x': center_x, 'y': center_y}
    coordinates2 = {'x': center_x + int(radius), 'y': center_y}

    return coordinates, coordinates2


def processing(img, coordinates, coordinates2):
    """
    Run RFNLD detection on a fundus image.

    Parameters
    ----------
    img : np.ndarray (H, W, 3), uint8, BGR/RGB fundus image.
    coordinates : dict with 'x', 'y' — optic disc center.
    coordinates2 : dict with 'x', 'y' — point on disc edge (defines radius).

    Returns
    -------
    dict with keys:
        image        – np.ndarray (H, W, 3), annotated image with RFNLD lines
        defects_found – bool, whether RFNLD defects were detected
        defect_count  – int, number of defect clusters detected
    """
    model = get_model()
    N = 32

    orig_img = img.copy()
    (contrast_enhanced_green_fundus, blood_vessels) = extract_bv(img)
    (ret, output) = cv2.threshold(blood_vessels, 0, 255, cv2.THRESH_BINARY_INV)
    dst = cv2.inpaint(contrast_enhanced_green_fundus, output, 3, cv2.INPAINT_NS)
    cv2.equalizeHist(dst)
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize = (5, 5))
    img1 = clahe.apply(dst)
    img2 = img1.copy()
    print('RFNLD: Preprocessing complete!')

    rx = coordinates['x']
    ry = coordinates['y']
    cx = coordinates2['x']
    cy = coordinates2['y']
    x = rx
    y = ry
    r = int(sqrt((cx - rx) * (cx - rx) + (cy - ry) * (cy - ry)))
    drawing = 2
    print('RFNLD: ROI generated!')
    img3 = orig_img.copy()
    img = img2 / 255
    (height, width) = img.shape
    img_point = np.zeros((height, width), np.uint8)
    cv2.circle(img_point, (x, y), int(3 * r), 255, -1)
    cv2.circle(img_point, (x, y), r, 0, -1)
    pt = np.transpose(np.where(np.equal(img_point, 255)))
    pt = pt.astype(np.int32)
    patch_predict = []
    print('RFNLD: Given to model')
    for j in range(0, len(pt)):
        patch = img[pt[j][0] - N // 2:pt[j][0] + N // 2, pt[j][1] - N // 2:pt[j][1] + N // 2]
        (p, q) = patch.shape
        if p == N and q == N:
            patch_predict.append(patch)
        else:
            padded_patch = np.lib.pad(patch, ((ceil((N - p) / 2), (N - p) // 2), (ceil((N - q) / 2), (N - q) // 2)), 'constant')
            patch_predict.append(padded_patch)
    print(f'RFNLD: Patch extraction completed — {len(patch_predict)} patches')

    # Process patches in batches to avoid OOM on large images
    BATCH_SIZE = 1024
    c = []
    total_patches = len(patch_predict)
    for batch_start in range(0, total_patches, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_patches)
        batch = np.array(patch_predict[batch_start:batch_end])
        bp, bq, bs = batch.shape
        batch = batch.reshape(bp, bq, bs, 1)
        batch_preds = model.predict(batch, batch_size=256, verbose=0)
        for k in range(len(batch_preds)):
            if batch_preds[k][1] > 0.9:
                idx = batch_start + k
                c.append((pt[idx][1], pt[idx][0]))
        if (batch_start // BATCH_SIZE) % 10 == 0:
            print(f'RFNLD: Predicted {min(batch_end, total_patches)}/{total_patches} patches')
    # Free the patch list from memory
    del patch_predict
    print('RFNLD: Pixelwise predictions complete!')
    c = np.asarray(c, dtype = np.float32)
    thresh = 4

    try:
        clusters = fclusterdata(c, thresh, criterion = 'distance')
    except:
        clusters = np.asarray([
            0])

    l = len(np.unique(clusters))
    cluster_points = []
    for j in range(0, l):
        points = []
        for k in range(0, len(c)):
            if clusters[k] == j + 1:
                points.append([
                    c[k][0],
                    c[k][1]])
        points = np.asarray(points)
        cluster_points.append(points)

    defect_count = 0
    max_cl_len = 0
    for j in range(0, l):
        if len(cluster_points[j]) > max_cl_len:
            max_cl_len = len(cluster_points[j])
    if max_cl_len > 80:
        k = len(cluster_points)
        j = 0
        while j < k:
            if len(cluster_points[j]) < int(0.4 * max_cl_len):
                removearray(cluster_points, cluster_points[j])
            else:
                j = j + 1
            k = len(cluster_points)
        param = []
        for j in range(0, len(cluster_points)):
            (slope, intercept) = np.polyfit(cluster_points[j][:, 0], cluster_points[j][:, 1], 1, rcond = None, full =False, w = None, cov = False)
            param.append((slope, intercept))
        
        defect_count = len(cluster_points)
        for j in range(0, len(cluster_points)):
            minn = sqrt((cluster_points[j][0][0] - x) ** 2 + (cluster_points[j][0][1] - y) ** 2)
            maxx = sqrt((cluster_points[j][0][0] - x) ** 2 + (cluster_points[j][0][1] - y) ** 2)
            for k in range(0, len(cluster_points[j])):
                if sqrt((cluster_points[j][k][0] - x) ** 2 + (cluster_points[j][k][1] - y) ** 2) <= minn:
                    minn = sqrt((cluster_points[j][k][0] - x) ** 2 + (cluster_points[j][k][1] - y) ** 2)
                    min_pt = (cluster_points[j][k][0], cluster_points[j][k][1])
                if sqrt((cluster_points[j][k][0] - x) ** 2 + (cluster_points[j][k][1] - y) ** 2) >= maxx:
                    maxx = sqrt((cluster_points[j][k][0] - x) ** 2 + (cluster_points[j][k][1] - y) ** 2)
                    max_pt = (cluster_points[j][k][0], cluster_points[j][k][1])
            q1 = ((param[j][0] * min_pt[0] - min_pt[1]) + param[j][1]) / (param[j][0] ** 2 + 1)
            q2 = ((param[j][0] * max_pt[0] - max_pt[1]) + param[j][1]) / (param[j][0] ** 2 + 1)
            p1 = (min_pt[0] - param[j][0] * q1, q1 + min_pt[1])
            p2 = (max_pt[0] - param[j][0] * q2, q2 + max_pt[1])
            print('RFNLD: Clustering completed')
            cv2.line(img3, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 4)
        
        return {
            "image": img3,
            "defects_found": True,
            "defect_count": defect_count,
        }
    else:
        return {
            "image": img3,
            "defects_found": False,
            "defect_count": 0,
        }