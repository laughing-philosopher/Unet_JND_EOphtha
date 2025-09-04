import cv2
import tensorflow as tf
import numpy as np
from math import sqrt, ceil
from scipy.cluster.hierarchy import fclusterdata
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../BTP/processing
ROOT_DIR = os.path.dirname(CURRENT_DIR)                    # .../BTP
model_path = os.path.join(ROOT_DIR, "models", "retinet_9010.h5")

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

def processing(img, coordinates, coordinates2):
    model = tf.keras.models.load_model(model_path)
    N = 32

    # img = cv2.resize(img, (0, 0), fx = 0.33, fy = 0.33)
    orig_img = img.copy()
    (contrast_enhanced_green_fundus, blood_vessels) = extract_bv(img)
    (ret, output) = cv2.threshold(blood_vessels, 0, 255, cv2.THRESH_BINARY_INV)
    dst = cv2.inpaint(contrast_enhanced_green_fundus, output, 3, cv2.INPAINT_NS)
    cv2.equalizeHist(dst)
    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize = (5, 5))
    img1 = clahe.apply(dst)
    img2 = img1.copy()
    print('Preprocessing complete!')

    rx = coordinates['x']
    ry = coordinates['y']
    cx = coordinates2['x']
    cy = coordinates2['y']
    x = rx
    y = ry
    r = int(sqrt((cx - rx) * (cx - rx) + (cy - ry) * (cy - ry)))
    drawing = 2
    print('ROI generated!')
    img3 = orig_img.copy()
    img = img2 / 255
    (height, width) = img.shape
    img_point = np.zeros((height, width), np.uint8)
    cv2.circle(img_point, (x, y), int(3 * r), 255, -1)
    cv2.circle(img_point, (x, y), r, 0, -1)
    pt = np.transpose(np.where(np.equal(img_point, 255)))
    pt = pt.astype(np.int32)
    patch_predict = []
    print('Given to model')
    for j in range(0, len(pt)):
        patch = img[pt[j][0] - N // 2:pt[j][0] + N // 2, pt[j][1] - N // 2:pt[j][1] + N // 2]
        (p, q) = patch.shape
        if p == N and q == N:
            patch_predict.append(patch)
        else:
            padded_patch = np.lib.pad(patch, ((ceil((N - p) / 2), (N - p) // 2), (ceil((N - q) / 2), (N - q) // 2)), 'constant')
            patch_predict.append(padded_patch)
    print('Predictions completed')
    patch_predict = np.array(patch_predict)
    p, q, s = patch_predict.shape
    patch_predict = patch_predict.reshape(p, q, s, 1)
    predictions = model.predict(patch_predict, batch_size = None, verbose = 1, steps = None)
    print('Pixelwise predictions complete!')
    c = []
    for k in range(0, len(patch_predict)):
        if predictions[k][1] > 0.9:
            c.append((pt[k][1], pt[k][0]))
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
            print('Clustering completed')
            cv2.line(img3, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 4)
        
        return img3
    else:
        return img3