import os, sys
import cv2
import torch
import numpy as np
import pandas as pd
import joblib
import segmentation_models_pytorch as smp

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "models", "best_unet_refuge2.pth")
RF_MODEL_PATH = os.path.join(BASE_PATH, "models", "glaucoma_rf_model.pkl")

IMG_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_rf_model = None

def get_model():
    global _model
    if _model is None:
        _model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=3,
            activation=None,
        )
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ODOC Model not found at: {MODEL_PATH}")
            
        state = torch.load(MODEL_PATH, map_location=device)
        _model.load_state_dict(state)
        _model.to(device)
        _model.eval()
    return _model

def get_rf_model():
    global _rf_model
    if _rf_model is None:
        if not os.path.exists(RF_MODEL_PATH):
            raise FileNotFoundError(f"RF Model not found at: {RF_MODEL_PATH}")
        _rf_model = joblib.load(RF_MODEL_PATH)
    return _rf_model

def to_probs(arr, axis=0):
    exp = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
    return exp / (exp.sum(axis=axis, keepdims=True) + 1e-8)

def clean_mask(mask_binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1: return np.zeros_like(mask_binary)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)

def extract_clinical_features(pred_mask):
    raw_od = ((pred_mask == 1) | (pred_mask == 2)).astype(np.uint8)
    raw_oc = (pred_mask == 2).astype(np.uint8)
    
    od_mask = clean_mask(raw_od)
    oc_mask = clean_mask(raw_oc)
    
    features = {
        'v_cdr': 0.0, 'area_cdr': 0.0, 'disc_area_pct': 0.0, 'rim_area_pct': 0.0,
        'inf_rim_pct': 0.0, 'sup_rim_pct': 0.0, 'nasal_temp_1_pct': 0.0, 'nasal_temp_2_pct': 0.0,
        'od_eccentricity': 0.0, 'od_solidity': 0.0, 'oc_eccentricity': 0.0, 'oc_solidity': 0.0
    }
    
    total_img_area = pred_mask.shape[0] * pred_mask.shape[1]
    disc_area = np.sum(od_mask)
    cup_area = np.sum(oc_mask)
    
    if disc_area == 0: return None # Indicate failure to find OD
        
    features['disc_area_pct'] = (disc_area / total_img_area) * 100
    features['area_cdr'] = cup_area / disc_area
    
    def get_vertical_diameter(m):
        rows = np.any(m, axis=1)
        if not rows.any(): return 0
        idx = np.where(rows)[0]
        return int(idx[-1] - idx[0] + 1)
        
    od_vd = get_vertical_diameter(od_mask)
    oc_vd = get_vertical_diameter(oc_mask)
    features['v_cdr'] = oc_vd / od_vd if od_vd > 0 else 0.0
    
    rim_mask = (od_mask > 0) & (oc_mask == 0)
    features['rim_area_pct'] = (np.sum(rim_mask) / total_img_area) * 100
    
    M = cv2.moments(od_mask)
    if M["m00"] != 0: cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else: cx, cy = pred_mask.shape[1]//2, pred_mask.shape[0]//2
        
    h, w = pred_mask.shape
    y, x = np.indices((h, w))
    angles = np.arctan2(y - cy, x - cx)
    
    features['inf_rim_pct'] = (np.sum(rim_mask & ((angles >= np.pi/4) & (angles <= 3*np.pi/4))) / total_img_area) * 100
    features['sup_rim_pct'] = (np.sum(rim_mask & ((angles >= -3*np.pi/4) & (angles <= -np.pi/4))) / total_img_area) * 100
    features['nasal_temp_1_pct'] = (np.sum(rim_mask & ((angles >= -np.pi/4) & (angles <= np.pi/4))) / total_img_area) * 100
    features['nasal_temp_2_pct'] = (np.sum(rim_mask & ((angles >= 3*np.pi/4) | (angles <= -3*np.pi/4))) / total_img_area) * 100
    
    def get_shape_features(m):
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0.0, 0.0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0.0
        if len(cnt) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
            a, b = ma/2, MA/2
            eccentricity = np.sqrt(abs(1 - (b**2)/(a**2))) if a > 0 and b > 0 else 0.0
        else: eccentricity = 0.0
        return eccentricity, solidity

    features['od_eccentricity'], features['od_solidity'] = get_shape_features(od_mask)
    if cup_area > 0: features['oc_eccentricity'], features['oc_solidity'] = get_shape_features(oc_mask)
        
    return features

def processing(image_rgb, threshold=0.5, batch_size=8):
    orig_h, orig_w = image_rgb.shape[:2]
    
    # 1. U-Net Segmentation
    img_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img_norm = (img_resized - MEAN) / STD
    img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    net = get_model()
    with torch.no_grad():
        logits = net(img_t)
        
    out_np = logits.squeeze().cpu().numpy()
    probs = to_probs(out_np, axis=0)
    
    pred = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    pred[probs[1] >= threshold] = 1   
    pred[probs[2] >= threshold] = 2   

    # 2. Extract Features
    features = extract_clinical_features(pred)
    rf_pred_label = "Unknown"
    rf_prob = 0.0
    
    # 3. Random Forest Classification
    if features is not None:
        rf = get_rf_model()
        # Convert dictionary to DataFrame to maintain exact feature names/order
        features_df = pd.DataFrame([features])
        rf_prob = rf.predict_proba(features_df)[0, 1]
        
        # Apply the optimized 0.35 threshold
        rf_pred_label = "Glaucoma" if rf_prob >= 0.35 else "Normal"

    # 4. Create Overlays
    full_res_pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    output_img = image_rgb.copy()

    mask_od = (full_res_pred == 1).astype(np.uint8)
    contours_od, _ = cv2.findContours(mask_od, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_img, contours_od, -1, (0, 255, 0), 3)

    mask_oc = (full_res_pred == 2).astype(np.uint8)
    contours_oc, _ = cv2.findContours(mask_oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_img, contours_oc, -1, (255, 0, 0), 3)

    return output_img, features, rf_pred_label, rf_prob, full_res_pred