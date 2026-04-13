import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================
MODEL_PATH = "models/best_unet_refuge2.pth"
TEST_IMAGE = r"C:\Users\AYUSHI JAIN\Downloads\T0001.jpg"  # Replace with a real test image in your directory

# ==========================================
# PREPROCESSING CONSTANTS
# ==========================================
IMG_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THRESHOLD = 0.5

def preprocess_image(image_bgr):
    """Resizes, normalizes, and converts image to PyTorch tensor."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img_norm = (img_resized - MEAN) / STD
    img_t = torch.from_numpy(img_norm.transpose(2, 0, 1))
    return img_t.unsqueeze(0)

def to_probs(arr, axis=0):
    """Applies softmax along the specified axis."""
    exp = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
    return exp / (exp.sum(axis=axis, keepdims=True) + 1e-8)

def run_test():
    # 1. Validate paths
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Error: Could not find image '{TEST_IMAGE}'. Please add a test image.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Could not find model at '{MODEL_PATH}'. Check your paths.")
        return

    print(f"✅ Loading image: {TEST_IMAGE}")
    image_bgr = cv2.imread(TEST_IMAGE)
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 2. Load Model
    print("⏳ Loading U-Net (ResNet34) model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=3,
        activation=None,
    )
    
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 3. Run Inference
    print("⏳ Running inference...")
    tensor = preprocess_image(image_bgr).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        
    out_np = logits.squeeze().cpu().numpy()

    # 4. Process Output
    print(f"✅ Raw output shape: {out_np.shape}")
    probs = to_probs(out_np, axis=0)
    p_od = probs[1]
    p_oc = probs[2]
    
    print(f"   Optic Disc (OD) Max Prob: {p_od.max():.4f}")
    print(f"   Optic Cup (OC) Max Prob:  {p_oc.max():.4f}")

    # Create masks based on threshold
    pred = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    pred[p_od >= THRESHOLD] = 1   # Optic Disc
    pred[p_oc >= THRESHOLD] = 2   # Optic Cup (Overrides OD)

    # Apply colors (Green for OD, Red for OC)
    color_mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    color_mask[pred == 1] = (0, 255, 0)   
    color_mask[pred == 2] = (255, 0, 0)   

    # Resize mask back to original image dimensions
    color_mask_resized = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Create Blended Overlay
    blend = cv2.addWeighted(image_rgb, 0.6, color_mask_resized, 0.4, 0)

    # 5. Visualization
    print("📈 Plotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(color_mask_resized)
    axes[1].set_title("Segmentation Mask\n(Green=Disc, Red=Cup)")
    axes[1].axis("off")
    
    axes[2].imshow(blend)
    axes[2].set_title("Blended Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()