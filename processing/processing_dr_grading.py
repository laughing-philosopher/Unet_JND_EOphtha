import os
import sys

# Load the EfficientNetB6 model from local .bin file
def load_dr_model():
    import torch  # lazy import to avoid startup issues when packaging
    import timm   # lazy import
    # Resolve models directory robustly for both dev and PyInstaller-frozen builds
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.abspath(os.path.join(base_dir, os.pardir))
    # If running from a PyInstaller bundle, _MEIPASS points to the temp dir with bundled files
    if hasattr(sys, "_MEIPASS"):
        app_root = sys._MEIPASS
    model_path = os.path.join(app_root, "models", "pytorch_model_effb6.bin")
    # Create with 1000 classes to match typical EfficientNet checkpoints
    model = timm.create_model("efficientnet_b6", pretrained=False, num_classes=1000)
    state = torch.load(model_path, map_location=torch.device("cpu"))
    # If checkpoint was saved with DataParallel
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strip potential 'module.' prefix
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    # Replace classifier head to 5 classes after loading
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, 5)
    model.eval()
    return model

# Preprocess image and make prediction
def predict_dr_severity(image_path, model):
    import torch  # lazy import
    import torchvision.transforms as transforms  # lazy import
    from PIL import Image  # lazy import

    transform = transforms.Compose([
        transforms.Resize((528, 528)),  # EfficientNet-B6 input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        severity = predicted.item()

    severity_map = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }

    return severity_map.get(severity, "Unknown")
