import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import timm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../BTP/processing
ROOT_DIR = os.path.dirname(CURRENT_DIR)                    # .../BTP
model_path = os.path.join(ROOT_DIR, "models", "pytorch_model_effb6.bin")


def load_dr_model():
    # Step 1: Build model with 5-class head FIRST
    model = timm.create_model("efficientnet_b6", pretrained=False, num_classes=5)

    state = torch.load(model_path, map_location=torch.device("cpu"))

    # Unwrap nested state_dict if needed
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip 'module.' prefix from DataParallel checkpoints
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    # If checkpoint has 1000-class head, drop those weights and load the rest
    state = {
        k: v for k, v in state.items()
        if not k.startswith("classifier")
    }
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_dr_severity(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((528, 528)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        severity = predicted.item()

    severity_map = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR",
    }

    return severity_map.get(severity, "Unknown")