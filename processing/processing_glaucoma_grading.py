"""processing_glaucoma_grading.py
==================================
Glaucoma severity grading using a custom SE-ResNet50 model.
Model file : models/best_glaucoma_model.pth  (3-class output)

Architecture reconstructed from state dict inspection:
  backbone  : ResNet101 with per-stage Squeeze-Excitation attention (layer3 has 23 blocks)
  regressor : Linear(2048→512)→BN→ReLU→Drop→Linear(512→256)→BN→ReLU→Drop→Linear(256→3)

Classes: No Glaucoma | Glaucoma Suspect | Moderate Glaucoma
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

_BASE      = (os.environ.get('AAKHI_BASE_PATH') or
              os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(_BASE, "models", "best_glaucoma_model.pth")

SEVERITY_MAP = {
    0: "No Glaucoma",
    1: "Glaucoma Suspect",
    2: "Moderate Glaucoma",
}

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model  = None
_device = None


# ── Model definition ─────────────────────────────────────────────────────── #

class _SEBlock(nn.Module):
    """Channel Squeeze-Excitation block applied after a ResNet stage."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # fc.0
            nn.ReLU(inplace=True),                                    # fc.1 (no params)
            nn.Linear(channels // reduction, channels, bias=False),  # fc.2
            nn.Sigmoid(),                                             # fc.3 (no params)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        y = x.mean(dim=[2, 3])           # Global average pool
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class _Backbone(nn.Module):
    """ResNet101 backbone with per-stage SE attention.

    State dict structure:
      backbone.conv1.*
      backbone.bn1.*
      backbone.layer{1-4}.0.{0,1,2,...}.*   ← Bottleneck blocks
      backbone.layer{1-4}.1.fc.{0,2}.*      ← SE block
    ResNet101 block counts: [3, 4, 23, 3]
    """
    STAGE_CHANNELS = [256, 512, 1024, 2048]

    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        base = tvm.resnet101(weights=None)

        self.conv1 = base.conv1
        self.bn1   = base.bn1
        # Each stage: Sequential([ResNet stage blocks, SEBlock])
        self.layer1 = nn.Sequential(base.layer1, _SEBlock(256))
        self.layer2 = nn.Sequential(base.layer2, _SEBlock(512))
        self.layer3 = nn.Sequential(base.layer3, _SEBlock(1024))
        self.layer4 = nn.Sequential(base.layer4, _SEBlock(2048))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def _mlp_head(out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2048, 512),         # 0
        nn.BatchNorm1d(512),          # 1
        nn.ReLU(inplace=True),        # 2
        nn.Dropout(p=0.5),            # 3
        nn.Linear(512, 256),          # 4
        nn.BatchNorm1d(256),          # 5
        nn.ReLU(inplace=True),        # 6
        nn.Dropout(p=0.5),            # 7
        nn.Linear(256, out_features), # 8
    )


class _GlaucomaNet(nn.Module):
    """SE-ResNet101 backbone with a 3-class classifier head and a 1-output regressor head."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone   = _Backbone()
        self.classifier = _mlp_head(num_classes)  # classification head (used for grading)
        self.regressor  = _mlp_head(1)             # auxiliary regression head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ── Public API ────────────────────────────────────────────────────────────── #

def load_glaucoma_model() -> _GlaucomaNet:
    global _model, _device
    if _model is not None:
        return _model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(MODEL_PATH, map_location=_device)
    # Strip DataParallel prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model = _GlaucomaNet(num_classes=len(SEVERITY_MAP))
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"[Glaucoma] WARNING: {len(missing)} missing keys")
    model.to(_device)
    model.eval()

    _model = model
    print(f"[Glaucoma] Loaded SE-ResNet101 from {MODEL_PATH}  classes={len(SEVERITY_MAP)}")
    return _model


def predict_glaucoma_severity(image_path: str, model: _GlaucomaNet | None = None) -> str:
    if model is None:
        model = load_glaucoma_model()

    img    = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0).to(_device)

    model.eval()
    with torch.no_grad():
        out = model(tensor)
        idx = int(torch.argmax(out, dim=1).item())

    return SEVERITY_MAP.get(idx, f"Grade {idx}")
