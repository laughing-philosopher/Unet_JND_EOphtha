
from kagglehub import model_download

# Download pretrained EfficientNet-B6 model for DR grading
model_path = model_download("wlyyyyy/efficientnetb6/PyTorch/default")
print(model_path)
