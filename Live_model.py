import cv2
import torch as pt
from torch import nn
from torchvision import models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np

from PIL import Image

# ====================== CONFIG ======================
num_classes = 4
img_size = 480

model_path = "C:\\School\\3de ba\\mach\\taak\\models\\E_ResNet50_4_0.0002_200_480.pth"    # <-- pas dit aan

labels = ["class_0", "class_1", "class_2", "class_3"]  # <-- vul in!

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

# ====================== MODEL DEFINITIE (identiek aan training) ======================
def get_trained_resnet50(num_classes, device):
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)

    # Freeze alles
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze layer 4
    for name, param in model.named_parameters():
        if name.startswith("layer4"):
            param.requires_grad = True

    # Custom classifier
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feats, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    model.to(device)
    return model

# Laad model
model = get_trained_resnet50(num_classes, device)
checkpoint = pt.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ====================== TRANSFORM (identiek aan training) ======================
transform = Compose([
    Resize((img_size, img_size)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std =[0.229, 0.224, 0.225])
])

# ====================== REAL-TIME CAMERA LOOP ======================
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Preprocess: BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Apply transform
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with pt.no_grad():
        outputs = model(img_tensor)
        pred = outputs.argmax(1).item()

    # Label
    label = labels[pred]

    # Tekst op beeld
    cv2.putText(frame, f"q to quit, prediction: {label}", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 255, 225), 2)

    cv2.imshow("Realtime ResNet50", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
