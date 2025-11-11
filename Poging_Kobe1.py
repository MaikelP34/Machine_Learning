import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import random
from PIL import Image

# Setup device-agnostic code
if torch.cuda.is_available():
    device = "cuda"
    print("using GPU") 
else:
    device = "cpu"
    print("using CPU")

# Setup path to data folder
data_path = Path("C:\\School\\3de ba\\mach\\taak\\dataset")
image_path = data_path / 'train\\images'

if image_path.is_dir():
    print("directory found")
else:
    print("NO directory found")

# Setup train and testing paths
train_dir = data_path / "train"
test_dir = data_path / "test"

print(train_dir, test_dir)

random.seed(0)
image_path_list = list(image_path.glob("*.jpg"))
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.name #TODO juiste classificatie vinden

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")

plt.figure(figsize=(8, 8))
plt.imshow(img)  # Convert from CxHxW to HxWxC for display
plt.axis('off')
plt.show()