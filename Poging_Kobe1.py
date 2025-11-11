#TODO extra uitleg en opbouw https://www.learnpytorch.io/03_pytorch_computer_vision/#31-setup-loss-optimizer-and-evaluation-metrics
#algemeen
import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
#voor selecteren
import random
from PIL import Image
#voor verwerking
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import re

#print functie
def print_img(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)  # Convert from CxHxW to HxWxC for display
    plt.axis('off')
    plt.show()

def data():
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}") 
    print(f"Image width: {img.width}")

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
train_dir = data_path / "train\\images"
test_dir = data_path / "test\\images"

print(train_dir, test_dir)

random.seed(0)
image_path_list = list(image_path.glob("*.jpg"))
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.name #TODO juiste classificatie vinden

# 4. Open image
img = Image.open(random_image_path)


transform = transforms.Compose([
    transforms.ToTensor(),
])

dataTransform = transform(img)

#print_img(img)
#data()

#TODO copilot
# Vervang de ImageFolder deel met:
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in Path(img_dir).glob("*.jpg")]
        
        # Extract class labels from filenames (e.g. "img_2_..." -> "img_2")
        def extract_class(fname):
            m = re.search(r'(img_\d+)', fname.name)
            return m.group(1) if m else 'unknown'
        
        self._labels_str = [extract_class(f) for f in self.img_files]
        unique = sorted(set(self._labels_str), key=lambda x: int(x.split('_')[1]) if x.startswith('img_') else x)
        self.class_names = unique
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.labels = [self.class_to_idx[l] for l in self._labels_str]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        label_idx = self.labels[idx]
        return image, label_idx

# Create datasets met custom class
train_data = CustomImageDataset(train_dir, transform=transform)
test_data = CustomImageDataset(test_dir, transform=transform)

print(f"Train data: {len(train_data)} images, Classes: {train_data.class_names}")
print(f"Test data: {len(test_data)} images")