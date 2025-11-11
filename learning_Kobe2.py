# Import PyTorch
import torch
from torch import nn

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

#Copilot
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Get all image files from directory
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # Extract class labels from filenames (e.g. "img_2_..." -> "img_2")
        import re
        def extract_class(fname):
            m = re.search(r'(img_\d+)', fname)
            return m.group(1) if m else 'unknown'
        self._labels_str = [extract_class(f) for f in self.img_files]
        # Build class_names and mapping to indices
        unique = sorted(set(self._labels_str), key=lambda x: int(x.split('_')[1]) if x.startswith('img_') and x.split('_')[1].isdigit() else x)
        self.class_names = unique
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        # Map filenames to indices
        self.labels = [self.class_to_idx[l] for l in self._labels_str]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('L')  # grayscale anders 'RGB' voor kleur
        if self.transform:
            image = self.transform(image)
        label_idx = self.labels[idx]
        return image, label_idx  # Returns image and integer label index

# Define transforms
transform = transforms.Compose([#TODO werkelijk (480x640)
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
])

# Create dataset
dataset = CustomImageDataset(
    img_dir=r'C:\\School\\3de ba\\mach\\taak\\dataset\\train\\images',
    transform=transform
)

# print info
# Zie de eerste trainingssample
image, label_idx = dataset[0]
print(f"Filename: {dataset.img_files[0]}")
print(f"Label index: {label_idx}, Label name: {dataset.class_names[label_idx]}")

# Wat is de vorm van de afbeelding?
print(image.shape)

# Hoeveel samples zijn er?
print(len(dataset))

# Plot more images (gebruik label name)
torch.manual_seed(42)
fig = plt.figure(figsize=(8, 8))
rows, cols = 2, 3
i = 0
while (i < rows * cols and i < len(dataset)):
    image, label_idx = dataset[i]
    fig.add_subplot(rows, cols, i+1)
    plt.imshow(image.squeeze(), cmap='gray')  # grayscale single channel
    plt.title(dataset.class_names[label_idx])
    plt.axis(False)
    i = i + 1
plt.show()