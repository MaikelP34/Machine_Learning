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

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Get all image files from directory
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.img_files[idx]  # Returns image and filename

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create dataset
dataset = CustomImageDataset(
    img_dir=r'C:\\School\\3de ba\\mach\\taak\\dataset\\train\\images',
    transform=transform
)

# See first training sample
image, label = dataset[0]
print(image, label)
# What's the shape of the image?
print(image.shape)
# How many samples are there? 
print(len(dataset))