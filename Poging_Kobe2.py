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
        image = Image.open(img_path).convert('L')
        
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

# Zie de eerste trainingssample
image, label = dataset[0]
print(image, label)

# Wat is de vorm van de afbeelding?
print(image.shape)

# Hoeveel samples zijn er?
print(len(dataset))

# Toon de afbeelding
plt.figure(figsize=(8, 8))
# Om de afbeelding weer te geven, draai de dimensies om van (C, H, W) naar (H, W, C)
plt.imshow(image.permute(1, 2, 0), cmap = 'gray')  # Verander de volgorde naar (H, W, C)
plt.title(label)
plt.axis('off')  # Verberg de assen
plt.show()

# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(8, 8))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(dataset), size=[1]).item()
    img, label = dataset[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(label)
    plt.axis(False)