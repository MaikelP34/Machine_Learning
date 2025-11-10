import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

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

# Create dataloader
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

# Test loading an image
image, filename = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Filename: {filename}")

plt.figure(figsize=(8, 8))
plt.imshow(image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC for display
plt.title(filename)
plt.axis('off')
plt.show()

# Example of iterating through the dataloader
print("Processing all batches:")
for batch_idx, (batch_images, batch_filenames) in enumerate(dataloader):
    print(f"\nBatch {batch_idx + 1}:")
    print(f"Batch shape: {batch_images.shape}")
    print(f"Batch filenames: {batch_filenames}")
    
    # Optioneel: Toon alle afbeeldingen in deze batch
    plt.figure(figsize=(15, 5))
    for idx in range(batch_images.shape[0]):
        plt.subplot(1, batch_images.shape[0], idx + 1)
        plt.imshow(batch_images[idx].permute(1, 2, 0))
        plt.title(batch_filenames[idx])
        plt.axis('off')
    plt.show()