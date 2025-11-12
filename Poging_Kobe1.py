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


from typing import List

import pathlib

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

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataTransform = transform(img)

#print_img(img)
#data()

#TODO copilot, ook op de site
# in aparte file zetten?
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = data_dir / "images"
        self.transform = transform
        self.img_files = sorted([f for f in Path(data_dir / "images").glob("*.jpg")])

        # Extract class labels from corresponding .txt files
        def extract_class(img_file):
            # Get the base name without extension (e.g., "img_2_xyz.jpg" -> "img_2_xyz")
            base_name = img_file.stem
            # Look for corresponding .txt file (e.g., "img_2_xyz.txt")
            txt_file = data_dir / "labels" / f"{base_name}.txt"
            
            if txt_file.exists():
                with open(txt_file, 'r') as f:
                    class_label = f.read(1)
                    return class_label
            else:
                # Fallback: extract from filename if no .txt file
                m = re.search(r'(img_\d+)', base_name)
                return m.group(1) if m else 'unknown'
        
        self._labels_str = [extract_class(f) for f in self.img_files]
        unique = sorted(set(self._labels_str))
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

print(f"Train data: {len(train_data)} images, Classes: {train_data.class_to_idx}")
print(f"Test data: {len(test_data)} images, Classes: {test_data.class_to_idx}")
print(f"Train classes: {train_data.labels}")
print(f"Test classes: {test_data.labels}")

# 1. Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset, classes: List[str] = None, n: int = 10, display_shape: bool = True, seed: int = None):
    
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(20, 8))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()

# Display random images from ImageFolder created Dataset
#display_random_images(train_data, n=5, classes=train_data.class_names,seed=None)

# Turn train and test custom Dataset's into DataLoader's
from torch.utils.data import DataLoader
train_dataloader_custom = DataLoader(dataset=train_data, # use custom created train Dataset
                                     batch_size=1, # how many samples per batch?
                                     num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

test_dataloader_custom = DataLoader(dataset=test_data, # use custom created test Dataset
                                    batch_size=1, 
                                    num_workers=0, 
                                    shuffle=False) # don't usually need to shuffle testing data

# Get image and label from custom DataLoader
img_custom, label_custom = next(iter(train_dataloader_custom))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label_custom.shape}")

from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense 
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
])

# Don't need to perform augmentation on the test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

# Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Create simple transform
simple_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

