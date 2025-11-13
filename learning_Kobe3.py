import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import re
import numpy as np
import random
from typing import List
import requests

# Custom Image Dataset
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        data_dir = Path(data_dir)  # Convert string to Path object
        self.img_dir = data_dir / "images"
        self.labels_dir = data_dir / "labels"
        self.transform = transform
        self.img_files = sorted([f for f in self.img_dir.glob("*.jpg")])

        # Extract class labels from corresponding .txt files
        def extract_class(img_file):
            base_name = img_file.stem  # Get the base name without extension
            txt_file = self.labels_dir / f"{base_name}.txt"
            
            if txt_file.exists():
                with open(txt_file, 'r') as f:
                    class_label = f.read(1)  # Read only first character
                    return class_label
            else:
                # Fallback: extract from filename if no .txt file
                m = re.search(r'(img_\d+)', base_name)
                return m.group(1) if m else 'unknown'
        
        self._labels_str = [extract_class(f) for f in self.img_files]
        self.class_names = sorted(set(self._labels_str))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.labels = [self.class_to_idx[l] for l in self._labels_str]
        
        # Debug: print class info
        print(f"Classes found: {self.class_names}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"First 10 labels: {self._labels_str[:10]}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        label_idx = self.labels[idx]
        return image, label_idx

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),
])

# Create dataset
dataset = CustomImageDataset(
    data_dir="C:\\School\\3de ba\\mach\\taak\\dataset\\train",
    transform=transform
)

# Check the first training sample
image, label_idx = dataset[0]
print(f"Filename: {dataset.img_files[0]}")
print(f"Label index: {label_idx}, Label name: {dataset.class_names[label_idx]}")
print(f"Image shape: {image.shape}")
print(f"Number of samples: {len(dataset)}")

# Print dataset details
print(f"Image directory: {dataset.img_dir}")
print(f"Image directory exists: {dataset.img_dir.exists()}")
print(f"Number of images found: {len(dataset.img_files)}")
if len(dataset.img_files) > 0:
    print(f"First few images: {[f.name for f in dataset.img_files[:5]]}")
else:
    print("WARNING: No images found!")
    # List what's actually in the images directory
    if dataset.img_dir.exists():
        print(f"Contents of {dataset.img_dir}:")
        for item in dataset.img_dir.iterdir():
            print(f"  - {item.name}")

# Plot a few images
torch.manual_seed(42)
fig = plt.figure(figsize=(8, 8))
rows, cols = 2, 3
for i in range(rows * cols):
    if i < len(dataset):
        image, label_idx = dataset[i]
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(dataset.class_names[label_idx])
        ax.axis("off")
plt.show()

# Create DataLoaders
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Definition
class SimpleModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

# Model initialization
input_shape = 128 * 128  # Image size 128x128
hidden_units = 128  # Arbitrary choice for hidden layer size
output_shape = len(dataset.class_names)  # Number of classes

model = SimpleModel(input_shape, hidden_units, output_shape)
model.to("cpu")

# Helper function for evaluating accuracy
from helper_functions import accuracy_fn  # Assumes the helper function is already available

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

# Training Loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n-------")
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    train_loss /= len(train_dataloader)

    # Testing Loop
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in test_dataloader:
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

# Model Evaluation
def eval_model(model, data_loader, loss_fn, accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_loss": loss.item(), "model_acc": acc}

# Evaluate the model on test data
model_results = eval_model(model, test_dataloader, loss_fn, accuracy_fn)
print(model_results)
