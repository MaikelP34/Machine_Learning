#TODO testen met verschillende waarden, clean-up, ?zwart-wit?, ?nieuw model?, ptt
import os
import time
from collections import Counter

import torch as pt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomRotation, ColorJitter
)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ====================== CONFIG ======================
#name sequence
batch_size = 4
learning_rate = 2e-4
epochs = 20
img_size = 480

num_classes = 4

#base_url = "C:\\School\\3de ba\\mach\\taak\\dataset" #KOBE
base_url = "C:\\Users\\maike\\OneDrive\\Documents\\School\\Unif\\ML_2526\\Project\\dataset" #Maikel

#model_path = f"C:\\School\\3de ba\\mach\\taak\\models" #KOBE
model_path = "C:\\Users\\maike\\OneDrive\\Documents\\School\\Unif\\ML_2526\\Project\\models" #Maikel

model_save_path = os.path.join(model_path, f"ResNet18_{batch_size}_{learning_rate}_{epochs}_{img_size}.pth")

train_data_url = os.path.join(base_url, "train", "images")
valid_data_url = os.path.join(base_url, "valid", "images")
test_data_url  = os.path.join(base_url, "test",  "images")

train_labels_url = os.path.join(base_url, "train", "labels")
valid_labels_url = os.path.join(base_url, "valid", "labels")
test_labels_url  = os.path.join(base_url, "test",  "labels")

num_workers = min(4, os.cpu_count() or 0)  # safe default

# ====================== automatisering ======================
def automate(accuracy):
    R_accuracy = round(accuracy, 2)
    output_dir = "output"
    if not os.path.exists("output"):
        os.makedirs("output")

    data_path = os.path.join(output_dir, f"{R_accuracy}_RN50_data_{batch_size}_{learning_rate}_{epochs}_{img_size}" )
    #run directory
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    return data_path

# ====================== DATASET ======================
class ResnetDataset(Dataset):

    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    try:
                        label = int(line.split()[0])
                    except Exception:
                        label = 0

        if self.transform:
            image = self.transform(image)

        return image, pt.tensor(label, dtype=pt.long)

# ====================== TRANSFORMS ======================
train_transform = Compose([
    Resize((img_size, img_size)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(10),
    ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std =[0.229, 0.224, 0.225])
])

eval_transform = Compose([
    Resize((img_size, img_size)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std =[0.229, 0.224, 0.225])
])

# ====================== MODEL UTIL ======================
def get_resnet50(num_classes, device, pretrained=True, unfreeze_layer4=True):
    model = models.resnet50(pretrained=pretrained)
    # Freeze all params first
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze last block (layer4) if requested (fine-tuning)
    if unfreeze_layer4:
        for name, param in model.named_parameters():
            if name.startswith("layer4"):
                param.requires_grad = True
    # Replace head
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feats, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    model = model.to(device)
    return model

# ====================== TRAIN/VAL/TEST HELPERS ======================
def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    preds = []
    trues = []
    with pt.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device); labels = labels.to(device)
            out = model(imgs)
            p = out.argmax(dim=1)
            correct += (p == labels).sum().item()
            total += labels.size(0)
            preds.extend(p.cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, np.array(trues), np.array(preds)

def confusion_matrix_from_arrays(trues, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(trues, preds):
        cm[int(t), int(p)] += 1
    return cm

# ====================== MAIN ======================
def main():
    # Device
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets & loaders
    train_dataset = ResnetDataset(train_data_url, train_labels_url, transform=train_transform)
    valid_dataset = ResnetDataset(valid_data_url, valid_labels_url, transform=eval_transform)
    test_dataset  = ResnetDataset(test_data_url,  test_labels_url,  transform=eval_transform)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(valid_dataset)} | Test samples: {len(test_dataset)}")

    # Quick label distribution check
    try:
        labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
        print("Train label counts:", Counter(labels))
    except Exception as e:
        print("Could not compute label distribution:", e)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"))

    # Model, loss, optimizer
    model = get_resnet50(num_classes=num_classes, device=device, pretrained=True, unfreeze_layer4=True)
    criterion = nn.CrossEntropyLoss()
    # Only params that require_grad will be optimized (layer4 + fc)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_val = -1.0
    train_losses = []
    val_accuracies = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_loss = running_loss / max(1, batches)
        train_losses.append(avg_loss)

        val_acc, _, _ = compute_accuracy(model, valid_loader, device)
        val_accuracies.append(val_acc)
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:5.2f}% | Time elapsed: {elapsed/60:.1f} min")

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            pt.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_accuracy": val_acc
            }, model_save_path)
            print(f" -> New best model saved (val {val_acc:.2f}%)")

    # Load best model for final eval
    chk = pt.load(model_save_path, map_location=device)
    model.load_state_dict(chk["model_state"])

    test_acc, trues, preds = compute_accuracy(model, test_loader, device)
    print(f"\nFinal test accuracy: {test_acc:.2f}%")

    data_path = automate(best_val)

    cm = confusion_matrix_from_arrays(trues, preds, num_classes)
    cm_str = np.array2string(cm)
    fname = os.path.join(data_path, f"ConfusionMatrix_{batch_size}_{learning_rate}_{epochs}_{img_size}.txt")

    with open(fname, "w") as f:
        f.write(f"Final test accuracy: {test_acc:.2f}%\nConfusionMatrix_{batch_size}_{learning_rate}_{epochs}_{img_size}:\n   0 1 2 3\n{cm_str}\n")
    print("Confusion matrix (rows=true, cols=pred):\n" + cm_str)

    # Plot loss + val acc
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.title("Train loss per epoch")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.subplot(2,1,2)
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o')
    plt.title("Validation accuracy per epoch")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    fig_dir = os.path.join(data_path, f"training_summary_{batch_size}_{learning_rate}_{epochs}_{img_size}.png")
    plt.savefig(fig_dir)
    plt.show()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
