from PIL import Image
from torch.utils.data import Dataset
import re
import pathlib as Path

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
