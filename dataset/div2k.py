import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DIV2KTrainDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=96, scale=4, augment=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

        self.filenames = []
        if os.path.exists(lr_dir):
            for f in os.listdir(lr_dir):
                if f.endswith("x4.png"):
                    base = f.replace("x4.png", "")
                    if os.path.exists(os.path.join(hr_dir, f"{base}.png")):
                        self.filenames.append(base)

        self.filenames.sort()
        print(f"Found {len(self.filenames)} training image pairs")

    def __len__(self):
        # Data augmentation increases dataset size
        return len(self.filenames) * (8 if self.augment else 1)

    def __getitem__(self, idx):
        # Handle augmented indexing
        if self.augment:
            file_idx = idx // 8
            aug_idx = idx % 8
        else:
            file_idx = idx
            aug_idx = 0

        name = self.filenames[file_idx]
        hr_path = os.path.join(self.hr_dir, f"{name}.png")
        lr_path = os.path.join(self.lr_dir, f"{name}x4.png")

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # Get LR and HR crop sizes
        lr_crop_size = self.patch_size
        hr_crop_size = self.patch_size * self.scale

        # Random crop positions
        lr_w, lr_h = lr.size
        x = random.randint(0, lr_w - lr_crop_size)
        y = random.randint(0, lr_h - lr_crop_size)

        # Crop matching patches
        lr_patch = lr.crop((x, y, x + lr_crop_size, y + lr_crop_size))
        hr_patch = hr.crop((x * self.scale, y * self.scale,
                           (x + lr_crop_size) * self.scale,
                           (y + lr_crop_size) * self.scale))

        # Apply data augmentation
        if self.augment and aug_idx > 0:
            # Rotation (90, 180, 270 degrees)
            if aug_idx in [1, 3, 5, 7]:
                angle = 90 * ((aug_idx + 1) // 2)
                lr_patch = lr_patch.rotate(angle, expand=True)
                hr_patch = hr_patch.rotate(angle, expand=True)

            # Horizontal flip
            if aug_idx in [2, 3, 6, 7]:
                lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)

            # Vertical flip
            if aug_idx in [4, 5, 6, 7]:
                lr_patch = lr_patch.transpose(Image.FLIP_TOP_BOTTOM)
                hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)


class DIV2KValDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.to_tensor = transforms.ToTensor()

        self.filenames = []
        if os.path.exists(lr_dir):
            for f in os.listdir(lr_dir):
                if f.endswith("x4.png"):
                    base = f.replace("x4.png", "")
                    if os.path.exists(os.path.join(hr_dir, f"{base}.png")):
                        self.filenames.append(base)

        # Use all validation images
        self.filenames = sorted(self.filenames)
        print(f"Found {len(self.filenames)} validation image pairs")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        hr_path = os.path.join(self.hr_dir, f"{name}.png")
        lr_path = os.path.join(self.lr_dir, f"{name}x4.png")

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        return self.to_tensor(lr), self.to_tensor(hr), name