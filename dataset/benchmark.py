import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BenchmarkDataset(Dataset):
    """Dataset for benchmarks (Set5, Set14, BSD100, Urban100)"""

    def __init__(self, benchmark_dir, scale=4):
        self.benchmark_dir = benchmark_dir
        self.scale = scale
        self.to_tensor = transforms.ToTensor()

        # Load image pairs
        self.image_pairs = []
        self._load_srf_images(benchmark_dir, scale)

        print(f"Loaded {len(self.image_pairs)} image pairs from {os.path.basename(benchmark_dir)}")

    def _load_srf_images(self, benchmark_dir, scale):
        """Load images with proper naming"""
        # Find all HR and LR images
        hr_images = {}
        lr_images = {}

        for filename in os.listdir(benchmark_dir):
            if filename.endswith('.png') and f'SRF_{scale}' in filename:
                if filename.endswith(f'_SRF_{scale}_HR.png'):
                    base_name = filename.replace(f'_SRF_{scale}_HR.png', '')
                    hr_images[base_name] = os.path.join(benchmark_dir, filename)
                elif filename.endswith(f'_SRF_{scale}_LR.png'):
                    base_name = filename.replace(f'_SRF_{scale}_LR.png', '')
                    lr_images[base_name] = os.path.join(benchmark_dir, filename)

        # Match HR and LR pairs
        for base_name in sorted(hr_images.keys()):
            if base_name in lr_images:
                self.image_pairs.append({
                    'name': base_name,
                    'hr_path': hr_images[base_name],
                    'lr_path': lr_images[base_name]
                })
            else:
                print(f"Warning: No LR image found for {base_name}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]

        # Load HR and LR images
        hr = Image.open(pair['hr_path']).convert('RGB')
        lr = Image.open(pair['lr_path']).convert('RGB')

        return self.to_tensor(lr), self.to_tensor(hr), pair['name']