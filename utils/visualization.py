import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from tqdm import tqdm


def save_sr_examples(model, dataloader, save_dir="sr_results", max_batches=5, device="cuda"):
    """Save super-resolution examples"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            if len(batch) == 3:
                lr, hr, names = batch
            else:
                lr, hr = batch
                names = [f"img_{j}" for j in range(lr.size(0))]

            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr).clamp(0, 1)

            for j in range(lr.size(0)):
                idx = i * dataloader.batch_size + j
                name = names[j] if isinstance(names, list) else f"img_{idx:03d}"

                save_image(lr[j], os.path.join(save_dir, f"{name}_lr.png"))
                save_image(sr[j], os.path.join(save_dir, f"{name}_sr.png"))
                save_image(hr[j], os.path.join(save_dir, f"{name}_hr.png"))

    print(f"Saved {min(max_batches * dataloader.batch_size, len(dataloader.dataset))} image examples to '{save_dir}/'")


def save_comparison_images(lr_tensor, sr_tensor, hr_tensor, save_path, psnr_value=None):
    """Save a comparison image showing LR, SR, and HR side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(TF.to_pil_image(lr_tensor.cpu()))
    axes[0].set_title("LR Input", fontsize=12)
    axes[0].axis('off')

    title = "SR Output"
    if psnr_value is not None:
        title += f"\nPSNR: {psnr_value:.2f}dB"
    axes[1].imshow(TF.to_pil_image(sr_tensor.cpu()))
    axes[1].set_title(title, fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(TF.to_pil_image(hr_tensor.cpu()))
    axes[2].set_title("HR Ground Truth", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()