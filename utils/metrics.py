import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def calculate_psnr(img1, img2, data_range=1.0):
    """Calculate PSNR between two images"""
    return compare_psnr(img1, img2, data_range=data_range)

def calculate_ssim(img1, img2, data_range=1.0, win_size=3):
    """Calculate SSIM between two images"""
    return compare_ssim(img1, img2, channel_axis=-1, data_range=data_range, win_size=win_size)