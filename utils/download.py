import os
import shutil
import zipfile
import requests
from tqdm import tqdm
from config import Config


def download_file(url, filepath):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filepath, 'wb') as f, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def download_div2k_dataset(data_dir=None):
    """Download DIV2K dataset"""
    if data_dir is None:
        data_dir = Config.DIV2K_ROOT

    # Check if already downloaded
    if (os.path.exists(Config.TRAIN_HR_DIR) and
            os.path.exists(Config.VAL_HR_DIR)):
        print("DIV2K dataset already exists!")
        return {
            "train_hr": Config.TRAIN_HR_DIR,
            "train_lr": Config.TRAIN_LR_DIR,
            "val_hr": Config.VAL_HR_DIR,
            "val_lr": Config.VAL_LR_DIR
        }

    # Create directory
    os.makedirs(data_dir, exist_ok=True)

    # Download and extract
    for name, url in Config.DIV2K_URLS.items():
        zip_path = os.path.join(data_dir, f"{name}.zip")

        if not os.path.exists(zip_path.replace('.zip', '')):
            print(f"Downloading {name}...")
            download_file(url, zip_path)

            print(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

            # Remove zip to save space
            os.remove(zip_path)

    print("DIV2K dataset downloaded and extracted successfully!")

    # Return paths
    paths = {
        "train_hr": Config.TRAIN_HR_DIR,
        "train_lr": Config.TRAIN_LR_DIR,
        "val_hr": Config.VAL_HR_DIR,
        "val_lr": Config.VAL_LR_DIR
    }

    return paths