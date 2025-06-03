# ESRGAN Based Image Upscaler




## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```


## Quick Start

### 1. Training

Train the model with automatic dataset download:

```bash
python main.py --mode train
```

**Training Options:**
```bash
# Resume from checkpoint
python main.py --mode train --resume checkpoints/checkpoint_epoch_10.pth

# Skip pretraining phase
python main.py --mode train --skip-pretrain
```

### 2. Inference

Process single images or directories:

```bash
# Single image
python inference.py --model checkpoints/best_model.pth --input image.jpg

# Directory processing
python inference.py --model checkpoints/best_model.pth --input /path/to/images/

# Create comparison images
python inference.py --model checkpoints/best_model.pth --input image.jpg --comparison

# Large image processing with tiling
python inference.py --model checkpoints/best_model.pth --input large_image.jpg --tile-size 512
```

### 3. Evaluation

Evaluate trained models:

```bash
# Evaluate on DIV2K validation set
python main.py --mode eval --model-path checkpoints/best_model.pth --eval-div2k

# Evaluate on benchmark datasets
python main.py --mode eval --model-path checkpoints/best_model.pth --eval-benchmarks

# Compare with bicubic baseline
python main.py --mode compare --model-path checkpoints/best_model.pth
```

## Project Structure

```
esrgan/
├── config.py              # Configuration settings
├── main.py                # Main training/evaluation script
├── inference.py           # Inference script
├── requirements.txt       # Dependencies
├── models/
│   ├── generator.py       # ESRGAN Generator with RRDB blocks
│   ├── discriminator.py   # Discriminator network
│   └── vgg_extractor.py   # VGG feature extractor
├── dataset/
│   ├── div2k.py          # DIV2K dataset loader
│   └── benchmark.py      # Benchmark dataset loader
├── training/
│   ├── trainer.py        # Training logic
│   └── evaluator.py      # Evaluation metrics
├── utils/
│   ├── losses.py         # Loss functions
│   ├── metrics.py        # PSNR/SSIM calculations
│   ├── visualization.py  # Result visualization
│   └── download.py       # Dataset download utilities
├── data/                 # Datasets
├── checkpoints/          # Model checkpoints 
└── results/              # Training results
```

## Configuration

Key settings in `config.py`:

```python
# Model settings
SCALE_FACTOR = 4          
NUM_RRDB_BLOCKS = 5       
PATCH_SIZE = 96           
BATCH_SIZE = 4            

# Training phases
PRETRAIN_EPOCHS = 15      
GAN_EPOCHS = 50           

# For Memory optimization
USE_MIXED_PRECISION = True
GRADIENT_ACCUMULATION_STEPS = 2
```







