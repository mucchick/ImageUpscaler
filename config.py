import torch
import os


class Config:
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data paths
    DATA_ROOT = "data"
    RESULTS_ROOT = "results"
    CHECKPOINTS_ROOT = "checkpoints"

    # DIV2K dataset paths
    DIV2K_ROOT = os.path.join(DATA_ROOT, "DIV2K")
    TRAIN_HR_DIR = os.path.join(DIV2K_ROOT, "DIV2K_train_HR")
    TRAIN_LR_DIR = os.path.join(DIV2K_ROOT, "DIV2K_train_LR_bicubic", "X4")
    VAL_HR_DIR = os.path.join(DIV2K_ROOT, "DIV2K_valid_HR")
    VAL_LR_DIR = os.path.join(DIV2K_ROOT, "DIV2K_valid_LR_bicubic", "X4")

    # Benchmark dataset paths
    BENCHMARKS_ROOT = os.path.join(DATA_ROOT, "benchmarks")

    # Model configuration
    SCALE_FACTOR = 4
    NUM_RRDB_BLOCKS = 5
    CHANNELS = 64
    GROWTH_CHANNELS = 32

    # Training configuration
    PATCH_SIZE = 96
    BATCH_SIZE = 4
    NUM_WORKERS = 4

    # Memory optimization settings
    USE_MIXED_PRECISION = True
    GRADIENT_ACCUMULATION_STEPS = 2  # Simulate larger batch by accumulating gradients

    # Pretraining phase
    PRETRAIN_EPOCHS = 15
    PRETRAIN_LR = 1e-4

    # GAN training phase - Balanced for quality vs memory
    GAN_EPOCHS = 50
    GAN_G_LR = 1e-4
    GAN_D_LR = 4e-4

    # Learning rate scheduling
    LR_DECAY_STEP = 20
    LR_DECAY_GAMMA = 0.5

    # Loss weights - Balanced configuration
    PIXEL_WEIGHT = 1.0
    PERCEPTUAL_WEIGHT_MAX = 0.1
    ADVERSARIAL_WEIGHT_MAX = 0.005
    WEIGHT_RAMP_EPOCHS = 15

    # Validation and saving
    VAL_FREQUENCY = 5
    SAVE_EXAMPLES_FREQUENCY = 10
    MAX_SAVE_EXAMPLES = 4  # Reduced to save memory

    # Data augmentation settings
    AUGMENTATION_PROBABILITY = 0.7

    # Memory optimization settings
    GRADIENT_CLIP_VALUE = 0.5
    ACCUMULATION_STEPS = 2  # Gradient accumulation

    # EMA settings
    USE_EMA = False
    EMA_DECAY = 0.999

    # Discriminator training settings - Memory optimized
    D_TRAIN_RATIO = 1
    D_REG_EVERY = 32
    D_REG_WEIGHT = 5.0

    # Perceptual loss settings -
    PERCEPTUAL_LAYERS = [2, 7, 16]
    PERCEPTUAL_WEIGHTS = [1.0, 1.0, 1.0]

    # Memory optimization flags
    PIN_MEMORY = False
    BENCHMARK_CUDNN = True

    # Progressive growing settings
    PROGRESSIVE_GROWING = False
    INITIAL_PATCH_SIZE = 64

    # Quality control settings
    SAVE_BEST_N_MODELS = 2
    EARLY_STOPPING_PATIENCE = 10
    MIN_PSNR_THRESHOLD = 26.0

    # Inference settings
    INFERENCE_TILE_SIZE = 256
    INFERENCE_TILE_OVERLAP = 32
    INFERENCE_BATCH_SIZE = 1



    # Memory optimization methods
    @classmethod
    def setup_memory_optimization(cls):
        """Setup memory optimization for RTX 3080"""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()

            # Set memory fraction (use 90% of available memory)
            torch.cuda.set_per_process_memory_fraction(0.9)

            # Enable memory optimization
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            print(" Memory optimization enabled for RTX 3080")
            print(f"  Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  Memory fraction: 90%")
        else:
            print(" CUDA not available")

    @classmethod
    def print_config_summary(cls):
        """Print memory-optimized configuration summary"""
        print("=" * 60)
        print("ESRGAN CONFIG - MEMORY OPTIMIZED FOR RTX 3080")
        print("=" * 60)
        print(f"Model Architecture:")
        print(f"  - RRDB Blocks: {cls.NUM_RRDB_BLOCKS} (Memory Optimized)")
        print(f"  - Patch Size: {cls.PATCH_SIZE}x{cls.PATCH_SIZE}")
        print(f"  - Batch Size: {cls.BATCH_SIZE}")
        print(f"  - Scale Factor: {cls.SCALE_FACTOR}x")
        print(f"  - Mixed Precision: {cls.USE_MIXED_PRECISION}")
        print(f"")
        print(f"Memory Optimizations:")
        print(f"  - Gradient Accumulation: {cls.ACCUMULATION_STEPS} steps")
        print(f"  - Reduced Perceptual Layers: {len(cls.PERCEPTUAL_LAYERS)}")
        print(f"  - EMA Disabled: {not cls.USE_EMA}")
        print(f"  - Pin Memory: {cls.PIN_MEMORY}")
        print(f"")
        print(f"Training Settings:")
        print(f"  - Pretraining: {cls.PRETRAIN_EPOCHS} epochs")
        print(f"  - GAN Training: {cls.GAN_EPOCHS} epochs")
        print(f"")
        print(f"Expected Results (Memory Optimized):")
        print(f"  - DIV2K PSNR: 30-32 dB (good quality)")
        print(f"  - Set5 PSNR: 32-34 dB")
        print(f"  - Training Time: ~6-8 hours")
        print(f"  - Memory Usage: ~8-9 GB VRAM")
        print("=" * 60)

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_ROOT, cls.RESULTS_ROOT, cls.CHECKPOINTS_ROOT,
                          cls.BENCHMARKS_ROOT]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def apply_memory_optimizations(cls):
        """Apply all memory optimizations"""
        cls.setup_memory_optimization()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = cls.BENCHMARK_CUDNN
            torch.backends.cudnn.deterministic = False

            # Clear any existing allocations
            torch.cuda.empty_cache()

            print(" All memory optimizations applied")
        else:
            print(" CUDA not available, skipping GPU optimizations")