

import os
import argparse
import torch
from torch.utils.data import DataLoader

# Local imports
from config import Config
from models import ESRGANGenerator, Discriminator, VGGFeatureExtractor
from dataset import DIV2KTrainDataset, DIV2KValDataset, BenchmarkDataset
from training import ESRGANTrainer, Evaluator
from utils import download_div2k_dataset


def setup_datasets():
    """Download and setup all required datasets"""
    print("Setting up datasets...")

    # Create necessary directories
    Config.create_directories()

    # Download DIV2K dataset
    print("Checking DIV2K dataset...")
    paths = download_div2k_dataset()

    return paths


def create_data_loaders():
    """Create data loaders for training and validation"""
    print("Creating data loaders...")

    # Training dataset with augmentation
    train_dataset = DIV2KTrainDataset(
        hr_dir=Config.TRAIN_HR_DIR,
        lr_dir=Config.TRAIN_LR_DIR,
        patch_size=Config.PATCH_SIZE,
        scale=Config.SCALE_FACTOR,
        augment=True
    )

    # Validation dataset
    val_dataset = DIV2KValDataset(
        hr_dir=Config.VAL_HR_DIR,
        lr_dir=Config.VAL_LR_DIR
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


def create_models(device):
    """Create and initialize all models"""
    print("Initializing models...")

    generator = ESRGANGenerator(
        num_rrdb=Config.NUM_RRDB_BLOCKS,
        scale=Config.SCALE_FACTOR
    ).to(device)

    discriminator = Discriminator().to(device)

    vgg_extractor = VGGFeatureExtractor().to(device)

    # Print model info
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())

    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Using device: {device}")

    return generator, discriminator, vgg_extractor


def print_benchmark_summary(results):
    """Print a formatted table of benchmark results"""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<15} {'PSNR (dB)':<12} {'SSIM':<12}")
    print("-" * 40)

    for benchmark, metrics in results.items():
        psnr = metrics.get('PSNR', {}).get('mean', 0)
        ssim = metrics.get('SSIM', {}).get('mean', 0)
        print(f"{benchmark:<15} {psnr:<12.2f} {ssim:<12.4f}")

    print("=" * 60)


def evaluate_benchmarks(generator, device):
    """Evaluate model on standard benchmarks"""
    evaluator = Evaluator(device)
    all_benchmark_results = {}

    # List of benchmarks to evaluate
    benchmarks = ['Set5', 'Set14', 'BSD100', 'Urban100']

    for benchmark_name in benchmarks:
        benchmark_dir = os.path.join(Config.BENCHMARKS_ROOT, benchmark_name)

        # Check if benchmark exists
        if not os.path.exists(benchmark_dir):
            print(f"Warning: {benchmark_name} not found at {benchmark_dir}")
            print(f"Please download {benchmark_name} manually and place it in {benchmark_dir}")
            continue

        try:
            print(f"\nEvaluating on {benchmark_name}...")
            benchmark_dataset = BenchmarkDataset(benchmark_dir, scale=Config.SCALE_FACTOR)

            # Create dataloader
            from torch.utils.data import DataLoader
            benchmark_loader = DataLoader(benchmark_dataset, batch_size=1, shuffle=False)

            results = evaluator.comprehensive_evaluation(
                generator, benchmark_loader,
                save_results=True,
                results_dir=os.path.join(Config.RESULTS_ROOT, f"{benchmark_name}_results")
            )
            all_benchmark_results[benchmark_name] = results

        except Exception as e:
            print(f"Error evaluating {benchmark_name}: {e}")

    return all_benchmark_results


def train_model(args):
    """Main training function"""
    print("Starting ESRGAN training...")

    # Setup
    device = Config.DEVICE
    setup_datasets()
    train_loader, val_loader = create_data_loaders()
    generator, discriminator, vgg_extractor = create_models(device)

    # Initialize trainer
    trainer = ESRGANTrainer(generator, discriminator, vgg_extractor, device)

    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")

    # Phase 1: Generator pretraining
    if not args.skip_pretrain:
        print("\n" + "=" * 50)
        print("PHASE 1: GENERATOR PRETRAINING")
        print("=" * 50)
        trainer.pretrain_generator(
            train_loader,
            val_loader,
            num_epochs=Config.PRETRAIN_EPOCHS
        )

    # Phase 2: GAN training
    print("\n" + "=" * 50)
    print("PHASE 2: GAN TRAINING")
    print("=" * 50)
    trainer.train_gan(
        train_loader,
        val_loader,
        num_epochs=Config.GAN_EPOCHS
    )

    # Final evaluation on benchmarks
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON STANDARD BENCHMARKS")
    print("=" * 50)

    all_benchmark_results = evaluate_benchmarks(generator, device)

    # Print summary
    if all_benchmark_results:
        print_benchmark_summary(all_benchmark_results)

    # Save final model
    final_checkpoint = {
        'epoch': Config.GAN_EPOCHS,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'benchmark_results': all_benchmark_results,
        'config': {
            'num_rrdb': Config.NUM_RRDB_BLOCKS,
            'scale': Config.SCALE_FACTOR,
            'patch_size': Config.PATCH_SIZE
        }
    }

    final_path = os.path.join(Config.CHECKPOINTS_ROOT, 'final_esrgan_model.pth')
    torch.save(final_checkpoint, final_path)
    print(f"Final model saved to {final_path}")


def evaluate_model(args):
    """Evaluate a trained model"""
    print("Evaluating trained model...")

    device = Config.DEVICE

    # Load model
    if not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} not found!")
        return

    checkpoint = torch.load(args.model_path, map_location=device)

    # Create generator
    generator = ESRGANGenerator(
        num_rrdb=Config.NUM_RRDB_BLOCKS,
        scale=Config.SCALE_FACTOR
    ).to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    evaluator = Evaluator(device)

    # Evaluate on validation set
    if args.eval_div2k:
        setup_datasets()
        _, val_loader = create_data_loaders()
        print("Evaluating on DIV2K validation set...")
        evaluator.comprehensive_evaluation(
            generator, val_loader,
            save_results=True,
            results_dir=os.path.join(Config.RESULTS_ROOT, "evaluation_div2k_val")
        )

    # Evaluate on benchmarks
    if args.eval_benchmarks:
        all_benchmark_results = evaluate_benchmarks(generator, device)
        if all_benchmark_results:
            print_benchmark_summary(all_benchmark_results)


def compare_with_bicubic(args):
    """Compare model with bicubic baseline"""
    print("Comparing with bicubic interpolation...")

    device = Config.DEVICE

    # Bicubic baseline model
    class BicubicBaseline(torch.nn.Module):
        def __init__(self, scale=4):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return torch.nn.functional.interpolate(
                x, scale_factor=self.scale, mode='bicubic', align_corners=False
            )

    # Setup test data
    setup_datasets()
    _, val_loader = create_data_loaders()

    # Evaluate bicubic
    bicubic_model = BicubicBaseline(Config.SCALE_FACTOR).to(device)
    evaluator = Evaluator(device)

    print("Bicubic baseline results:")
    bicubic_results = evaluator.comprehensive_evaluation(
        bicubic_model, val_loader,
        save_results=True,
        results_dir=os.path.join(Config.RESULTS_ROOT, "bicubic_baseline")
    )

    # Compare with trained model if provided
    if args.model_path and os.path.exists(args.model_path):
        print("\nTrained model results:")
        checkpoint = torch.load(args.model_path, map_location=device)

        generator = ESRGANGenerator(
            num_rrdb=Config.NUM_RRDB_BLOCKS,
            scale=Config.SCALE_FACTOR
        ).to(device)

        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()

        model_results = evaluator.comprehensive_evaluation(
            generator, val_loader,
            save_results=True,
            results_dir=os.path.join(Config.RESULTS_ROOT, "model_vs_bicubic")
        )

        # Print comparison
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        print(f"Bicubic PSNR: {bicubic_results['PSNR']['mean']:.2f}dB")
        print(f"Model PSNR: {model_results['PSNR']['mean']:.2f}dB")
        print(f"Improvement: {model_results['PSNR']['mean'] - bicubic_results['PSNR']['mean']:.2f}dB")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='ESRGAN Desktop Training')
    parser.add_argument('--mode', choices=['train', 'eval', 'compare'],
                        default='train', help='Operation mode')
    parser.add_argument('--model-path', type=str,
                        help='Path to trained model for evaluation')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Skip generator pretraining phase')
    parser.add_argument('--eval-div2k', action='store_true',
                        help='Evaluate on DIV2K validation set')
    parser.add_argument('--eval-benchmarks', action='store_true',
                        help='Evaluate on standard benchmarks (Set5, Set14, etc)')

    args = parser.parse_args()

    print("ESRGAN Desktop Training")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("=" * 50)

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'eval':
        evaluate_model(args)
    elif args.mode == 'compare':
        compare_with_bicubic(args)


if __name__ == "__main__":
    main()