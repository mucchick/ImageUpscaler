import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import save_comparison_images
from config import Config


class Evaluator:
    def __init__(self, device):
        self.device = device

    def evaluate_sr(self, model, dataloader, verbose=True):
        """Basic evaluation returning average PSNR"""
        model.eval()
        psnr_total = 0.0
        ssim_total = 0.0
        count = 0

        with torch.no_grad():
            iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
            for batch in iterator:
                if len(batch) == 3:
                    lr, hr, _ = batch
                else:
                    lr, hr = batch

                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = model(lr).clamp(0, 1)

                for i in range(sr.size(0)):
                    sr_img = sr[i].cpu().permute(1, 2, 0).numpy()
                    hr_img = hr[i].cpu().permute(1, 2, 0).numpy()

                    psnr_total += calculate_psnr(hr_img, sr_img, data_range=1.0)
                    ssim_total += calculate_ssim(hr_img, sr_img, data_range=1.0, win_size=3)
                    count += 1

        avg_psnr = psnr_total / count
        avg_ssim = ssim_total / count

        if verbose:
            print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

        return avg_psnr

    def comprehensive_evaluation(self, model, test_loader, save_results=True, results_dir=None):
        """Comprehensive evaluation with detailed metrics and visualizations"""
        if results_dir is None:
            results_dir = os.path.join(Config.RESULTS_ROOT, "comprehensive_evaluation")

        model.eval()

        # Metrics storage
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        inference_times = []

        # Try to load LPIPS model for perceptual quality
        lpips_model = None
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').to(self.device)
            print("LPIPS model loaded for perceptual evaluation")
        except ImportError:
            print("LPIPS not installed. Skipping perceptual metric.")

        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc="Comprehensive Testing")):
                if len(batch) == 3:
                    lr, hr, names = batch
                else:
                    lr, hr = batch
                    names = [f"image_{idx}"]

                lr = lr.to(self.device)
                hr = hr.to(self.device)

                # Measure inference time
                if torch.cuda.is_available():
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)

                    start_time.record()
                    sr = model(lr).clamp(0, 1)
                    end_time.record()

                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                    inference_times.append(inference_time)
                else:
                    sr = model(lr).clamp(0, 1)

                # Calculate metrics
                for i in range(sr.size(0)):
                    # Convert to numpy for PSNR/SSIM
                    sr_np = sr[i].cpu().permute(1, 2, 0).numpy()
                    hr_np = hr[i].cpu().permute(1, 2, 0).numpy()

                    # PSNR
                    psnr = calculate_psnr(hr_np, sr_np, data_range=1.0)
                    psnr_scores.append(psnr)

                    # SSIM
                    ssim = calculate_ssim(hr_np, sr_np, data_range=1.0)
                    ssim_scores.append(ssim)

                    # LPIPS (if available)
                    if lpips_model is not None:
                        lpips_score = lpips_model(sr[i:i + 1], hr[i:i + 1]).item()
                        lpips_scores.append(lpips_score)

                    # Save comparison images for first 10 results
                    if save_results and idx < 10:
                        name = names[i] if isinstance(names, list) else f"img_{idx:03d}"
                        comparison_path = os.path.join(results_dir, "images", f"{name}_comparison.png")
                        save_comparison_images(lr[i], sr[i], hr[i], comparison_path, psnr)

        # Calculate statistics
        results = {
            "PSNR": {
                "mean": float(np.mean(psnr_scores)),
                "std": float(np.std(psnr_scores)),
                "min": float(np.min(psnr_scores)),
                "max": float(np.max(psnr_scores))
            },
            "SSIM": {
                "mean": float(np.mean(ssim_scores)),
                "std": float(np.std(ssim_scores)),
                "min": float(np.min(ssim_scores)),
                "max": float(np.max(ssim_scores))
            }
        }

        if inference_times:
            results["Inference Time (ms)"] = {
                "mean": float(np.mean(inference_times)),
                "std": float(np.std(inference_times))
            }

        if lpips_scores:
            results["LPIPS"] = {
                "mean": float(np.mean(lpips_scores)),
                "std": float(np.std(lpips_scores))
            }

        # Print results
        print("\n" + "=" * 50)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 50)

        for metric, values in results.items():
            print(f"\n{metric}:")
            for stat, value in values.items():
                print(f"  {stat}: {value:.4f}")


        if save_results:
            with open(os.path.join(results_dir, "evaluation_results.json"), 'w') as f:
                json.dump(results, f, indent=4)

            # Create metrics distribution plots
            self._create_metrics_plots(psnr_scores, ssim_scores, results, results_dir)

        return results

    def _create_metrics_plots(self, psnr_scores, ssim_scores, results, results_dir):
        """Create histogram plots for metrics distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PSNR histogram
        axes[0].hist(psnr_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(results["PSNR"]["mean"], color='red', linestyle='--',
                        label=f'Mean: {results["PSNR"]["mean"]:.2f}')
        axes[0].set_xlabel('PSNR (dB)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('PSNR Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # SSIM histogram
        axes[1].hist(ssim_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(results["SSIM"]["mean"], color='red', linestyle='--',
                        label=f'Mean: {results["SSIM"]["mean"]:.4f}')
        axes[1].set_xlabel('SSIM')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('SSIM Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "metrics_distribution.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def evaluate_on_benchmark(self, model, benchmark_name, benchmark_dataset):
        """Evaluate model on standard SR benchmarks"""
        print(f"\nEvaluating on {benchmark_name}...")

        from torch.utils.data import DataLoader
        benchmark_loader = DataLoader(benchmark_dataset, batch_size=1, shuffle=False)

        print(f"Found {len(benchmark_dataset)} images in {benchmark_name}")

        # Run comprehensive evaluation
        results_dir = os.path.join(Config.RESULTS_ROOT, f"{benchmark_name}_results")
        results = self.comprehensive_evaluation(model, benchmark_loader,
                                                save_results=True, results_dir=results_dir)

        return results