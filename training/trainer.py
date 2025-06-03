import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.losses import CharbonnierLoss
from utils.visualization import save_sr_examples
from config import Config


class ESRGANTrainer:
    def __init__(self, generator, discriminator, vgg_extractor, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.vgg = vgg_extractor.to(device)
        self.device = device

        # Loss functions
        self.charbonnier_loss = CharbonnierLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Create directories
        os.makedirs(Config.CHECKPOINTS_ROOT, exist_ok=True)

        self.best_psnr = 0

    def pretrain_generator(self, train_loader, val_loader=None, num_epochs=10):
        """Pretrain generator with pixel loss only"""
        print("Starting generator pretraining...")

        self.generator.train()
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=Config.PRETRAIN_LR, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = self.generator(lr)
                loss = self.charbonnier_loss(sr, hr)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / len(train_loader)

            print(f"[Epoch {epoch + 1}] Avg Charbonnier Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")

            # Validate if validation loader provided
            if val_loader and (epoch + 1) % Config.VAL_FREQUENCY == 0:
                from .evaluator import Evaluator
                evaluator = Evaluator(self.device)
                val_psnr = evaluator.evaluate_sr(self.generator, val_loader, verbose=False)
                print(f"Validation PSNR: {val_psnr:.2f}dB")

        # Save pretrained model
        torch.save(self.generator.state_dict(), os.path.join(Config.CHECKPOINTS_ROOT, 'pretrained_generator.pth'))
        print("Generator pretraining completed!")

    def train_gan(self, train_loader, val_loader=None, num_epochs=50):
        """Train full ESRGAN with adversarial and perceptual losses"""
        print("Starting GAN training...")

        # Optimizers
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=Config.GAN_G_LR, betas=(0.9, 0.999))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=Config.GAN_D_LR, betas=(0.9, 0.999))

        # Learning rate schedulers
        g_scheduler = torch.optim.lr_scheduler.StepLR(g_opt, step_size=20, gamma=0.5)
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_opt, step_size=20, gamma=0.5)

        for epoch in range(1, num_epochs + 1):
            self.generator.train()
            self.discriminator.train()
            total_g_loss = 0.0
            total_d_loss = 0.0

            weights = self._get_loss_weights(epoch)

            for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                # Train Discriminator
                for _ in range(2):  # Train discriminator 2 times per generator update
                    with torch.no_grad():
                        fake_sr = self.generator(lr).detach()

                    real_out = self.discriminator(hr)
                    fake_out = self.discriminator(fake_sr)

                    # Relativistic average discriminator
                    real_label = torch.ones_like(real_out)
                    fake_label = torch.zeros_like(fake_out)

                    d_loss_real = self.bce_loss(real_out - fake_out.mean(), real_label)
                    d_loss_fake = self.bce_loss(fake_out - real_out.mean(), fake_label)
                    d_loss = (d_loss_real + d_loss_fake) / 2

                    d_opt.zero_grad()
                    d_loss.backward()
                    d_opt.step()

                    total_d_loss += d_loss.item()

                # Train Generator
                fake_sr = self.generator(lr)
                fake_out = self.discriminator(fake_sr)
                real_out = self.discriminator(hr)

                # Relativistic average GAN loss
                gan_g_loss = self.bce_loss(fake_out - real_out.mean(), real_label)

                # Multi-scale perceptual loss
                vgg_fake_features = self.vgg(fake_sr)
                vgg_real_features = self.vgg(hr)

                perceptual_loss = 0
                for fake_feat, real_feat in zip(vgg_fake_features, vgg_real_features):
                    perceptual_loss += self.charbonnier_loss(fake_feat, real_feat)
                perceptual_loss /= len(vgg_fake_features)

                # Pixel loss
                pixel_loss = self.charbonnier_loss(fake_sr, hr)

                # Total Generator loss
                g_loss = (weights['pixel'] * pixel_loss +
                          weights['perceptual'] * perceptual_loss +
                          weights['adversarial'] * gan_g_loss)

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

                total_g_loss += g_loss.item()

            # Update learning rates
            g_scheduler.step()
            d_scheduler.step()

            # Print progress
            avg_g_loss = total_g_loss / len(train_loader)
            avg_d_loss = total_d_loss / (len(train_loader) * 2)

            print(f"[Epoch {epoch}] G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")
            print(
                f"  Weights - Pixel: {weights['pixel']:.3f}, Perceptual: {weights['perceptual']:.3f}, Adversarial: {weights['adversarial']:.3f}")

            # Validation and saving
            if epoch % Config.VAL_FREQUENCY == 0:
                if val_loader:
                    from .evaluator import Evaluator
                    evaluator = Evaluator(self.device)
                    val_psnr = evaluator.evaluate_sr(self.generator, val_loader, verbose=False)

                    if val_psnr > self.best_psnr:
                        self.best_psnr = val_psnr
                        self.save_checkpoint(epoch, val_psnr, is_best=True)
                        print(f"  Saved best model with PSNR: {self.best_psnr:.2f}dB")
                    else:
                        print(f"  Validation PSNR: {val_psnr:.2f}dB")

                # Save regular checkpoint
                self.save_checkpoint(epoch, val_psnr if val_loader else 0)

            # Save example images
            if epoch % Config.SAVE_EXAMPLES_FREQUENCY == 0 and val_loader:
                example_dir = os.path.join(Config.RESULTS_ROOT, f"epoch_{epoch}_examples")
                save_sr_examples(self.generator, val_loader, example_dir,
                                 max_batches=Config.MAX_SAVE_EXAMPLES, device=self.device)

        print("GAN training completed!")

    def _get_loss_weights(self, epoch):
        """Progressive loss weights"""
        progress = min(epoch / Config.WEIGHT_RAMP_EPOCHS, 1.0)
        return {
            'pixel': Config.PIXEL_WEIGHT,
            'perceptual': Config.PERCEPTUAL_WEIGHT_MAX * progress,
            'adversarial': Config.ADVERSARIAL_WEIGHT_MAX * progress
        }

    def save_checkpoint(self, epoch, psnr, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'psnr': psnr,
            'best_psnr': self.best_psnr
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(Config.CHECKPOINTS_ROOT, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(Config.CHECKPOINTS_ROOT, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        if 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        if 'best_psnr' in checkpoint:
            self.best_psnr = checkpoint['best_psnr']

        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint.get('epoch', 0)