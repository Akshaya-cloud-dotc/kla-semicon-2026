import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import NAFNet
from dataset import get_dataloaders

import numpy as np

def compute_psnr(mse):
    if mse == 0:
        return float('inf')
    return -10.0 * math.log10(mse)

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        device = img1.device
        if self.window.device != device:
            self.window = self.window.to(device)
            
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(1).mean(1).mean(1)

class CombinedLoss(nn.Module):
    def __init__(self, loss_type='charbonnier_ssim', ssim_weight=0.1, fft_weight=0.05):
        super(CombinedLoss, self).__init__()
        self.loss_type = loss_type
        self.ssim_weight = ssim_weight
        self.fft_weight = fft_weight
        
        self.l1 = nn.L1Loss()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()

    def forward(self, x, y):
        if self.loss_type == 'l1':
            return self.l1(x, y)
        elif self.loss_type == 'charbonnier':
            return self.charbonnier(x, y)
        elif self.loss_type == 'charbonnier_ssim':
            return self.charbonnier(x, y) + self.ssim_weight * self.ssim(x, y)
        elif self.loss_type == 'charbonnier_ssim_fft':
            base_loss = self.charbonnier(x, y) + self.ssim_weight * self.ssim(x, y)
            x_fft = torch.fft.rfft2(x, dim=(-2, -1))
            y_fft = torch.fft.rfft2(y, dim=(-2, -1))
            x_mag = torch.abs(x_fft)
            y_mag = torch.abs(y_fft)
            fft_loss = self.l1(x_mag, y_mag)
            return base_loss + self.fft_weight * fft_loss
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Deterministic random seed set to: {seed}")

def main():
    parser = argparse.ArgumentParser(description="Train NAFNet for NoisyLR Restoration")
    parser.add_argument("--noisy_dir", type=str, default=r"c:\Users\aksha\Downloads\Test_NoisyLR\train_data\train\NoisyLR", help="Path to noisy images directory")
    parser.add_argument("--gt_dir", type=str, default=r"c:\Users\aksha\Downloads\Test_NoisyLR\train_data\train\GT", help="Path to ground truth images directory")
    parser.add_argument("--batch_size", type=str, default="16", help="Batch size for training")
    parser.add_argument("--epochs", type=str, default="30", help="Number of training epochs")
    parser.add_argument("--lr", type=str, default="1e-3", help="Initial learning rate")
    parser.add_argument("--width", type=str, default="16", help="NAFNet base channel width")
    parser.add_argument("--checkpoint", type=str, default="best_nafnet.pth", help="Path to save best checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--use_synthetic", action="store_true", help="Use on-the-fly synthetic data generation for pre-training")
    parser.add_argument("--load_weights", type=str, default=None, help="Path to pre-trained weights to load model from (resets optimizer and epochs)")
    parser.add_argument("--loss_type", type=str, choices=["l1", "charbonnier", "charbonnier_ssim", "charbonnier_ssim_fft"], default="charbonnier_ssim", help="Loss configuration to train with")
    parser.add_argument("--ssim_weight", type=float, default=0.1, help="Weight for the SSIM loss term")
    parser.add_argument("--fft_weight", type=float, default=0.05, help="Weight for the FFT loss term")
    
    # We parse manually to avoid conflicts in notebook-like/interactive environments
    args, unknown = parser.parse_known_args()
    
    # Set seed for training reproducibility
    set_seed(42)
    
    # Convert string arguments to correct types
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)
    width = int(args.width)
    checkpoint_path = args.checkpoint
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    print(f"Active Loss Configuration: {args.loss_type} (ssim_weight={args.ssim_weight}, fft_weight={args.fft_weight})")
    
    # Dataloaders
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(args.noisy_dir, args.gt_dir, batch_size=batch_size, val_split=0.1, use_synthetic=args.use_synthetic)
    print(f"Dataset loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model: Competitive depth configuration (enc=[2,2,4], middle=8, dec=[2,2,2])
    model = NAFNet(img_channel=1, width=width, middle_blk_num=8, enc_blk_nums=[2, 2, 4], dec_blk_nums=[2, 2, 2], upscale=2)
    model = model.to(device)
    
    # Print model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NAFNet model initialized with {num_params:,} trainable parameters.")
    
    criterion = CombinedLoss(loss_type=args.loss_type, ssim_weight=args.ssim_weight, fft_weight=args.fft_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    start_epoch = 1
    best_psnr = -float('inf')
    
    # Load pre-trained weights (without resuming optimizer/epochs)
    if args.load_weights and os.path.exists(args.load_weights):
        print(f"Loading pre-trained weights from: {args.load_weights}")
        checkpoint = torch.load(args.load_weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # Resume training
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', -float('inf'))
        print(f"Resumed from epoch {start_epoch} with best validation PSNR: {best_psnr:.4f} dB")
        
    print("Starting training pipeline...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, (noisy, gt, _) in enumerate(train_loader):
            noisy = noisy.to(device)
            gt = gt.to(device)
            
            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - epoch_start_time
                print(f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.6f} | Time: {elapsed:.1f}s")
                
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for noisy, gt, _ in val_loader:
                noisy = noisy.to(device)
                gt = gt.to(device)
                pred = model(noisy)
                
                # Clip values to valid image range [0.0, 1.0] before computing metric
                pred = torch.clamp(pred, 0.0, 1.0)
                mse = F.mse_loss(pred, gt)
                val_mse += mse.item()
                
        avg_val_mse = val_mse / len(val_loader)
        val_psnr = compute_psnr(avg_val_mse)
        epoch_time = time.time() - epoch_start_time
        
        print(f"--- Epoch {epoch} Summary: Train Loss: {avg_train_loss:.6f} | Val PSNR: {val_psnr:.4f} dB | Epoch Time: {epoch_time:.1f}s ---")
        
        # Save best checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'train_loss': avg_train_loss
            }, checkpoint_path)
            print(f"====> New best model saved to {checkpoint_path} with PSNR: {best_psnr:.4f} dB!")

if __name__ == "__main__":
    main()
