import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class NoisyLRDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir=None, augment=False):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.augment = augment
        
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.npy')])
        if gt_dir is not None:
            self.gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])
            assert len(self.noisy_files) == len(self.gt_files), "Noisy and GT file counts must match!"
        else:
            self.gt_files = None

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        noisy_img = np.load(noisy_path) # shape: (128, 128)
        
        if self.gt_files is not None:
            gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
            gt_img = np.load(gt_path) # shape: (256, 256)
        else:
            gt_img = None
            
        # Data Augmentation (flips and rotations)
        if self.augment and gt_img is not None:
            if np.random.rand() > 0.5:
                noisy_img = np.fliplr(noisy_img).copy()
                gt_img = np.fliplr(gt_img).copy()
            if np.random.rand() > 0.5:
                noisy_img = np.flipud(noisy_img).copy()
                gt_img = np.flipud(gt_img).copy()
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                noisy_img = np.rot90(noisy_img, rot_k).copy()
                gt_img = np.rot90(gt_img, rot_k).copy()

        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).float()
        
        if gt_img is not None:
            gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).float()
            return noisy_tensor, gt_tensor, self.noisy_files[idx]
        else:
            return noisy_tensor, self.noisy_files[idx]

class OnTheFlyDegradationDataset(Dataset):
    """
    Generates synthetic training pairs on-the-fly by applying estimated blur kernel
    and Poisson noise profile to ground truth images.
    """
    def __init__(self, gt_dir, kernel_path="estimated_kernel.npy", noise_profile_path="noise_profile.npy", augment=True):
        self.gt_dir = gt_dir
        self.augment = augment
        self.gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])
        
        # Load blur kernel
        if os.path.exists(kernel_path):
            kernel_np = np.load(kernel_path)
            print(f"Loaded estimated kernel of shape {kernel_np.shape} from {kernel_path}")
        else:
            # Fallback to standard 15x15 Gaussian blur kernel if not found
            print("Warning: estimated_kernel.npy not found! Creating default Gaussian kernel...")
            x = np.linspace(-3, 3, 15)
            gauss = np.exp(-x**2 / 2.0)
            kernel_2d = np.outer(gauss, gauss)
            kernel_np = kernel_2d / np.sum(kernel_2d)
            
        self.kernel_tensor = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0).float()
        self.kernel_size = kernel_np.shape[0]
        self.padding = self.kernel_size // 2
        
        # Load noise parameters (Poisson noise slope)
        if os.path.exists(noise_profile_path):
            noise_params = np.load(noise_profile_path)
            self.noise_slope = noise_params[0] # variance = slope * intensity
            print(f"Loaded noise profile from {noise_profile_path}: variance = {self.noise_slope:.6f} * intensity")
        else:
            self.noise_slope = 0.001614 # fallback value from estimation
            print(f"Warning: noise_profile.npy not found! Defaulting to noise slope: {self.noise_slope:.6f}")

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        gt_img = np.load(gt_path) # (256, 256)
        
        # Augment the ground truth image first
        if self.augment:
            if np.random.rand() > 0.5:
                gt_img = np.fliplr(gt_img).copy()
            if np.random.rand() > 0.5:
                gt_img = np.flipud(gt_img).copy()
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                gt_img = np.rot90(gt_img, rot_k).copy()
                
        # Convert ground truth to PyTorch tensor (1, H, W)
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).float()
        
        # Apply Blur Kernel
        with torch.no_grad():
            # Define degradation functions
            def apply_speckle(img):
                # Widen speckle noise range to [0.0, 0.1] for better OOD coverage
                std = np.random.uniform(0.0, 0.1)
                noise = torch.randn_like(img) * std
                return img * (1.0 + noise)

            def apply_downsample_and_blur(img):
                if img.shape[-1] == 256:
                    img_batch = img.unsqueeze(0)
                    # Randomize interpolation downsampling modes
                    mode = np.random.choice(['kernel', 'bilinear', 'bicubic', 'area'])
                    if mode == 'kernel':
                        # Anti-aliasing conv with estimated kernel + slice downsampling
                        blurred = F.conv2d(img_batch, self.kernel_tensor, padding=self.padding)
                        return blurred[:, :, ::2, ::2].squeeze(0)
                    else:
                        # Standard PyTorch interpolation methods
                        align_corners = False if mode in ['bilinear', 'bicubic'] else None
                        downsampled = F.interpolate(img_batch, size=(128, 128), mode=mode, align_corners=align_corners)
                        return downsampled.squeeze(0)
                else:
                    # If already downsampled, apply estimated kernel without downsampling
                    img_batch = img.unsqueeze(0)
                    return F.conv2d(img_batch, self.kernel_tensor, padding=self.padding).squeeze(0)

            def apply_gaussian(img):
                # Widen Gaussian noise range to [0.0, 0.08] for better OOD coverage
                std = np.random.uniform(0.0, 0.08)
                noise = torch.randn_like(img) * std
                return img + noise

            # Define operators list
            operators = [apply_speckle, apply_downsample_and_blur, apply_gaussian]
            
            # Randomly shuffle the order of operations for out-of-distribution robustness
            if np.random.rand() > 0.5:
                np.random.shuffle(operators)
                
            x = gt_tensor
            for op in operators:
                x = op(x)
                
            # If for some reason downsampling wasn't run first, force downsample to 128x128
            if x.shape[-1] != 128:
                x = F.interpolate(x.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
                
            # IMPORTANT: Do not clamp noisy_lr to [0.0, 1.0] as real noisy inputs naturally exceed this range.
            noisy_lr = x
            
        return noisy_lr, gt_tensor, self.gt_files[idx]

def get_dataloaders(noisy_dir, gt_dir, batch_size=16, val_split=0.1, num_workers=0, use_synthetic=False):
    if use_synthetic:
        # Pretrain mode: uses OnTheFlyDegradationDataset on the full GT training images
        print("Using On-The-Fly degradation simulation for pre-training...")
        dataset = OnTheFlyDegradationDataset(gt_dir, augment=True)
        val_dataset = OnTheFlyDegradationDataset(gt_dir, augment=False)
    else:
        # Fine-tune/Standard mode: uses the given real noisy-GT pairs
        print("Using real noisy/GT pairs for fine-tuning...")
        dataset = NoisyLRDataset(noisy_dir, gt_dir, augment=True)
        val_dataset = NoisyLRDataset(noisy_dir, gt_dir, augment=False)
    
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_size = int(np.floor(val_split * num_samples))
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    noisy_path = r"c:\Users\aksha\Downloads\Test_NoisyLR\train_data\train\NoisyLR"
    gt_path = r"c:\Users\aksha\Downloads\Test_NoisyLR\train_data\train\GT"
    
    if os.path.exists(noisy_path) and os.path.exists(gt_path):
        # Test real dataset
        print("\n--- Testing Real Dataloader ---")
        train_loader, val_loader = get_dataloaders(noisy_path, gt_path, batch_size=4, val_split=0.1, use_synthetic=False)
        for x, y, name in train_loader:
            print("Real Batch LR shape:", x.shape)
            print("Real Batch GT shape:", y.shape)
            break
            
        # Test synthetic dataset
        print("\n--- Testing On-the-Fly Synthetic Dataloader ---")
        train_loader_synth, val_loader_synth = get_dataloaders(noisy_path, gt_path, batch_size=4, val_split=0.1, use_synthetic=True)
        for x, y, name in train_loader_synth:
            print("Synthetic Batch LR shape:", x.shape)
            print("Synthetic Batch GT shape:", y.shape)
            break
        print("\nAll dataloaders tested successfully!")
    else:
        print("Directories not found, skipping self-check.")
