import os
import zipfile
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from model import NAFNet

def tta_forward(model, x):
    # x shape: (1, 1, H, W)
    outputs = []
    
    # The 8 combinations of flips and rotations (TTA)
    # 0: Original
    # 1: Rot90
    # 2: Rot180
    # 3: Rot270
    # 4: Horizontal Flip
    # 5: Horizontal Flip + Rot90
    # 6: Horizontal Flip + Rot180
    # 7: Horizontal Flip + Rot270
    for i in range(8):
        x_aug = x.clone()
        
        # Apply augmentation
        if i >= 4:
            x_aug = torch.flip(x_aug, dims=[3]) # flip horizontally (W dimension)
        rot_k = i % 4
        if rot_k > 0:
            x_aug = torch.rot90(x_aug, k=rot_k, dims=[2, 3]) # rotate in H-W plane
            
        # Model prediction
        pred = model(x_aug)
        
        # Reverse the augmentation on the prediction
        if rot_k > 0:
            pred = torch.rot90(pred, k=-rot_k, dims=[2, 3]) # reverse rotation
        if i >= 4:
            pred = torch.flip(pred, dims=[3]) # reverse flip
            
        outputs.append(pred)
        
    # Average all 8 predictions
    mean_out = torch.mean(torch.stack(outputs), dim=0)
    return mean_out

def main():
    parser = argparse.ArgumentParser(description="NAFNet TTA Inference and Submission Generator")
    parser.add_argument("--test_dir", type=str, default=r"c:\Users\aksha\Downloads\Test_NoisyLR\NoisyLR", help="Path to noisy test images directory")
    parser.add_argument("--checkpoint", type=str, default="pretrain_nafnet.pth", help="Path to model checkpoint")
    parser.add_argument("--out_dir", type=str, default=r"c:\Users\aksha\Downloads\Test_NoisyLR\predictions", help="Directory to save predicted .npy files")
    parser.add_argument("--zip_name", type=str, default=r"c:\Users\aksha\Downloads\Test_NoisyLR\submission_tta.zip", help="Path to save submission_tta.zip")
    parser.add_argument("--width", type=str, default="32", help="NAFNet base channel width")
    
    args, unknown = parser.parse_known_args()
    
    width = int(args.width)
    checkpoint_path = args.checkpoint
    out_dir = args.out_dir
    zip_path = args.zip_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running TTA inference on device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found!")
        return
        
    print("Loading model and checkpoint...")
    # Model: Competitive depth configuration (enc=[2,2,4], middle=8, dec=[2,2,2])
    model = NAFNet(img_channel=1, width=width, middle_blk_num=8, enc_blk_nums=[2, 2, 4], dec_blk_nums=[2, 2, 2], upscale=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    os.makedirs(out_dir, exist_ok=True)
    test_files = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.npy')])
    print(f"Found {len(test_files)} test files.")
    
    print("Running Test-Time Augmentation (TTA) inference...")
    t_start = time.time()
    
    with torch.no_grad():
        for f in test_files:
            in_path = os.path.join(args.test_dir, f)
            noisy_img = np.load(in_path) # (128, 128)
            
            x = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Forward pass with Test-Time Augmentation (TTA)
            pred = tta_forward(model, x)
            pred = torch.clamp(pred, 0.0, 1.0)
            
            out_img = pred.squeeze(0).squeeze(0).cpu().numpy()
            
            out_path = os.path.join(out_dir, f)
            np.save(out_path, out_img)
            
    t_total = time.time() - t_start
    fps = len(test_files) / t_total
    print(f"TTA Inference complete! Processed {len(test_files)} images in {t_total:.2f}s ({fps:.2f} FPS).")
    
    # Zip results for submission
    print(f"Zipping predictions to {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in test_files:
                out_path = os.path.join(out_dir, f)
                z.write(out_path, arcname=f)
        print(f"Submission zip generated successfully at: {zip_path}")
    except Exception as e:
        print("Error during zipping:", e)

if __name__ == "__main__":
    main()
