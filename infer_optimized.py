import os
import zipfile
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from model import NAFNet

def main():
    parser = argparse.ArgumentParser(description="NAFNet High-Throughput Inference and Submission Generator")
    parser.add_argument("--test_dir", type=str, default=r"c:\Users\aksha\Downloads\Test_NoisyLR\NoisyLR", help="Path to noisy test images directory")
    parser.add_argument("--checkpoint", type=str, default="pretrain_nafnet.pth", help="Path to model checkpoint")
    parser.add_argument("--out_dir", type=str, default=r"c:\Users\aksha\Downloads\Test_NoisyLR\predictions", help="Directory to save predicted .npy files")
    parser.add_argument("--zip_name", type=str, default=r"c:\Users\aksha\Downloads\Test_NoisyLR\submission.zip", help="Path to save submission.zip")
    parser.add_argument("--width", type=str, default="16", help="NAFNet base channel width")
    parser.add_argument("--compile", action="store_true", help="Compile model using torch.compile for maximum throughput")
    parser.add_argument("--channels_last", action="store_true", help="Use channels-last memory format for faster execution")
    parser.add_argument("--bf16", action="store_true", help="Use BFloat16 mixed precision for inference")
    
    args, unknown = parser.parse_known_args()
    
    width = int(args.width)
    checkpoint_path = args.checkpoint
    out_dir = args.out_dir
    zip_path = args.zip_name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running high-throughput inference on device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found. Please train the model first!")
        return
        
    # Model: Competitive depth configuration (enc=[2,2,4], middle=8, dec=[2,2,2])
    model = NAFNet(img_channel=1, width=width, middle_blk_num=8, enc_blk_nums=[2, 2, 4], dec_blk_nums=[2, 2, 2], upscale=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Apply Stage 5 Throughput Optimizations
    if args.channels_last:
        print("Applying Channels-Last memory format optimization...")
        model = model.to(memory_format=torch.channels_last)
        
    if args.compile:
        print("Compiling model using torch.compile (max-autotune)...")
        try:
            model = torch.compile(model, mode="max-autotune")
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}). Falling back to uncompiled model.")
            
    os.makedirs(out_dir, exist_ok=True)
    test_files = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.npy')])
    print(f"Found {len(test_files)} test files in {args.test_dir}.")
    
    # Warmup pass (important for compiled models to trigger JIT compilation)
    if args.compile:
        print("Running warmup pass...")
        warmup_input = torch.randn(1, 1, 128, 128).to(device)
        if args.channels_last:
            warmup_input = warmup_input.to(memory_format=torch.channels_last)
        with torch.no_grad():
            if args.bf16 and device.type == 'cuda':
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    _ = model(warmup_input)
            else:
                _ = model(warmup_input)
        print("Warmup complete!")

    print("Running inference...")
    t_start = time.time()
    
    with torch.no_grad():
        for f in test_files:
            in_path = os.path.join(args.test_dir, f)
            noisy_img = np.load(in_path) # (128, 128)
            
            x = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).float().to(device)
            
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
                
            # Autocast BF16
            if args.bf16:
                # BF16 is supported natively on CPU (PyTorch 1.10+) and CUDA (Ampere+)
                # Use general torch.amp.autocast for compatibility
                dtype = torch.bfloat16 if (device.type == 'cuda' or torch.cuda.is_available()) else torch.float32
                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    pred = model(x)
            else:
                pred = model(x)
                
            pred = torch.clamp(pred, 0.0, 1.0)
            
            # Convert back to float32 and squeeze
            out_img = pred.squeeze(0).squeeze(0).float().cpu().numpy()
            
            # Save predictions
            out_path = os.path.join(out_dir, f)
            np.save(out_path, out_img)
            
    t_total = time.time() - t_start
    fps = len(test_files) / t_total
    print(f"Inference complete! Processed {len(test_files)} images in {t_total:.2f}s ({fps:.2f} FPS).")
    
    # Verify shape of first prediction
    if len(test_files) > 0:
        first_pred = os.path.join(out_dir, test_files[0])
        pred_data = np.load(first_pred)
        print(f"Verified prediction shape: {pred_data.shape}, dtype: {pred_data.dtype}")
        
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
