import os
import argparse
import time
import numpy as np
import torch
import contextlib
from model import NAFNet

def main():
    parser = argparse.ArgumentParser(description="KLA Wafer Restoration Evaluation Script")
    
    # Support positional arguments (made optional via nargs='?' so optional aliases can be used instead)
    parser.add_argument("input_dir_pos", nargs="?", default=None, help="Input directory containing degraded .npy files")
    parser.add_argument("output_dir_pos", nargs="?", default=None, help="Output directory to save restored files")
    
    # Optional flags (aliases)
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory containing degraded .npy files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save restored files")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model weights (.pth)")
    parser.add_argument("--width", type=int, default=32, help="NAFNet base channel width")
    parser.add_argument("--use_tta", action="store_true", help="Enable Test-Time Augmentation (improves quality)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Resolve input and output directories (checking optional flags first, then positionals)
    input_dir = args.input_dir if args.input_dir is not None else args.input_dir_pos
    output_dir = args.output_dir if args.output_dir is not None else args.output_dir_pos
    
    if not input_dir or not output_dir:
        print("Error: Please provide both the input directory path and the output directory path.")
        print("Usage: python run.py <input_dir> <output_dir>")
        print("Or:    python run.py --input_dir <input_dir> --output_dir <output_dir>")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # Resolve checkpoint path relative to the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        # If relative, resolve against script_dir
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.normpath(os.path.join(script_dir, checkpoint_path))
    else:
        checkpoint_path = os.path.join(script_dir, "models", "finetuned_nafnet.pth")
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        return
        
    # Instantiate model: NAFNet width=32, 4,953,381 parameters (enc=[2,2,4], middle=8, dec=[2,2,2], upscale=2)
    model = NAFNet(
        img_channel=1, 
        width=args.width, 
        middle_blk_num=8, 
        enc_blk_nums=[2, 2, 4], 
        dec_blk_nums=[2, 2, 2], 
        upscale=2
    )
    
    # Load state dict (adding weights_only=True to avoid warnings)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Check if checkpoint is a dictionary containing 'model_state_dict' or is the state dict itself
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all test files
    test_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    print(f"Found {len(test_files)} test files to process.")
    
    if len(test_files) == 0:
        print("Warning: No .npy files found in the input directory.")
        return
        
    print(f"Starting inference (Batch Size: {args.batch_size}, TTA Enabled: {args.use_tta})...")
    
    total_inference_time = 0.0
    t_e2e_start = time.time()
    
    # Setup device autocast context for bf16 on CUDA
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    
    # Use inference mode
    with torch.inference_mode():
        for i in range(0, len(test_files), args.batch_size):
            batch_files = test_files[i:i+args.batch_size]
            
            # Load batch of images from disk
            batch_imgs = []
            for f in batch_files:
                in_path = os.path.join(input_dir, f)
                noisy_img = np.load(in_path) # Shape: (128, 128)
                batch_imgs.append(noisy_img)
                
            # Stack into float32 tensor of shape [B, 1, 128, 128] without clipping input values
            x = torch.from_numpy(np.stack(batch_imgs)).unsqueeze(1).float().to(device)
            
            # Sync CUDA before starting inference timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_inf_start = time.time()
            
            with autocast_context:
                if args.use_tta:
                    # 8-way Test-Time Augmentation on batch
                    outputs = []
                    for j in range(8):
                        x_aug = x.clone()
                        if j >= 4:
                            x_aug = torch.flip(x_aug, dims=[3])
                        rot_k = j % 4
                        if rot_k > 0:
                            x_aug = torch.rot90(x_aug, k=rot_k, dims=[2, 3])
                            
                        pred = model(x_aug)
                        
                        if rot_k > 0:
                            pred = torch.rot90(pred, k=-rot_k, dims=[2, 3])
                        if j >= 4:
                            pred = torch.flip(pred, dims=[3])
                        outputs.append(pred)
                    pred = torch.mean(torch.stack(outputs), dim=0)
                else:
                    pred = model(x)
            
            # Sync CUDA before stopping inference timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_inference_time += (time.time() - t_inf_start)
            
            # Clamp output to [0, 1]
            pred = torch.clamp(pred, 0.0, 1.0)
            
            # Cast back to float32 before converting to numpy
            pred_np = pred.float().cpu().numpy() # Shape: [B, 1, 256, 256]
            
            # Save batch outputs to disk individually
            for k, f in enumerate(batch_files):
                out_img = pred_np[k, 0] # Shape: (256, 256)
                
                # Verify sanity: Check for NaNs or Infs
                if np.isnan(out_img).any() or np.isinf(out_img).any():
                    print(f"Warning: Output for {f} contains NaN or Inf values. Replacing with zeros.")
                    out_img = np.nan_to_num(out_img, nan=0.0, posinf=1.0, neginf=0.0)
                
                out_path = os.path.join(output_dir, f)
                np.save(out_path, out_img)
                
    t_e2e_total = time.time() - t_e2e_start
    
    # Calculate FPS metrics
    num_images = len(test_files)
    pure_inf_fps = num_images / total_inference_time if total_inference_time > 0 else 0
    e2e_fps = num_images / t_e2e_total if t_e2e_total > 0 else 0
    
    print("\n" + "="*40)
    print("INFERENCE RUN COMPLETED")
    print("="*40)
    print(f"Total processed:         {num_images} images")
    print(f"Pure Inference time:     {total_inference_time:.4f} seconds")
    print(f"End-to-End time:         {t_e2e_total:.4f} seconds")
    print(f"Pure Inference (FPS):    {pure_inf_fps:.2f} images/second")
    print(f"End-to-End (FPS):        {e2e_fps:.2f} images/second")
    print(f"Restored outputs saved:  {output_dir}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
