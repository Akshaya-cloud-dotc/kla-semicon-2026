import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_input = os.path.join(script_dir, "temp_smoke_input")
temp_output = os.path.join(script_dir, "temp_smoke_output")
os.makedirs(temp_input, exist_ok=True)
os.makedirs(temp_output, exist_ok=True)

# 1. Create a simulated degraded image (128x128 float32 array)
# We draw a grid pattern and add heavy noise
grid = np.zeros((128, 128), dtype=np.float32)
grid[::8, :] = 1.0
grid[:, ::8] = 1.0

# Add multiplicative speckle noise and additive Gaussian noise
np.random.seed(42)
speckle = np.random.normal(1.0, 0.15, (128, 128)).astype(np.float32)
gaussian = np.random.normal(0.0, 0.05, (128, 128)).astype(np.float32)
degraded = (grid * speckle) + gaussian

# Save as .npy
np.save(os.path.join(temp_input, "simulated_degraded.npy"), degraded)
print("[1/3] Generated simulated degraded test image in temp_smoke_input/.")

# 2. Run the evaluation script using the local model and weights
# First check the models directory (cross-machine compatible)
weights_path = os.path.join(script_dir, "models", "finetuned_nafnet.pth")
if not os.path.exists(weights_path):
    # Fallback to local user Downloads folder
    weights_path = r"C:\Users\aksha\Downloads\finetuned_nafnet.pth"

print(f"[2/3] Running run.py on simulated image using weights: {weights_path}...")

cmd = [
    sys.executable,
    os.path.join(script_dir, "run.py"),
    temp_input,
    temp_output,
    "--checkpoint", weights_path
]

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)

if result.returncode != 0:
    print(f"Error executing run.py: {result.stderr}")
    exit(1)

# 3. Load the restored image and verify it
restored_path = os.path.join(temp_output, "simulated_degraded.npy")
if not os.path.exists(restored_path):
    print("Error: Restored image was not saved!")
    exit(1)

restored = np.load(restored_path)
print("[3/3] Restored output loaded successfully.")
print(f"   -> Input Shape: (128, 128) | Output Shape: {restored.shape}")
print(f"   -> Value range: Min={restored.min():.4f}, Max={restored.max():.4f}")

# Verification assertions
assert restored.shape == (256, 256), f"Output shape should be (256, 256) but got {restored.shape}"
assert restored.min() >= 0.0 and restored.max() <= 1.0, "Output values must be clamped between [0.0, 1.0]"

# Draw a comparison plot
fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
axes[0].imshow(degraded, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Simulated Degraded Input (128x128)")
axes[0].axis('off')

axes[1].imshow(restored, cmap='gray', vmin=0, vmax=1)
axes[1].set_title("NAFNet Restored Output (256x256)")
axes[1].axis('off')

plt.tight_layout()
output_img = os.path.join(script_dir, "smoke_test_result.png")
plt.savefig(output_img)
print(f"\n*** Visual comparison saved to: {output_img}")
print("SUCCESS: Model runs cleanly, loads weights, processes dimensions, and clamps output correctly!")

# Cleanup temp files
try:
    os.remove(os.path.join(temp_input, "simulated_degraded.npy"))
    os.remove(os.path.join(temp_output, "simulated_degraded.npy"))
    os.rmdir(temp_input)
    os.rmdir(temp_output)
except Exception as e:
    pass
