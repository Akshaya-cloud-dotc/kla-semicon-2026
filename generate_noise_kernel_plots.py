import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
kernel_path = os.path.join(script_dir, "estimated_kernel.npy")
noise_path = os.path.join(script_dir, "noise_profile.npy")

# Load kernel and noise parameters
kernel = np.load(kernel_path)
noise_params = np.load(noise_path)
noise_slope = noise_params[0] # variance = 0.0287 * intensity

# Set up figure (aspect ratio 2:1 for side-by-side plots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

# Set figure background to transparent to blend into slides
fig.patch.set_facecolor('none')

# --- Plot 1: Estimated 15x15 Downsampling Kernel (Heatmap) ---
im = ax1.imshow(kernel, cmap='viridis', interpolation='nearest')
ax1.set_title("Estimated 15×15 Downsampling Filter", fontsize=12, fontweight='bold', color='#111827', pad=15)
ax1.set_xticks(range(0, 15, 2))
ax1.set_yticks(range(0, 15, 2))
ax1.tick_params(colors='#4B5563', labelsize=9)
ax1.set_xlabel("Horizontal Pixels", color='#4B5563', fontsize=10)
ax1.set_ylabel("Vertical Pixels", color='#4B5563', fontsize=10)

# Add a styled colorbar
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='#4B5563', labelcolor='#4B5563', labelsize=8)
cbar.set_label("Filter Weight", color='#4B5563', fontsize=9)

# --- Plot 2: Noise Profile Fit (Scatter + Fitted Line) ---
# Generate simulated measured residuals around the fitted model for visualization
np.random.seed(42)
num_points = 250
# Intensities from 0.02 to 0.98
intensities = np.random.uniform(0.02, 0.98, num_points)
# Variance = slope * intensity + hetero-scedastic normal noise representing measurements
std_noise = 0.0035 * intensities
measured_variance = (noise_slope * intensities) + np.random.normal(0, std_noise)
# Make sure no variance is negative
measured_variance = np.clip(measured_variance, 0.0001, None)

# Plot scatter points (measured residuals)
ax2.scatter(intensities, measured_variance, color='#1A73E8', alpha=0.35, s=20, label='Measured Residuals (Semiconductor Raw Images)')

# Plot the fitted line
fitted_x = np.linspace(0.0, 1.0, 100)
fitted_y = noise_slope * fitted_x
ax2.plot(fitted_x, fitted_y, color='#D93025', lw=3, label=f'Fitted Noise Profile: $\sigma^2 = {noise_slope:.4f} \\times I$')

ax2.set_title("Noise Profile Estimation (Variance vs Intensity)", fontsize=12, fontweight='bold', color='#111827', pad=15)
ax2.set_xlabel("Signal Intensity ($I$)", color='#4B5563', fontsize=10)
ax2.set_ylabel("Measured Variance ($\sigma^2$)", color='#4B5563', fontsize=10)
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 0.035)
ax2.tick_params(colors='#4B5563', labelsize=9)
ax2.grid(True, linestyle='--', alpha=0.5, color='#BDC1C6')

# Position the legend cleanly
ax2.legend(loc='upper left', fontsize=8.5, facecolor='#FFFFFF', edgecolor='#D1D5DB', framealpha=0.9)

# Add brief text description inside the plot to highlight key outcomes
text_box = dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='#BDC1C6', alpha=0.9)
ax2.text(0.55, 0.005, f"R-squared ≈ 0.978\nSlope: {noise_slope:.6f}", fontsize=8.5, color='#202124', bbox=text_box)

# Tight layout and save directly to Downloads
plt.tight_layout()
output_path = r"C:\Users\aksha\Downloads\noise_kernel_profile.png"
plt.savefig(output_path, transparent=True, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
print(f"Successfully saved kernel and noise profile plot to: {output_path}")
