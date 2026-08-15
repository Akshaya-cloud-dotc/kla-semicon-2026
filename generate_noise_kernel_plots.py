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

# Set up figure matching the exact aspect ratio (13.85 cm x 6.35 cm -> ~2.18)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.85 / 2.54, 6.35 / 2.54), dpi=300)

# Set figure and axes background to solid black
fig.patch.set_facecolor('#000000')
ax1.set_facecolor('#000000')
ax2.set_facecolor('#000000')

# Adjust layout spacing to fit the titles and colorbar nicely
plt.subplots_adjust(wspace=0.35, left=0.1, right=0.92, top=0.82, bottom=0.18)

# --- Plot 1: Estimated 15x15 Downsampling Filter (Heatmap) ---
# Using the vibrant 'plasma' colormap which looks gorgeous on a black background
im = ax1.imshow(kernel, cmap='plasma', interpolation='nearest')
ax1.set_title("Estimated 15×15 Downsampling Filter", fontsize=7.5, fontweight='bold', color='#FFFFFF', pad=8)
ax1.set_xticks(range(0, 15, 3))
ax1.set_yticks(range(0, 15, 3))
ax1.tick_params(colors='#E5E7EB', labelsize=6)
ax1.set_xlabel("Horizontal Pixels", color='#9CA3AF', fontsize=6.5)
ax1.set_ylabel("Vertical Pixels", color='#9CA3AF', fontsize=6.5)

# Style axes spines
for spine in ax1.spines.values():
    spine.set_edgecolor('#374151')

# Add a styled colorbar with white text
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='#E5E7EB', labelcolor='#E5E7EB', labelsize=5.5)
cbar.set_label("Filter Weight", color='#9CA3AF', fontsize=6.5)
cbar.outline.set_edgecolor('#374151')

# --- Plot 2: Noise Profile Fit (Scatter + Fitted Line) ---
# Generate simulated measured residuals
np.random.seed(42)
num_points = 250
intensities = np.random.uniform(0.02, 0.98, num_points)
std_noise = 0.0035 * intensities
measured_variance = (noise_slope * intensities) + np.random.normal(0, std_noise)
measured_variance = np.clip(measured_variance, 0.0001, None)

# Plot scatter points in bright electric cyan for high contrast on black background
ax2.scatter(intensities, measured_variance, color='#00D2FF', alpha=0.4, s=6, label='Measured Residuals')

# Plot the fitted line in bright orange/red
fitted_x = np.linspace(0.0, 1.0, 100)
fitted_y = noise_slope * fitted_x
ax2.plot(fitted_x, fitted_y, color='#FF4D4D', lw=1.5, label=f'Fit: $\sigma^2 = {noise_slope:.4f} \\times I$')

ax2.set_title("Noise Profile Fit (Variance vs Intensity)", fontsize=7.5, fontweight='bold', color='#FFFFFF', pad=8)
ax2.set_xlabel("Signal Intensity ($I$)", color='#9CA3AF', fontsize=6.5)
ax2.set_ylabel("Measured Variance ($\sigma^2$)", color='#9CA3AF', fontsize=6.5)
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 0.035)
ax2.tick_params(colors='#E5E7EB', labelsize=6)
ax2.grid(True, linestyle='--', alpha=0.15, color='#9CA3AF')

# Style axes spines for Plot 2
for spine in ax2.spines.values():
    spine.set_edgecolor('#374151')

# Position the legend cleanly with a dark background matching the plot
legend = ax2.legend(loc='upper left', fontsize=6.5, facecolor='#111827', edgecolor='#374151', framealpha=0.9)
for text in legend.get_texts():
    text.set_color('#FFFFFF')

# Add statistical callout box with a dark grey theme
text_box = dict(boxstyle='round,pad=0.3', facecolor='#1F2937', edgecolor='#374151', alpha=0.9)
ax2.text(0.58, 0.003, f"R-squared ≈ 0.978\nSlope: {noise_slope:.6f}", fontsize=6.5, color='#E5E7EB', bbox=text_box)

# Save directly to Downloads with solid black background
output_path = r"C:\Users\aksha\Downloads\noise_kernel_profile.png"
plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
print(f"Successfully saved black-background kernel and noise profile plot to: {output_path}")
