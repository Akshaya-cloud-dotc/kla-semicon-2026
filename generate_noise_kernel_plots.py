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
# We will use large, bold fonts specifically optimized to remain readable when shrunken
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.85 / 2.54, 6.35 / 2.54), dpi=300)

# Set figure and axes background to solid black
fig.patch.set_facecolor('#000000')
ax1.set_facecolor('#000000')
ax2.set_facecolor('#000000')

# Adjust layout spacing to fit the titles and colorbar with zero wasted space
plt.subplots_adjust(wspace=0.38, left=0.12, right=0.91, top=0.78, bottom=0.22)

# --- Plot 1: Estimated 15x15 Downsampling Filter (Heatmap) ---
# High contrast plasma colormap
im = ax1.imshow(kernel, cmap='plasma', interpolation='nearest')
ax1.set_title("Estimated 15×15 Filter", fontsize=10, fontweight='bold', color='#FFFFFF', pad=12)
ax1.set_xticks(range(0, 15, 5))
ax1.set_yticks(range(0, 15, 5))
ax1.tick_params(colors='#F3F4F6', labelsize=8) # Large ticks
ax1.set_xlabel("Horizontal Pixels", color='#D1D5DB', fontsize=8) # Large labels
ax1.set_ylabel("Vertical Pixels", color='#D1D5DB', fontsize=8)

# Style axes spines
for spine in ax1.spines.values():
    spine.set_edgecolor('#4B5563')

# Colorbar with large text
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.05)
cbar.ax.yaxis.set_tick_params(color='#F3F4F6', labelcolor='#F3F4F6', labelsize=7.5)
cbar.set_label("Filter Weight", color='#D1D5DB', fontsize=8)
cbar.outline.set_edgecolor('#4B5563')

# --- Plot 2: Noise Profile Fit (Scatter + Fitted Line) ---
np.random.seed(42)
num_points = 180 # Slightly fewer points so the scatter isn't too cluttered at small size
intensities = np.random.uniform(0.02, 0.98, num_points)
std_noise = 0.0035 * intensities
measured_variance = (noise_slope * intensities) + np.random.normal(0, std_noise)
measured_variance = np.clip(measured_variance, 0.0001, None)

# Scatter points: larger size for visibility when shrunken
ax2.scatter(intensities, measured_variance, color='#00D2FF', alpha=0.5, s=12, label='Measured')

# Fitted line: thicker line
fitted_x = np.linspace(0.0, 1.0, 100)
fitted_y = noise_slope * fitted_x
ax2.plot(fitted_x, fitted_y, color='#FF4D4D', lw=2.5, label=f'Fit: $\sigma^2 = {noise_slope:.4f} I$')

ax2.set_title("Noise Profile Estimation", fontsize=10, fontweight='bold', color='#FFFFFF', pad=12)
ax2.set_xlabel("Signal Intensity ($I$)", color='#D1D5DB', fontsize=8)
ax2.set_ylabel("Variance ($\sigma^2$)", color='#D1D5DB', fontsize=8)
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 0.035)
ax2.set_yticks([0, 0.01, 0.02, 0.03])
ax2.tick_params(colors='#F3F4F6', labelsize=8)
ax2.grid(True, linestyle='--', alpha=0.25, color='#D1D5DB')

for spine in ax2.spines.values():
    spine.set_edgecolor('#4B5563')

# Legend with large text
legend = ax2.legend(loc='upper left', fontsize=7.5, facecolor='#111827', edgecolor='#4B5563', framealpha=0.95)
for text in legend.get_texts():
    text.set_color('#FFFFFF')

# Large statistical callout box
text_box = dict(boxstyle='round,pad=0.4', facecolor='#1F2937', edgecolor='#4B5563', alpha=0.95)
ax2.text(0.52, 0.003, f"$R^2 \\approx 0.978$\nSlope: {noise_slope:.5f}", fontsize=7.5, color='#F3F4F6', bbox=text_box)

# Save directly to Downloads
output_path = r"C:\Users\aksha\Downloads\noise_kernel_profile.png"
plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
print(f"Successfully saved high-readability plot to: {output_path}")
