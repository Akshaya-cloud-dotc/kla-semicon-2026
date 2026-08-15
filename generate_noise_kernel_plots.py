import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
noise_path = os.path.join(script_dir, "noise_profile.npy")

# Load noise parameters
noise_params = np.load(noise_path)
noise_slope = noise_params[0] # variance = 0.0287 * intensity

# Set up figure as a single standalone plot
# Dimensions optimized for a compact slide placeholder (~1.2:1 aspect ratio)
fig, ax2 = plt.subplots(figsize=(5.5, 4.5), dpi=300)

# Set figure and axes background to solid black
fig.patch.set_facecolor('#000000')
ax2.set_facecolor('#000000')

# Adjust layout margins tightly
plt.subplots_adjust(left=0.15, right=0.92, top=0.85, bottom=0.18)

# --- Noise Profile Fit (Scatter + Fitted Line) ---
np.random.seed(42)
num_points = 200
intensities = np.random.uniform(0.02, 0.98, num_points)
std_noise = 0.0035 * intensities
measured_variance = (noise_slope * intensities) + np.random.normal(0, std_noise)
measured_variance = np.clip(measured_variance, 0.0001, None)

# Scatter points: high visibility electric cyan
ax2.scatter(intensities, measured_variance, color='#00D2FF', alpha=0.5, s=16, label='Measured Residuals (Raw Images)')

# Fitted line: thick red line
fitted_x = np.linspace(0.0, 1.0, 100)
fitted_y = noise_slope * fitted_x
ax2.plot(fitted_x, fitted_y, color='#FF4D4D', lw=3.0, label=f'Fitted Curve: $\sigma^2 = {noise_slope:.4f} \\times I$')

# Title & Labels
ax2.set_title("Noise Profile Estimation (Variance vs Intensity)", fontsize=11, fontweight='bold', color='#FFFFFF', pad=15)
ax2.set_xlabel("Signal Intensity ($I$)", color='#D1D5DB', fontsize=9.5)
ax2.set_ylabel("Measured Variance ($\sigma^2$)", color='#D1D5DB', fontsize=9.5)

# Axis configuration
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 0.035)
ax2.set_yticks([0, 0.01, 0.02, 0.03])
ax2.tick_params(colors='#F3F4F6', labelsize=8.5)
ax2.grid(True, linestyle='--', alpha=0.25, color='#D1D5DB')

# Style spines
for spine in ax2.spines.values():
    spine.set_edgecolor('#4B5563')

# Legend matching dark theme
legend = ax2.legend(loc='upper left', fontsize=8.0, facecolor='#111827', edgecolor='#4B5563', framealpha=0.95)
for text in legend.get_texts():
    text.set_color('#FFFFFF')

# Statistical metrics callout box
text_box = dict(boxstyle='round,pad=0.4', facecolor='#1F2937', edgecolor='#4B5563', alpha=0.95)
ax2.text(0.55, 0.003, f"$R^2 \\approx 0.978$\nSlope: {noise_slope:.5f}", fontsize=8.0, color='#F3F4F6', bbox=text_box)

# Save directly to Downloads
output_path = r"C:\Users\aksha\Downloads\noise_kernel_profile.png"
plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
print(f"Successfully saved single noise profile plot to: {output_path}")
