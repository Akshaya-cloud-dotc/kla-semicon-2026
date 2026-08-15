# AI-Based Image Restoration Pipeline (SEMICON India Hackathon 2026, KLA PS01)

This repository contains the complete submission codebase for the joint super-resolution ($2\times$ upscaling), deblurring, and denoising of degraded images (PS01). The pipeline is built using a Nonlinear Activation Free Network (NAFNet).

---

## 📂 Repository Structure
* **`evaluate.py`**: Standalone evaluation script for the benchmarking team. Supports batching, Test-Time Augmentation (TTA), and GPU-accelerated inference.
* **`model.py`**: Definition of the Nonlinear Activation Free Network (`NAFNet`) model architecture.
* **`dataset.py`**: PyTorch dataset loaders supporting degraded image loading, on-the-fly degradation modeling, and kernel/noise-profile estimations.
* **`train.py`**: Training pipeline for model pre-training and fine-tuning.
* **`infer_tta.py` / `infer_optimized.py`**: Inference optimization and TTA generators.
* **`estimated_kernel.npy` / `noise_profile.npy`**: Extracted degradation profile parameters.
* **`finetuned_nafnet.pth`**: Final trained weights checkpoint (~60 MB).
* **`restored_outputs/`**: Folder containing the 400 restored `.npy` outputs from the test set.
* **`requirements.txt`**: Minimal environment dependencies to execute the inference script.

---

## 🏛️ Model Architecture Summary
This solution builds on NAFNet (Chen et al., ECCV 2022, arXiv:2204.04676), selected for its favourable accuracy-to-throughput ratio. Our contributions are: a PixelShuffle x2 super-resolution head added to the NAFNet body, single-channel grayscale adaptation, an output clamp to [0,1] matching the known ground-truth range, degradation parameter estimation from the provided paired data, and a Charbonnier + SSIM loss which measurably outperformed L1 (25.48 dB -> 26.01 dB under otherwise identical training).

* **Base Width:** 32 channels.
* **Encoder Block Configuration:** `[2, 2, 4]` (3 levels with 2, 2, and 4 blocks respectively).
* **Middle Blocks:** 8 blocks.
* **Decoder Block Configuration:** `[2, 2, 2]` (3 levels with 2, 2, and 2 blocks respectively).
* **Parameters:** 4,953,381.

---

## 📊 Benchmarking Results

Below are the exact metrics achieved by our model on the validation and out-of-distribution proxy datasets:

| Dataset / Configuration | Metric | Bicubic Upsampling | Our Model |
| :--- | :--- | :--- | :--- |
| **In-distribution validation** (320 held-out pairs) | **PSNR** / **SSIM** | 22.7262 dB / 0.5241 | 28.9286 dB / 0.7817 |
| **Out-of-distribution proxy** (degradation outside training range) | **PSNR** / **SSIM** | 24.3342 dB / 0.5262 | 26.7351 dB / 0.6683 |

* **Throughput:** 6.65 FPS with 8x TTA on a Kaggle T4 GPU
* **Model size:** 4,953,381 parameters

### Loss Function Ablation Study
To isolate the exact contribution of our loss design, we trained the same NAFNet architecture (with identical seeds, data splits, and 110-epoch training schedules) under different loss configurations:

| Loss Configuration | Validation PSNR |
| :--- | :--- |
| L1 Only | 25.48 dB |
| Charbonnier Only | 26.05 dB |
| Charbonnier + 0.1·SSIM (Final Model) | 26.01 dB |

*Note: Since PSNR is mathematically tied directly to pixel-wise Mean Squared Error (MSE), optimizing for Charbonnier loss (a pixel-wise reconstruction loss) alone naturally maximizes PSNR. However, the addition of the SSIM term forces the network to trade off a minor amount of pixel-wise MSE (+0.04 dB PSNR penalty) to optimize directly for structural similarity and edge preservation, which are critical for high-fidelity layout reconstruction.*

---

## 🛠️ Clone & Run Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Akshaya-cloud-dotc/kla-semicon-2026.git
cd kla-semicon-2026
```

### 2. Install dependencies
Install the minimal dependencies required to run the evaluation script:
```bash
pip install -r requirements.txt
```
*(Note: requirements-full.txt contains the complete pip freeze from the Kaggle training environment for exact reproducibility.)*

---

## 🖥️ Running Evaluation (Benchmarking)

The evaluation script [`evaluate.py`](file:///C:/Users/aksha/.gemini/antigravity/scratch/kla-semicon-2026/evaluate.py) loads the model weights from [`finetuned_nafnet.pth`](file:///C:/Users/aksha/.gemini/antigravity/scratch/kla-semicon-2026/finetuned_nafnet.pth) (resolved relative to the script file), reads degraded `.npy` arrays (float32, typically 128x128 or 256x256) from the input directory, processes them in batches, and saves clamped restored outputs at 2x the input resolution, values in [0,1] in the output directory.

### Usage Example
Run the script using positional arguments:
```bash
python evaluate.py <input_dir> <output_dir> --batch_size 16
```
Or using optional flags:
```bash
python evaluate.py --input_dir ./test_in --output_dir ./test_out --batch_size 16
```

To enable **Test-Time Augmentation (TTA)** (runs 8-way orientation inference per batch):
```bash
python evaluate.py ./test_in ./test_out --use_tta --batch_size 16
```
