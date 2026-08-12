# AI-Based Wafer Image Restoration Pipeline (SEMICON India Hackathon 2026, KLA PS01)

This repository contains the complete submission codebase for the joint super-resolution ($2\times$ upscaling), deblurring, and denoising of semiconductor wafer inspection layout images (PS01). The pipeline is built using a Nonlinear Activation Free Network (NAFNet).

---

## 📂 Repository Structure
* **`evaluate.py`**: Standalone evaluation script for the benchmarking team. Supports batching, Test-Time Augmentation (TTA), and high-precision float32/bf16 operations.
* **`model.py`**: Definition of the Nonlinear Activation Free Network (`NAFNet`) model architecture.
* **`dataset.py`**: PyTorch dataset loaders supporting wafer image loading, on-the-fly degradation modeling, and kernel/noise-profile estimations.
* **`train.py`**: Training pipeline for model pre-training and fine-tuning.
* **`infer_tta.py` / `infer_optimized.py`**: Inference optimization and TTA generators.
* **`estimated_kernel.npy` / `noise_profile.npy`**: Extracted degradation profile parameters.
* **`finetuned_nafnet.pth`**: Final trained weights checkpoint (~60 MB).
* **`restored_outputs/`**: Folder containing the 400 restored `.npy` outputs from the test set.
* **`requirements.txt`**: Minimal environment dependencies to execute the inference script.

---

## 🏛️ Model Architecture Summary
We implement the Nonlinear Activation Free Network (**NAFNet**), optimized for high-throughput image restoration:
* **Base Width:** 32 channels.
* **Encoder Block Configuration:** `[2, 2, 4]` (3 levels with 2, 2, and 4 blocks respectively).
* **Middle Blocks:** 8 blocks.
* **Decoder Block Configuration:** `[2, 2, 2]` (3 levels with 2, 2, and 2 blocks respectively).
* **Upscaler:** Sub-pixel Convolution (PixelShuffle) with an upscale factor of 2.
* **Parameters:** 4,953,381.
* **Key Design Principles:** Bypasses computationally heavy activation functions (such as GELU/ReLU) in favor of a multiplication-based **Simple Gate**, and simplifies spatial attention to a parameter-free **Simplified Channel Attention**.

---

## 📊 Benchmarking Results

Below are the exact metrics achieved by our model on the validation and out-of-distribution proxy datasets:

| Dataset / Configuration | Metric | Bicubic Upsampling | Our Model |
| :--- | :--- | :--- | :--- |
| **In-distribution validation** (320 held-out pairs) | **PSNR** / **SSIM** | 22.7262 dB / 0.5241 | 28.9286 dB / 0.7817 |
| **Out-of-distribution proxy** (degradation outside training range) | **PSNR** / **SSIM** | 24.3342 dB / 0.5262 | 26.7351 dB / 0.6683 |

* **Throughput:** 6.65 FPS with 8x TTA on a Kaggle T4 GPU
* **Model size:** 4,953,381 parameters

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
*(Note: A complete pip freeze requirements file containing the exact Kaggle training environment versions can be provided as `requirements-full.txt` for absolute reproducibility.)*

---

## 🖥️ Running Evaluation (Benchmarking)

The evaluation script [`evaluate.py`](file:///C:/Users/aksha/.gemini/antigravity/scratch/kla-semicon-2026/evaluate.py) loads the model weights from [`finetuned_nafnet.pth`](file:///C:/Users/aksha/.gemini/antigravity/scratch/kla-semicon-2026/finetuned_nafnet.pth) (resolved relative to the script file), reads the degraded `.npy` arrays (shape `128x128`, float32) from the input directory, processes them in batches, and saves the clamped restored outputs (shape `256x256`, values in `[0, 1]`) in the output directory.

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
