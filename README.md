# Hardware Efficiency vs. Topological Vulnerability: Evaluating Kolmogorov-Arnold Networks under Extreme Fourier-Domain Truncation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![Release](https://img.shields.io/badge/Release-v1.0.0-green.svg)](https://github.com/YourUsername/Fourier-ViT-KAN-Compression/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation for the manuscript:  
**"Hardware Efficiency vs. Topological Vulnerability: Evaluating Kolmogorov-Arnold Networks under Extreme Fourier-Domain Truncation"** (Submitted to *MDPI Electronics*).

---

## 📌 Abstract
This repository evaluates the architectural resilience of **Kolmogorov-Arnold Networks (KANs)** compared to standard **Multi-Layer Perceptrons (MLPs)** under aggressive frequency-domain compression. We introduce the **Banded Compression Transform (BCT)** module, which truncates peripheral high-frequency spatial components and projects features via trainable complex weights in the Fourier domain.

Key findings include:
- **Macroscopic Vision Tasks (CIFAR-10, MNIST, EuroSAT):** BCT achieves over 93% head parameter reduction (e.g., from 184,320 to 11,904 in ResNet50+KAN) and slashes inference latency by nearly 46% with negligible F1 score degradation.
- **Sensitive Medical Diagnostics (ISIC 2019):** BCT triggers a catastrophic **Diagnostic Collapse** in ViT+KAN architectures (Macro F1 plummets from 78.93% to 70.06%, $p < 0.001$), systematically hallucinating malignancy. In stark contrast, globally-oriented ViT+MLPs remain clinically resilient under identical truncation (F1 drops by only 0.28%, $p = 0.727$).
- **Pixel-Level Attribution:** Gradient-based SHAP explainability confirms that KAN's B-Spline grids critically anchor to high-frequency micro-textures, which are eliminated during extreme Fourier-domain truncation.

---

## 📁 Repository Structure

```text
Fourier-ViT-KAN-Compression/
├── models/
│   ├── bct_module.py                       # BCT spectral pooling and Fourier projection layers
│   ├── kan_layer.py                        # Minimal KAN layer utilizing B-Spline parameterization
│   ├── vit_kan.py                          # Baseline ViT+KAN and proposed ViT+BCT+KAN models
│   ├── vit_mlp.py                          # Baseline ViT+MLP and proposed ViT+BCT+MLP models
│   └── resnet_kan.py                       # ResNet50+KAN and ResNet50+BCT+KAN implementations
├── ablation_scripts/                       # End-to-end ablation pipelines
│   ├── isic2019_vit_bct_kan_ablation.py    # Evaluates KAN diagnostic collapse on skin lesions
│   ├── isic2019_vit_bct_mlp_ablation.py    # Demonstrates MLP clinical resilience
│   ├── resnet50_comprehensive_ablation.py  # ResNet50 ablation study (MNIST, EuroSAT)
│   ├── vit_comprehensive_ablation.py       # ViT ablation study (CIFAR-10, MNIST, EuroSAT)
│   └── SHAP_analysis_isic2019.py           # Generates pixel-level attribution maps (Figure 8)
├── visualization/                          # Publication-ready figure generators
│   ├── generate_figure5_isic_dichotomy.py  # Figure 5: F1 drop vs. parameter compression
│   ├── generate_figure6_isic2019_learning_curve.py # Figure 6: Training dynamics over 20 epochs
│   ├── generate_figure7_isic_confusion_matrix.py   # Figure 7: Dual confusion matrix comparison
│   └── plot_PSD.py                         # Figure 1: Power Spectral Density (PSD) analysis
├── weights/                                # Pre-trained model weight management
│   ├── download_weights.sh                 # Bash script to fetch v1.0.0 weights via wget
│   └── README.md                           # Hosting & manual download instructions
├── logs/                                   # Empirical logs matching manuscript statistics
│   ├── ISIC_2019_Ablation_Results.txt
│   ├── ResNet50_Ablation_Results.txt
│   ├── ViT_Ablation_Results.txt
│   └── original_raw_logs/                  # Raw, unedited terminal outputs from exploratory phase
├── requirements.txt                        # Computational dependencies
└── README.md
```

---

## ⚙️ Installation & Environment Setup

Clone this repository and configure the isolated Anaconda environment:

```bash
# Clone the repository
git clone [https://github.com/YourUsername/Fourier-ViT-KAN-Compression.git](https://github.com/YourUsername/Fourier-ViT-KAN-Compression.git)
cd Fourier-ViT-KAN-Compression

# Create and activate the conda environment
conda create -n fourier_kan python=3.10 -y
conda activate fourier_kan

# Install dependencies
pip install -r requirements.txt
```

---

## 💾 Pre-trained Model Weights (Release v1.0.0)

Due to GitHub's 100 MB file limit, complete ViT checkpoints containing pre-trained weights (`best_baseline_vit_kan.pth` and `best_proposed_vit_bct_kan.pth`, ~340 MB each) are hosted via [GitHub Releases v1.0.0](https://github.com/YourUsername/Fourier-ViT-KAN-Compression/releases/tag/v1.0.0).

To download the weights automatically:
```bash
cd weights
chmod +x download_weights.sh
./download_weights.sh
cd ..
```
*Note: Place the downloaded `.pth` files in the repository root directory to run the visualization scripts directly without re-training.*

---

## 🚀 Reproducing Key Results

### 1. Generating Publication Figures
To replicate the figures from the manuscript using pre-trained weights:
```bash
# Figure 1: Power Spectral Density (PSD) analysis across datasets
python visualization/plot_PSD.py

# Figure 5: Diagnostic Resilience Dichotomy (Macro F1 vs. Hardware)
python visualization/generate_figure5_isic_dichotomy.py

# Figure 6: Training loss and validation accuracy learning curves
python visualization/generate_figure6_isic2019_learning_curve.py

# Figure 7: Class-wise Confusion Matrices (Diagnosing Malignancy Hallucination)
python visualization/generate_figure7_isic_confusion_matrix.py

# Figure 8: Gradient-based SHAP Feature Attribution Maps
python ablation_scripts/SHAP_analysis_isic2019.py
```

### 2. Running Comprehensive Ablation Studies
To execute complete 5-seed, 20-epoch training and validation pipelines:
```bash
# Run multi-dataset ResNet50 benchmark (MNIST, EuroSAT)
python ablation_scripts/resnet50_comprehensive_ablation.py

# Run multi-dataset ViT benchmark (Targeting specific datasets like CIFAR-10, MNIST, EuroSAT)
python ablation_scripts/vit_comprehensive_ablation.py --dataset CIFAR-10
python ablation_scripts/vit_comprehensive_ablation.py --dataset EuroSAT

# Run medical ViT ablation benchmark on ISIC 2019
python ablation_scripts/isic2019_vit_bct_kan_ablation.py
python ablation_scripts/isic2019_vit_bct_mlp_ablation.py
```

---

## ⚠️ Transparency & Code Refactoring Note
**Authenticity of Experimental Logs:** To maintain academic transparency and reproducibility, unedited terminal outputs from the original experimental runs are cataloged in `logs/original_raw_logs/`.

The primary experiments were executed across an exploratory codebase during model development. For this public release, the codebase has been refactored, modularized, and documented in English following clean-code guidelines. Running the current scripts will generate outputs matching the metrics and statistical values reported in the manuscript and in `logs/`, while featuring refined logging output.

---

## 📖 Citation
If you use this repository or find our BCT Fourier-compression framework useful in your research, please cite our article:

```bibtex
@article{zontul2026hardware,
  title={Hardware Efficiency vs. Topological Vulnerability: Evaluating Kolmogorov-Arnold Networks under Extreme Fourier-Domain Truncation},
  author={Zontul, Metin},
  journal={Electronics},
  year={2026},
  publisher={MDPI}
}
```