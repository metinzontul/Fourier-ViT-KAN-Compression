# Pre-trained Model Weights (ISIC 2019)

Due to GitHub's strict file size limit (100 MB), the full Vision Transformer (ViT) backbones and their respective classification heads (`.pth` files) cannot be directly hosted in this repository. 

To ensure full reproducibility of the diagnostic collapse analysis (Figure 7 and Figure 8) without requiring weeks of retraining, the official pre-trained weights are hosted via **GitHub Releases (v1.0.0)**.

## Required Files
To run the evaluation and Explainable AI (SHAP) scripts successfully, you need the following files:
1. `best_baseline_vit_kan.pth` (Uncompressed Baseline)
2. `best_proposed_vit_bct_kan.pth` (Compressed Proposed Model)

## How to Download

### Option 1: Automated Download (Linux / macOS)
Simply execute the provided shell script inside this directory:
```bash
cd weights
chmod +x download_weights.sh
./download_weights.sh