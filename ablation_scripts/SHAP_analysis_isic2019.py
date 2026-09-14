# -*- coding: utf-8 -*-
"""
SHAP Explainability Analysis: Identifying Diagnostic Collapse in KAN Topology
Dataset: ISIC 2019

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: 
- Evaluates the pixel-level focus of ViT+KAN architectures before and after Extreme Fourier Truncation (BCT).
- Specifically isolates the Basal Cell Carcinoma (BCC) class to demonstrate the hallucination of malignancy 
  caused by the erasure of high-frequency diagnostic anchors (micro-textures).
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. SETTINGS AND DATA PATH
# =============================================================================
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust this path according to your local ISIC 2019 dataset location
DATA_DIR = r"./data/ISIC2019"

# =============================================================================
# 2. MODEL ARCHITECTURES
# =============================================================================

class BCTLayer(nn.Module):
    """Banded Compression Transform (Spectral Truncation)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.complex_weight = nn.Parameter(torch.randn(out_features // 2 + 1, dtype=torch.cfloat) * 0.02)
        
    def forward(self, x):
        x_freq = torch.fft.rfft(x, dim=-1)
        x_freq_trunc = x_freq[:, :self.out_features // 2 + 1]
        x_freq_trunc = x_freq_trunc * self.complex_weight
        return torch.fft.irfft(x_freq_trunc, n=self.out_features, dim=-1)

class KANHead(nn.Module):
    """Minimal Kolmogorov-Arnold Network (KAN) Proxy Head"""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.kan_layers = nn.Sequential(
            nn.Linear(in_features, in_features), 
            nn.SiLU(), 
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.kan_layers(x)

class ViT_KAN_Baseline(nn.Module):
    """Uncompressed Baseline Architecture"""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None)
        self.backbone.heads.head = nn.Identity()
        self.kan_head = KANHead(in_features=768, num_classes=num_classes)
        
    def forward(self, x):
        return self.kan_head(self.backbone(x))

class ViT_BCT_KAN_Proposed(nn.Module):
    """Compressed Proposed Architecture (Subject to Diagnostic Collapse)"""
    def __init__(self, num_classes, compressed_dim=256):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None)
        self.backbone.heads.head = nn.Identity()
        self.bct = BCTLayer(in_features=768, out_features=compressed_dim)
        self.kan_head = KANHead(in_features=compressed_dim, num_classes=num_classes)
        
    def forward(self, x):
        return self.kan_head(self.bct(self.backbone(x)))

class TargetClassWrapper(nn.Module):
    """Wraps the model to isolate and return the gradients for a specific target class."""
    def __init__(self, model, target_class_idx):
        super().__init__()
        self.model = model
        self.target_class_idx = target_class_idx

    def forward(self, x):
        return self.model(x)[:, self.target_class_idx].unsqueeze(1)

# =============================================================================
# 3. TEST DATA LOADING & TARGET SELECTION
# =============================================================================

def get_target_images(data_dir):
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
    generator = torch.Generator().manual_seed(42)
    val_size = len(full_val_dataset) // 2
    test_size = len(full_val_dataset) - val_size
    _, test_dataset = random_split(full_val_dataset, [val_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Generate background distribution for SHAP (approx. 32 images)
    background_images, _ = next(iter(test_loader))
    background_images = background_images.to(DEVICE)
    
    # Isolate a specific Basal Cell Carcinoma (BCC) image as the target
    target_image = None
    for images, labels in test_loader:
        for idx, label in enumerate(labels):
            if label.item() == 2:  # Index 2 corresponds to the BCC Class
                target_image = images[idx].unsqueeze(0).to(DEVICE)
                break
        if target_image is not None:
            break
            
    return background_images, target_image

# =============================================================================
# 4. SHAP GRADIENT EXTRACTION & PLOTTING
# =============================================================================

if __name__ == '__main__':
    print(">> Initializing Architectures and Loading Trained Weights...")
    
    model_baseline = ViT_KAN_Baseline(num_classes=NUM_CLASSES).to(DEVICE)
    # Ensure the weights exist in your directory before running
    if os.path.exists('best_baseline_vit_kan.pth'):
        model_baseline.load_state_dict(torch.load('best_baseline_vit_kan.pth', map_location=DEVICE))
    model_baseline.eval()

    model_proposed = ViT_BCT_KAN_Proposed(num_classes=NUM_CLASSES, compressed_dim=256).to(DEVICE)
    if os.path.exists('best_proposed_vit_bct_kan.pth'):
        model_proposed.load_state_dict(torch.load('best_proposed_vit_bct_kan.pth', map_location=DEVICE))
    model_proposed.eval()

    # Wrap models to observe ONLY the BCC (Index 2) prediction gradients
    bcc_model_base = TargetClassWrapper(model_baseline, target_class_idx=2)
    bcc_model_prop = TargetClassWrapper(model_proposed, target_class_idx=2)

    print(">> Selecting Evaluation Images from ISIC 2019...")
    try:
        background, test_image = get_target_images(DATA_DIR)
    except FileNotFoundError:
        print(f"\nERROR: ISIC 2019 Dataset not found at {DATA_DIR}.")
        print("Please run 'isic2019_vit_bct_kan_ablation.py' first to download the dataset.")
        sys.exit(1)

    print(">> Calculating SHAP gradients for Baseline Model (BCC Focus)...")
    explainer_base = shap.GradientExplainer(bcc_model_base, background)
    shap_vals_base = explainer_base.shap_values(test_image)

    print(">> Calculating SHAP gradients for Proposed Model (Degraded Focus)...")
    explainer_prop = shap.GradientExplainer(bcc_model_prop, background)
    shap_vals_prop = explainer_prop.shap_values(test_image)

    # ---------------------------------------------------------
    # DIMENSIONALITY SQUEEZE (Formatting for SHAP Visualizer)
    # ---------------------------------------------------------
    if isinstance(shap_vals_base, list):
        shap_vals_base = shap_vals_base[0]
    if isinstance(shap_vals_prop, list):
        shap_vals_prop = shap_vals_prop[0]

    # Squeeze unnecessary dimensions: forces shape to (3, 224, 224)
    shap_vals_base = np.squeeze(np.array(shap_vals_base))
    shap_vals_prop = np.squeeze(np.array(shap_vals_prop))
    
    # SHAP image_plot requires (H, W, C) format: (3, 224, 224) -> (224, 224, 3)
    shap_numpy_base = np.transpose(shap_vals_base, (1, 2, 0))
    shap_numpy_prop = np.transpose(shap_vals_prop, (1, 2, 0))
    
    # Format original test image
    test_numpy = np.squeeze(test_image.cpu().numpy())
    test_numpy = np.transpose(test_numpy, (1, 2, 0))

    print(">> Rendering Publication-Ready Plot...")
    
    # Render the dual-heatmap comparison
    shap.image_plot(
        [shap_numpy_base, shap_numpy_prop], 
        -test_numpy, 
        labels=['Baseline ViT+KAN\n(BCC Diagnostic Focus)', 'Proposed ViT+BCT+KAN\n(Degraded Focus)'],
        show=False
    )
    
    output_filename = 'Figure_8_SHAP_Analysis.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"\n[SUCCESS] SHAP diagnostic attribution maps saved to '{output_filename}'.")