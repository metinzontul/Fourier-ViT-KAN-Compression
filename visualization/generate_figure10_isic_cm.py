# -*- coding: utf-8 -*-
"""
Confusion Matrix Visualization: Exploratory Diagnostic Error Analysis
Dataset: ISIC 2019

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: 
- Evaluates the class-wise diagnostic shifts induced by Fourier-domain truncation.
- Uses proxy configurations to generate comparative confusion matrices (Figure 10).
- Highlights the hallucination of malignancy (e.g., benign lesions misclassified as MEL) 
  caused by the structural compression.
- Automatically generates publication-ready plots with strict academic formatting.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# =============================================================================
# PUBLICATION FORMATTING SETTINGS
# =============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# =============================================================================
# 1. SETTINGS AND DATA PATH
# =============================================================================
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust this path according to your local ISIC 2019 dataset location
DATA_DIR = r"./data/ISIC2019"

# =============================================================================
# 2. PROXY MODEL ARCHITECTURES
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
    """Uncompressed Proxy Architecture"""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None)
        self.backbone.heads.head = nn.Identity()
        self.kan_head = KANHead(in_features=768, num_classes=num_classes)
        
    def forward(self, x):
        return self.kan_head(self.backbone(x))

class ViT_BCT_KAN_Proposed(nn.Module):
    """BCT-Integrated Proxy Architecture"""
    def __init__(self, num_classes, compressed_dim=256):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None)
        self.backbone.heads.head = nn.Identity()
        self.bct = BCTLayer(in_features=768, out_features=compressed_dim)
        self.kan_head = KANHead(in_features=compressed_dim, num_classes=num_classes)
        
    def forward(self, x):
        return self.kan_head(self.bct(self.backbone(x)))

# =============================================================================
# 3. PREDICTIONS AND PLOTTING
# =============================================================================

def get_predictions(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

def plot_dual_confusion_matrix(y_true_base, y_pred_base, y_true_prop, y_pred_prop, class_names):
    cm_base = confusion_matrix(y_true_base, y_pred_base)
    cm_prop = confusion_matrix(y_true_prop, y_pred_prop)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'wspace': 0.15})
    
    # Subplot 1: Uncompressed proxy configuration
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[0].set_title('(a) Uncompressed proxy configuration', pad=20)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Subplot 2: BCT-integrated proxy configuration
    cbar_ax = fig.add_axes([.92, .15, .02, .7]) # Custom colorbar
    sns.heatmap(cm_prop, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
                xticklabels=class_names, yticklabels=class_names, cbar=True, cbar_ax=cbar_ax)
    axes[1].set_title('(b) BCT-integrated proxy configuration', pad=20)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    # Reorder tick labels and apply bold formatting
    ordered_classes = sorted(class_names)
    for ax in axes:
        ax.set_xticklabels(ordered_classes, rotation=0, fontweight='bold')
        ax.set_yticklabels(ordered_classes, rotation=90, va='center', fontweight='bold')

    output_filename = 'figure_10.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"\n[SUCCESS] Exploratory dual confusion matrix saved to '{output_filename}'.")


if __name__ == '__main__':
    print(">> Initializing Proxy Architectures and Loading Trained Weights...")
    
    model_baseline = ViT_KAN_Baseline(num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists('best_baseline_vit_kan.pth'):
        model_baseline.load_state_dict(torch.load('best_baseline_vit_kan.pth', map_location=DEVICE))
        
    model_proposed = ViT_BCT_KAN_Proposed(num_classes=NUM_CLASSES, compressed_dim=256).to(DEVICE)
    if os.path.exists('best_proposed_vit_bct_kan.pth'):
        model_proposed.load_state_dict(torch.load('best_proposed_vit_bct_kan.pth', map_location=DEVICE))

    print(">> Preparing validation subset from ISIC 2019...")
    try:
        transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        full_val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform)
        CLASS_NAMES = full_val_dataset.classes 
        
        generator = torch.Generator().manual_seed(42)
        val_size = len(full_val_dataset) // 2
        test_size = len(full_val_dataset) - val_size
        _, test_dataset = random_split(full_val_dataset, [val_size, test_size], generator=generator)
        
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    except Exception as e:
        print(f"\nERROR: Could not load ISIC 2019 Dataset from {DATA_DIR}.")
        print("Please ensure you have run the ablation scripts to generate the data directory.")
        sys.exit(1)

    print(">> Generating predictions for Uncompressed Proxy Configuration...")
    y_true_base, y_pred_base = get_predictions(model_baseline, test_loader)
    
    print(">> Generating predictions for BCT-integrated Proxy Configuration...")
    y_true_prop, y_pred_prop = get_predictions(model_proposed, test_loader)

    print(">> Rendering Publication-Ready Plot...")
    plot_dual_confusion_matrix(y_true_base, y_pred_base, y_true_prop, y_pred_prop, CLASS_NAMES)