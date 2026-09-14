# -*- coding: utf-8 -*-
"""
Generate Figure 7: Confusion Matrix Comparison for ISIC 2019

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: 
- Evaluates the models on the unseen test set and generates dual confusion matrices.
- Visually highlights the "Diagnostic Collapse" where the proposed ViT+BCT+KAN model 
  hallucinates malignancy (e.g., misclassifying Benign conditions as Malignant Carcinomas) 
  due to the removal of high-frequency spatial anchors.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif', 'axes.labelsize': 14,
    'axes.titlesize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10
})

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Relative path suitable for GitHub structure
DATA_DIR = "./data/ISIC2019"

# =============================================================================
# 2. STANDALONE MODEL ARCHITECTURES FOR EVALUATION
# =============================================================================
class BCTLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.complex_weight = nn.Parameter(torch.randn(out_features // 2 + 1, dtype=torch.cfloat) * 0.02)

    def forward(self, x):
        x_freq = torch.fft.rfft(x, dim=-1)
        x_freq_trunc = x_freq[:, :self.out_features // 2 + 1]
        x_freq_trunc = x_freq_trunc * self.complex_weight
        x_compressed = torch.fft.irfft(x_freq_trunc, n=self.out_features, dim=-1)
        return x_compressed

class KANHead(nn.Module):
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
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None) 
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity()
        self.kan_head = KANHead(in_features=in_features, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.kan_head(x)

class ViT_BCT_KAN_Proposed(nn.Module):
    def __init__(self, num_classes, compressed_dim=256):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity()
        self.bct = BCTLayer(in_features=in_features, out_features=compressed_dim)
        self.kan_head = KANHead(in_features=compressed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bct(x)
        return self.kan_head(x)

# =============================================================================
# 3. DATALOADER (UNSEEN TEST SET ONLY)
# =============================================================================
def get_test_loader(data_dir, batch_size):
    print(f">> Loading true test set (Leakage-Free, Seed=42): {data_dir}")
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        full_val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_test_transform)
    except FileNotFoundError:
        print(f"Dataset not found at {data_dir}. Please run the training script first to download.")
        return None

    val_size = len(full_val_dataset) // 2
    test_size = len(full_val_dataset) - val_size
    
    # Must use identical seed from training script to ensure identical split
    generator = torch.Generator().manual_seed(42)
    _, test_dataset = random_split(full_val_dataset, [val_size, test_size], generator=generator)

    print(f">> Unseen Test Set Size: {len(test_dataset)} images")
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# =============================================================================
# 4. INFERENCE FUNCTION
# =============================================================================
def evaluate_model(model, dataloader):
    model.to(DEVICE)
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))

# =============================================================================
# 5. MATRIX PLOTTING
# =============================================================================
def plot_figure_7(cm_baseline, cm_bct):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=600)
    cmap = "Blues"

    sns.heatmap(cm_baseline, annot=True, fmt='g', cmap=cmap, ax=ax1, 
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False, annot_kws={"size": 11})
    ax1.set_title('(a) Baseline ViT+KAN\n(No Compression)', pad=15)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    sns.heatmap(cm_bct, annot=True, fmt='g', cmap=cmap, ax=ax2, 
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=True, annot_kws={"size": 11})
    ax2.set_title('(b) Proposed ViT+BCT+KAN\n(Severe Diagnostic Collapse)', pad=15)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    # Highlight the specific shift from Carcinoma (BCC) to Benign (BKL)
    # x = 4 (BKL column), y = 2 (BCC row)
    ax2.add_patch(plt.Rectangle((4, 2), 1, 1, fill=False, edgecolor='red', lw=4, clip_on=False))
    
    # Adjust arrow to point accurately
    ax2.annotate('Critical Misclassifications\n(Carcinoma \u2192 Benign)', 
                 xy=(4.5, 2.0),           
                 xytext=(2.5, 0.5),       
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 color='red', weight='bold', ha='center', va='center')

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    output_path = os.path.join("figures", "Figure_7_ISIC2019_CM.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\n[SUCCESS] Saved Figure 7 to {output_path}")
    plt.show()

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    test_loader = get_test_loader(DATA_DIR, BATCH_SIZE)
    
    if test_loader is not None:
        print("\n>> Initializing models and loading '.pth' weights from disk...")
        model_baseline = ViT_KAN_Baseline(num_classes=NUM_CLASSES)
        model_proposed = ViT_BCT_KAN_Proposed(num_classes=NUM_CLASSES, compressed_dim=256)
        
        try:
            model_baseline.load_state_dict(torch.load('best_baseline_vit_kan.pth', map_location=DEVICE))
            model_proposed.load_state_dict(torch.load('best_proposed_vit_bct_kan.pth', map_location=DEVICE))
        except FileNotFoundError:
            print("ERROR: Pre-trained weights not found. You must run 'isic2019_vit_bct_kan_ablation.py' first.")
            exit(1)
        
        print("\n>> Running inference on the unseen Test set (Please wait)...")
        cm_baseline = evaluate_model(model_baseline, test_loader)
        cm_bct = evaluate_model(model_proposed, test_loader)
        
        print("\n>> Rendering graphic...")
        plot_figure_7(cm_baseline, cm_bct)