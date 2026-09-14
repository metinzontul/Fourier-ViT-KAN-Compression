# -*- coding: utf-8 -*-
"""
Generate Figure 6: ISIC 2019 Learning Curves

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: 
- Plots the training dynamics (loss and accuracy) over 20 epochs for the ISIC 2019 dataset,
  illustrating the optimization volatility of KANs versus the stability of MLPs.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# 1. PUBLICATION STANDARDS (MATPLOTLIB SETTINGS)
# =============================================================================
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

epochs = np.arange(1, 21)

# =============================================================================
# 2. LOG DATA (AVERAGED OVER 5 RUNS ON ISIC 2019)
# =============================================================================
# Training Loss Data
loss_vit_mlp = [1.301, 0.929, 0.749, 0.593, 0.491, 0.385, 0.335, 0.272, 0.240, 0.197,
                0.194, 0.147, 0.149, 0.133, 0.133, 0.107, 0.083, 0.123, 0.112, 0.078]

loss_vit_kan = [1.610, 1.077, 0.849, 0.681, 0.560, 0.461, 0.358, 0.293, 0.255, 0.198,
                0.198, 0.178, 0.168, 0.137, 0.107, 0.128, 0.091, 0.140, 0.061, 0.105]

# Validation Accuracy Data (%)
acc_vit_mlp = [62.13, 68.31, 68.69, 72.58, 72.17, 74.54, 77.34, 77.20, 77.49, 77.54,
               77.50, 79.35, 78.53, 78.72, 78.76, 80.70, 78.76, 78.36, 81.85, 80.08]

acc_vit_kan = [61.01, 63.83, 66.27, 70.63, 71.91, 70.22, 75.04, 75.38, 76.99, 78.33,
               77.85, 78.49, 79.75, 79.33, 80.22, 80.81, 76.62, 81.83, 79.09, 78.63]

# =============================================================================
# 3. PLOTTING THE GRAPHS
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=600)

# --- Panel (a): Training Loss ---
ax1.plot(epochs, loss_vit_mlp, marker='o', markersize=5, linestyle='-', linewidth=2, color='#2b5c8f', label='ViT+MLP (Baseline)')
ax1.plot(epochs, loss_vit_kan, marker='s', markersize=5, linestyle='--', linewidth=2, color='#d9534f', label='ViT+KAN (Baseline)')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss')
ax1.set_title('(a) Training Loss over 20 Epochs (ISIC 2019)')
ax1.set_xticks(np.arange(2, 22, 2))
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# --- Panel (b): Validation Accuracy ---
ax2.plot(epochs, acc_vit_mlp, marker='o', markersize=5, linestyle='-', linewidth=2, color='#2b5c8f', label='ViT+MLP (Baseline)')
ax2.plot(epochs, acc_vit_kan, marker='s', markersize=5, linestyle='--', linewidth=2, color='#d9534f', label='ViT+KAN (Baseline)')

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Validation Accuracy (%)')
ax2.set_title('(b) Validation Accuracy over 20 Epochs (ISIC 2019)')
ax2.set_xticks(np.arange(2, 22, 2))
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.7)

# =============================================================================
# 4. SAVE AND SHOW
# =============================================================================
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
output_path = os.path.join("figures", "Figure_6_ISIC2019_Learning_Curve.png")
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"Successfully saved figure to {output_path}")
plt.show()