# -*- coding: utf-8 -*-
"""
Generate Figure 9: ISIC 2019 Learning Curves
Publication-ready formatting with distinct markers for ViT+Linear and ViT+KAN.

Author: Prof. Dr. Metin Zontul
Date: September 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12

# Note: Replace with actual parsed log data
epochs = np.arange(1, 21)
# Mock data closely resembling the provided figure
loss_linear = [1.3, 0.93, 0.75, 0.59, 0.49, 0.39, 0.33, 0.27, 0.24, 0.20, 0.20, 0.15, 0.14, 0.13, 0.12, 0.11, 0.08, 0.12, 0.11, 0.08]
loss_kan = [1.61, 1.08, 0.85, 0.68, 0.56, 0.45, 0.36, 0.29, 0.25, 0.20, 0.20, 0.17, 0.17, 0.14, 0.11, 0.12, 0.08, 0.13, 0.06, 0.10]
acc_linear = [62.0, 68.3, 68.7, 72.6, 72.2, 74.9, 77.9, 77.7, 78.0, 78.0, 77.9, 79.2, 78.4, 78.6, 78.6, 80.6, 78.8, 78.3, 81.3, 80.1]
acc_kan = [61.0, 63.8, 66.2, 70.6, 71.9, 70.1, 75.1, 75.4, 77.0, 78.2, 78.0, 78.4, 79.6, 79.2, 80.3, 80.8, 76.7, 81.6, 78.8, 81.0]

color_linear = '#2c6085' # Koyu Mavi
color_kan = '#cc4b51'    # Koyu Kırmızı

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# --- Left Plot: Training Loss ---
axes[0].plot(epochs, loss_linear, '-o', color=color_linear, linewidth=2.5, markersize=7, label='ViT+Linear (Baseline)')
axes[0].plot(epochs, loss_kan, '--s', color=color_kan, linewidth=2.5, markersize=7, label='ViT+KAN (Baseline)')
axes[0].set_title('(a) Training Loss over 20 Epochs (ISIC 2019)', pad=10)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Training Loss')
axes[0].set_xticks(np.arange(2, 21, 2))
axes[0].grid(linestyle='--', alpha=0.5)
legend = axes[0].legend()
for text in legend.get_texts(): text.set_fontweight('bold')

# --- Right Plot: Validation Accuracy ---
axes[1].plot(epochs, acc_linear, '-o', color=color_linear, linewidth=2.5, markersize=7, label='ViT+Linear (Baseline)')
axes[1].plot(epochs, acc_kan, '--s', color=color_kan, linewidth=2.5, markersize=7, label='ViT+KAN (Baseline)')
axes[1].set_title('(b) Validation Accuracy over 20 Epochs (ISIC 2019)', pad=10)
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Validation Accuracy (%)')
axes[1].set_xticks(np.arange(2, 21, 2))
axes[1].grid(linestyle='--', alpha=0.5)
legend = axes[1].legend(loc='lower right')
for text in legend.get_texts(): text.set_fontweight('bold')

plt.tight_layout()
plt.savefig('figure_09.png', dpi=600, bbox_inches='tight')
print(">> Figure 9 (Learning Curves) generated successfully.")