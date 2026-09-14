# -*- coding: utf-8 -*-
"""
Generate Figure 5: ISIC 2019 Diagnostic Resilience Dichotomy

Author: Prof. Dr. Metin Zontul
Date: September 2026
Description: 
- Visualizes the trade-off between parameter compression and diagnostic resilience 
  on the ISIC 2019 dataset, comparing ViT+KAN and ViT+MLP architectures.
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

# Real data extracted from Table 4 (ISIC 2019 Ablation Logs)
labels = ['ViT + KAN Head', 'ViT + MLP Head']
f1_base = [0.7893, 0.7749]
f1_bct = [0.7006, 0.7721]

params_base = [55296, 6152]
params_bct = [9600, 1416]

x = np.arange(len(labels))
width = 0.35

# Create a 1x2 grid for subplots with 600 DPI resolution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=600)

# =============================================================================
# 2. PANEL (a): MACRO F1 SCORES
# =============================================================================
rects1_f1 = ax1.bar(x - width/2, f1_base, width, label='Baseline (Uncompressed)', color='#2b5c8f', edgecolor='black')
rects2_f1 = ax1.bar(x + width/2, f1_bct, width, label='Proposed (BCT Compressed)', color='#d9534f', edgecolor='black')

ax1.set_ylabel('Macro F1 Score')
ax1.set_title('(a) Diagnostic Resilience (ISIC 2019)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim([0.65, 0.82]) 
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Autolabel F1 scores on top of bars
def autolabel_f1(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel_f1(rects1_f1, ax1)
autolabel_f1(rects2_f1, ax1)

# Highlight F1 drop (Diagnostic Collapse) for KAN
ax1.annotate('', xy=(x[0] + width/2, 0.7006), xytext=(x[0] - width/2, 0.7893),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
ax1.text(x[0], 0.74, 'Severe\nCollapse', ha='center', va='center', color='black', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# Highlight F1 stability for MLP
ax1.annotate('', xy=(x[1] + width/2, 0.7721), xytext=(x[1] - width/2, 0.7749),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
ax1.text(x[1], 0.79, 'Stable', ha='center', va='center', color='black', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# =============================================================================
# 3. PANEL (b): HARDWARE COMPRESSION (LOG SCALE)
# =============================================================================
rects1_p = ax2.bar(x - width/2, params_base, width, label='Baseline', color='#2b5c8f', edgecolor='black')
rects2_p = ax2.bar(x + width/2, params_bct, width, label='Proposed (BCT)', color='#d9534f', edgecolor='black')

ax2.set_ylabel('Head Parameter Count (Log Scale)')
ax2.set_title('(b) Hardware Compression')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_yscale('log')
ax2.legend(loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Autolabel parameter counts
def autolabel_p(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel_p(rects1_p, ax2)
autolabel_p(rects2_p, ax2)

# =============================================================================
# 4. SAVE AND SHOW
# =============================================================================
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
output_path = os.path.join("figures", "Figure_5_ISIC2019_Dichotomy.png")
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"Successfully saved figure to {output_path}")
plt.show()