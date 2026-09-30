# -*- coding: utf-8 -*-
"""
Generate Figure 4: CIFAR-10 Macro F1 and Parameter Count Comparison
Publication-ready formatting with serif fonts and distinct academic markers.

Author: Prof. Dr. Metin Zontul
Date: September 2026
"""

import matplotlib.pyplot as plt
import numpy as np

# Makale standartları için akademik font ve stil ayarları
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

models = ['ResNet50+KAN', 'ViT+KAN', 'ViT+MLP']
f1_baseline = [94.9340, 93.2860, 94.8880]
f1_bct = [94.6000, 92.9760, 94.0060]

params_baseline = [184320, 85837056, 85806346]
params_bct = [11904, 85808640, 85803530]

x = np.arange(len(models))
width = 0.35

color_base = '#32628d'  # Koyu Mavi
color_bct = '#ce5356'   # Koyu Kırmızı

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Sol Grafik: Macro F1 ---
rects1 = axes[0].bar(x - width/2, f1_baseline, width, label='Baseline', color=color_base, edgecolor='black', zorder=3)
rects2 = axes[0].bar(x + width/2, f1_bct, width, label='BCT-integrated', color=color_bct, edgecolor='black', zorder=3)

axes[0].set_ylabel('Macro F1 Score (%)')
axes[0].set_title('(a) Macro F1 Scores on CIFAR-10', pad=15)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].set_ylim(91, 96)
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

autolabel(rects1, axes[0])
autolabel(rects2, axes[0])

# --- Sağ Grafik: Parametreler ---
rects3 = axes[1].bar(x - width/2, params_baseline, width, label='Baseline', color=color_base, edgecolor='black', zorder=3)
rects4 = axes[1].bar(x + width/2, params_bct, width, label='BCT-integrated', color=color_bct, edgecolor='black', zorder=3)

axes[1].set_ylabel('Reported Parameter Count (Log Scale)')
axes[1].set_title('(b) Parameter-Count Comparison', pad=15)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Ok ve açıklama kutusu (ResNet50 düşüşü için)
axes[1].annotate('93.5% parameter reduction\n0.3340 pp Macro F1 decrease',
                 xy=(x[0] + width/2, params_bct[0]), 
                 xytext=(x[0] + width*1.5, params_bct[0] * 50),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                 fontsize=10, fontweight='bold', ha='left',
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1))

plt.tight_layout()
plt.savefig('figure_04.png', dpi=600, bbox_inches='tight')
print(">> Figure 4 (CIFAR-10) generated successfully.")