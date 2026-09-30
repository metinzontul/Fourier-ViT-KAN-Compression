# -*- coding: utf-8 -*-
"""
Generate Figure 8: ISIC 2019 Performance-Parameter Trade-off
Publication-ready formatting with serif fonts.

Author: Prof. Dr. Metin Zontul
Date: September 2026
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

models = ['ViT+KAN', 'ViT+Linear']
f1_without = [78.9320, 77.4900]
f1_with = [70.0600, 77.2060]
params_without = [55296, 6152]
params_with = [9600, 1416]

x = np.arange(len(models))
width = 0.35

color_base = '#32628d'
color_bct = '#ce5356'

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Left Plot: F1 Scores ---
rects1 = axes[0].bar(x - width/2, f1_without, width, label='Without BCT', color=color_base, edgecolor='black', zorder=3)
rects2 = axes[0].bar(x + width/2, f1_with, width, label='With BCT', color=color_bct, edgecolor='black', zorder=3)

axes[0].set_ylabel('Macro F1 score (%)')
axes[0].set_title('(a) ISIC 2019 validation Macro F1', pad=15)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].set_ylim(68, 81)
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Delta annotations
axes[0].text(x[0], 73.8, r'$\Delta = -8.872$ pp', ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=1))
axes[0].text(x[1], 79.1, r'$\Delta = -0.284$ pp', ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=1))

for rect in rects1 + rects2:
    height = rect.get_height()
    axes[0].annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=11)

# --- Right Plot: Parameter Counts ---
rects3 = axes[1].bar(x - width/2, params_without, width, label='Without BCT', color=color_base, edgecolor='black', zorder=3)
rects4 = axes[1].bar(x + width/2, params_with, width, label='With BCT', color=color_bct, edgecolor='black', zorder=3)

axes[1].set_ylabel('Registered module parameters (log scale)')
axes[1].set_title('(b) Registered classifier-module parameters', pad=15)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Parameter text boxes
axes[1].text(x[0], 25000, '82.6% reduction', ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=1))
axes[1].text(x[1], 2500, '77.0% reduction', ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=1))

for rect in rects3 + rects4:
    height = rect.get_height()
    axes[1].annotate(f'{int(height):,}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('figure_08.png', dpi=600, bbox_inches='tight')
print(">> Figure 8 (ISIC 2019 Dichotomy) generated successfully.")